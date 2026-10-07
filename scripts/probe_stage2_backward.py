#!/usr/bin/env python3
"""One real training example: CPU-offloaded full-model backward feasibility only."""
import os,sys,json,time,traceback,subprocess,gc
from pathlib import Path
from functools import partial
import torch, torch.distributed as td
import torch.utils.checkpoint
import pandas as pd
import h5py
import numpy as np
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, CPUOffload, MixedPrecision
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper, CheckpointImpl
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/mindhier/MindHier'))
import dist
dist.get_device=lambda:torch.device('cpu')
from models.switti import HierarchyAR
from models.basic_hierarchy_ar import AdaLNSelfCrossAttn
from models.vqvae import VQVAE
from models.fmri_c2f_encoder import FMRI2CLIP

def main():
    out=ROOT/'runs/stage2-backward-probe-wrapped';out.mkdir(exist_ok=True)
    if (out/'status.json').exists(): raise RuntimeError('Use archived evidence and a fresh run directory before rerunning')
    assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()
    result={'phase':'preparing_real_training_input','pid':os.getpid(),'start_time':time.time(),'formal_training':False,'effective_batch':1,'test_data_used':False,'git_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()}
    def save():
        result['updated_at']=time.time();tmp=out/'status.tmp';tmp.write_text(json.dumps(result,indent=2));tmp.replace(out/'status.json')
    save();torch.set_num_threads(4);torch.manual_seed(42);device=torch.device('cuda:0');torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.43,device)
    try:
        patches=(1,2,3,4,6,9,13,18,24,32)
        row=pd.read_csv(ROOT/'manifests/subj01_train.csv').iloc[0]
        result.update(nsd_id=int(row.nsd_id),beta_row=int(row.beta_row));save()
        with h5py.File(ROOT/'data/nsd/betas_all_subj01_fp32_renorm.hdf5') as f:x=torch.tensor(f['betas'][int(row.beta_row)]).unsqueeze(0).to(device)
        fmri=FMRI2CLIP(input_dim=15724,multi_subj=False)
        ckpt=torch.load(ROOT/'runs/s01-stage1-paper-seed42/latest.pt',map_location='cpu',weights_only=False,mmap=True)
        fmri.load_state_dict(ckpt['model_state_dict'],strict=True);del ckpt
        fmri.to(device).eval().requires_grad_(False)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
            features=fmri(x);context=[t.float() for t in features[-1]];pooled=features[0].float().mean(1)
        del fmri,features,x;gc.collect();torch.cuda.empty_cache()
        with h5py.File(ROOT/'data/nsd_stimuli.hdf5') as f:im=Image.fromarray(f['imgBrick'][int(row.nsd_id)])
        # Deterministic full image is only for memory/backward probe, not the formal augmentation policy.
        im=im.resize((512,512),Image.Resampling.BICUBIC)
        image=torch.from_numpy(np.array(im).copy()).permute(2,0,1).unsqueeze(0).float().to(device)/127.5-1
        vae=VQVAE(vocab_size=4096,z_channels=32,ch=160,share_quant_resi=4,v_patch_nums=patches,test_mode=True)
        vae.load_state_dict(load_file(ROOT/'pretrained/vqvae/model.safetensors'),strict=True);vae.to(device).eval().requires_grad_(False)
        with torch.no_grad():
            ids=vae.img_to_idxBl(image,noise_std=0.0);target=torch.cat(ids,dim=1);teacher=vae.quantize.idxBl_to_hierarchy_ar_input(ids)
        del vae,ids,image;gc.collect();torch.cuda.empty_cache()
        result['phase']='loading_generator';save()
        torch.manual_seed(42)
        model=HierarchyAR(depth=30,embed_dim=1920,num_heads=30,patch_nums=patches,use_crop_cond=False);model.init_weights(init_std=-1)
        audit=json.loads((ROOT/'manifests/switti-adaptation.json').read_text());allowed={x['name'] for x in audit['missing']+audit['shape_mismatch']}
        state=model.state_dict();loaded=[]
        with safe_open(ROOT/'pretrained/switti/model.safetensors',framework='pt',device='cpu') as f:
            for k in f.keys():
                if k in state and tuple(f.get_slice(k).get_shape())==tuple(state[k].shape):
                    t=f.get_tensor(k)
                    if k=='attn_bias_for_masking':assert torch.equal(t,state[k])
                    else:assert torch.isfinite(t).all()
                    state[k].copy_(t);loaded.append(k)
        assert set(state)-set(loaded)==allowed
        del state,t;gc.collect()
        rendezvous=out/'rendezvous'
        td.init_process_group('nccl',init_method='file://'+str(rendezvous),rank=0,world_size=1)
        model.freqs_cis=model.freqs_cis.to(device);model.disable_gradient_checkpointing()
        for i,block in enumerate(model.blocks):
            model.blocks[i]=FSDP(checkpoint_wrapper(block,checkpoint_impl=CheckpointImpl.NO_REENTRANT),cpu_offload=CPUOffload(offload_params=True),mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,buffer_dtype=torch.float32,cast_forward_inputs=True),device_id=device)
        model=FSDP(model,cpu_offload=CPUOffload(offload_params=True),device_id=device)
        model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.9,.95),weight_decay=.01,foreach=False)
        result['phase']='real_forward_backward';save();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
        with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
            logits=model(teacher,context,pooled,None)
            loss=torch.nn.functional.cross_entropy(logits.float().reshape(-1,4096),target.reshape(-1))
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())
        result.update(loss=loss.item(),backward_seconds=time.monotonic()-start,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,phase='optimizer_step');save()
        optimizer.step()
        assert all(torch.isfinite(p).all().item() for p in model.parameters())
        result.update(phase='single_example_update_passed',seconds=time.monotonic()-start,gradients_on_cpu=all(p.grad is None or p.grad.device.type=='cpu' for p in model.parameters()),checkpoint_resume_validated=False,full_batch80_validated=False);save()
    except BaseException as e:
        result.update(phase='failed',error=repr(e),traceback=traceback.format_exc());save();raise
    finally:
        if td.is_initialized():td.destroy_process_group()
if __name__=='__main__':main()
