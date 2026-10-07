#!/usr/bin/env python3
"""One real training example: CPU-offloaded full-model backward feasibility only."""
import os,sys,json,time,traceback,subprocess,gc,argparse,hashlib
from cpu_gradient_accumulator import CPUGradientAccumulator
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
    ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    out=ROOT/'runs/stage2-batch80-cpu-checkpoint';out.mkdir(exist_ok=True)
    if (out/'status.json').exists() and not args.resume: raise RuntimeError('Use archived evidence and a fresh run directory before rerunning')
    assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()
    result={'phase':'preparing_real_training_input','pid':os.getpid(),'start_time':time.time(),'formal_training':False,'effective_batch':80,'test_data_used':False,'git_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()}
    def save():
        result['updated_at']=time.time();tmp=out/'status.tmp';tmp.write_text(json.dumps(result,indent=2));tmp.replace(out/'status.json')
    save();torch.set_num_threads(4);torch.manual_seed(42);device=torch.device('cuda:0');torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.43,device)
    try:
        patches=(1,2,3,4,6,9,13,18,24,32)
        inputs_path=out/'inputs.pt'
        if inputs_path.exists():
            inputs=torch.load(inputs_path,map_location='cpu',weights_only=False)
        else:
            rows=pd.read_csv(ROOT/'manifests/subj01_train.csv').iloc[:80]
            fmri=FMRI2CLIP(input_dim=15724,multi_subj=False)
            ckpt=torch.load(ROOT/'runs/s01-stage1-paper-seed42/latest.pt',map_location='cpu',weights_only=False,mmap=True)
            fmri.load_state_dict(ckpt['model_state_dict'],strict=True);del ckpt
            fmri.to(device).eval().requires_grad_(False);inputs=[]
            with h5py.File(ROOT/'data/nsd/betas_all_subj01_fp32_renorm.hdf5') as f,torch.no_grad():
                for _,row in rows.iterrows():
                    x=torch.tensor(f['betas'][int(row.beta_row)]).unsqueeze(0).to(device)
                    with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):features=fmri(x)
                    inputs.append({'nsd_id':int(row.nsd_id),'beta_row':int(row.beta_row),'context':[t.float().cpu() for t in features[-1]],'pooled':features[0].float().mean(1).cpu()})
            del fmri,features,x;gc.collect();torch.cuda.empty_cache()
            vae=VQVAE(vocab_size=4096,z_channels=32,ch=160,share_quant_resi=4,v_patch_nums=patches,test_mode=True)
            vae.load_state_dict(load_file(ROOT/'pretrained/vqvae/model.safetensors'),strict=True);vae.to(device).eval().requires_grad_(False)
            with h5py.File(ROOT/'data/nsd_stimuli.hdf5') as f,torch.no_grad():
                for i,item in enumerate(inputs):
                    im=Image.fromarray(f['imgBrick'][item['nsd_id']]).resize((512,512),Image.Resampling.BICUBIC)
                    image=torch.from_numpy(np.array(im).copy()).permute(2,0,1).unsqueeze(0).float().to(device)/127.5-1
                    ids=vae.img_to_idxBl(image,noise_std=0.0)
                    item['target']=torch.cat(ids,dim=1).cpu();item['teacher']=vae.quantize.idxBl_to_hierarchy_ar_input(ids).cpu()
                    result['prepared_samples']=i+1;save()
            del vae,ids,image;gc.collect();torch.cuda.empty_cache()
            torch.save(inputs,inputs_path)
        assert len(inputs)==80
        result['inputs_sha256']=hashlib.file_digest(inputs_path.open('rb'),'sha256').hexdigest();save()
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
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1.0)
        def forward(item):
            with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
                return model(item['teacher'].to(device),[t.to(device) for t in item['context']],item['pooled'].to(device),None)
        def evaluate():
            model.eval()
            with torch.no_grad():output=forward(inputs[0]).float().cpu()
            model.train();return output
        if args.resume:
            result['phase']='restoring_checkpoint';save()
            checkpoint=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False,mmap=True)
            assert checkpoint['step']==1 and checkpoint['inputs_sha256']==result['inputs_sha256']
            assert checkpoint['git_commit']==result['git_commit']
            assert checkpoint['format']=='fsdp_world1_cpu_flat_v1'
            params=dict(model.named_parameters());buffers=dict(model.named_buffers())
            assert set(params)==set(checkpoint['parameters']) and set(buffers)==set(checkpoint['buffers'])
            with torch.no_grad():
                for name,p in params.items():
                    saved=checkpoint['parameters'][name]
                    assert p.shape==saved.shape and p.dtype==saved.dtype
                    p.copy_(saved)
                for name,b in buffers.items():b.copy_(checkpoint['buffers'][name])
            optimizer.load_state_dict(checkpoint['optimizer']);scheduler.load_state_dict(checkpoint['scheduler'])
            torch.set_rng_state(checkpoint['torch_rng']);torch.cuda.set_rng_state(checkpoint['cuda_rng'])
            assert all(int(s['step'])==1 for s in optimizer.state.values())
            # Compare all moments, not merely optimizer step counters.
            actual=optimizer.state_dict()
            for key,value in checkpoint['optimizer']['state'].items():
                for name,tensor in value.items():
                    if torch.is_tensor(tensor):assert torch.equal(actual['state'][key][name],tensor)
            reference=checkpoint['reference_logits'];del actual,checkpoint;gc.collect()
            restored=evaluate();err=(restored-reference).abs().max().item()
            assert torch.allclose(restored,reference,rtol=1e-4,atol=1e-4),err
            result.update(phase='batch80_checkpoint_restore_passed',checkpoint_resume_validated=True,full_batch80_validated=True,restored_output_max_error=err);save();return
        result['phase']='accumulating_batch80';save();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
        acc=CPUGradientAccumulator(model.parameters(),80)
        for i,item in enumerate(inputs):
            optimizer.zero_grad(set_to_none=True)
            logits=forward(item)
            loss=torch.nn.functional.cross_entropy(logits.float().reshape(-1,4096),item['target'].to(device).reshape(-1))
            assert torch.isfinite(loss);loss.backward();acc.add()
            metric={'microbatch':i+1,'loss':loss.item(),'elapsed_seconds':time.monotonic()-start,'time':time.time(),'gpu_peak_gib':torch.cuda.max_memory_allocated()/2**30}
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(metric)+'\n')
            result.update(metric);save();del logits,loss
        acc.apply();result['phase']='optimizer_step';save();optimizer.step();scheduler.step()
        assert all(torch.isfinite(p).all().item() for p in model.parameters())
        optimizer.zero_grad(set_to_none=True);acc.release();del acc;gc.collect()
        reference=evaluate();result['phase']='saving_checkpoint';save()
        # All master parameters already reside on CPU after the offloaded step.
        # Avoid full_state_dict unsharding, which gathers them onto the shared GPU.
        assert all(p.device.type=='cpu' for p in model.parameters())
        payload={'format':'fsdp_world1_cpu_flat_v1','parameters':{n:p.detach().cpu() for n,p in model.named_parameters()},'buffers':{n:b.detach().cpu() for n,b in model.named_buffers()},'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'step':1,'next_sample':80,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'inputs_sha256':result['inputs_sha256'],'git_commit':result['git_commit'],'reference_logits':reference,'effective_batch':80,'world_size':1,'seed':42}
        torch.save(payload,out/'checkpoint.partial');os.replace(out/'checkpoint.partial',out/'checkpoint.pt')
        result.update(phase='batch80_saved_restore_pending',checkpoint_bytes=(out/'checkpoint.pt').stat().st_size,checkpoint_sha256=hashlib.file_digest((out/'checkpoint.pt').open('rb'),'sha256').hexdigest(),seconds=time.monotonic()-start,full_batch80_validated=True,checkpoint_resume_validated=False);save()
    except BaseException as e:
        result.update(phase='failed',error=repr(e),traceback=traceback.format_exc());save();raise
    finally:
        if td.is_initialized():td.destroy_process_group()
if __name__=='__main__':main()
