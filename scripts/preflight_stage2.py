#!/usr/bin/env python3
"""Actual weight loading and full-resolution forward probe; not Stage 2 training."""
import json,sys,time,subprocess,traceback
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import load_file
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/mindhier/MindHier'))
import dist
dist.get_device=lambda:torch.device('cpu')
from models.switti import HierarchyAR
from models.vqvae import VQVAE
from models.fmri_c2f_encoder import FMRI2CLIP

def main():
    out=ROOT/'runs/stage2-preflight';out.mkdir(exist_ok=True)
    result={'start_time':time.time(),'phase':'weight_loading','training_started':False,'input':'synthetic full-resolution shape probe','git_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()}
    def save(): (out/'status.json').write_text(json.dumps(result,indent=2))
    save();torch.set_num_threads(4);torch.manual_seed(42)
    try:
        patches=(1,2,3,4,6,9,13,18,24,32)
        model=HierarchyAR(depth=30,embed_dim=1920,num_heads=30,patch_nums=patches,use_crop_cond=False)
        model.init_weights(init_std=-1)
        audit=json.loads((ROOT/'manifests/switti-adaptation.json').read_text())
        allowed={x['name'] for x in audit['missing']+audit['shape_mismatch']}
        state=model.state_dict(); loaded=[]; skipped=[]
        with safe_open(ROOT/'pretrained/switti/model.safetensors',framework='pt',device='cpu') as f:
            for k in f.keys():
                if k in state and tuple(f.get_slice(k).get_shape())==tuple(state[k].shape):
                    t=f.get_tensor(k); assert torch.isfinite(t).all(),k
                    state[k].copy_(t);loaded.append(k)
                else:skipped.append(k)
        missing=set(state)-set(loaded)
        assert missing==allowed,(missing ^ allowed)
        assert set(skipped)==set(audit['unexpected'])|{x['name'] for x in audit['shape_mismatch']}
        params=dict(model.named_parameters());coverage=sum(params[k].numel() for k in loaded if k in params)/sum(p.numel() for p in params.values())
        assert coverage>=0.96
        result.update(loaded_keys=len(loaded),initialized_keys=sorted(missing),coverage=coverage,initialization='author init_weights(init_std=-1), seed 42; changed conditioning layers retain initialization')
        vae=VQVAE(vocab_size=4096,z_channels=32,ch=160,share_quant_resi=4,v_patch_nums=patches,test_mode=True)
        vae.load_state_dict(load_file(ROOT/'pretrained/vqvae/model.safetensors'),strict=True)
        result['vae_strict_load']=True;del vae
        fmri=FMRI2CLIP(input_dim=15724,multi_subj=False)
        ckpt=torch.load(ROOT/'runs/s01-stage1-paper-seed42/latest.pt',map_location='cpu',weights_only=False,mmap=True)
        fmri.load_state_dict(ckpt['model_state_dict'],strict=True)
        result['fmri_strict_load']=True;del fmri,ckpt
        result['phase']='forward_probe';save()
        device=torch.device('cuda:0');free,total=torch.cuda.mem_get_info(device)
        result['free_gpu_bytes_before']=free
        if free<14*2**30:
            result.update(phase='waiting_for_memory',required_free_gib=14);save();return
        torch.cuda.set_per_process_memory_fraction(0.45,device)
        model=model.to(device).eval()
        model.freqs_cis=model.freqs_cis.to(device)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(torch.randn(1,model.L-1,32,device=device),[torch.randn(1,257,768,device=device) for _ in range(4)],torch.randn(1,768,device=device),None)
        assert torch.isfinite(logits).all()
        result.update(phase='forward_passed_backward_pending',logits_shape=list(logits.shape),peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,finished_at=time.time())
        save();print(json.dumps(result),flush=True)
    except BaseException as exc:
        result.update(phase='failed',error=repr(exc),traceback=traceback.format_exc());save();raise
if __name__=='__main__':main()
