"""Export world1 flat pilot weights on CPU and compare independent inference logits."""
import sys,json,re,time,hashlib,traceback,gc
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/mindhier/MindHier'))
import dist
dist.get_device=lambda:torch.device('cpu')
from models.switti import HierarchyAR

def main():
    out=ROOT/'artifacts/s01-stage2-pilot-seed42';out.mkdir(parents=True,exist_ok=True)
    report={'phase':'exporting','formal_training':False,'started_at':time.time()}
    def save(): (out/'export-status.json').write_text(json.dumps(report,indent=2))
    save();torch.set_num_threads(4)
    try:
        source=ROOT/'runs/s01-stage2-pilot-seed42/checkpoint-step-100.pt'
        ck=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
        assert ck['format']=='fsdp_world1_cpu_flat_v1' and ck['step']==100 and ck['pilot']
        model=HierarchyAR(depth=30,embed_dim=1920,num_heads=30,patch_nums=(1,2,3,4,6,9,13,18,24,32),use_crop_cond=False)
        def native(name):return name.replace('_fsdp_wrapped_module.','').replace('_checkpoint_wrapped_module.','')
        params=dict(model.named_parameters());copied=set()
        with torch.no_grad():
            for key,flat in ck['parameters'].items():
                name=native(key);assert name.endswith('_flat_param'),key
                prefix=name[:-len('_flat_param')]
                if prefix:group=[(n,p) for n,p in params.items() if n.startswith(prefix)]
                else:group=[(n,p) for n,p in params.items() if not n.startswith('blocks.')]
                assert sum(p.numel() for _,p in group)==flat.numel(),key
                offset=0
                for n,p in group:
                    assert n not in copied;nval=flat[offset:offset+p.numel()].view_as(p)
                    assert torch.isfinite(nval).all(),n
                    p.copy_(nval);offset+=p.numel();copied.add(n)
            assert copied==set(params)
            buffers=dict(model.named_buffers())
            assert {native(n) for n in ck['buffers']}==set(buffers)
            for n,t in ck['buffers'].items():buffers[native(n)].copy_(t)
        reference=ck['reference_logits'].clone();del ck;gc.collect()
        path=out/'generator-fp32.pt';torch.save(model.state_dict(),out/'generator-fp32.partial');(out/'generator-fp32.partial').replace(path)
        # Reload the saved artifact itself before numerical comparison.
        loaded=torch.load(path,map_location='cpu',weights_only=True,mmap=True);model.load_state_dict(loaded,strict=True);del loaded
        report.update(phase='validating_export',export_bytes=path.stat().st_size,parameter_keys=len(copied));save()
        device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(.65,device)
        model.to(device).eval();model.freqs_cis=model.freqs_cis.to(device)
        def cast_inputs(module,args,kwargs):
            def cast(x):return x.to(torch.bfloat16) if torch.is_tensor(x) and x.is_floating_point() else x
            return tuple(cast(x) for x in args),{k:cast(v) for k,v in kwargs.items()}
        for block in model.blocks:
            block.to(torch.bfloat16);block.register_forward_pre_hook(cast_inputs,with_kwargs=True)
        sample=torch.load(ROOT/'runs/stage2-batch80-cpu-checkpoint/inputs.pt',map_location='cpu',weights_only=False)[0]
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
            value=model(sample['teacher'].to(device),[x.to(device) for x in sample['context']],sample['pooled'].to(device),None).float().cpu()
        error=(value-reference).abs().max().item();report['max_absolute_error']=error
        assert torch.allclose(value,reference,atol=1e-4,rtol=1e-4),error
        report.update(phase='export_verified',sha256=hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),finished_at=time.time());save()
    except BaseException as e:
        report.update(phase='failed',error=repr(e),traceback=traceback.format_exc());save();raise
if __name__=='__main__':main()
