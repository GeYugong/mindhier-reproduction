"""Fresh-pretrained Stage 2 training with deterministic full-trial stream."""
import os,sys,json,time,gc,hashlib,subprocess,traceback,argparse,random
from pathlib import Path
import numpy as np,torch,torch.distributed as td
from safetensors import safe_open
from safetensors.torch import load_file
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,CPUOffload,MixedPrecision
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper,CheckpointImpl
from cpu_gradient_accumulator import CPUGradientAccumulator
from stage2_data import Stream,lr_ratio
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src/mindhier/MindHier'))
import dist
dist.get_device=lambda:torch.device('cpu')
from models.switti import HierarchyAR
from models.vqvae import VQVAE
from models.fmri_c2f_encoder import FMRI2CLIP

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--steps',type=int,default=9000);ap.add_argument('--microbatch',type=int,default=4);ap.add_argument('--resume',action='store_true');ap.add_argument('--validation',action='store_true');args=ap.parse_args()
    assert 80%args.microbatch==0 and (args.validation or args.steps==9000)
    out=ROOT/'runs'/args.run;out.mkdir(exist_ok=True)
    assert args.resume or not (out/'status.json').exists(),'Run already exists'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    config=json.loads((ROOT/'configs/stage2-paper-s01.json').read_text());config.update(microbatch=args.microbatch,validation=args.validation)
    config_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    r={'phase':'loading','pid':os.getpid(),'started_at':time.time(),'formal_training':not args.validation,'git_commit':code,'completed_steps':0,'target_steps':args.steps,'microbatch_size':args.microbatch,'effective_batch':80}
    def save():
        r['updated_at']=time.time();(out/'status.tmp').write_text(json.dumps(r,indent=2));(out/'status.tmp').replace(out/'status.json')
    save();(out/'config.json').write_text(json.dumps(config,indent=2));torch.set_num_threads(4)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(.85,device)
    try:
        patches=tuple(config['patch_nums']);stream=Stream()
        fmri=FMRI2CLIP(input_dim=15724,multi_subj=False)
        ck=torch.load(ROOT/'runs/s01-stage1-paper-seed42/latest.pt',map_location='cpu',weights_only=False,mmap=True);fmri.load_state_dict(ck['model_state_dict'],strict=True);del ck
        fmri.to(device).eval().requires_grad_(False)
        vae=VQVAE(vocab_size=4096,z_channels=32,ch=160,share_quant_resi=4,v_patch_nums=patches,test_mode=True)
        vae.load_state_dict(load_file(ROOT/'pretrained/vqvae/model.safetensors'),strict=True);vae.to(device).eval().requires_grad_(False)
        def prepare(position,count):
            images,betas,records=stream.batch(position,count)
            with torch.no_grad():
                with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):features=fmri(betas.to(device))
                context=[t.float().detach() for t in features[-1]][::-1];pooled=features[0].float().mean(1)
                ids=vae.img_to_idxBl(images.to(device),noise_std=None);target=torch.cat(ids,1);teacher=vae.quantize.idxBl_to_hierarchy_ar_input(ids)
            return teacher,context,pooled,target,records
        reference_inputs=prepare(0,1)
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
        td.init_process_group('nccl',init_method='file://'+str(out/('rendezvous-'+str(os.getpid()))),rank=0,world_size=1)
        model.freqs_cis=model.freqs_cis.to(device);model.disable_gradient_checkpointing()
        for i,block in enumerate(model.blocks):
            model.blocks[i]=FSDP(checkpoint_wrapper(block,checkpoint_impl=CheckpointImpl.NO_REENTRANT),cpu_offload=CPUOffload(offload_params=True),mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,buffer_dtype=torch.float32,cast_forward_inputs=True),device_id=device)
        model=FSDP(model,cpu_offload=CPUOffload(offload_params=True),device_id=device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.9,.95),weight_decay=0,foreach=False)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_ratio)
        def forward(data):
            with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):return model(data[0],data[1],data[2],None)
        def evaluate():
            model.eval()
            with torch.no_grad():value=forward(reference_inputs).float().cpu()
            model.train();return value
        step0=0
        if args.resume:
            ck=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False,mmap=True)
            assert ck['config_hash']==config_hash and ck['git_commit']==code
            params=dict(model.named_parameters());buffers=dict(model.named_buffers());assert params.keys()==ck['parameters'].keys() and buffers.keys()==ck['buffers'].keys()
            with torch.no_grad():
                for n,p in params.items():assert p.shape==ck['parameters'][n].shape;p.copy_(ck['parameters'][n])
                for n,b in buffers.items():b.copy_(ck['buffers'][n])
            optimizer.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler']);step0=ck['step'];assert ck['next_sample']==step0*80
            actual=optimizer.state_dict()
            for key,values in ck['optimizer']['state'].items():
                for name,t in values.items():
                    if torch.is_tensor(t):assert torch.equal(actual['state'][key][name],t)
            assert scheduler.last_epoch==step0
            reference=ck['reference_logits'];torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state(ck['cuda_rng']);random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng'])
            assert stream.spec(step0*80)==ck['next_sample_spec']
            del actual,ck;gc.collect();restored=evaluate();err=(restored-reference).abs().max().item();assert torch.allclose(restored,reference,atol=1e-4,rtol=1e-4),err
            r.update(resume_verified=True,restored_output_max_error=err,restored_step=step0);save()
        model.train()
        for step in range(step0+1,args.steps+1):
            start=time.monotonic();r.update(phase='validation_training' if args.validation else 'formal_training',completed_steps=step-1,active_step=step);save();torch.cuda.reset_peak_memory_stats()
            acc=CPUGradientAccumulator(model.parameters(),80);losses=[];records=[];lr=optimizer.param_groups[0]['lr']
            for micro in range(0,80,args.microbatch):
                optimizer.zero_grad(set_to_none=True);data=prepare((step-1)*80+micro,args.microbatch);logits=forward(data)
                loss=torch.nn.functional.cross_entropy(logits.float().reshape(-1,4096),data[3].reshape(-1));assert torch.isfinite(loss)
                loss.backward();acc.add(args.microbatch);losses.append(loss.item());records.extend(data[4]);del logits,loss,data
                r.update(completed_micro_samples=micro+args.microbatch);save()
            acc.apply();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True,foreach=False);optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);acc.release();del acc;gc.collect()
            assert all(torch.isfinite(p).all().item() for p in model.parameters())
            assert all(p.grad is None for p in fmri.parameters()) and all(p.grad is None for p in vae.parameters())
            metric={'step':step,'loss':sum(losses)/len(losses),'lr':lr,'grad_norm':float(norm),'seconds':time.monotonic()-start,'time':time.time(),'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,'next_sample':step*80}
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(metric)+'\n')
            with (out/'samples.jsonl').open('a') as f:f.write(json.dumps({'step':step,'samples':records})+'\n')
            r.update(metric,completed_steps=step);save()
            if args.validation or step==1 or step%100==0 or step==9000:
                reference=evaluate();r['phase']='saving_checkpoint';save();assert all(p.device.type=='cpu' for p in model.parameters())
                payload={'format':'fsdp_world1_cpu_flat_v1','parameters':{n:p.detach().cpu() for n,p in model.named_parameters()},'buffers':{n:b.detach().cpu() for n,b in model.named_buffers()},'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'step':step,'next_sample':step*80,'next_sample_spec':stream.spec(step*80),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'config_hash':config_hash,'git_commit':code,'reference_logits':reference,'effective_batch':80,'world_size':1,'seed':42,'pilot':args.validation}
                torch.save(payload,out/'checkpoint.partial');os.replace(out/'checkpoint.partial',out/'checkpoint.pt')
                (out/'checkpoint.json').write_text(json.dumps({'step':step,'bytes':(out/'checkpoint.pt').stat().st_size,'time':time.time(),'config_hash':config_hash}))
                if not args.validation and step%500==0:os.link(out/'checkpoint.pt',out/f'checkpoint-step-{step:04d}.pt')
                del payload,reference;gc.collect()
        r.update(phase='validation_completed' if args.validation else 'formal_completed',completed_steps=args.steps);save()
    except BaseException as e:
        r.update(phase='failed',error=repr(e),traceback=traceback.format_exc());save();raise
    finally:
        if td.is_initialized():td.destroy_process_group()
if __name__=='__main__':main()
