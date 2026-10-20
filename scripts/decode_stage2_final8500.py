"""Decode all 1000 S1 test images using fixed step 8500; retain every candidate."""
import sys,os,json,time,traceback,subprocess,hashlib,gc,argparse,fcntl
from pathlib import Path
import torch,numpy as np,pandas as pd,h5py
from PIL import Image
from safetensors.torch import load_file
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/mindhier/MindHier'))
import dist
dist.get_device=lambda:torch.device('cpu')
from models.switti import HierarchyAR
from models.vqvae import VQVAE
from models.fmri_c2f_encoder import FMRI2CLIP
from models.pipeline import MindHierPipeline

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidates',type=int,choices=[1,4],required=True);args=ap.parse_args()
    out=ROOT/f'artifacts/s01-final8500-n{args.candidates}';out.mkdir(parents=True,exist_ok=True)
    lock=(out/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    export=json.loads((ROOT/'artifacts/s01-stage2-final8500/export-status.json').read_text())
    assert export['phase']=='export_verified' and export['checkpoint_step']==8500 and export['max_absolute_error']==0
    assert json.loads((ROOT/'runs/s01-stage2-8500-export/state.json').read_text())['exit_code']==0
    assert not (out/'status.json').exists(), 'Do not overwrite a previous decode run'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    r={'phase':'loading','pid':os.getpid(),'started_at':time.time(),'formal_training':False,'test_data_used':True,'gt_used_for_generation':False,'candidates_per_image':args.candidates,'checkpoint_step':8500,'training_budget':8500,'original_budget':9000,'generator_sha256':export['sha256'],'subject_batch':1,'seed_reset_each_image':59,'response_aggregation':'mean of three available trials','precision':'BF16 blocks and autocast, FP32 root','image_order':'ascending nsd_id','context_order':'forward','mask':'global query offset per scale','target_samples':1000,'completed_samples':0,'seed_base':59,'cfg':6,'top_k':400,'top_p':.95,'turn_on_cfg_start_si':1,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    def save():
        r['updated_at']=time.time();(out/'status.tmp').write_text(json.dumps(r,indent=2));(out/'status.tmp').replace(out/'status.json')
    save();torch.set_num_threads(4);torch.manual_seed(42);device=torch.device('cuda:0');torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.75,device)
    try:
        weight=ROOT/'artifacts/s01-stage2-final8500/generator-fp32.pt'
        assert hashlib.file_digest(weight.open('rb'),'sha256').hexdigest()==export['sha256']
        patches=(1,2,3,4,6,9,13,18,24,32)
        model=HierarchyAR(depth=30,embed_dim=1920,num_heads=30,patch_nums=patches,use_crop_cond=False)
        model.load_state_dict(torch.load(ROOT/'artifacts/s01-stage2-final8500/generator-fp32.pt',map_location='cpu',weights_only=True,mmap=True),strict=True)
        fmri=FMRI2CLIP(input_dim=15724,multi_subj=False)
        ck=torch.load(ROOT/'runs/s01-stage1-paper-seed42/latest.pt',map_location='cpu',weights_only=False,mmap=True);fmri.load_state_dict(ck['model_state_dict'],strict=True);del ck;gc.collect()
        vae=VQVAE(vocab_size=4096,z_channels=32,ch=160,share_quant_resi=4,v_patch_nums=patches,test_mode=True)
        vae.load_state_dict(load_file(ROOT/'pretrained/vqvae/model.safetensors'),strict=True)
        pipeline=MindHierPipeline(model,vae,fmri,device=device,clip_model_name_or_path=str(ROOT/'pretrained/clip-vit-large-patch14'),context_order='forward')
        model.to(device);vae.to(device);fmri.to(device);model.freqs_cis=model.freqs_cis.to(device);model.rng=torch.Generator(device=device)
        def cast_inputs(module,args,kwargs):
            def cast(x):return x.to(torch.bfloat16) if torch.is_tensor(x) and x.is_floating_point() else x
            return tuple(cast(x) for x in args),{k:cast(v) for k,v in kwargs.items()}
        for block in model.blocks:
            block.to(torch.bfloat16);block.register_forward_pre_hook(cast_inputs,with_kwargs=True)
        manifest=ROOT/'manifests/subj01_test_trials.csv'
        rows=pd.read_csv(manifest);train=pd.read_csv(ROOT/'manifests/subj01_train.csv')
        assert len(rows)==3000 and rows.nsd_id.nunique()==1000 and not set(rows.nsd_id)&set(train.nsd_id)
        groups=list(rows.groupby('nsd_id',sort=True));assert all(len(g)==3 for _,g in groups)
        r['test_manifest_sha256']=hashlib.file_digest(manifest.open('rb'),'sha256').hexdigest()
        r['phase']='decoding';save()
        with h5py.File(ROOT/'data/nsd/betas_all_subj01_fp32_renorm.hdf5') as f,h5py.File(ROOT/'data/nsd_stimuli.hdf5') as images,torch.inference_mode():
            for i,(nsd_id,group) in enumerate(groups):
                beta_rows=sorted(int(v) for v in group.beta_row)
                x=torch.from_numpy(np.stack([f['betas'][j] for j in beta_rows])).mean(0,keepdim=True).to(device)
                prompt=x.repeat_interleave(args.candidates,dim=0)
                torch.cuda.synchronize();start=time.time()
                with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
                    result=pipeline(prompt,seed=59,cfg=6,top_k=400,top_p=.95,turn_on_cfg_start_si=1,return_pil=False)
                torch.cuda.synchronize();elapsed=time.time()-start
                result=result.float().cpu();assert result.shape==(args.candidates,3,512,512)
                assert torch.isfinite(result).all() and result.min()>=0 and result.max()<=1
                name=f'{i:04d}-nsd{int(nsd_id)}';pt=out/(name+'.pt');assert not pt.exists()
                torch.save(result,out/(name+'.partial'));(out/(name+'.partial')).replace(pt)
                pngs=[]
                for c in range(args.candidates):
                    png=out/(name+f'-candidate{c}.png')
                    Image.fromarray((result[c].permute(1,2,0).numpy()*255).round().astype(np.uint8)).save(png);pngs.append(png.name)
                Image.fromarray(images['imgBrick'][int(nsd_id)]).save(out/(name+'-gt.png'))
                record={'index':i,'nsd_id':int(nsd_id),'beta_rows':beta_rows,'seed':59,'seconds':elapsed,'tensor':pt.name,'candidate_pngs':pngs,'gt_png':name+'-gt.png','sha256':hashlib.file_digest(pt.open('rb'),'sha256').hexdigest()}
                with (out/'samples.jsonl').open('a') as log:log.write(json.dumps(record)+'\n')
                r['completed_samples']=i+1;save()
        r.update(phase='decoded_metrics_pending',finished_at=time.time(),peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30);save()
    except BaseException as e:
        r.update(phase='failed',error=repr(e),traceback=traceback.format_exc());save();raise
if __name__=='__main__':main()
