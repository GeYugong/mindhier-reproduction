"""Freeze the training reference's VQ targets once; not a validation set."""
import hashlib,json,os,sys,time,subprocess
from pathlib import Path
import torch
from safetensors.torch import load_file
ROOT=Path(__file__).resolve().parents[1]
SOURCE='06e95b83a14b07fef51ebdb4c299b6df9914bb34'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    out=ROOT/'runs/checkpoint-review/reference-target.json'
    if out.exists():
        print(json.dumps({'status':'already_prepared','path':str(out)}));return
    frozen=ROOT/'runs/recovery-source-06e95b8'
    for name in ['scripts/stage2_data.py','src/mindhier/MindHier/models/vqvae.py','src/mindhier/MindHier/models/quant.py']:
        expected=subprocess.check_output(['git','show',SOURCE+':'+name],cwd=ROOT)
        assert (frozen/name).read_bytes()==expected,name
    sys.path.insert(0,str(frozen/'scripts'))
    from stage2_data import Stream
    sys.path.insert(0,str(frozen/'src/mindhier/MindHier'))
    import dist
    dist.get_device=lambda:torch.device('cpu')
    from models.vqvae import VQVAE
    torch.set_num_threads(2)
    stream=Stream();images,betas,records=stream.batch(0,1)
    assert records[0]['unconditional'] is False
    config=json.loads((ROOT/'configs/stage2-paper-s01.json').read_text())
    patches=tuple(config['patch_nums'])
    # One image on the same physical GPU and precision as the trainer, without gradients.
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    free,total=torch.cuda.mem_get_info()
    if free < 6*2**30:raise RuntimeError('Defer reference target preparation: need at least 6 GiB free')
    torch.cuda.set_per_process_memory_fraction(.12,device)
    vae=VQVAE(vocab_size=4096,z_channels=32,ch=160,share_quant_resi=4,v_patch_nums=patches,test_mode=True)
    weight=ROOT/'pretrained/vqvae/model.safetensors'
    vae.load_state_dict(load_file(weight),strict=True);vae.to(device).eval().requires_grad_(False)
    with torch.inference_mode():
        a=torch.cat(vae.img_to_idxBl(images.to(device),noise_std=None),1).cpu()
        b=torch.cat(vae.img_to_idxBl(images.to(device),noise_std=None),1).cpu()
    assert torch.equal(a,b) and list(a.shape)==[1,2240]
    result={'schema_version':1,'purpose':'fixed training reference only; not independent validation',
            'source_commit':SOURCE,'position':0,'record':records[0],
            'image_tensor_sha256':hashlib.sha256(images.numpy().tobytes()).hexdigest(),
            'beta_tensor_sha256':hashlib.sha256(betas.numpy().tobytes()).hexdigest(),
            'target_sha256':hashlib.sha256(a.numpy().tobytes()).hexdigest(),
            'vqvae_weights_sha256':sha(weight),'train_manifest_sha256':sha(ROOT/'manifests/subj01_train.csv'),
            'patch_nums':list(patches),'targets':a.tolist()[0],
            'precision':'CUDA FP32, autocast disabled, noise_std=None',
            'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'repeat_targets_equal':True,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,
            'prepared_at':time.time()}
    out.parent.mkdir(parents=True,exist_ok=True)
    temp=out.with_suffix('.partial');temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(out)
    stream.betas.close();stream.images.close()
    print(json.dumps({k:v for k,v in result.items() if k!='targets'},indent=2))
if __name__=='__main__':main()
