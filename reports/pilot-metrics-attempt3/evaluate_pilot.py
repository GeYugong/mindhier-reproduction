"""Offline eight-metric evaluation and sanity checks on saved training-only decodes."""
import os,json,time,hashlib,traceback,subprocess,shutil
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import models,transforms as T
from torchvision.models.feature_extraction import create_feature_extractor
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity
import clip
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/s01-stage2-pilot-metrics'

def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def paired_corr(a,b):
    a=a.astype(np.float64);b=b.astype(np.float64);a=a-a.mean(1,keepdims=True);b=b-b.mean(1,keepdims=True)
    return (a*b).sum(1)/np.sqrt((a*a).sum(1)*(b*b).sum(1))
def identify(gt,pred):
    gt=gt.astype(np.float64);pred=pred.astype(np.float64)
    gt-=gt.mean(1,keepdims=True);pred-=pred.mean(1,keepdims=True)
    gt/=np.linalg.norm(gt,axis=1,keepdims=True);pred/=np.linalg.norm(pred,axis=1,keepdims=True)
    r=gt@pred.T;success=r<np.diag(r)[None,:];np.fill_diagonal(success,False)
    return float(success.sum()/(len(gt)*(len(gt)-1)))
def pixels(gt,rec):
    a=T.Resize(425,antialias=True)(gt);b=T.Resize(425,antialias=True)(rec)
    corr=paired_corr(a.flatten(1).numpy(),b.flatten(1).numpy()).mean()
    a=rgb2gray(a.permute(0,2,3,1).numpy());b=rgb2gray(b.permute(0,2,3,1).numpy())
    ssim=np.mean([structural_similarity(x,y,gaussian_weights=True,sigma=1.5,use_sample_covariance=False,data_range=1.0) for x,y in zip(a,b)])
    return {'PixCorr':float(corr),'SSIM':float(ssim)}
def main():
    OUT.mkdir(exist_ok=True);report={'phase':'preparing','started_at':time.time(),'pid':os.getpid(),'formal_result':False,'test_data_used':False,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'metrics':{},'sanity':{},'weights':[],'preprocess':'all GT/recons bilinear resize 256 antialias=True; then author metric-specific resize/normalize; FP32 extraction; FP64 correlations'}
    def save():
        report['updated_at']=time.time();(OUT/'status.json').write_text(json.dumps(report,indent=2))
    save();torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    report["tf32"]=False
    try:
        cache=ROOT/'pretrained/metrics';cache.mkdir(exist_ok=True)
        needed=['alexnet-owt-7be5be79.pth','inception_v3_google-0cc3c7bd.pth','efficientnet_b1_rwightman-533bc792.pth','swav_800ep_pretrain.pth.tar','ViT-L-14.pt']
        for name in needed:
            source=Path.home()/('.cache/clip' if name.endswith('.pt') else '.cache/torch/hub/checkpoints')/name
            dest=cache/name
            if not dest.exists():shutil.copy2(source,dest)
            report['weights'].append({'name':name,'bytes':dest.stat().st_size,'sha256':digest(dest),'source':str(source)})
        source=ROOT/'artifacts/s01-stage2-pilot-decode';rows=[json.loads(l) for l in (source/'samples.jsonl').read_text().splitlines()]
        assert len(rows)==16 and len({r['nsd_id'] for r in rows})==16
        rec=torch.cat([torch.load(source/r['tensor'],weights_only=True) for r in rows]);gt=torch.stack([T.ToTensor()(Image.open(source/(Path(r['tensor']).stem+'-gt.png'))) for r in rows])
        gt=T.Resize(256,antialias=True)(gt);rec=T.Resize(256,antialias=True)(rec)
        report['sample_ids']=[r['nsd_id'] for r in rows];report['metrics'].update(pixels(gt,rec))
        report['sanity']['pixels_identity']=pixels(gt,gt);report['sanity']['pixels_shift']=pixels(gt,gt.roll(1,0))
        assert abs(report['sanity']['pixels_identity']['PixCorr']-1)<1e-8 and abs(report['sanity']['pixels_identity']['SSIM']-1)<1e-8
        save();device='cpu';report['feature_device']='cpu';torch.backends.mkldnn.enabled=False
        normalize=T.Normalize([.485,.456,.406],[.229,.224,.225])
        specifications=[('AlexNet',256,['features.4','features.11'],['AN2','AN5']),('Inception',342,['avgpool'],['Incep']),('CLIP',224,[None],['CLIP']),('Eff',255,['avgpool'],['Eff']),('SwAV',224,['avgpool'],['SwAV'])]
        for kind,size,nodes,names in specifications:
            report['phase']='extracting_'+kind;save()
            norm=normalize
            if kind=='AlexNet':m=models.alexnet(weights=None);m.load_state_dict(torch.load(cache/needed[0],weights_only=True))
            elif kind=='Inception':m=models.inception_v3(weights=None,init_weights=False,transform_input=True);m.load_state_dict(torch.load(cache/needed[1],weights_only=True))
            elif kind=='Eff':m=models.efficientnet_b1(weights=None);m.load_state_dict(torch.load(cache/needed[2],weights_only=True))
            elif kind=='SwAV':
                m=models.resnet50(weights=None);s=torch.load(cache/needed[3],weights_only=True);s={k.replace('module.',''):v for k,v in s.items()};missing,extra=m.load_state_dict(s,strict=False);assert set(missing)<=set(['fc.weight','fc.bias']) and not extra,(missing,extra)
            else:
                m,_=clip.load(str(cache/needed[4]),device='cpu',jit=False);m=m.float();norm=T.Normalize([.48145466,.4578275,.40821073],[.26862954,.26130258,.27577711])
            if kind!='CLIP':m=create_feature_extractor(m,return_nodes=nodes)
            m=m.to(device).eval().requires_grad_(False)
            def extract(images,batch):
                result={name:[] for name in names}
                with torch.inference_mode():
                    for x in images.split(batch):
                        x=norm(T.Resize(size,antialias=True)(x)).to(device)
                        values=m.encode_image(x) if kind=='CLIP' else m(x)
                        for node,name in zip(nodes,names):result[name].append((values if node is None else values[node]).float().flatten(1).cpu().numpy())
                return {k:np.concatenate(v) for k,v in result.items()}
            a=extract(gt,4);b=extract(rec,4);ref=extract(gt[:4],1)
            for name in names:
                error=float(np.max(np.abs(a[name][:4]-ref[name])));scale=float(np.max(np.abs(ref[name])))
                assert np.allclose(a[name][:4],ref[name],atol=2e-4,rtol=2e-4),(name,error,scale)
                np.savez(OUT/(name+'-features.npz'),gt=a[name],reconstruction=b[name],nsd_ids=report['sample_ids'])
                dist=name in ('Eff','SwAV');fn=(lambda x,y:float((1-paired_corr(x,y)).mean())) if dist else identify
                report['metrics'][name]=fn(a[name],b[name]);same=fn(a[name],a[name]);shift=fn(a[name],np.roll(a[name],1,axis=0))
                assert abs(same-(0 if dist else 1))<1e-8,(name,same)
                if not dist:
                    r=np.corrcoef(a[name],b[name])[:16,16:];z=r<np.diag(r)[None,:];np.fill_diagonal(z,False);assert abs(report['metrics'][name]-z.sum()/240)<1e-12
                report['sanity'][name]={'identity':same,'shift':shift,'batch_max_error':error}
            del m;torch.cuda.empty_cache();save()
        assert len(report['metrics'])==8 and all(np.isfinite(v) for v in report['metrics'].values())
        report.update(phase='eight_metrics_sanity_passed',finished_at=time.time());save()
    except BaseException as e:
        report.update(phase='failed',error=repr(e),traceback=traceback.format_exc());save();raise
if __name__=='__main__':main()
