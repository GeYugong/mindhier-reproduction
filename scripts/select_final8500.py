"""Offline GT-assisted selection; preserve all four original candidates."""
import os,json,time,hashlib,subprocess,traceback,fcntl
from pathlib import Path
import torch,clip
from PIL import Image
from torchvision import transforms as T
ROOT=Path(__file__).resolve().parents[1]
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 out=ROOT/'artifacts/s01-final8500-n4-selected';out.mkdir(exist_ok=True)
 lock=(out/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (out/'status.json').exists()
 report={'phase':'loading','pid':os.getpid(),'started_at':time.time(),'completed_samples':0,'gt_assisted':True,'checkpoint_step':8500,'selection':'OpenAI CLIP ViT-L/14 global embedding cosine; first argmax on ties','precision':'GPU FP32 TF32 off','gt_preprocess':'PIL bilinear 512 then tensor; CLIP bicubic224 antialias=None center crop normalize','git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
 def save():
  report['updated_at']=time.time();(out/'status.tmp').write_text(json.dumps(report,indent=2));(out/'status.tmp').replace(out/'status.json')
 save();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 try:
  source=ROOT/'artifacts/s01-final8500-n4';status=json.loads((source/'status.json').read_text());assert status['completed_samples']==1000 and status['phase']=='decoded_metrics_pending'
  rows=[json.loads(l) for l in (source/'samples.jsonl').read_text().splitlines()];assert len(rows)==1000 and len({x['nsd_id'] for x in rows})==1000
  weight=ROOT/'pretrained/metrics/ViT-L-14.pt';report['clip_sha256']=digest(weight)
  model,_=clip.load(str(weight),device='cpu',jit=False);model=model.float().to('cuda:0').eval().requires_grad_(False)
  pre=T.Compose([T.Resize(224,interpolation=T.InterpolationMode.BICUBIC,antialias=None),T.CenterCrop((224,224)),T.Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))])
  gt_pre=T.Compose([T.Resize((512,512)),T.ToTensor()])
  report['phase']='selecting';save()
  with torch.inference_mode():
   for i,row in enumerate(rows):
    assert row['index']==i
    p=source/row['tensor'];assert digest(p)==row['sha256']
    candidates=torch.load(p,map_location='cpu',weights_only=True);assert candidates.shape==(4,3,512,512) and torch.isfinite(candidates).all()
    gt=gt_pre(Image.open(source/row['gt_png']).convert('RGB')).unsqueeze(0)
    features=model.encode_image(pre(torch.cat([gt,candidates]).cuda())).float()
    features=torch.nn.functional.normalize(features,dim=-1);scores=torch.cosine_similarity(features[:1],features[1:],dim=-1).cpu();assert torch.isfinite(scores).all()
    pick=int(scores.argmax());name=row['tensor'];torch.save(candidates[pick:pick+1],out/name)
    os.link(source/row['gt_png'],out/row['gt_png']);os.link(source/row['candidate_pngs'][pick],out/(Path(name).stem+'-selected.png'))
    result=dict(row,tensor=name,sha256=digest(out/name),selected_index=pick,selection_scores=scores.tolist(),source_tensor_sha256=row['sha256'])
    with (out/'samples.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
    report['completed_samples']=i+1;save()
  report.update(phase='selection_complete',finished_at=time.time());save()
 except BaseException as e:
  report.update(phase='failed',error=repr(e),traceback=traceback.format_exc());save();raise
if __name__=='__main__':main()
