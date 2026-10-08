"""Deterministic exposure-indexed sampling, crops and classifier-free dropout."""
import math,hashlib
import numpy as np
import pandas as pd
import h5py,torch
from PIL import Image
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class Stream:
    def __init__(self,seed=42):
        self.seed=seed;self.rows=pd.read_csv(ROOT/'manifests/subj01_train.csv');self.permutations={}
        self.betas=h5py.File(ROOT/'data/nsd/betas_all_subj01_fp32_renorm.hdf5','r');self.images=h5py.File(ROOT/'data/nsd_stimuli.hdf5','r')
    def spec(self,position):
        epoch,index=divmod(position,len(self.rows))
        if epoch not in self.permutations:
            self.permutations={epoch:torch.randperm(len(self.rows),generator=torch.Generator().manual_seed(self.seed+epoch)).tolist()}
        row=self.rows.iloc[self.permutations[epoch][index]]
        g=torch.Generator().manual_seed(self.seed*1000000007+position)
        top,left=torch.randint(0,65,(2,),generator=g).tolist();drop=bool(torch.rand((),generator=g)<.1)
        return {'position':position,'epoch':epoch,'nsd_id':int(row.nsd_id),'beta_row':int(row.beta_row),'top':top,'left':left,'unconditional':drop}
    def batch(self,position,count):
        records=[self.spec(position+i) for i in range(count)];images=[];betas=[]
        for r in records:
            im=Image.fromarray(self.images['imgBrick'][r['nsd_id']]).resize((576,576),Image.Resampling.LANCZOS)
            im=im.crop((r['left'],r['top'],r['left']+512,r['top']+512))
            images.append(torch.from_numpy(np.array(im).copy()).permute(2,0,1).float()/127.5-1)
            beta=torch.from_numpy(self.betas['betas'][r['beta_row']].copy())
            if r['unconditional']:beta.zero_()
            betas.append(beta)
        return torch.stack(images),torch.stack(betas),records

def lr_ratio(completed):
    if completed<100:return .005+.995*completed/100
    progress=min(1,(completed-100)/(8999-100))
    return .01+.99*(1+math.cos(math.pi*progress))/2

def check():
    s=Stream();a,b,records=s.batch(26998,4);restored=Stream();c,d,again=restored.batch(26998,4)
    assert records==again and torch.equal(a,c) and torch.equal(b,d)
    ids=[s.spec(i)['beta_row'] for i in range(27000)];assert len(set(ids))==27000
    assert lr_ratio(0)==.005 and lr_ratio(100)==1 and lr_ratio(8999)==.01
    return {'passed':True,'crop_and_dropout_replay':True,'cross_epoch_replay':True,'unique_trials_in_epoch':len(set(ids)),'sample_specs':records}
if __name__=='__main__':
    import json
    print(json.dumps(check()))
