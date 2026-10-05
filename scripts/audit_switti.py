#!/usr/bin/env python3
"""Audit original Switti-to-MindHier parameter compatibility without allocating weights."""
import json,struct,sys,math
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src/mindhier/MindHier"))
import dist
dist.get_device=lambda:torch.device("cpu")
from models.switti import HierarchyAR
old_linspace=torch.linspace
def cpu_linspace(*args,**kwargs):
    kwargs["device"]="cpu";return old_linspace(*args,**kwargs)
torch.linspace=cpu_linspace
with torch.device("meta"):
    model=HierarchyAR(depth=30,embed_dim=1920,num_heads=30,patch_nums=(1,2,3,4,6,9,13,18,24,32),use_crop_cond=False)
torch.linspace=old_linspace
with (ROOT/"pretrained/switti/model.safetensors").open("rb") as f:
    n=struct.unpack("<Q",f.read(8))[0];header=json.loads(f.read(n))
state=model.state_dict();params=dict(model.named_parameters())
matching=[];missing=[];mismatch=[]
for name,p in state.items():
    if name not in header:missing.append({"name":name,"shape":list(p.shape)})
    elif list(p.shape)!=header[name]["shape"]:mismatch.append({"name":name,"source":header[name]["shape"],"target":list(p.shape)})
    else:matching.append(name)
unexpected=[k for k in header if k not in state and k!="__metadata__"]
total=sum(p.numel() for p in params.values());covered=sum(params[k].numel() for k in matching if k in params)
result={"source":"yresearch/Switti","target_depth":30,"target_embed_dim":1920,"target_context_dim":768,"total_target_parameters":total,"matching_parameters":covered,"coverage":covered/total,"matching_keys":len(matching),"missing":missing,"shape_mismatch":mismatch,"unexpected":unexpected,"interpretation":"Original text conditioning uses 2048-dimensional context and 1280-dimensional pooled embeddings; MindHier conditions on 768-dimensional fMRI features. Incompatible conditioning tensors require documented initialization; all other core weights must match."}
(ROOT/"manifests/switti-adaptation.json").write_text(json.dumps(result,indent=2))
print(json.dumps({"total":total,"coverage":covered/total,"missing_count":len(missing),"shape_mismatch_count":len(mismatch),"unexpected_count":len(unexpected),"example_mismatches":mismatch[:4]}))
