#!/usr/bin/env python3
"""Validate official NSD volume indices and prepare captions without rewriting source data."""
import argparse,csv,hashlib,io,json,tarfile,zipfile,collections,time
from pathlib import Path
import numpy as np,pandas as pd,h5py,requests
ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path("/data/matengyu/geyugong/neuroadapter-subject1-research/data/raw/nsd/experiments/nsd")
def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda:f.read(8*1024*1024),b""):h.update(c)
    return h.hexdigest()
def captions(meta):
    out=ROOT/"data/captions_73k.json"
    if out.exists():return
    target=ROOT/"data/annotations_trainval2017.zip"
    if not target.exists():
        part=target.with_suffix(".zip.partial")
        url="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        with requests.get(url,stream=True,timeout=(15,120)) as response:
            response.raise_for_status()
            with part.open("wb") as f:
                for c in response.iter_content(4*1024*1024):f.write(c)
        with zipfile.ZipFile(part) as z:
            if z.testzip() is not None:raise ValueError("COCO archive CRC failed")
        part.replace(target)
    lookup=collections.defaultdict(list)
    with zipfile.ZipFile(target) as z:
        for split in ["train2017","val2017"]:
            payload=json.loads(z.read("annotations/captions_"+split+".json"))
            for a in payload["annotations"]:lookup[int(a["image_id"])].append(a["caption"])
    result=[lookup[int(i)] for i in meta.cocoId]
    assert len(result)==73000 and min(map(len,result))>=3
    out.write_text(json.dumps(result,ensure_ascii=False),encoding="utf-8")
    (ROOT/"manifests/captions.json").write_text(json.dumps({"source":"http://images.cocodataset.org/annotations/annotations_trainval2017.zip","archive_sha256":digest(target),"captions_sha256":digest(out),"mapping_sha256":digest(SOURCE/"nsd_stim_info_merged.csv"),"selection":"first 3 annotations in original COCO annotation order; indexed by NSD ID; no test-derived tuning"},indent=2))
def read_shards(paths,split):
    rows=[]
    for path in paths:
        with tarfile.open(path) as t:
            for m in t:
                if m.name.endswith(".behav.npy"):
                    a=np.load(io.BytesIO(t.extractfile(m).read()),allow_pickle=False).reshape(-1)
                    assert np.isfinite(a[[0,5]]).all()
                    rows.append({"nsd_id":int(a[0]),"beta_row":int(a[5]),"session":int(a[1]),"trial":int(a[4]),"split":split,"shard":str(path.relative_to(ROOT)),"key":m.name.removesuffix(".behav.npy")})
    return rows
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--subject",type=int,default=1);ap.add_argument("--captions-only",action="store_true");args=ap.parse_args()
    meta=pd.read_csv(SOURCE/"nsd_stim_info_merged.csv")
    assert len(meta)==73000 and np.array_equal(meta.nsdId.to_numpy(),np.arange(73000))
    if args.captions_only:captions(meta);return
    s=f"subj{args.subject:02d}";base=ROOT/"data/nsd"
    paths=[base/"wds"/s/"train"/f"{i}.tar" for i in range(40)]
    train=read_shards(paths,"train");test=read_shards([base/"wds"/s/"new_test/0.tar"],"test")
    tset={r["nsd_id"] for r in train};vset={r["nsd_id"] for r in test}
    assert not tset.intersection(vset),"train/test image leakage"
    assert len(vset)==1000,(len(vset),"test images")
    assert len({r["beta_row"] for r in train})==len(train),"repeated train beta rows"
    with h5py.File(base/f"betas_all_{s}_fp32_renorm.hdf5","r") as f:
        shape=list(f["betas"].shape)
        assert min(r["beta_row"] for r in train+test)>=0
        assert max(r["beta_row"] for r in train+test)<shape[0]
        if args.subject==1:assert shape[1]==15724,shape
        for offset in range(0,shape[0],512):assert np.isfinite(f["betas"][offset:offset+512]).all()
    for r in train+test:r["coco_id"]=int(meta.iloc[r["nsd_id"]].cocoId)
    for name,rows in [("train",train),("test_trials",test)]:
        path=ROOT/f"manifests/{s}_{name}.csv"
        with path.open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
    ids=np.array(sorted(tset|vset),dtype=np.int64);np.save(ROOT/f"data/{s}_image_ids.npy",ids)
    counts=collections.Counter(r["nsd_id"] for r in test)
    result={"subject":args.subject,"beta_shape":shape,"train_trials":len(train),"train_unique_images":len(tset),"test_trials":len(test),"test_unique_images":len(vset),"overlap_images":0,"repetition_histogram":dict(collections.Counter(counts.values())),"train_manifest_sha256":digest(ROOT/f"manifests/{s}_train.csv"),"test_manifest_sha256":digest(ROOT/f"manifests/{s}_test_trials.csv"),"nsd_id_indexing":"zero-based","beta_row_indexing":"zero-based"}
    (ROOT/f"manifests/{s}_data_audit.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    captions(meta)
if __name__=="__main__":main()
