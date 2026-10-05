#!/usr/bin/env python3
"""Download pinned public assets with resumable transfers and content verification."""
import argparse,concurrent.futures,hashlib,json,os,time,threading
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1]
LOCK=threading.Lock()
def emit(**kw):
    with LOCK: print(json.dumps({"time":time.time(),**kw}),flush=True)
def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""):h.update(chunk)
    return h.hexdigest()
def fetch(item):
    dst=ROOT/item["destination"]
    assert dst.resolve().is_relative_to(ROOT)
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists() and dst.stat().st_size==item["size"]:
        sha=digest(dst)
        if not item.get("sha256") or sha==item["sha256"]:return {**item,"actual_sha256":sha}
        raise RuntimeError("Existing file checksum mismatch: "+str(dst))
    part=dst.with_name(dst.name+".partial")
    prefix="datasets/" if item["kind"]=="dataset" else ""
    url="https://hf-mirror.com/"+prefix+item["repo"]+"/resolve/"+item["revision"]+"/"+item["filename"]
    for attempt in range(12):
        try:
            offset=part.stat().st_size if part.exists() else 0
            headers={"Range":f"bytes={offset}-"} if offset else {}
            with requests.get(url,headers=headers,stream=True,timeout=(15,90)) as r:
                r.raise_for_status()
                if offset and r.status_code!=206:offset=0
                emit(event="download",file=item["destination"],offset=offset,total=item["size"])
                with part.open("ab" if offset else "wb") as f:
                    for chunk in r.iter_content(4*1024*1024):
                        if chunk:f.write(chunk)
            if part.stat().st_size!=item["size"]:raise RuntimeError("size mismatch")
            sha=digest(part)
            if item.get("sha256") and sha!=item["sha256"]:raise RuntimeError("SHA-256 mismatch")
            if not item.get("sha256") and item.get("git_blob"):
                blob=hashlib.sha1(b"blob "+str(part.stat().st_size).encode()+b"\0"+part.read_bytes()).hexdigest()
                if blob!=item["git_blob"]:raise RuntimeError("Git blob mismatch")
            part.replace(dst)
            result={**item,"actual_sha256":sha}
            verified=ROOT/"runs/bootstrap/verified"
            verified.mkdir(exist_ok=True)
            (verified/(hashlib.sha1(item["destination"].encode()).hexdigest()+".json")).write_text(json.dumps(result,indent=2))
            emit(event="verified",file=item["destination"],sha256=sha)
            return result
        except Exception as e:
            emit(event="retry",file=item["destination"],attempt=attempt+1,error=str(e).split("?")[0][:160])
            if attempt==11:raise
            time.sleep(min(5*(attempt+1),30))
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--workers",type=int,default=3);args=ap.parse_args()
    items=json.loads((ROOT/"manifests/downloads.json").read_text())
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        results=list(pool.map(fetch,items))
    (ROOT/"manifests/assets-verified.json").write_text(json.dumps(results,indent=2))
    (ROOT/"runs/bootstrap/downloads.complete").write_text("success\n")
    emit(event="all_complete",count=len(results))
if __name__=="__main__":main()
