#!/usr/bin/env python3
"""Compact read-only experiment status for the hourly monitor."""
import json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
out={"checked_at":time.time(),"server_commit":subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),"git_status":subprocess.check_output(["git","-C",str(ROOT),"status","--short"],text=True).splitlines()}
for name,rel in {"pipeline":"runs/pipeline/state.json","cache":"data/clip_targets_subj01/progress.json","data":"manifests/subj01_data_audit.json","memory_resume":"reports/stage1-memory-resume.json","validation":"reports/stage1-validation.json"}.items():
    p=ROOT/rel
    if p.exists():out[name]=json.loads(p.read_text())
out["runs"]={}
for p in (ROOT/"runs").glob("s01-stage1-*"):
    status=p/"status.json"
    value=json.loads(status.read_text()) if status.exists() else {}
    pidpath=p/"pid"
    if pidpath.exists():
        pid=int(pidpath.read_text());value["pid"]=pid
        try:os.kill(pid,0);value["process_alive"]=True
        except ProcessLookupError:value["process_alive"]=False
    if value:out["runs"][p.name]=value
out["downloads_complete"]=(ROOT/"runs/bootstrap/downloads.complete").exists()
out["cache_complete"]=(ROOT/"data/clip_targets_subj01/complete.json").exists()
out["gpu"]=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.total,memory.used,utilization.gpu","--format=csv,noheader"],text=True).splitlines()
stat=os.statvfs(ROOT);out["disk_free_gib"]=stat.f_bavail*stat.f_frsize/2**30
print(json.dumps(out,ensure_ascii=False,indent=2))
