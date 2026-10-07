#!/usr/bin/env python3
"""Compact read-only experiment status for the hourly monitor."""
import json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
out={"checked_at":time.time(),"server_commit":subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),"git_status":subprocess.check_output(["git","-C",str(ROOT),"status","--short"],text=True).splitlines()}
for name,rel in {"pipeline":"runs/pipeline/state.json","cache":"data/clip_targets_subj01/progress.json","data":"manifests/subj01_data_audit.json","memory_resume":"reports/stage1-memory-resume.json","validation":"reports/stage1-validation.json"}.items():
    p=ROOT/rel
    if p.exists():out[name]=json.loads(p.read_text())
stage2=ROOT/"runs/stage2-preflight/status.json"
if stage2.exists():
    raw=json.loads(stage2.read_text())
    out["stage2_preflight"]={k:v for k,v in raw.items() if k not in ["initialized_keys","traceback"]}
    out["stage2_preflight"]["status_mtime"]=stage2.stat().st_mtime
backward=ROOT/"runs/stage2-backward-probe-wrapped/status.json"
if backward.exists():
    raw=json.loads(backward.read_text())
    out["stage2_backward_probe"]={k:v for k,v in raw.items() if k != "traceback"}
    if "pid" in raw:
        try:os.kill(raw["pid"],0);out["stage2_backward_probe"]["process_alive"]=True
        except ProcessLookupError:out["stage2_backward_probe"]["process_alive"]=False
batch80=ROOT/"runs/stage2-batch80-cpu-checkpoint/status.json"
if batch80.exists():
    value=json.loads(batch80.read_text());value.pop("traceback",None)
    if "pid" in value:
        try:os.kill(value["pid"],0);value["process_alive"]=True
        except ProcessLookupError:value["process_alive"]=False
    out["stage2_batch80_validation"]=value
pilot=ROOT/"runs/s01-stage2-pilot-seed42/status.json"
if pilot.exists():
    value=json.loads(pilot.read_text());value.pop("traceback",None)
    try:os.kill(value["pid"],0);value["process_alive"]=True
    except ProcessLookupError:value["process_alive"]=False
    out["stage2_pilot"]=value
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
