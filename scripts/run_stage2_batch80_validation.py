#!/usr/bin/env python3
"""Run one batch80 validation update then verify restoration in a fresh process."""
import subprocess,os,json,time,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
run=ROOT/'runs/stage2-batch80-resume';run.mkdir(exist_ok=True)
lock=(run/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
(run/'supervisor.pid').write_text(str(os.getpid()))
python=ROOT/'.envs/mindhier/bin/python'
with (run/'stdout.log').open('a',buffering=1) as log:
    for args in [[],['--resume']]:
        code=subprocess.call([str(python),str(ROOT/'scripts/validate_stage2_batch80.py'),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        with (run/'exit-codes.jsonl').open('a') as f:f.write(json.dumps({'args':args,'exit_code':code,'time':time.time()})+'\n')
        if code:raise SystemExit(code)
