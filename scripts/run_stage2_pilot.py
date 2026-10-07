import os,json,subprocess,time,fcntl
from pathlib import Path
r=Path(__file__).resolve().parents[1];run=r/'runs/s01-stage2-pilot-seed42';run.mkdir(exist_ok=True)
lock=(run/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
(run/'supervisor.pid').write_text(str(os.getpid()))
with (run/'stdout.log').open('a') as log:
    code=subprocess.call([str(r/'.envs/mindhier/bin/python'),str(r/'scripts/train_stage2_pilot.py')],cwd=r,stdout=log,stderr=subprocess.STDOUT)
(run/'exit-code.json').write_text(json.dumps({'exit_code':code,'time':time.time()}))
raise SystemExit(code)
