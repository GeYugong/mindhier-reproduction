from pathlib import Path
import subprocess,time,json,os,fcntl
root=Path('/data/matengyu/geyugong/mindhier-reproduction');out=root/'runs/s01-stage2-8500-restore-gate'
lock=(out/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
(out/'supervisor.pid').write_text(str(os.getpid()))
for _ in range(900):
 s=json.loads((root/'runs/user-stop-8500-20261020/state.json').read_text())
 if s['phase']=='stopped_checkpoint_preserved':break
 assert s['phase']=='stopped_hashing_checkpoint',s
 time.sleep(2)
else:raise RuntimeError('stop hash timeout')
assert not Path('/proc/20304').exists()
assert not (out/'status.json').exists()
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0'
with (out/'restore.log').open('ab',buffering=0) as log:
 p=subprocess.Popen([str(root/'.envs/mindhier/bin/python'),'-u',str(out/'run_restore_only.py')],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
 (out/'child.pid').write_text(str(p.pid));code=p.wait()
(out/'exit.json').write_text(json.dumps({'exit_code':code,'finished_at':time.time()}))
