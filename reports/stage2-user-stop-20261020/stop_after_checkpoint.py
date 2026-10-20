import os,signal,time,json,pathlib,shutil,hashlib,fcntl,traceback,subprocess
ROOT=pathlib.Path('/data/matengyu/geyugong/mindhier-reproduction');OUT=ROOT/'runs/user-stop-8500-20261020';RUN=ROOT/'runs/s01-stage2-paper-seed42'
def write(name,obj):
 p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2)+'\n');t.replace(p)
lock=(OUT/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
pid=int((RUN/'child.pid').read_text());proc=pathlib.Path('/proc')/str(pid)
assert os.readlink(proc/'cwd')==str(ROOT/'runs/recovery-source-06e95b8')
assert b'scripts/train_stage2_formal.py' in (proc/'cmdline').read_bytes()
assert pid==20304
import ctypes
lib=ctypes.CDLL(None,use_errno=True)
fd=lib.pidfd_open(pid,0)
if fd<0:raise OSError(ctypes.get_errno(),'pidfd_open')
def send_signal(fd,sig):
 if lib.pidfd_send_signal(fd,int(sig),None,0)!=0:raise OSError(ctypes.get_errno(),'pidfd_send_signal')
state={'watcher_pid':os.getpid(),'trainer_pid':pid,'target_step':8500,'requested_at':time.time(),'reason':'Explicit user instruction: save complete checkpoint then stop','automatic_resume_forbidden':True,'original_budget':9000,'phase':'waiting_checkpoint'}
(OUT/'watcher.pid').write_text(str(os.getpid()));write('state.json',state)
try:
 while True:
  send_signal(fd,0)
  try:c=json.loads((RUN/'checkpoint.json').read_text())
  except json.JSONDecodeError:
   time.sleep(1);continue
  if c['step']>=8500 and (RUN/f"checkpoint-step-{c['step']:04d}.pt").exists():
   assert c['step']==8500
   assert (RUN/'checkpoint.pt').stat().st_size==c['bytes']
   send_signal(fd,signal.SIGSTOP)
   for _ in range(30):
    if '\nState:\tT' in (proc/'status').read_text():break
    time.sleep(1)
   else:raise RuntimeError('Trainer suspension not confirmed')
   assert json.loads((RUN/'checkpoint.json').read_text())==c
   assert os.path.samefile(RUN/'checkpoint.pt',RUN/'checkpoint-step-8500.pt')
   with (RUN/'checkpoint.pt').open('rb') as f:os.fsync(f.fileno())
   os.link(RUN/'checkpoint.pt',OUT/'checkpoint-8500.pt')
   for name in ['status.json','checkpoint.json','metrics.jsonl','samples.jsonl','stdout-formal.log','child.pid']:shutil.copy2(RUN/name,OUT/name)
   state.update(phase='checkpoint_preserved_trainer_suspended',checkpoint=c,suspended_at=time.time());write('state.json',state)
   send_signal(fd,signal.SIGTERM);send_signal(fd,signal.SIGCONT)
   for i in range(60):
    if not proc.exists() or '\nState:\tZ' in (proc/'status').read_text():break
    time.sleep(1)
   else:raise RuntimeError('Trainer did not exit after SIGTERM; no escalation performed')
   state.update(phase='stopped_hashing_checkpoint',stopped_at=time.time());write('state.json',state)
   hashes={}
   for p in sorted(OUT.iterdir()):
    if p.name in ['checkpoint-8500.pt','status.json','checkpoint.json','metrics.jsonl','samples.jsonl','stdout-formal.log']:
     h=hashlib.sha256()
     with p.open('rb') as f:
      for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
     hashes[p.name]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
   write('sha256.json',hashes)
   state.update(phase='stopped_checkpoint_preserved',finished_at=time.time(),full_checkpoint_sha256=hashes['checkpoint-8500.pt']['sha256'],resume_validation_performed=False);write('state.json',state)
   break
  state.update(last_poll_at=time.time(),latest_checkpoint=c['step']);write('state.json',state);time.sleep(2)
except BaseException:
 state.update(phase='needs_attention',error=traceback.format_exc(),updated_at=time.time());write('state.json',state);raise
