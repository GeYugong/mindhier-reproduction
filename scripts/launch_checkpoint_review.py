"""Start one detached checkpoint reviewer; never start or stop the trainer."""
import fcntl,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    out=ROOT/'runs/checkpoint-review';out.mkdir(exist_ok=True)
    lock=(out/'launcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    try:
        pid=int((out/'monitor.pid').read_text())
        proc=Path('/proc')/str(pid)
        args=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        if 'scripts/review_stage2_checkpoints.py --watch' in args and (proc/'cwd').resolve()==ROOT:
            print(json.dumps({'status':'already_running','pid':pid}));return
    except (OSError,ValueError):pass
    if not (out/'reference-target.json').exists():raise RuntimeError('Prepare the fixed reference target before launching')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    with (out/'monitor.log').open('ab',buffering=0) as log:
        child=subprocess.Popen([str(ROOT/'.envs/mindhier/bin/python'),'-u','scripts/review_stage2_checkpoints.py','--watch','--interval','20'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    result={'pid':child.pid,'started_at':time.time(),'command':'scripts/review_stage2_checkpoints.py --watch --interval 20','cpu_only':True}
    (out/'launcher.json').write_text(json.dumps(result,indent=2)+'\n')
    for _ in range(15):
        if child.poll() is not None:raise RuntimeError('Reviewer exited; inspect monitor.log')
        try:
            state=json.loads((out/'monitor-status.json').read_text())
            if state['pid']==child.pid:
                if state['phase']=='evaluation_failed':raise RuntimeError(state['last_error'])
                result['phase']=state['phase'];print(json.dumps(result,indent=2));return
        except (OSError,json.JSONDecodeError):pass
        time.sleep(1)
    raise RuntimeError('Reviewer startup not confirmed; inspect PID/status before any retry')
if __name__=='__main__':main()
