"""Gate the formal run on two real updates and independent-process restore."""
import json,os,subprocess,time,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'runs/stage2-formal-supervisor';out.mkdir(exist_ok=True)
lock=(out/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
(out/'supervisor.pid').write_text(str(os.getpid()))
python=str(ROOT/'.envs/mindhier/bin/python')
phases=[('validate',['--run','s01-stage2-formal-validation-mb4','--validation','--steps','1']),('restore',['--run','s01-stage2-formal-validation-mb4','--validation','--steps','2','--resume']),('formal',['--run','s01-stage2-paper-seed42'])]
assert not (ROOT/'runs/s01-stage2-formal-validation-mb4/status.json').exists()
for phase,args in phases:
    if phase=='formal':
        v=ROOT/'runs/s01-stage2-formal-validation-mb4';s=json.loads((v/'status.json').read_text())
        assert s['phase']=='validation_completed' and s['completed_steps']==2 and s.get('resume_verified') and s['restored_step']==1
        metrics=[json.loads(x) for x in (v/'metrics.jsonl').read_text().splitlines()];assert [x['step'] for x in metrics]==[1,2]
    state={'stage':'stage2_'+phase,'formal_training':phase=='formal','supervisor_pid':os.getpid(),'updated_at':time.time()}
    (ROOT/'runs/pipeline/state.json').write_text(json.dumps(state,indent=2))
    target=ROOT/'runs'/args[1];target.mkdir(exist_ok=True)
    with (target/('stdout-'+phase+'.log')).open('ab') as log:
        proc=subprocess.Popen([python,'-u','scripts/train_stage2_formal.py',*args,'--microbatch','4'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        (target/'child.pid').write_text(str(proc.pid));code=proc.wait()
    with (out/'exit-codes.jsonl').open('a') as f:f.write(json.dumps({'phase':phase,'exit_code':code,'time':time.time()})+'\n')
    if code:
        state.update(stage='stage2_'+phase+'_failed',exit_code=code,updated_at=time.time());(ROOT/'runs/pipeline/state.json').write_text(json.dumps(state,indent=2));raise SystemExit(code)
state.update(stage='stage2_formal_completed',updated_at=time.time());(ROOT/'runs/pipeline/state.json').write_text(json.dumps(state,indent=2))
