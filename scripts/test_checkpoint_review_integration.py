"""Isolated publication and duplicate-lock test; never touches a real run."""
import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]

def main():
    torch.set_num_threads(1)
    fixture=ROOT/'runs'/('checkpoint-review-fixture-'+str(time.time_ns()))
    (fixture/'scripts').mkdir(parents=True)
    shutil.copy2(ROOT/'scripts/review_stage2_checkpoints.py',fixture/'scripts/review_stage2_checkpoints.py')
    run=fixture/'runs/s01-stage2-paper-seed42';run.mkdir(parents=True)
    out=fixture/'runs/checkpoint-review';out.mkdir(parents=True)
    targets=torch.zeros((1,2240),dtype=torch.int64)
    doc={'source_commit':'06e95b83a14b07fef51ebdb4c299b6df9914bb34','record':{'unconditional':False,'synthetic_fixture':True},'targets':targets.tolist()[0],
         'target_sha256':hashlib.sha256(targets.numpy().tobytes()).hexdigest(),'patch_nums':[1,2,3,4,6,9,13,18,24,32]}
    (out/'reference-target.json').write_text(json.dumps(doc))
    logits=torch.zeros(1,2240,4096)
    def publish(step):
        payload={'format':'fsdp_world1_cpu_flat_v1','git_commit':doc['source_commit'],'config_hash':'e08b20d94e6981ee6c3abdeb8b2d2cd9bd13e82f62ca8581ae3df82ca43b5a57','step':step,'world_size':1,'seed':42,'pilot':False,'next_sample':step*80,'reference_logits':logits,'synthetic_fixture':True}
        torch.save(payload,run/'checkpoint.partial');os.replace(run/'checkpoint.partial',run/'checkpoint.pt')
        rows=[{'step':i,'loss':8.-i*.001,'lr':1e-4,'grad_norm':1.,'seconds':1.} for i in range(1,step+1)]
        (run/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        (run/'checkpoint.json').write_text(json.dumps({'step':step,'bytes':(run/'checkpoint.pt').stat().st_size,'config_hash':payload['config_hash']}))
        (run/'status.json').write_text(json.dumps({'phase':'formal_training','completed_steps':step}))
    def wait_step(step,proc):
        deadline=time.time()+25
        while time.time()<deadline:
            if proc.poll() is not None:raise RuntimeError('Fixture monitor exited unexpectedly')
            p=out/'monitor-status.json'
            if p.exists():
                s=json.loads(p.read_text())
                if s.get('latest_evaluated_step')==step and s['phase']=='watching':return s
            time.sleep(.2)
        raise RuntimeError('Fixture event not observed')
    publish(100)
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
    cmd=[str(ROOT/'.envs/mindhier/bin/python'),'-u',str(fixture/'scripts/review_stage2_checkpoints.py'),'--watch','--interval','5']
    with (fixture/'stdout.log').open('wb') as log:
        proc=subprocess.Popen(cmd,cwd=fixture,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            first=wait_step(100,proc)
            publish(200);second=wait_step(200,proc)
            duplicate=subprocess.run(cmd,cwd=fixture,env=env,capture_output=True,text=True,timeout=20)
            assert duplicate.returncode != 0 and 'refusing duplicate' in duplicate.stderr
            time.sleep(5.5)
            reports=list((out/'reviews').glob('*.json'))
            assert len(reports)==2
            result={'passed':True,'synthetic_fixture':str(fixture.relative_to(ROOT)),
                    'detected_steps':[first['latest_evaluated_step'],second['latest_evaluated_step']],
                    'duplicate_launch_rejected':True,'same_checkpoint_not_duplicated':True,
                    'reports':len(reports),'real_training_untouched':True}
            p=ROOT/'reports/checkpoint-review-20261014/integration-test.json'
            p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
        finally:
            # Only the exact fixture child created above is terminated.
            proc.terminate();proc.wait(timeout=15)
if __name__=='__main__':main()
