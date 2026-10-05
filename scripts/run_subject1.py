#!/usr/bin/env python3
"""Run resource validation, recoverable smoke training and the fixed Stage 1 budget."""
import fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PY=str(ROOT/".envs/mindhier/bin/python")
BASE=ROOT/"runs/pipeline";BASE.mkdir(parents=True,exist_ok=True)
def state(stage,**kw):
    value={"stage":stage,"updated_at":time.time(),**kw}
    p=BASE/"state.json";t=p.with_suffix(".tmp");t.write_text(json.dumps(value,indent=2));t.replace(p)
    print(json.dumps(value),flush=True)
def commit_generated(message):
    lines=subprocess.check_output(["git","-C",str(ROOT),"status","--porcelain"],text=True).splitlines()
    if not lines:return
    paths=[line[3:] for line in lines]
    if any(not p.startswith(("manifests/","reports/","experiments/")) for p in paths):
        raise RuntimeError("Uncommitted source/config changes: freeze code before launching a new run")
    subprocess.run(["git","-C",str(ROOT),"add","--",*paths],check=True)
    subprocess.run(["git","-C",str(ROOT),"commit","-m",message],check=True)
def wait_memory(min_free=9500):
    while True:
        raw=subprocess.check_output(["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True)
        free=[int(x) for x in raw.splitlines()]
        if free[1]>=min_free:return
        state("waiting_gpu_memory",gpu=1,free_mib=free[1],required_mib=min_free)
        time.sleep(30)
def child(run_id,extra):
    run=ROOT/"runs"/run_id;run.mkdir(exist_ok=True)
    if (run/"complete.json").exists():return
    wait_memory()
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES="1",PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="4",TOKENIZERS_PARALLELISM="false")
    cmd=[PY,"-u",str(ROOT/"scripts/train_stage1.py"),"--run-id",run_id,*extra]
    if (run/"latest.pt").exists():cmd.append("--resume")
    with (run/"stdout.log").open("ab") as log:
        p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        (run/"pid").write_text(str(p.pid))
        state("training",run_id=run_id,pid=p.pid,command=cmd)
        rc=p.wait()
    if rc:raise RuntimeError(f"{run_id} exited with {rc}; see stdout.log")
def main():
    lock=(BASE/"pipeline.lock").open("w")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (BASE/"pid").write_text(str(os.getpid()))
    try:
        cache=ROOT/"data/clip_targets_subj01"
        while not (cache/"progress.json").exists():state("waiting_initial_clip_cache");time.sleep(30)
        commit_generated("docs(data): record verified download and data manifests")
        # Three full 512-sample updates; explicitly stop and resume after the first.
        rid="s01-stage1-memory-resume-check"
        child(rid,["--smoke","--steps","3","--limit-trials","512","--batch","512","--microbatch","8","--stop-after","1"])
        child(rid,["--smoke","--steps","3","--limit-trials","512","--batch","512","--microbatch","8"])
        rows=[json.loads(x) for x in (ROOT/"runs"/rid/"metrics.jsonl").read_text().splitlines()]
        assert len(rows)==3 and rows[-1]["step"]==3
        (ROOT/"reports/stage1-memory-resume.json").write_text(json.dumps({"passed":True,"updates":3,"global_contrastive_batch":512,"resumed_after_update":1,"peak_gpu_gib":max(r["gpu_peak_gib"] for r in rows),"seconds_per_update":[r["elapsed_step_seconds"] for r in rows],"losses":[r["loss"] for r in rows]},indent=2))
        while not (cache/"complete.json").exists():
            progress=json.loads((cache/"progress.json").read_text())
            state("waiting_full_clip_cache",progress=progress)
            cache_pid=int((ROOT/"runs/bootstrap/clip-cache.pid").read_text())
            try:os.kill(cache_pid,0)
            except ProcessLookupError:raise RuntimeError("CLIP cache process stopped before completion")
            time.sleep(30)
        commit_generated("docs(validation): record full-batch memory and resume checks")
        child("s01-stage1-validation-seed42",["--smoke","--steps","100","--limit-trials","512","--batch","64","--microbatch","8"])
        smoke=ROOT/"runs/s01-stage1-validation-seed42"
        metrics=[json.loads(x) for x in (smoke/"metrics.jsonl").read_text().splitlines()]
        assert metrics[-1]["step"]==100
        (ROOT/"reports/stage1-validation.json").write_text(json.dumps({"passed":True,"updates":100,"loss_start":metrics[0]["loss"],"loss_end":metrics[-1]["loss"],"test_data_used":False,"purpose":"training, checkpoint and numerical stability validation; not a test-set result"},indent=2))
        commit_generated("docs(validation): record stage 1 smoke training")
        child("s01-stage1-paper-seed42",[])
        state("stage1_complete_stage2_preflight_required",run_id="s01-stage1-paper-seed42",next="Validate Stage 2 weight adaptation and memory before generator training; do not mark full reproduction complete.")
    except BaseException as e:
        state("failed",error=repr(e));traceback.print_exc();raise
if __name__=="__main__":main()
