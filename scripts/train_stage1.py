#!/usr/bin/env python3
"""Train the released hierarchical fMRI encoder with auditable full-batch losses."""
import argparse,hashlib,json,math,os,random,subprocess,sys,time
from pathlib import Path
import h5py,numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src/mindhier/brainencoder"))
from model.new_model import fMRI2CLIP
from stage1_core import cached_backward
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda:f.read(8*1024*1024),b""):h.update(c)
    return h.hexdigest()
def atomic_json(path,data):
    tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(data,indent=2),encoding="utf-8");tmp.replace(path)
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",default="configs/stage1-subj01.json");ap.add_argument("--run-id",required=True);ap.add_argument("--resume",action="store_true");ap.add_argument("--steps",type=int);ap.add_argument("--limit-trials",type=int);ap.add_argument("--batch",type=int);ap.add_argument("--microbatch",type=int);ap.add_argument("--smoke",action="store_true");ap.add_argument("--stop-after",type=int);args=ap.parse_args()
    cfg=json.loads((ROOT/args.config).read_text())
    if args.batch:cfg["global_batch"]=args.batch
    if args.microbatch:cfg["microbatch"]=args.microbatch
    if (args.steps or args.limit_trials or args.batch) and not args.smoke:raise ValueError("Partial budget requires --smoke")
    cfg.update({"smoke":args.smoke,"steps_limit":args.steps,"limit_trials":args.limit_trials})
    run=ROOT/"runs"/args.run_id
    run.mkdir(parents=True,exist_ok=True)
    if (run/"latest.pt").exists() and not args.resume:raise RuntimeError("Existing run checkpoint: use --resume or new run ID")
    commit=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()
    if subprocess.check_output(["git","-C",str(ROOT),"status","--porcelain"],text=True).strip():raise RuntimeError("Commit code changes before starting a run")
    torch.set_num_threads(4);random.seed(cfg["seed"]);np.random.seed(cfg["seed"]);torch.manual_seed(cfg["seed"]);torch.cuda.manual_seed_all(cfg["seed"])
    device=torch.device(cfg["device"]);torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(.72,device)
    torch.backends.cuda.matmul.allow_tf32=True
    subject=f"subj{cfg['subject']:02d}"
    manifest=ROOT/f"manifests/{subject}_train.csv";rows=pd.read_csv(manifest)
    targets_dir=ROOT/f"data/clip_targets_{subject}"
    image_ids=np.load(targets_dir/"image_ids.npy")
    if not (targets_dir/"complete.json").exists():
        if not args.smoke:raise RuntimeError("CLIP target cache incomplete")
        completed=json.loads((targets_dir/"progress.json").read_text())["completed"]
        rows=rows[rows.nsd_id.isin(image_ids[:completed])].copy()
    if args.limit_trials:rows=rows.iloc[:args.limit_trials].copy()
    lookup={int(i):k for k,i in enumerate(image_ids)}
    subset_file=run/"train_rows.csv"
    if args.resume and subset_file.exists():rows=pd.read_csv(subset_file)
    elif args.smoke:rows.to_csv(subset_file,index=False,lineterminator="\n")
    cache_rows=np.array([lookup[int(i)] for i in rows.nsd_id],dtype=np.int64)
    image_maps=[np.load(targets_dir/f"image_layer_{layer}.npy",mmap_mode="r") for layer in [12,16,20,24]]
    text_maps=[np.load(targets_dir/f"text_caption_{k}.npy",mmap_mode="r") for k in range(3)]
    with h5py.File(ROOT/f"data/nsd/betas_all_{subject}_fp32_renorm.hdf5","r") as f:betas=torch.from_numpy(f["betas"][:])
    model=fMRI2CLIP(input_dim=betas.shape[1],d_model=768,fmri_seq_len=100,image_seq_len=257,text_seq_len=77,num_layers=4,multi_subj=False).to(device)
    parameters=sum(p.numel() for p in model.parameters())
    no_decay=["bias","LayerNorm.bias","LayerNorm.weight"]
    groups=[{"params":[p for n,p in model.named_parameters() if not any(x in n for x in no_decay)],"weight_decay":cfg["weight_decay"]},{"params":[p for n,p in model.named_parameters() if any(x in n for x in no_decay)],"weight_decay":0.}]
    optimizer=torch.optim.AdamW(groups,lr=cfg["lr"],betas=tuple(cfg["adam_betas"]),foreach=False)
    steps_per_epoch=len(rows)//cfg["global_batch"]
    assert steps_per_epoch>0
    total_steps=cfg["epochs"]*steps_per_epoch if not args.steps else args.steps
    warmup=min(cfg["warmup_epochs"]*steps_per_epoch,max(1,total_steps//10))
    def lr_factor(step):
        if step<warmup:return .01+.99*step/max(1,warmup)
        progress=min(1,(step-warmup)/max(1,total_steps-warmup-1))
        return cfg["min_lr_ratio"]+(1-cfg["min_lr_ratio"])*.5*(1+math.cos(math.pi*progress))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_factor)
    step=0;epoch_start=0;batch_start=0
    resolved={**cfg,"git_commit":commit,"run_id":args.run_id,"train_manifest_sha256":sha(manifest),"parameter_count":parameters,"train_trials_used":len(rows),"steps_per_epoch":steps_per_epoch,"total_optimizer_steps":total_steps,"warmup_updates":warmup,"command":sys.argv,"CUDA_VISIBLE_DEVICES":os.environ.get("CUDA_VISIBLE_DEVICES",""),"torch":torch.__version__,"start_time":time.time()}
    if args.resume:
        state=torch.load(run/"latest.pt",map_location="cpu",weights_only=False)
        for key in ["subject","global_batch","lr","seed","epochs","mse_scale","softclip_temperature"]:
            if state["config"][key]!=resolved[key]:raise ValueError("Resume config mismatch: "+key)
        if state["config"]["train_manifest_sha256"]!=resolved["train_manifest_sha256"]:raise ValueError("Data manifest changed")
        model.load_state_dict(state["model_state_dict"],strict=True);optimizer.load_state_dict(state["optimizer"]);scheduler.load_state_dict(state["scheduler"])
        step=state["step"];epoch_start=state["next_epoch"];batch_start=state["next_batch"]
        torch.set_rng_state(state["torch_rng"]);torch.cuda.set_rng_state(state["cuda_rng"],device);random.setstate(state["python_rng"]);np.random.set_state(state["numpy_rng"])
        del state
    atomic_json(run/"config.json",resolved)
    print(json.dumps({"event":"start",**resolved}),flush=True)
    logfile=(run/"metrics.jsonl").open("a",buffering=1)
    model.train()
    if step>=total_steps:
        print("ALREADY_COMPLETE",flush=True);return
    def save(next_epoch,next_batch,milestone=False):
        payload={"model_state_dict":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"step":step,"next_epoch":next_epoch,"next_batch":next_batch,"config":resolved,"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state(device),"python_rng":random.getstate(),"numpy_rng":np.random.get_state()}
        tmp=run/"latest.pt.partial";torch.save(payload,tmp);os.replace(tmp,run/"latest.pt")
        if milestone:
            path=run/f"epoch-{next_epoch:03d}.pt"
            if not path.exists():os.link(run/"latest.pt",path)
        atomic_json(run/"checkpoint.json",{"step":step,"next_epoch":next_epoch,"next_batch":next_batch,"time":time.time(),"bytes":(run/"latest.pt").stat().st_size})
    try:
        for epoch in range(epoch_start,cfg["epochs"] if not args.smoke else max(cfg["epochs"],total_steps//steps_per_epoch+1)):
            order=np.random.default_rng(cfg["seed"]+epoch).permutation(len(rows))
            for batch_index in range(batch_start if epoch==epoch_start else 0,steps_per_epoch):
                start=time.monotonic();chosen=order[batch_index*cfg["global_batch"]:(batch_index+1)*cfg["global_batch"]]
                x=betas[torch.from_numpy(rows.beta_row.to_numpy()[chosen].copy())]
                ii=cache_rows[chosen]
                # Match the author's per-minibatch cycling through the first three captions.
                caption_index=batch_index%3
                target_images=[torch.from_numpy(np.array(m[ii],copy=True)) for m in image_maps]
                targets=[target_images[-1],torch.from_numpy(np.array(text_maps[caption_index][ii],copy=True)),*target_images]
                optimizer.zero_grad(set_to_none=True)
                metrics=cached_backward(model,x,targets,cfg["microbatch"],device,cfg["mse_scale"],cfg["softclip_temperature"],True)
                finite=all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())
                if not finite or not math.isfinite(metrics["loss"]):raise FloatingPointError("Non-finite loss/gradient")
                optimizer.step();scheduler.step();step+=1
                torch.cuda.synchronize(device)
                info={"event":"train","step":step,"epoch":epoch+1,"batch_index":batch_index,"lr":optimizer.param_groups[0]["lr"],"elapsed_step_seconds":time.monotonic()-start,"gpu_peak_gib":torch.cuda.max_memory_allocated(device)/2**30,"time":time.time(),**metrics}
                logfile.write(json.dumps(info)+"\n");print(json.dumps(info),flush=True)
                atomic_json(run/"status.json",{"state":"running",**info})
                at_end=batch_index+1==steps_per_epoch
                if at_end or step%50==0 or step>=total_steps:
                    save(epoch+1 if at_end else epoch,0 if at_end else batch_index+1,milestone=at_end and (epoch+1)%cfg["checkpoint_epochs"]==0)
                del targets,target_images
                if args.stop_after and step>=args.stop_after and step<total_steps:
                    save(epoch+1 if at_end else epoch,0 if at_end else batch_index+1)
                    atomic_json(run/"status.json",{"state":"paused_for_resume_validation","step":step,"time":time.time()})
                    return
                if step>=total_steps:
                    atomic_json(run/"status.json",{"state":"completed","step":step,"epoch":epoch+1,"time":time.time(),"smoke":args.smoke})
                    (run/"complete.json").write_text(json.dumps({"step":step,"checkpoint":"latest.pt","git_commit":commit,"smoke":args.smoke}))
                    return
    except BaseException as e:
        atomic_json(run/"status.json",{"state":"failed","step":step,"error":repr(e),"time":time.time()})
        raise
    finally:logfile.close()
if __name__=="__main__":main()
