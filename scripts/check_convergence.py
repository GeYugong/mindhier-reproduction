"""Read-only training-loss diagnostics; never label training data as validation."""
import argparse, json, math, statistics
from pathlib import Path
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parents[1]
def block(rows):
    return {"first_step": rows[0]["step"], "last_step": rows[-1]["step"], "count": len(rows), "loss_mean": statistics.mean(r["loss"] for r in rows), "loss_std": statistics.pstdev(r["loss"] for r in rows), "lr_last": rows[-1]["lr"]}
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output"); args = ap.parse_args()
    data = {}
    for stage, name, width in [("stage1", "s01-stage1-paper-seed42", 1300), ("stage2", "s01-stage2-paper-seed42", 500)]:
        raw = (ROOT / "runs" / name / "metrics.jsonl").read_bytes()
        # Ignore an incomplete final JSONL record while the trainer is appending.
        lines = raw.splitlines() if raw.endswith(b"\n") else raw.splitlines()[:-1]
        rows = [json.loads(line) for line in lines if line.strip()]
        if not rows: raise RuntimeError("No complete metrics: " + name)
        if [r["step"] for r in rows] != list(range(1, len(rows) + 1)): raise RuntimeError("Non-contiguous steps: " + name)
        if any(not math.isfinite(r["loss"]) for r in rows): raise RuntimeError("Non-finite loss: " + name)
        data[stage] = {"run": name, "completed_metrics": len(rows), "blocks": [block(rows[a:a+width]) for a in range(0,len(rows),width)], "recent_windows": {str(n): {"previous": block(rows[-2*n:-n]), "latest": block(rows[-n:])} for n in (100,500) if len(rows)>=2*n}}
    windows = data["stage2"]["blocks"]
    full = [b for b in windows if b["count"] == 500]
    gains = [(a["loss_mean"] - b["loss_mean"]) / a["loss_mean"] for a,b in zip(full,full[1:])]
    review = []
    if len(gains)>=2 and all(g<=-0.01 for g in gains[-2:]): review.append("training_loss_worsening_review")
    if len(gains)>=2 and all(abs(g)<0.001 for g in gains[-2:]): review.append("training_loss_plateau_review")
    result = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "scope": "training optimization only; no independent validation/generalization conclusion", "review_thresholds": "Two consecutive 500-step block comparisons: relative loss increases >=1%, or absolute relative changes <0.1%; operational review triggers, not validated early-stopping criteria", "review_flags": review, "stage2_full_block_relative_improvements": gains, **data}
    if args.output:
        out = ROOT / args.output; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__ == "__main__": main()
