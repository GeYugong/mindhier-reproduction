"""Evaluate saved training-reference logits and convergence after each checkpoint.

Runs outside the immutable training process. CPU only; does not restore a model,
change RNG state in the trainer, select on test data, or stop training itself.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SOURCE = '06e95b83a14b07fef51ebdb4c299b6df9914bb34'
CONFIG = 'e08b20d94e6981ee6c3abdeb8b2d2cd9bd13e82f62ca8581ae3df82ca43b5a57'
POLICY = {
    'version': 1, 'window_steps': [100, 500], 'confirmation_steps': 500,
    'consecutive_comparisons': 2, 'worsening_relative': 0.01,
    'plateau_absolute_relative': 0.001,
    'scope': 'operational training-review triggers, not validated early stopping or generalization criteria',
    'automatic_training_mutation': False,
}

def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)

def read_complete_jsonl(path):
    raw = path.read_bytes()
    lines = raw.splitlines()
    if not raw.endswith(b'\n'): lines = lines[:-1]
    return [json.loads(line) for line in lines if line.strip()]

def relative_gain(old, new):
    return (old - new) / max(abs(old), 1e-12)

def loss_block(rows):
    return {'first_step': rows[0]['step'], 'last_step': rows[-1]['step'],
            'count': len(rows), 'loss_mean': statistics.mean(r['loss'] for r in rows),
            'loss_std': statistics.pstdev(r['loss'] for r in rows),
            'lr_last': rows[-1]['lr']}

def repeated_flags(gains, prefix):
    if len(gains) < POLICY['consecutive_comparisons']: return []
    tail = gains[-POLICY['consecutive_comparisons']:]
    flags = []
    if all(g <= -POLICY['worsening_relative'] for g in tail):
        flags.append(prefix + '_worsening')
    if all(abs(g) < POLICY['plateau_absolute_relative'] for g in tail):
        flags.append(prefix + '_plateau')
    return flags

def training_review(rows, step):
    rows = [r for r in rows if r['step'] <= step]
    if [r['step'] for r in rows] != list(range(1, step + 1)):
        raise ValueError('Missing, duplicate or non-contiguous completed metrics')
    for row in rows:
        for key in ['loss', 'lr', 'grad_norm', 'seconds']:
            if not math.isfinite(row[key]): raise ValueError('Non-finite training ' + key)
    windows = {}
    for n in POLICY['window_steps']:
        if len(rows) >= 2*n:
            old, new = loss_block(rows[-2*n:-n]), loss_block(rows[-n:])
            windows[str(n)] = {'previous': old, 'latest': new,
                              'relative_improvement': relative_gain(old['loss_mean'], new['loss_mean'])}
    width = POLICY['confirmation_steps']
    blocks = [loss_block(rows[a:a+width]) for a in range(0, len(rows)-width+1, width)]
    gains = [relative_gain(a['loss_mean'], b['loss_mean']) for a,b in zip(blocks, blocks[1:])]
    return {'evaluated_through_step': step, 'windows': windows,
            'complete_500_step_blocks': blocks, 'block_relative_improvements': gains,
            'review_flags': repeated_flags(gains, 'training_loss'),
            'last_loss': rows[-1]['loss'], 'last_lr': rows[-1]['lr'],
            'last_grad_norm': rows[-1]['grad_norm'], 'all_checked_values_finite': True}

def reference_metrics(logits, target, patch_nums):
    import torch
    import torch.nn.functional as F
    expected = (1, sum(p*p for p in patch_nums), 4096)
    if tuple(logits.shape) != expected or tuple(target.shape) != expected[:2]:
        raise ValueError('Reference tensor shape mismatch')
    if not torch.isfinite(logits).all(): raise ValueError('Non-finite reference logits')
    if target.min() < 0 or target.max() >= 4096: raise ValueError('Invalid VQ target')
    scores = logits.float().reshape(-1, 4096)
    truth = target.reshape(-1)
    loss = F.cross_entropy(scores, truth, reduction='none')
    correct = (scores.argmax(1) == truth).float()
    logp = scores.log_softmax(-1)
    entropy = -(logp.exp() * logp).sum(-1)
    per_scale = []
    start = 0
    for p in patch_nums:
        stop = start + p*p
        per_scale.append({'patch_size': p, 'tokens': p*p,
                          'ce': float(loss[start:stop].mean()),
                          'top1_accuracy': float(correct[start:stop].mean())})
        start = stop
    return {'n_images': 1, 'tokens': len(truth), 'split': 'training',
            'teacher_forced': True, 'ce': float(loss.mean()),
            'top1_accuracy': float(correct.mean()),
            'mean_predictive_entropy': float(entropy.mean()), 'per_scale': per_scale,
            'limitation': 'one already-seen fixed training crop; no independent validation, free-running reconstruction or eight-metric conclusion'}

def decision(training, reference, history, step):
    # Only compare distinct, consecutive 500-step milestones for confirmation.
    milestones = {r['checkpoint']['step']: r['reference']['ce'] for r in history
                  if r['checkpoint']['step'] <= step and r['checkpoint']['step'] % 500 == 0}
    if step % 500 == 0: milestones[step] = reference['ce']
    pairs = list(sorted(milestones.items()))
    recent = pairs[-3:]
    gains = [relative_gain(a[1], b[1]) for a,b in zip(recent,recent[1:])]
    contiguous = len(recent) == 3 and all(b[0]-a[0] == 500 for a,b in zip(recent,recent[1:]))
    ref_flags = repeated_flags(gains, 'fixed_train_reference') if contiguous else []
    flags = training['review_flags'] + ref_flags
    prior = [r for r in history if r['checkpoint']['step'] < step]
    previous = max(prior, key=lambda r:r['checkpoint']['step']) if prior else None
    delta = None if previous is None else {
        'previous_step': previous['checkpoint']['step'], 'current_step': step,
        'relative_ce_improvement': relative_gain(previous['reference']['ce'],reference['ce']),
        'top1_accuracy_change': reference['top1_accuracy']-previous['reference']['top1_accuracy']}
    if training['review_flags'] and ref_flags:
        action, reason = 'review_before_continuing', 'Both training windows and the fixed probe warrant checkpoint-level review; inspect direction, LR and scale metrics before a pause decision.'
    elif flags:
        action, reason = 'review_signal_disagreement', 'A persistent training or reference signal needs review; a single probe cannot establish generalization or justify automatic early stopping.'
    elif training['windows'].get('500', {}).get('relative_improvement', 0) > 0.001:
        action, reason = 'continue', 'Recent 500-step training CE is improving; no persistent review trigger.'
    else:
        action, reason = 'observe', 'No confirmed persistent trigger; continue collecting evidence and review at the next checkpoint.'
    return {'action': action, 'reason': reason, 'review_flags': flags,
            'previous_checkpoint_comparison': delta,
            'reference_milestone_steps': [x[0] for x in recent],
            'reference_milestone_relative_improvements': gains,
            'reference_confirmation_available': contiguous,
            'auto_stop': False, 'generalization_assessed': False}

def identity(path):
    s = path.stat()
    return {'inode': s.st_ino, 'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns}

def evaluate(path, expected_step, run, output, target_doc, target, history, monitor_commit, marker=None):
    import torch
    before = identity(path)
    if marker is not None and before['bytes'] != marker['bytes']:
        raise RuntimeError('Checkpoint publication in progress (size mismatch)')
    ck = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    for key, value in [('format','fsdp_world1_cpu_flat_v1'), ('git_commit',SOURCE),
                       ('config_hash',CONFIG), ('step',expected_step), ('world_size',1), ('seed',42), ('pilot',False)]:
        if ck.get(key) != value: raise ValueError('Checkpoint mismatch: '+key)
    if ck['next_sample'] != expected_step*80: raise ValueError('Checkpoint sample position mismatch')
    reference = ck['reference_logits'].clone()
    del ck
    gc.collect()
    if identity(path) != before: raise RuntimeError('Checkpoint changed while reading; retry')
    if marker is not None and json.loads((run/'checkpoint.json').read_text()) != marker:
        raise RuntimeError('Checkpoint marker changed while reading; retry')
    logits_sha = hashlib.sha256(reference.numpy().tobytes()).hexdigest()
    review_id = f'{expected_step:06d}-{logits_sha[:16]}'
    report_path = output/'reviews'/(review_id+'.json')
    if report_path.exists():
        result=json.loads(report_path.read_text())
        if result['reference_target_sha256'] != target_doc['target_sha256']:
            raise ValueError('Reference changed for an existing review')
        return result
    train = training_review(read_complete_jsonl(run/'metrics.jsonl'), expected_step)
    ref = reference_metrics(reference, target, target_doc['patch_nums'])
    del reference
    report = {'schema_version':1, 'review_id':review_id, 'evaluated_at':time.time(),
              'run':run.name, 'monitor_commit':monitor_commit, 'policy':POLICY,
              'monitor_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'monitor_worktree_dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),
              'trigger':'retained_checkpoint_backfill' if marker is None else 'checkpoint_detected',
              'checkpoint':{'step':expected_step, 'path':str(path.relative_to(ROOT)),
                            **before, 'source_commit':SOURCE, 'config_hash':CONFIG,
                            'reference_logits_sha256':logits_sha,
                            'full_checkpoint_sha256_verified':False,
                            'evaluation_mode':'CPU mmap of saved reference logits; no model restore'},
              'reference_target_sha256':target_doc['target_sha256'],
              'reference_sample':target_doc['record'], 'training':train, 'reference':ref,
              'decision':decision(train,ref,history,expected_step)}
    atomic_json(report_path,report)
    return report

def main():
    import fcntl
    import torch
    torch.set_num_threads(1)
    ap=argparse.ArgumentParser()
    ap.add_argument('--watch',action='store_true')
    ap.add_argument('--backfill',action='store_true')
    ap.add_argument('--interval',type=int,default=20)
    args=ap.parse_args()
    if args.interval < 5: raise ValueError('Poll interval must be at least five seconds')
    run=ROOT/'runs/s01-stage2-paper-seed42'
    output=ROOT/'runs/checkpoint-review';output.mkdir(exist_ok=True)
    lock=(output/'monitor.lock').open('a')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise SystemExit('Checkpoint review already active; refusing duplicate')
    (output/'monitor.pid').write_text(str(os.getpid()))
    os.nice(10)
    target_doc=json.loads((output/'reference-target.json').read_text())
    if target_doc['source_commit'] != SOURCE or target_doc['record']['unconditional']:
        raise ValueError('Unexpected fixed training reference')
    target=torch.tensor([target_doc['targets']],dtype=torch.int64)
    if hashlib.sha256(target.numpy().tobytes()).hexdigest()!=target_doc['target_sha256']:
        raise ValueError('Reference target checksum mismatch')
    monitor_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    loaded=[json.loads(p.read_text()) for p in sorted((output/'reviews').glob('*.json'))]
    by_step={}
    for item in sorted(loaded,key=lambda r:r['evaluated_at']):by_step[item['checkpoint']['step']]=item
    history=sorted(by_step.values(),key=lambda r:r['checkpoint']['step'])
    seen=None
    state={'pid':os.getpid(),'started_at':time.time(),'monitor_commit':monitor_commit,
           'training_source_commit':SOURCE,'interval_seconds':args.interval,'mode':'CPU-only saved-logit diagnostics'}
    pending=[]
    if args.backfill:
        pending=[(p,int(p.stem.split('-')[-1]),None) for p in sorted(run.glob('checkpoint-step-*.pt'))]
    while True:
        try:
            marker=json.loads((run/'checkpoint.json').read_text())
            current=(run/'checkpoint.pt',marker['step'],marker)
            fingerprint=(marker['step'],*identity(current[0]).values())
            todo=pending + ([current] if fingerprint != seen else [])
            pending=[]
            for path,step,meta in todo:
                report=evaluate(path,step,run,output,target_doc,target,history,monitor_commit,meta)
                history=[r for r in history if r['checkpoint']['step'] != step]+[report]
                history.sort(key=lambda r:r['checkpoint']['step'])
                latest=history[-1]
                atomic_json(output/'latest.json',latest)
                index=[{'step':r['checkpoint']['step'],'review_id':r['review_id'],
                        'reference_ce':r['reference']['ce'],'reference_top1':r['reference']['top1_accuracy'],
                        'action':r['decision']['action']} for r in history]
                atomic_json(output/'index.json',index)
                print(json.dumps({'evaluated_step':step,'reference_ce':report['reference']['ce'],
                                  'decision':report['decision']['action']},ensure_ascii=False),flush=True)
            seen=fingerprint
            state.update(phase='watching' if args.watch else 'checked',last_poll_at=time.time(),
                         latest_checkpoint_step=marker['step'],latest_evaluated_step=history[-1]['checkpoint']['step'],
                         latest_decision=history[-1]['decision'],last_error=None)
            status=json.loads((run/'status.json').read_text())
            finished=status['phase']=='formal_completed' and marker['step']==9000 and history[-1]['checkpoint']['step']==9000
            if finished:state['phase']='completed'
            atomic_json(output/'monitor-status.json',state)
            if not args.watch or finished:return
        except Exception as exc:
            state.update(phase='evaluation_failed',last_poll_at=time.time(),last_error=repr(exc),traceback=traceback.format_exc())
            atomic_json(output/'monitor-status.json',state)
            print(json.dumps({'evaluation_error':repr(exc)},ensure_ascii=False),flush=True)
            if not args.watch:raise
        time.sleep(args.interval)

if __name__=='__main__': main()
