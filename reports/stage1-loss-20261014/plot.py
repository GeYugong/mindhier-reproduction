"""Plot all Stage 1 training observations; never infer validation performance."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--qa-tools-dir', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.qa_tools_dir.resolve()))
from audit_panel_alignment import require_matplotlib_panel_alignment

root = Path(__file__).resolve().parents[2]
out = Path(__file__).resolve().parent
source = root / 'reports/stage1-final/metrics.jsonl'
rows = [json.loads(line) for line in source.read_text().splitlines()]
assert len(rows) == 15600
assert [r['step'] for r in rows] == list(range(1, 15601))
keys = ['loss', *[f'mse_{i}' for i in range(6)], 'softclip_0', 'softclip_1', 'lr']
assert all(np.isfinite(r[k]) for r in rows for k in keys)
assert all(r[f'softclip_{i}'] == 0 for r in rows for i in range(2, 6))
assert all(abs(r['loss'] - sum(r[f'mse_{i}'] + r[f'softclip_{i}'] for i in range(6))) < 1e-5 for r in rows)
epochs = []
for ep in range(1, 301):
    group = [r for r in rows if r['epoch'] == ep]
    assert len(group) == 52
    entry = {'epoch': ep, 'updates': len(group)}
    entry.update({k: float(np.mean([r[k] for r in group])) for k in keys})
    entry['loss_sd_across_batches'] = float(np.std([r['loss'] for r in group], ddof=1))
    epochs.append(entry)
with (out / 'epoch-metrics.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(epochs[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(epochs)
summary = {
    'run_id': 's01-stage1-paper-seed42',
    'source': 'reports/stage1-final/metrics.jsonl',
    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'rows': len(rows), 'epochs': len(epochs), 'updates_per_epoch': 52,
    'seed': 42, 'split': 'training only; no independent validation curve',
    'aggregation': 'arithmetic mean of all 52 optimizer-update losses per epoch; no smoothing',
    'excluded_rows': 0, 'nonfinite_values': 0,
    'component_definition': 'six L2-normalized MSE terms, each already multiplied by 200000, plus image and text SoftCLIP terms',
    'selected_epochs': {ep: epochs[ep-1] for ep in [1, 2, 10, 25, 50, 100, 150, 200, 250, 275, 300]},
    'relative_loss_reduction_percent': {
        f'{start}_to_300': 100 * (1 - epochs[-1]['loss']/epochs[start-1]['loss'])
        for start in [1, 100, 150, 200, 250, 275]
    },
    'last_25_epoch_mean_range': [min(r['loss'] for r in epochs[-25:]), max(r['loss'] for r in epochs[-25:])],
    'interpretation_limit': 'training-objective improvement cannot establish generalization, reconstruction quality or an optimal stopping epoch',
}
(out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

# Contract: quantify diminishing training-loss improvement, then localize it to components.
# Quantitative grid: full trajectory, late detail, weighted MSE, and contrastive loss.
# Single seed: lines are epoch means, not replicate estimates; no confidence intervals.
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
    'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.7, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
    'savefig.facecolor': 'white',
})
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.9))
fig.subplots_adjust(left=0.10, right=0.98, bottom=0.11, top=0.86, wspace=0.29, hspace=0.57)
x = np.arange(1, 301)
y = np.array([r['loss'] for r in epochs])
blue = '#24648D'
orange = '#C87932'
for label, ax in zip('abcd', axes.flat):
    ax.annotate(label, xy=(0, 1), xycoords='axes fraction', xytext=(-27, 12),
                textcoords='offset points', fontsize=10, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.grid(axis='y', color='#E8E8E8', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=3)

ax = axes[0, 0]
ax.plot(x, y, color=blue, linewidth=1.6)
ax.set(title='Total training loss: full run', ylabel='Total loss', xlim=(1, 300), ylim=(0, 28))
ax.set_xticks([1, 50, 100, 150, 200, 250, 300])

ax = axes[0, 1]
ax.plot(x[199:], y[199:], color=blue, linewidth=1.3)
ax.set(title='Late training: epochs 200-300', ylabel='Total loss (zoomed)', xlim=(200, 300), ylim=(3.69, 3.92))
ax.set_xticks([200, 225, 250, 275, 300])
ax.set_yticks([3.70, 3.75, 3.80, 3.85, 3.90])
ax.grid(False)
ax.text(0.43, 0.80, 'Epoch 250 to 300\nLoss decrease: 0.86%', transform=ax.transAxes, fontsize=8)

ax = axes[1, 0]
colors = [blue, orange, '#679F96', '#8B78AA', '#B2A14E', '#6F7983']
labels = ['Image', 'Text', 'CLIP 12', 'CLIP 16', 'CLIP 20', 'CLIP 24']
styles = ['-', '-', '--', '-.', ':', '--']
for i, (color, label, style) in enumerate(zip(colors, labels, styles)):
    z = np.array([r[f'mse_{i}'] for r in epochs])
    assert np.all(z > 0)
    ax.plot(x, z, color=color, label=label, linestyle=style, linewidth=1.1)
ax.set(title='MSE components (weighted)', ylabel='Weighted MSE (log scale)', xlim=(1, 300), yscale='log', ylim=(0.5, 9))
ax.set_xticks([1, 100, 200, 300])
ax.set_yticks([0.5, 1, 2, 5])
ax.set_yticklabels(['0.5', '1', '2', '5'])
ax.minorticks_off()
ax.legend(loc='upper right', ncol=2, frameon=False, columnspacing=1.0, handlelength=1.7)

ax = axes[1, 1]
for i, (color, label) in enumerate([(blue, 'Image'), (orange, 'Text')]):
    z = np.array([r[f'softclip_{i}'] for r in epochs])
    assert np.all(z > 0)
    ax.plot(x, z, color=color, label=label, linewidth=1.2)
ax.set(title='SoftCLIP components', ylabel='SoftCLIP loss (log scale)', xlim=(1, 300), yscale='log', ylim=(0.015, 12))
ax.set_xticks([1, 100, 200, 300])
ax.set_yticks([0.02, 0.1, 1, 10])
ax.set_yticklabels(['0.02', '0.1', '1', '10'])
ax.minorticks_off()
ax.legend(loc='upper right', frameon=False)

fig.suptitle('MindHier Stage 1 | Subject 1 | Seed 42', fontsize=12, x=0.10, y=0.98, ha='left', fontweight='bold')
fig.text(0.10, 0.93, '300 epochs / 15,600 updates / 52 updates per epoch / batch 512', fontsize=8)
fig.text(0.10, 0.025, 'All curves: epoch means, without smoothing. Training data only; no validation curve.', fontsize=8, color='#555555')
fig.canvas.draw()
require_matplotlib_panel_alignment(fig, json_out=out / 'loss.alignment.json',
                                   tolerance_pt=1.5, gutter_tolerance_pt=1.5,
                                   require_panel_labels=True, strict=True)
fig.savefig(out / 'stage1-loss.png', dpi=300)
fig.savefig(out / 'stage1-loss.svg')
fig.savefig(out / 'stage1-loss.pdf')
plt.close(fig)
print(json.dumps({'selected_losses': {k: v['loss'] for k, v in summary['selected_epochs'].items()}, 'reduction_percent': summary['relative_loss_reduction_percent']}, indent=2))
