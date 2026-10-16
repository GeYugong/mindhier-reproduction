"""Snapshot and plot completed Stage 2 training losses; no validation claims."""
import argparse,json,hashlib,sys,statistics,time
from pathlib import Path
from datetime import datetime,timezone,timedelta
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap=argparse.ArgumentParser();ap.add_argument('--qa-tools-dir',type=Path,required=True);args=ap.parse_args()
sys.path.insert(0,str(args.qa_tools_dir.resolve()))
from audit_panel_alignment import require_matplotlib_panel_alignment
ROOT=Path(__file__).resolve().parents[2];out=Path(__file__).resolve().parent
run=ROOT/'runs/s01-stage2-paper-seed42';source=out/'metrics.jsonl'
if not source.exists():
    status=json.loads((run/'status.json').read_text());raw=(run/'metrics.jsonl').read_bytes()
    lines=raw.splitlines() if raw.endswith(b'\n') else raw.splitlines()[:-1]
    lines=[line for line in lines if json.loads(line)['step']<=status['completed_steps']]
    source.write_bytes(b'\n'.join(lines)+b'\n')
    (out/'snapshot.json').write_text(json.dumps({'captured_at':time.time(),'status':status,'checkpoint':json.loads((run/'checkpoint.json').read_text())},indent=2)+'\n')
rows=[json.loads(line) for line in source.read_text().splitlines()]
assert [r['step'] for r in rows]==list(range(1,len(rows)+1))
x=np.array([r['step'] for r in rows]);y=np.array([r['loss'] for r in rows]);assert np.isfinite(y).all()
assert len(rows)>=1000
smooth=np.convolve(y,np.ones(100)/100,mode='valid')
snap=json.loads((out/'snapshot.json').read_text())
summary={'completed_steps':len(rows),'target_steps':9000,'metric':'mean training VQ token cross-entropy',
         'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'rows':len(rows),'excluded_complete_rows':snap.get('rows_after_status_boundary_excluded',0),
         'rolling_mean':'trailing 100 steps, shown from step 100; raw observations all shown',
         'first_100_mean':float(y[:100].mean()),'last_100_mean':float(y[-100:].mean()),'previous_100_mean':float(y[-200:-100].mean()),
         'previous_500_mean':float(y[-1000:-500].mean()),'last_500_mean':float(y[-500:].mean()),
         'recent_500_relative_improvement_percent':float(100*(1-y[-500:].mean()/y[-1000:-500].mean())),
         'last_step_loss':float(y[-1]),'last_step_lr':rows[-1]['lr'],'no_independent_validation':True,
         'snapshot_beijing':datetime.fromtimestamp(snap['captured_at'],timezone(timedelta(hours=8))).isoformat()}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':8,
                     'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
                     'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.linewidth':.7,'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})
fig,axes=plt.subplots(1,2,figsize=(7.2,3.9))
fig.subplots_adjust(left=.09,right=.98,bottom=.19,top=.72,wspace=.25)
blue='#24648D';gray='#B8C2CB'
for i,ax in enumerate(axes):
    ax.annotate('ab'[i],(0,1),xycoords='axes fraction',xytext=(-23,10),textcoords='offset points',fontsize=10,fontweight='bold')
    ax.set_xlabel('Completed optimizer steps');ax.set_ylabel('Training cross-entropy')
    ax.grid(axis='y',color='#E9E9E9',linewidth=.5);ax.set_axisbelow(True)
    start=0 if i==0 else max(0,len(rows)-1500)
    ax.plot(x[start:],y[start:],color=gray,linewidth=.5,label='Per-step loss')
    keep=x[99:]>=x[start]
    ax.plot(x[99:][keep],smooth[keep],color=blue,linewidth=1.5,label='Trailing 100-step mean')
    ax.set_xlim(x[start],x[-1])
for ax in axes:
    ax.axvline(5000,color='#B47736',linestyle='--',linewidth=.8)
axes[0].set_title('Full training trajectory')
axes[1].set_title('Latest 1,500 updates (zoomed)')
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper left',bbox_to_anchor=(.08,.85),ncol=2,frameon=False)
fig.suptitle('MindHier Stage 2 | Subject 1 | Seed 42',x=.09,y=.98,ha='left',fontsize=12,fontweight='bold')
fig.text(.09,.91,f'{len(rows):,} / 9,000 updates | batch 80 | checkpoint {snap["checkpoint"]["step"]:,}',fontsize=8)
fig.text(.09,.075,'Training data only. No independent validation or reconstruction-quality curve.',fontsize=8,color='#555555')
fig.text(.09,.025,'Snapshot: '+summary['snapshot_beijing'][:19].replace('T',' ')+' Beijing time | Lower is better',fontsize=7,color='#555555')
fig.canvas.draw()
require_matplotlib_panel_alignment(fig,json_out=out/'loss.alignment.json',tolerance_pt=1.5,gutter_tolerance_pt=1.5,require_panel_labels=True,strict=True)
fig.savefig(out/'stage2-loss.png',dpi=300)
fig.savefig(out/'stage2-loss.svg')
fig.savefig(out/'stage2-loss.pdf')
plt.close(fig)
print(json.dumps(summary,indent=2))
