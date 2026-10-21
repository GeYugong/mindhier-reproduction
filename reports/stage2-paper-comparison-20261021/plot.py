from pathlib import Path
import sys,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0,'runs/figure-qa')
from audit_panel_alignment import require_matplotlib_panel_alignment
out=Path(__file__).resolve().parent;rows=json.loads((out/'samples.json').read_text())
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,axes=plt.subplots(4,3,figsize=(7.2,10.7));fig.subplots_adjust(left=.08,right=.98,bottom=.08,top=.91,wspace=.06,hspace=.22)
for j,title in enumerate(['Original image','N1 | no GT selection','N4 | GT-assisted']):axes[0,j].set_title(title,fontsize=10,pad=8)
for k,row in enumerate(rows):
 for j,tag in enumerate(['gt','n1','n4']):
  ax=axes[k,j];ax.imshow(Image.open(out/'images'/f"{row['index']:04d}-{tag}.png"));ax.set_xticks([]);ax.set_yticks([])
  for spine in ax.spines.values():spine.set_visible(False)
 axes[k,0].set_ylabel(f"NSD {row['nsd_id']}",fontsize=9,labelpad=8)
fig.suptitle('MindHier | S1 | Step 8,500',x=.08,y=.975,ha='left',fontsize=15,fontweight='bold')
fig.text(.08,.945,'Fixed test positions 1, 251, 501, 751; selected before visual inspection.',fontsize=9)
fig.text(.08,.04,'N4 uses the original image only to select among four saved candidates.',fontsize=9)
fig.text(.08,.018,'All images shown without enhancement or selective cropping.',fontsize=8)
fig.canvas.draw();require_matplotlib_panel_alignment(fig,json_out=out/'alignment.json',strict=True,require_panel_labels=False)
fig.savefig(out/'recon-comparison.png',dpi=300);fig.savefig(out/'recon-comparison.pdf');fig.savefig(out/'recon-comparison.svg');plt.close(fig)
