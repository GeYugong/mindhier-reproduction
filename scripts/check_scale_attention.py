"""Compare full-sequence cross-attention with cached scale-wise execution."""
import sys,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/mindhier/MindHier'))
from models.basic_switti import CrossAttention
torch.set_num_threads(4);torch.manual_seed(42)
m=CrossAttention(embed_dim=16,context_dim=12,num_heads=4,qk_norm=True).eval()
lengths=[n*n for n in (1,2,3,4,6,9,13,18,24,32)]
errors=[]
with torch.no_grad():
    for batch,groups in ((2,4),(1,4),(1,2)):
        x=torch.randn(batch,sum(lengths),16);context=torch.randn(batch,groups*3,12)
        m.kv_caching(False);whole=m(x,context,groups)
        m.kv_caching(True);parts=[];offset=0
        for length in lengths:
            parts.append(m(x[:,offset:offset+length],context,groups));offset+=length
        split=torch.cat(parts,1);error=(whole-split).abs().max().item();errors.append(error)
        assert torch.allclose(whole,split,atol=1e-6,rtol=1e-5),error
        m.kv_caching(False);assert m.query_offset==0
result={'passed':True,'max_absolute_errors':errors,'checks':['all ten scales match full attention','cache reset across batches','mask rebuild across context group counts']}
print(json.dumps(result));(ROOT/'reports/stage2-pilot-final/scale-attention-check.json').write_text(json.dumps(result,indent=2))
