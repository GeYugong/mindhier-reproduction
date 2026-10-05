#!/usr/bin/env python3
"""Small real CUDA backward pass and distributed collective check."""
import json,os,torch,torch.distributed as dist
torch.set_num_threads(2)
rank=int(os.environ.get("LOCAL_RANK",0));torch.cuda.set_device(rank)
x=torch.randn(64,64,device=f"cuda:{rank}",requires_grad=True)
with torch.autocast("cuda",dtype=torch.bfloat16):loss=(x@x.T).square().mean()
loss.backward()
assert torch.isfinite(x.grad).all()
if "RANK" in os.environ:
    dist.init_process_group("nccl")
    y=torch.tensor(float(rank+1),device=f"cuda:{rank}")
    dist.all_reduce(y)
    assert y.item()==3
    dist.destroy_process_group()
else:y=torch.tensor(0)
print(json.dumps({"rank":rank,"cuda_backward_finite":True,"bf16":torch.cuda.is_bf16_supported(),"capability":torch.cuda.get_device_capability(rank),"all_reduce":y.item()}),flush=True)
