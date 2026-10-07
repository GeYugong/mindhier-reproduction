"""Explicit FP32 CPU gradient averaging independent of FSDP accumulation semantics."""
import torch
class CPUGradientAccumulator:
    def __init__(self, parameters, expected_samples):
        self.parameters=list(parameters);self.expected_samples=expected_samples;self.samples=0;self.buffers=[None]*len(self.parameters)
    @torch.no_grad()
    def add(self, samples=1):
        if self.samples+samples>self.expected_samples:raise ValueError('Too many accumulated samples')
        for i,p in enumerate(self.parameters):
            if p.grad is None:continue
            g=p.grad.detach()
            if g.device.type!='cpu' or g.dtype!=torch.float32:raise RuntimeError('Expected offloaded FP32 gradient')
            if not torch.isfinite(g).all():raise FloatingPointError('Nonfinite gradient')
            if self.buffers[i] is None:self.buffers[i]=g.clone().mul_(samples/self.expected_samples)
            else:self.buffers[i].add_(g,alpha=samples/self.expected_samples)
            p.grad=None
        self.samples+=samples
    def apply(self):
        if self.samples!=self.expected_samples:raise RuntimeError('Incomplete effective batch')
        for p,g in zip(self.parameters,self.buffers):p.grad=g
    def release(self):self.buffers=[]

def test():
    import copy,json
    torch.manual_seed(42)
    ref=torch.nn.Sequential(torch.nn.Linear(7,13),torch.nn.GELU(),torch.nn.Linear(13,5))
    micro=copy.deepcopy(ref);x=torch.randn(80,7);y=torch.randn(80,5)
    torch.nn.functional.mse_loss(ref(x),y).backward()
    acc=CPUGradientAccumulator(micro.parameters(),80)
    for i in range(80):
        micro.zero_grad(set_to_none=True);torch.nn.functional.mse_loss(micro(x[i:i+1]),y[i:i+1]).backward();acc.add()
    acc.apply();error=max((a.grad-b.grad).abs().max().item() for a,b in zip(ref.parameters(),micro.parameters()))
    assert error<1e-6,error
    a=torch.optim.AdamW(ref.parameters(),lr=1e-4,betas=(.9,.95));b=torch.optim.AdamW(micro.parameters(),lr=1e-4,betas=(.9,.95));a.step();b.step()
    update_error=max((a-b).abs().max().item() for a,b in zip(ref.parameters(),micro.parameters()))
    assert update_error<1e-6,update_error
    return {'samples':80,'gradient_max_absolute_error':error,'adam_update_max_absolute_error':update_error,'passed':True,'scope':'FP32 CPU accumulator versus full batch; does not validate full generator BF16 numerics'}
if __name__=='__main__':
    import json
    print(json.dumps(test()))
