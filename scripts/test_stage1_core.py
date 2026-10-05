"""Compare cached and retained-graph gradients with full cross-sample losses."""
import copy,json,torch
from stage1_core import cached_backward,flatten_outputs,loss_for_branch
class Toy(torch.nn.Module):
    def __init__(self,dropout):
        super().__init__();self.shared=torch.nn.Sequential(torch.nn.Linear(5,12),torch.nn.GELU(),torch.nn.Dropout(dropout));self.heads=torch.nn.ModuleList([torch.nn.Linear(12,7) for _ in range(6)])
    def forward(self,x):
        h=self.shared(x);o=[f(h) for f in self.heads];return o[0],o[1],o[2:]
def main():
    torch.set_num_threads(2)
    for dropout in [0.,.2]:
        torch.manual_seed(10);a=Toy(dropout);b=copy.deepcopy(a);x=torch.randn(8,5);targets=[torch.randn(8,7) for _ in range(6)]
        torch.manual_seed(22)
        # With dropout, preserve the same per-microbatch RNG draw sequence.
        parts=[flatten_outputs(a(chunk)) for chunk in x.split(2)] if dropout else [flatten_outputs(a(x))]
        out=[torch.cat([p[k] for p in parts]) for k in range(6)]
        loss=sum(loss_for_branch(p,t,k,2.,.2)[0] for k,(p,t) in enumerate(zip(out,targets)));loss.backward()
        torch.manual_seed(22);m=cached_backward(b,x,targets,2,torch.device("cpu"),mse_scale=2.,temp=.2,amp=False)
        errors=[(p.grad-q.grad).abs().max().item() for p,q in zip(a.parameters(),b.parameters())]
        for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=2e-5,atol=2e-6)
        assert abs(loss.item()-m["loss"])<2e-5
        print(json.dumps({"dropout":dropout,"max_gradient_absolute_error":max(errors),"loss":m["loss"],"passed":True}))
if __name__=="__main__":main()
