"""Full-batch contrastive losses with RNG-preserving recomputation."""
import torch
import torch.nn.functional as F
def flatten_outputs(output):
    image,text,layers=output
    return (image,text,*layers)
def loss_for_branch(pred,target,index,mse_scale=200000.0,temp=0.005):
    p=F.normalize(pred.float().flatten(1),dim=-1)
    t=F.normalize(target.float().flatten(1),dim=-1)
    mse=F.mse_loss(p,t)*mse_scale
    soft=p.new_zeros(())
    if index<2:
        labels=((t@t.T)/temp).softmax(-1)
        logits=(p@t.T)/temp
        soft=(-(logits.log_softmax(-1)*labels).sum(-1).mean()-(logits.T.log_softmax(-1)*labels).sum(-1).mean())/2
    return mse+soft,mse.detach(),soft.detach()
def capture_rng(device):
    return (torch.get_rng_state(),torch.cuda.get_rng_state(device) if device.type=="cuda" else None)
def restore_rng(state,device):
    torch.set_rng_state(state[0])
    if state[1] is not None:torch.cuda.set_rng_state(state[1],device)
def cached_backward(model,inputs,targets,microbatch,device,mse_scale=200000.,temp=0.005,amp=True):
    chunks=[];rng=[]
    def autocast():return torch.autocast(device.type,dtype=torch.bfloat16,enabled=amp)
    with torch.no_grad():
        for offset in range(0,len(inputs),microbatch):
            rng.append(capture_rng(device))
            with autocast():out=flatten_outputs(model(inputs[offset:offset+microbatch].to(device)))
            chunks.append(tuple(x.detach().cpu() for x in out))
    final_rng=capture_rng(device)
    predictions=[torch.cat([o[k] for o in chunks]) for k in range(len(chunks[0]))]
    del chunks
    gradients=[];metrics={}
    for k,(prediction,target) in enumerate(zip(predictions,targets)):
        p=prediction.to(device=device,dtype=torch.float32).requires_grad_(True)
        t=target.to(device=device,dtype=torch.float32)
        loss,mse,soft=loss_for_branch(p,t,k,mse_scale,temp)
        loss.backward()
        gradients.append(p.grad.detach().cpu())
        metrics[f"mse_{k}"]=float(mse);metrics[f"softclip_{k}"]=float(soft)
        del p,t,loss
    del predictions
    for chunk_index,offset in enumerate(range(0,len(inputs),microbatch)):
        restore_rng(rng[chunk_index],device)
        with autocast():out=flatten_outputs(model(inputs[offset:offset+microbatch].to(device)))
        torch.autograd.backward(out,[g[offset:offset+microbatch].to(device=device,dtype=o.dtype) for g,o in zip(gradients,out)])
    restore_rng(final_rng,device)
    metrics["loss"]=sum(metrics.values())
    return metrics
