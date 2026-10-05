#!/usr/bin/env python3
"""Cache frozen CLIP targets with the preprocessing used by the released Stage 1."""
import argparse,json,time,sys
from pathlib import Path
import h5py,numpy as np,torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPTokenizer,CLIPTextModelWithProjection,CLIPVisionModelWithProjection,CLIPImageProcessor
ROOT=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--subject",type=int,default=1);ap.add_argument("--batch",type=int,default=8);ap.add_argument("--device",default="cuda:0");args=ap.parse_args()
    torch.set_num_threads(4);device=torch.device(args.device)
    ids=np.load(ROOT/f"data/subj{args.subject:02d}_image_ids.npy")
    captions=json.loads((ROOT/"data/captions_73k.json").read_text())
    folder=ROOT/f"data/clip_targets_subj{args.subject:02d}";folder.mkdir(exist_ok=True)
    if (folder/"complete.json").exists():print("CACHE_COMPLETE",flush=True);return
    path=str(ROOT/"pretrained/clip-vit-large-patch14")
    image_model=CLIPVisionModelWithProjection.from_pretrained(path,local_files_only=True,attn_implementation="eager").eval().to(device)
    text_model=CLIPTextModelWithProjection.from_pretrained(path,local_files_only=True,attn_implementation="eager").eval().to(device)
    image_model.requires_grad_(False);text_model.requires_grad_(False)
    tokenizer=CLIPTokenizer.from_pretrained(path,local_files_only=True)
    processor=CLIPImageProcessor.from_pretrained(path,local_files_only=True)
    # The upstream Stage 1 loader uses mid_reso=1 and final_reso=512 on square NSD images.
    resize=transforms.Compose([transforms.Resize(512,interpolation=transforms.InterpolationMode.LANCZOS),transforms.ToTensor()])
    progress=folder/"progress.json"
    start=json.loads(progress.read_text())["completed"] if progress.exists() else 0
    image_maps=[];text_maps=[]
    for k in range(4):
        p=folder/f"image_layer_{[12,16,20,24][k]}.npy"
        image_maps.append(np.lib.format.open_memmap(p,mode="r+" if p.exists() else "w+",dtype=np.float32,shape=(len(ids),257,768)))
    for k in range(3):
        p=folder/f"text_caption_{k}.npy"
        text_maps.append(np.lib.format.open_memmap(p,mode="r+" if p.exists() else "w+",dtype=np.float32,shape=(len(ids),77,768)))
    np.save(folder/"image_ids.npy",ids)
    torch.backends.cuda.matmul.allow_tf32=True
    with h5py.File(ROOT/"data/nsd_stimuli.hdf5","r") as stimuli,torch.inference_mode():
        for offset in range(start,len(ids),args.batch):
            batch_ids=ids[offset:offset+args.batch]
            inputs=torch.stack([resize(Image.fromarray(stimuli["imgBrick"][int(i)])) for i in batch_ids])
            pixel=processor(images=inputs,return_tensors="pt",do_rescale=False).pixel_values.to(device)
            out=image_model(pixel_values=pixel,output_hidden_states=True)
            for k,layer in enumerate([12,16,20,24]):
                x=image_model.visual_projection(image_model.vision_model.post_layernorm(out.hidden_states[layer]))
                image_maps[k][offset:offset+len(batch_ids)]=x.float().cpu().numpy()
            for k in range(3):
                toks=tokenizer([captions[int(i)][k] for i in batch_ids],truncation=True,max_length=77,padding="max_length",return_tensors="pt").to(device)
                txt=text_model(**toks).last_hidden_state
                text_maps[k][offset:offset+len(batch_ids)]=txt.float().cpu().numpy()
            completed=offset+len(batch_ids)
            if completed%128<args.batch or completed==len(ids):
                for m in image_maps+text_maps:m.flush()
                temp=progress.with_suffix(".tmp");temp.write_text(json.dumps({"completed":completed,"total":len(ids),"time":time.time()}));temp.replace(progress)
                print(json.dumps({"event":"cache","completed":completed,"total":len(ids),"gpu_peak_gib":torch.cuda.max_memory_allocated()/2**30}),flush=True)
    result={"images":len(ids),"image_layers":[12,16,20,24],"image_shape":[257,768],"text_shape":[77,768],"dtype":"float32","compute_dtype":"float32","preprocessing":"PIL Resize(512,LANCZOS), ToTensor, CLIPImageProcessor(do_rescale=False)","caption_selection":"first 3 original COCO annotations"}
    (folder/"complete.json").write_text(json.dumps(result,indent=2))
    (ROOT/f"manifests/clip_targets_subj{args.subject:02d}.json").write_text(json.dumps(result,indent=2))
    print("CACHE_COMPLETE",flush=True)
if __name__=="__main__":main()
