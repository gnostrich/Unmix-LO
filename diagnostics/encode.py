"""Encode the physics world with ViT + MiniLM keeping PER-TOKEN (pre-pool) representations.
Writes Z.npy (world state), vit.npy (n,197,768), mini.npy (n,64,384), mask.npy (n,64).
Needs world.py (scene sim + render + describe), preserved on branch archive/pre-nuke:
  git show archive/pre-nuke:virtualworld/world.py > diagnostics/world.py
Then, from diagnostics/:  python encode.py && python channel_diag.py"""
import os,sys,numpy as np,torch
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoModel, AutoTokenizer
import world as W
torch.set_num_threads(os.cpu_count() or 4)
N=320; IMEAN=np.array([0.485,0.456,0.406]);ISTD=np.array([0.229,0.224,0.225])
d=W.collect(n_rollouts=16,T=21,seed0=0); sp,sc=d["s_prev"][:N],d["s_cur"][:N]
Z=np.stack([W.scene_features(s) for s in sc]); np.save("Z.npy",Z)
n=len(sc)
@torch.no_grad()
def vit():
    m=AutoModel.from_pretrained("google/vit-base-patch16-224").eval();o=[]
    for i in range(0,n,16):
        fr=np.stack([W.render(sp[j],sc[j]) for j in range(i,min(i+16,n))])
        x=torch.tensor(((fr-IMEAN)/ISTD).transpose(0,3,1,2),dtype=torch.float32)
        o.append(m(pixel_values=x).last_hidden_state.numpy().astype(np.float32))
        print("vit",i,flush=True)
    return np.concatenate(o)
@torch.no_grad()
def mini():
    tok=AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2");m=AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").eval()
    tx=[W.describe(sc[i],sp[i]) for i in range(n)];H=[];M=[]
    for i in range(0,n,32):
        e=tok(tx[i:i+32],return_tensors="pt",padding="max_length",truncation=True,max_length=64)
        H.append(m(**e).last_hidden_state.numpy().astype(np.float32));M.append(e["attention_mask"].numpy())
    return np.concatenate(H),np.concatenate(M)
np.save("vit.npy",vit()); h,m=mini(); np.save("mini.npy",h); np.save("mask.npy",m)
print("encoded",n,"frames")
