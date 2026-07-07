import numpy as np, os
_here = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(_here, '_shared_neural_mlp.py')).read().split("print(")[0])
NF,TPF=6,6
fam_U=[make_family_subspace(100+f) for f in range(NF)]
teach=[[make_teacher(fam_U[f],1000*f+t) for t in range(TPF)] for f in range(NF)]
w0=flat(init_params(999))

def fisher_top(teacher,k=8,n=80,seed0=0):
    # empirical Fisher = second moment of minibatch gradients at shared point (the operator natgrad sees)
    G=[]
    for i in range(n):
        r=np.random.default_rng(seed0+i); x=r.normal(0,1,(16,D)); y=teacher_fwd(teacher,x)
        G.append(fwd_back(unflat(w0),x,y)[1])
    G=np.stack(G); G=G-G.mean(0,keepdims=True)
    # top-k right singular vectors of G = top eigenvectors of Fisher
    return np.linalg.svd(G,full_matrices=False)[2][:k]
def delta_top(teacher,k=8,n=80,seed0=0):
    # first-order: pseudo-gradient directions (the earlier test's invariant)
    Δ=[]
    for i in range(n):
        r=np.random.default_rng(seed0+i); v=w0.copy()
        for _ in range(10):
            x=r.normal(0,1,(16,D)); y=teacher_fwd(teacher,x); v=v-0.05*fwd_back(unflat(v),x,y)[1]
        Δ.append(v-w0)
    Δ=np.stack(Δ); Δ=Δ-Δ.mean(0,keepdims=True)
    return np.linalg.svd(Δ,full_matrices=False)[2][:k]
def ov(A,B):
    s=np.linalg.svd(A@B.T,compute_uv=False); return float(np.mean(s**2))

for name,fn in [("DELTA (gradient dirs)",delta_top),("OPERATOR (Fisher/curvature)",fisher_top)]:
    S=[[fn(teach[f][t],seed0=13*(f*TPF+t)) for t in range(TPF)] for f in range(NF)]
    win=np.array([ov(S[f][a],S[f][b]) for f in range(NF) for a in range(TPF) for b in range(a+1,TPF)])
    acr=np.array([ov(S[f1][a],S[f2][a]) for f1 in range(NF) for f2 in range(f1+1,NF) for a in range(TPF)])
    print(f"{name:30s}: within={win.mean():.3f} across={acr.mean():.3f} "
          f"ratio={win.mean()/acr.mean():.2f}x sep={(win.mean()-acr.mean())/(acr.std()+1e-9):.2f}sd")
