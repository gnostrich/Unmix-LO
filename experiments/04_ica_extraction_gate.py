# experiment 04 — ICA extraction gate (conditional positive). See experiments/README.md
import numpy as np
from sklearn.decomposition import FastICA
from scipy.optimize import linear_sum_assignment
rng=np.random.default_rng(3)
D, P, kappa = 24, 16, 40.0
U,_=np.linalg.qr(rng.normal(0,1,(D,P))); U=U[:,:P]
def Hb(active):
    A=np.eye(D)
    for i in active: A=A+(kappa-1)*np.outer(U[:,i],U[:,i])
    return A
def ica_data(tasks, M=300, lock=False):
    X=[]
    for act in tasks:
        for _ in range(M):
            s=rng.laplace(size=len(act))
            if lock: s[:]=s[0]                      # CORRELATED loadings -> true identifiability failure
            v=sum(s[j]*U[:,act[j]] for j in range(len(act)))+rng.normal(0,0.02,D)
            X.append(v)
    return np.array(X)
def ica_lib(tasks, lock=False):
    X=ica_data(tasks, lock=lock)
    ica=FastICA(n_components=P, whiten='unit-variance', max_iter=2000, random_state=0)
    ica.fit(X); A=ica.mixing_
    return A/(np.linalg.norm(A,axis=0,keepdims=True)+1e-12)
def indiv(lib):
    C=np.abs(U.T@lib); r,c=linear_sum_assignment(-C); return C[r,c].mean()
def damp_dirs(dirs):                                 # PSD damper on the SPAN of dirs (orthonormalized)
    Q,_=np.linalg.qr(dirs); Q=Q[:,:dirs.shape[1]]
    return np.eye(D)-(1-1/kappa)*Q@Q.T
def routed(H,lib,thr=5.0):                           # damp only INDIVIDUAL active dirs (near-orthonormal subset -> PSD)
    d=np.diag(lib.T@H@lib); act=[lib[:,k] for k in range(lib.shape[1]) if d[k]>thr]
    if not act: return np.eye(D)
    return damp_dirs(np.stack(act,axis=1))
def it(H,ws,Pm=None,tol=1e-3,maxit=6000):
    M=Pm@H if Pm is not None else H
    lam=np.linalg.eigvalsh(M)
    if lam.min()<=1e-9: return maxit                 # non-PSD guard
    lr=1.8/lam.max(); w=np.zeros(D); L0=0.5*(w-ws)@H@(w-ws)
    for i in range(1,maxit+1):
        g=H@(w-ws); w=w-lr*(Pm@g); 
        if 0.5*(w-ws)@H@(w-ws)<tol*L0: return i
    return maxit
def itv(H,ws,tol=1e-3,maxit=6000):
    lr=1.8/np.linalg.eigvalsh(H).max(); w=np.zeros(D); L0=0.5*(w-ws)@H@(w-ws)
    for i in range(1,maxit+1):
        w=w-lr*(H@(w-ws))
        if 0.5*(w-ws)@H@(w-ws)<tol*L0: return i
    return maxit
def evaluate(lib, held, tag):
    monoP=damp_dirs(lib)
    sl=[]; sm=[]
    for h in held:
        ws=rng.normal(0,1,D); H=Hb(h); b=itv(H,ws)
        sl.append(b/it(H,ws,routed(H,lib))); sm.append(b/it(H,ws,monoP))
    print(f"   {tag}: individual-recovery={indiv(lib):.3f}  routed={np.mean(sl):.2f}x  "
          f"monolithic(subspace)={np.mean(sm):.2f}x  advantage={np.mean(sl)/max(np.mean(sm),1e-6):.2f}x")

train=[]
for i in range(P): train.append((i,(i+1)%P))
for i in range(0,P,2): train.append((i,(i+5)%P,(i+9)%P))
held=[tuple(sorted(rng.choice(P,size=rng.integers(2,4),replace=False))) for _ in range(30)]
held=[h for h in held if h not in train]
print("SEPARATING, sparse activation, independent loadings (ICA-identifiable):")
evaluate(ica_lib(train), held, "ICA          ")
print("\nSAME tasks but CORRELATED loadings (true identifiability failure -> ICA should break):")
evaluate(ica_lib(train, lock=True), held, "ICA (locked) ")
