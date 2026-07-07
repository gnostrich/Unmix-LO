import numpy as np
exec(open('gate0.py').read().split("all_codes=")[0])   # reuse H_of, learn_library, precond_from_library, iters, ov, active_subspace
rng=np.random.default_rng(1)
# SPARSE regime: many primitives, few active per task -> routing SHOULD matter (monolithic over-damps)
Pbig=16
Ubig,_=np.linalg.qr(rng.normal(0,1,(D,Pbig))); Ubig=Ubig[:,:Pbig]
kappa=40.0
def Hb(active):
    A=np.eye(D)
    for i in active: A=A+(kappa-1)*np.outer(Ubig[:,i],Ubig[:,i])
    return A
def act_sub(H,k):
    ev,V=np.linalg.eigh(H); return V[:,-k:]
def learn_lib(tasks):
    Pi=np.zeros((D,D))
    for act in tasks:
        S=act_sub(Hb(act),len(act)); Pi+=S@S.T
    Pi/=len(tasks); ev,V=np.linalg.eigh(Pi); ev=ev[::-1]; V=V[:,::-1]
    keep=ev>0.5/len(tasks); return V[:,keep]
def precond(H,lib):
    d=np.diag(lib.T@H@lib); Pm=np.eye(D)
    for k in range(lib.shape[1]):
        if d[k]>5.0: Pm=Pm-(1-1/kappa)*np.outer(lib[:,k],lib[:,k])
    return Pm
def it(H,ws,Pm=None,tol=1e-3,maxit=6000):
    M=Pm@H if Pm is not None else H; lr=1.8/np.linalg.eigvalsh(M).max()
    w=np.zeros(D); L0=0.5*(w-ws)@H@(w-ws)
    for i in range(1,maxit+1):
        g=H@(w-ws); w=w-lr*(Pm@g if Pm is not None else g)
        if 0.5*(w-ws)@H@(w-ws)<tol*L0: return i
    return maxit

# training tasks: random pairs/triples of primitives covering all 16, separating; held-out: unseen combos
train=[]
for i in range(Pbig): train.append((i,(i+1)%Pbig))          # consecutive pairs (covers all, separating)
for i in range(0,Pbig,2): train.append((i,(i+5)%Pbig,(i+9)%Pbig))
held=[tuple(sorted(rng.choice(Pbig,size=rng.integers(2,4),replace=False))) for _ in range(30)]
held=[h for h in held if h not in train]

lib=learn_lib(train); Krec=lib.shape[1]
monoP=np.eye(D)
for k in range(Krec): monoP=monoP-(1-1/kappa)*np.outer(lib[:,k],lib[:,k])   # monolithic: damp ALL
sp_lib=[]; sp_mono=[]
for h in held:
    ws=rng.normal(0,1,D); H=Hb(h); b=it(H,ws)
    sp_lib.append(b/it(H,ws,precond(H,lib))); sp_mono.append(b/it(H,ws,monoP))
print("SPARSE regime: 16 primitives, 2-3 active per task (routing SHOULD matter)")
print(f"  library recovered = {Krec}  (true = {Pbig}; #train tasks = {len(train)})")
print(f"  recovered-vs-true subspace overlap = {ov(lib,Ubig):.3f}")
print(f"  HELD-OUT ({len(held)} unseen combos): routed-library = {np.mean(sp_lib):.2f}x   monolithic(no-routing) = {np.mean(sp_mono):.2f}x")
print(f"  routing advantage = {np.mean(sp_lib)/np.mean(sp_mono):.2f}x  (>1 => composition/routing genuinely helps)")
