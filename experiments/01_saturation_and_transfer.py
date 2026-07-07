# experiment 01 — saturation + transfer (quadratic compositional tasks). See experiments/README.md
import numpy as np, itertools
rng=np.random.default_rng(0)
D, P, kappa = 24, 4, 40.0          # dim, #primitives, ill-conditioning along active primitives
U,_=np.linalg.qr(rng.normal(0,1,(D,P))); U=U[:,:P]     # P orthonormal primitive directions
def H_of(code):                     # task Hessian = I + ill-conditioning along ACTIVE primitives (a composition)
    A=np.eye(D)
    for i in range(P):
        if code[i]: A=A+(kappa-1)*np.outer(U[:,i],U[:,i])
    return A
def active_subspace(H,k):           # what the optimizer 'sees': top-curvature directions of the task
    ev,V=np.linalg.eigh(H); return V[:,-k:]            # top-k eigvecs

# ---- LIBRARY LEARNING = average projector onto training tasks' active subspaces (the shared-core extractor) ----
def learn_library(train_codes, floor=None):
    Pi=np.zeros((D,D))
    for c in train_codes:
        k=sum(c)                                        # dims of this task's active subspace
        S=active_subspace(H_of(c),k)
        Pi+=S@S.T
    Pi/=len(train_codes)
    ev,V=np.linalg.eigh(Pi)
    ev=ev[::-1]; V=V[:,::-1]
    if floor is None: floor=0.5/len(train_codes)        # a direction must recur; natural floor
    keep=ev>floor
    return V[:,keep], ev[keep]                           # recovered library directions + usage weights

def precond_from_library(H, lib):                        # ROUTER: damp library dirs this task is ill-conditioned along
    d=np.diag(lib.T@H@lib)                                # curvature of H along each library dir
    Pm=np.eye(D)
    for k in range(lib.shape[1]):
        if d[k] > 5.0:                                    # active for this task
            Pm=Pm-(1-1/kappa)*np.outer(lib[:,k],lib[:,k])
    return Pm
def iters(H,wstar,Pm=None,tol=1e-3,maxit=4000):
    M=Pm@H if Pm is not None else H
    lr=1.8/np.linalg.eigvalsh(M).max(); w=np.zeros(D); L0=0.5*(w-wstar)@H@(w-wstar)
    for i in range(1,maxit+1):
        g=H@(w-wstar); w=w-lr*(Pm@g if Pm is not None else g)
        if 0.5*(w-wstar)@H@(w-wstar)<tol*L0: return i
    return maxit
def ov(A,B):
    s=np.linalg.svd(A.T@B,compute_uv=False); return float(np.mean(s**2))

all_codes=[c for c in itertools.product([0,1],repeat=P) if sum(c)>=1]
def run(train_codes, label):
    held=[c for c in all_codes if c not in train_codes]
    lib,w=learn_library(train_codes)
    Krec=lib.shape[1]
    rec_overlap=ov(lib,U)                                # did we recover the TRUE primitives?
    # held-out generalization: speedup on compositions never seen in training
    sp_lib=[]; sp_mono=[]
    lib_all = U*0  # monolithic baseline: damp EVERY recovered dir always (no routing)
    monoP=np.eye(D)
    for k in range(lib.shape[1]): monoP=monoP-(1-1/kappa)*np.outer(lib[:,k],lib[:,k])
    for c in held:
        ws=rng.normal(0,1,D); H=H_of(c)
        base=iters(H,ws); libr=iters(H,ws,precond_from_library(H,lib)); mono=iters(H,ws,monoP)
        sp_lib.append(base/libr); sp_mono.append(base/mono)
    print(f"[{label}]")
    print(f"   library size recovered = {Krec}   (true #primitives = {P};  #training tasks = {len(train_codes)})")
    print(f"   recovered-vs-TRUE-primitive overlap = {rec_overlap:.3f}   usage weights = {np.round(w,2)}")
    print(f"   HELD-OUT compositions ({len(held)}): routed-library speedup = {np.mean(sp_lib):.2f}x   "
          f"monolithic(no-routing) speedup = {np.mean(sp_mono):.2f}x\n")

# SEPARATING training set: singletons + varied pairs -> primitives are individually identifiable
sep_train=[(1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1),(1,1,0,0),(0,1,1,0),(1,0,1,0),(1,1,1,1)]
run(sep_train, "SEPARATING training compositions (primitives appear in varied combos)")

# NON-SEPARATING: primitives locked in fixed pairs {1,2} and {3,4} -> cannot separate individuals
nonsep_train=[(1,1,0,0),(0,0,1,1),(1,1,1,1)]
run(nonsep_train, "NON-SEPARATING training (primitives 1&2 and 3&4 always co-occur)")
