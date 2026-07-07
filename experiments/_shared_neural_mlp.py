import numpy as np
rng = np.random.default_rng(0)

# ---------- tiny 2-layer MLP (pure numpy), manual backprop ----------
D, Hd = 20, 24          # input dim, hidden
def init_params(seed):
    r = np.random.default_rng(seed)
    W1 = r.normal(0,1/np.sqrt(D),(D,Hd)); b1=np.zeros(Hd)
    W2 = r.normal(0,1/np.sqrt(Hd),(Hd,1)); b2=np.zeros(1)
    return [W1,b1,W2,b2]
def flat(p): return np.concatenate([x.ravel() for x in p])
def unflat(v):
    i=0;out=[]
    for shp in [(D,Hd),(Hd,),(Hd,1),(1,)]:
        n=int(np.prod(shp)); out.append(v[i:i+n].reshape(shp)); i+=n
    return out
def fwd_back(p,x,y):
    W1,b1,W2,b2=p
    z=x@W1+b1; h=np.tanh(z); yh=h@W2+b2
    e=yh-y; L=np.mean(e**2)
    dyh=2*e/len(x)
    dW2=h.T@dyh; db2=dyh.sum(0)
    dh=dyh@W2.T; dz=dh*(1-h**2)
    dW1=x.T@dz; db1=dz.sum(0)
    return L, flat([dW1,db1,dW2,db2])
def teacher_fwd(p,x):
    W1,b1,W2,b2=p; return np.tanh(x@W1+b1)@W2+b2

# ---------- task families: family = shared 5-dim input subspace ----------
r_fam=5
def make_family_subspace(seed):
    A=np.random.default_rng(seed).normal(0,1,(D,r_fam))
    U,_=np.linalg.qr(A); return U           # D x r_fam orthonormal
def make_teacher(U,seed):
    # teacher depends ONLY on projection of x onto U  (family-shared relevant subspace)
    r=np.random.default_rng(seed)
    W1=U@r.normal(0,1,(r_fam,Hd))           # first layer sees only U-subspace
    b1=r.normal(0,0.3,Hd); W2=r.normal(0,1,(Hd,1)); b2=r.normal(0,0.3,1)
    return [W1,b1,W2,b2]

# ---------- pseudo-gradient generator: DiLoCo inner loop ----------
theta0 = flat(init_params(999))             # shared global init (fixed)
Hlocal, inner_lr, batch = 10, 0.05, 64
def pseudo_grad(teacher, seed):
    r=np.random.default_rng(seed)
    v=theta0.copy()
    for _ in range(Hlocal):
        x=r.normal(0,1,(batch,D)); y=teacher_fwd(teacher,x)
        _,g=fwd_back(unflat(v),x,y); v=v-inner_lr*g
    return v-theta0                          # pseudo-gradient (Delta)

def task_cloud(teacher, M, seed0):
    return np.stack([pseudo_grad(teacher, seed0+i) for i in range(M)])  # M x P

# ---------- analysis helpers ----------
def spectrum(X):                # X: n x P, return singular values of centered X
    Xc=X-X.mean(0,keepdims=True)
    return np.linalg.svd(Xc,full_matrices=False)[1]
def participation_ratio(s):     # effective rank
    p=s**2; return (p.sum()**2)/(np.sum(p**2)+1e-30)
def thresh_rank(s,floor):       # count atoms above noise floor
    return int((s>floor).sum())
def subspace(X,k):              # top-k right singular vecs (directions in param space)
    Xc=X-X.mean(0,keepdims=True)
    Vt=np.linalg.svd(Xc,full_matrices=False)[2]; return Vt[:k]
def subspace_overlap(A,B):      # mean cos^2 of principal angles, in [0,1]
    M=A@B.T; s=np.linalg.svd(M,compute_uv=False); return float(np.mean(s**2))

print("param dim P =", theta0.size)
np.save('theta0.npy',theta0)
