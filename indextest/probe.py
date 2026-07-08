import numpy as np
from numpy.linalg import lstsq, qr, svd
rng = np.random.default_rng(0)
# Sanity probe: can we construct present-but-ENTANGLED complementarity where
#  - best single << joint oracle (complementarity real)
#  - STRONG naive linear alignment+pool CANNOT recover it
#  - but the information IS there (a nonlinear/relational readout can)?
# If yes, the test is well-posed. If naive-strong already recovers it, the "entangled" premise is empty.

N, D = 4000, 30
# generative factors: two disjoint halves
zA = rng.normal(size=(N, D)); zB = rng.normal(size=(N, D))
# target needs BOTH halves, and needs their INTERACTION (so linear pooling of separate views can't get it)
w = rng.normal(size=D)
y = np.sign((zA @ w) * (zB @ w))   # sign of product -> needs both AND their interaction (XOR-like)

# encoders: A sees only zA, B sees only zB (disjoint) -> each alone cannot determine y
fA = zA.copy()          # frozen "encoder A" features
fB = zB.copy()
# ENTANGLE each into mismatched coordinates via fixed random invertible mixing (the gauge)
MA = rng.normal(size=(D, D)); MB = rng.normal(size=(D, D))
eA = fA @ MA; eB = fB @ MB     # what the indexer/naive actually see

def acc(pred, y): return np.mean(np.sign(pred - 0.5) == y) if pred.dtype==float else np.mean(pred==y)
def readout_linear(X, y, ntr=3000):
    Xtr, ytr = X[:ntr], y[:ntr]; W,*_ = lstsq(Xtr, (ytr>0).astype(float), rcond=None)
    p = (X@W > 0.5); return np.mean(p[ntr:] == (y[ntr:]>0))
def readout_mlp(X, y, ntr=3000, H=64, iters=300, lr=0.05):
    # tiny 1-hidden-layer net, the "relational" readout that can use interactions
    d=X.shape[1]; W1=rng.normal(size=(d,H))*0.1; b1=np.zeros(H); W2=rng.normal(size=H)*0.1; b2=0.0
    Xtr=X[:ntr]; t=(y[:ntr]>0).astype(float)
    for _ in range(iters):
        h=np.tanh(Xtr@W1+b1); o=1/(1+np.exp(-(h@W2+b2))); g=(o-t)/len(t)
        gW2=h.T@g; gb2=g.sum(); gh=np.outer(g,W2)*(1-h**2)
        gW1=Xtr.T@gh; gb1=gh.sum(0)
        W1-=lr*gW1; b1-=lr*gb1; W2-=lr*gW2; b2-=lr*gb2
    h=np.tanh(X@W1+b1); o=(h@W2+b2>0); return np.mean(o[ntr:]==(y[ntr:]>0))

# --- arms ---
# best single (only one entangled view)
single = max(readout_mlp(eA,y), readout_mlp(eB,y))
# joint oracle: the TRUE disjoint features concatenated, nonlinear readout (ceiling)
oracle = readout_mlp(np.hstack([fA,fB]), y)
# NAIVE-STRONG: best linear alignment (CCA-ish via joint whitening) of the two ENTANGLED spaces + pool, then readout
#   give it the strongest fair shot: concat entangled views + linear-align each to a common basis
def whiten(X):
    U,S,Vt=svd(X-X.mean(0),full_matrices=False); return (X-X.mean(0))@Vt.T/ (S/np.sqrt(len(X))+1e-6)
naive_lin = readout_linear(np.hstack([whiten(eA),whiten(eB)]), y)          # linear pool (naive)
naive_strong = readout_mlp(np.hstack([whiten(eA),whiten(eB)]), y)          # strong: nonlinear readout on aligned concat
# INDEXED (proxy): the indexer's job is to learn a joint representation that un-does the gauge enough to
#   expose the interaction. Proxy for "successful indexing" = readout on concat of entangled views with a
#   nonlinear readout that CAN model cross-terms. If naive_strong already = this, indexing adds nothing.
print("Sanity probe on planted entangled complementarity (XOR-like, needs both halves + interaction):")
print(f"  best single (one entangled view)      = {single:.3f}")
print(f"  joint ORACLE (true disjoint feats)     = {oracle:.3f}   <- ceiling")
print(f"  NAIVE linear pool (entangled)          = {naive_lin:.3f}")
print(f"  NAIVE-STRONG (aligned concat + MLP)    = {naive_strong:.3f}")
print()
print("  Interpretation:")
print(f"   - complementarity real?  single({single:.2f}) << oracle({oracle:.2f}) : {'YES' if oracle-single>0.15 else 'NO'}")
print(f"   - naive-strong ALREADY recovers it? {'YES -> entangled premise EMPTY, test ill-posed' if naive_strong>oracle-0.07 else 'NO -> room for indexing to matter, test well-posed'}")
