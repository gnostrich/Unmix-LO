import numpy as np
from numpy.linalg import lstsq, svd
rng = np.random.default_rng(0)
# Does a TASK-AWARE redundancy-penalized aggregator recover synergy that the BLIND indexer couldn't?
# Recreate the exact regime that killed the blind version, then give the aggregator the task signal
# and the redundancy penalty, and see if it now beats best-single AND strong-naive.

N, D = 6000, 20
zA = rng.normal(size=(N,D)); zB = rng.normal(size=(N,D))
w = rng.normal(size=D)
# synergy target: needs BOTH views' interaction (the hard case). Use a learnable-but-cross form.
y = ((zA@w) * (zB@w) > 0).astype(float)   # needs cross-view product -> genuine synergy
MA = rng.normal(size=(D,D)); MB = rng.normal(size=(D,D))   # gauge entanglement
eA = zA@MA; eB = zB@MB

def mlp(X, y, ntr=4500, H=128, iters=600, lr=0.1, seed=0):
    r=np.random.default_rng(seed); d=X.shape[1]
    W1=r.normal(size=(d,H))*0.1; b1=np.zeros(H); W2=r.normal(size=H)*0.1; b2=0.0
    Xtr=X[:ntr]; t=y[:ntr]
    for _ in range(iters):
        h=np.tanh(Xtr@W1+b1); o=1/(1+np.exp(-(h@W2+b2))); g=(o-t)/len(t)
        gW2=h.T@g; gb2=g.sum(); gh=np.outer(g,W2)*(1-h**2); gW1=Xtr.T@gh; gb1=gh.sum(0)
        W1-=lr*gW1;b1-=lr*gb1;W2-=lr*gW2;b2-=lr*gb2
    h=np.tanh(X@W1+b1); p=(1/(1+np.exp(-(h@W2+b2)))>0.5)
    return np.mean(p[ntr:]==(y[ntr:]>0.5))

def whiten(X):
    U,S,Vt=svd(X-X.mean(0),full_matrices=False); return (X-X.mean(0))@Vt.T/(S/np.sqrt(len(X))+1e-6)

wA, wB = whiten(eA), whiten(eB)
single = max(mlp(wA,y), mlp(wB,y))
oracle = mlp(np.hstack([zA,zB]), y)                       # true feats, ceiling
naive  = mlp(np.hstack([wA,wB]), y)                       # strong naive: aligned concat + MLP

# BLIND indexer: whitened views + ALL cross products, blind to y (the version that FAILED)
cross_all = (wA[:,:,None]*wB[:,None,:]).reshape(N,-1)      # all D*D cross terms
blind = mlp(np.hstack([wA,wB,cross_all]), y)

# TASK-AWARE aggregator: SELECT cross-terms by task relevance (redundancy-penalized proxy):
#   rank cross-terms by correlation with residual-of-single (info neither single has), keep top-k
res = y[:4500] - (y[:4500].mean())
corr = np.abs(cross_all[:4500].T @ (res - res.mean())) / (np.linalg.norm(cross_all[:4500],axis=0)+1e-9)
topk = np.argsort(corr)[-15:]                              # keep only 15 task-relevant cross-terms
taskaware = mlp(np.hstack([wA,wB,cross_all[:,topk]]), y)

print("Does TASK-AWARE selection recover synergy the BLIND indexer lost?")
print(f"  best-single            = {single:.3f}")
print(f"  joint ORACLE (ceiling) = {oracle:.3f}")
print(f"  strong-naive (concat)  = {naive:.3f}")
print(f"  BLIND indexer (all X)  = {blind:.3f}   <- the version that failed (feature bloat)")
print(f"  TASK-AWARE (top-15 X)  = {taskaware:.3f}   <- redundancy-penalized selection")
print()
print(f"  synergy real?         single({single:.2f}) << oracle({oracle:.2f}): {'YES' if oracle-single>0.15 else 'NO'}")
print(f"  task-aware > naive?   {taskaware:.3f} vs {naive:.3f}: {'YES ->recovers synergy' if taskaware>naive+0.03 else 'NO'}")
print(f"  task-aware > blind?   {taskaware:.3f} vs {blind:.3f}: {'YES ->selection beats bloat' if taskaware>blind+0.03 else 'NO'}")
