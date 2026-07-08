import numpy as np
rng = np.random.default_rng(0)
# ---------------------------------------------------------------------------
# CLAIM: the routing/settling memory of the inference federation has Mori-Zwanzig
# structure -> streaming (routed models) + MEMORY KERNEL (projected-out models'
# delayed influence) + noise (self-consistency residual). And the useful bet:
# that memory kernel is LOW-RANK / short-memory (atomicity) => a cheap fast index
# suffices, and its noise term predicts when to re-settle (repair criterion).
#
# TEST on a concrete coupled federation:
#  - N models, each a node with hidden state; they are COUPLED (a model's state
#    depends on others' states -> that's the "settling" dynamics).
#  - Track only a routed subset (the "resolved" variables). MZ says the skipped
#    models' effect on the tracked ones = convolution with a memory kernel.
#  - Q1: does an explicit MZ memory-kernel closure (track subset + learned kernel
#        over its own history) reproduce the true settled answer WITHOUT running
#        the full federation each time?  (=> the index amortizes settling)
#  - Q2: is that kernel LOW-RANK (few atoms enough)?  (=> cheap index)
#  - Q3: does the closure's RESIDUAL (MZ noise) flag the queries where it's wrong?
#        (=> principled repair/slow-path trigger)
# ---------------------------------------------------------------------------
N = 20          # models
d = 1           # 1-d hidden state per model (keep minimal/interpretable)
K = 6           # routed/tracked subset size
T = 40          # settling iterations (the true dynamics horizon)
Q = 400         # queries

# coupling matrix: each model's update depends on a sparse set of others (the federation graph)
A = rng.normal(0, 1, (N,N)) * (rng.random((N,N)) < 0.25)
A = 0.9 * A / (np.abs(np.linalg.eigvals(A)).max())    # contractive => settling converges

def settle(x0, steps=T):
    # true full-federation settling: x <- A x + input, to fixed point
    xs=[x0.copy()]; x=x0.copy()
    for _ in range(steps):
        x = A@x + x0*0.1
        xs.append(x.copy())
    return np.array(xs)   # (T+1, N)

tracked = np.arange(K)                 # the routed subset we resolve
proj = np.zeros((N,N)); proj[tracked,tracked]=1  # projection P onto tracked vars

# generate data: many queries (random inputs), record full settled trajectory of tracked vars
X0 = rng.normal(size=(Q,N))
traj = np.array([settle(X0[q]) for q in range(Q)])      # (Q, T+1, N)
tracked_traj = traj[:,:,tracked]                        # (Q, T+1, K)

# --- Build the MORI-ZWANZIG closure: predict tracked_var(t+1) from a MEMORY window
#     of tracked vars over the last L steps (the memory kernel), WITHOUT the other N-K models.
def build_mz(L):
    Xs, Ys = [], []
    for q in range(Q):
        for t in range(L, T):
            window = tracked_traj[q, t-L:t, :].reshape(-1)   # last L tracked states (the memory)
            Xs.append(window); Ys.append(tracked_traj[q, t, :])
    Xs=np.array(Xs); Ys=np.array(Ys)
    Kmat,*_ = np.linalg.lstsq(Xs, Ys, rcond=None)          # the MZ memory kernel (linear)
    pred = Xs@Kmat
    resid = np.linalg.norm(pred-Ys,axis=1)
    err = np.mean(resid)/ (np.mean(np.linalg.norm(Ys,axis=1))+1e-9)
    return Kmat, err, Xs, Ys, resid

print("MORI-ZWANZIG INDEX test on a coupled inference federation")
print(f"  {N} models, tracking routed subset of {K}; true settling horizon T={T}\n")

# Q1 + memory length: does a short memory kernel reproduce settled dynamics of tracked vars?
for L in [1,2,3,5,8]:
    Kmat,err,Xs,Ys,resid = build_mz(L)
    # Q2: rank of the kernel (how many atoms carry it)
    sv = np.linalg.svd(Kmat, compute_uv=False)
    eff_rank = int((sv> 0.01*sv.max()).sum())
    print(f"  memory L={L}: closure rel-error={err:.3f}   kernel eff-rank={eff_rank}/{Kmat.shape[0]}  (low => compressible index)")

# Markovian baseline (L=1, no memory) vs best memory -> does MEMORY matter (true MZ signature)?
K1,e1,*_ = build_mz(1); K5,e5,Xs,Ys,resid = build_mz(5)
print(f"\n  Markovian (no memory) err={e1:.3f}  vs  with-memory err={e5:.3f}  -> memory helps by {e1/e5:.2f}x  (>1 => genuine MZ memory)")

# Q3: does the closure RESIDUAL (MZ noise) flag the hard/wrong queries? correlate residual with true settling difficulty
# 'difficulty' = how far the tracked vars move over settling (large move = coupled-in harder to close)
diff = np.linalg.norm(tracked_traj[:,-1,:]-tracked_traj[:,0,:],axis=1)
# per-query mean residual from L=5 closure
per_q=[]
idx=0
for q in range(Q):
    n = T-5
    per_q.append(resid[idx:idx+n].mean()); idx+=n
per_q=np.array(per_q)
c = np.corrcoef(per_q, diff)[0,1]
print(f"  corr( closure residual , settling difficulty ) = {c:.3f}   (>0 => noise term flags when to re-settle)")
