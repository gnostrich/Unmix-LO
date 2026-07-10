"""
CHANNEL-STRUCTURE DIAGNOSTIC (read-only; does NOT touch field.py).
Question: we feed the field ONE mean-pooled vector per model. Does keeping per-token/channel
structure give the field genuinely richer REAL-model input, or is it mostly redundant/noise?

Reproduces every number in DIAGNOSTIC.md from cached encodings:
  Z.npy   (n,26)      world state (medium features)
  vit.npy (n,197,768) ViT per-token, pre-pool
  mini.npy(n,64,384)  MiniLM per-token, pre-pool   mask.npy (n,64)
Encodings are produced by encode.py (needs world.py from branch archive/pre-nuke). numpy only.
"""
import numpy as np

Z0 = (lambda z: (z - z.mean(0)) / (z.std(0) + 1e-8))(np.load("Z.npy"))
VIT = np.load("vit.npy"); MINI = np.load("mini.npy"); MASK = np.load("mask.npy")
n = len(Z0)


def r2(P, T): return float(1 - ((T - P) ** 2).sum() / (((T - T.mean(0)) ** 2).sum() + 1e-9))


def pca_scores(X, k, base):                       # top-k PCA coords via Gram (n small, features wide)
    Xc = X - X[base].mean(0); G = Xc @ Xc.T; ev, U = np.linalg.eigh(G)
    idx = np.argsort(ev)[::-1][:k]; ev = np.clip(ev[idx], 1e-9, None)
    return U[:, idx] * np.sqrt(ev)


def hr2(X, Z, train, test, lam=5.0):              # held-out R^2 predicting world state Z from X
    Xs = pca_scores(X, min(64, X.shape[1], len(train) - 1), train) if X.shape[1] > 64 else X
    A = Xs[train]; W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Z[train])
    return r2(Xs[test] @ W, Z[test])


# ---------- (1) how much does mean-pooling DESTROY? within-frame vs between-frame token variance
def pooled_of(tok, mask=None):
    return tok.mean(1) if mask is None else (tok * mask[:, :, None]).sum(1) / np.clip(mask[:, :, None].sum(1), 1, None)


def destruction(name, tok, mask=None):
    n_, T_, _ = tok.shape
    valid = np.ones((n_, T_), bool) if mask is None else mask.astype(bool)
    pooled = pooled_of(tok, mask)
    between = np.var(pooled, 0).sum()
    within = np.mean([np.var(tok[i][valid[i]], 0).sum() for i in range(n_)])
    print(f"  [{name}] within/between token-var = {within / (between + 1e-9):.1f} "
          f"({'real per-token spread pooling discards' if within > between else 'tokens ~redundant'})")


# ---------- (2) per-token world-signal: contiguous (cross-rollout) split, PCA-conditioned
def per_token(name, tok, mask=None):
    n_, T_, _ = tok.shape; a = int(0.6 * n_); TR = np.arange(a); TE = np.arange(a, n_)
    pooled = pooled_of(tok, mask); full = tok.reshape(n_, -1)
    rp = hr2(pooled, Z0, TR, TE); rf = hr2(full, Z0, TR, TE)
    st = []
    for t in range(T_):
        if mask is not None and mask[:, t].sum() < n_ * 0.6: st.append(np.nan); continue
        st.append(hr2(tok[:, t, :], Z0, TR, TE))
    st = np.array(st); good = st[~np.isnan(st)]
    print(f"  [{name}] (cross-rollout split) pooled R^2={rp:.3f}  FULL-token-concat R^2={rf:.3f}  "
          f"per-token max={np.nanmax(st):.3f} median={np.nanmedian(good):.3f} frac<=0(noise/dead)={np.mean(good <= 0):.2f}")


# ---------- (3) does a SELECTED subset beat pool-all? no-leakage + random control, reshuffled iid splits
def subset_test(tok, K=(8, 20)):
    pool_all = tok.mean(1); T = tok.shape[1]
    acc = {"pool_all": [], **{f"sel{k}": [] for k in K}, **{f"rand{k}": [] for k in K}}
    for seed in range(8):
        rng = np.random.default_rng(seed)
        p = rng.permutation(n); Z = Z0[p]; V = tok[p]; P = pool_all[p]
        a, b = int(0.4 * n), int(0.7 * n)
        FIT, SEL, TEST = np.arange(a), np.arange(a, b), np.arange(b, n); trAll = np.concatenate([FIT, SEL])
        acc["pool_all"].append(hr2(P, Z, trAll, TEST))
        sig = np.array([hr2(V[:, t, :], Z, FIT, SEL) for t in range(T)])   # rank on FIT->SEL only
        order = np.argsort(sig)[::-1]
        for k in K:
            acc[f"sel{k}"].append(hr2(V[:, order[:k], :].mean(1), Z, trAll, TEST))
            acc[f"rand{k}"].append(np.mean([hr2(V[:, rng.choice(T, k, False), :].mean(1), Z, trAll, TEST) for _ in range(6)]))
    for key, v in acc.items():
        v = np.array(v); print(f"    {key:9s}: R^2 mean={v.mean():+.3f} std={v.std():.3f}")
    for k in K:
        s, rd = np.array(acc[f"sel{k}"]), np.array(acc[f"rand{k}"])
        print(f"    selected-{k} vs random-{k}: {s.mean():+.3f} vs {rd.mean():+.3f}  win-rate={np.mean(s > rd):.2f}  "
              f"(both vs pool-all {np.array(acc['pool_all']).mean():+.3f})")


print("=" * 78); print("(1) POOLING DESTRUCTION"); print("=" * 78)
destruction("ViT", VIT); destruction("MiniLM", MINI, MASK)
print("\n" + "=" * 78); print("(2) PER-TOKEN WORLD-SIGNAL (cross-rollout held-out)"); print("=" * 78)
per_token("ViT", VIT); per_token("MiniLM", MINI, MASK)
print("\n" + "=" * 78); print("(3) ViT: does a SELECTED patch subset beat pool-all? (iid, no-leak, random control)"); print("=" * 78)
subset_test(VIT)
