"""
read_instrument.py — fresh, PCA-conditioned read-out for the trace (conditional expectation E).

The trace is read from I/O by presenting BOTH orders of a pair and splitting the output:
  commuting(x,y) = 1/2 ( o(x,y) + o(y,x) )   -- the canonical / order-free "average" (part a)
  residue(x,y)   = 1/2 ( o(x,y) - o(y,x) )   -- the non-commuting order-dependence      (part b)

Metrics (PCA-conditioned, held-out — the swirl lesson: else the read fabricates at small N):
  fit_read(features, target)  -> train R^2 and HELD-OUT R^2 of a linear read of `target` from `features`.
  eff_rank(M)                 -> participation ratio (sum s)^2 / sum(s^2) of the singular values of M.
No reuse of any prior instrument. numpy only.
"""
import numpy as np

ATOMIC_EFFRANK_FRAC = 0.4     # residue is "low-rank" if eff_rank < 0.4 * m
ATOMIC_READ_R2 = 0.3          # residue is "readable" (generalizes) if held-out R^2 >= 0.3
FUNGIBLE_REL_NORM = 0.05      # residue is "zero" (fungible) if ||residue|| / ||commuting|| < this


def _r2(pred, target):
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean(0)) ** 2).sum()
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def _pca_fit_transform(X, tr, k):
    """PCA-condition (fit on TRAIN only) to top-k so the linear read is well-posed (n_train >> features)."""
    k = int(min(k, X.shape[1], len(tr) - 1))
    mu = X[tr].mean(0)
    Xc = X - mu
    _, _, Vt = np.linalg.svd(Xc[tr], full_matrices=False)
    return Xc @ Vt[:k].T


def fit_read(features, target, seed=0, train_frac=0.6, pca_k=64, lam=1.0):
    """Linear read of `target` from `features`, PCA-conditioned, with a held-out split. Returns train and
    held-out R^2 — the readability of `target` as a coherent function of `features`."""
    X = np.asarray(features, float); Y = np.asarray(target, float)
    n = len(X); rng = np.random.default_rng(seed)
    idx = rng.permutation(n); ntr = int(train_frac * n); tr, te = idx[:ntr], idx[ntr:]
    Xc = _pca_fit_transform(X, tr, pca_k)
    Phi = np.concatenate([Xc, np.ones((n, 1))], axis=1)
    W = np.linalg.solve(Phi[tr].T @ Phi[tr] + lam * np.eye(Phi.shape[1]), Phi[tr].T @ Y[tr])
    return _r2(Phi[tr] @ W, Y[tr]), _r2(Phi[te] @ W, Y[te])


def eff_rank(M):
    """Participation ratio of the singular values of M (1 = one direction; full = spread / noise)."""
    s = np.linalg.svd(M - M.mean(0), compute_uv=False)
    s = s[s > 0]
    return float((s.sum() ** 2) / (s ** 2).sum()) if s.size else 0.0


def classify(residue_rel_norm, effrank_residue, readability, m):
    """The atomic dial on the non-commuting residue (part b): zero=fungible, low-rank+readable=atomic,
    else noise."""
    if residue_rel_norm < FUNGIBLE_REL_NORM:
        return "FUNGIBLE"
    if effrank_residue < ATOMIC_EFFRANK_FRAC * m and readability >= ATOMIC_READ_R2:
        return "ATOMIC"
    return "NOISE"
