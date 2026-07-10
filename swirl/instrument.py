"""
instrument.py — the swirl instrument, fresh and self-contained (numpy only).

Given a frozen fragment's embeddings of physics states and the physics targets (next-states):
  - fit a linear (ridge) readout  embedding -> next-state  on a TRAIN split;
  - readout_R2  = R^2 of the readout on TRAIN (does the fragment predict the physics at all);
  - swirl       = the readout residual on the HELD-OUT split (the fragment's deviation from true physics);
  - eff_rank    = effective rank of the swirl matrix = participation ratio (sum s)^2 / sum(s^2) of its
                  singular values (1 = one direction, D = spread across all -> noise);
  - heldout_R2  = R^2 of the readout on the HELD-OUT split (the adjudicator: coherent function of state,
                  or coincidence).

Operational verdict, written explicitly:
    ATOMIC  iff  eff_rank < 0.4 * D  AND  heldout_R2 >= 0.3
    else NOISE
where D is the target dimension (the next-state dimension).
"""
import numpy as np

ATOMIC_EFFRANK_FRAC = 0.4      # eff_rank must be below 0.4 * D
ATOMIC_HELDOUT_R2 = 0.3        # heldout_R2 must be at least this


def _r2(pred, target):
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean(0)) ** 2).sum()
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def _ridge(X, Y, lam=1.0):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)


def _pca_fit_transform(emb, tr, k):
    """PCA-condition the embeddings (fit on TRAIN only) to top-k components so the linear readout is
    well-posed (n_train >> features) — without this, a high-dim embedding overfits and the instrument
    fabricates low-rank structure even from noise. Returns the conditioned embeddings for all rows."""
    k = int(min(k, emb.shape[1], len(tr) - 1))
    mu = emb[tr].mean(0)
    Xc = emb - mu
    _, _, Vt = np.linalg.svd(Xc[tr], full_matrices=False)
    return Xc @ Vt[:k].T


def _eff_rank(M):
    """Participation ratio of the singular values of M: (sum s)^2 / sum(s^2)."""
    s = np.linalg.svd(M, compute_uv=False)
    s = s[s > 0]
    if s.size == 0:
        return 0.0
    return float((s.sum() ** 2) / (s ** 2).sum())


def measure(embeddings, targets, seed=0, train_frac=0.6, pca_k=64):
    """embeddings: (n, d_embed) fragment features. targets: (n, D) next-states. Returns the swirl metrics
    and the pre-committed ATOMIC/NOISE verdict. Embeddings are PCA-conditioned (fit on train) to keep the
    linear readout well-posed."""
    emb = np.asarray(embeddings, dtype=np.float64)
    Y = np.asarray(targets, dtype=np.float64)
    n, D = Y.shape
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    ntr = int(train_frac * n)
    tr, te = idx[:ntr], idx[ntr:]

    emb = _pca_fit_transform(emb, tr, pca_k)               # well-posed conditioning (fit on train)
    Phi = np.concatenate([emb, np.ones((n, 1))], axis=1)   # bias column
    W = _ridge(Phi[tr], Y[tr])

    readout_r2 = _r2(Phi[tr] @ W, Y[tr])                   # TRAIN fit
    pred_te = Phi[te] @ W
    heldout_r2 = _r2(pred_te, Y[te])                       # HELD-OUT (adjudicator)
    swirl = Y[te] - pred_te                                # the swirl = held-out readout residual
    eff_rank = _eff_rank(swirl - swirl.mean(0))
    swirl_rel_norm = float(np.linalg.norm(swirl) / (np.linalg.norm(Y[te]) + 1e-12))

    atomic = (eff_rank < ATOMIC_EFFRANK_FRAC * D) and (heldout_r2 >= ATOMIC_HELDOUT_R2)
    return {"readout_R2": readout_r2, "heldout_R2": heldout_r2, "eff_rank": eff_rank,
            "D": int(D), "atomic_effrank_threshold": ATOMIC_EFFRANK_FRAC * D,
            "swirl_rel_norm": swirl_rel_norm, "atomic": bool(atomic)}
