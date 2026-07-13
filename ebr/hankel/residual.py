"""
hankel/residual.py — the load-bearing computation (§6). Per prompt, per member v of edge e:
  1. Gram from the normalized cost by double-centering (classical MDS): G_v = -1/2 · J D_v J, J = I - 11^T/n.
     Derived from D only -> gauge-invariant (never touches coordinates).
  2. Deflate the anchor-reconstructed subspace (columns of pi_{v,e}, whitened): Ghat = G - P G P.
  3. Residual moments h_k = tr(Ghat^k)/n, k=1..k_max (k_max=6), from eigenvalues of Ghat.
Across members -> a vector per prompt; across a window of W prompts -> a series; its block-Hankel spectrum vs
the measured noise floor gives the above-floor rank = the McMillan degree of the traffic's residual (the
self-sizing headline). Reuses the block-Hankel construction validated in io_trace/stream_trace.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "io_trace"))
import stream_trace as ST   # block_hankel, validated Hankel machinery

K_MAX = 6


def gram_from_D(D):
    """Coordinate-free Gram via double-centering (classical MDS). Gauge-invariant."""
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    return -0.5 * J @ D @ J


def deflate(G, pi, tol=1e-9):
    """Ghat = G - P G P, P = orthogonal projector onto the column space of pi (the anchor-reconstructed
    subspace pulled back through the coupling)."""
    U, s, _ = np.linalg.svd(pi, full_matrices=False)
    r = int((s > tol * s.max()).sum()) if s.max() > 0 else 0
    if r == 0:
        return G
    Q = U[:, :r]                      # orthonormal basis of col(pi)
    PG = Q @ (Q.T @ G)
    return G - PG - PG.T + Q @ (Q.T @ G @ Q) @ Q.T


def residual_moments(G, pi, k_max=K_MAX):
    """[tr(Ghat^k)/n for k=1..k_max] from the eigenvalues of the deflated Gram."""
    Ghat = deflate(G, pi)
    n = Ghat.shape[0]
    ev = np.linalg.eigvalsh((Ghat + Ghat.T) / 2)
    return np.array([(ev ** k).sum() / n for k in range(1, k_max + 1)])


def residual_vector(Ds, pis, k_max=K_MAX):
    """Stack residual moments across the members of an edge -> one vector per prompt.
    Ds, pis: lists over members of (normalized D_v, coupling pi_{v,e})."""
    return np.concatenate([residual_moments(gram_from_D(D), pi, k_max) for D, pi in zip(Ds, pis)])


def hankel_spectrum(series, L=12):
    """Block-Hankel singular spectrum of a windowed, mean-centered vector series (W×p) over lags 0..L,
    via the covariance sequence R(k) = <y_{t+k}, y_t> (stochastic-realization Hankel; reuses io_trace).
    Returns singular values sorted descending."""
    Y = np.asarray(series, dtype=np.float64)
    Y = Y - Y.mean(0)
    Y = Y / (Y.std(0) + 1e-12)         # z-score channels: moments span k=1..6 with disparate scales
    W, p = Y.shape
    kmax = 2 * L + 1
    R = np.zeros((kmax + 1, p, p))
    for k in range(kmax + 1):
        if W - k > 0:
            R[k] = Y[k:].T @ Y[:W - k] / (W - k)
    H = ST.block_hankel(R, L)          # Hankel of R[1..] (memory part), validated construction
    return np.linalg.svd(H, compute_uv=False)


def above_floor_rank(series, floor, L=12):
    sv = hankel_spectrum(series, L=L)
    return int((sv > floor).sum()), sv
