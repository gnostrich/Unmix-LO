"""
resolvent.py — the s4c-resolvent realization layer (built from scratch; the named module was absent).

A TASK DISTRIBUTION is a discrete-time operator-valued transfer function (resolvent)
    G(z) = C (zI - A)^{-1} B ,   Markov params  h_k = C A^{k-1} B  (k>=1),  h_0 = 0.
Its McMillan degree = dim(A) for a minimal realization = rank of the block-Hankel matrix of {h_k}
= atomic-support size of its resolvent (number of poles = eig(A)).

- ATOMIC distribution: a known minimal (A0,B0,C0) of order r  =>  McMillan degree = r EXACTLY (ground truth).
- CONTINUOUS-SPECTRUM distribution: a dense continuum of poles (no atomic support) => McMillan degree = inf;
  the Hankel spectrum decays smoothly with no clean gap.

Block-Hankel + Ho-Kalman (ERA) give the minimal realization from Markov parameters. numpy only.
"""
import numpy as np


# ------------------------------------------------------------------ task distributions (poles-first)
def atomic_system(r, p=3, seed=0, rho=0.9):
    """Known minimal realization of order r -> McMillan degree = r exactly. Returns (A0,B0,C0)."""
    rng = np.random.default_rng(seed)
    # stable A0 with spectral radius rho and well-separated poles (clean atomic support)
    A = rng.normal(size=(r, r))
    ev = np.linalg.eigvals(A)
    A = A * (rho / (np.max(np.abs(ev)) + 1e-9))
    B = rng.normal(size=(r, p)) / np.sqrt(p)
    C = rng.normal(size=(p, r)) / np.sqrt(r)
    return A, B, C


def markov_from_system(A, B, C, K):
    """Markov parameters h_1..h_K, each p x p:  h_k = C A^{k-1} B."""
    p = C.shape[0]
    h = np.zeros((K, p, p))
    M = np.eye(A.shape[0])
    for k in range(K):
        h[k] = C @ M @ B
        M = M @ A
    return h


def continuous_markov(K, p=3, seed=0, npoles=400, a=0.9):
    """Continuous-spectrum task distribution: a DENSE continuum of poles in (-a,a), no atomic support.
    h_k = (1/npoles) sum_m R_m lambda_m^{k-1}, lambda_m ~ Uniform(-a,a), R_m random rank-1 (p x p).
    McMillan degree = infinity; Hankel singular values decay smoothly (no gap)."""
    rng = np.random.default_rng(seed)
    lam = np.linspace(-a, a, npoles) + rng.normal(scale=1e-3, size=npoles)   # dense support
    U = rng.normal(size=(npoles, p)); V = rng.normal(size=(npoles, p))
    h = np.zeros((K, p, p))
    powk = np.ones(npoles)
    for k in range(K):
        # sum_m u_m v_m^T * lambda^{k-1}
        h[k] = (U * powk[:, None]).T @ V / npoles
        powk = powk * lam
    return h


# ------------------------------------------------------------------ block-Hankel + Ho-Kalman (ERA)
def block_hankel(h, L, shift=0):
    """Block-Hankel of Markov params: H[i,j] = h[i+j+shift], i,j in [0,L). Shape (L*p, L*p).
    Needs h of length >= 2L+shift."""
    p = h.shape[1]
    H = np.zeros((L * p, L * p))
    for i in range(L):
        for j in range(L):
            H[i*p:(i+1)*p, j*p:(j+1)*p] = h[i + j + shift]
    return H


def hankel_svals(h, L=None):
    """Singular values of the block-Hankel matrix of {h_k} (k indexed from 1: pass h with h[0]=h_1)."""
    K = h.shape[0]
    if L is None:
        L = K // 2
    L = min(L, (K) // 2)
    return np.linalg.svd(block_hankel(h, L), compute_uv=False), L


def ho_kalman(h, n, L=None):
    """Ho-Kalman / ERA: minimal order-n realization (A,B,C) from Markov params h_1.. .
    Uses H0 = Hankel(h_1..) and H1 = shifted Hankel(h_2..)."""
    K = h.shape[0]; p = h.shape[1]
    if L is None:
        L = K // 2
    L = min(L, (K - 1) // 2)
    H0 = block_hankel(h, L, shift=0)     # blocks h_{1..}
    H1 = block_hankel(h, L, shift=1)     # blocks h_{2..}
    U, S, Vt = np.linalg.svd(H0)
    n = max(1, min(n, np.sum(S > 1e-12)))
    U1 = U[:, :n]; S1 = S[:n]; V1 = Vt[:n].T
    sq = np.sqrt(S1)
    O = U1 * sq                          # observability  (L*p, n)
    R = (V1 * sq).T                      # reachability    (n, L*p)
    Oinv = (U1 / sq).T                   # pinv(O)  = diag(1/sq) U1^T
    Rinv = (V1 / sq)                     # pinv(R)  = V1 diag(1/sq)
    A = Oinv @ H1 @ Rinv
    C = O[:p, :]
    B = R[:, :p]
    return A, B, C, S


def mcmillan_degree_true(kind, r=None):
    """Ground-truth McMillan degree: r for atomic, inf for continuous."""
    return float("inf") if kind == "continuous" else int(r)
