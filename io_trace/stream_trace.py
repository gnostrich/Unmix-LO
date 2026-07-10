"""
stream_trace.py — the trace machinery built ON the I/O stream (fit-free; no read-out head, no free knobs).

Given only a stream (u_t, y_t):
  1. memory response from the stream itself:  h_k = (1/T) sum_t y_{t+k} u_t^T   (k >= 1; h_0 = instantaneous
     map, set aside — memory is what remains after the instant).  For white input this IS the Markov/memory
     sequence of the MZ closure of y on u.
  2. block-Hankel of h_{1..} -> singular spectrum -> atoms above a SELF-CALIBRATING permutation floor
     (circularly shift the output stream, recompute the top Hankel singular value; floor = a quantile of that
     null distribution — the stream's own marginals define "no temporal structure").
  3. Ho-Kalman realization of the above-floor part -> the memory kernel's POLES.
The natural read = (atom count, poles). Nothing is trained. numpy only.
"""
import numpy as np

FLOOR_SHIFTS = 40      # permutation-null draws (circular shifts of y)
FLOOR_Q = 99.0         # floor = this percentile of null top singular values


def est_markov(u, y, kmax):
    """h_k = (1/T) sum_t y_{t+k} u_t^T for k = 0..kmax. Shapes: u (T,p), y (T,q) -> h (kmax+1, q, p)."""
    T = len(u)
    h = np.zeros((kmax + 1, y.shape[1], u.shape[1]))
    for k in range(kmax + 1):
        h[k] = y[k:].T @ u[:T - k] / (T - k)
    return h


def block_hankel(h, L, shift=0):
    """Hankel of the MEMORY sequence h[1..]: block (i,j) = h[1 + i + j + shift]."""
    q, p = h.shape[1], h.shape[2]
    H = np.zeros((L * q, L * p))
    for i in range(L):
        for j in range(L):
            H[i * q:(i + 1) * q, j * p:(j + 1) * p] = h[1 + i + j + shift]
    return H


def permutation_floor(u, y, kmax, L, seed=0):
    """The self-calibrating noise floor: destroy temporal structure with the stream's own marginals
    (large random circular shifts of y), recompute the top Hankel singular value each time."""
    rng = np.random.default_rng(seed)
    T = len(y)
    tops = []
    for _ in range(FLOOR_SHIFTS):
        s = int(rng.integers(T // 4, 3 * T // 4))
        hn = est_markov(u, np.roll(y, s, axis=0), kmax)
        tops.append(np.linalg.svd(block_hankel(hn, L), compute_uv=False)[0])
    return float(np.percentile(tops, FLOOR_Q))


def ho_kalman(h, n, L):
    """Minimal order-n realization of the memory sequence h[1..] -> (A,B,C). Poles = eig(A)."""
    H0 = block_hankel(h, L, shift=0)
    H1 = block_hankel(h, L, shift=1)
    U, S, Vt = np.linalg.svd(H0)
    n = max(1, min(n, int((S > 1e-12).sum())))
    sq = np.sqrt(S[:n])
    O = U[:, :n] * sq
    A = (U[:, :n] / sq).T @ H1 @ (Vt[:n].T / sq)
    q, p = h.shape[1], h.shape[2]
    return A, (Vt[:n].T * sq).T[:, :p], O[:q, :]


def read_trace(u, y, kmax=25, L=12, seed=0):
    """The natural read of the stream's memory kernel: atoms above the permutation floor + their poles.
    Returns dict with order, poles, spectral gap at the cut, floor, singular values."""
    h = est_markov(u, y, kmax)
    sv = np.linalg.svd(block_hankel(h, L), compute_uv=False)
    floor = permutation_floor(u, y, kmax, L, seed=seed)
    order = int((sv > floor).sum())
    gap = float(sv[order - 1] / (sv[order] + 1e-15)) if 0 < order < len(sv) else 0.0
    poles = np.array([])
    if order > 0:
        A, _, _ = ho_kalman(h, order, L)
        poles = np.linalg.eigvals(A)
    return {"order": order, "poles": poles, "gap": gap, "floor": floor, "svals": sv, "h": h}


def pole_match_error(true_poles, rec_poles):
    """Mean distance from each true pole to its nearest recovered pole (inf if none recovered)."""
    if len(rec_poles) == 0:
        return float("inf")
    return float(np.mean([np.min(np.abs(rec_poles - tp)) for tp in true_poles]))
