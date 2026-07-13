"""
geometry/gram.py — gauge-faithful geometry helpers derived from the normalized cost D only (never
coordinates). Used by the Frank–Wolfe ORACLE (events/frankwolfe.py) to propose atom directions. These are
pure geometry, not the abandoned rank instrument.
"""
import numpy as np


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
    Q = U[:, :r]
    PG = Q @ (Q.T @ G)
    return G - PG - PG.T + Q @ (Q.T @ G @ Q) @ Q.T
