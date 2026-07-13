"""
geometry/clouds.py — the ONLY module that touches model outputs. Converts a point cloud at a port into a
normalized cost matrix + masses (D, w). Nothing downstream may consume coordinates (invariant-interface
rule, §0); this is what makes every logged quantity gauge-invariant (G0).

Gauge / scramble group G0 on a model's interface representation X (n×d):
    X -> (s · X · Q) + t   with Q orthogonal (incl. coordinate permutation), s>0, t a constant shift.
D[i,j] = ||x_i - x_j||^2 is invariant to Q and t; the global scale s^2 is divided out by the median
normalization. Point relabelling permutes rows/cols of D (a relabelling the downstream is equivariant to).
So (D, w) is invariant to the scramble group up to point permutation — proved here, tested in tests/.
"""
import numpy as np


def cloud_to_Dw(X):
    """X: (n, d) cloud -> (D_normalized, w_uniform). D = squared-euclidean, normalized by its median.
    Median normalization makes scale a gauge dimension (mandatory, §0)."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    sq = (X * X).sum(1)
    D = sq[:, None] + sq[None, :] - 2.0 * X @ X.T
    D = np.maximum(D, 0.0)
    np.fill_diagonal(D, 0.0)
    med = np.median(D[np.triu_indices(n, 1)])
    if med <= 0:
        med = 1.0
    D = D / med
    w = np.full(n, 1.0 / n)
    return D, w


def scramble(X, seed=0, shift=True):
    """Apply a random element of the G0 scramble group to a representation (for the gauge-invariance gate)."""
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    A = rng.normal(size=(d, d))
    Q, _ = np.linalg.qr(A)                     # random orthogonal
    if rng.random() < 0.5:                      # include an improper rotation / coordinate permutation
        Q[:, 0] *= -1
    s = float(np.exp(rng.normal() * 0.5))       # positive rescale
    t = rng.normal(size=(1, d)) if shift else 0.0
    return s * (X @ Q) + t
