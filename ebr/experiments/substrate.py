"""
substrate.py — known-degree traffic for validating the instrument (the spec's positive-control logic: you
cannot check 'active rank = McMillan degree of traffic' against opaque models whose degree is unknown).

A hidden order-r linear system drives a latent trajectory u(t) over PROMPT time (its McMillan degree = r =
the 'diversity'). Each prompt t: n augmented inputs around u(t); each of K frozen nonlinear models maps them
to a feature cloud -> port cost D_v(t). Raising K = more views of the SAME latent (degree stays r); raising
r = more latent modes (degree grows). This is io_trace's atomic generator lifted through a model/port layer.
"""
import numpy as np


def latent_traffic(T, r, d_lat=6, seed=0, rho=0.9, noise=0.02):
    """Order-r hidden linear system over prompt time -> latent u(t) in R^{d_lat} with McMillan degree r."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(r, r)); A *= rho / np.max(np.abs(np.linalg.eigvals(A)))
    B = rng.normal(size=(r, r)) / np.sqrt(r)
    Cmap = rng.normal(size=(d_lat, r)) / np.sqrt(r)
    x = np.zeros(r); U = []
    for _ in range(T):
        U.append(Cmap @ x); x = A @ x + B @ rng.normal(size=r)
    U = np.array(U)
    return U + noise * rng.normal(size=U.shape)


class FrozenModel:
    """A fixed random nonlinear map R^{d_lat} -> R^{d_out} (a sealed black box)."""
    def __init__(self, d_lat, d_out, seed):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(size=(d_lat, 32)) / np.sqrt(d_lat)
        self.b1 = rng.normal(size=32) * 0.1
        self.W2 = rng.normal(size=(32, d_out)) / np.sqrt(32)

    def __call__(self, Z):
        return np.tanh(Z @ self.W1 + self.b1) @ self.W2

    def tap(self, Z):
        """Pre-logits LINEAR tap (§12 fix): features before the nonlinearity. The invariant moments are then
        ~quadratic in the latent -> observable degree r(r+1)/2 (resolvable), not a high sym-power."""
        return Z @ self.W1 + self.b1


def make_models(K, d_lat=6, d_out=12, seed0=100):
    return [FrozenModel(d_lat, d_out, seed=seed0 + i) for i in range(K)]


class DiverseModel:
    """A heterogeneous frozen black box — one of several distinct architectures, so 'flat rank in K' means
    the anchor found shared structure ACROSS real representational differences, not across near-clones."""
    def __init__(self, kind, d_lat, d_out, seed):
        rng = np.random.default_rng(seed)
        self.kind = kind
        self.W1 = rng.normal(size=(d_lat, 40)) / np.sqrt(d_lat)
        self.b1 = rng.normal(size=40) * 0.1
        self.W2 = rng.normal(size=(40, d_out)) / np.sqrt(40)
        self.Wd = rng.normal(size=(d_out, d_out)) / np.sqrt(d_out)
        self.freq = rng.normal(size=(d_lat, 40)) * 1.5

    def __call__(self, Z):
        if self.kind == "relu_deep":
            h = np.maximum(Z @ self.W1 + self.b1, 0) @ self.W2
            return np.maximum(h @ self.Wd, 0)
        if self.kind == "quadratic":
            h = Z @ self.W1
            return (h * h) @ self.W2
        if self.kind == "fourier":
            return np.sin(Z @ self.freq) @ self.W2
        if self.kind == "tanh":
            return np.tanh(Z @ self.W1 + self.b1) @ self.W2
        return (Z @ self.W1 + self.b1) @ self.W2          # linear

    def tap(self, Z):
        """Pre-logits linear tap (§12 fix): observable ~quadratic in the latent."""
        return Z @ self.W1 + self.b1


DIVERSE_KINDS = ["tanh", "relu_deep", "quadratic", "fourier", "linear"]


def make_diverse_models(K, d_lat=6, d_out=12, seed0=200):
    """K genuinely different architectures (cycled from DIVERSE_KINDS)."""
    return [DiverseModel(DIVERSE_KINDS[i % len(DIVERSE_KINDS)], d_lat, d_out, seed=seed0 + i) for i in range(K)]


def prompt_clouds(models, u_row, n=256, jitter=0.15, seed=0):
    """One prompt -> n augmented inputs around the latent u_row -> one cloud per model (n points each)."""
    rng = np.random.default_rng(seed)
    aug = u_row[None, :] + jitter * rng.normal(size=(n, len(u_row)))
    return [M(aug) for M in models]
