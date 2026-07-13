"""Probe: with the SHARED anchor pooling K members, is the residual-Hankel rank K-invariant and
diversity-sensitive (the G1 dissociation)? Runs the real per-prompt equilibration loop (warm-started
anchor, coupling continuity), no growth yet — fixed m0 anchor, measure residual above-floor rank."""
import numpy as np
from ..geometry.clouds import cloud_to_Dw
from ..hankel import residual as H
from ..energy import functional as EN
from . import substrate as S

RNG = np.random.default_rng(7)
PROBE = None
GK = None


def _warp_inputs(u_row, n, warp=0.3):
    global PROBE, GK
    if PROBE is None or PROBE.shape[0] != n:
        PROBE = RNG.normal(size=(n, 6)); GK = [RNG.normal(size=(6, 6)) for _ in range(6)]
    M = np.eye(6) + sum(u_row[k] * GK[k] for k in range(6)) * warp
    return PROBE @ M


def residual_series(K, r, T=200, n=120, m0=4, eps=0.08, seed=0, warmup=40, diverse=False):
    """Run the real per-prompt shared-anchor equilibration (warm-started, coupling continuity) and return
    (residual_moment_series, mean_Lyapunov_monotone_fraction). diverse=True uses heterogeneous architectures."""
    U = S.latent_traffic(T, r, seed=seed)
    models = S.make_diverse_models(K, seed0=200) if diverse else S.make_models(K, seed0=100)
    De = np.random.default_rng(1).random((m0, m0)); De = (De + De.T) / 2; np.fill_diagonal(De, 0)
    De /= np.median(De[np.triu_indices(m0, 1)]) if m0 > 1 else 1.0
    a = np.full(m0, 1.0 / m0); abar = a.copy()
    pis = None
    series, fmono = [], []
    for t in range(T):
        inp = _warp_inputs(U[t], n)
        Dws = [cloud_to_Dw(M(inp)) for M in models]
        Ds = [d for d, _ in Dws]; ws = [w for _, w in Dws]
        pis, De, a, ftr, conv = EN.equilibrate(Ds, ws, De, a, abar, eps=eps, pis0=pis)
        abar = 0.95 * abar + 0.05 * a                  # slow Polyak reference
        fmono.append(all(ftr[i + 1] <= ftr[i] + 1e-9 for i in range(len(ftr) - 1)))
        if t >= warmup:
            series.append(H.residual_vector(Ds, pis))
    return np.array(series), float(np.mean(fmono))


def run(K, r, T=200, n=120, m0=4, eps=0.08, seed=0, warmup=40, diverse=False):
    series, mono = residual_series(K, r, T=T, n=n, m0=m0, eps=eps, seed=seed, warmup=warmup, diverse=diverse)
    rank = int((H.hankel_spectrum(series) > _floor(series)).sum())
    return rank, mono


def _floor(series, L=12, draws=40, seed=0):
    rng = np.random.default_rng(seed)
    return np.percentile([H.hankel_spectrum(series[rng.permutation(len(series))], L=L)[0] for _ in range(draws)], 95)
