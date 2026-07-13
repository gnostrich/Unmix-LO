"""
P2 — corrected diversity leg. The old test expected raw invariant rank = latent degree r ([proven-negative],
closed). P1 showed invariant rank = symmetric-power degree. So here: single model (K=1, no K-inflation),
NO anchor deflation (m0=0, so nothing absorbs diversity — frozen-capacity note), doubled budget. Prediction
(pre-registered): invariant-observable rank is MONOTONE increasing in r and super-linear (toward the sym
values, truncated by floor/T). Decoder r_hat = (-1+sqrt(1+8·rank))/2 (quadratic inversion).
"""
import numpy as np
from ..geometry.clouds import cloud_to_Dw
from ..hankel import residual as H
from . import substrate as S
from . import g1_probe as G1


def invariant_series(r, T=300, n=140, seed=0, tap=False, jitter=0.15):
    """Per-prompt invariant moments (full Gram, NO anchor deflation) from one model — the raw invariant
    observable of the traffic. tap=True uses the pre-logits LINEAR readout (§12 fix): observable ~quadratic
    in the latent (degree r(r+1)/2, resolvable) instead of a high sym-power from the tanh nonlinearity."""
    U = S.latent_traffic(T, r, seed=seed)
    model = S.make_models(1, seed0=100)[0]
    out = []
    for t in range(T):
        inp = G1._warp_inputs(U[t], n)
        X = model.tap(inp) if tap else model(inp)
        D, w = cloud_to_Dw(X)
        out.append(H.residual_moments(H.gram_from_D(D), np.zeros((n, 1))))
    return np.array(out)


def _floor(series, L=20, draws=80, seed=0):
    rng = np.random.default_rng(seed)
    return np.percentile([H.hankel_spectrum(series[rng.permutation(len(series))], L=L)[0] for _ in range(draws)], 95)


def decode(rank):
    return (-1 + np.sqrt(1 + 8 * rank)) / 2


def run(rs=(2, 3, 4), T=1200, L=20, seed=0, tap=True):
    print(f"P2 corrected diversity leg (K=1, no anchor, T={T}, L={L}, pre-logits tap={tap}):")
    print(f"  sym prediction r(r+1)/2 = {[r * (r + 1) // 2 for r in rs]}")
    ranks = []
    for r in rs:
        s = invariant_series(r, T=T, seed=seed, tap=tap)
        fl = _floor(s, L=L)
        rank = int((H.hankel_spectrum(s, L=L) > fl).sum())
        ranks.append(rank)
        print(f"  latent r={r}: invariant rank={rank}  (decoded r_hat={decode(rank):.1f}, pred sym "
              f"{r * (r + 1) // 2})  floor={fl:.2f}")
    monotone = all(ranks[i + 1] > ranks[i] for i in range(len(ranks) - 1))
    superlin = ranks[-1] > rs[-1]
    print(f"P2 verdict: {'PASS' if monotone else 'FAIL'} — rank monotone in diversity {ranks} "
          f"({'super-linear' if superlin else 'not super-linear'})")
    return {"ranks": ranks, "monotone": bool(monotone), "superlinear": bool(superlin), "T": T, "tap": tap}


if __name__ == "__main__":
    run()
