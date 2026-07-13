"""
phase_zero.py — G0 calibration (§8), the gate everything is downstream of. Runs on known-degree substrate
traffic. Produces: gauge-scramble verdict, floors (φ_H via time-permutation null), positive control
(mismatched batch must fire), and φ_solver (restart spread of couplings). Nothing else may run until G0 passes.
Run: python -m ebr.phase_zero
"""
import numpy as np
from .geometry.clouds import cloud_to_Dw, scramble
from .hankel import residual as H
from .transport import gw
from .experiments import substrate as S
from .experiments import g1_probe as G1


def gauge_gate(seed=0, n=120):
    """Scramble one model's representation; every (D, residual-moment) must be invariant to numerical
    precision. Passes by construction (Gram double-centered from normalized D). Hard gate."""
    U = S.latent_traffic(30, 3, seed=seed)
    models = S.make_models(3, seed0=100)
    inp = G1._warp_inputs(U[5], n)
    worst = 0.0
    for M in models:
        X = M(inp)
        D, w = cloud_to_Dw(X)
        Ds = cloud_to_Dw(scramble(X, seed=seed + 1))[0]
        worst = max(worst, float(np.abs(D - Ds).max()))
        # residual moments (no coupling => full Gram) must also be invariant
        m1 = H.residual_moments(H.gram_from_D(D), np.zeros((n, 1)))
        m2 = H.residual_moments(H.gram_from_D(Ds), np.zeros((n, 1)))
        rel = np.abs(m1 - m2) / (np.abs(m1) + 1e-9)      # RELATIVE (6th moments are ~1e6; tight abs is unfair)
        worst = max(worst, float(rel.max()))
    return worst, worst < 1e-6


def floors(series, draws=100, L=12):
    """φ_H = 95th percentile of top Hankel singular value under the time-permutation null (destroys
    prompt-time structure, keeps marginals)."""
    rng = np.random.default_rng(0)
    tops = [H.hankel_spectrum(series[rng.permutation(len(series))], L=L)[0] for _ in range(draws)]
    return float(np.percentile(tops, 95))


def positive_control(series, floor):
    """A deliberately structured (un-permuted) series must exceed the floor — the detector can fire."""
    top = H.hankel_spectrum(series)[0]
    return top, top > floor


def solver_floor(D, De, w, a, eps, R=16):
    """φ_solver = spread (std) of GW cost across random-restart couplings on a fixed cloud (§7)."""
    costs = []
    for r in range(R):
        rng = np.random.default_rng(r)
        pi0 = rng.random((D.shape[0], De.shape[0])); pi0 *= (w / pi0.sum(1))[:, None]
        pi, _ = gw.equilibrate_coupling(D, De, w, a, eps, pi0=pi0, j_sink=5)
        costs.append(gw.gw_cost(D, De, pi))
    return float(np.std(costs)), float(np.median(costs))


def run():
    print("=" * 66); print("PHASE ZERO (G0) — must pass before anything runs"); print("=" * 66)
    worst, gok = gauge_gate()
    print(f"[gauge-scramble] worst |Δ| over (D, residual moments) = {worst:.2e}  ->  {'PASS' if gok else 'FAIL'}")

    # a real residual series to set/exercise the floor + positive control
    series, _mono = G1.residual_series(3, 3, T=120, m0=1, seed=0)
    phiH = floors(series)
    top, pcok = positive_control(series, phiH)
    print(f"[floor φ_H]      {phiH:.3f}   [positive control] top={top:.3f} > φ_H : {'PASS' if pcok else 'FAIL'}")

    # solver floor on one prompt's coupling
    U = S.latent_traffic(10, 3, seed=0); models = S.make_models(3, seed0=100)
    D, w = cloud_to_Dw(models[0](G1._warp_inputs(U[3], 120)))
    De = np.random.default_rng(1).random((4, 4)); De = (De + De.T) / 2
    np.fill_diagonal(De, 0); De /= np.median(De[np.triu_indices(4, 1)])
    sstd, smed = solver_floor(D, De, w, np.full(4, 0.25), 0.08)
    calibratable = sstd < smed
    print(f"[φ_solver]       restart std={sstd:.4f}  median cost={smed:.4f}  "
          f"-> {'CALIBRATABLE' if calibratable else 'UNCALIBRATABLE'}")

    g0 = gok and pcok and calibratable
    print("-" * 66)
    print(f"G0 verdict: {'PASS' if g0 else 'FAIL'}")
    return {"gauge_worst": worst, "gauge_pass": bool(gok), "phi_H": phiH,
            "positive_control_pass": bool(pcok), "phi_solver_std": sstd,
            "phi_solver_med": smed, "calibratable": bool(calibratable), "G0": bool(g0)}


if __name__ == "__main__":
    run()
