"""
Validation of the Frank–Wolfe re-derivation (spec v1.1 #1): structural growth driven by F ALONE.
Reproduces: self-quenching (each accepted atom strictly lowers F), K-invariant self-sizing, and the
mechanism/instrument separation (anchor count = spatial complexity of the shared geometry, NOT the traffic's
temporal McMillan degree — that is the pole instrument's job). Run: python -m ebr.experiments.fw_selfsize
"""
import numpy as np
from ..geometry.clouds import cloud_to_Dw
from ..events import frankwolfe as FW
from . import substrate as S


def snapshot(K, r, n=100, W=25, seed=0):
    """Slow shared geometry: normalized D averaged over a window, per member (heterogeneous architectures)."""
    U = S.latent_traffic(60, r, seed=seed)
    models = S.make_diverse_models(K, seed0=200)
    accD = [np.zeros((n, n)) for _ in range(K)]
    for t in range(5, 5 + W):
        inp = S.warp_inputs(U[t], n)
        for i, M in enumerate(models):
            accD[i] += cloud_to_Dw(M(inp))[0]
    return [D / W for D in accD], [np.full(n, 1.0 / n) for _ in range(K)]


def _grow(Ds, ws):
    return FW.grow(Ds, ws, np.array([[0.0]]), np.array([1.0]), np.array([1.0]), max_atoms=10)


def run():
    print("Frank–Wolfe structural events — driven by F alone (no Hankel). Self-quench + K-invariance.")
    print("  K-invariance leg (r=3): anchor count should be flat in K")
    counts = []
    for K in (2, 3, 5):
        res = _grow(*snapshot(K, 3, seed=0))
        counts.append(res["n_atoms"])
        print(f"    K={K}: {res['n_atoms']} atoms (active {res['active']}); F {[round(x, 2) for x in res['F_trace']]}")
    flat = max(counts) - min(counts) == 0
    mono = all(all(t['F_trace'][i + 1] < t['F_trace'][i] for i in range(len(t['F_trace']) - 1))
               for t in [_grow(*snapshot(3, 3, seed=0))])
    print(f"  K-invariance: {'PASS (flat)' if flat else 'partial'} {counts};  F strictly decreasing per atom: {mono}")
    print("  (anchor count = spatial complexity of the SHARED geometry, K-invariant — distinct from the")
    print("   traffic's temporal McMillan degree, which is the pole instrument's job, P5.)")
    return {"K_counts": counts, "K_flat": bool(flat), "F_monotone": bool(mono)}


if __name__ == "__main__":
    run()
