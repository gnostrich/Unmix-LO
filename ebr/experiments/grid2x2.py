"""
Corrected G1 — the 2×2 selective dissociation (PREREG_2x2.md). Two independent knobs:
  g = within-prompt geometric richness (probe has g spatial clusters) -> should drive ATOM count (FW/F).
  r = across-prompt dynamical diversity (latent temporal degree)      -> should drive POLE count (instrument).
Each instrument should be sensitive to its own axis and flat on the other (and on K).
Run: python -m ebr.experiments.grid2x2
"""
import numpy as np
from ..geometry.clouds import cloud_to_Dw
from ..hankel import residual as H
from ..events import frankwolfe as FW
from . import substrate as S

RNG = np.random.default_rng(11)
_BASIS = [RNG.normal(size=(6, 6)) for _ in range(6)]


def clustered_probe(n, g, spread=1.4, seed=0):
    """Probe cloud with g spatial clusters -> per-prompt geometry of complexity ~g."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(g, 6)) * spread
    assign = np.repeat(np.arange(g), int(np.ceil(n / g)))[:n]
    return centers[assign] + 0.25 * rng.normal(size=(n, 6))


def _warp(u_row, warp=0.3):
    return np.eye(6) + sum(u_row[k] * _BASIS[k] for k in range(6)) * warp


def clouds_over_prompts(g, r, K, T=90, n=96, seed=0):
    """Return per-member averaged geometry (for atom count) and the invariant-moment series (for pole count)."""
    U = S.latent_traffic(T, r, seed=seed)
    models = S.make_diverse_models(K, seed0=200)
    probe = clustered_probe(n, g, seed=seed)
    accD = [np.zeros((n, n)) for _ in range(K)]
    series = []
    for t in range(T):
        inp = probe @ _warp(U[t])
        vec = []
        for i, M in enumerate(models):
            D, w = cloud_to_Dw(M(inp))
            accD[i] += D
            vec.append(H.residual_moments(H.gram_from_D(D), np.zeros((n, 1))))
        series.append(np.concatenate(vec))
    Ds = [D / T for D in accD]
    ws = [np.full(n, 1.0 / n) for _ in range(K)]
    return Ds, ws, np.array(series)


def atom_count(Ds, ws):
    res = FW.grow(Ds, ws, np.array([[0.0]]), np.array([1.0]), np.array([1.0]), max_atoms=9, n_outer=12)
    return res["active"]


def pole_count(series, L=18):
    """Resolved temporal modes = above-floor rank of the covariance Hankel of the invariant moments."""
    rng = np.random.default_rng(0)
    fl = np.percentile([H.hankel_spectrum(series[rng.permutation(len(series))], L=L)[0] for _ in range(60)], 95)
    return int((H.hankel_spectrum(series, L=L) > fl).sum())


def run(gs=(2, 3, 4), rs=(2, 3, 4), K=3, seed=0):
    print("Corrected G1 — 2×2 selective dissociation (atom count = spatial g; pole count = temporal r)\n")
    AT = np.zeros((len(gs), len(rs)), int)
    PO = np.zeros((len(gs), len(rs)), int)
    for ig, g in enumerate(gs):
        for ir, r in enumerate(rs):
            Ds, ws, series = clouds_over_prompts(g, r, K, seed=seed)
            AT[ig, ir] = atom_count(Ds, ws)
            PO[ig, ir] = pole_count(series)
            print(f"  g={g} r={r}: atoms={AT[ig, ir]}  poles={PO[ig, ir]}", flush=True)
    print("\n  ATOM count grid [rows g, cols r]:\n", AT)
    print("  POLE count grid [rows g, cols r]:\n", PO)
    # selective sensitivity: atoms ~ g (row-monotone), flat in r (cols); poles ~ r (col-monotone), flat in g
    atoms_track_g = all(AT[:, ir].tolist() == sorted(AT[:, ir].tolist()) for ir in range(len(rs))) and (AT.max(0) - AT.min(0)).max() <= 999
    atoms_flat_r = (AT.max(1) - AT.min(1)).max() <= 1
    poles_track_r = all(PO[ig, :].tolist() == sorted(PO[ig, :].tolist()) for ig in range(len(gs)))
    poles_flat_g = (PO.max(0) - PO.min(0)).max() <= 1
    print(f"\n  P6a atoms track g (row-monotone) & flat in r (spread≤1): {atoms_track_g} & {atoms_flat_r}")
    print(f"  P6b poles track r (col-monotone) & flat in g (spread≤1): {poles_track_r} & {poles_flat_g}")
    return {"atoms": AT.tolist(), "poles": PO.tolist(),
            "P6a": bool(atoms_track_g and atoms_flat_r), "P6b": bool(poles_track_r and poles_flat_g)}


if __name__ == "__main__":
    run()
