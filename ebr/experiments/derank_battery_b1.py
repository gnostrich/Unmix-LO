"""
experiments/derank_battery_b1.py — TRUST BATTERY leg B1 (audit classification: experimental CONTROL /
GAUGE GATE; instrument, NOT mechanism). This file MEASURES an invariance of the D_e-effective-rank spatial
meter; it adds no new mechanism and no new QC. Every numeric function it calls is reused verbatim from
atom_observable_search.py (equilibrate_fixed_m, eff_rank, the (b) D_e eff-rank readout) and clouds.py
(cloud_to_Dw, scramble).

PRE-REGISTRATION under test: PREREG_derank_battery.md, section "B1 — Scramble (gauge)". The equilibrated
anchor's D_e effective rank is built from the gauge-normalized (D, w) only, so it must be INVARIANT to the G0
scramble group (orthogonal x perm x scale x shift) applied to a member's raw coordinates, to numerical
precision. Committed prediction: |Δ eff-rank| < 1e-6. PASS iff invariant.

WHAT THIS FEEDS (the honest, non-trivial part): the scramble is applied to the RAW COORDINATE cloud X BEFORE
cloud_to_Dw, not to D. Scrambling D directly would be a trivially-passing shim (D is invariant by
construction). Here we re-derive each member's coordinates X = Z @ Wmap, then compare the FULL downstream
pipeline
    X            -> cloud_to_Dw -> (D, w) -> equilibrate_fixed_m(m=12) -> svd(De) -> eff_rank
    scramble(X)  -> cloud_to_Dw -> (D, w) -> equilibrate_fixed_m(m=12) -> svd(De) -> eff_rank
so the invariance must survive the entire equilibration, not just the first step. For a fair comparison the
anchor initialization (De0, a0) is held identical across the clean/scrambled runs (same seed into
equilibrate_fixed_m); the ONLY difference is whether the member coordinates were scrambled.

DEFENDED CONSTANTS (no undefended magic; no shims; no force-pass flags):
    * m = 12: the same generous fixed anchor budget as readout (b) in atom_observable_search.py.
    * r in {2, 6}: two planted ranks (one low, one mid) — enough to show invariance is content-independent.
    * K = 3 members, SEEDS = [0, 1, 2]: reuse the substrate's member count and seed set.
    * scramble seed per (r, cloud-seed): SCR_SEED_BASE + sd, an arbitrary but fixed offset so the scramble is
      a genuine nontrivial G0 element (random orthogonal + improper flip + rescale + shift), not the identity.
    * PASS RULE (to the digit, from the prereg): B1 passes iff max over all (r, seed, member-set) of
      |eff_rank_clean - eff_rank_scrambled| < 1e-6.

Run: python -m ebr.experiments.derank_battery_b1
"""
import numpy as np

from ..geometry.clouds import cloud_to_Dw, scramble
from .atom_observable_search import (
    equilibrate_fixed_m,
    eff_rank,
    K,
    N,
    D_AMB,
    M_MAX,
    SEEDS,
)

R_LEVELS = [2, 6]
SCR_SEED_BASE = 777          # fixed offset so scramble(X, seed) is a genuine non-identity G0 element
PASS_TOL = 1e-6              # committed prediction from PREREG_derank_battery.md, B1


def planted_coords(r, K=K, n=N, d=D_AMB, seed=0):
    """LOCAL copy of atom_observable_search.planted_clouds that returns the underlying COORDINATE clouds Xk
    (Z @ Wmap) per member, instead of the (D, w) pairs. Same RNG stream and construction as the original, so
    cloud_to_Dw(planted_coords(...)[k]) reproduces planted_clouds(...) exactly. Returning X is what lets the
    scramble bite on the raw coordinates before cloud_to_Dw."""
    rng = np.random.default_rng(seed * 131 + r)
    Xs = []
    for _k in range(K):
        Z = rng.normal(size=(n, r))              # N(0, I_r)
        Wmap = rng.normal(size=(r, d))           # fixed random r x d embedding (member-specific)
        Xs.append(Z @ Wmap)
    return Xs


def derank_from_coords(Xs, seed):
    """The readout (b) pipeline, driven from coordinate clouds: cloud_to_Dw per member, equilibrate one shared
    m=M_MAX anchor (init seeded by `seed`), eff_rank of the anchor cost's singular values."""
    Ds, ws = zip(*(cloud_to_Dw(X) for X in Xs))
    _pis, De, _a = equilibrate_fixed_m(list(Ds), list(ws), M_MAX, seed=seed)
    sv = np.linalg.svd(De, compute_uv=False)
    return eff_rank(sv)


def run():
    print("=" * 78)
    print("TRUST BATTERY B1 — Scramble (gauge) gate for the D_e eff-rank spatial meter")
    print(f"  planted rank-r clouds, r in {R_LEVELS}, K={K}, n={N}, d={D_AMB}, m={M_MAX}, seeds={SEEDS}")
    print("  scramble applied to RAW COORDINATES before cloud_to_Dw; anchor init held identical.")
    print(f"  PREREG B1: |Δ eff-rank| < {PASS_TOL:g}. PASS iff invariant.")
    print("=" * 78)

    all_deltas = []
    print("\n  r  | seed | eff-rank clean | eff-rank scrambled |     |Δ eff-rank|")
    print("  " + "-" * 66)
    per_r = {}
    for r in R_LEVELS:
        r_deltas = []
        for sd in SEEDS:
            Xs = planted_coords(r, seed=sd)
            clean = derank_from_coords(Xs, seed=sd)

            # apply a genuine G0 element to ALL members' raw coordinates, then rerun the full pipeline
            Xs_scr = [scramble(X, seed=SCR_SEED_BASE + sd) for X in Xs]
            scr = derank_from_coords(Xs_scr, seed=sd)

            delta = abs(clean - scr)
            r_deltas.append(delta)
            all_deltas.append(delta)
            print(f"  {r:>1d}  |  {sd:>2d}  |   {clean:12.8f} |    {scr:12.8f}    |   {delta:.3e}")
        per_r[r] = (float(np.mean([derank_from_coords(planted_coords(r, seed=sd), seed=sd) for sd in SEEDS])),
                    float(max(r_deltas)))

    print("\n  per-r summary (seed-averaged clean eff-rank, max |Δ| over seeds):")
    for r in R_LEVELS:
        mean_clean, max_d = per_r[r]
        print(f"    r={r}: clean eff-rank ≈ {mean_clean:.6f}   max|Δ| = {max_d:.3e}")

    worst = max(all_deltas)
    verdict = worst < PASS_TOL
    print("\n" + "=" * 78)
    print(f"  worst |Δ eff-rank| over all cells = {worst:.3e}   (PASS tol {PASS_TOL:g})")
    print(f"  B1 VERDICT: {'PASS' if verdict else 'FAIL'} — D_e eff-rank is "
          f"{'invariant' if verdict else 'NOT invariant'} to the G0 scramble on raw coordinates.")
    print("=" * 78)
    return {"per_r": per_r, "worst_delta": worst, "pass": verdict}


if __name__ == "__main__":
    run()
