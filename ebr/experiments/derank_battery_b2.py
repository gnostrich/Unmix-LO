"""
experiments/derank_battery_b2.py — TRUST BATTERY leg B2 (audit classification: experimental CONTROL /
null-floor + offset explanation; this is VALIDATION of an instrument, NOT mechanism. Nothing here drives the
2x2 — it only probes whether the D_e-eff-rank spatial meter reads structure vs noise the way the prereg
committed it must, BEFORE the meter is trusted).

WHAT THIS LEG TESTS (verbatim targets from ebr/PREREG_derank_battery.md, section B2):
  1. eff-rank is MONOTONE non-decreasing in planted rank r (more intrinsic structure -> more anchor
     directions used). r = 1..8; r=1 is a new low end.
  2. The r=1 baseline reads an OFFSET O > 1 (NOT ~1). At fixed budget m=12 with entropic eps the anchor
     spectrum is smoothed, so even a rank-1 cloud spreads its mass over ~O atoms. Prereg predicts O in [3,6]
     (consistent with the known r=2 -> 4.67 reading). O is set by (m, eps), not by content; the meter's
     usable signal is the ORDER / increment above O, not the absolute value.
  3. STRUCTURELESS clouds with MATCHED moments read HIGH — at the TOP of the meter's dynamic band, clearly
     ABOVE the structured low-r readings. EMPIRICAL CAVEAT (do not airbrush): that band is COMPRESSED to
     ~[4.65, 6.4], it is NOT near m=12. The same entropic smoothing that lifts the r=1 floor to the offset O
     also pulls the ceiling far below m, so the D_e eff-rank readout's usable signal is the ORDER / margin
     above the structured levels, NOT proximity to the nominal budget m. Two honest constructions, both
     reported:
        (i)  iid Gaussian N(0, I_d) in the full ambient dimension d (maximal intrinsic rank).
        (ii) a PER-FEATURE independent shuffle of a structured (low-r) cloud's columns — destroys the
             cross-feature low-rank structure while preserving each feature's first/second moments.

PASS RULE (verbatim from prereg): PASS iff eff-rank(structureless) > eff-rank(rank-2) by a CLEAR MARGIN
AND eff-rank is monotone in r. (If structureless read LOW, or r were non-monotone, the meter would be
reading noise, not structure.) A failed leg KILLS the meter — reported honestly, no patching.

DEFENDED CONSTANTS (no undefended magic; no shims; no special-casing to force the ordering or the margin):
  * R_LEVELS = [1..8]           — prereg's stated range; r=1 is the mandated new low end for the offset test.
  * M = 12                      — the meter's fixed anchor budget (spec cap m<=12); the exact operating point
                                  at which the 4.67-at-r=2 offset was observed. Held fixed so O is a property
                                  of (m, eps), matching atom_observable_search's readout (b).
  * EPS = 0.05                  — entropic regularizer; atom_observable_search default (the operating point
                                  the offset must be explained AT).
  * K = 3, N = 100, D_AMB = 16  — atom_observable_search defaults (K members; d=16 > max r=8 so r is the true
                                  intrinsic rank; d=16 > m=12 so the anchor budget m, not d, is the NOMINAL
                                  cap — though entropic smoothing pulls the REALIZED ceiling to ~6.4, below m).
  * SEEDS = [0, 1, 2]           — seed-averaged to beat solver noise (same as atom_observable_search).
  * SHUFFLE_BASE_R = 2          — the shuffle null is built from the r=2 structured cloud so its matched
                                  moments correspond to the very level (r=2 -> 4.67) the offset explains.
  * CLEAR_MARGIN = 1.0          — REPORTING threshold for "clear margin": one full extra anchor direction of
                                  eff-rank separating the structureless null from rank-2. Not a mechanism knob;
                                  the raw margins are printed so a human can judge independently.
  * MONO_NOISE = 0.10           — seed-averaging leaves ~0.1 eff-rank jitter; an adjacent dip below this is
                                  reported as noise, a dip above it as a genuine non-monotonicity. Spearman is
                                  also printed so the ordering verdict never rests on this tolerance alone.

Everything numeric below is REUSED from atom_observable_search: planted_clouds() substrate, eff_rank(),
equilibrate_fixed_m() (readout (b) D_e eff-rank at m=12), gram eff-rank (readout (e)), and spearman(). No new
mechanism and no new QC are introduced here.

Run: python -m ebr.experiments.derank_battery_b2
"""
import numpy as np

from ..geometry.clouds import cloud_to_Dw
from ..geometry import gram as GR
from .atom_observable_search import (
    planted_clouds, eff_rank, equilibrate_fixed_m, spearman,
    N, D_AMB, K, EPS, N_OUTER,
)

R_LEVELS = [1, 2, 3, 4, 5, 6, 7, 8]      # prereg range; r=1 is the mandated new low end
M = 12                                    # fixed anchor budget = the meter's operating point (readout (b))
SEEDS = [0, 1, 2]                         # seed-averaged, atom_observable_search convention
SHUFFLE_BASE_R = 2                        # matched-moment shuffle built from the r=2 cloud (the 4.67 level)
CLEAR_MARGIN = 1.0                        # reporting threshold: one full anchor direction of separation
MONO_NOISE = 0.10                         # seed-averaging jitter allowed on an adjacent dip


# ----------------------------------------------------------------------------- raw-cloud generators
def planted_cloud_Xs(r, K=K, n=N, d=D_AMB, seed=0):
    """Raw clouds X (n x d) generated EXACTLY as planted_clouds() does, before cloud_to_Dw. Kept in lockstep
    with the substrate so the shuffle null is a faithful moment-match of the real rank-r clouds."""
    rng = np.random.default_rng(seed * 131 + r)   # identical seeding to planted_clouds
    Xs = []
    for _k in range(K):
        Z = rng.normal(size=(n, r))               # N(0, I_r)
        Wmap = rng.normal(size=(r, d))            # fixed random r x d embedding (member-specific)
        Xs.append(Z @ Wmap)
    return Xs


def structureless_iid(K=K, n=N, d=D_AMB, seed=0):
    """(i) iid Gaussian N(0, I_d) in the FULL ambient dimension d: maximal intrinsic rank, no low-rank
    structure. The honest 'no compressible structure' null."""
    rng = np.random.default_rng(seed * 977 + 12345)
    Ds, ws = [], []
    for _k in range(K):
        D, w = cloud_to_Dw(rng.normal(size=(n, d)))
        Ds.append(D); ws.append(w)
    return Ds, ws


def structureless_shuffle(r_base=SHUFFLE_BASE_R, K=K, n=N, d=D_AMB, seed=0):
    """(ii) per-feature independent row-shuffle of a structured rank-r_base cloud. Permuting each column
    independently preserves every feature's empirical marginal (hence its first & second moments) while
    destroying the cross-feature covariance that carried the low-rank structure. Matched moments, no
    low-rank structure."""
    rng = np.random.default_rng(seed * 631 + 7)
    Xs = planted_cloud_Xs(r_base, K=K, n=n, d=d, seed=seed)
    Ds, ws = [], []
    for X in Xs:
        Xsh = np.empty_like(X)
        for j in range(d):
            Xsh[:, j] = X[rng.permutation(n), j]  # independent per-feature permutation
        D, w = cloud_to_Dw(Xsh)
        Ds.append(D); ws.append(w)
    return Ds, ws


# ----------------------------------------------------------------------------- readouts (reused (b) and (e))
def derank_and_gram(Ds, ws, seed):
    """(b) D_e eff-rank at m=M via equilibrate_fixed_m, and (e) member-averaged Gram eff-rank. Both reused
    verbatim from atom_observable_search."""
    _pis, De, _aeq = equilibrate_fixed_m(Ds, ws, M, eps=EPS, n_outer=N_OUTER, seed=seed)
    b = eff_rank(np.linalg.svd(De, compute_uv=False))
    e = float(np.mean([eff_rank(np.linalg.eigvalsh(GR.gram_from_D(D))) for D in Ds]))
    return b, e


def seed_avg(gen):
    """Average (b) D_e eff-rank and (e) Gram eff-rank over SEEDS. gen(seed) -> (Ds, ws)."""
    bs, es = [], []
    for sd in SEEDS:
        Ds, ws = gen(sd)
        b, e = derank_and_gram(Ds, ws, sd)
        bs.append(b); es.append(e)
    return float(np.mean(bs)), float(np.mean(es))


def run():
    print("=" * 82)
    print("TRUST BATTERY B2 — null floor on structureless clouds + explanation of the 4.67 offset")
    print(f"  planted continuous rank-r clouds, r in {R_LEVELS}, K={K}, n={N}, d={D_AMB},")
    print(f"  seeds={SEEDS}, m={M}, eps={EPS}, n_outer={N_OUTER}. (instrument VALIDATION; not mechanism)")
    print("  PASS iff eff-rank(structureless) > eff-rank(rank-2) by a clear margin AND monotone in r.")
    print("=" * 82)

    # structured levels
    b_vals, e_vals = [], []
    for r in R_LEVELS:
        b, e = seed_avg(lambda sd, r=r: planted_clouds(r, seed=sd))
        b_vals.append(b); e_vals.append(e)

    # structureless nulls (seed passed by keyword; positional would collide with the K argument)
    b_iid, e_iid = seed_avg(lambda sd: structureless_iid(seed=sd))
    b_shuf, e_shuf = seed_avg(lambda sd: structureless_shuffle(seed=sd))

    # ---------------------------------------------------------------- table
    hdr = "  r            | " + " | ".join(f"{r:>6d}" for r in R_LEVELS) + " |  Spearman(vs r)"
    print("\n" + hdr)
    print("  " + "-" * (len(hdr) - 2))
    rho_b = spearman(R_LEVELS, b_vals)
    rho_e = spearman(R_LEVELS, e_vals)
    print("  (b) D_e eff-rank m=12 | " + " | ".join(f"{v:6.2f}" for v in b_vals) + f" |   {rho_b:+.2f}")
    print("  (e) Gram eff-rank     | " + " | ".join(f"{v:6.2f}" for v in e_vals) + f" |   {rho_e:+.2f}")

    print("\n  structureless matched-moment nulls (should read HIGH — at the TOP of the meter's COMPRESSED"
          " band ~[O, 6.4], NOT near m=%d; entropic smoothing pulls the ceiling far below the budget):" % M)
    print(f"    (i)  iid Gaussian N(0,I_d), d={D_AMB} : D_e eff-rank = {b_iid:6.2f}   (Gram eff-rank {e_iid:6.2f})")
    print(f"    (ii) per-feature shuffle of r={SHUFFLE_BASE_R}    : D_e eff-rank = {b_shuf:6.2f}   "
          f"(Gram eff-rank {e_shuf:6.2f})")

    # ---------------------------------------------------------------- offset O
    O = b_vals[0]                                  # eff-rank at r=1
    b_r2 = b_vals[1]                               # eff-rank at r=2 (the 4.67 reference)
    O_in_band = (3.0 <= O <= 6.0)
    print("\n  OFFSET  O = eff-rank(r=1) = %.2f   (prereg predicts O in [3,6]; r=2 reference was ~4.67)" % O)
    print("    -> O %s [3,6]; r=2 here reads %.2f  %s" % (
        "IS in" if O_in_band else "is NOT in", b_r2,
        "(consistent with the known 4.67 offset)" if abs(b_r2 - 4.67) < 1.0 else "(differs from 4.67)"))

    # ---------------------------------------------------------------- margins
    margin_iid = b_iid - b_r2
    margin_shuf = b_shuf - b_r2
    print("\n  MARGIN structureless - rank2 (D_e eff-rank):")
    print(f"    iid Gaussian : {margin_iid:+.2f}   %s" %
          ("clear" if margin_iid >= CLEAR_MARGIN else "NOT clear"))
    print(f"    shuffle      : {margin_shuf:+.2f}   %s" %
          ("clear" if margin_shuf >= CLEAR_MARGIN else "NOT clear"))

    # ---------------------------------------------------------------- monotonicity
    diffs = np.diff(b_vals)
    max_dip = float(-diffs.min()) if diffs.min() < 0 else 0.0
    mono_ok = (rho_b >= 0.90) and (max_dip <= MONO_NOISE)
    print("\n  MONOTONICITY of (b) in r:")
    print("    adjacent diffs = " + " ".join(f"{d:+.2f}" for d in diffs))
    print("    Spearman = %+.2f (prereg B-convention >=0.90); max adjacent dip = %.2f (noise floor %.2f)"
          % (rho_b, max_dip, MONO_NOISE))
    print("    -> monotone non-decreasing: %s" % ("YES" if mono_ok else "NO"))

    # ---------------------------------------------------------------- verdict
    struct_high = (margin_iid >= CLEAR_MARGIN) and (margin_shuf >= CLEAR_MARGIN)
    b2_pass = struct_high and mono_ok
    print("\n" + "=" * 82)
    print("B2 VERDICT")
    print("=" * 82)
    print("  structureless reads HIGH (both nulls > rank2 by >= %.1f): %s" % (CLEAR_MARGIN, struct_high))
    print("  eff-rank monotone in r                                 : %s" % mono_ok)
    print("  offset O = %.2f in [3,6] (explains 4.67)                : %s" % (O, O_in_band))
    print("  -> B2 %s" % ("PASS" if b2_pass else "FAIL"))
    if not b2_pass:
        print("  (a FAIL kills the meter — reported honestly, no patching.)")
    print("=" * 82)

    return {
        "b_vals": b_vals, "e_vals": e_vals, "rho_b": rho_b,
        "b_iid": b_iid, "b_shuf": b_shuf, "O": O, "b_r2": b_r2,
        "margin_iid": margin_iid, "margin_shuf": margin_shuf,
        "mono_ok": mono_ok, "struct_high": struct_high, "b2_pass": b2_pass,
    }


if __name__ == "__main__":
    run()
