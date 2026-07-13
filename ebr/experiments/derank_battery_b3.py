"""
experiments/derank_battery_b3.py — trust-battery leg B3 (audit classification: experimental CONTROL /
operating-point robustness of an INSTRUMENT, not a mechanism change). This is "the test its predecessor
failed": the FW active-atom-count meter INVERTED / FLATTENED under exactly this sweep (its Spearman went
negative / to ~0), which is what killed it as the spatial-complexity meter. B3 asks whether the candidate
replacement — the equilibrated anchor's D_e EFFECTIVE RANK at fixed budget m=12 — keeps its monotone ORDER
when the operating point moves.

NOTHING here is new mechanism and NO new QC is introduced. We reuse verbatim from atom_observable_search.py:
planted_clouds() (the planted continuous rank-r substrate), eff_rank() (participation-ratio effective count),
spearman() (Pearson-on-ranks, which already returns 0.0 for a zero-variance / flat readout — reused exactly,
not re-guarded), and equilibrate_fixed_m() (the (b) D_e eff-rank readout via energy.equilibrate at fixed m).

WHAT B3 PREDICTS (from ebr/PREREG_derank_battery.md, section B3, to the digit): the monotonic order survives
operating-point changes. The absolute eff-rank values MAY shift (the offset O moves with eps/n) but the SIGN
and ordering must not invert or flatten. Concretely: Spearman(D_e eff-rank, r) >= 0.90 at EVERY cell.

SWEEP GRID (mirrors the prereg exactly; each choice defended in one line):
    * eps in {0.02, 0.05, 0.10} — the entropic-regularization operating points named in the prereg / the exact
      eps values the atom-count predecessor was swept over (atom_operating_point.py EPS_GRID).
    * n in {80, 160} — cloud size (points/member); the prereg's n sweep and atom_operating_point.py N_GRID.
    * r in R_LEVELS = {2,3,4,6,8} — the fixed planted ranks from atom_observable_search.py (ground truth).
    * m = 12 — the fixed generous anchor budget of the (b) readout (spec cap m<=12).
    * SEEDS = [0,1,2] — seed-averaged to beat solver noise, matching atom_observable_search.py.
    * n_outer convergence check: one cell is recomputed at a HIGHER n_outer to confirm the ordering is stable
      under more equilibration (not an artifact of under-convergence).

PASS RULE (verbatim from the prereg): PASS iff Spearman >= 0.90 across ALL cells. A SINGLE cell below 0.90
fails the leg — and a failed leg kills the meter (reported dead like its predecessor, no patching). There is
no cherry-picking of cells, no shim, no special-casing: every one of the six cells is reported straight, and
the overall verdict is min(Spearman over cells) >= 0.90.

Run: python -m ebr.experiments.derank_battery_b3
"""
import numpy as np

from .atom_observable_search import (
    R_LEVELS,
    SEEDS,
    M_MAX,
    N_OUTER,
    planted_clouds,
    eff_rank,
    spearman,
    equilibrate_fixed_m,
)

# ---- operating-point sweep grid (mirrors PREREG_derank_battery.md section B3) -----------------------------
EPS_GRID = (0.02, 0.05, 0.10)
N_GRID = (80, 160)
M_FIXED = M_MAX               # fixed generous anchor budget of the (b) D_e eff-rank readout (m=12)
PASS_THRESHOLD = 0.90         # prereg: PASS iff Spearman >= 0.90 across ALL cells

# n_outer convergence check: recompute ONE cell at a higher n_outer than the sweep default.
CHECK_CELL = (0.05, 160)      # a mid-eps, larger-n cell
N_OUTER_HIGH = 2 * N_OUTER    # 24 (> the sweep's N_OUTER=12) — confirms ordering is not an under-convergence artifact


def derank_for_cell(eps, n, n_outer=N_OUTER):
    """Seed-averaged D_e effective rank at m=M_FIXED for each planted r, at operating point (eps, n).

    Reuses the (b) readout verbatim: equilibrate a fixed-m anchor, then eff_rank of its singular values.
    Returns the per-r eff-rank vector aligned to R_LEVELS.
    """
    vals = []
    for r in R_LEVELS:
        seed_ranks = []
        for sd in SEEDS:
            Ds, ws = planted_clouds(r, n=n, seed=sd)
            _pis, De, _a = equilibrate_fixed_m(Ds, ws, M_FIXED, eps=eps, n_outer=n_outer, seed=sd)
            sv = np.linalg.svd(De, compute_uv=False)
            seed_ranks.append(eff_rank(sv))
        vals.append(float(np.mean(seed_ranks)))
    return vals


def run():
    print("=" * 96)
    print("B3 — OPERATING-POINT SWEEP (the test its predecessor failed): does D_e eff-rank keep its ORDER?")
    print(f"  D_e effective rank at fixed m={M_FIXED}; planted ranks r in {R_LEVELS}; seeds={SEEDS}, n_outer={N_OUTER}.")
    print(f"  sweep eps in {list(EPS_GRID)} x n in {list(N_GRID)} (six cells).")
    print(f"  PASS iff Spearman(D_e eff-rank, r) >= {PASS_THRESHOLD:.2f} at EVERY cell; a single cell below "
          f"{PASS_THRESHOLD:.2f} FAILS the leg.")
    print(f"  contrast: the FW atom-count predecessor INVERTED / FLATTENED under this exact sweep.")
    print("=" * 96)

    header = ("  cell (eps, n)   | " + " | ".join(f"r={r:<4d}" for r in R_LEVELS)
              + " |  Spearman | cell")
    print("\n" + header)
    print("  " + "-" * (len(header) - 2))

    cell_rhos = {}
    for eps in EPS_GRID:
        for n in N_GRID:
            vals = derank_for_cell(eps, n)
            rho = spearman(vals, R_LEVELS)
            cell_rhos[(eps, n)] = rho
            cells = " | ".join(f"{v:6.2f}" for v in vals)
            tag = "PASS" if rho >= PASS_THRESHOLD else "FAIL"
            print(f"  eps={eps:<4} n={n:<4} | {cells} |  {rho:+.3f}  | {tag}")

    # ---------------------------------------------------- n_outer convergence check (one cell, higher n_outer)
    ceps, cn = CHECK_CELL
    vals_hi = derank_for_cell(ceps, cn, n_outer=N_OUTER_HIGH)
    rho_hi = spearman(vals_hi, R_LEVELS)
    rho_lo = cell_rhos[CHECK_CELL]
    print("\n  n_outer convergence check (ordering stability under more equilibration):")
    print(f"    cell (eps={ceps}, n={cn}):  n_outer={N_OUTER} -> Spearman {rho_lo:+.3f}   |   "
          f"n_outer={N_OUTER_HIGH} -> Spearman {rho_hi:+.3f}")
    print(f"    higher-n_outer eff-rank vector: " + " ".join(f"{v:.2f}" for v in vals_hi))
    conv_ok = rho_hi >= PASS_THRESHOLD
    print(f"    ordering at higher n_outer {'STABLE (>= %.2f)' % PASS_THRESHOLD if conv_ok else 'DROPS below %.2f' % PASS_THRESHOLD}"
          f" — {'not an under-convergence artifact.' if conv_ok else 'CONVERGENCE CONCERN.'}")

    # ---------------------------------------------------- overall verdict
    min_rho = min(cell_rhos.values())
    min_cell = min(cell_rhos, key=cell_rhos.get)
    verdict = "PASS" if min_rho >= PASS_THRESHOLD else "FAIL"

    print("\n" + "=" * 96)
    print("B3 VERDICT")
    print("=" * 96)
    print(f"  min Spearman over the six cells = {min_rho:+.3f}  at cell eps={min_cell[0]}, n={min_cell[1]}")
    if verdict == "PASS":
        print(f"  ALL cells >= {PASS_THRESHOLD:.2f}: the D_e eff-rank ORDER survives the operating-point sweep.")
        print("  Unlike the FW atom-count predecessor, it does NOT invert or flatten. B3 PASSES.")
    else:
        print(f"  cell eps={min_cell[0]}, n={min_cell[1]} is BELOW {PASS_THRESHOLD:.2f}: the order failed at an "
              f"operating point.")
        print("  A single failing cell kills the meter — reported dead, no patching. B3 FAILS.")
    print(f"\n  B3 = {verdict}   (PASS iff min Spearman over cells >= {PASS_THRESHOLD:.2f})")
    print("=" * 96)

    return {"cell_rhos": cell_rhos, "min_rho": min_rho, "min_cell": min_cell,
            "convergence": {"cell": CHECK_CELL, "rho_lo": rho_lo, "rho_hi": rho_hi},
            "verdict": verdict}


if __name__ == "__main__":
    run()
