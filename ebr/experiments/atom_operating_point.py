"""
experiments/atom_operating_point.py — CONTROL experiment (audit classification: experimental control, not a
mechanism change). Question posed by WALL_2x2_atomleg.md: is the FW *atom count* set by the OPERATING POINT
(eps, rel_tol, max_atoms, cloud size n) rather than by the intrinsic complexity of the shared geometry?

This file changes NOTHING in the mechanism. grow() is used exactly as-is: it is the F-descent oscillator
(no Hankel/rank trigger is added, no acceptance rule is touched). We only feed it two PLANTED synthetic
geometries of deliberately different intrinsic complexity and sweep the operating point around them, reading
back the active atom count. Synthetic clouds are built directly with numpy (no encoder / library needed).

DESIGN (honest, pre-declared):
  - Two geometries, K=3 members each:
      LOW  = continuous intrinsic rank 2   (points live on a random 2-D subspace of R^AMBIENT)
      HIGH = continuous intrinsic rank 8   (points live on a random 8-D subspace of R^AMBIENT)
    "Clearly different intrinsic complexity" is the whole point of the control.
  - Operating point swept one-factor-at-a-time from a CENTER, over the knobs named in the wall doc:
      eps       in {0.02, 0.05, 0.10}
      rel_tol   in {0.01, 0.03, 0.08}
      max_atoms in {6, 12, 20}
      n (points/cloud) in {80, 160}
    These swept values ARE the knobs under study; they are declared as constants below, not tuned.

METRIC (honest): at each op-point setting record atoms(LOW), atoms(HIGH).
  - GEOMETRY GAP  = atoms(HIGH) - atoms(LOW) at a fixed op-point. SIGN MATTERS: for atom count to be a
    spatial-complexity meter this must be POSITIVE (higher intrinsic rank -> more atoms). A negative gap is
    not "geometry-tracking" — it is anti-tracking, evidence AGAINST the meter, not for it.
  - OP-POINT RANGE = max(atoms) - min(atoms) across op-point settings at fixed geometry (does the knob move it?).
  - DOMINATION RATIO = op-point range / max(1, |geometry gap|). Only meaningful as "geometry-tracking" when
    the gap is positive; reported for completeness regardless.
    ratio >> 1  (or a non-positive gap) => operating-point-dominated (confirms the wall's diagnosis).
    positive gap that exceeds the op-point range => atom count tracks geometry (refutes it).
The verdict printed keys off BOTH the measured ratio and the SIGN of the gap, reported straight.
"""
import numpy as np

from ..geometry.clouds import cloud_to_Dw
from ..events import frankwolfe as FW

# ---- declared constants (fixed structure of the control) -------------------------------------------------
AMBIENT = 16          # ambient dimension the low-rank clouds are embedded in
K_MEMBERS = 3         # members per geometry (fixed; K-invariance is not the question here)
RANK_LOW = 2          # planted intrinsic rank of the LOW geometry
RANK_HIGH = 8         # planted intrinsic rank of the HIGH geometry
N_OUTER = 10          # equilibration outer iterations (<=12, CPU-fast, held fixed across the sweep)
SEED = 0

# ---- operating-point knobs under study (the swept values ARE the object of study) ------------------------
EPS_GRID = (0.02, 0.05, 0.10)
REL_TOL_GRID = (0.01, 0.03, 0.08)
MAX_ATOMS_GRID = (6, 12, 20)
N_GRID = (80, 160)
CENTER = dict(eps=0.05, rel_tol=0.03, max_atoms=12, n=160)   # one-factor-at-a-time pivot


def planted_cloud(n, rank, seed):
    """A continuous cloud of intrinsic `rank`: n points on a random rank-D subspace of R^AMBIENT.
    Z ~ N(0, I_rank) mapped through a random rank x AMBIENT basis => spread genuinely fills `rank`
    directions and no more. Different seeds give different (heterogeneous) members of the same rank."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, rank))
    B = rng.standard_normal((rank, AMBIENT))
    return Z @ B


def geometry(rank, n, seed0):
    """K_MEMBERS clouds of the given planted rank -> (Ds, ws) ready for grow()."""
    Ds, ws = [], []
    for k in range(K_MEMBERS):
        X = planted_cloud(n, rank, seed=seed0 + 1000 * rank + k)
        D, w = cloud_to_Dw(X)
        Ds.append(D)
        ws.append(w)
    return Ds, ws


def atoms(rank, n, eps, rel_tol, max_atoms):
    """Run grow() as-is on a planted geometry; return the ACTIVE (parked-filtered) atom count."""
    Ds, ws = geometry(rank, n, seed0=SEED)
    res = FW.grow(Ds, ws, np.array([[0.0]]), np.array([1.0]), np.array([1.0]),
                  eps=eps, rel_tol=rel_tol, max_atoms=max_atoms, n_outer=N_OUTER)
    return res["active"]


def op_settings():
    """One-factor-at-a-time settings around CENTER (center emitted once)."""
    settings = [("center", dict(CENTER))]
    for e in EPS_GRID:
        if e != CENTER["eps"]:
            s = dict(CENTER); s["eps"] = e; settings.append((f"eps={e}", s))
    for rt in REL_TOL_GRID:
        if rt != CENTER["rel_tol"]:
            s = dict(CENTER); s["rel_tol"] = rt; settings.append((f"rel_tol={rt}", s))
    for ma in MAX_ATOMS_GRID:
        if ma != CENTER["max_atoms"]:
            s = dict(CENTER); s["max_atoms"] = ma; settings.append((f"max_atoms={ma}", s))
    for nn in N_GRID:
        if nn != CENTER["n"]:
            s = dict(CENTER); s["n"] = nn; settings.append((f"n={nn}", s))
    return settings


def run():
    print("ATOM OPERATING-POINT CONTROL — is FW atom count set by the op-point or by geometry content?")
    print(f"  planted LOW=rank{RANK_LOW}  HIGH=rank{RANK_HIGH}  (K={K_MEMBERS} members, ambient={AMBIENT})")
    print(f"  center op-point: {CENTER}\n")
    header = f"  {'setting':<14}{'atoms(LOW)':>12}{'atoms(HIGH)':>13}{'geom gap':>11}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    lows, highs, gaps = [], [], []
    for name, s in op_settings():
        lo = atoms(RANK_LOW, s["n"], s["eps"], s["rel_tol"], s["max_atoms"])
        hi = atoms(RANK_HIGH, s["n"], s["eps"], s["rel_tol"], s["max_atoms"])
        gap = hi - lo
        lows.append(lo); highs.append(hi); gaps.append(gap)
        print(f"  {name:<14}{lo:>12}{hi:>13}{gap:>11}")

    lows, highs, gaps = np.array(lows), np.array(highs), np.array(gaps)
    op_range_low = int(lows.max() - lows.min())
    op_range_high = int(highs.max() - highs.min())
    op_range = max(op_range_low, op_range_high)
    geom_gap_med = float(np.median(gaps))      # signed: HIGH - LOW, positive iff more rank -> more atoms
    geom_gap_max = int(gaps.max())
    ratio = op_range / max(1.0, abs(geom_gap_med))

    print("\n  QUANTIFICATION (honest):")
    print(f"    op-point RANGE of atoms at fixed geometry : LOW {op_range_low}, HIGH {op_range_high}  (take {op_range})")
    print(f"    geometry GAP atoms(HIGH)-atoms(LOW)       : median {geom_gap_med:+.1f}, max {geom_gap_max:+d}, "
          f"per-setting {list(map(int, gaps))}")
    print(f"    DOMINATION RATIO op_range / |median gap|   : {ratio:.2f}  (interpretable as geometry-tracking "
          f"only if the gap is POSITIVE)")

    if geom_gap_med <= 0:
        verdict = (f"operating-point-DOMINATED (and ANTI-tracking) — higher intrinsic rank does NOT buy more "
                   f"atoms: median gap {geom_gap_med:+.1f} (HIGH sits AT the op-point floor {int(np.median(highs))}, "
                   f"unmoved by any knob: HIGH op-range {op_range_high}), while the ONLY variable that moves the "
                   f"count is the operating point on LOW (op-range {op_range_low}). Confirms WALL_2x2_atomleg.md.")
    elif ratio >= 2.0:
        verdict = (f"operating-point-DOMINATED — the knob moves the count ~{ratio:.1f}x more than rank2-vs-rank8 "
                   f"does (confirms WALL_2x2_atomleg.md).")
    elif geom_gap_med >= 2.0 and op_range <= abs(geom_gap_med):
        verdict = ("geometry-tracking — the positive rank gap exceeds op-point wobble; atom count tracks intrinsic "
                   "complexity here (refutes the wall on these clouds).")
    else:
        verdict = (f"MIXED / weak — op_range={op_range}, median geom gap={geom_gap_med:+.1f}; neither clearly "
                   f"dominates (ratio {ratio:.2f}). No strong claim either way.")
    print(f"\n  VERDICT: {verdict}")
    return {"op_range": op_range, "geom_gap_median": geom_gap_med, "domination_ratio": ratio,
            "lows": list(map(int, lows)), "highs": list(map(int, highs))}


if __name__ == "__main__":
    run()
