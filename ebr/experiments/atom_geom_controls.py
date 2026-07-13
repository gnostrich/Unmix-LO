"""
experiments/atom_geom_controls.py — EXPERIMENTAL CONTROL (null-hypothesis probe), not a mechanism claim.

QUESTION (honest): does the Frank-Wolfe active-atom count (events/frankwolfe.py: grow, an F-DRIVEN
conditional-gradient mechanism) track ANY planted GEOMETRIC parameter of a shared point cloud? A prior leg
(WALL_2x2_atomleg.md) found the count flat (~3) on encoder clouds, and a discrete-rank control gave
rank 2/4/6 -> atoms 3,3,3. This file pushes the same question on PURELY SYNTHETIC clouds (numpy only, no
encoder/model/library) with three DISTINCT geometry knobs, from m=1, and reports what actually happens.

CONTROL DISCIPLINE (audit protocol):
  * grow() is used verbatim as the F-driven mechanism. NO Hankel / rank / spectral trigger is added — the
    atom count is whatever strict F-descent self-quenches to. If it does not move with the knob, that is the
    finding (a clean null), and it is reported as "flat", NOT patched.
  * Undefended constants: the two rel_tol values are STATED and are the experimental variable
    (0.02 = frankwolfe's registered default acceptance floor; 0.05 = a stricter floor, i.e. demand each atom
    pay more before it is accepted). N and max_atoms are a CPU budget (N<=120), not tuned to a result.
  * Knobs are NOT tuned to force tracking. Same N, same ambient dim, same noise across a knob's sweep; only
    the planted geometric parameter changes.

THREE KNOBS (K=3 members per value; each member = an independent random cloud realization sharing the same
planted structure, so FW pools a shared geometry across members):
  1. CONTINUOUS rank r  : points ~ N(0, I_r) pushed through a FIXED random r->d linear map (an r-dim
                          continuous manifold in d-space). r = 2,4,6,8.
  2. DISCRETE clusters k: k orthogonal, equal-scale, equidistant cluster centers; points round-robin assigned
                          + isotropic noise. k = 2,4,6,8.
  3. HIERARCHICAL g     : balanced binary hierarchy of depth g embedded with g orthogonal level-directions at
                          GEOMETRICALLY separated scales -> g distinct pairwise-distance scales. g = 2,3,4.

Run: python -m ebr.experiments.atom_geom_controls
"""
import numpy as np

from ..geometry.clouds import cloud_to_Dw
from ..events import frankwolfe as FW

# ---- CPU budget (defended as budget, not tuned to a result) ------------------------------------------
N = 96            # points per member cloud (<=120)
D_AMB = 20        # ambient dimension the synthetic clouds live in (fixed across every sweep)
K = 3             # members per knob value (shared-structure pooling)
NOISE = 0.05      # isotropic per-point jitter, identical across every sweep
MAX_ATOMS = 10    # growth ceiling (budget)
N_OUTER = 12      # equilibration outer iters (<=14, budget)
REL_TOLS = (0.02, 0.05)   # acceptance floors: default (registered) and stricter


# ---- synthetic cloud generators (numpy only) ---------------------------------------------------------
def _orthobasis(d, k, seed):
    """k orthonormal directions in R^d (columns), from a random orthogonal frame."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    return Q[:, :k]


def cloud_rank(r, member_seed, map_seed=1000):
    """Knob 1 — continuous rank r. Z ~ N(0, I_r) mapped by a FIXED random r x d map (shared across members
    of this r). Independent Z per member -> a shared r-dim continuous manifold, different sample each member."""
    M = np.random.default_rng(map_seed + r).normal(size=(r, D_AMB)) / np.sqrt(r)
    Z = np.random.default_rng(member_seed).normal(size=(N, r))
    return Z @ M


def cloud_clusters(k, member_seed):
    """Knob 2 — k discrete clusters. Centers = k orthogonal equal-scale unit directions (mutually
    equidistant); points round-robin assigned, isotropic noise. Independent noise per member."""
    C = _orthobasis(D_AMB, k, seed=2000)          # shared centers across members of this k
    rng = np.random.default_rng(member_seed)
    idx = np.arange(N) % k                          # round-robin assignment
    X = C[:, idx].T.copy()                          # (N, d) each row a center
    X += NOISE * rng.normal(size=X.shape)
    return X


def cloud_hier(g, member_seed):
    """Knob 3 — hierarchical, g distinct distance-scales. Balanced binary hierarchy of depth g: g orthogonal
    level-directions at GEOMETRICALLY separated scales rho**level, so level splits sit at g separated radii.
    Each point gets a g-bit address (round-robin over the 2**g leaves); coordinate = sum_l (b_l-0.5)*scale_l*u_l."""
    U = _orthobasis(D_AMB, g, seed=3000)            # shared level directions across members of this g
    rho = 3.0                                       # geometric scale separation between levels
    scales = rho ** np.arange(g)[::-1]              # level 0 (top split) widest
    rng = np.random.default_rng(member_seed)
    addr = np.arange(N) % (2 ** g)                  # round-robin over leaves
    bits = ((addr[:, None] >> np.arange(g)[None, :]) & 1).astype(float)  # (N, g)
    X = ((bits - 0.5) * scales[None, :]) @ U.T      # (N, d)
    X += NOISE * rng.normal(size=X.shape)
    return X


# ---- driver ------------------------------------------------------------------------------------------
def _members(gen, value):
    """Build K independent member clouds for one knob value -> (Ds, ws)."""
    Ds, ws = [], []
    for m in range(K):
        X = gen(value, member_seed=10 * value + m)
        D, w = cloud_to_Dw(X)
        Ds.append(D); ws.append(w)
    return Ds, ws


def _atoms(Ds, ws, rel_tol):
    res = FW.grow(Ds, ws, np.array([[0.0]]), np.array([1.0]), np.array([1.0]),
                  max_atoms=MAX_ATOMS, n_outer=N_OUTER, rel_tol=rel_tol)
    return res["active"], res["n_atoms"]


def _verdict(values, active_by_tol):
    """Honest classification from the DEFAULT-tol (rel_tol=0.02) active counts.
    'tracks'  : non-decreasing AND spans >=2 distinct values across the sweep (roughly follows the knob).
    'flat'    : identical count at every knob value (operating-point / floor dominated).
    'other'   : moves but not monotone -> describe."""
    a = active_by_tol[REL_TOLS[0]]
    if len(set(a)) == 1:
        return f"flat (active == {a[0]} at every value; operating-point/floor-dominated, knob ignored)"
    mono = all(a[i + 1] >= a[i] for i in range(len(a) - 1))
    if mono and (a[-1] - a[0]) >= 1:
        return f"tracks (non-decreasing {a[0]}->{a[-1]} across the sweep)"
    return f"other (non-monotone / weak: {a} — moves but does not cleanly follow the knob)"


def sweep(name, gen, values):
    print(f"\n== KNOB: {name} ==  (values {list(values)}; N={N}, d={D_AMB}, K={K})")
    header = "  value |" + "".join(f"  rel_tol={t}  " for t in REL_TOLS)
    print(header)
    print("        |" + "".join("  active/total " for _ in REL_TOLS))
    active_by_tol = {t: [] for t in REL_TOLS}
    for v in values:
        Ds, ws = _members(gen, v)
        cells = []
        for t in REL_TOLS:
            act, tot = _atoms(Ds, ws, t)
            active_by_tol[t].append(act)
            cells.append(f"   {act}/{tot}      ")
        print(f"   {v:>3}  |" + "".join(cells))
    verdict = _verdict(values, active_by_tol)
    print(f"  VERDICT [{name}]: {verdict}")
    return {"values": list(values), "active_by_tol": active_by_tol, "verdict": verdict}


def run():
    print("FW atom-count vs planted GEOMETRY — synthetic-cloud CONTROL (grow() F-driven, no rank trigger).")
    print("Reading a null honestly: if 'active' does not move with the knob, that IS the result.")
    out = {}
    out["rank"] = sweep("continuous rank r", cloud_rank, (2, 4, 6, 8))
    out["clusters"] = sweep("discrete clusters k", cloud_clusters, (2, 4, 6, 8))
    out["hier"] = sweep("hierarchical depth g", cloud_hier, (2, 3, 4))
    print("\n== SUMMARY (honest per-knob verdicts, rel_tol=0.02 default) ==")
    for k, v in out.items():
        print(f"  {k:9s}: {v['verdict']}")
    return out


if __name__ == "__main__":
    run()
