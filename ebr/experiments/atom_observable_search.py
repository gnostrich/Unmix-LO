"""
experiments/atom_observable_search.py — INSTRUMENT SEARCH (audit classification: experimental CONTROL /
instrument search; NONE of the readouts below drives mechanism — they only MEASURE a planted axis).

CONTEXT (the wall being interrogated): WALL_2x2_atomleg.md reports that the Frank–Wolfe active-atom count
does NOT track planted geometric complexity — it sits at ~3, dominated by the (eps, rel_tol) operating point,
independent of the geometry. The spec's "atom count = spatial complexity of the shared geometry" claim
therefore has no instrument. The wall's option (1) names an UNTRIED principled candidate: the equilibrium GW
*transport cost* as a function of anchor budget m — a rate–distortion curve whose KNEE could define spatial
complexity. This file tests that candidate head-on, alongside the other F-/geometry-derived readouts.

THE HONEST QUESTION this file answers: is there ANY F-derived (or geometry-derived) readout that cleanly
tracks the ground-truth complexity r of a planted continuous rank-r cloud — and in particular does the
rate–distortion KNEE work? If the knee tracks r, that is the real spatial-complexity meter the spec needs.

SUBSTRATE (planted, continuous, ground truth = r):
    For each r in {2,3,4,6,8}, K=3 members. Each member is an INDEPENDENT realization: n points ~ N(0, I_r)
    pushed through a FIXED random r x d linear map (its own draw per member). The intrinsic dimension of every
    member's cloud is exactly r, so the ground-truth spatial complexity is r and is present in the geometry
    (readout (e) is the sanity check that it is).

READOUTS (each tested for MONOTONE tracking of r; reported with a Spearman rank correlation vs r):
    (a) BASELINE NULL — FW active atom count, grown from m=1 (the wall's suspect; expected NOT to track).
    (b) Equilibrated anchor D_e EFFECTIVE RANK at fixed generous m=12 (participation ratio exp(entropy) of the
        singular-value distribution of D_e), via energy.equilibrate.
    (c) INDUCED anchor-mass participation ratio (exp(entropy) of the equilibrium column marginal a) at m=12.
    (d) RATE–DISTORTION KNEE — equilibrate a fixed-m anchor for m = 1..12, record equilibrium total GW
        transport cost D(m) = sum_v gw_cost(D_v, D_e, pi_v). D(m) decreases in m; the KNEE is the elbow.
    (e) PURE-GEOMETRY baseline — effective rank of the cloud's double-centered Gram directly (must track r;
        proves r is recoverable from the geometry at all, independent of F).

DEFENDED CONSTANTS (no undefended magic; no shims):
    * m range for the RD curve: m = 1..12 (spec cap m<=12; m=1 is the degenerate single-anchor lower budget).
    * KNEE RULE (stated explicitly, Kneedle/elbow): on the curve (m, D(m)), m=1..M, take the chord from
      (1, D(1)) to (M, D(M)); the knee is the m of MAXIMUM perpendicular distance from the curve to that chord.
      This is the standard elbow rule; no free threshold.
    * effective rank = exp(Shannon entropy of the nonneg spectrum normalized to a probability vector) —
      a smooth participation-ratio-style count, = k for k equal components, = 1 for a rank-1 spectrum.
    * TRACKS-r verdict: Spearman(readout, r) >= 0.90 AND value strictly larger at r=8 than at r=2. Otherwise
      the readout does NOT track r. (Spearman on 5 planted levels; averaged over cloud seeds to beat solver
      noise.)
    * sizes: n=100 points, d=16 ambient (> max r), K=3, n_outer<=12, eps=0.05, SEEDS averaged for stability.

Run: python -m ebr.experiments.atom_observable_search
"""
import numpy as np

from ..geometry.clouds import cloud_to_Dw
from ..geometry import gram as GR
from ..energy import functional as EN
from ..transport import gw
from ..events import frankwolfe as FW

R_LEVELS = [2, 3, 4, 6, 8]
N = 100
D_AMB = 16
K = 3
M_MAX = 12
N_OUTER = 12
EPS = 0.05
SEEDS = [0, 1, 2]


# ----------------------------------------------------------------------------- substrate + small numerics
def planted_clouds(r, K=K, n=N, d=D_AMB, seed=0):
    """K independent continuous rank-r members: N(0,I_r) mapped by a fixed random r x d map (own draw each)."""
    rng = np.random.default_rng(seed * 131 + r)
    Ds, ws = [], []
    for _k in range(K):
        Z = rng.normal(size=(n, r))              # N(0, I_r)
        Wmap = rng.normal(size=(r, d))           # fixed random r x d embedding (member-specific)
        D, w = cloud_to_Dw(Z @ Wmap)
        Ds.append(D)
        ws.append(w)
    return Ds, ws


def eff_rank(vals):
    """Participation-ratio effective count: exp(Shannon entropy of the nonneg spectrum as a distribution)."""
    v = np.clip(np.asarray(vals, dtype=np.float64), 0.0, None)
    s = v.sum()
    if s <= 0:
        return 1.0
    p = v / s
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def spearman(x, y):
    """Spearman rank correlation (no scipy): Pearson on the ranks. Handles the 5-level monotonicity check."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / den) if den > 0 else 0.0


# ----------------------------------------------------------------------------- fixed-m equilibration helper
def equilibrate_fixed_m(Ds, ws, m, eps=EPS, n_outer=N_OUTER, seed=0):
    """Equilibrate ONE shared anchor of fixed budget m across the members. Returns (pis, De, a)."""
    if m == 1:
        De0 = np.array([[0.0]])                  # single atom: zero self-distance
    else:
        rng = np.random.default_rng(9000 + seed * 17 + m)
        De0, _ = cloud_to_Dw(rng.normal(size=(m, 8)))   # random valid normalized anchor cost
    a0 = np.full(m, 1.0 / m)
    abar = np.full(m, 1.0 / m)
    pis, De, a, _ftr, _conv = EN.equilibrate(Ds, ws, De0, a0, abar, eps=eps, n_outer=n_outer)
    return pis, De, a


def total_transport_cost(Ds, De, pis):
    """D(m): equilibrium total GW transport cost, summed over members."""
    return float(sum(gw.gw_cost(D, De, pi) for D, pi in zip(Ds, pis)))


def knee_m(ms, Dvals):
    """Kneedle/elbow: m of MAX perpendicular distance from the curve (m, D(m)) to the chord (m1,D1)->(mM,DM)."""
    ms = np.asarray(ms, float); Dv = np.asarray(Dvals, float)
    p1 = np.array([ms[0], Dv[0]]); pM = np.array([ms[-1], Dv[-1]])
    v = pM - p1
    vn = v / (np.linalg.norm(v) + 1e-12)
    dist = []
    for i in range(len(ms)):
        w = np.array([ms[i], Dv[i]]) - p1
        proj = w - (w @ vn) * vn                 # component perpendicular to the chord
        dist.append(np.linalg.norm(proj))
    return int(ms[int(np.argmax(dist))]), np.asarray(dist)


# ----------------------------------------------------------------------------- readouts (per r, seed-averaged)
def readouts_for_r(r):
    """Compute all five readouts for planted complexity r, averaged over SEEDS."""
    a_atoms, b_derank, c_amass, e_gram = [], [], [], []
    rd_curves = []
    for sd in SEEDS:
        Ds, ws = planted_clouds(r, seed=sd)

        # (a) FW active atom count grown from m=1
        res = FW.grow(Ds, ws, np.array([[0.0]]), np.array([1.0]), np.array([1.0]),
                      eps=EPS, max_atoms=M_MAX, n_outer=N_OUTER)
        a_atoms.append(res["active"])

        # (b) & (c): one generous fixed-m anchor at m = M_MAX
        pis, De, aeq = equilibrate_fixed_m(Ds, ws, M_MAX, seed=sd)
        sv = np.linalg.svd(De, compute_uv=False)
        b_derank.append(eff_rank(sv))
        c_amass.append(eff_rank(aeq))

        # (e) pure-geometry: effective rank of each member's double-centered Gram, member-averaged
        grr = []
        for D in Ds:
            ev = np.linalg.eigvalsh(GR.gram_from_D(D))
            grr.append(eff_rank(ev))
        e_gram.append(np.mean(grr))

        # (d) rate-distortion curve D(m), m = 1..M_MAX
        curve = []
        for m in range(1, M_MAX + 1):
            pm, Dem, _am = equilibrate_fixed_m(Ds, ws, m, seed=sd)
            curve.append(total_transport_cost(Ds, Dem, pm))
        rd_curves.append(curve)

    rd_mean = np.mean(np.array(rd_curves), axis=0)
    km, _dist = knee_m(list(range(1, M_MAX + 1)), rd_mean)
    return {
        "a_atoms": float(np.mean(a_atoms)),
        "b_derank": float(np.mean(b_derank)),
        "c_amass": float(np.mean(c_amass)),
        "d_knee": float(km),
        "d_curve": rd_mean,
        "e_gram": float(np.mean(e_gram)),
    }


def _verdict(vals):
    rho = spearman(R_LEVELS, vals)
    tracks = (rho >= 0.90) and (vals[-1] > vals[0])
    return rho, tracks


def run():
    print("=" * 78)
    print("ATOM-OBSERVABLE SEARCH — does ANY F-/geometry-derived readout track planted r?")
    print(f"  planted continuous rank-r clouds, r in {R_LEVELS}, K={K}, n={N}, d={D_AMB},")
    print(f"  seeds={SEEDS}, eps={EPS}, n_outer={N_OUTER}, m<= {M_MAX}. (instrument search; none is mechanism)")
    print("=" * 78)

    rows = {r: readouts_for_r(r) for r in R_LEVELS}

    def col(key):
        return [rows[r][key] for r in R_LEVELS]

    header = "  r          | " + " | ".join(f"{r:>6d}" for r in R_LEVELS) + " |  Spearman  verdict"
    print("\n" + header)
    print("  " + "-" * (len(header) - 2))

    lines = [
        ("(a) FW active atoms   [NULL]", "a_atoms", "%6.2f"),
        ("(b) D_e eff-rank m=12       ", "b_derank", "%6.2f"),
        ("(c) anchor-mass PR   m=12   ", "c_amass", "%6.2f"),
        ("(d) rate-distortion knee-m  ", "d_knee", "%6.0f"),
        ("(e) Gram eff-rank [geometry]", "e_gram", "%6.2f"),
    ]
    verdicts = {}
    for label, key, fmt in lines:
        vals = col(key)
        rho, tracks = _verdict(vals)
        verdicts[key] = (rho, tracks)
        cells = " | ".join(fmt % v for v in vals)
        tag = "TRACKS r" if tracks else "does NOT"
        print(f"  {label} | {cells} |  {rho:+.2f}     {tag}")

    print("\n  rate-distortion curves D(m), m=1..12 (seed-averaged; knee = max dist to chord):")
    for r in R_LEVELS:
        cv = rows[r]["d_curve"]
        print(f"    r={r}: " + " ".join(f"{x:5.2f}" for x in cv) + f"   -> knee m={int(rows[r]['d_knee'])}")

    # ------------------------------------------------------------------- honest headline
    print("\n" + "=" * 78)
    print("HEADLINE")
    print("=" * 78)
    geom_ok = verdicts["e_gram"][1]
    knee_ok = verdicts["d_knee"][1]
    f_derived_ok = [k for k in ("a_atoms", "b_derank", "c_amass", "d_knee") if verdicts[k][1]]

    if geom_ok:
        print("  (e) pure-geometry Gram eff-rank TRACKS r: the complexity r IS present and recoverable")
        print("      from the raw cloud geometry (sanity holds).")
    else:
        print("  (e) pure-geometry Gram eff-rank does NOT track r: r is not even cleanly present in the")
        print("      geometry as generated — treat all F-derived nulls below as inconclusive.")

    if knee_ok:
        print("  (d) THE RATE-DISTORTION KNEE TRACKS r. This is the win: the equilibrium GW transport-cost")
        print("      curve's elbow is a faithful F-derived spatial-complexity meter (wall option 1 succeeds).")
    else:
        print("  (d) the rate-distortion knee does NOT track r cleanly — the elbow is operating-point / floor")
        print("      dominated just like the atom count; wall option 1 (RD knee) FAILS on these clouds.")

    if f_derived_ok:
        print(f"  F-derived readouts that track r (monotone, Spearman>=0.90): {f_derived_ok}.")
        if verdicts["b_derank"][1]:
            bv = col("b_derank")
            print("  >> (b) equilibrated anchor D_e EFFECTIVE RANK is a faithful F-derived spatial-complexity")
            print(f"     meter: monotone in r ({bv[0]:.2f}->{bv[-1]:.2f}). CAVEAT: absolute value is compressed")
            print("     and inflated at low r (the fixed m=12 budget forces spread) — it tracks r ordinally,")
            print("     it does NOT recover r on the nose. This, not the atom count or the knee, is the win.")
    else:
        print("  NO F-derived readout (a,b,c,d) tracks r: spatial complexity is present in the geometry (e)")
        print("  but the equilibrated-anchor / FW readouts do not expose it. The atom leg stays a null.")
    print("=" * 78)

    return {"rows": rows, "verdicts": verdicts}


if __name__ == "__main__":
    run()
