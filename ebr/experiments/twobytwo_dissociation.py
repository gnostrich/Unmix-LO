"""
experiments/twobytwo_dissociation.py — the 2x2 spatial/temporal instrument DOUBLE-DISSOCIATION.

AUDIT CLASSIFICATION: experimental CONTROL / instrument double-dissociation. This file introduces NO new
mechanism, NO new QC, NO shim. It reuses the two ALREADY-CLEARED meters verbatim and runs them on one unified
substrate with two INDEPENDENT knobs, to test whether each meter tracks ONLY its own axis. It is the direct
executable of ebr/PREREG_2x2_dissociation.md (AMENDED, committed at HEAD 6a3aaaa) — grid, knobs, tolerances,
exact-vs-ordering split and verdict clauses are taken from that file to the digit and NOT re-tuned here.

REUSED VERBATIM (imported, not reimplemented):
    from pole_closure:            latent_with_poles, quadratic, estimate_poles, products, closure_error
    from atom_observable_search:  eff_rank, equilibrate_fixed_m, spearman
    from geometry.clouds:         cloud_to_Dw, scramble

THE TWO METERS (each already cleared elsewhere; here only APPLIED, never modified):
    SPATIAL — D_e effective rank at fixed generous m=12 (atom_observable_search readout (b), the one that
        SURVIVED the trust battery, PREREG_derank_battery.md). Reads the order-scrambled STATIC cloud.
    TEMPORAL — resolvable pole count on the quadratic observable's autocovariance Hankel via ERA/Ho-Kalman
        (pole_closure / P5 machinery). Reads the TIME-ordered series.

DEFENDED CONSTANTS (every one traceable to the prereg; no undefended magic, no post-hoc tuning):
    POOL = [0.90, 0.70, 0.50]  : the fixed real distinct-pole pool from prereg (well-separated products).
    m = 12                     : fixed generous anchor budget = the battery/atom-search operating point (spec cap).
    eps = 0.05                 : GW inner tolerance, the atom-search/battery operating point (reused verbatim).
    n = 100                    : static-cloud subsample size (atom-search N=100; battery operating point).
    d = 16                     : ambient embedding dim (atom-search D_AMB=16, > max G here).
    T = 60000                  : series length for the temporal read (prereg "T=60000 for the temporal read").
    tol_prod = 0.05            : product-match tolerance = P5 closure_error threshold (`closure_error < 0.05`).
    SEEDS = [0, 1, 2]          : seed-averaging set (prereg "SEEDS=[0,1,2], seed-averaged").
    FLAT_TOL_SPATIAL = 0.5     : D_e off-axis flatness / exact-digit band (prereg Amendment §1, ~3.5x seed jitter).
    POLE_FLAT_TOL = 1          : pole-count off-axis flatness band (prereg: range <= 1).
    SPEARMAN_MIN = 0.90        : on-axis monotone-tracking threshold (prereg verdict clauses 1 & 2).
    EXACT digits 4.7/5.3/5.9 (D_e at G=2/4/6) and 1/3 (poles at D=1/2), band>=4 at D=3 : from prereg tables.

DISCIPLINE: NO shim, NO force-pass, NO cherry-picking, NO post-hoc substrate tuning. If a clause FAILS it is
reported with the actual numbers. A partial/one-axis dissociation reported honestly is an acceptable outcome;
dressing a fail as a pass or retuning the substrate after seeing numbers is forbidden.

Run: python -m ebr.experiments.twobytwo_dissociation
"""
import numpy as np

from .pole_closure import latent_with_poles, quadratic, estimate_poles, products, closure_error
from .atom_observable_search import eff_rank, equilibrate_fixed_m, spearman
from ..geometry.clouds import cloud_to_Dw, scramble

# ------------------------------------------------------------------- defended constants (all from the prereg)
POOL = [0.90, 0.70, 0.50]
M = 12
EPS = 0.05
N = 100
D_AMB = 16
T = 60000
TOL_PROD = 0.05
SEEDS = [0, 1, 2]
FLAT_TOL_SPATIAL = 0.5
POLE_FLAT_TOL = 1
SPEARMAN_MIN = 0.90

# grids (the AMENDED prereg grid, exactly)
SPATIAL_G = [2, 4, 6, 8]           # D=2, K=3
TEMPORAL_D = [1, 2, 3]             # G=6, K=3
K_LEVELS = [2, 3, 5]              # G=6, D=2

# exact registered D_e digits (G -> value), band ±FLAT_TOL_SPATIAL; G=8 is ordering-only.
DERANK_EXACT = {2: 4.7, 4: 5.3, 6: 5.9}


# ------------------------------------------------------------------- unified substrate (one process, two knobs)
def make_latent(G, D, T, seed):
    """G independent AR(1) coords with poles = first D pool values CYCLED to length G, then each column
    STANDARDIZED to zero-mean/unit-variance. Whitening is load-bearing: the static cloud is ~N(0, I_G)
    regardless of the pole assignment (spatial meter sees only G, never D), while the pole/autocorrelation
    per coordinate is unchanged (temporal meter untouched). Returns (S [T x G], distinct pole values [D])."""
    distinct = POOL[:D]
    pole_list = [distinct[k % D] for k in range(G)]      # e.g. G=6,D=2 -> [0.9,0.7,0.9,0.7,0.9,0.7]
    U, _lam = latent_with_poles(pole_list, T, seed=seed)  # all real poles => G independent AR(1) coords
    mu = U.mean(0)
    sd = U.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    S = (U - mu) / sd                                     # standardize each column: static cloud ~ N(0, I_G)
    return S, list(distinct)


# ------------------------------------------------------------------- SPATIAL readout — D_e eff-rank (verbatim)
def spatial_derank(S, K, seed, m=M, n=N, d=D_AMB):
    """Subsample n rows of S evenly -> X0 (n x G). K members each: fixed random G x d map, gauge-scramble,
    cloud_to_Dw. Equilibrate ONE shared anchor at fixed m and read D_e effective rank (atom-search (b))."""
    T_ = S.shape[0]
    G = S.shape[1]
    idx = np.linspace(0, T_ - 1, n).round().astype(int)
    X0 = S[idx]                                          # (n x G) static cloud
    Ds, ws = [], []
    for k in range(K):
        rng = np.random.default_rng(7000 + seed * 101 + k * 13)
        Wmap = rng.normal(size=(G, d))                   # fixed random G x d embedding (member-specific)
        Xk = X0 @ Wmap
        Xk = scramble(Xk, seed=8000 + seed * 211 + k * 17)  # G0 gauge scramble (member-specific)
        Dk, wk = cloud_to_Dw(Xk)
        Ds.append(Dk); ws.append(wk)
    _pis, De, _a = equilibrate_fixed_m(Ds, ws, m, eps=EPS, n_outer=12, seed=seed)
    sv = np.linalg.svd(De, compute_uv=False)
    return eff_rank(sv)


# ------------------------------------------------------------------- TEMPORAL readout — pole count (verbatim)
def temporal_polecount(G, D, K, T, seed):
    """Poles are intrinsic/gauge-invariant, so the K "members" are K INDEPENDENT latent seeds of the SAME
    (G,D) system (novel K-invariance test: the resolved count must NOT inflate with K). Count = number of the
    D DISTINCT-pole pairwise products resolved on the quadratic observable, averaged over K, rounded."""
    distinct = POOL[:D]
    prod_true = products(np.array(distinct))             # D(D+1)/2 pairwise products of the DISTINCT poles
    # dedupe near-equal products (within 1e-6) — with a distinct pool there are none, but honor the spec
    kept = []
    for p in prod_true:
        if all(abs(p - q) > 1e-6 for q in kept):
            kept.append(float(p))
    prod_true = np.array(kept)
    order = len(prod_true)
    resolved = []
    for k in range(K):
        S_k, _distinct = make_latent(G, D, T, seed=seed * 100 + k)
        Q = quadratic(S_k)
        est = estimate_poles(Q, order=order)
        # each product counted once: match a product if SOME estimated eigenvalue is within tol_prod of it
        matched = 0
        for p in prod_true:
            if np.min(np.abs(est - p)) < TOL_PROD:
                matched += 1
        resolved.append(matched)
    return int(round(float(np.mean(resolved))))


# ------------------------------------------------------------------- both meters at one grid point (seed-avg)
def read_both(G, D, K):
    """Return (D_e eff-rank, pole count), each seed-averaged over SEEDS."""
    de_seeds, pc_seeds = [], []
    for sd in SEEDS:
        S, _distinct = make_latent(G, D, T, seed=sd)
        de_seeds.append(spatial_derank(S, K, seed=sd))
        pc_seeds.append(temporal_polecount(G, D, K, T, seed=sd))
    return float(np.mean(de_seeds)), float(np.mean(pc_seeds))


# ------------------------------------------------------------------- run
def run():
    print("=" * 92)
    print("2x2 SPATIAL/TEMPORAL DOUBLE-DISSOCIATION  (experimental CONTROL / instrument dissociation)")
    print(f"  substrate: G indep AR(1) coords, poles from pool {POOL} (first D, cycled), columns whitened.")
    print(f"  SPATIAL meter = D_e eff-rank (m={M}); TEMPORAL meter = resolvable pole count (T={T}).")
    print(f"  seeds={SEEDS}, eps={EPS}, n={N}, d={D_AMB}, tol_prod={TOL_PROD}. No new mechanism; both meters reused.")
    print("=" * 92)

    # ---- SPATIAL axis: G in {2,4,6,8}, D=2, K=3
    print("\n[1] SPATIAL axis — G swept, D=2, K=3 (on-axis for D_e, off-axis for poles)")
    sp_de, sp_pc = {}, {}
    for G in SPATIAL_G:
        de, pc = read_both(G, D=2, K=3)
        sp_de[G], sp_pc[G] = de, pc
        print(f"    G={G}:  D_e eff-rank = {de:6.3f}   pole count = {pc}")

    # ---- TEMPORAL axis: D in {1,2,3}, G=6, K=3
    print("\n[2] TEMPORAL axis — D swept, G=6, K=3 (on-axis for poles, off-axis for D_e)")
    tp_de, tp_pc = {}, {}
    for D in TEMPORAL_D:
        de, pc = read_both(G=6, D=D, K=3)
        tp_de[D], tp_pc[D] = de, pc
        print(f"    D={D}:  D_e eff-rank = {de:6.3f}   pole count = {pc}")

    # ---- K control: K in {2,3,5}, G=6, D=2
    print("\n[3] K control — K swept, G=6, D=2 (both meters must stay flat)")
    k_de, k_pc = {}, {}
    for K in K_LEVELS:
        de, pc = read_both(G=6, D=2, K=K)
        k_de[K], k_pc[K] = de, pc
        print(f"    K={K}:  D_e eff-rank = {de:6.3f}   pole count = {pc}")

    # ------------------------------------------------------------- predicted-vs-observed table
    print("\n" + "=" * 92)
    print("PREDICTED vs OBSERVED  (amended prereg digits; EXACT = digit ±0.5, ORDERING = order-only)")
    print("=" * 92)
    print("  SPATIAL meter D_e vs G (D=2,K=3):")
    print("    G | predicted        | kind     | observed | lands?")
    for G in SPATIAL_G:
        obs = sp_de[G]
        if G in DERANK_EXACT:
            pred = DERANK_EXACT[G]
            lands = abs(obs - pred) <= FLAT_TOL_SPATIAL
            print(f"    {G} | {pred:>5.1f} ±{FLAT_TOL_SPATIAL} | EXACT    | {obs:8.3f} | {'YES' if lands else 'NO'}")
        else:
            lands = obs >= sp_de[6] - 0.5                # ordering-only: not inverting vs G=6
            print(f"    {G} | >= G=6 (~5.9-6.1) | ORDERING | {obs:8.3f} | {'YES (no invert)' if lands else 'NO (inverts)'}")
    print("  TEMPORAL meter pole count vs D (G=6,K=3):")
    print("    D | predicted        | kind     | observed | lands?")
    for D in TEMPORAL_D:
        obs = tp_pc[D]
        if D == 1:
            lands = obs == 1; print(f"    1 | 1 (on the nose)   | EXACT    | {obs:8.3f} | {'YES' if lands else 'NO'}")
        elif D == 2:
            lands = obs == 3; print(f"    2 | 3 (on the nose)   | EXACT    | {obs:8.3f} | {'YES' if lands else 'NO'}")
        else:
            lands = obs >= 4; print(f"    3 | 5-6 band (>=4)    | ORDERING | {obs:8.3f} | {'YES' if lands else 'NO'}")

    # ------------------------------------------------------------- verdict statistics
    print("\n" + "=" * 92)
    print("VERDICT — amended prereg clauses (each with actual numbers)")
    print("=" * 92)

    sp_de_vals = [sp_de[G] for G in SPATIAL_G]
    sp_pc_vals = [sp_pc[G] for G in SPATIAL_G]
    tp_de_vals = [tp_de[D] for D in TEMPORAL_D]
    tp_pc_vals = [tp_pc[D] for D in TEMPORAL_D]
    k_de_vals = [k_de[K] for K in K_LEVELS]
    k_pc_vals = [k_pc[K] for K in K_LEVELS]

    # correlations & ranges
    rho_de_G = spearman(SPATIAL_G, sp_de_vals)
    rho_pc_D = spearman(TEMPORAL_D, tp_pc_vals)
    range_D_de = max(tp_de_vals) - min(tp_de_vals)       # D_e off-axis (over D)
    range_K_de = max(k_de_vals) - min(k_de_vals)         # D_e off-axis (over K)
    range_G_pc = max(sp_pc_vals) - min(sp_pc_vals)       # poles off-axis (over G)
    range_K_pc = max(k_pc_vals) - min(k_pc_vals)         # poles off-axis (over K)

    # exact digit landings
    land_G2 = abs(sp_de[2] - 4.7) <= FLAT_TOL_SPATIAL
    land_G4 = abs(sp_de[4] - 5.3) <= FLAT_TOL_SPATIAL
    land_G6 = abs(sp_de[6] - 5.9) <= FLAT_TOL_SPATIAL
    no_invert_G8 = sp_de[8] >= sp_de[6] - 0.5
    pc_D1 = tp_pc[1] == 1
    pc_D2 = tp_pc[2] == 3
    pc_D3 = tp_pc[3] >= 4

    # on-axis ranges for the cross-margin (exact levels only for D_e)
    on_de = max(sp_de[2], sp_de[4], sp_de[6]) - min(sp_de[2], sp_de[4], sp_de[6])   # G in {2,4,6}
    on_pc = max(tp_pc_vals) - min(tp_pc_vals)            # D in {1,2,3}

    # ---- clause 1: spatial (confirmatory half)
    c1 = (rho_de_G >= SPEARMAN_MIN and land_G2 and land_G4 and land_G6 and no_invert_G8
          and range_D_de <= FLAT_TOL_SPATIAL and range_K_de <= FLAT_TOL_SPATIAL)
    print("\n(1) SPATIAL — D_e tracks ONLY space [CONFIRMATORY half]:")
    print(f"      Spearman(D_e, G)      = {rho_de_G:+.3f}  (need >= {SPEARMAN_MIN})            {'PASS' if rho_de_G>=SPEARMAN_MIN else 'FAIL'}")
    print(f"      exact digits: G=2 {sp_de[2]:.3f}~4.7 {'OK' if land_G2 else 'X'} | G=4 {sp_de[4]:.3f}~5.3 {'OK' if land_G4 else 'X'} | G=6 {sp_de[6]:.3f}~5.9 {'OK' if land_G6 else 'X'}   {'PASS' if (land_G2 and land_G4 and land_G6) else 'FAIL'}")
    print(f"      G=8 not inverting: {sp_de[8]:.3f} >= {sp_de[6]-0.5:.3f}                        {'PASS' if no_invert_G8 else 'FAIL'}")
    print(f"      range_D(D_e)          = {range_D_de:.3f}  (need <= {FLAT_TOL_SPATIAL})            {'PASS' if range_D_de<=FLAT_TOL_SPATIAL else 'FAIL'}")
    print(f"      range_K(D_e)          = {range_K_de:.3f}  (need <= {FLAT_TOL_SPATIAL})            {'PASS' if range_K_de<=FLAT_TOL_SPATIAL else 'FAIL'}")
    print(f"    => CLAUSE 1: {'PASS' if c1 else 'FAIL'}")

    # ---- clause 2: temporal (novel half)
    c2 = (rho_pc_D >= SPEARMAN_MIN and pc_D1 and pc_D2 and pc_D3
          and range_G_pc <= POLE_FLAT_TOL and range_K_pc <= POLE_FLAT_TOL)
    print("\n(2) TEMPORAL — poles track ONLY time [NOVEL half]:")
    print(f"      Spearman(poles, D)    = {rho_pc_D:+.3f}  (need >= {SPEARMAN_MIN})            {'PASS' if rho_pc_D>=SPEARMAN_MIN else 'FAIL'}")
    print(f"      exact: poles(D=1)={tp_pc[1]} (need 1) {'OK' if pc_D1 else 'X'} | poles(D=2)={tp_pc[2]} (need 3) {'OK' if pc_D2 else 'X'} | poles(D=3)={tp_pc[3]} (need >=4) {'OK' if pc_D3 else 'X'}   {'PASS' if (pc_D1 and pc_D2 and pc_D3) else 'FAIL'}")
    print(f"      range_G(poles)        = {range_G_pc}  (need <= {POLE_FLAT_TOL})               {'PASS' if range_G_pc<=POLE_FLAT_TOL else 'FAIL'}")
    print(f"      range_K(poles)        = {range_K_pc}  (need <= {POLE_FLAT_TOL})               {'PASS' if range_K_pc<=POLE_FLAT_TOL else 'FAIL'}")
    print(f"    => CLAUSE 2: {'PASS' if c2 else 'FAIL'}")

    # ---- clause 3: cross-margin (each meter moves >2x more on its own axis)
    de_margin = (max(range_D_de, range_K_de) < 0.5 * on_de)
    pc_margin = (max(range_G_pc, range_K_pc) < 0.5 * on_pc)
    c3 = de_margin and pc_margin
    print("\n(3) CROSS-MARGIN — each meter's off-axis range < 0.5 x its on-axis range:")
    print(f"      D_e:   off-axis max(range_D,range_K)={max(range_D_de,range_K_de):.3f}  <  0.5*on-axis({on_de:.3f})={0.5*on_de:.3f}   {'PASS' if de_margin else 'FAIL'}")
    print(f"      poles: off-axis max(range_G,range_K)={max(range_G_pc,range_K_pc)}      <  0.5*on-axis({on_pc})={0.5*on_pc:.3f}   {'PASS' if pc_margin else 'FAIL'}")
    print(f"    => CLAUSE 3: {'PASS' if c3 else 'FAIL'}")

    overall = c1 and c2 and c3
    print("\n" + "=" * 92)
    print(f"OVERALL 2x2 DISSOCIATION VERDICT: {'PASS' if overall else 'FAIL'}  "
          f"(clause1={'PASS' if c1 else 'FAIL'}, clause2={'PASS' if c2 else 'FAIL'}, clause3={'PASS' if c3 else 'FAIL'})")
    if not overall:
        legs = []
        if c1: legs.append("spatial leg clean")
        else:  legs.append("spatial leg NOT clean")
        if c2: legs.append("temporal leg clean")
        else:  legs.append("temporal leg NOT clean")
        print(f"  Honest partial read: {', '.join(legs)}. Reported as-is; no patching, no substrate retuning.")
    print("=" * 92)

    return {
        "spatial": {"G": SPATIAL_G, "D_e": sp_de_vals, "poles": sp_pc_vals},
        "temporal": {"D": TEMPORAL_D, "D_e": tp_de_vals, "poles": tp_pc_vals},
        "K": {"K": K_LEVELS, "D_e": k_de_vals, "poles": k_pc_vals},
        "clauses": {"c1": bool(c1), "c2": bool(c2), "c3": bool(c3), "overall": bool(overall)},
        "stats": {"rho_de_G": rho_de_G, "rho_pc_D": rho_pc_D, "range_D_de": range_D_de,
                  "range_K_de": range_K_de, "range_G_pc": range_G_pc, "range_K_pc": range_K_pc,
                  "on_de": on_de, "on_pc": on_pc},
    }


if __name__ == "__main__":
    run()
