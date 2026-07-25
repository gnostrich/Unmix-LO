"""
probe_b.py — MEASUREMENT ONLY (EBR decision probes, PREREG-PROBES.md @ 9b24e8e).

Is the measured disagreement member-carried holonomy, or a solver-schedule artifact?
Reuses the validated G4 path (experiments/g4_meter) verbatim on its own synthetic
clone-vs-disjoint substrate. Adds no mechanism; the warm variant uses only the
EXISTING `pis0` argument of functional.equilibrate. Faithful-or-wipe.

B1 cold floor : g4's cycle_cost is ALREADY cold (no pis0 in the path) -> the
                separation it reports IS the cold number. A warm variant (seed
                couplings from a prior solve via the existing pis0 arg) quantifies
                path-debt.
B2 schedule   : member-update order is provably a no-op (all members updated
                against a frozen anchor within a sweep, engine/functional) -> we
                MEASURE its variance (expect ~0). Block order (pi -> De/a) is
                forced by data dependency and NOT engine-exposed; permuting it
                needs a re-parameterized solver = out of scope -> reported as
                unmeasurable-without-building, per the directive stop rule.
"""
import csv
import itertools
import numpy as np

from ebr.energy import functional as EN
from ebr.experiments import g4_meter as G4


def separation(Ds_d, ws_d, Ds_c, ws_c, mA, mB, eps=0.08):
    c_disj = G4.cycle_cost(Ds_d, ws_d, mA, mB, eps=eps)
    c_clone = G4.cycle_cost(Ds_c, ws_c, mA, mB, eps=eps)
    sep = c_disj / max(c_clone, 1e-9)
    return c_disj, c_clone, sep


def cycle_cost_ordered(Ds, ws, mA, mB, order_A, order_B, eps=0.08, m=4, n_outer=15, init_seed=1):
    """cycle_cost with EXPLICIT member update order (permutes the list handed to the solver).
    Uses only EN.equilibrate; no mechanism added."""
    def anchor(idx_ordered):
        rng = np.random.default_rng(init_seed)
        De = rng.random((m, m)); De = (De + De.T) / 2
        np.fill_diagonal(De, 0); De /= np.median(De[np.triu_indices(m, 1)])
        a = np.full(m, 1.0 / m)
        pis, De, a, _f, _c = EN.equilibrate([Ds[i] for i in idx_ordered], [ws[i] for i in idx_ordered],
                                            De, a, a.copy(), eps=eps, n_outer=n_outer)
        return dict(zip(idx_ordered, pis)), a
    piA, aA = anchor(order_A)
    piB, aB = anchor(order_B)
    shared = [i for i in mA if i in mB]
    costs = [np.linalg.norm(G4._self_coupling(piA[i], aA) - G4._self_coupling(piB[i], aB)) for i in shared]
    return float(np.mean(costs))


def cycle_cost_warm(Ds, ws, mA, mB, eps=0.08, m=4, n_outer=15, init_seed=1):
    """Warm variant: solve cold, then RE-solve seeded from the prior couplings (existing pis0 arg).
    Path-debt = |warm - cold| final residue."""
    def anchor(idx):
        rng = np.random.default_rng(init_seed)
        De = rng.random((m, m)); De = (De + De.T) / 2
        np.fill_diagonal(De, 0); De /= np.median(De[np.triu_indices(m, 1)])
        a = np.full(m, 1.0 / m)
        Dl, wl = [Ds[i] for i in idx], [ws[i] for i in idx]
        pis1, De1, a1, _f, _c = EN.equilibrate(Dl, wl, De, a, a.copy(), eps=eps, n_outer=n_outer)
        pis2, De2, a2, _f2, _c2 = EN.equilibrate(Dl, wl, De1, a1.copy(), a.copy(),
                                                 eps=eps, n_outer=n_outer, pis0=pis1)
        return dict(zip(idx, pis2)), a2
    piA, aA = anchor(mA)
    piB, aB = anchor(mB)
    shared = [i for i in mA if i in mB]
    costs = [np.linalg.norm(G4._self_coupling(piA[i], aA) - G4._self_coupling(piB[i], aB)) for i in shared]
    return float(np.mean(costs))


def main():
    from ebr.experiments import substrate as S
    mA, mB = [0, 1, 2], [1, 2]
    eps = 0.08

    # ---------------- B1 cold floor (validated path is already cold) ----------------
    print("========== B1 COLD FLOOR ==========")
    seps, disjs, clones = [], [], []
    for sd in range(5):
        disj = S.make_diverse_models(3, seed0=200 + 10 * sd)
        Dd, wd = G4.build_clouds(disj, n=110, seed=sd)
        clone = [S.make_diverse_models(1, seed0=200 + 10 * sd)[0]] * 3
        Dc, wc = G4.build_clouds(clone, n=110, seed=sd)
        cd, cc, sep = separation(Dd, wd, Dc, wc, mA, mB, eps)
        disjs.append(cd); clones.append(cc); seps.append(sep)
        print(f"  substrate {sd}: disjoint={cd:.4f}  clone={cc:.4f}  separation={sep:.1f}×")
    seps = np.array(seps)
    print(f"  cold separation: median {np.median(seps):.1f}×  mean {seps.mean():.1f}×  "
          f"min {seps.min():.1f}×  max {seps.max():.1f}×")
    print(f"  floor-is-real threshold (prereg) = 3× ; cold median {'PASS' if np.median(seps) > 3 else 'FAIL'}")

    # warm-vs-cold path-debt on substrate 0
    disj = S.make_diverse_models(3, seed0=200); Dd, wd = G4.build_clouds(disj, n=110, seed=0)
    cold0 = G4.cycle_cost(Dd, wd, mA, mB, eps=eps)
    warm0 = cycle_cost_warm(Dd, wd, mA, mB, eps=eps)
    print(f"  warm-vs-cold (disjoint residue): cold={cold0:.4f}  warm={warm0:.4f}  "
          f"path-debt |Δ|={abs(warm0 - cold0):.4f}  ({100*abs(warm0-cold0)/max(cold0,1e-9):.1f}% of cold)")

    # ---------------- B2 schedule permutation ----------------
    print("\n========== B2 SCHEDULE PERMUTATION ==========")
    disj = S.make_diverse_models(3, seed0=200); Dd, wd = G4.build_clouds(disj, n=110, seed=0)
    clone = [S.make_diverse_models(1, seed0=200)[0]] * 3; Dc, wc = G4.build_clouds(clone, n=110, seed=0)
    perms = list(itertools.permutations([0, 1, 2]))          # 6 member-order permutations
    perm_res = []
    for pm in perms:
        # remap member ids consistently for edges A and B under this order
        oA = [p for p in pm if p in mA]
        oB = [p for p in pm if p in mB]
        r = cycle_cost_ordered(Dd, wd, mA, mB, oA, oB, eps=eps)
        perm_res.append((pm, r))
        print(f"  member order {pm}: disjoint residue = {r:.6f}")
    rv = np.array([r for _, r in perm_res])
    cv = float(rv.std() / max(rv.mean(), 1e-12))
    print(f"  member-order residue: mean {rv.mean():.6f}  std {rv.std():.2e}  CV {cv:.2e}")
    print("  => member-update order is a structural no-op (all members updated vs a frozen anchor).")
    print("  BLOCK-order (pi -> De/a) is forced by data dependency and not engine-exposed;")
    print("  permuting it requires a re-parameterized solver = OUT OF SCOPE (unmeasurable without building).")

    # ---------------- B3 joint verdict ----------------
    print("\n========== B3 JOINT VERDICT ==========")
    cold_ok = np.median(seps) > 3
    sched_stable = cv < 0.10
    print(f"  cold floor > 3× : {cold_ok} (median {np.median(seps):.1f}×)")
    print(f"  schedule-stable (member-order CV < 0.10): {sched_stable} (CV {cv:.2e})")
    print(f"  block-order non-commutativity: NOT TESTED (schedule forced/unexposed; would need a build)")
    if cold_ok and sched_stable:
        verdict = ("member-carried w.r.t. the reachable schedule axis (member order); "
                   "block-order artifact NOT excluded — unmeasurable without building")
    else:
        verdict = "schedule-tracking or floor-below-3× — see numbers"
    print(f"  VERDICT: {verdict}")

    # ---------------- CSV ----------------
    with open("probes/probe_b_cold.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["substrate_seed", "disjoint", "clone", "separation_x"])
        for i in range(len(seps)):
            w.writerow([i, disjs[i], clones[i], seps[i]])
    with open("probes/probe_b_perms.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["member_order", "disjoint_residue"])
        for pm, r in perm_res:
            w.writerow(["-".join(map(str, pm)), r])
    print("\nwrote probes/probe_b_{cold,perms}.csv")


if __name__ == "__main__":
    main()
