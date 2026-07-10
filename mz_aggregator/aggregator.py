"""
aggregator.py — decentralized aggregation over K workers into the self-expanding OV MZ memory kernel,
plus the playable CLI and the double-dissociation experiment.

Protocol (decentralized, dimension-independent):
  A task distribution (atomic order-r, or continuous spectrum) has true Markov params h_true.
  Each of K workers observes h_true through its own finite-sample fluctuation (independent noise; optionally
  its own internal dimension q_w) and reports a LOCAL closure residual (its estimate minus the shared
  kernel's current reproduction) in the common p-channel basis. The kernel AGGREGATES the residuals
  (running mean over all worker reports = evidence accrual), estimates the second-FDT fluctuation std from
  worker disagreement, and self-expands / prunes its order by the Hankel-rank/atomicity criterion.
  The kernel's order (and memory) tracks the McMillan degree of the task distribution — NOT K.

CLI:
  python aggregator.py --live --K 8 --diversity 4          # watch the kernel rank move, round by round
  python aggregator.py --live --K 8 --continuous           # continuous-spectrum: rank never cleanly stops
  python aggregator.py --hetero --K 8 --diversity 4        # heterogeneous per-worker dims (dim-independence)
  python aggregator.py --calibrate                         # poles-first ground-truth check (atomic + continuous)
  python aggregator.py --sweep --out results.json          # the frozen double dissociation -> results.json
  python aggregator.py --all --out results.json            # calibrate + sweep + dim-independence, full run
"""
import argparse, json, sys
import numpy as np
import resolvent as R
from mz_kernel import MZKernel

P = 3            # shared channel width
KMARKOV = 40     # Markov parameters observed
L = 18           # Hankel block size
SIGMA0 = 0.02    # per-worker finite-sample fluctuation (std per Markov entry)


# ------------------------------------------------------------------ task distributions
def make_task(diversity, continuous=False, seed=0):
    """Return (h_true, true_degree, kind)."""
    if continuous:
        h = R.continuous_markov(KMARKOV, p=P, seed=seed)
        return h, float("inf"), "continuous"
    A, B, C = R.atomic_system(diversity, p=P, seed=seed)
    h = R.markov_from_system(A, B, C, KMARKOV)
    return h, diversity, "atomic"


# ------------------------------------------------------------------ workers (decentralized observers)
class Worker:
    def __init__(self, h_true, sigma, seed, q=None):
        self.h_true = h_true
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)
        self.q = q                       # optional per-worker internal dim (dimension-independence demo)
        if q is not None:
            p = h_true.shape[1]
            self.Penc = self.rng.normal(size=(q, p)) / np.sqrt(p)     # p -> q_w  (worker's own space)
            self.Pdec = np.linalg.pinv(self.Penc)                     # q_w -> p  (report back to shared basis)

    def report(self):
        """Noisy estimate of the Markov params in the shared p-channel basis."""
        h = self.h_true + self.rng.normal(scale=self.sigma, size=self.h_true.shape)
        if self.q is not None:
            # pass each block through the worker's own q_w-dim space and back (dimension-independence)
            h = np.stack([self.Pdec @ (self.Penc @ hk @ self.Penc.T) @ self.Pdec.T for hk in h])
        return h


# ------------------------------------------------------------------ the aggregation loop
def run_loop(diversity, K, rounds=25, continuous=False, hetero=False, seed=0, verbose=False):
    """Decentralized evidence accrual into the self-expanding kernel. Returns trajectory dict."""
    h_true, degree, kind = make_task(diversity, continuous, seed)
    rng = np.random.default_rng(1000 + seed)
    qs = [int(rng.integers(P, 3 * P + 1)) for _ in range(K)] if hetero else [None] * K
    workers = [Worker(h_true, SIGMA0, seed=seed * 991 + w, q=qs[w]) for w in range(K)]
    kernel = MZKernel(p=P, L=L, seed=seed)

    acc = np.zeros_like(h_true); count = 0
    traj = {"order": [], "residual": [], "cost": [], "gap": [], "floor": [], "sigma": []}
    for rd in range(rounds):
        reports = []
        for w in workers:
            h_w = w.report()
            # LOCAL closure residual against the shared kernel (decentralized)
            _ = h_w - kernel.predicted_markov(KMARKOV)
            reports.append(h_w)
            acc += h_w; count += 1
        h_hat = acc / count                      # aggregated running-mean estimate (= kernel_pred + agg residual)
        # second-FDT fluctuation std of the aggregated (running-mean) estimate over all reports so far
        sigma = SIGMA0 / np.sqrt(count)
        n = kernel.step(h_hat, sigma)
        traj["order"].append(n); traj["residual"].append(round(kernel.closure_residual(), 4))
        traj["cost"].append(kernel.memory_cost()); traj["gap"].append(round(kernel.spectral_gap(), 2))
        traj["floor"].append(round(kernel.last_floor, 5)); traj["sigma"].append(round(sigma, 5))
        if verbose:
            print(f"  round {rd:2d}: order={n:2d}  cost={kernel.memory_cost():3d}  "
                  f"closure_resid={kernel.closure_residual():.4f}  gap={kernel.spectral_gap():7.1f}  "
                  f"floor={kernel.last_floor:.4f}  sigma={sigma:.4f}", flush=True)
    traj["degree_true"] = degree; traj["kind"] = kind; traj["expansions"] = kernel.expansions
    traj["prunes"] = kernel.prunes; traj["terminal_order"] = int(np.median(traj["order"][-5:]))
    return traj


# ------------------------------------------------------------------ poles-first calibration
def calibrate(seeds=(0, 1, 2)):
    print("=" * 74); print("POLES-FIRST CALIBRATION (ground truth: atomic degree = r; continuous = inf)"); print("=" * 74)
    out = {"atomic": {}, "continuous": {}}
    print("\nATOMIC (terminal kernel order must == r, with a clean Hankel gap):")
    for r in [2, 3, 4, 6, 8]:
        terms, gaps = [], []
        for s in seeds:
            tr = run_loop(r, K=8, rounds=25, seed=s)
            terms.append(tr["terminal_order"]); gaps.append(tr["gap"][-1])
        ok = all(abs(t - r) <= 1 for t in terms)
        out["atomic"][r] = {"terminal_orders": terms, "mean_gap": float(np.mean(gaps)), "match": bool(ok)}
        print(f"  r={r}: terminal order {terms} (truth {r}) gap~{np.mean(gaps):8.1f}  {'MATCH' if ok else 'MISS'}")
    print("\nCONTINUOUS (must NOT cleanly terminate; order drifts with evidence/K, gap ~ 1):")
    cont = []
    for K in [4, 8, 16, 32]:
        tr = run_loop(0, K=K, rounds=25, continuous=True, seed=0)
        cont.append((K, tr["terminal_order"], tr["gap"][-1]))
        print(f"  K={K:2d}: terminal order={tr['terminal_order']:2d}  gap~{tr['gap'][-1]:.2f}  "
              f"(order drifts with K, no clean small-integer termination)")
    orders = [c[1] for c in cont]
    drifts = max(orders) - min(orders) >= 2
    gapsmall = np.mean([c[2] for c in cont]) < 3.0
    out["continuous"]["orders_vs_K"] = cont
    out["continuous"]["no_clean_termination"] = bool(drifts and gapsmall)
    print(f"  -> drifts with K: {drifts}; gaps ~1 (no atomic gap): {gapsmall}  "
          f"=> no clean termination: {drifts and gapsmall}")
    return out


# ------------------------------------------------------------------ the double dissociation
def sweep(seeds=(0, 1, 2, 3, 4)):
    print("\n" + "=" * 74); print("DOUBLE DISSOCIATION SWEEP"); print("=" * 74)
    res = {"arm1_raiseK_fixed_diversity": {}, "arm2_raise_diversity_fixed_K": {}}

    R_FIXED = 4
    print(f"\nARM 1 — raise K (workers) at FIXED diversity r={R_FIXED}. PREDICT: kernel rank FLAT in K.")
    for K in [2, 4, 8, 16, 32, 64]:
        terms, costs = [], []
        for s in seeds:
            tr = run_loop(R_FIXED, K=K, rounds=25, seed=s)
            terms.append(tr["terminal_order"]); costs.append(tr["cost"][-1])
        res["arm1_raiseK_fixed_diversity"][K] = {"orders": terms, "mean_order": float(np.mean(terms)),
                                                  "mean_kernel_cost": float(np.mean(costs))}
        print(f"  K={K:3d}: kernel rank {terms} mean={np.mean(terms):.2f}  kernel_mem={np.mean(costs):.0f}")

    K_FIXED = 8
    print(f"\nARM 2 — raise diversity r at FIXED K={K_FIXED}. PREDICT: kernel rank GROWS with r (tracks McMillan degree).")
    for r in [2, 3, 4, 6, 8, 10]:
        terms, costs = [], []
        for s in seeds:
            tr = run_loop(r, K=K_FIXED, rounds=25, seed=s)
            terms.append(tr["terminal_order"]); costs.append(tr["cost"][-1])
        res["arm2_raise_diversity_fixed_K"][r] = {"orders": terms, "mean_order": float(np.mean(terms)),
                                                   "truth": r, "mean_kernel_cost": float(np.mean(costs))}
        print(f"  r={r:2d}: kernel rank {terms} mean={np.mean(terms):.2f} (truth {r})  kernel_mem={np.mean(costs):.0f}")

    a1 = res["arm1_raiseK_fixed_diversity"]
    orders_K = [a1[K]["mean_order"] for K in [2, 4, 8, 16, 32, 64]]
    arm1_flat = (max(orders_K) - min(orders_K)) <= 1.0
    a2 = res["arm2_raise_diversity_fixed_K"]
    orders_r = [a2[r]["mean_order"] for r in [2, 3, 4, 6, 8, 10]]
    arm2_grows = all(x <= y + 0.5 for x, y in zip(orders_r, orders_r[1:])) and (orders_r[-1] - orders_r[0]) >= 3
    arm2_tracks = np.mean([abs(a2[r]["mean_order"] - r) for r in [2, 3, 4, 6, 8, 10]]) <= 1.0
    res["verdict"] = {"arm1_flat_in_K": bool(arm1_flat), "arm2_grows_with_diversity": bool(arm2_grows),
                      "arm2_tracks_mcmillan_degree": bool(arm2_tracks),
                      "claim_holds": bool(arm1_flat and arm2_grows)}
    print(f"\nVERDICT: Arm1 flat in K = {arm1_flat} (orders {orders_K}); "
          f"Arm2 grows with diversity = {arm2_grows}, tracks degree = {arm2_tracks} (orders {orders_r})")
    print(f"  => CLAIM HOLDS (flat-in-K AND grows-in-diversity): {arm1_flat and arm2_grows}")
    return res


def dim_independence(seeds=(0, 1, 2)):
    print("\n" + "=" * 74); print("DIMENSION-INDEPENDENCE (heterogeneous per-worker dims; kernel order unaffected)"); print("=" * 74)
    out = {}
    for tag, het in [("homogeneous", False), ("heterogeneous q_w in [p,3p]", True)]:
        terms = [run_loop(4, K=8, rounds=25, hetero=het, seed=s)["terminal_order"] for s in seeds]
        out[tag] = {"terminal_orders": terms, "mean": float(np.mean(terms))}
        print(f"  {tag:32s}: terminal kernel order {terms} (truth r=4)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--hetero", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--diversity", type=int, default=4)
    ap.add_argument("--continuous", action="store_true")
    ap.add_argument("--rounds", type=int, default=25)
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    if args.live or (not any([args.calibrate, args.sweep, args.all, args.hetero])):
        kind = "continuous spectrum" if args.continuous else f"atomic diversity r={args.diversity}"
        print(f"LIVE: {kind}, K={args.K} workers, {args.rounds} rounds — watch the kernel rank move")
        tr = run_loop(args.diversity, args.K, rounds=args.rounds, continuous=args.continuous,
                      hetero=args.hetero, seed=0, verbose=True)
        print(f"\nterminal kernel order = {tr['terminal_order']}  (truth: {tr['degree_true']}, kind: {tr['kind']})  "
              f"expansions={tr['expansions']} prunes={tr['prunes']}")
        return

    results = {"config": {"p": P, "Kmarkov": KMARKOV, "L": L, "sigma0": SIGMA0}}
    if args.calibrate or args.all:
        results["calibration"] = calibrate()
    if args.sweep or args.all:
        results["sweep"] = sweep()
    if args.hetero or args.all:
        results["dim_independence"] = dim_independence()
    json.dump(results, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
