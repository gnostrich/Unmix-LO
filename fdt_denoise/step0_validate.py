"""
STEP 0 -- GATES EVERYTHING. Validate the proper multivariate second-FDT estimator on ground-truth
processes with KNOWN FDT status. The estimator MUST score the FDT-holds process HIGH and every
FDT-fails process LOW. If it cannot separate them, NO real-model result is trustworthy.

Ground truths:
  (a) OU-reversible : symmetric-drift equilibrium OU (linear generator + white noise, detailed balance)
                      -> 2nd-FDT holds BY CONSTRUCTION           -> expect HIGH.
  (b) random-walk-diff : difference of two independent random walks (unit root, NO dissipation)
                      -> fluctuation with no matched dissipation  -> FDT FAILS -> expect LOW.
  (c) rotational-driven : non-symmetric drift with a rotation (probability currents, broken detailed
                      balance) -> non-equilibrium               -> FDT FAILS -> expect LOW.
  (d) white-noise : memoryless fluctuation (no generator/memory) -> the noise floor -> expect LOW.

The v1 probe FAILED here (its crude autocorrelation proxy gave OU only +0.178, ambiguous). We also
document WHY a scalar residual-reconstruction estimator is invalid (it is CIRCULAR: it scores both a
FDT-holds and a FDT-fails scalar process at cosine 1.000), which is why the estimator is multivariate.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fdt_estimator import fdt_fraction

SEED = 0
NR, T, D = 60, 400, 8       # rollouts x steps x dim  (mimics the real per-rollout trajectory structure)
rng = np.random.default_rng(SEED)


def _sim(Phi, sigma=0.5):
    rolls = []
    for _ in range(NR):
        X = np.zeros((T, D)); X[0] = rng.normal(size=D)
        for t in range(1, T):
            X[t] = Phi @ X[t - 1] + sigma * rng.normal(size=D)
        rolls.append(X)
    return rolls


def ou_reversible():
    B = rng.normal(size=(D, D)); B = (B + B.T) / 2
    w, V = np.linalg.eigh(B)
    w = 0.3 + 0.5 * (w - w.min()) / (w.max() - w.min() + 1e-9)   # eigenvalues in [0.3,0.8]
    return _sim(V @ np.diag(w) @ V.T)                             # symmetric -> reversible, dissipative


def rw_diff():
    rolls = []
    for _ in range(NR):
        g1 = np.cumsum(rng.normal(size=(T, D)), 0) * 0.05
        g2 = np.cumsum(rng.normal(size=(T, D)), 0) * 0.05
        rolls.append(np.tanh(g1) - np.tanh(g2))
    return rolls


def rotational():
    Phi = np.zeros((D, D)); th = 0.6
    for i in range(0, D, 2):
        Phi[i, i] = 0.7 * np.cos(th); Phi[i, i + 1] = -0.7 * np.sin(th)
        Phi[i + 1, i] = 0.7 * np.sin(th); Phi[i + 1, i + 1] = 0.7 * np.cos(th)
    return _sim(Phi)


def white():
    return [rng.normal(size=(T, D)) for _ in range(NR)]


def main():
    cases = [("OU-reversible (FDT holds)   [expect HIGH]", ou_reversible, "HIGH"),
             ("random-walk-diff (no dissip)[expect LOW]", rw_diff, "LOW"),
             ("rotational-driven (currents)[expect LOW]", rotational, "LOW"),
             ("white-noise (no memory)     [expect LOW]", white, "LOW")]
    print(f"STEP-0 FDT estimator validation  (k={D} modes, {NR} rollouts x {T} steps)\n")
    out = {}
    for name, fn, expect in cases:
        r = fdt_fraction(fn(), k=D)
        out[name.split(" [")[0].strip()] = {k: (v if np.isscalar(v) else None)
                                            for k, v in r.items() if k in
                                            ("frac_soft", "frac_hard", "mean_absmu", "mean_rev", "current_frac")}
        print(f"  {name}: FDT_frac={r['frac_soft']:.3f} (hard={r['frac_hard']:.2f})  "
              f"<|mu|>={r['mean_absmu']:.3f} rev={r['mean_rev']:.3f} curr={r['current_frac']:.3f}")

    ou = out["OU-reversible (FDT holds)"]["frac_soft"]
    fails = [out[k]["frac_soft"] for k in out if "OU-reversible" not in k]
    gap = ou - max(fails)
    passed = (ou >= 0.5) and (max(fails) <= 0.3) and (gap >= 0.3)
    print(f"\n  OU (FDT-holds) = {ou:.3f}  vs  max(FDT-fails) = {max(fails):.3f}   gap = {gap:.3f}")
    print(f"  STEP-0 VALIDATION: {'PASS -- estimator cleanly separates FDT-holds from FDT-fails' if passed else 'FAIL -- do NOT trust real results'}")
    out["_verdict"] = {"ou_frac": ou, "max_fail_frac": float(max(fails)), "gap": float(gap), "passed": bool(passed)}
    json.dump(out, open(os.path.join(HERE, "step0_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
