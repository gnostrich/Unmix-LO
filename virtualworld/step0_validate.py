"""
STEP-0 (MANDATORY) — validate the structured/noise detector on synthetic INJECT-KNOBS BEFORE wiring
any real modality. Same Step-0 discipline as fdt_denoise.

Setup: a real engine world-state trajectory (shared medium, dim D). Build a residual `d` in that medium:
  - NOISE knob        : d = state-independent random scatter  -> must tag NOISE.
  - STRUCTURED(lin)   : d = alpha * (s @ B) , B a fixed low-rank (r=2) map of the STATE -> must tag
                        STRUCTURED (reproducible low-rank AND held-out predictable from state).
  - STRUCTURED(regime): d = a coarse binary REGIME derived from the state, times a fixed direction ->
                        must tag STRUCTURED (a hidden state-dependent distinction, not perfectly linear).

The calibration point: run at D=6 (smoke-loop's failing dim) and D>=20-32. Show captured-vs-baseline
ALONE misfires at D=6, and the combined (captured AND held-out-R^2>=0.3) detector separates cleanly at
D>=20-32. If it does not separate there, the detector is fixed before real modalities are wired.
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import world as W
from detector import classify, R2_MIN


def make_state(D, n, seed):
    """An AR world-state trajectory of dim D (stand-in shared medium), standardised."""
    rng = np.random.default_rng(seed)
    s = np.zeros((n, D))
    for t in range(1, n):
        s[t] = 0.9 * s[t - 1] + 0.3 * rng.normal(size=D)
    s = (s - s.mean(0)) / (s.std(0) + 1e-9)
    return s


def inject(kind, s, seed, alpha=1.0):
    """Return a decoherence residual `d` (n, D) of the requested kind."""
    rng = np.random.default_rng(seed)
    n, D = s.shape
    if kind == "noise":
        return alpha * rng.normal(size=(n, D))
    if kind == "structured_linear":
        r = 2
        B = rng.normal(size=(D, r)) @ rng.normal(size=(r, D))   # rank-2 map of the STATE
        d = s @ B
        d += 0.15 * np.linalg.norm(d) / np.sqrt(d.size) * rng.normal(size=(n, D))  # small nuisance
        return alpha * d
    if kind == "structured_regime":
        w = rng.normal(size=D)
        regime = (s @ w > 0).astype(float) * 2 - 1              # hidden binary regime from state
        direction = rng.normal(size=D)
        d = np.outer(regime, direction)
        d += 0.15 * np.linalg.norm(d) / np.sqrt(d.size) * rng.normal(size=(n, D))
        return alpha * d
    if kind == "structured_diffuse":
        # FULL-rank but fully state-predictable: reproduces the smoke_loop failure mode where
        # captured-vs-baseline ALONE misses it (eff-rank high -> captured ~ base), yet held-out R^2
        # from the state catches it. This is exactly why the held-out condition is required.
        B = rng.normal(size=(D, D))                            # full-rank map of the STATE
        d = s @ B
        d += 0.15 * np.linalg.norm(d) / np.sqrt(d.size) * rng.normal(size=(n, D))
        return alpha * d
    raise ValueError(kind)


def run(D, n=1000, seed=0):
    s = make_state(D, n, seed)
    ntr = n // 2
    train, test = np.arange(ntr), np.arange(ntr, n)
    rows = {}
    for kind in ("noise", "structured_linear", "structured_regime", "structured_diffuse"):
        d = inject(kind, s, seed + 1)
        rows[kind] = classify(d, s, train, test, D=D)
    return rows


def main():
    print(f"STEP-0 detector validation  (STRUCTURED requires captured>1.3*base AND held-out R^2>={R2_MIN})\n")
    results = {"r2_min": R2_MIN, "by_D": {}}
    header = f"{'D':>4} {'inject':<20}{'eff':>6}{'captured':>10}{'base':>7}{'cap/base':>9}{'heldoutR2':>11}  {'captured-only':>14}  verdict"
    for D in (6, 20, 24, 32):
        print(header)
        results["by_D"][D] = {}
        for kind, r in run(D).items():
            cap_only = "STRUCTURED" if r["captured"] > 1.3 * r["baseline"] else "NOISE"
            print(f"{D:>4} {kind:<20}{r['eff_rank']:>6.1f}{r['captured']:>10.2f}{r['baseline']:>7.2f}"
                  f"{r['captured_vs_base']:>9.2f}{r['heldout_r2']:>11.3f}  {cap_only:>14}  {r['verdict']}")
            results["by_D"][D][kind] = {**r, "captured_only_verdict": cap_only}
        print()

    # PASS criterion at the working dims: every structured knob -> STRUCTURED, noise -> NOISE, D in {20,24,32}
    struct_kinds = ("structured_linear", "structured_regime", "structured_diffuse")
    ok = True
    min_struct_r2, max_noise_r2 = 1.0, -1.0
    for D in (20, 24, 32):
        rr = results["by_D"][D]
        for k in struct_kinds:
            ok &= rr[k]["structured"]
            min_struct_r2 = min(min_struct_r2, rr[k]["heldout_r2"])
        ok &= not rr["noise"]["structured"]
        max_noise_r2 = max(max_noise_r2, rr["noise"]["heldout_r2"])
    results["separates_at_D>=20"] = bool(ok)
    results["min_structured_heldout_r2_D>=20"] = float(min_struct_r2)
    results["max_noise_heldout_r2_D>=20"] = float(max_noise_r2)
    print(f"COMBINED detector separates structured vs noise at D in {{20,24,32}}: {ok}")
    print(f"Held-out R^2 (the robust condition) is the decisive separator: every STRUCTURED knob R^2 "
          f">= {min_struct_r2:.2f} (>= {R2_MIN}); every NOISE knob R^2 <= {max_noise_r2:.2f} (< {R2_MIN}).")
    print("\nSTEP-0 VERDICT:", "PASS — detector validated, safe to wire real modalities."
          if ok else "FAIL — fix the detector before proceeding.")
    json.dump(results, open(os.path.join(os.path.dirname(__file__), "step0_results.json"), "w"),
              indent=1, default=float)


if __name__ == "__main__":
    main()
