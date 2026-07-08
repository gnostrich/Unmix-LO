"""
STAGE 1 — VALIDITY GATE for the indexing-value test (family frozen in FAMILY.md, thresholds in
PREREG.md / BRIEF.md). For each of the 12 pre-specified planting configs, measure best-single,
true-oracle, entangled-oracle, and naive-strong, and check the three validity conditions. We are
NOT comparing indexing here — only asking whether the informative regime is blindly constructible.

Conditions per config:
  (i)   true_oracle - best_single >= 0.15   (complementarity real)
  (ii)  true_oracle - naive_strong >= 0.10  (a strong naive baseline genuinely fails)
  (iii) true_oracle >= 0.80                 (oracle reachable — info present & learnable)
A config is VALID iff all three hold. Stage 1 PASSES iff >= 1 config is valid.
"""
import os, json
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
D = 24
SEED = 0


def readout(X, y, ntr):
    clf = make_pipeline(StandardScaler(),
                        MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400,
                                      early_stopping=True, n_iter_no_change=15, random_state=0))
    clf.fit(X[:ntr], y[:ntr])
    return float(clf.score(X[ntr:], y[ntr:]))


def whiten(X):
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt.T / (S / np.sqrt(len(X)) + 1e-6)


def make_target(kind, zA, zB, rng):
    wA, wB, wg = rng.normal(size=D), rng.normal(size=D), rng.normal(size=D)
    gA, gB = zA @ wA, zB @ wB
    if kind == "additive":
        y = np.sign(gA + gB)
    elif kind == "xor":
        y = np.sign(gA * gB)
    else:  # gate
        y = np.sign(np.where(zA @ wg > 0, gB, -gB))
    return (y > 0).astype(int)


def entangle(kind, fA, fB, rng):
    if kind == "E1":
        return fA @ rng.normal(size=(D, D)), fB @ rng.normal(size=(D, D))
    if kind == "E2":
        M = rng.normal(size=(2 * D, 2 * D))
        E = np.hstack([fA, fB]) @ M
        return E[:, :D], E[:, D:]
    if kind == "E3":
        return np.tanh(fA @ rng.normal(size=(D, D))), np.tanh(fB @ rng.normal(size=(D, D)))
    if kind == "E4":
        k = D // 2
        return fA @ rng.normal(size=(D, k)), fB @ rng.normal(size=(D, k))
    raise ValueError(kind)


FAMILY = [
    ("c1", "additive", "E1", 4000), ("c2", "additive", "E4", 4000),
    ("c3", "xor", "E1", 4000), ("c4", "xor", "E2", 4000), ("c5", "xor", "E3", 4000),
    ("c6", "xor", "E4", 4000), ("c7", "gate", "E1", 4000), ("c8", "gate", "E2", 4000),
    ("c9", "gate", "E3", 4000), ("c10", "xor", "E1", 800), ("c11", "xor", "E3", 800),
    ("c12", "gate", "E4", 4000),
]
NTEST = 1000


def run_config(cid, target, ent, ntr):
    rng = np.random.default_rng(SEED)                 # same seed per config: planting is blind & fixed
    N = ntr + NTEST
    zA, zB = rng.normal(size=(N, D)), rng.normal(size=(N, D))
    y = make_target(target, zA, zB, rng)
    fA, fB = zA.copy(), zB.copy()                     # "frozen encoder" features = the true factors
    eA, eB = entangle(ent, fA, fB, rng)
    true_oracle = readout(np.hstack([fA, fB]), y, ntr)
    single = max(readout(eA, y, ntr), readout(eB, y, ntr))
    entangled_oracle = readout(np.hstack([eA, eB]), y, ntr)          # universal readout on entangled
    naive_strong = readout(np.hstack([whiten(eA), whiten(eB)]), y, ntr)  # best linear align + MLP
    c_i = (true_oracle - single) >= 0.15
    c_ii = (true_oracle - naive_strong) >= 0.10
    c_iii = true_oracle >= 0.80
    return {
        "config": cid, "target": target, "entangle": ent, "n_train": ntr,
        "best_single": single, "true_oracle": true_oracle,
        "entangled_oracle": entangled_oracle, "naive_strong": naive_strong,
        "info_retained (entangled_oracle/true_oracle)": entangled_oracle / true_oracle,
        "naive_reaches_entangled_max (naive/entangled_oracle)": naive_strong / (entangled_oracle + 1e-9),
        "cond_i_complementarity": bool(c_i), "cond_ii_naive_fails": bool(c_ii),
        "cond_iii_oracle_reachable": bool(c_iii), "valid": bool(c_i and c_ii and c_iii),
    }


def main():
    rows = []
    print(f"{'cfg':4s} {'target':8s} {'ent':3s} {'single':>7s} {'true_orc':>8s} "
          f"{'ent_orc':>8s} {'naive':>7s} | i  ii iii  VALID")
    for cid, target, ent, ntr in FAMILY:
        r = run_config(cid, target, ent, ntr)
        rows.append(r)
        print(f"{cid:4s} {target:8s} {ent:3s} {r['best_single']:7.3f} {r['true_oracle']:8.3f} "
              f"{r['entangled_oracle']:8.3f} {r['naive_strong']:7.3f} | "
              f"{'Y' if r['cond_i_complementarity'] else '.':2s} "
              f"{'Y' if r['cond_ii_naive_fails'] else '.':2s} "
              f"{'Y' if r['cond_iii_oracle_reachable'] else '.':3s} "
              f"{'*** VALID ***' if r['valid'] else ''}", flush=True)
    valid = [r for r in rows if r["valid"]]
    result = {"family_size": len(rows), "valid_configs": [r["config"] for r in valid],
              "stage1_pass": len(valid) > 0, "rows": rows}
    print(f"\nSTAGE 1 VALIDITY: {len(valid)}/{len(rows)} configs valid -> "
          f"{'PASS -> proceed to Stage 2 on valid configs' if valid else 'FAIL -> informative regime NOT blindly constructible; composition thesis closes'}")
    # the key structural diagnostic
    inv = [r for r in rows if r["entangle"] in ("E1", "E2", "E3")]
    print(f"\nDIagnostic — on invertible-gauge configs, naive_strong / entangled_oracle = "
          f"{np.mean([r['naive_reaches_entangled_max (naive/entangled_oracle)'] for r in inv]):.3f} "
          f"(≈1 => naive is already the universal readout; no room for a blind indexer)")
    json.dump(result, open(os.path.join(HERE, "stage1_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
