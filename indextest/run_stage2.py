"""
STAGE 2 — RESULTS TEST (only the Stage-1-valid configs). Thresholds frozen in PREREG.md.

Arms (equal readout capacity + equal tuning budget; all use the same MLP readout):
  best_single    : floor (readout on one entangled view)
  naive_strong   : whiten/align both entangled views + MLP  (the brief's strong naive)
  naive_poly     : + WITHIN-view degree-2 products (same feature budget as indexed, but no
                   cross-view terms) — fairness control: isolates whether a WIN is specifically
                   cross-view relational structure vs. just more polynomial capacity
  INDEXED        : Baur×MZ connective tissue = whiten + CROSS-view bilinear products (blind to y
                   and to the mixing) + MLP. The relational frame between the two entangled spaces.
  true_oracle    : ceiling (readout on the true unentangled features)

Plus the MANDATORY no-complementarity control per config: a twin whose target is determined by
ONE encoder alone. Indexed must NOT beat naive there (anti-hallucination).

PRE-COMMITTED PASS (all required): indexed >= 1.15x naive_strong on complementarity configs AND
indexed approaches oracle AND indexed does NOT beat naive on the control. FAIL on any.
CIs from multiple seeds.
"""
import os, json
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
D = 24
R = 10                      # PCA rank per view for the connective frame
NTEST = 1000


def readout(X, y, ntr, seed):
    clf = make_pipeline(StandardScaler(),
                        MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400,
                                      early_stopping=True, n_iter_no_change=15, random_state=seed))
    clf.fit(X[:ntr], y[:ntr])
    return float(clf.score(X[ntr:], y[ntr:]))


def whiten(X):
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt.T / (S / np.sqrt(len(X)) + 1e-6)


def pca_r(X, r):
    r = min(r, X.shape[1])
    return StandardScaler().fit_transform(PCA(n_components=r, random_state=0).fit_transform(X))


def cross_products(A, B):
    return (A[:, :, None] * B[:, None, :]).reshape(len(A), -1)   # all a_i * b_j


def within_products(A):
    idx = np.triu_indices(A.shape[1])
    return (A[:, :, None] * A[:, None, :]).reshape(len(A), -1)[:, np.ravel_multi_index(idx, (A.shape[1], A.shape[1]))]


def make_target(kind, zA, zB, rng):
    wA, wB, wg = rng.normal(size=D), rng.normal(size=D), rng.normal(size=D)
    gA, gB = zA @ wA, zB @ wB
    if kind == "additive":
        y = np.sign(gA + gB)
    elif kind == "xor":
        y = np.sign(gA * gB)
    elif kind == "gate":
        y = np.sign(np.where(zA @ wg > 0, gB, -gB))
    else:  # single_A — no-complementarity control target
        y = np.sign(gA)
    return (y > 0).astype(int)


def entangle(kind, fA, fB, rng):
    if kind == "E1":
        return fA @ rng.normal(size=(D, D)), fB @ rng.normal(size=(D, D))
    if kind == "E2":
        M = rng.normal(size=(2 * D, 2 * D)); E = np.hstack([fA, fB]) @ M
        return E[:, :D], E[:, D:]
    if kind == "E3":
        return np.tanh(fA @ rng.normal(size=(D, D))), np.tanh(fB @ rng.normal(size=(D, D)))
    k = D // 2
    return fA @ rng.normal(size=(D, k)), fB @ rng.normal(size=(D, k))


def arms(target, ent, ntr, seed):
    rng = np.random.default_rng(1000 + seed)
    N = ntr + NTEST
    zA, zB = rng.normal(size=(N, D)), rng.normal(size=(N, D))
    y = make_target(target, zA, zB, rng)
    eA, eB = entangle(ent, zA, zB, rng)
    wA, wB = whiten(eA), whiten(eB)
    pA, pB = pca_r(eA, R), pca_r(eB, R)
    lin = np.hstack([wA, wB])
    naive = readout(lin, y, ntr, seed)
    poly = readout(np.hstack([lin, within_products(pA), within_products(pB)]), y, ntr, seed)
    indexed = readout(np.hstack([lin, cross_products(pA, pB)]), y, ntr, seed)
    single = max(readout(eA, y, ntr, seed), readout(eB, y, ntr, seed))
    oracle = readout(np.hstack([zA, zB]), y, ntr, seed)
    return dict(single=single, naive=naive, poly=poly, indexed=indexed, oracle=oracle)


# Stage-1-valid configs (from stage1_results.json): invertible-gauge (the interesting ones) + lossy
MAIN = [("c4", "xor", "E2", 4000), ("c5", "xor", "E3", 4000),
        ("c8", "gate", "E2", 4000), ("c9", "gate", "E3", 4000)]
LOSSY = [("c2", "additive", "E4", 4000), ("c6", "xor", "E4", 4000), ("c12", "gate", "E4", 4000)]
SEEDS_MAIN, SEEDS_LOSSY = 4, 2


def agg(runs):
    return {k: [float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))]
            for k in runs[0]}


def main():
    results = {"configs": {}, "controls": {}}
    print(f"{'cfg':4s} {'tgt':5s}{'ent':4s} {'single':>7s}{'naive':>8s}{'poly':>8s}{'indexed':>8s}"
          f"{'oracle':>8s} {'idx/naive':>10s}")
    for group, seeds in [(MAIN, SEEDS_MAIN), (LOSSY, SEEDS_LOSSY)]:
        for cid, target, ent, ntr in group:
            runs = [arms(target, ent, ntr, s) for s in range(seeds)]
            a = agg(runs)
            ratio = a["indexed"][0] / (a["naive"][0] + 1e-9)
            gap_closed = (a["indexed"][0] - a["naive"][0]) / (a["oracle"][0] - a["naive"][0] + 1e-9)
            results["configs"][cid] = {"target": target, "entangle": ent, **a,
                                       "indexed_over_naive": ratio, "oracle_gap_closed": gap_closed}
            print(f"{cid:4s} {target:5s}{ent:4s} {a['single'][0]:7.3f}{a['naive'][0]:8.3f}"
                  f"{a['poly'][0]:8.3f}{a['indexed'][0]:8.3f}{a['oracle'][0]:8.3f} {ratio:9.2f}x", flush=True)
            # matched no-complementarity control (target = A alone), same entanglement
            cruns = [arms("single_A", ent, ntr, s) for s in range(seeds)]
            ca = agg(cruns)
            cratio = ca["indexed"][0] / (ca["naive"][0] + 1e-9)
            results["controls"][cid] = {**ca, "indexed_over_naive": cratio}

    # verdict on the invertible-gauge complementarity configs (the ones where info is present)
    comp = {c: results["configs"][c] for c in [m[0] for m in MAIN]}
    a_pass = all(v["indexed_over_naive"] >= 1.15 for v in comp.values())
    b_frac = np.mean([v["oracle_gap_closed"] for v in comp.values()])
    b_pass = b_frac >= 0.30
    ctrl_ok = all(results["controls"][c]["indexed_over_naive"] <= 1.05 for c in comp)
    overall = a_pass and b_pass and ctrl_ok
    results["verdict"] = {
        "a_indexed_ge_1.15x_naive_all_comp_configs": bool(a_pass),
        "b_indexed_approaches_oracle_meanfrac": float(b_frac), "b_pass": bool(b_pass),
        "c_no_hallucination_on_controls": bool(ctrl_ok), "PASS": bool(overall)}
    print(f"\n(a) indexed>=1.15x naive on ALL comp configs: {a_pass}")
    print(f"(b) indexed closes {100*b_frac:.0f}% of naive->oracle gap (>=30%): {b_pass}")
    print(f"(c) no hallucination on controls (indexed<=1.05x naive): {ctrl_ok}")
    for c in comp:
        cc = results["controls"][c]
        print(f"    control {c}: naive {cc['naive'][0]:.3f} indexed {cc['indexed'][0]:.3f} "
              f"({cc['indexed_over_naive']:.2f}x)")
    print(f"\nSTAGE 2: {'PASS — the narrow band is INHABITABLE (not that real tasks live there)' if overall else 'FAIL — indexing does not beat strong naive / or hallucinates'}")
    json.dump(results, open(os.path.join(HERE, "stage2_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
