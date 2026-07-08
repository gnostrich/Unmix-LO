"""
GATE ZERO (cold-split) — the confound-controlled re-test (thresholds in PREREG.md).

Same encoders (ESM-2, ChemBERTa), same DAVIS data, same two conditions as the in-distribution
gate. The ONLY change: entity-disjoint splits so no drug/target in test was seen in train,
removing the marginal-promiscuity leakage that inflated single-specialist accuracy in-dist.

Runs cold-drug, cold-target, and cold-pair; reports each with the full decomposition and a
side-by-side vs the in-distribution numbers. Encoder-collapse guard included (per PREREG.md).
"""
import os, sys, json, pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from embed_specialists import embed

DATA = os.path.join(HERE, "..", "gate0", "data")
PKD_THRESH = 7.0
SEED = 0


def load():
    lig = json.load(open(os.path.join(DATA, "ligands_can.txt")))
    pro = json.load(open(os.path.join(DATA, "proteins.txt")))
    Y = np.array(pickle.load(open(os.path.join(DATA, "Y"), "rb"), encoding="latin1"), dtype=float)
    label = (-np.log10(Y / 1e9) >= PKD_THRESH).astype(int)     # (n_drug, n_prot)
    Ep = embed("esm2_protein", [pro[p] for p in pro])
    Em = embed("chemberta_mol", [lig[d] for d in lig])
    return Ep, Em, label


def disjoint3(n, rng, fr=(0.6, 0.15, 0.25)):
    perm = rng.permutation(n)
    a, b = int(fr[0] * n), int((fr[0] + fr[1]) * n)
    return perm[:a], perm[a:b], perm[b:]


def tune(prob, y):
    best_t, best = 0.5, -1
    for t in np.quantile(prob, np.linspace(0.05, 0.95, 37)):
        ba = balanced_accuracy_score(y, (prob >= t).astype(int))
        if ba > best:
            best, best_t = ba, t
    return best_t


def probes_on(feat, y, tr, va, te):
    out = {}
    for tag in ("protein_only", "molecule_only", "union"):
        X = feat[tag]
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
        pv, pt = clf.predict_proba(sc.transform(X[va]))[:, 1], clf.predict_proba(sc.transform(X[te]))[:, 1]
        out[tag] = {"thr": tune(pv, y[va]), "pt": pt,
                    "auroc": float(roc_auc_score(y[te], pt)) if len(np.unique(y[te])) > 1 else float("nan"),
                    "auprc": float(average_precision_score(y[te], pt)) if y[te].sum() else float("nan")}
    return out


def evaluate(Ep, Em, label, mode, rng):
    nd, npr = label.shape
    di, pi = np.meshgrid(np.arange(nd), np.arange(npr), indexing="ij")
    di, pi, y = di.ravel(), pi.ravel(), label.ravel()
    dtr, dva, dte = disjoint3(nd, rng)
    ptr, pva, pte = disjoint3(npr, rng)
    ds, vs, ts = set(dtr), set(dva), set(dte)
    ps, qs, rs = set(ptr), set(pva), set(pte)
    if mode == "cold_drug":                       # targets shared; drugs disjoint
        m_tr = np.isin(di, dtr); m_va = np.isin(di, dva); m_te = np.isin(di, dte)
    elif mode == "cold_target":                   # drugs shared; targets disjoint
        m_tr = np.isin(pi, ptr); m_va = np.isin(pi, pva); m_te = np.isin(pi, pte)
    else:                                         # cold_pair: BOTH disjoint (cross pairs dropped)
        m_tr = np.isin(di, dtr) & np.isin(pi, ptr)
        m_va = np.isin(di, dva) & np.isin(pi, pva)
        m_te = np.isin(di, dte) & np.isin(pi, pte)
    tr, va, te = np.where(m_tr)[0], np.where(m_va)[0], np.where(m_te)[0]
    feat = {"protein_only": Ep[pi], "molecule_only": Em[di],
            "union": np.concatenate([Ep[pi], Em[di]], axis=1)}
    P = probes_on(feat, y, tr, va, te)

    yte = y[te]
    best_single_name = max(("protein_only", "molecule_only"), key=lambda k: P[k]["auprc"])
    best_single = P[best_single_name]["auprc"]
    ratio = P["union"]["auprc"] / (best_single + 1e-9)

    # split-knowledge fraction on class-balanced cold-test subset
    pos = np.where(yte == 1)[0]; neg = np.where(yte == 0)[0]
    nb = min(len(pos), len(neg))
    if nb == 0:
        return {"error": "cold-test has one class only", "n_test": int(len(te))}
    bal = np.concatenate([pos[:nb], rng.permutation(neg)[:nb]])
    cor = lambda tag: ((P[tag]["pt"][bal] >= P[tag]["thr"]).astype(int) == yte[bal])
    cp, cm, cu = cor("protein_only"), cor("molecule_only"), cor("union")
    bw = (~cp) & (~cm)
    split_frac = float((bw & cu).mean())

    c1, c2 = split_frac >= 0.30, ratio >= 1.10
    return {
        "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
        "test_binder_rate": float(yte.mean()), "balanced_subset_n": int(len(bal)),
        "auroc": {k: P[k]["auroc"] for k in P}, "auprc": {k: P[k]["auprc"] for k in P},
        "best_single": {"name": best_single_name, "auprc": best_single},
        "union_over_best_single_auprc_ratio": float(ratio),
        "split_knowledge_fraction": split_frac,
        "decomposition": {
            "p_protein_correct": float(cp.mean()), "p_molecule_correct": float(cm.mean()),
            "p_union_correct": float(cu.mean()), "p_both_singles_wrong": float(bw.mean()),
            "union_rescue_rate_when_both_wrong": float(cu[bw].mean()) if bw.sum() else 0.0,
            "single_error_correlation": float(np.corrcoef((~cp).astype(float), (~cm).astype(float))[0, 1]),
        },
        "cond1_split_ge_0.30": bool(c1), "cond2_auprc_ratio_ge_1.10": bool(c2),
        "pass": bool(c1 and c2),
    }


def main():
    Ep, Em, label = load()
    print(f"DAVIS embeddings: protein {Ep.shape}, molecule {Em.shape}, binder rate {label.mean():.3f}\n")
    res = {"in_distribution_reference": {"split_knowledge_fraction": 0.005,
                                         "union_over_best_single_auprc_ratio": 1.37,
                                         "single_error_correlation": -0.18,
                                         "p_both_singles_wrong": 0.051}}
    for mode in ("cold_drug", "cold_target", "cold_pair"):
        r = evaluate(Ep, Em, label, mode, np.random.default_rng(SEED))
        res[mode] = r
        if "error" in r:
            print(f"{mode}: {r['error']}"); continue
        d = r["decomposition"]
        print(f"{mode}: test n={r['n_test']} binder {r['test_binder_rate']:.3f}")
        print(f"  AUROC prot {r['auroc']['protein_only']:.3f} mol {r['auroc']['molecule_only']:.3f} "
              f"uni {r['auroc']['union']:.3f} | AUPRC prot {r['auprc']['protein_only']:.3f} "
              f"mol {r['auprc']['molecule_only']:.3f} uni {r['auprc']['union']:.3f}")
        print(f"  split-frac {r['split_knowledge_fraction']:.3f} (>=0.30 {'PASS' if r['cond1_split_ge_0.30'] else 'FAIL'}) | "
              f"AUPRC ratio {r['union_over_best_single_auprc_ratio']:.2f} (>=1.10 {'PASS' if r['cond2_auprc_ratio_ge_1.10'] else 'FAIL'})")
        print(f"  both-wrong {d['p_both_singles_wrong']:.3f}, rescue {d['union_rescue_rate_when_both_wrong']:.3f}, "
              f"err-corr {d['single_error_correlation']:.3f}  -> {mode} {'PASS' if r['pass'] else 'FAIL'}\n")

    # overall: primary modes are cold_drug and cold_target (cold_pair supporting)
    primary = [res["cold_drug"], res["cold_target"]]
    overall = all(m.get("pass") for m in primary)
    res["overall_pass_primary_cold_splits"] = bool(overall)
    # encoder-collapse guard
    collapsed = any(m.get("auroc", {}).get("union", 1) < 0.6 for m in primary if "auroc" in m)
    res["encoder_collapse_flag"] = bool(collapsed)
    print(f"OVERALL (cold_drug AND cold_target): {'PASS -> gate re-opens' if overall else 'FAIL -> confound-controlled negative'}")
    if collapsed:
        print("  NOTE: union AUROC < 0.6 on a cold split — frozen encoders may not transfer to unseen "
              "entities; this is a DIFFERENT finding (see PREREG.md), not the split-knowledge verdict.")
    json.dump(res, open(os.path.join(HERE, "gate0cold_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
