"""
GATE ZERO — do DAVIS DTI queries genuinely need multiple specialists? (thresholds in PREREG.md)

Embeds the 442 targets (ESM-2) and 68 drugs (ChemBERTa), builds per-pair features, trains
identical logistic-regression probes on protein-only / molecule-only / union, and applies the
two pre-committed conditions. Writes gate0_results.json.
"""
import os, sys, json, pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from embed_specialists import embed

DATA = os.path.join(HERE, "data")
PKD_THRESH = 7.0
SEED = 0


def load_davis():
    ligands = json.load(open(os.path.join(DATA, "ligands_can.txt")))
    proteins = json.load(open(os.path.join(DATA, "proteins.txt")))
    Y = np.array(pickle.load(open(os.path.join(DATA, "Y"), "rb"), encoding="latin1"), dtype=float)
    drug_ids, prot_ids = list(ligands), list(proteins)
    pKd = -np.log10(Y / 1e9)
    label = (pKd >= PKD_THRESH).astype(int)          # (n_drug, n_prot)
    return ligands, proteins, drug_ids, prot_ids, label


def tune_threshold(prob, y):
    best_t, best_ba = 0.5, -1
    for t in np.quantile(prob, np.linspace(0.05, 0.95, 37)):
        ba = balanced_accuracy_score(y, (prob >= t).astype(int))
        if ba > best_ba:
            best_ba, best_t = ba, t
    return best_t


def fit_probe(Xtr, ytr):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf.fit(sc.transform(Xtr), ytr)
    return sc, clf


def prob_of(sc, clf, X):
    return clf.predict_proba(sc.transform(X))[:, 1]


def main():
    ligands, proteins, drug_ids, prot_ids, label = load_davis()
    print(f"DAVIS: {len(drug_ids)} drugs x {len(prot_ids)} targets, "
          f"binder rate {label.mean():.3f}", flush=True)

    Ep = embed("esm2_protein", [proteins[p] for p in prot_ids])       # (442, 320)
    Em = embed("chemberta_mol", [ligands[d] for d in drug_ids])       # (68, 384)
    print(f"protein emb {Ep.shape}, molecule emb {Em.shape}", flush=True)

    # all pairs
    di, pi = np.meshgrid(np.arange(len(drug_ids)), np.arange(len(prot_ids)), indexing="ij")
    di, pi, y = di.ravel(), pi.ravel(), label.ravel()
    Xp = Ep[pi]                                   # protein-only
    Xm = Em[di]                                   # molecule-only
    Xu = np.concatenate([Xp, Xm], axis=1)         # union

    # stratified 70/15/15 split, seed 0
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(y))
    pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
    def split3(a):
        n = len(a); return a[:int(.7*n)], a[int(.7*n):int(.85*n)], a[int(.85*n):]
    trp, vap, tep = split3(pos); trn, van, ten = split3(neg)
    tr = np.concatenate([trp, trn]); va = np.concatenate([vap, van]); te = np.concatenate([tep, ten])

    res = {"n_pairs": len(y), "binder_rate": float(y.mean()),
           "n_train": len(tr), "n_val": len(va), "n_test": len(te)}
    probes = {}
    for tagname, X in [("protein_only", Xp), ("molecule_only", Xm), ("union", Xu)]:
        sc, clf = fit_probe(X[tr], y[tr])
        pv, pt = prob_of(sc, clf, X[va]), prob_of(sc, clf, X[te])
        thr = tune_threshold(pv, y[va])
        probes[tagname] = {"thr": thr, "prob_test": pt}
        res[tagname] = {"auroc": float(roc_auc_score(y[te], pt)),
                        "auprc": float(average_precision_score(y[te], pt))}
        print(f"  {tagname:14s}: AUROC {res[tagname]['auroc']:.3f}  AUPRC {res[tagname]['auprc']:.3f}", flush=True)

    best_single = max(res["protein_only"]["auprc"], res["molecule_only"]["auprc"])
    best_single_name = "protein_only" if res["protein_only"]["auprc"] >= res["molecule_only"]["auprc"] else "molecule_only"

    # --- condition 1: split-knowledge fraction on class-balanced test subset
    yte = y[te]
    pos_te = np.where(yte == 1)[0]; neg_te = np.where(yte == 0)[0]
    nb = min(len(pos_te), len(neg_te))
    bal = np.concatenate([pos_te[:nb], rng.permutation(neg_te)[:nb]])
    def correct(tag):
        p = probes[tag]["prob_test"][bal]
        return (p >= probes[tag]["thr"]).astype(int) == yte[bal]
    cp, cm, cu = correct("protein_only"), correct("molecule_only"), correct("union")
    split_mask = (~cp) & (~cm) & cu
    split_fraction = float(split_mask.mean())
    res["balanced_subset_n"] = int(len(bal))
    res["split_knowledge_fraction"] = split_fraction

    # --- condition 2: pooling beats best-single
    union_auprc = res["union"]["auprc"]
    auprc_ratio = union_auprc / (best_single + 1e-9)
    res["best_single"] = {"name": best_single_name, "auprc": best_single}
    res["union_over_best_single_auprc_ratio"] = auprc_ratio

    c1 = split_fraction >= 0.30
    c2 = auprc_ratio >= 1.10
    res["pass"] = bool(c1 and c2)
    print(f"\n  condition 1 split-knowledge fraction = {split_fraction:.3f} (>= 0.30): {'PASS' if c1 else 'FAIL'}")
    print(f"  condition 2 union AUPRC {union_auprc:.3f} / best-single ({best_single_name}) {best_single:.3f} "
          f"= {auprc_ratio:.2f}x (>= 1.10): {'PASS' if c2 else 'FAIL'}")
    print(f"\n  GATE ZERO: {'PASS -> proceed to experiment/' if res['pass'] else 'FAIL -> no composite need; stop and report'}")

    for tag in probes:                           # strip arrays before dumping
        probes[tag].pop("prob_test", None)
    json.dump(res, open(os.path.join(HERE, "gate0_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
