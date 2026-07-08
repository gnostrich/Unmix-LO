"""
P1 reconciliation on DAVIS DTI — is cold-split complementarity real, and does it depend on the
readout? BIOMESH gate0cold used a LINEAR logistic probe and found union <= best-single ("composition
hurts"). Here we test whether an INTERACTION-CAPABLE (MLP) readout changes that, across ALL three
cold splits including the strictest (cold-pair). Binding is inherently interactive, so a linear pool
may miss complementarity a nonlinear readout captures. This determines the synergy P1 honestly and
reconciles it with the BIOMESH result.

For each cold split x readout: best-single vs joint-oracle, reported as balanced accuracy (primary,
P1 gap>=0.15), AUROC, AUPRC. Same readout for single and oracle => the prereg capacity control is
inherent (a gap is cross-model info, not capacity).
"""
import os, sys, json, pickle
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "biomesh"))
from embed_specialists import embed
D = os.path.join(HERE, "..", "biomesh", "gate0", "data")
SEED = 0


def readouts(Xtr, ytr, Xte, yte):
    out = {}
    for name, clf in [("mlp", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400,
                                            early_stopping=True, n_iter_no_change=15, random_state=0)),
                      ("logistic", LogisticRegression(max_iter=2000, class_weight="balanced"))]:
        pipe = make_pipeline(StandardScaler(), clf)
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]
        out[name] = {"balacc": float(balanced_accuracy_score(yte, pipe.predict(Xte))),
                     "auroc": float(roc_auc_score(yte, p)),
                     "auprc": float(average_precision_score(yte, p))}
    return out


def main():
    lig = json.load(open(os.path.join(D, "ligands_can.txt")))
    pro = json.load(open(os.path.join(D, "proteins.txt")))
    Y = np.array(pickle.load(open(os.path.join(D, "Y"), "rb"), encoding="latin1"), dtype=float)
    lab = (-np.log10(Y / 1e9) >= 7).astype(int)
    Ep = embed("esm2_protein", [pro[p] for p in pro])
    Em = embed("chemberta_mol", [lig[d] for d in lig])
    nd, npr = lab.shape
    di, pi = np.meshgrid(np.arange(nd), np.arange(npr), indexing="ij")
    di, pi, y = di.ravel(), pi.ravel(), lab.ravel()
    XA, XB = Ep[pi], Em[di]
    XU = np.concatenate([XA, XB], 1)
    rng = np.random.default_rng(SEED)
    dperm, pperm = rng.permutation(nd), rng.permutation(npr)
    dtr, dte = set(dperm[:int(.6 * nd)]), set(dperm[int(.6 * nd):])
    ptr, pte = set(pperm[:int(.6 * npr)]), set(pperm[int(.6 * npr):])

    splits = {
        "cold_drug":   (np.isin(di, list(dtr)), np.isin(di, list(dte))),
        "cold_target": (np.isin(pi, list(ptr)), np.isin(pi, list(pte))),
        "cold_pair":   (np.isin(di, list(dtr)) & np.isin(pi, list(ptr)),
                        np.isin(di, list(dte)) & np.isin(pi, list(pte))),
    }
    results = {}
    print(f"{'split':11s} {'readout':8s} {'single_bal':>10s} {'oracle_bal':>10s} {'gap':>7s} "
          f"{'orc_auc':>8s} P1(>=.15)")
    for sname, (mtr, mte) in splits.items():
        tr, te = np.where(mtr)[0], np.where(mte)[0]
        rA = readouts(XA[tr], y[tr], XA[te], y[te])
        rB = readouts(XB[tr], y[tr], XB[te], y[te])
        rU = readouts(XU[tr], y[tr], XU[te], y[te])
        results[sname] = {"single_A": rA, "single_B": rB, "oracle": rU,
                          "n_train": int(len(tr)), "n_test": int(len(te))}
        for ro in ("mlp", "logistic"):
            best_single = max(rA[ro]["balacc"], rB[ro]["balacc"])
            gap = rU[ro]["balacc"] - best_single
            results[sname][f"P1_{ro}"] = {"gap": gap, "holds": bool(gap >= 0.15)}
            print(f"{sname:11s} {ro:8s} {best_single:10.3f} {rU[ro]['balacc']:10.3f} {gap:+7.3f} "
                  f"{rU[ro]['auroc']:8.3f} {'HOLDS' if gap >= 0.15 else 'fails'}")
    # the decisive question: does P1 hold under the STRICTEST split (cold_pair) with a strong readout?
    cp_mlp = results["cold_pair"]["P1_mlp"]["holds"]
    results["P1_decisive_cold_pair_mlp"] = bool(cp_mlp)
    print(f"\nDecisive (cold_pair, MLP): P1 {'HOLDS -> complementarity real under the strictest split' if cp_mlp else 'FAILS -> the cold-target hold was the shared-drug marginal; no robust complementarity'}")
    print("Reconciliation with BIOMESH: compare mlp vs logistic rows — if MLP holds where logistic "
          "fails, DTI complementarity is interactive (a linear pool misses it).")
    json.dump(results, open(os.path.join(HERE, "davis_p1_full_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
