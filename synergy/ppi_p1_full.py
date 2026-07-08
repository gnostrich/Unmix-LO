"""
P1 on D-SCRIPT human PPI — properly powered. PPI is genuinely combinatorial (a single protein
cannot determine a pairwise interaction), the fairest venue for cold-split complementarity.

Powering fix vs the first underpowered run (256 pairs): use the top-2000 highest-degree proteins
(a dense subgraph → thousands of within-split pairs; hub-bias makes best-single STRONGER, i.e.
conservative for P1), split disjoint (cold), balance 1:1. Same interaction-capable + linear
readouts as the DAVIS reconciliation. P1 holds iff oracle balanced-acc − best-single ≥ 0.15.
"""
import os, sys, json, random
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "biomesh"))
from embed_specialists import embed
DS = os.path.join(HERE, "data")
SEED = 0
K_PROT = 2000


def load_fasta(path):
    seqs, cur, buf = {}, None, []
    for line in open(path):
        if line.startswith(">"):
            if cur:
                seqs[cur] = "".join(buf)
            cur, buf = line[1:].strip(), []
        else:
            buf.append(line.strip())
    if cur:
        seqs[cur] = "".join(buf)
    return seqs


def readouts(Xtr, ytr, Xte, yte):
    out = {}
    for name, clf in [("mlp", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400,
                                            early_stopping=True, n_iter_no_change=15, random_state=0)),
                      ("logistic", LogisticRegression(max_iter=2000, class_weight="balanced"))]:
        pipe = make_pipeline(StandardScaler(), clf); pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]
        out[name] = {"balacc": float(balanced_accuracy_score(yte, pipe.predict(Xte))),
                     "auroc": float(roc_auc_score(yte, p)),
                     "auprc": float(average_precision_score(yte, p))}
    return out


def main():
    seqs = load_fasta(os.path.join(DS, "ppi_seqs.fasta"))
    rng = random.Random(SEED)
    pool = [l.split("\t") for l in open(os.path.join(DS, "ppi_train.tsv"))] + \
           [l.split("\t") for l in open(os.path.join(DS, "ppi_test.tsv"))]
    pool = [(a, b, int(c)) for a, b, c in pool if a in seqs and b in seqs]
    deg = {}
    for a, b, _ in pool:
        deg[a] = deg.get(a, 0) + 1; deg[b] = deg.get(b, 0) + 1
    top = sorted(deg, key=deg.get, reverse=True)[:K_PROT]
    rng.shuffle(top)
    ptr, pte = set(top[:int(.62 * K_PROT)]), set(top[int(.62 * K_PROT):])   # disjoint COLD split

    def collect(pset):
        pos = [(a, b) for a, b, c in pool if c == 1 and a in pset and b in pset]
        neg = [(a, b) for a, b, c in pool if c == 0 and a in pset and b in pset]
        rng.shuffle(pos); rng.shuffle(neg)
        n = min(len(pos), len(neg))
        pos, neg = pos[:n], neg[:n]                                          # 1:1 balance
        pairs = [(a, b, 1) for a, b in pos] + [(a, b, 0) for a, b in neg]
        rng.shuffle(pairs)
        return pairs
    train_pairs, test_pairs = collect(ptr), collect(pte)
    used = sorted({p for a, b, _ in train_pairs + test_pairs for p in (a, b)})
    print(f"PPI powered: {len(train_pairs)} train, {len(test_pairs)} test pairs; "
          f"{len(used)} proteins to embed (cold: {len(ptr)}tr/{len(pte)}te disjoint)", flush=True)
    E = embed("esm2_protein", [seqs[p][:1022] for p in used])
    idx = {p: i for i, p in enumerate(used)}

    def feats(pairs):
        A = np.array([E[idx[a]] for a, b, c in pairs]); B = np.array([E[idx[b]] for a, b, c in pairs])
        return A, B, np.array([c for a, b, c in pairs])
    Atr, Btr, ytr = feats(train_pairs); Ate, Bte, yte = feats(test_pairs)
    rA = readouts(Atr, ytr, Ate, yte); rB = readouts(Btr, ytr, Bte, yte)
    rU = readouts(np.hstack([Atr, Btr]), ytr, np.hstack([Ate, Bte]), yte)
    res = {"n_train": len(train_pairs), "n_test": len(test_pairs), "n_proteins": len(used),
           "single_A": rA, "single_B": rB, "oracle": rU}
    print(f"{'readout':8s} {'single_bal':>10s} {'oracle_bal':>10s} {'gap':>7s} {'orc_auc':>8s} P1(>=.15)")
    for ro in ("mlp", "logistic"):
        best = max(rA[ro]["balacc"], rB[ro]["balacc"]); gap = rU[ro]["balacc"] - best
        res[f"P1_{ro}"] = {"gap": gap, "holds": bool(gap >= 0.15)}
        print(f"{ro:8s} {best:10.3f} {rU[ro]['balacc']:10.3f} {gap:+7.3f} {rU[ro]['auroc']:8.3f} "
              f"{'HOLDS' if gap >= 0.15 else 'fails'}")
    holds = res["P1_mlp"]["holds"]
    res["P1_holds"] = bool(holds)
    print(f"\nPPI P1 (powered, cold): {'HOLDS -> genuine combinatorial complementarity; build aggregator' if holds else 'FAILS -> even a genuinely combinatorial task lacks cold-split complementarity these frozen encoders capture'}")
    json.dump(res, open(os.path.join(HERE, "ppi_p1_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
