"""
SYNERGY — precondition P1: does complementarity EXIST under a cold (entity-disjoint) split?
(PREREG.md P1). If best-single << joint-oracle (gap >= 0.15) on the cold split, complementarity is
real and we may build the aggregator. If not -> STOP (BIOMESH-DTI lesson: no synergy to deliver).

Tested on two real tasks:
  DAVIS DTI (protein+molecule) — the canonical composite task (cached embeddings). BIOMESH already
    found cold-split union <= best-single; re-confirmed here with a strong MLP readout.
  D-SCRIPT human PPI (protein+protein) — genuinely COMBINATORIAL: a single protein cannot determine
    a pairwise interaction (no marginal shortcut like DTI promiscuity), so P1 has a real chance.
    Cold split = disjoint protein sets between train and test.
"""
import os, sys, json, pickle, random
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "biomesh"))
from embed_specialists import embed
SEED = 0


def readout(Xtr, ytr, Xte, yte, seed=0):
    clf = make_pipeline(StandardScaler(),
                        MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400,
                                      early_stopping=True, n_iter_no_change=15, random_state=seed))
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    return (float(roc_auc_score(yte, p)) if len(np.unique(yte)) > 1 else float("nan"),
            float(balanced_accuracy_score(yte, clf.predict(Xte))))


def p1_verdict(name, singleA, singleB, oracle):
    best_single = max(singleA[1], singleB[1])              # balanced-accuracy primary
    gap = oracle[1] - best_single
    holds = gap >= 0.15
    print(f"  {name}: single_A bal={singleA[1]:.3f}(auc {singleA[0]:.3f}) "
          f"single_B bal={singleB[1]:.3f}(auc {singleB[0]:.3f}) "
          f"oracle bal={oracle[1]:.3f}(auc {oracle[0]:.3f})  gap={gap:+.3f}  "
          f"P1 {'HOLDS' if holds else 'FAILS'}")
    return {"single_A": singleA, "single_B": singleB, "oracle": oracle,
            "best_single_balacc": best_single, "gap": gap, "P1_holds": bool(holds)}


# ---------------- DAVIS DTI (cached embeddings, cold split) ----------------
def davis_p1():
    D = os.path.join(HERE, "..", "gate0", "data")
    lig = json.load(open(os.path.join(D, "ligands_can.txt")))
    pro = json.load(open(os.path.join(D, "proteins.txt")))
    Y = np.array(pickle.load(open(os.path.join(D, "Y"), "rb"), encoding="latin1"), dtype=float)
    lab = (-np.log10(Y / 1e9) >= 7).astype(int)
    Ep = embed("esm2_protein", [pro[p] for p in pro])
    Em = embed("chemberta_mol", [lig[d] for d in lig])
    nd, npr = lab.shape
    rng = np.random.default_rng(SEED)
    dtr = set(rng.permutation(nd)[:int(.6 * nd)]); pte = rng.permutation(npr)
    ptr, ptes = set(pte[:int(.6 * npr)]), set(pte[int(.6 * npr):])
    # cold-target split: drugs shared, targets disjoint (BIOMESH's most complementarity-favorable)
    di, pi = np.meshgrid(np.arange(nd), np.arange(npr), indexing="ij")
    di, pi, y = di.ravel(), pi.ravel(), lab.ravel()
    m_tr = np.isin(pi, list(ptr)); m_te = np.isin(pi, list(ptes))
    tr, te = np.where(m_tr)[0], np.where(m_te)[0]
    XA = Ep[pi]; XB = Em[di]; XU = np.concatenate([XA, XB], 1)
    sA = readout(XA[tr], y[tr], XA[te], y[te])
    sB = readout(XB[tr], y[tr], XB[te], y[te])
    orc = readout(XU[tr], y[tr], XU[te], y[te])
    return p1_verdict("DAVIS-DTI cold-target", sA, sB, orc)


# ---------------- D-SCRIPT PPI (genuinely combinatorial, cold split) ----------------
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


def ppi_p1():
    dsdir = os.path.join(HERE, "data")
    seqs = load_fasta(os.path.join(dsdir, "ppi_seqs.fasta"))
    rng = random.Random(SEED)

    def read_pairs(fn, cap):
        rows = [l.split("\t") for l in open(os.path.join(dsdir, fn))]
        rows = [(a, b, int(c)) for a, b, c in rows if a in seqs and b in seqs]
        rng.shuffle(rows)
        return rows[:cap]
    pool = read_pairs("ppi_train.tsv", 120000) + read_pairs("ppi_test.tsv", 52725)

    deg = {}
    for a, b, _ in pool:
        deg[a] = deg.get(a, 0) + 1; deg[b] = deg.get(b, 0) + 1
    prots = [p for p, d in deg.items() if d >= 3]
    rng.shuffle(prots)
    prots = prots[:2400]
    ptr = set(prots[:1500]); pte = set(prots[1500:])          # disjoint COLD split

    def build(pool, pset, want_pos):
        pos = [(a, b) for a, b, c in pool if c == 1 and a in pset and b in pset]
        neg = [(a, b) for a, b, c in pool if c == 0 and a in pset and b in pset]
        rng.shuffle(pos); rng.shuffle(neg)
        pos = pos[:want_pos]; neg = neg[:len(pos)]            # 1:1 balance
        pairs = [(a, b, 1) for a, b in pos] + [(a, b, 0) for a, b in neg]
        rng.shuffle(pairs)
        return pairs
    train_pairs = build(pool, ptr, 3000)
    test_pairs = build(pool, pte, 900)
    used = sorted({p for a, b, _ in train_pairs + test_pairs for p in (a, b)})
    print(f"  PPI: {len(train_pairs)} train pairs, {len(test_pairs)} test pairs, "
          f"{len(used)} proteins to embed (cold: {len(ptr)}tr/{len(pte)}te disjoint)", flush=True)
    E = embed("esm2_protein", [seqs[p][:1022] for p in used])
    idx = {p: i for i, p in enumerate(used)}

    def feats(pairs):
        A = np.array([E[idx[a]] for a, b, c in pairs])
        B = np.array([E[idx[b]] for a, b, c in pairs])
        y = np.array([c for a, b, c in pairs])
        return A, B, y
    Atr, Btr, ytr = feats(train_pairs); Ate, Bte, yte = feats(test_pairs)
    sA = readout(Atr, ytr, Ate, yte)
    sB = readout(Btr, ytr, Bte, yte)
    orc = readout(np.hstack([Atr, Btr]), ytr, np.hstack([Ate, Bte]), yte)
    return p1_verdict("PPI cold-both", sA, sB, orc)


def main():
    print("P1 precondition — complementarity under cold split (gap >= 0.15 to proceed):")
    res = {"davis_dti": davis_p1(), "ppi": ppi_p1()}
    any_holds = res["davis_dti"]["P1_holds"] or res["ppi"]["P1_holds"]
    res["P1_any_task_holds"] = bool(any_holds)
    print(f"\nP1 overall: {'HOLDS on >=1 real task -> may build aggregator on it' if any_holds else 'FAILS on all tested real tasks -> STOP (no complementarity to deliver)'}")
    json.dump(res, open(os.path.join(HERE, "p1_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
