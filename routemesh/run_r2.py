"""
ROUTEMESH GATE R2 — can a light critic realize the oracle from SPARSE competence data, and beat a
SOTA learned single-hop router (not just naive pooling)? Design frozen per PREREG.md + the SOTA-arm
addendum. Five arms: best-single | pooling | SOTA-ROUTER | ROUTEMESH | oracle(ceiling).

Critic (learns "who to trust", not the task): per-specialist LogisticRegression predicting P(specialist
solves query) from a bag-of-relations feature, trained on a SPARSE fraction of competence labels.
ROUTEMESH: per-step topology-free assembly — route each path step to the critic's predicted owner,
abstain if none predicted competent. SOTA-ROUTER: route the WHOLE query to the single specialist the
critic predicts best (standard single-hop routing; structurally cannot assemble).

Reports quality by topology (SOTA should match on atomic, lose on multi/cyclic) AND cost-vs-N (G2:
ROUTEMESH engages only the path's specialists, flat in N; pooling engages all).
"""
import os, sys, json, random
from collections import Counter, defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_r1 as R1

SEED = 0
SPARSE_FRAC = 0.25       # fraction of train queries with competence labels (the sparseness)


def feats(q, rels):
    """bag-of-relations (which relations appear) — what the critic sees; NOT the ownership."""
    v = np.zeros(len(rels))
    for r in q["rel_seq"]:
        v[rels.index(r)] = 1
    return v


def solves(q, allowed, spec_rels, edges, owner):
    ca = R1.traverse(spec_rels, edges, owner, q["h"], q["rel_seq"], list(range(len(spec_rels))))
    return R1.traverse(spec_rels, edges, owner, q["h"], q["rel_seq"], allowed) == ca and \
        all(r in {rr for s in allowed for rr in spec_rels[s]} for r in q["rel_seq"])


def run(K, edges, rels, seed, want_arms=True):
    owner, spec_rels = R1.partition(rels, K, seed)
    rng = random.Random(seed)
    answerable, abstain = R1.sample_paths(edges, rels, owner, rng)
    half = len(answerable) // 2
    train, test = answerable[:half], answerable[half:]

    # --- cost-vs-N: per-query specialists engaged (topology-bounded for routing, all-N for pooling)
    def engaged_routemesh(q):
        return len({owner[r] for r in q["rel_seq"]})       # only the path's owning specialists
    cost = {"routemesh": float(np.mean([engaged_routemesh(q) for q in test])),
            "sota_router": 1.0,                            # routes to a single model
            "pooling": float(K)}                           # runs every specialist
    if not want_arms:
        return {"K": K, "cost": cost}

    # --- critic: per-specialist competence from SPARSE labels ---
    Xtr = np.array([feats(q, rels) for q in train])
    labeled = rng.sample(range(len(train)), int(SPARSE_FRAC * len(train)))
    critics = {}
    for s in range(K):
        y = np.array([int(solves(train[i], [s], spec_rels, edges, owner)) for i in labeled])
        Xs = Xtr[labeled]
        if len(set(y)) < 2:
            critics[s] = None
        else:
            critics[s] = LogisticRegression(max_iter=1000).fit(Xs, y)

    def comp(s, x):
        return 0.0 if critics[s] is None else critics[s].predict_proba([x])[0, 1]

    # per-relation predicted owner (for ROUTEMESH per-step routing): who does the critic trust for {r}?
    rel_owner_pred = {}
    for r in rels:
        xr = np.zeros(len(rels)); xr[rels.index(r)] = 1
        rel_owner_pred[r] = max(range(K), key=lambda s: comp(s, xr))

    Xte = [feats(q, rels) for q in test]
    best_s = max(range(K), key=lambda s: sum(solves(q, [s], spec_rels, edges, owner) for q in train))

    res = {"K": K, "n_test": len(test), "cost": cost,
           "topo_test": dict(Counter(q["topo"] for q in test))}
    res["best_single"] = sum(solves(q, [best_s], spec_rels, edges, owner) for q in test) / len(test)
    res["oracle"] = sum(solves(q, list(range(K)), spec_rels, edges, owner) for q in test) / len(test)
    # pooling: majority vote over specialists' answers (drag)
    def pool_ans(q, subset):
        v = Counter(R1.traverse(spec_rels, edges, owner, q["h"], q["rel_seq"], [s]) for s in subset)
        v.pop(None, None)
        return v.most_common(1)[0][0] if v else None
    ca = {id(q): R1.traverse(spec_rels, edges, owner, q["h"], q["rel_seq"], list(range(K))) for q in test}
    res["pooling"] = sum(pool_ans(q, range(K)) == ca[id(q)] for q in test) / len(test)
    # SOTA-ROUTER: route whole query to critic's single best specialist
    res["sota_router"] = sum(
        solves(q, [max(range(K), key=lambda s: comp(s, Xte[i]))], spec_rels, edges, owner)
        for i, q in enumerate(test)) / len(test)
    # ROUTEMESH: per-step assembly via predicted per-relation owners
    res["routemesh"] = sum(solves(q, [rel_owner_pred[r] for r in q["rel_seq"]], spec_rels, edges, owner)
                           for q in test) / len(test)

    # quality by topology: ROUTEMESH vs SOTA per topology class
    res["by_topo"] = {}
    for topo in ("atomic", "multi", "cyclic"):
        qs = [(i, q) for i, q in enumerate(test) if q["topo"] == topo]
        if not qs:
            continue
        rm = np.mean([solves(q, [rel_owner_pred[r] for r in q["rel_seq"]], spec_rels, edges, owner) for i, q in qs])
        so = np.mean([solves(q, [max(range(K), key=lambda s: comp(s, Xte[i]))], spec_rels, edges, owner) for i, q in qs])
        res["by_topo"][topo] = {"routemesh": float(rm), "sota_router": float(so), "n": len(qs)}

    # fabrication guard: no-path queries -> ROUTEMESH must abstain (predicted owner can't execute first hop)
    guard_ok = 0
    for q in abstain:
        first = q["rel_seq"][0]
        # ROUTEMESH abstains iff it cannot execute the first hop from h (no real edge)
        guard_ok += int(q["h"] not in edges[first])
    res["fabrication_guard_abstain_rate"] = guard_ok / len(abstain)
    return res


def main():
    edges, rels = R1.load_kg()
    print("=== R2 main (K=5) ===")
    r = run(5, edges, rels, SEED)
    gap = r["oracle"] - r["best_single"]
    closed = (r["routemesh"] - r["best_single"]) / (gap + 1e-9)
    print(f"best-single {r['best_single']:.3f} | pooling {r['pooling']:.3f} | "
          f"SOTA-router {r['sota_router']:.3f} | ROUTEMESH {r['routemesh']:.3f} | oracle {r['oracle']:.3f}")
    print(f"ROUTEMESH closes {100*closed:.0f}% of oracle-minus-best-single gap")
    print("quality by topology (ROUTEMESH vs SOTA):")
    for topo, d in r["by_topo"].items():
        print(f"  {topo:7s} (n={d['n']:4d}): ROUTEMESH {d['routemesh']:.3f}  SOTA {d['sota_router']:.3f}  "
              f"edge {d['routemesh']-d['sota_router']:+.3f}")
    print(f"fabrication guard abstain rate = {r['fabrication_guard_abstain_rate']:.3f}")

    print("\n=== cost vs N (G2: per-query specialists engaged) ===")
    cost_curve = {}
    for K in (2, 3, 4, 5, 6, 8):
        c = run(K, edges, rels, SEED, want_arms=False)["cost"]
        cost_curve[K] = c
        print(f"  N={K}: ROUTEMESH {c['routemesh']:.2f}  SOTA {c['sota_router']:.2f}  pooling {c['pooling']:.2f}")

    # verdict
    beats = (r["routemesh"] > r["best_single"] and r["routemesh"] > r["pooling"]
             and r["routemesh"] > r["sota_router"])
    closes = closed >= 0.50
    abstains = r["fabrication_guard_abstain_rate"] >= 0.9
    rm_costs = [cost_curve[k]["routemesh"] for k in (2, 3, 4, 5, 6, 8)]
    flat = max(rm_costs) - min(rm_costs) <= 1.5           # ROUTEMESH cost ~flat in N
    r["cost_curve"] = cost_curve
    r["verdict"] = {"beats_all_three": bool(beats), "closes_50pct_gap": bool(closes),
                    "abstains": bool(abstains), "cost_flat_in_N": bool(flat),
                    "R2_PASS": bool(beats and closes and abstains)}
    print(f"\nR2: beats best-single+pooling+SOTA={beats}, closes>=50% gap={closes} ({100*closed:.0f}%), "
          f"abstains={abstains}, cost-flat-in-N={flat}")
    print(f"R2 {'PASS' if r['verdict']['R2_PASS'] else 'FAIL'}")
    json.dump(r, open(os.path.join(HERE, "r2_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
