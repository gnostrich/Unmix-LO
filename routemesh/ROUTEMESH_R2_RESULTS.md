# ROUTEMESH R2 — RESULTS: **PASS** — a light critic realizes the oracle and beats SOTA routing (structurally, on assembly)

Run 2026-07-08 per PREREG.md + the SOTA-arm addendum (frozen before the run). WN18RR, 5 specialists,
sparse competence labels (25% of train). Five arms, equal features/budget. Full numbers in
r2_results.json.

## Result (K=5)

| arm | accuracy | what it is |
|---|---|---|
| best-single-overall | 0.260 | floor |
| pooling (majority vote, all specialists) | 0.357 | weak incumbent — suffers ignorance-drag |
| **SOTA-ROUTER** (learned query→single model) | 0.577 | strong incumbent — single-hop routing |
| **ROUTEMESH** (critic-fed topology-free assembly) | **0.993** | ours |
| oracle (ceiling = reachable union) | 1.000 | |

ROUTEMESH closes **99%** of the oracle-minus-best-single gap. R2 PASS (beats best-single AND pooling
AND SOTA-ROUTER; closes ≥50% of the gap; abstains correctly). A **light** critic (per-specialist
logistic regression on a bag-of-relations feature, trained on 25% competence labels) realizes the
oracle — the sparse-data realizability the prereg flagged as the untested piece.

## The distinctive claim, measured (quality edge over SOTA is STRUCTURAL, not marginal)
Edge = ROUTEMESH − SOTA-ROUTER, per topology:

| topology | n | ROUTEMESH | SOTA-ROUTER | edge |
|---|---|---|---|---|
| atomic | 574 | 1.000 | 1.000 | **+0.000** |
| multi-hop | 474 | 0.983 | 0.000 | **+0.983** |
| cyclic-iterated | 450 | 0.996 | 0.644 | **+0.351** |

SOTA single-hop routing **matches ROUTEMESH exactly on atomic queries** and **collapses on multi-hop**
(it routes to one model, which cannot execute a path spanning multiple specialists). The entire quality
advantage is **multi-hop/cyclic union-retrieval** — capturing reachable knowledge a single-hop router
structurally cannot. This is the honest form of "superior to SOTA": not a marginal quality gap on the
same queries, but a structural win on the queries that require assembly.

## Cost vs N (G2 — the scale-free edge)
Per-query specialists engaged, as the federation grows:

| N | ROUTEMESH | SOTA-ROUTER | pooling |
|---|---|---|---|
| 2 | 1.19 | 1.00 | 2.00 |
| 4 | 1.42 | 1.00 | 4.00 |
| 6 | 1.43 | 1.00 | 6.00 |
| 8 | 1.45 | 1.00 | 8.00 |

ROUTEMESH's per-query cost is **flat in N** (~1.4, bounded by path topology, not federation size) —
the G2 property, on a real routing task. Pooling scales O(N) and still loses on drag. So ROUTEMESH
delivers assembly-quality at single-hop-order cost.

Fabrication guard: 1.000 of injected no-path queries are correctly unanswerable (ROUTEMESH abstains
rather than fabricate a cyclic answer — the G1 boundary).

## Honest scope (pre-committed, carried from R1)
- This lives in the **constructed disjoint-compositional regime** (relation ownership partitioned by
  design). R2 shows that *given* such structure, (a) a light critic realizes the oracle from sparse
  data, (b) topology-free routing beats SOTA single-hop routing specifically via multi-hop/cyclic
  assembly, (c) at flat cost in N. It does **not** show real federations have this structure — the
  flat real-task tests (BIOMESH, synergy) showed frozen encoders do **not**, beyond marginals.
- Per the honest guard: in this construction SOTA-ROUTER (0.577) is *stronger* than the ~9%-over-best
  real-world routers, because atomic routing here is clean; ROUTEMESH's edge over it is nonetheless
  **structural** (assembly on multi-hop/cyclic), not an inflated marginal gap. On an atomic-only task
  the quality edge would vanish and only the cost-scaling advantage would remain — reported, not hidden.
- **No ">"**: the ceiling is the reachable union ("="); the oracle sits at it, ROUTEMESH approaches it
  (0.993), neither exceeds it. This is union-without-drag, not new capability — consistent with the
  surviving G2/G3 (infrastructure), orthogonal to the dead ">" composition thesis.

## What ROUTEMESH is
The first conditional **positive** of the program — but a routing/retrieval result, not a capability
one: **when** a federation holds genuinely disjoint, compositional competence and queries require
assembly, a critic-fed topology-free router captures the reachable union that best-single and pooling
miss and that SOTA single-hop routing structurally cannot assemble — at cost flat in N. It is
infrastructure (cheaper, more complete routing), squarely in the G2/G3 lane, and it does not revive
the "> best-single via joint capability" thesis that four gates killed.
