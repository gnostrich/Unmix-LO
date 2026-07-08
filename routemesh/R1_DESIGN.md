# ROUTEMESH R1 — instantiation design (frozen before running; implements the topology-free upgrade)

Per BRIEF.md + PREREG.md + the topology-free/cyclic strengthening. This file fixes the concrete
real-data realization and the difficulty-only baseline BEFORE any R1 run. Frozen thresholds are the
PREREG ones (oracle beats BOTH best-single-overall AND difficulty-only by ≥ 0.10).

## Why a knowledge-graph reachability task
The topology-free oracle (free in cardinality AND topology, cycles allowed, measuring the *complete
reachable union*) needs compositional structure — a flat classification task has no multi-hop or
cyclic reachability to measure. A real KG provides atomic / multi-hop / iterated(cyclic) / abstain
routes with verifiable answers, and the G1 boundary maps exactly: every traversal step retrieves a
**real edge held by some member** (never generates); abstain on anything outside the reachable closure.

## Data (real)
WN18RR (real WordNet facts): 93,003 triples, 11 relations, from the standard KG-completion release.
`_hypernym` (34,796 edges) and `_derivationally_related_form` (29,715) dominate; `_hypernym` is
transitive → iterated traversal is genuine.

## Specialists (K=5, disjoint relation ownership)
The 11 relations are partitioned into 5 disjoint groups (fixed, seed-0), so each specialist "owns"
a subset of relations and can traverse only its own edges. A specialist's per-query competence is
GROUND-TRUTH (it can answer iff the query's required edges are all in its relation set) — this makes
R1 an opportunity/ceiling test, exactly as the prereg specifies (the oracle knows true competence).

## Queries (typed-path entity prediction — the answer depends on relation TYPE, not just connectivity)
A query is (start entity h, relation-path [r1,…,rk]) with answer = the actual endpoint reached by
following that exact typed path in the KG (verifiable). Sampled from real paths. Topology classes:
- **atomic** (k=1): one relation, one specialist.
- **multi-hop** (k≥2, relations span ≥2 specialists): needs assembly across specialists.
- **cyclic/iterated**: path that iterates a transitive relation and/or revisits specialists in a
  loop to reach an answer only reachable by feedback (e.g., mixed-relation closure requiring
  A→B→A… to a fixpoint). Each step retrieves a real edge.
- **abstain**: no valid path exists in the federation (injected unreachable queries) → must abstain.

Note (analytical, reported): typed-path prediction is used precisely because pure *reachability*
(does any path exist) is relation-agnostic — untyped connectivity would match the oracle, so
reachability could not distinguish model-specific competence from difficulty. Typed-path answers
require knowing WHICH relations/specialists to compose, which is where model-specific competence lives.

## Arms
1. **best-single-overall** — the single specialist owning the most query-answering relations; solves
   a query iff it owns every relation on the path.
2. **difficulty-only router** (the crux baseline; frozen definition): routes using ONLY a
   model-agnostic difficulty signal = the query's path length k (a proxy for "how hard"), never which
   relations/specialists are needed. Strongest fair form: for each difficulty stratum k, route all
   queries to the FIXED policy best for that k averaged over queries (single best specialist per k, or
   abstain), chosen on a train split, applied to test. It can exploit "k-hop queries are harder" but
   NOT "this k-hop query needs relations owned by specialists {A,C}".
3. **ORACLE ROUTER** (topology-free) — free in cardinality (1 / assemble-K / abstain) and topology
   (multi-hop + iterated/cyclic); solves iff the path's relations are covered by the federation and
   the path is valid; abstains on unreachable queries. Measures the complete reachable union.

## Metric
Per-query correctness = produces the exact answer entity (abstain = 0 credit on answerable queries;
abstain = correct on injected-unreachable queries). Report overall accuracy + the route breakdown
(single-hop / multi-hop / cyclic / abstain fractions of the oracle's routes).

## Pre-committed verdict (frozen)
- **R1 PASS** iff ORACLE ≥ best-single-overall + 0.10 AND ORACLE ≥ difficulty-only + 0.10.
- **R1 FAIL** (oracle ≈ difficulty-only) → competence is a shared difficulty marginal (P1's ghost at
  the routing level) → STOP, do not build R2.

## Mandatory controls (cyclic-aware)
- FABRICATION GUARD: injected no-path queries must ABSTAIN; a confident answer (esp. via a spurious
  cycle) = FAIL. (The ground-truth oracle only traverses real edges, so it passes by construction;
  the guard is load-bearing for R2's learned router.)
- ATOMIC-VS-TOPOLOGY: atomic queries must route single-hop; multi-hop/cyclic routes firing on atomic
  queries and "winning" = overfitting → disqualify.
- DRAG CONTROL: pooling (use all specialists, no selection) must improve when restricted to the
  competent subset — proving ignorance-drag is the mechanism routing fixes.
- NO ">" : the oracle never exceeds the reachable union (what members hold); if any arm appears to,
  investigate for leak/fabrication.

## Honesty note (pre-committed)
Because relation ownership is partitioned by design, a favorable R1 largely re-validates the
*mechanism* on real KG facts; it is NOT by itself evidence that less-constructed federations exhibit
exploitable disjoint competence (BIOMESH/synergy showed flat real encoders do NOT, beyond marginals).
A pass earns the right to test R2 (can a light critic realize the oracle from SPARSE competence data —
the genuinely untested piece); it is treated as a claim to falsify (leak / difficulty artifact), as
the DAVIS cold-target episode was.
