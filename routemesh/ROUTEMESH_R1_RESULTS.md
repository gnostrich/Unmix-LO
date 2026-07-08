# ROUTEMESH R1 — RESULTS: **PASS** (routable structure exists), with an honest construction caveat

Run 2026-07-08 per R1_DESIGN.md / PREREG.md (frozen before the run). WN18RR real KG (93,003
WordNet triples, 11 relations partitioned across 5 specialists), 3,000 answerable + 1,000 abstain
typed-path queries. Full numbers in r1_results.json.

## Result

| arm | accuracy | note |
|---|---|---|
| best-single-overall (spec3, owns _hypernym) | 0.254 | one specialist's own-relation coverage |
| **difficulty-only router** | **0.254** | route by path-length only — identical to best-single |
| **ORACLE (topology-free: 1 / assemble-K / cycles / abstain)** | **1.000** | assembles the reachable union |

- oracle − best-single = **+0.746** (≥ 0.10 ✓)
- oracle − difficulty-only = **+0.746** (≥ 0.10 ✓)
- **R1 PASS.**

Route breakdown (test): 574 atomic / 486 multi-hop / 437 cyclic-iterated / 3 same-specialist-multi.
Drag control: pooling 0.345 → **0.585** when restricted to the competent subset (+0.240) — ignorance-
drag is the mechanism, and selection fixes it. Atomic→single-hop solve rate 1.000 (atomic queries
route single-path, not spuriously multi). Fabrication guard: 1.000 of injected no-path queries are
correctly unanswerable (the ground-truth oracle only traverses real edges — the G1 boundary holds).

## What the difficulty-only = best-single identity means (the crux, passed cleanly)
The prereg's real contest was oracle vs difficulty-only, not oracle vs best-single. Here they are
**identical** (0.254) — a difficulty-only router, knowing only how many hops a query needs, collapses
to best-single because it cannot identify which specific specialists own a multi-hop path's relations.
So the oracle's entire +0.746 gain is **model-specific assembly**, with **zero** difficulty artifact.
This is the strongest form of an R1 pass: the routable structure is real and not a difficulty marginal.

## The honest caveat (pre-committed, not post-hoc)
This is largely a pass **by construction**: relation ownership is disjoint by design, so assembling
the reachable union trivially beats any single specialist. R1 therefore validates that the topology-
free/cyclic routing *mechanism* is real and well-behaved on real KG facts (correct route breakdown,
drag control, fabrication guard) — it does **not** by itself show that less-constructed federations
exhibit exploitable disjoint competence. The flat real-task tests earlier this program (BIOMESH,
synergy) showed frozen encoders do **not** have disjoint competence beyond marginals. So R1 delineates
*when* routing helps — given genuinely disjoint, compositional competence — rather than resurrecting
the dead ">" thesis (this is not ">": every sub-answer is one member's; the ceiling is "=", the
reachable union, and the oracle sits exactly at it).

Treated as a claim to falsify (per the DAVIS cold-target discipline): checked and clean — not a
difficulty artifact (difficulty-only = best-single), not a leak (the oracle's competence knowledge is
the *definition* of the opportunity gate; whether it is *realizable* from sparse data is R2's job).

## Consequence
R1 passes → proceed to R2: can a light learned critic realize the oracle from SPARSE competence data,
and does the resulting router beat not just pooling but a **SOTA learned single-hop router** — with the
quality edge coming specifically from multi-hop/cyclic union-retrieval, and the cost edge from flat-in-N
scaling (G2)? R2 is the genuinely untested piece.
