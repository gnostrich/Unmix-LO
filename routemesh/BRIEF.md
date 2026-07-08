# ROUTEMESH — Claude Code brief

Read PREREG.md (frozen thresholds) and run probe.py first — it validates the R1 MECHANISM in sandbox:
under disjoint competence, free-cardinality oracle routing beats best-single (+0.405, easy) AND
difficulty-only (+0.102, the REAL contest, barely clears). Cardinality fell out (69% single / 31% multi,
unforced). The probe proves the mechanism is coherent; it does NOT prove real specialists have disjoint
competence -- that is exactly what R1 tests on real data.

## Why this is NOT the 4 dead composition tests
Dead thesis was ">": exceed best-single via JOINT capability (killed 4x; P1 = joint-beyond-single is thin).
THIS drops ">". Target = per-query UNION-without-ignorance-drag. Failures were caused by ignorance-DRAG
(pooling diluted the knowledgeable model with ignorant ones), not absent knowledge. Fix = ROUTE (select the
competent, suppress the ignorant); cardinality (single/multi/abstain) EMERGES from one correctness objective.
Ceiling is "=" (per-query best-achievable-from-constituents), never ">". Disjoint-competence != synergy:
P1 failing does NOT imply R1 fails -- a model can solely own a query-slice with zero joint info. That's the bet.

## R1 (test FIRST; the REAL baseline is difficulty-only, NOT best-single)
Build the free-cardinality ORACLE ROUTER (route-to-1 / assemble-K-disjoint-experts / abstain) on real
frozen specialists with per-query labels. R1 PASS iff oracle beats BOTH best-single-overall AND a
difficulty-only router by >= 0.10. The best-single margin will be easy; the DIFFICULTY-ONLY margin is the
crux (does MODEL-SPECIFIC competence exist beyond "some queries are just hard"). Report emergent cardinality
split. Treat a favorable R1 as a CLAIM TO FALSIFY (as the DAVIS cold-target episode was): check it's not a
label leak or a difficulty artifact. R1 FAIL (oracle ~= difficulty-only) => competence is a shared difficulty
marginal => STOP, negative, do not build R2.

## R2 (only if R1 passes): can a light critic realize the oracle from SPARSE data?
Loss oracle from HF I/O: each frozen model's linked/benchmark data as sparse (input->correct?) competence
labels. Learned critic (synthetic-gradient-style) INTERPOLATES per-model competence to queries outside each
model's benchmark -- learning "who to trust", not the task (far smaller signal; this is the sparseness fix).
Baur x MZ descends routing memory on the critic's signal; streaming=single-path, memory=multi-path, so
cardinality is emergent. Arms: best-single | pooling | ROUTEMESH | oracle(ceiling). R2 PASS iff ROUTEMESH
beats best-single AND pooling, closes >=50% of the oracle-minus-best-single gap, AND abstains on no-knowledge queries.

## Mandatory controls
- FABRICATION GUARD: no-constituent-can-answer queries -> must ABSTAIN, not emit confident multi-path. Confident-wrong = FAIL.
- ATOMIC-VS-COMPOSITE: atomic queries should route single-path, decomposable multi-path. Multi-path winning on
  atomic queries = overfitting -> disqualify.
- DRAG CONTROL: show pooling improves when restricted to the competent subset -> proves ignorance-drag is the mechanism.
- NO ">" : ROUTEMESH must not exceed per-query best-achievable. If it seems to -> leak/fabrication, investigate not celebrate.

## Setup & discipline
4-8 frozen specialists with plausibly DISJOINT competence regions (each best on its own sub-domain slice),
task with verifiable per-query labels. Pre-register, commit artifacts before runs, honest RED = success,
stop if R1 fails. Surviving-so-far results (G2 scale-free cost, G3 compression) are unaffected either way.
