# ROUTEMESH — PRE-REGISTRATION (commit BEFORE any run code)
#
# THESIS SHIFT (why this is NOT a repeat of the 4 dead composition tests):
# The dead thesis was ">": virtual model EXCEEDS best constituent by extracting JOINT capability.
# Killed 4 ways (pooling, settling, blind index, task-aware aggregation); P1 showed joint-beyond-single
# complementarity is thin. THIS test drops ">" entirely. New target:
#   Reach the UNION of what constituents individually know, per query, WITHOUT ignorance-drag.
# The failures were caused by IGNORANCE-DRAG (pooling let ignorant models dilute the knowledgeable one),
# not by absent knowledge. So the obstruction is ROUTING, not synergy. Target = per-query correct routing:
#   - atomic query, one model knows -> route to it (single path)
#   - decomposable query, pieces across models -> assemble disjoint expert pieces (multi-path)
#   - no model knows -> abstain (fabrication guard)
# Cardinality (single vs multi vs abstain) is EMERGENT from one objective (correctness), never forced.
# Ceiling is per-query best-achievable-from-constituents; we NEVER claim > best single on a query.

## The one objective
A correctness loss oracle. Baur x MZ descends routing paths into an MZ memory kernel:
  streaming term = single/direct path ; memory term = multi-hop/through-other-models path.
Minimizing correctness-loss => descent uses multi-path WHERE it lowers loss (decomposable queries),
single-path where multi only adds drag (atomic), abstain where no route is low-loss (absent knowledge).
The MZ streaming/memory split IS the atomic/composite dial; we read cardinality OUT of it, never set it.

## GATE R1 — OPPORTUNITY (test FIRST; if fails, STOP — do not build the critic)
Question: is there routable+assemblable competence structure at all, beyond the difficulty marginal?
Build an ORACLE ROUTER that knows ground-truth per-query competence and is FREE to choose cardinality
(route to 1 model, assemble K models' disjoint-correct pieces, or abstain). Compare to best-single-OVERALL.
  - DISTINGUISH from the difficulty marginal: also report a "difficulty-only" baseline (route by query
    difficulty, ignoring which model). If oracle-router ~= difficulty-only, competence is not model-specific
    -> nothing to route -> FAIL.
PRE-COMMITTED: R1 PASS iff oracle-router beats BOTH best-single-overall AND difficulty-only by >= 0.10
  (real per-query MODEL-SPECIFIC competence exists). Report the split of oracle routes: what fraction were
  single-path vs multi-path vs abstain (this shows whether decomposable competence exists, and it falls out).
  R1 FAIL => competence is a shared difficulty marginal (P1's ghost at the routing level); STOP, report negative.

## GATE R2 — REALIZABILITY (only if R1 passes)
Can a LIGHT learned critic, fed by SPARSE constituent competence data, approach the oracle router?
- Loss oracle from HF I/O: for each frozen model, use its linked/benchmark data as sparse
  (input -> model-correct?) labels. This is the competence signal, and it is SPARSE by design.
- SYNTHETIC-GRADIENT / learned critic for sparseness: the critic predicts per-model competence on
  queries OUTSIDE each model's benchmark by interpolating from the sparse labels (this is the untried
  piece — a competence-interpolator, learning "who to trust" not the task, far smaller than task labels).
- Baur x MZ descends routing memory using the critic's competence signal as the loss oracle.
Arms (equal capacity/budget):
  1. best-single-overall (floor)
  2. POOLING (the incumbent that suffers ignorance-drag)
  3. ROUTEMESH: critic-fed Baur x MZ, emergent cardinality
  4. ORACLE ROUTER (ceiling, from R1)
PRE-COMMITTED: R2 PASS iff ROUTEMESH beats BOTH best-single-overall AND pooling, AND approaches oracle
  (closes >= 50% of the oracle-minus-best-single gap), AND correctly ABSTAINS on no-knowledge queries.

## Mandatory controls (guards against the failure modes re-entering)
- FABRICATION GUARD: inject queries NO constituent can answer; ROUTEMESH must ABSTAIN, not emit a
  confident multi-path answer. A confident-wrong multi-path route is a FAIL (this is the G3/G1 guard at routing level).
- ATOMIC-VS-COMPOSITE READOUT: verify emergent cardinality is sensible — atomic queries (single-expertise)
  should get single-path routes; decomposable queries should get multi-path. If multi-path fires on atomic
  queries and "wins", it's overfitting/fabricating -> disqualify.
- DRAG CONTROL: confirm pooling loses specifically because it includes ignorant models (show pooling
  improves if restricted to the competent subset) -> proves the mechanism is ignorance-drag, and routing fixes it.
- NO ">" CLAIM: ROUTEMESH must NOT exceed per-query best-single-achievable. If it appears to, that's leakage
  or fabrication -> investigate, do not celebrate. The honest ceiling is union-without-drag, = not >.

## Setup
- 4-8 frozen specialists with genuinely DISJOINT competence regions (this is the precondition for routing
  to help — models that each own a chunk of query-space). Prefer specialists on different sub-domains where
  each is clearly best on its own slice (so per-query model-specific competence plausibly exists).
  NOTE: disjoint-competence is DIFFERENT from synergy/P1 — a model can be sole-expert on a slice with zero
  joint-beyond-single info. P1 failing does NOT imply R1 fails. That divergence is the whole bet.
- Task with verifiable per-query labels so competence is measurable.

## Discipline
Pre-register, commit artifacts before runs, honest RED = success, stop if R1 fails (don't build R2 on a null).
Keep every invalidated run. R1 is the real risk; be as adversarial to a favorable R1 as the DAVIS cold-target
episode was (a surprising R1 pass gets MORE scrutiny — check it's not a difficulty marginal or a label leak).
