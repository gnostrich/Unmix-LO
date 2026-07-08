# BIOMESH — build brief (biomedical specialist composition; cash the validated VIRTUALMESH win)

## Context (read first)
Prior project VIRTUALMESH ran 4 gates. Surviving, validated claims ONLY:
- **G2 [PASS, scoped]**: the routing/memory kernel's closure complexity is INDEPENDENT of federation
  size (kernel rank flat across N=4->10, linear regime). => "cost doesn't grow as you add models."
- **G3 [PASS, amended]**: transitive pathways distill into direct edges that CACHE the composite at
  lower inference cost, bounded EXACTLY by the chain ceiling. => COMPRESSION of existing reachable
  knowledge, NEVER new capability.
Refuted, do NOT use or claim:
- **G1 [FAIL]**: settling/fact-passing amplifies hallucination without calibrated ignorance. NO settling
  layer. NO "models reason together." NO capability-generation claims.

## What BIOMESH claims (and ONLY this)
A scale-invariant compression-and-routing layer over FROZEN biomedical specialist encoders that
composes their REPRESENTATIONS at cost independent of the number of specialists — beating published
baselines (agentic orchestration; static embedding pooling) on the COST-vs-SCALE axis at equal accuracy.
NOT "generates new capability." NOT "reasons." The win is: same accuracy, flat cost as N grows.

## The cluster (frame-connected biomedical specialists on HF)
Complementary knowledge, overlapping representational structure (chemistry/sequence = the real frame):
- ESM-2 (proteins), ChemBERTa (small molecules / SMILES), scGPT (single-cell), DNABERT (genomic),
  + BioBERT/PubMedBERT (biomedical text) as connective tissue.
Pick 4-6. NOTE: these do NOT share a base => heterogeneous frame (this folds in the G3-heterogeneous
test for free — alignment is the real cost, measure it).

## Non-negotiable discipline
- Pre-commit thresholds BEFORE running. Honest RED is success.
- Claim only G2/G3-validated properties. Settling stays excluded. No capability-generation language.
- Watch degenerate wins: a cost-advantage at DEGRADED accuracy is not a win; equal-accuracy is the bar.
- Every baseline must be the FAIR one (same models, same task) — see baselines/.

## Structure (fan out agents)
- **gate0/** : GATE ZERO — are the biomedical composite queries GENUINELY split-knowledge? (pacing gate)
- **experiment/** : the cost-vs-scale demonstration (kernel-routing composition vs baselines)
- **baselines/** : orchestration (Het-MedAgent-style) + static pooling (BioVERSE-style) + best-single
See each folder's README.

---

## OUTCOME (2026-07-08 — GATE ZERO run)

**GATE ZERO: FAIL (as pre-registered) → project stopped before the cost-vs-scale experiment.**

On DAVIS drug-target interaction (30,056 protein×drug pairs, ESM-2 + ChemBERTa specialists):

| condition | measured | threshold | result |
|---|---|---|---|
| split-knowledge fraction (per-instance) | 0.005 | ≥ 0.30 | FAIL |
| union AUPRC / best-single AUPRC | 1.37× | ≥ 1.10 | PASS |

Composition improves aggregate discrimination (union 1.37× the best single) but does **not**
create a class of only-jointly-solvable queries: single-specialist errors are anti-correlated
(−0.18), both are wrong only 5% of the time, and the union rescues just 10% of those. The
strict split-knowledge premise BIOMESH exists to serve is absent on this task — DTI single
baselines are inflated by marginal promiscuity structure. Per the frozen decision rule, the
cost-vs-scale experiment (`experiment/`) does **not** run. Full analysis in
`gate0/GATE0_RESULTS.md`; two candidate follow-ups (marginal-free task; re-scoped
aggregate-accuracy claim) are noted there, each as a NEW pre-registration.

The pacing gate did its job: it stopped a beautiful cheap-composition layer from being built for
queries that don't need composing.

## OUTCOME (2026-07-08 — GATE ZERO cold-split, confound-controlled re-test)

**GATE ZERO cold-split: FAIL decisively → the negative is now confound-controlled.**

Entity-disjoint splits (no drug/target shared between train and test) remove the marginal-
promiscuity leakage that inflated the in-distribution single baselines. Result: composition
doesn't just fail to be *necessary* — it *hurts*. Union AUPRC falls **below** best-single in
every mode (cold-drug 0.88×, cold-target 0.63×, cold-pair 0.69×), and split-knowledge fraction
stays ~0. Union AUROC holds at 0.65–0.76 (no encoder collapse — the guard doesn't trip), so the
verdict is clean. The in-distribution 1.37× "composition helps" gain **inverts** under the fair
split, showing that gain was itself marginal memorization. This also undercuts the option-3
re-scope: there is no aggregate-accuracy gain to deliver cheaply. Full analysis in
`gate0cold/GATE0COLD_RESULTS.md`. **STOP is now decisive.**
