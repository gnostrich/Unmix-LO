# SYNERGISTIC AGGREGATOR — PRE-REGISTRATION v2 (real-data only; commit BEFORE any run code)
#
# WHY v2: sandbox probing (probe.py) could NOT construct a valid SYNTHETIC synergy task twice over —
# either the oracle was unlearnable (chance) or a strong naive baseline already captured it, and the
# blind/high-feature arms won by OVERFITTING, not synergy. That construction difficulty is itself a
# signal that learnable-synergy-naive-can't-capture is a THIN regime. So we do NOT test on synthetic
# planted synergy. We test on a REAL biomedical task where synergy, if present, is planted by nature,
# with entity-disjoint (cold) splits so overfitting cannot masquerade as synergy.

## The claim (both halves required to PASS)
(A) SYNERGY: a TASK-AWARE redundancy-penalized aggregator over frozen specialists beats best-single
    (>= 1.10x) and approaches the joint-oracle, on a real composite task WITH confound-controlled splits.
(B) SCALE-FREE COST: its per-query cost (FLOPs/calls/latency + routing-memory size) grows sub-linearly
    in N and strictly flatter than pooling/orchestration.
Novel part is (B) ON a working (A) — task-aware synergy itself is occupied (BioVERSE/MoE). If (A) holds
but (B) doesn't -> occupied competitor, no unique claim. If neither -> negative. Report honestly.

## Preconditions (gate BEFORE claiming anything — these killed prior overclaims)
P1. COMPLEMENTARITY EXISTS: on the real task, best-single << joint-oracle (gap >= 0.15) under the
    COLD/entity-disjoint split. If not -> STOP (no synergy to deliver; the BIOMESH-DTI lesson). Do not
    proceed to build the aggregator on a task with no complementarity.
P2. COLD SPLIT MANDATORY: entity-disjoint train/test (cold-drug/cold-target or domain-disjoint), the
    established-correct methodology. In-distribution numbers are reported ONLY alongside cold numbers,
    never as the verdict (in-dist inflated BIOMESH's 1.37x which inverted to <1 under cold split).

## Setup
- 4-8 FROZEN specialists, frame-connectable, biomedical/molecular (ESM-2, ChemBERTa, scGPT, DNABERT,
  PubMedBERT). Real composite task with verifiable labels needing multiple specialists (e.g. a curated
  multi-modal biomedical prediction where protein + molecule + text each contribute).
- Task signal seeded from the models' own linked datasets / cited benchmarks (HF metadata) -> task-aware.

## Aggregator (the descent that CAN prove synergy)
Minimize combined-readout TASK loss + lambda * redundancy penalty (suppress features predictable from any
single specialist alone; reward cross-specialist info with positive conditional-MI w.r.t. target given each
single model). Feature SELECTION by task-relevance is capped (top-k) and the cap is chosen by CV on the
COLD-TRAIN split only, NEVER on test -> closes the overfitting hole the probe exposed. Routing memory
low-rank/MZ so per-query cost ~ kernel rank, not N.

## Arms (equal readout capacity + equal tuning + SAME feature-count budget everywhere -> no overfit edge)
1. best-single  2. static pooling (BioVERSE-style)  3. orchestration-style [optional]
4. TASK-AWARE AGGREGATOR (ours).  Sweep N=2..6.

## Anti-fooling controls (mandatory — the probe showed these are essential)
- EQUAL-FEATURE-COUNT: every arm gets the same total feature budget, so a win cannot be raw capacity/overfit.
- NO-COMPLEMENTARITY control: a task/subset answerable by one specialist -> aggregator must NOT beat
  best-single (else overfitting-as-synergy -> disqualify).
- CAPACITY control: equal-capacity readout on best-single features must NOT close the gap (else it's
  capacity not cross-model info).
- COLD-SPLIT: overfitting cannot masquerade as synergy because test entities are unseen.

## Verdict (frozen; STOP either way)
PASS iff P1 holds AND (A) aggregator >= 1.10x best-single approaching oracle on COLD split AND (B) cost
sub-linear in N and flatter than baselines, AND all controls pass. Else report which half failed:
  - P1 fails -> "no complementarity in this task" (BIOMESH-class negative).
  - A fails  -> "task-aware aggregation doesn't capture the complementarity cheaply" (negative).
  - B fails  -> "synergy real but not scale-free -> occupied competitor, no unique claim."
Read BioVERSE/Het-MedAgent directly; confirm neither already has N-independent-cost synergy before claiming it.

## Discipline
Pre-register, commit design artifacts before their runs, keep invalidated runs on record, honest RED = success.
Note in the writeup: two synthetic-construction attempts FAILED to build learnable-naive-hard synergy, which
is corroborating (not conclusive) evidence the regime is thin; the real-data test is the fair arbiter.
