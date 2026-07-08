# SYNERGISTIC AGGREGATOR — RESULTS: precondition P1 **FAILS** on both real tasks → STOP

Run 2026-07-08 per PREREG.md (v2, real-data-only, cold-split-mandatory). The task-aware aggregator
was **not built**: its precondition — complementarity must exist under a cold split — fails on every
real task tested, with both a linear and an interaction-capable readout. This is the disciplined
outcome the prereg specifies ("P1 fails → no complementarity in this task → STOP").

## P1 precondition (complementarity under cold split; gap ≥ 0.15 balanced-acc to proceed)

| task | split | readout | best-single | oracle | gap | verdict |
|---|---|---|---|---|---|---|
| DAVIS DTI | cold-drug | MLP | 0.500 | 0.521 | +0.021 | fails |
| DAVIS DTI | cold-target (shared drugs) | MLP | 0.549 | 0.727 | **+0.177** | *holds — but leaky* |
| DAVIS DTI | **cold-pair (strictest)** | MLP | 0.500 | 0.518 | +0.018 | **fails** |
| DAVIS DTI | cold-pair | logistic | 0.637 | 0.543 | −0.094 | fails |
| PPI (D-SCRIPT) | cold (powered, 15k pairs) | MLP | 0.747 | 0.797 | +0.049 | **fails** |
| PPI (D-SCRIPT) | cold | logistic | 0.741 | 0.777 | +0.036 | fails |

## The one surprise, chased down and dissolved
The first pass flagged DAVIS **cold-target** as P1-holding (gap +0.177) with the interaction-capable
MLP — apparently contradicting the BIOMESH linear-probe negative. Treated as a claim to falsify, not
a green light: re-run across all three cold splits, the hold appears **only** on cold-target (where
drugs are shared train/test) and **vanishes under the strictest cold-pair split** (gap +0.018, oracle
AUROC 0.609 ≈ chance) with both readouts. So the cold-target hold was the **shared-drug marginal**,
exactly the leak the strictest split exists to catch. BIOMESH's negative stands — and is now confirmed
not to be a linear-probe artifact (the MLP fails cold-pair too).

## Why P1 fails even on a genuinely combinatorial task (PPI)
PPI was chosen as the fairest venue: a single protein cannot determine a pairwise interaction in
principle, so complementarity "should" be forced. Yet best-single reaches 0.747 balanced accuracy —
because high-degree **hub** proteins are identifiable from sequence, and "is this a hub" predicts
"likely interacts" and generalizes cold. The genuine joint signal beyond that marginal is only +0.049.
The recurring biomedical pattern: **entity marginals** (drug promiscuity, protein hub-ness) dominate
the learnable, cold-generalizing signal, leaving little per-query complementarity for an aggregator to
capture — cheaply or otherwise.

## Verdict and consequences (per the frozen decision rule)
- **P1 FAILS on all real tasks tested** → STOP. The task-aware aggregator (claim A) and its scale-free
  cost (claim B) were **not** evaluated, because there is no complementarity to capture — building the
  aggregator would optimize the cost of a null operation (the exact trap the prereg's P1 gate prevents).
- Two frozen encoders (ESM-2 8M, ChemBERTa), one task pair (DTI) plus one genuinely-combinatorial task
  (PPI), warm and cold splits, linear and nonlinear readouts. The negative is confound-controlled and
  consistent across all of them.
- Corroborating (per prereg): the synthetic probe could not construct valid learnable-naive-hard
  synergy either (oracle unlearnable; high-feature arms won by overfitting above the oracle ceiling).

## Where this leaves the composition thesis
This is the **fourth** confound-controlled negative (naive pooling / settling / blind indexing /
task-aware aggregation), and the first to fail at the *precondition* — there isn't even complementarity
to serve. See ../COMPOSITION_THESIS.md (updated). Frozen-model composition remains infrastructure
(G2 scale-free cost, G3 compression of reachable knowledge), not new capability.
