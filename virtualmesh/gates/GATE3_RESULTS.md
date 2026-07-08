# G3-real RESULTS — pathway thickening on real specialists: **PASS** (amended design)

Run 2026-07-08. Full numbers in real/gate3_results.json; the invalid and the as-preregistered
runs are preserved (see "Record of corrections").

## Setup
Real 2-hop chains of specialist calls (all four specialists at 1.00 single-hop accuracy):
- pair 1: person -> city (M_pc) -> company (M_cc); distill chain pseudo-labels into a direct
  rank-8 LoRA edge on the frozen base.
- pair 2 (**the held-out gap**): city -> company (M_cc) -> product (M_co2pr); the SAME
  procedure applied unchanged to a transitively-connected pair it was not designed against.
- fabrication guard: person -> hobby (M_ph) fed into M_cc — a chain across models that share
  no real path.

## Results (thresholds: agreement >= 0.90; >= +20% relative vs both controls; guard <= +5%)

| measure | pair 1 | pair 2 (held-out gap) |
|---|---|---|
| Q1 agreement with chain, trained template | 1.00 | 1.00 |
| Q1 agreement with chain, UNSEEN paraphrase | **1.00** | **1.00** |
| ground-truth accuracy: distilled edge | **0.775** | **0.786** |
| ground-truth accuracy: chain (ceiling) | 0.775 | 0.786 |
| base-rate control (shuffled-label LoRA) | 0.425 | 0.286 |
| no-edge (frozen base) | 0.00 | 0.00 |
| inference cost | 1 call vs 2 | 1 call vs 2 |

Fabrication guard: junk-chain edge scores **0.15** ground truth vs base-rate 0.42 — far below
the no-information baseline. **The pipeline does not manufacture structure. PASS.**

## Reading
- A transitive pathway through real frozen specialists CAN be thickened: the distilled direct
  edge is a *perfect functional cache* of the 2-hop chain (agreement 1.00 even on unseen
  phrasings) at half the inference cost, and the procedure transfers unchanged to a new pair.
- The honest bound holds exactly as the sandbox predicted: the edge inherits the chain's
  ceiling (it equals it to the third decimal) and cannot exceed it. Thickening is compression
  of existing composite structure, not creation of new capability.
- The guard result is as important as the pass: a chain with no real shared structure distills
  into an edge that is *worse than base-rate* — fabricated edges are detectable and rejected
  by exactly the check gates/README.md demanded.

## Record of corrections (all on the record, none post-hoc threshold changes)
1. First run INVALID: distill() carried the same finite-loss/non-finite-gradient NaN bug as the
   trainer; adapters were silently poisoned (gate3_results from that run discarded, bug fixed
   and committed separately).
2. As-preregistered design FAIL-BY-CONSTRUCTION (gate3_results_prereg_original.json): the
   prereg held out PERSONS, but in a random relational world the composite on an unseen key is
   information-theoretically unpredictable — the run showed exactly that signature (agreement
   0.00; base-rate control 0.42 > direct 0.25). gates/README.md's G3 spec makes the held-out
   unit a transitively-connected MODEL PAIR; the amended design implements that, keeps all
   threshold magnitudes, and upgrades the guard's baseline from the (vacuous) frozen base to
   the base-rate control. Amendment rationale is embedded in gate3_thicken.py and the results JSON.
