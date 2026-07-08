# GATE ZERO (cold-split) — PRE-REGISTRATION
# Commit this file BEFORE running anything. Thresholds are frozen here.

## Motivation (what the prior run found, and why this run)
Gate-zero on DAVIS DTI (in-distribution split) FAILED: per-instance split-knowledge fraction = 0.005
(bar >= 0.30), while union/best-single AUPRC = 1.37x (bar >= 1.10, passed). Diagnosis: single-specialist
accuracy was INFLATED by marginal promiscuity structure (promiscuous drugs bind many targets;
promiscuous kinases bind many drugs), so one input usually determines the label -> the union rarely adds
per-query necessity. Cold-drug / cold-target splits are the ESTABLISHED-CORRECT DTI methodology that
removes exactly this leakage. This is NOT a relaxation of the prior test; it is the FAIRER version of the
same test. If strict complementarity is real, it should appear here; if it does not, the negative is
confound-controlled and decisive.

## Design (frozen)
- Same encoders: ESM-2 (protein, frozen) + ChemBERTa (drug, frozen). Same DAVIS DTI data.
- SPLIT: cold-drug AND cold-target (report both, and cold-pair if feasible). No drug or target seen in
  train appears in test -> removes marginal promiscuity leakage. Use scaffold/cluster split if standard.
- Retrain ONLY the light readout heads on the cold-train split; encoders stay frozen. Evaluate on cold-test.

## Pre-committed measurements + thresholds (BOTH required, gate PASSES only if both pass)
1. **per-instance split-knowledge fraction** on cold-test: fraction of test pairs where NO single
   specialist reaches acceptable accuracy but the UNION does.  PASS iff >= 0.30.
2. **union AUPRC / best-single AUPRC** on cold-test.  PASS iff >= 1.10.

## Required decomposition (report regardless of verdict)
- error correlation between the two specialists on cold-test (prior: -0.18 in-dist)
- both-wrong rate; union-rescue rate among both-wrong (prior: 5% both-wrong, 10% rescued -> 0.5%)
- compare in-distribution vs cold-split numbers side by side (does removing promiscuity raise the split fraction?)

## Decision (frozen)
- BOTH pass -> split-knowledge property is REAL under the fair split; gate-zero re-opens; proceed to the
  cost-vs-scale experiment (biomesh/experiment/). Strong outcome.
- Fails -> the negative is now CONFOUND-CONTROLLED and decisive: biomedical DTI lacks per-query
  split-knowledge even on the correct split. STOP. Publishable strong negative. The "cheaper aggregate
  accuracy" re-scope (option 3) becomes the only honest remaining pitch, to be decided separately.

## Discipline
- Honest RED is success. Do NOT tune the split or threshold to pass. Keep prior in-dist results on record.
- If cold-split accuracy collapses for BOTH specialists (encoders don't generalize to unseen drugs/targets),
  that is a DIFFERENT finding (frozen encoders don't transfer cold) -> report it, do not conflate with
  the split-knowledge question. Note it explicitly.
