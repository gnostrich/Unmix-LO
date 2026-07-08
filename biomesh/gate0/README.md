# GATE ZERO — are the composite queries GENUINELY split-knowledge?

## Why this gates everything
In VIRTUALMESH G1, on the split-knowledge setup, POOLING ITSELF lost to best-single — meaning the
"composition beats parts" premise did not even fire for the baseline. If biomedical composite queries
don't genuinely require multiple specialists, then a cheap composition layer is moot (nobody needs it).
So BEFORE demonstrating cheap composition, PROVE the queries need composition at all.

## Setup
Curate a task set of biomedical queries whose answers plausibly require MULTIPLE specialists' knowledge
(e.g. drug-property / protein-function / molecule-target questions spanning sequence + structure +
molecular + text). Use tasks with VERIFIABLE answers (labels exist) — no self-consistency-only scoring.

Candidate task families (verifiable, composite):
- protein-ligand / drug-target interaction (needs protein encoder + molecule encoder)
- molecular property + biomedical-context (ChemBERTa + PubMedBERT)
- gene/protein function (scGPT or DNABERT + ESM-2 + text)
Prefer existing labeled benchmarks so scoring is honest.

## The measurement + PRE-COMMITTED thresholds
For each query, measure how well EACH single specialist alone answers it, vs. an oracle that has all.
- **Split-knowledge fraction** = fraction of queries where NO single specialist reaches acceptable
  accuracy but the UNION (oracle with all specialists' features) does.
- **PASS iff split-knowledge fraction >= 0.3** (a meaningful chunk genuinely needs multiple specialists).
- Also report: does simple POOLING of specialist features beat best-single on these queries?
  (If pooling ALSO loses — the G1 bonus-negative recurring — flag it: composition itself is not helping,
   and the whole premise is in question, not just our method.)

## Decision
- PASS (>=0.3 split, and pooling beats best-single) => genuine composite need exists; proceed to experiment/.
- FAIL (queries answerable by single specialists, OR pooling loses like G1) => there is no composite
  need to serve cheaply; STOP and report — the biomedical cluster does not exhibit split-knowledge on
  these tasks, so a composition layer (cheap or not) has no customer. This is a real, publishable finding.

This gate protects against building a beautiful cheap-composition layer for queries nobody needs composed.
