# Sandbox results (2026-07-08, this repo as committed)

All three prototypes run clean. G1/G2 reproduce the claims in README_prototypes.md; G3 (built
here — it had no prototype) passes after two documented test-construction corrections.

## G1 — settling vs pooling (split knowledge)
- 8 models, each knows 6/30 dims (union 21/30): best-single 0.894, pooling 0.818,
  settling 0.543 rel-error -> **settling 1.50x over pooling**. Matches the recorded sandbox claim.

## G2 — MZ kernel closure
- 20 coupled models, tracked subset K=6, horizon T=40: closure error 0.510 (L=1) -> 0.000 (L=5);
  kernel eff-rank 6 = K at every L (rank tracks routed subset, not federation size N=20);
  memory essential (Markovian 0.510 vs L=5 exact); residual-difficulty corr 0.291.

## G3 — pathway thickening / gap-filling (new prototype)
- Q1 THICKEN: distilled direct edge matches the A->B->C composite to 0.0000 held-out rel-error.
- Q2 GENERALIZE: on through-B-carryable dims, 97% gain vs both no-edge and random-edge;
  **bound honestly measured**: A-C-shared dims invisible to B stay at rel-error 0.99 — a
  transitive edge cannot carry them, and doesn't pretend to.
- Q3 GAP-FILL: fusing the edge-transported estimate with C's own reading, swept across the
  source-quality ratio: +24% at parity, +0% worst-case harm (fusion weights use only
  node-known noise levels, no oracle).
- Q4 FABRICATION GUARD: distilling an edge to a node sharing nothing with A gains -1% vs
  no-edge — the pipeline does not manufacture structure.
- Corrections on the record (in the prototype header): v1's world made Q2 unpassable by
  construction (disjoint A-C latents) and its Q3 degenerate-passable (C read the query
  directly); v2's Q3 used a single arbitrary noise level and oracle fusion weights.
  Thresholds were never changed to rescue a result.

## Sandbox verdicts (clean/linear — NOT real-model evidence)
| gate | sandbox | what the real gate must now show |
|---|---|---|
| G1 | PASS 1.50x | real specialists, real split knowledge, + one-step ablation arm |
| G2 | PASS (rank=K, memory essential) | nonlinear real-representation dynamics, rank vs K not N |
| G3 | PASS (with measured bound) | real distilled edges generalize; guard still refuses fabrication |
