# CROSS-MODEL AMBIGUITY RESOLUTION — RESULTS: **COINCIDING (null, as predicted)**

Run 2026-07-09 per xresolve/PREREG.md (frozen before the run; bound to CONSTRUCT.md as a COMPONENT
precondition test, not the construct). Real frozen encoders on CIFAR-10 (ground-truth-distinct = different
classes). Full numbers in xresolve_results.json. The NULL was the *expected* outcome (the session's
convergence evidence, BIOMESH/synergy), so a positive would have triggered the hardest scrutiny; none did.

## Measure
Among 135,241 ground-truth-different (different-class) image pairs: find A-aliased pairs (bottom-5% of A's
cosine distances — A can't tell them apart despite different classes); resolution score = fraction that B
distinguishes (B-distance above B's own median → capacity-normalized, baseline = 0.5 by construction).
score ≫ 0.5 → B RESOLVES A's aliasing (INDEPENDENT). score ≤ ~0.5 → COINCIDING. Both directions.

## Result — every direction is COINCIDING, and not marginally

| A → B (does B distinguish what A aliases?) | resolution score | baseline | B-percentile of A-aliased | A–B dist corr |
|---|---|---|---|---|
| ViT → text (cross-modality) | 0.167 | 0.50 | 0.221 | 0.363 |
| text → ViT | 0.133 | 0.50 | 0.194 | 0.363 |
| ViT → DINO (cross-arch, transformer) | 0.286 | 0.50 | 0.329 | 0.156 |
| DINO → ViT | 0.251 | 0.50 | 0.306 | 0.156 |
| ViT → ResNet (transformer vs CNN) | 0.124 | 0.50 | 0.204 | 0.389 |
| ResNet → ViT | 0.155 | 0.50 | 0.222 | 0.389 |
| DINO → ResNet | 0.121 | 0.50 | 0.208 | 0.421 |
| ResNet → DINO | 0.149 | 0.50 | 0.225 | 0.421 |

**VERDICT: COINCIDING (null).** Every resolution score (0.12–0.29) is *well below* the 0.5 baseline, and
every B-percentile of A-aliased pairs is below 0.5, across cross-modality AND cross-architecture
(transformer↔transformer, transformer↔CNN, vision↔text) pairs.

## What it means — stronger than "independent-fails", it's active convergence
Scores below baseline are not merely "B distinguishes A-aliased pairs no better than random" (which would be
score ≈ 0.5, independent ambiguities). They are **below** random: pairs that A aliases are pairs that B *also*
finds hard — the models alias the **same** pairs. The positive A–B distance correlations (0.16–0.42) confirm
it directly: different frozen encoders order pairs similarly. What is ambiguous for one model is ambiguous for
another, so cross-model re-framing has nothing to resolve. This is the Platonic-convergence null, confirmed at
scale on real models — the aliasing-structure-level extension of BIOMESH/synergy's convergence finding.

The caption=label confound on the vision↔text arm is naturally controlled by the baseline normalization (text
distinguishing everything by class inflates its median too), and the arm lands COINCIDING anyway.

## Verdict and consequence (per the pre-committed rule)
- **COINCIDING → the cross-model paraconsistency-resolution extension is TOY-BOUND** on real frozen encoders.
  A second model does not resolve the first's genuine aliasing; their ambiguities coincide. The conviction that
  cross-model re-framing dissolves irreducible single-frame ambiguity is, on this evidence, in the construction,
  not in the world. Honest close of the extension.
- **The single-model paraconsistency dissociation remains banked** (reproduced this run: hold beats collapse
  **+0.059** on irreversible-ambiguity tasks, ties on reconstruction). That real, un-occupied result stands
  regardless — it was never contingent on cross-model independence.

No mandatory-scrutiny controls were triggered (no direction cleared score ≫ baseline). Consistent with the
session's spine: frozen models converge; composition/re-framing accesses reachable structure, never new.
