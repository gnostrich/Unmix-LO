# CROSS-MODEL AMBIGUITY RESOLUTION — PRE-REGISTRATION (commit BEFORE run)
#
# BIND TO thoughtworld_construct/CONSTRUCT.md. This is a COMPONENT test (precondition), NOT the construct.
#
# HONESTY FRAME (read first): this session has TWO relevant prior results that make the NULL the expected
# outcome, so a POSITIVE here demands the hardest scrutiny (surprising-positive-gets-more-scrutiny rule):
#   - BIOMESH/synergy: independently-trained frozen encoders CONVERGE; complementarity beyond marginals is thin.
#   - The composition-negative arc: frozen models are more SIMILAR than hoped.
# => PRIOR = real models' ambiguities likely COINCIDE (Platonic convergence) -> cross-model resolution
#    would be TOY-BOUND. We are testing whether the data overturns that prior. Designed to say NO.

## What established this test (do not re-litigate)
- Single-model PARACONSISTENCY dissociation is REAL and un-occupied: holding two hypotheses beats collapsing
  them (+0.059) ONLY on irreversible-ambiguity tasks where evidence is interpretable only via held hypotheses
  (ties on pure reconstruction). This survived scrutiny. [validated in sandbox]
- The EXTENSION "a second model resolves the first's paraconsistency" requires the two models' ambiguities to
  be INDEPENDENT. If they COINCIDE, re-framing has nothing to resolve. THIS is the untested precondition.
- Sandbox attempts to measure resolution via raw distance FAILED (conflated with shared-latent proximity) --
  the valid measure is CONDITIONAL DISTINGUISHABILITY controlling for latent distance (below). Real models required.

## The question (single, falsifiable)
Take two REAL frozen models with genuinely DIFFERENT typings of the SAME input (e.g. a vision encoder and a
text/language encoder reading the same scene; or two very different-architecture encoders). Among input pairs
that model A maps to near-identical representations (genuine A-aliasing) BUT that are actually DIFFERENT inputs,
does model B DISTINGUISH them (INDEPENDENT ambiguity -> resolution real) or does B ALSO alias them
(COINCIDING ambiguity -> Platonic convergence -> conviction toy-bound)?

## Valid measure (fixes the broken sandbox metric)
For many input pairs (x_i, x_j) that are GENUINELY DIFFERENT (known distinct inputs):
1. Find A-ALIASED pairs: ||repA(x_i) - repA(x_j)|| in the bottom 5% (A can't tell them apart).
2. CONTROL for confound: restrict to A-aliased pairs that are ALSO genuinely different (verify via a ground-truth
   label / known distinct source images -- NOT via any model's rep). This removes "they're just identical inputs".
3. RESOLUTION SCORE = fraction of these genuinely-different-but-A-aliased pairs that B distinguishes
   (||repB(x_i)-repB(x_j)|| above B's median pair distance). 
   - score ~ chance/baseline (pairs distinguished at the base rate) => COINCIDING (null).
   - score >> baseline => B RESOLVES A's aliasing (independent ambiguity, effect real).
4. SYMMETRIZE: also measure A-resolves-B (does A distinguish B's aliased pairs). Report both directions.

## Arms / models (pick genuinely different typings; the more different, the fairer the test)
- Pair 1: a VISION encoder (e.g. ViT/DINO) and a TEXT encoder reading a caption of the same image (different modality).
- Pair 2 (control for "just different modality"): two SAME-modality but DIFFERENT-architecture vision encoders
  (e.g. ViT vs a CNN/ResNet) -- tests whether resolution needs cross-MODALITY or just cross-ARCHITECTURE.
- Use a dataset with KNOWN-distinct inputs so "genuinely different" is ground-truth, not model-inferred.

## PRE-COMMITTED verdict
- COINCIDING (NULL, expected): resolution score ~ baseline (B distinguishes A-aliased pairs no better than random
  pairs). => cross-model paraconsistency-resolution is TOY-BOUND; the surviving real result is the single-model
  dissociation only. Report as the honest close of the extension.
- INDEPENDENT (SURPRISING positive): resolution score >> baseline in at least one direction, ROBUST to the
  ground-truth-distinct control (not an artifact of aliased pairs being identical inputs). => cross-model
  resolution is REAL; the conviction is earned; proceed to test whether the fluid's PARACONSISTENT form beats
  classical multiple-hypothesis tracking (the occupancy question, a SEPARATE next gate).
- MANDATORY scrutiny on a positive (it contradicts the session's convergence prior): re-run the ground-truth-
  distinct control; confirm it's not driven by a handful of degenerate pairs; confirm B's "resolution" isn't just
  B having higher capacity/dimension (normalize by B's overall distinguishing power). A positive that survives
  ALL of these is real; anything less is the convergence prior reasserting.

## Controls (mandatory)
- GROUND-TRUTH-DISTINCT: A-aliased pairs must be verified different by a label OUTSIDE any model (else "aliased"
  = "identical input" and resolution is trivially impossible/meaningless).
- CAPACITY normalization: normalize resolution by B's median pair-distinguishing scale (a higher-dim B shouldn't
  win by scale alone).
- BASELINE: resolution score for RANDOM (non-A-aliased) pairs -> the score for A-aliased pairs must EXCEED this.
- SYMMETRY: report both A-resolves-B and B-resolves-A; a real effect need not be symmetric but report both.

## Scope & discipline
COMPONENT test of a PRECONDITION, not the construct (bind to CONSTRUCT.md; do not build the fluid/tape here).
Pre-register, commit before running. The NULL is the EXPECTED outcome given the session's convergence evidence;
a positive gets MORE scrutiny, not less. Honest COINCIDING is a real finding (closes the cross-model extension,
keeps the single-model dissociation). Keep the single-model paraconsistency dissociation on record as the
surviving un-occupied result regardless of this outcome.
