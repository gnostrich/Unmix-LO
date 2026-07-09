# CROSS-MODEL AMBIGUITY RESOLUTION — Claude Code brief

BIND FIRST to thoughtworld_construct/CONSTRUCT.md (canonical spec; do NOT flatten/build the construct here).
This is a COMPONENT precondition test.

## Context (two validated sandbox results included)
- paraconsistency_dissociation_probe.py: VALID. Holding two hypotheses beats collapsing them (+0.059) ONLY on
  irreversible-ambiguity tasks (ties on pure reconstruction). This single-model result is REAL and un-occupied.
- broken_measure_v1.py: INVALID (kept for the lesson). Raw-distance resolution metric conflates cross-model
  resolution with shared-latent proximity -> all regimes scored ~0.65. The valid measure is conditional
  distinguishability with a GROUND-TRUTH-DISTINCT control (see PREREG).

## The one question
Do two REAL frozen models with different typings have INDEPENDENT ambiguities (B resolves A's aliasing ->
cross-model paraconsistency-resolution is real) or COINCIDING ambiguities (Platonic convergence -> the idea is
toy-bound)? The session's convergence evidence (BIOMESH/synergy) predicts COINCIDING = the NULL. Test whether
data overturns that. A POSITIVE contradicts the prior and gets the HARDEST scrutiny.

## Do
Follow PREREG.md exactly. Real models, real dataset with GROUND-TRUTH-distinct inputs. Measure conditional
distinguishability (fraction of genuinely-different-but-A-aliased pairs that B distinguishes above baseline),
capacity-normalized, both directions. Two model pairs: (1) vision vs text (cross-modality), (2) two different
vision architectures (cross-architecture control). Pre-committed verdict: COINCIDING (null, expected) vs
INDEPENDENT (surprising, needs all controls to survive).

## Verdict
- COINCIDING: resolution ~ baseline -> cross-model extension toy-bound; keep single-model dissociation as the
  surviving result. Honest close.
- INDEPENDENT: resolution >> baseline, survives ground-truth-distinct + capacity + baseline + degeneracy checks
  -> real; next gate = does the fluid's PARACONSISTENT form beat classical multiple-hypothesis tracking (separate).

## Discipline
Component not construct. Null is expected; positive gets more scrutiny. Ground-truth-distinct control is the key
guard (else "aliased"="identical input"). Capacity-normalize (B shouldn't win by dimension). Report both directions.
Commit measure + results + a table. Keep the single-model paraconsistency dissociation on record regardless.
