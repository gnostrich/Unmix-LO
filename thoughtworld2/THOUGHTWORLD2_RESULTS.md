# THOUGHTWORLD-2 — RESULTS: **NOISE (generalized negative)** — dynamics-trained fragments too; but the LLM comes closest

Run 2026-07-09 per PREREG.md (frozen before the run). SAME engine + SAME validated instrument as
THOUGHTWORLD (imported unchanged); only the fragments changed to ones that actually model dynamics.
Full numbers in thoughtworld2_results.json.

## The question
THOUGHTWORLD found NOISE on frozen *vision* encoders — but ViTs have no dynamics to deviate *with*, so
that was uninformative. Do fragments that actually model dynamics (an LLM predicting next-state from a
scene description; a video model trained on dynamics) carry **atomic** deviation, or is even their
world-deviation structureless? Verdict frozen: ATOMIC iff eff-rank < 8 AND held-out R² ≥ 0.3 AND
distinguishable from the per-fragment random control (and the ViT floor of eff-rank ~16.4).

## Result (readout R² reported first — does the fragment even predict physics?)

| fragment | readout R² | eff-rank(A) | held-out R² | control eff-rank | directed-frac | verdict |
|---|---|---|---|---|---|---|
| ViT floor (THOUGHTWORLD ref) | −0.10 | 16.4 | 0.44 | 16.9 | 0.20 | NOISE |
| **F2 Qwen2.5-0.5B (LLM)** | **0.566** | **13.06** | **0.163** | 17.98 | **0.45** | **NOISE** |
| F1 VideoMAE (video) | 0.335 | 16.14 | 0.615 | 17.47 | 0.33 | NOISE |

**Both dynamics-fragments → NOISE. Verdict: GENERALIZED NEGATIVE.** Per-fragment random controls give
eff-rank ~17.5–18 (NOISE), so no fabrication. Anti-trivial passed (dev rel-norm 0.48 / 0.59).

## The honest nuance: the LLM is the most-structured fragment tested — but still fails the bar
Unlike the vision encoders, the LLM is the *informative cell* the prereg wanted, and it does not behave
like the noise floor:
- It **predicts the physics** (readout R² 0.57, vs ViT −0.10) — a linear readout on its representation
  recovers next-state well (the description carries positions+velocities; dynamics is near-linear).
- Its deviation is **genuinely more concentrated than the floor**: eff-rank 13.06 vs ViT 16.4, and — the
  real test — **below its own random control** (13.06 vs 17.98, a gap of ~4.9 ≫ the 1.5 threshold). So on
  the eff-rank / distinguishability conditions, the LLM *passes*.
- It is the **most directed**: directed-frac 0.45 (vs 0.20 for the ViT floor) — the antisymmetric
  (noncommutative) part is a larger share.
- **But held-out R² = 0.163 < 0.30**: the concentration is NOT a coherent generalizing function of state.
  The low-rank structure exists but does not predict held-out deviations. Per the pre-committed AND, that
  is NOISE — this is exactly the failure the two-condition verdict is designed to catch (low-rank
  coincidence without predictivity), the reason v1's shuffle control was replaced by held-out R².

So the LLM shows a **partial, sub-threshold signal**: concentrated (beats its control), directed, and
physics-predicting — but not coherently state-predictable. Not atomic, but not the flat null the ViTs
gave either. Reported as-is, not inflated.

## VideoMAE: full-rank engine leakage despite dynamics training
VideoMAE predicts physics somewhat (readout R² 0.335) and has a *high* held-out R² (0.615) — but its
eff-rank is 16.14, at the ViT floor. The high R² here is the **engine's own dynamics leaking through**
(broad, near-full-rank linear structure), NOT atomicity — precisely the confound the prereg flagged, and
why held-out R² alone is insufficient and the eff-rank condition is load-bearing. So a video model trained
on real dynamics deviates from this engine with the same full-rank signature as an appearance encoder.

## Reading (per the pre-committed decisions)
**GENERALIZED NEGATIVE.** No frozen model tested — appearance encoder (ViT/DINO), video model (VideoMAE),
or language model (Qwen) — carries atomic world-structure beyond the engine that clears the bar. The
world-structure lives in the engine; frozen models add (near-)structureless deviation to it. This is the
world-model axis of the program's finding, now general across fragment types, not just vision.

The one caveat carried honestly: the LLM's sub-threshold-but-non-null signal (concentrated + directed +
predictive-of-physics, failing only held-out coherence) is the single place across two experiments where a
frozen fragment departs meaningfully from the noise floor. The prereg's pre-registered follow-up — does
atomicity ignite as the seed densifies, or with domain-matched fragments / better alignment — is where that
hint would be pursued; nothing here clears the bar, and the frozen thresholds stand.

## Where it sits
Same shape as the whole program, now on the world-structure axis and generalized across fragment types:
frozen-model composition/reference is infrastructure over *reachable* content (G2/G3/ROUTEMESH), not a
source of *new* structure — task capability (four ">" negatives) or world-model structure (THOUGHTWORLD 1+2).
See ../COMPOSITION_THESIS.md and ../report.md.
