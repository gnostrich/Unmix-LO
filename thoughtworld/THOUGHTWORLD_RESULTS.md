# THOUGHTWORLD — RESULTS: **NOISE** (honest negative — the world-model analogue of the composition negatives)

Run 2026-07-09 per PREREG.md (frozen before the run). Seed engine: numpy rigid-body physics (5 balls,
gravity, wall + ball-ball collisions; deterministic, directed; state D=20). Fragments: two frozen HF
vision encoders on 2-frame-overlay renders of engine states. Only training: a ridge readout (alignment).
Full numbers in thoughtworld_results.json. Instrument validated separately (curv_probe_v2.py).

## The question
A dense self-consistent engine fixes the gauge, so each frozen fragment's **deviation** from true
dynamics, dev(s) = fragment_pred(s) − engine_next(s), is well-defined. Is dev **atomic** (low-rank,
held-out-predictable — real directed world-structure the fragment carries beyond the engine) or
**structureless noise** (just off-distribution error)? Pre-committed: ATOMIC iff eff-rank(A) < 0.4·D
(=8) AND held-out R² ≥ 0.3, else NOISE.

## Result

| fragment | readout pred-R² | dev rel-norm | eff-rank(A) | held-out R² | directed-frac | verdict |
|---|---|---|---|---|---|---|
| google/vit-base-patch16-224 | −0.10 | 0.77 | **16.41** | 0.435 | 0.20 | **NOISE** |
| facebook/dino-vitb16 | 0.15 | 0.67 | **16.37** | 0.478 | 0.20 | **NOISE** |
| random-fragment (control) | — | — | 16.88 | 0.084 | — | NOISE ✓ |

**THOUGHTWORLD verdict: NOISE.** Both real fragments fail the atomicity (eff-rank) condition; the
control passes (no fabrication).

## Why NOISE — the mechanism (unambiguous)
- The frozen ViTs barely predict the physics at all: readout pred-R² is **−0.10** (ViT-base) and
  **0.15** (DINO) — a linear map from their features to next-state does essentially no better than the
  mean. Off-the-shelf vision encoders do not encode this world's dynamics in a linearly-decodable way.
- So the deviation dev ≈ (constant) − engine_next(s): it is dominated by the **engine's own full-rank
  dynamics** showing through, not by any structure the fragment adds. Hence eff-rank(A) ≈ 16.4 (≈ 0.82·D,
  spread across nearly all directions), far above the atomic threshold of 8.
- **The decisive tell**: the real fragments are statistically **indistinguishable from the random-
  fragment control** on eff-rank (16.4 / 16.4 vs 16.9). The frozen models add no concentrated structure
  over the null. The directed-frac is also identical (0.20) — no special directed structure either.
- The held-out R² is decent (0.44–0.48) but this is the *same engine-dynamics leakage* (dev is a
  broad, near-full-rank linear function of state), not atomic structure — which is exactly why the
  prereg required BOTH conditions. The control's low R² (0.08) reflects its norm-matched randomness
  washing out even that leakage; the eff-rank condition is what cleanly separates all three.

Anti-trivial check passed: dev rel-norm 0.67–0.77 (fragments are not reproducing the engine; the object
is nonzero). Directedness: F/A ≈ 0.20 for real fragments and the null alike — no directed atomic core.

## Reading (per the pre-committed reads)
**NOISE**: frozen models' world-deviations, referenced against a coherent physics seed, are
structureless — no atomic (concentrated, directed) world-structure beyond the engine. This is the
honest negative the prereg anticipated: the **world-model analogue of this program's composition
negatives**. Just as frozen models hold no *new joint capability* to compose (four dead ">" tests),
their *deviations from a shared world-model* hold no *atomic directed structure* to seed a thought-world
— on this minimal seed, with these two vision fragments.

## Honest scope
One minimal seed (2D 5-ball physics), two general vision encoders, single-frame-pair inputs. The prereg
flags the percolation question — *does atomicity ignite as the seed densifies, or do niche/domain models
attach on their sub-regions?* — as an explicit FOLLOW-UP, not this experiment. This experiment tested the
single precondition (is the deviation atomic at all) and the answer, confound-controlled, is no. A denser
seed or domain-matched fragments could differ; nothing here suggests they would, and the fragments being
indistinguishable from the random null is a strong-form negative.

## Where it sits
Consistent with the program's spine: frozen-model composition is infrastructure over *reachable* content
(G2/G3/ROUTEMESH), not a source of *new* structure — capability (the ">" tests) or world-structure (this).
See ../COMPOSITION_THESIS.md and ../report.md.
