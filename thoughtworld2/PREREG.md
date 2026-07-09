# THOUGHTWORLD-2 — PRE-REGISTRATION (commit BEFORE run code)
# Follow-up to THOUGHTWORLD (which returned NOISE on frozen VISION encoders).

## What THOUGHTWORLD established (the anchor for this run)
Frozen VISION encoders (ViT-base, DINO-ViT) deviate from a physics engine's dynamics at eff-rank ~16.4
(0.82*D), STATISTICALLY INDISTINGUISHABLE from a random-fragment control (16.4 vs 16.9). Diagnosis: image
encoders barely predict physics (readout R^2 -0.10, 0.15), so their "deviation" is the engine's own
full-rank dynamics leaking through = the null's signature. Held-out R^2 was decent (0.44-0.48) but that is
leakage, not concentrated structure -- which is why BOTH conditions (low eff-rank AND held-out R^2) are required.
=> ESTABLISHED NOISE FLOOR: eff-rank ~16.4-16.9. Any fragment is only interesting if it BEATS this floor.

## The question THOUGHTWORLD did NOT answer (this run)
ViTs are APPEARANCE encoders -- they have no dynamics to deviate WITH, so NOISE was almost expected.
The live question: do fragments that ACTUALLY MODEL DYNAMICS carry ATOMIC deviation from the engine, or
is even their world-deviation structureless?
Test fragment types that plausibly hold dynamical world-structure:
  (F1) a VIDEO / next-frame prediction model (trained on dynamics)
  (F2) a small LANGUAGE model predicting next physical STATE from a text description of the scene
       (explicit causal commonsense -- the "language extends the seed" fragment)
  (F3) OPTIONAL if runnable: a tiny learned WORLD MODEL (DreamerV3-scale RSSM) -- the fragment MOST likely
       to have atomic deviation, because it is literally a dynamics model.

## Setup (reuse THOUGHTWORLD engine + instrument UNCHANGED -- only fragments change)
- SEED engine: SAME numpy rigid-body engine (5 balls, gravity, wall+ball-ball collisions, deterministic,
  directed, D=20), 2-frame-overlay renders. DO NOT change the engine (keeps the noise floor comparable).
- FRAGMENTS: F1 (video/frame-predictor, small, HF, CPU), F2 (small LLM, e.g. Qwen2.5-0.5B, describing/
  predicting next state in text mapped to engine coords). F3 if a CPU-runnable RSSM/Dreamer checkpoint exists.
- ALIGNMENT: same lightweight ridge readout (fragment-repr -> engine-state) as the only training.
- MEASURE (identical instrument, VALIDATED in thoughtworld v2 probe): deviation A = fragment_pred - engine_next;
  report eff-rank(A), held-out R^2 (fit 50% / test 50%), directed-frac ||F||/||A|| with F=(A-A^T)/2.

## PRE-COMMITTED verdict (unchanged, TWO conditions, PLUS the noise-floor reference)
ATOMIC iff: eff-rank(A) < 8 (i.e. < 0.4*D) AND held-out R^2 >= 0.3
  AND (new) eff-rank is DISTINGUISHABLE FROM THE NOISE FLOOR: eff-rank(A) < 12 with a gap to the
  random-fragment control's eff-rank that is statistically real (report the control per fragment).
Else NOISE.
MANDATORY controls: random-fragment null per fragment (must give NOISE ~ floor, else fabrication -> disqualify);
anti-trivial (dev != 0; fragment not perfectly matching engine); readout R^2 reported (does the fragment even
predict physics -- if readout R^2 <= 0 like the ViTs, the fragment has no dynamics and NOISE is expected-not-informative).

## Decisions (both outcomes strong)
- If dynamics-fragments ALSO give NOISE indistinguishable from the ViT floor:
  => GENERALIZED NEGATIVE: no frozen model, even dynamics-trained ones, carries atomic world-structure beyond
     the engine. Strong, general result. Fold into synthesis as the world-structure axis, now general.
- If any dynamics-fragment gives ATOMIC (beats floor, low eff-rank, held-out predictive, ideally directed):
  => you've found WHICH fragment types carry world-structure. The thought-world has content FOR THOSE fragments.
     Report which, and the directed-frac (is the structure directed/noncommutative -- the actual claim).
- Report readout R^2 per fragment FIRST: a fragment with readout R^2 <= 0 has no dynamics and its NOISE is
  uninformative (like the ViTs); the informative test is fragments that DO predict physics (readout R^2 > 0)
  yet whose deviation is still NOISE (or ATOMIC).

## Discipline
Reuse the SAME engine and the SAME validated instrument -- only swap fragments -- so results are directly
comparable to the ViT noise floor. Pre-commit thresholds. Random-fragment control per fragment is the guard.
Honest NOISE is a real (and now GENERAL) finding. Commit fragments+alignment+measurement+RESULTS.md, and a
table with the ViT floor row included for direct comparison.
