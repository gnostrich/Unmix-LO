# THOUGHTWORLD — PRE-REGISTRATION (commit BEFORE run code)
# Question (characterization, NOT capability): when frozen models (fragments) are referenced against
# a dense, self-consistent, directed SEED (a physics/game engine), does their DEVIATION from the engine
# (the "swirl" / curvature) have ATOMIC structure (real directed world-structure the fragments carry
# beyond the engine) or is it STRUCTURELESS NOISE (fragments just wrong off-distribution)?
#
# This is the precondition for the whole "thought-world" idea. If deviation is atomic -> there is a
# real directed structure to build on. If noise -> honest negative (frozen models' world-deviations
# are structureless), the world-model analogue of this program's composition negatives.

## Math object
Engine = flat reference connection nabla_0 (self-consistent, directed, rollable). Each fragment i has
its own implied dynamics nabla_i. The object is A_i = nabla_i - nabla_0 (deviation 1-form / connection);
its curvature/atomicity is the question. The engine FIXES THE GAUGE -> A_i is well-defined (without a
coherent reference, deviation is gauge-ambiguous; this is why the dense seed is mathematically necessary).

## SANDBOX LESSON (see curv_probe_v1_flawed.py)
The effective-rank measurement WORKS (atomic dev -> eff-rank ~3; noise -> eff-rank ~full-D). But the
pairing-shuffle control was WRONG: shuffling destroys state-dependence yet leaves the subspace rank intact,
so it can't tell "real coherent curvature" from "low-rank coincidence." CORRECTED verdict below uses
HELD-OUT PREDICTIVITY as the guard, not shuffle.

## Setup
- SEED: minimal dense+self-consistent+DIRECTED engine. Start simplest: a small rigid-body physics sim
  (pybullet/pymunk) or a deterministic gridworld with consistent dynamics. Must be ROLLABLE (produce
  state trajectories) and SELF-CONSISTENT (deterministic). Keep tiny (few objects, gravity, collisions).
- FRAGMENTS: 2-3 frozen HF models with SOME overlap with engine physical content (a vision encoder on
  rendered states; an LLM predicting/describing next-state in text; optionally a frame-prediction model).
- ALIGNMENT (only training allowed): a lightweight learned map fragment-repr -> engine-state-space,
  fit on an overlap anchor set (rendered/described states <-> engine states). This is projection onto nabla_0.

## Measurement
1. Roll engine -> trajectories (engine IS the data generator; no external dataset).
2. For each fragment: given engine state s_t, get fragment's predicted s_{t+1} (mapped into engine coords).
   Deviation dev(s) = fragment_pred(s) - engine_next(s). Collect over many states.
3. Fit the connection A: dev ~ f(s) (linear first: dev ~ s @ A^T; report nonlinear residual too).
4. Report the SPECTRUM of A: effective rank (participation ratio), and the antisymmetric part F=(A-A^T)/2
   (the directed/noncommutative curvature) and its rank.

## PRE-COMMITTED verdict (TWO conditions, both required for ATOMIC)
(1) LOW EFFECTIVE RANK: eff-rank(A) < 0.4 * D (deviation concentrates in few directions).
(2) HELD-OUT PREDICTIVITY: fit A on 50% of states, test on held-out 50%; the map must predict held-out
    deviations with R^2 >= 0.3 (the connection is a REAL FUNCTION of state, not a coincidence).
ATOMIC (signal) iff (1) AND (2).  NOISE (structureless) iff not(1) OR not(2).
CONTROLS (mandatory):
- RANDOM-FRAGMENT null: replace fragment predictions with random vectors of matched norm -> must give
  NOISE (high rank, low held-out R^2). If random gives "atomic", the pipeline fabricates -> disqualify.
- ANTI-TRIVIAL: confirm dev != 0 (fragment isn't just reproducing engine) AND fragment isn't perfectly
  matching engine (then there's no object). Report ||dev||/||engine_next||.
- DIRECTEDNESS: report F=(A-A^T)/2 magnitude vs symmetric part; the CLAIM is directed structure, so if
  all structure is symmetric (F~0) the "directed" part is absent -> note it (still could be atomic-but-undirected).

## Reads (both outcomes informative)
- ATOMIC + directed (F significant): fragments carry real directed world-structure beyond the engine ->
  the thought-world has content; the atomic core IS the seed of it. Proceed to characterize it further.
- ATOMIC + undirected: real shared structure but not directed -> it's fusion-territory, not thought-world.
- NOISE: frozen models' world-deviations are structureless -> honest negative; report as world-model
  analogue of the composition negatives.

## Multi-scale note (for later, not this experiment)
Niche models (bio etc.) would attach on the sub-region of state-space where their domain overlaps the
engine, contributing A on their slice -> the object is naturally multi-scale. NOT tested here; this
experiment is the single precondition (is the deviation atomic at all) on 2-3 physical fragments.

## Discipline
Characterization not capability. Pre-commit thresholds. Random-fragment control is the key guard.
Honest NOISE verdict is a real finding, not a failure. Commit engine + alignment + measurement + RESULTS.md.
