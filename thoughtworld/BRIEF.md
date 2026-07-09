# THOUGHTWORLD — Claude Code brief

Read PREREG.md (frozen thresholds), then run curv_probe_v2.py — the VALIDATED instrument. It confirms the
measurement (effective-rank AND held-out predictivity of the deviation connection) cleanly separates:
atomic->ATOMIC, noise->NOISE, mixed->ATOMIC, random-fragment-control->NOISE (no fabrication). v1_flawed.py
is kept to show the pairing-shuffle control that FAILED and why (shuffle leaves subspace rank intact).

## What this tests (characterization, NOT capability)
When frozen HF models (fragments) are referenced against a dense self-consistent DIRECTED seed (a physics
engine), is their DEVIATION from the engine (the "swirl"/curvature) ATOMIC (real directed world-structure
beyond the engine) or STRUCTURELESS NOISE (just off-distribution error)? This is the single precondition
for the whole thought-world idea. Both outcomes are informative; NOISE is an honest negative, not a failure.

## Build
1. SEED engine: smallest dense+self-consistent+ROLLABLE+DIRECTED sim (pybullet/pymunk rigid-body, or a
   deterministic gridworld). Few objects, gravity, collisions. Must produce state trajectories.
2. FRAGMENTS: 2-3 frozen HF models overlapping the engine's physical content (vision encoder on rendered
   states; an LLM predicting next-state; optionally a frame-predictor). Frozen.
3. ALIGNMENT (only training): lightweight map fragment-repr -> engine-state-space on an overlap anchor set.
4. MEASURE per fragment: deviation dev(s)=fragment_pred(s)-engine_next(s); fit connection A (dev~f(s));
   report eff-rank(A), held-out R^2 (fit on 50% states, test on 50%), and directed fraction ||F||/||A||
   with F=(A-A^T)/2.

## Verdict (pre-committed, TWO conditions for ATOMIC)
ATOMIC iff eff-rank(A) < 0.4*D AND held-out R^2 >= 0.3. Else NOISE.
MANDATORY controls: random-fragment null (must give NOISE, else pipeline fabricates -> disqualify);
anti-trivial (dev != 0 and fragment != perfect engine match); report directed-frac (is the atomic
structure directed/noncommutative, the actual claim, or merely symmetric).

## Reads
- ATOMIC + directed: fragments carry real directed world-structure beyond the engine -> thought-world has
  content; the atomic core is its seed. Characterize further (later: does atomicity grow as seed densifies?
  do niche/bio models attach on their sub-regions? -- NOT this experiment).
- ATOMIC + symmetric-only: real shared structure but fusion-territory, not directed thought-world.
- NOISE: frozen models' world-deviations are structureless -> honest negative, world-model analogue of
  this program's composition negatives.

## Discipline
Characterization not capability. Random-fragment control is the key guard (validated in probe: gives NOISE).
Pre-commit thresholds. Honest NOISE is a real finding. Commit engine+alignment+measurement+RESULTS.md.
Keep the seed minimal first; the "how dense must the seed be to ignite structure" percolation question is
a FOLLOW-UP (vary seed complexity, see if atomicity of the deviation increases) -- only after this precondition.
