# FDT / NATIVE-DENOISING TEST — PRE-REGISTRATION (commit BEFORE run)
# BIND to thoughtworld_construct/CONSTRUCT.md. COMPONENT test of a THEORY claim, not the construct.

## The claim under test
The Baur/MZ process denoises the disagreement-swirl NATIVELY (not via a bolted-on threshold) IFF the
second-fluctuation-dissipation relation holds for the disagreement dynamics: memory kernel K(t) linked to
noise autocorrelation <xi(t)xi(0)>. Theory result (from sandbox reasoning): FDT holds EXACTLY on the
disagreement-component that INHERITS THE SHARED WORLD'S GENERATOR -> that component = F_gauge (reproducible,
content-bearing); the FDT-violating component = F_noise (idiosyncratic, vacuous). So native denoising applies
to exactly the content and rejects exactly the noise -- IF the FDT relation actually holds on real disagreement.

## What this measures
On REAL frozen models viewing a world WITH a generator (the physics engine), what FRACTION of the two-model
disagreement satisfies the second-FDT relation (= inherits the generator = F_gauge, natively denoisable) vs
violates it (= F_noise)? This is the F_gauge/F_noise ratio, measured properly.

## CRITICAL: validate the FDT estimator FIRST (the sandbox failed here)
A prior sandbox FDT estimator FAILED its own ground-truth check (an OU process that MUST satisfy FDT came back
weak/ambiguous). So STEP 0 is mandatory:
  STEP 0: build an OU / linear-generator-plus-noise process (FDT holds BY CONSTRUCTION) and a pure-random-walk-
          difference process (FDT FAILS by construction). The estimator MUST give high FDT-score on OU and low
          on the random difference. If it cannot separate these two known cases, the estimator is invalid and
          NO real-model result can be trusted -- fix the estimator before proceeding. Report the validation.
  A valid FDT estimator likely needs: proper memory-kernel extraction (e.g. via the projection K(t) from the
  orthogonal-dynamics / Volterra route, or a Prony/ARMA fit), and the noise term as the true residual after
  the resolved propagator. Do NOT use a crude autocorrelation proxy (that is what failed).

## Setup (only after STEP 0 passes)
- WORLD: the physics engine (reuse from thoughtworld; it HAS a generator -- deterministic dynamics).
- Two REAL frozen models with different typings reading the engine states (e.g. a vision encoder on renders
  and a text/LLM encoder on descriptions; or two different-architecture vision encoders as a control).
- Align each to a common space (lightweight learned map, train split).
- DISAGREEMENT trajectory: d_t = repA_aligned(state_t) - repB_aligned(state_t) over the engine trajectory.
- Extract MZ memory kernel K(t) and noise term xi from d_t (the VALIDATED estimator from STEP 0).
- FDT SCORE = shape-correlation (or a proper linear-response check) between K(t) and <xi(t)xi(0)>.

## PRE-COMMITTED reads
- Decompose d_t: the FDT-satisfying subspace (F_gauge, generator-inherited) vs FDT-violating (F_noise).
- Report the FRACTION of disagreement variance that is FDT-satisfying (the F_gauge/F_noise ratio).
- HIGH FDT-satisfying fraction -> native denoising is PRINCIPLED for real-model disagreement (suspicion holds):
  the MZ noise floor is real, and the Baur process natively separates content from noise. Strong positive.
- LOW FDT-satisfying fraction -> the disagreement is mostly idiosyncratic non-generator noise; the "noise floor"
  is a heuristic threshold, denoising is not native. Honest downgrade of the naturality claim.
- MID -> report the fraction; that fraction IS the natively-denoisable content share.

## Controls (mandatory)
- STEP 0 estimator validation (OU-positive, random-diff-negative) -- GATES everything. No STEP 0, no result.
- RANDOM-MODEL control: replace one model with random features -> disagreement should be MOSTLY FDT-violating
  (F_noise); if random-model disagreement scores HIGH FDT, the estimator is fabricating -> disqualify.
- CROSS-ARCH vs CROSS-MODAL: run both model pairs; report if generator-inheritance differs by pair type.
- Capacity/scale normalization as usual.

## Scope & discipline
COMPONENT test of the native-denoising theory claim; NOT the construct (do not build the fluid/tape).
STEP 0 estimator validation is non-negotiable and gates the whole run (the sandbox estimator failed it).
Pre-register; the random-model control is the fabrication guard. Both outcomes are informative. Commit
estimator + validation + results + a table (FDT-satisfying fraction per model pair, with controls).
