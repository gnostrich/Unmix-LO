# FDT / NATIVE-DENOISING — Claude Code brief (safe to run IN PARALLEL with xresolve)

BIND FIRST to thoughtworld_construct/CONSTRUCT.md. COMPONENT test of a THEORY claim (does the Baur/MZ
process denoise the disagreement-swirl NATIVELY), not the construct itself. Independent of xresolve --
can run in parallel.

## The theory being tested
Native denoising is principled IFF the second-FDT relation holds for the disagreement dynamics (memory kernel
linked to noise autocorrelation). Theory: FDT holds EXACTLY on the disagreement-component that inherits the
shared world's GENERATOR (= F_gauge = content); the FDT-violating part is F_noise. So the MZ process natively
separates content from noise -- IF FDT actually holds on real disagreement. This run measures the fraction.

## fdt_probe_v1_failed_validation.py -- READ THE LESSON
Included and FLAGGED FAILED: its crude autocorrelation-based FDT estimator did NOT validate on ground truth
(an OU process that MUST satisfy FDT came back weak/ambiguous, +0.178, not clearly positive). So a proper
estimator is required, and it MUST pass STEP 0 before any real-model result is believed.

## Do (in order)
STEP 0 (GATES EVERYTHING): build an OU/linear-generator+noise process (FDT holds by construction) and a
  random-walk-difference process (FDT fails by construction). Build a PROPER FDT estimator (proper memory-kernel
  extraction -- projection/Volterra or Prony/ARMA fit -- and true residual noise, NOT crude autocorrelation).
  It MUST score OU high and random-diff low. If it can't separate these known cases, FIX IT before proceeding.
  Report the validation numbers. NO valid STEP 0 -> NO trustworthy real result.
STEP 1: physics engine (reuse thoughtworld; it has a generator). Two real frozen models, different typings,
  aligned to a common space. Disagreement d_t = repA(state_t) - repB(state_t) over the trajectory.
STEP 2: extract memory kernel K(t) + noise xi from d_t with the VALIDATED estimator; compute the FDT relation;
  decompose d_t into FDT-satisfying (F_gauge, generator-inherited) vs FDT-violating (F_noise) subspaces.
STEP 3: report the FRACTION of disagreement variance that is FDT-satisfying (the F_gauge/F_noise ratio).

## Verdict
- HIGH FDT-satisfying fraction -> native denoising PRINCIPLED for real-model disagreement (suspicion holds).
- LOW -> disagreement mostly idiosyncratic noise; noise floor is heuristic, not native. Honest downgrade.
- MID -> that fraction IS the natively-denoisable content share; report it.

## Controls (mandatory)
- STEP 0 estimator validation GATES the run (the v1 estimator failed it -- do not repeat).
- RANDOM-MODEL control: one model -> random features; its disagreement must score MOSTLY FDT-violating.
  If random-model disagreement scores HIGH FDT -> estimator fabricating -> disqualify.
- Run cross-arch AND cross-modal model pairs; report if generator-inheritance differs by pair type.

## Discipline
Component not construct. STEP 0 non-negotiable. Random-model control is the fabrication guard. Both outcomes
informative. Commit estimator + STEP-0 validation + results + table.
