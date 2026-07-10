# PRE-REGISTRATION — self-expanding OV Mori–Zwanzig memory kernel as a dimension-independent
# decentralized aggregator. Frozen BEFORE run code, per standing discipline.

## The object (one thing)
An operator-valued (OV), low-rank **Mori–Zwanzig memory kernel** is the aggregator state. It observes a
stream of closure residuals from K decentralized workers and maintains a minimal state-space realization
(A, B, C) — a resolvent G(z) = C(zI−A)⁻¹B — of the shared task dynamics.

**Self-expansion by Hankel-rank / atomicity (the mechanism, not a stub):**
- A new state dimension is **appended** when the top singular value of the *closure-residual* block-Hankel
  matrix clears the **second-FDT noise floor** (the Hankel signal level attributable to fluctuation, set
  self-consistently from the residual/worker-disagreement covariance — no free knob).
- **Balanced-truncation prune**: states whose Hankel singular value falls below the floor are truncated.
- Terminal kernel order == number of Hankel singular values above the FDT floor == the estimated
  **McMillan degree** of the task distribution's resolvent (its atomic-support size).

**Decentralized & dimension-independent:** K workers each observe the shared task dynamics through their own
(noisy, possibly different-dimensional) measurement and feed closure residuals into the shared kernel in a
common p-channel basis. The kernel state dimension is **not** tied to worker count K or to any per-worker
dimension — only to the McMillan degree of the task distribution.

## Reuse (named modules absent; lineage present)
The literal `ov-ssm-stage0` / `s4c-resolvent` files are NOT in the repo. The load-bearing primitives ARE:
the MZ closure-residual + kernel eff-rank realization in `archive/pre-nuke:virtualmesh/gates/real/
gate2_mzkernel.py` (rank-vs-width grows, rank-vs-federation-size flat) and the atomicity/McMillan-degree
framing in `CONTEXT.md` (#2). We reuse the block-Hankel closure primitive and implement the resolvent
(Ho-Kalman/ERA) realization from scratch. This is stated plainly, not hidden.

## The differentiating experiment — the DOUBLE DISSOCIATION (main runnable)
Cost := kernel state dimension n (memory O(n²+np)); communication O(K·p) reported separately. The CLAIM
under test: **n tracks the McMillan degree / atomic-support size of the task distribution's resolvent, NOT
the worker count.**

- **ARM 1 — raise K (workers) at FIXED task-diversity r.**
  Frozen prediction: **kernel rank n is FLAT in K** (n ≈ r for all K; variation ≤ ±1 across the K sweep).
  Rationale: more workers lower the noise floor by averaging but add no new atomic modes; the r signal poles
  stay above the floor, no new modes appear.

- **ARM 2 — raise task-diversity r at FIXED K.**
  Frozen prediction: **kernel rank n GROWS with r**, tracking the known McMillan degree (monotone increasing;
  n(r) ≈ r within tolerance over the calibrated range).

- **Verdict rule (frozen):** the CLAIM HOLDS iff **(Arm 1 flat in K) AND (Arm 2 grows with r)**. If Arm 1 is
  not flat (rank rises with workers) → the aggregator's cost is contaminated by worker count → report Arm-1
  FAIL. If Arm 2 does not grow (rank blind to diversity) → the kernel isn't tracking degree → report Arm-2
  FAIL. Anything other than both-hold is reported as which arm failed. RED is an honest outcome.

## Poles-first calibration (checkable ground truth, run BEFORE reading the sweep)
- **Atomic control (positive arm):** a synthetic task distribution built from a KNOWN minimal realization
  (A₀∈ℝ^{r×r}, B₀, C₀), so McMillan degree = r exactly (poles = eig(A₀)). Frozen prediction: the
  self-expanding kernel's **terminal order equals r** (±1) across r in a calibrated set, and the Hankel
  spectrum shows a clean gap at r (σ_r ≫ floor ≫ σ_{r+1}).
- **Continuous-spectrum control (negative arm of the atomicity dial):** a task distribution with a
  continuous pole density (no atomic support; McMillan degree = ∞). Frozen prediction: the Hankel spectrum
  **decays smoothly with no clean gap**, the terminal order does **NOT** cleanly terminate at a small
  integer, and it **drifts with the floor / observation length / K** (the atomicity criterion is vacuous —
  as it should be when there is no atomic support). If instead it terminates at a stable small integer, the
  atomicity dial is broken (false termination) → report it.

## Playable harness (must be runnable)
A CLI: dial K and task-diversity, watch kernel rank / cost / closure residual live; add workers; change the
task distribution; run the full sweep. Deliver `results.json` + `RESULTS.md` with the two dissociation curves
and the honest verdict. Dimension-independence is demonstrated by a heterogeneous-worker-dimension run.

## Discipline
Frozen predictions committed before run code. Poles-first ground truth checked before the real sweep. No
relabeling a no-op as a result; a surprising positive gets more scrutiny; honest nulls/REDs are results. If a
load-bearing piece were truly missing and unbuildable, archive with the blocker named — but the object is
buildable here, so it is built and tested against ground truth.
