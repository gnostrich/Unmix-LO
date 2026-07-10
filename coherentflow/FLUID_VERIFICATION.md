# FLUID_VERIFICATION — the theoretically-correct feedback fluid meets its acceptance criteria

Built 2026-07-09 per the correctness-first directive. Bound to `thoughtworld_construct/CONSTRUCT.md`.
Module: `coherentflow/fluid_settle.py` (runnable, CPU, deterministic; dumps `fluid_settle_results.json`).
**No existing code was modified** — the shipped averaging settle (`coherentflow.settle`) is left intact and
correctly labelled feed-forward averaging (see `MECHANISM_CHECK.md`). This is a **separate, correct-by-
construction** module; utility / wiring-to-real-models is a deliberate later step.

## The object (models as OPERATORS, not fixed targets)
Each model `i` is an operator `Rᵢ` that reads the shared state through its frame and writes back a frame-
transformed contribution, so models drive each other. Coupled flow and its Jacobian:
```
s ← s + step · Σ wᵢ (Rᵢ − I) s        J(w) = I + step · Σ wᵢ (Rᵢ − I)
```
Because the frame operators can be **non-normal / gain>1** (a model insisting past the state, or read-dim ≠
write-dim in a cycle), `J(w)` **can exceed spectral radius 1** for incompatible frames — the property the
averaging settle (`J = (1−step)I + step·mean(Pᵢ)`, symmetric projectors, eigenvalues ≤ 1) structurally cannot
have. That is the whole difference between the fluid and occupied consensus fusion.

## Acceptance criteria — ALL PASS (the definition of "correct", independent of utility)

### [1] MUTUAL INSTABILITY — the coupled radius CAN exceed 1  ✅
| model set | coupled ρ(J) | reading |
|---|---|---|
| agreeing frames (fluid ops) | 0.978 | contracts (compatible) |
| **conflicting frames (fluid ops)** | **1.088** | **amplifies — genuine mutual instability** |
| minimal 2D skew (read-dim ≠ write-dim cycle) | **1.044** | `|λ| = √(1+step²) > 1`, provable |
| averaging (shipped settle, same-shape) | 0.978 | **≤ 1 by construction — cannot ever be unstable** |

The acceptance test is met: a model set exists for which the coupled fluid is unstable (ρ > 1), and the
averaging Jacobian is bounded ≤ 1. The instability comes from **frame-conflict** (non-normal coupling), exactly
as required — naive symmetric/contractive operators would not produce it.

### [2] FLUID EXCLUSION — descending the instability routes around the destabilizer  ✅
3 compatible models + 1 rogue (non-normal, gain>1). At equal weights ρ = **1.015 (unstable)**. The fluid
descends its own measured growth over the routing simplex (power-method growth + finite-difference gradient,
robust to the *oscillatory* leading eigenvalue). Result: **rogue weight → 0.0000 (excluded)**, ρ → **0.970
(< 1, stabilized)**. This is the behaviour the averaging settle cannot produce — it has no instability to
descend, so a corrupt model is simply averaged in (the bug flagged in `MECHANISM_CHECK.md`, empirically: a 5σ
model shifts the averaging consensus by ~1.0 rather than being routed around).

Distinction that emerges from the *same* descent: an **asymmetric rogue** is zeroed out (exclusion); **symmetric
rivals** are instead **balanced** to marginal stability (neither excluded) — the seed of held-superposition.

### [3] INTRINSIC OUTPUT — query = perturbation, re-settle, equilibrium-SHIFT is the answer  ✅
No external readout head. Query perturbs the settled state; the fluid re-settles; the shift is the answer.
- **(a) agreed query → collapses to consensus** (response norm **0.0000**): where models agree, the shift
  moves into agreement and the contested content is zero — the ordinary consensus answer.
- **(b) adversarial query → bounded** (response norm **1.96** for input magnitudes 5, 50, 500 over 1000 steps):
  the saturation caps it; instability-descent + bounded settle mean an adversarial query cannot blow up.
- **(c) contested query → HELD-SUPERPOSITION** (rank-2, singular values **[5.65, 5.25]**, branches **±1.39**):
  where two rival models structurally differ, the response does **not** collapse to one consensus midpoint —
  it holds **both** branches. Over a sweep of contested queries the responses span the full 2-D rival plane
  (two significant singular values), and the branch set retains both ±.

## Honest theoretical finding (reported, not hidden)
Criteria **1-2 are properties of the LINEAR coupled operator** and hold for the linear feedback flow.
Criterion **3c (robust held-superposition) is NOT achievable by linear feedback.** A linear flow can hold a
contested subspace only at *fragile exact-marginal stability* (ρ = 1, measure-zero): any drift makes one branch
dominate and the superposition collapses to a single direction (verified — a ρ≈1.008 linear hold amplified one
branch away over a few hundred steps). **Robust held-superposition requires NONLINEAR MULTISTABILITY** — a
bounded saturation that turns rival model directions into two stable wells, with the consensus fixed point
**unstable on contested subspaces** so the fluid *commits* to a held branch rather than blandly averaging to
the midpoint. So `fluid_settle`'s intrinsic settle is nonlinear (saturating) by necessity.

This is a real result about the theory, not a workaround: **the paraconsistent hold is a nonlinear, multistable
phenomenon.** Averaging (linear, contractive, single fixed point) cannot produce it; genuine feedback with
saturation can. It also sharpens the construct's claim — "held-superposition where interfaces carry structured
decoherence" is precisely bistability of the settle on the contested subspace, an emergent property of
stability-seeking relaxation, not a bolted-on branch-tracker.

## Status
The theoretically-correct fluid exists as a separate, verified module and meets all three acceptance criteria,
with the one criterion that linear feedback *cannot* meet (robust held-superposition) identified and met by the
necessary nonlinearity. Next (later, deliberate): ask what it is useful for — wiring the operators to real
frozen models (each model's read/write frame from its encoder), and whether real convergent models ever produce
the frame-conflict needed for non-trivial routing (expected: rarely, per the convergence findings — but now
that is a question we can actually pose to the *correct* mechanism, not to averaging).
