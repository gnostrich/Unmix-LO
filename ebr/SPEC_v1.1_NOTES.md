# Spec v1.1 — architecture amendments (the "derives from F?" audit)

The v1 spec is ~90% derived from the single functional, ~10% duct-tape — and the duct-tape sat on the
candidate-original element (self-sizing). These amendments remove it.

## #1 — Structural events derive from F (IMPLEMENTED, `events/frankwolfe.py`)
v1's §3 fired growth off a **second statistic** (residual Hankel σ_{m+1} vs φ_H) — a parallel decision channel
that can disagree with F. Replaced by a **conditional-gradient (Frank–Wolfe) step on the anchor measure**:
propose the atom = dominant unexplained residual direction (linearized-F steepest descent on support space),
accept iff re-equilibrated F strictly decreases net of the τ mass-creation cost already in F. Grow / park /
revive / merge are then ONE move: support adaptation of an unbalanced measure under the same descent.

**Validated:** self-quenching (each accepted atom strictly lowers F: e.g. 10.1→6.2→4.0, then stops); the
Hankel is never consulted by the mechanism. **K-invariant self-sizing: 3,3,3 atoms across K=2,3,5** (cleaner
than the Hankel readout's 1,3,3).

**Mechanism/instrument separation (the payoff).** Anchor count (F/FW) = spatial complexity of the shared
geometry, K-invariant. The traffic's temporal McMillan degree = the **pole instrument** (P5, multiplicative
closure). These are different quantities; v1 conflated them. Consequences: (a) events no longer wait on the
stochastic degree law — they never read the broken statistic; (b) the Hankel/poles gate CLAIMS, never drive
MACHINERY; (c) the old rank-based diversity leg was a category error (spatial vs temporal).

## #2 — Pressure accumulator deleted
λ (growth leak), the P>1 capacity constant, and the 10× parking clock do not derive from F. Temporal
persistence is already intrinsic: ā/B̄ Polyak averages + τ/η KL-tethers mean a transient cannot move the slow
state enough to make a support change F-profitable. The grow/park timescale asymmetry survives as the
creation-vs-annihilation cost asymmetry in the unbalanced term — ONE pre-registered constant, not three.

## #3 — L3 out of the loss
v1 trained θ partly on held-out predictivity while G5 gated on it (training on the exam). L3 leaves the loss;
it survives only as G5's held-out metric. Router trains on the amortization gap alone (the intrinsic term).

## Borderline — accepted with honesty flags
- **Lyapunov is ENFORCED, not derived.** The raw mirror / barycenter steps overshoot; the backtracking guard
  (67%→100% monotone) is now part of the *definition* of the B/a block updates. The "everything is one
  I-projection" claim is aspirational at the B/a blocks — the spec must say so.
- Freeze rule, φ_solver subtraction: experimental controls, extrinsic to the system, correctly so.
- Two-edge stage-0 topology: labeled scaffolding the events should eventually discover.
- Revival null: with revival now an F-comparison (#1), the separate null is belt-and-suspenders — keep only as
  a phase-zero calibration of the linearization oracle, not a runtime second criterion.
- Sym-power decoder: retired to the deterministic regime; the pole readout (P5) replaces it in the stochastic
  regime.

## Full amendment list (six)
1. Structural events = Frank–Wolfe on the anchor measure (one authority). **[done]**
2. Pressure accumulator + its three constants deleted; asymmetry = one unbalanced-cost constant. **[done in FW]**
3. L3 removed from the loss; survives as G5 metric only.
4. Lyapunov guards normative; "one projection" flagged aspirational at B/a blocks.
5. §6 deterministic (data-Hankel) vs stochastic (covariance-Hankel) distinction explicit.
6. §6 readout = pole-estimation with predicted-resolvable subset; rank a summary statistic, never the gate.
