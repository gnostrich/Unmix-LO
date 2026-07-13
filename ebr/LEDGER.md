# EBR claims ledger

Status tags: [proven] · [proven-negative] (closed, do not re-litigate) · [candidate] · [open] · [partial].

## Instrument / theory
- [proven] **P1 sym-power law (deterministic/Koopman regime).** Invariant (relational, ≥quadratic)
  observable's McMillan degree = symmetric-power degree of the latent, EXACT integers: linear r={2,3,4},
  quadratic r(r+1)/2={3,6,10}, lin+quad r(r+3)/2={5,9,14}. Decoder r=(−1+√(1+8·rank))/2. **Deterministic
  regime only** (data-Hankel on autonomous trajectory).
- [proven-negative] **Sym-power law transfers to the covariance-Hankel (stochastic) regime.** Bridge test:
  stochastic degree-r latent, quadratic observable, no model → covariance-Hankel rank non-monotone/decreasing
  (r=2,3,4 → 5,3,3 at T=6000), not r(r+1)/2. Closed. The *rank readout* is the fragile part.
- [candidate → testing as P5] **Multiplicative closure (Wick law).** For a linear-Gaussian degree-r latent
  with poles {λ_i}, the quadratic observable's covariance modes are the pairwise products {λ_iλ_j} (Isserlis).
  Mode count ~r(r+1)/2 but magnitudes are products (faster decay, smaller residue) → resolvable subset is
  pole-geometry- and floor-dependent. This *predicts* the [5,3,3] shadow as the resolvability shadow of the
  product-pole law. Read POLES, not rank (ERA / Ho-Kalman). Pre-registered in PREREG_P5.md.
- [open] **Stochastic invariant-rank ↔ latent-diversity map.** Superseded in practice by the pole readout if
  P5 holds; rank retires to a summary statistic.

## Architecture / mechanism (both load-bearing candidate-original elements now have empirical legs)
- [partial] **Self-sizing via K-invariant pooling.** Homogeneous members: residual rank flat at 1 across K
  (bare detector inflates 2.2→3.2). Heterogeneous members (P3): rank 1,3,3 over K=2,3,5 — **saturates** for
  K≥3 but low-K transient; misses flat-≤1. Saturation ≠ flatness logged as a real nuance. (Hypothesis to test
  under the pole readout: the low-K transient is a resolvability artifact, not a pooling failure.)
- [passed] **Cycle-cost holonomy meter (G4).** Clone 0.056 < floor (clone+3σ) 0.320 < disjoint 1.141,
  **20.4× separation** with a real solver floor. The disagreement meter is validated as an instrument.

## Spec v1.1 amendment queue (five)
1. Sym-power decoder — tag **deterministic-regime-only**.
2. Lyapunov backtracking guards — **normative** (67%→100% monotone is the evidence).
3. Frozen-anchor sweep rule (identical capacity across diversity cells).
4. §6: **deterministic (data-Hankel) vs stochastic (covariance-Hankel)** distinction made explicit.
5. §6 readout: **pole-estimation with predicted-resolvable-subset**, rank kept only as a summary, never the
   gate quantity.
