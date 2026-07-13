# EBR claims ledger

Status tags: [proven] · [proven-negative] (closed, do not re-litigate) · [candidate] · [open] · [partial].

## Corrections (the wall was information — write it, don't absorb it silently)
- [proven-negative, original form] **Candidate-original element (a): "anchor count = McMillan degree of
  traffic."** The build separated two quantities this claim conflated: **atom count** = spatial complexity of
  the shared per-prompt geometry (F/FW mechanism), and **pole count** = temporal McMillan degree of the
  traffic across prompts (pole instrument, P5). Corrected form: "(a1) atom count self-sizes to the shared
  geometry's spatial complexity, K-invariant [validated 3,3,3]; (a2) the pole instrument reports the traffic's
  temporal McMillan degree via multiplicative closure [P5 proven]." The v1 single-number claim is retired.

## Gate decision (mid-July go/no-go — recorded, not drifted)
- **EBR is the NeurReps instance.** Both load-bearing candidate-original elements are empirically legged:
  F-driven self-sizing with clean K-invariance (3,3,3) and the holonomy meter at 20.4× separation; plus one
  derived law (P5) and one exact theorem (P1). The aggregator is repositioned as the **decentralized-training
  application** of the same validated core (E_B + self-sizing + gauge-invariant interface), not a competing
  instance. The corrected-G1 2×2 (below) is the submission spine — pre-registered, runs next.

## Instrument / theory
- [proven] **P1 sym-power law (deterministic/Koopman regime).** Invariant (relational, ≥quadratic)
  observable's McMillan degree = symmetric-power degree of the latent, EXACT integers: linear r={2,3,4},
  quadratic r(r+1)/2={3,6,10}, lin+quad r(r+3)/2={5,9,14}. Decoder r=(−1+√(1+8·rank))/2. **Deterministic
  regime only** (data-Hankel on autonomous trajectory).
- [proven-negative] **Sym-power law transfers to the covariance-Hankel (stochastic) regime.** Bridge test:
  stochastic degree-r latent, quadratic observable, no model → covariance-Hankel rank non-monotone/decreasing
  (r=2,3,4 → 5,3,3 at T=6000), not r(r+1)/2. Closed. The *rank readout* is the fragile part.
- [proven] **Multiplicative closure (Wick law), P5.** For a linear-Gaussian degree-r latent with poles {λ_i},
  the quadratic observable's covariance modes are the pairwise products {λ_iλ_j} (Isserlis). Estimated poles
  lie on the product set to <0.02 up to the resolvable order; resolved in |·| order (top products first,
  monotone in T); generators recovered as multiplicative square-roots (λ1≈0.855/0.85, |λ2|≈0.652/0.65). The
  [5,3,3] shadow IS the resolvability shadow of the product-pole law. **Read poles, not rank.**
- [resolved] **Stochastic invariant-rank ↔ latent-diversity map.** Closed: the readout is the pole set with a
  floor-aware predicted-resolvable subset; rank demoted to a summary statistic. Diversity leg rebuildable on
  pole closure (next).

## Architecture / mechanism (both load-bearing candidate-original elements now have empirical legs)
- [proven] **Self-sizing via F-driven Frank–Wolfe growth (v1.1 #1).** Structural events re-derived as a
  conditional-gradient step on the anchor measure — one authority (F), no second statistic. Self-quenching
  (each accepted atom strictly lowers F, then stops); Hankel never consulted by the mechanism. **K-invariant
  self-sizing: 3,3,3 atoms across K=2,3,5** with heterogeneous members (cleaner than the Hankel readout's
  1,3,3). Anchor count = spatial complexity of the shared geometry (K-invariant), cleanly SEPARATED from the
  traffic's temporal McMillan degree (the pole instrument, P5). `events/frankwolfe.py`.
- [superseded] The earlier residual-rank self-sizing (1,3,3 partial) — the atom-count-as-temporal-degree
  reading was a category error; resolved by #1's mechanism/instrument split.
- [passed] **Cycle-cost holonomy meter (G4).** Clone 0.056 < floor (clone+3σ) 0.320 < disjoint 1.141,
  **20.4× separation** with a real solver floor. The disagreement meter is validated as an instrument.

## Spec v1.1 amendment queue (five)
1. Sym-power decoder — tag **deterministic-regime-only**.
2. Lyapunov backtracking guards — **normative** (67%→100% monotone is the evidence).
3. Frozen-anchor sweep rule (identical capacity across diversity cells).
4. §6: **deterministic (data-Hankel) vs stochastic (covariance-Hankel)** distinction made explicit.
5. §6 readout: **pole-estimation with predicted-resolvable-subset**, rank kept only as a summary, never the
   gate quantity.
