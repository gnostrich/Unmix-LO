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
- [proven] **F-driven Frank–Wolfe growth is a single-authority self-quenching mechanism (v1.1 #1).**
  Structural events re-derived as a conditional-gradient step on the anchor measure — one authority (F), no
  second statistic. Self-quenching (each accepted atom strictly lowers F, then stops); Hankel never consulted.
  `events/frankwolfe.py`.
- [holds] **Atom count is K-invariant** — flat across K=2,3,5, does not inflate with members (the bare
  detector did, 2.2→3.2). A real pooling property.
- [resolved-by-substitution — corrects an earlier over-claim] **Atom count tracks the shared geometry's spatial
  complexity.** WALL_2x2_atomleg.md: across 4 geometric knobs × 3 ε × discrete+continuous readouts, atom count
  is operating-point-dominated (~3–4), does not track any geometric-richness knob. The earlier "3,3,3 = clean
  geometric self-sizing" reading is retracted; the value ~3 is the operating-point floor. K-invariance stands;
  geometric-sizing does not. **Resolved not by rescuing atom count but by substituting the instrument:** the
  equilibrated anchor's **D_e effective rank** IS the spatial-complexity meter (next entry). Atom count stays a
  K-invariant pooling property only.
- [superseded] The earlier residual-rank self-sizing (1,3,3 partial) — the atom-count-as-temporal-degree
  reading was a category error; resolved by #1's mechanism/instrument split.
- [derived — unimplemented] **Hyperedge spawn/merge as level-2 FW.** Same conditional-gradient move on a
  measure over port-subsets: oracle (residual co-clustering) proposes a subset U; accept iff instantiating its
  sub-anchor Z_U with the γ gluing term strictly decreases F net of Z_U's creation cost. No wall found;
  requires multi-edge F + gluing (single-edge today). The cleanest open mechanism thread — once built, the
  "one move" claim becomes true at both levels. Sequenced AFTER the single-edge 2×2.
- [passed] **Cycle-cost holonomy meter (G4).** Clone 0.056 < floor (clone+3σ) 0.320 < disjoint 1.141,
  **20.4× separation** with a real solver floor. The disagreement meter is validated as an instrument.
- [validated — full trust battery] **D_e effective rank is the SPATIAL-complexity instrument.** Participation
  ratio exp(H) of the equilibrated anchor cost's singular values at fixed budget m=12. Survived a
  pre-registered three-leg battery (PREREG_derank_battery.md; `experiments/derank_battery_b{1,2,3}.py`): B1
  scramble gauge-invariant to <3.6e-15; B2 null-floor — monotone in planted rank (Spearman +1.00), structureless
  matched-moment clouds read high (6.36 > rank-2's 4.67), baseline offset O=4.65 explains the 4.67-at-r=2
  reading; B3 operating-point — Spearman(eff-rank, r) = +1.00 at every ε×n cell (the exact sweep that killed the
  atom count). **Honest caveat (recorded, not airbrushed):** the realized dynamic band is COMPRESSED to
  ~[4.65, 6.4], not near m — the usable signal is the ORDER/margin, not the absolute value.
- [partial — one-axis dissociation, HONEST negative] **The 2×2 spatial/temporal dissociation.** Pre-registered
  to the digit (PREREG_2x2_dissociation.md, amended before running); `experiments/twobytwo_dissociation.py`.
  Unified substrate: G independent whitened AR(1) coords carrying D distinct pole values (spatial knob G,
  temporal knob D, both independent). **SPATIAL leg PASS** — D_e eff-rank tracks ONLY geometry: Spearman(D_e,G)
  = +1.00, exact digits land (4.66/5.49/5.82 vs 4.7/5.3/5.9), flat in D (range 0.14) and K (range 0.04). The
  spatial meter is validated end-to-end as a dissociable instrument. **TEMPORAL leg FAIL (as pre-registered,
  reported straight)** — the ERA/P5 pole COUNT is not invariant to spatial multiplicity: at G>D the duplicated
  dominant pole (0.9→product 0.81) saturates the top-order ERA singular subspace, crowding out sub-dominant
  products (0.63, 0.49), so poles(D=2)=1 not 3 and range_G(poles)=2.0 (need ≤1). The prereg's risk section
  anticipated only the D=3 smallest-product under-resolving, not this G>D crowding — an honest, recorded miss.
  **Outcome:** a clean ONE-AXIS dissociation (spatial validated; temporal not). Per covenant, recorded not
  patched: a G-invariant pole-count readout, if pursued, is a genuine design fork needing a FRESH
  pre-registration before any re-run — not a post-hoc order bump.

## Untied cross-modal routing (the dictionary-free thesis) — two pre-registered negatives, one boundary law
- [partial — honest negative with real signal] **Overlap routing (PREREG_overlap_routing.md, run at 316f648).**
  Untied collective F over 10 member-channels (3 modalities; CLIP + SigLIP + physics micro-world as
  independent bridges), NO pairing dictionary. Forward render→text: shape .476 and motion .465 rise well above
  chance BECAUSE of the added overlap (Δ +.198/+.146; motion can only ride the world's dynamics axis — the
  overlap thesis measurably works for these), but color stays near chance (.351, Δ+.052 < the .15 bar) and
  reverse (text→vision) transfers nothing. Registered verdict: the tied dictionary stays load-bearing.
  Gauge survives untied routing (|ΔF| = 1.2e-14).
- [diagnosed + fix falsified] **Color-symmetry cycle (PREREG_color_symmetry.md, both legs run).** D-leg
  CONFIRMED the under-encoding mechanism: ρ_color < ρ_shape on ALL pixel-vision members (ViT 1.16 vs 1.72,
  CLIP 1.22 vs 1.29, SigLIP 1.25 vs 1.44) — color barely structures pixel-vision relational geometry; the
  exchangeability mechanism was NOT confirmed (3/10). F-leg (world v2: color covaries with speed) FAILED its
  registered bars: color 0.319 (Δ −0.032), and the covariance skewed the motion-word distribution enough to
  degrade motion (.465→.337). Registered branch: **covariance-insufficient — deeper wall.**
- **Boundary law (the two negatives made quantitative):** untied relational routing transfers the attributes
  that STRUCTURE members' intrinsic geometries (shape, motion); an attribute relationally under-encoded at the
  members (color in pixel-vision) cannot ride the shared anchor even when handed a covariance route — the tie
  (matched probes) remains the mechanism for exactly those attributes. Candidate next move if ever pursued
  (needs its own prereg): R2 channel-blocking to expose a color-dominant feature-group channel, raising
  ρ_color at the member — attacking the diagnosed under-encoding directly, with spec machinery, not a shim.

## Spec v1.1 amendment queue (five)
1. Sym-power decoder — tag **deterministic-regime-only**.
2. Lyapunov backtracking guards — **normative** (67%→100% monotone is the evidence).
3. Frozen-anchor sweep rule (identical capacity across diversity cells).
4. §6: **deterministic (data-Hankel) vs stochastic (covariance-Hankel)** distinction made explicit.
5. §6 readout: **pole-estimation with predicted-resolvable-subset**, rank kept only as a summary, never the
   gate quantity.
