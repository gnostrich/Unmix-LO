# EBR-v1.1 — Equilibrium Barycentric Router (supersedes v1)

Supersedes ebr-spec-v1. Rewritten under the single-authority principle: the whole system is descent on ONE
functional F, and everything else is instrument, oracle, or experimental control. Every constant that survives
is listed with its derivation (§10); anything without one was deleted, not defended.

## 0. Single-authority principle (standing law)

There is ONE decision-maker: the functional **F**. Every behaviour — coupling, gain, mass, support, topology —
is an F-descent move. Everything else is exactly one of:
- **INSTRUMENT** — measures and reports (Hankel spectra, poles, cycle costs, rank summaries, floors, gates).
  May gate CLAIMS and experiments. **May never drive mechanism.**
- **ORACLE / ACCELERATOR** — proposes or warm-starts (FW proposal directions, router inits). May be any
  heuristic, because acceptance is always strict F-descent. Affects speed, never outcomes.
- **EXPERIMENTAL CONTROL** — freeze windows, φ_solver subtraction, nulls, pre-registration. Extrinsic to the
  system, intrinsic to the epistemics. Lives in `experiments/`, never in `energy/`.

Litmus (run on every design decision): (1) can it DISAGREE with F about what the system should do? → forbidden.
(2) does each constant derive from F, a measured null, or a pre-registered choice? if none → shim, re-derive.
(3) if deleted, does the system misbehave (mechanism) or do we just stop knowing something (instrument)?
mislabeling is a violation. (4) does the prose claim more structure than the code instantiates? → fix prose.

## 1. The functional (data-derived tethers ONLY — no learned quantity in F)

F(π, B, a; x) = Σ_e Σ_{v∈m(e)} ⟨GW(D_v, D_e), π_{v,e}⟩        (transport)
             + ε Σ KL(π ‖ w_v ⊗ a_e)                          (entropy)
             + τ Σ_e KL(a_e ‖ ā_e)                            (mass plasticity; ā = Polyak of past equilibria)
             + γ Σ_overlaps KL(composite ‖ direct)            (gluing)
             + η Σ KL(B_{v,e} ‖ B̄_{v,e})                      (gain inertia; **B̄ = Polyak of past equilibrium
                                                               B's — DATA-derived, NOT R_θ**)

**FIX-1:** the v1 term η·KL(B ‖ R_θ(ι)) is REMOVED. A learned router inside F is a second authority. B is
tethered only to a data-derived slow reference B̄. The router R_θ is an ORACLE (warm starts only, §4); its
amortization gap is MEASURED, never optimized through the equilibrium.

**Optimization = block-coordinate MIRROR DESCENT on F with backtracking (FIX-3, honest).** The blocks are NOT
exact I-projections; the raw mirror/barycenter steps overshoot (measured 67% monotone → 100% with a
backtracking line search on every block). Guaranteed: monotone descent. Not guaranteed: exactness. F is the
Lyapunov function by construction of the guard, not by projection identity.

## 2. Structural mechanism = Frank–Wolfe support adaptation (the ONLY structural authority)

The anchor is a free-support unbalanced measure. Every structural move is a conditional-gradient (FW) step on
it under the SAME F:
- **atoms (grow / park / revive) — IMPLEMENTED, VALIDATED.** Oracle proposes the atom (dominant unexplained
  residual direction — any heuristic); accept iff re-equilibrated F strictly decreases net of the τ
  mass-creation cost. Self-quenching (each accepted atom strictly lowers F, then stops). The Hankel is NEVER
  consulted by the mechanism. Park = unbalanced a-block driving mass→0; grow/park asymmetry = the
  creation-vs-annihilation cost asymmetry in the unbalanced term (ONE constant, not a separate clock — the v1
  pressure accumulator, λ, P>1, and 10× clock are DELETED).
- **hyperedges (spawn / merge) — DERIVED, NOT YET IMPLEMENTED (FIX-2).** Same move on a level-2 measure over
  port-subsets: oracle = residual co-clustering proposes a subset U; accept iff instantiating its sub-anchor
  Z_U (with the γ gluing term) strictly decreases F net of Z_U's creation cost. No wall found. Until built,
  the two-edge topology is a LABELED experimental fixture (`experiments/g4_meter.py`), not discovered
  structure; "spawn/merge are one move" is NOT claimed of the code.

## 3. Mechanism / instrument split (first-class — corrects v1's central conflation)

v1's candidate-original element (a), "active anchor count = McMillan degree of traffic," conflated two
DIFFERENT quantities. Recorded as a corrected error, not silently rewritten:
- **atom count** = spatial complexity of the shared per-prompt geometry. MECHANISM-side (F/FW). K-invariant
  [validated: 3,3,3 atoms across K=2,3,5, heterogeneous members].
- **pole set** = temporal McMillan degree of the traffic across prompts. INSTRUMENT-side (covariance-Hankel /
  ERA). Read as POLES via multiplicative closure [P5 proven: poles lie on {λ_iλ_j} to <0.02; generators
  recovered]. Rank is a SUMMARY statistic only, never a gate quantity.

## 4. Router = pure amortizer (oracle)

DeepSets over per-port instrument blocks → per-(v,e) warm-start B̂ and Sinkhorn potentials. Trained on the
MEASURED amortization gap KL(B_final ‖ R_θ). L3 predictivity is OUT of all losses (survives as the G5 metric
only). Dual-estimator check (implicit vs unrolled gradients) is a halt condition. R_θ contributes ZERO terms
to F; deleting it changes speed, never the equilibrium.

## 5. Corrected G1 — the 2×2 selective dissociation (submission spine)

| knob ↑ | atom count (mechanism) | pole set (instrument) |
|---|---|---|
| within-prompt geometric richness | grows | flat |
| across-prompt dynamical diversity | flat | grows |
| K (members) | flat [3,3,3 done] | flat (to check) |

Each instrument selectively sensitive to its own axis. Pre-registered to the digit (predicted atom counts from
geometry; predicted resolvable pole subset from Wick residues + measured floor) before running.

## 6. Instruments (gate claims; never mechanism)

Residual block-Hankel (§6 of v1) DEMOTED to instrument. Readout is **pole estimation with a floor-aware
predicted-resolvable subset**; rank is a summary only. **Deterministic (data-Hankel) vs stochastic
(covariance-Hankel) regimes are distinct**: the sym-power rank decoder [P1] is a DETERMINISTIC-regime
instrument; the stochastic regime uses pole closure [P5]. Moment channels are z-scored — a fixed, declared
instrument choice with its own scramble-invariance CI test. Gauge/scramble group includes **shift** (code
tests it). Cycle-cost holonomy meter validated [G4, 20.4× separation].

## 8-11. Calibration, gates, discipline (carried from v1)

Phase zero (G0) gauge-scramble + floors (φ_H, φ_F, φ_cyc, φ_solver) + positive control + revival null —
**revival null demoted to phase-zero calibration of the FW oracle; runtime revival is F-descent alone**. Gates
in order G0→G5; commit-before-run; append-only registry; freeze rule during measurement.

## 10. Constants — each with its derivation (no undefended constant survives)

| constant | value | derivation |
|---|---|---|
| ε (entropy) | grid @ phase-zero, frozen | pre-registered (positive-control × φ_solver) |
| τ (mass plasticity) | pre-registered | sets creation/annihilation asymmetry (ONE constant, replaces λ/P/clock) |
| γ (gluing), η (gain inertia) | 1 | pre-registered |
| j_sink, T_max, W, L, k_max | 5, 20, 50, 12, 6 | pre-registered |
| floors φ_H, φ_F, φ_cyc, φ_solver | measured | null-ensemble 95th percentile |
| **DELETED** | λ, P>1 capacity, 10× parking clock | no F-derivation → removed (persistence is intrinsic via ā/τ) |

## Changelog v1 → v1.1
FIX-1 router→oracle (no learned term in F). FIX-2 hyperedges derived (impl pending; fixture labeled).
FIX-3 Lyapunov honest (mirror descent + backtracking, not I-projections). FIX-4 z-score declared + tested,
shift in scramble group, revival null → phase-zero, sym-power decoder deterministic-only. Pressure accumulator
+ 3 constants deleted. Candidate-(a) conflation corrected via the mechanism/instrument split. G1 → 2×2.
