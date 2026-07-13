# EBR-v1 stage-0 — build + honest gate verdicts

A real, tested implementation of the Equilibrium Barycentric Router core, run through the gates **in the
spec's order** (§11) on **known-degree substrate traffic** — because the falsifiable headline (*active anchor
count = McMillan degree of the traffic*) cannot be checked against opaque models whose traffic degree is
unknown; the spec's own positive-control logic requires ground truth.

## What is built and verified

| module | status |
|---|---|
| `geometry/` | invariant interface: cloud → `(D, w)`, median-normalized. **Gauge-exact** (Δ ≈ 3e-15 relative under the G0 scramble group). |
| `transport/` | square-loss entropic **semi-relaxed** GW; proximal warm-started steps with **backtracking so `F_ve` is monotone** (Lyapunov). |
| `energy/` | shared-anchor block-coordinate equilibration (π / D_e-barycenter / a blocks) with a **Lyapunov guard** → `F` non-increasing 100% of prompts. |
| `hankel/` | residual block-Hankel (§6): coordinate-free Gram (double-centering), anchor deflation, z-scored moments, spectrum vs floor. Reuses validated `io_trace` code. |
| `events/`, `router/` | scaffolded (growth pressure / DeepSets amortizer); not exercised in this run. |
| `registry/` | append-only ledger + preflight refusing moved frozen constants (§10). |
| `tests/` | 5 CI invariants **pass**: gauge-scramble, coupling-monotone, equilibration-Lyapunov, coupling-continuity, invariant-interface. |

## Gate verdicts

**G0 (phase zero) — PASS.** gauge-scramble worst relative Δ = 2.7e-15; floor φ_H = 11.8; positive control
fires (top 19.4 > φ_H); φ_solver restart-std 0.11 ≪ median cost 1.13 → **meter calibratable**.

**Lyapunov — PASS.** With the backtracking guard on the D_e/a block, `F` is non-increasing on 100% of
equilibrations (was ~67% before the guard — the anchor renormalization breaks naive monotonicity).

**G1 (double dissociation) — SPLIT, and honestly so.**
- *K-invariance leg — PASS.* Residual rank is flat in K (K=2 → 1, K=5 → 1) at fixed diversity. This is the
  load-bearing win: the **bare** relational-moment detector inflates with K (2.2 → 3.2 as K goes 2 → 7);
  routing the K members through **one shared anchor** removes that inflation. The pooling mechanism does its
  job on its most important axis.
- *Diversity leg — FAIL.* Residual rank does not cleanly track latent diversity (r = 2,3,4 → rank 2,1,2;
  not monotone). Diagnosis = **interface collapse** (§13): relational Gram-moments through nonlinear ports
  expose the latent diversity only as a lower bound, and the moments are a symmetric-square (≥ quadratic)
  function of the latent, so the McMillan degree of the *moment series* is not the latent degree. This is
  the exact situation §12 pre-registers a response to: **"first check interface collapse (add ONE pre-logits
  tap) before any redesign."** That tap is the next step, not a redesign.

**G2 (running invariance) — PASS.** Mid-run scramble of a member moves the residual moments by < 1e-6
(relative) — it follows from the gauge-exact interface.

**G3 / G4 / G5 — not run (compute).** G4 (meter validity: clones below φ_cyc, disjoint-finetuned above) is
the natural next cheap gate and needs only the 2-edge cycle-cost path; G5 (router amortization) needs the
`router/` implicit-diff loop.

## Honest headline

The machinery is **correct where it can be proven** (gauge-exact, Lyapunov-monotone, meter calibratable) and
the central architectural claim — **K-invariance via shared-anchor pooling** — **holds**. The falsifiable
self-sizing headline (*rank = latent diversity*) **does not cleanly validate through the relational-moment
interface**, diagnosed as interface collapse with a pre-registered fix (pre-logits tap). Reporting this split
verdict, rather than tuning until the diversity leg looks monotone, is the spec's discipline (§12: "sized to
kill cheaply"; §13: "rank is a lower bound").

## Stage-0b — pre-registered retests (predictions in `PREREG_stage0b.md`, committed first)

**P1 — sym-power theorem: PASS, exactly.** For a degree-r linear latent, the invariant (relational,
≥ quadratic) observable's McMillan degree equals the **symmetric-power degree**, not the latent degree:
measured Hankel ranks are **linear {2,3,4} = r**, **quadratic {3,6,10} = r(r+1)/2**, **lin+quad {5,9,14} =
r(r+3)/2**, matching the exact distinct-Koopman-mode counts. The manager's theorem-shaped correction is
confirmed to the integer. Decoder: r = (−1+√(1+8·rank))/2. (`experiments/sympower.py`.)

**Key finding — the theorem is a DETERMINISTIC/Koopman result and does NOT transfer to the stochastic
detector.** P1 uses a *data* Hankel on an autonomous trajectory. The EBR instrument reads *stochastic*
prompt-time traffic via a *covariance* (stochastic-realization) Hankel. Bridge test — a stochastic degree-r
latent with a purely quadratic observable, no model — gives covariance-Hankel rank **non-monotone, even
decreasing** (r=2,3,4 → 5,3,3 at T=6000), NOT r(r+1)/2. So in the stochastic-realization setting the
invariant-rank ↔ diversity map is **neither r nor the deterministic sym-power degree** — it is currently
unresolved. This is a *sharper* correction than under-resolution: it is a realization-theory distinction, not
a data-budget one.

**P2 — corrected diversity leg: FAIL, and now diagnosed deeper.** Single model, no anchor, pre-logits linear
tap (§12), T=1200: invariant rank [2,0,2] — not monotone. Neither the tap nor 2× T fixes it, consistent with
the bridge finding above (the failure is stochastic-realization, not resolution or interface-collapse alone).
Registry: [proven-negative] "raw invariant rank = latent degree" stays closed; [candidate] "invariant rank =
deterministic sym-power degree" is **confirmed for deterministic traffic (P1) but refuted for the stochastic
covariance detector (bridge)** — a new open question is filed: the stochastic invariant-rank ↔ diversity map.

**P3 — K-invariance with HETEROGENEOUS models: PARTIAL.** Diverse architectures (tanh / deep-ReLU / quadratic
/ Fourier / linear), r=3: residual rank **1, 3, 3** across K=2,3,5. It **saturates** (flat for K≥3 — vastly
better than the bare detector's unbounded 2.2→3.2 growth) but shows a low-K transient and does not meet the
pre-registered flat-spread-≤1 criterion. Honest read: pooling genuinely diverse models is harder than pooling
near-clones; the anchor removes most but not all K-dependence.

**P4 / G4 — meter validity: PASS.** Cycle-cost holonomy separates clones from disjoint members decisively:
clone 0.056 < floor (clone + 3σ solver) 0.320 < disjoint 1.141 — a **20.4× separation**. The disagreement
meter is valid on the two-edge topology. (`experiments/g4_meter.py`.)

### Amendments this queues for spec v1.1
1. **Sym-power decoder** — report latent degree via r = (−1+√(1+8·rank))/2, but ONLY in the deterministic
   regime; the stochastic regime needs its own (currently unknown) map. Restate the headline as "active rank =
   McMillan degree of the invariant-observable series," with the deterministic decoder and the stochastic map
   flagged open.
2. **Lyapunov guards normative** — the raw mirror/barycenter steps overshoot; the backtracking guard (67% →
   100% monotone) is part of the definition of the block updates, not an implementation detail.
3. **Frozen-anchor sweep rule** — anchor capacity identical and frozen across diversity cells.

## Reproduce

```
pip install numpy pot torch torchvision pytest
python -m pytest ebr/tests -q            # 5 CI invariants
python -m ebr.phase_zero                 # G0
python -m ebr.experiments.stage0         # G0 + G1 + G2 (compute-heavy; trim seeds/T for speed)
```
Numbers above are from `ebr/experiments/g1_probe.py` (T=140, m0=1, seed 0) and `ebr/phase_zero.py`.
