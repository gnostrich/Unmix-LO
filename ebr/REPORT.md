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

## Reproduce

```
pip install numpy pot torch torchvision pytest
python -m pytest ebr/tests -q            # 5 CI invariants
python -m ebr.phase_zero                 # G0
python -m ebr.experiments.stage0         # G0 + G1 + G2 (compute-heavy; trim seeds/T for speed)
```
Numbers above are from `ebr/experiments/g1_probe.py` (T=140, m0=1, seed 0) and `ebr/phase_zero.py`.
