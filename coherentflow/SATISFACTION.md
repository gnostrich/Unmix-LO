# SATISFACTION BATTERY — against the REAL coherentflow build (internal check, NOT a gate)

Run 2026-07-09 per `coherentflow/SATISFACTION_BRIEF.md`, exercising the **actual committed functions**
(`coherentflow.structured` / `settle` / `combined_read` / `make_interface`), not a reimplementation.
Reproduce: `python coherentflow/run_satisfaction.py` (CPU, seconds). Sandbox stand-ins that shipped with
the brief: `satisfaction_battery_sandbox.py`, `satisfaction_sweep_sandbox.py`.

## The 7 checks — ALL PASS on the real build
| # | check | result | detail |
|---|---|---|---|
| T1 | coverage-union > best-single | **PASS** | union R² 0.494 vs best-single 0.280 |
| T2 | coherent → honest no-op | **PASS** | held 0/3, circ-norm 0.0000 |
| T3 | structured → combined read > consensus | **PASS** | held 1/3, consensus 0.510, combined **1.000** (naive-mean 0.510) |
| T4 | noise → rejected (no G1) | **PASS** | held 0/3, circ-norm 0.0000 |
| T5 | settling contracts (tail-slope, init off fixed point) | **PASS** | res 30.20 → 0.019; tail 0.080 → 0.019 |
| T6 | circulation concentrated (not sprayed) | **PASS** | top-direction energy fraction 0.839 |
| T7 | pure-noise falsification (must NOT hold) | **PASS** | held 0/3 |

## Trust-critical #1 — FALSE-POSITIVE rate (the headline)
**0.0% over 40 seeds** of coherent input (never holds anything, circ-norm 0). The no-fabrication guarantee
holds on the real build: it does **not** invent structure where there is none. This is the strongest and
most important result — it is what makes the object trustworthy. Confirmed to match the sandbox (0%).

## Trust-critical #2 — detection sensitivity + read-payoff (40 seeds/strength)
| injection strength | detect % | payoff vs removed-consensus | payoff vs naive-mean (fair) |
|---|---|---|---|
| 0.5 | 0% | +0.000 | +0.000 |
| 1.0 | 0% | +0.000 | +0.000 |
| 1.2 | 62% | +0.499 | +0.453 |
| 1.4 | 95% | +0.497 | +0.452 |
| 1.6 | 100% | +0.497 | +0.453 |
| 2.0 | 100% | +0.498 | +0.459 |
| 4.0 | 100% | +0.497 | +0.478 |
| 6.0 | 100% | +0.496 | +0.482 |

**Detection floor: sharp, centered ~1.2–1.4× noise scale** (0% below 1.0×, 62% at 1.2×, 95% at 1.4×,
100% by 1.6×).

## Two honest discrepancies with the sandbox — reported, not forced to match
The real build **holds structured decoherence OUT of the consensus** (target = mean(fᵢ − heldᵢ)); the
sandbox **folds it back into the state** (state ← mean + circ). That one design difference explains both
discrepancies, and it means the sandbox *understated* the real object:

1. **Sharper, lower detection floor** — real ~1.2–1.4× vs sandbox ~2–3×. Keeping the branch in the
   disagreement (not absorbing it into the state) makes the held-out structure cleaner to detect.
2. **Much larger read-payoff** — real ~+0.50 (flat) vs sandbox +0.05 shrinking to +0.01. The sandbox's
   small payoff was an artifact of *its own* design degrading the combined channel (structure folded into
   the state leaks into consensus and shrinks the circulated channel). The real build keeps them separate.

**Is the large payoff suspicious?** The brief warned it would be. I checked the suspicion three ways and
it holds up as **real, not a weak-baseline artifact**:
- The **fair naive-mean-of-interfaces baseline also sits at chance** (0.510 in T3; +0.45–0.48 payoff
  against it across the sweep). Plain averaging genuinely cannot recover the minority, alignment-orthogonal
  branch — so beating it is a real capability, not a rigged baseline.
- The read is a **held-out** train/test linear probe (fit on the train half, scored on the test half) — not
  circular.
- The magnitude reflects a **clean, strong, perfectly-reproducible injected signal** in a synthetic regime;
  it is not evidence the object delivers +0.5 anywhere real.

## What this build can and can't be trusted to do (plain English)
This build can be trusted to **not fabricate**: on coherent input and on pure noise it holds nothing (0%
false positives across 40 seeds, T2/T4/T7). When a genuine, concentrated, held-out-predictable distinction
is injected into one interface above ~1.3× the noise scale, it **reliably detects it (≈100% by 1.6×), holds
it in a concentrated tape channel (not sprayed, 0.84 top-dir), and surfaces it in the combined read that
neither the consensus nor a naive average of the interfaces can recover**. Its settling **converges and
never blows up** (T5; fixed point is seed-dependent, not always tightly contractive — reported as-is).

What it **cannot** be trusted to do: deliver that payoff on **real frozen/convergent models**. There the
precondition is not met — the interfaces agree (F_gauge ≈ 0, per xresolve/biomesh/synergy), so the object
correctly **no-ops**. The large synthetic payoff is a property of the injected-structure regime, not a
capability the object conjures from convergent models. In one line: **trustworthy because it refuses to
fabricate; useful exactly when a real held-out distinction is present, and honestly inert when it is not.**
