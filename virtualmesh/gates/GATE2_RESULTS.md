# G2-real RESULTS — MZ kernel on real-representation settling dynamics: **PASS** (scoped)

Run 2026-07-08 per REAL_PREREG.md. Nodes = (model, layer) hidden-state spaces of the four
specialists + frozen base (layers 10/16/22), embedding 240 shared probes, PCA to d=40/node;
channels = spectrally-capped ridge maps fitted on half the probes; damped coupled settling to
T=40 on the held-out half. Full numbers in real/gate2_results.json.

## Pre-registered measurements

| measure | result | criterion |
|---|---|---|
| closure error vs memory length | L=1: 0.084 -> L=2: **0.000** (exact) | some L<=8 < 0.15 ✓ |
| memory necessity | Markovian 0.084 vs L=2 exact | memory helps ✓ |
| kernel eff-rank vs routed width K | 80 / 160 / 240 for K=2/4/6 | grows with K ✓ |
| kernel eff-rank vs federation size N | **120 / 120 / 120 / 120** for N=4/6/8/10 | flat ±2 ✓ |
| residual-difficulty correlation | -0.07 (sandbox: +0.29) | reported, not gating |

**Verdict: PASS on all pre-registered criteria.** The settling dynamics of a real-geometry
federation are closable from a short history of the routed subset alone, and the closure's
complexity does not grow with federation size — the scale-invariance claim the "innumerable
models" story needs.

## Honest scope limits (stated before anyone asks)

1. **Rank-at-cap**: eff-rank equals the output-dimension cap K·d in every configuration. So
   "rank grows with K" is partly dimensional necessity, and the STRONGER atomicity reading —
   a compressed index with rank << K·d — is NOT demonstrated. What is demonstrated is exactly
   the pre-registered pair: short-memory closability + N-independence.
2. **Linear instantiation**: per prereg, the coupled dynamics use linear (ridge) channels over
   real model geometry with linear damping. The geometry is real; the dynamics are not
   nonlinear. gates/README.md's fully nonlinear settling lived in G1's text-space protocol,
   which failed upstream (hallucination cascade) before any kernel question could be posed.
   A nonlinear-dynamics G2 remains untested.
3. The repair-trigger signal (residual ~ difficulty) did not replicate on real geometry
   (-0.07). The corresponding spec clause (G2-C bonus) stays unpromoted.

## Consequence for the merge
Promote to spec/paper in the SCOPED form: "settling over frame-aligned real representation
spaces with linear channels admits an exact short-memory MZ closure over the routed subset,
with closure complexity independent of federation size (N=4..10, 15 real nodes available)."
Do not promote: atomicity-as-compression (rank << cap), residual-as-repair-trigger, or any
claim about nonlinear settling dynamics.
