# RESULTS — orbit: operator + MZ-kernel instrument over the physics corpus (honest, local)

Prereg frozen before run code (commit `cceab00`). Built in the mold the repo pointer ("The Basin") made
concrete: an **instrument** (atlas + transfer operator + spectral macros + Mori–Zwanzig memory kernel) over a
real-coupling **corpus**, **orbited** into a **rendered artifact**, decided against a **κ=0 ablation**. Same
MZ-kernel machinery as `mz_aggregator`; real substrate; artifact at the end. numpy/scipy only. Reproduce:
`python run.py`. Local object — no cross-thread reconciliation.

## Poles-first calibration — the kernel is validated
The memory kernel (autocorrelation of the resolved slow coordinate → damped-oscillator fit, CV order ≤ 3) is
the load-bearing adjudicator; validate it before trusting any real number.

| calibration corpus | expected | measured ω | outcome |
|---|---|---|---|
| **null** (i.i.d. random states, no dynamics) | no oscillatory mode | **0.000** (γ=1.9, fast decay) | ✓ no structure |
| **periodic** (circular motion, period 25) | ω ≈ 2π/25 = 0.251 | **0.253**, autocorr cleanly oscillates (1.0 → −0.52 at half-period → +1.0 at lag 25) | ✓ period recovered |

**Kernel calibration PASSES**: it finds an oscillatory slow mode exactly when the corpus has one observable in
its state, and finds none in noise. (The spectral-gap *count* detector is unreliable on these near-degenerate
operators — it flags "no clean gap" everywhere — so it is **not** relied on; ω-recovery is the validated dial.)

## Real substrate — the κ=0 ablation → (b) NULL

| corpus | ω (slow mode) | κ=0 miss | κ=0.4 miss | margin |
|---|---|---|---|---|
| **driven** (fast collisions + slow gravity modulation, period 30) | 0.044 | 4.925 ± 0.105 | 4.813 ± 0.177 | **+0.023** |
| **plain billiards** | 0.029 | 4.659 ± 0.192 | 4.546 ± 0.167 | **+0.024** |

`miss` = continuation-miss: how far each emitted state is from the true physical successor of the previous,
in units of the mean step size (lower = the orbit follows real dynamics). κ = memory strength; **κ=0 is the
memoryless P-walk** (bit-identical rng); the tilt scale is corpus-calibrated to the operator's own logit
spread, so κ *can* steer.

**Verdict: (b) NULL.** On both physics corpora the MZ momentum ties the κ=0 ablation within noise (margin
+0.02 vs σ≈0.02–0.04 — a marginal inertia effect, not beyond noise). The cause is structural, and it is the
whole point: **ω ≈ 0 on the physics corpus** — there is no *state-observable phrase-scale mode* for the memory
to exploit. The driven corpus *has* a real slow mode (the gravity phase), but that phase is exogenous time,
**not recoverable from an instantaneous `(pos,vel)` state under a time-homogeneous operator**, so the resolved
coordinate's autocorrelation just decays monotonically and the kernel finds no oscillation. Billiards is
Markovian/chaotic; its slow structure isn't in the instantaneous state the way a music phrase is in its audio.

## The dissociation (why this is a finding, not a broken instrument)
Same instrument, three corpora: **periodic → ω recovered** (0.253≈0.251, oscillating autocorr); **null → ω=0**;
**physics → ω≈0** (monotone-decaying autocorr). The machinery detects phrase-scale memory precisely when it is
observable in the corpus's state, and correctly reports its absence otherwise. The **κ=0 ablation is the
adjudicator** and it ties — honestly — because there is nothing for the memory to carry.

## The artifact
`artifact_kappa0.png` / `artifact_kappa04.png` (+ GIFs): the orbit rendered as a "set" of physics frames
(green = current positions, red ghost = velocity). Every frame is on-manifold (concatenative emission), but the
*sequence* is jumpy (miss ≈ 4.9× a physical step) and κ=0 vs κ=0.4 look alike — the visible signature of the
(b) null: no phrase-scale memory to smooth the transitions.

## This is the Basin lesson, ported and confirmed on my own substrate
The Basin earns its keep because a music corpus's slow structure (rhythm → phrase → arc) is **real and
observable** in its features, so its MZ kernel finds oscillatory modes and momentum can steer. Ported here: the
identical machinery, validated on an oscillatory control, **finds nothing on mechanical physics** because the
slow coupling isn't observable in the instantaneous state. The kernel earns its keep **only where the slow
coupling is real and observable** — the exact discipline the pointer was for. Honest (b) is the outcome.

## Scope
Local: an instrument-over-a-corpus with a κ=0 ablation and a rendered artifact. Not the resolvent conjecture,
not the auction, not the trace claim. The next move that could flip (b)→(a) is a substrate whose slow mode is
state-observable (window the features so exogenous phase becomes visible, or a corpus with genuine intrinsic
slow structure) — deferred, not faked.
