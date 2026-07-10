# PRE-REGISTRATION — orbit: an operator + MZ-kernel instrument over the physics corpus (frozen before run code)

Resuming the thread in the mold the repo pointer ("The Basin") made concrete: stop probing a frozen model's
reading of the physics (that read NOISE — see `swirl/`); build the **operator + memory-kernel instrument over
the real-coupling corpus itself**, orbit it into a **rendered artifact**, and decide it against a **κ=0
ablation**. Same validated machinery as `mz_aggregator` (Mori–Zwanzig memory kernel), real substrate, artifact
at the end. numpy/scipy only. Local object; no cross-thread grand-conflation.

## The corpus (real coupling, already in-repo)
`engine.py`-style 2D rigid-body physics (5 balls, gravity, elastic wall + ball–ball collisions). Genuine
multiscale coupling (ballistic drift ↔ collision events). Deterministic, renderable. The corpus is a bank of
real trajectories; the instrument is built from it, nothing hand-labeled.

## The instrument (built, not imposed)
1. **Charts (atlas):** window the corpus states, k-means into charts; soft (Gaussian) membership. Chart count
   capped so each chart has enough support (≥ N windows/chart).
2. **Transfer operator P:** row-stochastic chart→chart transition operator estimated from the corpus flow
   (within-trajectory only). Eigen-spectrum + **spectral-gap** macro/basin detection (`|λ_i|/|λ_{i+1}|`; if the
   largest relative drop is not ≥ 1.3× the median, declare "no clear gap" and flag it — K is **measured, not
   imposed**).
3. **Memory kernel (MZ):** autocorrelation of the resolved (slow-macro) coordinate → damped-oscillator fit,
   order chosen by **track-held-out CV** (cap ≤ 3). (γ, ω) come from the fitted modes — nothing hand-picked.

## The orbit (the dynamics → artifact)
A walk on the instrument: step through P; add the momentum tilt `β·ψ·p` with `p ← e^{−γ}p + Δa − ω²a` driven by
the walk's own motion (the measured kernel modes). Emit a concrete state per step; **render** the emitted
trajectory to frames → an artifact (a watchable video/strip of generated physics). `κ` scales the memory tilt;
**κ = 0 exactly reproduces the memoryless walk** (unit-checked).

## Pre-registered outcomes (frozen, Basin-style; the κ=0 ablation is the deciding experiment)
The instrument is validated poles-first before any real verdict: a **null corpus** (i.i.d. random states, no
dynamics) must yield **no clean spectral gap** and a memory kernel consistent with **no structure**; a
**known-periodic corpus** must yield a clean gap and a kernel recovering its period. Only then the real corpus:

- **(a) POSITIVE** — the kernel-on orbit is measurably **more coherent** than the κ=0 ablation on a
  pre-committed objective (generated states stay on the physics manifold / respect its invariants markedly
  better than κ=0), AND the difference is visible in the rendered artifact. → memory earns its keep.
- **(b) NULL** — kernel-on ties κ=0 on the objective (memory adds nothing measurable). Honest no-op.
- **(c) UNSTABLE** — the memory tilt destabilizes the orbit (accumulation/collapse). Report it as a diagnosed
  failure mode (as The Basin honestly did), not a hidden result.

Decision rule: **POSITIVE requires beating the κ=0 ablation on the frozen objective by a real margin AND a
visible artifact difference.** Coordinate-level structure that does NOT translate to a better-looking orbit is
reported as (b)/partial — the Basin's "coordinate success ≠ audible arcs" honesty, ported to the visual artifact.

## Discipline
Frozen PREREG before run code. Poles-first (null = no gap/no structure; periodic positive control) before any
real-corpus verdict. κ=0 ablation is the adjudicator; the artifact is shown. K measured not imposed; kernel
modes corpus-fit, no free knobs. Honest (b)/(c) are success outcomes. Local object — not the resolvent
conjecture, not the auction, not the trace claim; no cross-thread reconciliation.
