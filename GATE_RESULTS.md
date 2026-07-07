# GATE RESULTS — real-gradient decision experiment (2026-07-07)

**Verdict: RED. Real per-task gradients do not contain stable + individual + reused operator
components at the scale tested. Recommendation: pivot to the robustness reframe
(non-destructive learned merge under worker heterogeneity). Do not build the compositional
system.**

Pre-committed thresholds from GATE.md / the run spec were applied unmodified; nothing was tuned
after seeing real numbers. Honest negatives were the goal; this is one.

## Setup

Two independent runs, both 9 tasks = 3 genres x 3 tasks (code: sklearn/numpy/scipy source;
math: arithmetic/algebra/sequence generators; prose: Austen/Aurelius/Melville), minibatch
gradients wrt LoRA-B (rank 8, attention projections, fixed seed => shared comparable space)
at one fixed base checkpoint, batch 4 x 256 tokens:

| run | base model | grads/task | P | pooled n (post-hygiene) |
|---|---|---|---|---|
| primary | Qwen2.5-0.5B-Instruct | 200 | 393,216 | 1,780 |
| secondary | gpt2 (124M) | 120 | 221,184 | 1,080 |

Pipeline: per-cloud norm normalization -> pooled PCA r=100 -> **integrity guard (a)**: regress
out task-generic axes (uniform per-task variance; gpt2: 3 axes = 38% of subspace variance,
Qwen: 0 axes) -> FastICA K=30 -> the three checks. Hygiene: non-finite and >10x-median-norm
rows dropped (Qwen loss-spike minibatches reached |g| ~ 1e12; <3% of rows dropped per task).

## The three numbers (primary run, pre-committed config r=100, K=30)

| check | measured | bar | result |
|---|---|---|---|
| **STABLE** — bootstrap matched cosine | **0.814** | >= 0.8 | **PASS** |
| **INDIVIDUAL** — max pairwise \|cos\| | **0.882** (median 0.075; kurtosis 4.38; smeared 0.03) | < 0.65, kurt > 0, not smeared | **FAIL** |
| **REUSED** — held-out recon residual | **0.582** (10.7/30 active; cross-genre 0.40) | < 0.3, sparse, cross-genre | **FAIL** |

Secondary run (gpt2): STABLE 0.644 FAIL, INDIVIDUAL smeared 0.60 FAIL, REUSED 0.337 FAIL.

## Sensitivity annex (verdict is config-robust)

| config | STABLE | max overlap | REUSED residual | verdict |
|---|---|---|---|---|
| qwen r=100 K=30 (headline) | 0.814 | 0.882 | 0.582 | RED |
| qwen r=100 K=15 | 0.834 | 0.803 | 0.772 | RED |
| qwen r=100 K=50 | 0.719 | 0.899 | 0.483 | RED |
| qwen r=50 K=15 | 0.831 | 0.803 | 0.685 | RED |
| gpt2 r=100 K=30 | 0.644 | 0.663 | 0.337 | RED |
| gpt2 r=100 K=15 | 0.755 | 0.548 | 0.435 | RED |
| gpt2 r=50 K=15 | 0.778 | 0.549 | 0.359 | RED |

The >=0.80 overlapping pair persists at K=15, so it is not an ICA over-specification artifact.
REUSED residual *worsens* as K shrinks — a small library spans held-out tasks even less well.

## Honest reading — what the data actually said

Not a flat null. The failure is specific:

- **Not the Gaussian null.** Loading kurtosis 4.4 (Qwen) / 1.3 (gpt2) > 0: ICA had traction;
  there is real non-Gaussian structure in gradient space. Guard (b) satisfied.
- **Components are individual-ish and stable on the instruct model**: concentrated on ~2.8/9
  tasks (smearing 0.03), stable at 0.81 — genuine, repeatable task-cluster structure exists.
- **Experiment 05's diversity prediction REPLICATES on real gradients**: extraction stability
  rises as genres are pooled — 0.512 (1 genre) -> 0.657 (2) -> 0.808 (3) on Qwen;
  0.478 -> 0.517 -> 0.642 on gpt2. The diversity solvent works.
- **But the reuse premise fails.** Held-out tasks are NOT sparse combinations of the same
  components (residual 0.48-0.77 across configs vs bar 0.3). What the extractor finds are
  (mostly) per-task-cluster signatures plus at least one stubbornly fused pair — not a shared
  compositional basis that recombines to explain unseen tasks. The world's compositionality,
  at this scale, does not reach into weight space the way the thesis needs.

This is the same shape as experiment 03's negative prior (shared structure thin/generic),
sharpened: with an instruct model and genre diversity you get stable individual structure,
but it is task-cluster identity, not reusable operators.

## Confidence and caveats

Medium confidence in the verdict at this scale; the direction is consistent across two models,
two P's, and seven analysis configs, and REUSED fails by 2x, not marginally. Caveats, per the
compute-limited clause (CPU-only box):

- Base models at/below the spec floor (0.5B instruct + 124M); rank-8 LoRA-B subspace only.
- Minibatch gradients at the base checkpoint (a GATE.md-sanctioned option), not adapter deltas
  after k inner steps; fine-tuned-trainee pseudo-gradients could in principle decouple better.
- Corpora are small slices (package source, synthetic math, Gutenberg), 200 grads/task.
- The rising diversity curve leaves one honest escape hatch: stability was still climbing at
  3 genres. A 1.5B+ run with 5-6 genres and pseudo-gradients is the only follow-up that could
  plausibly flip REUSED; nothing at this scale suggests it would.

## Decision (per the pre-committed rule)

**Any fail -> robustness reframe.** Two of three fail, robustly. The compositional
self-refining optimizer should NOT be built. The valuable salvage:

1. The extractor + gate harness (this repo) — reusable measurement infrastructure.
2. The replicated diversity->stability curve — the multi-environment identifiability mechanism
   is real in weight space, even though the sources it identifies are not reusable primitives.
3. The reframe target: a learned merge/aggregator that preserves worker differences under
   heterogeneity (non-destructive amalgamation), which needs none of the failed premises and
   addresses a real gap in decentralized-training stacks (see README.md decision rule).
