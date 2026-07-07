# STABILITY GATE — do enacted type-boundaries reproduce across seeds? (pre-registered)

**Status: PRE-REGISTERED, no results yet.** This section is committed before any training run;
the Results section below is empty until the runs finish. Thresholds here are frozen.

## Claim under test

When learned channels wire the latent spaces of several frozen models into a cyclic graph —
trained by plain backprop on cycle-consistency + structure-preservation + non-degeneracy —
the type-boundaries the routing enacts (the partition structure the coherent channels respect)
are REAL and REPRODUCIBLE, not seed-dependent artifacts. This is the precondition for the
"navigator enacts types → stability promotes them → crystallize into Rzk" architecture.

## Setup (compute-adapted, CPU-only box)

- **Frozen encoders (3, heterogeneous):** GPT-2 (124M, causal-LM, mean-pooled, 768d),
  Qwen2.5-0.5B-Instruct (mean-pooled, 896d), all-MiniLM-L6-v2 (contrastive, 384d).
  Encoders are never trained; embeddings are extracted once and cached.
  (The spec's cross-modal variant needs GPU hours this box lacks; three architecturally and
  training-objective-distinct model spaces is the minimal substrate and is flagged as such.)
- **Corpus:** ~4,200 text snippets with natural (unlabeled-for-training) category structure:
  3 prose sources (Austen / Aurelius / Melville), 3 code sources (sklearn / numpy / scipy),
  3 math generators (arithmetic / algebra / sequences). 9 latent categories, used ONLY for
  post-hoc diagnostics, never in training.
- **Channels:** 6 learned maps (all ordered pairs of the 3 spaces), 2-layer MLPs.
  Cyclic graph: both 3-cycles (A→B→C→A, A→C→B→A) and all 2-cycles (A→B→A, ...).
- **Losses (plain Adam, no learned optimizer):**
  1. cycle-consistency: MSE between round-trips and identity, all cycles;
  2. structure-preservation: match batch cosine-similarity matrices before/after each channel
     (relative-representations style);
  3. non-degeneracy: hinge penalty on per-feature batch std below a floor (anti-collapse).
  NO paired-alignment loss — pairing is used only for evaluation diagnostics.
- **Runs:** K=8 seeds; each run resamples the training set (bootstrap) AND reinitializes
  channels. A fixed held-out evaluation set (never trained on) is shared by all runs.

## The one measurement (frozen before running)

Enacted boundaries := for each run, map all eval samples into a common space via the trained
channels (fuse: concatenate each sample's image in all 3 spaces after routing into space A),
k-means with k=9 (the corpus's source count) → a partition of the SAME eval set per run.

**STABLE** = mean pairwise adjusted-Rand index (ARI) across the K runs' partitions.
Secondary: channel-map functional agreement — mean cosine between different seeds' channel
outputs on eval inputs (channels are comparable as functions because encoder spaces are fixed).

### Pre-committed thresholds
- **PASS: cross-seed ARI >= 0.8** (boundaries reproduce → real emergent types)
- **FAIL: ARI < 0.8** (seed-dependent habits → nothing to crystallize)

### Validity guards (a "stable" result does not count unless BOTH hold)
1. **No collapse:** each trained channel's output keeps effective rank >= 10 and per-feature
   std ratio (output/input space) >= 0.1; neighborhood preservation (top-10 NN Jaccard,
   source vs mapped, eval set) must exceed the untrained-channel baseline.
2. **No pass-through/steganography artifact:** cycle loss must be non-trivially low while
   structure preservation holds on HELD-OUT data; report the untrained-channels control
   (random-init, no training) — trained ARI must exceed the control ARI to attribute
   stability to training rather than to the data's own geometry.

### Controls reported alongside
- **Raw-geometry control:** k-means on each RAW encoder space across the same bootstrap
  resamples → ARI. If raw ARI is already >= trained ARI, the channels add no enacted
  structure beyond what the frozen encoders carry (report honestly; the gate is then about
  whether routing PRESERVES it, and the claim of navigator-ENACTED types is weakened).
- **Untrained-channels control:** same pipeline, channels at init.

## Decision rule (frozen)
- PASS + guards hold → the navigator-enacts-stable-types precondition holds; recommend the
  crystallization prototype (write stable boundaries as Rzk/Agda terms) as the next gate.
- FAIL or guards violated → the emergent-typing architecture is unfounded on this evidence;
  recommend the seeded structure-preserving merge as the sole deliverable and close the
  directed-type thread.

Honest RED is the goal, not a pass. Thresholds will not be adjusted after results exist.

---

## Results (2026-07-07, K=8 runs — see stability-gate/results_stability.json)

**VERDICT: FAIL — by the attribution guard, not the headline number.** The nominal
stability is high, but the controls show it is not the navigator's.

| quantity | value |
|---|---|
| cross-seed ARI (trained channels) | **0.948** ± 0.05 |
| **untrained-channel control ARI** | **0.987** (HIGHER than trained) |
| raw-geometry control ARI (gpt2/qwen/minilm) | 0.919 / 0.945 / 0.888 |
| channel functional agreement across seeds | **0.005–0.026** (≈ zero) |
| anti-collapse guards (eff. rank 137–200, std 0.64–0.71, NN-Jaccard 0.60–0.69) | pass |
| post-hoc ARI vs source labels (diagnostic) | 0.76 |

Reading, in one line: **what is stable is not enacted, and what is enacted is not stable.**

- The eval-set partitions reproduce almost perfectly across seeds (0.948) — but untrained
  random channels reproduce them *better* (0.987), and each raw frozen space alone gives
  0.89–0.95. The stability belongs to the corpus geometry inside the frozen encoders
  (partitions track the 9 sources at ARI 0.76), which the fused representation exposes with
  or without training.
- The channels themselves — the only thing training creates, the candidate "enacted
  boundaries" — agree across seeds at cosine ≈ 0.01. Every run learns a functionally
  different routing that satisfies the same objectives. Cycle loss plateaus ~0.70 while
  structure loss goes to ~0.007: the objectives constrain the channels only up to a large
  symmetry group, and the seeds land at arbitrary points of it.
- No collapse and no pass-through cheating (guards pass; neighborhoods genuinely preserved
  at Jaccard ~0.67): the runs are healthy. The instability is intrinsic to the setup, not
  an optimization pathology.

The verdict line originally printed by run_gate.py said PASS because the code implemented
only the collapse guards; the attribution clause ("trained ARI must exceed the control ARI
to attribute stability to training") was in the pre-registration from the start and is
decisive. The pre-registration is the authority; the code was corrected to match it.

### Caveats
- CPU-scale: three text-family encoders (no true cross-modal pair), 4.2k snippets, 400 Adam
  steps, K=8. A larger run could tighten the numbers, but the channel-agreement figure
  (~0.01 against a threshold-free comparison) is not a marginal miss — the objectives as
  specified simply do not pin down a canonical routing.
- The corpus has strong intrinsic cluster structure; a harder corpus would LOWER all ARIs,
  which cannot rescue the attribution gap.

### Decision (per the frozen rule)
FAIL → **the emergent-typing architecture ("navigator enacts types, stability promotes
them, crystallize into Rzk") is unfounded on this evidence.** The boundaries worth
crystallizing already live in the frozen encoders; the navigator adds nothing stable of its
own. Recommendation: ship the seeded structure-preserving merge as the sole deliverable and
close the directed-type thread. If it is ever reopened, the objective must include a
canonicalization pressure (something that collapses the symmetry group of solutions —
e.g. anchor-based relative representations or explicit gauge fixing) before stability of
enacted structure is even a well-posed target.
