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

## Results

*(empty at pre-registration; filled by stability-gate/run_gate.py output after K runs)*
