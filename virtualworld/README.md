# Virtual World Model — a playable multimodal instrument

A single small 2D physics world (balls, gravity, elastic wall + ball collisions) seen through **four
genuinely-different direct-view modalities of the same events**, stitched into one coherent world
estimate you can poke. Built to run easily on CPU in Claude Code; everything is precomputed by one
Python script that emits a light JSON the viewer loads.

Binds to `../thoughtworld_construct/CONSTRUCT.md`. Reuses the numpy physics engine from
`../thoughtworld/engine.py` (unchanged as a file; N raised at runtime so the shared world-state medium
is D=26) and the frozen small CPU models already cached (`google/vit-base-patch16-224`,
`sentence-transformers/all-MiniLM-L6-v2`).

## HONEST LABEL — read this first

Real frozen small models mostly **AGREE**. After each modality is aligned to the shared medium, the
pairwise disagreement (decoherence) is state-independent **NOISE** — in this build **0 of 6** natural
modality pairs are STRUCTURED (consistent with the session's *xresolve* null: cross-model ambiguities
coincide). So:

- The **visible, robust win is the coverage-union stitch** (fill-missing-modalities / drop-one). That
  is the thing to look at.
- **STRUCTURED extension appears mainly when YOU inject it** with the knob. That is an *injected*
  distinction — **NOT** one the convergent models supplied naturally. The dashboard and this README say
  so plainly and never dress injected structure as if it arose from the models.

## Two layers — kept visibly separate

| Layer | What | Status |
|---|---|---|
| **VALIDATED** (single-step) | stitch (coherent fuse) / classify structured-vs-noise (D≥20 **and** held-out-predictivity R²≥0.3) / extend-paraconsistent / reject-noise | passed synthetic **Step-0** validation |
| **EXPERIMENTAL** (recurrent) | the tape ≡ Mori-Zwanzig memory as one object; self-expansion by a Hankel-SV-vs-noise-floor criterion; contraction native to the dynamics | **UNVALIDATED** — reduces toward classical linear state-space (Kalman/subspace-ID) filtering; **never** presented as validated |

The recurrent layer runs **alongside** the validated behaviors, never merged into them. Do not read its
numbers as a construct result.

## Step-0 detector validation (mandatory, done BEFORE wiring real modalities)

`python step0_validate.py` — validates the structured/noise detector on synthetic inject-knobs. The
smoke_loop calibration note showed captured-vs-baseline **alone** fails at small D (baseline eff/D too
high). The fix: use D≥20–32 **and** add held-out-predictivity R²≥0.3 as a second condition.

Result: at **D ∈ {20, 24, 32}** the detector separates cleanly —
- every STRUCTURED knob (`structured_linear`, `structured_regime`, `structured_diffuse`) → **STRUCTURED**, held-out R² ≥ **0.54**;
- injected `noise` → **NOISE**, held-out R² ≤ **−0.06**.

Held-out-R²-from-state is the decisive separator: a genuine hidden distinction is a reproducible
function of the world state (predictable); random noise is not. **PASS.**

## What to poke (dashboard.html)

Open `dashboard.html` in a browser (it loads `data.js`, produced by the build script). Sections:

- **(a) Stitched world estimate vs ground truth** — the fused estimate tracks true scene quantities
  (mean height, mean speed, kinetic energy, near-floor count, occupancy) over a held-out rollout.
  Overall test R² ≈ **0.45** (velocity/energy dims **0.75**, spatial dims **0.33**).
- **(b) Per-modality unique contribution (drop-one)** — remove a modality and watch the union degrade.
  **Text** is the spatial workhorse (drop it → spatial R² 0.33→0.13); **audio** adds collision/velocity
  info. Vision and time-series are largely *redundant* given the others (honest: the hand-written
  qualitative text and the collision features already carry their information). The union of all four
  still beats any single modality — that is the coverage win. A coverage heatmap shows which scene
  dimensions each modality can read (spatial block vs velocity block vs collision dims).
- **(c) Decoherence map** — all 6 pairs tagged STRUCTURED/NOISE with the held-out-R² and captured/base
  numbers. All NOISE here (the honest result).
- **(d) Knobs** — pick a modality and an action:
  - **inject NOISE** → tagged **NOISE** (held-out R²≈0); rejecting it (drop the flagged modality,
    denoise toward consensus) restores R² vs a naive fuse that trusts the corruption.
  - **inject a STRUCTURED hidden distinction** (a rank-2 signal that is a reproducible function of the
    true world state) → tagged **STRUCTURED** (held-out R² ≈ 0.4); the construct **EXTENDS**
    paraconsistently — holds the coherent stitch *and* exposes the extra distinction (recovered
    held-out R² ≈ 1.0). Reminder shown on-screen: *you* injected this.
- **EXPERIMENTAL panel** (dashed, badged UNVALIDATED) — the recurrent MZ/tape probe: the tape
  self-expands to a small order via Hankel SVs above a surrogate noise floor; the memory kernel closes
  through-time structure in the streaming residual (spectral radius < 1 ⇒ native settling). This closure
  **is** classical linear state-space filtering; reported honestly, not as a validated capability.

## Reproduce

```bash
python step0_validate.py        # Step-0 detector validation  -> step0_results.json
python build_virtualworld.py    # encode 4 modalities (cached), align, stitch, knobs, MZ probe
                                #  -> virtualworld_data.json + data.js
# open dashboard.html in a browser
```

First `build` run encodes ~1150 frames with the frozen ViT on CPU (~2 min) and caches features under
`feat_cache/` (git-ignored); subsequent runs are seconds.

## Files

- `world.py` — physics world (reuses the engine), the four modality feature extractors, and the
  permutation-invariant scene medium (balls are indistinguishable, so an ordered per-ball state is not
  identifiable from vision/text/audio; invariant scene features are).
- `detector.py` — the Step-0-validated structured/noise classifier (captured-vs-baseline **and**
  held-out-R²-from-state ≥ 0.3).
- `step0_validate.py` — mandatory Step-0 synthetic validation (`step0_results.json`).
- `build_virtualworld.py` — the build: encode → align (ridge, the only training) → stitch → drop-one →
  decoherence → knobs → emits `virtualworld_data.json` + `data.js`.
- `mz_fluid.py` — the EXPERIMENTAL (unvalidated) recurrent MZ/tape layer.
- `dashboard.html` — self-contained viewer.
- `smoke_loop.py` / `BRIEF.md` / `PREREG.md` — original design + light smoke.

## Binding to CONSTRUCT.md (non-negotiables)

This is a **component-level instrument**, not the whole construct. Assessed against the CONSTRUCT.md
non-negotiable structure: the VALIDATED single-step layer is memoryless, so (1) tape≡memory and (2)
self-expansion are **not** in it — they live only in the EXPERIMENTAL layer, labeled unvalidated. (3)
Baur descent: partial — a loss (ridge) populates the alignment maps, but modality cardinality is
hand-set, not emergent. (4) Loss = the engine's **own** state/next-state (the seed's own grounding), not
an invented judge — honored. (5) Faithfulness is native (the classifier is the validated single-step
instrument; the recurrent layer's contraction is a spectral-radius penalty, a loss term, not a verify
phase). No claim beyond the seed+modalities' closure.
