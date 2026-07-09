# VIRTUAL WORLD MODEL — build spec (playable seed, NOT a gate)
# BIND to thoughtworld_construct/CONSTRUCT.md. This BUILDS the optimal seed to PLAY with, applying the
# VALIDATED construct mechanism (stitch / extend-paraconsistently / reject-noise -- all three passed
# synthetic validation). This is an INSTRUMENT to explore, not a pass/fail experiment.

## Goal
A single "virtual world model" you can poke: a small physics world that emits MULTIPLE direct-bridge
modalities, frozen small models read each, and the construct stitches them into ONE coherent world +
shows where they decohere and whether that decoherence is structured (extends) or noise (rejected).
Must run EASILY in Claude Code -- small models, CPU, no huge runs, cached.

## Seed = optimal DIRECT-FIT modality diversity (within the "not too intricate" range)
The world is a simple 2D physics sim (pymunk or numpy rigid-body: a few balls, gravity, walls, collisions).
It emits FOUR modalities that are all DIRECT views of the SAME events (buildable correspondence, per the
physical-bridge law -- no orphan modalities):
  M1 VISION   : rendered frames (numpy circle-draw or PIL) -> small vision encoder (MobileNetV3-small or
                 a CLIP/ViT-small, CPU). 
  M2 TEXT     : QUALITATIVE description ("two balls near left wall, one moving right") -> small text encoder
                 (all-MiniLM-L6-v2 or Model2Vec, CPU). Qualitative ONLY -- NO coordinates (avoids the
                 velocity/arithmetic-handoff confound that bit thoughtworld-2).
  M3 AUDIO    : collision-event signal (a synthesized click/tone per collision, or just the event's
                 spectral features) -> tiny audio encoder (Whisper-tiny encoder or CLAP, CPU) OR, to keep
                 it light, a hand-features vector (onset times/energies). 
  M4 TIME-SER : physical quantities over a short window (positions/speeds/energies over the last k frames)
                 -> a tiny learned encoder or raw features. Direct view of the physics through the time axis.
NOTE: use the pre-fused option if convenient: llm-semantic-router/multi-modal-embed-small (text+image+audio
in one 384-dim space, ~116M, CPU) can supply M1/M2/M3 already aligned, saving alignment work.

## The construct applied (the VALIDATED mechanism -- reuse, do not reinvent)
For each frame/state, each modality is ADMITTED to a shared medium via a lightweight learned alignment
(fit on a train split; the ONLY training). Then:
  STITCH  : coherent fuse where modalities agree -> the unified world estimate (the shared world-model).
  DECOHERE: compute pairwise disagreement fields.
  CLASSIFY (validated detector): is each decoherence STRUCTURED (reproducible low-rank subspace on held-out,
            captured >> baseline) or NOISE (spreads full-rank ~ baseline)?
  EXTEND  : where STRUCTURED, HOLD both framings (paraconsistent) and expose the extra distinction it carries.
  REJECT  : where NOISE, denoise toward the coherent stitch, do NOT extend.
(These are exactly the three behaviors that passed synthetic validation: don't-invent / extend-on-structure /
reject-noise. Reuse that logic.)

## Playable outputs (the point -- something to poke)
A small local dashboard / notebook / HTML that lets you:
  1. Step the physics world; see the STITCHED unified world estimate vs ground truth (reconstruction quality).
  2. See PER-MODALITY contributions: what each modality uniquely fills in (coverage-complementarity --
     "fill in missing modalities", the reliable coherent win).
  3. See the DECOHERENCE MAP: where modalities disagree, and each disagreement tagged STRUCTURED (extends) or
     NOISE (rejected), with the captured-vs-baseline number.
  4. KNOBS to play: add/remove a modality (watch coverage change); inject noise into a modality (watch it get
     tagged NOISE + rejected); inject a structured hidden distinction (watch it get tagged STRUCTURED +
     extended). This lets you SEE the three validated behaviors live on the real seed.
  5. HONEST readout: since real small models likely mostly AGREE (convergence) or produce NOISE decoherence,
     the dashboard should HONESTLY show that structured-decoherence is RARE on these inputs -- the coherent
     stitch (coverage-union) is the main visible win; structured extension mostly appears when YOU inject it
     (the knob), which is the honest demonstration that the machinery works but real convergent models rarely
     supply structured decoherence (consistent with xresolve). Do NOT fake structured decoherence as if it
     arose naturally.

## Constraints (must run easily in Claude Code)
- Small CPU models only (MobileNet/MiniLM/Whisper-tiny/Model2Vec class). Cache all encodings.
- Small world (a few hundred to ~1-2k frames max). No multi-model heavy runs.
- Everything reproducible from one script + a light viewer. Commit code + a short README of what to poke.

## Discipline (bind to CONSTRUCT.md)
- Reuse the VALIDATED stitch/extend/reject logic; do NOT reinvent or flatten into an ad-hoc pipeline.
- Coverage-union (fill missing modalities) is the reliable win -- feature it. Structured-decoherence extension
  is the RARE bonus -- show it honestly (mostly via the inject-knob), not as if convergent models supply it.
- No overclaim: this is an INSTRUMENT to see the construct behaviors on a real seed, not evidence that frozen
  models decohere usefully (they mostly don't -- xresolve). Label the coherent win and the injected-decoherence
  demo distinctly.

## CALIBRATION NOTE (from smoke_loop.py -- read before building the detector)
The captured-vs-baseline structured/noise detector needs the state dimension D to be reasonably large,
else baseline (= eff_rank/D) is too high and structured decoherence is not separable from noise (at D=6 the
injected-structured case was MISSED -- baseline 0.83). FIXES for the build:
  - Use a higher-dimensional world state / medium (D >= 20-32, e.g. more balls or richer per-ball features),
    so baseline is low and structure stands out; AND/OR
  - Add HELD-OUT PREDICTIVITY as a second detector condition (the thoughtworld-validated instrument: fit the
    decoherence's subspace on train, require it predicts held-out decoherence R^2 >= 0.3) -- this is the
    robust criterion; captured-vs-baseline alone is not enough at small D.
  - VALIDATE the detector on the inject-knobs FIRST (inject known-structured -> must tag STRUCTURED; inject
    known-noise -> must tag NOISE) before wiring it to real-model decoherence. If the knobs don't separate,
    fix the detector before proceeding (same STEP-0 discipline as fdt_denoise).
