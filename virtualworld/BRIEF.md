# VIRTUAL WORLD MODEL — Claude Code brief (playable seed; light, CPU, no huge runs)

BIND FIRST to thoughtworld_construct/CONSTRUCT.md. This BUILDS an instrument to PLAY with -- the optimal
maximal-direct-modality seed with the VALIDATED construct mechanism applied. NOT a gate. Read PREREG.md.

## What to build
A small 2D physics world (pymunk or numpy rigid-body: a few balls, gravity, walls, collisions) emitting FOUR
direct-view modalities of the SAME events, each read by a SMALL CPU model, stitched by the validated construct
into one virtual world model you can poke.
  M1 vision (rendered frames -> MobileNetV3-small or small ViT/CLIP)
  M2 text (QUALITATIVE description, NO coordinates -> all-MiniLM-L6-v2 / Model2Vec)
  M3 audio (collision-event click/features -> Whisper-tiny encoder / CLAP, or hand-features to stay light)
  M4 time-series (positions/speeds/energies over last k frames -> tiny encoder or raw features)
OPTIONAL shortcut: llm-semantic-router/multi-modal-embed-small gives text+image+audio pre-aligned in 384-dim.

## Apply the VALIDATED construct (reuse, don't reinvent -- it passed synthetic validation)
Admit each modality to a shared medium (lightweight learned align, train split = only training). Then:
  STITCH (coherent fuse where they agree) ; DECOHERE (pairwise disagreement) ; CLASSIFY structured vs noise
  (VALIDATED detector -- see calibration note: needs D>=20-32 AND/OR held-out-predictivity R^2>=0.3) ;
  EXTEND paraconsistently where STRUCTURED (hold both) ; REJECT/denoise where NOISE.
smoke_loop.py confirms the loop is light (<1s numpy) and shows the three behaviors; it ALSO shows the detector
fails to separate at small D -- so build with higher D and the held-out condition, and VALIDATE the detector
on the inject-knobs before wiring real decoherence.

## Playable dashboard (the point)
Local notebook or single HTML/JS viewer with:
  1. Step the world; STITCHED world estimate vs ground truth.
  2. Per-modality UNIQUE contribution (drop-one coverage -- "fill missing modalities", the reliable win).
  3. Decoherence map: each disagreement tagged STRUCTURED(extend) / NOISE(reject) + the number.
  4. KNOBS: add/remove a modality; inject noise into one (watch -> NOISE, rejected); inject a structured
     hidden distinction (watch -> STRUCTURED, extended). Live demo of the three validated behaviors.
  5. HONEST label: real small models mostly AGREE or give NOISE decoherence (convergence, per xresolve) --
     so the coherent coverage-union is the main visible win; structured EXTENSION mostly shows when YOU inject
     it. Do NOT present injected structure as if convergent models supplied it naturally.

## Constraints
Small CPU models, cache encodings, small world (<= ~1-2k frames), reproducible from one script + light viewer.
Validate the detector on knobs first (STEP-0 style). Commit code + README of what to poke.

## Discipline
Reuse validated stitch/extend/reject; no ad-hoc flattening. Feature the coherent coverage win; show structured
extension honestly (mostly via inject-knob). No overclaim that frozen models decohere usefully -- they mostly
don't. This is an instrument to SEE the mechanism on a real seed, and to PLAY.
