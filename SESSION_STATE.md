# SESSION STATE — resume anchor (durable across compaction)

Branch: **main** (all work pushed here; `claude/unzip-archive-xov5w0` == main historically).
Git: a **steward subagent** (id may change) handles commits — pre-register/commit-before-run, trailers
(`Co-Authored-By: Claude Fable 5` + `Claude-Session:`), push to main, fetch+rebase on reject, never force,
no PRs. Every experiment: PREREG committed BEFORE run; controls; honest null = real finding; caches gitignored.

## The through-line (see report.md + COMPOSITION_THESIS.md — the canonical ledgers)
Frozen-model composition is **infrastructure over reachable content, never new knowledge** (capability or
world-structure). 11 gates done across two axes.

## Completed gates (all on main, each with RESULTS.md + json)
1. UNMIX — gradient operators: RED. 2. VIRTUALMESH G1 settling: RED. 3. G2 scale-free cost: PASS(scoped).
4. G3 compression: PASS(amended). 5. BIOMESH naive pooling: RED (cold-split). 6. indextest blind index: RED.
7. synergy task-aware aggregator: RED (P1 fails). 8. ROUTEMESH union-routing: PASS(scoped) — beats SOTA on
multi-hop/cyclic, flat cost. 9. THOUGHTWORLD vision-deviation: NOISE. 10. THOUGHTWORLD-2 dynamics-fragment
(LLM/video): NOISE generalized (LLM signal = prompt-arithmetic, velocity-confound control confirmed).
11. xresolve cross-model ambiguity: COINCIDING null (models alias SAME pairs; single-model paraconsistency
+0.059 banked). thoughtworld_construct/CONSTRUCT.md = canonical spec (construct itself UNVALIDATED; open joints J1-J5).

## IN FLIGHT
- **fdt_denoise** (component, native-denoising FDT theory test): STEP-0 estimator VALIDATED & committed
  (8a57bf8) — OU 0.81 vs FDT-violating ≤0.18, gap 0.64, passed. Real-model run (run_fdt.py) was still
  running at last check; **when done: read fdt_denoise/fdt_results.json, write FDT_RESULTS.md, commit to main**,
  add row 12 to report.md. Controls: random-model fabrication guard + cross-arch/cross-modal pairs.
  (Was launched as a parallel general-purpose agent; agent stopped, run left in background.)

## NEXT / PENDING BUILD (not started)
- **virtualworld** (spec in virtualworld/PREREG.md, BRIEF.md, smoke_loop.py — committed): a PLAYABLE INSTRUMENT
  (not a gate). Build a small 2D physics world (numpy rigid-body / reuse thoughtworld/engine.py) emitting 4
  modalities (vision→small ViT/MobileNet, text→all-MiniLM, audio→collision features, time-series), align to a
  shared medium (lightweight learned map = only training), STITCH where agree, CLASSIFY decoherence
  structured(low-rank+heldout-R²≥0.3)/noise, EXTEND paraconsistently where structured, REJECT where noise.
  Dashboard (single HTML/JS or notebook): stitched-vs-ground-truth, drop-one per-modality contribution,
  decoherence map, KNOBS (add/remove modality, inject noise→NOISE-rejected, inject hidden distinction→STRUCTURED-
  extended). **Step-0 discipline: validate the structured/noise detector on the inject-knobs BEFORE wiring real
  decoherence; use D≥20-32 AND held-out-R²≥0.3 (smoke_loop.py flagged the detector doesn't separate at small D).**
  HONEST LABEL required: real small models mostly agree/noise (convergence, per xresolve) → coverage-union is the
  main real win; structured extension mostly appears only when the user injects it — do NOT present injected
  structure as convergent-model-supplied.
  ** OWNER STEERING (2026-07-09, binding): do NOT flatten the construct into a static per-frame stitch/classify/
  extend pipeline — that is the forbidden flattening error. Keep TWO clearly-separated layers: (1) VALIDATED
  single-step behaviors (stitch/classify[D≥20 + heldout-R²≥0.3]/extend-paraconsistent/reject) — present as
  validated; (2) EXPERIMENTAL recurrent Baur/MZ fluid (tape≡MZ memory as ONE object, memory kernel through-time,
  self-expansion by Hankel-SV-vs-noise-floor, contraction/settling native) — label "unvalidated; probes showed
  it reduces toward classical state-space filtering." Faithfulness = a LOSS TERM, not a generate/verify phase.
  Loss = seed models' own grounding, not arbitrary/invented-judge data. Dashboard/README must keep the two
  layers visibly distinct; never dress the recurrent part as validated. Build agent was steered to report a
  diff against CONSTRUCT.md's non-negotiable-structure items 1-5 before proceeding. **

## Repo map
report.md (headline ledger), COMPOSITION_THESIS.md (synthesis), per-dir RESULTS.md. thoughtworld/engine.py =
reusable physics engine. biomesh/embed_specialists.py = frozen-encoder embedding helper. Data/caches gitignored.
