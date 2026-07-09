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
- (none) — fdt_denoise, virtualworld, coherentflow all landed; see below.

## coherentflow (whole-object one-shot, observe-don't-prove — NOT a gate, like virtualworld)
Built the COMPLETE construct object as ONE loop (Baur-written MZ tape, recurrent settle under internal
coherence loss, guards INSIDE, combined read across interfaces). On frame-diverse INJECTED input:
SETTLES (residual 2.666→0.003), SURFACES+HOLDS structure (1/3 held, only the branch-carrier), COMBINED
READ recovers the branch 1.000 vs consensus-collapse 0.483 (+0.517 payoff — held superposition surfaces
what a single-frame collapse loses). Controls: COHERENT→honest no-op (0/3 held, circ 0); NOISE→rejected
by concentration (full-rank eff 23/24), nothing written to tape, no G1. OWNER EqProp STEERING honored:
routing kept FIXED/initialized (labeled, not learning); learning mechanism = equilibrium-response
(two-settles-and-a-difference, NO backprop-through-time). Honest point-4 flag: response reciprocity
asymmetry ≈0.19–0.23, β-INDEPENDENT (structural, not numerical) → EqProp-LIKE but NOT textbook: paraconsistent
holding makes it a constrained relaxation, not a clean scalar-energy flow. RESULTS.md + json committed.
- SATISFACTION battery (internal check vs the REAL build, NOT a gate): 7/7 checks PASS; FALSE-POSITIVE rate
  0.0%/40 seeds (no-fabrication headline confirmed); detection floor sharp ~1.2–1.4× noise (100% by 1.6×).
  Read-payoff LARGE (~+0.50 vs consensus, ~+0.46 vs the FAIR naive-mean baseline which sits at chance) —
  investigated per the brief's "large payoff = suspicious" warning and found REAL not artifact (held-out
  probe; naive averaging genuinely can't recover the minority alignment-orthogonal branch). Discrepancy vs
  sandbox (sharper floor, bigger payoff) explained: real build holds structure OUT of consensus; sandbox
  folds it in (degrading its own channel). Honest caveat: payoff is regime-specific (injected structure);
  on real convergent models it no-ops. SATISFACTION.md + run_satisfaction.py committed. Added non-breaking
  settle(init=) to enable the contraction-from-transient test against the real dynamics.

## DONE since last state
- **fdt_denoise** (gate 12, component native-denoising FDT test): **MID → DOWNGRADE (null-leaning)**.
  Real disagreement pairs FDT-frac 0.49–0.54 (never near FDT-holds 0.81); covariance-matched-noise
  control already scores 0.43 so genuine content above noise ~0.06–0.11; vision readout R² negative;
  literal-random fabrication guard floors at 0.13 (estimator sound, null trustworthy). MZ "noise floor"
  = heuristic threshold, not native separation. FDT_RESULTS.md + fdt_results.json committed, report row 12.
  (First run died on the Qwen download stall; re-run with ViT/DINO cached completed clean.)
- **virtualworld** (playable instrument, NOT a gate): built + pushed (24c2387). Coverage-union stitch
  R²=0.445 (best single = text 0.337); drop-one text +0.140 / audio +0.034; NATURAL structured
  decoherence 0/6 (matches xresolve convergence null); inject-knobs live (noise→rejected 4/4,
  structured→extended 4/4); Step-0 detector validated (D∈{20,24,32}, held-out-R²≥0.3). MZ recurrent
  layer kept SEPARATE + badged UNVALIDATED. Layer-separation PROVEN: VW_MZ=0 → validated payload
  BIT-IDENTICAL (18887 bytes) to VW_MZ=1; MZ is a pure sink, no entanglement (flag committed f9f078a).

## world_ux.html (interactive instrument — marries virtualworld real models + coherentflow settle semantics)
Self-contained single-file UX (virtualworld/world_ux.html + interactive_data.js exported by build_virtualworld.py).
Physics scene + decoherence-map are the visual heroes. Runs stitch/classify/settle/combined-read LIVE in JS on
real ViT/MiniLM-derived aligned vectors (encoders can't run in-browser). STOCK-TAKING (answered): REAL models =
vision ViT-base (google/vit-base-patch16-224), text MiniLM (all-MiniLM-L6-v2); audio & timeseries = HAND-FEATURES.
The interactive SETTLING loop lives in coherentflow (synthetic interfaces); virtualworld is single-step +
experimental MZ. KEY HONEST FINDING (empirically established): on REAL convergent modalities the pairwise
disagreements are large, mid-rank (eff 12-15) and world-UNpredictable, so injected structure gets BURIED → the
object correctly NO-OPS (matches the convergence thesis). What works live on real data: coverage-union (stitch
0.36 on shown subset / 0.445 full-test ref, drop-one) + NOISE detection (injected noise → eff→21-25 → NOISE·
rejected, 0 held = no-fabrication live). The STRUCTURED-held-and-surfaced mechanism only bites on SEPARABLE
senses → provided a REAL vs SYNTHETIC-separable mode toggle; synth mode shows STRUCTURED·held + combined-read
beats consensus. Detection = eff-rank 3-way (struct=low-rank+sizeable, noise=full-rank+large, else consensus).
Verified via Playwright (Chromium): no console errors, all behaviors + both modes correct. Includes an I/O-flow
panel (world 24-dim → 4 senses [ViT 768/MiniLM 384/audio 28/ts 84] → medium 26 → stitch/settle → readout;
"nothing set in stone": D & N runtime params, medium self-expanding, senses add/remove-able).

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
