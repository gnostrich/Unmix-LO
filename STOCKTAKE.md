# STOCKTAKE — the whole setup in one place (code state + theory reconciliation)

Full stock-take 2026-07-09, verified against the committed code on `main`. Bind to `CONSTRUCT.md` + `THEORY.md`.
No building in this pass — this is the single view of what the code is, what the theory now is, and where they
diverge.

---

## PART 1 — verified code state (read from the real modules)

### Module inventory (what each ACTUALLY implements)
| module | what it really is | mechanism | tested / status |
|---|---|---|---|
| `virtualworld/world.py` | seed world + 4 direct-view modalities; **medium = feature REGISTRY** | physics engine reuse; `scene_features` from `SCENE_REGISTRY` (D=len, tag-derived indices, `append_feature` grows) | ✅ registry verified bit-identical at D=26; runs at D=24/27 |
| `virtualworld/build_virtualworld.py` | real-encoder coverage-union build | ViT-base + MiniLM (frozen) + hand features → ridge→26-dim medium → stitch/drop-one/classify/knobs + MZ probe + interactive export | ✅ stitch R²=0.445, drop-one text +0.14, natural 0/6; **coverage-union = the validated win** |
| `virtualworld/detector.py` | structured/noise classifier (Step-0) | captured-vs-baseline **AND** held-out-R²-from-state | ✅ Step-0 validated |
| `virtualworld/mz_fluid.py` | EXPERIMENTAL recurrent MZ/tape probe | block-Hankel SV order + **linear** ridge memory-kernel closure | ⚠️ UNVALIDATED; "reduces toward classical linear state-space filtering" |
| `virtualworld/settle_real.py` | coherentflow.settle on the REAL aligned vectors | (imports the shipped settle) | ✅ honest **NO-OP** on real convergent senses (0 held); separable control fires; noise rejected |
| `virtualworld/world_ux.html` | interactive instrument (scene + decoherence map) | live JS stitch/classify/settle/combined-read on real vectors | ✅ Playwright-verified; honest framing |
| `coherentflow/coherentflow.py` | `settle` (dispatch) + `structured` guard + `combined_read` | **DEFAULT = the operator-feedback FLUID** (Step 1, via `fluid_pipeline`); `mechanism='averaging'` kept byte-identical behind a flag; read still a **fitted-lstsq-probe** (INV6, later step) | ✅ INV2 fixed; ⚠️ read still a fitted head |
| `coherentflow/fluid_pipeline.py` | wires `fluid_settle` into the pipeline recurrence | derives operator Rᵢ from each interface (`f≈z·Rᵢᵀ`); coupled feedback `S←S·Jᵀ`; instability-descent routing | ✅ INV2 PASS: incompatible→ρ=1.09>1, aligned→0.998, rogue excluded; averaging byte-identical to HEAD |
| `coherentflow/fluid_settle.py` | **the theoretically-correct feedback fluid** | models as OPERATORS; coupled ρ can exceed 1; instability-descent; nonlinear multistable intrinsic settle | ✅ all 3 acceptance criteria PASS (`FLUID_VERIFICATION.md`); **SEPARATE, not wired** |
| `coherentflow/` satisfaction (`run_satisfaction.py`, sandboxes) | battery vs the shipped object | — | ✅ 7/7, **0% false-positive** (a property of the GUARD, not the fluid) |
| `conformance/run_conformance.py` | executable test per theory-invariant | — | ✅ built; live tally **PASS 4 / PARTIAL 1 / FAIL 2** (INV2 fixed by Step 1) |

Related but SEPARATE experiments (not the construct code): `STABILITY_GATE.md` (pre-registered type-boundary
reproducibility gate, no results yet), `AGDA_RESULTS.md` (object re-hosted on the Agda oracle substrate), the
original `GATE.md`/`src/extractor.py` gradient-unmixing project, `report.md`/`COMPOSITION_THESIS.md` (the 12-gate
ledger). These inform the program but are not the fluid/construct implementation.

### Conformance suite — live PASS/FAIL (which invariants the code satisfies)
| # | invariant | status | gap |
|---|---|---|---|
| 1 | resizable / self-expanding medium | ✅ PASS | — |
| 2 | genuine feedback recurrence, not averaging | ✅ PASS | Step 1: `fluid_pipeline` wires the fluid as the default settle; averaging byte-identical behind a flag |
| 3 | faithfulness is a loss TERM, not a phase | 🟡 PARTIAL | shipped guard-in-loop OK; **fluid** stabilization is a pre-phase |
| 4 | MZ memory = the tape, ONE object | ❌ FAIL | memory is a transient dict; tape is a separate module |
| 5 | loss = models' own grounding | ✅ PASS | — |
| 6 | intrinsic output (not a fitted head) | ❌ FAIL | shipped `combined_read` is an lstsq probe; fluid equilibrium-shift correct but unwired |
| 7 | frozen interfaces, everything else flexible | ✅ PASS | — |

### Registry state
- **D is a knob** (`SCENE_D = len(SCENE_REGISTRY)`; index groups tag-derived; `append_feature` grows the medium).
  Verified bit-identical at the default 26; runs end-to-end at 24 and 27.
- **Input/medium registries:** the *medium* is a declared registry. The *input modalities* are a list (`MODS`)
  with per-modality raw→26 ridge maps (encoders frozen, medium-side fitted); adding a modality needs no medium
  reshape. There is **no separate "input registry" object** yet — modalities are a plain list; a formal input
  registry mirroring the medium registry is *specified-by-analogy but not built*.

### Built & verified vs Specified-not-built vs Assumed-but-regressed
| category | items |
|---|---|
| **BUILT & VERIFIED** | registry medium (D-knob); real-encoder coverage-union (0.445, the honest win); Step-0 detector; satisfaction guard (0% FP); interactive UX; `settle_real` honest no-op; **`fluid_settle` (all 3 acceptance criteria)**; conformance suite |
| **SPECIFIED, NOT BUILT** | tape≡MZ-memory as one object (INV4); self-expansion trigger (J2); multi-seed-model loss combination (J1); graduation rule (J3); abstraction-as-holonomy; **terrain read (T3)**; **trace tail-motion output (T5 full)**; **streaming I/O (T6)**; formal input registry |
| **ASSUMED-CORRECT BUT ACTUALLY REGRESSED** | the recurrence was **averaging not feedback** (INV2 — caught by probe, `fluid_settle` built, and now **WIRED as default** = FIXED, INV2 PASS); the output is a **fitted probe head** not intrinsic (INV6, newly surfaced); memory is a **transient dict** not the tape (INV4, newly surfaced); dimensionality was **glued** (INV1, fixed by registry) |

---

## PART 2 — theory reconciliation (was it in CONSTRUCT? added? contradicts?)

Full detail in `THEORY.md`; CONSTRUCT.md now carries a pointer + the resolved tensions. Summary map:

| theory item | in CONSTRUCT.md before? | action | contradiction resolved |
|---|---|---|---|
| **T1** mutual-instability = Baur objective | IMPLICIT (#3/#5) | added to THEORY.md; CONSTRUCT gloss updated | **#4** content-loss reading superseded by "instability of grounded coupling" (grounding stands) |
| **T2** fluid exclusion (emergent, no reject) | IMPLICIT (#3/#5) | added | none (mechanism of #3's emergent topology) |
| **T3** per-query stability **terrain** | **NEW** | added | **the informative object relocated** from content-swirl/held-structure to stability geometry — the biggest update |
| **T4** information relocation (point vs dynamics) | **NEW** | added | none (compatible with streaming/memory split) |
| **T5** trace-native output (tail motion) | IMPLICIT (#5) | added | **supersedes** the fitted-probe head (INV6); probe kept only as external scoring |
| **T6** streaming / anytime I/O | **NEW** | added | held-superposition = the CYCLING termination (dynamical, not static) |
| **T7** field/settling mental model | **NEW** (as explicit framing) | added | none |

### The three contradictions that matter (now flagged, to resolve deliberately — not in this pass)
1. **Objective:** #4's implicit content-reconstruction loss ⟂ T1's instability objective → resolved by
   *grounded-coupling instability* (grounding kept, content-loss reading dropped).
2. **Informative object:** old swirl/held-structure ⟂ T3 terrain → held-structure is the *divergent-case* read;
   the terrain (a confidence field non-vacuous on convergent models) is the general, still-untested object. This
   reframes what the convergence nulls mean: they close the *content-swirl* question, NOT the terrain question.
3. **Self-expansion:** #2's Hankel-SV-vs-FDT trigger ⟂ the empirics (`fdt_denoise` MID; `mz_fluid`→linear
   filtering) → candidate re-tie to persistent instability modes of the terrain (open joint J2, unbuilt).

---

## The one-line picture
**Code:** a validated coverage-union + guard + a correct-but-unwired feedback fluid; the shipped recurrence is
still averaging with a fitted-head read (regressions caught and mapped, not yet fixed).
**Theory:** the informative object is the **per-query stability terrain**, read from the **settling trace's tail
motion**, streamed until it converges/cycles/ times-out — grounded in the models' own coupling, non-vacuous even
when they agree.
**Divergence:** conformance INV2/INV4/INV6 are the exact points where code ≠ theory; T3/T5/T6 are the theory not
yet built. Fix order (deliberate): wire `fluid_settle` (INV2+INV6), fold descent into the loop (INV3), unify
held-memory into a resizable tape (INV4), then build the terrain read + tail-motion output + streaming I/O
(T3/T5/T6) — targeting `THEORY.md`, not the pre-theory spec.
