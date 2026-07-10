# IO_STOCKTAKE — dimensional & wiring facts from the SHIPPED code (not the spec)

Read against the committed `virtualworld/` and `coherentflow/` on main. Bound to
`thoughtworld_construct/CONSTRUCT.md`. Every number below was verified by running/greping the real code;
file:line citations included. **Headline correction up front:** the two builds are *separate codebases with
different medium dimensions* — virtualworld's shared medium is **D=26** (not 24; 24 is the engine state), and
coherentflow's is a *separate* hard-coded **D=24**. There is no single shared `D` across the two.

> **Mechanism regression (see `MECHANISM_CHECK.md`):** the shipped settle is **feed-forward averaging** (proven
> identical to `mean(f_i)` with no held structure) + a state-dependent held-subtraction — **not** the genuine
> model→model feedback fluid CONSTRUCT #1-2 requires. It is structurally contractive (cannot be mutually
> unstable) and non-exclusionary (a corrupt model is averaged in, not routed around). Prior "no-op on real
> models" / satisfaction results validated the **guard** and the **coverage-union/held read** (real); the
> **fluid's** routing/exclusion was never instantiated or tested. Resolve deliberately before building on the
> "fluid" story.

## 1. Models actually wired (per modality)

| modality | source (as committed) | real neural encoder? | native output dim | frozen? |
|---|---|---|---|---|
| **vision** | `google/vit-base-patch16-224` — **ViT-BASE** (not ViT-small), `AutoModel`, `last_hidden_state.mean(1)` | **REAL encoder** | **768** | yes (`@torch.no_grad`, `.eval()`, features cached) |
| **text** | `sentence-transformers/all-MiniLM-L6-v2` — MiniLM-L6, attention-mask mean-pool | **REAL encoder** | **384** | yes (`@torch.no_grad`, `.eval()`) |
| **audio** | `world.audio_features(hist)` — hand-crafted collision/impact features | **HAND-FEATURES** (no model; **no Whisper**) | **28** (7 feats × K=4-frame window) | n/a (deterministic function of state) |
| **timeseries** | `world.timeseries_features(hist)` — hand-crafted velocity/speed/energy features | **HAND-FEATURES** (no encoder; derived from raw velocities/energies) | **84** ((2N+N+3)=21 × K=4) | n/a |

Citations: `virtualworld/build_virtualworld.py:35-36` (model ids), `:46,51` (ViT `from_pretrained().eval()`,
`last_hidden_state.mean(1)`), `:60,64` (MiniLM), `virtualworld/world.py:104-133` (audio, 7×K), `:137-153`
(timeseries, 21×K). **coherentflow has NO real models** — its "interfaces" are synthetic
(`make_interface`, `coherentflow.py:42`: random-orthogonal rotation of a latent + noise).

*Two real neural encoders (vision, text); two hand-feature views (audio, time-series).* Both encoders are
frozen — only cached, never trained.

## 2. The medium dimension D (shipped value)

- **virtualworld shared medium: D = 26.** Set at `build_virtualworld.py:132` `D = scene.shape[1]`, where
  `scene = W.scene_features(...)`. Since the registry refactor, the 26 = `len(SCENE_REGISTRY)` — `SCENE_LABELS`
  / `SCENE_D` / `SCENE_POS`/`VEL`/`COLL` are all *derived* in `_refresh_scene_index()` (no literal tuple); the
  registry declares 9 occupancy + 4 spatial + 4 wall-count + 3 motion + 2 energy + 2 velocity + 2 collision =
  **26**. It is a **permutation-invariant scene descriptor**, not the engine state.
- **Engine state = 24** (`ENG.D = 4*N`, N=6 balls × [x,y,vx,vy], `world.py:30`). The "D=24" in comments/spec
  refers to this engine state, *not* the medium. (The sandbox's "24" was the engine/synthetic dim.)
- **coherentflow medium: D = 24**, hard-coded constant `coherentflow.py:32` `D, T = 24, 600`. Unrelated to
  virtualworld's 26 — a separate synthetic space.

## 3. The interfaces (native_dim_i → D map)

Confirmed: each interface is a **ridge-alignment of a frozen model's native embedding to the shared medium**,
model-side frozen, only the medium-side projection fitted.

- **virtualworld interface** = `fit_ridge` / `apply_ridge` (`build_virtualworld.py:71-92`). Pipeline:
  `native (T, native_dim_i)` → standardize (μ,σ on train) → **PCA to n_pca=48** (only if native_dim > 48; so
  768→48, 384→48, 84→48, 28→28) → append bias → **ridge (λ=10)** to the standardized 26-dim medium. The fitted
  object is the tuple `(μ, σ, P, A)` — **only P (PCA basis) and A (ridge map) are learned**, on the train
  rollouts (`rollout < 0.6·N_ROLLOUTS`, `:134`). The model-side (ViT/MiniLM/features) is frozen and cached.
- **coherentflow interface** = `make_interface` (`coherentflow.py:42`): `v = z@R + noise` (R = fixed random
  orthogonal = the frozen "model-side"), then `A = lstsq(v → z)` (the fitted medium-side), returns `v@A`.
  native = medium = 24.

So the "everything frozen except the medium-side projection" contract **is** how it's built.

## 4. The I/O chain, actual shapes (virtualworld; n = N_ROLLOUTS·(T−1) = 26·44 = **1144** frames)

| stage | shape | where |
|---|---|---|
| engine states `s_cur` | (1144, **24**) | `world.collect`, `build:126` |
| medium target `Y` (standardized scene_features) | (1144, **26**) | `build:129,138` |
| vision: render → ViT | frames (1144,224,224,3) → `last_hidden_state` (1144,197,768) → mean(1) → (1144, **768**) | `build:141,49-51` |
| text: describe → MiniLM | (1144 strings) → (1144, seq, 384) → mean-pool → (1144, **384**) | `build:142,62-64` |
| audio / timeseries | (1144, **28**) / (1144, **84**) | `build:146-147` |
| **interface** each modality | (1144, native_dim) → ridge → **(1144, 26)** = `aligned[m]` | `build:153-158` |
| **stitch** (precision-weighted per-dim fuse) | (1144, **26**) | `build:164`, `stitch()` |
| **settle** (coherentflow port — see note) | `state` (n, **26**); `memory` = dict `{modality → held (n, 26)}` | `coherentflow.settle` |
| **combined read** | `consensus_view=state` (n,26) ⊕ `held=concat(memory.values())` (n, 26·k) → `combined_view` (n, 26+26k) | `coherentflow.combined_read:159-162` |
| **query output** | the shipped `combined_read` returns `(consensus_acc, combined_acc, held_dim)` for a query probe; underlying representation = `(state vector, memory dict)` | `:162` |

**Important:** the **shipped `virtualworld` build has NO recurrent settle** — it ships stitch + `classify`
(structured/noise) + drop-one + the *experimental* MZ probe (`MZ.run`, `build:242`). The recurrent
settle/hold/combined-read lives **only in `coherentflow`** (synthetic interfaces). The UX ported coherentflow's
settle onto virtualworld's aligned vectors; that port is the marriage, not a shipped virtualworld function.

## 5. Currently flexible vs hard-coded (against "everything sizewise flexible EXCEPT the interface contract")

| knob | should be | currently | gap? |
|---|---|---|---|
| **model-side of each interface frozen** | frozen | frozen (no_grad/eval/cached; coherentflow R fixed) | ✅ matches |
| **medium-side projection auto-refits** | auto | ✅ ridge fits to whatever `Y` dim it's handed (`fit_ridge` reads `Ytr` shape) | ✅ matches |
| **n_models (add/remove a model)** | free, no medium reshape | ✅ `MODS` list / `ifaces` list; medium is model-independent, ridge auto-fits the new model → **no medium reshape needed** | ✅ matches |
| **T (frames / rollouts)** | free | ✅ `N_ROLLOUTS,T` module constants (`build:34`); everything reads `n` from shape; no reshape | ✅ matches |
| **held-rank** | free / adaptive | ✅ data-driven: `structured()` eff = participation ratio, `P` = top-eff subspace (`coherentflow:60-64`); MZ tape order = Hankel-SV-vs-noise-floor (`mz_fluid.self_expand_order`) | ✅ matches (adaptive, not fixed) |
| **D (medium dimension)** | a free scalar knob | ✅ **FIXED (registry refactor).** The medium is now a declared **FEATURE REGISTRY** — `SCENE_REGISTRY` = list of `Feature(name, tags, fn)`; `SCENE_D = len(SCENE_REGISTRY)` (`world.py`). Changing D = changing the registry; the encoders are D-agnostic and the ridge interface auto-refits to the new width. Verified: default registry reproduces D=26 & all numbers bit-identically; D=27 (append) and D=24 (drop) run end-to-end. | ✅ **resolved** |
| **D-dependent index maps** | derive from D | ✅ **FIXED.** `SCENE_POS/SCENE_VEL/SCENE_COLL` are now **derived from each feature's `tags`** (`_refresh_scene_index()`), not fixed ranges — they track the registry automatically (e.g. appending a `coll`-tagged feature extends `SCENE_COLL`). | ✅ **resolved** |
| **medium is resizable / self-expanding** | yes (CONSTRUCT non-negotiable #2) | ⚠️ **structure now supports growth** — `append_feature(feature)` grows D→D+1 and refreshes the index views (the hook a self-expansion step would call). The self-expansion *logic* (when/what to append) is intentionally **not wired yet**. So the precondition is in place; the growth policy is the remaining step. | partial (precondition met) |
| **coherentflow D** | free | ⚠️ hard-coded constant `D,T=24,600` (`coherentflow:32`), but the math is fully D-generic (reads from shape) → one-line change, self-consistent | minor |
| **single shared D across builds** | one parameter | ❌ two separate D's (26 vs 24), two codebases, no shared config | **GAP** |
| **interface bottleneck / regularization** | knob | ⚠️ `n_pca=48`, `λ=10` hard-coded defaults (`fit_ridge`); `K=4` window, `DAMP=0.5`, `ITERS=18` constants | minor |

**Direct answers to the three questions:**
- *Change D without touching model wiring?* — **YES (as of the registry refactor).** The medium is a declared
  feature registry whose length *is* D; the index groups derive from tags. Edit the registry, and the D-agnostic
  encoders + auto-refitting ridge adapt with nothing else touched. Verified at D=24/26/27.
- *Add/remove a model without reshaping the medium?* — **Yes.** The medium is model-independent; add an encoder,
  append to `MODS`, provide its raw features, and its ridge auto-fits to the medium. No reshape.
- *Change T freely?* — **Yes.** `N_ROLLOUTS`/`T` are constants; nothing reshapes on T.

## 6. The read / output representation

Confirmed at `coherentflow.combined_read:149-162`:
- `consensus_view = state` — the settled unified estimate (n, D), with held structure **removed** from the
  consensus target during settling (`settle:` `coherent = mean(ifaces[i] - memory[i])`), so a consensus-collapse
  cannot see the held distinction.
- `held = np.concatenate([memory[i] ...])` — the **held-superposition**: one (n, D) block per interface whose
  disagreement was tagged STRUCTURED. **Multiple structured interfaces → multiple held branches**, concatenated.
- `combined_view = concatenate([consensus_view, held])` → shape (n, D + D·k) for k held interfaces.

So the output is **consensus where interfaces cohere + a held set (multiple branches) where structured
decoherence exists** — exactly as claimed. The shipped function returns *query-probe accuracies*
`(consensus_acc, combined_acc, held_dim)`; the underlying **representation is `(state vector, memory dict of
held vectors)`**, not a single flat vector.

## Divergences from the "everything flexible except interfaces" principle
1. ~~**D is not a scalar knob**~~ — **RESOLVED (registry refactor).** The medium is now a declared feature
   registry (`SCENE_REGISTRY`, `world.py`): `D = len(registry)`, index groups derive from per-feature `tags`,
   and `append_feature()` grows the medium. Verified bit-identical at D=26 and end-to-end at D=24/27.
2. **The medium's self-expansion is not wired** — the *structure* now supports growth (`append_feature`), so
   the CONSTRUCT #2 precondition is met, but the growth *policy* (when to append a mode, e.g. Hankel-SV /
   noise-floor) is intentionally not implemented in this step. Partial.
3. **Two separate D's / two codebases** (virtualworld registry-D=26, coherentflow hard-coded D=24) with no
   shared config — the settle is a UX-side port, not a shipped virtualworld function. Still open.
4. **Interface hyperparameters hard-coded** (`n_pca=48`, `λ=10`, `K=4`, `DAMP`, `ITERS`) — fine as defaults,
   but list them as knobs if the UX exposes them. Still open (minor).

Everything *else* already honors the principle: model-side frozen, medium-side auto-refitting, **D now a
registry knob**, n_models / T / held-rank all free.
