# GATE — the one-day, parallelizable decision experiment

**Goal:** measure whether real per-task gradients contain *stable, individual, reused* operator
components. This is the single fact everything rides on. It is ~1 day with parallel agents, and
the extractor you build here IS the prototype's spine (not throwaway).

## Design

Two axes matter, so the task set must vary both:
- **domains/genres** (to test cross-domain decoupling — the diversity solvent)
- **compositions within domain** (to test reuse on held-out combinations)

### Step 1 — trainees (parallelizable, cheap)
Fine-tune ~6-12 small LoRA adapters on a small base model (e.g. a 0.5B-1.5B instruct model),
spanning genres:
- 2-3 code tasks, 2-3 math tasks, 2-3 prose/NL tasks (and optionally a 4th genre).
Keep them tiny (LoRA rank 8-16, a few hundred steps). Fix the SAME base checkpoint for all so
gradients live in a comparable space. Parallelize across agents/GPUs.

### Step 2 — collect gradients
At the shared base checkpoint (and optionally at 1-2 later checkpoints), for each task collect a
CLOUD of minibatch gradients (or pseudo-gradients = adapter delta after k inner steps).
Flatten to vectors. ~100-300 per task. Store as `grads/{genre}/{task}.npy`, shape (n, P).
Recommended: restrict to a parameter subspace (e.g. project to top-r PCA of the pooled gradients,
r ~ 50-200) so ICA is well-conditioned — this mirrors the low-rank operator assumption.

### Step 3 — extract (use src/extractor.py)
Pool gradients across all tasks/genres, whiten, run FastICA (n_components ~ r). This is the exact
method validated in experiments 04 and 05.

### Step 4 — the three checks (this is the decision)

1. **STABLE** — re-run extraction on bootstrap resamples / different seeds. Components are stable
   if the Amari distance (or matched-component cosine) between runs is high (matched cosine > ~0.8).
   Unstable components = no real sources.

2. **INDIVIDUAL** — components must not be stuck at ~0.707 pairwise (the fused/45-degrees signature
   from experiment 04's control). Check pairwise |cos| between recovered components is low, and that
   each explains a concentrated, sparse set of tasks (a real primitive is used by SOME tasks, not
   smeared across all). Kurtosis of loadings should be high (non-Gaussian = separable source).

3. **REUSED** — split tasks into train/held-out *combinations*. Fit the library on train tasks;
   check held-out tasks' gradients are well-reconstructed as a SPARSE combination of the SAME
   components (low residual, few active components each), AND that the same components recur across
   different genres (a component that fires for both a code task and a math task = genuine reuse).
   The diversity test (experiment 05) predicts recovery should IMPROVE as more genres are pooled —
   verify that curve rises on real gradients.

### Decision rule
- **All three pass** -> the world's compositionality reaches weight space and decouples across
  domains. Build the full self-refining optimizer around this extractor. Green light.
- **Any fails** (correlated/unstable/not-reused) -> fall back to the robustness reframe
  (non-destructive learned merge). Do not build the compositional system.

## Notes / gotchas
- Watch for the trivial task-generic component (there will be a big shared direction that is NOT
  family-specific — see experiment 03). Regress it out or verify individuality per check 2.
- Gaussian loadings => ICA cannot separate. If everything looks Gaussian, that itself is a
  (negative) answer.
- Keep the base checkpoint fixed; comparing gradients across different base weights is apples-to-oranges.
- If compute-limited, even 6 adapters across 3 genres is enough for a rough yes/no.
