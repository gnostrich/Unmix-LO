# CONTEXT — theory arc and honest ledger

This file is the compressed provenance so an agent (or future human) can pick up the thread
without re-deriving it. Epistemic tags are used throughout: [proven], [occupied]=exists in the
literature, [candidate]=plausibly-original conjunction, [open]=untested, [reject]=ruled out.

## The object, in one line

A learned optimizer whose reusable weight-space operators form a self-expanding library,
extracted by blind source separation, made identifiable by trainee/domain diversity, and
composed per-task by a sparse router.

## How it was arrived at (why each piece is here, not decoration)

1. **Composition is the substrate.** A learned optimizer is a program composing reusable
   update-operations. General-vs-specialized, s-m-n, interpreter/Futamura. [occupied]
2. **Atomicity dial.** Number of reusable primitives = McMillan degree = rank of an
   operator-valued memory kernel = atomicity of its resolvent (Kronecker scalar / MSY
   operator-valued). Self-expansion = add a primitive when residual clears a noise floor. [occupied math]
3. **Grounding via trainees.** A closed compositional object is vacuous (self-consistent,
   meaningless). Making it operate on real trainees, judged by trainee error on retained data,
   gives it a semantics. This "endogeneity hack" is the actual content of the object. [candidate]
4. **The value hinges on identifiability.** "Refine into clean reusable pieces" is unidentifiable
   from a scalar penalty alone (Locatello 2019). Structural bias (low-rank operators + sparse
   routing) is required to break the degeneracy. [occupied]
5. **The natural solvent is diversity, not a gradient trick.** Multi-environment / multi-trainee
   variation restores identifiability (Hyvarinen-Sasaki-Turner; iVAE; causal representation
   learning). Active/curious task-generation breaks residual correlations (interventional
   identifiability). Decorrelation pressure (Barlow/VICReg) operationalizes it. Synthetic gradient
   is the WRONG tool here (it decouples in time, not sources). [reject synthetic-gradient-as-solver;
   occupied multi-env identifiability]

## What the experiments actually established

- Saturation (K -> #primitives) works. [proven, toy]
- Second-order extraction (projector-averaging = "amalgamated core") recovers only the subspace;
  routing on it HURTS. Higher-order (ICA) is mandatory. [proven, toy — a real design correction]
- With ICA + separable primitives: routed composition ~20x over vanilla; monolithic (no routing)
  ~1x. Collapses to 0.01x when primitives are correlated (identifiability failure). [proven, toy]
- Diversity restores separability: recovery 0.33 -> 0.78 as genres pooled; truly-fused pairs
  stay at 0.707 (correctly unseparable). [proven, genre-calibrated simulation]
- On a real-ish neural net, shared gradient structure is thin and generic (family signal
  ~1.04-1.13x, high absolute overlap is task-generic). [proven negative — the reason for the gate]

## The honest reduction

The value-bearing core of the object IS blind source separation (ICA) of weight-space operators.
- [occupied as machinery] — ICA/BSS, multi-environment identifiability.
- [conditional] — needs non-Gaussian, independent, cross-domain-decoupling primitive loadings.
- [open] — whether real gradients satisfy this. Never measured. Toys lean thin.

The compression/saturation half works. The compositional-value half is a conditional yes whose
condition (real-gradient ICA-identifiability) is the single untested fact.

## Quarantine (do not re-sell as novel)

- de Rham / Wu / Berger decomposition: holonomy-reducible <=> factorization; irreducible holonomy
  components = the factors. The "swirl = non-factorization" identity is this classical theorem.
  Only the non-normal/indecomposable corner is open, and it's actively-worked (SUSY/pseudo-Riemannian),
  not ours. [occupied]
- EqProp / holonomy learning rules (Laborieux-Zenke). [occupied]
- Learned optimizers: VeLO, Celo/Celo2, muLO, PyLO — monolithic, frozen-after-meta-train,
  NOT compositional. [occupied]
- Compositional/modular meta-learning, neural module nets, modular lifelong RL. [occupied]
- Categorical deep learning (backprop-as-functor, categorical compositional RL). [occupied]
  The unconnected pure-math bridge (operad <-> atomicity/resolvent) is the one open cell, high adjacency risk.

## Naturality rubric (used to reject tacked-on machinery)

A piece is natural iff: (a) implied by the object, not added to fix a symptom; (b) past-only or
checkable-at-apply-time (no smuggled future signal); (c) removing it degrades the object;
(d) no free hyperparameter not set by the object's own spectrum. Anything failing this is [tacked-on].
