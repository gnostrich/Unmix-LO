# UNMIX

**Separating the reusable operators of learning out of the gradient mixture.**

Blind source separation splits a mixture back into its pure sources — and only works because
the sources are really there. UNMIX asks whether the *reusable operations of learning*
(the weight-space "moves" a training process applies, shared across tasks) can be
separated out of the mixture of gradients that many diverse trainees produce — and, if so,
composed and reused as a library by a learned optimizer.

This repo is a **decision gate**, not a finished system. The theory is worked out, the
mechanism is validated on toys, and exactly one empirical fact remains untested. The gate
either green-lights the full prototype or cheaply kills it.

---

## The one-paragraph thesis

A learned optimizer can be seen as a program that composes reusable weight-space operations
to train models. If tasks in the wild are compositions of a small set of shared operations,
that optimizer could accumulate a **library of reusable primitives** and get more valuable the
more it is used (a compounding public good). The catch is identifiability: the useful primitives
must be *separable* in gradient space, and gradients from a single domain have their primitives
**fused** (correlated). The natural solvent for that is **trainee/domain diversity** — the same
primitive coupled to different partners across domains becomes separable when domains are pooled
(multi-environment ICA identifiability). Frontier training already supplies maximal diversity.
Whether real gradients actually decouple this way is the single open bet.

## What is proven (see `experiments/`)

- **The mechanism works given separable input.** ICA extraction + routed composition gives
  ~20x speedup over plain GD on engineered-compositional tasks, and correctly collapses when
  primitives are not separable. `04_ica_extraction_gate.py`.
- **Second-order extraction is insufficient; higher-order (ICA) is required.** Projector-averaging
  ("amalgamated core") recovers only the *subspace*, not individual primitives, and on that basis
  compositional routing *hurts*. `01`, `02`.
- **Library size saturates at the true primitive count**, not the task count (compression works).
  `01`, `02`.
- **Diversity restores separability, quantifiably.** Individual-skill recovery climbs 0.33 -> 0.78
  as genres are pooled, and truly-always-fused factors correctly stay fused (0.707). Calibrated to
  realistic LLM training genres. `05_diversity_restores_identifiability.py`.
- **Naturally negative prior on the raw claim.** On a small neural net, shared structure in
  *gradients* is thin and what is shared is *generic* (family-specific signal ~1.04-1.13x).
  `03_operator_vs_delta_neural.py`. This is why the gate below exists.

## What is NOT proven (the gate)

> Do **real** per-task gradients (real models, real data) contain **stable, individual, reused**
> operator components — i.e., does the world's compositionality reach into weight space and
> decouple across domains the way the simulation assumes?

Every green result above is upstream of this. It has never been measured on real gradients, and
the one real-adjacent measurement (`03`) leaned thin. **Do not build the full system before
running the gate.** See `GATE.md`.

## Decision rule

- **Gate passes** (components stable + individual + reused across held-out domain combos)
  -> the extractor you built *is* the de-risked prototype core; build the full self-refining
  optimizer around it.
- **Gate fails** (components correlated/unstable) -> fall back to the **robustness reframe**:
  a learned merge/aggregator that doesn't destroy worker differences under heterogeneity
  (non-destructive amalgamation). This needs none of the compositional machinery and still fills
  a real gap in decentralized-training stacks.

## Layout

```
README.md      - this file
CONTEXT.md     - compressed theory arc + honest epistemic ledger (read this second)
GATE.md        - precise one-day spec for the real-gradient gate (the job to run)
requirements.txt
experiments/   - the toys that establish the mechanism and the negative prior
src/           - the extractor spine (runs on synthetic now; hook for real gradients)
```

## Quickstart

```bash
pip install -r requirements.txt
python experiments/05_diversity_restores_identifiability.py   # the encouraging one
python experiments/04_ica_extraction_gate.py                  # the conditional positive
python src/extractor.py                                       # the spine, on synthetic data
# then: implement GATE.md (real LoRA gradients) — that's the whole decision.
```
