# THOUGHTWORLD-2 — Claude Code brief

Read PREREG.md first. This is a TIGHT follow-up to THOUGHTWORLD (which returned NOISE on frozen VISION
encoders, eff-rank ~16.4, indistinguishable from a random control). REUSE that project's engine and the
validated curvature instrument UNCHANGED -- only swap the fragments. If the thoughtworld package/repo is
available, import its engine.py, alignment, and measurement code directly; do not rebuild them.

## The one question
ViTs are appearance encoders (no dynamics to deviate with) -> NOISE was expected. Do fragments that ACTUALLY
MODEL DYNAMICS carry ATOMIC deviation from the engine, or is even their world-deviation structureless?

## Fragments to test (swap in; keep engine + instrument identical)
- F1: a small VIDEO / next-frame prediction model (HF, CPU-runnable) -- trained on dynamics.
- F2: a small LANGUAGE model (Qwen2.5-0.5B) predicting the next physical STATE from a text description of the
      scene (positions/velocities described -> next positions), mapped to engine coords via the ridge readout.
- F3 (optional, only if a CPU-runnable checkpoint exists): a tiny learned WORLD MODEL / RSSM (DreamerV3-scale).

## Instrument (identical, already validated)
deviation A = fragment_pred(state) - engine_next(state); report eff-rank(A), held-out R^2 (fit 50%/test 50%),
directed-frac ||F||/||A|| (F=(A-A^T)/2). Random-fragment control PER fragment.

## Verdict (pre-committed)
Report readout R^2 FIRST per fragment (does it even predict physics). Then:
ATOMIC iff eff-rank < 8 AND held-out R^2 >= 0.3 AND eff-rank distinguishable from the random control / ViT floor.
Else NOISE. Include the ViT floor row (eff-rank ~16.4) in the results table for direct comparison.

## Decisions (both strong)
- dynamics-fragments ALSO NOISE ~ floor -> GENERALIZED NEGATIVE (no frozen model carries atomic world-structure
  beyond the engine). Fold into synthesis; the world-structure axis is now general.
- any dynamics-fragment ATOMIC -> found WHICH fragment types carry world-structure; report which + directed-frac.
- a fragment with readout R^2 <= 0 has no dynamics -> its NOISE is uninformative (like ViTs); the INFORMATIVE
  cell is a fragment that predicts physics (readout R^2 > 0) yet whose deviation is still NOISE, or is ATOMIC.

## Discipline
Same engine, same instrument, only fragments change -> directly comparable to the ViT floor. Pre-commit thresholds.
Per-fragment random control is the fabrication guard. Honest NOISE is a real, now-general finding. Commit
fragments + alignment + measurement + RESULTS.md with the ViT-floor comparison row.
