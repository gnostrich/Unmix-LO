# Reproduced results (2026-07-07, this repo as committed)

All five experiments run end-to-end; numbers below are from this run and match the claims
in `README.md` / `../CONTEXT.md`. (02 and 03 originally referenced pre-rename filenames
`gate0.py`/`toy.py`; fixed to point at `01_saturation_and_transfer.py` and
`_shared_neural_mlp.py`.)

## 01 — saturation and transfer
- Separating training: library saturates at K=4 (true 4, tasks 8), true-primitive overlap 1.000;
  held-out routed 2.73x vs monolithic 2.93x.
- Non-separating training: overlap still 1.000 (subspace recovered) but routing loses:
  routed 1.43x vs monolithic 2.26x.

## 02 — sparse routing fails under second-order extraction
- 16 primitives, 2-3 active/task: library recovers the full subspace (overlap 1.000) but
  routed = 0.27x vs monolithic = 1.11x — routing advantage 0.24x. Second-order recovers the
  subspace, not individuals; routing on it actively hurts.

## 03 — operator vs delta on a neural net (the negative prior)
- DELTA (gradient dirs):    within-family 0.206 vs across 0.183 — ratio 1.13x, sep 0.37 sd.
- OPERATOR (Fisher):        within-family 0.632 vs across 0.610 — ratio 1.04x, sep 0.55 sd.
- Family-specific signal is thin and the high absolute overlap is task-generic.

## 04 — ICA extraction gate (the conditional positive)
- Separable loadings: individual-recovery 1.000, routed 20.60x, monolithic 1.07x — advantage 19.29x.
- Correlated loadings (locked): recovery 0.733, routed 0.01x — collapses exactly as identifiability
  theory predicts.

## 05 — diversity restores identifiability
- Recovery vs genres pooled: 0.328 / 0.568 / 0.640 / 0.738 / 0.747 / 0.783 (1 -> 6 genres).
- Control (pair fused in ALL genres): both stay at 0.707 — correctly never separated.

## extractor self-test (`../src/extractor.py`)
- STABLE 0.997, INDIVIDUAL max-overlap 0.705 / kurtosis 9.33, REUSED residual 0.055 with
  3.5 active comps/task, ground-truth recovery 0.859.

The real-gradient gate itself is implemented in `../gate/` — see `../gate/README.md`.
