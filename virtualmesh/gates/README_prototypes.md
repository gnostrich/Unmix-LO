# Validated sandbox prototypes (starting points for the real gates)
- G1_reconcile_prototype.py : settling vs pooling on split knowledge. PASSED clean (1.5x).
  Real version: swap synthetic split-dim world for real specialist models + shared frame; ADD the
  one-step-confidence ablation arm; measure frame cost.
- G2_mzkernel_prototype.py  : MZ closure reproduces settling; kernel rank tracks routed subset not N;
  memory necessary; residual~difficulty. PASSED clean (linear-exact). Real version: nonlinear real
  models, approximate low-rank kernel, vary N to confirm rank scales with K not N.
- G3 has NO prototype yet (pathway thickening / gap-filling) — build from gates/README.md spec. Highest risk.
