# NATIVE-DENOISING FDT — RESULTS: **MID → DOWNGRADE (null-leaning)**

Run 2026-07-09 per `fdt_denoise/PREREG.md` (frozen before the run; component test bound to
CONSTRUCT.md, not the construct). The claim under test: cross-model disagreement `d_t = predA - predB`
is a *fluctuation* whose FDT-satisfying part (`F_gauge`) is generator-inherited content and whose
FDT-violating part (`F_noise`) is vacuous, so the Baur/MZ process denoises **natively** (a real noise
floor, not a bolted-on threshold) — *iff* the FDT relation actually holds on real disagreement.
Full numbers in `fdt_results.json`. STEP-0 estimator validation (`step0_results.json`) passed and was
committed **before** this run (OU/FDT-holds 0.813 vs FDT-violating ≤0.175, gap 0.638).

## The estimator scale (from STEP-0, the anchors)
`FDT-satisfying fraction` ≈ **0.81** when FDT holds by construction (OU) and ≈ **0.17** when it is
violated by construction (random-walk-diff / rotational currents / white noise). That is the band the
real-model numbers must be read against.

## Result — real disagreement lands MID, and the fabrication-adjacent control eats most of it

| pair | FDT-satisfying frac | reversibility | irrev. current | \|d\|rel |
|---|---|---|---|---|
| ViT–DINO (cross-ARCH) | **0.503** | 0.559 | 0.154 | 0.379 |
| ViT–Qwen (cross-MODAL) | **0.541** | 0.679 | 0.102 | 0.523 |
| DINO–Qwen (cross-MODAL) | **0.486** | 0.741 | 0.080 | 0.505 |
| **CONTROL** ViT vs RANDOM-features (literal) | **0.132** | 0.206 | 0.362 | 0.978 |
| **CONTROL** DINO vs matched-noise prediction | **0.431** | 0.513 | 0.155 | 0.687 |

State-readout held-out R²: ViT **−0.151**, DINO **−0.074**, Qwen **+0.139** (Qwen is handed
positions/velocities in the prompt — prereg-noted confound; the vision models barely track the state).

## Verdict — MID fraction, but not native denoising
- **The fabrication guard PASSES.** Literal random features score **0.132** — correctly *FDT-violating*
  (right at the STEP-0 violating floor of 0.17), with high irreversible current (0.362). The estimator is
  **not** fabricating FDT structure out of noise. The null it returns is trustworthy.
- **Real pairs are MID (~0.49–0.54)** — halfway between clean-FDT (0.81) and violating (0.17), never
  near the "FDT-holds" regime. Per the pre-committed MID rule, that fraction *is* the reported
  natively-denoisable share — but it must be read net of the matched-noise control.
- **The matched-noise prediction control scores 0.431** — covariance-matched *white noise* already
  reproduces most of the real pairs' "FDT-satisfying" fraction. The genuine generator-inherited content
  above covariance-matched noise is only **~0.06–0.11**, not ~0.5.
- **Vision state-readout R² is negative** (ViT −0.15, DINO −0.07): the frozen vision encoders barely
  read the world state at all, so there is little real generator content in their disagreement to *be*
  FDT-structured — consistent with the small real-over-noise margin.

**Conclusion: DOWNGRADE of the naturality claim (null-leaning).** The "FDT-satisfying fraction" of real
cross-model disagreement is mostly generic second-moment (covariance) structure, not generator-inherited
content: it sits far below the FDT-holds regime and only marginally above a covariance-matched-noise
fabrication baseline. Native FDT denoising is therefore **not principled** on real frozen-model
disagreement here — the MZ "noise floor" behaves as a heuristic threshold, not a native content/noise
separation. The estimator is sound (the literal-random guard floors correctly), so this is an informative
null, not a measurement failure.

## Where it sits in the spine
Consistent with the session through-line and the xresolve null: frozen-model composition/denoising
accesses **reachable second-moment structure**, it does not manufacture new generator content. The
recurrent Baur/MZ object's own probes (see `virtualworld/mz_fluid.py`) already reduce toward classical
linear state-space filtering; this FDT test reaches the same place from the fluctuation-dissipation side —
the disagreement is (mostly) noise with realistic covariance, not a content-bearing FDT fluctuation.
