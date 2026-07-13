# Pre-registration — the 2×2 spatial/temporal dissociation (committed BEFORE running, to the digit)

The D_e-effective-rank spatial meter SURVIVED its trust battery (PREREG_derank_battery.md; commit on
`main`). It is now cleared as the 2×2 **spatial leg**. This file registers the full 2×2 — a double
dissociation — with specific predicted numbers, committed before any 2×2 number exists. Predictions are
derived from PRIOR COMMITTED calibrations only (the battery curve for the spatial meter; P5 product theory for
the temporal meter); the 2×2 run is NOT peeked at before this is committed.

No new mechanism, no new QC. The two meters are reused verbatim: D_e eff-rank via
`atom_observable_search.equilibrate_fixed_m` + `eff_rank`; pole count via `pole_closure` ERA/Ho-Kalman on the
quadratic observable's autocovariance Hankel. This is an experimental CONTROL (instrument dissociation), not a
mechanism change.

## Amendment (committed BEFORE any 2×2 number — this file supersedes the first-cut prereg)
Three changes were made before running, none of them peeking at 2×2 output:
1. **Stated flatness tolerance with provenance.** The geometry-leg (D_e eff-rank) off-axis flatness band is set
   to **FLAT_TOL_SPATIAL = 0.5**, justified as ~3.5× the meter's MEASURED seed jitter (~0.13: from the B1
   battery run the r=2 seed spread was 4.63–4.71 ≈ 0.08 and r=6 was 5.80–5.93 ≈ 0.13) and ≈ 1/2.7 of the
   on-axis swing (5.86−4.67 ≈ 1.19 over the exact-prediction G levels). So "flat" means "moves less than a
   third of a seed-jitter-normalized on-axis step", not an arbitrary 0.5.
2. **Knob values read off the calibration curve; EXACT numbers only where the curve is RESOLVABLE, ordering
   elsewhere.** The battery curve (r→eff-rank) is 4.65/4.67/5.10/5.34/5.71/5.86/5.89/6.04 for r=1..8. Adjacent
   levels separated by ≳ 3× the seed jitter are "resolvable" and get an exact-number prediction; levels whose
   gap is within jitter get an ORDERING-only prediction (≥, no digit). This is applied to both legs below.
3. **K-leg split into a confirmatory half and a novel half** (see the K control section): the spatial-meter
   K-invariance is confirmatory (re-confirms the established pooling invariance of the atom-count lineage); the
   temporal-meter K-invariance is NOVEL (first test that independent same-pole realizations don't inflate the
   resolved-product count).

## Unified substrate (one generative process, two INDEPENDENT knobs)

Latent = **G independent AR(1) coordinates** s_1..s_G(t), each s_k(t+1) = λ_{c(k)} s_k(t) + ε_k(t) with its
own independent Gaussian drive, **then each coordinate whitened to unit stationary variance** (divide by
σ/√(1−λ²)). Whitening is load-bearing: it removes the per-pole variance (a coord with λ=0.9 is intrinsically
larger than one with λ=0.5), so the STATIC cloud is ~N(0, I_G) regardless of the pole assignment — the spatial
meter then sees only G, never D. Whitening leaves each coordinate's pole/autocorrelation unchanged, so the
temporal meter is untouched.

- **Spatial knob G** = number of latent coordinates (independent geometric directions). Intrinsic dimension of
  the static cloud {s(t)} = G.
- **Temporal knob D** = number of DISTINCT pole VALUES among the G coordinates (coords D+1..G reuse values
  from the distinct set; G ≥ D). Coords sharing a pole are independent (own drive) → geometrically distinct
  but temporally one mode.
- **Control K** = number of members. Each member observes the SAME latent through its own fixed random G×d
  embedding + a G0 gauge scramble (only (D,w) crosses; poles are preserved by a linear map).

Distinct pole VALUES are drawn from the fixed real pool {0.90, 0.70, 0.50} in order (D ≤ 3 here uses only these
three; products are well separated). n=100 static-cloud points (subsampled from the series) for the spatial
read; T=60000 for the temporal read. SEEDS=[0,1,2], seed-averaged.

Two readouts on the SAME data object:
- **SPATIAL** (order-scrambled → pure static cloud): cloud_to_Dw → equilibrate_fixed_m(m=12) → D_e eff-rank.
- **TEMPORAL** (time order preserved): quadratic observable q={s_i s_j} → autocovariance Hankel → ERA poles →
  count of distinct resolvable poles on the product set (P5 machinery, `closure_error < 0.05`).

## The 2×2 grid (+ K control)

| sweep            | knob varied           | fixed        | role                                   |
|------------------|-----------------------|--------------|----------------------------------------|
| SPATIAL axis     | G ∈ {2, 4, 6, 8}      | D=2, K=3     | on-axis for D_e, off-axis for poles    |
| TEMPORAL axis    | D ∈ {1, 2, 3}         | G=6, K=3     | on-axis for poles, off-axis for D_e    |
| K control        | K ∈ {2, 3, 5}         | G=6, D=2     | both flat — CONFIRMATORY (D_e) + NOVEL (poles) |

## Registered predictions — TO THE DIGIT

### Spatial leg — D_e eff-rank (from the committed battery calibration)
The whitened AR static cloud of intrinsic rank G should reproduce the battery's continuous-rank curve (r=G):
battery read 4.67 / 5.34 / 5.86 / 6.04 at r = 2 / 4 / 6 / 8. Adjacent-gap resolvability (seed jitter ~0.13):
4.67→5.34 (gap 0.67, ≳5× jitter → resolvable), 5.34→5.86 (gap 0.52, ≳4× → resolvable), 5.86→6.04 (gap 0.18,
~1.4× jitter → NOT resolvable). So G=2,4,6 get EXACT predictions; G=8 gets an ORDERING-only prediction.

| G | predicted D_e eff-rank | kind | tol |
|---|------------------------|------|-----|
| 2 | 4.7  | EXACT    | ±FLAT_TOL_SPATIAL (0.5) |
| 4 | 5.3  | EXACT    | ±0.5 |
| 6 | 5.9  | EXACT    | ±0.5 |
| 8 | ≥ value at G=6 (~5.9–6.1) | ORDERING only | plateau — no distinct digit registered |

- **On-axis:** Spearman(D_e eff-rank, G) ≥ 0.90, monotone non-decreasing across all four G (the plateau at G=8
  must not INVERT, but need not exceed G=6 by a resolvable margin). The three EXACT digits must each land within
  ±0.5; the G=8 point is judged on ordering only. (Battery lesson: ORDER is the registered signal.)
- **Off-axis FLAT in D:** at fixed G=6, sweeping D ∈ {1,2,3}, range(D_e eff-rank) ≤ FLAT_TOL_SPATIAL = 0.5 — the
  whitened cloud is rank-6 regardless of poles, so the meter reads ~5.9 at every D. (Tolerance provenance: ~3.5×
  the measured seed jitter, ≪ the 1.19 on-axis swing over the exact levels — see Amendment §1.)
- **Off-axis FLAT in K:** at fixed (G=6,D=2), sweeping K ∈ {2,3,5}, range(D_e eff-rank) ≤ FLAT_TOL_SPATIAL = 0.5.
  (This is the CONFIRMATORY half of the K-leg — see the K control section.)

### Temporal leg — pole count (from P5 product theory)
Distinct real poles are read off the pool {0.90, 0.70, 0.50} in order. D distinct latent poles → distinct
pairwise products {λ_iλ_j : i≤j} number **D(D+1)/2**. Resolvability (P5b, top-|·| prefix at finite T=60000):
- D=1: pole 0.90 → 1 product {0.81}. Magnitude 0.81 large → RESOLVABLE → EXACT 1.
- D=2: poles {0.90,0.70} → 3 products {0.81, 0.63, 0.49}, all |·| ≥ 0.49 → RESOLVABLE → EXACT 3.
- D=3: poles {0.90,0.70,0.50} → 6 products {0.81,0.63,0.49,0.45,0.35,0.25}. The top 5 (|·| ≥ 0.35) resolve at
  T=60000; the smallest, 0.25, sits near the resolvability floor → NOT reliably resolvable → ORDERING/BAND, not
  a single digit.

| D | distinct products D(D+1)/2 | predicted resolvable pole count | kind |
|---|----------------------------|---------------------------------|------|
| 1 | 1  | 1   | EXACT |
| 2 | 3  | 3   | EXACT |
| 3 | 6  | 5–6 (0.25 product may under-resolve) | ORDERING/BAND (≥ 4, i.e. strictly > D=2) |

- **On-axis:** Spearman(pole count, D) ≥ 0.90, monotone; the two EXACT points must hit 1 and 3 on the nose, and
  D=3 must exceed D=2 (count ≥ 4). count(D=3) − count(D=1) ≥ 3.
- **Off-axis FLAT in G:** at fixed D=2, sweeping G ∈ {2,4,6,8}, range(pole count) ≤ 1 — the distinct product
  VALUES depend only on the 2 distinct poles, not on how many coords carry them, so the count stays 3.
- **Off-axis FLAT in K:** at fixed (G=6,D=2), sweeping K ∈ {2,3,5}, range(pole count) ≤ 1. (This is the NOVEL
  half of the K-leg — see the K control section.)

### K control — both meters flat, split into a confirmatory and a novel half
The K sweep (K ∈ {2,3,5}, fixed G=6, D=2) must leave BOTH meters flat. The two halves are NOT the same
epistemic claim:
- **Confirmatory half — spatial D_e eff-rank flat in K.** The atom-count lineage already established a pooling
  invariance (atom count K-invariant, flat 3,3,3 across K=2,3,5 — LEDGER "[holds] atom count is K-invariant").
  This half RE-CONFIRMS that pooling invariance for the D_e-eff-rank spatial meter: adding members must not
  inflate the read geometric complexity. PASS iff range_K(D_e eff-rank) ≤ FLAT_TOL_SPATIAL = 0.5.
- **Novel half — temporal pole count flat in K.** Never tested: whether K INDEPENDENT same-pole latent
  realizations (K "members" each an independent draw of the identical dynamics) inflate the resolved-product
  count. The claim is that they do NOT — pooling more evidence sharpens estimation but does not add spurious
  distinct products. PASS iff range_K(pole count) ≤ 1. This is a genuinely new instrument property, flagged as
  such so a pass counts as a first demonstration and a fail counts as a real (not merely re-confirmatory) miss.

## Dissociation verdict — PASS iff ALL of:
1. **Spatial meter tracks ONLY space:** Spearman(D_e, G) ≥ 0.90 (monotone, plateau at G=8 may be ordering-only)
   AND the three EXACT digits land within ±FLAT_TOL_SPATIAL (G=2→4.7, G=4→5.3, G=6→5.9, each ±0.5) AND
   range_D(D_e) ≤ FLAT_TOL_SPATIAL (0.5) AND range_K(D_e) ≤ FLAT_TOL_SPATIAL (0.5) [K-leg CONFIRMATORY half].
2. **Temporal meter tracks ONLY time:** Spearman(poles, D) ≥ 0.90 AND the two EXACT points hit 1 (D=1) and 3
   (D=2) on the nose AND D=3 count ≥ 4 AND range_G(poles) ≤ 1 AND range_K(poles) ≤ 1 [K-leg NOVEL half].
3. **Cross-margin:** for each meter, off-axis range < 0.5 × on-axis range (each meter moves at least twice as
   much along its own axis as along the other's). On-axis range for D_e is over the EXACT G levels {2,4,6}
   (excludes the G=8 plateau); for poles it is over D {1,2,3}.

If any clause fails, the 2×2 is reported as a partial or failed dissociation — honestly, no patching. A
one-axis result (temporal leg clean, spatial leg not, or vice versa) is reported as exactly that. The forbidden
move is dressing a failed clause as a pass or tuning the substrate after seeing the numbers.

## Known registered risks (stated before the run)
- The spatial meter was calibrated on continuous N(0,I_r) clouds; the whitened AR stationary cloud is also
  per-coordinate Gaussian, so the curve should transfer — but if the absolute values shift outside ±0.5 while
  the ORDER holds, clause 1's Spearman/flatness can still pass on order; the absolute-value miss is recorded.
- Temporal resolvability at D=3 depends on the smallest product magnitude resolving at T=60000; the 5–6 band
  admits the under-resolved case. If even D=3 under-resolves to < 4, the on-axis margin (≥3) is at risk and
  that is a genuine (recorded) failure, not to be rescued by raising T post hoc.
