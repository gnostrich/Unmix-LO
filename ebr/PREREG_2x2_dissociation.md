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

Distinct pole VALUES are drawn from the fixed pool {0.90, 0.70, 0.50, 0.80·e^{±iπ/4}} in order, so products are
well separated. n=100 static-cloud points (subsampled from the series) for the spatial read; T=60000 for the
temporal read. SEEDS=[0,1,2], seed-averaged.

Two readouts on the SAME data object:
- **SPATIAL** (order-scrambled → pure static cloud): cloud_to_Dw → equilibrate_fixed_m(m=12) → D_e eff-rank.
- **TEMPORAL** (time order preserved): quadratic observable q={s_i s_j} → autocovariance Hankel → ERA poles →
  count of distinct resolvable poles on the product set (P5 machinery, `closure_error < 0.05`).

## The 2×2 grid (+ K control)

| sweep            | knob varied           | fixed        |
|------------------|-----------------------|--------------|
| SPATIAL axis     | G ∈ {2, 4, 6, 8}      | D=2, K=3     |
| TEMPORAL axis    | D ∈ {1, 2, 3}         | G=6, K=3     |
| K control        | K ∈ {2, 3, 5}         | G=6, D=2     |

## Registered predictions — TO THE DIGIT

### Spatial leg — D_e eff-rank (from the committed battery calibration)
The whitened AR static cloud of intrinsic rank G should reproduce the battery's continuous-rank curve
(r=G): battery read 4.67 / 5.34 / 5.86 / 6.04 at r = 2 / 4 / 6 / 8. Registered:

| G | predicted D_e eff-rank | tol |
|---|------------------------|-----|
| 2 | 4.7  | ±0.5 |
| 4 | 5.3  | ±0.5 |
| 6 | 5.9  | ±0.5 |
| 8 | 6.0  | ±0.5 |

- **On-axis:** Spearman(D_e eff-rank, G) ≥ 0.90, monotone non-decreasing. (The battery's lesson: the ORDER is
  the registered signal; absolute values may shift with the AR substrate but the order must hold.)
- **Off-axis FLAT in D:** at fixed G=6, sweeping D ∈ {1,2,3}, range(D_e eff-rank) ≤ 0.5 — the whitened cloud is
  rank-6 regardless of poles, so the meter reads ~5.9 at every D.
- **Off-axis FLAT in K:** at fixed (G=6,D=2), sweeping K ∈ {2,3,5}, range(D_e eff-rank) ≤ 0.5.

### Temporal leg — pole count (from P5 product theory)
D distinct latent poles → distinct pairwise products {λ_iλ_j : i≤j} number **D(D+1)/2** (generic, well-
separated pool). The resolvable count is the top-|·| prefix at finite T; at T=60000 with the separated pool it
reaches the ceiling. Registered:

| D | distinct products D(D+1)/2 | predicted resolvable pole count |
|---|----------------------------|---------------------------------|
| 1 | 1  | 1 |
| 2 | 3  | 3 |
| 3 | 6  | 5–6 (smallest-|·| product may under-resolve at finite T) |

- **On-axis:** Spearman(pole count, D) ≥ 0.90, monotone; count(D=3) − count(D=1) ≥ 3.
- **Off-axis FLAT in G:** at fixed D=2, sweeping G ∈ {2,4,6,8}, range(pole count) ≤ 1 — the distinct product
  VALUES depend only on the 2 distinct poles, not on how many coords carry them, so the count stays 3.
- **Off-axis FLAT in K:** at fixed (G=6,D=2), sweeping K ∈ {2,3,5}, range(pole count) ≤ 1.

## Dissociation verdict — PASS iff ALL of:
1. **Spatial meter tracks ONLY space:** Spearman(D_e, G) ≥ 0.90 AND range_D(D_e) ≤ 0.5 AND range_K(D_e) ≤ 0.5.
2. **Temporal meter tracks ONLY time:** Spearman(poles, D) ≥ 0.90 AND range_G(poles) ≤ 1 AND range_K(poles) ≤ 1.
3. **Cross-margin:** for each meter, off-axis range < 0.5 × on-axis range (each meter moves at least twice as
   much along its own axis as along the other's).

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
