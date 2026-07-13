# Pre-registration — trust battery for the D_e-effective-rank spatial meter (committed BEFORE running)

The candidate spatial-complexity meter is the equilibrated anchor's **D_e effective rank** (participation
ratio exp(H) of D_e's singular values, at a fixed anchor budget m). It tracked planted rank Spearman +1.00 in
`atom_observable_search.py`. Its predecessor (FW atom count) died of operating-point dependence; this meter
must survive the same tests before it is used in the 2×2. Substrate: planted continuous rank-r clouds
(N(0,I_r) mapped by a fixed random r×d map), K=3 members, seed-averaged.

## B1 — Scramble (gauge). 
**Predict:** eff-rank is invariant to the G0 scramble group (orthogonal×perm×scale×shift) on any member, to
numerical precision. D_e is built from the gauge-normalized (D,w) only, so |Δ eff-rank| < 1e-6. **PASS iff**
invariant.

## B2 — Null floor on structureless clouds + explanation of the 4.67 offset.
Feed (i) planted rank r = 1..8, and (ii) STRUCTURELESS clouds with MATCHED moments (iid Gaussian / a
per-feature shuffle of a structured cloud — same first/second moments, no low-rank structure).
**Predict:**
- eff-rank is **monotone non-decreasing in r** (more intrinsic structure → more anchor directions used).
- The **r=1 baseline reads an OFFSET O > 1**, not ≈1. Hypothesis for the 4.67-at-r=2 offset: at a FIXED
  budget m=12 with entropic ε, the anchor spectrum is smoothed — even a rank-1/2 cloud spreads mass over ~O
  atoms, so eff-rank = O + (increment in r). O is set by (m, ε), NOT by content. **Predict O ∈ [3, 6]**
  (consistent with r=2 → 4.67). The meter's usable signal is the ORDER / increment above O, not the absolute
  value.
- **Structureless matched-moment clouds read HIGH — near the ceiling C (≈ full rank / near m), clearly ABOVE
  the structured low-r readings.** This is the correct null: no compressible structure → the anchor uses all
  atoms. **PASS iff** eff-rank(structureless) > eff-rank(rank-2) by a clear margin AND eff-rank is monotone in
  r. (If structureless read LOW, or r were non-monotone, the meter would be reading noise, not structure.)

## B3 — Operating-point sweep (the test its predecessor failed).
Sweep ε ∈ {0.02, 0.05, 0.10} × n ∈ {80, 160} (and n_outer as a convergence check) at fixed planted ranks.
**Predict:** the **monotonic ORDER survives** — Spearman(eff-rank, r) ≥ 0.90 at EVERY operating point. Absolute
eff-rank values may shift with ε/n (the offset O moves), but the SIGN and ordering must not invert or flatten
(atom count inverted/flattened here — that killed it). **PASS iff** Spearman ≥ 0.90 across all cells.

## Battery verdict
The meter **survives** iff B1 ∧ B2 ∧ B3 all PASS. Only then is it used as the 2×2 spatial leg. If any leg
fails, the meter is reported dead like its predecessor — no patching.

## If it survives → 2×2 (pre-registered separately, to the digit, before running)
- spatial leg: D_e-eff-rank ↑ with geometric richness, flat with temporal diversity and K.
- temporal leg: pole count (P5 multiplicative closure) ↑ with dynamical diversity, flat with geometry and K.
- K leg: both flat.
Predicted values/orders committed in a separate prereg before that run.
