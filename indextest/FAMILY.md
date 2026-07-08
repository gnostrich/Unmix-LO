# STAGE 1 — the FIXED planting family (frozen BEFORE checking any indexing/naive comparison)

Per BRIEF.md's hard constraint: this list of 12 configs is specified and frozen before running the
validity measurements. We are NOT tuning to make indexing win; we are asking whether ANY blind
config in this pre-specified family satisfies all three validity conditions:

- (i)  complementarity real:      true_oracle − best_single ≥ 0.15
- (ii) strong naive fails:        true_oracle − naive_strong ≥ 0.10
- (iii) oracle reachable:         true_oracle ≥ 0.80

Readout (identical for every arm, equal capacity + budget): StandardScaler → MLP(128,64) with
early stopping, calibrated blind on true features (oracle reaches 0.88–0.95 across target types;
single-view 0.48–0.76 — real gaps confirmed before freezing).

## Target types (on true disjoint factors zA→f_A, zB→f_B)
- **additive**: y = sign(zA·wA + zB·wB) — needs both, no interaction.
- **xor**:      y = sign((zA·wA)(zB·wB)) — needs both + their product.
- **gate**:     y = sign( [zA·wg>0] ? zB·w : −zB·w ) — zA multiplexes which zB-projection sets y.

## Entanglement types (the "gauge" the indexer/naive never see)
- **E1** per-view linear invertible:  e = f @ M            (M random Gaussian, invertible w.h.p.)
- **E2** cross-view linear invertible: [eA,eB] = [fA,fB] @ M_joint, resplit (mixes across views)
- **E3** nonlinear invertible:        e = tanh(f @ M)      (monotone elementwise → invertible in principle)
- **E4** lossy projection:            e = f @ P, P is D×(D/2) (destroys half the information)

## The 12 frozen configs
| # | target | entangle | n_train | note |
|---|---|---|---|---|
| c1 | additive | E1 | 4000 | linear target, invertible gauge |
| c2 | additive | E4 | 4000 | lossy — info-present check |
| c3 | xor | E1 | 4000 | interaction, invertible gauge |
| c4 | xor | E2 | 4000 | interaction, cross-view gauge |
| c5 | xor | E3 | 4000 | interaction, nonlinear gauge (hardest for linear align) |
| c6 | xor | E4 | 4000 | lossy |
| c7 | gate | E1 | 4000 | conditional, invertible gauge |
| c8 | gate | E2 | 4000 | conditional, cross-view gauge |
| c9 | gate | E3 | 4000 | conditional, nonlinear gauge |
| c10 | xor | E1 | 800 | sample-limited (semi-supervised regime) |
| c11 | xor | E3 | 800 | sample-limited + nonlinear gauge |
| c12 | gate | E4 | 4000 | lossy |

## Diagnostic reported for every config (to make the info argument explicit)
Besides the three arms, we also compute **entangled_oracle** = the same MLP on the raw entangled
concat [eA,eB] — the information-theoretic max extractable from the entangled features by any
readout. The "present-but-entangled-but-naive-hard" band requires BOTH entangled_oracle ≈
true_oracle (info present) AND naive_strong ≪ entangled_oracle (naive can't reach the max). If
naive_strong ≈ entangled_oracle everywhere, naive is already the universal readout and the band
is empty — indexing (which sees the same entangled features and less) cannot inhabit it.

Frozen 2026-07-08. Verdict computed strictly from these conditions on these configs; no config
added, dropped, or re-parameterized after seeing an indexing/naive comparison.
