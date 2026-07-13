# Stage-0b pre-registration (committed BEFORE running)

Follows the manager amendment: the invariant-observable rank tracks the **symmetric-power degree** of the
latent, not the latent degree, because gauge invariance forces every observable to be relational (≥ quadratic
in latent coordinates) — Koopman/Carleman. Predictions registered here; results in REPORT.md.

For a degree-r linear latent (eigenvalues λ_1..λ_r, generic), the distinct eigenvalues of the observable's
Koopman operator are:
- linear part {λ_i}: r of them
- quadratic part {λ_i λ_j, i≤j}: r(r+1)/2 of them
So an observable containing linear + quadratic terms has McMillan degree **r(r+3)/2** (generic, before
floor/T truncation); a purely quadratic (relational/Gram) observable has degree **r(r+1)/2**.

| r | pure-quadratic r(r+1)/2 | linear+quadratic r(r+3)/2 |
|---|---|---|
| 2 | 3 | 5 |
| 3 | 6 | 9 |
| 4 | 10 | 14 |

## Registered predictions

**P1 — sym-power theorem (clean observable, numpy, no floor truncation).** Degree-r linear latent, purely
quadratic observable of the state → block-Hankel rank **exactly r(r+1)/2** = {3,6,10} for r={2,3,4}. A
linear+quadratic observable → **r(r+3)/2** = {5,9,14}. PASS iff the clean-observable ranks equal these
(generic λ). This is the decoder-map validation; it either confirms the theorem or refutes it.

**P2 — corrected diversity leg (model pipeline).** Single model (K=1, removes the K-inflation confound),
**no anchor deflation** (m0=0, so the anchor cannot absorb diversity — addresses the frozen-capacity note),
doubled budget (T≈300, L=20 lags). Prediction: invariant-observable rank is **monotone increasing in r**
and moves toward the sym predictions (truncated by floor and T, so absolute values may be below {3,6,10}).
PASS iff rank(r=2) < rank(r=3) < rank(r=4) (corrected G1 diversity leg). The old "[proven-negative] raw rank
= latent degree" stays closed.

**P3 — K-invariance with HETEROGENEOUS models.** Replace identical MLPs with diverse architectures
(deep/shallow ReLU, quadratic, Fourier, localized). Prediction: after shared-anchor deflation, residual rank
**flat in K** (spread ≤ 1) at fixed diversity — confirming the K-invariance pass is not an artifact of
near-clone models. PASS iff flat across K∈{2,3,5} with heterogeneous members.

**P4 — G4 meter validity (two-edge topology).** Cycle cost = GW discrepancy of the composite self-coupling
v→e_A→w→e_B→v from identity, net of φ_solver. Prediction: **clones** (an edge of identical models) → cycle
cost **below** (φ_cyc + φ_solver); **disjoint** (genuinely different models) → **above**. PASS iff
clone_cost < floor < disjoint_cost.

Frozen constants unchanged (§10). Anchor capacity identical and frozen across all diversity cells (freeze
rule applied to the sweep, not only measurement windows).
