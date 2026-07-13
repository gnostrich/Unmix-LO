# P5 pre-registration — multiplicative closure (Wick law), committed BEFORE running

**Claim.** For a linear-Gaussian degree-r latent u(t+1)=A u(t)+ε with poles {λ_i} = eig(A), the quadratic
observable q(t) = {u_i u_j : i≤j} (mean-centered) has an autocovariance sequence whose modes are the
**pairwise products {λ_iλ_j : i≤j}** (Isserlis/Wick: Cov(q_{t+k}, q_t) is a sum of products of two latent
cross-covariances R(k)∝A^k, so modes multiply). Therefore poles estimated (ERA / Ho-Kalman) from the
covariance Hankel of q must exhibit **multiplicative closure**: every estimated pole ≈ some λ_iλ_j.

**Substrate (fixed).** r=3 latent with KNOWN poles:
- λ1 = 0.85 (real)
- λ2, λ3 = 0.65 · e^{±iπ/4} (complex pair, |·|=0.65, angle 45°)

**Predicted product set {λ_iλ_j} (to the digit):**
| product | value | magnitude |
|---|---|---|
| λ1·λ1 | 0.7225 | 0.7225 |
| λ1·λ2, λ1·λ3 | 0.5525·e^{±iπ/4} | 0.5525 |
| λ2·λ3 | 0.4225 (real) | 0.4225 |
| λ2·λ2, λ3·λ3 | 0.4225·e^{±iπ/2} | 0.4225 |

Six distinct products; magnitudes {0.7225, 0.5525(×2), 0.4225(×3)}.

**Registered predictions:**
- **P5a — closure.** Every estimated pole is within 5% of some listed product. PASS iff max nearest-product
  error < 0.05 across the resolved set.
- **P5b — resolvability order.** Products resolve in magnitude order as T grows (slower-decaying = larger
  magnitude resolve first): at moderate T the resolvable subset is {0.7225, 0.5525·e^{±iπ/4}} (3 poles);
  the 0.4225 group resolves only at larger T. PASS iff the resolved subset at each T is the top-|·| prefix of
  the product set (floor-aware), and count is monotone non-decreasing in T.
- **P5c — decoder retirement.** Given closure, latent poles are recoverable as the "square roots" of the
  product set (the generators of the multiplicative closure); rank is demoted to a summary statistic. PASS iff
  the generating latent poles {0.85, 0.65·e^{±iπ/4}} are recoverable from the estimated product poles.

**Hypothesis tagged for later (P3 transient).** Once the readout is pole-based, re-examine whether P3's low-K
transient (rank 1,3,3) is a resolvability artifact rather than a pooling failure — test, don't assume.

Frozen constants unchanged (§10). This supersedes the rank-against-floor readout for the stochastic regime
(spec amendment #5).
