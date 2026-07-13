# Corrected G1 — the 2×2 selective-dissociation (pre-registered BEFORE running)

Post mechanism/instrument split, "rank" is two quantities with two instruments. G1 is therefore a 2×2 (three
knobs), each instrument selectively sensitive to its own axis:

| knob ↑ | atom count (FW / F mechanism) | pole count (covariance-Hankel instrument) |
|---|---|---|
| **within-prompt geometric richness g** (spatial structure per cloud) | **grows** | flat |
| **across-prompt dynamical diversity r** (latent modes over the prompt stream) | flat | **grows** |
| **K** (number of members) | flat [3,3,3 validated] | flat (to check) |

Two independent knobs on the substrate:
- **g** — the probe cloud has g spatial clusters; more clusters = richer per-prompt geometry. Averaged over a
  window (zero-mean temporal warp), the shared geometry has ~g-block structure → FW self-sizes to ~g atoms.
- **r** — the latent trajectory u(t) has temporal McMillan degree r; it warps the clusters over prompts. The
  invariant-moment series inherits temporal degree ~sym-power(r) → pole instrument reads it. Zero-mean warp
  ⇒ r does not change the averaged geometry ⇒ atom count flat in r.

## Registered predictions (3×3 grid g,r ∈ {2,3,4}, K=3; plus K sweep at g=r=3)
- **P6a — atom count tracks g, not r.** atom_count monotone increasing across g at each r; spread across r at
  each g ≤ 1. Selective: corr(atom_count, g) strongly positive, corr(atom_count, r) ≈ 0.
- **P6b — pole count tracks r, not g.** resolved-pole count monotone non-decreasing across r at each g; spread
  across g at each r ≤ 1 (up to resolvability). Selective: corr(pole_count, r) positive, corr(pole_count, g)
  ≈ 0. (Pole count = resolved poles of the covariance Hankel of the invariant moments; resolvability shadow
  per P5 may cap absolute values, so the test is monotone-in-r and flat-in-g, not exact integers.)
- **P6c — K flat on both.** atom_count and pole_count each flat (spread ≤ 1) across K ∈ {2,3,5} at g=r=3.

PASS (corrected G1) = each instrument sensitive to its own axis and flat on the other two. This is a STRONGER
headline than v1's 1D leg: a two-axis selective dissociation. Frozen constants unchanged (§10). If P6b is
non-monotone, that is the pole-count-readout's resolvability limit (diagnose via P5's floor-aware subset),
logged, not tuned away.
