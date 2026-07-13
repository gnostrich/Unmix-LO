# WALL: the 2×2 mechanism (atom/geometric) leg does not instantiate as designed

Reported per the covenant (blocker honestly reported > patched green result). The 2×2 was NOT run; its
mechanism leg premise failed calibration. No prereg was committed for a leg that doesn't hold.

## The wall
The mechanism-side readout "atom count = spatial complexity of the shared geometry" does **not** cleanly track
any geometric-richness knob I could construct. Atom count hovers at 3–4 across:
- **4 geometric knobs**: random clusters, orthogonal equal-scale clusters, rank-g subspace probe, multiscale
  (g distinct distance scales). Sample: orthogonal clusters g=2,3,4,5 → atoms 3,3,4,4; multiscale
  scale-count 4,6,8,8 → atoms 3,3,3,3; rank-g → 9,6,4 (over-grows on continuous/degenerate spread).
- **3 ε values** (0.02/0.04/0.08) → 3,4,3,3 / 3,4,4,4 / 3,3,4,3. No ε makes it track g.
- **discrete and continuous readouts**: FW atom count (above); equilibrated anchor D_e effective rank
  (~5.0,5.0,5.26,5.25 — flat); induced-mass participation ratio (2.77,5.37,5.28,6.49 — noisy, not clean).

## Diagnosis (localized, not hand-wavy)
- **Not the model.** The pre-logits tap preserves input geometry (input eff-rank 2.53 → tap eff-rank 2.44).
  The geometry reaches the anchor.
- **Atom count is set by the (ε, rel_tol) operating point, ~independent of geometric content.** Under entropic
  blur the anchor needs only ~3 atoms to match a smoothed distance distribution regardless of g. First-
  principles: GW self-sizing responds to the DISTANCE-DISTRIBUTION structure, which my cluster/dimension/scale
  knobs did not move in the way I assumed (equidistant clusters are bimodal for any g; multiscale scale-count
  didn't map to atom count either). The re-derived "distance-scale" knob also failed — so the wall is not just
  "wrong knob," it is that no simple geometric parameter drove the readout.

## Honesty correction (I over-claimed earlier — retract cleanly)
The prior "K-invariant self-sizing: 3,3,3 atoms across K=2,3,5" — the FLATNESS in K is real and non-trivial
(more members do not inflate the count; the bare detector did, 2.2→3.2). But the VALUE ~3 is consistent with
the operating-point floor, so interpreting 3,3,3 as *meaningful sizing to the shared geometry's spatial
complexity* is **UNSUPPORTED**. Corrected ledger status:
- [holds] atom count is K-invariant (flat, does not inflate with members) — a real pooling property.
- [open / not demonstrated] atom count meaningfully tracks geometric complexity. The wall above is the
  evidence against the strong reading.

## Update: the proposed geometric re-scoring ALSO does not hold (planted-rank control)
A later proposal was that atom count is fine as a *spatial-complexity* meter — it should respond to GEOMETRIC
manipulation even if not to content — with a control: planted rank 2 vs 6 → atoms ~2 vs ~6. Ran it (discrete
rank-r clusters, K=3 members): **r = 2,4,6 → atoms 3,3,3** (rel_tol 0.02) / **3,2,2** (rel_tol 0.05). Atom
count does NOT track planted geometric rank either — it is robustly ~3, operating-point-dominated. So the
re-scoring is not validated: FW atom count tracks neither content diversity NOR planted geometric rank on
these clouds; flat-at-~3 is the honest reading (the distance distribution of r equidistant clusters is bimodal
for any r, so GW self-sizing sees ~constant structure). The atom criterion stays a documented null.

## RESOLUTION: the readout was wrong, not the mechanism (parallel control sweep)
Three independent synthetic-cloud controls (`experiments/atom_geom_controls.py`, `atom_operating_point.py`,
`atom_observable_search.py`) settle the atom question:
- **The discrete atom COUNT tracks no geometric parameter** — flat/anti-tracking across continuous rank,
  cluster multiplicity, and hierarchical distance-scale depth (Spearman −0.80 vs planted rank); it is set by
  the (ε, rel_tol, max_atoms, n) operating point (op-point range dominates the geometry gap, and the gap is
  even *negative* — higher rank buys fewer atoms because continuous spread over-grows at low rank then pins).
- **The rate–distortion knee** (the fork-memo candidate: GW cost vs anchor budget, elbow = complexity) **also
  FAILS** — floor-pinned at m=3 for every rank (Spearman 0). Wall-option "RD knee" is dead.
- **But an F-derived readout DOES track planted rank, cleanly: the equilibrated anchor's D_e effective rank at
  fixed budget** (participation ratio of D_e's singular values, m=12) — **Spearman +1.00** (4.67→6.04 as
  r=2→8), matching the pure-geometry Gram effective rank (sanity: the complexity is present and recoverable).

**Conclusion.** Spatial complexity is an INSTRUMENT read off the equilibrated anchor geometry (D_e effective
rank), NOT the discrete self-sizing atom count — which is a floor-dominated mechanism artifact. This fits the
mechanism/instrument split exactly: the atom count self-quenches (mechanism, F-driven) but does not meter
complexity; the D_e spectrum meters it (instrument). The 2×2's spatial leg should be re-specified on the
D_e-effective-rank readout, and "atom count = spatial complexity" is formally retired.

## Consequence for the 2×2
The submission-spine 2×2 as designed has a clean TEMPORAL leg (pole closure, P5 [proven]) but its SPATIAL
(mechanism/atom) leg is not instantiated. Options (manager's call — a genuine design fork, not something to
patch around):
1. **Different mechanism observable.** Find an F-derived quantity that provably tracks a geometric invariant.
   Continuous anchor readouts tried and failed here; a principled candidate is untried: the GW *transport
   cost* at equilibrium as a function of anchor budget (an information-curve / rate–distortion readout), whose
   knee could define spatial complexity. Needs derivation.
2. **Re-examine the claim.** The wall may be saying atom count is simply not a spatial-complexity meter — in
   which case the corrected G1 is a ONE-axis dissociation (temporal/pole leg only), and "atom count = spatial
   complexity, K-invariant" is downgraded to "atom count is K-invariant," full stop.
3. **A geometry knob that provably moves the distance distribution** in the way GW self-sizing reads — requires
   first deriving what functional of the distance distribution the FW count actually tracks, then constructing
   the knob to move exactly that. One session of derivation, pre-registered.

I did not run the 2×2 or pre-register the atom leg, because pre-registering a prediction for a readout that
does not track its intended axis would be the forbidden move.
