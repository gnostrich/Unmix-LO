# INDEXING-VALUE TEST — RESULTS: Stage 1 PASS (validity), Stage 2 **FAIL** (decisive)

Run 2026-07-08. Design frozen and committed before each run (PREREG.md, FAMILY.md, run_stage1.py,
run_stage2.py all pushed before their outputs). This was the final composition-thesis test; it
closes it.

## Stage 1 — validity gate: PASS (7/12 configs constructible)
Before comparing indexing, we proved the informative regime is blindly constructible: for each of
12 pre-specified planting configs, is complementarity real (oracle − single ≥ 0.15), does a strong
naive fail (oracle − naive ≥ 0.10), is the oracle reachable (≥ 0.80)? Seven configs satisfied all
three. **But** the built-in diagnostic already showed the catch: on invertible-gauge configs
`naive_strong / entangled_oracle = 1.09` — the strong naive baseline already extracts the maximum
any readout can pull from the entangled features. The "valid" configs were valid for two reasons:
lossy gauges (E4, info actually destroyed) or a fixed-capacity *learnability* gap in the rotated
basis (E2/E3) — neither of which leaves room for a blind indexer.

## Stage 2 — results test: FAIL (indexing is inert-to-harmful; it does not hallucinate)

| config | target/gauge | single | naive-strong | indexed | oracle | indexed/naive |
|---|---|---|---|---|---|---|
| c4 | xor / E2 | 0.610 | 0.765 | 0.628 | 0.909 | **0.82×** |
| c5 | xor / E3 | 0.521 | 0.579 | 0.540 | 0.909 | **0.93×** |
| c8 | gate / E2 | 0.591 | 0.744 | 0.579 | 0.905 | **0.78×** |
| c9 | gate / E3 | 0.510 | 0.589 | 0.525 | 0.905 | **0.89×** |
| c2 | additive / E4 (lossy) | 0.677 | 0.760 | 0.733 | 0.954 | 0.96× |
| c6 | xor / E4 (lossy) | 0.522 | 0.627 | 0.614 | 0.902 | 0.98× |
| c12 | gate / E4 (lossy) | 0.507 | 0.625 | 0.597 | 0.912 | 0.96× |

Frozen verdict conditions:
- (a) indexed ≥ 1.15× naive on ALL complementarity configs → **FALSE** (indexed is *below* naive,
  0.78–0.93×, everywhere).
- (b) indexed approaches oracle → **FALSE** (indexed *moves away* — closes −58% of the naive→oracle
  gap on average).
- (c) no hallucination on the no-complementarity controls → **TRUE** (indexed ≤ naive on every
  control, 0.96–0.99×).

**STAGE 2 FAIL** on (a) and (b). Indexing is inert-to-harmful, not hallucinatory.

## Why indexing loses — the airtight mechanism
The indexed feature set is a strict **superset** of the naive set (whitened views + all cross-view
products), and it still loses. Because the connective frame is built **blind to y**, it exposes
*all* ~100 cross-view products indiscriminately; a fixed-capacity readout does worse on that
bloated, mostly-irrelevant feature set than a label-informed MLP does on clean aligned features.
The within-view-polynomial fairness control confirms it is not a capacity artifact — degree-2
features in general hurt here (0.49–0.60). The strong naive baseline's decisive advantage is
simply that its readout **sees y** and the indexer's frame does not: without task signal, a
connective frame cannot preferentially surface the task-relevant cross-terms, and surfacing all of
them is worse than letting a capable readout learn the few that matter from clean features.

The one place indexed ≈ naive (lossy E4 configs) is where the information was destroyed by the
gauge — so *nothing* can win, indexer included.

## Verdict
The narrow band — present-but-entangled-but-recoverable complementarity a strong naive cannot
reach — is **not inhabited by a blind indexer**. Even in the ideal constructed regime, engineered
to be maximally favorable, indexing does not beat strong naive; it is worse. Per the frozen rule,
STOP. The composition thesis is closed. (See ../COMPOSITION_THESIS.md for the full synthesis across
all three attack surfaces.)

## Honesty confirmations (per BRIEF.md)
- Naive was genuinely strong, not a strawman: it beats best-single substantially (e.g. 0.765 vs
  0.610 on c4) and closes a real fraction of the oracle gap.
- Anti-hallucination guard passed: indexing did not beat naive on any no-complementarity control.
- Prior negatives kept on record: BIOMESH cold-split (composition hurts on the fair split),
  VIRTUALMESH G1 (settling amplifies hallucination).
