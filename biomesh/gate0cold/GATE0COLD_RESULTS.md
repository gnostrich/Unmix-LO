# GATE ZERO (cold-split) RESULTS — confound-controlled re-test: **FAIL, decisively**

Run 2026-07-08 per PREREG.md (frozen before the run). Full numbers in gate0cold_results.json.
This is the fair version of the in-distribution gate (entity-disjoint splits remove the
marginal-promiscuity leakage). The negative is now confound-controlled — and it is stronger than
the in-distribution one, not weaker.

## Verdict per split mode (both conditions frozen: split-frac >= 0.30 AND union/best-single AUPRC >= 1.10)

| mode | union AUROC | AUPRC: prot / mol / union | best-single | union/best ratio | split-frac | verdict |
|---|---|---|---|---|---|---|
| cold-drug   | 0.653 | 0.198 / 0.147 / 0.175 | protein 0.198 | **0.88** | 0.000 | FAIL |
| cold-target | 0.758 | 0.081 / 0.286 / 0.179 | molecule 0.286 | **0.63** | 0.011 | FAIL |
| cold-pair   | 0.549 | 0.106 / 0.165 / 0.114 | molecule 0.165 | **0.69** | 0.005 | FAIL |

**OVERALL (cold-drug AND cold-target): FAIL.** Both pre-committed conditions miss in every mode.

## The decisive finding: composition doesn't just fail to be necessary — it HURTS
On the fair split, **union AUPRC is BELOW best-single in all three modes** (ratios 0.88 / 0.63 /
0.69, all < 1.0). Naive concatenation lets whichever specialist faces unseen entities drag down
the one that doesn't:
- cold-drug: the molecule encoder can no longer memorize per-drug promiscuity → molecule-only
  collapses (AUROC 0.79→0.59); union is dragged below protein-only.
- cold-target: the protein encoder faces unseen targets → protein-only near chance (0.569);
  union is dragged below the still-strong molecule-only.
This is the VIRTUALMESH-G1 "pooling loses to best-single" bonus-negative, now reproduced under
the confound-controlled split on real biomedical encoders.

## Encoder-collapse guard (PREREG.md): NOT triggered — the verdict is clean
Union AUROC stays 0.653 (cold-drug) and 0.758 (cold-target), both >= 0.6. The frozen encoders
retain real signal on unseen entities — enough that a composition benefit, if it existed, would
show. It does not. So this is a genuine split-knowledge verdict, not a "frozen encoders don't
transfer cold" artifact. (Protein-only does hit chance on unseen targets, 0.499–0.569; that
degradation is real but the union still has signal, so the attribution holds.)

## In-distribution vs cold-split, side by side
| quantity | in-distribution | cold-drug | cold-target |
|---|---|---|---|
| union / best-single AUPRC | **1.37×** | 0.88× | 0.63× |
| split-knowledge fraction | 0.005 | 0.000 | 0.011 |
| both-singles-wrong rate | 0.051 | 0.110 | 0.103 |
| single error correlation | −0.18 | −0.22 | −0.08 |

Removing the confound RAISED the both-wrong rate (0.05→0.11, singles are genuinely weaker) but
the union still fails to rescue those cases (rescue rate 0.0–0.11), so the split fraction stays
~0. And the in-distribution 1.37× aggregate gain **inverts** to <1.0 — confirming that gain was
itself largely marginal memorization, not composition.

## What this decides
1. **The split-knowledge premise is dead on this task, confound-controlled.** No per-query need
   for multiple specialists survives the fair split. The in-distribution negative was not a
   metric artifact — it is real and sharper under the correct methodology.
2. **The option-3 re-scope ("cheaper aggregate accuracy") is also undercut.** There is no
   aggregate accuracy gain to deliver cheaply: composition is below best-single on the fair
   split. A cost-vs-scale layer would be optimizing the cost of a negative-value operation.
3. **STOP is now decisive**, not provisional. The honest, publishable finding: on DAVIS DTI with
   frozen heterogeneous encoders (ESM-2 + ChemBERTa), specialist composition provides no
   per-query split-knowledge and no aggregate-accuracy benefit once marginal leakage is removed;
   the apparent in-distribution benefit was promiscuity memorization.

## Honest scope of the negative
One task (DAVIS), one small encoder pair (ESM-2-8M + ChemBERTa), random entity-disjoint cold
splits (not scaffold/cluster — a stricter variant not run here; it would only make the split
harder, not easier, so it cannot rescue the positive). Larger encoders or a genuinely
multi-modal task (where no single input can in principle determine the label) could differ — but
DTI, the canonical "needs both drug and protein" task, does not exhibit the property. That is the
result.
