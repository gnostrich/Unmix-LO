# GATE ZERO — PRE-REGISTRATION (thresholds frozen before any run)

Frozen 2026-07-08, before embeddings are computed. Per biomesh/README.md discipline: honest
RED is success; no threshold is moved after seeing a number.

## Question
Do biomedical composite queries GENUINELY require multiple specialists? If a single specialist
suffices, a cheap composition layer has no customer (the VIRTUALMESH G1 lesson: on split-knowledge
queries, pooling itself lost to best-single — the composition premise never fired).

## Task (verifiable, labeled, genuinely composite)
DAVIS drug-target interaction (Öztürk et al. DeepDTA data): 68 drugs (SMILES) × 442 kinase
targets (amino-acid sequences) = 30,056 pairs. Label = binder iff pKd >= 7 (Kd <= 100 nM), the
standard DAVIS binarization; prevalence 8.3%. Predicting binding provably needs BOTH a molecule
view (which drug) and a protein view (which target) — the textbook split-knowledge structure.

## Specialists (the two complementary encoders the task needs)
- ESM-2 (facebook/esm2_t6_8M_UR50D), frozen — protein sequence → mean-pooled residue embedding.
- ChemBERTa (DeepChem/ChemBERTa-77M-MLM), frozen — SMILES → mean-pooled token embedding.
Heterogeneous frame (no shared base) — the honest biomedical setting.

## Protocol
- Split: random pair split stratified by label, 70/15/15 train/val/test, seed 0. (Warm setting:
  this makes the single-specialist baselines STRONGER — a protein-only probe can learn per-target
  promiscuity, a drug-only probe per-drug polypharmacology — so it is the conservative direction
  for the composition claim.)
- Probes: L2-regularized logistic regression on (a) protein-only, (b) molecule-only,
  (c) union = concatenated features. Identical probe class for all three (fair).
- Per-probe decision threshold tuned on VAL for balanced accuracy; never on test.

## Pre-committed thresholds (BOTH must hold to PASS)
1. **Composition-needed** — split-knowledge fraction >= 0.30 on a class-balanced test subset
   (equal binders / non-binders, so the metric measures need-for-composition, not easy-negative
   dilution). A test instance is "split-knowledge" iff BOTH single-specialist probes misclassify
   it AND the union probe classifies it correctly.
2. **Pooling helps** — union AUPRC >= 1.10 × best-single AUPRC (composition meaningfully beats
   the best individual specialist). Report AUROC too.

## Decision
- PASS (split-fraction >= 0.30 AND union AUPRC >= 1.10 × best-single) → genuine composite need
  exists; proceed to experiment/ (the cost-vs-scale demonstration).
- FAIL (either condition misses) → the biomedical cluster does not exhibit split-knowledge on
  this task; a composition layer has no customer here. STOP and report as a real negative
  (this is the VIRTUALMESH-G1 recurrence, and publishable).

## Degenerate-win guards
- Report single-specialist AUPRCs explicitly; if a single specialist already saturates
  (AUPRC > 0.8), "union wins" would be marginal and the composite need is weak — flag it.
- Class-balanced denominator is fixed here, before running, precisely so a 92%-negative test set
  cannot inflate or deflate the split fraction post-hoc.
