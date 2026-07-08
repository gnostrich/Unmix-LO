# Synergy aggregator — run report (2026-07-08)

Standalone report for the most recent run: the **task-aware synergistic aggregator** test. This was
the fairest remaining shot at the composition thesis — it fixed the flaw that killed the prior
(blind) indexer by making the aggregator *task-aware*, and it moved to *real* biomedical data with
*cold* splits so overfitting could not masquerade as synergy. It ends at the precondition: there is
no complementarity to aggregate.

**Bottom line: precondition P1 FAILS on every real task tested. The aggregator was not built.**

---

## What was being tested and why

Prior result (indextest): a *blind* indexed connective frame cannot beat a strong naive readout —
blind to the label, it surfaces every cross-term indiscriminately and the bloat hurts. The obvious
fix: make the aggregator **task-aware** (select cross-features by relevance to the target, with a
redundancy penalty). The synergy prereg (v2, real-data-only) tests exactly that, but gates it behind
a precondition:

> **P1 — complementarity must exist under a cold split.** best-single ≪ joint-oracle (gap ≥ 0.15
> balanced accuracy), with train/test entities disjoint. If P1 fails, there is nothing for any
> aggregator (cheap or not) to capture → STOP. (This is the BIOMESH-DTI lesson, pre-registered.)

The claim structure: (A) the aggregator matches the joint-oracle at ≥1.10× best-single on a cold
split; (B) it does so at scale-free cost. The novel part is (B) *on a working* (A). But both are moot
if P1 fails — so P1 is the whole ballgame, and it did not pass.

## Setup

- **Encoders (frozen):** ESM-2 8M (protein), ChemBERTa-77M (molecule) — attention-masked mean-pooled
  embeddings, no fine-tuning.
- **Readouts (both, for reconciliation):** an interaction-capable MLP(128,64) and a linear logistic
  probe. Same readout for single and oracle arms → the prereg's capacity control is inherent (any
  gap is cross-model information, not readout capacity).
- **Tasks:** DAVIS drug-target interaction (the canonical composite task) and D-SCRIPT human
  protein-protein interaction (a genuinely *combinatorial* task — a single protein cannot determine a
  pairwise interaction in principle, so complementarity is "forced" if it exists anywhere).
- **Splits:** entity-disjoint cold splits (the established-correct methodology); DAVIS run at all
  three (cold-drug, cold-target, cold-pair); PPI at cold-both.

## Results

### DAVIS DTI — P1 across all three cold splits × both readouts

| split | readout | best-single | oracle | oracle AUROC | gap | P1 |
|---|---|---|---|---|---|---|
| cold-drug | MLP | 0.500 | 0.521 | 0.672 | +0.021 | fails |
| cold-drug | logistic | 0.627 | 0.621 | 0.682 | −0.006 | fails |
| cold-target (shared drugs) | MLP | 0.549 | 0.727 | 0.894 | **+0.177** | *holds — leaky* |
| cold-target | logistic | 0.719 | 0.706 | 0.764 | −0.014 | fails |
| **cold-pair (strictest)** | MLP | 0.500 | 0.518 | 0.609 | +0.018 | **fails** |
| cold-pair | logistic | 0.637 | 0.543 | 0.580 | −0.094 | fails |

### D-SCRIPT PPI — P1, properly powered (10,766 train / 4,306 test cold pairs, 2,000 proteins)

| readout | best-single | oracle | oracle AUROC | gap | P1 |
|---|---|---|---|---|---|
| MLP | 0.747 | 0.797 | 0.889 | **+0.049** | **fails** |
| logistic | 0.741 | 0.777 | 0.861 | +0.036 | fails |

## The one surprise — chased down, not banked

The first pass flagged DAVIS **cold-target** as P1-*holding* (gap +0.177 with the interaction-capable
MLP) — which appeared to contradict the earlier BIOMESH cold-split negative and tempt a "composition
works after all" conclusion. It was treated as a claim to falsify:

- The hold appears **only** on cold-target, where *drugs are shared* between train and test.
- Under the strictest **cold-pair** split (both drug and target unseen), it collapses to +0.018 with
  the oracle at AUROC 0.609 ≈ chance — with **both** readouts.

So the +0.177 was the **shared-drug marginal** (drug-promiscuity memorization), exactly the leak the
strict split exists to catch. It is not complementarity. **BIOMESH's negative stands**, and the
MLP-vs-logistic comparison confirms it was never a linear-probe artifact — the interaction-capable
readout also fails cold-pair.

## Why even a genuinely combinatorial task (PPI) fails P1

PPI was the fair venue: predicting whether protein A binds protein B *inherently* needs both. Yet
best-single already reaches 0.747 balanced accuracy — because high-degree **hub** proteins are
identifiable from sequence, and "is this a hub" predicts "likely interacts" and generalizes to unseen
proteins. The genuine joint-beyond-marginal signal is only **+0.049**. The recurring biomedical
pattern: **entity marginals** (drug promiscuity, protein hub-ness) dominate the cold-generalizing
signal, leaving almost no per-query complementarity for an aggregator to serve.

Corroborating (per prereg): the synthetic probe could not construct valid learnable-naive-hard
synergy either — the oracle was unlearnable and the high-feature arms "won" by overfitting *above* the
oracle ceiling, which is why the prereg mandated real data.

## Verdict and consequence

- **P1 fails on all real tasks tested** (DAVIS cold-pair +0.02, PPI +0.05; both ≪ 0.15), with both a
  linear and an interaction-capable readout.
- Per the frozen decision rule, **STOP** — the task-aware aggregator (A) and its scale-free cost (B)
  were not evaluated, because building them would optimize the cost of a null operation. This is the
  exact trap the P1 gate exists to prevent.
- The negative is confound-controlled and consistent across two encoders, two tasks (one genuinely
  combinatorial), warm and cold splits, and two readout classes.

## Where it sits in the larger picture

This is the **fourth** confound-controlled negative against the composition thesis (after naive
pooling / BIOMESH, settling / VIRTUALMESH G1, and blind indexing / indextest), and the first to fail
at the *precondition* — there isn't even complementarity to capture. Together with indextest it shows
the composition band is empty from both directions: constructed-but-uninhabitable (indextest) and
real-but-absent (this run). Frozen-model composition remains **infrastructure** (G2 scale-free cost,
G3 compression of reachable knowledge), **not new capability**.

Full cross-project synthesis: `../COMPOSITION_THESIS.md` and `../report.md`. Numbers:
`p1_results.json`, `davis_p1_full_results.json`, `ppi_p1_results.json`. Design (frozen before runs):
`PREREG.md`. Runners: `run_p1.py`, `davis_p1_full.py`, `ppi_p1_full.py`.
