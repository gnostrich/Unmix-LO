# Can you compose frozen models into more than the best of their parts?

**A single-session research report. 2026-07-08. Branch `claude/unzip-archive-xov5w0`.**

One question ran through five projects and a dozen pre-registered gates: **can independently-trained,
frozen models be composed into new joint capability?** The answer, established by four
confound-controlled negatives and defended against its own most tempting positives, is **no**.
Composition of frozen models is *infrastructure* — cheaper routing, compression of already-reachable
knowledge — not *intelligence*. This report is the complete ledger.

---

## TL;DR

| # | project | mechanism under test | verdict | the killing fact |
|---|---|---|---|---|
| 1 | UNMIX | reusable operators separable from gradients | **RED** | Real per-task gradients: components unstable (0.64), smeared (0.60), dense reconstruction. Diversity curve replicates but sources aren't reusable primitives. |
| 2 | VIRTUALMESH G1 | settling / models "reasoning together" | **RED** | Without calibrated ignorance, recurrent settling amplifies hallucination — fact-precision 0.018 at 5 rounds. Pooling also lost to best-single. |
| 3 | VIRTUALMESH G2 | routing-memory cost is scale-free | **PASS (scoped)** | Exact short-memory closure; kernel rank flat (120) across N=4→10. Linear regime; rank-at-cap caveat. |
| 4 | VIRTUALMESH G3 | pathway thickening = compression | **PASS (amended)** | Distilled edges cache real 2-hop chains at half cost, bounded exactly by the chain ceiling; fabrication guard refuses junk edges. |
| 5 | BIOMESH | naive pooling of biomedical encoders | **RED** | On the confound-controlled cold split, union is *below* best-single (0.63–0.88×). The in-distribution 1.37× gain was marginal-promiscuity memorization; it inverts under cold split. |
| 6 | indextest | blind indexed connective tissue | **RED** | Even a steelman cross-view bilinear indexer, on complementarity engineered to be maximally favorable, is *worse* than a strong naive readout (0.78–0.93×). |
| 7 | synergy | task-aware redundancy-penalized aggregator | **RED (at precondition)** | Complementarity doesn't survive the strictest cold split on any real task (DAVIS cold-pair +0.02, PPI +0.05, ≪ 0.15). Marginals dominate. |
| 8 | ROUTEMESH | union-without-drag routing (drops ">") | **PASS (scoped)** | A *different* thesis: reach the reachable union, ceiling "=". Light critic realizes the oracle from sparse data (99% of gap); beats a SOTA single-hop router structurally via multi-hop/cyclic assembly (atomic +0.00, multi +0.98, cyclic +0.35) at cost flat in N. Scoped to the constructed disjoint-compositional regime. |
| 9 | THOUGHTWORLD | do frozen models' deviations from a physics seed carry atomic world-structure? | **NOISE (negative)** | Against a gauge-fixing physics engine, two frozen vision encoders' deviations are near-full-rank (eff-rank 16.4/20) and *indistinguishable from the random-fragment control* — no atomic directed structure. The world-model analogue of the composition negatives. |

**Survives:** G2 (scale-free cost) + G3 (compression of reachable knowledge) + ROUTEMESH (union-without-
drag routing, a conditional positive). **Dead:** every attempt to extract *new* capability — pooling,
settling, blind indexing, task-aware aggregation — and now *new world-structure* (THOUGHTWORLD). The line
is sharp: composition buys cheaper and more complete access to **reachable** knowledge (infrastructure),
never **new** knowledge — capability or world-structure (intelligence).

---

## 1. UNMIX — can reusable "operators of learning" be separated from gradients?

The seed project. Thesis: a learned optimizer could accumulate a library of reusable weight-space
primitives, extracted by ICA from the gradients of many diverse trainees. Five toy experiments
established the mechanism (ICA extraction + routed composition ~20× on engineered-compositional
tasks; diversity restores identifiability 0.33→0.78) but also a **negative prior**: on real-ish
neural nets, shared gradient structure is thin and generic.

**The gate (real gradients).** Collected 1,780 real minibatch LoRA gradients from a Qwen2.5-0.5B
federation (9 tasks × 3 genres: code/math/prose), fixed shared checkpoint, then ran the three checks:

- **STABLE** — bootstrap matched cosine **0.814** (borderline).
- **INDIVIDUAL** — max pairwise overlap 0.882, smeared fraction **0.60**: components fire on ~all tasks, not a concentrated few. **FAIL.**
- **REUSED** — held-out reconstruction residual **0.582**, dense (10.7/30 active). **FAIL.**

Notably *not* a Gaussian null (kurtosis 4.4 — ICA had traction) and the diversity→stability curve
*replicated* on real gradients (0.51→0.66→0.81 as genres pool). But what ICA finds is stable
**task-cluster identity**, not reusable **operators**. The world's compositionality does not reach
into weight space the way the thesis needs. → **Robustness reframe**, not the compositional optimizer.

## 2–4. VIRTUALMESH — settling, scale-free cost, thickening

Reframed the question as distributed *inference*: unify frozen specialists into one queryable
"virtual model." Three real-model gates on Qwen2.5-0.5B LoRA specialists (person→city→company→product):

- **G1 (settling > pooling): RED.** Best-single 0.15, pooling 0.05, one-step 0.00, settling 0.00.
  Fact-precision of the settling scratchpad: **0.018** — 2,171 facts, 98% false. The sandbox's 1.5×
  win silently assumed *calibrated ignorance* (models know what they don't know); real 0.5B
  specialists confabulate at near-full confidence, and iteration compounds the hallucination.
  Bonus negative: pooling itself lost to best-single.
- **G2 (MZ kernel is low-rank & scale-free): PASS, scoped.** On real hidden-state settling dynamics,
  an exact short-memory closure (L=2) reproduces settling, and kernel effective rank stays **flat at
  120 across N=4→10** — cost independent of federation size. Scoped honestly: linear channels,
  rank-at-cap (not the stronger atomicity claim), repair-trigger signal did not replicate.
- **G3 (thickening = compression): PASS, amended.** Distilled direct edges are a **perfect functional
  cache** of real 2-hop specialist chains (agreement 1.00 incl. unseen paraphrases), at half the
  inference cost, bounded exactly by the chain ceiling (0.775/0.786). The fabrication guard: a junk
  chain distills to *below* base rate — the mesh refuses to invent structure. Amended after the
  as-preregistered design was fail-by-construction (documented, original kept on record).

Merge discipline: only G2/G3 laws promoted into the spec (scoped/amended, with certificates); G1 laws
moved to a refuted register, not deleted; the MVP demos only the passed mechanism.

## 5. BIOMESH — does naive composition help on real biomedical encoders?

Cashed the G2/G3 wins on frozen biomedical encoders (ESM-2 + ChemBERTa), gated by **GATE ZERO**: are
the queries genuinely split-knowledge? On DAVIS drug-target interaction (30,056 pairs):

- **In-distribution:** union AUPRC 1.37× best-single (looked promising), but per-instance
  split-knowledge fraction 0.005 — composition helps *ranking*, doesn't create only-jointly-solvable
  queries.
- **Cold split (confound-controlled): decisively RED.** Union AUPRC falls *below* best-single in
  every mode (cold-drug 0.88×, cold-target 0.63×, cold-pair 0.69×). The 1.37× gain **inverts** — it
  was drug/protein marginal-promiscuity memorization. Union AUROC stays 0.65–0.76 (no encoder
  collapse), so the verdict is clean. → No composite customer; stop.

## 6. indextest — can *blind* indexed connective tissue surface entangled complementarity?

The deepest test. Two stages, everything committed before it ran.

- **Stage 1 (validity gate): PASS.** 7/12 pre-specified planting configs proved the informative
  regime is blindly constructible (real complementarity, failing strong-naive, reachable oracle).
- **Stage 2 (results test): FAIL, decisively.** A steelman blind cross-view bilinear indexer is
  *worse* than a strong naive readout on every complementarity config (0.78–0.93×), and passes the
  anti-hallucination control (inert, not fabricating). The mechanism is airtight: the indexed feature
  set is a literal **superset** of the naive set, yet loses — because blind to `y`, it surfaces all
  ~100 cross-terms indiscriminately, and that bloat degrades a fixed-capacity readout more than clean
  aligned features fed to a label-informed MLP. **A composer blind to the task cannot beat a readout
  that sees it.**

## 7. synergy — a *task-aware* aggregator on *real* nature-planted synergy

Fixed the indextest blindness (aggregator now uses `y`), tested on real data with cold splits, gated
on precondition **P1** (complementarity must survive the cold split).

- **DAVIS DTI:** P1 appeared to *hold* on cold-target (+0.177) — a surprising positive. Treated as a
  claim to falsify: it vanished under the strictest **cold-pair** split (+0.018), with both linear and
  MLP readouts. It was a shared-drug marginal. BIOMESH stands, and is confirmed not a linear-probe
  artifact.
- **PPI (D-SCRIPT, genuinely combinatorial, properly powered, 15k cold pairs):** P1 **fails** too
  (+0.049). Best-single is already 0.747 — protein **hub-ness** is a marginal a single embedding
  captures; the joint-beyond-marginal signal is thin.
- → P1 fails on every real task; the aggregator was **not built** (per the frozen rule — don't
  optimize the cost of a null operation).

---

## What it all means

The composition band is empty **from both directions**:

- **Constructed-but-uninhabitable** (indextest): where the ideal regime *can* be built, a composer
  blind to the task cannot beat a readout that sees it. The limit is informational, not empirical.
- **Real-but-absent** (BIOMESH, synergy): where the task is *real*, the complementarity isn't there —
  entity marginals (drug promiscuity, protein hub-ness) dominate the cold-generalizing signal, and the
  joint-beyond-marginal residue is < 0.05 balanced accuracy.

**Frozen-model composition is a cost-and-routing story (G2/G3), not a capability story.** You can make
an already-reachable composite computation cheaper (G3) and flatter in cost as you add models (G2). You
cannot conjure joint capability that wasn't already reachable — not by pooling, settling, blind
indexing, or task-aware aggregation.

## Why these negatives are trustworthy

- **Pre-registration**: every threshold frozen and committed *before* its run; design artifacts
  (prereg → family → runners) each landed before their outputs, so nothing could be tuned to a result.
- **Confound control**: cold/entity-disjoint splits throughout; anti-hallucination, fairness
  (equal-feature-count), and capacity controls gated every positive-looking signal.
- **Adversarial toward positives**: the two most tempting wins — BIOMESH's in-distribution 1.37× and
  synergy's DAVIS cold-target +0.177 — were both chased down and shown to be marginal leakage. A
  surprising positive got *more* scrutiny, not less.
- **Failures kept on record**: two NaN-poisoned training runs and one fail-by-construction split were
  caught, fixed, and preserved, not hidden.

## Artifacts (all on `claude/unzip-archive-xov5w0`)

```
report.md                     - this file
COMPOSITION_THESIS.md         - the synthesis
UNMIX (root)                  - GATE_RESULTS.md, gate/, experiments/, src/extractor.py
virtualmesh/                  - gates/ (G1/G2/G3 results + prereg), spec/, paper/, mvp/
biomesh/                      - gate0/ (in-dist) + gate0cold/ (confound-controlled) + GATE0*_RESULTS.md
indextest/                    - PREREG, FAMILY, Stage-1/2 runners + INDEXTEST_RESULTS.md
synergy/                      - PREREG, P1 runners (DTI + PPI) + SYNERGY_RESULTS.md
```

Per-gate RESULTS.md files carry the full numbers; every JSON output is committed alongside its runner.
