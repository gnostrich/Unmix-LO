# PRE-REGISTRATION — real-model gates (committed BEFORE any real run)

Frozen 2026-07-08, before the first specialist finishes training. Thresholds below are final;
if a measurement design turns out broken (as happened twice in the G3 sandbox), the fix and the
reason are documented and the threshold stays.

## Common setup
- Base: Qwen2.5-0.5B-Instruct, CPU box. Specialists = rank-8 LoRA finetunes of the SAME frozen
  base (shared frame = shared base/tokenizer; frame cost ~0 — report this as the frame artifact).
- Synthetic relational world (ground truth fully known, so honest scoring is possible):
  entities person -> city -> company -> product, plus person -> hobby as an unrelated relation.
  Specialists: M_pc (person->city), M_cc (city->company), M_cp (company->product),
  M_ph (person->hobby). Each trained ONLY on its relation's single-hop QA pairs.
- Split-knowledge queries: 2-hop and 3-hop compositions never seen by any single model
  ("Which company is based in the city where X lives?"). Ground-truth scored, exact match.

## G1-real — settling vs pooling
Arms (identical query set, identical models):
  1. best single model (floor)
  2. POOLING: each model answers the full query one-shot; confidence-weighted vote.
  3. ONE-STEP: one round of confident contributions to a shared scratchpad, then one answer
     pass — reconciliation WITHOUT iteration (the ablation the sandbox flagged).
  4. SETTLING: iterate contribute->reconcile on the shared scratchpad to fixed point (<=5
     rounds); a model contributes an assertion only when its answer confidence (mean token
     logprob) clears a fixed threshold set on TRAINING relations only, never tuned on queries.
- PASS iff SETTLING accuracy >= POOLING accuracy * 1.10 (>=10% relative), on >=40 split queries.
- Report SETTLING vs ONE-STEP verbatim; if equal, the claim shrinks to "reconciliation > pooling".
- Degenerate-win guards: report hallucinated-fact rate entering the scratchpad; a settling win
  driven by confidently-wrong chains (accuracy up but fact-precision < 0.8) is a FAIL.

## G2-real — MZ kernel on real-representation settling dynamics
- Nodes = specialists' frame-aligned representation spaces (hidden-state embeddings of shared
  probe inputs at a fixed layer); channels = ridge maps between spaces (as in G3 sandbox but on
  REAL model geometry); settling = damped coupled iteration x_i <- a*x_i + (1-a)*mean_j W_ji x_j
  + input, run to T=40.
- Fit linear MZ closure on a tracked subset K over history length L (exact sandbox protocol).
- PASS iff (a) some L <= 8 gives closure rel-error < 0.15, AND (b) kernel eff-rank grows with K
  (routed width) and stays flat (+/-2) as N grows from 4 -> 8-10 nodes (add layers/models as nodes).
- Also report residual-vs-difficulty correlation (>0 expected; not a pass condition).

## G3-real — pathway thickening on real specialists
- Thicken: distill the chain M_pc -> M_cc (person->company, 2 hops of REAL model calls, using
  M_pc's answer as M_cc's input) into a direct LoRA edge trained ONLY on chain pseudo-labels
  from a TRAIN split of persons.
- Q1 PASS: distilled edge matches the chain's answers on held-out persons >= 90% (tolerance).
- Q2 PASS: on held-out persons, distilled-edge GROUND-TRUTH accuracy beats (i) the frozen base
  with the same prompt (no-edge) and (ii) a LoRA trained on shuffled pseudo-labels (random-edge)
  by >= 20% relative each.
- Q4 GUARD: identically distill the broken chain M_ph -> M_cc (person->hobby feeding a
  city->company model — no real path). Its held-out ground-truth accuracy must NOT beat
  no-edge by > 5% relative. If it does, the pipeline fabricates; G3 FAILS regardless of Q1/Q2.
- The chain's own ground-truth accuracy is reported as the ceiling (the bound analog).

## Decision rule (per virtualmesh/README.md)
Each gate PASS -> promote to spec (validated law) + paper (result) + MVP (feature).
Any FAIL -> drop from spec, report as characterized negative in paper, omit from MVP.
Honest RED is a success of the process.
