# G1-real RESULTS — settling vs pooling on real specialists: **FAIL**

Run 2026-07-08 per REAL_PREREG.md (thresholds frozen before the run). Federation: 4 rank-8
LoRA specialists on frozen Qwen2.5-0.5B-Instruct (p2c, c2co, co2pr, p2h), all at 1.00
single-hop accuracy on their own relation. 40 split-knowledge queries (20 two-hop person->
company, 20 three-hop person->product), ground-truth scored. Full numbers in
real/gate1_results.json.

## Pre-registered arms

| arm | accuracy | fact-precision | facts admitted |
|---|---|---|---|
| best single model | 0.15 | — | — |
| POOLING (confidence-weighted vote) | 0.05 | — | — |
| ONE-STEP reconciliation | 0.00 | 0.547 | 64 |
| SETTLING (<=5 rounds to fixed point) | **0.00** | **0.018** | 2,171 |

Pre-registered pass: settling >= pooling x 1.10 AND fact-precision >= 0.8 AND settling >
best-single. **All three conditions fail.** Settling vs pooling relative gain: -100%.

## What actually happened (the characterized negative)

1. **Confidence calibration on real 0.5B specialists is nearly nonexistent.** Calibrated on
   training relations only (per prereg): correct-key vs wrong-type-key mean logprob gaps of
   0.03-0.07 nats (p2h: -0.206 vs -0.211). A LoRA specialist asked about a wrong-type entity
   confabulates an answer of its own relation type at almost its trained confidence.
2. **Iteration amplifies confabulation.** Each admitted junk fact adds an entity to the shared
   state; next round every model confabulates about it. The scratchpad grew to ~54 facts/query,
   98.2% false — a positive-feedback hallucination cascade. One-step admitted 64 facts at 0.55
   precision; five rounds admitted 2,171 at 0.018.
3. **Pooling was ALSO worse than the best single model** (0.05 vs 0.15): three of four
   specialists cannot answer any multi-hop query, and their confident wrong votes drown the
   sometimes-right one. At this scale, both aggregation baselines lose to argmax-model.
4. The sandbox's 1.50x settling win silently assumed **calibrated ignorance** — a model there
   contributed only on dims it truly knew (mask=0 elsewhere). Real small specialists have no
   such mask. That assumption, not the settling algebra, was the load-bearing part.

## Verdict and consequences (per the pre-registered decision rule)

- **G1 FAIL.** The settling/reconciliation law (spec Law G1-B and the graded refinement
  claims gated on G1) is NOT promoted to the spec; it is recorded in the paper as this
  characterized negative; the MVP omits settling.
- The honest residual claim is only: "on split-knowledge multi-hop queries, one-shot pooling
  of small specialists is worse than best-single, and recurrent settling without calibrated
  confidence is worse still — it amplifies hallucination."
- Bound for any future retry: the gate showed the missing precondition is per-fact calibration
  (abstention), not more iteration or a better frame. A retry must gate contributions on a
  verifier at >=0.8 fact-precision BEFORE settling gets another test. That is a different,
  pre-registerable experiment; nothing here licenses building the settling layer.
