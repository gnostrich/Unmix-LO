# EXPERIMENT — the cost-vs-scale demonstration (only runs if GATE ZERO passes)

## The single claim to demonstrate
At EQUAL task accuracy, the kernel-routing composition layer's inference/coordination cost stays
FLAT as the number of specialists N grows, while orchestration and pooling baselines' cost grows.
(This is the G2 property — scale-invariance of the routing memory — cashed as a real advantage.)

## Method (the BIOMESH layer — G2/G3 only, NO settling)
1. **Frame alignment**: align each frozen specialist encoder into a shared space via a small anchor
   set (relative-representations / a few hundred shared probe entities run through all specialists).
   MEASURE the alignment cost per specialist — this is the honest maintained-artifact cost.
2. **Kernel routing (G2)**: build the low-rank MZ-style routing memory over the aligned specialists;
   a query engages a routed subset; per-query cost governed by kernel rank, NOT by N. Verify rank
   stays flat as N grows (replicate G2's flat-rank finding on real biomedical encoders — this is also
   the G3-heterogeneous test, since these models don't share a base).
3. **Edge compression (G3)**: for frequently-used transitive specialist paths, distill direct edges
   (cache composite at lower cost). Bounded by the ceiling — compression only, verify it doesn't
   fabricate (guard from G3).
4. Decode to the task answer.

## Baselines (baselines/ — must be FAIR: same specialists, same task, same accuracy target)
- **best-single**: the strongest individual specialist (floor).
- **static pooling (BioVERSE-style)**: align all specialists once, pool features, decode. Cost grows
  with N (must process all N every query).
- **agentic orchestration (Het-MedAgent-style)**: an LLM calls each specialist as a tool and stitches
  outputs. Cost grows with N (a call per specialist) and adds LLM-orchestration overhead.

## The measurement + PRE-COMMITTED
Sweep N = 2,3,4,5,6 specialists. For each, at a fixed task-accuracy target, measure:
- **accuracy** (all methods must hit the same target — a cheaper method at lower accuracy is NOT a win)
- **cost**: per-query FLOPs / calls / latency AND coordination/memory cost vs N.
PASS iff: BIOMESH matches baseline accuracy AND its cost curve vs N is FLATTER (sub-linear where
baselines are linear-or-worse), with the crossover N stated (beyond how many specialists BIOMESH wins).
Report the alignment cost honestly as the upfront price.

FAIL iff: BIOMESH cost also grows with N (G2 didn't transfer to real encoders), OR it only matches
baseline cost, OR it needs accuracy sacrifice to be cheaper. Then the scale-free advantage is not real
on biomedical encoders — report as a scoped negative.

## Honesty clauses (keep in the writeup)
- Claim ONLY the cost-vs-scale advantage at equal accuracy. NOT new capability (G3 ceiling). NOT reasoning (G1).
- These baselines (BioVERSE, Het-MedAgent) are 2025-26 — read them directly; confirm neither already
  has an N-independent-cost composition (the distinctive axis) before claiming it's unoccupied.
- If GATE ZERO showed pooling loses to best-single, the honest framing shifts to "cheap composition
  where composition helps" — do not overclaim a general biomedical win.
