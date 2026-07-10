# RESULTS — the I/O stream as the track: does its closure converge to the MZ memory? (honest)

Prereg frozen before run code (commit `159587e`). The direction under test: the "track" is the model's **I/O
stream**, not the model; build the trace machinery **on the stream**; the memory kernel should **emerge from
the MZ closure**, read fit-free (no trained read-out head, no free knobs — the noise floor is a permutation
null computed from the stream's own marginals). Poles-first, synthetic generators, machinery never sees them.
Reproduce: `python run.py` (frozen arms) — post-hoc sweep in `posthoc_T48k.json`.

## Frozen verdict first, plainly: **FAILS at the frozen T** (Arm 1), with Arms 2–3 clean
At the pre-registered stream length T = 12,000:

| arm | frozen prediction | outcome |
|---|---|---|
| 1 — atomic hidden memory (r = 2,3,4,6) | order = r (±1) AND pole err < 0.1, all seeds | **FAILS**: r=2 exact (err ≤ 0.01); r=3,4,6 undercount by 1–2 with pole err up to 0.27 on some seeds |
| 2 — memoryless generator + shuffled-atomic validity | order = 0 | **PASSES**: 0 everywhere — the floor never fabricates memory |
| 3 — continuous spectrum (no atomic support) | must NOT terminate; drifts with T | **PASSES**: order 4→6→7 as T grows 3k→48k, gap stays ~1.5–2 |

By the frozen all-three rule, **the claim as pre-registered does not hold at T = 12k.** That is the verdict of
record.

## Diagnosis (labeled post-hoc): the misses are weak poles under an honest floor — and recovery CONVERGES
- **What was actually planted:** the random stable generators contain near-zero poles (|λ| = 0.052, 0.099,
  0.17, 0.29 in exactly the missed cells). A |λ|≈0.1 mode's memory decays in one step; at T=12k its Hankel mass
  sits genuinely below the finite-sample permutation floor. The estimator is being **conservative, not deaf**.
- **Convergence in T (the direction's actual claim — "converges to MZ memory"):**
  - T = 12k: 7/12 atomic cells recovered.
  - T = 48k: **11/12** (r=2: exact; r=3: exact; r=6: within ±1, pole err ≤ 0.099; only r=4 seed 2 — weakest
    pole 0.29 — still short).
  - T = 192k (worst r=6 cell): **exact** — order 6, pole err 0.007 (from order 4, err 0.177 at 12k).
- **Direction of error is one-sided:** across every cell and every T, the read **undercounts weak memory and
  never over-counts** — memoryless and shuffled streams read 0 at all T; the continuous control never
  terminates. No fabrication anywhere.

## What this adds up to (both statements, no blending)
1. **The pre-registered fixed-T criterion failed.** Frozen prediction was too strong for streams whose
   generators carry near-floor poles; that is a fact about the frozen test and it stands.
2. **The convergence property the direction pointed at is supported**: the closure built on the I/O stream
   alone — fit-free, self-calibrated floor — recovers the hidden generator's memory **order and pole
   locations** as the stream grows, degrades gracefully (undercounting weak modes) when data is short, and
   discriminates memoryless / no-atomic-support streams perfectly. "Somehow converges to MZ memory" is, on
   this ground truth, literal: it converges, monotonically, from below.

## The natural read, as realized here
Query-side nothing was fitted: memory sequence = input–output cross-correlation of the stream; atoms = Hankel
singular values above a **permutation-null floor** (the stream's own marginals define "no temporal structure");
poles = Ho-Kalman on the above-floor part. No training, no train/test regression, no hand-set threshold.

## Scope
Local, synthetic, poles-first. Says nothing about any real LLM (no provenance-logged real I/O in this session —
that remains UNTESTABLE, per `trace_read/`). The region-to-region **relative trace** (two streams on shared
inputs) is the stated next step, unbuilt. Not the resolvent conjecture, not the auction; same words across
threads ≠ same object.
