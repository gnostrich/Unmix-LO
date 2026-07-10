# PRE-REGISTRATION — the I/O stream is the track: MZ closure ON the stream, read fit-free (frozen before run code)

Direction (high-level, from the thread): in the Basin you read the *tracks* and build the machinery right upon
them. Here the "track" is **the model's I/O stream** — outputs over a given input distribution — **not the model
itself**. The model (the "virtual thing") stays a black box; the machinery is built on its I/O. Mori–Zwanzig is
the reason this is well-posed: MZ **is** projection onto the observed variables. The resolved variable here IS
the I/O stream; the model's internal computation is exactly what gets projected out. MZ's closure theorem says
the resolved stream then obeys a closed law with a **memory kernel**, and that kernel **is the trace of the
projected-out internals**. So the memory "somehow converges to MZ memory" is not a metaphor — it is forced by
the projection, IF the machinery is honest. This experiment tests exactly that, poles-first.

## The object
- **Track**: an I/O stream `(u_t, y_t)` — inputs drawn from a known distribution (the "training data" role),
  outputs from a black-box generator (the "model" role). We never open the generator.
- **Machinery built on the stream** (nothing fitted, no read-out head, no train/test regression):
  1. Memory response from the stream itself: `ĥ_k = (1/T) Σ_t y_{t+k} u_tᵀ` (input-output cross-correlation;
     for white input this IS the Markov/memory sequence of the closure). `h_0` (the instantaneous map) is set
     aside; **memory = k ≥ 1**.
  2. Block-Hankel of `ĥ_{1..}` → singular spectrum → **atoms above a self-calibrating noise floor** = the
     memory's atomic support. Realization (Ho-Kalman) of the above-floor part → the kernel's **poles**.
  3. **Noise floor is a permutation null, not a knob**: circularly shift / shuffle the output stream in time,
     recompute the Hankel top singular value; the floor is a quantile of that distribution. Destroying temporal
     structure with the stream's own marginals is the natural second-FDT floor — self-calibrated, no free
     hyperparameter.
- **The natural read**: the recovered atomic support (count + pole locations) of the stream's memory kernel.
  Fit-free — the read is correlation + spectrum of the object's own closure; there is nothing to train.

## Poles-first ground truth (all generators hidden from the machinery; only streams observed)
1. **ATOMIC memory, rank r** — hidden linear state-space `x_{t+1}=Ax_t+Bu_t, y_t=Cx_t (+ noise)` with known
   McMillan degree r and known poles = eig(A), for r ∈ {2, 3, 4, 6}.
   **Frozen prediction:** recovered order = r (±1) and recovered poles match eig(A) (small matching error).
2. **MEMORYLESS generator** — `y_t = M u_t + noise` (a static map, no hidden state).
   **Frozen prediction:** recovered order = 0 (all memory Hankel mass below the permutation floor).
3. **PERMUTATION NULL** — any stream with outputs shuffled in time.
   **Frozen prediction:** order = 0 by construction of the floor (validity check that the floor works).
4. **CONTINUOUS-SPECTRUM memory** (dense pole continuum, no atomic support) — the negative arm of the atomicity
   dial. **Frozen prediction:** recovered order does NOT terminate at a stable small integer — it drifts up as
   T grows (floor drops) and shows no clean spectral gap.

## Frozen verdict
- **CONVERGES-TO-MZ-MEMORY (the claim holds)** iff: (1) atomic generators' order AND poles are recovered from
  the stream alone across r; (2) memoryless reads 0; (3) the continuous control does NOT cleanly terminate.
  All three — recovery without discrimination is fabrication; discrimination without recovery is deafness.
- Anything else: report which arm failed, plainly. Honest failure = the "converges to MZ memory" intuition is
  not realized by this machinery, and that is a result.

## Scope-lock
Local: does the MZ closure built on an I/O stream recover the generator's memory structure fit-free, on
synthetic ground truth. It is not the resolvent conjecture, not the auction, not a claim about any real LLM
(no real-model I/O with logged provenance is in this session — per trace_read, that remains UNTESTABLE, not
NOISE). The region-to-region **relative trace** (two streams on shared inputs) is the stated next step and is
out of scope for this prereg. Same words across threads ≠ same object; no cross-thread reconciliation.

## Discipline
Prereg frozen before run code. The floor is self-calibrated (permutation null), no free knobs. A surprising
positive gets more scrutiny (vary T, noise, seeds). Honest nulls are results. Report all arms' numbers.
