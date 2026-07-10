# RESULTS — Qwen's swirl: atomic or noise? (velocity-withholding, N=1200, honest)

Pre-registration frozen in `PREREG.md` before run code (commit `c5e3732`). Fresh engine (`engine.py`) and fresh
instrument (`instrument.py`); nothing external. One local experiment: is `Qwen2.5-0.5B-Instruct`'s deviation
from the physics seed (the "swirl") a genuine atomic world-structure object, or noise / prompt-arithmetic?
Full N = 1200 states. Reproduce: `python run.py --n 1200`.

**ATOMIC iff `eff_rank < 0.4·D (=8)` AND `heldout_R2 ≥ 0.3`** (D = target dim = 20). Otherwise NOISE.

## Results (N = 1200) — readout_R2 reported first

| fragment | readout_R2 | heldout_R2 | eff_rank (D=20) | verdict |
|---|---|---|---|---|
| **NULL** (gaussian, state-independent) | 0.081 | −0.047 | 18.05 | **NOISE (valid)** |
| **V+ Qwen** (velocities given) | 0.520 | 0.435 | **16.92** | **NOISE** |
| V+ linear baseline (pos+vel) | 0.864 | 0.856 | 10.12 | NOISE |
| **V− Qwen** (velocities withheld) | 0.378 | 0.254 | **16.06** | **NOISE** |
| V− linear baseline (pos only) | 0.398 | 0.413 | 10.12 | NOISE |

## Verdict (pre-committed): NOISE
Qwen's swirl is **not atomic — in either arm.** The load-bearing failure is `eff_rank`: the deviation matrix
sits at **16.9 (V+) / 16.1 (V−)** out of D=20 — spread across nearly all state directions, nowhere near the
`< 8` atomic bar. The swirl is structureless in both arms.

## The instrument does not fabricate (null valid)
The state-independent Gaussian null reads **NOISE** (heldout R² −0.047, eff_rank 18.05 ≈ full): with no world
information the readout cannot generalize and the residual is full-rank. The instrument declares nothing
atomic from noise, so the ATOMIC/NOISE calls above are trustworthy. (A random *linear projection of the raw
state* is **not** a valid null — it linearly preserves the near-linear physics — so the null is
state-independent random features. Noted for honesty.)

## readout_R2 first (per the discipline)
- **V+**: Qwen predicts the physics at readout 0.520 / held-out 0.435 — it *can* read the dynamics off a
  prompt that contains the velocities. But a plain linear map on the same numbers does far better (held-out
  0.856), so the LLM adds nothing over arithmetic on the handed values.
- **V−**: with velocities withheld, Qwen's held-out prediction drops to **0.254 — below the 0.3 bar** — and
  it does **not beat a linear map on positions alone** (0.413). So Qwen contributes no world-knowledge
  structure beyond what a trivial positions-only linear readout already extracts (next-position ≈
  current-position for a small timestep). Its swirl is uninformative.

## The decisive V+ → V− comparison
`readout_R2 0.520 → 0.378 ; held-out R² 0.435 → 0.254 (falls below 0.3) ; eff_rank 16.92 → 16.06`.

There is **no atomic signature in V+ to vanish** — the swirl was already high-rank (16.9, not < 8) with
velocities *given*. So this is the stronger form of the negative: not "atomic in V+, gone in V−," but
**structureless in both arms**, with the readout coherence additionally collapsing below threshold once
velocities are withheld. Either way the pre-committed rule returns **NOISE / prompt-arithmetic**.

## On the earlier suggestive smoke
A prior small-N smoke suggested a low eff-rank / high readout "signal." That was a **small-N fabrication
artifact**: at N=48 this same instrument reads *everything* — including the Gaussian null — as ATOMIC
(eff_rank ~3–5), because the readout is wildly over-parameterized on a few dozen states. The fix (used here)
is PCA-conditioning the readout and running the full N=1200 so the held-out adjudicator is stable; the null
then correctly reads NOISE and the spurious low eff-rank vanishes. The apparent atomicity did not survive a
stable, non-fabricating measurement — before velocity-withholding even entered.

## Scope
This is the local atomic/noise call for **Qwen's swirl on this engine only**. Honest NOISE is the outcome. It
is not connected to any other claim.
