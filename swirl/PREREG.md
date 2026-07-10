# PRE-REGISTRATION — Qwen "swirl": atomic world-structure or noise? (frozen BEFORE run code)

One local experiment. Fresh, self-contained: a minimal numpy physics engine (`engine.py`) and a fresh swirl
instrument (`instrument.py`). The verdict is about **Qwen2.5-0.5B's deviation on this engine only** — it is not
connected to any broader claim.

## Setup
- **Engine:** 2D rigid-body — 5 balls in a unit box under gravity, elastic wall + ball-ball collisions.
  Deterministic given a seed. `state` = positions + velocities of all balls (D = 4·N = 20). The coherent
  dynamics the fragment is measured against.
- **Fragment:** `Qwen2.5-0.5B-Instruct`, frozen. A state is encoded by feeding a text `describe(state)` prompt
  and taking a mean-pooled hidden-state embedding.
- **Instrument** (`instrument.py`): fit a linear readout embedding → next-state on a train split.
  - `readout_R2` = train-fit R² (does the fragment predict the physics at all).
  - **swirl** = the readout residual (fragment's deviation from true physics) on the held-out split.
  - `eff_rank` = effective rank of the swirl matrix = participation ratio (Σσ)²/Σσ² of its singular values.
  - `heldout_R2` = readout R² on the held-out split (the adjudicator: coherent function of state, or coincidence).
  - **ATOMIC iff `eff_rank < 0.4·D` AND `heldout_R2 ≥ 0.3`; otherwise NOISE.** (D = target dim = 20; 0.4·D = 8.)

## The two arms (same states, same engine, same instrument)
- **Arm V+ (velocities given):** `describe` includes each ball's position AND velocity. The LLM is handed the
  velocities. POSITIVE CONTROL (shows the pipeline can produce a readout).
- **Arm V− (velocities withheld):** `describe` includes positions ONLY — no velocities, no derived motion
  quantities (no speed, direction, or Δ). The LLM must infer dynamics from world-knowledge. LOAD-BEARING ARM.
- **Random-fragment control** (a fixed random-projection "encoder" of the raw state) is run in BOTH arms. It is
  the null and must come out NOISE.
- N: target ~1000–1200 states so the held-out adjudicator is stable. If CPU time is tight, report the N reached
  — do not fabricate.

## Report order (frozen)
`readout_R2` FIRST per arm. If Qwen cannot predict the physics in V− (readout_R2 ≤ 0 or near-zero), its swirl is
uninformative and the outcome is NOISE — stated plainly, not glossed as a positive.

## Frozen verdict
- The ATOMIC criterion (`eff_rank < 0.4·D` AND `heldout_R2 ≥ 0.3`) is the load-bearing verdict applied to the
  **V− arm**.
- **Decisive comparison V+ → V−:** does any atomic signature in V+ SURVIVE into V−, or VANISH?
  - **VANISH** (eff-rank rises toward full and/or held-out R² drops below 0.3 when velocities are withheld)
    → **NOISE / prompt-arithmetic** — the signal was reading velocities off the prompt, not world-knowledge.
  - **SURVIVE** (both V− conditions hold, velocities withheld) → **genuine atomic world-structure** (first real
    positive); only then report directed-frac.

## Discipline
Frozen PREREG committed before run code. Random-fragment control in both arms. `readout_R2` reported first.
Honest NOISE is a success outcome, not a failure. Report both arms' full numbers + the explicit V+→V−
survive/vanish comparison + the pre-committed verdict. Partial N reported honestly if the run can't finish. This
verdict is the local atomic/noise call for Qwen's swirl on this engine — nothing more.
