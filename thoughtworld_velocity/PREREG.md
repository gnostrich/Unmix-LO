# PRE-REGISTRATION — THOUGHTWORLD velocity-withholding arm (frozen BEFORE run code)

Continues design commit `8b826a8` / `9c396a5`. Reuses the physics engine (`engine.py`) and the validated
instrument (`instrument.py` = the unchanged `measure()` + `connection_verdict()` from `run_thoughtworld.py`)
**byte-for-byte unchanged**, both restored verbatim from `origin/archive/pre-nuke`. The ONLY new code is the
V− prompt and the two-arm harness. Full N = 1200 states (the 250-state smoke had an unstable held-out
adjudicator).

## The question
Is Qwen2.5-0.5B's deviation from the physics seed (the "swirl") a genuine atomic world-structure object, or
is it noise / prompt-arithmetic? The prior V+ smoke (readout R²≈0.69, low eff-rank) is suspect because
velocities were handed to the LLM in the prompt — the readout may be arithmetic on the handed numbers
(next_pos ≈ pos + vel·dt), not world-knowledge. The already-run arithmetic control confirmed the confound is
plausible; the decisive test — an actual velocity-WITHHELD LLM re-encode — has not been run. This runs it.

## The two arms (same 1200 states, same engine, same instrument)
- **Arm V+ (velocities given):** the original prompt — each ball's position AND velocity, "predict next
  positions and velocities." Reproduces the smoke/TW2 condition. POSITIVE CONTROL (shows the pipeline works).
- **Arm V− (velocities withheld):** identical states, but the prompt gives ONLY the current positions and the
  physics description — NO velocities, NO derived motion quantities (no speed, no direction, no Δ). The LLM
  must infer dynamics from world-knowledge, not read them off the prompt. LOAD-BEARING ARM.

Per arm, per fragment (Qwen2.5-0.5B as F2; VideoMAE V+ reported from the frozen TW2 run as reference), the
unchanged instrument reports: readout R² (does it predict the physics at all), deviation eff-rank (vs the
established ViT noise floor ≈ 16.4 and vs a matched random-fragment control run in the SAME arm), held-out R²
(the adjudicator: is the low-rank structure a coherent function of state, or low-rank coincidence), and
directed-frac. The random-fragment control is kept in BOTH arms.

## Report order (frozen)
Readout R² is reported FIRST per arm/fragment. If a fragment cannot predict the physics in V− (readout R² ≤ 0
or near-zero), its deviation is uninformative and the outcome is NOISE — stated plainly, not glossed as a
positive.

## Pre-registered verdict (frozen)
- **ATOMIC (genuine stable swirl)** requires, in the **V− arm**, ALL THREE:
  1. eff-rank clearly below the 16.4 ViT floor, AND
  2. eff-rank below its own matched random control (real gap), AND
  3. held-out R² ≥ 0.3.
  (velocities withheld — this is the only arm that can certify ATOMIC.)
- **NOISE** iff, in V−: held-out R² < 0.3, OR eff-rank collapses back toward the 16.4 floor once velocities are
  withheld (loses its gap to floor/control).
- **Decisive comparison V+ → V−:** if an atomic-looking signature is present in V+ but VANISHES in V−
  (eff-rank rises toward floor and/or held-out R² drops below 0.3), the verdict is **NOISE / prompt-arithmetic**
  — the V+ signal was reading velocities off the prompt, not world-knowledge. If the atomic signature SURVIVES
  velocity-withholding (all three V− conditions hold), that is the first real positive — a genuine atomic
  world-structure object — and only then do we report directed-frac for the noncommutative structure.

## Discipline
Frozen PREREG committed before any run code. Engine + instrument reused UNCHANGED (verbatim from archive).
Random-fragment control in both arms. Readout R² reported first. Honest NOISE is a success outcome, not a
failure. Both arms' full numbers + the explicit V+→V− comparison + the pre-committed verdict reported. If the
run cannot complete (timeout/cost), report partial with N reached — do NOT fabricate held-out numbers.
