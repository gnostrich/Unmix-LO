# field — a settling-field object, rebuilt from the concept

Fresh start (2026-07-09): rebuilt from the CONCEPT, no test suite, theory drives the code. Full prior history
(the construct: fluid, conformance suite, theory docs, the whole build) is preserved on branch
`archive/pre-nuke` and tag `construct-pre-nuke`.

## The object
Models are **forces** (operators) tensioning a shared **field**; the field **settles** under bounded coupled
feedback (genuine — it can go unstable, not averaging); you **probe** with a query and **read** the settling
trace's tail motion (point = consensus, tremble = competing branches). Per-query **terrain** = how stability
varies with the query. See `field.py`.

- `field.py` — the object (settle, tail-read, terrain, streaming). numpy only.
- `explore.py` — feeds it convergent / conflicting / real / injected regimes; reproduces the numbers.
- `EXPLORE.md` — honest writeup of what it does, with the plain **average** as the null throughout.

## What it does (see EXPLORE.md for numbers)
- **Faithful:** conflicting frames drive coupled ρ = 1.09 > 1 (averaging structurally can't) — it's the real
  feedback object, not averaging in disguise.
- **Honest no-op on real:** the four real modalities converge (ρ = 0.96 ≤ 1) → it settles to consensus and
  **ties the plain average**.
- **One distinct signal:** an agree-vs-conflict terrain the average is blind to — but non-vacuous only on
  genuinely conflicting frames, which real convergent models don't produce.
- **The payoff fails:** held-superposition (recovering competing branches from the tremble) does **not** beat
  the average (≈ chance). Elegant mechanism, no real fuel on real input.

## Standing baseline (unchanged, real)
**Coverage-union** — fusing complementary modalities beats best-single (stitch R² 0.445 > 0.337, in
`data/real_modalities.js`) — is the genuinely useful result and does not depend on the field object. The prior
program's full ledger is in `report.md` and `COMPOSITION_THESIS.md`.
