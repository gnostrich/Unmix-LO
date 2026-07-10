# STEP 4 — HALT at gate #2: held-superposition does NOT fire from the trace tail-motion (a real finding)

Self-driving build, terrain-first ordering (4→5→3→6). **STEP 4 gate #2 FAILED → HALT** per the standing rule
("if it does NOT fire even in the unstable regime, HALT and report — a real finding, not something to paper
over"). No mechanism code was changed; all probes were scratchpad-only, tree clean. Steps 5/3/6 NOT attempted.

## STEP 0 decision (done): 4 → 5 → 3 → 6 confirmed
The tail motion is `s_final − s_{final−1}` off the fluid state trajectory (the settle loop exposes `prev`/`S`
each iteration), **independent of the held-memory dict**. So Step 4 needs only the fluid trajectory, not the
unified tape (Step 3). Terrain-first ordering stands.

## What passed
- **Consensus / soft read:** on convergent input the tail motion is small; on divergent it is large. The
  *magnitude* channel of the trace-native read is fine.
- **INV6 read path** (replacing the fitted lstsq probe with a tail-motion read) is mechanically feasible for
  the consensus and soft cases.

## What FAILED (gate #2): held-superposition does not fire on contested input
I built genuinely contested (coupled ρ>1) federations myself and ran the tail-motion read under THREE settle
formulations. In none does held-superposition fire (branch recovery ≈ chance):

| settle formulation | contested tail behaviour | branch recovered from tail? |
|---|---|---|
| linear `S←S·Jᵀ` (+ in-loop descent) | **blows up** (|Δ| 4.5e3 → 1.2e4, unbounded) | — (diverges) |
| tanh-bounded increment | state still accumulates, |Δ| → 85 | acc **0.507** (chance) |
| norm-ball projection | collapses to the **single** dominant eigen-direction | acc **0.520 / 0.517** (chance) |

Convergent controls behave (tail → small / →0). But contested input never produces a *bounded tremble whose
directions recover the competing branches* — it diverges, drifts, or collapses to one direction. Held-
superposition (proven to work in `fluid_settle.acceptance_3`) requires an **explicitly multistable field**
(hand-constructed bistable wells per rival direction), which does NOT emerge from operators derived off real
interface vectors.

## ROOT CAUSE (the important finding) — the operator derivation discards the held-structure
The Step-1 wiring derives each operator as `Rᵢᵀ = lstsq(z, f)` — the **medium-linear** part of the interface.
An injected hidden distinction (a branch independent of the medium `z`) lives in the **residual** `f − z·Rᵢᵀ`,
which the derivation throws away. Measured directly:

```
injected branch correlation with operator-reconstruction (KEPT):     0.092
injected branch correlation with residual (DISCARDED by Rᵢ):         0.966
```

So the operator-fluid is **structurally blind to medium-orthogonal content** — exactly where content
held-structure / competing branches live. The trace tail-motion therefore cannot surface a held-superposition
of a hidden distinction: the distinction was discarded at the operator-derivation step, before the settle.

## What this SHARPENS about the theory (not a dead end — a clarification)
This cleanly separates two things THEORY.md had bundled:
- **Frame-conflict terrain (T3, medium-linear):** the operator-fluid DOES carry this — the coupled-operator
  stability geometry (ρ, damped/amplifying/swirling directions) is a property of the derived operators `Rᵢ`.
  So the **terrain read (Step 5) is still plausibly buildable** on the current wiring.
- **Content held-superposition (T5/T6 cycling, medium-orthogonal):** the operator-fluid does NOT carry this,
  because the branch content is discarded by `lstsq(z, f)`. Surfacing it requires the operators (or the settle)
  to **retain the medium-orthogonal residual**.

So the informative-object relocation (STOCKTAKE contradiction #2) has a sharper form: the operator-fluid reads
the *frame-conflict terrain* but not *content branches*. Held-superposition of injected content is not a wiring
detail — it needs a different operator construction.

## Candidate fixes (for deliberate decision — NOT auto-applied)
1. **Retain the residual:** augment each interface's operator with its medium-orthogonal residual subspace
   (the part `lstsq` discards), so the settle can hold/circulate content branches — then re-test the
   tail-motion held-superposition. This is a redesign of `interface_operator`, re-gated on INV2/INV3/no-op/no-fab.
2. **Explicit multistable field from interface conflict:** derive rival directions from the interface
   disagreement's top modes and build the bistable field (as `fluid_settle.acceptance_3` does by hand) from
   them — so held-superposition emerges as bistability, not from linear feedback.
3. **Accept the split:** ship the trace-native read for CONSENSUS/SOFT (which works) and the **terrain read
   (Step 5)** as the medium-linear informative object, and document that content held-superposition needs
   option 1 or 2. (This would make INV6 a consensus/soft/terrain read, with held-superposition explicitly
   scoped to the residual-retaining redesign.)

## Guarantees intact (nothing broken by the halt)
No mechanism code changed. Conformance still PASS 5 / FAIL 2 (INV2/INV3 PASS from Steps 1-2). No-fabrication
(0% FP), convergent no-op (0/4 held, 0.8485), and coverage-union (0.445) all untouched.

**Recommendation:** decide between candidate fix 1 (retain residual — most faithful to "content held-
superposition") and reordering to build the **terrain read (Step 5)** first (which the current operators DO
support), before revisiting Step 4's held-superposition. This is the deliberate decision the halt surfaces.
