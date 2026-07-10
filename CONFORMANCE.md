# CONFORMANCE — code vs. CONSTRUCT.md theory-invariants (the drift map)

Executable suite: `conformance/run_conformance.py` (run it; dumps `conformance/conformance_results.json`).
Bind to `thoughtworld_construct/CONSTRUCT.md`. **Methodological purpose:** two regressions (dimensionality
glued; recurrence was averaging not feedback) were caught only by after-the-fact probes. This suite turns each
theory-invariant into a test that **cannot silently break**, and surfaces the **full** drift surface at once —
not just the invariants we happened to probe. This pass **establishes the honest map; it does not fix the
FAILs** (those are resolved deliberately, in order, afterwards).

## Current state — run 2026-07-09

| # | invariant (CONSTRUCT ref) | status | one-line gap |
|---|---|---|---|
| 1 | resizable / self-expanding medium (#2) | ✅ **PASS** | registry-driven D, grows, no hard-coded dims |
| 2 | genuine feedback recurrence, not averaging | ❌ **FAIL** | wired settle is averaging; correct fluid exists but unwired |
| 3 | faithfulness is a loss TERM, not a phase (#5) | 🟡 **PARTIAL** | shipped guard-in-loop OK; fluid stabilization is a pre-phase |
| 4 | MZ memory = the tape, ONE object (#1) | ❌ **FAIL** | memory is a transient dict; tape is a separate module |
| 5 | loss = models' own grounding (#4) | ✅ **PASS** | objectives read intrinsic signals; no external judge |
| 6 | intrinsic output (from the terrain) | ❌ **FAIL** | shipped read is a fitted probe head; fluid shift is correct but unwired |
| 7 | frozen interfaces, everything else flexible | ✅ **PASS** | encoders frozen; D / n_models / T / held-rank all knobs |

**Tally: PASS 3 · PARTIAL 1 · FAIL 3.** The drift surface is wider than the two regressions we caught by probe:
INV4 (MZ=tape) and INV6 (intrinsic output) are **newly surfaced** here — they were silently violated because the
system was built feature-first. That is exactly what the suite is for.

## Per-invariant: what the code does vs. what the theory requires

### ✅ 1 — resizable / self-expanding medium (#2)
- **Code:** the medium is `SCENE_REGISTRY`; `SCENE_D = len(registry)`; `SCENE_POS/VEL/COLL` derive from tags;
  `append_feature` grows D; the build derives `D = scene.shape[1]`.
- **Theory:** D a registry knob, medium can grow, nothing hard-codes a dimension. **Met** (registry refactor).

### ❌ 2 — genuine feedback recurrence, not averaging
- **Code:** the WIRED recurrence (`coherentflow.settle`) is **feed-forward averaging** — incompatible frames
  `z, −z` contract to their mean (max-diff < 1e-6), Jacobian bounded ≤ 1, cannot be unstable or exclude. The
  **correct fluid EXISTS separately** (`fluid_settle.py`) and passes: coupled ρ can exceed 1 (1.088) and a
  destabilizer is excluded (rogue weight → 0.000).
- **Theory:** the recurrence used by the pipeline must be model→model feedback (coupled ρ>1 achievable;
  destabilizer routed around, not averaged in).
- **Resolution:** wire `fluid_settle` in place of the averaging settle (deliberate later step; see
  `MECHANISM_CHECK.md` / `FLUID_VERIFICATION.md`).

### 🟡 3 — faithfulness is a loss TERM, not a phase (#5)
- **Code:** `coherentflow.settle` calls the guard `structured()` **inside** the ITERS loop (one loop, no
  separate verify pass) — good. **But** `fluid_settle`'s stabilization (`instability_descent`) is currently a
  **separate pre-phase** computing `w` before the settle runs — a contraction *phase*, not a term inside the
  settle loop.
- **Theory:** the contraction/anti-hallucination guard is a term inside the single objective/loop, never a
  separate phase.
- **Resolution:** fold the instability-descent INTO the fluid settle loop (adapt `w` each step) so stabilization
  is native, not a pre-phase.

### ❌ 4 — MZ memory = the tape, ONE object (#1)
- **Code:** `coherentflow.settle`'s `memory` is a **per-step dict** of held projections (transient, recomputed
  each iteration) — a held-structure store, not a persistent resizable tape. The MZ kernel / tape (block-Hankel,
  self-expansion) lives in a **separate** module `virtualworld/mz_fluid.py`. Two different structures.
- **Theory:** the NTM-like resizable tape and the MZ memory kernel are the **same** structure; the tape's
  read/write dynamics ARE the kernel (streaming/memory split intrinsic).
- **Resolution:** unify — make the fluid's held memory a persistent resizable tape whose read/write IS the
  through-time kernel, not a transient dict + a separate Hankel module.

### ✅ 5 — loss = models' own grounding (#4)
- **Code:** the guard reads held-out predictivity from the shared medium `z` (grounding); the fluid's objective
  is coupling instability (spectral radius / measured growth) — both intrinsic. The ridge aligns to the physics
  engine's own scene state (seed grounding). No external labels/judge. **Met.**

### ❌ 6 — intrinsic output (read from the terrain)
- **Code:** the SHIPPED read (`coherentflow.combined_read`) fits a **linear probe via lstsq** to predict a
  target — an external fitted readout head, not intrinsic. The **correct intrinsic output EXISTS** in
  `fluid_settle`: the answer is the **equilibrium shift** of the settled state (query = perturbation), no head.
- **Theory:** the output emerges from the settled dynamics (equilibrium shift), not a separately-trained probe.
- **Resolution:** use the fluid's equilibrium-shift output; retire the fitted-probe read (keep it only as an
  external *scoring* tool for evaluation, clearly separated from the object's output).

### ✅ 7 — frozen interfaces, everything else flexible
- **Code:** encoders frozen (`@torch.no_grad` + `.eval()`, cached); only the medium-side ridge is fitted; D is a
  registry knob, `n_models` is a list, T is module constants — runtime-variable with no model rewiring. **Met.**

## The fix order (deliberate, biggest-leverage first — NOT done in this pass)
1. **INV2 + INV6 together (the fluid):** wire `fluid_settle` as the recurrence and take the output from its
   equilibrium shift. This flips both FAILs and is the largest fidelity gain. (The correct object already exists
   and is verified.)
2. **INV3:** fold instability-descent into the fluid settle loop (stabilization as a term, not a phase).
3. **INV4:** unify held-memory into a persistent resizable MZ tape (one object).
Then re-run the suite: the target is 7/7 PASS, and the suite runs on every future change so drift is caught
immediately, not by after-the-fact probes.

## Keeping it faithful
`conformance/run_conformance.py` is the tripwire. Run it before every commit that touches the settle, the
medium, the memory, the objective, or the output. A silently-averaging recurrence, a hard-coded dimension, a
bolted-on readout head, or a two-phase verify gadget will each flip an invariant to FAIL loudly.
