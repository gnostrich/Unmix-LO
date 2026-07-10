# MECHANISM_CHECK — is the shipped settle the FLUID (feedback) or averaging (feed-forward)?

Regression check requested 2026-07-09. Bound to `thoughtworld_construct/CONSTRUCT.md`. **No code changed** — this
is a verify-and-report on the *committed* settle (`coherentflow/coherentflow.py:71-107`, used verbatim by
`virtualworld/settle_real.py` and the satisfaction battery).

## Verdict (plain): FEED-FORWARD AVERAGING + a self-referential held-subtraction. NOT model→model feedback.

The shipped settle is **not** the genuine feedback fluid the construct calls for. It cannot be mutually unstable
and it cannot do fluid-exclusion (route around a destabilizing model). This is a **regression from CONSTRUCT
non-negotiable #1-2** (the tape≡MZ memory recurrence, streaming *and* through-other-models memory term).

## The literal update rule (coherentflow.py:85-105)
```
state = mean(ifaces)                                    # ifaces = FIXED pre-aligned views f_i
for _ in range(ITERS):
    prev = state.copy()
    for i, f in enumerate(ifaces):                     # ← every model reads the SAME `state` ...
        d = f - state                                  #   `state` is NOT written inside this loop
        is_s, P, _ = structured(d, z)
        if is_s: memory[i] = d @ P                     #   held part = model i's OWN disagreement, projected
    coherent = mean([ ifaces[i] - memory.get(i,0) ])   # ← mean of FIXED views minus each one's own held part
    state = prev + DAMP*(coherent - prev)              # ← state written ONCE, after all models
```
Three structural facts, each fatal to the "feedback" reading:
1. **The interfaces `f_i` are fixed vectors**, never transformed by the evolving state. Each model pulls the
   state toward its own fixed target — the textbook feed-forward-averaging shape `state ← state + step·mean(f_i − state)`.
2. **All models read the same `state` per step (simultaneous / Jacobi).** `state` is read in `d = f - state`
   for every `i` and updated only *after* the loop. Model B's contribution does **not** depend on the state
   model A shaped this step — there is no intra-step, inter-model loop.
3. **The only inter-model coupling is the shared mean, across steps** — and it is contractive (below). A
   model's held part `memory[i]` depends on `state` (so the map is state-*dependent* when structure is held),
   but it is model i acting on **its own** disagreement `(f_i − state)P_i`; no model-j operator touches it.

## Empirical confirmation (ran against the real `cf.settle`, D=24, T=600)
| test | result | reading |
|---|---|---|
| **A. within-step read** | by trace: all models read `state` from prev step | simultaneous, no B-sees-A feedback |
| **B. mutual instability** — maximally incompatible frames (`z`, `−z`) | settle **contracts to their mean 0** (max\|state\|=0.0000), does not diverge/oscillate | structurally cannot be mutually unstable |
| **C. feed-forward identity** — settle (guard off) vs `mean(f_i)` | **max abs diff = 0.000000** | with no held structure it IS exactly averaging toward the fixed mean |
| **D. fluid-exclusion** — add a 5σ-corrupt model | **0 held**, but consensus **shifts by 1.005** from including it | the corrupt view is **averaged in**, not routed around |

Linearized: with the held-projectors ~constant near equilibrium, `coherent(state) = c + state·M`,
`M = mean_i P_i` (PSD, eigenvalues ∈ [0,1]), so the Jacobian `(1−DAMP)I + DAMP·M` has eigenvalues in
`[1−DAMP, 1]` — **always contractive**. The held-subtraction changes *where* it settles (holds structure out
of the consensus), never *whether* it settles. (Consistent with the earlier EqProp probe: response asymmetry
~0.2, i.e. *mostly* conservative/near-averaging, with only a mild non-conservative wrinkle from `P_i(state)`.)

## What the prior results actually tested
Both the "honest no-op on real models" (`settle_real`) and the satisfaction battery were measured on **this
averaging + held-subtraction mechanism** — so:
- **The GUARD results are genuine.** The structured/noise classifier (0% false-positive over 40 seeds) and the
  finding that real convergent senses carry **no** structured decoherence (0/4 held, ho all negative) are real
  and validated. That part stands.
- **The FLUID was never tested.** On real models 0 structure was held, so the settle collapsed to **pure
  averaging** — which by construction (tests B–D) does nothing routing-like. So the "no-op on real models" is
  *"the guard found no structured decoherence, and averaging structurally does nothing"* — **not** *"the fluid
  correctly found nothing to route around."* The distinction the regression check flags is exactly right.
- **The satisfaction +0.5 combined-vs-consensus** comes from the **held-subtraction** (holding structure out of
  the consensus so the combined read recovers it), **not** from feedback routing. Also genuine, also not a fluid
  test. Mutual-instability and fluid-exclusion were never exhibited or measured — the mechanism cannot produce them.

## Note: even the "experimental fluid" is not operator-feedback
CONSTRUCT's actual fluid is the self-expanding operator-valued MZ memory (`virtualworld/mz_fluid.py`, already
labeled UNVALIDATED). But that layer is a **post-hoc linear memory-kernel closure** of the streaming residual
(Hankel-SV order + ridge closure) that the build itself reports "reduces toward classical linear state-space
filtering." So **neither** shipped component implements genuine model→model operator feedback: the settle is
feed-forward averaging, and the MZ layer is a linear through-time closure. The feedback fluid is not shipped.

## What it would take to make the coupling genuine feedback (do NOT auto-apply — decide deliberately)
1. **Models as operators, not fixed targets.** Replace fixed `f_i` with a state-dependent contribution
   `O_i(state)` — decode the current shared state through model i's frame and re-encode — so a model's write
   depends on the state other models are shaping.
2. **Genuine intra-step coupling** — sequential (Gauss-Seidel) or a jointly-coupled operator `state ← state +
   step·Σ_i O_i(state)` whose combined spectrum **can exceed 1** for incompatible frames (so the flow can be
   mutually unstable, not contractive-by-construction).
3. **Emergent exclusion from stability-seeking** — the update must be able to **down-weight or drop** a model's
   contribution from the consensus when including it destabilizes the flow (routing around it), rather than the
   current guard which only skips *circulating* a rejected model's held part while still averaging its raw view in.
Only with (1)-(3) does the object gain mutual flow-instability and fluid-exclusion — the properties that make it
the construct's fluid rather than occupied consensus fusion.

## Bottom line
Feed-forward averaging (proven identical to `mean(f_i)` with no held structure) plus a state-dependent held-
subtraction — **not** the feedback fluid. It is structurally contractive (cannot be mutually unstable) and
non-exclusionary (destabilizing models are averaged in). The validated results are about the **guard** and the
**coverage-union/held read**, which are real; the **fluid's** routing/exclusion has never been instantiated or
tested. Flagging as a regression to resolve deliberately before building further on the "fluid" story.
