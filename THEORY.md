# THEORY — the canonical theory layer developed 2026-07-09 (binds equally to CONSTRUCT.md)

This captures theory figured out AFTER `thoughtworld_construct/CONSTRUCT.md` was written, so it is not lost.
It **binds equally** to CONSTRUCT.md. Where it refines or supersedes an existing non-negotiable, the
reconciliation is stated explicitly (and mirrored in CONSTRUCT.md's new pointer section). The through-line of
this layer: **the informative object is the stability geometry of the coupled fluid, not a content residual** —
and the output is read from the settling trace itself, streamed until it stabilizes.

Status legend per item: **[NEW]** not in CONSTRUCT.md · **[IMPLICIT]** latent in CONSTRUCT but not explicit ·
**[TENSION]** needs reconciling with an existing non-negotiable (stated inline).

---

## T1. Mutual-instability IS the Baur objective  [IMPLICIT + TENSION with #4]
The loss the Baur process descends is the **mutual flow-instability of the coupled system** — the spectral
radius of the coupled operator `J(w) = I + step·Σ wᵢ(Rᵢ − I)` — **not** a content-reconstruction loss. It is
**non-vacuous even when content-disagreement (held-structure) is zero**: two models can perfectly agree on
content yet couple into an unstable or a trivially-stable flow, and that geometry is the signal.
- **CONSTRUCT status:** #3 says "descend a loss," #5 says contraction is a term — but the *primary objective
  being coupling-instability* is not stated; #4 reads as a content/training-data loss.
- **Reconciliation with #4 (loss = seed models' own grounding):** NO contradiction once framed right. The
  objective is instability, but it is **grounded** because the coupling operators `Rᵢ` ARE the seed models' own
  (trained) read/write frames — the instability is measured on their *grounded interaction*, never on arbitrary
  data or an external judge. So #4's grounding requirement holds; #4's *implicit content-reconstruction reading
  is superseded* by "instability of the grounded coupling."
- **Code:** implemented in `coherentflow/fluid_settle.py` (`instability_descent` descends the coupled spectral
  radius). The shipped `coherentflow.settle` does NOT do this (it averages) — conformance INV2 FAIL.

## T2. Fluid exclusion (emergent, no explicit reject)  [IMPLICIT]
Descending the instability **naturally routes AROUND a destabilizing model** — its routing weight → 0 — rather
than averaging it in. Emergent robustness: there is **no explicit reject step**; exclusion falls out of
stability-seeking. (Symmetric rivals are instead *balanced*, not excluded — the seed of held-superposition.)
- **CONSTRUCT status:** IMPLICIT in #3 ("topology emerges, never forced") and #5 ("stay contractive on the
  seed-consistent manifold"). Not stated as a required, testable behavior. No contradiction.
- **Required behavior of the correct fluid** (acceptance criterion). Contrast the averaging bug: a corrupt model
  is averaged in (shifts the consensus), never routed around (see `MECHANISM_CHECK.md`).
- **Code:** verified in `fluid_settle.py` (rogue weight → 0.000, ρ → 0.970).

## T3. Per-query stability terrain (the informative object)  [NEW — central shift]
For a query, the informative object is the **local stability terrain along the query direction**: which
directions are **damped/agreed** (contractive, consensus), which are **amplifying/contested** (expanding, rival),
which are **swirling/path-dependent** (non-normal, rotational/ambiguous). This terrain **varies per query** and
is **non-vacuous even on convergent models** — it is a *confidence gradient* on convergent models and *full
contested-structure* on divergent ones. **Content-averaging is blind to it** (averaging returns one point).
- **CONSTRUCT status:** NEW. "query" and "terrain" are absent from CONSTRUCT.
- **Central reconciliation (the informative object RELOCATED):** CONSTRUCT's informative content was the
  **content swirl / fragment-deviation / held-structure** (what THOUGHTWORLD probed as single-fragment
  deviation; NOISE on convergent vision). T3 says the informative object is the **stability geometry**, which is
  informative *even where content-deviation is zero*. So: **held-structure is the divergent-case reading of a
  more general object — the terrain.** This is why the convergence nulls (xresolve/biomesh/THOUGHTWORLD) are
  not the end of the story: convergent models have a *flat-but-graded* terrain (a confidence field), not
  nothing. Averaging is blind to that field; the fluid reads it.

## T4. Information relocation (agree → point; diverge → dynamics)  [NEW]
Same object, read two ways. When the models' forces **converge**, the information is in the **equilibrium POINT**
(the consensus). When they **diverge**, the point is uninformative (a bland midpoint) and the information
**relocates to the DYNAMICS AROUND the point** — the residual "tremble"/instability modes.
- **CONSTRUCT status:** NEW framing; compatible with the streaming/memory split (streaming ≈ the point,
  memory ≈ the through-time dynamics). No contradiction.
- **Consequence:** an object that only reads the equilibrium point (averaging, or a fitted head on the settled
  state) throws away the divergent-case information. The fluid must read the point AND its surrounding motion.

## T5. Trace-native output (the tail motion IS the answer)  [IMPLICIT — supersedes fitted-head reads]
The output is read from the **trace** — the recorded settling trajectory — specifically the **residual
settling-motion at the end**. If the trace has **stopped**, the answer is the **point** (consensus). If it is
**still moving**, the **motion IS the answer**: its **directions** = the competing branches, its **magnitude** =
uncertainty, its **rotation** = ambiguity/path-dependence. There is **no external readout head** — the output is
the trace's own tail motion, and its **derivative w.r.t. the query is the terrain (T3)**.
- **CONSTRUCT status:** IMPLICIT in #5 ("the native MZ process settles via its own dynamics") but the
  *output = trace tail-motion, no head* is not stated.
- **Directly resolves conformance INV6 (intrinsic output):** the shipped `combined_read` fits an `lstsq` probe
  head (FAIL). The correct output is the trace tail-motion / equilibrium-shift — implemented in `fluid_settle`
  (equilibrium shift), to be generalized to the full tail-motion read. **Supersedes** any fitted-head output.

## T6. Streaming / anytime I/O  [NEW]
The read is **not call-return**; it is a **stream of refining states** that runs until stabilization. It
terminates in one of three ways, and **the MANNER of stabilization is part of the answer**:
- **CONVERGING** → confident consensus (a limit point) — the answer is that point.
- **CYCLING** → contested — the **cycle's branches ARE the held-superposition** (the divergent read, T4).
- **BUDGET cutoff** → best-so-far point + residual motion = an explicit **uncertainty** estimate.
Anytime: consume the stream until the residual motion says "enough."
- **CONSTRUCT status:** NEW (I/O model unspecified in CONSTRUCT). No contradiction.
- **Reconciliation with held-superposition:** held-superposition (the paraconsistent hold) is precisely the
  **CYCLING termination** — it is a *dynamical* phenomenon (T5/T4), consistent with the finding in
  `FLUID_VERIFICATION.md` that robust held-superposition requires nonlinear multistability, not a static branch.

## T7. Field / settling mental model (the intuition)  [NEW — framing]
Models are **forces tensioning a shared field**; the field **settles** under those forces; you **read the
resting position where forces agree** (the field is taut) and **read the tremble where they diverge** (the field
is slack / has vibrating modes). The **query = where you probe the field**.
- **CONSTRUCT status:** NEW as explicit intuition ("the fluid" and "settles" exist; the forces-on-a-field
  picture does not). The canonical mental model for the whole layer; no contradiction.

---

## Cross-cutting contradictions / tensions to resolve deliberately
1. **Objective framing (#4 vs T1):** CONSTRUCT #4's implicit content-reconstruction reading is **superseded** by
   "instability of the grounded coupling." #4's *grounding* requirement stands. → Update #4's gloss (done in the
   CONSTRUCT pointer section).
2. **Informative object (old swirl/held-structure vs T3 terrain):** the informative object **relocated** from
   content-deviation to stability geometry. Held-structure is the *divergent-case* reading (T4/T6-cycling), not
   the whole object. → The convergence nulls do not close the construct; the terrain (confidence field) is the
   remaining, untested object. This is the single most important update.
3. **Self-expansion criterion (#2 Hankel-SV-vs-FDT vs the empirics):** `fdt_denoise` found native FDT denoising
   is MID/DOWNGRADE and `mz_fluid` "reduces toward classical linear state-space filtering." So #2's literal
   trigger is empirically weak. **Candidate reconciliation (open joint J2):** tie self-expansion to *persistent
   instability modes of the terrain* (a new atom = a new stable expanding/swirling direction that recurs across
   queries), rather than the Hankel-SV of a content residual. Not yet built — flag, don't resolve here.
4. **Output (INV6 fitted head vs T5 trace-native):** the shipped fitted-probe read contradicts T5. → the fitted
   probe is retained only as an *external scoring tool for evaluation*, never as the object's output.

## What this layer implies for the build order (not built in this pass)
- The correct fluid (`fluid_settle`) already implements T1 (instability objective) and T2 (exclusion) and a
  first form of T5 (equilibrium-shift output). Wiring it (conformance INV2 + INV6) is the biggest fidelity step.
- The **terrain read (T3)**, the **trace tail-motion output (T5 full)**, and **streaming I/O (T6)** are theory
  **not yet built** — they are the next design targets, and they must not be missed by targeting the old spec.
- T3's claim ("non-vacuous even on convergent models — a confidence gradient") is a **testable prediction** that
  distinguishes the terrain read from the content-averaging nulls. It is the first thing to probe once the
  terrain read exists.
