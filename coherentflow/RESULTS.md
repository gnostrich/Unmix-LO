# COHERENTFLOW — the WHOLE object, one-shot (observe-don't-prove)

Run 2026-07-09 per `coherentflow/PREREG.md` (design frozen & committed before the build). Bound to
`thoughtworld_construct/CONSTRUCT.md`. This is **not a gate and not a proof** — it instantiates the
complete construct object as ONE loop and watches what it does on frame-diverse input. Numbers in
`coherentflow_results.json`; reproduce with `python coherentflow/coherentflow.py` (CPU, seconds).

## The object (one loop, guards inside)
A Baur descent writes a natural Mori-Zwanzig memory (the resizable "tape") holding the recurrent routing
of information across model **interfaces** that each connect to a shared medium. Driven purely by an
**internal coherence loss** (settle toward the state maximally consistent with every interface's
grounding — no task, no labels), the recurrent flow **settles** to a unified world-state. An answer is an
**optimal combined read**: consensus where interfaces cohere, **paraconsistent held-superposition** where
they carry structured decoherence. CONSTRUCT mapping: streaming term = consensus settling; memory term =
held structured circulation; tape ≡ memory (one object); self-expansion = a memory mode per interface
whose disagreement clears the structured criterion; faithfulness = the damped (contractive) update is
native settling, guards live *inside* the loop.

## The three behaviors (on frame-diverse INJECTED input — one interface carries a hidden binary branch)
| behavior | observation |
|---|---|
| **SETTLE** | residual **2.666 → 0.003**, tail-contracting (observed, not proven). |
| **SURFACE + HOLD** | **1/3** interfaces held STRUCTURED (correctly: only the branch-carrier), tape written to dim 24, circ-norm 23.6. |
| **COMBINED READ** | consensus-collapse recovers the branch **0.483** (chance — the minority branch is diluted out of consensus); combined read (consensus ⊕ held superposition) recovers it **1.000**. **Payoff +0.517.** |

The combined read surfaces held structure a consensus-collapse **completely loses** — that is the object's
payoff, and the fix over `smoke_oneshot.py` (which folded the held structure back into the consensus, so
its combined read barely beat consensus, 0.568 vs 0.528). Here the structured decoherence is held
**separately in the tape**, kept out of the consensus, so only the combined read can extract it.

## The two controls (mandatory)
- **COHERENT input (no injected structure) → HONEST NO-OP.** 0/3 held, circ-norm 0.000, consensus =
  combined = 0.483 (≈chance). It settles to consensus and circulates nothing. Correct — **not hacked** to
  fake activity. This is the expected behavior on frozen/converged models (F_gauge ≈ 0, per xresolve).
- **NOISE input (one interface corrupted with unstructured noise) → REJECT, no amplification (no G1).**
  0/3 held, circ-norm 0.000 (nothing written to the tape → nothing to amplify). Discriminator: the
  corrupted interface is **full-rank (eff-rank 23/24), cap 0.96 < 1.44 threshold → NOISE**, while the
  injected carrier is **concentrated (eff-rank 11/24, cap 0.83 > 0.69) AND held-out predictable (R² +0.57)
  → STRUCTURED, held**. The guard's concentration criterion is what rejects the noise.

## EqProp-native learning mechanism (owner steering, 2026-07-09)
Per the steering: the memory/routing, *if learned*, must be learned by **equilibrium-response**, NOT
backprop-through-the-settling and NOT a fixed heuristic dressed as learning. For this first run the
routing is **fixed/initialized** (labeled as such) and we observe the settling + read behaviors; the
EqProp probe (`eqprop_probe`) demonstrates the native learning signal and reports the honest flag.

- **Equilibrium-response learning signal (point 2).** Settle to a free equilibrium s*, add a weak nudge
  β toward the coherence objective, settle to s^β, and use **R = (s^β − s*)/β** as the local routing
  signal — **two settles and a difference, no iteration unrolled or differentiated**. R overlaps the held
  routing directions, so it is a usable learning signal for the memory.
- **Honest flag — is the settling a clean scalar-energy flow, or constrained relaxation (point 4)?** The
  exact criterion for "gradient flow of a scalar energy" is a **symmetric response operator**
  (Maxwell/Onsager reciprocity): ⟨R_u, v⟩ = ⟨R_v, u⟩. Measured **response asymmetry ≈ 0.19–0.23**, and
  critically it is **β-independent** (0.225 → 0.228 as β drops from 0.05 to 0.001) and seed-robust — so
  it is **not** a finite-nudge numerical artifact but a **genuine, structural non-conservative component**.
  - **Verdict: EqProp-*like* but not textbook.** The guarded settling is *mostly* conservative (~78%
    reciprocal) — equilibrium-response is the right, native learning mechanism and it works — but holding
    structured decoherence introduces a real, stable non-conservative part (from the state-dependence of
    the structured projectors / the hold constraint). So the object is a **constrained relaxation on a
    manifold**, not a clean gradient descent of a single scalar energy. Textbook EqProp's clean-energy
    precondition is *mildly but genuinely* violated by the paraconsistent hold — interesting either way,
    exactly the "see what happens" payoff of the EqProp reframing.

## Honest framing
Everything here is on **injected** frame-diversity, where the precondition (a hidden distinction one
interface carries) is met so the object can act. On real frozen/converged models it would **no-op**
(the coherent control), consistent with the session's convergence findings (xresolve, biomesh, synergy).
The object does exactly what it should: settle, hold real structure, reject noise, and surface held
structure in the combined read that a single-frame collapse loses — and its native learning is
equilibrium-response, which here behaves as a *near*-conservative constrained relaxation, not a clean
energy flow.
