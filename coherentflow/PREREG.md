# COHERENTFLOW — one-shot build of the WHOLE OBJECT (fast experiment, observe-don't-prove)
# BIND to thoughtworld_construct/CONSTRUCT.md. Build the COMPLETE object as ONE loop. NOT pieces, NOT a gate,
# NOT a proof. Instantiate the aesthetic object faithfully and WATCH what it does on frame-diverse input.

## The object (complete, theoretically closed -- build it faithfully, do not flatten or re-scope)
A Baur process writes a natural Mori-Zwanzig memory (NTM-like but primitive/natural) encoding the RECURRENT
routing of information through a set of models. Driven PURELY by an INTERNAL COHERENCE LOSS -- settle into the
state maximally consistent with every model's own grounding -- the recurrent flow reaches a SETTLED internal
state (the unified world-model), living in a shared medium each model connects to through its INTERFACE.
The objective is entirely internal settling; it produces NO output per se. An answer is an OPTIMAL COMBINED
READ ACROSS THE INTERFACES: for a query, read the settled state through all interfaces and combine into the
view of maximal cross-interface consistency -> CONSENSUS where interfaces cohere, PARACONSISTENT HELD-
SUPERPOSITION where they carry structured decoherence. Settling integrates; combined-read extracts.

## Faithful build requirements (all in ONE loop -- guards INSIDE, not beside)
1. INTERFACES: each model connects to a shared medium via its own map (native<->medium). Different models =
   different interfaces (frames). Use several small/synthetic frames for speed; the point is frame-diversity.
2. INTERNAL COHERENCE LOSS: settle toward the state maximally consistent with each model's grounding. Operational
   form: minimise UNSTRUCTURED cross-interface decoherence (each model "unsurprised" by the settled state);
   HOLD structured decoherence (do not average it away). No external task, no labels. Purely internal.
3. RECURRENT SETTLING with GUARDS IN THE LOOP (this is what makes recurrence safe, not a separate layer):
   - classify each interface's disagreement with current state: STRUCTURED (concentrated AND held-out-predictable)
     vs NOISE. 
   - HOLD + circulate structured decoherence; REJECT noise (do NOT circulate it -> this defeats G1).
   - CONTRACTION: damped update so the flow converges. Do NOT prove contraction -- OBSERVE it (tail residual
     decreasing). If it blows up on some input, REPORT that empirically (that's a finding, not a failure).
4. COMBINED READ (get this RIGHT -- the smoke shortcut under-extracted): for a query, read the settled state
   through EACH interface, then COMBINE across interfaces into the maximally-self-consistent view. Where
   interfaces agree -> consensus. Where they carry structured decoherence -> HELD SUPERPOSITION (keep both
   interface-views, do not collapse). The read must SURFACE the held structure that a single-frame/consensus
   collapse would lose. (smoke_oneshot.py shows structure IS surfaced (circ large) but the crude linear read
   under-extracts -> build the proper cross-interface combination.)

## What to WATCH (observe-don't-prove; both outcomes interesting)
Run on FRAME-DIVERSE INJECTED input (inject structured decoherence -- a hidden distinction one interface carries)
because frozen/converged models no-op (F_gauge~0, per xresolve) -- test where the precondition is MET so the
object can actually act. Then OBSERVE:
  - Does it SETTLE (stable, tail-contracting)? Report the residual curve. (Blow-up = report it, don't hide.)
  - Does it SURFACE + HOLD the injected structure (not average it away)?
  - Does the COMBINED READ recover the held structure BETTER than a consensus-collapse read? (This is the
    payoff: the object's output is richer than any single-frame collapse because it holds cross-frame structure.)
  - CONTROL: on COHERENT input (no injected structure) it must NO-OP -- settle to consensus, circulate nothing.
    That honest no-op is CORRECT, not a failure. Do NOT hack it to appear to do something.
  - CONTROL: on NOISE input (corrupted interface) it must REJECT -- not circulate, not amplify (no G1).

## Discipline (fast experiment, faithful object)
- ONE loop, complete object, guards INSIDE. Do NOT flatten into separate stitch/extend pieces; do NOT quarantine
  the recurrence beside guards -- the guards ARE what makes the recurrence the object.
- No proofs, no thresholds-as-gates -- OBSERVE behavior and report it. Convergence-to-no-op on coherent input is
  a valid, honest outcome (never hacked to fake activity).
- Test on injected frame-diversity (mechanism can act); note explicitly that frozen models would no-op.
- Small/fast (synthetic or tiny models, CPU, <~1k samples). Reproducible from one script + a short readout view.
- Report the residual curve, the surfaced-structure norm, and the combined-read-vs-consensus comparison.
- Commit code + RESULTS.md with the three behaviors (settle / surface-hold / combined-read) + the two controls.
