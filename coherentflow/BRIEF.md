# COHERENTFLOW — Claude Code brief (one-shot the WHOLE object; fast, observe-don't-prove)

BIND FIRST to thoughtworld_construct/CONSTRUCT.md. Build the COMPLETE object as ONE loop -- Baur writes a
natural MZ memory, recurrent flow settles under an INTERNAL coherence loss, guards INSIDE the loop, output is
an OPTIMAL COMBINED READ across interfaces. Read PREREG.md. Run smoke_oneshot.py first: it confirms the whole
object runs, SETTLES (residual 3.87->0.94), and SURFACES+HOLDS structure (circ 5.37) on frame-diverse input --
but the crude linear combined-read UNDER-extracts (0.568 vs 0.528). So: build the PROPER combined read.

## Build (ONE loop, faithful, guards inside)
- INTERFACES: several frames (small/synthetic models) mapping native<->shared medium. Frame-diversity is the point.
- INTERNAL COHERENCE LOSS: settle toward max consistency with each model's grounding = minimise UNSTRUCTURED
  decoherence, HOLD structured. No external task/labels. Purely internal.
- RECURRENT SETTLING + GUARDS IN LOOP: classify each disagreement STRUCTURED(hold+circulate) vs NOISE(reject,
  don't circulate -> defeats G1); damped update -> OBSERVE contraction (don't prove it; report blow-ups).
- COMBINED READ (get right): read settled state through EACH interface, COMBINE into max-self-consistent view
  -> consensus where they agree, HELD SUPERPOSITION where structured decoherence. Must surface held structure
  a consensus-collapse loses. (smoke shows structure is there; proper read must extract it -- linear shortcut didn't.)

## Watch (observe-don't-prove)
Run on FRAME-DIVERSE INJECTED input (frozen models no-op per xresolve; test where precondition is met). Observe:
settles (residual curve)? surfaces+holds injected structure? combined-read beats consensus-collapse at recovering
held structure? CONTROLS: coherent input -> HONEST NO-OP (settle to consensus, circulate nothing -- do NOT hack
to fake activity); noise input -> REJECT (no circulate, no amplify).

## Discipline
ONE loop, complete object, guards inside (not quarantined). No proofs/gates -- OBSERVE and report. No-op on
coherent input is CORRECT, never faked. Small/fast/CPU/reproducible. Commit code + RESULTS.md (three behaviors +
two controls + residual curve).
