# INDEXING-VALUE TEST — PRE-REGISTRATION (commit BEFORE any run code)
# Question: does a Baur x Mori-Zwanzig INDEXING phase over frozen models surface
# present-but-ENTANGLED complementarity that a STRONG naive baseline cannot?
# This tests the ACTUAL object (indexed connective tissue), not naive pooling.

## The one claim
Indexing adds value IFF it wins where complementarity is real-but-entangled AND does NOT
win where complementarity is absent. Both halves required. A win on both = the indexer
hallucinates structure (disqualifying). A win on neither = the indexer is inert.

## Design integrity (guards against designer-intent leaking into the result)
G-A. PLANTING IS INDEPENDENT OF RECOVERY. The complementarity is planted by a process with
     NO shared structure with the Baur/MZ recovery: two frozen "models" are trained on DISJOINT
     causal halves of a target, then their representations are entangled by a RANDOM invertible
     mixing the indexer never sees. The indexer must recover usefulness without knowing the mixing.
G-B. NAIVE POOLING IS STRONG, NOT A STRAWMAN. The naive baseline gets: best linear alignment
     (Procrustes/CCA), tuned combination, same readout capacity, and the SAME compute/tuning budget
     as the indexer. If indexing wins, it must beat a fully-fair naive baseline.
G-C. MANDATORY NEGATIVE CONTROL. A twin task with NO complementarity (both models see the same
     information; target is determined by either alone). Indexing MUST NOT beat naive there.
G-D. FROZEN THRESHOLDS + STOP-EITHER-WAY. No re-running with adjusted knobs. One run. Stop on any verdict.

## Setup (constructed, controlled)
- Two frozen encoders f_A, f_B (small nets), each trained on a DISJOINT half of the generative
  factors of a synthetic target y. Neither alone determines y; TOGETHER they do -> complementarity
  is real BY CONSTRUCTION (verify: best-single accuracy << joint-oracle accuracy).
- ENTANGLE: present each encoder's features through a fixed random invertible mixing M_A, M_B
  (the "gauge" — independently-trained models live in mismatched coordinates). The complementary
  signal is now present but rotated into a form naive pooling cannot linearly combine to recover.
- INDEXING PHASE: run ARBITRARY probe data through both frozen encoders; Baur x MZ process builds
  the connective memory (learns the relational frame between the two entangled spaces) WITHOUT
  seeing y and WITHOUT seeing M_A,M_B. Then a light readout on the indexed joint representation.

## Arms (all get equal readout capacity + equal tuning budget)
1. best-single (floor)
2. NAIVE-STRONG: best linear alignment (CCA/Procrustes) of the two entangled spaces + tuned pool + readout
3. INDEXED: Baur x MZ indexed frame + readout
Run on BOTH the complementarity task and the no-complementarity control.

## PRE-COMMITTED verdict (frozen)
INDEXING-VALUE = PASS iff ALL of:
  (a) complementarity task: INDEXED accuracy >= 1.15x NAIVE-STRONG (clear, non-noise margin), AND
  (b) INDEXED approaches joint-oracle (recovers a real fraction of the planted complementarity), AND
  (c) no-complementarity control: INDEXED does NOT beat NAIVE-STRONG (within noise) -- the anti-hallucination guard.
FAIL iff any of: indexed ~= naive on the complementarity task (inert / entanglement not recoverable),
  OR indexed beats naive on the control (hallucinating structure -> whole result disqualified),
  OR both-single already ~= joint (complementarity wasn't actually planted -> task invalid, rebuild not re-judge).

## What a PASS means (kept modest -- guard against inflation)
Only that the narrow band (present-but-entangled complementarity) is INHABITABLE and indexing can
work it where strong-naive cannot. NOT that real tasks live there. A pass earns the right to hunt a
real task next; it does not prove real-world value. A FAIL is dispositive: across naive (dead, BIOMESH),
settling (dead, G1), and now indexed-on-planted-complementarity, the composition thesis is closed.

## Report regardless
- best-single / naive-strong / indexed accuracy on BOTH tasks, joint-oracle ceiling, margins with CIs.
- confirm the planting worked (single << oracle on complementarity task; single ~= oracle on control).
- confirm naive-strong was actually strong (report its alignment quality) so a win isn't a strawman.
