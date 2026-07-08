# INDEXING-VALUE TEST — Claude Code brief

Read PREREG.md first (frozen thresholds) and run probe.py (the sandbox attempt that found the test
is ill-posed as first designed). Your job is NOT to make indexing look good. Your job is to run a
VALIDITY GATE, then only if it passes, the results test. Honesty of construction is the pass condition.

## STAGE 1 — VALIDITY GATE (this is the real gate; most likely outcome per probe.py is FAIL)
Question: can "present-but-entangled complementarity" be planted by a method BLIND to the recovery
mechanism, such that:
  (i)  complementarity is real:            best-single << joint-oracle (gap >= 0.15), AND
  (ii) a STRONG naive baseline fails:      naive-strong (best linear alignment CCA/Procrustes of the
       entangled spaces + tuned nonlinear readout, same budget) stays well below oracle, AND
  (iii) the oracle is actually reachable:  a readout on the TRUE (unentangled) features hits high acc
       (so the information is present and learnable in principle).

HARD CONSTRAINT (the integrity of the whole thing): the planting must be chosen BLIND — you may vary
the generative structure, entanglement mixing, dimensionality, sample size, readout capacity, but you
must NOT tune the interaction to the point where the choice is made BECAUSE indexed-beats-naive. The
planting family must be specified and fixed BEFORE checking whether indexing wins. Run a BOUNDED search
(say <= 12 pre-specified planting configs, listed before running) over generative structures that are
plausible/natural (additive+interaction mixes, disjoint-factor targets, varied entanglement), and for
each just measure (i)-(iii). You are searching for whether ANY blind config satisfies (i)-(iii) — a
task that is genuinely complementary, genuinely naive-hard, genuinely oracle-reachable.

VALIDITY VERDICT:
- If NO config in the fixed family satisfies (i)-(iii): the informative regime is NOT blindly
  constructible -> STOP. This is the answer: the "present-but-entangled-but-recoverable" band is not
  robustly plantable without aiming, consistent with DTI finding it empty. Composition thesis closes.
- If >=1 config satisfies (i)-(iii): the regime IS constructible -> proceed to STAGE 2 on those configs.

## STAGE 2 — RESULTS TEST (only if Stage 1 found valid configs)
On each valid config, three arms, equal readout capacity + equal tuning budget:
  best-single | naive-strong (fair, tuned) | INDEXED (Baur x MZ: run arbitrary probe data through the
  two frozen encoders, build the connective memory WITHOUT seeing y or the mixings, then readout).
Plus the MANDATORY negative control: a no-complementarity twin (target determined by either encoder
alone). Indexed MUST NOT beat naive there.
PASS iff ALL: indexed >= 1.15x naive-strong on the complementarity configs; indexed approaches oracle;
indexed does NOT beat naive on the no-complementarity control (anti-hallucination). Fail on any.
STOP either way. A pass means only "the narrow band is inhabitable," NOT "real tasks live there."

## Report
Stage-1 table (every config: single, oracle, naive-strong, verdict on i-iii). If Stage 2 ran: the three
arms + control with CIs. Confirm naive-strong was actually strong (report its alignment quality) so no
strawman. Keep prior negatives (BIOMESH cold-split, VIRTUALMESH G1) on record; this is the final
composition-thesis test either way.

## If everything fails / closes
Assemble the full honest negative: composition of frozen models creates no value via naive pooling
(BIOMESH), settling (G1), or indexed connective tissue (this test / non-constructibility). Surviving
validated results unchanged: G2 (scale-free routing cost), G3 (compression of reachable knowledge).
Infrastructure, not intelligence. That is the complete, honest deliverable.
