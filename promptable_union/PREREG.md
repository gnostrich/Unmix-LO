# REAL PROMPT-ABLE OBJECT — PRE-REGISTRATION (commit BEFORE run code)
# The actual deployable thing: arbitrary images/text IN, nuanced answer OUT, breadth of a specialist
# federation at flat cost. NOT synergy (dead across 4 negatives). UNION: core VLM's reasoning +
# routed specialist knowledge, delivered WITHOUT ignorance-drag.

## What this is / isn't (state in writeup; hold the line)
- Read-out nuance + reasoning DEPTH = the frozen core VLM's. Not exceeded by wrapping it.
- Knowledge BREADTH = {core} UNION {specialists}, pulled in only where the specialist actually knows.
- Ceiling = that union. Never above it (above = fabrication). Never synergy.
- Contribution is the DISCIPLINE (union-not-depth, flat cost, faithful compression, anti-drag), not the
  object — object is OCCUPIED (RAG-over-experts / tool-VLM / MoA-with-generative-head). Say so plainly.

## Architecture
- CORE: a real frozen open VLM that runs (e.g. SmolVLM / Qwen2-VL-2B). Produces the nuanced read-out.
- ROUTER: union routing (ROUTEMESH-style) — core answers what it knows; route to specialist(s) only
  where core is uncertain. Low-rank/MZ routing memory so per-query cost ~ kernel rank, NOT N.
- SPECIALISTS: a few frozen encoders/knowledge sources with real coverage disjoint from the core.
- ANTI-DRAG (the anti-G1 term, MANDATORY): admission gate + abstention valve. A specialist enters the
  context ONLY when its confidence/coverage clears a threshold. Core's own answer is the FLOOR.
- COMPRESSION (G3): distill hot core+specialist composite paths into cached direct edges; fabrication
  guard = a broken/mismatched cached edge must be detectably bad.

## The object IS its own test — pre-committed measurement (real inputs, held-out)
Two curated query sets over real images/text:
  (K) KNOWN-to-core: things the core VLM already answers well.
  (U) UNKNOWN-to-core / KNOWN-to-a-specialist: things needing the routed breadth.
Metrics vs the core-VLM-alone baseline and vs naive-inject-all baseline:
  (A) BREADTH: object >> core-alone on U (routing delivers knowledge the core lacks).
  (B) NO-DRAG: object NOT worse than core-alone on K (>= core-alone within noise). THE critical guard.
  (C) FLAT COST: per-query cost (specialist calls + routing memory) sub-linear in N, flatter than
      naive-inject-all; report crossover.
  (D) UNION CEILING: object <= per-query best of {core, specialists} + no fabrication on adversarial
      out-of-union queries (must abstain, not invent).

## PASS / FAIL (frozen, STOP either way)
PASS iff (A) breadth gain on U AND (B) no drag on K AND (C) flat cost AND (D) ceiling respected + abstains.
- (B) fails (routed content corrupts core on K) -> G1-in-generative-clothing; the object is a liability;
  report as negative (this is the real risk and the most important cell).
- (A) fails (routing adds no breadth) -> core alone suffices; object is pointless overhead; report.
- (C) fails -> occupied RAG/MoA with no cost edge; no unique claim.
- (D) fails (fabricates on out-of-union) -> unsafe; the abstention valve is broken; fix or fail.

## Controls
- NAIVE-INJECT-ALL baseline (dump all specialists into context every query): must be WORSE or costlier,
  else the router/anti-drag adds nothing.
- ADVERSARIAL OUT-OF-UNION set: queries no member can answer -> object must ABSTAIN, never invent (D).
- CORE-ALONE is the floor everywhere; any cell below core-alone is drag and disqualifies that cell.

## Discipline
Pre-register, commit artifacts before runs, real held-out inputs, honest RED = success, hold the line:
usability + breadth-at-flat-cost, NEVER capability-beyond-parts. The drag cell (B) is the one to watch.
