# VIRTUALMESH — build brief (multi-agent, gate-controlled)

## What this is
A distributed-INFERENCE construct: unify many independently-pretrained frozen models into one
queryable "virtual model" by coupling them through a shared frame and letting a Baur-trained,
Mori-Zwanzig-structured process settle them to a path-coherent joint state — routing through
short high-overlap paths, thickening real pathways, filling interpolable gaps. NOT a router
(routers select+pool one-shot, flattened); this RECONCILES (settles to a fixed point). NOT
merging (merging collapses to one static model); this keeps models distinct and couples them.

## Non-negotiable discipline (read before doing anything)
Three prior pre-registered gates in this line all came back RED/FAIL when tested honestly
(gradient compositionality; oracle-substrate reuse; navigator-enacted stability). The failure
mode was always: elegant machinery built on an unvalidated claim that floats. So:
- **The formal spec (Track B) is DOWNSTREAM of the gates (Track A).** Formalize ONLY claims a
  gate has passed. A red gate means that part is NOT written into the spec or the paper.
- Pre-commit thresholds BEFORE running. Honest RED is a success, not a failure.
- Watch for degenerate wins: collapse, steganographic pass-through, "stable" that is just frozen
  corpus geometry (see gate G2 attribution clause). A stable-looking result that is an artifact is a FAIL.
- Do not let a beautiful Rzk spec substitute for a measured number.

## Parallelization (fan out agents; these are independent until the merge)
- Agents A1-A3: the three real-model gates (gates/). START HERE — they pace everything.
- Agents B1-B2: paper scaffold + Rzk spec skeleton for the RESULT-INDEPENDENT parts only
  (construct definition, positioning vs adjacent fields, type skeleton). May run concurrently but
  must NOT formalize any claim still gated.
- Agent C1: MVP integration (mvp/) — assemble the passing pieces into a runnable virtual-mesh demo.
- Merge step (after gates return): promote only PASSED claims into spec + paper; prune the rest.

## Success = a paper that PAIRS a precise formal construct (Rzk spec of validated claims) with
## real-model evidence. Spec-alone = vapor. MVP-alone = a demo the graph-MoE field absorbs. Both.

See gates/README.md, spec/README.md, paper/README.md, mvp/README.md for each track.

---

## OUTCOME (2026-07-08 — gates run, merge executed)

| gate | verdict | one line |
|---|---|---|
| G1 reconcile>pooling | **FAIL** | settling is a hallucination amplifier without calibrated ignorance (fact-precision 0.018; pooling also lost to best-single) — `gates/GATE1_RESULTS.md` |
| G2 MZ kernel | **PASS (scoped)** | exact L=2 closure; kernel complexity flat across N=4..10; linear instantiation, rank-at-cap caveat — `gates/GATE2_RESULTS.md` |
| G3 thickening | **PASS (amended)** | distilled edges perfectly cache real 2-hop chains at half cost, transfer to a held-out pair, and the guard refuses fabricated edges — `gates/GATE3_RESULTS.md` |

Merge protocol executed: `spec/virtualmesh.md` promotes only the scoped G2/G3 laws (refuted
G1 laws live in its §II.R register), `paper/paper.md` reports all three with the FAIL as a
headline result, and `mvp/demo_thicken.py` demos the passed mechanism only. Pre-registration,
amendments, and two invalidated runs are all on the record in `gates/`.
