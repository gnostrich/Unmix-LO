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
