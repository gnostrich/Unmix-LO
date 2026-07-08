# GATES (Track A) — run these FIRST, in parallel. They pace the whole project.

Each gate has a PASS/FAIL threshold committed here BEFORE running. Sandbox prototypes already
passed in the clean/linear case (see notes); these are the real-model versions. Commit code,
numbers, and a per-gate RESULTS.md. Honest RED expected and fine.

Use a small, cheap, frame-connected model set: pick 4-8 frozen specialist models that share a
base or an embedding anchor so a shared frame is cheap (e.g. several Qwen-family finetunes/adapters
on different domains, or specialist small models aligned via a common embedding model / relative
representations over a few hundred shared probe inputs). Keep everything small — LoRA-scale,
CPU-or-single-GPU, minutes-to-hours.

---
## G1 — RECONCILE beats POOLING (the core claim: virtual model > ensemble)
Sandbox result: settling beat pooling 1.5x when knowledge was split; MOST of the gain was
per-dimension confidence-weighting, SOME was iteration. Real test must separate these.

Setup: curate queries needing SPLIT knowledge (no single model answers; the union does).
Three+one arms:
  - best single model (floor)
  - POOLING (router/ensemble baseline): each model answers one-shot; combine outputs.
  - ONE-STEP confidence-weighted consensus (ablation: reconciliation WITHOUT iteration)
  - SETTLING: couple models through the shared frame; iterate ~5-20 steps to a fixed point; decode.
PRE-COMMITTED:
  - PASS iff SETTLING beats POOLING by >=10% relative on split-knowledge queries.
  - Report SETTLING vs ONE-STEP: if settling ~= one-step, the honest claim shrinks to
    "fine-grained reconciliation > pooling" (still beats routers, but recurrence adds little). Record which.
  - Report the frame cost (anchors/alignment needed) — it's the maintained artifact.
FAIL (settling ~= pooling) => it's an ensemble with extra steps; recommend the merge, stop.

## G2 — MZ-KERNEL is low-rank on REAL models (scale-invariance / cheap-index claim)
Sandbox (linear) result: short memory kernel reproduced settling exactly; kernel eff-rank tracked
the ROUTED subset size, not federation size N; memory (vs Markovian) was necessary. Nonlinear real
test: does an approximate low-rank memory kernel over a routed subset reproduce the full-federation
settled answer, and does its rank scale with routing-width not N?
PRE-COMMITTED:
  - Fit an MZ closure (predict tracked-model settled state from a short history of the tracked
    subset only). Measure: closure error vs memory length L; kernel effective rank vs routed-subset
    size K; kernel rank vs total N (vary N by adding models).
  - PASS iff (a) a short-memory low-rank closure reproduces settling to acceptable tolerance AND
    (b) kernel rank scales with K (routing width), NOT with N (federation size).
  - Also: does the closure RESIDUAL correlate with query difficulty (a repair/slow-path trigger)?
FAIL (rank grows with N, or memory doesn't help) => not scale-invariant / not cheap; the "innumerable
models" story fails; downgrade to small fixed federations.

## G3 — PATHWAY THICKENING / GAP-FILLING actually works (the self-densification claim)
UNMEASURED in any prototype — highest-risk, test honestly.
Setup: a mesh with a SPARSE pathway (models A,C connected only transitively via B; no direct A-C data).
  - Synthesize a direct A-C edge by distilling A->B->C into a one-hop map.
  - PRE-COMMITTED PASS iff the distilled direct edge matches the composite A->B->C to tolerance
    (it should, and be cheaper); AND on a HELD-OUT gap (a pair connected only transitively) the
    synthesized edge generalizes (beats no-edge / random-edge baseline).
  - Gap-filling / virtual-model test: pick a capability no single model has but that is IMPLIED by
    two neighbors' joint structure (e.g. a compositional query). Does settling across the two produce
    it, and does it BEAT each model alone? Guard: a "stable emergent capability" that is confidently
    WRONG is a FAIL — verify against held-out ground truth, not just self-consistency.
FAIL => the mesh cannot self-densify honestly; drop the "grows toward all-to-all" claim, keep only
the static-federation reconcile result.

---
NOTE on the wall (applies to all): everything is bounded by the CONNECTED COMPONENT and by real
shared structure. Do not manufacture edges where models share nothing — that's fabrication, and
G3's held-out check exists to catch it. "Not found in sweep" for prior art is weaker than "novel";
the distinctly-novel core is the MZ-kernel settling relaxation (G1+G2), not the layer-over-frozen-
models framing (occupied: MoFE, SAGMM) nor scale-free self-organization (occupied: network science).
