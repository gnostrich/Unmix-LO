# VIRTUALMESH: a virtual-model layer over frozen models via memory-kernel settling

**Status: SCAFFOLD.** Only result-independent sections are drafted (Sections 1, 2, 5, 6).
Sections 3 and 4 are stubs awaiting the three pre-registered gates in `gates/`. Per the build
brief (`virtualmesh/README.md`), no claim enters this paper's spec or evidence sections until
its gate has passed; a RED gate means the corresponding claim is reported as a characterized
negative and pruned from the construct's asserted properties. Nothing in this file will be
retro-fitted to results: thresholds quoted in Section 4 are copied verbatim from
`gates/README.md`, which was committed before any real-model run.

---

## 1. The construct

### 1.1 One paragraph

We define a **virtual-model layer**: a way to make many independently-pretrained, *frozen*
models jointly queryable as one virtual model, without merging their weights and without
selecting among them. The models are coupled through a shared representation frame; a query is
answered by letting the coupled system *settle* — iterate a reconciliation dynamics to a
path-coherent fixed point — rather than by pooling one-shot outputs. The settling dynamics is
given a Mori–Zwanzig (MZ) structure: the influence of the untracked remainder of the federation
on a routed subset of models is summarized by a memory kernel, and the conjectured low rank of
that kernel is what would make inference cost scale with routing width rather than federation
size. Sparse connectivity is (conjecturally) densified by *pathway thickening*: distilling
frequently-used multi-hop routes into direct edges, restricted to gaps that are interpolable
from existing shared structure. Everything past "coupled settling over a shared frame" in that
description is gated (Section 4) and is asserted nowhere in this paper until measured.

### 1.2 Objects: frozen models and frame-aligned representation spaces

The primitive objects are **frozen models** — independently pretrained networks whose weights
are never modified by the construct. What the layer actually operates on is not the weights but
each model's **representation space**, made mutually addressable through a **shared frame**: a
common coordinate system obtained by aligning each model's space against a small set of shared
anchors (relative representations over a few hundred probe inputs, or a shared base/embedding
model where the federation permits it). The frame is the construct's one maintained artifact:
it is cheap when models are frame-connected (shared base, shared anchor set) and is the
explicit cost we report (G1's "frame cost" clause). An object, precisely, is a pair
(frozen model, its frame-aligned representation space); the model contributes knowledge, the
frame contributes addressability.

### 1.3 Morphisms: lossy directed channels

Models are connected by **channels**: maps between frame-aligned representation spaces. Two
properties are definitional, not incidental:

- **Directed.** A channel from A to B is not, in general, invertible; A→B→A ≠ id. Losslessly
  reversible channels are a derived special case (the invertible fragment), not the ambient
  assumption. This is why the eventual formalization targets directed type theory (spec/):
  the symmetric identity type of cubical settings cannot express the generic case.
- **Lossy, gradedly.** Each channel carries a tolerance grade — how much structure it is
  permitted to collapse. Composition of channels is composition of paths through the mesh, and
  grades are intended to compose additively (triangle-inequality style, Lawvere-metric
  enrichment). The additive-composition *law* is conjectural until gated; the *definition*
  (channels carry grades; paths accumulate them) is part of the construct.

A path A→B→C is therefore a first-class citizen: a composite channel with an accumulated
grade, generally worse than a hypothetical direct edge, and never assumed reversible.

### 1.4 Settling to a path-coherent fixed point

A query is not routed to one model and not pooled across models. It is injected into the
frame, and the coupled system is iterated: each model updates its state given the
frame-transported states of its neighbors, for on the order of 5–20 steps, until the joint
state stops moving. The target of this relaxation is **path-coherence**: a joint state in
which every path between two objects transports the state to (graded-tolerance) agreement —
the diagram commutes up to the accumulated grades. Reconciliation, not selection: models that
disagree are pulled toward a mutually consistent state weighted by per-dimension confidence,
rather than having one winner chosen or their outputs averaged after the fact. This is
fixed-point inference in the sense of Deep Equilibrium Models, applied *across* frozen models
rather than within one network's layers. Which fixed point is reached is gauge-dependent — it
requires the anchor set to be well-posed (Section 5.3).

Two honest decompositions of "settling" are built into the evaluation rather than papered
over: (i) settling vs. *one-step* confidence-weighted consensus isolates what iteration adds
beyond fine-grained reconciliation (G1's ablation arm); (ii) settling vs. pooling isolates
what reconciliation adds beyond ensembling at all. The construct survives with a smaller claim
if iteration adds little; it does not survive if reconciliation adds nothing.

### 1.5 MZ memory-kernel routing

Settling over the full federation is unaffordable and unnecessary if most models are
irrelevant to a query. The Mori–Zwanzig formalism gives the principled reduction: project the
dynamics onto a **routed subset** of tracked models, and replace the untracked remainder by a
**memory kernel** — the tracked subset's next state is predicted from a short history of the
tracked subset alone, with the kernel carrying the integrated-out influence of everything
else. Three properties of this kernel are the load-bearing conjectures, all under G2:
the memory must actually be necessary (a Markovian closure should not suffice); a *short*
memory and a *low-rank* kernel must reproduce full-federation settling to tolerance; and the
kernel's effective rank must scale with the routing width K, not the federation size N. If the
rank tracks K, the kernel is a cheap index and the construct scales to large federations; if
it tracks N, the "innumerable models" story fails and the construct degrades to small fixed
federations (this fallback is stated, not hidden). A secondary conjecture, also under G2: the
closure *residual* correlates with query difficulty, giving a repair/slow-path trigger — route
wider exactly when the cheap index is insufficient.

### 1.6 Pathway thickening

A mesh built from real shared structure will be sparse: many model pairs are connected only
transitively (A–C only via B). **Pathway thickening** is the conjectured self-densification
mechanism: distill a frequently-traversed composite path A→B→C into a direct one-hop channel
A→C, which should match the composite to tolerance while being cheaper, and should generalize
to held-out transitively-connected pairs. The same machinery underwrites **gap-filling**: a
capability no single model has, but which is implied by two neighbors' joint structure, should
be producible by settling across them. Both claims are entirely unmeasured — no sandbox
prototype exists (G3 is the highest-risk gate) — and both are bounded by interpolability: only
gaps inside the convex hull of existing shared structure may be filled. Manufacturing an edge
between models that share nothing is fabrication, and G3's held-out and ground-truth checks
exist to catch exactly that (Section 5.4). If G3 reds, the "grows toward all-to-all" claim is
dropped and the construct keeps only the static-federation reconcile result.

---

## 2. Positioning

The layer-over-frozen-models framing is occupied territory; the claim to novelty must be
placed precisely or not at all. The distinctly-ours core, if the gates validate it, is:
**MZ-memory-kernel settling relaxation over frozen models, cost-bounded by kernel atomicity.**
Not the framing, not the graph, not the frozen experts — the settling dynamics plus the
memory-kernel scaling law.

### 2.1 Adjacent occupied fields (what this is NOT)

1. **Routers / MoE / mixture-of-agents.** Select-and-pool: a gating function picks one or a
   few experts, outputs are combined one-shot, and the interaction structure is flattened.
   There is no joint state, no iteration, no mutual constraint between experts at inference
   time. We settle and reconcile: models constrain each other through the frame until a fixed
   point, and the answer is decoded from the reconciled joint state.
2. **Model merging** (weight averaging, task arithmetic, TIES/DARE-style). Collapses the
   federation into one static model; the merged object cannot keep contradictory
   specializations distinct, and the merge is done once, offline. We keep models distinct and
   frozen, and couple them per-query; disagreement is resolved at inference by settling, not
   destroyed at merge time.
3. **Graph-MoE / expert-interaction graphs** — the closest live front, flagged as such. Here a
   graph over experts informs routing, so interaction structure is no longer flat. But the
   inference remains one-shot: the graph shapes which experts fire and how outputs combine; it
   does not run a relaxation to a fixed point, and there is no memory-kernel reduction of the
   untracked federation. What we add is settling-to-fixed-point plus memory-kernel dynamics.
   **Adjacency warning (kept in the final paper):** this field is moving fast; several
   candidate citations are 2025–26 and may be closer than their abstracts show. The graph-MoE
   "expert interactions" line in particular must be verified against the papers directly — not
   abstracts — before the settling/kernel distinction is claimed as uncontested. Until that
   verification is done, the distinction is asserted only as "not found in our sweep."
4. **MoFE / SAGMM** (layers over frozen experts). These already occupy "a trainable layer over
   frozen expert models" — which is why that framing is explicitly *not* our claimed novelty.
   They are gating layers: router-shaped, select-and-combine. We reconcile. The differentiator
   is the dynamics, and it must be earned by G1 (settling > pooling), not asserted.
5. **Scale-free / self-organizing networks** (network science). The right mathematics for
   connectivity growth and preferential thickening of used paths — and occupied as such. We do
   not claim the network-science mechanism; if G3 passes, we claim its *application* to
   representation-space channels between frozen models, with the interpolability bound. If G3
   fails, we cite the field and claim nothing.

### 2.2 Ancestors (what this builds ON, and does not reinvent)

1. **Relative representations** (and the structure-alignment line behind them): the shared
   frame. Anchor-based alignment of independently-trained representation spaces is their
   result; we consume it as infrastructure and report its cost. The prior stability gate in
   this repo (Section 6) independently rediscovered why anchors matter: without an
   anchor/gauge, "which coherent state" is not even well-posed.
2. **Deep Equilibrium Models**: fixed-point inference. DEQs establish that the useful output
   of a computation can be defined as the fixed point of a relaxation rather than the end of a
   feedforward pass, with the machinery to make that stable and differentiable. We lift the
   idea from layers-within-one-model to channels-between-frozen-models. The fixed-point
   *concept* is theirs; the cross-model settling and its empirical value are what our gates
   test.
3. **Graded modal type theory** (Moon–Eades–Orchard GrTT; Atkey's QTT; graded-modal Agda
   formalizations): the grades. Tolerance-annotated channels with semiring-valued grade
   composition are an *instance* of existing graded modal dependent type theory. The spec
   (Section 3) cites and instantiates; it does not reinvent. The one genuinely novel seam the
   spec may claim — a certification modality grounded in an *external empirical* valuation
   rather than an internal resource — is formalized only after gates, and is flagged in
   spec/README.md as the boundary-touching piece.

### 2.3 Honesty clauses (binding on all novelty claims in this paper)

- **"Not found in sweep" ≠ "novel."** Our prior-art search is a sweep, not a proof of absence.
  Adjacency is increasing, especially on the graph-MoE front, and the burden stays on us:
  every "distinctly ours" claim in this paper is implicitly qualified by the sweep's coverage
  and dated by it.
- Several works we position against are 2025–26 preprints whose abstracts may undersell their
  overlap with settling/kernel dynamics. The graph-MoE expert-interactions paper(s) must be
  read in full before the final version asserts the distinction of Section 2.1(3).
- The distinctly-novel core is the MZ-kernel settling relaxation (G1+G2) — *not* the
  layer-over-frozen-models framing (occupied: MoFE, SAGMM) and *not* scale-free
  self-organization (occupied: network science).

---

## 3. Formal specification [AWAITING GATES]

This section imports from `spec/` (Rzk, spec-level: directed homs for lossy channels, graded
tolerance composition, path-coherence as graded diagram commutation, the MZ-kernel atomicity
property, and the externally-valued certification modality) and will contain **only
gate-validated laws**. The result-independent type skeleton (objects, directed morphisms,
path-coherence as a definition) may be stated; every structural *law* — additive grade
composition, kernel low-rank/atomicity, thickening soundness — enters this section if and only
if the corresponding gate passes, and is otherwise reported in Section 4 as a characterized
negative and labeled conjectural or removed. Per the build discipline: a beautiful Rzk spec is
not a substitute for a measured number, and nothing is formalized that is not measured.
**[AWAITING GATES — see gates/ and spec/].**

---

## 4. Evidence [PENDING]

Three pre-registered gates, thresholds committed in `gates/README.md` before any real-model
run and quoted verbatim below. Results will be reported honestly, including any RED — a
characterized negative is publishable and protects the integrity of whatever survives.
Sandbox prototypes are labeled as what they are: synthetic, clean/linear, and **not evidence
about real models**; they justify running the real gates, nothing more.

### 4.1 G1 — Reconcile beats pooling [PENDING — see gates/]

Pre-committed threshold (verbatim from gates/README.md):

> PASS iff SETTLING beats POOLING by >=10% relative on split-knowledge queries.
>
> Report SETTLING vs ONE-STEP: if settling ~= one-step, the honest claim shrinks to
> "fine-grained reconciliation > pooling" (still beats routers, but recurrence adds little).
> Record which.
>
> Report the frame cost (anchors/alignment needed) — it's the maintained artifact.
>
> FAIL (settling ~= pooling) => it's an ensemble with extra steps; recommend the merge, stop.

Sandbox (clean/linear, synthetic split-dimension world — **not a real-model result**):
passed; settling beat pooling 1.5x when knowledge was split, with most of the gain from
per-dimension confidence-weighting and some from iteration. The real test must separate these
two contributions (hence the one-step ablation arm).

Real-model result: **[PENDING]**.

### 4.2 G2 — MZ kernel is low-rank on real models [PENDING — see gates/]

Pre-committed threshold (verbatim from gates/README.md):

> Fit an MZ closure (predict tracked-model settled state from a short history of the tracked
> subset only). Measure: closure error vs memory length L; kernel effective rank vs
> routed-subset size K; kernel rank vs total N (vary N by adding models).
>
> PASS iff (a) a short-memory low-rank closure reproduces settling to acceptable tolerance AND
> (b) kernel rank scales with K (routing width), NOT with N (federation size).
>
> Also: does the closure RESIDUAL correlate with query difficulty (a repair/slow-path trigger)?
>
> FAIL (rank grows with N, or memory doesn't help) => not scale-invariant / not cheap; the
> "innumerable models" story fails; downgrade to small fixed federations.

Sandbox (clean/linear, exact — **not a real-model result**): passed; a short memory kernel
reproduced settling exactly, kernel effective rank tracked the routed subset size rather than
federation size, memory (vs. Markovian) was necessary, and residual tracked difficulty. The
real test is nonlinear and approximate; none of the linear exactness transfers by argument.

Real-model result: **[PENDING]**.

### 4.3 G3 — Pathway thickening / gap-filling [PENDING — see gates/]

No prototype exists for G3, sandbox or otherwise; this claim is unmeasured in any setting and
is the highest-risk gate.

Pre-committed threshold (verbatim from gates/README.md):

> PRE-COMMITTED PASS iff the distilled direct edge matches the composite A->B->C to tolerance
> (it should, and be cheaper); AND on a HELD-OUT gap (a pair connected only transitively) the
> synthesized edge generalizes (beats no-edge / random-edge baseline).
>
> Gap-filling / virtual-model test: pick a capability no single model has but that is IMPLIED
> by two neighbors' joint structure (e.g. a compositional query). Does settling across the two
> produce it, and does it BEAT each model alone? Guard: a "stable emergent capability" that is
> confidently WRONG is a FAIL — verify against held-out ground truth, not just
> self-consistency.
>
> FAIL => the mesh cannot self-densify honestly; drop the "grows toward all-to-all" claim,
> keep only the static-federation reconcile result.

Real-model result: **[PENDING]**.

---

## 5. Bounds

Stated plainly, because the construct's honest scope is part of the contribution.

### 5.1 Connected component only

The virtual model is bounded by the connected component of the mesh under *real shared
structure*. Models reachable only through channels that do not exist — because the models
share no base, no anchor-alignable structure, no overlapping competence — are simply outside
the virtual model. There is no mechanism, and we claim none, for coupling models that share
nothing.

### 5.2 Interpolable gaps only

Gap-filling (if G3 passes) operates strictly inside the convex hull of existing structure:
a missing capability can be synthesized only when it is implied by the joint structure of
connected neighbors. Extrapolation beyond what any combination of the federation's models
supports is out of scope by construction, and no thickened edge may be created where the
composite path does not already exist.

### 5.3 Anchor / gauge dependence

*Which* path-coherent state the system settles to is gauge-dependent: the coupling objectives
constrain the joint state only up to a symmetry group, and the anchor set (the shared frame)
is what fixes the gauge. Without it, "the" coherent state is not well-defined — different runs
settle to different, mutually incompatible reconciliations. This is not a hypothetical: the
prior stability gate in this repo (Section 6) failed on precisely this point, with training
objectives that pinned down solutions only up to a large symmetry group and seeds landing at
arbitrary points of it. The frame is therefore load-bearing, and its cost (anchors, alignment
maintenance) is reported as part of any headline number, per G1.

### 5.4 Fabrication guard

Emergent capabilities require the external consistency check or they fabricate. A settled
state can be stable, self-consistent, and confidently wrong; self-consistency is not evidence.
Every claimed emergent capability must be verified against held-out ground truth (G3's guard
clause), every synthesized edge must beat no-edge and random-edge baselines on held-out gaps,
and a "stable" result whose stability is attributable to frozen corpus geometry rather than to
the mechanism is a FAIL (the attribution lesson of the stability gate, Section 6). The
fabrication guard is why the certification modality in spec/ is grounded in an *external*
empirical valuation and not internalized.

---

## 6. Related program: the three RED gates behind this discipline

This construct is the fourth pre-registered gate cycle in this repository, and the first three
all returned honest RED. They are cited here not as failures to bury but as the epistemic
lineage: each one killed an elegant piece of machinery that was floating on an unvalidated
claim, and each shaped a rule this paper obeys. Machinery in this program is trusted only
after a gate passes.

1. **Gradient compositionality** (`GATE_RESULTS.md`, 2026-07-07): RED. Real per-task gradients
   at the scale tested contain stable, individual task-cluster structure, but held-out tasks
   are not sparse combinations of a shared component basis (REUSED residual 0.48–0.77 vs. bar
   0.3, robust across two models and seven analysis configs). The compositional self-refining
   optimizer was not built. Lesson inherited here: the world's structure does not
   automatically reach into the substrate the way a thesis needs; measure the load-bearing
   premise first.
2. **Oracle-substrate reuse** (`AGDA_RESULTS.md`, 2026-07-07): RED at the loop level (held-out
   reuse 2.2%, cross-domain 0%), with the failure honestly localized to the composer (Agsy's
   8.7% composability ceiling), not the substrate — whose human corpus is unambiguously
   compositional. Lesson inherited here: bound what your negative actually bounds, and report
   the asterisk; a RED that localizes is more useful than a vague green.
3. **Navigator-enacted stability** (`STABILITY_GATE.md`, 2026-07-07): FAIL by the attribution
   guard, not the headline number. Cross-seed partition stability was 0.948 — but untrained
   channels scored 0.987 and raw frozen-encoder geometry 0.89–0.95, while the trained channels
   themselves agreed across seeds at cosine ≈ 0.01: "what is stable is not enacted, and what
   is enacted is not stable." Lessons inherited here directly: (a) a stable-looking result
   attributable to frozen corpus geometry is a FAIL — hence G2's attribution-style controls
   and Section 5.4; (b) objectives that constrain solutions only up to a symmetry group need
   explicit gauge fixing — hence the anchor/frame of Section 5.3, which is exactly the
   canonicalization pressure that gate's post-mortem prescribed.

The common failure mode was always the same: elegant machinery built on an unvalidated claim
that floats. The present paper is structured so that it *cannot* do that — Sections 3 and 4
are empty until the gates fill them, and if the gates red, this document becomes a precise
construct definition paired with characterized negatives, which is the honest deliverable.

---

*Scaffold notes (remove in final): Section 3 imports from spec/ after the merge step; Section
4 subsections are filled from per-gate RESULTS.md files, numbers unedited; Section 2.1(3)
requires the direct graph-MoE paper verification before final claims; citations to be added
for relative representations, DEQ, GrTT/QTT, MoFE, SAGMM, and the graph-MoE line during the
merge step.*
