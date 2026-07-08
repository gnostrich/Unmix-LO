# VIRTUALMESH: thickening a mesh of frozen models — and why settling fails without calibrated ignorance

**Status: MERGED (2026-07-08).** The three pre-registered real-model gates have run and the
merge protocol (`virtualmesh/paper/README.md`) has been executed on this document:
**G1 FAIL** — the core settling/reconciliation claim is refuted at this scale; the mechanism
(a positive-feedback hallucination cascade under uncalibrated confidence) is characterized and
reported as a headline result, not a footnote (Section 4.1); **G2 PASS, scoped** — exact
short-memory MZ closure with federation-size-independent complexity, linear instantiation only
(Section 4.2); **G3 PASS, amended design** — pathway thickening as exact compression of
composite structure, bounded by the chain ceiling, amendment on the record (Section 4.3).
Per the build brief (`virtualmesh/README.md`), only gate-validated claims are promoted to the
spec (Section 3); the refuted claim is pruned from the construct's asserted properties and
kept as a characterized negative. Nothing was retro-fitted: thresholds quoted in Section 4 are
copied verbatim from `gates/README.md`, committed before any real-model run; all design
corrections and the one pre-registration amendment are documented in the per-gate results
files and summarized where they matter.

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
be producible by settling across them. Both claims entered the program unmeasured (G3 was the
highest-risk gate; see Section 4.3 for what survived) and both are bounded by interpolability: only
gaps inside the convex hull of existing shared structure may be filled. Manufacturing an edge
between models that share nothing is fabrication, and G3's held-out and ground-truth checks
exist to catch exactly that (Section 5.4). If G3 reds, the "grows toward all-to-all" claim is
dropped and the construct keeps only the static-federation reconcile result.

---

## 2. Positioning

The layer-over-frozen-models framing is occupied territory; the claim to novelty must be
placed precisely or not at all. The distinctly-ours core, as pre-registered, was:
**MZ-memory-kernel settling relaxation over frozen models, cost-bounded by kernel atomicity.**
Not the framing, not the graph, not the frozen experts — the settling dynamics plus the
memory-kernel scaling law.

**Post-merge note (binding on this section).** The gates validated only part of that core.
The settling half FAILED on real specialists (G1, Section 4.1): the comparisons below that
turn on "we settle and reconcile" describe the *construct as designed*, and the design is now
known to require a precondition — calibrated per-fact ignorance — that the tested scale does
not provide (Section 5.5). The kernel half passed only in scoped, linear-instantiation form
(G2, Section 4.2), and thickening passed as compression-not-capability (G3, Section 4.3).
No positioning claim below may be read as an empirical superiority claim over the adjacent
fields; the empirical record is Section 4, including its headline negative.

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

## 3. Formal specification [MERGED 2026-07-08]

This section imports from `spec/` (Rzk, spec-level) and contains **only gate-validated laws**.
The result-independent type skeleton stands as drafted: objects, directed non-invertible
morphisms, graded composition as *definition*, path-coherence as *definition*. The merge
disposition of the structural laws is:

- **Refuted — NOT in the spec (G1 FAIL).** The settling/reconciliation law (spec Law G1-B:
  coupled settling reaches a path-coherent fixed point that improves on pooling) and every
  graded-refinement claim gated on G1 — including the additive grade-composition law, whose
  planned empirical validation path ran through G1 — are not promoted. They are recorded as
  the characterized negative of Section 4.1. Grade composition remains in the skeleton as a
  labeled *conjecture*, not an asserted law.
- **Promoted, scoped (G2 PASS).** The MZ-closure law, in exactly the scoped form of
  GATE2_RESULTS.md: settling over frame-aligned real representation spaces *with linear
  channels* admits an exact short-memory (L=2) closure over the routed subset, with closure
  complexity independent of federation size (rank flat at 120 across N=4..10). NOT promoted:
  atomicity-as-compression (rank << K·d was not shown — rank sat at the dimensional cap),
  the residual-as-repair-trigger clause (did not replicate, r=-0.07), and any claim about
  nonlinear settling dynamics.
- **Promoted, amended (G3 PASS).** The thickening-soundness law: a distilled direct edge is a
  functional cache of the composite path (agreement 1.00 incl. unseen paraphrase), bounded
  exactly by the chain's ground-truth ceiling, at lower inference cost; and the
  fabrication-guard law: an edge distilled across models sharing no real path scores below
  base-rate and is thereby detectable. Promoted under the amended held-out design
  (held-out unit = transitively-connected model pair, per gates/README.md's own G3 wording),
  with the amendment and the discarded as-preregistered run on the record in GATE3_RESULTS.md.

The externally-valued certification modality remains formalizable in principle but now lacks
its intended principal client (G1's settled states); it is retained in spec/ as scaffolding
for the G3-validated guard semantics only. Per the build discipline: a beautiful Rzk spec is
not a substitute for a measured number, and nothing is formalized here that was not measured.

---

## 4. Evidence

Three pre-registered gates, thresholds committed in `gates/README.md` before any real-model
run and quoted verbatim below; real runs 2026-07-08 (GATE1_RESULTS.md, GATE2_RESULTS.md,
GATE3_RESULTS.md; sandbox record in SANDBOX_RESULTS.md). Scoreboard: **G1 FAIL, G2 PASS
(scoped), G3 PASS (amended design)**. The G1 negative — with its mechanism — is a headline
result of this paper, reported at the same prominence as the passes. Sandbox prototypes are
labeled as what they are: synthetic, clean/linear, and **not evidence about real models**;
G1's sandbox is doubly instructive because the real gate identified exactly which sandbox
assumption was load-bearing and false.

### 4.1 G1 — Reconcile beats pooling: **FAIL** (headline negative)

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
per-dimension confidence-weighting and some from iteration (reproduced 2026-07-08:
settling 1.50x over pooling, SANDBOX_RESULTS.md).

**Real-model result (2026-07-08, GATE1_RESULTS.md): FAIL — all three pre-registered pass
conditions fail; settling vs. pooling relative gain is -100%.** Federation: four rank-8 LoRA
specialists on frozen Qwen2.5-0.5B-Instruct, each at 1.00 single-hop accuracy on its own
relation; 40 split-knowledge multi-hop queries, ground-truth scored.

| arm | accuracy | fact-precision | facts admitted |
|---|---|---|---|
| best single model | **0.15** | — | — |
| POOLING (confidence-weighted vote) | 0.05 | — | — |
| ONE-STEP reconciliation | 0.00 | 0.547 | 64 |
| SETTLING (<=5 rounds to fixed point) | **0.00** | **0.018** | 2,171 |

The characterized mechanism, which is the result:

1. **Confidence calibration on real 0.5B specialists is nearly nonexistent.** Correct-key vs.
   wrong-type-key mean logprob gaps of 0.03–0.07 nats; a LoRA specialist asked about a
   wrong-type entity confabulates an answer of its own relation type at almost its trained
   confidence.
2. **Iteration amplifies confabulation.** Each admitted junk fact seeds further confabulation
   next round — a positive-feedback hallucination cascade. One step admitted 64 facts at 0.547
   precision; five rounds admitted 2,171 at 0.018 (~54 facts/query, 98.2% false).
3. **Pooling ALSO lost to the best single model** (0.05 vs. 0.15): three of four specialists
   cannot answer any multi-hop query, and their confident wrong votes drown the
   sometimes-right one. At this scale both aggregation baselines lose to argmax-model — a
   finding that cuts against ensembling generally, not only against our construct.
4. **The sandbox's 1.50x win silently assumed calibrated ignorance** — a sandbox model
   contributed only on dimensions it truly knew (mask=0 elsewhere). Real small specialists
   have no such mask. That assumption, not the settling algebra, was the load-bearing part.

Consequences (per the pre-registered decision rule): the settling/reconciliation law is not
promoted to the spec (Section 3); the MVP omits settling. The honest residual claim is only:
on split-knowledge multi-hop queries at this scale, one-shot pooling of small specialists is
worse than best-single, and recurrent settling without calibrated confidence is worse still —
it amplifies hallucination. Bound for any retry: the missing precondition is per-fact
calibration (abstention), not more iteration or a better frame; a retry must gate
contributions on a verifier at >=0.8 fact-precision *before* settling gets another test, as a
separately pre-registered experiment. Nothing here licenses building the settling layer.

### 4.2 G2 — MZ kernel is low-rank on real models: **PASS (scoped)**

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
federation size, memory (vs. Markovian) was necessary, and residual tracked difficulty
(r=+0.29).

**Real-model result (2026-07-08, GATE2_RESULTS.md): PASS on all pre-registered criteria.**
Nodes = (model, layer) hidden-state spaces of the four specialists plus the frozen base
(layers 10/16/22), 240 shared probes, PCA to d=40 per node; channels = spectrally-capped
ridge maps fitted on half the probes; damped coupled settling to T=40 on the held-out half.

| measure | result | criterion |
|---|---|---|
| closure error vs. memory length | L=1: 0.084 -> **L=2: 0.000 (exact)** | some L<=8 < 0.15 — pass |
| memory necessity | Markovian 0.084 vs. L=2 exact | memory helps — pass |
| kernel eff-rank vs. routed width K | 80 / 160 / 240 for K=2/4/6 | grows with K — pass |
| kernel eff-rank vs. federation size N | **120 / 120 / 120 / 120** for N=4/6/8/10 | flat ±2 — pass |
| residual-difficulty correlation | **-0.07** (sandbox: +0.29) | reported, not gating |

The settling dynamics of a real-geometry federation are closable from a short history of the
routed subset alone, and the closure's complexity does not grow with federation size — the
scale-invariance property the "innumerable models" story needs.

Honest scope limits (from GATE2_RESULTS.md, binding on every use of this result):

1. **Rank-at-cap.** Effective rank equals the output-dimension cap K·d in every
   configuration, so "rank grows with K" is partly dimensional necessity. The STRONGER
   atomicity reading — a compressed index with rank << K·d — is NOT demonstrated. What is
   demonstrated is exactly the pre-registered pair: short-memory closability plus
   N-independence.
2. **Linear instantiation.** The coupled dynamics use linear (ridge) channels over real model
   geometry with linear damping. The geometry is real; the dynamics are not nonlinear. The
   fully nonlinear settling of gates/README.md lived in G1's text-space protocol, which failed
   upstream (hallucination cascade) before any kernel question could be posed there. A
   nonlinear-dynamics G2 remains untested.
3. **The repair-trigger signal did not replicate** on real geometry (-0.07 vs. sandbox +0.29).
   The corresponding spec clause stays unpromoted.

Promoted claim, scoped form only: settling over frame-aligned real representation spaces with
linear channels admits an exact short-memory MZ closure over the routed subset, with closure
complexity independent of federation size (N=4..10, 15 real nodes available). Not promoted:
atomicity-as-compression, residual-as-repair-trigger, nonlinear settling.

### 4.3 G3 — Pathway thickening / gap-filling: **PASS (amended design)**

G3 entered the program with no prototype (the highest-risk gate). A sandbox prototype was
built and passed on 2026-07-08 (SANDBOX_RESULTS.md, with two documented test-construction
corrections and one honestly measured bound: dimensions invisible to the intermediate model
stay at rel-error 0.99 — a transitive edge cannot carry them and does not pretend to).

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

**Real-model result (2026-07-08, GATE3_RESULTS.md): PASS under an amended held-out design;
the amendment is on the record and documented below.** Real 2-hop chains of specialist calls
(all four specialists at 1.00 single-hop accuracy): pair 1 = person -> city -> company,
distilled into a direct rank-8 LoRA edge on the frozen base; pair 2 = the held-out gap
(city -> company -> product), the same procedure applied unchanged to a transitively-connected
pair it was not designed against; fabrication guard = person -> hobby fed into the
city-to-company model — a chain across models sharing no real path. Thresholds: agreement
>= 0.90; >= +20% relative vs. both controls; guard <= +5%.

| measure | pair 1 | pair 2 (held-out gap) |
|---|---|---|
| agreement with chain, trained template | 1.00 | 1.00 |
| agreement with chain, UNSEEN paraphrase | **1.00** | **1.00** |
| ground-truth accuracy: distilled edge | **0.775** | **0.786** |
| ground-truth accuracy: chain (ceiling) | 0.775 | 0.786 |
| base-rate control (shuffled-label LoRA) | 0.425 | 0.286 |
| no-edge (frozen base) | 0.00 | 0.00 |
| inference cost | 1 call vs. 2 | 1 call vs. 2 |

Fabrication guard: the junk-chain edge scores **0.15** ground truth vs. base-rate **0.42** —
far below the no-information baseline. The pipeline does not manufacture structure; fabricated
edges are detectable and rejected by exactly the check gates/README.md demanded.

Reading. A transitive pathway through real frozen specialists CAN be thickened: the distilled
direct edge is a perfect functional cache of the 2-hop chain (agreement 1.00 even on unseen
phrasings) at half the inference cost, and the procedure transfers unchanged to a held-out
pair. The honest bound holds exactly: the edge's ground-truth accuracy equals the chain's
ceiling to the third decimal and cannot exceed it. **Thickening is compression of existing
composite structure, not creation of new capability** (promoted as a bound, Section 5.6).
The guard result is as important as the pass.

Record of corrections (none are post-hoc threshold changes; full detail in GATE3_RESULTS.md):

1. A first run was INVALID (a finite-loss/non-finite-gradient NaN bug silently poisoned the
   distilled adapters); that run's results were discarded and the bug fixed separately.
2. The as-preregistered design was FAIL-BY-CONSTRUCTION (preserved in
   gate3_results_prereg_original.json): it held out PERSONS, but in a random relational world
   the composite on an unseen key is information-theoretically unpredictable, and the run
   showed exactly that signature (agreement 0.00; base-rate control 0.42 > direct 0.25).
   gates/README.md's own G3 wording makes the held-out unit a transitively-connected MODEL
   PAIR; the amended design implements that, keeps all threshold magnitudes unchanged, and
   upgrades the guard's baseline from the vacuous frozen base to the base-rate control.
   Because this is a pre-registration amendment rather than a clean pre-registered pass, the
   verdict is reported as PASS (amended design), never as an unqualified PASS.

Note on scope: the gap-filling arm of gates/README.md's G3 spec ("does settling across the two
produce it") was mooted upstream by G1 — settling itself failed — so what G3 validates is the
distillation/thickening mechanism, not settling-mediated emergence. The gap-filling-via-
settling claim is therefore neither passed nor failed on real models; it is unreachable until
G1's precondition (Section 5.5) is met.

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

Gap-filling operates strictly inside the convex hull of existing structure:
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
empirical valuation and not internalized. Post-merge, this guard has an empirical
demonstration: G3's junk-chain edge scored 0.15 against a 0.42 base-rate — fabricated
structure is not merely forbidden, it is detectable (Section 4.3).

### 5.5 Settling requires calibrated ignorance (empirical bound, from G1)

Settling is only well-founded over models that know what they do not know. The reconciliation
dynamics assumes each participant contributes on the dimensions it genuinely knows and
abstains elsewhere; real small specialists exhibit no such mask (correct-key vs.
wrong-type-key confidence gaps of 0.03–0.07 nats, Section 4.1). Absent per-fact calibration,
settling is not merely useless but actively harmful — **a hallucination amplifier**: each
uncalibrated contribution seeds further confabulation on the next round, and the
positive-feedback cascade drove fact-precision from 0.547 (one step) to 0.018 (five steps).
Any future settling construction must place a calibration/abstention verifier (>=0.8
fact-precision on admitted facts) *upstream* of the coupling loop as a precondition, not as a
post-hoc filter. Until that precondition is met on real models, the settling layer of
Section 1.4 is a definition without an instance.

### 5.6 Thickening is compression, bounded by the chain ceiling (empirical bound, from G3)

A distilled direct edge is exactly a functional cache of the composite path it replaces: it
matches the chain's ground-truth accuracy to the third decimal (0.775/0.775 and 0.786/0.786
on the held-out pair) and can never exceed it. Thickening buys inference cost (one call
instead of two) and transfer of the *procedure* to new transitively-connected pairs — it
never buys new capability. Anything the composite path gets wrong, the thickened edge gets
wrong; anything invisible to the intermediate model stays invisible (the sandbox measured
this at rel-error 0.99 on B-invisible dimensions). "Self-densification" therefore means the
mesh compresses its existing composite structure toward direct connectivity; it does not mean
the mesh grows knowledge.

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
that floats. The present paper was structured so that it *could not* do that — Sections 3 and
4 stayed empty until the gates filled them — and the discipline paid out on schedule: the
program's own core claim (G1) redded, exactly as three of its ancestors did, and this
document is now what the protocol said it would be in that case — a precise construct
definition paired with a characterized headline negative (Section 4.1) and two scoped,
gate-validated survivals (Sections 4.2, 4.3). The G1 failure also rhymes with its lineage:
like the stability gate, the load-bearing ingredient turned out to live somewhere other than
the mechanism (there, frozen corpus geometry; here, an assumed calibration mask the sandbox
provided for free and real models do not).

---

*Editorial notes (remove in final): merge executed 2026-07-08 — Section 4 filled from
GATE1/GATE2/GATE3_RESULTS.md with numbers unedited, Section 3 updated to the promoted/refuted
ledger, bounds 5.5–5.6 added from the gate mechanisms. Still open before a final version:
Section 2.1(3)'s direct graph-MoE paper verification; citations for relative representations,
DEQ, GrTT/QTT, MoFE, SAGMM, and the graph-MoE line; and a decision on whether the title's
"memory-kernel settling" should be revised given that settling failed its gate while the
kernel result survives only in scoped form.*
