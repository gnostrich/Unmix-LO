# VIRTUALMESH — spec-level formal skeleton (Track B)

**Format note.** Rzk tooling is not available in this environment, so this file is the `.md`
fallback permitted by `spec/README.md`: a typed specification written in Rzk-style syntax inside
code blocks, precise enough to transcribe into a `.rzk` module verbatim. Ambient theory is the
Riehl–Shulman simplicial (directed) type theory as implemented in the Rzk proof assistant
(`#lang rzk-1`): a directed interval `2` with endpoints `0₂`, `1₂`, tope logic, shapes
`Δ¹`, `Δ²`, and extension types `(t : Δ¹) → A [φ ↦ a]`. We freely use the standard sHoTT
library notions `hom`, `hom2`, `is-segal`, `is-rezk`. Statements that a real Rzk file would take
as module parameters or `#postulate`s are marked as such; nothing here requires a construction
that Rzk cannot express at spec level.

**STATUS (2026-07-08 — POST-MERGE; all three real gates have returned).**
Verdicts, per `gates/GATE1_RESULTS.md`, `gates/GATE2_RESULTS.md`, `gates/GATE3_RESULTS.md`:

- **G1: FAIL.** Settling amplifies hallucination on real specialists (fact-precision **0.018**
  vs required 0.8; settling vs pooling **−100%** relative; pooling itself lost to best-single).
  Missing precondition: *calibrated ignorance*. All G1-gated laws are **REFUTED AT SCALE** and
  live in §II.R (the refuted/unpromoted register) — recorded, not silently deleted.
- **G2: PASS (scoped).** Exact short-memory MZ closure at L=2; kernel eff-rank flat (120) across
  N=4..10 while tracking routed width K. Promoted **only** in the scoped form of that file's
  "Consequence for the merge" (linear ridge channels over real representation geometry). NOT
  promoted: atomicity-as-compression (rank << cap), residual-as-repair-trigger, nonlinear dynamics.
- **G3: PASS (amended prereg).** The distilled direct edge is a *perfect functional cache* of the
  2-hop chain (agreement 1.00 including unseen paraphrases), inherits the chain's ground-truth
  ceiling exactly (0.775/0.786 — never exceeds it), and the fabrication guard rejects junk chains
  (0.15, below the 0.42 base rate). The prereg amendment (held-out unit = transitively-connected
  MODEL PAIR, per `gates/README.md`; guard baseline upgraded to the base-rate control) is on the
  record in that file's "Record of corrections".

Consequences for this document:

- **Part I** is result-independent and unchanged by the merge.
- **Part II** laws now carry one of three markers: `[PROMOTED — Gn PASS (scope)]` with a
  `#postulate`d inhabitant citing the gate's RESULTS.md; `[CONJECTURAL — UNMEASURED]` (stated,
  never asserted); or `[REFUTED AT SCALE]` / `[UNPROMOTED]`, collected in §II.R.
- Per the paper-track flag: **grade additivity (Law G1-A) was never separately measured** — G1's
  text-space run failed upstream (hallucination cascade) before tolerance-composition could be
  posed. It therefore stays `[CONJECTURAL — UNMEASURED]`, overriding its pre-written
  "drop on G1 fail" note: it is neither evidence-backed nor refuted; it is unmeasured.

---

# PART I — RESULT-INDEPENDENT CORE

This part defines *what the construct is*, independently of whether it works. It makes the novelty
legible against routers, merging, and graph-MoE: the mesh is a **directed** graph of **lossy
channels** between **frozen models**, whose composites are required to agree only **up to a graded
tolerance** — and the theory never pretends the channels are invertible.

## I.0 Ambient conventions

```rzk
#lang rzk-1

-- Standard sHoTT definitions assumed (Riehl–Shulman 2017; rzk sHoTT library):
--
--   hom (A : U) (x y : A) : U
--     := (t : Δ¹) → A [t ≡ 0₂ ↦ x, t ≡ 1₂ ↦ y]
--
--   hom2 (A : U) (x y z : A) (f : hom A x y) (g : hom A y z) (h : hom A x z) : U
--     := (composite-witness triangles over Δ²)
--
--   is-segal A   : composites exist and are unique up to contractible choice
--   is-rezk  A   : is-segal A, and identifications coincide with isomorphisms
```

Directedness is the load-bearing choice: `hom A x y` and `hom A y x` are unrelated types, and no
rule of the theory manufactures inverses. This is exactly the geometry of lossy channels, and it is
*not* expressible in symmetric (cubical) identity types without hand-encoding a category.

## I.1 Objects — models as frame-aligned representation spaces

The mesh lives in a single Rezk type `𝓜` whose elements are the *frame-aligned representation
spaces* of the participating models (a frozen model is identified with its representation space
after alignment into the shared frame). The shared frame is itself an object, and frame-alignment
is the requirement that every model has a distinguished readout channel into it.

```rzk
#postulate 𝓜 : U                      -- the type of frame-aligned model spaces
#postulate is-rezk-𝓜 : is-rezk 𝓜      -- directed category structure, univalent on the iso fragment

#postulate Frame : 𝓜                   -- the shared frame (anchor/probe coordinate space)

-- frame-alignment: every model has a canonical readout into the shared frame
#postulate anchor : (M : 𝓜) → hom 𝓜 M Frame
```

Intended semantics (not formalized; recorded for the paper): an element of `𝓜` is a pair
(representation space of a frozen model, alignment map fitted on a few hundred shared probe
inputs — relative representations over an embedding anchor). `Frame` is the probe-coordinate
space; `anchor M` is the fitted alignment map. The *frame cost* (anchors + alignment) is the
maintained artifact that G1 was required to report; it is a semantic quantity, not a type.

## I.2 Morphisms — channels as directed, non-invertible homs

```rzk
-- A channel from M to N is a directed hom in the mesh type.
#def Channel (M N : 𝓜) : U
  := hom 𝓜 M N
```

**Non-invertibility is the default, not a side condition.** For `f : Channel M N` and
`g : Channel N M`, the composite `g ∘ f : Channel M M` is an ordinary endo-channel; *no rule*
identifies it with `id M`. The type `Channel M N` may be inhabited while `Channel N M` is empty.
In symbols, the theory is one in which

```rzk
-- NOT a law of the theory (recorded to make the design intent explicit):
--   (f : Channel M N) (g : Channel N M) → (comp g f = id-hom 𝓜 M)     -- ✗ no such rule
```

Intended semantics: a channel is a lossy translation of one model's states into another's
(a fitted read-write probe / stochastic map). A roundtrip is a contraction onto the shared
subspace of the two models — informative, but not the identity. This is the design point that
rules out modeling the mesh in a groupoid: `A → B → A ≠ id` is directed-native.

## I.3 Composition — paths through the mesh

The mesh proper is `𝓜` together with a chosen generating graph of *realized* channels (the edges
that have actually been fitted; not every hom is realized).

```rzk
-- A mesh structure on 𝓜: which channels are realized as fitted edges.
#def MeshStruct : U
  := (M : 𝓜) → (N : 𝓜) → (Channel M N → Prop)     -- "is a realized edge"

#variable E : MeshStruct

-- Paths: finite composable chains of realized edges (spec-level; a real .rzk file
-- would give the standard inductive presentation, here written informally).
#def MeshPath (M N : 𝓜) : U
  :=  chains  M = M₀ –f₁→ M₁ –f₂→ … –fₖ→ Mₖ = N   with each fᵢ realized per E
      (k = 0 gives the identity path; k = 1 gives a single edge)

-- Composition: Segal structure gives each chain a composite, unique up to
-- contractible choice; associativity and unitality hold automatically.
#def compose (M N : 𝓜) (p : MeshPath M N) : Channel M N
  := iterated Segal composite of the chain p
```

Because `𝓜` is Segal, there is no coherence bureaucracy to discharge at spec level: composites
exist, and all reassociations of a chain yield identical composites up to contractible ambiguity.
"Routing through short high-overlap paths" is, formally, just: selecting a `MeshPath` and taking
its composite.

## I.4 The invertible fragment — equivalences as a derived special case

Lossless-reversible channels are not an extra primitive; they are carved out of `Channel`.

```rzk
#def is-iso (M N : 𝓜) (f : Channel M N) : U
  := Σ (g : Channel N M) ,
       product (comp g f = id-hom 𝓜 M) (comp f g = id-hom 𝓜 N)

#def Iso (M N : 𝓜) : U
  := Σ (f : Channel M N) , is-iso M N f
```

By Rezk-completeness of `𝓜` (`is-rezk-𝓜`), identifications of models coincide with isomorphisms:

```rzk
--   (M = N)  ≃  Iso M N        -- directed univalence on the invertible core
```

So the symmetric world (model identity, lossless translation) is recovered *inside* the directed
one as a derived special case — the correct containment. Routers and merging live entirely in
degenerate fragments of this picture: a router never composes (paths of length ≤ 1, outputs
pooled outside the mesh); merging quotients `𝓜` to a point. The mesh is the non-degenerate case.

## I.5 Path-coherence — the SHAPE of path isogeny (grade left as a parameter)

Two routes between the same pair of models should agree *up to a controlled tolerance*. Part I
fixes only the **shape** of this property: what kind of thing the tolerance is (an element of an
abstract preordered parameter type) and what kind of statement agreement is. Nothing in this
section asserts that any mesh *is* path-coherent. (Post-merge note: the only measured instance of
this predicate is the G3 two-hop configuration promoted in §II.2; the general property remains a
predicate, not a theorem.)

```rzk
-- Abstract tolerance parameter (Part I commits to nothing beyond a preorder):
#postulate Tol : U
#postulate leq-Tol : Tol → Tol → Prop            -- δ ≤ δ' : "δ is a tighter tolerance"

-- Graded agreement of parallel channels: a Tol-indexed, proposition-valued,
-- reflexive relation, monotone in the tolerance.
#postulate Agree
  : (δ : Tol) → (M N : 𝓜) → Channel M N → Channel M N → Prop

#postulate Agree-refl                              -- every channel agrees with itself
  : (δ : Tol) (M N : 𝓜) (f : Channel M N) → Agree δ M N f f

#postulate Agree-mono                              -- loosening the tolerance preserves agreement
  : (δ δ' : Tol) → leq-Tol δ δ' →
    (M N : 𝓜) (f g : Channel M N) → Agree δ M N f g → Agree δ' M N f g

-- THE SHAPE of path isogeny: a mesh E is path-isogenous at grade δ iff any two
-- mesh-paths between the same endpoints have δ-agreeing composites.
#def is-path-isogenous (E : MeshStruct) (δ : Tol) : U
  := (M N : 𝓜) → (p q : MeshPath M N) →
       Agree δ M N (compose M N p) (compose M N q)
```

`is-path-isogenous E δ` is a *predicate definition*, not a theorem: Part I introduces the type;
no `δ` is claimed to inhabit it for any real mesh in general. "Path isogeny" names the intended
reading: structure-preserving agreement of routes up to controlled collapse — diagrams commute up
to grade, with the grade a first-class parameter rather than a hand-wave.

*End of the result-independent core. Everything above may be transcribed and published regardless
of gate outcomes.*

---

# PART II — GATED LAYERS (post-merge, 2026-07-08)

Part II now carries three classes of material, each explicitly marked:

- `[PROMOTED — Gn PASS (scope)]` — the statement has a `#postulate`d inhabitant, with an inline
  citation to the gate's RESULTS.md and its measured numbers, valid **only within the stated
  scope**. Scope predicates are part of the statement, not footnotes.
- `[CONJECTURAL — UNMEASURED]` — stated as a type, never asserted; no gate has measured it.
- `[REFUTED AT SCALE]` / `[UNPROMOTED]` — collected in §II.R with a pointer to the characterized
  negative. Refuted statements are retained (as types) for the record; they are never inhabited.

Merge protocol executed: gate passed → statement postulated with citation; gate failed → law moved
to §II.R with its failure characterization; scoped pass → only the scoped restatement promoted,
with the unscoped stronger reading left in §II.R as unpromoted.

## II.1 Graded / tolerance layer — `[CONJECTURAL — UNMEASURED]`

`-- [CONJECTURAL — UNMEASURED after G1 FAIL: G1's run collapsed upstream (hallucination cascade, gates/GATE1_RESULTS.md) BEFORE tolerance-composition was ever measured. Per the paper-track flag, this layer is retained as stated-not-asserted: it is unmeasured, not refuted. Its pre-written "drop on G1 fail" note is superseded by this merge decision. No law in this subsection has an inhabitant.]`

This layer would refine Part I's abstract `Tol` into a **partially ordered semiring** of grades
and turn the mesh into a **Lawvere-metric-enriched** directed category: every channel carries a
grade (its lossiness/tolerance), and composition *adds* grades.

**This is an instance, not an invention.** The grading discipline instantiated here is exactly the
semiring-parameterized grading of existing graded modal dependent type theories, and the spec
inherits its rules from them rather than re-deriving them:

- **GrTT** — B. Moon, H. Eades III, D. Orchard, *Graded Modal Dependent Type Theory*, ESOP 2021:
  general semiring-graded modalities over dependent types; our grade algebra is its semiring
  parameter, instantiated at `([0,∞], ≥, 0, +, ·)`.
- **QTT** — R. Atkey, *Syntax and Semantics of Quantitative Type Theory*, LICS 2018 (after
  McBride's *I Got Plenty o' Nuttin'*): the resource-annotated judgment discipline we mimic when
  annotating channels rather than variables.
- **Graded-modal Agda** — A. Abel, N. A. Danielsson, O. Eriksson et al., *A Graded Modal Dependent
  Type Theory with a Universe and Erasure, Formalized* (ICFP 2023, Agda formalization): the
  mechanized reference for the structural laws (subsumption, grade monotonicity) that our graded
  homs must satisfy; the de-risking shallow embedding (Appendix A) targets this development.
- The additive-composition/enrichment reading is **Lawvere metric enrichment** — F. W. Lawvere,
  *Metric spaces, generalized logic, and closed categories*, 1973: hom-objects valued in
  `([0,∞], ≥, +, 0)`.

```rzk
-- Grade algebra: a partially ordered semiring, postulated abstractly;
-- intended instance: ([0,∞], order ≥-as-refinement, 0, +, ·)  — Lawvere.
#postulate G : U
#postulate leq-G  : G → G → Prop
#postulate zero-G : G                       -- grade of identities / lossless channels
#postulate add-G  : G → G → G               -- sequential composition of losses
#postulate mul-G  : G → G → G               -- (reserved: parallel/scaling; unused until needed)
-- (po-semiring axioms: assoc, comm of add-G, units, monotonicity — as in GrTT §2, elided)

-- Part I's Tol is instantiated: Tol := G, leq-Tol := leq-G.

-- Graded channels: channels indexed by an upper bound on their lossiness.
#postulate chan : (g : G) → (M N : 𝓜) → U

#postulate chan-sub                        -- subsumption along the grade order
  : (g h : G) → leq-G g h → (M N : 𝓜) → chan g M N → chan h M N

#postulate chan-forget                     -- every graded channel is a channel
  : (g : G) (M N : 𝓜) → chan g M N → Channel M N

#postulate id-chan : (M : 𝓜) → chan zero-G M M          -- identities are lossless
```

### Law G1-A (graded composition / triangle inequality)
`-- [CONJECTURAL — UNMEASURED: never separately measured. G1's real run failed upstream of any tolerance-composition measurement (the only empirical support remains the SANDBOX composite ≤ sum-of-hops result, which promotes nothing). Stated, not asserted; a future gate must measure additivity directly before this may be postulated.]`

```rzk
#def Statement-G1-A : U
  := (g h : G) → (M N P : 𝓜) →
     chan g M N → chan h N P → chan (add-G g h) M P
-- i.e. grade(q ∘ p) ≤ grade(p) + grade(q): composing along a path degrades
-- at worst additively.
-- NO inhabitant is postulated.
```

### Law G1-B (reconciliation refines pooling — the former core claim)
`-- [REFUTED AT SCALE — see §II.R.1. Moved out of the live spec per the merge; the typed statement is preserved in the register, uninhabited, with the characterized negative from gates/GATE1_RESULTS.md.]`

## II.2 Densification layer — `[PROMOTED — G3 PASS (amended prereg)]`

Pathway thickening on real specialists passed, in an amended design whose held-out unit is a
transitively-connected **model pair** (as `gates/README.md` specified; the original prereg's
held-out-*persons* design was fail-by-construction on a random relational world — see
`gates/GATE3_RESULTS.md`, "Record of corrections", which also records the invalidated first run).
The promoted statements are written in Part I vocabulary (`Channel`, `compose`, `Agree`) and do
**not** use the unvalidated additive grade algebra of §II.1.

```rzk
-- Scope predicates for the tested regime (semantic; part of every promoted statement):
#postulate transitively-connected : (E : MeshStruct) (A B C : 𝓜) → Prop
   -- A→B and B→C realized in E; no direct A→C edge; real shared structure via B
#postulate distilled-edge : (E : MeshStruct) (A C : 𝓜) → Channel A C → Prop
   -- fitted by distilling chain pseudo-labels into a direct low-rank adapter edge
#postulate δ-G3 : Tol      -- the pre-registered agreement tolerance (threshold 0.90)
```

### Law G3-D (thickening: the distilled direct edge is a functional cache of the chain)
`-- [PROMOTED — G3 PASS 2026-07-08, gates/GATE3_RESULTS.md: agreement with chain = 1.00 on BOTH the design pair and the held-out gap pair, INCLUDING unseen paraphrases (threshold 0.90); inference cost 1 call vs 2. Scope: 2-hop chains of specialists each at 1.00 single-hop accuracy, LoRA-scale distillation on a shared frozen base.]`

```rzk
#def Statement-G3-D : U
  := (E : MeshStruct) (A B C : 𝓜) → transitively-connected E A B C →
     (p : MeshPath A C via B) →                       -- the 2-hop chain
     Σ (d : Channel A C) ,
       product (distilled-edge E A C d)
               (Agree δ-G3 A C d (compose A C p))     -- functional-cache agreement

#postulate law-G3-D : Statement-G3-D
-- PROMOTED per gates/GATE3_RESULTS.md: Q1 agreement 1.00 (trained template) and 1.00
-- (unseen paraphrase) on pair 1 (person→city→company) AND pair 2, the held-out gap
-- (city→company→product), same procedure applied unchanged.
```

### Law G3-C (ceiling: thickening is compression, not creation)
`-- [PROMOTED — G3 PASS 2026-07-08 (this is the honest BOUND that ships with G3-D and must never be dropped from paper or spec): the distilled edge inherits the chain's ground-truth ceiling EXACTLY and cannot exceed it — 0.775 = 0.775 (pair 1), 0.786 = 0.786 (held-out pair). Thickening compresses existing composite structure; it does not create capability.]`

```rzk
#def Statement-G3-C : U
  := (E : MeshStruct) (A B C : 𝓜) (p : MeshPath A C via B)
     (d : Channel A C) → distilled-edge E A C d → Agree δ-G3 A C d (compose A C p) →
       gt-accuracy d = gt-accuracy (compose A C p)
-- gt-accuracy: ground-truth valuation (EXTERNAL — see §II.4; never internalized)

#postulate law-G3-C : Statement-G3-C
-- PROMOTED per gates/GATE3_RESULTS.md (edge = chain ceiling to the third decimal;
-- both far above base-rate control 0.425/0.286 and no-edge 0.00).
```

### Law G3-G (fabrication guard: junk chains are detectable and rejected)
`-- [PROMOTED — G3 PASS 2026-07-08: a chain across models sharing NO real path distills to an edge scoring 0.15 ground truth, far BELOW the 0.42 base-rate control — fabricated edges are detectable by exactly the check gates/README.md demanded. Guard baseline was upgraded (on the record) from the vacuous frozen base to the base-rate control.]`

```rzk
#def Statement-G3-G : U
  := (E : MeshStruct) (A B C : 𝓜) → NOT (transitively-connected E A B C) →
     (d : Channel A C) → distilled-edge E A C d →
       lt (gt-accuracy d) (base-rate A C)
-- the mesh does NOT manufacture structure where models share nothing;
-- below-base-rate distillation is the rejection signature.

#postulate law-G3-G : Statement-G3-G
-- PROMOTED per gates/GATE3_RESULTS.md (guard: 0.15 vs base-rate 0.42).
```

## II.3 MZ-kernel layer — `[PROMOTED — G2 PASS (scoped)]`

`-- [Scope of every promotion in this subsection, per gates/GATE2_RESULTS.md "Consequence for the merge": LINEAR instantiation — spectrally-capped ridge channels over frame-aligned REAL representation geometry ((model, layer) hidden-state spaces, 240 shared probes, PCA d=40/node), damped linear settling; federation sizes N = 4..10 (15 real nodes available). Nonlinear settling dynamics remain UNTESTED (G1's nonlinear text-space protocol failed upstream). Statements are guarded by LinearRegime.]`

The settling dynamics carries a memory: the influence of untracked models on a routed subset is
summarized by a **Mori–Zwanzig memory kernel** (Zwanzig 1961; Mori 1965 — projection-operator
closure of a partially observed dynamical system). What G2 validated is the pre-registered pair —
**short-memory closability** and **N-independence of the closure's complexity** — which is the
scale-invariance the "cost scales with routing width, not federation size" claim needs, *in the
linear regime*.

```rzk
-- Spec-level dynamical interface (semantic details live in gates/G2, not here):
#postulate State   : 𝓜 → U                        -- per-model settled-state space
#postulate Routing : U                             -- a routed subset of the federation
#postulate width   : Routing → ℕ                   -- K: number of routed models
#postulate size    : MeshStruct → ℕ                -- N: federation size
#postulate d-node  : ℕ                             -- per-node embedding dimension (tested: 40)

#postulate LinearRegime : MeshStruct → Prop
   -- scope predicate: linear (ridge) channels over real model geometry, damped
   -- linear settling — the regime actually tested in gates/GATE2_RESULTS.md

#postulate Kernel  : MeshStruct → Routing → (L : ℕ) → U
   -- the fitted memory kernel of history-length L closing the routed subset's dynamics
#postulate effrank : (E : MeshStruct) (r : Routing) (L : ℕ) → Kernel E r L → ℕ
#postulate closure-err : (E : MeshStruct) (r : Routing) (L : ℕ) → Kernel E r L → G
   -- discrepancy between kernel-predicted and full-federation settled states
```

### Law G2-A (short-memory closure adequacy — scoped)
`-- [PROMOTED — G2 PASS 2026-07-08, gates/GATE2_RESULTS.md: closure error 0.084 at L=1, 0.000 (EXACT) at L=2, vs pre-committed criterion "some L ≤ 8 with error < 0.15"; memory necessary (Markovian closure stuck at 0.084). Scope: LinearRegime only.]`

```rzk
#def Statement-G2-A-scoped : U
  := (E : MeshStruct) → LinearRegime E → (r : Routing) →
     Σ (L₀ : ℕ) , product (leq-ℕ L₀ 2)
       (Σ (K : Kernel E r L₀) , leq-G (closure-err E r L₀ K) ε-closure)
-- ε-closure: the pre-committed tolerance (0.15); measured value 0.000 at L₀ = 2.

#postulate law-G2-A : Statement-G2-A-scoped
-- PROMOTED per gates/GATE2_RESULTS.md (exact L=2 closure on held-out probes, T=40).
```

### Law G2-B (N-independence of closure complexity — scoped; NOT the compression reading)
`-- [PROMOTED — G2 PASS 2026-07-08, gates/GATE2_RESULTS.md: eff-rank 120/120/120/120 (flat ±2) for N=4/6/8/10 with routed subset fixed; eff-rank tracks K (80/160/240 for K=2/4/6). Scope: LinearRegime. WHAT IS NOT PROMOTED: eff-rank sat AT the dimensional cap K·d-node in every configuration, so the stronger atomicity-as-COMPRESSION reading (rank << K·d) is UNDEMONSTRATED and stays in §II.R.6.]`

```rzk
#def Statement-G2-B-scoped : U
  := (E E' : MeshStruct) → LinearRegime E → LinearRegime E' →
     extends E' E →                                  -- E' adds models; size E' > size E
     (r : Routing) (L : ℕ) (K : Kernel E r L) (K' : Kernel E' r L) →
       product (effrank E' r L K' = effrank E r L K)          -- flat under growing N
               (leq-ℕ (effrank E r L K) (mul-ℕ (width r) d-node))
-- The second conjunct is the CAP bound — honest but partly dimensional necessity;
-- the promoted content is the first conjunct: closure complexity is a function of
-- the routed subset, invariant under federation growth (the scale-invariance claim,
-- in its linear-regime form).

#postulate law-G2-B : Statement-G2-B-scoped
-- PROMOTED per gates/GATE2_RESULTS.md (rank flat across N=4..10; grows with K).
```

### Law G2-C (residual as repair signal)
`-- [UNPROMOTED — see §II.R.5: did not replicate on real geometry (correlation −0.07 vs sandbox +0.29). Statement preserved in the register, uninhabited.]`

## II.4 The certification modality `cert` — `[PARTIALLY PROMOTED; grade index retired]`

The genuinely novel, boundary-touching seam. `cert T` is the type of *certificates* that `T` is
stable-to-tolerance — certified not by a proof but by an **external empirical oracle** (the gate
protocol: pre-registered thresholds, held-out ground truth) run against the frozen models. The
graded-modal literature (GrTT/QTT/Abel et al., §II.1) tracks *internal* resources; a modality
whose introduction is grounded in an *external valuation* is not in that literature.

**Merge consequence for the modality's shape.** The original design indexed the modality by a
tolerance grade, `cert ε T`, with a monotonicity rule gated on G1 (the grade order is meaningful
only if G1 calibrates it). G1 failed, so — exactly as the pre-written failure note prescribed —
the modality is **weakened to the unindexed form** `cert T`: a certificate holds *at the tolerance
it was measured at* (recorded in the citing comment, semantically), with no internal loosening
rule. The graded form and its rules are preserved, uninhabited, in §II.R.8–10.

**Intended semantics (sketch, not formalized).** Interpret the theory in presheaves over a small
category `W` of *worlds*, where a world is a frozen-model federation state (the given models, with
fitted frame and edges); world morphisms are federation extensions/refittings. `cert T` at world
`w` is inhabited iff the oracle's valuation, run at `w`, returned PASS for `T`.

**Exactly two external ports remain, by design:**

```rzk
-- PORT 1: the given models — the world-indexed family of frozen models.
#postulate W : U                                   -- worlds = frozen federation states
#postulate Models : W → U                          -- the given models at each world

-- PORT 2: the valuation — the empirical oracle. META-LEVEL ONLY.
--   val : (w : W) → (T : U) → {PASS, FAIL}        (with its tolerance recorded externally)
-- val is NOT a term of the theory. It has no formation rule, no type, no name
-- in the term language. It acts only through the axiom schema cert-oracle below.
-- DO NOT internalize the valuation: internalizing it would (a) fake an
-- introduction rule the empirical process does not license, and (b) collapse
-- the novel seam back into ordinary internal grading.
```

```rzk
-- Formation (weakened, unindexed — see merge consequence above):
#postulate cert : U → U
```

### Rule C-INTRO (external introduction only — the oracle axiom schema, NOW INSTANTIATED)
`-- [PROMOTED-BY-INSTANTIATION 2026-07-08: the schema itself is a design rule; it gains content only through actually-passed gate runs. Two runs passed; certificates are minted below, each citing its RESULTS.md. G1's run FAILED, so no certificate exists for any settling/reconciliation claim — and none may be minted without a new pre-registered run.]`

```rzk
-- There is NO internal introduction rule: no term of the theory constructs
-- cert T from a proof of T or from anything else internal.
-- Certificates enter ONLY as postulates, one per passed oracle run:

#postulate cert-oracle-G2 : cert Statement-G2-A-scoped
-- minted per gates/GATE2_RESULTS.md (PASS, all pre-registered criteria; measured
-- tolerance: closure error 0.000 at L=2 against pre-committed 0.15; linear regime)

#postulate cert-oracle-G2b : cert Statement-G2-B-scoped
-- minted per gates/GATE2_RESULTS.md (rank flat 120 across N=4..10; linear regime)

#postulate cert-oracle-G3 : cert Statement-G3-D
-- minted per gates/GATE3_RESULTS.md (PASS amended prereg; measured agreement 1.00
-- against pre-registered threshold 0.90, incl. unseen paraphrases and held-out pair;
-- ceiling and guard clauses law-G3-C / law-G3-G certified by the same run)

-- Guard (from G3, now VALIDATED as a guard): a run whose "stability" is
-- self-consistency without held-out ground truth does NOT license an instantiation.
-- G3's fabrication-guard result (junk-chain edge at 0.15 vs base-rate 0.42) is the
-- empirical demonstration that this check has teeth. (Anti-steganography clause.)
```

### Rule C-NOELIM (no counit: certification is not proof)
`-- [Design invariant, not gated — it can never be promoted INTO an eliminator by any gate outcome. Empirical evidence never yields an inhabitant of T. G1's failure is the cautionary instance: a "settled" fixed point looked stable and was 98.2% false.]`

```rzk
-- NOT a rule of the theory:
--   (T : U) → cert T → T                     -- ✗ deliberately absent
```

### Rule C-FUNC (certificates survive federation growth — scoped)
`-- [PROMOTED — gated on G2, which PASSED scoped: the warrant is exactly law-G2-B's N-invariance (the routed subset's closure, hence the measured stability, is unchanged by growing N from 4 to 10). Scope: LinearRegime worlds, routed subset fixed, N within the tested range; outside that regime certificates remain world-pinned.]`

```rzk
#def Statement-C-FUNC-scoped : U
  := (w w' : W) → extends w' w →              -- w' adds models to w
     routed-subset-fixed w w' → LinearRegime-at w' →
     (T : U) → cert-at w T → cert-at w' T
-- (cert-at : W → U → U is the world-indexed form of cert in the intended
--  presheaf semantics; cert T ≡ cert-at at the ambient world)

#postulate law-C-FUNC : Statement-C-FUNC-scoped
-- PROMOTED per gates/GATE2_RESULTS.md (eff-rank and closure invariant, N=4..10).
```

### Rules C-MONO, C-COMP, C-PAIR
`-- [UNPROMOTED — all were gated on G1, which FAILED. Per their pre-written failure notes: C-MONO's failure weakens cert to the unindexed form used above; C-COMP is dropped (certificates do NOT propagate along channels — each holds only at the configuration where it was measured); C-PAIR is left unstated. Statements preserved in §II.R.8–10.]`

## II.R — REFUTED AT SCALE / UNPROMOTED register

Nothing in this section is asserted. Statements are preserved as types for the record; none has an
inhabitant. Each entry carries its one-line disposition.

### II.R.1 Law G1-B (settling refines pooling) — REFUTED AT SCALE
`-- [REFUTED AT SCALE — gates/GATE1_RESULTS.md: settling AMPLIFIES hallucination (fact-precision 0.018 vs required 0.8; 2,171 facts admitted, 98.2% false; settling vs pooling −100% relative). Characterized missing precondition: CALIBRATED IGNORANCE — real 0.5B specialists confabulate at near-trained confidence on wrong-type keys (logprob gaps 0.03–0.07 nats); the sandbox's 1.5x win silently assumed a knowledge mask. Any retry must gate contributions on a verifier at ≥0.8 fact-precision BEFORE settling is re-tested — a different, pre-registerable experiment. Nothing here licenses building the settling layer.]`

```rzk
-- REFUTED — retained for the record; NO inhabitant, and none may be postulated.
#postulate Query  : U
#postulate Answer : U
#postulate err : Answer → Query → G          -- EXTERNAL valuation (see §II.4)
#postulate pool     : MeshStruct → Query → Answer
#postulate one-step : MeshStruct → Query → Answer
#postulate settle   : MeshStruct → Query → Answer

#def Statement-G1-B : U                       -- REFUTED AT SCALE
  := (q : SplitKnowledgeQuery) →
       leq-G (err (settle E q) q) (scale 0.9 (err (pool E q) q))
```

### II.R.2 Law G1-B' (one-step reconciliation refines pooling) — REFUTED AT SCALE
`-- [REFUTED AT SCALE — same run: one-step accuracy 0.00 vs pooling 0.05; the pre-planned "weakened claim" fallback is ALSO dead at this scale.]`

```rzk
#def Statement-G1-B' : U                      -- REFUTED AT SCALE
  := (q : SplitKnowledgeQuery) →
       leq-G (err (one-step E q) q) (scale 0.9 (err (pool E q) q))
```

### II.R.3 Baseline ordering note — REFUTED AT SCALE
`-- [gates/GATE1_RESULTS.md finding 3: POOLING itself lost to the best single model (0.05 vs 0.15) — three of four specialists cannot answer any multi-hop query and their confident wrong votes drown the sometimes-right one. At this scale both aggregation baselines lose to argmax-model. Any future claim must beat best-single, not just pooling.]`

### II.R.4 Scope correction to the pre-written failure note
`-- [The original G1-B failure note said "all of Part II removed" on outright G1 failure. That was wrong as written: G2 and G3 ran on protocols that do not depend on text-space settling (linear representation-space dynamics; chain distillation) and passed on their own pre-registered terms. The merge therefore removes ONLY the settling/reconciliation-as-inference layer; the kernel and densification layers stand, scoped as promoted above.]`

### II.R.5 Law G2-C (residual as repair signal) — NOT REPLICATED
`-- [UNPROMOTED — gates/GATE2_RESULTS.md: residual–difficulty correlation −0.07 on real geometry vs +0.29 in sandbox; reported, not gating; per its pre-written note it is dropped from the live spec, recorded here.]`

```rzk
#def Statement-G2-C : U                       -- NOT REPLICATED; no inhabitant
  := monotone-correlation (closure-residual) (query-difficulty)
```

### II.R.6 Atomicity-as-compression (rank << cap) — UNDEMONSTRATED
`-- [UNPROMOTED — gates/GATE2_RESULTS.md scope limit 1: eff-rank equaled the output-dimension cap K·d-node in EVERY configuration, so "rank grows with K" is partly dimensional necessity. The compressed-index reading below is NOT demonstrated and must not be claimed in the paper.]`

```rzk
#def Statement-G2-B-compression : U           -- UNDEMONSTRATED; no inhabitant
  := (E : MeshStruct) (r : Routing) (L : ℕ) (K : Kernel E r L) →
       lt-ℕ (effrank E r L K) (mul-ℕ (width r) d-node)     -- strictly below cap
```

### II.R.7 Nonlinear-dynamics kernel claims — UNTESTED
`-- [UNPROMOTED — gates/GATE2_RESULTS.md scope limit 2: the tested dynamics are linear (ridge channels, linear damping) over real geometry; the fully nonlinear settling of gates/README.md lived in G1's text-space protocol, which failed upstream before any kernel question could be posed. A nonlinear-dynamics G2 remains an open, pre-registerable experiment.]`

### II.R.8 Rule C-MONO (tolerance monotonicity) — UNPROMOTED (weakening applied)
`-- [UNPROMOTED — was gated on G1 (grade order calibration), which FAILED. Per its pre-written note, cert is weakened to the unindexed form in §II.4; the graded statement is preserved here.]`

```rzk
#def Statement-C-MONO : U                     -- UNPROMOTED; no inhabitant
  := (ε ε' : G) → leq-G ε ε' → (T : U) → cert-graded ε T → cert-graded ε' T
-- cert-graded : (ε : G) → U → U  is the retired graded formation.
```

### II.R.9 Rule C-COMP (certificate transport along channels) — DROPPED
`-- [UNPROMOTED — needed Law G1-A's additive law (unmeasured) and G1's calibration (failed). Per its pre-written note: certificates do NOT propagate along channels; each holds only at the object/world where it was measured. Preserved here, uninhabited.]`

```rzk
#def Statement-C-COMP : U                     -- DROPPED; no inhabitant
  := (g : G) (M N : 𝓜) (f : chan g M N) →
     (P : 𝓜 → U) → (ε : G) →
       cert-graded ε (P M) → cert-graded (add-G ε g) (P N)
```

### II.R.10 Rule C-PAIR (combining certificates) — LEFT UNSTATED
`-- [UNPROMOTED — was optional and gated on G1; per its pre-written note it is left unstated rather than weakened. Recorded here as a name only; no statement is carried forward.]`

### Gate → law outcome table (merge of 2026-07-08)

| Law / rule | Gate | Verdict | Disposition |
|---|---|---|---|
| G1-A graded composition (triangle) | G1 | never measured (run failed upstream) | `[CONJECTURAL — UNMEASURED]`, §II.1 (paper-track flag honored) |
| G1-B settling refines pooling | G1 | **FAIL** | REFUTED AT SCALE → §II.R.1 |
| G1-B' one-step refines pooling | G1 | **FAIL** | REFUTED AT SCALE → §II.R.2 |
| G3-D thickening (functional cache) | G3 | **PASS** (amended) | PROMOTED, §II.2 (`law-G3-D`) |
| G3-C ceiling (compression, not creation) | G3 | **PASS** | PROMOTED, §II.2 (`law-G3-C`) — honest bound, ships with G3-D |
| G3-G fabrication guard | G3 | **PASS** | PROMOTED, §II.2 (`law-G3-G`) |
| G2-A short-memory closure | G2(a) | **PASS** (scoped) | PROMOTED, §II.3 (`law-G2-A`); LinearRegime only |
| G2-B N-independence | G2(b) | **PASS** (scoped) | PROMOTED, §II.3 (`law-G2-B`); compression reading NOT promoted (§II.R.6) |
| G2-C residual = repair signal | G2 (bonus) | not replicated (−0.07) | UNPROMOTED → §II.R.5 |
| C-INTRO oracle schema | per run | G2, G3 passed | INSTANTIATED ×3 (`cert-oracle-G2/G2b/G3`); no settling certificate exists |
| C-NOELIM no counit | — (design invariant) | — | retained; G1 is its cautionary instance |
| C-MONO tolerance monotonicity | G1 | **FAIL** | weakening applied (cert unindexed) → §II.R.8 |
| C-COMP transport along channels | G1 (+G2 cost) | **FAIL** (G1) | DROPPED → §II.R.9 |
| C-PAIR combining certificates | G1 (optional) | **FAIL** (G1) | left unstated → §II.R.10 |
| C-FUNC stability under federation growth | G2 | **PASS** (scoped) | PROMOTED, §II.4 (`law-C-FUNC`); LinearRegime, N=4..10 |

---

## Appendix A — Why Rzk and not Agda (rationale note, from spec/README.md; for the paper)

Building this in Agda, the graded machinery already exists (Abel–Danielsson–Eriksson's formalized
graded modal dependent type theory, and the GrTT/QTT lineage behind it), but directedness does
not: lossy non-invertible channels would have to be hand-encoded as a category-with-annotations,
and the contribution dilutes into encoding bookkeeping. Building in Rzk, directedness is native —
`hom` is a primitive shape, non-invertibility is the default, and the Segal/Rezk structure
discharges composition coherence for free — so the contribution stays ONE clean layer: grading
plus an externally-grounded certification modality over directed types. Rzk is the right home
because it isolates exactly the novel seam. Recommended de-risking path: shallow-embed the graded
and certification rules in Agda first (against the Abel et al. development) to check the rules
stand up, then present the construct in Rzk for the clean statement. (Post-merge note: with the
graded layer unmeasured and the modality unindexed, the Agda de-risking target shrinks to the
unindexed cert rules plus the scoped kernel laws — smaller than originally planned.)

## Appendix B — Transcription notes for a future `.rzk` file

- Part I transcribes directly: `#postulate` blocks as written; `MeshPath`/`compose` via the
  standard inductive-chain encoding over the sHoTT library's `comp-is-segal`.
- PROMOTED laws transcribe as `#def Statement-… : U := …` **plus** their
  `#postulate law-… : Statement-…` inhabitants, each carrying its RESULTS.md citation comment
  (add the results-JSON path and commit hash at transcription time: `real/gate2_results.json`,
  `real/gate3_results.json`).
- `[CONJECTURAL — UNMEASURED]` laws transcribe as statements ONLY — no inhabitant.
- §II.R entries transcribe as statements in a clearly-marked `-- REFUTED / UNPROMOTED` module
  section (or a separate `virtualmesh-register.rzk`), never inhabited; keep the disposition
  comments verbatim — the register is part of the scientific record, not dead code.
- `G`'s intended instance `([0,∞], ≥, 0, +)` stays semantic (a note), since real-number arithmetic
  is not the point; all spec-level reasoning uses the abstract po-semiring interface.
- Scope predicates (`LinearRegime`, `transitively-connected`, `distilled-edge`) are honest
  `#postulate`d Props: they mark exactly where the empirical warrant ends. Do not "simplify"
  them away during transcription — an unguarded law would claim more than any gate measured.
