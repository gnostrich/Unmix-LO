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

**STATUS (2026-07-08).** No gate (G1/G2/G3, see `gates/README.md`) has returned a real-model
result. Therefore, per the project discipline ("formalize ONLY claims a gate has passed"):

- **Part I** below is result-independent structure. It asserts no empirical claim; it is safe now.
- **Part II** contains ONLY skeletons and *typed statements* of laws. Every law is labeled
  `[CONJECTURAL — gated on Gn]` and is **stated, not asserted**: in a transcribed `.rzk` file each
  would be a *type* (a `#def` of the statement), never a `#postulate` of an inhabitant, until its
  gate passes. Each law carries a one-line note saying which gate promotes it and what it becomes
  if that gate fails.

Sandbox prototypes (`gates/README_prototypes.md`) passed in the clean/linear case; sandbox passes
do **not** promote anything here. Only the real-model gates do.

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
maintained artifact that G1 is required to report; it is a semantic quantity, not a type.

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
abstract preordered parameter type) and what kind of statement agreement is. Which tolerance
algebra to use, and whether composition interacts with it additively, is Part II material
(gated on G1). Nothing in this section asserts that any mesh *is* path-coherent.

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
no `δ` is claimed to inhabit it for any real mesh. "Path isogeny" names the intended reading:
structure-preserving agreement of routes up to controlled collapse — diagrams commute up to
grade, with the grade a first-class parameter rather than a hand-wave.

*End of the result-independent core. Everything above may be transcribed and published regardless
of gate outcomes.*

---

# PART II — GATED LAYERS

**Everything in Part II is conjectural.** Each law below is a *typed statement* — in a transcribed
`.rzk` file, a `#def` whose body is the statement's type — and is inhabited (`#postulate`d as an
axiom reflecting measured evidence, or proved from such axioms) **only after** the named gate
passes. The one-line comment on each law names its gate and its failure behavior.

Promotion protocol: gate passes → the law's statement is postulated with a citation to the gate's
`RESULTS.md` (commit hash + measured numbers). Gate fails → the law is dropped or weakened exactly
as noted, and this file is edited accordingly at the merge step.

## II.1 Graded / tolerance layer — `[CONJECTURAL — gated on G1]`

This layer refines Part I's abstract `Tol` into a **partially ordered semiring** of grades and
turns the mesh into a **Lawvere-metric-enriched** directed category: every channel carries a
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

The delta over that literature is small and deliberate: those systems grade *variable use inside
the type theory*; here the same algebra grades *directed homs between external models*. Nothing
about the algebra is new — which is the point.

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
`-- [CONJECTURAL — gated on G1: promoted iff G1 passes; empirical warrant is G1's tolerance-composition arm (sandbox: composite ≤ sum-of-hops held 100%, real-model version pending). If G1 FAILS: dropped — grades revert to Part I's bare preordered annotation with NO additive law, and Lawvere enrichment is removed from spec and paper.]`

```rzk
#def Statement-G1-A : U
  := (g h : G) → (M N P : 𝓜) →
     chan g M N → chan h N P → chan (add-G g h) M P
-- i.e. grade(q ∘ p) ≤ grade(p) + grade(q): composing along a path degrades
-- at worst additively. This is what makes "short high-overlap paths" a
-- well-typed optimization objective (minimize summed grade over MeshPath).
```

### Law G1-B (reconciliation refines pooling — the core claim, typed)
`-- [CONJECTURAL — gated on G1: promoted iff SETTLING beats POOLING by ≥10% relative on split-knowledge queries. If G1's ablation shows SETTLING ≈ ONE-STEP: weakened to Statement-G1-B' (one-step confidence-weighted consensus refines pooling; recurrence clause dropped). If G1 FAILS outright (settling ≈ pooling): dropped entirely — the construct is an ensemble with extra steps; Part II.1–II.3 are all removed and only Part I survives as a definitional note.]`

```rzk
-- Query answering: postulated interface (semantic; kept opaque at spec level).
#postulate Query  : U
#postulate Answer : U
#postulate err : Answer → Query → G          -- answer-quality grade w.r.t. ground truth
                                             -- (EXTERNAL valuation — see II.3; never internalized)

#postulate pool     : MeshStruct → Query → Answer   -- one-shot combine (router/ensemble baseline)
#postulate one-step : MeshStruct → Query → Answer   -- confidence-weighted consensus, no iteration
#postulate settle   : MeshStruct → Query → Answer   -- coupled fixed-point settling through Frame

#def Statement-G1-B : U
  := (q : SplitKnowledgeQuery) →
       leq-G (err (settle E q) q) (scale 0.9 (err (pool E q) q))
-- with SplitKnowledgeQuery ⊂ Query the pre-registered split-knowledge set and
-- scale 0.9 reflecting the pre-committed ≥10%-relative threshold.

#def Statement-G1-B' : U            -- the weakened form, if settling ≈ one-step
  := (q : SplitKnowledgeQuery) →
       leq-G (err (one-step E q) q) (scale 0.9 (err (pool E q) q))
```

### Law G3-D (densification: synthesized edges match composites)
`-- [CONJECTURAL — gated on G3: promoted iff the distilled direct A→C edge matches the composite A→B→C to tolerance AND generalizes on a held-out transitively-connected pair (beats no-edge/random-edge). If G3 FAILS: dropped — the "mesh grows toward all-to-all / self-densification" claim is removed; the spec keeps only the static-federation mesh, and MeshStruct E is presented as fixed rather than thickening.]`

```rzk
#def Statement-G3-D : U
  := (A B C : 𝓜) (δ : G) →
     (f : chan g A B) → (h : chan h' B C) →        -- a sparse 2-hop pathway
     Σ (d : chan δ A C) ,                          -- a synthesized ONE-hop edge
       Agree δ A C (chan-forget δ A C d)
                   (chan-forget (add-G g h') A C (graded-comp f h))
-- plus (semantic side condition, per G3's guard): d is fitted by distillation,
-- validated on HELD-OUT ground truth — self-consistency alone does NOT witness this Σ.
```

## II.2 MZ-kernel atomicity — `[CONJECTURAL — gated on G2]`

The settling dynamics carries a memory: the influence of untracked models on a routed subset is
summarized by a **Mori–Zwanzig memory kernel** (Zwanzig 1961; Mori 1965 — projection-operator
closure of a partially observed dynamical system). The scale-invariance claim of the whole
construct — cost scales with **routing width**, not **federation size** — is exactly a rank
property of this kernel object, stated here as a typed property and asserted never.

```rzk
-- Spec-level dynamical interface (semantic details live in gates/G2, not here):
#postulate State   : 𝓜 → U                        -- per-model settled-state space
#postulate Routing : U                             -- a routed subset of the federation
#postulate width   : Routing → ℕ                   -- K: number of routed models
#postulate size    : MeshStruct → ℕ                -- N: federation size

#postulate Kernel  : MeshStruct → Routing → (L : ℕ) → U
   -- the fitted memory kernel of history-length L closing the routed subset's dynamics
#postulate effrank : (E : MeshStruct) (r : Routing) (L : ℕ) → Kernel E r L → ℕ
#postulate closure-err : (E : MeshStruct) (r : Routing) (L : ℕ) → Kernel E r L → G
   -- discrepancy between kernel-predicted and full-federation settled states
```

### Law G2-A (short-memory closure adequacy)
`-- [CONJECTURAL — gated on G2(a): promoted iff a short-memory low-rank closure reproduces full-federation settling within the pre-committed tolerance on real nonlinear models. If G2(a) FAILS (memory doesn't help / no short closure): dropped — settling must be simulated on the full federation; the kernel object is deleted from the spec.]`

```rzk
#def Statement-G2-A : U
  := (E : MeshStruct) (r : Routing) →
     Σ (L₀ : ℕ) , Σ (K : Kernel E r L₀) ,
       leq-G (closure-err E r L₀ K) ε-closure          -- ε-closure: pre-committed tolerance
```

### Law G2-B (atomicity: kernel rank scales with routing width, not federation size)
`-- [CONJECTURAL — gated on G2(b): promoted iff measured effrank tracks K as N is varied by adding models. If G2(b) FAILS (rank grows with N): dropped — the "innumerable models / scale-invariant index" story is removed from spec and paper; weakened to an explicit small-fixed-federation scope note (cost bounded by N, no atomicity).]`

```rzk
#def Statement-G2-B : U
  := Σ (c : ℕ) ,                                       -- a constant independent of size E
     (E E' : MeshStruct) (r : Routing) (L : ℕ) (K : Kernel E r L) (K' : Kernel E' r L) →
       -- for any two federations containing the same routed subset r
       -- (E' extends E by adding models; size E' > size E):
       product (effrank E  r L K  ≤ mul-ℕ c (width r))
               (effrank E' r L K' ≤ mul-ℕ c (width r))
-- "atomicity": the kernel's effective rank is a function of the routing width K,
-- invariant under growing N. This — not the layer-over-frozen-models framing —
-- is the distinctly novel core (with G1), per gates/README.md's prior-art note.
```

### Law G2-C (residual as repair signal) — optional clause
`-- [CONJECTURAL — gated on G2's residual sub-measurement (a bonus clause, not part of PASS/FAIL): promoted iff closure residual correlates with query difficulty. If it fails: silently dropped; no other law depends on it.]`

```rzk
#def Statement-G2-C : U
  := monotone-correlation (closure-residual) (query-difficulty)
-- (stated semantically; a slow-path/repair trigger, not load-bearing structure)
```

## II.3 The certification modality `cert_ε` — `[CONJECTURAL — gated on ALL of G1, G2, G3]`

The genuinely novel, boundary-touching seam. `cert ε T` is the type of *certificates* that `T` is
stable-to-tolerance-`ε` — certified not by a proof but by an **external empirical oracle** (the
stability gate / self-consistency-plus-ground-truth check run against the frozen models). The
graded-modal literature (GrTT/QTT/Abel et al., §II.1) tracks *internal* resources; a modality
whose introduction is grounded in an *external valuation* is not in that literature, and that is
the contribution — so it is also the part most strictly gated.

**Intended semantics (sketch, not formalized).** Interpret the theory in presheaves over a small
category `W` of *worlds*, where a world is a frozen-model federation state (the given models, with
fitted frame and edges); world morphisms are federation extensions/refittings. `cert ε T` at world
`w` is inhabited iff the oracle's valuation, run at `w`, returns PASS at tolerance `ε` for `T`.

**Exactly two external ports remain, by design:**

```rzk
-- PORT 1: the given models — the world-indexed family of frozen models.
#postulate W : U                                   -- worlds = frozen federation states
#postulate Models : W → U                          -- the given models at each world

-- PORT 2: the valuation — the empirical oracle. META-LEVEL ONLY.
--   val : (w : W) → (T : U) → (ε : G) → {PASS, FAIL}
-- val is NOT a term of the theory. It has no formation rule, no type, no name
-- in the term language. It acts only through the axiom schema cert-oracle below.
-- DO NOT internalize the valuation: internalizing it would (a) fake an
-- introduction rule the empirical process does not license, and (b) collapse
-- the novel seam back into ordinary internal grading.
```

```rzk
-- Formation (the modality itself):
#postulate cert : (ε : G) → U → U
```

### Rule C-INTRO (external introduction only — the oracle axiom schema)
`-- [CONJECTURAL — gated on ALL gates: the schema is only ever instantiated by an actually-passed gate run; each instantiation cites a RESULTS.md. If any gate FAILS, the corresponding certificates are simply never minted — the schema itself needs no weakening, but if ALL gates fail, cert is dropped from the spec entirely (a modality with no possible introductions is dead weight).]`

```rzk
-- There is NO internal introduction rule: no term of the theory constructs
-- cert ε T from a proof of T or from anything else internal.
-- Certificates enter ONLY as postulates, one per passed oracle run:
--
--   for each gate run ρ with verdict PASS at tolerance ε on claim T_ρ :
--     #postulate cert-oracle-ρ : cert ε T_ρ        -- cites gates/<Gn>/RESULTS.md @ commit
--
-- Guard (from G3): a run whose "stability" is self-consistency without held-out
-- ground truth does NOT license an instantiation — confidently-wrong-but-stable
-- is a FAIL at the oracle, hence no certificate. (Anti-steganography clause.)
```

### Rule C-NOELIM (no counit: certification is not proof)
`-- [Design invariant, not gated — but recorded here because it is a *rule about* the gated modality: it can never be promoted INTO an eliminator by any gate outcome. Empirical evidence at tolerance ε never yields an inhabitant of T.]`

```rzk
-- NOT a rule of the theory:
--   (ε : G) (T : U) → cert ε T → T          -- ✗ deliberately absent, at every ε
```

### Rule C-MONO (monotonicity in the tolerance)
`-- [CONJECTURAL — gated on G1: meaningful only if G1 validates the grade order as tracking real tolerance (the same measurements that warrant G1-A calibrate leq-G). If G1 FAILS: weakened — cert loses its grade index and degenerates to an unindexed cert T ("certified at the tolerance it was measured at", no loosening rule).]`

```rzk
#def Statement-C-MONO : U
  := (ε ε' : G) → leq-G ε ε' → (T : U) → cert ε T → cert ε' T
-- a certificate at a tight tolerance is a certificate at any looser one
```

### Rule C-COMP (interaction with graded composition: certificates degrade additively along channels)
`-- [CONJECTURAL — gated on G1 (needs Law G1-A's additive law) AND on G2 for its intended cheap implementation (transport of a certificate along a route is evaluated through the routed-subset kernel, so its COST claim needs G2-B). If G1 FAILS: dropped — certificates do not propagate; each holds only at the world/object where it was measured. If G1 passes but G2 FAILS: the rule may be promoted but its scope note must say transport costs O(N), not O(K).]`

```rzk
-- P : 𝓜 → U a model-indexed claim family; transport of claims along channels
-- is the semantic action of chan on P (postulated with the family, elided here).
#def Statement-C-COMP : U
  := (g : G) (M N : 𝓜) (f : chan g M N) →
     (P : 𝓜 → U) → (ε : G) →
       cert ε (P M) → cert (add-G ε g) (P N)
-- routing a certified claim through a channel of grade g costs exactly g of tolerance
```

### Rule C-PAIR (lax monoidality: combining certificates)
`-- [CONJECTURAL — gated on G1 (candidate rule; promote only if G1's measured tolerance-composition data supports combining independent certificates at the joined grade — this is a STRICTLY optional rule; if the data is silent, leave it unstated rather than weakened). If G1 FAILS: dropped with the rest of the graded layer.]`

```rzk
#def Statement-C-PAIR : U
  := (ε ε' : G) (T S : U) →
     cert ε T → cert ε' S → cert (join-G ε ε') (product T S)
-- join-G: the order-theoretic join (max) in (G, leq-G)
```

### Rule C-FUNC (functoriality along world extension)
`-- [CONJECTURAL — gated on G2: a certificate minted at federation w should survive extending the federation (adding models) precisely because the kernel is atomic — the routed subset's closure, hence the measured stability, is unchanged by growing N (Law G2-B). If G2 FAILS: dropped — every certificate is world-pinned, and any change to the federation voids all outstanding certificates.]`

```rzk
#def Statement-C-FUNC : U
  := (w w' : W) → extends w' w →              -- w' adds models to w, routed subset fixed
     (ε : G) (T : U) → cert-at w ε T → cert-at w' ε T
-- (cert-at : W → G → U → U is the world-indexed form of cert in the intended
--  presheaf semantics; cert ε T ≡ cert-at at the ambient world)
```

### Gate → law promotion table (summary of the inline notes above)

| Law / rule | Gate | If the gate fails |
|---|---|---|
| G1-A graded composition (triangle) | G1 | dropped; grades revert to bare preorder, no enrichment |
| G1-B settling refines pooling | G1 | ablation-weakened to G1-B' (one-step), or dropped entirely (construct = ensemble; all of Part II removed) |
| G3-D densification (edge synthesis) | G3 | dropped; static federation only, no self-densification claim |
| G2-A short-memory closure | G2(a) | dropped; kernel object deleted |
| G2-B atomicity (rank ~ K, not N) | G2(b) | dropped; scope-noted to small fixed federations |
| G2-C residual = repair signal | G2 (bonus) | silently dropped |
| C-INTRO oracle schema | all (per-run) | never instantiated for failed runs; cert deleted if no run passes |
| C-MONO tolerance monotonicity | G1 | weakened to unindexed `cert T` |
| C-COMP transport along channels | G1 (+G2 for cost) | dropped (G1 fail) / cost note weakened to O(N) (G2 fail) |
| C-PAIR combining certificates | G1 (optional) | left unstated |
| C-FUNC stability under federation growth | G2 | dropped; certificates world-pinned |

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
stand up, then present the construct in Rzk for the clean statement.

## Appendix B — Transcription notes for a future `.rzk` file

- Part I transcribes directly: `#postulate` blocks as written; `MeshPath`/`compose` via the
  standard inductive-chain encoding over the sHoTT library's `comp-is-segal`.
- Part II laws transcribe as `#def Statement-… : U := …` ONLY (statements-as-types). Inhabitants
  (`#postulate law-… : Statement-…`) are added one per passed gate, each with a comment citing
  the gate's `RESULTS.md` and commit hash. A red gate means the corresponding `#postulate` is
  never written; this file's failure notes say what, if anything, replaces the statement.
- `G`'s intended instance `([0,∞], ≥, 0, +)` stays semantic (a note), since real-number arithmetic
  is not the point; all spec-level reasoning uses the abstract po-semiring interface.
