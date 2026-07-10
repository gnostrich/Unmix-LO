# THE CONSTRUCT — canonical spec (bind to this; do NOT flatten into a cobbled pipeline)

This is ONE process, not a pipeline of gadgets. Every prior Claude Code test touched a COMPONENT
(routing, fusion, single-fragment deviation). None has tested THE CONSTRUCT. Keep that distinction.

## >>> CANONICAL THEORY LAYER (2026-07-09) — see THEORY.md; BINDS EQUALLY <<<
A layer of theory developed after this spec was written is captured in **`THEORY.md`** and binds equally. The
seven items: **T1** mutual-instability IS the Baur objective (the loss is the coupled operator's spectral
radius, grounded via the models' own frames — non-vacuous even at zero content-disagreement); **T2** fluid
exclusion (descending instability routes AROUND a destabilizer, weight→0, no explicit reject); **T3** per-query
**stability terrain** is the informative object (damped/agreed vs amplifying/contested vs swirling — non-vacuous
even on convergent models, a confidence gradient); **T4** information relocation (agree→equilibrium POINT,
diverge→the DYNAMICS around it); **T5** trace-native output (the settling trace's tail MOTION is the answer;
directions=branches, magnitude=uncertainty, rotation=ambiguity; NO fitted head); **T6** streaming/anytime I/O
(converge=consensus / cycle=held-superposition / budget=uncertainty; the manner of stabilization is part of the
answer); **T7** the field/settling mental model (models = forces tensioning a shared field; read the taut
resting position, read the tremble where slack).

**Reconciliations that update the items below** (details in THEORY.md):
- Non-negotiable **#4**: its GROUNDING requirement stands, but its implicit "content-reconstruction loss"
  reading is **superseded by T1** — the objective is instability of the *grounded coupling*.
- The **informative object relocated**: CONSTRUCT's "swirl / single-fragment deviation / held-structure" is the
  **divergent-case reading (T4/T6-cycling)** of the more general object, the **terrain (T3)**. Convergence nulls
  (xresolve/THOUGHTWORLD) therefore do NOT close the construct — the graded terrain on convergent models is the
  untested object.
- Non-negotiable **#2** self-expansion (Hankel-SV vs FDT floor) is empirically weak (`fdt_denoise` MID/DOWNGRADE;
  `mz_fluid` reduces to linear filtering); candidate re-tie to *persistent instability modes of the terrain*
  (open joint J2).
- **Output**: any fitted-probe readout head contradicts **T5** and is retained only as an external *scoring*
  tool, never as the object's output.

Current code-vs-theory fidelity is tracked by the executable **conformance suite** (`conformance/`,
`CONFORMANCE.md`) and the module inventory in **`STOCKTAKE.md`**.

## The one object (say it as one sentence)
A SELF-EXPANDING OPERATOR-VALUED MORI-ZWANZIG MEMORY living on an NTM-LIKE RESIZABLE TAPE (the "fluid"),
POPULATED BY THE BAUR DESCENT against the LOSS CARRIED BY THE SEED MODELS' OWN TRAINING DATA, where the
streaming/memory split natively handles propose-vs-settle and single-vs-composite, and the memory
SELF-EXPANDS by an atomicity / Hankel-rank criterion.

## Non-negotiable structure (regression guard — if a design violates these it has FLATTENED the construct)
1. TAPE = MEMORY = ONE OBJECT. The NTM-like resizable tape and the MZ memory kernel are the SAME thing.
   The tape's read/write dynamics ARE the MZ kernel. Do NOT model "a memory module" bolted onto a tape.
   - streaming term = direct/current contribution (single-hop)
   - memory term    = delayed, through-other-models / through-past contribution (multi-hop, cyclic)
2. RESIZABLE / SELF-EXPANDING. Memory is not fixed-width. It grows: new mode appended when the closure
   residual's Hankel singular value clears the second-FDT noise floor; balanced-truncation prune below floor.
   One spectral atom = one state dimension. (This is the atomicity dial as the expansion criterion.)
3. BAUR DESCENT POPULATES IT. Nothing is hand-wired. Routing paths / compositions get WRITTEN into the
   memory by descending a loss. Cardinality AND topology (single / multi / cyclic) EMERGE from the objective,
   never forced. The MZ streaming/memory split natively carries the atomic-vs-composite distinction.
4. LOSS COMES FROM THE SEED MODELS' OWN TRAINING DATA. Not arbitrary data, not an invented external "judge."
   Each seed-included model carries its OWN loss signal from its OWN training data; the descent uses those.
   Seed models bring their grounding WITH them.
5. FAITHFULNESS IS A LOSS TERM, NOT A PHASE. The anti-hallucination / contraction guard is a TERM INSIDE
   the single Baur objective (penalize non-convergence / seed-inconsistent / unadmitted writes). It is NOT
   a separate "verify pass." The native MZ process converges (settles) via its own dynamics; the term just
   keeps it contractive on the seed-consistent manifold. Off that manifold it need not contract.
   DO NOT re-introduce a two-phase generate/verify pipeline -- that was a FLATTENING error. Propose-and-settle
   is native to the streaming/memory decomposition.

## Seed (current vs designed)
- CURRENT (built): seed = the physics engine ONLY (numpy rigid-body, 5 balls, gravity, collisions, D=20).
  Coherent, self-consistent, directed, rollable. This is the flat reference nabla_0.
- DESIGNED (not yet built): seed EXTENDS to the COHERENT CORES of admitted models across modalities. A model
  (or a REGION of a model) GRADUATES from fragment -> seed where it is (a) internally self-consistent,
  (b) consistent with existing seed on the overlap, (c) connected (shares substrate). Where it deviates,
  that region stays FRAGMENT (the swirl). Same model can be partly seed, partly fragment, per-region.
  The graduation rule (fragment-region -> seed-region) is a REAL open joint, not yet implemented.

## Seed membership law (the physical-bridge principle -- do not lose this)
Everything attaches through a PHYSICAL BRIDGE. Abstraction-depth scales with substrate-depth. Far
abstractions (math, symbolic, computation) do NOT attach as external modalities -- they EMERGE as
high-level regularities of a rich-enough physical substrate (Minecraft-analog-computer style). Direct-fit
modalities for a ball-and-wall world: vision (renders), text (describes), audio (collision sounds),
time-series (physical quantities over time). Chemistry/tabular/computation need much richer substrate =
OPEN FRONTIER (the substrate-depth -> attachable-abstraction map is uncharted).

## Abstraction = stable holonomy of re-framing (the emergence hypothesis)
A re-framing = the fluid representing the same state under a different model's typing. Abstraction emerges
when composing re-framings AROUND A LOOP does not return home (holonomy != 0) AND that non-return is STABLE
across states (a persistent reusable coordinate). Test = atomicity of the re-framing-loop holonomy (the SAME
instrument, pointed at loops not edges). Thinking = loose-typing / permissible-inconsistency traversal of this
curvature; verification = imposing consistency to keep survivors. Generative permits inconsistency; the loss
term disposes.

## Closure bound (honesty ceiling -- never claim past it)
The construct reaches the DEDUCTIVE CLOSURE of the seed+fragments' knowledge under re-framing/composition --
potentially vast, but NEVER beyond it. It ASSEMBLES/RE-FRAMES reachable structure; it does NOT create
knowledge no member holds. "Abstraction emerges" = abstractions DERIVABLE FROM THE GROUNDED SUBSTRATE emerge
(arithmetic from counting objects), not arbitrary abstraction from nowhere. No ">" over the closure = fabrication.

## What has and hasn't been tested (anti-regression ledger)
TESTED (components, all with pre-registered gates):
- routing / union-without-drag (ROUTEMESH): conditional PASS on disjoint-competence tasks; flat cost in N.
- fusion / blend-by-superposition (fluid probe): beats routing 2.5x on BLEND tasks (granted alignment).
- single-VISION-fragment deviation from engine (THOUGHTWORLD): NOISE (eff-rank ~16.4 = random floor).
- single-DYNAMICS-fragment deviation (THOUGHTWORLD-2, LLM/video): [was running; user stopped it].
  NOTE: that probe has a KNOWN CONFOUND on its positive branch -- the LLM is HANDED velocities in the prompt,
  so a "structured deviation" could be prompt-arithmetic, not world-knowledge. Velocity-withholding control
  REQUIRED before any ATOMIC reading is believed.
NOT TESTED (the construct itself):
- the tape = MZ memory as ONE object
- the Baur descent populating it
- self-expansion by the Hankel/atomicity criterion
- multiple seed models contributing their OWN training losses into one descent
- faithfulness/contraction as a loss term
- the fragment-region -> seed-region graduation rule
- abstraction as stable re-framing holonomy
=> THE CONSTRUCT AS A WHOLE IS DESIGNED AND ARGUED BUT UNVALIDATED. No single component probe changes this.

## Open joints (the real underspecified parts -- work these, don't re-litigate settled pieces)
J1. How do multiple seed models' INDIVIDUAL training losses combine into ONE Baur descent objective?
J2. Exact self-expansion trigger: when does the tape add a dimension? (Hankel SV vs second-FDT noise floor --
    needs a concrete estimator on real memory state.)
J3. The graduation rule: how does a fragment-region become a seed-region operationally?
J4. The substrate-depth -> attachable-abstraction map (the emergence frontier; needs the densification dial).
J5. Rich-substrate seeds need GROUNDED initial data, not arbitrary -- how much, and of what.

## Standing discipline (bind to this on EVERY Claude Code run)
- Do NOT flatten the single MZ/tape process into a hand-built pipeline or two-phase verify gadget.
- Faithfulness/contraction = a LOSS TERM, never a separate phase.
- Loss = seed models' OWN training data, never an invented external judge, never arbitrary data (for the rich case).
- Test the CONSTRUCT vs a COMPONENT explicitly; state which every time.
- Closure bound: assemble/re-frame reachable structure, never claim knowledge no member holds.
- Pre-register; random/anti-fabrication control on every positive; honest NOISE/RED is a real finding.
- A surprising positive gets MORE scrutiny (the velocity-handoff confound is the live example).
