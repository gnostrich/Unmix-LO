# SPEC (Track B) — Rzk formalization of the construct (SPEC-LEVEL, not full metatheory)

## Scope discipline
Formalize the *specification* of the object, not a mechanized metatheory. Goal: make the construct
PRECISE and its novelty LEGIBLE, distinguishing it from routers / merging / graph-MoE. Do NOT attempt
a full verified graded-modal-directed kernel extension — that is genuinely large. Shallow, spec-level.

## What to formalize (ONLY the parts a gate has passed — coordinate with gates/)
Directed type theory (Rzk) is the right home because channels are LOSSY/NON-INVERTIBLE (A->B->A != id),
which is directed-native and NOT expressible in symmetric (cubical) identity.

Core skeleton (result-independent — safe to draft now):
- **Objects** = models (or their frame-aligned representation spaces).
- **Morphisms** = channels, as DIRECTED homs (non-invertible). Composition = path through the mesh.
- **Equivalence** = the invertible fragment (lossless-reversible channels) — derived special case.
- **Path-coherence** = the diagram commuting up to graded tolerance (paths between two objects agree).
  This is the intended "path isogeny": structure-preserving agreement of routes up to controlled collapse.

Graded/quantitative layer (formalize AFTER G1/G2 pass — this is the tolerance grade):
- Each morphism carries a GRADE (tolerance), valued in a partially-ordered semiring (metric/[0,inf]).
- This is an INSTANCE of existing graded modal dependent type theory (Moon-Eades-Orchard GrTT;
  Atkey QTT; Abel-Danielsson-Eriksson graded-modal, formalized in Agda). Cite, instantiate — do NOT reinvent.
- Composition ADDS grades (triangle-inequality / Lawvere metric enrichment). G1's tolerance-composition
  sandbox result (composite <= sum-of-hops held 100%) is the empirical justification for this rule.

MZ-kernel / atomicity (formalize AFTER G2 passes):
- The routing memory = a Mori-Zwanzig memory kernel; its FINITE/LOW RANK (atomicity) = the property
  that makes cost scale with routing-width not federation-size. State as a property of the kernel object.

Certification modality (the genuinely novel, boundary-touching piece — formalize carefully, AFTER gates):
- A modality "T cert_eps" meaning: T is certified stable-to-tolerance by an EXTERNAL empirical oracle
  (the stability gate / self-consistency check), not by a proof. Interpreted in a semantics whose worlds
  are the frozen models. This is NOT in the graded-modal literature (they track internal resources);
  a modality grounded in an external valuation is the novel seam. Two external ports remain, correctly:
  (1) the given models, (2) the valuation. Everything else internal. Do not try to internalize the valuation.

## Rationale note to include in the paper (Agda vs Rzk)
Built in Agda: graded machinery exists but directedness must be hand-encoded -> contribution dilutes.
Built in Rzk: directedness is native -> the contribution is ONE clean layer (grading+certification over
directed types). Rzk is the right home because it isolates the novelty. (Recommended: shallow-embed the
graded/cert modality in Agda first to de-risk the rules, then present in Rzk for the clean contribution.)

## Deliverable
spec/virtualmesh.rzk (or .md if Rzk tooling blocks) : the typed skeleton + the passed structural laws,
with a clear marker of which laws are gate-validated vs conjectural. Conjectural laws are labeled, not asserted.
