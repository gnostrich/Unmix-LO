# WALL: cross-modal transfer to silent heterogeneous models (R5 readout)

Reported per the covenant — a blocker honestly stated, not shimmed. The demo runs end-to-end and most of
R1–R5 works; ONE piece (the silent-model cross-modal panel) hit a fundamental wall.

## What works (real, on the live demo)
- **F-loop on real 256-point clouds:** equilibrates, Lyapunov-monotone (backtracking guard holds on real data).
- **Channel routing B (R4):** per-prompt gains adapt (e.g. vit B=[0.51,0.49], mobilenet [0.54,0.46]).
- **Within-modality coherence:** a dog image → vit, mobilenet, AND clip_vision all top-exemplar `dog` — three
  different vision families independently agreeing through the shared anchor.
- **Consensus + session line;** **R3 gauge guarantee** user-visible (`--scramble` → |ΔF| ≈ 1e-14).
- **CLIP-tower cross-modal transfer in isolation:** with clean class-anchored captions, clip_vision→clip_text
  transfers (dog image → "a photo of a dog"). Proven.

## The wall
Feeding a dog image, the SILENT heterogeneous text models (minilm, clip_text) inside the full 5-port anchor
surface the WRONG class ("a photo of a ship / automobile / frog"), not dog. Cross-modal transfer to silent
heterogeneous ports does not work.

## Why (localized across ~8 experiments)
- **Relational-only GW discards the cross-modal correspondence.** R3 keeps only within-library geometry D_v;
  which library point matches which across modalities (the semantic alignment) lives in the *cross* positions,
  which never cross the interface. GW then aligns each port's D_v to the anchor **independently**, with no
  signal tying image-i to caption-i — so concepts land on arbitrary, per-port-inconsistent atoms.
- **It works only when ports share geometry.** CLIP's two towers are trained into one space, so
  D_vision ≈ D_text and GW's identity coupling aligns dog↔dog. Heterogeneous models (vit vs minilm) have
  unrelated geometries; GW cross-alignment is then arbitrary.
- **Paired probes make it trivial, not solved.** If all ports literally share the 256-concept index (paired),
  you can read the paired side directly — but real frozen models each have their OWN behavioral support with
  no shared index, which is the whole premise. Free-form captions fail worse still: a caption's off-object
  content ("…in a kitchen") dominates its embedding and destroys even the class alignment.

This is the same tension the earlier multimodal thread hit (it needed CCA on a shared subspace — i.e. it used
the cross-modal alignment that gauge-invariance forbids here).

## Re-derivation options (manager's call — a claim/design fork, not a patch)
1. **CLIP as the declared cross-modal bridge.** Route cross-modal traffic through CLIP's aligned towers (the
   channel-native model), other models consensus within-modality. Faithful (CLIP is R2's ideal channel pair),
   and it demonstrably transfers — but the cross-modal claim then scopes to CLIP, not "any heterogeneous pair."
2. **Tied couplings for shared-support (paired) measures.** When ports share a probe index, one coupling to
   the anchor (concept→atom) instead of independent per-port couplings — forces cross-port consistency. This
   is a real mechanism addition (R3 for paired n-measures), needs derivation + a Lyapunov proof.
3. **Accept a one-axis demo.** Within-modality consensus (dog→dog across vision families) + gauge + meter is
   itself a real result; scope the silent-model panel to same-modality silent members and state the
   cross-modal limitation.

I did not dress up the failing panel or claim the acceptance test's cross-modal criterion passes. Everything
above is on the live demo.
