# Pre-registration — collective routing through modality OVERLAP (committed BEFORE running)

**Thesis under test (the architecture's real claim).** Cross-modal routing should emerge from diverse members
with partial modality OVERLAP, routed collectively by F — *not* from an imposed cross-member pairing
dictionary. No member is a translator; the ensemble routes. The current demo leans on a hand-paired
dictionary (CIFAR image_i ↔ caption_i, forced equal via the TIED coupling) precisely because its only genuine
cross-modal overlap is CLIP — one thin bridge. **Add overlap and the dictionary should become unnecessary.**

**The overlap-maker.** `demo/world.py` — a 2D physics micro-world. From one ground-truth scene it emits three
paired modalities: a **render** (overlaps vision), a **caption** (overlaps text), and a **dynamics signature**
(a third modality nobody else speaks). Its cross-modal consistency is ground truth (the simulator's own state),
not a curated dictionary — so it is a *natural* bridge, like CLIP's two towers, plus a new axis.

## What is and isn't new
- **Not new mechanism.** Routing is the EXISTING untied `engine.equilibrate` (block-coordinate F descent).
  The world member is a new MEMBER (probe library + per-channel encoders producing (D,w) like every other
  port). No new authority, no new QC.
- **The whole point of "untied":** each member aligns its own probe-geometry to the shared anchor
  INDEPENDENTLY. There is NO constraint forcing member A's event-i onto the same atom as member B's event-i.
  The only thing that can bridge vision↔text is the cross-modal consistency *inside* the multimodal members
  (CLIP; the world model). This is the honest opposite of the tied dictionary.

## Substrate
Shared reference set = **N=256 world scenes** (`world.event(seed)`), seeds fixed. Each member encodes the
events IN ITS OWN MODALITY, independently. Member roster is chosen to MAXIMIZE cross-modal diversity within
Claude-Code CPU/download tractability (one new download = SigLIP; all else local):
- **vision** (pixels): ViT, MobileNet, CLIP-vision, SigLIP-vision — encode each scene's **render**;
- **text** (language): MiniLM, CLIP-text, SigLIP-text — encode each scene's **caption**;
- **dynamics** (trajectories, a THIRD modality): the world member's dynamics channel;
- the **world member** encodes each scene across **render / caption / dynamics** (internally consistent → a
  ground-truth bridge). CLIP and SigLIP are two further, independently-trained vision↔text bridges.
Only (D, w) crosses (R3). Anchor budget m=14. Seed set {0,1,2} for the routing solver; scenes held fixed.

**Registered next diversity axis (not this run): AUDIO.** The world can synthesize a waveform (bounce events →
clicks, speed → pitch); a small audio encoder would add a 4th modality and further overlap. Deferred to keep
this run tractable; noted so adding it later is a pre-registered extension, not a scope surprise.

## Conditions (the core contrast)
- **A — thin overlap (control):** members {ViT, MobileNet, MiniLM, CLIP-vision, CLIP-text}. CLIP is the ONLY
  cross-modal bridge. Untied. This is the regime where cross-modal transfer was FRAGILE (the dog→ship wall).
- **B — rich overlap (thesis):** A **+ SigLIP (a second, independently-trained vision↔text dual-encoder
  bridge) + the world member** (render+caption+dynamics). Untied. Now THREE independent bridges (CLIP, SigLIP,
  world) plus a shared dynamics axis. B bundles two enrichers deliberately (the thesis is "rich overlap
  recovers routing," not "which model"); per-member ablation (world-only vs SigLIP-only) is a registered
  follow-up, not this run.

## Task & metric
Hold out K=24 fresh scenes (unseen seeds). For each, feed its **render** to the vision members only; read the
**silent text members'** top-1 retrieved caption. Score attribute matches against the input scene's PRIMARY
object:
- **color match** (4 colors → chance 25%),
- **shape match** (3 shapes → chance 33%),
- **motion match** (direction words → chance ≈ 40% given the phrase set).
Also run the reverse (feed caption → read silent vision members' top render's ground-truth attributes).
All accuracies are seed-averaged over the 3 solver seeds. Gauge check: scramble one member's features → require
|ΔF| < 1e-6 (R3 must survive untied too).

## Registered predictions
This is a NEW experiment with NO prior calibration curve, so — unlike the battery/2×2 — I register **direction,
chance levels, and falsifiable thresholds**, NOT exact digits (registering fake digits with no calibration
would be the dishonest move). Predicted:
- **B recovers cross-modal transfer:** color match ≥ **0.50** and shape match ≥ **0.45** in condition B
  (both well above the 0.25 / 0.33 chance floors), stable across the 3 seeds (std ≤ 0.15).
- **Overlap is what does it (A vs B):** color match improves by ≥ **0.15** from A to B (B − A ≥ 0.15). If A is
  already high, the thin-CLIP bridge sufficed and the world member is redundant — reported honestly, not spun.
- **Gauge survives:** |ΔF| < 1e-6 under scramble in both conditions.
- **Reverse direction** (text→vision) tracks the same direction (B ≥ A), reported but weighted second.

## Verdict — PASS iff ALL:
1. B color ≥ 0.50 AND B shape ≥ 0.45 (above chance, stable).
2. B − A color ≥ 0.15 (the overlap, not CLIP alone, carries it).
3. |ΔF| < 1e-6 under scramble.
If (1) holds but (2) fails → "CLIP overlap already sufficed; world member redundant" (honest partial). If (1)
fails → "collective untied routing does NOT recover transfer even with added overlap; the dictionary is still
load-bearing" (the wall bites again — reported straight, no patching, and the tied dictionary demo stands as
the honest baseline).

## Registered risks
- The orbit/free-lunch ambiguity that broke the CLIP-only bridge may persist even with two bridges — richer
  overlap is a hypothesis, not a theorem. A clean FAIL here is a real result about how much overlap collective
  routing needs.
- The world member's render encoder and ViT see the same pixels but with unrelated geometries; their
  relational overlap is assumed, not guaranteed. If the world-render geometry does not correlate with ViT's,
  the world member won't bridge and B≈A — that too is reported.
