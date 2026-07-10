# EXPLORE — what the field object actually does (honest, average-null throughout)

Rebuilt from the concept (no conformance suite, no tests to pass). `field.py` = the object; `explore.py`
reproduces every number below. The one non-negotiable was faithfulness: *if the coupled flow can't reach
spectral radius > 1 for conflicting frames, it's averaging in disguise.* Stance: exploratory — report what it
does, with the plain average as the null. No relabeling a no-op as a result; injected structure labeled as
injected.

## The object (one thing)
Models are FORCES (operators Rᵢ = frames) tensioning a shared field. The field settles under bounded coupled
feedback `s ← s + step·Σ wᵢ·cap·tanh((Rᵢ s − s)/cap)` — each model reads the current state through its frame and
writes back a bounded correction, so models drive each other and the flow CAN be unstable (not averaging). You
probe with a query and read the trace's TAIL MOTION (stopped→point/consensus, trembling→branches, rotating→
ambiguous); the per-query TERRAIN is `‖q Jᵀ‖` (how stability varies with the query); the read streams until it
converges/cycles/times-out.

## What it DOES — regime by regime (numbers from `explore.py`, D=8)

| regime | coupled ρ | terrain (contested-frac) | tail-read | vs average |
|---|---|---|---|---|
| **convergent frames (agree)** | 0.955 ≤1 | flat, std 0.000, 0% contested | CONSENSUS 12/12 | ties (honest no-op) |
| **conflicting frames (diverge)** | **1.090 >1** | contested, 95% amplifying | SOFT (unresolved pull) 12/12 | terrain ≠ average (see below) |
| **identical control** | 1.000 | flat, 0% contested | CONSENSUS 12/12 | ties (correct) |
| **REAL modalities** (ViT/MiniLM/audio/ts) | **0.962 ≤1** | std 0.032 (< identical-control 0.051) | — | ties the average, no-op |

### 1. It is FAITHFUL — a genuine feedback object, not averaging ✅
Conflicting frames drive the coupled operator to **ρ = 1.090 > 1**; convergent frames give ρ = 0.955 ≤ 1.
Averaging is structurally capped at ≤ 1 and can never do this. So the object passes the one real check: it is
the feedback fluid, not averaging in disguise.

### 2. It honestly NO-OPS on agreement — and on real models ✅ (but this is the null)
Convergent frames and the four **real modalities** both settle to CONSENSUS: coupled ρ ≤ 1, tail motion → a
point, terrain flat. The real-modality terrain std (0.032) is *below* the identical-input control (0.051), so
there is no real conflict signal there. The field ties the plain average on real input. This is the expected,
honest outcome (real independently-trained models converge — the whole prior program's finding).

### 3. Its one behaviour distinct from the average: an agree-vs-conflict TERRAIN — but only on genuine conflict
On conflicting frames the terrain is contested (95% of query directions amplify, ρ>1); on agreeing/identical
frames it is flat. The plain average returns the *same consensus point* whether the models agreed or fought to
get there — it is blind to that distinction. So the terrain carries a real agreement/confidence signal the
average lacks. **But** it is non-vacuous *only* when the frames genuinely conflict (ρ>1), and real modalities
don't reach that regime — their terrain is flat (below the control). So the signal exists but is silent on real
input.

### 4. Held-superposition does NOT beat the average ❌ (the payoff fails)
The paraconsistent payoff — recovering competing branches from the tremble — does not fire usefully:
- **Fair test** (a hidden ± distinction carried by one FRAME, neutral query): field settled-state recovers the
  branch at **0.550**, the average null at **0.562** — both near chance, **field does not beat the average**.
- On conflicting frames the tail-read is **SOFT (unresolved pull)**, not a clean branch-recovering tremble.
- Consistent with the earlier finding (archived): a settling read collapses/drifts/blows-up on real-derived
  operators; robust branch-holding needs explicit hand-built bistable wells, which do not emerge from the frames.

## Verdict — plainly, no hedge
**It does X:** the object is a *genuine feedback fluid* (reaches ρ>1 on conflicting frames — not averaging), it
*honestly no-ops to consensus* on agreement, and its *terrain distinguishes agree from conflict* — a per-query
signal the average is structurally blind to.

**It doesn't do Y:** on **real** independently-trained models the frames converge, so it no-ops and **ties the
average** — the contested terrain is silent there. And the **held-superposition payoff does not beat the
average** even on synthetic/injected conflict (branch recovery ~chance, tie with the average). The interesting
behaviour (contested terrain) is real but appears *only* on genuinely conflicting frames, which real models
don't produce, and it does not translate into a usable read that beats averaging.

**Net:** an elegant, faithful mechanism whose only average-beating signal (agree-vs-conflict terrain) requires
fuel — genuinely conflicting frames — that real convergent models don't supply. On real input it is an honest
no-op that ties the average. The paraconsistent held-superposition, the thing that would have made it more than
averaging on real data, does not fire.

## Standing baseline (unchanged, genuinely useful)
**Coverage-union** — fusing complementary modalities beats best-single — remains the real, un-blocked win
(stitch R² **0.445** > best-single 0.337, in `data/real_modalities.js`). That result does not depend on the
field object and stands on its own. The prior program's ledger is in `report.md` / `COMPOSITION_THESIS.md`.
Full construct history (fluid, conformance, theory, the regression saga) is preserved on branch
`archive/pre-nuke`.
