# DIAGNOSTIC — channel/token structure vs. one mean-pooled vector (findings + recommendation)

**Question (from the brief):** we feed the field ONE mean-pooled vector per model (ViT 768, MiniLM 384),
which averages away all per-token/channel structure *before* the field sees it. Empirically, on the REAL
models: does keeping channel-level structure give the field genuinely richer input, or are the channels
mostly redundant/noise? **Diagnose and recommend a granularity — do NOT change the field architecture.**

Read-only pass. Nothing in `field.py`/`explore.py` changed. Reproduce with
`diagnostics/encode.py` (needs `world.py` from `archive/pre-nuke`) then `diagnostics/channel_diag.py`.
Setup: 320 physics-world frames, ViT `last_hidden_state` (197 tokens × 768), MiniLM (64 tokens × 384),
target = the 26-dim world state Z. Held-out R² is PCA-conditioned (top-64 via Gram-SVD, ridge λ=5) because
n≪features. The **plain mean-pool is the null throughout.**

## Finding 1 — pooling does throw away real per-token spread
Within-frame token variance vs between-frame pooled variance:

| model | within / between token-var | reading |
|---|---|---|
| ViT | **1.9** | tokens are not redundant copies — pooling discards real spread |
| MiniLM | **165.7** | huge per-token spread (CLS vs word tokens); pooling collapses it hard |

So structurally there *is* something being averaged away. Finding 1 only says the spread exists — not that
it carries **world-signal**. Findings 2–3 test that.

## Finding 2 — but almost none of that spread is world-signal (93–98% of tokens are noise/dead)
Held-out R² predicting the world state, cross-rollout split (train on some rollouts, test on unseen ones):

| model | pooled | full token-concat | per-token max | per-token median | frac tokens ≤0 (noise/dead) |
|---|---|---|---|---|---|
| ViT | 0.015 | **−0.692** | 0.093 | −0.368 | **0.93** |
| MiniLM | −0.018 | **−1.736** | 0.019 | −0.169 | **0.98** |

Exposing the **full** token grid is strongly *worse* than the pool (negative R²) — it hands the field 93–98%
noise dimensions. Per-token, the best single ViT patch reaches only 0.093 and the best MiniLM token 0.019.
**Naively exposing all channels would degrade the field, not enrich it.** This kills option (c) outright.

## Finding 3 — a *selected* ViT patch subset beats the pool, but mostly via noise-dilution, not localized signal
The sharper question: is there a **principled subset** that beats the coarse pool? Tested with an
honest 3-way split (FIT ranks nothing / fits ridge, SEL ranks patches by signal, TEST reports — selection
never sees TEST) **plus a random same-size subset control**, averaged over 8 reshuffled iid splits:

| input to the pool | held-out R² (mean ± std over 8 splits) |
|---|---|
| **pool ALL 197 tokens** (current null) | 0.444 ± 0.013 |
| pool top-8 *signal-selected* patches | **0.483 ± 0.015** |
| pool top-20 *signal-selected* patches | **0.485 ± 0.017** |
| pool 8 *random* patches (control) | 0.462 ± 0.013 |
| pool 20 *random* patches (control) | 0.464 ± 0.013 |

Two things, and the control is what matters:
1. **Any small subset beats pool-all.** Even *random* 8/20-patch pools (0.462/0.464) beat pooling all 197
   (0.444). Most of the gain is just **less noise dilution** — pooling over 197 tokens (93% noise) drags the
   summary toward mush; pooling fewer tokens recovers ~0.02 R², and you get that for free from *any* subset.
2. **Selection adds a real but small extra.** Signal-selected subsets (0.483–0.485) beat random subsets
   (0.462–0.464) with win-rate 0.75–0.88 across splits — so some patches genuinely carry more world-signal
   than average, but the *selection-specific* edge is only ~+0.02 R² on top of the free dilution win.

(Note the two split regimes: pool-all scores 0.444 under **iid** reshuffled splits but only 0.015 under the
**cross-rollout** split of Finding 2 — the ViT pooled features transfer poorly to *unseen rollouts*. That
weak cross-rollout transfer is itself a caution: the patch signal is partly rollout-specific, not a clean
generalizing world-readout.)

## Signal/noise split — summary
- **ViT:** ~7% of patches carry weak world-signal (the ball/moving-object patches); ~93% are
  background/noise/dead. Structure exists but is sparse and weak (best patch R²≈0.09).
- **MiniLM:** ~2% of tokens above zero, max 0.019 — **essentially no per-token world-signal.** The text
  branch describes the world but its per-token structure adds nothing over the pool (and the pool itself
  barely reads the continuous world state).

## Recommendation
**Split by model — this is option (b) for ViT (narrowly), option (a) for MiniLM, and definitely NOT (c).**

- **MiniLM → (a) coarse is fine.** Per-token structure is noise for world-state purposes (98% ≤0, max
  0.019). Keep the single pooled vector; there is nothing to expose.
- **ViT → (b) a principled subset, but with a caveat that it's barely worth it.** The signal-carrying patches
  are real and selectable (win-rate ~0.8 vs random), but the honest, control-adjusted payoff over the
  current pool is only **~+0.02–0.04 R²**, and roughly half of that is generic noise-dilution any subset
  gives you. If we expose more ViT input, expose **a small pooled subset of the top signal patches (≈8–20),
  not the raw tokens** — never the full 197-token grid (that's option (c), and it's *worse* than the pool at
  −0.69).
- **(c) full channel-level is rejected.** Empirically degrades to negative R² for both models.

### The load-bearing caveat for the field specifically
This diagnostic measures *predictive richness of the features* (a coverage-union / baseline-R² axis). It does
**not** show the field's distinctive behavior would benefit. Per `EXPLORE.md`, the field only beats the
average when frames genuinely **conflict** (coupled ρ>1); real independently-trained models converge, so it
no-ops. Finer ViT patches are all **one model's** views — they will converge among themselves, not
manufacture ρ>1. So richer patch input plausibly helps the **standing coverage-union baseline** (better
features → better stitch), but there is no evidence it supplies the *conflicting-frame fuel* the field needs
to beat averaging. I would **not** re-architect the field around patch-forces on the strength of a ~0.02 R²
feature gain.

**Bottom line:** channels are *mostly* redundant/noise (Finding 2); a thin ViT patch subset carries a little
independent signal (Finding 3) but the control shows most of the apparent win is dilution, not localization;
MiniLM has none. Recommend **keeping the pooled field input as-is** and, *if* we chase the ViT subset at all,
doing it in the coverage-union baseline (where feature-R² is the metric), not by changing the field object.
Awaiting your call before any architecture change.
