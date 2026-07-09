# SETTLE_REAL — coherentflow's settling object on virtualworld's REAL frozen interfaces

**Observe-don't-prove. NOT a gate, NOT a proof.** Bound to `thoughtworld_construct/CONSTRUCT.md`;
context in `IO_STOCKTAKE.md`. Runnable: `python3 virtualworld/settle_real.py` (CPU, ~0.3 s, deterministic;
dumps `settle_real_results.json`).

## What this actually is

`settle_real.py` **imports and calls the REAL committed coherentflow functions** — `structured`, `settle`,
`combined_read`, `make_interface` — the settling object is **not** reimplemented. coherentflow's dimensional
module-globals `(D, T, NTR)` are its declared knobs; we point them at virtualworld's shape (`D=26, n=264,
NTR=132`) exactly as `IO_STOCKTAKE.md` notes the math is fully shape-generic. Nothing in the settling / guard /
read math is altered.

The interfaces are virtualworld's **REAL aligned modality vectors** — ViT-base (vision) + MiniLM (text) +
hand-feature (audio, timeseries) views, each ridge-aligned to the shared 26-dim scene medium — parsed
straight out of `interactive_data.js`. `Y` (standardized scene medium) is the world latent `z`.

**Framing:** the coherentflow object + guard are **VALIDATED** committed component code (0% false-positive
hold-rate over 40 seeds, per `false_positive_ref`). Wiring it onto the real frozen interfaces and reading the
result is **EXPERIMENTAL** observation. This script watches; it does not gate or prove.

## Results (exact numbers)

### 1. REAL modalities — honest NO-OP (as expected)

Settle the 4 real aligned vectors (vision/text/audio/timeseries), `z = Y`:

| quantity | value |
|---|---|
| residual (settle) | `0.0000 -> 0.0000` (flat; tail-contract = False) |
| interfaces held STRUCTURED | **0 / 4** |
| circulation norm | `0.0000` |
| consensus read acc | `0.8485` |
| combined (consensus + held) read acc | `0.8485` |
| combined − consensus | **`+0.0000`** (held_dim = 0; combined is bit-identical to consensus) |

Per-modality `cf.structured()` verdict (why nothing is held):

| modality | eff-rank | cap | held-out predictability `ho` | verdict |
|---|---|---|---|---|
| vision | 14 / 26 | 0.944 | **−1.986** | not held |
| text | 14 / 26 | 0.919 | **−1.512** | not held |
| audio | 12 / 26 | 0.928 | **−0.637** | not held |
| timeseries | 13 / 26 | 0.928 | **−1.397** | not held |

The residual is flat because with nothing held the naive consensus is *already* the coherence fixed point — the
object no-ops from iteration 0. The disagreements are **mid-rank (eff 12–14)** and, decisively, **world-UNpredictable**
(`ho` far below the `+0.30` floor — the concentration `cap` actually clears its threshold, but `ho` fails hard).
So the guard holds nothing, combined == consensus, and the object correctly no-ops. This matches the shipped
`natural_structured_count = 0` and the `honest_label` / xresolve null: **real convergent small models, after
alignment, disagree only by state-independent noise.**

### 2. SEPARABLE control — the SAME code path DOES fire (proves wiring, not a bug)

**2a — decisive, at virtualworld's exact scale.** coherentflow's own separable construction
(`make_interface` + a hidden binary branch injected into ONE interface) on an **isotropic** latent at `D=26,
n=264`. Only the latent differs from case 1 (isotropic-separable vs real-convergent `Y`):

| quantity | value |
|---|---|
| interfaces held STRUCTURED | **1 / 3** (interface 0), tail-contracts = True |
| carrier guard | eff-rank 12/26, cap `0.833 > 0.692`, `ho = +0.596 > +0.30` -> STRUCTURED |
| consensus read acc | `0.4924` (chance — consensus-collapse loses the branch) |
| combined read acc | **`1.0000`** |
| combined − consensus | **`+0.5076`** |

The same functions, at the same scale, **hold the structured decoherence and the combined read decisively
beats consensus-collapse.** So the real-case no-op is the **DATA** (real-model convergence), not a wiring bug.

**2b — corroboration on the real Y medium.** Injecting a branch into interfaces built with `make_interface`
**on `Y`**: the guard **still fires** (1/3 held, carrier eff-rank 11/26, `ho = +0.836`). The combined − consensus
payoff is only `+0.0076` — modest because `Y`'s real low-rank structure plus `n = 264` (132 train frames) limit
the joint consensus+held probe synergy that reaches `+0.5` at isotropic scale. The point stands: **the hold path
fires on Y-based interfaces too.**

### 3. NOISE guard — stays REJECTED (no fabrication)

Corrupt the real `vision` interface with unstructured, high-variance noise (`3.0 · N(0,1)`), `z = Y`:

| quantity | value |
|---|---|
| interfaces held STRUCTURED | **0 / 4** |
| circulation norm | `0.0000` |
| corrupted-interface guard | eff-rank **25/26** (near full-rank, diffuse), `ho = −1.542` (world-UNpredictable) -> NOISE |

Unstructured corruption is **not** written to the tape — the object does not fabricate structure.

## What this shows (plain English)

The settling **mechanism is real and correctly wired**. On a synthetic **separable** input the same code path
holds structured decoherence and the combined read decisively beats consensus-collapse (`+0.51`); unstructured
noise stays rejected (0 held). On virtualworld's **real, convergent frozen senses it is honestly INERT** — their
post-alignment disagreements are mid-rank and world-unpredictable, so nothing is held and combined == consensus
(bit-identical). **The object correctly no-ops on the real modalities.** That inertness is a property of the
DATA (real-model convergence), demonstrated by the separable control firing on the identical code at the
identical scale — not a bug, not a broken wiring.

Observe-don't-prove, not a gate. The coherentflow object is VALIDATED (component-level); wiring it onto the real
interfaces and reading the result is EXPERIMENTAL observation.
