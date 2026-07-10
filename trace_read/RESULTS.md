# RESULTS — trace read-out: distinguishing gate + poles-first calibration (honest, local)

Pre-registration frozen before run code (commit `06f3bf9`). This reports two things and then **stops at the
real-model boundary**, per the frozen preconditions: (1) the preflight-by-hand distinguishing verdict, and (2)
the poles-first calibration of a fresh read-out instrument. **No real-model number is produced** — the data
that would make one interpretable is not present, and per the PREREG that is *untestable*, not NOISE.

## 1. Preflight-by-hand (registry absent) — DISTINGUISHABLE
`preflight`/`registry` do not exist in the repo (the registry build did not land), so the dead-composition
flag was checked by hand against the buried corpse (`report.md`, `COMPOSITION_THESIS.md`).

- **Dead / buried:** combination beating parts (">") — synergy, capability-beyond-parts. Killed across
  BIOMESH / synergy; *"composition is a cost-and-routing story, not a capability story."*
- **This object:** whether a **single model's** conditional expectation `E` is finitely-readable, and (pairwise)
  whether the non-commuting residue has structure. Scored as **readability + residue-atomicity (structure)** —
  on the *"="* side (readability of what is already in one model), never as task-performance gain over parts.
- **Bright line (hard stop, frozen):** if any metric ever compares **"A+B performance vs best-single on a
  task,"** the run has relitigated the corpse and must STOP. "Read the trace so models compose (to beat parts)"
  = the dead synergy claim in new vocabulary; "read a single model's trace to test its readability" = the new,
  legitimate claim.
- **Verdict: DISTINGUISHABLE on paper.** Proceeding to calibration only, under that bright line.

## 2. Poles-first calibration — PASSES
Fresh, PCA-conditioned read-out (`read_instrument.py`); no prior instrument reused. Synthetic conditional
expectations `E` with known structure; each maps an ordered token pair to an output; the read presents both
orders and splits it into commuting (part a) and non-commuting residue (part b). Atomic dial on the residue:
`ATOMIC iff eff_rank(residue) < 0.4·m (=8) AND held-out readability ≥ 0.3; residue≈0 → FUNGIBLE; else NOISE.`

| E (ground truth) | residue rel-norm | eff_rank(residue) | readability (b) | commuting read (a) | verdict | expected |
|---|---|---|---|---|---|---|
| **fungible** (commuting E) | 0.005 (≈0) | — | — | 1.00 | **FUNGIBLE** | FUNGIBLE ✓ |
| **atomic** (known rank-3 residue) | 2.01 | **2.63 ≈ 3** | **1.00** | 1.00 | **ATOMIC** | ATOMIC ✓ |
| **noise** (untied full-rank residue) | 0.73 | 19.79 ≈ full | 0.001 | 1.00 | **NOISE** | NOISE ✓ |
| **null** (random features on the atomic residue) | — | — | **−0.033** | — | **NOISE (valid)** | NOISE ✓ |

The instrument (a) recovers the *known* rank of an atomic residue (2.63 ≈ 3), (b) calls a commuting E fungible
(residue ≈ 0), (c) calls an untied residue noise (full-rank, unreadable), and (d) the state-independent null
cannot read even the atomic residue (held-out −0.033) — so it does **not** fabricate readability. The commuting
part (a) is trivially readable (1.00) in every case — the occupied "average" side, reported separately as
required. **Calibration passes: the instrument is trustworthy.**

## 3. Real-model boundary — STOP (untestable, not NOISE)
Per the frozen preconditions, a real-model number is interpretable only on **provenance-logged,
input-overlapping I/O**:
- **No model I/O with logged provenance is present** in this repo/session (no registry, no linked-dataset/widget
  I/O). A model with no honest linked I/O is **untestable** — reported as such, **not** as a NOISE data point. A
  documentation/data gap must not masquerade as a negative result.
- The **relative trace** (model A read against B) is defined **only on shared inputs**. With no two models'
  I/O samples in hand, there is no input overlap to read; the code would (correctly) detect and refuse a
  comparison rather than fabricate one on disjoint inputs.

So: the instrument is built fresh and **validated poles-first**, but the real-model / relative-trace test is
**not run** — the load-bearing data is absent. To proceed, supply per-model I/O with logged provenance and, for
the relative read, at least two models whose inputs overlap.

## Scope-lock
This is a **local** instrument validation: a fresh read-out that recovers known residue structure and whose null
reads NOISE. It **does not** confirm the resolvent conjecture, is **not** the auction, is **not** the
aggregator-cost claim, and the atomic residue here is a **structural rhyme** with Conjecture-1, **not** evidence
for it. No cross-thread reconciliation. Honest outcome: instrument validated; real-model call held as
**untestable** pending provenance-logged, input-overlapping I/O.

Reproduce: `python calibrate.py`.
