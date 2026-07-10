# PRE-REGISTRATION — reading a model's trace (conditional expectation) from I/O (frozen BEFORE run code)

One local test. Fresh, self-contained. The question: **is a single model's input→output map a finitely-readable
(atomic) conditional expectation, or does reading it need as many parameters as samples (memorization)?** And,
for a pair on shared inputs: **does the relative (non-commuting) residue between two models have atomic
structure?** Nothing here confirms any broader conjecture (see Scope-lock).

## The object — a static landscape queried by conditioning, NOT a descent
The trace is a **static landscape queried by conditioning**, not a gradient/Baur descent. Do not pattern-match
to adjoint/gradient code. The query operator is a **conditional expectation** `E : B → B'` (region-to-region),
read at the input's region. It is **NOT multiplicative** — `E(xy) ≠ E(x)E(y)` in general — and that
non-multiplicativity is **where the content lives**, not a nuisance to smooth over. Any code or metric that
implicitly assumes `E(xy)=E(x)E(y)` (e.g. treating the read as a plain order-free linear projection) has thrown
the object away.

## The read splits in two — report BOTH separately
- **(a) commuting / canonical part** — the plain low-rank read-out, the order-free "average" answer. This is the
  **cheap, occupied** side.
- **(b) non-commuting residue** — the order-dependence of conditioning (the antisymmetric / `E(xy)−E(yx)` part).
  This is the **candidate-original** content.
A result that only reports (a) has measured the occupied thing. The code must report (a) and (b) separately.

## The atomic dial — applied to the residue (b)
- residue **zero** → input is fungible → the trace is a lookup table (no order structure).
- residue **nonzero but full-rank** → noise.
- residue **nonzero and low-rank** → the real object (atomic).
This is the same atomic dial as elsewhere in the program, applied to the non-commuting residue. **It is a
STRUCTURAL RHYME with Conjecture-1, NOT evidence for it.** Do not conflate. (Guard against the recurring
grand-conflation failure.)

## The read-out is a probe — hard preconditions (learned from the swirl run)
The swirl run just proved *this class of instrument fabricates at small N* (an over-parameterized read-out
called the null ATOMIC at N=48). Therefore, as hard preconditions:
- The read is valid **only if it generalizes to held-out I/O from the same model** (held-out adjudicator).
- The **null = state-independent random features** must read NOISE under the identical pipeline. NOT a random
  projection of the raw input — that leaks the input's information and is not a valid null (swirl correction).
- **Poles-first:** verify the null reads NOISE **and** a known-readable positive control reads ATOMIC BEFORE
  trusting any real model's number.
- Build the read-out **fresh, PCA-conditioned** so the held-out adjudicator is stable. Do **not** reuse any
  prior instrument.

## I/O is an empirical sample of E itself — what "held-out" means
The I/O sample is an **empirical sample of `E` itself**, so "held-out" is **not** a train/test split of an
external task — it is **held-out observations of the model's own conditional expectation**. This is
**self-calibration**: the model's own linked data both teaches and certifies the read. No external benchmark is
needed or wanted.

## Provenance — untestable ≠ NOISE
Linked-dataset / widget I/O is self-reported and of uneven quality. **Log where each model's I/O came from.** A
model with no honest linked I/O is **untestable**, and must be reported as such — **not** as a NOISE data point.
A documentation gap must never masquerade as a negative result.

## The relative trace — needs shared inputs, defined only on the overlap
The **relative** trace (model A read against model B on shared inputs) is the pairwise content. It requires
**overlapping inputs** between the two models' I/O samples. If two models' inputs do not overlap, each can be
read absolutely but the relative read is **undefined** — the code must **detect and report** disjoint inputs,
**not** fabricate a comparison. The residue of the relative read (where A and B don't commute) is the cross-model
structure, and it is only defined on the input overlap.

## Frozen verdict (poles-first ground truth, then real models if data exists)
Calibration (synthetic E's with known structure), all under the identical PCA-conditioned pipeline:
- **null** (state-independent random features) → must read **NOISE**.
- **fungible control** (multiplicative/commuting E) → residue (b) ≈ **zero** (lookup table).
- **atomic positive control** (non-commuting E with a KNOWN low-rank residue of rank r) → residue (b) nonzero
  and **low-rank**, recovered rank ≈ r (held-out).
- **noise control** (non-commuting E with a full-rank residue) → residue (b) nonzero but **full-rank** → NOISE.
If calibration does not cleanly separate these four, the instrument is not trustworthy and NO real-model number
is reported. Real-model / relative-trace runs proceed **only** on provenance-logged, input-overlapping I/O.

## Scope-lock (mandatory in RESULTS)
This is a **local** test of: (i) can a low-rank read learned from a model's I/O generalize on held-out I/O of
that same model, and (ii) does the relative read between a pair have atomic residue on their shared inputs. It
**does not** confirm the resolvent conjecture, is **not** the auction, is **not** the aggregator-cost claim. Same
words ("trace", "atomic", "residue") across threads ≠ same object. RESULTS state the finding **locally**, with
**no** cross-thread reconciliation.

## The distinguishing test (preflight-by-hand — the load-bearing guard)
`preflight`/`registry` are absent (the registry build did not land), so the dead-composition flag is checked by
hand. The buried corpse (`report.md`, `COMPOSITION_THESIS.md`) is **combination beating parts (">")** — synergy,
capability-beyond-parts, killed across BIOMESH/synergy; *"composition is a cost-and-routing story, not a
capability story."* The surviving results are all *"="*: readability / reach / compression of already-reachable
knowledge.

- **This object is on the "=" side:** it asks whether a **single model's** conditional expectation is
  finitely-readable, and (pairwise) whether the non-commuting residue has structure. It is scored as
  **readability + residue-atomicity (structure)**, never as task-performance gain over parts.
- **Bright line (hard stop):** if any metric ever compares **"A+B performance vs best-single on a task,"** the
  run has relitigated the corpse and must STOP. "Reading the trace so models compose" (to beat parts) is the
  dead synergy claim in new vocabulary; "reading a single model's trace to test its readability" is the new,
  legitimate claim. This doc commits to the latter and forbids the former.
- **Verdict: DISTINGUISHABLE on paper** under this bright line → proceed to calibration. (Were it not
  distinguishable, the instruction is to STOP before running.)

## Discipline
Frozen PREREG committed before run code. Poles-first calibration before any real-model number. State-independent
null must read NOISE. Report (a) and (b) separately. Untestable ≠ NOISE. Relative read only on input overlap.
Honest NOISE / fungible / untestable are all valid outcomes. No cross-thread reconciliation in RESULTS.
