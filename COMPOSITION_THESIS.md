# The composition thesis — closed. A complete, honest ledger.

Across three projects and many pre-registered gates, one question was under test: **can you
compose independently-trained frozen models into more than the best of their parts?** The answer,
established by repeated confound-controlled negatives, is **no** — not by pooling, not by settling,
not by indexed connective tissue. What survives is real but modest: composition is *infrastructure*
(cheaper routing, compression of already-reachable knowledge), not *intelligence* (new capability).

## The three attack surfaces, all dead

| mechanism | where tested | verdict | the killing fact |
|---|---|---|---|
| **naive pooling** | BIOMESH gate0 (DAVIS DTI, real encoders) | FAIL | On the confound-controlled cold split, union is *below* best-single (0.63–0.88×). The in-distribution 1.37× "gain" was marginal-promiscuity memorization; it inverts once leakage is removed. |
| **settling / fact-passing** | VIRTUALMESH G1 (real LoRA specialists) | FAIL | Without calibrated ignorance, recurrent settling amplifies hallucination (fact-precision 0.018 at 5 rounds); it does not average toward truth. Pooling also lost to best-single. |
| **indexed connective tissue** | indextest (planted, ideal regime) | FAIL | Even a steelman blind cross-view bilinear indexer, on complementarity engineered to be maximally favorable, is *worse* than a strong naive readout (0.78–0.93×). Blind to y, a connective frame cannot surface the task-relevant cross-terms. |

The indextest is the decisive one: it removed every real-world excuse (small models, hard data,
weak encoders) by *constructing* the ideal regime — genuine complementarity, a reachable oracle, a
gauge that hides the signal from linear pooling — and indexing *still* could not beat naive. The
failure is not empirical bad luck; it is informational. The composer is always blind to the task
signal the readout already has, so it cannot add value the readout couldn't extract itself.

## What survives (validated, unchanged)

- **VIRTUALMESH G2 [PASS, scoped]** — the routing/memory kernel's cost is independent of federation
  size (kernel rank flat across N=4→10, linear regime). *Cost doesn't grow as you add models.*
- **VIRTUALMESH G3 [PASS, amended]** — transitive pathways distill into direct edges that cache the
  composite at lower inference cost, bounded exactly by the chain ceiling. *Compression of
  already-reachable knowledge — never new capability, and the fabrication guard refuses to invent
  edges where none exist.*

Both are **infrastructure**: they make an existing, reachable computation cheaper or flatter in
cost. Neither creates capability that wasn't already present in the models' reachable composition.

## The one-line conclusion
Frozen-model composition is a cost-and-routing story (G2/G3), not a capability story (G1/BIOMESH/
indextest). Every attempt to extract *new* joint capability from frozen parts — by pooling,
settling, or blind indexing — failed under confound control, because a composer blind to the task
cannot beat a readout that sees it. The honest deliverable is the negative plus the two surviving
infrastructure results.

## Discipline record (why these negatives are trustworthy)
Every threshold was pre-registered and committed before its run; two invalidated runs (NaN-poisoned
adapters; a fail-by-construction split) were caught, fixed, and kept on record; anti-hallucination
and fairness controls gated every positive-looking result; the one in-distribution "win" (BIOMESH
1.37×) was chased down and shown to be leakage. The gates did their job — they stopped three
beautiful systems from being built on premises that do not hold.
