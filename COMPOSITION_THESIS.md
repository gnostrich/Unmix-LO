# The composition thesis — closed. A complete, honest ledger.

Across three projects and many pre-registered gates, one question was under test: **can you
compose independently-trained frozen models into more than the best of their parts?** The answer,
established by repeated confound-controlled negatives, is **no** — not by pooling, not by settling,
not by indexed connective tissue. What survives is real but modest: composition is *infrastructure*
(cheaper routing, compression of already-reachable knowledge), not *intelligence* (new capability).

## The four attack surfaces, all dead

| mechanism | where tested | verdict | the killing fact |
|---|---|---|---|
| **naive pooling** | BIOMESH gate0 (DAVIS DTI, real encoders) | FAIL | On the confound-controlled cold split, union is *below* best-single (0.63–0.88×). The in-distribution 1.37× "gain" was marginal-promiscuity memorization; it inverts once leakage is removed. |
| **settling / fact-passing** | VIRTUALMESH G1 (real LoRA specialists) | FAIL | Without calibrated ignorance, recurrent settling amplifies hallucination (fact-precision 0.018 at 5 rounds); it does not average toward truth. Pooling also lost to best-single. |
| **indexed connective tissue** | indextest (planted, ideal regime) | FAIL | Even a steelman blind cross-view bilinear indexer, on complementarity engineered to be maximally favorable, is *worse* than a strong naive readout (0.78–0.93×). Blind to y, a connective frame cannot surface the task-relevant cross-terms. |
| **task-aware aggregation** | synergy P1 (DAVIS DTI + D-SCRIPT PPI, cold) | FAIL | Fails at the *precondition*: complementarity does not survive the strictest cold split on either task (DAVIS cold-pair gap +0.02; PPI +0.05, both ≪ 0.15). Even a genuinely combinatorial task (PPI) is marginal-dominated (hub-ness). No complementarity to aggregate, cheaply or otherwise. |

Two of these are decisive from opposite ends. The **indextest** removed every real-world excuse by
*constructing* the ideal regime — genuine complementarity, a reachable oracle, a gauge that hides the
signal from linear pooling — and indexing *still* could not beat naive; the failure is informational,
not empirical. The **synergy P1** test comes from the other side: on *real* biomedical tasks under
honest cold splits, the complementarity the indexer would need does not even exist — entity marginals
(drug promiscuity, protein hub-ness) dominate, and the joint-beyond-marginal signal is < 0.05 balanced
accuracy. Constructed-but-uninhabitable and real-but-absent: the band is empty from both directions.
The composer is always blind to the task
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
Frozen-model composition is a cost-and-routing story (G2/G3), not a capability story
(G1/BIOMESH/indextest/synergy). Every attempt to extract *new* joint capability from frozen parts —
by pooling, settling, blind indexing, or task-aware aggregation — failed under confound control:
where the regime is constructible the composer cannot inhabit it (blind to the task, it cannot beat
a readout that sees it), and where the task is real the complementarity is not there to begin with
(marginals dominate under honest cold splits). The honest deliverable is the negative plus the two
surviving infrastructure results.

## Discipline record (why these negatives are trustworthy)
Every threshold was pre-registered and committed before its run; two invalidated runs (NaN-poisoned
adapters; a fail-by-construction split) were caught, fixed, and kept on record; anti-hallucination
and fairness controls gated every positive-looking result; the one in-distribution "win" (BIOMESH
1.37×) was chased down and shown to be leakage. The gates did their job — they stopped three
beautiful systems from being built on premises that do not hold.
