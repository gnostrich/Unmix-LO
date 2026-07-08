# GATE ZERO RESULTS — do DAVIS DTI queries genuinely need multiple specialists? **FAIL** (as pre-registered)

Run 2026-07-08 per PREREG.md (thresholds frozen before the run). Full numbers in
gate0_results.json. Per the BIOMESH discipline the pre-committed rule stands unmodified; this is
an honest RED, and it stops the project before the cost-vs-scale experiment.

## The two pre-committed conditions (BOTH required to PASS)

| condition | measured | threshold | result |
|---|---|---|---|
| 1 — per-instance split-knowledge fraction (balanced subset) | **0.005** | >= 0.30 | **FAIL** |
| 2 — union AUPRC / best-single AUPRC | 0.417 / 0.304 = **1.37×** | >= 1.10 | PASS |

**GATE ZERO FAILS** (condition 1 misses by ~60×). Per the frozen decision rule, do NOT proceed
to experiment/; report the negative.

## Probe performance (DAVIS, 30,056 pairs, 8.3% binders, warm stratified split)
| probe | AUROC | AUPRC |
|---|---|---|
| protein-only (ESM-2) | 0.711 | 0.209 |
| molecule-only (ChemBERTa) | 0.794 | 0.304 |
| union (concat) | 0.870 | 0.417 |

## Why it failed — the honest mechanism (decomposition, balanced subset n=752)
- P(protein correct) 0.684, P(molecule correct) 0.722, P(union correct) 0.811.
- **P(both singles wrong) = 0.051** — the two specialists are rarely wrong at the same time.
- **Single-specialist errors are ANTI-correlated (−0.18)**: when the protein view fails, the
  molecule view usually covers, and vice versa (exactly one wrong on 49% of instances).
- Of the few both-wrong instances, the union rescues only **10.5%** → split fraction 0.005.

So composition **improves aggregate discrimination** (union AUPRC 1.37× the best single, a real
and expected DTI result) but it does **not** unlock a distinct class of *only-jointly-solvable*
queries. The union's gain is ranking refinement on instances at least one specialist already got
right — not the strict complementarity the split-knowledge premise requires.

## What this means (and the honest scope of the negative)
The pre-registered "customer" — queries that no single specialist can answer but the union can —
is essentially absent on DAVIS DTI (0.5% of balanced instances, 0.1% of the natural test set).
The premise BIOMESH exists to serve (cheap composition *where composition is necessary*) does not
fire on this task, for a characterizable reason: **DTI single-specialist baselines are inflated
by marginal structure** (promiscuous drugs bind many targets; promiscuous kinases bind many
drugs), so whichever single input a probe has usually determines the binary label well enough.
This is the VIRTUALMESH-G1 lesson recurring in a new key: composition helping *on average* is not
the same as composition being *necessary per query*.

Note the tension was designed-in and honored: PREREG.md deliberately chose the warm split
(single baselines stronger = conservative for the composition claim), and the conservative choice
is exactly what made strict complementarity rare. The verdict is not moved.

## Decision
**STOP.** Do not build the cost-vs-scale experiment on the claim of a validated split-knowledge
need — that customer is not here. Recorded as a real, publishable negative.

## Candidate follow-ups (each a NEW pre-registration, not a retroactive edit of this gate)
1. **Marginal-free split-knowledge task.** A cold-drug or cold-target split (or a task where
   per-entity promiscuity is uninformative) would test strict complementarity without the
   marginal inflation that depressed condition 1 here. If strict complementarity appears there,
   re-open gate0 under that pre-registration.
2. **Re-scope the BIOMESH claim to aggregate-accuracy.** Condition 2 passed strongly; a layer
   that delivers the union's 1.37× AUPRC gain at N-independent cost is a coherent, weaker claim —
   but it must be pre-registered as "cheaper aggregate-accuracy," NOT "serving queries that need
   composition," and it must still clear the cost-vs-scale bar in experiment/. This is a scope
   decision for the owner, not something to assume.
