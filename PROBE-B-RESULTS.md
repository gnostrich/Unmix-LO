# PROBE-B-RESULTS.md — member-carried holonomy or solver-schedule artifact?

**Verdict: member-carried on every reachable axis; one axis (block order)
unmeasurable without building.** Measurement only; nothing built or fixed.
Code under test: `ebr/` @ origin/main (validated G4 path,
`experiments/g4_meter`). Script: `probes/probe_b.py`. Raw:
`probes/probe_b_{cold,perms}.csv`. Substrate: g4's own synthetic
clone-vs-disjoint models (the substrate the 20.4× was validated on).

## B1 — cold floor

The validated `g4_meter.cycle_cost` path calls `EN.equilibrate(...)` with **no
`pis0`** → it is **already cold** (no warm-starting of couplings or Sinkhorn
potentials anywhere in it). So the separation it reports *is* the cold number.

| substrate seed | disjoint | clone | separation |
|---|---|---|---|
| 0 (canonical) | 1.1407 | 0.0561 | **20.4×** |
| 1 | 1.1538 | 0.0952 | 12.1× |
| 2 | 1.3477 | 0.0866 | 15.6× |
| 3 | 1.5257 | 0.0241 | 63.3× |
| 4 | 1.2030 | 0.0284 | 42.4× |

- **Cold separation: median 20.4×, mean 30.7×, min 12.1×, max 63.3×.**
- Pre-registered "floor is real" threshold = 3×. Cold median 20.4× → **PASS** by a
  wide margin at every seed (min 12.1× ≫ 3×).
- The canonical seed reproduces the validated **20.4×** exactly.

**Warm-vs-cold path-debt** (warm variant seeded from a prior solve via the
existing `pis0` arg): cold 1.1407 vs warm 1.1354 → |Δ| = 0.0054 = **0.5% of
cold**. Warm-starting does not inflate the residue; the signal is not path-debt.

## B2 — schedule permutation

| member update order | disjoint residue |
|---|---|
| (0,1,2) | 1.140734 |
| (0,2,1) | 1.140734 |
| (1,0,2) | 1.140734 |
| (1,2,0) | 1.140734 |
| (2,0,1) | 1.140734 |
| (2,1,0) | 1.140734 |

- **Member-order residue: mean 1.140734, std 2.22e-16, CV 1.95e-16** — invariant
  to machine precision. The residue is **not** a member-order artifact.
- This is expected structurally: within each solver sweep, *all* members'
  couplings are updated against a **frozen** anchor (engine/functional) before the
  De/a block, so member-update order cannot matter. The measurement confirms it.
- **Block order (π → De/a)** is the only axis on which the block-coordinate solver
  could exhibit non-commutativity. It is **forced by data dependency** (the De and
  a updates consume π) and is **not exposed as a solver parameter**. Permuting it
  requires a re-parameterized solver — **new capability, out of scope**. Per the
  directive stop rule this is reported as **unmeasurable without building X**, not
  built. Note that because the schedule is essentially *forced*, there is no
  alternative admissible schedule to compare against; the block-order
  artifact question cannot be settled by permutation in the existing code.

## B3 — joint verdict

- Cold floor > 3× : **True** (median 20.4×, min 12.1×).
- Schedule-stable on the reachable axis (member-order CV < 0.10) : **True** (CV 1.95e-16).
- Warm-start is not the cause : **True** (0.5% path-debt).
- Block-order non-commutativity : **NOT TESTED** — schedule forced/unexposed;
  would need a build.

**Conclusion.** The separation is a real, robust, cold-floor measurement (median
20.4×, ≥12× at every seed), and it is excluded from being a *member-order* or a
*warm-start* artifact. It is member-carried with respect to every schedule axis
the existing code lets us vary. The single residual caveat — whether it is an
artifact of the solver's *forced block order* — is genuinely unmeasurable without
building a re-parameterized solver, which this session is not authorized to do.
That caveat is a real limit on the certification, and is reported rather than
papered over.
