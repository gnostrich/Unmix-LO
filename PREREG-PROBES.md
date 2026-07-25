# PREREG-PROBES.md — EBR decision probes (measurement only)

**Session type:** measurement. Not authorized to build, fix, or refactor anything.
**Date issued / registered:** 2026-07-25
**Code under test:** `ebr/` at commit `2a28682` (origin/main).
**Working branch:** `main` (the branch the EBR codebase lives on).

This file is committed BEFORE any probe runs. Every threshold below is fixed now.
Any threshold not written here is inadmissible afterward. No tolerance will be
adjusted after seeing results (doing so = automatic session failure).

---

## Pre-registered predictions and thresholds (fill-in from the directive)

```
PROBE A predicted verdict (G / M / D):        M
PROBE A invertibility tolerance:              0.10   (relative Frobenius, defn below)
PROBE A cocycle tolerance:                    0.10   (relative Frobenius, defn below)
PROBE B predicted cold-restart separation:    12×
PROBE B "floor is real" threshold:            3×     (disjoint/clone separation must exceed this)
PROBE B predicted schedule sensitivity:       CV ≈ 0 across reachable permutations (see B note)
```

---

## Operational definitions (fixed in advance, so nothing is chosen post-hoc)

### Probe A objects
- The transport object between a member `v` and the anchor is the coupling
  `π_v` returned by `gw.equilibrate_coupling` — shape `n_v × m`
  (`n_v` = member support size, `m` = anchor atoms). It is **not square** and
  has **rank ≤ m**. No inverse of any `π` is formed anywhere in the codebase
  (confirmed by inventory; the only composite is `_self_coupling`).
- Because **no member→member transition exists**, the pairwise transition used
  in A1/A2/A4 is *synthesized through the hub* as a MEASUREMENT (not a build):
  `T_{ij} := π_j · diag(1/a) · π_iᵀ`  (shape `n_j × n_i`), the natural
  generalization of the existing `_self_coupling` (which is the `i=j` case).
  This is analysis of existing couplings; it routes nothing.
- **Identity reference for A1:** `I` on member `i`'s own support (`n_i × n_i`).

### Probe A tolerances
- **Invertibility (A1):** a pair passes iff
  `‖T_{ij} ∘ T_{ji} − I‖_F / ‖I‖_F ≤ 0.10`  (i.e. ≤ 0.10·√n_i in raw Frobenius).
  Prediction: measured values ≫ 0.10 (composition is rank ≤ m ≪ n_i, so it
  cannot approximate a full-rank identity) → **not invertible → not G**.
- **Associativity (A2):** report `‖(T∘T)∘T − T∘(T∘T)‖_F / ‖T∘T∘T‖_F` on length-3
  chains; no pass/fail bar (diagnostic), reported as a distribution.
- **Triples (A3):** count genuine triples under BOTH readings and report both:
  (a) strict — three members mutually sharing a **direct** channel (star topology
  ⇒ expected 0), and (b) hub — three members sharing the **anchor atoms** as a
  common target (expected `C(K,3) > 0`). If the strict count is 0 AND the hub
  triples are cone-degenerate (all overlaps factor through the single apex), that
  is the structural signature of **verdict D**. If hub triples support a
  non-degenerate 2-complex, the cocycle test A4 decides M vs G.
- **Cocycle (A4):** a triple passes iff
  `‖T_{ki} ∘ T_{jk} ∘ T_{ij} − I‖_F / ‖I‖_F ≤ 0.10`.
  Prediction: fails (same rank obstruction).

### Probe A verdict rule (fixed)
- **G** iff A1 passes for a majority of pairs AND A4 passes for a majority of triples.
- **D** iff A3 strict-triple count is 0 and the hub triples are cone-degenerate
  (no genuine degree-2 among members) → H¹ language is bookkeeping only.
- **M** otherwise: transitions are non-invertible transport; the closure defect
  is a distance from closure, not a cohomology class. The 20.4× stays a real
  measurement; the cohomological justification does not survive.
- Predicted: **M** (transitions are lossy transport; triples exist through the
  hub so not strictly D, but composition is non-invertible so not G).

### Probe B objects and procedure
- Signal = `g4_meter.cycle_cost` (the validated holonomy meter); separation =
  `disjoint / clone`. Warm reference = the current pipeline's separation.
- **B1 cold restart:** run cycle_cost with NO warm-start of couplings
  (`pi0=None`, the code's default) and no reuse of Sinkhorn potentials across
  runs; compare separation to a warm-started variant (couplings seeded from a
  prior solve). The cold number is the floor; warm−cold is path-debt.
- **B2 schedule permutation:** hold members, probes, wiring, init fixed; vary the
  block-coordinate update order across N≥5 permutations; report residue per
  permutation and its variance (CV = std/mean).
- **B "floor is real":** the cold-restart disjoint/clone separation must exceed
  **3×** to count as a real member-carried floor.

### Probe B note (declared in advance — a reachability limit, not a result)
Inventory shows the engine updates **all** members' couplings against a **frozen**
anchor within each sweep (engine.py:117–123) before the De/a block, so
**member-update order is provably order-independent** — permuting it is a
mathematical no-op (predicted CV ≈ 0, machine precision). The only schedule
freedom that could exhibit non-commutativity is the **block order (π→De→a)**,
which the engine does not expose as a parameter. Per the directive's stop rule,
if exercising block-order permutation requires new capability, that sub-measurement
will be reported as **"unmeasurable without building X"** — a valid result — rather
than built. B2 will therefore report: (i) measured member-order CV (expect ≈0),
and (ii) an explicit statement of whether block-order non-commutativity is
reachable without modifying the solver.

---

## Stop conditions honored (from the directive)
- Probe A returns **D** → stop; do not run B.
- Probe A returns **M** → run B anyway; flag that paper claim language needs downgrading.
- Any tempting fix → stop and report. No shims, no parallel code paths, faithful-or-wipe.
- Any probe unrunnable without new capability → say so and stop that probe.

A clean negative (A = M or D, or B = schedule-tracking) is a full-value outcome.
