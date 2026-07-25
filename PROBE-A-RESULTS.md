# PROBE-A-RESULTS.md — are the transitions groupoid-like or lossy?

**Verdict: M (metric).** Predicted M (PREREG-PROBES.md @ `9b24e8e`). Measurement
only; nothing built or fixed. Code under test: `ebr/` @ origin/main. Script:
`probes/probe_a.py`. Raw numbers: `probes/probe_a_{pairs,triples,selfloop}.csv`.

Substrate: real frozen ports (vit, mobilenet, minilm, clip_vision, clip_text),
one shared anchor equilibrated over all five via the validated path
(`demo.meter._anchor` → `engine.equilibrate`), `m = 6` anchor atoms, member
support capped at 128.

## A0 — inventory (what actually transports)

- The transition object is the **entropic semi-relaxed GW coupling `π_v`** returned
  by `gw.equilibrate_coupling`. Shape **`128 × 6`** for every member — **not
  square**, **rank ≤ 6**.
- Each `π_v` couples a member to the **shared anchor**, not to another member.
  **No member→member transition is stored anywhere.** Cross-member maps had to be
  *synthesized* for measurement as `T_ij = π_j diag(1/a) π_iᵀ` (the `i≠j`
  generalization of the code's own `_self_coupling`).
- **No inverse of any `π` is formed anywhere in `ebr/`** (confirmed by inventory).

## A1 — invertibility  (tol 0.10, relative Frobenius)

| quantity | mean | min | max | within tol |
|---|---|---|---|---|
| self-loop `‖S_i − I‖` (i=j round-trip) | **0.9950** | — | — | — |
| pair round-trip `‖T_ji∘T_ij − I‖` | **0.9962** | 0.9959 | 0.9966 | **0/20** |

Even a member's **own** round-trip through the anchor is ~1.0 away from identity.
No pair is invertible; every round-trip is essentially maximally far from `I`
(a rank-6 composite cannot approximate a rank-128 identity).

## A2 — associativity (length-3 bracketing)

`‖(T∘T)∘T − T∘(T∘T)‖ / ‖·‖`: mean **6.75e-16**, max 1.04e-15 over 120 chains.
Matrix composition is associative, so bracketing is a non-test; the substantive
lossiness is the non-invertibility in A1, not a bracketing defect. Reported for
completeness per the directive.

## A3 — triple overlaps

- **Strict triples** (three members sharing a *direct* channel): **0** — the
  wiring is a star; every overlap factors through the single anchor apex.
- **Hub triples** (three members sharing the anchor atoms as common target):
  **10 = C(5,3)**.

## A4 — cocycle condition (around the 10 hub triples)

`‖T_ki∘T_jk∘T_ij − I‖`: mean **0.9961**, min 0.9961, max 0.9966. **0/10** within
tol 0.10. Composing around a triple lands ~1.0 from identity — the cocycle
condition fails as hard as invertibility does.

## Verdict rule (from prereg) and result

- Not **G**: invertibility fails 0/20 and cocycle fails 0/10 (need a majority of each).
- Not strictly **D**: hub triples do exist (10), so degree-2 is *askable* — it just
  doesn't close.
- **M**: the inter-member transitions are non-invertible transport plans; the
  closure defect is a **distance from closure, not a cohomology class**.

## What it implies for the paper's claim language

The cohomological vocabulary — H¹, "the residue is a cocycle not a coboundary,"
obstruction *class* — is **not literally licensed** by this engine. The objects
that transport between members are rank-≤6 GW couplings that are neither
invertible (A1: `‖round-trip − I‖ ≈ 0.996`, 0/20) nor cocyclic (A4: 0/10), and
the wiring is a hub-mediated star with **zero direct triple overlaps** (A3), so
there is no genuine groupoid or Čech 2-complex among the members. The measured
20.4× separation **remains a real measurement** — it is a *distance from
closure*, a metric defect between two anchor reconstructions of a member — but it
must be described as such, not as the norm of an H¹ obstruction class. Paper
claim language should be **downgraded from cohomological to metric**: "closure
defect / holonomy-like distance," not "cocycle." Per the prereg stop rule (A = M
→ run B anyway, flag downgrade), Probe B proceeds.
