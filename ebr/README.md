# ebr — Equilibrium Barycentric Router (stage-0)

Implementation of EBR-v1 (`ebr-spec-v1`). One variational principle: a single functional `F` over transport
couplings, a shared anchor's geometry/masses, and channel gains; block-coordinate I-projections minimize it;
`F` is the Lyapunov function. The falsifiable headline is *self-sizing*: the active anchor count should equal
the McMillan degree of the traffic, read as the above-noise-floor rank of a residual block-Hankel spectrum.

**Hard rule (invariant interface, §0):** nothing downstream of `geometry/` consumes coordinates — only
normalized cost matrices `(D, w)` cross any boundary. This is what makes every logged quantity gauge-invariant
(G0), and it is enforced by module boundary + CI fixture.

```
ebr/
  geometry/   cloud -> (normalized D, w). ONLY module touching model outputs.
  transport/  square-loss entropic semi-relaxed GW; monotone proximal steps.
  energy/     F assembly + shared-anchor block-coordinate loop (Lyapunov-guarded).
  hankel/     residual block-Hankel (§6); reuses io_trace/stream_trace.
  events/     growth pressure / park / revive / spawn / merge (scaffold).
  router/     DeepSets amortizer + implicit diff (scaffold).
  registry/   append-only ledger + preflight (frozen constants, §10).
  experiments/ substrate (known-degree traffic) + stage-0 harness + G1 probe.
  tests/      5 CI invariants (gauge, Lyapunov, coupling-continuity, interface).
```

Status and honest gate verdicts: see `REPORT.md`. Short version — G0 PASS, Lyapunov PASS, G2 PASS,
G1 K-invariance PASS but diversity leg FAIL (diagnosed interface collapse, pre-registered fix = pre-logits
tap), G3/G4/G5 scaffolded.

```
python -m pytest ebr/tests -q     # invariants
python -m ebr.phase_zero          # G0
```
