# ebr — Equilibrium Barycentric Router (stage-0)

Implementation of EBR (authoritative spec: **`ebr-spec-v1.1.md`**). Single-authority principle: ONE functional
`F` decides everything; all else is instrument (reports/gates claims), oracle (proposes/warm-starts;
acceptance is strict F-descent), or experimental control. Optimization is block-coordinate **mirror descent on
F with a backtracking line search** guaranteeing monotone descent — the blocks are NOT exact I-projections
(FIX-3). Structural growth is **Frank–Wolfe support adaptation** on the anchor measure, driven by F alone; the
Hankel/poles are instrument, never mechanism. The corrected headline is a mechanism/instrument split: atom
count = spatial complexity of the shared geometry (K-invariant); the pole set = temporal McMillan degree of
traffic (multiplicative closure). See `ebr-spec-v1.1.md` and `LEDGER.md`.

**Hard rule (invariant interface, §0):** nothing downstream of `geometry/` consumes coordinates — only
normalized cost matrices `(D, w)` cross any boundary. This is what makes every logged quantity gauge-invariant
(G0), and it is enforced by module boundary + CI fixture.

```
ebr/
  geometry/   cloud -> (normalized D, w). ONLY module touching model outputs.
  transport/  square-loss entropic semi-relaxed GW; monotone proximal steps.
  energy/     F assembly + shared-anchor block-coordinate loop (Lyapunov-guarded).
  hankel/     residual block-Hankel (§6); reuses io_trace/stream_trace.
  events/     Frank–Wolfe support adaptation (atoms: grow/park/revive — F-driven, validated).
  router/     DeepSets warm-starter (oracle; no term in F) — scaffold.
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
