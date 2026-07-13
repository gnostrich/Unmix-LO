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
  geometry/   cloud -> (normalized D, w) + gauge-faithful Gram helpers. ONLY module touching model outputs.
  transport/  square-loss entropic semi-relaxed GW; backtracked monotone mirror steps.
  energy/     F assembly + shared-anchor block-coordinate loop (Lyapunov-guarded).
  events/     Frank–Wolfe support adaptation (atoms: grow/park/revive — F-driven, validated).
  registry/   append-only ledger + preflight (frozen constants).
  experiments/ substrate + validated controls (fw_selfsize, g4_meter meter, sympower/pole_closure P5).
  tests/      CI invariants (gauge, Lyapunov, coupling-continuity, interface, gram gauge).
```

Status, honest verdicts, and corrections: see `LEDGER.md`, `AUDIT_VERDICT.md`, and `WALL_2x2_atomleg.md`.
Proven: gauge-exact interface, Lyapunov-monotone equilibration, F-driven FW self-quench + K-invariance,
G4 disagreement meter (20.4×), P5 pole closure. Open: the demo (R2/R4/R5 readout on real models).

```
python -m pytest ebr/tests -q     # invariants
```
