# Audit-or-wipe verdict (Part 1) — recorded before build resumes

**Decision: KEEP + DELETE.** The substrate passed real gates (gauge exact, Lyapunov monotone, G4 meter 20.4×,
P5 pole closure); wiping validated instruments would be waste. R1–R5 can be built on the kept skeleton without
shims — the gaps (R2 channels, R4 per-prompt B adaptation, R5 full readout) are UNBUILT, not mis-built.

## Audit against R1–R5 + SPIRIT
| req | status | where |
|---|---|---|
| R1 hypergraph connectors expand/contract intrinsically | atom-level FW built + validated (self-quench, K-invariant); hyperedge level derived-unimplemented | `events/frankwolfe.py` |
| R2 all-to-all channel-blocked gains B | **NOT BUILT** (B vestigial) — demo step 3 | — |
| R3 measure couplings through anchors, gauge-invariant | built + gauge-exact | `geometry/`, `transport/gw.py`, `energy/functional.py` |
| R4 channels trained/adapted per-prompt, recurrent | **NOT BUILT** — demo step 3 | — |
| R5 any-subset input, equilibrate, readout (consensus / per-model / meter) | meter built (cycle cost); consensus + per-model panels **NOT BUILT** — demo step 4 | `experiments/g4_meter.py`, `transport.barycentric_pushforward` |
| SPIRIT one authority, no coordinates, no shims | holds in kept code | CI invariants |

## KEEP (faithful, load-bearing for R1–R5 or cited proven results)
geometry/clouds.py, geometry/gram.py (FW oracle helpers relocated from hankel), transport/gw.py,
energy/functional.py, events/frankwolfe.py, registry/registry.py, tests/test_core.py (gauge/Lyapunov/
coupling/interface/gram invariants), experiments/{substrate,g4_meter,fw_selfsize,sympower,pole_closure}.py,
docs {ebr-spec-v1.1.md, LEDGER.md, README.md, PREREG_P5.md, WALL_2x2_atomleg.md}.

## DELETE (served the abandoned rank-gate mechanism / walled / superseded / stale) — deleted, not deprecated
- `hankel/` (residual-rank instrument) — its gauge-faithful helpers (gram_from_D, deflate) relocated to
  `geometry/gram.py`; the rank-spectrum functions deleted.
- `experiments/g1_probe.py` (residual-rank self-sizing, superseded by FW), `diversity_retest.py` (P2 failed),
  `grid2x2.py` (walled 2×2 atom leg), `stage0.py` (rank-based G0–G5 harness), `phase_zero.py` (rank-instrument
  floors; gauge covered by CI).
- Stale docs: `REPORT.md`, `SPEC_v1.1_NOTES.md` (both superseded by ebr-spec-v1.1 + LEDGER), `PREREG_stage0b.md`,
  `PREREG_2x2.md` (prereg for superseded/walled experiments).
- The z-scored-Hankel CI test (tested the deleted instrument) → replaced by a gauge test on `geometry/gram`.

CI green after cleanup (6 invariants). Kept modules import. No wipe.
