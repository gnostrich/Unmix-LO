# mz_aggregator — self-expanding OV Mori–Zwanzig memory kernel as a dimension-independent decentralized aggregator

Playable. numpy only. Frozen predictions in `PREREG.md` (committed before run code); honest verdict + curves in
`RESULTS.md`; raw numbers in `results.json`.

## Play it
```bash
python aggregator.py --live --K 8  --diversity 4      # watch the kernel rank lock to the McMillan degree
python aggregator.py --live --K 64 --continuous       # continuous spectrum: rank drifts, never terminates
python aggregator.py --live --K 8  --diversity 8      # dial diversity up -> rank grows
python aggregator.py --hetero --K 8 --diversity 4     # heterogeneous per-worker dims -> rank unchanged
python aggregator.py --calibrate                      # poles-first ground-truth check (atomic + continuous)
python aggregator.py --sweep                          # the double dissociation
python aggregator.py --all --out results.json         # everything -> results.json
```

## The pieces
- `resolvent.py` — task distributions as resolvents `G(z)=C(zI−A)⁻¹B`; atomic (known McMillan degree r) and
  continuous-spectrum; block-Hankel + Ho-Kalman/ERA realization; ground-truth degree.
- `mz_kernel.py` — the self-expanding OV MZ kernel: append a state when the closure-residual Hankel singular
  value clears the second-FDT noise floor; balanced-truncation prune below it.
- `aggregator.py` — K decentralized workers feed closure residuals; the kernel aggregates, self-expands/prunes;
  CLI + calibration + the double-dissociation sweep.

## The result (see RESULTS.md)
Kernel rank is **FLAT in worker count K** (4,4,4,4,4,4 for K=2→64) and **GROWS with task-diversity** tracking
the McMillan degree (2,3,4,6,8,9.8 for r=2→10). Cost tracks the atomic-support size of the task distribution's
resolvent, not the worker count. Continuous-spectrum control: rank does not cleanly terminate (the atomicity
dial's negative arm). Both frozen predictions held.

## Lineage
Names `ov-ssm-stage0` / `s4c-resolvent` were absent from the repo; the block-Hankel closure primitive is reused
from `archive/pre-nuke:virtualmesh/gates/real/gate2_mzkernel.py` and the atomicity/McMillan framing from
`CONTEXT.md`. The Ho-Kalman resolvent realization is implemented from scratch. Stated plainly.
