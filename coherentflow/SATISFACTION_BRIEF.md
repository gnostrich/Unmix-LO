# SATISFACTION BATTERY — run against the REAL coherentflow/virtualworld build (internal check, not a gate)

BIND to thoughtworld_construct/CONSTRUCT.md. This is for OUR OWN satisfaction that the SHIPPED code behaves as
believed -- run the battery on the ACTUAL committed coherentflow loop (and virtualworld where relevant), NOT a
reimplementation. Two sandbox scripts included (battery_sandbox.py, sweep_sandbox.py) show the tests + the
numbers I got on synthetic stand-in interfaces; reproduce them against the real build.

## Run these 7 checks against the REAL coherentflow settle/classify/read code
Use the actual committed functions (the real is_structured/settle/combined-read), not fresh copies.
- T1 COVERAGE-UNION > best-single: complementary modalities fused beat the best single modality. (sandbox: 0.494 vs 0.280)
- T2 COHERENT -> HONEST NO-OP: agreeing inputs -> holds nothing, circ~0, combined-read = consensus. (sandbox: held 0)
- T3 STRUCTURED -> READ BEATS CONSENSUS: injected structured decoherence recovered by combined read, not by
  consensus-collapse. (sandbox: works WHEN detected; see sweep for the detection floor)
- T4 NOISE -> REJECTED (no G1): corrupted interface -> held 0, nothing circulated, no amplification. (sandbox: held 0)
- T5 SETTLING STABLE: init away from the stitch, confirm it CONVERGES (comes down, never blows up). Use tail-slope,
  NOT first-vs-last (that's the known measurement bug). (sandbox: converges from ~30 to single digits; fixed point
  is seed-dependent, not always tight -- report that honestly)
- T6 CIRCULATION CONCENTRATED: when it holds, circulated energy concentrates in few directions (not sprayed). (sandbox: 0.77 top-dir)
- T7 PURE-NOISE FALSIFICATION: noise dressed as structure -> must NOT be held. (sandbox: held 0)

## The TWO trust-critical measurements (reproduce exactly, these are the point)
1. FALSE-POSITIVE RATE on coherent input across >=30 seeds -> MUST be ~0% (the no-fabrication guarantee).
   (sandbox: 0% across 30 seeds -- the strongest result; confirm the real build matches.)
2. DETECTION-SENSITIVITY SWEEP: detect-rate vs injection strength across >=30 seeds/strength. Report the curve.
   (sandbox: strength 1.0->0%, 2.0->43%, 3.0->93%, 4.0->97% -- a clean floor at ~2-3x noise scale.)
   ALSO report read-payoff (combined - consensus) per strength. (sandbox: real but MODEST, +0.05 near threshold,
   shrinking to +0.01 at high strength as consensus starts leaking the signal -- report honestly, don't inflate.)

## Honest reporting (do not dress up)
- The no-fabrication result (0% false positives, pure-noise-never-held) is the headline -- it's what makes the
  system trustworthy. Confirm it on the real build.
- The read-payoff is REAL but SMALL (single-digit accuracy points) and regime-dependent. Report the actual number;
  a large payoff would be SUSPICIOUS, not good.
- Settling converges/stable but not always sharply contractive (seed-dependent fixed point). Report as-is.
- If the real build's numbers DIFFER from the sandbox (e.g. real detector floor is different, or false-positives
  > 0%), that's an important finding -- report the discrepancy, don't force a match.

## Deliverable
Commit SATISFACTION.md next to the code: the 7-check table (real build), the false-positive rate, the
detection-sensitivity curve, the read-payoff-vs-strength curve, and a plain-English "what this build can and
can't be trusted to do" paragraph. Internal check -- no report.md gate row.
