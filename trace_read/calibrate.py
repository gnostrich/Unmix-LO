"""
calibrate.py — POLES-FIRST calibration of the trace read-out, BEFORE any real-model number.

Synthetic conditional expectations E with KNOWN structure. Each E maps an ORDERED token pair (x,y) to an
output o(x,y) in R^m. The read-out sees the input token embeddings (its view of the I/O), presents both
orders, and splits the output into commuting (part a) and residue (part b). Ground truth:

  FUNGIBLE : commuting E (output symmetric in the pair)          -> residue ~ 0        (lookup table)
  ATOMIC   : non-commuting, residue is a KNOWN rank-r linear map -> residue low-rank + readable  (the object)
  NOISE    : non-commuting, residue is per-pair random (untied)  -> residue full-rank + unreadable
  NULL     : read the ATOMIC residue from STATE-INDEPENDENT random features -> must fail (NOISE)

The instrument is trustworthy iff it returns FUNGIBLE / ATOMIC(rank~r) / NOISE / NULL=NOISE respectively.
If it does not cleanly separate these, NO real-model number may be reported (per PREREG).
"""
import json
import numpy as np
import read_instrument as R

T = 40          # tokens
d = 16          # token embedding dim (the read-out's view of the input)
m = 20          # output dim
R_ATOMIC = 3    # known rank of the atomic residue (ground truth)
NPAIRS = 1500
OBS_NOISE = 0.01


def build(kind, seed=0):
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(T, d))                       # token embeddings (input representation)
    Ws = rng.normal(size=(m, d)) / np.sqrt(d)           # commuting (order-free) map
    Gr = (rng.normal(size=(m, R_ATOMIC)) @ rng.normal(size=(R_ATOMIC, d))) / np.sqrt(d)  # rank-r residue map
    Gfull = rng.normal(size=(m, d)) / np.sqrt(d)
    pair_noise = {}                                     # per-unordered-pair random residue (untied to tokens)

    def out(x, y):
        c = Ws @ (emb[x] + emb[y])                      # commuting part
        if kind == "fungible":
            rho = np.zeros(m)
        elif kind == "atomic":
            rho = Gr @ (emb[x] - emb[y])                # rank-r, antisymmetric, readable
        elif kind == "noise":
            key = (min(x, y), max(x, y))
            if key not in pair_noise:
                pair_noise[key] = np.random.default_rng(hash(key) % (2**32)).normal(size=m)
            rho = pair_noise[key] * (1.0 if x < y else -1.0)   # full-rank, antisym, untied to embeddings
        else:
            rho = np.zeros(m)
        return c + rho + OBS_NOISE * rng.normal(size=m)

    return emb, out


def run_kind(kind, seed=0):
    emb, out = build(kind, seed)
    rng = np.random.default_rng(100 + seed)
    pairs = []
    while len(pairs) < NPAIRS:
        x, y = rng.integers(0, T, 2)
        if x != y:
            pairs.append((int(x), int(y)))
    o_xy = np.array([out(x, y) for x, y in pairs])
    o_yx = np.array([out(y, x) for x, y in pairs])
    commuting = 0.5 * (o_xy + o_yx)
    residue = 0.5 * (o_xy - o_yx)

    sig = np.stack([emb[x] - emb[y] for x, y in pairs])     # order-signed feature (reads the residue)
    sym = np.stack([emb[x] + emb[y] for x, y in pairs])     # order-free feature (reads the commuting part)
    rand_feat = np.random.default_rng(7).normal(size=(len(pairs), 32))  # state-independent null features

    res_rel = float(np.linalg.norm(residue) / (np.linalg.norm(commuting) + 1e-12))
    eff_res = R.eff_rank(residue)
    _, read_b = R.fit_read(sig, residue, seed=seed)         # readability of the residue (part b)
    _, read_a = R.fit_read(sym, commuting, seed=seed)       # the commuting/occupied read (part a)
    _, read_null = R.fit_read(rand_feat, residue, seed=seed)  # null: random features on the residue
    verdict = R.classify(res_rel, eff_res, read_b, m)
    return {"kind": kind, "residue_rel_norm": round(res_rel, 4), "eff_rank_residue": round(eff_res, 2),
            "readability_residue_b": round(read_b, 3), "commuting_read_a": round(read_a, 3),
            "null_read": round(read_null, 3), "verdict": verdict}


def main():
    print("=" * 78)
    print(f"POLES-FIRST CALIBRATION (m={m}, atomic residue rank r={R_ATOMIC}, {NPAIRS} pairs)")
    print("=" * 78)
    expect = {"fungible": "FUNGIBLE", "atomic": "ATOMIC", "noise": "NOISE"}
    rows = {k: run_kind(k, seed=0) for k in ["fungible", "atomic", "noise"]}
    print(f"\n{'E kind':<10}{'resid_norm':>12}{'eff_rank_b':>12}{'read_b':>9}{'read_a':>9}{'null':>8}  verdict  (expect)")
    ok = True
    for k, v in rows.items():
        got = v["verdict"]; exp = expect[k]; good = got == exp; ok = ok and good
        print(f"{k:<10}{v['residue_rel_norm']:>12}{v['eff_rank_residue']:>12}{v['readability_residue_b']:>9}"
              f"{v['commuting_read_a']:>9}{v['null_read']:>8}  {got:<9}({exp}) {'ok' if good else 'MISMATCH'}")
    # null guard: the state-independent null must fail to read even the ATOMIC residue
    null_atomic = rows["atomic"]["null_read"]
    null_valid = null_atomic < R.ATOMIC_READ_R2
    print(f"\nnull guard: reading the ATOMIC residue from state-independent random features -> "
          f"held-out R^2 {null_atomic} (< {R.ATOMIC_READ_R2} required) : {'NOISE (valid)' if null_valid else 'FABRICATES (invalid)'}")
    calibrated = ok and null_valid
    print(f"\nCALIBRATION {'PASSES' if calibrated else 'FAILS'} — "
          + ("instrument separates fungible / atomic / noise and the null reads NOISE. "
             "It is trustworthy; a real-model number would be interpretable IF provenance-logged, "
             "input-overlapping I/O existed." if calibrated else
             "instrument does NOT cleanly separate the controls; no real-model number may be reported."))
    out = {"config": {"T": T, "d": d, "m": m, "atomic_rank": R_ATOMIC, "npairs": NPAIRS},
           "rows": rows, "null_read_on_atomic": null_atomic, "null_valid": bool(null_valid),
           "calibration_passes": bool(calibrated)}
    json.dump(out, open("calibration_results.json", "w"), indent=1)
    print("\nwrote calibration_results.json")


if __name__ == "__main__":
    main()
