"""
THOUGHTWORLD-2 — REQUIRED velocity-confound control (mandated by thoughtworld_construct/CONSTRUCT.md
anti-regression ledger). COMPONENT-level sharpening of TW2, NOT a construct test.

The confound: the LLM fragment was HANDED velocities in its prompt, so its structured deviation
(eff-rank 13.1 < floor, readout R^2 0.57, directed-frac 0.45) could be PROMPT-ARITHMETIC on the handed
numbers, not world-knowledge. Decisive check: measure the SAME instrument on a PURE-ARITHMETIC fragment
= a linear map on the exact numbers the LLM was handed (the raw state [pos,vel]). If the arithmetic
baseline reproduces (or exceeds) the LLM's deviation structure, the LLM adds no world-structure over
arithmetic -> confound confirmed. Also a velocity-WITHHELD arithmetic baseline (positions over 2 frames,
velocity inferable but not handed) for completeness.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "thoughtworld"))
sys.path.insert(0, HERE)
import engine as ENG
import run_thoughtworld as TWM       # SAME validated instrument (measure)
N = 1200


def main():
    prev, cur, nxt = ENG.collect(n_rollouts=40, T=55, seed0=0)
    prev, cur, nxt = prev[:N], cur[:N], nxt[:N]
    D = ENG.D
    npos = 2 * ENG.N

    variants = {
        # pure arithmetic on the HANDED numbers (pos+vel) — what the velocity-handed LLM was given
        "arith_handed_posvel": cur,
        # arithmetic on positions over TWO frames (velocity inferable, not handed) — matched to a withheld LLM
        "arith_withheld_2frame_pos": np.concatenate([prev[:, :npos], cur[:, :npos]], axis=1),
        # arithmetic on a SINGLE frame's positions only (velocity truly absent — should fail to predict)
        "arith_single_frame_pos": cur[:, :npos],
    }
    res = {"N": N, "D": D,
           "llm_handed_reference": {"eff_rank": 13.06, "heldout_r2": 0.163,
                                    "readout_pred_r2": 0.566, "directed_frac": 0.45},
           "variants": {}}
    print(f"{'variant':<28}{'readoutR2':>10}{'eff-rank':>10}{'heldoutR2':>11}{'dir-frac':>9}")
    print(f"{'LLM handed (TW2 ref)':<28}{0.566:>10.3f}{13.06:>10.2f}{0.163:>11.3f}{0.45:>9.2f}")
    for name, phi in variants.items():
        v = TWM.measure(phi.astype(np.float64), cur, nxt, D, np.random.default_rng(0))
        res["variants"][name] = v
        print(f"{name:<28}{v['readout_pred_r2']:>10.3f}{v['eff_rank']:>10.2f}"
              f"{v['heldout_r2']:>11.3f}{v['directed_frac']:>9.2f}")

    ah = res["variants"]["arith_handed_posvel"]
    # confound confirmed if pure arithmetic on the handed numbers matches/exceeds the LLM's structure
    confound = (ah["eff_rank"] <= 13.06 + 1.0) and (ah["readout_pred_r2"] >= 0.566 - 0.1)
    res["confound_confirmed"] = bool(confound)
    print(f"\nArithmetic-on-handed-numbers eff-rank {ah['eff_rank']:.2f} vs LLM 13.06; "
          f"readout R^2 {ah['readout_pred_r2']:.3f} vs LLM 0.566.")
    print("CONFOUND CONFIRMED: the LLM's structured deviation is reproduced by pure arithmetic on the "
          "handed numbers -> it is prompt-arithmetic, NOT world-knowledge."
          if confound else
          "LLM structure EXCEEDS arithmetic baseline -> a residual world-structure claim would survive; "
          "run the velocity-withheld LLM re-encode to probe further.")
    json.dump(res, open(os.path.join(HERE, "velocity_control_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
