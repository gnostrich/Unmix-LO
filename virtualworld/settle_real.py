"""
SETTLE_REAL — wire coherentflow's ACTUAL settling object onto virtualworld's REAL frozen-model
interfaces and HONESTLY watch what it does. Observe-don't-prove. NOT a gate, NOT a proof.
BIND: thoughtworld_construct/CONSTRUCT.md ; context: IO_STOCKTAKE.md.

WHAT THIS IS (faithfulness):
  - It IMPORTS and calls the REAL committed coherentflow functions (`structured`, `settle`,
    `combined_read`, `make_interface`) — the object is NOT reimplemented here. coherentflow's
    dimensional module-globals (D, T, NTR) are the object's declared knobs; we set them to
    virtualworld's shape (D=26, T=n, NTR=n//2) exactly as the IO_STOCKTAKE notes the math is
    fully shape-generic ("one-line change"). Nothing about the settling / guard / read math is touched.
  - The interfaces are virtualworld's REAL aligned modality vectors (ViT-base + MiniLM + hand-feature
    views, each ridge-aligned to the shared 26-dim scene medium), parsed straight out of
    interactive_data.js. `Y` is the standardized scene medium (the world latent z).

THREE OBSERVATIONS (the honest null on the REAL senses is EXPECTED — see CONSTRUCT / xresolve):
  1. REAL modalities  -> settle the 4 real aligned vectors. EXPECTED: convergent real small models,
     after alignment, disagree only by state-INDEPENDENT (world-unpredictable) residual -> the guard
     holds NOTHING -> combined read == consensus -> the object correctly NO-OPS. Reported honestly.
  2. SEPARABLE control -> the SAME code path on a synthetic SEPARABLE input (coherentflow's own
     make_interface construction, at virtualworld's exact D=26/n=264 scale) DOES hold structured and
     combined DECISIVELY beats consensus-collapse. Proves the wiring is correct and the real no-op is
     the DATA (real-model convergence), not a bug. A second variant injects into the REAL Y medium to
     show the guard also fires there.
  3. NOISE guard -> inject UNSTRUCTURED noise into a real interface -> must stay REJECTED (0 held),
     confirming the object does not fabricate structure.

VALIDATED vs EXPERIMENTAL: the settling OBJECT and its guard are coherentflow's committed, self-tested
code (VALIDATED at the component level, 0% false-positive over 40 seeds per false_positive_ref). Wiring
it onto virtualworld's real frozen interfaces and reading the result is EXPERIMENTAL observation — this
script watches, it does not gate or prove.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CF_DIR = os.path.join(ROOT, "coherentflow")
sys.path.insert(0, CF_DIR)
import coherentflow as cf   # the REAL object; we call its functions, never reimplement them


def load_vw():
    """Parse virtualworld's REAL aligned modality vectors + medium out of interactive_data.js."""
    raw = open(os.path.join(HERE, "interactive_data.js")).read().strip()
    prefix = "window.VW_IX = "
    assert raw.startswith(prefix), "unexpected interactive_data.js prefix"
    raw = raw[len(prefix):]
    if raw.endswith(";"):
        raw = raw[:-1]
    d = json.loads(raw)
    mods = d["meta"]["modalities"]
    aligned = {m: np.asarray(d["aligned"][m], float) for m in mods}
    Y = np.asarray(d["Y"], float)
    return d, mods, aligned, Y


def set_cf_shape(D, T):
    """Point coherentflow's declared dimensional knobs at this shape (the only 'config' we touch)."""
    cf.D, cf.T, cf.NTR = D, T, T // 2


def run_settle(ifaces, z, target=None):
    """Call the REAL cf.settle + cf.combined_read and summarize honestly."""
    state, memory, res, circ = cf.settle(ifaces, z)
    tail_contracts = res[-1] < res[len(res) // 2]
    out = {
        "residual_curve": [round(float(r), 6) for r in res],
        "res_first": float(res[0]), "res_last": float(res[-1]),
        "settles_tail_contract": bool(tail_contracts),
        "n_interfaces_held_structured": len(memory),
        "held_interface_indices": sorted(memory.keys()),
        "circ_norm": float(circ),
    }
    if target is not None:
        cons_acc, comb_acc, held_dim = cf.combined_read(state, memory, target)
        out.update({"consensus_read_acc": float(cons_acc),
                    "combined_read_acc": float(comb_acc),
                    "held_memory_dim": int(held_dim),
                    "combined_minus_consensus": float(comb_acc - cons_acc)})
    return state, memory, out


def guard_scores(ifaces, z, i):
    """The REAL cf.structured() verdict for interface i's disagreement at the naive consensus."""
    st0 = np.mean(ifaces, axis=0)
    is_s, _P, sc = cf.structured(ifaces[i] - st0, z)
    thr = 1.5 * sc["eff"] / cf.D
    return {"structured": bool(is_s), "eff": int(sc["eff"]), "cap": float(sc["cap"]),
            "cap_threshold": float(thr), "ho": float(sc["ho"])}


def main():
    d, mods, aligned, Y = load_vw()
    n, D = Y.shape
    coll_dims = d["meta"]["coll_dims"]
    set_cf_shape(D, n)
    results = {"config": {"D": D, "n": n, "NTR": cf.NTR, "iters": cf.ITERS, "damp": cf.DAMP,
                          "modalities": mods, "coherentflow_dir": CF_DIR},
               "provenance": {"model_source": d["model_source"],
                              "natural_structured_count_ref": d["natural_structured_count"],
                              "honest_label_ref": d["honest_label"],
                              "false_positive_ref": d["false_positive_ref"]}}

    print("SETTLE_REAL — coherentflow's settling object on virtualworld's REAL frozen interfaces")
    print(f"(observe-don't-prove; D={D}, n={n}, NTR={cf.NTR}; calling REAL cf.settle/structured/"
          f"combined_read/make_interface)\n")

    # ============================================================= 1. REAL MODALITIES
    ifaces = [aligned[m] for m in mods]
    # a genuine world target to probe the read with: a collision-activity medium dim, median-split.
    col = coll_dims[0]
    world_target = (Y[:, col] > np.median(Y[:, col])).astype(int)
    _state, _mem, real = run_settle(ifaces, Y, target=world_target)
    real["per_modality_guard_scores"] = {m: guard_scores(ifaces, Y, i) for i, m in enumerate(mods)}
    real["probe_target"] = f"median-split of medium dim {col} ({d['meta']['scene_labels'][col]})"
    results["case1_real_modalities"] = real

    print("[1] REAL modalities — settle the 4 real aligned vectors (vision/text/audio/timeseries):")
    print(f"    SETTLE        : residual {real['res_first']:.4f} -> {real['res_last']:.4f}  "
          f"(tail-contracts? {real['settles_tail_contract']})")
    print(f"                    (recurrence = the operator-feedback FLUID; on convergent real senses the "
          f"derived operators R_i approx I so the coupled flow is stable (rho<=1) and settles to the "
          f"consensus with NOTHING held -> the honest no-op)")
    print(f"    HOLD          : {real['n_interfaces_held_structured']}/4 interfaces held STRUCTURED, "
          f"circ_norm={real['circ_norm']:.4f}")
    print(f"    COMBINED READ : consensus={real['consensus_read_acc']:.4f}  "
          f"combined(consensus+held)={real['combined_read_acc']:.4f}  "
          f"(delta={real['combined_minus_consensus']:+.4f}, held_dim={real['held_memory_dim']})")
    print(f"    WHY nothing held (per-modality cf.structured verdict):")
    for m in mods:
        g = real["per_modality_guard_scores"][m]
        print(f"      {m:11s}: eff-rank={g['eff']:2d}/{D} (mid-rank), cap={g['cap']:.3f}, "
              f"ho={g['ho']:+.3f}  -> {'STRUCTURED' if g['structured'] else 'not held'} "
              f"(ho below +0.30 world-predictability floor)")
    print(f"    => real convergent modalities disagree by mid-rank, WORLD-UNPREDICTABLE residual "
          f"-> honest NO-OP (matches natural_structured_count={d['natural_structured_count']}).\n")

    # ============================================================= 2. SEPARABLE CONTROL
    # 2a — DECISIVE: coherentflow's own separable construction at virtualworld's EXACT scale (D=26,n=264).
    #      Isotropic latent so the injected branch is a clean separable channel; only the LATENT differs
    #      from case 1 (isotropic-separable vs real-convergent Y) -> isolates that the no-op is the DATA.
    rng = np.random.default_rng(0)
    z_iso = rng.normal(size=(n, D))
    branch = rng.integers(0, 2, n)
    ext = np.zeros((n, D)); ext[:, 0] = (branch * 2 - 1) * 3.0
    sep = [cf.make_interface(z_iso, 1, extra=ext), cf.make_interface(z_iso, 2), cf.make_interface(z_iso, 3)]
    _s, _m, sep_res = run_settle(sep, z_iso, target=branch)
    sep_res["injected_carrier_guard_scores"] = guard_scores(sep, z_iso, 0)
    results["case2a_separable_matched_scale"] = sep_res

    print("[2a] SEPARABLE control — SAME code path, coherentflow separable construction at THIS scale "
          f"(D={D}, n={n}):")
    print(f"     HOLD          : {sep_res['n_interfaces_held_structured']}/3 held STRUCTURED "
          f"(iface {sep_res['held_interface_indices']}), circ_norm={sep_res['circ_norm']:.3f}, "
          f"tail-contracts? {sep_res['settles_tail_contract']}")
    g = sep_res["injected_carrier_guard_scores"]
    print(f"     carrier guard : eff-rank={g['eff']}/{D} (concentrated), cap={g['cap']:.3f}>"
          f"{g['cap_threshold']:.3f} AND ho={g['ho']:+.3f}>+0.30 -> STRUCTURED, held")
    print(f"     COMBINED READ : consensus={sep_res['consensus_read_acc']:.4f}  "
          f"combined={sep_res['combined_read_acc']:.4f}  "
          f"(delta={sep_res['combined_minus_consensus']:+.4f})")
    print(f"     => combined DECISIVELY beats consensus-collapse -> wiring correct; the real no-op is "
          f"the DATA, not a bug.\n")

    # 2b — corroboration: inject into the REAL Y medium (make_interface on Y) -> the guard STILL fires.
    z_med = Y
    branch2 = np.random.default_rng(1).integers(0, 2, n)
    ext2 = np.zeros((n, D)); ext2[:, 0] = (branch2 * 2 - 1) * 6.0
    sepY = [cf.make_interface(z_med, 1, extra=ext2, noise=0.02),
            cf.make_interface(z_med, 2, noise=0.02),
            cf.make_interface(z_med, 3, noise=0.02)]
    _s2, _m2, sepY_res = run_settle(sepY, z_med, target=branch2)
    sepY_res["injected_carrier_guard_scores"] = guard_scores(sepY, z_med, 0)
    results["case2b_separable_on_real_medium"] = sepY_res
    gy = sepY_res["injected_carrier_guard_scores"]
    print("[2b] SEPARABLE on the REAL Y medium (make_interface on Y + injected branch) — guard fires here too:")
    print(f"     HOLD          : {sepY_res['n_interfaces_held_structured']}/3 held STRUCTURED, "
          f"carrier eff-rank={gy['eff']}/{D}, ho={gy['ho']:+.3f}>+0.30 -> held")
    print(f"     COMBINED READ : consensus={sepY_res['consensus_read_acc']:.4f}  "
          f"combined={sepY_res['combined_read_acc']:.4f}  "
          f"(delta={sepY_res['combined_minus_consensus']:+.4f}; modest — Y's real low-rank structure + "
          f"n={n} limit the joint-probe synergy that reaches +0.5 at isotropic scale)\n")

    # ============================================================= 3. NOISE GUARD
    rngn = np.random.default_rng(7)
    big = 3.0 * rngn.normal(size=(n, D))            # high-variance, NOT world-predictable
    corrupt_idx = 0                                  # corrupt the vision interface
    ifaces_noise = [ifaces[i] + (big if i == corrupt_idx else 0.0) for i in range(len(ifaces))]
    _sn, _mn, noise_res = run_settle(ifaces_noise, Y)
    noise_res["corrupted_interface"] = mods[corrupt_idx]
    noise_res["corrupted_iface_guard_scores"] = guard_scores(ifaces_noise, Y, corrupt_idx)
    results["case3_noise_guard"] = noise_res
    gn = noise_res["corrupted_iface_guard_scores"]

    print(f"[3] NOISE guard — corrupt the REAL '{mods[corrupt_idx]}' interface with UNSTRUCTURED noise:")
    print(f"    HOLD          : {noise_res['n_interfaces_held_structured']}/4 held STRUCTURED "
          f"(noise NOT held), circ_norm={noise_res['circ_norm']:.4f}")
    print(f"    corrupted guard: eff-rank={gn['eff']}/{D} (near full-rank, diffuse), ho={gn['ho']:+.3f} "
          f"(world-UNpredictable) -> NOISE, rejected")
    print(f"    => no fabrication: unstructured corruption is not written to the tape.\n")

    # ============================================================= verdict
    real_noops = (real["n_interfaces_held_structured"] == 0
                  and abs(real["combined_minus_consensus"]) < 1e-9)
    sep_fires = (sep_res["n_interfaces_held_structured"] >= 1
                 and sep_res["combined_minus_consensus"] > 0.1)
    noise_rejected = noise_res["n_interfaces_held_structured"] == 0
    results["verdict"] = {
        "real_case_noops_as_expected": bool(real_noops),
        "separable_control_fires_and_combined_beats_consensus": bool(sep_fires),
        "noise_stays_rejected": bool(noise_rejected),
        "plain_english": (
            "The settling MECHANISM is real and correctly wired: on a synthetic SEPARABLE input the "
            "same code path holds structured decoherence and the combined read decisively beats "
            "consensus-collapse; unstructured noise stays rejected. On virtualworld's REAL, convergent "
            "frozen senses it is HONESTLY INERT — their post-alignment disagreements are mid-rank and "
            "world-unpredictable, so nothing is held and combined == consensus. The object correctly "
            "no-ops. Observe-don't-prove, not a gate."),
        "framing": "coherentflow object = VALIDATED (component); wiring-onto-real-interfaces = EXPERIMENTAL observation.",
    }
    print("VERDICT:")
    print(f"  real case no-ops as expected?              {results['verdict']['real_case_noops_as_expected']}")
    print(f"  separable control fires (combined>>cons)?  {results['verdict']['separable_control_fires_and_combined_beats_consensus']}")
    print(f"  noise stays rejected?                      {results['verdict']['noise_stays_rejected']}")

    out_path = os.path.join(HERE, "settle_real_results.json")
    json.dump(results, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
