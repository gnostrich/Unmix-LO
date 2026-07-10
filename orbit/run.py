"""
run.py — poles-first calibration (the kernel/omega adjudicator), then the real-corpus kappa=0 ablation +
rendered artifact + frozen verdict. Reproduce: python run.py  (writes results.json + artifact PNG/GIF).
"""
import json, argparse
import numpy as np
import physics as PH
import instrument as INS
import orbit as OR

SEED = 0


def build(name, states, bounds, k=48):
    inst = INS.build_instrument(states, bounds, k=k, seed=SEED)
    print(f"  [{name}] charts={inst['P'].shape[0]} K={inst['K']} gap_flagged={inst['gap_flagged']} "
          f"gamma={inst['gamma']:.3f} omega={inst['omega']:.3f}")
    return inst


def ablate(inst, seeds=range(6), n=200):
    ref = float(np.linalg.norm(np.diff(inst["states"], axis=0), axis=1).mean())
    out = {}
    for kappa in [0.0, 0.4]:
        coh = [OR.coherence(OR.generate(inst, n, kappa=kappa, seed=s)[0], ref) for s in seeds]
        out[f"kappa={kappa}"] = {"miss_mean": float(np.mean(coh)), "miss_std": float(np.std(coh))}
    m0, mk = out["kappa=0.0"]["miss_mean"], out["kappa=0.4"]["miss_mean"]
    out["margin"] = float((m0 - mk) / (m0 + 1e-12))
    return out


def calibrate():
    print("=" * 74); print("POLES-FIRST CALIBRATION (kernel/omega is the validated adjudicator)"); print("=" * 74)
    print("\nNULL (i.i.d. random states) -> expect no oscillatory mode (omega ~ 0):")
    iN = build("null", *PH.collect_null(2000, seed=0))
    print("\nPERIODIC (circular, period 25) -> expect omega ~ 2pi/25 = 0.251, autocorr oscillates:")
    iP = build("periodic", *PH.collect_periodic(40, 80, seed0=0, period=25))
    wtrue = 2 * np.pi / 25
    per_ok = abs(iP["omega"] - wtrue) / wtrue < 0.15
    null_ok = iN["omega"] < 0.05
    print(f"\n  periodic recovers omega ({iP['omega']:.3f} vs {wtrue:.3f}): {per_ok} ; "
          f"null has no oscillatory mode ({iN['omega']:.3f}<0.05): {null_ok}")
    passes = per_ok and null_ok
    print(f"  KERNEL CALIBRATION {'PASSES' if passes else 'FAILS'}  "
          f"(note: the spectral-gap COUNT detector is unreliable on these corpora and is not relied on)")
    return {"null_omega": iN["omega"], "periodic_omega": iP["omega"], "periodic_omega_true": wtrue,
            "periodic_recovers": bool(per_ok), "null_no_mode": bool(null_ok), "kernel_calibration_passes": bool(passes)}


def real(render=True):
    print("\n" + "=" * 74); print("REAL SUBSTRATE — kappa=0 ablation (the deciding experiment)"); print("=" * 74)
    res = {}
    corpora = {"driven_multiscale": PH.collect_driven(60, 90, seed0=0, period=30),
               "plain_billiards": PH.collect(60, 90, seed0=0)}
    insts = {}
    for name, (s, b) in corpora.items():
        inst = build(name, s, b); insts[name] = inst
        ab = ablate(inst)
        res[name] = {"omega": inst["omega"], "gamma": inst["gamma"], "gap_flagged": inst["gap_flagged"], **ab}
        print(f"    {name}: kappa=0 miss={ab['kappa=0.0']['miss_mean']:.3f}+/-{ab['kappa=0.0']['miss_std']:.3f}  "
              f"kappa=0.4 miss={ab['kappa=0.4']['miss_mean']:.3f}+/-{ab['kappa=0.4']['miss_std']:.3f}  "
              f"margin={ab['margin']:+.3f}")

    # frozen verdict, keyed on the validated adjudicator (omega) + the ablation margin vs its noise
    head = res["driven_multiscale"]
    osc = head["omega"] > 0.08
    margin = head["margin"]; noise = head["kappa=0.0"]["miss_std"] / head["kappa=0.0"]["miss_mean"]
    if not osc:
        verdict = ("(b) NULL — no state-observable phrase-scale mode on this substrate (omega~0), so the MZ "
                   "momentum has nothing oscillatory to exploit; kappa ties kappa=0 within noise. The kernel is "
                   "validated (periodic control), but mechanical physics is not music: the slow coupling isn't "
                   "observable in the instantaneous state.")
    elif margin > max(0.05, noise):
        verdict = "(a) POSITIVE — kernel-on beats kappa=0 beyond noise on continuation-miss; see artifact."
    elif margin < -noise:
        verdict = "(c) UNSTABLE — memory tilt raises churn vs kappa=0 (Basin's momentum finding, ported)."
    else:
        verdict = "(b) NULL — memory ties kappa=0 within noise."
    res["verdict"] = verdict
    print(f"\n  VERDICT: {verdict}")

    if render:
        inst = insts["driven_multiscale"]
        p0, g0 = OR.render_artifact(OR.generate(inst, 240, kappa=0.0, seed=0)[0], "artifact_kappa0.png")
        pk, gk = OR.render_artifact(OR.generate(inst, 240, kappa=0.4, seed=0)[0], "artifact_kappa04.png")
        res["artifacts"] = [p0, g0, pk, gk]
        print(f"  rendered: {p0} / {pk} (+ GIFs) — the orbit you can watch")
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no-render", action="store_true"); a = ap.parse_args()
    out = {"calibration": calibrate(), "real": real(render=not a.no_render)}
    json.dump(out, open("results.json", "w"), indent=1)
    print("\nwrote results.json")


if __name__ == "__main__":
    main()
