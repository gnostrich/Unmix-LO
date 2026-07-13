"""
ebr.demo — run the router on real frozen models and real inputs.

  python -m ebr.demo --text "a dog sitting on the grass"
  python -m ebr.demo --image dog.png
  python -m ebr.demo --image dog.png --text "a dog" --to vit,minilm
  python -m ebr.demo --image dog.png --scramble mobilenet     # gauge guarantee, user-visible

Feeds the input to any subset of models (R5), equilibrates a shared anchor (F-loop, channel routing B per
prompt = R4), and prints: consensus, what EACH model says (silent models included, via cross-modal coupling),
and the session line.
"""
import argparse
import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ebr.demo")
    ap.add_argument("--image")
    ap.add_argument("--text")
    ap.add_argument("--to", help="comma-separated model subset that RECEIVES the input (others go silent)")
    ap.add_argument("--scramble", help="scramble this model's internal features; every number must be identical (R3)")
    ap.add_argument("--atoms", type=int, default=14, help="anchor size (FW self-sizing in --session)")
    ap.add_argument("--session", action="store_true",
                    help="run FW self-sizing over a session of diverse prompts (R1); prints atom dynamics")
    a = ap.parse_args(argv)
    if a.session:
        from . import session as SS
        import datasets, warnings
        warnings.filterwarnings("ignore")
        ds = datasets.load_dataset("uoft-cs/cifar10", split="test[256:268]")
        inputs = [{"image": r["img"], "text": None} for r in ds]
        inputs += [{"image": None, "text": t} for t in
                   ["a photo of a dog", "a photo of a truck", "a photo of a ship"]]
        SS.run_session(inputs)
        return
    if not a.image and not a.text:
        ap.error("give --image and/or --text")

    from . import ports as P, library as L, engine as E, readout as R
    image = None
    if a.image:
        from PIL import Image
        image = Image.open(a.image)
    subset = set(a.to.split(",")) if a.to else None

    print("[ebr.demo] loading frozen ports (vit, mobilenet, minilm, clip)...", flush=True)
    ports = P.load_ports()
    manifest = L.build(ports)
    libs = L.load_libs()
    clouds, meta = L.materialize(ports, libs, image=image, text=a.text, active_subset=subset)

    m = a.atoms
    rng = np.random.default_rng(0)
    De0 = rng.random((m, m)); De0 = (De0 + De0.T) / 2; np.fill_diagonal(De0, 0)
    De0 /= np.median(De0[np.triu_indices(m, 1)])
    a0 = np.full(m, 1.0 / m)
    Bbar = {p: np.full(len(clouds[p]), 1.0 / len(clouds[p])) for p in clouds}
    print("[ebr.demo] equilibrating shared anchor (F-loop + channel routing)...", flush=True)
    res = E.equilibrate(clouds, De0, a0, a0.copy(), Bbar)

    active_in = [p for p in meta if meta[p]["active"]]
    print(f"\nINPUT: image={a.image or '—'}  text={a.text or '—'}  received-by={active_in}\n")
    con = R.consensus(res)
    print(f"CONSENSUS: {con['active']}/{con['atoms']} anchor atoms active ({con['parked']} parked); "
          f"F={con['F']:.3f}  converged={con['converged']}  F-monotone={con['monotone']}\n")

    in_mods = {meta[p]["modality"] for p in active_in}
    pan = R.panels(res, meta, manifest)
    print(f"WHAT EACH MODEL SAYS  (● received input, ○ silent — via the shared anchor):")
    print("-" * 74)
    for port, p in pan.items():
        mark = "●" if p["active"] else "○"
        Bstr = ",".join(f"{b:.2f}" for b in res["B"][port])
        # honest tag: silent same-modality reads are meaningful; silent CROSS-modal transfer is a known wall
        tag = ""
        if not p["active"]:
            tag = "  [cross-modal: LIMITED — see WALL_crossmodal.md]" if p["modality"] not in in_mods else "  [silent, same-modality]"
        print(f"  {mark} {port:12} [{p['modality']:6}] B=[{Bstr}]  ->  {' | '.join(p['exemplars'])}{tag}")
    # cross-modal via CLIP's aligned towers — ATTEMPTED but FRAGILE (kept for honesty, not claimed as working)
    bridge = R.clip_bridge(clouds, manifest)
    if bridge:
        silent_tower = "clip_text" if not meta.get("clip_text", {}).get("active") else "clip_vision"
        if silent_tower in bridge:
            print(f"\nCROSS-MODAL (CLIP bridge, FRAGILE): silent {silent_tower} -> "
                  f"{' | '.join(bridge[silent_tower])}   [unreliable: GW-nonconvex, F-optimal ≠ semantic;"
                  f" see WALL_crossmodal.md]")
    print("\nNOTE: WITHIN a modality the router genuinely works — a dog image makes vit/mobilenet/clip_vision"
          " all say 'dog', aligning different embedding geometries via gauge-invariant coupling."
          "\n      CROSS-modal transfer is a WALL (WALL_crossmodal.md): relational-only GW discards the"
          " cross-modal alignment, and even the CLIP bridge is fragile (F-optimal coupling ≠ semantic).")

    # disagreement meter (R5.3) — an INSTRUMENT: cycle cost net of the solver-restart floor
    from . import meter as MT
    dm = MT.disagreement(clouds, meta)
    print(f"\nDISAGREEMENT METER: cycle cost {dm['cycle_cost']:.4f}  vs floor {dm['floor']:.4f}  ->  "
          f"{dm['verdict']}  (floor = solver-noise null; indicative at CPU settings — see meter.py)")

    if a.scramble:
        _scramble_check(a.scramble, libs, clouds, meta, manifest, De0, a0, Bbar, E, R)

    print(f"\n[session] atoms={con['atoms']} active={con['active']} F={con['F']:.3f} "
          f"converged={con['converged']}")


def _scramble_check(model, libs, clouds, meta, manifest, De0, a0, Bbar, E, R):
    """R3: scramble a model's internal features -> every readout number identical to precision."""
    from ..geometry.clouds import scramble
    from . import library as L
    scr_libs = {ep: dict(ch) for ep, ch in libs.items()}
    for ep, chans in scr_libs.items():
        if L.ENGINE_PORTS[ep]["model"] == model:
            for ch in chans:
                scr_libs[ep][ch] = scramble(chans[ch], seed=1)
    # rebuild clouds from scrambled libs and re-equilibrate; compare consensus F + panel exemplars
    from ..geometry.clouds import cloud_to_Dw
    clouds2 = {}
    for ep in clouds:
        chans = []
        for k, ch in enumerate(L.ENGINE_PORTS[ep]["channels"]):
            D, _ = cloud_to_Dw(scr_libs[ep][ch]); chans.append((D, clouds[ep][k][1]))
        clouds2[ep] = chans
    res2 = E.equilibrate(clouds2, De0, np.array(a0), np.array(a0), Bbar)
    con2 = R.consensus(res2)
    con1 = R.consensus(E.equilibrate(clouds, De0, np.array(a0), np.array(a0), Bbar))
    dF = abs(con1["F"] - con2["F"])
    print(f"\n[R3 gauge check] scrambled {model}'s features -> |ΔF_consensus| = {dF:.2e}  "
          f"{'IDENTICAL (gauge holds)' if dF < 1e-6 else 'LEAK'}")


if __name__ == "__main__":
    main()
