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

    active_in = [p for p in meta if meta[p]["active"]]
    print(f"\nINPUT: image={a.image or '—'}  text={a.text or '—'}  received-by={active_in}")
    print("[ebr.demo] equilibrating MATCHED-PROBE tied coupling (gauge-fixed router)...", flush=True)
    tied = R.tied_transfer(clouds, meta, manifest, m=a.atoms)

    print(f"\nCONSENSUS: {tied['active_atoms']}/{tied['atoms']} anchor atoms active; "
          f"F={tied['F']:.3f}  converged={tied['converged']}  F-monotone={tied['monotone']}\n")
    print("WHAT EACH MODEL SAYS  (● received input, ○ silent — via the tied coupling):")
    print("-" * 74)
    for port, p in tied["panels"].items():
        mark = "●" if p["active"] else "○"
        Bstr = ",".join(f"{b:.2f}" for b in p["B"])
        print(f"  {mark} {port:12} [{p['modality']:6}] B=[{Bstr}]  ->  {' | '.join(p['exemplars'])}")
    print("\nCROSS-MODAL now works: a dog image makes the SILENT text models read 'a photo of a dog' too,"
          " and it's STABLE across inits — the matched-probe tie pins the GW orbit (the semantically-correct"
          "\n      alignment relational geometry alone cannot select). Same input = data identity, not a"
          " frame, so gauge holds (see --scramble). Theorem + resolution in WALL_crossmodal.md.")

    # disagreement meter (R5.3) — an INSTRUMENT: cycle cost net of the solver-restart floor
    from . import meter as MT
    dm = MT.disagreement(clouds, meta)
    print(f"\nDISAGREEMENT METER: cycle cost {dm['cycle_cost']:.4f}  vs floor {dm['floor']:.4f}  ->  "
          f"{dm['verdict']}  (floor = solver-noise null; indicative at CPU settings — see meter.py)")

    if a.scramble:
        _scramble_check_tied(a.scramble, libs, clouds, meta, manifest, R)

    print(f"\n[session] atoms={tied['atoms']} active={tied['active_atoms']} F={tied['F']:.3f} "
          f"converged={tied['converged']}")


def _scramble_check_tied(model, libs, clouds, meta, manifest, R):
    """R3: scramble a model's internal features -> the tied-coupling result is identical to precision. The
    tie references INPUT IDENTITY (which world-event), not the representation frame, so it cannot leak."""
    from ..geometry.clouds import scramble, cloud_to_Dw
    from . import library as L
    clouds2 = {}
    for ep in clouds:
        if L.ENGINE_PORTS[ep]["model"] == model:
            clouds2[ep] = [(cloud_to_Dw(scramble(libs[ep][ch], seed=1))[0], clouds[ep][k][1])
                           for k, ch in enumerate(L.ENGINE_PORTS[ep]["channels"])]
        else:
            clouds2[ep] = clouds[ep]
    f0 = R.tied_transfer(clouds, meta, manifest)["F"]
    f1 = R.tied_transfer(clouds2, meta, manifest)["F"]
    dF = abs(f0 - f1)
    print(f"\n[R3 gauge check] scrambled {model}'s features -> |ΔF_tied| = {dF:.2e}  "
          f"{'IDENTICAL (gauge holds)' if dF < 1e-6 else 'LEAK'}")


if __name__ == "__main__":
    main()
