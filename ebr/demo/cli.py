"""
ebr.demo — run the router on real frozen models and real inputs.

  python -m ebr.demo --text "a photo of a dog"
  python -m ebr.demo --image path/to/pic.jpg
  python -m ebr.demo --image pic.jpg --text "a dog" --to vit,minilm
  python -m ebr.demo --text "a cat" --scramble mobilenet     # gauge guarantee, user-visible

STEP 2 (this file): loads the models, materializes each port's cloud for the input, and prints the clouds'
gauge invariants + each model's top library exemplar. The F-loop / channel adaptation / FW / full readout
arrive in steps 3–4.
"""
import argparse
import numpy as np


def _entropy_bits(w):
    p = w[w > 0]
    return float(-(p * np.log2(p)).sum())


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ebr.demo")
    ap.add_argument("--image", help="path to an image file")
    ap.add_argument("--text", help="a sentence")
    ap.add_argument("--to", help="comma-separated model subset that RECEIVES the input (others go silent)")
    ap.add_argument("--scramble", help="scramble this model's internal features; every printed number must be identical (R3)")
    a = ap.parse_args(argv)
    if not a.image and not a.text:
        ap.error("give --image and/or --text")

    from . import ports as P
    from . import library as L
    from ..geometry.clouds import cloud_to_Dw, scramble

    image = None
    if a.image:
        from PIL import Image
        image = Image.open(a.image)
    text = a.text
    active_subset = set(a.to.split(",")) if a.to else None

    print("[ebr.demo] loading frozen ports (vit, mobilenet, minilm, clip)...", flush=True)
    ports = P.load_ports()
    manifest = L.build(ports)
    libs = L.load(manifest)
    clouds = L.materialize(ports, libs, manifest, image=image, text=text)

    print(f"\nINPUT: image={a.image or '—'}  text={a.text or '—'}"
          f"  active-subset={a.to or 'all applicable'}\n")
    header = f"{'port':14} {'modality':8} {'state':7} {'n':>4} {'D~med':>6} {'w-entropy':>9}  top exemplar"
    print(header); print("-" * len(header))
    for key, c in clouds.items():
        name = key.split("__")[0]
        # honor --to: a model not in the subset is forced silent (uniform w)
        active = c["active"] and (active_subset is None or name in active_subset)
        w = c["w"] if active else np.full(len(c["w"]), 1.0 / len(c["w"]))
        Dmed = float(np.median(c["D"][np.triu_indices(c["D"].shape[0], 1)]))
        top = ""
        if active:
            i = int(np.argmax(w))
            top = (manifest["texts"][i][:40] if c["lib_kind"] == "text"
                   else manifest["vision_labels"][i])
        print(f"{key:14} {c['lib_kind']:8} {'ACTIVE' if active else 'silent':7} "
              f"{len(w):>4} {Dmed:>6.2f} {_entropy_bits(w):>9.2f}  {top}")

    if a.scramble:
        # R3 user-visible guarantee: scramble the model's internal features -> D unchanged to precision
        key = next((k for k in libs if k.split('__')[0] == a.scramble), None)
        if key is None:
            print(f"\n[scramble] no such model '{a.scramble}'")
        else:
            D0, _ = cloud_to_Dw(libs[key])
            D1, _ = cloud_to_Dw(scramble(libs[key], seed=1))
            print(f"\n[R3 gauge check] scrambled {a.scramble}'s features (orthogonal×perm×scale×shift): "
                  f"max |ΔD| = {float(np.abs(D0 - D1).max()):.2e}  -> "
                  f"{'IDENTICAL (gauge holds)' if np.abs(D0 - D1).max() < 1e-9 else 'LEAK'}")

    print("\n[step 2] clouds materialized. F-loop + channel adaptation + FW + full readout: steps 3–4.")


if __name__ == "__main__":
    main()
