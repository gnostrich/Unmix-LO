"""
demo/library.py — probe libraries (each model's behavioral support, R5) and cloud materialization.

Each port has a fixed library of feature points (its responses to natural probes: CIFAR-10 images for vision,
COCO captions for text — natural, from the fast HF mirror; NEVER synthetic for vision; captions chosen so the
text library is semantically aligned with the vision probes). Built once, cached.

A prompt materializes one measure per port: the port's library REWEIGHTED toward the input (active model) or
left uniform (silent model — a reference cloud). Only (D, w) crosses to geometry: D = normalized pairwise
distances within the library (fixed per port); w = the reweighting. This is R5's "what each model says as a
reweighting of its own support."
"""
import os, json
import numpy as np

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
PROBE_N = 256


def _vision_probes(n=PROBE_N):
    import datasets
    ds = datasets.load_dataset("uoft-cs/cifar10", split=f"test[:{n}]")
    return [r["img"] for r in ds], [ds.features["label"].int2str(r["label"]) for r in ds]


def _text_probes(n=PROBE_N):
    """Natural image captions (COCO) — object/scene descriptions, semantically aligned with the CIFAR vision
    probes so the cross-modal panel is meaningful (a dog photo surfaces dog-adjacent sentences)."""
    import datasets
    ds = datasets.load_dataset("sentence-transformers/coco-captions", split=f"train[:{n}]")
    return [r["caption1"].strip() for r in ds]


# which library (vision/text) each port reads, and the channel used to encode it
PORT_LIB = {"vit": ("vision", 0), "mobilenet": ("vision", 0), "minilm": ("text", 0),
            "clip": ("both", None)}   # clip builds both


def build(ports, force=False):
    """Encode each port's probes -> cached feature libraries + a manifest (probe provenance for exemplars)."""
    os.makedirs(CACHE, exist_ok=True)
    manifest_path = os.path.join(CACHE, "manifest.json")
    if os.path.exists(manifest_path) and not force:
        return json.load(open(manifest_path))
    print("[library] building probe libraries (first run; cached after)...", flush=True)
    vimgs, vlabels = _vision_probes()
    texts = _text_probes()
    manifest = {"vision_labels": vlabels, "texts": texts, "n": PROBE_N, "libs": {}}
    for name, port in ports.items():
        if name == "clip":
            fv = port.encode(vimgs, channel=0); ft = port.encode(texts, channel=1)
            np.save(os.path.join(CACHE, "clip__vision.npy"), fv)
            np.save(os.path.join(CACHE, "clip__text.npy"), ft)
            manifest["libs"]["clip__vision"] = "vision"; manifest["libs"]["clip__text"] = "text"
        else:
            lib, ch = PORT_LIB[name]
            probes = vimgs if lib == "vision" else texts
            f = port.encode(probes, channel=ch)
            np.save(os.path.join(CACHE, f"{name}.npy"), f)
            manifest["libs"][name] = lib
        print(f"  [library] {name} encoded", flush=True)
    json.dump(manifest, open(manifest_path, "w"))
    return manifest


def load(manifest):
    return {key: np.load(os.path.join(CACHE, f"{key}.npy")) for key in manifest["libs"]}


def _reweight(L, f, temp_scale=1.0):
    """Softmax of the library toward the input feature f (the input reweights its own support)."""
    d2 = ((L - f[None, :]) ** 2).sum(1)
    temp = temp_scale * (np.median(d2) + 1e-9)
    wlog = -d2 / temp
    w = np.exp(wlog - wlog.max()); return w / w.sum()


def materialize(ports, libs, manifest, image=None, text=None):
    """Per (port-channel) key -> dict(D, w, active, lib_kind). D from geometry/clouds (only (D,w) crosses)."""
    from ..geometry.clouds import cloud_to_Dw
    out = {}
    feats = {}   # cache input features per (port, channel)
    for key, lib in libs.items():
        kind = manifest["libs"][key]
        name = key.split("__")[0]
        port = ports[name]
        D, _ = cloud_to_Dw(lib)                      # fixed library geometry (gauge-normalized)
        active = (kind == "vision" and image is not None) or (kind == "text" and text is not None)
        if active:
            ch = 1 if key == "clip__text" else (0 if key == "clip__vision" else PORT_LIB[name][1])
            inp = [image] if kind == "vision" else [text]
            f = port.encode(inp, channel=ch)[0]
            w = _reweight(lib, f)
        else:
            w = np.full(len(lib), 1.0 / len(lib))    # silent: uniform reference cloud
        out[key] = {"D": D, "w": w, "active": bool(active), "lib_kind": kind}
    return out
