"""
demo/library.py — probe libraries (each model's behavioral support, R5) and channel-aware cloud
materialization (R2).

Each engine-port carries C_v channels (feature groups where the interface decomposes; CLIP's two towers are
the cross-modal pair). A channel's library = the model's responses to natural probes: CIFAR-10 images for
vision (natural, fast HF mirror; NEVER synthetic), index-paired class-anchored captions ("a photo of a
{class}") for text. Free-form COCO captions were tried and break cross-modal alignment — see _paired_probes()
and WALL_crossmodal.md. Built once, cached.

A prompt materializes, per port-channel, a measure: the channel's library REWEIGHTED toward the input
(active model) or uniform (silent — a reference cloud). Only (D, w) crosses to geometry (R3).
"""
import os, json
import numpy as np

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
PROBE_N = 256

# engine-ports and their channels. model = which ports.py model; ch = encode channel index.
ENGINE_PORTS = {
    "vit":         {"model": "vit",       "modality": "vision", "channels": [0, 1]},   # cls, patch-mean
    "mobilenet":   {"model": "mobilenet", "modality": "vision", "channels": [0, 1]},   # penultimate, logits
    "minilm":      {"model": "minilm",    "modality": "text",   "channels": [0, 1]},   # mean, cls
    "clip_vision": {"model": "clip",      "modality": "vision", "channels": [0]},       # CLIP vision tower
    "clip_text":   {"model": "clip",      "modality": "text",   "channels": [1]},       # CLIP text tower
}


def _paired_probes(n=PROBE_N, seed=0):
    """PAIRED cross-modal probes: each CIFAR image is paired index-wise with a class-anchored natural caption
    ("a photo of a {class}"). Pairing is required for relational transfer through the anchor (the correspondence
    must be in the data, since gauge invariance discards absolute cross-modal alignment).

    LIMITATION (documented, not hidden): FREE-FORM captions (e.g. raw COCO) do NOT work here — a caption's
    off-object scene content ("...in a kitchen", "...on a boat") dominates its embedding geometry and destroys
    the class alignment that relational-only GW needs. So the text probes are class-anchored templates: natural
    sentences whose cross-modal correspondence lives in the shared class structure both modalities cluster by.
    Raw-COCO captions remain available via _coco_captions() for the (failing) comparison in REPORT."""
    import datasets
    ds = datasets.load_dataset("uoft-cs/cifar10", split=f"test[:{n}]")
    imgs = [r["img"] for r in ds]
    labels = [ds.features["label"].int2str(r["label"]) for r in ds]
    caps = [f"a photo of a {lab}" for lab in labels]
    return imgs, labels, caps


def _coco_captions(n=PROBE_N):
    import datasets
    return [r["caption1"].strip() for r in
            datasets.load_dataset("sentence-transformers/coco-captions", split=f"train[:{n}]")]


def build(ports, force=False):
    os.makedirs(CACHE, exist_ok=True)
    mpath = os.path.join(CACHE, "manifest.json")
    if os.path.exists(mpath) and not force:
        return json.load(open(mpath))
    print("[library] building channel probe libraries (first run; cached after)...", flush=True)
    vimgs, vlabels, texts = _paired_probes()   # index-paired image/caption (cross-modal bridge)
    for ep, cfg in ENGINE_PORTS.items():
        probes = vimgs if cfg["modality"] == "vision" else texts
        for ch in cfg["channels"]:
            f = ports[cfg["model"]].encode(probes, channel=ch)
            np.save(os.path.join(CACHE, f"{ep}__c{ch}.npy"), f)
        print(f"  [library] {ep} channels {cfg['channels']} encoded", flush=True)
    manifest = {"vision_labels": vlabels, "texts": texts, "n": PROBE_N}
    json.dump(manifest, open(mpath, "w"))
    return manifest


def load_libs():
    libs = {}
    for ep, cfg in ENGINE_PORTS.items():
        libs[ep] = {ch: np.load(os.path.join(CACHE, f"{ep}__c{ch}.npy")) for ch in cfg["channels"]}
    return libs


def _reweight(L, f):
    d2 = ((L - f[None, :]) ** 2).sum(1)
    temp = np.median(d2) + 1e-9
    wl = -d2 / temp
    w = np.exp(wl - wl.max()); return w / w.sum()


def materialize(ports, libs, image=None, text=None, active_subset=None):
    """Return clouds: port -> [(D_c, w_c) per channel], and meta: port -> dict(active, modality, primary_lib,
    primary_w). Only (D,w) crosses to geometry."""
    from ..geometry.clouds import cloud_to_Dw
    clouds, meta = {}, {}
    for ep, cfg in ENGINE_PORTS.items():
        mod = cfg["modality"]
        has_input = (mod == "vision" and image is not None) or (mod == "text" and text is not None)
        active = has_input and (active_subset is None or ep in active_subset
                                or cfg["model"] in active_subset)
        chans = []
        for ch in cfg["channels"]:
            L = libs[ep][ch]
            D, _ = cloud_to_Dw(L)
            if active:
                f = ports[cfg["model"]].encode([image if mod == "vision" else text], channel=ch)[0]
                w = _reweight(L, f)
            else:
                w = np.full(len(L), 1.0 / len(L))
            chans.append((D, w))
        clouds[ep] = chans
        c0 = cfg["channels"][0]
        meta[ep] = {"active": bool(active), "modality": mod,
                    "primary_lib": libs[ep][c0], "primary_w": chans[0][1]}
    return clouds, meta
