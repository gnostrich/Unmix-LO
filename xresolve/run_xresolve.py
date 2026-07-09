"""
CROSS-MODEL AMBIGUITY RESOLUTION — the real-models precondition test (design frozen in xresolve/PREREG.md).
COMPONENT test bound to thoughtworld_construct/CONSTRUCT.md; NOT the construct.

Question: do two real frozen models with different typings have INDEPENDENT ambiguities (B distinguishes
what A aliases -> cross-model resolution real) or COINCIDING ambiguities (Platonic convergence -> toy-bound)?
NULL = COINCIDING (expected, given the session's convergence evidence). A positive gets the hardest scrutiny.

Measure (prereg): among GROUND-TRUTH-different pairs (different CIFAR classes), find A-aliased pairs
(bottom-5% of A's cosine distances), and score the fraction that B distinguishes (B-distance above B's own
MEDIAN over different-class pairs -> capacity-normalized, baseline = 0.5 by construction). score >> 0.5 =>
B RESOLVES A's aliasing (INDEPENDENT); score <= ~0.5 => COINCIDING/null. Symmetrized, both directions.
"""
import os, sys, json
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "feat_cache"); os.makedirs(CACHE, exist_ok=True)
torch.set_num_threads(os.cpu_count() or 4)
SEED = 0
N_IMG = 2000
N_PAIRS = 150000
ALIAS_FRAC = 0.05
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]); IMAGENET_STD = np.array([0.229, 0.224, 0.225])
VISION = {"vit": "google/vit-base-patch16-224", "dino": "facebook/dino-vitb16",
          "resnet": "microsoft/resnet-18"}


def load_images():
    ds = load_dataset("uoft-cs/cifar10", split=f"test[:{N_IMG}]")
    names = ds.features["label"].names
    imgs = np.stack([np.array(im.convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
                     for im in ds["img"]])
    labels = np.array(ds["label"])
    return imgs, labels, names


@torch.no_grad()
def encode_vision(key, imgs):
    path = os.path.join(CACHE, f"{key}_{len(imgs)}.npy")
    if os.path.exists(path):
        return np.load(path)
    model = AutoModel.from_pretrained(VISION[key]).eval()
    out = []
    for i in range(0, len(imgs), 16):
        x = (imgs[i:i + 16] - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.tensor(x.transpose(0, 3, 1, 2), dtype=torch.float32)
        o = model(pixel_values=x)
        h = o.last_hidden_state
        feat = h.mean(1) if h.dim() == 3 else h.mean(dim=(2, 3))   # ViT/DINO tokens vs ResNet spatial map
        out.append(feat.reshape(len(x), -1).float().numpy())
        if (i // 16) % 20 == 0:
            print(f"    [{key}] {i}/{len(imgs)}", flush=True)
    F = np.concatenate(out).astype(np.float32); np.save(path, F); return F


@torch.no_grad()
def encode_text(labels, names):
    path = os.path.join(CACHE, f"text_{len(labels)}.npy")
    if os.path.exists(path):
        return np.load(path)
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").eval()
    texts = [f"a photo of a {names[l]}" for l in labels]
    out = []
    for i in range(0, len(texts), 64):
        enc = tok(texts[i:i + 64], return_tensors="pt", padding=True, truncation=True)
        h = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).float().numpy())
    F = np.concatenate(out).astype(np.float32); np.save(path, F); return F


def norm(F):
    return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)


def cos_dist(F, ii, jj):
    return 1.0 - (F[ii] * F[jj]).sum(1)


def resolve(A, B, ii, jj):
    """fraction of A-aliased (bottom-5% A-dist) different-class pairs that B distinguishes (> B median).
    baseline = 0.5 (B's median split). Also continuous B-percentile of A-aliased pairs."""
    dA, dB = cos_dist(A, ii, jj), cos_dist(B, ii, jj)
    k = int(ALIAS_FRAC * len(ii))
    aliased = np.argsort(dA)[:k]                    # A maps these closest (aliased)
    bmed = np.median(dB)
    score = float((dB[aliased] > bmed).mean())      # fraction B distinguishes
    # continuous: mean rank-percentile of A-aliased pairs within B's distance distribution
    order = np.argsort(np.argsort(dB)) / len(dB)
    b_percentile = float(order[aliased].mean())     # 0.5 = random; >0.5 = B distinguishes them more
    corr = float(np.corrcoef(dA, dB)[0, 1])         # A-B distance correlation (high => coinciding)
    return {"n_aliased": int(k), "resolution_score": score, "baseline": 0.5,
            "b_percentile_of_aliased": b_percentile, "AB_distance_corr": corr}


def main():
    imgs, labels, names = load_images()
    print(f"{N_IMG} CIFAR images, {len(set(labels))} classes", flush=True)
    reps = {k: norm(encode_vision(k, imgs)) for k in VISION}
    reps["text"] = norm(encode_text(labels, names))
    print("encoded:", {k: reps[k].shape[1] for k in reps}, flush=True)

    rng = np.random.default_rng(SEED)
    ii = rng.integers(0, N_IMG, N_PAIRS); jj = rng.integers(0, N_IMG, N_PAIRS)
    keep = (ii != jj) & (labels[ii] != labels[jj])   # GROUND-TRUTH-DISTINCT: different classes
    ii, jj = ii[keep], jj[keep]
    print(f"{len(ii)} ground-truth-different (different-class) pairs\n", flush=True)

    pairs = [("vit", "text"), ("text", "vit"),           # Pair 1: cross-modality
             ("vit", "dino"), ("dino", "vit"),           # Pair 2: cross-architecture (transformer)
             ("vit", "resnet"), ("resnet", "vit"),       #          cross-architecture (transformer vs CNN)
             ("dino", "resnet"), ("resnet", "dino")]
    res = {"n_pairs": int(len(ii)), "results": {}}
    print(f"{'A -> B (does B resolve A-aliased?)':<34}{'score':>8}{'base':>7}{'B%ile':>8}{'A-Bcorr':>9}  verdict")
    for A, B in pairs:
        r = resolve(reps[A], reps[B], ii, jj)
        res["results"][f"{A}->{B}"] = r
        indep = r["resolution_score"] > 0.60 and r["b_percentile_of_aliased"] > 0.58
        tag = "INDEPENDENT?" if indep else "coinciding/null"
        print(f"{A+' -> '+B:<34}{r['resolution_score']:>8.3f}{r['baseline']:>7.2f}"
              f"{r['b_percentile_of_aliased']:>8.3f}{r['AB_distance_corr']:>9.3f}  {tag}"
              + ("  [caption=label confound]" if 'text' in (A, B) else ""))

    # verdict: INDEPENDENT only if a NON-confounded (vision-vision) direction has score >> baseline
    vv = {k: v for k, v in res["results"].items() if "text" not in k}
    positive = any(v["resolution_score"] > 0.60 and v["b_percentile_of_aliased"] > 0.58 for v in vv.values())
    res["verdict"] = {"independent_positive": bool(positive),
                      "coinciding_null": bool(not positive),
                      "note": "text arms carry the caption=label confound; verdict rests on vision-vision"}
    print(f"\nVERDICT: {'INDEPENDENT (surprising positive — apply hardest scrutiny)' if positive else 'COINCIDING (null, expected) — cross-model resolution is toy-bound on real frozen encoders'}")
    print("  (single-model paraconsistency dissociation +0.059 remains banked regardless.)")
    json.dump(res, open(os.path.join(HERE, "xresolve_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
