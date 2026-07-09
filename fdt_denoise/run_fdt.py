"""
STEP 1-3 -- real frozen-model disagreement FDT measurement (gated by the PASSED STEP 0).

STEP 1: roll the physics engine (a real generator), keep per-rollout time-ordered trajectories.
        Encode each state with real frozen models of different typings:
          ViT  (google/vit-base-patch16-224) on (prev,cur) renders,
          DINO (facebook/dino-vitb16)        on the same renders   [cross-ARCH control pair vs ViT],
          Qwen (Qwen2.5-0.5B-Instruct)       on text descriptions  [cross-MODAL pair vs vision].
        Ridge-align each model's features to the engine state s_t (train on ALIGN rollouts).
        Disagreement trajectory  d_t = predA(s_t) - predB(s_t)  on held-out MEASURE rollouts.
STEP 2/3: run the VALIDATED multivariate 2nd-FDT estimator on the per-rollout d_t trajectories;
        report the FDT-satisfying variance fraction (F_gauge/F_noise) per model pair.

Controls: RANDOM-MODEL (two variants) as the fabrication guard; cross-arch vs cross-modal comparison.
NOTE (confound, per CONSTRUCT): Qwen is HANDED positions+velocities in the prompt, so its state readout
is near-exact -> the cross-modal disagreement ~ the vision model's error vs ground truth. Reported honestly.
"""
import os, sys, json, time
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "thoughtworld"))
import engine as ENG
from fdt_estimator import fdt_fraction

CACHE = os.path.join(HERE, "feat_cache"); os.makedirs(CACHE, exist_ok=True)
torch.set_num_threads(os.cpu_count() or 4)
SEED = 0
R, T = 30, 40                 # rollouts x steps  (~1170 states; CPU-bounded)
ALIGN_FRAC = 0.5              # first half of rollouts train the ridge alignment
K_MODES = 10                  # whitened modes for the FDT estimator
VIT, DINO, QWEN = "google/vit-base-patch16-224", "facebook/dino-vitb16", "Qwen/Qwen2.5-0.5B-Instruct"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]); IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def build_data():
    """Per-rollout time-ordered states; flat frames/descriptions with a rollout-id index."""
    states = [ENG.rollout(SEED + r, T) for r in range(R)]        # list of (T,20)
    rid, tgt, prevs, curs, descs = [], [], [], [], []
    for r in range(R):
        for t in range(1, T):
            rid.append(r); tgt.append(states[r][t])
            prevs.append(states[r][t - 1]); curs.append(states[r][t])
            descs.append(describe(states[r][t]))
    return (np.array(rid), np.array(tgt, np.float32),
            np.array(prevs, np.float32), np.array(curs, np.float32), descs)


def describe(s):
    p = s[:2 * ENG.N].reshape(ENG.N, 2); v = s[2 * ENG.N:].reshape(ENG.N, 2)
    parts = ["Physics: balls in a box, gravity pulls down, elastic wall and ball collisions."]
    for i in range(ENG.N):
        parts.append(f"Ball{i+1} at ({p[i,0]:.2f},{p[i,1]:.2f}) velocity ({v[i,0]:+.2f},{v[i,1]:+.2f}).")
    parts.append("Describe the current scene.")
    return " ".join(parts)


@torch.no_grad()
def encode_vision(model_id, prevs, curs):
    path = os.path.join(CACHE, f"{model_id.replace('/','_')}_{len(curs)}.npy")
    if os.path.exists(path):
        return np.load(path)
    model = AutoModel.from_pretrained(model_id).eval()
    out, t0 = [], time.time()
    for i in range(0, len(curs), 16):
        frames = np.stack([ENG.render(prevs[j], curs[j]) for j in range(i, min(i + 16, len(curs)))])
        x = (frames - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.tensor(x.transpose(0, 3, 1, 2), dtype=torch.float32)
        h = model(pixel_values=x).last_hidden_state
        out.append(h.mean(1).numpy())
        if (i // 16) % 15 == 0:
            print(f"    [{model_id}] {i}/{len(curs)} ({time.time()-t0:.0f}s)", flush=True)
    F = np.concatenate(out).astype(np.float32); np.save(path, F); return F


@torch.no_grad()
def encode_llm(descs):
    path = os.path.join(CACHE, f"qwen_{len(descs)}.npy")
    if os.path.exists(path):
        return np.load(path)
    tok = AutoTokenizer.from_pretrained(QWEN); model = AutoModel.from_pretrained(QWEN).eval()
    out, t0 = [], time.time()
    for i in range(0, len(descs), 16):
        enc = tok(descs[i:i + 16], return_tensors="pt", padding=True, truncation=True, max_length=256)
        h = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).float().numpy())
        if (i // 16) % 15 == 0:
            print(f"    [Qwen] {i}/{len(descs)} ({time.time()-t0:.0f}s)", flush=True)
    F = np.concatenate(out).astype(np.float32); np.save(path, F); return F


def ridge_align(phi, tgt, train_mask, lam=10.0):
    """Fit phi->state on train rows; return predictions for ALL rows and train R^2."""
    Phi = np.concatenate([phi, np.ones((len(phi), 1))], 1)
    A = Phi[train_mask]; Y = tgt[train_mask]
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Y)
    pred = Phi @ W
    te = ~train_mask
    r2 = float(1 - ((pred[te] - tgt[te]) ** 2).sum() / (((tgt[te] - tgt[te].mean(0)) ** 2).sum() + 1e-9))
    return pred, r2


def to_rollouts(vec, rid, measure_rollouts):
    """Split a flat (N,D) prediction/disagreement into per-rollout time-ordered trajectories."""
    return [vec[rid == r] for r in measure_rollouts]


def fdt_on_pair(predA, predB, rid, measure_rollouts):
    d = predA - predB
    rolls = to_rollouts(d, rid, measure_rollouts)
    res = fdt_fraction(rolls, k=K_MODES)
    res["disagree_rel_norm"] = float(np.linalg.norm(d) / (np.linalg.norm(predA) + 1e-9))
    return res


def main():
    rng = np.random.default_rng(SEED)
    rid, tgt, prevs, curs, descs = build_data()
    n = len(rid)
    train_mask = rid < int(ALIGN_FRAC * R)
    measure_rollouts = [r for r in range(R) if r >= int(ALIGN_FRAC * R)]
    print(f"states={n}, D={ENG.D}, rollouts={R}x{T}, align<{int(ALIGN_FRAC*R)}, measure={len(measure_rollouts)} rollouts", flush=True)
    # anti-trivial: engine moves
    step_disp = np.mean([np.linalg.norm(curs[i] - prevs[i]) for i in range(n)])
    print(f"mean ||s_cur - s_prev|| = {step_disp:.4f} (engine moves)")

    feats = {}
    print("encoding ViT ..."); feats["ViT"] = encode_vision(VIT, prevs, curs)
    print("encoding DINO ..."); feats["DINO"] = encode_vision(DINO, prevs, curs)
    print("encoding Qwen ..."); feats["Qwen"] = encode_llm(descs)

    preds, r2 = {}, {}
    for name, phi in feats.items():
        preds[name], r2[name] = ridge_align(phi, tgt, train_mask)
        print(f"  {name}: state-readout held-out R^2 = {r2[name]:+.3f}", flush=True)

    # random-model controls (both variants), built from the ViT slot
    scale = np.linalg.norm(feats["ViT"], axis=1, keepdims=True).mean() / np.sqrt(feats["ViT"].shape[1])
    rand_feat = rng.normal(0, scale, size=feats["ViT"].shape).astype(np.float32)
    pred_randfeat, _ = ridge_align(rand_feat, tgt, train_mask)           # literal: random FEATURES (ridge collapses to ~mean)
    # matched-noise PREDICTION: white noise matched to a real model's prediction covariance (true fabrication guard)
    cov = np.cov(preds["DINO"].T); L = np.linalg.cholesky(cov + 1e-6 * np.eye(cov.shape[0]))
    pred_randnoise = (rng.normal(size=preds["DINO"].shape) @ L.T + preds["DINO"].mean(0)).astype(np.float32)

    pairs = {
        "ViT-DINO (cross-ARCH)":   ("ViT", "DINO"),
        "ViT-Qwen (cross-MODAL)":  ("ViT", "Qwen"),
        "DINO-Qwen (cross-MODAL)": ("DINO", "Qwen"),
    }
    results = {"config": {"R": R, "T": T, "K_MODES": K_MODES, "align_frac": ALIGN_FRAC},
               "readout_r2": r2, "pairs": {}, "controls": {}}
    print("\n=== FDT-satisfying fraction per model pair ===")
    for label, (a, b) in pairs.items():
        res = fdt_on_pair(preds[a], preds[b], rid, measure_rollouts)
        results["pairs"][label] = {k: float(v) for k, v in res.items() if np.isscalar(v)}
        print(f"  {label:26s}: FDT_frac={res['frac_soft']:.3f} (hard={res['frac_hard']:.2f})  "
              f"<|mu|>={res['mean_absmu']:.3f} rev={res['mean_rev']:.3f} curr={res['current_frac']:.3f} "
              f"|d|rel={res['disagree_rel_norm']:.3f}")

    print("\n=== CONTROLS (random-model fabrication guard) ===")
    ctrls = {
        "ViT-vs-RANDOMfeat (literal)":     fdt_on_pair(preds["ViT"], pred_randfeat, rid, measure_rollouts),
        "DINO-vs-RANDOMnoise (matched)":   fdt_on_pair(preds["DINO"], pred_randnoise, rid, measure_rollouts),
    }
    for label, res in ctrls.items():
        results["controls"][label] = {k: float(v) for k, v in res.items() if np.isscalar(v)}
        print(f"  {label:32s}: FDT_frac={res['frac_soft']:.3f} (hard={res['frac_hard']:.2f})  "
              f"<|mu|>={res['mean_absmu']:.3f} rev={res['mean_rev']:.3f} curr={res['current_frac']:.3f}")

    json.dump(results, open(os.path.join(HERE, "fdt_results.json"), "w"), indent=1)
    print("\nwrote fdt_results.json")


if __name__ == "__main__":
    main()
