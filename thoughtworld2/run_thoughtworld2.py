"""
THOUGHTWORLD-2 — do fragments that ACTUALLY MODEL DYNAMICS carry atomic deviation, or is even their
world-deviation NOISE ~ the ViT floor? Reuses THOUGHTWORLD's engine + validated instrument UNCHANGED
(imports engine, measure, connection_verdict); only the fragments change.

Fragments:
  F2  LLM (Qwen2.5-0.5B-Instruct): text description of the scene (positions+velocities) -> last-hidden
      representation -> ridge readout -> next state. The "language extends the seed" fragment.
  F1  Video model (VideoMAE-base, trained on video dynamics): a 16-frame history clip -> representation
      -> ridge readout -> next state.
Report readout R^2 FIRST per fragment (readout R^2 <= 0 => no dynamics, NOISE uninformative like ViTs).
Verdict: ATOMIC iff eff-rank < 8 AND held-out R^2 >= 0.3 AND eff-rank < 12 with a real gap to the
per-fragment random control. Else NOISE. ViT floor row (eff-rank ~16.4) included for direct comparison.
"""
import os, sys, json, time
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
TW = os.path.join(HERE, "..", "thoughtworld")
sys.path.insert(0, TW)
import engine as ENG                          # SAME engine
import run_thoughtworld as TWM                # SAME instrument: measure(), connection_verdict()

CACHE = os.path.join(HERE, "feat_cache"); os.makedirs(CACHE, exist_ok=True)
torch.set_num_threads(os.cpu_count() or 4)
SEED = 0
N_LLM = 1200           # states for the LLM fragment (CPU-bounded)
N_VIDEO = 500          # states for the video fragment
VIT_FLOOR = 16.4        # established noise floor (THOUGHTWORLD ViT-base/DINO)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]); IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def describe(s):
    """Text description of a state (positions + velocities) — the LLM's view of the scene."""
    p = s[:2 * ENG.N].reshape(ENG.N, 2); v = s[2 * ENG.N:].reshape(ENG.N, 2)
    parts = ["Physics: balls in a box, gravity pulls down, elastic wall and ball collisions."]
    for i in range(ENG.N):
        parts.append(f"Ball{i+1} at ({p[i,0]:.2f},{p[i,1]:.2f}) velocity ({v[i,0]:+.2f},{v[i,1]:+.2f}).")
    parts.append("Predict the next positions and velocities after one small time step.")
    return " ".join(parts)


@torch.no_grad()
def encode_llm(states):
    path = os.path.join(CACHE, f"qwen_{len(states)}.npy")
    if os.path.exists(path):
        return np.load(path)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model = AutoModel.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct").eval()
    texts = [describe(s) for s in states]
    out, t0 = [], time.time()
    for i in range(0, len(texts), 16):
        enc = tok(texts[i:i + 16], return_tensors="pt", padding=True, truncation=True, max_length=256)
        h = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).float().numpy())
        if (i // 16) % 10 == 0:
            print(f"    [Qwen] {i+16}/{len(texts)} ({time.time()-t0:.0f}s)", flush=True)
    F = np.concatenate(out).astype(np.float32); np.save(path, F); return F


@torch.no_grad()
def encode_video(states_idx, all_prev, all_cur):
    """16-frame history clip ending at each state -> VideoMAE representation."""
    path = os.path.join(CACHE, f"videomae_{len(states_idx)}.npy")
    if os.path.exists(path):
        return np.load(path)
    model = AutoModel.from_pretrained("MCG-NJU/videomae-base").eval()
    out, t0 = [], time.time()
    for k, t in enumerate(states_idx):
        # build 16 frames: render the states leading up to t (clamp at 0), single-channel current pos
        idxs = [max(0, t - 15 + j) for j in range(16)]
        clip = np.stack([ENG.render(all_prev[j], all_cur[j]) for j in idxs])   # (16,224,224,3)
        x = (clip - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.tensor(x.transpose(0, 3, 1, 2)[None], dtype=torch.float32)    # (1,16,3,224,224)
        h = model(pixel_values=x).last_hidden_state                            # (1, tokens, 768)
        out.append(h.mean(1).numpy())
        if k % 40 == 0:
            print(f"    [VideoMAE] {k}/{len(states_idx)} ({time.time()-t0:.0f}s)", flush=True)
    F = np.concatenate(out).astype(np.float32); np.save(path, F); return F


def eval_fragment(name, phi, s_cur, s_next, D):
    v = TWM.measure(phi, s_cur, s_next, D, np.random.default_rng(SEED))
    # per-fragment random control (matched norm)
    scale = np.linalg.norm(phi, axis=1, keepdims=True).mean() / np.sqrt(phi.shape[1])
    phi_rand = np.random.default_rng(1).normal(0, scale, size=phi.shape).astype(np.float32)
    vr = TWM.measure(phi_rand, s_cur, s_next, D, np.random.default_rng(SEED))
    atomic = (v["eff_rank"] < 0.4 * D) and (v["heldout_r2"] >= 0.3) and \
             (v["eff_rank"] < 12) and (vr["eff_rank"] - v["eff_rank"] > 1.5)
    return {**v, "control_eff_rank": vr["eff_rank"], "control_heldout_r2": vr["heldout_r2"],
            "atomic": bool(atomic)}


def main():
    D = ENG.D
    prev, cur, nxt = ENG.collect(n_rollouts=40, T=55, seed0=0)
    print(f"states {cur.shape}, D={D}, ViT noise floor eff-rank ~{VIT_FLOOR}\n")
    res = {"D": D, "vit_floor_eff_rank": VIT_FLOOR, "fragments": {}}

    # F2 — LLM
    print("=== F2: Qwen2.5-0.5B-Instruct (language dynamics) ===", flush=True)
    phi_llm = encode_llm(cur[:N_LLM])
    res["fragments"]["F2_qwen_llm"] = eval_fragment("Qwen", phi_llm, cur[:N_LLM], nxt[:N_LLM], D)

    # F1 — Video
    print("\n=== F1: VideoMAE-base (video dynamics) ===", flush=True)
    idx = list(range(N_VIDEO))
    phi_vid = encode_video(idx, prev, cur)
    res["fragments"]["F1_videomae"] = eval_fragment("VideoMAE", phi_vid, cur[:N_VIDEO], nxt[:N_VIDEO], D)

    print(f"\n{'fragment':<20}{'readoutR2':>10}{'eff-rank':>10}{'heldoutR2':>11}{'ctrl-eff':>9}{'dir-frac':>9}  verdict")
    print(f"{'ViT floor (ref)':<20}{'-0.10':>10}{VIT_FLOOR:>10.1f}{'0.44':>11}{'16.9':>9}{'0.20':>9}  NOISE")
    for k, v in res["fragments"].items():
        print(f"{k:<20}{v['readout_pred_r2']:>10.3f}{v['eff_rank']:>10.2f}{v['heldout_r2']:>11.3f}"
              f"{v['control_eff_rank']:>9.2f}{v['directed_frac']:>9.2f}  {'ATOMIC' if v['atomic'] else 'NOISE'}")

    any_atomic = any(v["atomic"] for v in res["fragments"].values())
    res["verdict"] = {"any_dynamics_fragment_atomic": bool(any_atomic),
                      "generalized_negative": bool(not any_atomic)}
    print(f"\nTHOUGHTWORLD-2 verdict: "
          + ("ATOMIC found — some dynamics-fragment carries world-structure (see table)"
             if any_atomic else
             "NOISE — even dynamics-trained frozen models deviate structurelessly ~ the ViT floor. "
             "GENERALIZED NEGATIVE."))
    json.dump(res, open(os.path.join(HERE, "thoughtworld2_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
