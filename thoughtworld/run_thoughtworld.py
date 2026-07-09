"""
THOUGHTWORLD — the precondition experiment (design frozen in PREREG.md).
Roll the engine, render (prev,cur) frames, encode with FROZEN vision fragments, fit a lightweight
alignment/readout (fragment features -> engine next-state), measure the deviation connection
A (dev ~ s_cur), and apply the pre-committed verdict:
    ATOMIC iff eff-rank(A) < 0.4*D AND held-out R^2 >= 0.3, else NOISE.
Controls: random-fragment null (must be NOISE), anti-trivial (dev != 0, not perfect match), directed-frac.
"""
import os, sys, json, time
import numpy as np
import torch
from transformers import AutoModel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as ENG

CACHE = os.path.join(HERE, "feat_cache")
os.makedirs(CACHE, exist_ok=True)
torch.set_num_threads(os.cpu_count() or 4)
SEED = 0
FRAGMENTS = ["google/vit-base-patch16-224", "facebook/dino-vitb16"]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]); IMAGENET_STD = np.array([0.229, 0.224, 0.225])


@torch.no_grad()
def encode(model_id, frames):
    path = os.path.join(CACHE, f"{model_id.replace('/', '_')}_{len(frames)}.npy")
    if os.path.exists(path):
        return np.load(path)
    model = AutoModel.from_pretrained(model_id).eval()
    out, t0 = [], time.time()
    for i in range(0, len(frames), 16):
        batch = frames[i:i + 16]
        x = (batch - IMAGENET_MEAN) / IMAGENET_STD            # (B,224,224,3)
        x = torch.tensor(x.transpose(0, 3, 1, 2), dtype=torch.float32)
        h = model(pixel_values=x).last_hidden_state           # (B, tokens, hidden)
        out.append(h.mean(1).numpy())                         # mean-pool tokens
        if (i // 16) % 10 == 0:
            print(f"    [{model_id}] {i+len(batch)}/{len(frames)} ({time.time()-t0:.0f}s)", flush=True)
    F = np.concatenate(out).astype(np.float32)
    np.save(path, F)
    return F


def ridge(X, Y, lam=1.0):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)


def connection_verdict(dev, s_cur, D):
    """Fit A on 50% of states (dev ~ s_cur), held-out R^2 on the other 50%. Match probe v2 convention."""
    n = len(dev) // 2
    A = np.linalg.lstsq(s_cur[:n], dev[:n], rcond=None)[0]     # dev ~ s_cur @ A
    sv = np.linalg.svd(A, compute_uv=False)
    eff_rank = float((sv.sum() ** 2) / (sv ** 2).sum())
    pred = s_cur[n:] @ A
    ss_res = ((dev[n:] - pred) ** 2).sum()
    ss_tot = ((dev[n:] - dev[n:].mean(0)) ** 2).sum()
    r2 = float(1 - ss_res / ss_tot)
    F = 0.5 * (A - A.T)
    directed_frac = float(np.linalg.norm(F) / (np.linalg.norm(A) + 1e-9))
    atomic = (eff_rank < 0.4 * D) and (r2 >= 0.3)
    return {"eff_rank": eff_rank, "heldout_r2": r2, "directed_frac": directed_frac,
            "atomic": bool(atomic)}


def measure(phi, s_cur, s_next, D, rng):
    """phi: fragment features. Fit readout on ALIGN split, measure dev on MEASURE split."""
    n = len(phi)
    idx = rng.permutation(n)
    al, me = idx[:int(0.4 * n)], idx[int(0.4 * n):]
    Phi = np.concatenate([phi, np.ones((n, 1))], axis=1)      # bias
    W = ridge(Phi[al], s_next[al])
    dev = Phi[me] @ W - s_next[me]
    v = connection_verdict(dev, s_cur[me], D)
    v["dev_rel_norm"] = float(np.linalg.norm(dev) / (np.linalg.norm(s_next[me]) + 1e-9))
    v["readout_pred_r2"] = float(1 - ((Phi[me] @ W - s_next[me]) ** 2).sum() /
                                 ((s_next[me] - s_next[me].mean(0)) ** 2).sum())
    return v


def main():
    rng = np.random.default_rng(SEED)
    prev, cur, nxt = ENG.collect(n_rollouts=40, T=55, seed0=0)
    D = ENG.D
    print(f"states {cur.shape}, D={D}; rendering {len(cur)} frames ...", flush=True)
    frames = np.stack([ENG.render(prev[i], cur[i]) for i in range(len(cur))])
    # anti-trivial: engine motion is real
    print(f"mean ||s_next - s_cur|| = {np.linalg.norm(nxt - cur, axis=1).mean():.4f} (nonzero => engine moves)")

    res = {"D": D, "n_states": len(cur), "atomic_threshold_effrank": 0.4 * D, "fragments": {}}
    for mid in FRAGMENTS:
        print(f"\n=== fragment: {mid} ===", flush=True)
        phi = encode(mid, frames)
        v = measure(phi, cur, nxt, D, np.random.default_rng(SEED))
        res["fragments"][mid] = v
        print(f"  readout pred R^2 (fragment predicts next-state) = {v['readout_pred_r2']:.3f}")
        print(f"  dev rel-norm ||dev||/||s_next|| = {v['dev_rel_norm']:.3f} (anti-trivial: not ~0)")
        print(f"  eff-rank(A) = {v['eff_rank']:.2f} (< {0.4*D:.0f} for atomic)")
        print(f"  held-out R^2 = {v['heldout_r2']:.3f} (>= 0.3 for atomic)")
        print(f"  directed-frac ||F||/||A|| = {v['directed_frac']:.2f}")
        print(f"  -> {'ATOMIC' if v['atomic'] else 'NOISE'}")

    # CONTROL: random fragment (matched norm) must give NOISE
    phi0 = encode(FRAGMENTS[0], frames)
    scale = np.linalg.norm(phi0, axis=1, keepdims=True).mean() / np.sqrt(phi0.shape[1])
    phi_rand = np.random.default_rng(1).normal(0, scale, size=phi0.shape).astype(np.float32)
    vr = measure(phi_rand, cur, nxt, D, np.random.default_rng(SEED))
    res["random_fragment_control"] = vr
    print(f"\n=== CONTROL: random fragment ===")
    print(f"  eff-rank={vr['eff_rank']:.2f}  held-out R^2={vr['heldout_r2']:.3f}  "
          f"-> {'ATOMIC (CONTROL FAILED - pipeline fabricates!)' if vr['atomic'] else 'NOISE (control passes)'}")

    any_atomic = any(res["fragments"][m]["atomic"] for m in FRAGMENTS)
    control_ok = not vr["atomic"]
    res["verdict"] = {"any_fragment_atomic": bool(any_atomic), "control_passes": bool(control_ok),
                      "VALID": bool(control_ok)}
    print(f"\nTHOUGHTWORLD verdict: {'ATOMIC (some fragment carries structured world-deviation)' if any_atomic and control_ok else 'NOISE (frozen models world-deviations are structureless — honest negative)'}"
          + ("" if control_ok else "  [INVALID: control fabricated]"))
    json.dump(res, open(os.path.join(HERE, "thoughtworld_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
