"""
run.py — the one local experiment: is Qwen2.5-0.5B's swirl on this engine ATOMIC or NOISE, and does any
V+ signature survive velocity-withholding? Fresh engine + fresh instrument; nothing external.

Arms (same states):
  V+ : describe(state) gives positions AND velocities  (positive control)
  V- : describe(state) gives positions ONLY            (load-bearing; LLM must infer dynamics)
Random-fragment control (random projection of the raw state) in BOTH arms — the null, must be NOISE.
readout_R2 reported FIRST. Frozen verdict in PREREG.md.
"""
import os, sys, json, time, argparse
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as ENG
import instrument as INST

CACHE = os.path.join(HERE, "feat_cache"); os.makedirs(CACHE, exist_ok=True)
torch.set_num_threads(os.cpu_count() or 4)
QWEN = "Qwen/Qwen2.5-0.5B-Instruct"
SEED = 0


def describe_Vplus(s):
    p = s[:2 * ENG.N].reshape(ENG.N, 2); v = s[2 * ENG.N:].reshape(ENG.N, 2)
    parts = ["Physics: balls in a box, gravity pulls down, elastic wall and ball collisions."]
    for i in range(ENG.N):
        parts.append(f"Ball{i+1} at ({p[i,0]:.2f},{p[i,1]:.2f}) velocity ({v[i,0]:+.2f},{v[i,1]:+.2f}).")
    parts.append("Predict the next positions and velocities after one small time step.")
    return " ".join(parts)


def describe_Vminus(s):
    p = s[:2 * ENG.N].reshape(ENG.N, 2)
    parts = ["Physics: balls in a box, gravity pulls down, elastic wall and ball collisions."]
    for i in range(ENG.N):
        parts.append(f"Ball{i+1} at ({p[i,0]:.2f},{p[i,1]:.2f}).")
    parts.append("Predict the next positions after one small time step.")
    return " ".join(parts)


@torch.no_grad()
def encode_qwen(states, describe_fn, tag):
    path = os.path.join(CACHE, f"qwen_{tag}_{len(states)}.npy")
    if os.path.exists(path):
        return np.load(path)
    tok = AutoTokenizer.from_pretrained(QWEN)
    model = AutoModel.from_pretrained(QWEN).eval()
    texts = [describe_fn(s) for s in states]
    out, t0 = [], time.time()
    for i in range(0, len(texts), 16):
        enc = tok(texts[i:i + 16], return_tensors="pt", padding=True, truncation=True, max_length=256)
        h = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).float().numpy())
        if (i // 16) % 10 == 0:
            print(f"    [{tag}] {i+16}/{len(texts)} ({time.time()-t0:.0f}s)", flush=True)
    F = np.concatenate(out).astype(np.float32); np.save(path, F); return F


def gaussian_null(n):
    """The VALID null: state-INDEPENDENT random features (carry no world information). Must come out NOISE;
    if it reads ATOMIC the instrument is fabricating. (A random linear projection of the raw state is NOT a
    valid null — it linearly preserves the near-linear physics — so the null is state-independent noise.)"""
    return np.random.default_rng(123).normal(size=(n, 256)).astype(np.float32)


def linear_baseline(states, in_dim):
    """Interpretability reference (NOT a null): the raw information available to the arm's prompt as features
    (V+ = positions+velocities; V- = positions only). Shows the linear ceiling extractable from the prompt."""
    return states[:, :in_dim].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()
    N = 48 if args.smoke else args.n

    cur, nxt = ENG.collect(n_rollouts=40, T=max(31, N // 40 + 2), seed0=0)
    cur, nxt = cur[:N], nxt[:N]
    npos = 2 * ENG.N
    print(f"states={cur.shape} D_target={ENG.D} N={N}", flush=True)
    print(f"mean ||s_next - s_cur|| = {np.linalg.norm(nxt - cur, axis=1).mean():.4f} (engine moves)\n")

    res = {"N": N, "D": ENG.D, "atomic_criterion": "eff_rank < 0.4*D (=8) AND heldout_R2 >= 0.3",
           "null_control": INST.measure(gaussian_null(N), nxt, seed=SEED), "arms": {}}

    for tag, dfn, lin_in in [("Vplus", describe_Vplus, ENG.D), ("Vminus", describe_Vminus, npos)]:
        print(f"=== Arm {tag}: Qwen ({'velocities GIVEN' if tag=='Vplus' else 'velocities WITHHELD'}) ===",
              flush=True)
        phi = encode_qwen(cur, dfn, tag)
        qv = INST.measure(phi, nxt, seed=SEED)
        lb = INST.measure(linear_baseline(cur, lin_in), nxt, seed=SEED)   # linear ceiling from prompt info
        res["arms"][tag] = {"qwen": qv, "linear_baseline": lb}

    nc = res["null_control"]
    print(f"\n{'fragment':<26}{'readout_R2':>12}{'heldout_R2':>12}{'eff_rank':>10}{'D':>4}  verdict")
    print(f"{'NULL (gaussian, valid)':<26}{nc['readout_R2']:>12.3f}{nc['heldout_R2']:>12.3f}"
          f"{nc['eff_rank']:>10.2f}{nc['D']:>4}  {'ATOMIC(!! fabricates)' if nc['atomic'] else 'NOISE (valid)'}")
    for tag in ["Vplus", "Vminus"]:
        for who in ["qwen", "linear_baseline"]:
            v = res["arms"][tag][who]
            print(f"{tag+' '+who:<26}{v['readout_R2']:>12.3f}{v['heldout_R2']:>12.3f}{v['eff_rank']:>10.2f}"
                  f"{v['D']:>4}  {'ATOMIC' if v['atomic'] else 'NOISE'}")

    vp, vm = res["arms"]["Vplus"]["qwen"], res["arms"]["Vminus"]["qwen"]
    survives = vm["atomic"]
    vp_signature = vp["atomic"]
    if survives:
        verdict = "ATOMIC — Qwen's swirl survives velocity-withholding (genuine atomic world-structure)"
    elif vp_signature and not survives:
        verdict = ("NOISE / prompt-arithmetic — an atomic signature present in V+ VANISHES in V- "
                   "(the V+ signal was reading velocities off the prompt, not world-knowledge)")
    else:
        verdict = "NOISE — no atomic signature even in V+ (swirl is structureless / uninformative)"
    res["decisive"] = {
        "null_is_noise_valid": bool(not nc["atomic"]),
        "vplus_atomic": bool(vp_signature), "vminus_atomic": bool(survives),
        "readout_R2_Vplus": vp["readout_R2"], "readout_R2_Vminus": vm["readout_R2"],
        "heldout_R2_Vplus": vp["heldout_R2"], "heldout_R2_Vminus": vm["heldout_R2"],
        "eff_rank_Vplus": vp["eff_rank"], "eff_rank_Vminus": vm["eff_rank"], "verdict": verdict}
    print(f"\nV+ -> V- : readout_R2 {vp['readout_R2']:.3f} -> {vm['readout_R2']:.3f} ; "
          f"heldout_R2 {vp['heldout_R2']:.3f} -> {vm['heldout_R2']:.3f} (atomic needs >=0.3) ; "
          f"eff_rank {vp['eff_rank']:.2f} -> {vm['eff_rank']:.2f} (atomic needs <8)")
    print(f"\nVERDICT: {verdict}")
    json.dump(res, open(os.path.join(HERE, args.out), "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
