"""
G2-real — MZ memory kernel on settling dynamics over REAL model representation spaces
(thresholds in ../REAL_PREREG.md).

Nodes = (specialist adapter or base, layer) hidden-state spaces: each node embeds every probe
input as the last-token hidden state at its layer, PCA'd to d dims. Channels between nodes are
ridge maps fitted on shared probes (the frame). Settling = damped coupled iteration
  x_i <- a x_i + (1-a) mean_j W_(j->i) x_j + b u_i
run to T on held-out probes — REAL geometry, sandbox protocol.

Measured (pre-registered): closure rel-error vs memory length L (pass: some L<=8 < 0.15);
kernel eff-rank vs routed width K (must grow) and vs federation size N (must stay flat +/-2
as N goes 4 -> 10); residual-vs-difficulty correlation (reported, not gating).
"""
import os, json, itertools, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("VM_BASE", "Qwen/Qwen2.5-0.5B-Instruct")
RELS = ["p2c", "c2co", "co2pr", "p2h"]
LAYERS = [10, 16, 22]
D_NODE = 40          # per-node PCA dim
N_PROBE = 240
T_SETTLE = 40
DAMP = 0.55
torch.set_num_threads(os.cpu_count() or 4)


def make_probes(world, n=N_PROBE):
    import random
    rng = random.Random(5)
    probes = []
    pools = [("In which city does {} live?", world["persons"]),
             ("Which company is based in {}?", world["cities"]),
             ("What product does {} make?", world["companies"]),
             ("What hobby does {} practice?", world["persons"]),
             ("Tell me about {}.", world["persons"] + world["cities"] + world["companies"])]
    while len(probes) < n:
        t, pool = rng.choice(pools)
        probes.append("Question: " + t.format(rng.choice(pool)) + " Answer:")
    return probes


@torch.no_grad()
def embed_all(model, tok, probes, layers):
    """Return {layer: (n_probe, hidden)} last-token hidden states."""
    outs = {L: [] for L in layers}
    model.eval()
    for p in probes:
        enc = tok(p, return_tensors="pt")
        hs = model(**enc, output_hidden_states=True).hidden_states
        for L in layers:
            outs[L].append(hs[L][0, -1].float().numpy())
    return {L: np.array(v) for L, v in outs.items()}


def pca(X, d):
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt[:d].T
    return Z / (Z.std() + 1e-9)


def ridge(X, Y, lam=1e-2):
    W = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)
    s = np.linalg.svd(W, compute_uv=False)[0]
    return W / max(1.0, s / 0.95)          # spectral cap => contraction, settling converges


def settle_traj(nodes_u, W, T=T_SETTLE):
    """nodes_u: list of (nq, d) inputs per node. Returns traj (T+1, n_nodes, nq, d)."""
    n = len(nodes_u)
    x = [u.copy() for u in nodes_u]
    traj = [np.stack(x)]
    for _ in range(T):
        xn = []
        for i in range(n):
            coupled = np.mean([x[j] @ W[(j, i)] for j in range(n) if j != i], axis=0)
            xn.append(DAMP * x[i] + (1 - DAMP) * coupled + 0.1 * nodes_u[i])
        x = xn
        traj.append(np.stack(x))
    return np.array(traj)


def mz_closure(tracked_traj, L):
    """tracked_traj: (T+1, K, nq, d) -> lstsq closure over L-history. Returns (err, effrank)."""
    T1, K, nq, d = tracked_traj.shape
    flat = tracked_traj.transpose(2, 0, 1, 3).reshape(nq, T1, K * d)   # (nq, T+1, Kd)
    Xs, Ys = [], []
    for q in range(nq):
        for t in range(L, T1 - 1):
            Xs.append(flat[q, t - L:t].reshape(-1)); Ys.append(flat[q, t])
    Xs, Ys = np.array(Xs), np.array(Ys)
    Kmat, *_ = np.linalg.lstsq(Xs, Ys, rcond=None)
    pred = Xs @ Kmat
    err = np.linalg.norm(pred - Ys) / (np.linalg.norm(Ys) + 1e-9)
    sv = np.linalg.svd(Kmat, compute_uv=False)
    effrank = int((sv > 0.01 * sv[0]).sum())
    resid_per_q = np.linalg.norm((pred - Ys).reshape(nq, -1), axis=1)
    return err, effrank, Kmat, resid_per_q


def main():
    world = json.load(open(os.path.join(HERE, "world.json")))
    tok = AutoTokenizer.from_pretrained(BASE)
    probes = make_probes(world)
    t0 = time.time()

    # build node bank: (model_name, layer) -> embeddings
    bank = {}
    base = AutoModelForCausalLM.from_pretrained(BASE)
    for L, X in embed_all(base, tok, probes, LAYERS).items():
        bank[("base", L)] = pca(X, D_NODE)
    print(f"base embedded ({time.time()-t0:.0f}s)", flush=True)
    for rel in RELS:
        m = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(BASE),
                                      os.path.join(HERE, "adapters", rel))
        for L, X in embed_all(m, tok, probes, LAYERS).items():
            bank[(rel, L)] = pca(X, D_NODE)
        del m
        print(f"{rel} embedded ({time.time()-t0:.0f}s)", flush=True)

    node_order = [(r, L) for r in RELS + ["base"] for L in LAYERS]   # 15 available
    ntr = N_PROBE // 2                                               # channel-fit half vs dynamics half
    res = {"nodes_available": [f"{r}:{L}" for r, L in node_order], "d_node": D_NODE}

    def build_system(nodes):
        W = {}
        for i, j in itertools.permutations(range(len(nodes)), 2):
            W[(i, j)] = ridge(bank[nodes[i]][:ntr], bank[nodes[j]][:ntr])
        u = [bank[nd][ntr:] for nd in nodes]
        return settle_traj(u, W)

    # (a) closure error vs memory length, N=8, K=4
    nodes8 = node_order[:8]
    traj8 = build_system(nodes8)
    print(f"settling dynamics built for N=8 ({time.time()-t0:.0f}s)", flush=True)
    res["closure_vs_L"] = {}
    best_err = 1e9
    for L in [1, 2, 3, 5, 8]:
        err, rank, _, resid = mz_closure(traj8[:, :4], L)
        res["closure_vs_L"][L] = {"err": float(err), "effrank": rank}
        best_err = min(best_err, err)
        print(f"  L={L}: closure err {err:.3f}, eff-rank {rank}", flush=True)
    markov = res["closure_vs_L"][1]["err"]; mem = res["closure_vs_L"][5]["err"]
    res["memory_helps"] = float(markov / max(mem, 1e-9))

    # (b) rank vs K (routed width), N=8, L=5
    res["rank_vs_K"] = {}
    for K in [2, 4, 6]:
        _, rank, _, _ = mz_closure(traj8[:, :K], 5)
        res["rank_vs_K"][K] = rank
        print(f"  K={K}: eff-rank {rank}", flush=True)

    # (c) rank vs N (federation size), K=3 fixed, L=5
    res["rank_vs_N"] = {}
    for N in [4, 6, 8, 10]:
        traj = build_system(node_order[:N]) if N != 8 else traj8
        err, rank, _, resid = mz_closure(traj[:, :3], 5)
        res["rank_vs_N"][N] = {"effrank": rank, "err": float(err)}
        print(f"  N={N}: eff-rank {rank} (err {err:.3f}) ({time.time()-t0:.0f}s)", flush=True)

    # (d) residual vs difficulty (difficulty = how far tracked nodes move during settling)
    err, rank, _, resid = mz_closure(traj8[:, :4], 5)
    move = np.linalg.norm((traj8[-1, :4] - traj8[0, :4]).transpose(1, 0, 2).reshape(resid.shape[0], -1), axis=1)
    res["residual_difficulty_corr"] = float(np.corrcoef(resid, move)[0, 1])

    ranks_N = [res["rank_vs_N"][N]["effrank"] for N in [4, 6, 8, 10]]
    ranks_K = [res["rank_vs_K"][K] for K in [2, 4, 6]]
    res["pass"] = bool(best_err < 0.15
                       and ranks_K[0] < ranks_K[-1]
                       and max(ranks_N) - min(ranks_N) <= 2)
    print(f"\nG2-real: best closure err {best_err:.3f} (<0.15), memory helps {res['memory_helps']:.1f}x, "
          f"rank vs K {ranks_K}, rank vs N {ranks_N}, resid-difficulty corr "
          f"{res['residual_difficulty_corr']:.2f} -> {'PASS' if res['pass'] else 'FAIL'}")
    json.dump(res, open(os.path.join(HERE, "gate2_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
