"""
STABILITY_GATE steps 2-3 — train cyclic channels K times, measure boundary stability.

Pre-registered in ../STABILITY_GATE.md (committed before this ran): PASS iff cross-seed
ARI of eval-set partitions >= 0.8 AND the validity guards hold. Controls: untrained
channels, raw-geometry k-means. No paired-alignment loss anywhere.
"""
import os, json, itertools
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

HERE = os.path.dirname(os.path.abspath(__file__))
SPACES = ["gpt2", "qwen", "minilm"]
K_RUNS = int(os.environ.get("SG_RUNS", 8))
STEPS = int(os.environ.get("SG_STEPS", 400))
BATCH = 256
N_EVAL = 1000
K_CLUST = 9
torch.set_num_threads(os.cpu_count() or 4)


def load_spaces():
    embs = {}
    for s in SPACES:
        E = np.load(os.path.join(HERE, f"emb_{s}.npy")).astype(np.float32)
        E = (E - E.mean(0)) / (E.std(0) + 1e-6)          # standardize each frozen space
        embs[s] = E
    labels = json.load(open(os.path.join(HERE, "corpus_meta.json")))["labels"]
    return embs, np.array(labels)


class Channel(nn.Module):
    def __init__(self, din, dout, hidden=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, hidden), nn.GELU(), nn.Linear(hidden, dout))

    def forward(self, x):
        return self.net(x)


def cos_sim_matrix(X):
    Xn = X / (X.norm(dim=1, keepdim=True) + 1e-8)
    return Xn @ Xn.T


def train_run(embs_train, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dims = {s: embs_train[s].shape[1] for s in SPACES}
    ch = nn.ModuleDict({f"{a}__{b}": Channel(dims[a], dims[b])
                        for a in SPACES for b in SPACES if a != b})
    opt = torch.optim.Adam(ch.parameters(), lr=1e-3)
    n = len(embs_train[SPACES[0]])
    boot = rng.integers(0, n, n)                          # bootstrap resample per run
    T = {s: torch.tensor(embs_train[s][boot]) for s in SPACES}

    cycles3 = [("gpt2", "qwen", "minilm"), ("gpt2", "minilm", "qwen")]
    logs = {}
    for step in range(STEPS):
        idx = torch.tensor(rng.integers(0, n, BATCH))
        x = {s: T[s][idx] for s in SPACES}
        loss_cyc = 0.0
        for a, b in itertools.permutations(SPACES, 2):    # 2-cycles
            back = ch[f"{b}__{a}"](ch[f"{a}__{b}"](x[a]))
            loss_cyc = loss_cyc + ((back - x[a]) ** 2).mean()
        for a, b, c in cycles3:                           # 3-cycles
            rt = ch[f"{c}__{a}"](ch[f"{b}__{c}"](ch[f"{a}__{b}"](x[a])))
            loss_cyc = loss_cyc + ((rt - x[a]) ** 2).mean()
        loss_struct = 0.0
        for a, b in itertools.permutations(SPACES, 2):    # structure preservation
            loss_struct = loss_struct + ((cos_sim_matrix(x[a])
                                          - cos_sim_matrix(ch[f"{a}__{b}"](x[a]))) ** 2).mean()
        loss_deg = 0.0
        for a, b in itertools.permutations(SPACES, 2):    # anti-collapse variance floor
            std = ch[f"{a}__{b}"](x[a]).std(0)
            loss_deg = loss_deg + torch.relu(0.1 - std).mean()
        loss = loss_cyc + loss_struct + loss_deg
        opt.zero_grad(); loss.backward(); opt.step()
        if step == STEPS - 1:
            logs = {"cycle": float(loss_cyc), "struct": float(loss_struct),
                    "deg": float(loss_deg)}
    return ch.eval(), logs


@torch.no_grad()
def fused_rep(ch, embs, idx):
    """Route every space's view into space A (gpt2) and concatenate — the enacted geometry."""
    xs = {s: torch.tensor(embs[s][idx]) for s in SPACES}
    parts = [xs["gpt2"]] + [ch[f"{s}__gpt2"](xs[s]) for s in SPACES if s != "gpt2"]
    return torch.cat(parts, dim=1).numpy()


@torch.no_grad()
def guards(ch, embs, eval_idx, ref_nn):
    """Anti-collapse + structure-preservation on held-out data. ref_nn: raw top-10 NN sets."""
    out = {}
    for a, b in itertools.permutations(SPACES, 2):
        Y = ch[f"{a}__{b}"](torch.tensor(embs[a][eval_idx])).numpy()
        s = np.linalg.svd(Y - Y.mean(0), compute_uv=False)
        eff_rank = float((s.sum() ** 2) / ((s ** 2).sum() + 1e-9))
        std_ratio = float(Y.std(0).mean())                # inputs standardized to ~1
        Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9)
        nn_map = np.argsort(-(Yn @ Yn.T), axis=1)[:, 1:11]
        jac = np.mean([len(set(nn_map[i]) & ref_nn[a][i]) / len(set(nn_map[i]) | ref_nn[a][i])
                       for i in range(0, len(eval_idx), 5)])
        out[f"{a}->{b}"] = {"eff_rank": round(eff_rank, 1), "std": round(std_ratio, 3),
                            "nn_jaccard_vs_source": round(float(jac), 3)}
    return out


def partition(train_rep, eval_rep):
    km = KMeans(n_clusters=K_CLUST, n_init=10, random_state=0).fit(train_rep)
    return km.predict(eval_rep)


def mean_pairwise_ari(parts):
    vals = [adjusted_rand_score(parts[i], parts[j])
            for i in range(len(parts)) for j in range(i + 1, len(parts))]
    return float(np.mean(vals)), float(np.std(vals))


def main():
    embs, labels = load_spaces()
    n = len(labels)
    rng = np.random.default_rng(123)
    eval_idx = rng.choice(n, N_EVAL, replace=False)
    train_idx = np.setdiff1d(np.arange(n), eval_idx)
    embs_train = {s: embs[s][train_idx] for s in SPACES}
    print(f"{n} samples: {len(train_idx)} train pool, {N_EVAL} eval (fixed)", flush=True)

    ref_nn = {}
    for a in SPACES:
        X = embs[a][eval_idx]
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        nnm = np.argsort(-(Xn @ Xn.T), axis=1)[:, 1:11]
        ref_nn[a] = {i: set(nnm[i]) for i in range(len(eval_idx))}

    parts_tr, parts_untr, chan_agree, run_logs, guard_reports = [], [], [], [], []
    trained_outputs = []
    for k in range(K_RUNS):
        ch, logs = train_run(embs_train, seed=k)
        run_logs.append(logs)
        tr_rep_train = fused_rep(ch, embs, train_idx)
        tr_rep_eval = fused_rep(ch, embs, eval_idx)
        parts_tr.append(partition(tr_rep_train, tr_rep_eval))
        guard_reports.append(guards(ch, embs, eval_idx, ref_nn))
        with torch.no_grad():
            trained_outputs.append({f"{a}__{b}": ch[f"{a}__{b}"](
                torch.tensor(embs[a][eval_idx])).numpy()
                for a, b in itertools.permutations(SPACES, 2)})
        # untrained control (same seed, no training)
        torch.manual_seed(k)
        dims = {s: embs[s].shape[1] for s in SPACES}
        ch0 = nn.ModuleDict({f"{a}__{b}": Channel(dims[a], dims[b])
                             for a in SPACES for b in SPACES if a != b}).eval()
        parts_untr.append(partition(fused_rep(ch0, embs, train_idx),
                                    fused_rep(ch0, embs, eval_idx)))
        print(f"run {k}: losses={logs}", flush=True)

    # channel functional agreement across runs
    for a, b in itertools.permutations(SPACES, 2):
        sims = []
        for i in range(K_RUNS):
            for j in range(i + 1, K_RUNS):
                Yi, Yj = trained_outputs[i][f"{a}__{b}"], trained_outputs[j][f"{a}__{b}"]
                num = (Yi * Yj).sum(1)
                den = np.linalg.norm(Yi, axis=1) * np.linalg.norm(Yj, axis=1) + 1e-9
                sims.append(float(np.mean(num / den)))
        chan_agree.append((f"{a}->{b}", round(float(np.mean(sims)), 3)))

    ari, ari_std = mean_pairwise_ari(parts_tr)
    ari_untr, _ = mean_pairwise_ari(parts_untr)
    # raw-geometry control: k-means on each raw space, bootstrap-fitted
    raw_ari = {}
    for a in SPACES:
        ps = []
        for k in range(K_RUNS):
            b = np.random.default_rng(k).integers(0, len(train_idx), len(train_idx))
            ps.append(partition(embs[a][train_idx][b], embs[a][eval_idx]))
        raw_ari[a] = round(mean_pairwise_ari(ps)[0], 3)
    # post-hoc diagnostic only: agreement of trained partitions with source labels
    label_ari = float(np.mean([adjusted_rand_score(labels[eval_idx], p) for p in parts_tr]))

    res = {"k_runs": K_RUNS, "steps": STEPS,
           "cross_seed_ARI": round(ari, 3), "ARI_std": round(ari_std, 3),
           "untrained_control_ARI": round(ari_untr, 3),
           "raw_geometry_control_ARI": raw_ari,
           "channel_functional_agreement": chan_agree,
           "guards_last_run": guard_reports[-1],
           "final_losses_per_run": run_logs,
           "posthoc_ARI_vs_source_labels": round(label_ari, 3)}
    ok_collapse = all(g["eff_rank"] >= 10 and g["std"] >= 0.1
                      for rep in guard_reports for g in rep.values())
    # pre-registered guard 2 (attribution): stability only counts as ENACTED if the trained
    # pipeline beats the untrained-channel control; otherwise it is the data's own geometry
    ok_attrib = ari > ari_untr
    res["guards_pass"] = bool(ok_collapse and ok_attrib)
    res["guard_detail"] = {"no_collapse": bool(ok_collapse),
                           "attribution_trained_beats_untrained": bool(ok_attrib)}
    res["gate"] = "PASS" if (ari >= 0.8 and ok_collapse and ok_attrib) else "FAIL"
    json.dump(res, open(os.path.join(HERE, "results_stability.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
    print(f"\nGATE: {res['gate']}  (pre-registered: PASS iff ARI>=0.8 AND guards hold)")


if __name__ == "__main__":
    main()
