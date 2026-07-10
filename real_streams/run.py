"""
run.py — first REAL-model streams through the validated io_trace reader (instrument UNCHANGED).
Three streams per PREREG.md: MiniLM memoryless (frozen: 0), MiniLM sliding-window W=4 (frozen: cutoff at the
window edge), Qwen2.5-0.5B digit-token stream (exploratory). Writes results.json + dashboard.png.
"""
import os, sys, json, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "io_trace"))
import stream_trace as ST                      # the validated reader, unchanged

torch.set_num_threads(os.cpu_count() or 4)
T_MINI = 3000
KMAX, L = 25, 12
Q_PROJ = 8
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)


def project_q(Y, q=Q_PROJ):
    """Fixed linear channel reduction of the output stream (disclosed; a projection cannot create memory)."""
    Yc = Y - Y.mean(0)
    _, _, Vt = np.linalg.svd(Yc[:1000], full_matrices=False)
    return Yc @ Vt[:q].T


@torch.no_grad()
def minilm_stream(texts, tag):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    m = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").eval()
    out = []
    for i in range(0, len(texts), 64):
        enc = tok(texts[i:i + 64], return_tensors="pt", padding=True, truncation=True, max_length=64)
        h = m(**enc).last_hidden_state
        msk = enc["attention_mask"].unsqueeze(-1).float()
        out.append(((h * msk).sum(1) / msk.sum(1).clamp(min=1)).numpy())
        if (i // 64) % 10 == 0:
            log(f"  [{tag}] {i+64}/{len(texts)}")
    return np.concatenate(out)


def stream_minilm_memoryless(seed=0):
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(T_MINI, 3))
    texts = [f"sensor a={x[0]:+.2f} b={x[1]:+.2f} c={x[2]:+.2f}." for x in u]
    y = project_q(minilm_stream(texts, "memoryless"))
    return u, y, texts


def stream_minilm_window(seed=0, W=4):
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(T_MINI, 3))
    texts = []
    for t in range(T_MINI):
        parts = []
        for k, label in enumerate(["now", "before", "earlier", "earliest"][:W]):
            x = u[max(0, t - k)]
            parts.append(f"{label} a={x[0]:+.2f} b={x[1]:+.2f} c={x[2]:+.2f}")
        texts.append("readings " + "; ".join(parts) + ".")
    y = project_q(minilm_stream(texts, "window4"))
    return u, y, texts


@torch.no_grad()
def stream_qwen_digits(n_seq=6, T_seq=1024, seed=0):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    m = AutoModel.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct").eval()
    dig_ids = []
    for d in range(10):
        ids = tok(f"{d}", add_special_tokens=False)["input_ids"]
        assert len(ids) == 1, f"digit '{d}' not a single token: {ids}"
        dig_ids.append(ids[0])
    rng = np.random.default_rng(seed)
    us, ys, sample = [], [], None
    for s in range(n_seq):
        digs = rng.integers(0, 10, T_seq)
        ids = torch.tensor([[dig_ids[d] for d in digs]])
        h = m(input_ids=ids).last_hidden_state[0].float().numpy()  # (T_seq, 896)
        us.append(((digs - 4.5) / 2.87)[:, None])                  # centered/scaled scalar input
        ys.append(h)
        if sample is None:
            sample = " ".join(str(d) for d in digs[:40]) + " ..."
        log(f"  [qwen] sequence {s+1}/{n_seq}")
    Vfit = np.concatenate(ys)[:2000]
    _, _, Vt = np.linalg.svd(Vfit - Vfit.mean(0), full_matrices=False)
    ys = [ (y - Vfit.mean(0)) @ Vt[:Q_PROJ].T for y in ys ]
    return us, ys, sample


# ---- multi-sequence read (averaged memory response; same reader math, floor per PREREG) ----
def read_multi(us, ys, kmax=KMAX, Lh=L, seed=0):
    hs = [ST.est_markov(u, y, kmax) for u, y in zip(us, ys)]
    h = np.mean(hs, axis=0)
    sv = np.linalg.svd(ST.block_hankel(h, Lh), compute_uv=False)
    rng = np.random.default_rng(seed); tops = []
    for _ in range(ST.FLOOR_SHIFTS):
        hn = np.mean([ST.est_markov(u, np.roll(y, int(rng.integers(len(y)//4, 3*len(y)//4)), axis=0), kmax)
                      for u, y in zip(us, ys)], axis=0)
        tops.append(np.linalg.svd(ST.block_hankel(hn, Lh), compute_uv=False)[0])
    floor = float(np.percentile(tops, ST.FLOOR_Q))
    order = int((sv > floor).sum())
    gap = float(sv[order-1] / (sv[order] + 1e-15)) if 0 < order < len(sv) else 0.0
    return {"order": order, "gap": gap, "floor": floor, "svals": sv, "h": h}


def dashboard(panels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(panels), 3, figsize=(15, 3.4 * len(panels)))
    for r, P in enumerate(panels):
        ax = axes[r]
        # col 1: literal I/O — input trace + output channel heat-strip
        ax[0].plot(P["u_show"], lw=0.7, color="#333")
        ax[0].set_title(f"{P['name']} — INPUT (first 200 steps)", fontsize=9)
        ax2 = ax[0].twinx(); ax2.imshow(P["y_show"].T, aspect="auto", cmap="magma",
                                        extent=[0, len(P["u_show"]), -3, 3], alpha=0.55)
        ax2.set_yticks([]); ax[0].text(0.01, -0.28, P["io_text"][:150], transform=ax[0].transAxes,
                                       fontsize=6.5, family="monospace")
        # col 2: memory response vs lag + floor
        mag = [np.linalg.norm(P["h"][k]) for k in range(1, len(P["h"]))]
        ax[1].bar(range(1, len(P["h"])), mag, color="#2a6fdb")
        ax[1].axhline(P["floor"] / np.sqrt(L), color="crimson", ls="--", lw=1, label="floor (scaled)")
        if P.get("cutoff"): ax[1].axvline(P["cutoff"] - 0.5, color="green", ls=":", lw=1.5, label=f"window edge k={P['cutoff']}")
        ax[1].set_title("memory response |h_k| vs lag", fontsize=9); ax[1].legend(fontsize=7)
        # col 3: Hankel spectrum + floor -> order
        nsv = min(15, len(P["svals"]))
        ax[2].semilogy(np.arange(1, nsv + 1), P["svals"][:nsv], "o-", ms=4, color="#333")
        ax[2].axhline(P["floor"], color="crimson", ls="--", lw=1)
        ax[2].set_title(f"Hankel spectrum -> ORDER = {P['order']}", fontsize=9)
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


def main():
    res = {}
    panels = []

    log("STREAM 1: MiniLM memoryless (frozen: order 0)")
    u, y, texts = stream_minilm_memoryless()
    rd = ST.read_trace(u, y, kmax=KMAX, L=L, seed=0)
    res["minilm_memoryless"] = {"order": rd["order"], "floor": rd["floor"], "top_sv": float(rd["svals"][0])}
    log(f"  -> order {rd['order']} (frozen prediction: 0)")
    panels.append({"name": "MiniLM memoryless", "u_show": u[:200, 0], "y_show": y[:200],
                   "io_text": "in:  " + texts[5], "h": rd["h"], "floor": rd["floor"],
                   "svals": rd["svals"], "order": rd["order"]})

    log("STREAM 2: MiniLM sliding-window W=4 (frozen: cutoff at k=4)")
    u, y, texts = stream_minilm_window()
    rd = ST.read_trace(u, y, kmax=KMAX, L=L, seed=0)
    mag = np.array([np.linalg.norm(rd["h"][k]) for k in range(1, KMAX + 1)])
    inside, outside = float(mag[:3].min()), float(mag[3:].max())
    res["minilm_window4"] = {"order": rd["order"], "mem_lags_1_3_min": inside, "mem_lags_4plus_max": outside,
                             "cutoff_clean": bool(inside > 3 * outside)}
    log(f"  -> order {rd['order']}; |h| lags1-3 min {inside:.4f} vs lags4+ max {outside:.4f}")
    panels.append({"name": "MiniLM window W=4", "u_show": u[:200, 0], "y_show": y[:200],
                   "io_text": "in:  " + texts[5], "h": rd["h"], "floor": rd["floor"],
                   "svals": rd["svals"], "order": rd["order"], "cutoff": 4})

    log("STREAM 3: Qwen2.5-0.5B digit-token stream (exploratory)")
    us, ys, sample = stream_qwen_digits()
    rd = read_multi(us, ys)
    res["qwen_digits"] = {"order": rd["order"], "gap": rd["gap"], "floor": rd["floor"],
                          "svals_top10": [float(x) for x in rd["svals"][:10]]}
    log(f"  -> order {rd['order']}  gap {rd['gap']:.2f}")
    panels.append({"name": "Qwen2.5-0.5B digits", "u_show": us[0][:200, 0], "y_show": ys[0][:200],
                   "io_text": "in:  " + sample, "h": rd["h"], "floor": rd["floor"],
                   "svals": rd["svals"], "order": rd["order"]})

    dashboard(panels, os.path.join(HERE, "dashboard.png"))
    json.dump(res, open(os.path.join(HERE, "results.json"), "w"), indent=1)
    log("wrote results.json + dashboard.png")


if __name__ == "__main__":
    main()
