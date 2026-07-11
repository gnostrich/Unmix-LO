"""
multimodal.py — the same world, watched through TWO real encoders. Answers "don't we have multiple
modalities?": diagnostics/encode.py observes each world state through a vision encoder (ViT) AND a language
encoder (MiniLM). Here we put both on the reader's input side.

A colored world latent u_t (K factors) smoothly sets the positions of the balls -> a watchable scene. The
same scene is observed two ways:
  vision  f^V_t = proj( ViT(render(state_t)) )         google/vit-base-patch16-224
  text    f^L_t = proj( MiniLM(describe(state_t)) )     sentence-transformers/all-MiniLM-L6-v2
Both are correlated (the world is smooth) and are different nonlinear encodings of the same u_t.

A watcher with PLANTED poles is driven by u_t and emits y_t. The reader (io_trace + correlated_read,
deconvolving) recovers the watcher's memory from (modality-feature, y_t). The multi-modal question: does the
read agree across vision and language (memory is a property of the watched system, not the modality), and
does fusing the two help when one modality is degraded?

Encoding is the slow part; features are cached to feats.npz. Run: python multimodal.py [--T 2000]
"""
import os, sys, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "orbit"))
sys.path.insert(0, os.path.join(HERE, "..", "io_trace"))
import physics as PH
import stream_trace as ST
import correlated_read as CR

K = 4                      # world factors
NBALL = PH.N
CACHE = os.path.join(HERE, "feats.npz")

# regions for the text modality (graded words -> MiniLM embeds them smoothly)
XR = ["far-left", "left", "center-left", "center", "center-right", "right", "far-right"]
YR = ["at the bottom", "low", "lower-mid", "mid-height", "upper-mid", "high", "at the top"]
SP = ["still", "drifting", "moving", "moving fast"]


def colored_latent(T, a=0.8, seed=0):
    rng = np.random.default_rng(seed)
    e = rng.normal(size=(T, K)); u = np.zeros((T, K))
    for t in range(1, T):
        u[t] = a * u[t - 1] + np.sqrt(1 - a * a) * e[t]
    return u


def positions(u_row, Gx, Gy, cen):
    """Map the K-factor latent to N ball positions inside the box (smooth, clamped)."""
    px = np.clip(cen[:, 0] + Gx @ u_row, PH.R, 1 - PH.R)
    py = np.clip(cen[:, 1] + Gy @ u_row, PH.R, 1 - PH.R)
    return np.stack([px, py], 1)


def scene(u, seed=0):
    """u -> list of world states [pos(2N), vel(2N)] (vel = finite difference, for the render ghost)."""
    rng = np.random.default_rng(seed)
    Gx = rng.normal(size=(NBALL, K)) * 0.18
    Gy = rng.normal(size=(NBALL, K)) * 0.18
    cen = rng.uniform(0.3, 0.7, (NBALL, 2))
    P = np.array([positions(u[t], Gx, Gy, cen) for t in range(len(u))])
    V = np.vstack([np.zeros((1, NBALL, 2)), np.diff(P, axis=0)])
    return [np.concatenate([P[t].ravel(), V[t].ravel()]) for t in range(len(u))]


def describe(state):
    p = state[:2 * NBALL].reshape(NBALL, 2); v = state[2 * NBALL:].reshape(NBALL, 2)
    parts = []
    for i in range(NBALL):
        xi = XR[min(int(p[i, 0] * len(XR)), len(XR) - 1)]
        yi = YR[min(int(p[i, 1] * len(YR)), len(YR) - 1)]
        sp = SP[min(int(np.hypot(*v[i]) / 0.03), len(SP) - 1)]
        parts.append(f"ball {i} is {xi} {yi}, {sp}")
    return "; ".join(parts)


def encode(states, seed=0, batch=32):
    """Real ViT + MiniLM features per frame, mean-pooled (full 768 / 384 dims). Cached in full so the
    dimensionality reduction can be re-tried offline."""
    import torch, torch.nn.functional as Fn
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(os.cpu_count() or 4)

    vit = AutoModel.from_pretrained("google/vit-base-patch16-224").eval()
    IM = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    IS = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    V = []
    with torch.no_grad():
        for i in range(0, len(states), batch):
            fr = np.stack([PH.render(s) for s in states[i:i + batch]]).astype(np.float32) / 255
            x = torch.tensor(fr.transpose(0, 3, 1, 2))
            x = Fn.interpolate(x, size=224, mode="bilinear", align_corners=False)
            x = (x - IM) / IS
            V.append(vit(pixel_values=x).last_hidden_state.mean(1).numpy())
            print(f"  vit {i + batch}/{len(states)}", end="\r", flush=True)
    V = np.concatenate(V); print()

    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    mlm = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").eval()
    txt = [describe(s) for s in states]; L = []
    with torch.no_grad():
        for i in range(0, len(txt), batch):
            e = tok(txt[i:i + batch], return_tensors="pt", padding=True, truncation=True, max_length=64)
            h = mlm(**e).last_hidden_state
            m = e["attention_mask"].unsqueeze(-1).float()
            L.append(((h * m).sum(1) / m.sum(1)).numpy())
            print(f"  minilm {i + batch}/{len(txt)}", end="\r", flush=True)
    L = np.concatenate(L); print()
    return V, L, txt[0]


def reduce(F, dim, seed=0):
    """Top-`dim` PCA of an encoder-feature stream, standardized. The scene's only variation is the ball
    motion, so the leading components track the world; a random projection would bury it in noise."""
    Fc = F - F.mean(0)
    U, S, Vt = np.linalg.svd(Fc, full_matrices=False)
    Z = Fc @ Vt[:dim].T
    return Z / (Z.std(0) + 1e-9)


def watcher_pair(u, r=0.8, period=12, seed=0):
    """Planted oscillatory watcher (complex pole pair) driven by the world latent u -> output y_t."""
    th = 2 * np.pi / period
    A = r * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    rng = np.random.default_rng(seed + 5)
    C = rng.normal(size=(CR.Q_OUT, 2)) / np.sqrt(2)
    B = rng.normal(size=(2, u.shape[1])) / np.sqrt(u.shape[1])
    x = np.zeros(2); ys = []; nr = np.random.default_rng(seed + 3)
    for t in range(len(u)):
        ys.append(C @ x + CR.NOISE * nr.normal(size=CR.Q_OUT)); x = A @ x + B @ u[t]
    return np.array(ys), np.linalg.eigvals(A)


def build(T=2000, seed=0, force=False):
    if os.path.exists(CACHE) and not force:
        d = np.load(CACHE, allow_pickle=True)
        if int(d["T"]) == T:
            return d
    u = colored_latent(T, seed=seed)
    states = scene(u, seed=seed)
    V, L, sample_txt = encode(states, seed=seed)
    y, tp = watcher_pair(u, seed=seed)
    strip = np.stack([PH.render(states[t]) for t in range(0, 40, 4)])
    np.savez(CACHE, T=T, u=u, V=V, L=L, y=y, tp=tp, strip=strip, sample_txt=sample_txt)
    return np.load(CACHE, allow_pickle=True)


def _whiten(X, k):
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(k, int((S > 1e-6).sum()))
    return U[:, :k] * np.sqrt(len(X))


def cca(A, B, k=100, d=4):
    """Canonical correlation between two encoder-feature streams. Returns the two sets of canonical
    variates and the canonical correlations. No supervision — the world state is never used."""
    Wa, Wb = _whiten(A, k), _whiten(B, k)
    M = Wa.T @ Wb / len(A)
    U, S, Vt = np.linalg.svd(M)
    return Wa @ U[:, :d], Wb @ Vt[:d].T, S[:d]


def shared_latent(A, B, k=100, d=4):
    """The cross-modal shared latent: average of the two canonical variates. What vision and language agree
    on is the common cause — the world."""
    za, zb, _ = cca(A, B, k=k, d=d)
    return (za + zb) / 2


def r2(f, target):
    F = np.c_[f, np.ones(len(f))]
    W, _, _, _ = np.linalg.lstsq(F, target, rcond=None)
    pred = F @ W
    return 1 - ((target - pred) ** 2).sum(0) / ((target - target.mean(0)) ** 2).sum(0)


def read_all(d, dim=8, k=150, sdim=4, seed=0):
    """Read the watcher's memory through each raw modality and through the cross-modal shared latent."""
    y, tp = d["y"], d["tp"]
    fV, fL = reduce(d["V"], dim), reduce(d["L"], dim)
    shared = shared_latent(d["V"], d["L"], k=k, d=sdim)
    out = {}
    for name, f in [("vision", fV), ("text", fL), ("shared", shared)]:
        rd = CR.read_trace_deconv(f, y, seed=seed)
        rd["pole_err"] = ST.pole_match_error(tp, rd["poles"])
        out[name] = rd
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--T", type=int, default=2000)
    ap.add_argument("--force", action="store_true"); a = ap.parse_args()
    d = build(T=a.T, force=a.force)
    print(f"\nsample description: {str(d['sample_txt'])}")
    print(f"planted watcher poles: {np.round(np.sort_complex(d['tp']), 3)}\n")
    res = read_all(d)
    for name, rd in res.items():
        pe = f"{rd['pole_err']:.3f}" if np.isfinite(rd['pole_err']) else "inf"
        pl = np.round(np.sort_complex(rd['poles']), 3) if len(rd['poles']) else "[]"
        print(f"  {name:7s}: order {rd['order']} (planted 2), pole err {pe}   poles~{pl}")


if __name__ == "__main__":
    main()
