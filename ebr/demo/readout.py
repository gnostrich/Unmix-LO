"""
demo/readout.py — R5 readouts from an equilibrated anchor: consensus, per-model panels (what EACH model
says, silent ones included, in its own vocabulary), and the disagreement meter.

Per-model panel = barycentric pushforward: anchor mass a pushed back through port v's (B-weighted) coupling
onto v's own library support, r_v = normalize(Σ_c B_v[c]·(π_v^c a)). The top-mass library rows are the
exemplars v surfaces. A SILENT port's coupling was equilibrated to the shared anchor (shaped by the ACTIVE
ports), so its pushforward highlights its exemplars aligned with the cross-modal consensus.
"""
import numpy as np


def pushforward(res, port):
    B = res["B"][port]
    r = sum(B[c] * (pi @ res["a"]) for c, pi in enumerate(res["pis"][port]))
    s = r.sum()
    return r / s if s > 0 else r


def consensus(res, park_frac=0.1):
    a = res["a"]
    active = int((a > a.max() * park_frac).sum())
    return {"atoms": len(a), "active": active, "parked": len(a) - active,
            "F": res["F_trace"][-1] if res["F_trace"] else float("nan"),
            "converged": res["converged"], "monotone": res["monotone"]}


def clip_bridge(clouds, manifest, m=14, n_outer=30, restarts=6):
    """Attempt to route cross-modal transfer through CLIP's two towers (its jointly-trained shared space is
    the most favourable case). Equilibrate ONLY the two CLIP towers and pushforward to each; entropic GW is
    nonconvex, so we take `restarts` random inits and keep the LOWEST-F equilibrium (F is the arbiter). This is
    FRAGILE, NOT a working solution: the F-optimal coupling is not the semantically-faithful one — at the
    default m=14 it yields "deer" for a dog image even though CLIP itself reads dog. It transfers correctly at
    some (m, init) and not others. Kept for honesty and shown labelled FRAGILE; heterogeneous non-CLIP models
    do not transfer at all. See WALL_crossmodal.md."""
    from . import engine as E
    sub = {k: clouds[k] for k in ("clip_vision", "clip_text") if k in clouds}
    if len(sub) < 2:
        return None
    a0 = np.full(m, 1.0 / m)
    Bbar = {p: np.full(len(sub[p]), 1.0 / len(sub[p])) for p in sub}
    best = None
    for s in range(restarts):
        rng = np.random.default_rng(s)
        De = rng.random((m, m)); De = (De + De.T) / 2; np.fill_diagonal(De, 0)
        De /= np.median(De[np.triu_indices(m, 1)])
        res = E.equilibrate(sub, De, a0, a0.copy(), Bbar, n_outer=n_outer)
        Ff = res["F_trace"][-1] if res["F_trace"] else float("inf")
        if best is None or Ff < best[0]:
            best = (Ff, res)
    res = best[1]
    out = {"converged": res["converged"], "F": best[0]}
    for port in sub:
        r = sum(res["B"][port][c] * (pi @ res["a"]) for c, pi in enumerate(res["pis"][port]))
        idx = np.argsort(-r)[:3]
        labels = manifest["vision_labels"] if port == "clip_vision" else manifest["texts"]
        out[port] = [str(labels[i])[:40] for i in idx]
    return out


def panels(res, meta, manifest, topk=3):
    """port -> (active?, modality, [top-k exemplar strings])."""
    out = {}
    for port in res["pis"]:
        r = pushforward(res, port)
        idx = np.argsort(-r)[:topk]
        mod = meta[port]["modality"]
        labels = manifest["vision_labels"] if mod == "vision" else manifest["texts"]
        ex = [str(labels[i])[:46] for i in idx]
        out[port] = {"active": meta[port]["active"], "modality": mod, "exemplars": ex,
                     "top_mass": float(r[idx[0]])}
    return out
