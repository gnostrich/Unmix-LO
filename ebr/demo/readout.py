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
