"""
demo/readout.py — R5 readout via the MATCHED-PROBE tied coupling (the principled gauge-fixing). What each
model says, silent ones included, in its own vocabulary — cross-modally and stably, because the tie pins the
GW orbit that relational geometry alone cannot select. Only (D, w) crosses; the tie is input identity, not a
frame (R3 preserved — see --scramble).
"""
import numpy as np


def tied_transfer(clouds, meta, manifest, m=12, n_outer=30):
    """All members share the probe index (image_i and caption_i are the same world-event), so one shared
    coupling ties every member's assignment of event i to atoms — this pins the GW orbit and makes cross-modal
    transfer STABLE (a dog image -> silent text reads 'a photo of a dog' across inits, no fragility).
    Event-relevance r is the geometric-mean reweighting of the ACTIVE members; every member then reads its OWN
    side of the input-relevant events (events reweighted through the tied anchor: re = pi @ a)."""
    from . import engine as E
    active = [p for p in meta if meta[p]["active"]]
    if not active:
        return None
    n = len(clouds[active[0]][0][1])
    logr = np.zeros(n)
    for p in active:
        logr += np.log(clouds[p][0][1] + 1e-12)
    r = np.exp(logr - logr.max()); r /= r.sum()
    rng = np.random.default_rng(0)
    De = rng.random((m, m)); De = (De + De.T) / 2; np.fill_diagonal(De, 0)
    De /= np.median(De[np.triu_indices(m, 1)])
    a0 = np.full(m, 1.0 / m)
    Bbar = {p: np.full(len(clouds[p]), 1.0 / len(clouds[p])) for p in clouds}
    res = E.equilibrate_tied(clouds, r, De, a0, a0.copy(), Bbar, n_outer=n_outer)
    re = res["pi"] @ res["a"]
    idx = np.argsort(-re)[:3]
    a = res["a"]
    panels = {}
    for port in clouds:
        mod = meta[port]["modality"]
        labels = manifest["vision_labels"] if mod == "vision" else manifest["texts"]
        panels[port] = {"active": meta[port]["active"], "modality": mod, "B": res["B"][port],
                        "exemplars": [str(labels[i])[:44] for i in idx]}
    return {"panels": panels, "atoms": len(a), "active_atoms": int((a > a.max() * 0.1).sum()),
            "F": res["F_trace"][-1], "converged": res["converged"], "monotone": res["monotone"]}
