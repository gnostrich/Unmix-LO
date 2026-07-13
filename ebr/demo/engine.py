"""
demo/engine.py — the per-prompt F-loop on real clouds, channel-aware (R2 + R4). Same functional F as the
validated single-channel path (energy/functional is its C_v=1 special case); here each port carries C_v
channels and a per-port channel-gain simplex B_v that routes which channel the anchor listens to, adapted
every prompt (R4). One authority: every block is an F-descent step with a backtracking monotonicity guard.

F = Σ_v Σ_c B_v[c]·[GW(D_v^c, D_e) + ε KL(π_v^c ‖ w_v⊗a)] + τ KL(a‖ā) + η KL(B_v ‖ B̄_v)
  B_v ∈ simplex^{C_v};  B̄_v = data-derived slow reference (Polyak), NOT a learned router (FIX-1).

Blocks: π (per port-channel, semi-relaxed GW mirror step), B (routing: closed-form simplex mirror step),
D_e (GW barycenter, B-weighted), a (unbalanced masses). All backtracked so F is monotone non-increasing.
"""
import numpy as np
from ..transport import gw
from ..energy import functional as EN


def equilibrate_tied(clouds, r, De0, a0, abar, Bbar, eps=0.08, tau=1.0, eta=0.5, n_outer=25, j_sink=5):
    """MATCHED-PROBE tied couplings — the principled gauge-fixing. All members share the SAME probe index
    (the same world-events: image_i and caption_i are responses to event i). So there is ONE shared coupling
    pi (n_events × m) tying every member's assignment of event i to atoms — this pins which GW-orbit element is
    the semantically-faithful one (which relational-only GW cannot). "Same input" is an identity on the DATA
    side, not a frame on the representation side, so R3 holds: no coordinates cross.

    The tie is a convex (row-marginal) constraint on pi, so the pi-update stays an entropic projection (one
    more Csiszár set) — monotonicity inherited from the backtracking guard, no new Lyapunov proof needed.
    r = shared event-relevance (row-marginal target, from the active members). Returns shared pi + anchor + B.
    """
    m = len(a0); ports = list(clouds); n = len(r)
    De = np.array(De0, float)
    if De.shape[0] == 1:
        De = np.array([[0.0]])
    a = np.array(a0, float)
    Cv = {v: len(clouds[v]) for v in ports}
    B = {v: np.full(Cv[v], 1.0 / Cv[v]) for v in ports}
    pi = np.outer(r, a)

    def F(pi, De, a, B):
        f = tau * gw.kl(a, abar) + eps * gw.kl(pi, np.outer(r, a))
        for v in ports:
            for c, (Dc, _w) in enumerate(clouds[v]):
                f += B[v][c] * gw.gw_cost(Dc, De, pi)
            f += eta * gw.kl(B[v], Bbar[v])
        return f

    ftr = []; f_prev = None
    for it in range(n_outer):
        # pi-block: summed B-weighted GW gradient + entropy; entropic mirror step, row-marginal tied to r
        G = eps * np.log((pi + 1e-300) / (np.outer(r, a) + 1e-300))
        for v in ports:
            for c, (Dc, _w) in enumerate(clouds[v]):
                G += B[v][c] * gw.gw_grad(Dc, De, pi)
        f0 = F(pi, De, a, B); step = 1.0
        for _bt in range(30):
            K = pi * np.exp(-step * (G - G.min())); row = K.sum(1); row[row < 1e-300] = 1e-300
            cand = K * (r / row)[:, None]
            if F(cand, De, a, B) <= f0 + 1e-12:
                pi = cand; break
            step *= 0.5
        # B-block: route each member to its lowest-cost channel under the shared pi
        for v in ports:
            costs = np.array([gw.gw_cost(Dc, De, pi) for Dc, _w in clouds[v]])
            lb = np.log(np.maximum(Bbar[v], 1e-12)) - costs / max(eta, 1e-9)
            B[v] = np.exp(lb - lb.max()); B[v] /= B[v].sum()
        # De/a-block: B-weighted barycenter + col-marginal masses, backtracked
        num = np.zeros((m, m))
        for v in ports:
            for c, (Dc, _w) in enumerate(clouds[v]):
                num += B[v][c] * (pi.T @ Dc @ pi)
        denom = np.outer(a, a) * max(sum(Cv.values()), 1); denom[denom < 1e-12] = 1e-12
        De_c = 0.5 * (num / denom + (num / denom).T); np.fill_diagonal(De_c, 0.0)
        med = np.median(De_c[np.triu_indices(m, 1)]) if m > 1 else 1.0
        De_c = De_c / (med if med > 0 else 1.0)
        a_c = np.maximum((eps * pi.sum(0) + tau * abar) / (eps + tau), 1e-8)
        f_after = F(pi, De, a, B); al = 1.0
        for _bt in range(20):
            De_t = (1 - al) * De + al * De_c; a_t = np.maximum((1 - al) * a + al * a_c, 1e-8)
            if F(pi, De_t, a_t, B) <= f_after + 1e-12:
                De, a = De_t, a_t; break
            al *= 0.5
        f = F(pi, De, a, B); ftr.append(f)
        if f_prev is not None and abs(f_prev - f) < 1e-4 * (abs(f_prev) + 1e-9):
            break
        f_prev = f
    return {"pi": pi, "De": De, "a": a, "B": B, "F_trace": ftr, "converged": it < n_outer - 1,
            "monotone": all(ftr[i + 1] <= ftr[i] + 1e-9 for i in range(len(ftr) - 1))}


def _port_cost(chans, De, a, pis, eps):
    """Per-channel (GW + entropy) costs for one port -> vector over channels. chans: [(Dc, wc), ...]."""
    return np.array([gw.entropic_gw_value(Dc, De, wc, a, pi, eps) for (Dc, wc), pi in zip(chans, pis)])


def _F(clouds, De, a, abar, pis, B, Bbar, eps, tau, eta):
    f = tau * gw.kl(a, abar)
    for v, chans in clouds.items():
        costs = _port_cost(chans, De, a, pis[v], eps)
        f += float(B[v] @ costs) + eta * gw.kl(B[v], Bbar[v])
    return f


def equilibrate(clouds, De0, a0, abar, Bbar, eps=0.08, tau=1.0, eta=0.5,
                pis0=None, B0=None, n_outer=20, j_sink=5, tol=1e-4):
    """clouds: dict port -> [(D_v^c, w_v^c) for c in channels]. Returns anchor + couplings + gains + trace."""
    m = len(a0)
    De = np.array(De0, float)
    if De.shape[0] == 1:
        De = np.array([[0.0]])
    a = np.array(a0, float)
    ports = list(clouds)
    Cv = {v: len(clouds[v]) for v in ports}
    pis = {v: [None] * Cv[v] for v in ports} if pis0 is None else {v: [np.array(p) for p in pis0[v]] for v in ports}
    B = {v: (np.array(B0[v]) if B0 else np.full(Cv[v], 1.0 / Cv[v])) for v in ports}
    ftr = []
    f_prev = None
    for it in range(n_outer):
        # pi-block: each port-channel coupling is a monotone descent on its own F_ve
        for v in ports:
            pis[v] = [gw.equilibrate_coupling(Dc, De, wc, a, eps, pi0=p, j_sink=j_sink)[0]
                      for (Dc, wc), p in zip(clouds[v], pis[v])]
        # B-block: route to the lowest-cost channel, tethered to Bbar (simplex mirror step; closed form)
        for v in ports:
            costs = _port_cost(clouds[v], De, a, pis[v], eps)
            logB = np.log(np.maximum(Bbar[v], 1e-12)) - costs / max(eta, 1e-9)
            B[v] = np.exp(logB - logB.max()); B[v] /= B[v].sum()
        f_after = _F(clouds, De, a, abar, pis, B, Bbar, eps, tau, eta)
        # De/a-block: B-weighted GW barycenter + unbalanced masses, backtracked so F does not increase
        num = np.zeros((m, m)); Q = np.zeros(m)
        for v in ports:
            for c, ((Dc, wc), pi) in enumerate(zip(clouds[v], pis[v])):
                num += B[v][c] * (pi.T @ Dc @ pi); Q += B[v][c] * pi.sum(0)
        denom = np.outer(a, a) * max(sum(Cv.values()), 1); denom[denom < 1e-12] = 1e-12
        De_cand = 0.5 * (num / denom + (num / denom).T); np.fill_diagonal(De_cand, 0.0)
        med = np.median(De_cand[np.triu_indices(m, 1)]) if m > 1 else 1.0
        De_cand = De_cand / (med if med > 0 else 1.0)
        a_cand = np.maximum((eps * Q + tau * abar) / (eps * max(sum(Cv.values()), 1) + tau), 1e-8)
        alpha = 1.0
        for _bt in range(20):
            De_t = (1 - alpha) * De + alpha * De_cand
            a_t = np.maximum((1 - alpha) * a + alpha * a_cand, 1e-8)
            if _F(clouds, De_t, a_t, abar, pis, B, Bbar, eps, tau, eta) <= f_after + 1e-12:
                De, a = De_t, a_t; break
            alpha *= 0.5
        f = _F(clouds, De, a, abar, pis, B, Bbar, eps, tau, eta)
        ftr.append(f)
        if f_prev is not None and abs(f_prev - f) < tol * (abs(f_prev) + 1e-9):
            break
        f_prev = f
    return {"De": De, "a": a, "pis": pis, "B": B, "F_trace": ftr,
            "converged": it < n_outer - 1, "monotone": all(ftr[i + 1] <= ftr[i] + 1e-9 for i in range(len(ftr) - 1))}
