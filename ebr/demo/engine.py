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
