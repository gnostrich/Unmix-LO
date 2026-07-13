"""
energy/functional.py — the per-prompt equilibration of a SHARED anchor across the members of an edge, and
the F value it minimizes. This is where K-invariance is supposed to come from: the K members are pooled onto
ONE anchor by GW coupling, so shared structure is represented once; the residual measured AFTER deflating
that anchor should not inflate with K.

HONEST STATEMENT (FIX-3): the blocks are NOT exact I-projections. This is block-coordinate MIRROR DESCENT on
F, with a backtracking line search on every block guaranteeing monotone descent. The v1 "everything is one
I-projection" claim is aspirational; empirically the raw mirror/barycenter steps overshoot (67% monotone
without the guard → 100% with it). Only monotone descent is guaranteed, not exactness.
  pi-block : semi-relaxed entropic GW mirror step of each member D_v to the shared anchor D_e (backtracked).
  De-block : square-loss GW barycenter candidate, blended by backtracking so F does not increase.
  a-block  : anchor masses toward induced marginals (slow), backtracked; realizes self-sizing/parking.
F = Σ_v [GW(D_v,D_e;pi_v) + ε KL(pi_v‖w_v⊗a)] + τ KL(a‖ā).  Reported per prompt for the monotone-descent log.
"""
import numpy as np
from ..transport import gw


def gw_barycenter_De(Ds, pis, a):
    """Square-loss GW barycenter update for the anchor cost (Peyré 2016 closed form):
    D_e = [ Σ_v pi_v^T D_v pi_v ] / (a a^T), elementwise (averaged over members)."""
    m = len(a)
    num = np.zeros((m, m))
    for D, pi in zip(Ds, pis):
        num += pi.T @ D @ pi
    denom = np.outer(a, a) * len(Ds)
    denom[denom < 1e-12] = 1e-12
    De = num / denom
    De = 0.5 * (De + De.T)
    np.fill_diagonal(De, 0.0)
    return De


def F_value(Ds, ws, a, De, pis, abar, eps, tau):
    f = sum(gw.entropic_gw_value(D, De, w, a, pi, eps) for D, w, pi in zip(Ds, ws, pis))
    return f + tau * gw.kl(a, abar)


def equilibrate(Ds, ws, De0, a0, abar, eps=0.05, tau=1.0, pis0=None,
                n_outer=20, j_sink=5, tol=1e-4):
    """Block-coordinate descent to equilibrium for one prompt. Warm-started from pis0 (coupling continuity).
    Returns (pis, De, a, F_trace, converged)."""
    m = len(a0)
    De = np.array(De0, dtype=np.float64)
    a = np.array(a0, dtype=np.float64)
    pis = [None] * len(Ds) if pis0 is None else [np.array(p) for p in pis0]
    ftrace = []
    f_prev = None
    for it in range(n_outer):
        # pi-block (each coupling is a monotone descent on its own F_ve)
        pis = [gw.equilibrate_coupling(D, De, w, a, eps, pi0=p, j_sink=j_sink)[0]
               for D, w, p in zip(Ds, ws, pis)]
        f_after_pi = F_value(Ds, ws, a, De, pis, abar, eps, tau)
        # De/a-block: closed-form candidates, then backtrack-blend so F is non-increasing (Lyapunov guard)
        De_cand = gw_barycenter_De(Ds, pis, a)
        med = np.median(De_cand[np.triu_indices(m, 1)]) if m > 1 else 1.0
        De_cand = De_cand / (med if med > 0 else 1.0)
        Q = np.sum([pi.sum(0) for pi in pis], axis=0)
        a_cand = np.maximum((eps * Q + tau * abar) / (eps * len(Ds) + tau), 1e-8)
        alpha = 1.0
        for _bt in range(20):
            De_try = (1 - alpha) * De + alpha * De_cand
            a_try = np.maximum((1 - alpha) * a + alpha * a_cand, 1e-8)
            f_try = F_value(Ds, ws, a_try, De_try, pis, abar, eps, tau)
            if f_try <= f_after_pi + 1e-12:
                De, a = De_try, a_try
                break
            alpha *= 0.5
        f = F_value(Ds, ws, a, De, pis, abar, eps, tau)
        ftrace.append(f)
        if f_prev is not None and abs(f_prev - f) < tol * (abs(f_prev) + 1e-9):
            f_prev = f
            break
        f_prev = f
    converged = it < n_outer - 1
    return pis, De, a, ftrace, converged


def pis_or_uniform(pis, Ds, ws, m):
    return [np.outer(w, np.full(m, 1.0 / m)) if p is None else p for p, (D, w) in zip(pis, zip(Ds, ws))]
