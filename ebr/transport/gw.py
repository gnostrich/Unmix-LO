"""
transport/gw.py — square-loss entropic Gromov–Wasserstein primitives, semi-relaxed (hard cloud-side row
marginal = w, free anchor-side column marginal), with proximal warm-started updates so the caller owns the
functional F and its monotone descent (Lyapunov, §1). numpy only; consumes ONLY (D, w)-type data (no
coordinates ever cross), per the invariant-interface rule.

Square-loss GW cost for a coupling pi (n×m) between normalized costs Dv (n×n), De (m×m):
    C(pi) = sum_{i,j,k,l} (Dv[i,k]-De[j,l])^2 pi[i,j] pi[k,l]
Gradient (Peyré factorization, p=pi.sum(1), q=pi.sum(0)):
    grad[i,j] = 2( (Dv∘Dv) p )_i + 2( (De∘De) q )_j - 4 (Dv pi De^T)[i,j]
and C = 0.5 <grad, pi>.
"""
import numpy as np


def gw_grad(Dv, De, pi):
    p = pi.sum(1); q = pi.sum(0)
    return (2.0 * (Dv * Dv) @ p)[:, None] + (2.0 * (De * De) @ q)[None, :] - 4.0 * (Dv @ pi @ De.T)


def gw_cost(Dv, De, pi):
    return 0.5 * float((gw_grad(Dv, De, pi) * pi).sum())


def kl(pi, ref):
    """Generalized KL(pi ‖ ref) = sum pi log(pi/ref) - pi + ref, both nonnegative measures."""
    m = pi > 1e-300
    return float((pi[m] * np.log(pi[m] / ref[m])).sum() - pi.sum() + ref.sum())


def _f_ve(Dv, De, w, a, pi, eps):
    return gw_cost(Dv, De, pi) + eps * kl(pi, np.outer(w, a))


def semirelaxed_step(Dv, De, w, a, pi, eps, eta):
    """One mirror step on F_ve(pi) = GW + eps·KL(pi‖w⊗a) with step eta; enforce ONLY the row marginal
    (semi-relaxed: column/anchor marginal free). Total gradient includes the entropy term."""
    ref = np.outer(w, a)
    G = gw_grad(Dv, De, pi) + eps * np.log((pi + 1e-300) / (ref + 1e-300))
    K = pi * np.exp(-eta * (G - G.min()))
    row = K.sum(1)
    row[row < 1e-300] = 1e-300
    return K * (w / row)[:, None]


def equilibrate_coupling(Dv, De, w, a, eps, pi0=None, j_sink=5, eta0=1.0):
    """j_sink semi-relaxed mirror sweeps with backtracking so F_ve is monotone non-increasing (Lyapunov, §1).
    Warm-started from pi0 (coupling continuity, §7). Returns (pi, induced anchor mass q)."""
    m = De.shape[0]
    pi = np.outer(w, np.full(m, 1.0 / m)) if pi0 is None else np.array(pi0, dtype=np.float64)
    f = _f_ve(Dv, De, w, a, pi, eps)
    for _ in range(j_sink):
        eta = eta0
        for _bt in range(30):
            cand = semirelaxed_step(Dv, De, w, a, pi, eps, eta)
            fc = _f_ve(Dv, De, w, a, cand, eps)
            if fc <= f + 1e-12:
                pi, f = cand, fc
                break
            eta *= 0.5
    return pi, pi.sum(0)


def entropic_gw_value(Dv, De, w, a, pi, eps):
    """The (transport + entropy) contribution of one coupling to F: GW cost + eps·KL(pi ‖ w⊗a)."""
    ref = np.outer(w, a)
    return gw_cost(Dv, De, pi) + eps * kl(pi, ref)


def barycentric_pushforward(pi, a_e):
    """Message passing (§5.2): anchor mass pushed through the coupling rows onto the port's OWN support.
    r = normalize(pi @ a_e). No coordinates cross."""
    r = pi @ a_e
    s = r.sum()
    return r / s if s > 0 else r
