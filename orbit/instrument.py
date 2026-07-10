"""
instrument.py — build the instrument from the corpus: atlas (charts) + transfer operator P (spectral-gap
macros) + Mori-Zwanzig memory kernel (damped-oscillator fit). K is MEASURED, not imposed; kernel modes are
corpus-fit (no hand-picked constants).
"""
import numpy as np
from sklearn.cluster import KMeans


def build_atlas(states, k, seed=0):
    """Standardize states, k-means into charts. Returns (labels, centroids, mu, sd)."""
    mu = states.mean(0); sd = states.std(0) + 1e-8
    Z = (states - mu) / sd
    k = int(min(k, max(2, len(states) // 8)))          # >= ~8 windows/chart
    km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(Z)
    return km.labels_, km.cluster_centers_, mu, sd, Z


def transfer_operator(labels, bounds, k):
    """Row-stochastic chart->chart operator P, counted WITHIN trajectories only."""
    C = np.zeros((k, k))
    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        seg = labels[b0:b1]
        for a, c in zip(seg[:-1], seg[1:]):
            C[a, c] += 1
    rs = C.sum(1, keepdims=True)
    P = np.divide(C, rs, out=np.ones_like(C) / k, where=rs > 0)
    return P


def spectral_macros(P, band=(3, 6), drop_ratio=1.3):
    """Eigen-decompose P; find the macro count K just above the largest relative gap in `band`.
    Returns (K, gap_flagged, eigvals_sorted, psi) where psi = standardized subdominant right eigenvectors."""
    w, V = np.linalg.eig(P.T)                            # left eigvecs of P = right of P^T (stationary etc.)
    order = np.argsort(-w.real)
    w = w[order].real; V = V[:, order].real
    # candidate real-positive eigenvalues below the stationary mode (index 0 ~ 1.0)
    lam = w[1:]
    lam = lam[lam > 1e-6]
    K, flagged = 4, True
    if len(lam) >= 2:
        lo, hi = band
        drops = lam[:-1] / (lam[1:] + 1e-12)
        cand = range(max(1, lo - 1), min(hi, len(drops)))
        if cand:
            i_star = max(cand, key=lambda i: drops[i])
            med = np.median(drops[:max(hi, len(drops))]) + 1e-12
            if drops[i_star] >= drop_ratio * med:
                K, flagged = i_star + 1, False
    K = int(max(1, min(K, V.shape[1] - 1)))
    psi = V[:, 1:1 + K]
    psi = (psi - psi.mean(0)) / (psi.std(0) + 1e-8)     # standardize each macro column (a real coordinate choice)
    return K, flagged, w, psi


def resolved_series(labels, bounds, psi):
    """The slow (macro) coordinate a(t) along corpus trajectories: a[t] = psi[chart_t]."""
    return [psi[labels[b0:b1]] for b0, b1 in zip(bounds[:-1], bounds[1:])]


def _damped_cos(t, A, g, wcyc, c):
    return A * np.exp(-g * t) * np.cos(wcyc * t) + c


def fit_kernel(a_trajs, max_lag=40):
    """Autocorrelation of the resolved coordinate -> damped-oscillator fit. Returns (gamma, omega, autocorr).
    Track-held-out CV picks whether a real oscillatory mode beats a pure decay (order selection, cap tiny)."""
    from scipy.optimize import curve_fit
    def autocorr(trajs):
        acc = np.zeros(max_lag + 1); cnt = np.zeros(max_lag + 1)
        for a in trajs:
            x = a[:, 0] - a[:, 0].mean()                 # first macro
            for L in range(max_lag + 1):
                if len(x) > L:
                    acc[L] += np.dot(x[:len(x) - L], x[L:]); cnt[L] += len(x) - L
        c = acc / (cnt + 1e-12)
        return c / (c[0] + 1e-12)
    ntr = len(a_trajs)
    tr = a_trajs[:max(1, int(0.7 * ntr))]; te = a_trajs[max(1, int(0.7 * ntr)):] or tr
    c_tr = autocorr(tr); c_te = autocorr(te); t = np.arange(max_lag + 1)
    # order selection: damped cosine (oscillatory) vs pure decay, by held-out fit error
    try:
        po, _ = curve_fit(_damped_cos, t, c_tr, p0=[1, 0.05, 0.3, 0], maxfev=8000,
                          bounds=([0, 0, 0, -1], [2, 2, np.pi, 1]))
        err_osc = np.mean((_damped_cos(t, *po) - c_te) ** 2)
    except Exception:
        po, err_osc = [1, 0.1, 0.0, 0], 1e9
    pd = np.polyfit(t, np.log(np.clip(np.abs(c_tr), 1e-3, None)), 1)   # pure exp decay
    g_decay = max(1e-3, -pd[0]); err_dec = np.mean((np.exp(-g_decay * t) - c_te) ** 2)
    if err_osc <= err_dec:
        return float(po[1]), float(po[2]), autocorr(a_trajs)
    return float(g_decay), 0.0, autocorr(a_trajs)       # omega=0 => pure damping, no oscillatory memory


def build_instrument(states, bounds, k=64, seed=0):
    labels, cent, mu, sd, Z = build_atlas(states, k, seed)
    kk = cent.shape[0]
    P = transfer_operator(labels, bounds, kk)
    K, flagged, eig, psi = spectral_macros(P)
    a_trajs = resolved_series(labels, bounds, psi)
    gamma, omega, ac = fit_kernel(a_trajs)
    return {"labels": labels, "centroids": cent, "mu": mu, "sd": sd, "P": P, "K": K,
            "gap_flagged": bool(flagged), "eigvals": eig, "psi": psi, "gamma": gamma,
            "omega": omega, "autocorr": ac, "states": states, "bounds": bounds}
