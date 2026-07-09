"""
PROPER multivariate second-FDT estimator.

WHY multivariate: a scalar stationary Gaussian process can ALWAYS be written as an FDT-consistent GLE
(the residual-reconstruction test is circular -> both FDT-holds and FDT-fails scalar cases score 1.0).
Second-FDT / equilibrium is only falsifiable in the MULTIVARIATE dynamics:
  FDT holds  <=> the disagreement dynamics is (i) DISSIPATIVE-WITH-MEMORY (mean-reverting, bounded,
                 finite autocorrelation time: 0<|lambda|<1 -- rules out random walk |lambda|~1 [pure
                 fluctuation, no dissipation] and white noise |lambda|~0 [no memory]) AND
                (ii) REVERSIBLE / detailed-balance (no net probability current; the shared time-reversible
                 generator is inherited).  FDT-satisfying = generator-inherited content (F_gauge);
                 FDT-violating = undissipated or current-carrying idiosyncratic noise (F_noise).

Estimator (per data = list of within-rollout trajectories of vectors d_t in R^D):
  1. PCA-whiten d_t on the top-k modes -> stationary covariance C0 = I (clean reversibility test).
  2. Fit VAR(1) propagator Phi by least squares over within-rollout consecutive pairs (Mori projection;
     residual xi orthogonal to d_t by construction).
  3. Symmetric part S=(Phi+Phi^T)/2 (reversible dynamics), antisymmetric A=(Phi-Phi^T)/2 (currents).
  4. Eigendecompose S (real, orthogonal eigenvectors u_i, eigenvalues mu_i).
     per orthogonal mode i (equal variance under whitening):
       mem_i = 4*|mu_i|*(1-|mu_i|)  clamped[0,1]  (dissipative-WITH-memory quality; ->0 at |mu|=0 white
                                                    and |mu|=1 random-walk; peaks mid-relaxation)
       rev_i = ( ||S u_i|| / (||S u_i|| + CUR_WT*||A u_i||) )**REV_POW   (reversible fraction, sharpened)
       w_i   = mem_i * rev_i    (FDT-satisfying weight of the mode)
  FROZEN hyperparameters (set in STEP 0, before any real run): REV_POW=2, CUR_WT=2.0, thr=0.25, k modes.
  5. FDT-satisfying variance fraction = mean_i w_i   (equal variance per whitened mode).
     Also a hard count at a threshold, and the mean |mu| / current magnitude for reporting.
"""
import numpy as np


def _pairs(rolls, k_whiten=None):
    rolls = [np.asarray(r, float) for r in rolls]
    X = np.concatenate([r[:-1] for r in rolls], 0)
    Y = np.concatenate([r[1:] for r in rolls], 0)
    return X, Y


def _whiten(rolls, k):
    allx = np.concatenate([np.asarray(r, float) for r in rolls], 0)
    mu = allx.mean(0)
    C = np.cov((allx - mu).T)
    C = np.atleast_2d(C)
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    k = min(k, (w > 1e-9).sum())
    Wm = V[:, :k] / np.sqrt(w[:k] + 1e-12)   # whitening map (D x k)
    lam = w[:k]
    rolls_w = [((np.asarray(r, float) - mu) @ Wm) for r in rolls]
    return rolls_w, lam, k


REV_POW = 2      # FROZEN in STEP 0 before any real-model run
CUR_WT = 2.0
THR = 0.25


def fdt_fraction(rolls, k=10, thr=THR, ridge=1e-3):
    rolls_w, lam, k = _whiten(rolls, k)
    X, Y = _pairs(rolls_w)
    # VAR(1) least squares: Y ~ X Phi^T  -> Phi = (Y^T X)(X^T X)^-1
    XtX = X.T @ X + ridge * np.eye(k)
    Phi = (Y.T @ X) @ np.linalg.inv(XtX)          # k x k, Y_t = Phi X_{t-1}
    S = 0.5 * (Phi + Phi.T)
    A = 0.5 * (Phi - Phi.T)
    mu, U = np.linalg.eigh(S)                      # symmetric -> real
    mem, rev, w = [], [], []
    for i in range(k):
        ui = U[:, i]
        m = 4 * abs(mu[i]) * (1 - abs(mu[i]))
        m = float(np.clip(m, 0, 1))
        su = np.linalg.norm(S @ ui); au = np.linalg.norm(A @ ui)
        r = float((su / (su + CUR_WT * au + 1e-12)) ** REV_POW)
        mem.append(m); rev.append(r); w.append(m * r)
    w = np.array(w)
    frac_soft = float(w.mean())
    frac_hard = float((w >= thr).mean())
    return {"frac_soft": frac_soft, "frac_hard": frac_hard,
            "mean_absmu": float(np.mean(np.abs(mu))), "mean_rev": float(np.mean(rev)),
            "current_frac": float(np.linalg.norm(A) / (np.linalg.norm(Phi) + 1e-12)),
            "k": k, "mu": mu, "w": w}
