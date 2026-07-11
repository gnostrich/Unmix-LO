"""
correlated_read.py — the nuanced reader: recover a watcher's memory when the input is NOT white.

The white-drive demo (watched.py) handed the reader a white input, so the fit-free Markov estimate
  h_k = (1/T) sum_t y_{t+k} u_t^T
IS the memory kernel. The moment the input is a real feature scraped off the film (correlated in time,
because the physics is smooth), that same estimate is the cross-correlation R_yf(k), which is the true
kernel CONVOLVED with the input's own autocorrelation R_ff. Read naively, the recovered poles are pulled
toward the input's color — you read the input's spectrum, not the watcher's.

The fix is a deconvolution: the Markov params h_{0..K} are exactly the least-squares regression of y_t on the
stacked lags [f_t, f_{t-1}, ..., f_{t-K}]. For white f this reduces to the old estimator; for colored f it
divides out R_ff. Everything downstream (Hankel -> permutation floor -> Ho-Kalman -> poles) is reused
unchanged from stream_trace. The one honest cost: a ridge is needed when R_ff is near-singular (very smooth
input), which is a knob the white reader did not have; we keep it tiny and disclose it.

numpy only. `python correlated_read.py` runs the calibration (white control + colored control with known
poles: naive must fail, deconvolving must recover).
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "io_trace"))
import stream_trace as ST   # reuse block_hankel, ho_kalman, permutation-floor idea, pole_match_error

FLOOR_SHIFTS = 40
FLOOR_Q = 99.0


def est_markov_deconv(f, y, kmax, ridge=1e-3):
    """h_{0..kmax} = OLS regression of y_t on [f_t, f_{t-1}, ..., f_{t-kmax}]. Deconvolves R_ff.
    ridge is relative to the mean diagonal of the normal matrix (tiny; only for conditioning)."""
    T, p = f.shape
    q = y.shape[1]
    K = kmax
    # design: Phi[t] = [f_t, f_{t-1}, ..., f_{t-K}], valid for t >= K
    rows = T - K
    Phi = np.empty((rows, (K + 1) * p))
    for k in range(K + 1):
        Phi[:, k * p:(k + 1) * p] = f[K - k: T - k]
    Yt = y[K:T]
    G = Phi.T @ Phi
    lam = ridge * np.trace(G) / G.shape[0]
    H = np.linalg.solve(G + lam * np.eye(G.shape[0]), Phi.T @ Yt)   # ((K+1)p, q)
    h = np.zeros((K + 1, q, p))
    for k in range(K + 1):
        h[k] = H[k * p:(k + 1) * p, :].T
    return h


def _deconv_floor(f, y, kmax, L, ridge, seed=0):
    """Self-calibrating floor: circularly shift y (destroy y|f temporal link, keep marginals), refit, take
    the top Hankel singular value of the memory part. Floor = FLOOR_Q percentile of the null draws."""
    rng = np.random.default_rng(seed)
    T = len(y)
    tops = []
    for _ in range(FLOOR_SHIFTS):
        s = int(rng.integers(T // 4, 3 * T // 4))
        hn = est_markov_deconv(f, np.roll(y, s, axis=0), kmax, ridge)
        tops.append(np.linalg.svd(ST.block_hankel(hn, L), compute_uv=False)[0])
    return float(np.percentile(tops, FLOOR_Q))


def read_trace_deconv(f, y, kmax=25, L=12, seed=0, ridge=1e-3):
    """Same natural read as stream_trace.read_trace (atoms above the permutation floor + poles), but the
    Markov estimate deconvolves the (colored) input first."""
    h = est_markov_deconv(f, y, kmax, ridge)
    sv = np.linalg.svd(ST.block_hankel(h, L), compute_uv=False)
    floor = _deconv_floor(f, y, kmax, L, ridge, seed=seed)
    order = int((sv > floor).sum())
    gap = float(sv[order - 1] / (sv[order] + 1e-15)) if 0 < order < len(sv) else 0.0
    poles = np.array([])
    if order > 0:
        A, _, _ = ST.ho_kalman(h, order, L)
        poles = np.linalg.eigvals(A)
    return {"order": order, "poles": poles, "gap": gap, "floor": floor, "svals": sv, "h": h}


# ---------------- calibration ----------------
P_IN, Q_OUT, NOISE = 3, 3, 0.05


def make_watcher(r, seed, rho=0.85):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(r, r)); A *= rho / np.max(np.abs(np.linalg.eigvals(A)))
    B = rng.normal(size=(r, P_IN)) / np.sqrt(P_IN)
    C = rng.normal(size=(Q_OUT, r)) / np.sqrt(r)
    return A, B, C, np.linalg.eigvals(A)


def drive_through(A, B, C, f, seed=0):
    rng = np.random.default_rng(seed)
    r = A.shape[0]; x = np.zeros(r); ys = []
    for t in range(len(f)):
        ys.append(C @ x + NOISE * rng.normal(size=Q_OUT))
        x = A @ x + B @ f[t]
    return np.array(ys)


def colored_input(T, a, seed=0):
    """AR(1) colored input per channel: f_t = a f_{t-1} + sqrt(1-a^2) eps_t (unit variance, known color)."""
    rng = np.random.default_rng(seed)
    e = rng.normal(size=(T, P_IN)); f = np.zeros((T, P_IN))
    for t in range(1, T):
        f[t] = a * f[t - 1] + np.sqrt(1 - a * a) * e[t]
    return f


def main():
    T, r = 12000, 4
    A, B, C, tp = make_watcher(r, seed=1)

    print("=" * 74)
    print("CONTROL 1 — WHITE input: deconvolving reader must match the white reader (recover the poles)")
    print("=" * 74)
    fw = np.random.default_rng(0).normal(size=(T, P_IN))
    yw = drive_through(A, B, C, fw, seed=0)
    r_naive = ST.read_trace(fw, yw, seed=0)
    r_dec = read_trace_deconv(fw, yw, seed=0)
    print(f"  naive  : order {r_naive['order']} (truth {r}), pole err {ST.pole_match_error(tp, r_naive['poles']):.3f}")
    print(f"  deconv : order {r_dec['order']} (truth {r}), pole err {ST.pole_match_error(tp, r_dec['poles']):.3f}")

    print("\n" + "=" * 74)
    print("CONTROL 2 — COLORED input (AR(1), a=0.9): naive reads the input's color, deconv recovers")
    print("=" * 74)
    fc = colored_input(T, a=0.9, seed=0)
    yc = drive_through(A, B, C, fc, seed=0)
    r_naive = ST.read_trace(fc, yc, seed=0)
    r_dec = read_trace_deconv(fc, yc, seed=0)
    en, ed = ST.pole_match_error(tp, r_naive["poles"]), ST.pole_match_error(tp, r_dec["poles"])
    print(f"  naive  : order {r_naive['order']} (truth {r}), pole err {en:.3f}   "
          f"poles~{np.round(np.sort_complex(r_naive['poles']), 2) if len(r_naive['poles']) else '[]'}")
    print(f"  deconv : order {r_dec['order']} (truth {r}), pole err {ed:.3f}   "
          f"poles~{np.round(np.sort_complex(r_dec['poles']), 2) if len(r_dec['poles']) else '[]'}")
    print(f"\n  true watcher poles ~{np.round(np.sort_complex(tp), 2)}")
    print(f"  VERDICT: deconv beats naive on colored input: {ed < en - 0.02}  (naive err {en:.3f} -> deconv {ed:.3f})")


if __name__ == "__main__":
    main()
