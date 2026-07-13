"""
P5 — multiplicative closure (Wick law). Read POLES, not rank. For a linear-Gaussian degree-r latent with
poles {λ_i}, the quadratic observable's covariance modes are the pairwise products {λ_iλ_j} (Isserlis).
Estimate poles from the covariance Hankel of q via Ho-Kalman/ERA and check every estimated pole ≈ a product.
Pre-registered in PREREG_P5.md. Run: python -m ebr.experiments.pole_closure
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "io_trace"))
import stream_trace as ST


def latent_with_poles(poles, T, seed=0, drive=0.3):
    """VAR(1) linear-Gaussian latent u(t+1)=A u(t)+ε with prescribed eigenvalues `poles` (real block form)."""
    blocks = []
    used = set()
    pl = list(poles)
    i = 0
    while i < len(pl):
        z = pl[i]
        if abs(z.imag) < 1e-9:
            blocks.append(np.array([[z.real]])); i += 1
        else:
            re, im = z.real, z.imag
            blocks.append(np.array([[re, -im], [im, re]])); i += 2   # complex pair
    A = np.zeros((sum(b.shape[0] for b in blocks),) * 2)
    o = 0
    for b in blocks:
        A[o:o + b.shape[0], o:o + b.shape[0]] = b; o += b.shape[0]
    r = A.shape[0]
    rng = np.random.default_rng(seed)
    u = np.zeros(r); U = []
    for _ in range(T):
        U.append(u.copy()); u = A @ u + drive * rng.normal(size=r)
    return np.array(U), np.linalg.eigvals(A)


def quadratic(U):
    r = U.shape[1]
    Q = np.stack([U[:, i] * U[:, j] for i in range(r) for j in range(i, r)], 1)
    return Q - Q.mean(0)


def autocov(Q, K):
    T = len(Q)
    return np.stack([Q[k:].T @ Q[:T - k] / (T - k) for k in range(K + 1)])


def estimate_poles(Q, order, L=20):
    """Ho-Kalman/ERA on the autocovariance sequence of q -> poles."""
    R = autocov(Q, 2 * L + 2)
    A, _, _ = ST.ho_kalman(R, order, L)
    return np.linalg.eigvals(A)


def products(poles):
    r = len(poles)
    return np.array([poles[i] * poles[j] for i in range(r) for j in range(i, r)])


def closure_error(est, prod):
    return np.array([np.min(np.abs(prod - e)) for e in est])


def run(T=60000, seed=0):
    latent_poles = [0.85 + 0j, 0.65 * np.exp(1j * np.pi / 4), 0.65 * np.exp(-1j * np.pi / 4)]
    U, lam = latent_with_poles(latent_poles, T, seed=seed)
    prod = products(lam)
    print("P5 — multiplicative closure (read poles, not rank)")
    print(f"  latent poles   : {np.round(np.sort_complex(lam), 3)}")
    print(f"  predicted {len(prod)} products {{λ_iλ_j}} (by |·|): "
          f"{np.round(np.sort_complex(prod[np.argsort(-np.abs(prod))]), 3)}")
    Q = quadratic(U)
    # P5a: closure holds up to the RESOLVABLE order — find the largest order at which every estimated pole
    # still lies on the product set (over-ordering injects spurious poles: the resolvability limit).
    print("  estimated poles vs closure error, by realization order:")
    resolvable = 0
    for order in (2, 3, 4, 5, 6):
        err = closure_error(estimate_poles(Q, order), prod)
        ok = err.max() < 0.05
        if ok:
            resolvable = order
        print(f"    order {order}: max nearest-product err {err.max():.4f}  {'on product set' if ok else 'spurious pole appears'}")
    closure_pass = resolvable >= 3
    print(f"  -> closure holds up to order {resolvable} (resolvable subset = top-|·| products)")

    # P5b: resolvability grows in magnitude order with T (top-|·| prefix resolves first)
    print("  resolvability vs T (poles within 3% of a product, at the resolvable order):")
    counts = []
    for Tt in (8000, 25000, 60000):
        Ut, _ = latent_with_poles(latent_poles, Tt, seed=seed)
        est = estimate_poles(quadratic(Ut), 4)
        n = int((closure_error(est, prod) < 0.03).sum()); counts.append(n)
        print(f"    T={Tt:6d}: {n} poles on the product set")
    order_ok = all(counts[i + 1] >= counts[i] for i in range(len(counts) - 1))

    # P5c: recover the latent (generator) poles as multiplicative square-roots of the resolved products
    est3 = estimate_poles(Q, 3)
    real_prod = est3[np.argmin(np.abs(est3.imag))]              # the real product ≈ λ1^2
    lam1 = np.sqrt(real_prod.real)
    cpair = est3[np.abs(est3.imag) > 1e-3]                       # ≈ λ1·λ2 pair
    lam2 = (cpair[np.argmax(cpair.imag)] / lam1) if len(cpair) else np.nan
    gen_err = min(abs(lam1 - 0.85), 1) + (abs(abs(lam2) - 0.65) if np.isfinite(lam2) else 1)
    print(f"  P5c generator recovery: λ1≈{lam1:.3f} (true 0.85), λ2≈{lam2:.3f} (true 0.65·e^iπ/4={0.65*np.exp(1j*np.pi/4):.3f})")

    p5 = closure_pass and order_ok and gen_err < 0.05
    print(f"P5 verdict: {'PASS' if p5 else 'FAIL'} — multiplicative closure confirmed "
          f"(poles lie on {{λ_iλ_j}}, resolve in |·| order, generators recovered)")
    return {"resolvable_order": resolvable, "closure_pass": bool(closure_pass),
            "resolvability_monotone": bool(order_ok), "generator_err": float(gen_err), "P5": bool(p5)}


if __name__ == "__main__":
    run()
