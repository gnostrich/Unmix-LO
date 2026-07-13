"""
sympower.py — P1: does the invariant (relational, ≥ quadratic) observable's McMillan degree equal the
symmetric-power degree of the latent? Clean numpy test, no model/floor confounds.

Degree-r linear latent x(t+1)=A x(t) (generic eigenvalues). Observables:
  linear     : x_i(t)                          -> Koopman eigenvalues {λ_i}          -> degree r
  quadratic  : x_i x_j (i≤j)                   -> {λ_i λ_j}                          -> degree r(r+1)/2
  lin+quad   : both                            -> {λ_i} ∪ {λ_i λ_j}                  -> degree r(r+3)/2
Hankel rank of each observable series should equal the count of DISTINCT Koopman eigenvalues.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "io_trace"))
import stream_trace as ST


def _make_A(r, seed=0, rho=0.9):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(r, r)); A *= rho / np.max(np.abs(np.linalg.eigvals(A)))
    return A


def _traj_from(A, T, ic_seed):
    x = np.random.default_rng(1000 + ic_seed).normal(size=A.shape[0]); X = []
    for _ in range(T):
        X.append(x.copy()); x = A @ x
    return np.array(X)                          # same dynamics A, distinct initial condition


def _distinct(vals, tol=1e-6):
    vals = list(vals); keep = []
    for v in vals:
        if not any(abs(v - k) < tol for k in keep):
            keep.append(v)
    return len(keep)


def predicted_degree(lam, kind):
    """Exact distinct-Koopman-eigenvalue count for the observable (accounts for real-matrix coincidences)."""
    lin = list(lam)
    quad = [lam[i] * lam[j] for i in range(len(lam)) for j in range(i, len(lam))]
    if kind == "linear":
        return _distinct(lin)
    if kind == "quadratic":
        return _distinct(quad)
    return _distinct(lin + quad)


def _data_hankel(Y, L):
    W, p = Y.shape
    cols = W - L
    Hs = np.stack([Y[i:i + cols] for i in range(L + 1)])      # (L+1, cols, p)
    return Hs.transpose(0, 2, 1).reshape((L + 1) * p, cols)   # ((L+1)p, cols)


def _hankel_rank(series_list, L=16, tol=1e-6):
    """Data Hankel of a deterministic observable, columns pooled over several initial conditions so every
    Koopman mode is excited (residue-free). rank = distinct excited modes = McMillan degree."""
    Hm = np.concatenate([_data_hankel(np.asarray(Y, float), L) for Y in series_list], axis=1)
    sv = np.linalg.svd(Hm, compute_uv=False)
    return int((sv > tol * sv[0]).sum()), sv


def observables(X, kind):
    T, r = X.shape
    lin = X
    quad = np.stack([X[:, i] * X[:, j] for i in range(r) for j in range(i, r)], 1)
    if kind == "linear":
        return lin
    if kind == "quadratic":
        return quad
    return np.concatenate([lin, quad], 1)


def run(rs=(2, 3, 4), T=4000, L=16):
    print(f"P1 sym-power theorem (T={T}, L={L}) — measured rank vs exact distinct-Koopman-mode count:")
    print(f"{'r':>2} | {'linear':>14} | {'quadratic (invariant)':>22} | {'lin+quad':>16} | generic r(r+1)/2")
    out = {}; ok = True
    for r in rs:
        A = _make_A(r, seed=0)                                # ONE system, several ICs -> all modes excited
        lam = np.linalg.eigvals(A)
        trajs = [_traj_from(A, T, ic) for ic in range(4)]
        row = {}
        for kind in ("linear", "quadratic", "linquad"):
            got = _hankel_rank([observables(X, kind) for X in trajs], L)[0]
            pred = predicted_degree(lam, kind)
            row[kind] = (got, pred)
            if kind != "linear":
                ok &= (got == pred)
        out[r] = row
        print(f"{r:>2} | {row['linear'][0]:>4} (pred {row['linear'][1]}) | "
              f"{row['quadratic'][0]:>6} (pred {row['quadratic'][1]})        | "
              f"{row['linquad'][0]:>3} (pred {row['linquad'][1]})     | {r * (r + 1) // 2}")
    # the headline: invariant (quadratic) rank is the SYM-POWER degree, strictly super-linear in r
    q = [out[r]["quadratic"][0] for r in rs]
    superlin = all(q[i + 1] > q[i] for i in range(len(q) - 1)) and q[-1] > rs[-1]
    print(f"P1 verdict: {'PASS' if ok and superlin else 'FAIL'} — invariant rank = symmetric-power degree "
          f"(quadratic ranks {q} ≫ latent {list(rs)})")
    return out, bool(ok and superlin)


if __name__ == "__main__":
    run()
