"""
UNMIX extractor — the prototype spine.

Separates reusable weight-space operators from a pool of per-task gradient clouds via ICA,
and runs the three gate checks (STABLE / INDIVIDUAL / REUSED). Runs on synthetic data out of
the box; point `load_real_gradients` at real LoRA gradients to run the actual gate (see GATE.md).
"""
import numpy as np
from sklearn.decomposition import FastICA
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------- core extraction
def whiten_project(X, r):
    """Center + PCA-project to r dims (conditions ICA; mirrors low-rank operator assumption)."""
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    r = min(r, Vt.shape[0])
    return Xc @ Vt[:r].T, Vt[:r]           # projected data, projection basis (r x P)


def extract(X, n_components, seed=0):
    """X: (n_samples, P) pooled gradients. Returns components as directions in original P-space."""
    ica = FastICA(n_components=n_components, whiten="unit-variance",
                  max_iter=2000, random_state=seed)
    ica.fit(X)
    A = ica.mixing_                         # (P, n_components)
    A = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)
    return A, ica


def match(A, B):
    """Best 1-1 assignment cosine between two component sets (columns). Returns mean matched |cos|."""
    C = np.abs(A.T @ B)
    r, c = linear_sum_assignment(-C)
    return C[r, c].mean(), (r, c)


# ---------------------------------------------------------------- the three gate checks
def check_stable(X, n_components, n_boot=5):
    """Re-extract on bootstrap resamples; components stable if matched cosine across runs is high."""
    base, _ = extract(X, n_components, seed=0)
    sims = []
    for b in range(1, n_boot + 1):
        idx = np.random.default_rng(b).integers(0, len(X), len(X))
        Ab, _ = extract(X[idx], n_components, seed=b)
        sims.append(match(base, Ab)[0])
    return float(np.mean(sims)), base          # >~0.8 = stable


def check_individual(A, loadings):
    """Individual iff (a) low pairwise overlap (not 0.707-fused) and (b) non-Gaussian, sparse loadings."""
    K = A.shape[1]
    off = np.abs(A.T @ A)[~np.eye(K, dtype=bool)]
    max_pair_overlap = float(off.max())        # high (~0.7) => fused pair present
    # excess kurtosis of loadings: >0 = non-Gaussian (separable source); ~0 = Gaussian (unseparable)
    L = loadings - loadings.mean(0, keepdims=True)
    kurt = ((L**4).mean(0) / ((L**2).mean(0)**2 + 1e-12)) - 3.0
    return max_pair_overlap, float(np.median(kurt))


def check_reused(train_clouds, held_clouds, basis, A, thr=0.15):
    """
    Fit library on train tasks; test whether held-out task gradients reconstruct as a SPARSE
    combination of the SAME components, and whether components recur across tasks.
    Returns (mean held-out reconstruction residual, mean #active components per held task).
    """
    resid, nact = [], []
    for cloud in held_clouds:
        Xc = (cloud - cloud.mean(0, keepdims=True)) @ basis.T   # project to library space
        # each component's activation on this task = variance of loadings along it
        load = Xc @ A                                            # (n, K)
        act = load.var(0)
        active = act > thr * act.max()
        nact.append(int(active.sum()))
        # reconstruct using only active components
        Aa = A[:, active]
        proj = Xc @ Aa @ np.linalg.pinv(Aa)
        resid.append(float(np.linalg.norm(Xc - proj)**2 / (np.linalg.norm(Xc)**2 + 1e-12)))
    return float(np.mean(resid)), float(np.mean(nact))


# ---------------------------------------------------------------- data hooks
def synthetic_pool(n_prim=8, n_domains=6, D=32, M=400, seed=0, fused_everywhere=False):
    """Genre-structured synthetic gradients: shared primitives fused with DIFFERENT partners per domain."""
    rng = np.random.default_rng(seed)
    S, _ = np.linalg.qr(rng.normal(0, 1, (D, n_prim))); S = S[:, :n_prim]
    clouds = []
    for d in range(n_domains):
        active = list(rng.choice(n_prim, size=3, replace=False))
        a, b = (0, 1) if fused_everywhere else (active[0], active[1])
        X = []
        for _ in range(M):
            loads = {i: rng.laplace() for i in active}
            loads[b] = loads[a]                         # within-domain lock
            v = sum(loads[i] * S[:, i] for i in active) + rng.normal(0, 0.02, D)
            X.append(v)
        clouds.append(np.array(X))
    return clouds, S


def load_real_gradients(path):
    """GATE.md step 2: load grads/{genre}/{task}.npy -> (list of (n, P) arrays, list of (genre, task))."""
    import os, glob
    clouds, labels = [], []
    for f in sorted(glob.glob(os.path.join(path, "*", "*.npy"))):
        genre = os.path.basename(os.path.dirname(f))
        task = os.path.splitext(os.path.basename(f))[0]
        clouds.append(np.load(f).astype(np.float64))
        labels.append((genre, task))
    if not clouds:
        raise FileNotFoundError(f"no gradient clouds under {path} — run gate/collect_grads.py first")
    return clouds, labels


# ---------------------------------------------------------------- demo / self-test
if __name__ == "__main__":
    print("UNMIX extractor self-test on genre-structured synthetic gradients\n")
    clouds, S = synthetic_pool()
    X = np.vstack(clouds)
    Xp, basis = whiten_project(X, r=S.shape[1])
    K = S.shape[1]

    stab, A = check_stable(Xp, K)
    # recover components back in D-space to compare to ground truth S
    A_D = basis.T @ A; A_D /= (np.linalg.norm(A_D, axis=0, keepdims=True) + 1e-12)
    truth = match(S, A_D)[0]
    maxpair, kurt = check_individual(A, Xp @ A)
    train, held = clouds[:4], clouds[4:]
    resid, nact = check_reused(train, held, basis, A)

    print(f"  STABLE     : cross-run matched cosine = {stab:.3f}   (>0.8 good)")
    print(f"  INDIVIDUAL : max pairwise overlap = {maxpair:.3f} (~0.7 = fused/bad),"
          f" median loading excess-kurtosis = {kurt:.2f} (>0 = separable)")
    print(f"  REUSED     : held-out recon residual = {resid:.3f} (low good),"
          f" active comps/task = {nact:.1f} (sparse good)")
    print(f"  [ground-truth recovery vs planted primitives = {truth:.3f}]")
    print("\n  -> replace synthetic_pool with load_real_gradients (GATE.md) for the real decision.")
