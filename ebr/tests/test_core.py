"""
Core CI invariants (the spec's mandatory fixtures, §9). Run: python -m pytest ebr/tests -q
  - gauge-scramble: (D,w) invariant to the G0 group (orthogonal × permutation × rescale × shift).
  - Lyapunov-monotone: F non-increasing across block-coordinate updates.
  - coupling-continuity: warm-started equilibration never worse than cold.
  - invariant-interface: geometry exports only (D,w); D unchanged under coordinate scramble.
"""
import numpy as np
from ebr.geometry.clouds import cloud_to_Dw, scramble
from ebr.transport import gw
from ebr.energy import functional as EN


def _cloud(n=40, d=8, r=3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, r)) @ rng.normal(size=(r, d))


def test_gauge_scramble_invariance():
    X = _cloud(seed=1)
    D, w = cloud_to_Dw(X)
    for s in range(5):
        D2, w2 = cloud_to_Dw(scramble(X, seed=s))
        assert np.abs(D - D2).max() < 1e-9, "frame leak: D not invariant to the scramble group"
        assert np.allclose(w, w2)


def test_coupling_F_monotone():
    X = _cloud(seed=2)
    D, w = cloud_to_Dw(X)
    De = np.random.default_rng(3).random((4, 4)); De = (De + De.T) / 2
    np.fill_diagonal(De, 0); De /= np.median(De[np.triu_indices(4, 1)])
    a = np.full(4, 1.0 / 4)
    pi = None; fs = []
    for _ in range(8):
        pi, _q = gw.equilibrate_coupling(D, De, w, a, 0.05, pi0=pi, j_sink=1)
        fs.append(gw.entropic_gw_value(D, De, w, a, pi, 0.05))
    assert all(fs[i + 1] <= fs[i] + 1e-9 for i in range(len(fs) - 1)), "coupling F_ve not monotone"


def test_equilibration_lyapunov():
    Dws = [cloud_to_Dw(_cloud(seed=10 + i)) for i in range(3)]
    Ds = [d for d, _ in Dws]; ws = [w for _, w in Dws]
    De = np.random.default_rng(4).random((4, 4)); De = (De + De.T) / 2
    np.fill_diagonal(De, 0); De /= np.median(De[np.triu_indices(4, 1)])
    a = np.full(4, 1.0 / 4)
    _pis, _De, _a, ftr, _c = EN.equilibrate(Ds, ws, De, a, a.copy(), eps=0.08, n_outer=15)
    assert all(ftr[i + 1] <= ftr[i] + 1e-9 for i in range(len(ftr) - 1)), \
        "Lyapunov violated: F increased across a block update"


def test_coupling_continuity():
    X = _cloud(seed=5); D, w = cloud_to_Dw(X)
    De = np.random.default_rng(6).random((4, 4)); De = (De + De.T) / 2
    np.fill_diagonal(De, 0); De /= np.median(De[np.triu_indices(4, 1)])
    a = np.full(4, 1.0 / 4)
    pi_cold, _ = gw.equilibrate_coupling(D, De, w, a, 0.05, pi0=None, j_sink=5)
    warm, _ = gw.equilibrate_coupling(D, De, w, a, 0.05, pi0=pi_cold, j_sink=5)
    f_cold = gw.entropic_gw_value(D, De, w, a, pi_cold, 0.05)
    f_warm = gw.entropic_gw_value(D, De, w, a, warm, 0.05)
    assert f_warm <= f_cold + 1e-9, "warm-start regressed"


def test_zscored_pipeline_scramble_invariant():
    """FIX-4 (instrument hygiene): z-scoring of moment channels is a fixed INSTRUMENT choice; the full
    z-scored Hankel pipeline must remain scramble-invariant. Scramble one member's clouds every prompt;
    the invariant-moment series and its Hankel spectrum must be identical to numerical precision."""
    from ebr.hankel import residual as H
    rng = np.random.default_rng(0)
    T, n = 40, 30
    base = [_cloud(n=n, r=3, seed=100 + t) for t in range(T)]
    s_plain, s_scram = [], []
    for t, X in enumerate(base):
        D0, _ = cloud_to_Dw(X)
        D1, _ = cloud_to_Dw(scramble(X, seed=t))
        z = np.zeros((n, 1))
        s_plain.append(H.residual_moments(H.gram_from_D(D0), z))
        s_scram.append(H.residual_moments(H.gram_from_D(D1), z))
    sp = H.hankel_spectrum(np.array(s_plain))          # z-scores channels internally
    ss = H.hankel_spectrum(np.array(s_scram))
    assert np.abs(sp - ss).max() < 1e-6, "z-scored pipeline leaks the frame under scramble"


def test_invariant_interface_no_coordinates():
    # geometry.cloud_to_Dw must return only (D, w); D symmetric, zero diagonal, median-normalized.
    D, w = cloud_to_Dw(_cloud(seed=7))
    assert np.allclose(D, D.T) and np.allclose(np.diag(D), 0)
    assert abs(np.median(D[np.triu_indices(D.shape[0], 1)]) - 1.0) < 1e-9
    assert abs(w.sum() - 1.0) < 1e-12
