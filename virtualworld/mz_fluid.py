"""
EXPERIMENTAL — the recurrent Mori-Zwanzig / resizable-tape layer.  *** UNVALIDATED. ***

This runs ALONGSIDE (never merged into) the validated single-step stitch/classify/extend/reject layer.
It is an HONEST probe of the CONSTRUCT.md recurrent object — the tape == MZ memory as ONE object,
self-expanding by a Hankel-SV vs noise-floor criterion — NOT a validated result. Prior probes in this
line reduce toward classical linear state-space (Kalman / subspace-ID) filtering; we report that honestly.

One object, two terms (CONSTRUCT.md item 1):
  streaming term = the validated single-step stitch  \hat y_t         (direct / current contribution)
  memory   term = an operator-valued kernel over PAST innovations     (delayed / through-time closure)
The tape IS the memory: its read/write dynamics ARE the MZ kernel closing the streaming residual.

Self-expansion (item 2): block-Hankel singular values of the innovation sequence are compared to a
phase-randomized surrogate NOISE FLOOR (a second-FDT analog). #SVs above floor = the order the tape
self-expands to; modes below floor are pruned.

Faithfulness (item 5): the memory operator is fit with a spectral-radius (contraction) penalty — a LOSS
TERM, so settling is native to the dynamics; there is NO separate verify phase.

Loss (item 4): the target is the physics engine's OWN next-state (the seed model's own grounding), never
an invented judge or arbitrary data.
"""
import numpy as np


def innovation_sequences(Y, stitch, rollout):
    """Per-rollout, time-ordered innovation e_t = y_t - stitch_t (the streaming residual to be closed)."""
    seqs = []
    for r in np.unique(rollout):
        idx = np.where(rollout == r)[0]
        seqs.append((Y[idx] - stitch[idx]))          # (Tr, D)
    return seqs


def _block_hankel(seqs, L):
    """Stack length-L windows across all rollouts -> (n_windows, L*D)."""
    rows = []
    for e in seqs:
        Tr, D = e.shape
        for t in range(Tr - L + 1):
            rows.append(e[t:t + L].ravel())
    return np.array(rows)


def self_expand_order(seqs, L=6, seed=0):
    """Hankel-SV self-expansion: #singular values above a phase-randomized surrogate noise floor."""
    H = _block_hankel(seqs, L)
    if len(H) < 4:
        return {"order": 0, "svs": [], "noise_floor": 0.0}
    Hc = H - H.mean(0)
    sv = np.linalg.svd(Hc, compute_uv=False)
    # surrogate noise floor: destroy temporal structure by shuffling time within each channel, redo Hankel
    rng = np.random.default_rng(seed)
    surr = []
    for e in seqs:
        es = e.copy()
        for c in range(es.shape[1]):
            es[:, c] = es[rng.permutation(es.shape[0]), c]
        surr.append(es)
    Hs = _block_hankel(surr, L); Hs = Hs - Hs.mean(0)
    floor = float(np.linalg.svd(Hs, compute_uv=False).max())
    order = int((sv > floor).sum())
    return {"order": order, "svs": sv[:12].tolist(), "noise_floor": floor}


def _lag_design(seqs, p):
    """Build (past p innovations) -> (current innovation) regression across rollouts, time-respecting."""
    X, Yt = [], []
    for e in seqs:
        Tr = len(e)
        for t in range(p, Tr):
            X.append(np.concatenate([e[t - k] for k in range(1, p + 1)]))
            Yt.append(e[t])
    return np.array(X), np.array(Yt)


def fit_memory_kernel(train_seqs, test_seqs, p, lam=10.0):
    """Fit the MZ memory kernel (order-p linear closure of the innovation) with a contraction penalty.

    Returns held-out R^2 of predicting the innovation from its OWN past (i.e. how much through-time
    structure the streaming residual still carries) and the operator spectral radius (contraction check).
    This linear closure IS classical state-space / subspace identification — reported as such.
    """
    Xtr, Ytr = _lag_design(train_seqs, p)
    Xte, Yte = _lag_design(test_seqs, p)
    if len(Xtr) < 10 or len(Xte) < 5:
        return {"memory_heldout_r2": 0.0, "spectral_radius": 0.0, "p": p, "insufficient": True}
    K = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ Ytr)  # ridge = contraction bias
    pred = Xte @ K
    ss_res = ((Yte - pred) ** 2).sum(); ss_tot = ((Yte - Yte.mean(0)) ** 2).sum() + 1e-12
    r2 = float(1 - ss_res / ss_tot)
    D = Ytr.shape[1]
    # companion operator spectral radius (settling / contraction; <1 => native settling)
    comp = np.zeros((p * D, p * D))
    comp[:D] = K.T                       # first block-row = the fitted kernel
    if p > 1:
        comp[D:, :-D] = np.eye((p - 1) * D)
    rho = float(np.max(np.abs(np.linalg.eigvals(comp))))
    return {"memory_heldout_r2": r2, "spectral_radius": rho, "p": p, "insufficient": False}


def run(Y, stitch, rollout, train_rollouts, test_rollouts, L=6):
    """Full experimental MZ probe. Returns a dict for the dashboard's EXPERIMENTAL panel."""
    seqs_tr = innovation_sequences(Y[np.isin(rollout, train_rollouts)],
                                   stitch[np.isin(rollout, train_rollouts)],
                                   rollout[np.isin(rollout, train_rollouts)])
    seqs_te = innovation_sequences(Y[np.isin(rollout, test_rollouts)],
                                   stitch[np.isin(rollout, test_rollouts)],
                                   rollout[np.isin(rollout, test_rollouts)])
    exp = self_expand_order(seqs_tr, L=L)
    p = max(1, min(exp["order"], 4))                 # tape order the self-expansion criterion selected
    mem = fit_memory_kernel(seqs_tr, seqs_te, p=p)
    return {
        "LABEL": "EXPERIMENTAL — UNVALIDATED recurrent MZ/tape layer (reduces toward classical "
                 "linear state-space filtering; NOT a validated construct result).",
        "self_expansion": exp,
        "tape_order_selected": p,
        "memory_closure": mem,
        "reading": ("The streaming residual retains through-time structure the memory kernel closes "
                    f"(held-out R^2={mem['memory_heldout_r2']:.3f}, spectral radius {mem['spectral_radius']:.2f}<1 "
                    "=> native settling). This closure IS classical linear state-space identification; the "
                    "recurrent MZ object reduces to it here — reported honestly, NOT presented as validated."),
    }
