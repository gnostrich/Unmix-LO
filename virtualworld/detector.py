"""
DETECTOR — the validated structured/noise classifier for a decoherence residual.

This reuses the thoughtworld-validated instrument logic (held-out predictivity of a residual from the
world state) and adds the low-rank captured-vs-baseline signal. Per the smoke_loop calibration note,
captured-vs-baseline ALONE fails at small D (baseline eff/D is too high); the robust second condition
is HELD-OUT R^2 predicting the decoherence FROM THE WORLD STATE. STRUCTURED requires BOTH:

  (A) reproducible low-rank subspace : the top-eff-rank subspace fit on TRAIN captures, on HELD-OUT,
      markedly more energy than a random subspace of the same size would (captured > MARGIN * base).
  (B) held-out predictable from state: fit  d ~ s_cur @ Wmap  on TRAIN, held-out R^2 >= R2_MIN.

NOISE = fails (B) (state-independent scatter). This cleanly separates a state-dependent structured
hidden distinction (predictable, low-rank) from injected random noise (unpredictable, full-rank).
"""
import numpy as np

R2_MIN = 0.30
CAP_MARGIN = 1.3


def _eff_rank(sv):
    sv = sv[sv > 0]
    if sv.size == 0:
        return 1.0
    return float((sv.sum() ** 2) / (sv ** 2).sum())


def classify(d, s_cur, train, test, D=None, r2_min=R2_MIN, cap_margin=CAP_MARGIN):
    """Classify decoherence residual `d` (n, k) as STRUCTURED vs NOISE.

    d      : (n, k) residual (e.g. aligned_rep_m1 - aligned_rep_m2), in the shared-medium space.
    s_cur  : (n, Dw) world state per row (the predictor for condition B).
    train, test : index arrays into the rows.
    """
    if D is None:
        D = d.shape[1]
    dtr, dte = d[train], d[test]
    mu = dtr.mean(0)
    dtr_c, dte_c = dtr - mu, dte - mu

    # (A) low-rank reproducibility: train subspace generalises to held-out
    U, S, Vt = np.linalg.svd(dtr_c, full_matrices=False)
    eff = _eff_rank(S)
    k = max(1, int(round(eff)))
    P = Vt[:k].T @ Vt[:k]
    captured = float(((dte_c @ P) ** 2).sum() / ((dte_c ** 2).sum() + 1e-12))
    base = k / D                                   # a random k-subspace captures ~k/D of held-out energy
    lowrank_ok = captured > cap_margin * base

    # (B) held-out predictivity from world state (the robust condition)
    Xtr = np.concatenate([s_cur[train], np.ones((len(train), 1))], 1)
    Xte = np.concatenate([s_cur[test], np.ones((len(test), 1))], 1)
    Wmap = np.linalg.lstsq(Xtr, dtr, rcond=None)[0]
    pred = Xte @ Wmap
    ss_res = ((dte - pred) ** 2).sum()
    ss_tot = ((dte - dte.mean(0)) ** 2).sum() + 1e-12
    r2 = float(1 - ss_res / ss_tot)
    predictable = r2 >= r2_min

    structured = bool(lowrank_ok and predictable)
    return dict(
        eff_rank=eff, k=k, captured=captured, baseline=base, captured_vs_base=captured / (base + 1e-12),
        heldout_r2=r2, lowrank_ok=bool(lowrank_ok), predictable=bool(predictable),
        structured=structured, verdict=("STRUCTURED" if structured else "NOISE"),
        resid_norm=float(np.linalg.norm(dte) / np.sqrt(dte.size)),
    )
