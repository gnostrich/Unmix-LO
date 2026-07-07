"""
GATE.md steps 3-4 — extract components from the pooled real gradient clouds and run the decision.

Reads grads/{genre}/{task}.npy (from collect_grads.py), then:
  STABLE     — bootstrap re-extraction, matched-component cosine. PASS >= 0.8.
  INDIVIDUAL — after regressing out task-generic directions: max pairwise |cos| LOW
               (fail near the ~0.707 fused signature), median loading excess-kurtosis > 0
               (near-Gaussian loadings = ICA cannot separate = a real null, not a bug),
               usage concentrated on some tasks, not smeared over all.
               PASS = max overlap < 0.65 AND kurtosis > 0 AND not smeared.
  REUSED     — hold out one task per genre; held-out clouds reconstruct sparsely and with low
               residual from the SAME library; components recur across >=2 genres; stability
               rises as genres are pooled. PASS = residual < 0.3 AND sparse AND cross-genre.

Thresholds are pre-committed (GATE.md + CONTEXT.md exp 03 guards); do not tune them to pass.
Writes gate/results.json and prints the verdict.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from extractor import (whiten_project, extract, match, check_stable,
                       check_individual, check_reused, load_real_gradients)

R = int(os.environ.get("GATE_R", 100))         # PCA subspace dim (GATE.md: ~50-200)
K = int(os.environ.get("GATE_K", 30))          # ICA components
GRADS = os.environ.get("GATE_GRADS", os.path.join(HERE, "..", "grads"))
OUT = os.environ.get("GATE_RESULTS", os.path.join(HERE, "results.json"))
ACT_THR = 0.15                                  # activity threshold (same as check_reused)
GENERIC_UNIFORMITY = 0.25                       # min/max per-task variance ratio => task-generic


def guarded_project(clouds, r, top=5):
    """PCA-project the pool, then regress out task-generic directions (integrity guard a).

    A PCA axis along which EVERY task varies comparably is task-generic (loss scale, shared
    token statistics — experiment 03's 'high absolute overlap is task-generic'); a compositional
    primitive is used by SOME tasks and near-silent in the rest. Drop generic axes among the
    top `top` before ICA."""
    X = np.vstack(clouds)
    Xp, basis = whiten_project(X, r)
    Xmean = X.mean(0, keepdims=True)
    cl_p = [(c - Xmean) @ basis.T for c in clouds]
    generic = []
    for d in range(min(top, Xp.shape[1])):
        v = np.array([cp[:, d].var() for cp in cl_p])
        if v.min() / (v.max() + 1e-12) > GENERIC_UNIFORMITY:
            generic.append(d)
    tot = Xp.var(0).sum()
    var_removed = float(Xp.var(0)[generic].sum() / tot) if generic else 0.0
    keep = [i for i in range(Xp.shape[1]) if i not in generic]
    return Xp[:, keep], basis[keep], [cp[:, keep] for cp in cl_p], generic, var_removed


def task_activation(A, cl_p):
    """(K, n_tasks) activation = per-task variance of loadings on each component, row-normalized."""
    M = np.array([(cp @ A).var(0) for cp in cl_p]).T
    return M / (M.max(1, keepdims=True) + 1e-12)


def main():
    clouds, labels = load_real_gradients(GRADS)
    genres = [g for g, _ in labels]
    genre_names = sorted(set(genres), key=genres.index)
    print(f"loaded {len(clouds)} task clouds from {GRADS}: {labels}")

    # per-cloud normalization so no single task dominates the pool by gradient scale
    clouds = [c / (np.linalg.norm(c, axis=1, keepdims=True).mean() + 1e-12) for c in clouds]
    Xp, basis, cl_p, generic, var_removed = guarded_project(clouds, R)
    print(f"pooled {sum(len(c) for c in clouds)} x {clouds[0].shape[1]} -> PCA r={R}; "
          f"guard removed {len(generic)} task-generic axis(es) = {100*var_removed:.0f}% of subspace "
          f"variance -> {Xp.shape[1]} dims, ICA K={K}\n")
    res = {"n_tasks": len(clouds), "labels": labels, "r": R, "K": K,
           "generic_axes_removed": generic, "generic_variance_share": var_removed}

    # ---- 1 STABLE
    stab, A = check_stable(Xp, K)
    res["stable_matched_cosine"] = stab
    print(f"1 STABLE     : bootstrap matched cosine = {stab:.3f}   (PASS >= 0.8)")

    # ---- 2 INDIVIDUAL (on generic-regressed data)
    maxpair, kurt = check_individual(A, Xp @ A)
    off = np.abs(A.T @ A)[~np.eye(A.shape[1], dtype=bool)]
    medpair = float(np.median(off))
    M = task_activation(A, cl_p)
    active = M > ACT_THR
    usage = active.sum(1)
    smeared = float((usage >= len(clouds) - 1).mean())
    res.update(max_pairwise_overlap=maxpair, median_pairwise_overlap=medpair,
               median_loading_kurtosis=kurt,
               mean_tasks_per_component=float(usage.mean()), frac_smeared_components=smeared)
    print(f"2 INDIVIDUAL : max pairwise overlap = {maxpair:.3f} (PASS < 0.65; ~0.707 = fused), "
          f"median = {medpair:.3f}")
    print(f"               median loading excess-kurtosis = {kurt:.2f} (PASS > 0; ~0 = Gaussian = unseparable)")
    print(f"               tasks using each component: mean {usage.mean():.1f}/{len(clouds)}, "
          f"smeared(all-task) fraction = {smeared:.2f} (PASS < 0.5)")

    # ---- 3 REUSED (hold out one task per genre, fit on the rest, same guard)
    held_idx = [max(i for i, (gg, _) in enumerate(labels) if gg == g) for g in genre_names]
    train_idx = [i for i in range(len(clouds)) if i not in held_idx]
    Xtr_p, basis_tr, _, _, _ = guarded_project([clouds[i] for i in train_idx], R)
    A_tr, _ = extract(Xtr_p, K, seed=0)
    resid, nact = check_reused([clouds[i] for i in train_idx],
                               [clouds[i] for i in held_idx], basis_tr, A_tr)
    genre_act = np.zeros((A.shape[1], len(genre_names)))
    for j, (g, _) in enumerate(labels):
        genre_act[:, genre_names.index(g)] += active[:, j]
    crossgenre = float(((genre_act > 0).sum(1) >= 2).mean())
    res.update(heldout_tasks=[labels[i] for i in held_idx], heldout_recon_residual=resid,
               active_components_per_heldout_task=nact, frac_components_crossgenre=crossgenre)
    print(f"3 REUSED     : held-out {[labels[i] for i in held_idx]}")
    print(f"               recon residual = {resid:.3f} (PASS < 0.3), "
          f"active comps/task = {nact:.1f} (PASS < {K//2}: sparse)")
    print(f"               components active in >=2 genres = {crossgenre:.2f} (PASS > 0.2)")

    # ---- diversity curve (exp 05 analog): extraction stability as genres are pooled
    curve = {}
    for ng in range(1, len(genre_names) + 1):
        idx = [i for i, (g, _) in enumerate(labels) if g in genre_names[:ng]]
        Xg_p, _, _, _, _ = guarded_project([clouds[i] for i in idx], R)
        s, _ = check_stable(Xg_p, min(K, Xg_p.shape[1]), n_boot=3)
        curve[ng] = round(float(s), 3)
        print(f"   diversity  : {ng} genre(s) pooled -> stability {s:.3f}")
    res["diversity_stability_curve"] = curve

    # ---- verdict (pre-committed thresholds)
    p_stable = stab >= 0.8
    p_indiv = (kurt > 0) and (maxpair < 0.65) and (smeared < 0.5)
    p_reused = (resid < 0.3) and (nact < K * 0.5) and (crossgenre > 0.2)
    res["pass"] = {"stable": bool(p_stable), "individual": bool(p_indiv), "reused": bool(p_reused)}
    verdict = "GREEN — build the compositional system" if all(res["pass"].values()) else \
              "RED — fall back to the robustness reframe (non-destructive learned merge)"
    res["verdict"] = verdict
    print(f"\nVERDICT: STABLE={p_stable} INDIVIDUAL={p_indiv} REUSED={p_reused}\n -> {verdict}")

    json.dump(res, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
