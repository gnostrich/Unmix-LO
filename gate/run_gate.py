"""
GATE.md steps 3-4 — extract components from the pooled real gradient clouds and run the decision.

Reads grads/{genre}/{task}.npy (from collect_grads.py), then:
  STABLE     — bootstrap re-extraction, matched-component cosine (> ~0.8 passes)
  INDIVIDUAL — pairwise overlap not stuck at ~0.707; loadings non-Gaussian (kurtosis > 0);
               per-component task usage concentrated, not smeared; task-generic direction flagged
  REUSED     — hold out one task per genre; sparse low-residual reconstruction with the SAME
               library; components recurring across >=2 genres; diversity curve (1->3 genres)

Writes gate/results.json and prints the verdict.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from extractor import (whiten_project, extract, match, check_stable,
                       check_individual, check_reused, load_real_gradients)

R = int(os.environ.get("GATE_R", 50))          # PCA subspace dim (GATE.md: ~50-200)
K = int(os.environ.get("GATE_K", 30))          # ICA components
ACT_THR = 0.15                                  # activity threshold (same as check_reused)


def task_activation(A, basis, clouds):
    """(K, n_tasks) activation = per-task variance of loadings on each component, column-normalized."""
    acts = []
    for c in clouds:
        Xc = (c - c.mean(0, keepdims=True)) @ basis.T
        acts.append((Xc @ A).var(0))
    M = np.array(acts).T                         # (K, n_tasks)
    return M / (M.max(1, keepdims=True) + 1e-12)


def main():
    clouds, labels = load_real_gradients(os.path.join(HERE, "..", "grads"))
    genres = [g for g, _ in labels]
    genre_names = sorted(set(genres), key=genres.index)
    print(f"loaded {len(clouds)} task clouds: {labels}")

    # per-cloud normalization so no single task dominates the pool by gradient scale
    clouds = [c / (np.linalg.norm(c, axis=1, keepdims=True).mean() + 1e-12) for c in clouds]
    X = np.vstack(clouds)
    Xp, basis = whiten_project(X, R)
    print(f"pooled {X.shape} -> PCA r={Xp.shape[1]}, ICA K={K}\n")
    res = {"n_tasks": len(clouds), "labels": labels, "r": int(Xp.shape[1]), "K": K}

    # ---- 1 STABLE
    stab, A = check_stable(Xp, K)
    res["stable_matched_cosine"] = stab
    print(f"1 STABLE     : bootstrap matched cosine = {stab:.3f}   (pass > ~0.8)")

    # ---- 2 INDIVIDUAL
    maxpair, kurt = check_individual(A, Xp @ A)
    M = task_activation(A, basis, clouds)
    active = M > ACT_THR
    usage = active.sum(1)                                        # tasks using each component
    smeared = float((usage >= len(clouds) - 1).mean())           # fraction of near-universal comps
    # task-generic direction (GATE.md gotcha): component most aligned with the global mean gradient
    gmean = X.mean(0) @ basis.T
    gmean /= np.linalg.norm(gmean) + 1e-12
    generic_cos = float(np.abs(gmean @ A).max())
    res.update(max_pairwise_overlap=maxpair, median_loading_kurtosis=kurt,
               mean_tasks_per_component=float(usage.mean()), frac_smeared_components=smeared,
               generic_direction_max_cos=generic_cos)
    print(f"2 INDIVIDUAL : max pairwise overlap = {maxpair:.3f} (~0.707 = fused signature)")
    print(f"               median loading excess-kurtosis = {kurt:.2f} (pass > 0, Gaussian = unseparable)")
    print(f"               tasks using each component: mean {usage.mean():.1f}/{len(clouds)}, "
          f"smeared(all-task) fraction = {smeared:.2f}")
    print(f"               task-generic direction max |cos| with a component = {generic_cos:.2f}")

    # ---- 3 REUSED (hold out one task per genre, fit on the rest)
    held_idx = []
    for g in genre_names:
        held_idx.append(max(i for i, (gg, _) in enumerate(labels) if gg == g))
    train_idx = [i for i in range(len(clouds)) if i not in held_idx]
    Xtr = np.vstack([clouds[i] for i in train_idx])
    Xtr_p, basis_tr = whiten_project(Xtr, R)
    A_tr, _ = extract(Xtr_p, K, seed=0)
    resid, nact = check_reused([clouds[i] for i in train_idx],
                               [clouds[i] for i in held_idx], basis_tr, A_tr)
    # cross-genre recurrence on the full library
    genre_act = np.zeros((A.shape[1], len(genre_names)))
    for j, (g, _) in enumerate(labels):
        genre_act[:, genre_names.index(g)] += active[:, j]
    crossgenre = float(((genre_act > 0).sum(1) >= 2).mean())
    res.update(heldout_tasks=[labels[i] for i in held_idx], heldout_recon_residual=resid,
               active_components_per_heldout_task=nact, frac_components_crossgenre=crossgenre)
    print(f"3 REUSED     : held-out {[labels[i] for i in held_idx]}")
    print(f"               recon residual = {resid:.3f} (pass = low), active comps/task = {nact:.1f} (pass = sparse, << {K})")
    print(f"               components active in >=2 genres = {crossgenre:.2f}")

    # ---- diversity curve (exp 05 analog): stability as genres are pooled
    curve = {}
    for ng in range(1, len(genre_names) + 1):
        idx = [i for i, (g, _) in enumerate(labels) if g in genre_names[:ng]]
        Xg = np.vstack([clouds[i] for i in idx])
        Xg_p, _ = whiten_project(Xg, R)
        s, _ = check_stable(Xg_p, min(K, Xg_p.shape[1]), n_boot=3)
        curve[ng] = round(float(s), 3)
        print(f"   diversity  : {ng} genre(s) pooled -> stability {s:.3f}")
    res["diversity_stability_curve"] = curve

    # ---- verdict
    p_stable = stab > 0.8
    p_indiv = (kurt > 0) and (maxpair < 0.68 or maxpair > 0.73) and smeared < 0.5
    p_reused = (resid < 0.3) and (nact < K * 0.5) and crossgenre > 0.2
    res["pass"] = {"stable": bool(p_stable), "individual": bool(p_indiv), "reused": bool(p_reused)}
    verdict = "GREEN — build the compositional system" if all(res["pass"].values()) else \
              "RED — fall back to the robustness reframe (non-destructive learned merge)"
    res["verdict"] = verdict
    print(f"\nVERDICT: STABLE={p_stable} INDIVIDUAL={p_indiv} REUSED={p_reused}\n -> {verdict}")

    json.dump(res, open(os.path.join(HERE, "results.json"), "w"), indent=2)
    print(f"\nwrote {os.path.join(HERE, 'results.json')}")


if __name__ == "__main__":
    main()
