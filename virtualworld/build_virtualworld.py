"""
BUILD the virtual world model: read the FOUR direct-view modalities with frozen small CPU models,
align each to the shared medium (the world state) with a lightweight ridge map (the ONLY training),
then apply the VALIDATED construct behaviors and emit a light JSON the dashboard loads.

  STITCH   : precision-weighted per-dimension fuse (each state dim taken from whichever modality sees
             it best) -> ONE world estimate. This is the coverage-union — the reliable, honest win.
  DECOHERE : pairwise residual of aligned reps.
  CLASSIFY : detector.classify (Step-0 validated) -> STRUCTURED (extend) vs NOISE (reject).
  KNOBS    : drop a modality; inject NOISE into one (-> NOISE, rejected by the robust stitch);
             inject a STRUCTURED hidden distinction (-> STRUCTURED, extended / held-both).

HONEST LABEL: real frozen models mostly AGREE (post-alignment residual is state-independent NOISE), so
NATURAL structured decoherence is rare; the coverage-union stitch is the visible win, and STRUCTURED
extension appears mainly when the USER injects it (the knob). We do NOT present injected structure as
if convergent models supplied it.
"""
import os, sys, json, time
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import world as W
from detector import classify
import mz_fluid as MZ

CACHE = os.path.join(HERE, "feat_cache"); os.makedirs(CACHE, exist_ok=True)
torch.set_num_threads(os.cpu_count() or 4)
SEED = 0
MZ_ON = os.environ.get("VW_MZ", "1") != "0"     # VW_MZ=0 disables the EXPERIMENTAL recurrent layer entirely
OUT_SUF = os.environ.get("VW_SUF", "")           # output-filename suffix (separation-check runs write elsewhere)
N_ROLLOUTS, T = 26, 45
VISION_MODEL = "google/vit-base-patch16-224"
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]); IMAGENET_STD = np.array([0.229, 0.224, 0.225])
MODS = ["vision", "text", "audio", "timeseries"]


# --------------------------------------------------------------- encoders -----
@torch.no_grad()
def encode_vision(frames):
    path = os.path.join(CACHE, f"vision_{len(frames)}.npy")
    if os.path.exists(path): return np.load(path)
    model = AutoModel.from_pretrained(VISION_MODEL).eval()
    out, t0 = [], time.time()
    for i in range(0, len(frames), 16):
        x = (frames[i:i + 16] - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.tensor(x.transpose(0, 3, 1, 2), dtype=torch.float32)
        out.append(model(pixel_values=x).last_hidden_state.mean(1).numpy())
        if (i // 16) % 8 == 0: print(f"    [vision] {i+16}/{len(frames)} ({time.time()-t0:.0f}s)", flush=True)
    F = np.concatenate(out).astype(np.float32); np.save(path, F); return F


@torch.no_grad()
def encode_text(texts):
    path = os.path.join(CACHE, f"text_{len(texts)}.npy")
    if os.path.exists(path): return np.load(path)
    tok = AutoTokenizer.from_pretrained(TEXT_MODEL); model = AutoModel.from_pretrained(TEXT_MODEL).eval()
    out, t0 = [], time.time()
    for i in range(0, len(texts), 32):
        enc = tok(texts[i:i + 32], return_tensors="pt", padding=True, truncation=True, max_length=160)
        h = model(**enc).last_hidden_state; m = enc["attention_mask"].unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).numpy())
        if (i // 32) % 8 == 0: print(f"    [text] {i+32}/{len(texts)} ({time.time()-t0:.0f}s)", flush=True)
    F = np.concatenate(out).astype(np.float32); np.save(path, F); return F


# --------------------------------------------------------------- ridge --------
def fit_ridge(Xtr, Ytr, lam=10.0, n_pca=48):
    """Standardise + (optional) PCA-reduce features on train, then ridge to the standardised target.

    PCA on high-dim frozen features (vision 768, text 384) is essential: with a few hundred train
    frames a raw ridge overfits and generalises negatively across held-out rollouts.
    """
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    Xs = (Xtr - mu) / sd
    if n_pca and Xs.shape[1] > n_pca:
        U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
        P = Vt[:n_pca].T                                 # (feat, n_pca)
    else:
        P = np.eye(Xs.shape[1])
    Z = Xs @ P
    Z = np.concatenate([Z, np.ones((len(Z), 1))], 1)
    A = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ Ytr)
    return (mu, sd, P, A)


def apply_ridge(model, X):
    mu, sd, P, A = model
    Z = ((X - mu) / sd) @ P
    Z = np.concatenate([Z, np.ones((len(Z), 1))], 1)
    return Z @ A


def per_dim_r2(pred, truth):
    ss_res = ((truth - pred) ** 2).sum(0)
    ss_tot = ((truth - truth.mean(0)) ** 2).sum(0) + 1e-12
    return 1 - ss_res / ss_tot


def r2(pred, truth):
    return float(1 - ((truth - pred) ** 2).sum() / (((truth - truth.mean(0)) ** 2).sum() + 1e-12))


def stitch(aligned, weights, mods):
    """Precision-weighted per-dim fuse over the listed modalities. weights[m]: (D,) train per-dim R^2."""
    D = next(iter(aligned.values())).shape[1]
    num = np.zeros((len(next(iter(aligned.values()))), D)); den = np.zeros(D)
    for m in mods:
        w = np.clip(weights[m], 0, None)                 # only trust dims a modality actually predicts
        num += aligned[m] * w; den += w
    fallback = den < 1e-6
    if fallback.any():                                    # no modality covers this dim -> equal mean
        eq = np.mean([aligned[m] for m in mods], axis=0)
        out = np.where(fallback, eq, num / np.where(den < 1e-6, 1, den))
    else:
        out = num / den
    return out


# --------------------------------------------------------------- main ---------
def main():
    print("rolling world + building modalities ...", flush=True)
    d = W.collect(N_ROLLOUTS, T, seed0=SEED)
    s_cur, s_prev, hist, rollout = d["s_cur"], d["s_prev"], d["hist"], d["rollout"]
    n = len(s_cur)
    # shared medium = permutation-invariant SCENE features (balls are indistinguishable, so an ordered
    # per-ball state is not identifiable from vision/text/audio)
    scene = np.stack([W.scene_features(s_cur[i]) for i in range(n)])
    D = scene.shape[1]
    # split by ROLLOUT so a test rollout is contiguous for the trajectory plot
    n_tr_roll = int(0.6 * N_ROLLOUTS)
    train = np.where(rollout < n_tr_roll)[0]; test = np.where(rollout >= n_tr_roll)[0]
    print(f"frames {n}, medium D {D}; train {len(train)} test {len(test)}", flush=True)

    # standardise the shared-medium target on train
    tmu, tsd = scene[train].mean(0), scene[train].std(0) + 1e-8
    Y = (scene - tmu) / tsd

    # raw modality features
    frames = np.stack([W.render(s_prev[i], s_cur[i]) for i in range(n)])
    texts = [W.describe(s_cur[i], s_prev[i]) for i in range(n)]
    raw = {
        "vision": encode_vision(frames),
        "text": encode_text(texts),
        "audio": np.stack([W.audio_features(hist[i]) for i in range(n)]),
        "timeseries": np.stack([W.timeseries_features(hist[i]) for i in range(n)]),
    }

    # align each modality -> shared medium; per-dim train R^2 = coverage weights
    aligned, weights, per_mod = {}, {}, {}
    for m in MODS:
        model = fit_ridge(raw[m][train], Y[train])
        a = apply_ridge(model, raw[m])
        aligned[m] = a
        weights[m] = per_dim_r2(a[train], Y[train])        # train per-dim coverage
        per_mod[m] = {"r2_overall": r2(a[test], Y[test]),
                      "per_dim_r2_test": per_dim_r2(a[test], Y[test]).tolist()}
        print(f"  {m:<11} align test R^2 = {per_mod[m]['r2_overall']:.3f}", flush=True)

    pos = np.array(W.SCENE_POS); vel = np.array(W.SCENE_VEL)   # spatial vs velocity/energy medium dims

    # STITCH (coverage union) + drop-one uniqueness
    st = stitch(aligned, weights, MODS)
    stitch_all = {"r2_overall": r2(st[test], Y[test]),
                  "r2_pos": r2(st[test][:, pos], Y[test][:, pos]),
                  "r2_vel": r2(st[test][:, vel], Y[test][:, vel]),
                  "per_dim_r2": per_dim_r2(st[test], Y[test]).tolist()}
    drop_one = []
    for m in MODS:
        others = [x for x in MODS if x != m]
        sm = stitch(aligned, weights, others)
        drop_one.append({"modality": m,
                         "r2_overall": r2(sm[test], Y[test]),
                         "delta": stitch_all["r2_overall"] - r2(sm[test], Y[test]),
                         "r2_pos": r2(sm[test][:, pos], Y[test][:, pos]),
                         "r2_vel": r2(sm[test][:, vel], Y[test][:, vel])})
    print(f"STITCH test R^2 = {stitch_all['r2_overall']:.3f} "
          f"(pos {stitch_all['r2_pos']:.3f}, vel {stitch_all['r2_vel']:.3f})")
    for x in drop_one:
        print(f"  drop {x['modality']:<11} R^2 {x['r2_overall']:.3f}  (uniqueness {x['delta']:+.3f})")

    # NATURAL decoherence map (expected: mostly NOISE — honest)
    natural = []
    for i in range(len(MODS)):
        for j in range(i + 1, len(MODS)):
            m1, m2 = MODS[i], MODS[j]
            res = classify(aligned[m1] - aligned[m2], s_cur, train, test, D=D)
            natural.append({"pair": f"{m1}~{m2}", **{k: res[k] for k in
                            ("verdict", "structured", "heldout_r2", "captured", "baseline",
                             "captured_vs_base", "eff_rank", "resid_norm")}})
    n_struct = sum(x["structured"] for x in natural)
    print(f"NATURAL decoherence: {n_struct}/{len(natural)} tagged STRUCTURED "
          f"(expected ~0 — frozen models agree / residual is noise)")

    # KNOBS -------------------------------------------------------------------
    rng = np.random.default_rng(SEED)
    knob_noise, knob_struct = [], []
    ref = "timeseries"                                    # decohere the poked modality against this ref
    scale = np.linalg.norm(aligned["vision"]) / np.sqrt(aligned["vision"].size)
    # standardised RAW engine state — the detector predicts residuals from THIS (the true world state),
    # so an injected STRUCTURED distinction must be a function of it to be (correctly) held-out-predictable.
    Sstd = (s_cur - s_cur[train].mean(0)) / (s_cur[train].std(0) + 1e-8)
    for m in MODS:
        other = ref if m != ref else "vision"

        # (i) inject NOISE into modality m -> should tag NOISE; REJECT = drop the flagged modality
        #     (denoise toward the coherent consensus), vs a naive fuse that TRUSTS the corrupted modality.
        noise = 1.5 * scale * rng.normal(size=aligned[m].shape)
        det_n = classify((aligned[m] + noise) - aligned[other], s_cur, train, test, D=D)
        a_corrupt = dict(aligned); a_corrupt[m] = aligned[m] + noise
        equal_w = {mm: np.ones(D) for mm in MODS}
        r_naive = r2(stitch(a_corrupt, equal_w, MODS)[test], Y[test])          # trusts the noise
        r_reject = r2(stitch(aligned, weights, [x for x in MODS if x != m])[test], Y[test])  # drop it
        knob_noise.append({"modality": m, "verdict": det_n["verdict"],
                           "heldout_r2": det_n["heldout_r2"], "captured_vs_base": det_n["captured_vs_base"],
                           "stitch_r2_naive_trust": r_naive, "stitch_r2_reject": r_reject,
                           "rejected": not det_n["structured"]})

        # (ii) inject a STRUCTURED hidden distinction: rank-2 and LINEAR IN THE RAW WORLD STATE, so it is
        #      genuinely reproducible + held-out predictable -> should tag STRUCTURED -> extend / hold both.
        Braw = rng.normal(size=(s_cur.shape[1], 2))
        Bdir = rng.normal(size=(2, D))
        raw = (Sstd @ Braw) @ Bdir                        # (n, D) rank-2 state-driven signal
        inj = 0.9 * scale * raw / (raw.std() + 1e-9)
        det_s = classify((aligned[m] + inj) - aligned[other], s_cur, train, test, D=D)
        recov = classify(inj, s_cur, train, test, D=D)    # EXTEND / hold-both: recover the distinction
        knob_struct.append({"modality": m, "verdict": det_s["verdict"],
                            "heldout_r2": det_s["heldout_r2"], "captured_vs_base": det_s["captured_vs_base"],
                            "extended": det_s["structured"],
                            "recovered_distinction_r2": recov["heldout_r2"]})
    print(f"KNOB inject-noise: {sum(k['rejected'] for k in knob_noise)}/{len(knob_noise)} correctly rejected as NOISE")
    print(f"KNOB inject-structured: {sum(k['extended'] for k in knob_struct)}/{len(knob_struct)} correctly extended as STRUCTURED")

    # EXPERIMENTAL (UNVALIDATED) recurrent MZ/tape layer — runs ALONGSIDE, kept SEPARATE, never merged.
    # It is a pure SINK: it reads the validated Y/stitch (as copies) and feeds NOTHING back, so every
    # validated number above is bit-identical whether it runs or not (VW_MZ=0 proves this empirically).
    if MZ_ON:
        print("\n[EXPERIMENTAL] recurrent MZ/tape probe (UNVALIDATED — reduces toward classical filtering):")
        experimental_mz = MZ.run(Y, st, rollout, np.arange(n_tr_roll), np.arange(n_tr_roll, N_ROLLOUTS))
        print(f"    tape self-expands to order {experimental_mz['tape_order_selected']} "
              f"(Hankel SVs>floor); memory-closure held-out R^2 = "
              f"{experimental_mz['memory_closure']['memory_heldout_r2']:.3f}, "
              f"spectral radius {experimental_mz['memory_closure']['spectral_radius']:.2f}")
    else:
        print("\n[EXPERIMENTAL] recurrent MZ/tape layer DISABLED (VW_MZ=0) — separation check.")
        experimental_mz = {"disabled": True,
                           "note": "recurrent layer turned OFF for the validated-layer separation check"}

    # trajectory sample (one test rollout) for the stitched-vs-truth plot, in ORIGINAL units
    viz_roll = n_tr_roll
    idx = np.where(rollout == viz_roll)[0]
    L = W.SCENE_LABELS
    comps = {"mean height (y_mean)": L.index("y_mean"),
             "mean speed": L.index("speed_mean"),
             "total kinetic energy": L.index("KE_total"),
             "balls near floor": L.index("n_floor"),
             "top-row occupancy": L.index("occ[2,0]")}
    st_orig = st * tsd + tmu
    trajectory = {"t": d["tidx"][idx].tolist(), "truth": {}, "stitch": {}}
    for name, c in comps.items():
        trajectory["truth"][name] = scene[idx, c].tolist()
        trajectory["stitch"][name] = st_orig[idx, c].tolist()

    out = {
        "meta": {"N": W.N, "D": D, "n_frames": n, "n_train": len(train), "n_test": len(test),
                 "n_rollouts": N_ROLLOUTS, "modalities": MODS,
                 "models": {"vision": VISION_MODEL, "text": TEXT_MODEL,
                            "audio": "hand-crafted collision/impact features",
                            "timeseries": "velocity/speed/energy window features"},
                 "detector": "STRUCTURED iff captured>1.3*base AND held-out R^2 from state >= 0.3 (Step-0 validated)"},
        "stitch": stitch_all,
        "drop_one": drop_one,
        "per_modality_align": per_mod,
        "coverage_dim_labels": list(W.SCENE_LABELS),
        "coverage_pos_dims": list(W.SCENE_POS), "coverage_vel_dims": list(W.SCENE_VEL),
        "coverage_matrix": {m: np.clip(weights[m], 0, 1).tolist() for m in MODS},
        "decoherence_natural": natural,
        "natural_structured_count": int(n_struct),
        "knob_inject_noise": knob_noise,
        "knob_inject_structured": knob_struct,
        "trajectory": trajectory,
        "experimental_mz": experimental_mz,
        "layers_note": ("TWO SEPARATE LAYERS. VALIDATED (single-step): stitch / classify (D>=20 + "
                        "held-out R^2>=0.3) / extend / reject — passed Step-0 synthetic validation. "
                        "EXPERIMENTAL (recurrent MZ/tape): UNVALIDATED, reduces toward classical linear "
                        "state-space filtering. The recurrent part is NEVER presented as validated."),
        "honest_label": (
            "Real frozen small models mostly AGREE: after alignment the pairwise residual is "
            "state-independent NOISE, so NATURAL structured decoherence is rare (see the map). The "
            "visible, robust win is the COVERAGE-UNION stitch (fill-missing-modalities, drop-one). "
            "STRUCTURED extension appears mainly when YOU inject it via the knob — that is an injected "
            "distinction, NOT one the convergent models supplied. Consistent with the xresolve null."),
    }

    def js(o):
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(o)

    json.dump(out, open(os.path.join(HERE, f"virtualworld_data{OUT_SUF}.json"), "w"), indent=1, default=js)
    with open(os.path.join(HERE, f"data{OUT_SUF}.js"), "w") as f:
        f.write("window.VW_DATA = ")
        json.dump(out, f, default=js)
        f.write(";")
    print("\nwrote virtualworld_data.json + data.js")


if __name__ == "__main__":
    main()
