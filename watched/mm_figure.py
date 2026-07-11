"""
mm_figure.py — the multi-modal picture (watched_multimodal.png). The same world state, watched through two
real encoders (ViT on the frame, MiniLM on a text description). What the two modalities SHARE is the world,
recovered with no supervision — and that shared latent is what lets the reader recover the hidden watcher's
dynamics, where a single raw modality cannot.

Uses the cached encoder features (feats.npz, T=4000). Reproduce: python multimodal.py --T 4000 ; python mm_figure.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "orbit"))
import physics as PH
import multimodal as MM

WIN = 200


def main():
    d = np.load(os.path.join(HERE, "feats.npz"), allow_pickle=True)
    u, V, L, tp = d["u"], d["V"], d["L"], d["tp"]
    strip = d["strip"]
    res = MM.read_all(d)
    za, zb, cc = MM.cca(V, L, k=150, d=4)
    shared = (za + zb) / 2
    # a lively description (frame 0 is static); recompute states cheaply (no encoders) for the text sample
    states = MM.scene(MM.colored_latent(int(d["T"]), seed=0), seed=0)
    sample = MM.describe(states[24])

    # align a shared canonical component to the true u factor it tracks
    Us = (u - u.mean(0)) / u.std(0)
    Ss = (shared - shared.mean(0)) / shared.std(0)
    corr = np.abs(Ss.T @ Us) / len(u)
    si, ui = np.unravel_index(np.argmax(corr), corr.shape)
    sign = np.sign((Ss[:, si] * Us[:, ui]).mean())

    fig = plt.figure(figsize=(16, 7.6))
    gs = GridSpec(1, 3, figure=fig, left=0.05, right=0.97, top=0.52, bottom=0.09, wspace=0.28)

    # ---- top: the two modalities of the same scene ----
    strip_gs = GridSpec(1, len(strip), figure=fig, left=0.035, right=0.985, top=0.90, bottom=0.68, wspace=0.12)
    for k in range(len(strip)):
        ax = fig.add_subplot(strip_gs[0, k])
        ax.imshow(strip[k], interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"t={k * 4}", fontsize=7.5)
        for sp in ax.spines.values():
            sp.set_edgecolor("#2a6fdb"); sp.set_linewidth(1.3)
    fig.text(0.5, 0.945, "ONE world, TWO real encoders — ViT sees the frame, MiniLM reads the description",
             ha="center", fontsize=13, weight="bold")
    fig.text(0.5, 0.925, "vision modality: ViT(render)   ·   the SAME scene", ha="center", fontsize=9.5,
             color="#2a6fdb")
    fig.text(0.5, 0.635, f'language modality: MiniLM("{sample}")',
             ha="center", fontsize=8.5, color="#0a8f3c", style="italic")

    # ---- bottom-left: canonical correlations (the encoders agree) ----
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(range(1, len(cc) + 1), cc, color="#6a3fb5", width=0.6)
    ax1.set_ylim(0, 1); ax1.set_xticks(range(1, len(cc) + 1))
    ax1.set_xlabel("canonical component", fontsize=9); ax1.set_ylabel("correlation", fontsize=9)
    ax1.set_title("the two encoders share the world\n(vision <-> language canonical corr)", fontsize=10)
    for i, c in enumerate(cc):
        ax1.text(i + 1, c + 0.02, f"{c:.2f}", ha="center", fontsize=8)

    # ---- bottom-mid: the shared latent IS the world ----
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(Us[:WIN, ui], color="#111", lw=1.4, label="true world factor u")
    ax2.plot(sign * Ss[:WIN, si], color="#6a3fb5", lw=1.2, alpha=0.85,
             label="shared latent (CCA, no supervision)")
    ax2.set_yticks([]); ax2.set_xlabel("time step", fontsize=9)
    ax2.legend(fontsize=7.5, loc="upper right")
    ax2.set_title(f"the shared subspace IS the world\n(recovered blind; corr {corr[si, ui]:.2f}, "
                  f"u-R2 {MM.r2(shared, u).mean():.2f})", fontsize=10)

    # ---- bottom-right: reading the hidden watcher ----
    ax3 = fig.add_subplot(gs[0, 2])
    th = np.linspace(0, 2 * np.pi, 200); ax3.plot(np.cos(th), np.sin(th), color="#bbb", lw=1)
    ax3.scatter(tp.real, tp.imag, marker="x", s=110, color="black", label="planted watcher poles", zorder=5)
    tpo = res["text"]["poles"]
    if len(tpo):
        ax3.scatter(tpo.real, tpo.imag, marker="s", s=95, facecolors="none", edgecolors="#c33", lw=1.7,
                    label=f"text only (order {res['text']['order']})")
    spo = res["shared"]["poles"]
    if len(spo):
        ax3.scatter(spo.real, spo.imag, marker="o", s=125, facecolors="none", edgecolors="#0a8f3c", lw=2.0,
                    label=f"shared latent (order {res['shared']['order']}, err {res['shared']['pole_err']:.3f})")
    ax3.set_aspect("equal"); ax3.set_xlim(-1.15, 1.15); ax3.set_ylim(-1.15, 1.15)
    ax3.set_xticks([-1, 0, 1]); ax3.set_yticks([-1, 0, 1])
    ax3.set_title(f"reading the hidden watcher\nvision alone: order {res['vision']['order']} (below floor); "
                  "shared recovers", fontsize=10)
    ax3.legend(fontsize=7, loc="lower left")

    fig.suptitle("Multiple modalities: what vision and language SHARE is the world — and only the shared latent reads the hidden dynamics",
                 fontsize=12.5, y=0.995)
    fig.savefig(os.path.join(HERE, "watched_multimodal.png"), dpi=115, bbox_inches="tight")
    print(f"wrote watched_multimodal.png  | vision o{res['vision']['order']} text o{res['text']['order']} "
          f"shared o{res['shared']['order']} err {res['shared']['pole_err']:.3f}")


if __name__ == "__main__":
    main()
