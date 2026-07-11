"""
visualize.py — the one-screen picture: a recognizable modality on the input side (a bouncing-ball film),
the model in the middle (a watcher with hidden memory, never opened), and the reader's verdict on the right
(memory order + poles recovered from the raw streams alone). Writes watched.png.

Left-to-right is exactly the pipeline: FILM (u_t made physical) -> WATCHER (u_t in, y_t out; x_t hidden) ->
READER (sees only u_t, y_t -> order + poles). The film strip is shown next to the read so the input side is
a modality your eyes parse, not an abstract squiggle — yet the reader works from the streams regardless.
Reproduce: python visualize.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "orbit"))
import physics as PH
import watched as WM

SHOW = 160        # stream steps to display
N_FRAMES = 10     # film-strip frames
FRAME_STRIDE = 3  # spacing between shown frames


def stack(ax, X, title, color, hidden=False):
    for i in range(X.shape[1]):
        ax.plot(X[:SHOW, i] + 3.2 * i, lw=0.8, color=color, alpha=0.7 if hidden else 0.9)
    ax.set_yticks([]); ax.set_xlabel("time step", fontsize=8); ax.set_title(title, fontsize=10)
    if hidden:
        ax.set_facecolor("#f2f2f2")
        ax.text(0.5, 0.985, "INSIDE THE MODEL — INVISIBLE TO THE READER", transform=ax.transAxes,
                ha="center", va="top", fontsize=7.5, color="#888", weight="bold")


def poles(ax, true_p, rec_p, title):
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color="#bbb", lw=1)
    ax.scatter(true_p.real, true_p.imag, marker="x", s=95, color="black", label="planted (hidden)")
    if len(rec_p):
        ax.scatter(rec_p.real, rec_p.imag, marker="o", s=115, facecolors="none",
                   edgecolors="crimson", lw=1.7, label="recovered from I/O")
    ax.set_aspect("equal"); ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15)
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
    ax.set_title(title, fontsize=10); ax.legend(fontsize=7.5, loc="lower left")


def main():
    out = WM.run(T=10000, r=4, seed=0)
    rd = out["read"]

    fig = plt.figure(figsize=(17, 7.8))
    gs = GridSpec(2, N_FRAMES, figure=fig, height_ratios=[0.62, 1.35], hspace=0.32, wspace=0.18)

    # ---- top band: the recognizable modality (film strip) ----
    for k in range(N_FRAMES):
        ax = fig.add_subplot(gs[0, k])
        s = out["states"][k * FRAME_STRIDE]
        ax.imshow(PH.render(s), interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"t={k * FRAME_STRIDE}", fontsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#2a6fdb"); sp.set_linewidth(1.4)
    fig.text(0.5, 0.905, "INPUT — a bouncing-ball film you can watch (the white drive u_t, made physical)",
             ha="center", fontsize=12, color="#2a6fdb", weight="bold")

    # ---- bottom band: input -> watcher (hidden) -> output -> reader ----
    ax_u = fig.add_subplot(gs[1, 0:3])
    ax_x = fig.add_subplot(gs[1, 3:6])
    ax_y = fig.add_subplot(gs[1, 6:8])
    ax_p = fig.add_subplot(gs[1, 8:10])

    stack(ax_u, out["u"], "the same drive u_t, as a stream  (3 white channels)", "#2a6fdb")
    stack(ax_x, out["x"], f"WATCHER hidden state x_t  (r={out['true_order']} memory modes)", "#777", hidden=True)
    stack(ax_y, out["y"], "WATCHER output y_t  (3 channels)", "#0a8f3c")
    ok = abs(rd["order"] - out["true_order"]) <= 1 and rd["pole_err"] < 0.1
    poles(ax_p, out["true_poles"], rd["poles"],
          f"READER saw only u,y\n-> order {rd['order']} (planted {out['true_order']}), "
          f"pole err {rd['pole_err']:.3f}\n{'RECOVERED' if ok else 'MISS'}")

    fig.suptitle("Recognizable modality in  ·  the model never opened  ·  its memory read from the streams alone",
                 fontsize=13.5, y=0.975)
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "watched.png"),
                dpi=115, bbox_inches="tight")
    print(f"wrote watched.png  (order {rd['order']}/{out['true_order']}, pole err {rd['pole_err']:.3f})")


if __name__ == "__main__":
    main()
