"""
nuanced.py — the honest, harder version: the watcher reads a feature off the actual film PIXELS, so its
input is correlated (the physics is smooth), not white. Read that stream naively and you recover the INPUT's
color instead of the watcher; deconvolve the input's autocorrelation and the watcher's true poles return.

One screen (watched_nuanced.png):
  top  : the recognizable film — a smooth, watchable physics scene the watcher actually looks at
  below: the pixel feature f_t (correlated, with its slowly-decaying autocorrelation) -> watcher output y_t
         -> NAIVE read (wrong: reads the input's color) vs DECONVOLVED read (recovers the planted poles)

The poles are planted, so both reads are checkable. Reproduce: python nuanced.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "orbit"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "io_trace"))
import physics as PH
import stream_trace as ST
import watched as WM
import correlated_read as CR

SHOW = 160
N_FRAMES = 10
FRAME_STRIDE = 4
KICK, DAMP = 0.08, 0.005
T, R_TRUE = 12000, 4


def run(seed=0):
    rng = np.random.default_rng(seed)
    u = WM.white_drive(T, seed=seed)
    states = WM.film(u, damp=DAMP, kick=KICK, seed=seed)          # smooth, watchable film
    Proj = rng.normal(size=(3, PH.IMG * PH.IMG * 3)) / np.sqrt(PH.IMG * PH.IMG * 3)
    f = np.array([Proj @ PH.render(s).astype(np.float32).ravel() for s in states])
    f = f - f.mean(0)                                             # the pixel feature (correlated input)
    A, B, C, tp = CR.make_watcher(R_TRUE, seed=seed + 1)
    y = CR.drive_through(A, B, C, f, seed=seed)
    naive = ST.read_trace(f, y, seed=seed)
    deconv = CR.read_trace_deconv(f, y, seed=seed)
    naive["pole_err"] = ST.pole_match_error(tp, naive["poles"])
    deconv["pole_err"] = ST.pole_match_error(tp, deconv["poles"])
    ac = float(np.mean([np.corrcoef(f[1:, i], f[:-1, i])[0, 1] for i in range(3)]))
    return dict(states=states, f=f, y=y, tp=tp, naive=naive, deconv=deconv, autocorr=ac)


def stack(ax, X, title, color, subtitle=None):
    for i in range(X.shape[1]):
        ax.plot(X[:SHOW, i] + 3.2 * i, lw=0.85, color=color, alpha=0.9)
    ax.set_yticks([]); ax.set_xlabel("time step", fontsize=8); ax.set_title(title, fontsize=10)
    if subtitle:
        ax.text(0.5, 0.985, subtitle, transform=ax.transAxes, ha="center", va="top",
                fontsize=8, color="#a23", weight="bold")


def poles(ax, tp, rec, title, good):
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color="#bbb", lw=1)
    ax.scatter(tp.real, tp.imag, marker="x", s=95, color="black", label="planted watcher poles")
    if len(rec):
        ax.scatter(rec.real, rec.imag, marker="o", s=115, facecolors="none",
                   edgecolors=("#0a8f3c" if good else "crimson"), lw=1.8,
                   label="recovered from (f,y)")
    ax.set_aspect("equal"); ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15)
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
    ax.set_title(title, fontsize=9.5, color=("#0a8f3c" if good else "crimson"))
    ax.legend(fontsize=7, loc="lower left")


def acorr_inset(ax, f):
    """Small inset: the input's own autocorrelation decays slowly (this is what the naive read mistakes
    for the watcher's memory); white input would be a spike at lag 0."""
    K = 20
    fc = f - f.mean(0)
    r0 = np.mean(np.sum(fc * fc, 1))
    rk = [np.mean(np.sum(fc[k:] * fc[:len(fc) - k], 1)) / r0 for k in range(K + 1)]
    ins = ax.inset_axes([0.55, 0.58, 0.42, 0.38])
    ins.bar(range(K + 1), rk, color="#a23", width=0.9)
    ins.axhline(0, color="#888", lw=0.6)
    ins.set_title("input autocorr R_ff(k)", fontsize=6.5)
    ins.tick_params(labelsize=5); ins.set_xticks([0, 10, 20])


def main():
    out = run()
    nv, dc = out["naive"], out["deconv"]

    fig = plt.figure(figsize=(17, 7.9))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[0.62, 1.35], hspace=0.34, wspace=0.26)

    # top: the film the watcher looks at
    strip = GridSpec(1, N_FRAMES, figure=fig, left=0.04, right=0.98, top=0.88, bottom=0.60, wspace=0.15)
    for k in range(N_FRAMES):
        ax = fig.add_subplot(strip[0, k])
        ax.imshow(PH.render(out["states"][k * FRAME_STRIDE]), interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"t={k * FRAME_STRIDE}", fontsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#a23"); sp.set_linewidth(1.4)
    fig.text(0.5, 0.905, "INPUT — the watcher reads a feature off these pixels (a smooth, watchable film)",
             ha="center", fontsize=12, color="#a23", weight="bold")

    ax_f = fig.add_subplot(gs[1, 0])
    ax_y = fig.add_subplot(gs[1, 1])
    ax_n = fig.add_subplot(gs[1, 2])
    ax_d = fig.add_subplot(gs[1, 3])

    stack(ax_f, out["f"], f"pixel feature f_t  (lag-1 rho ~ {out['autocorr']:.2f})", "#a23",
          subtitle="CORRELATED — not white")
    acorr_inset(ax_f, out["f"])
    stack(ax_y, out["y"], "watcher output y_t", "#0a8f3c")
    poles(ax_n, out["tp"], nv["poles"],
          f"NAIVE read of (f,y)\norder {nv['order']} (planted {R_TRUE}), pole err {nv['pole_err']:.2f}\n"
          f"reads the input's color, not the watcher", good=False)
    poles(ax_d, out["tp"], dc["poles"],
          f"DECONVOLVED read of (f,y)\norder {dc['order']} (planted {R_TRUE}), pole err {dc['pole_err']:.3f}\n"
          f"input autocorrelation divided out", good=True)

    fig.suptitle("The watcher reads the pixels, so its input is correlated — naive read fails, deconvolution recovers",
                 fontsize=13.5, y=0.975)
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "watched_nuanced.png"),
                dpi=115, bbox_inches="tight")
    print(f"wrote watched_nuanced.png  (autocorr {out['autocorr']:.2f}; "
          f"naive err {nv['pole_err']:.2f} order {nv['order']} -> deconv err {dc['pole_err']:.3f} order {dc['order']})")


if __name__ == "__main__":
    main()
