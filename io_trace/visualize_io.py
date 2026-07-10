"""
visualize_io.py — show the LITERAL I/O of the virtual generators from the io_trace experiment:
the input stream, the hidden layer (which the reader NEVER sees), the output stream, and what the
reader recovered from I/O alone (poles: true x vs recovered o). Pure visualization; no new claims.
"""
import sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import stream_trace as ST
from run import gen_atomic, gen_memoryless, gen_continuous, T_DEFAULT

SHOW = 160   # steps to display


def gen_atomic_with_hidden(r, T, seed=0, rho=0.85):
    """Same generator as run.py, but also return the hidden state (for display only)."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(r, r)); A *= rho / np.max(np.abs(np.linalg.eigvals(A)))
    B = rng.normal(size=(r, 3)) / np.sqrt(3)
    C = rng.normal(size=(3, r)) / np.sqrt(r)
    u = rng.normal(size=(T, 3)); x = np.zeros(r); ys, xs = [], []
    for t in range(T):
        xs.append(x.copy()); ys.append(C @ x + 0.05 * rng.normal(size=3))
        x = A @ x + B @ u[t]
    return u, np.array(ys), np.array(xs), np.linalg.eigvals(A)


def stack_traces(ax, X, title, color, hidden=False):
    n = X.shape[1]
    for i in range(n):
        ax.plot(X[:SHOW, i] + 3.2 * i, lw=0.8, color=color, alpha=0.9 if not hidden else 0.7)
    ax.set_yticks([]); ax.set_title(title, fontsize=10)
    if hidden:
        ax.set_facecolor("#f2f2f2")
        ax.text(0.5, 0.985, "INVISIBLE TO THE READER", transform=ax.transAxes, ha="center", va="top",
                fontsize=8, color="#888", weight="bold")


def pole_plot(ax, true_p, rec_p, title):
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color="#bbb", lw=1)
    if len(true_p):
        ax.scatter(true_p.real, true_p.imag, marker="x", s=90, color="black", label="true (hidden)")
    if len(rec_p):
        ax.scatter(rec_p.real, rec_p.imag, marker="o", s=110, facecolors="none",
                   edgecolors="crimson", lw=1.6, label="recovered from I/O")
    ax.set_aspect("equal"); ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15)
    ax.set_title(title, fontsize=10); ax.legend(fontsize=7, loc="lower left")


fig, axes = plt.subplots(3, 4, figsize=(17, 10))

# ---- row 1: ATOMIC r=4 ----
u, y, x, tp = gen_atomic_with_hidden(4, T_DEFAULT, seed=1)
rd = ST.read_trace(u, y, seed=1)
stack_traces(axes[0, 0], u, "INPUT u_t  (3 channels, white noise)", "#2a6fdb")
stack_traces(axes[0, 1], x, "HIDDEN LAYER x_t  (r=4 modes)", "#777", hidden=True)
stack_traces(axes[0, 2], y, "OUTPUT y_t  (3 channels)", "#0a8f3c")
pole_plot(axes[0, 3], tp, rd["poles"], f"reader saw only u,y -> order {rd['order']} (truth 4)")
axes[0, 0].set_ylabel("ATOMIC virtual model (r=4)", fontsize=11, weight="bold")

# ---- row 2: MEMORYLESS ----
u, y, _ = gen_memoryless(T_DEFAULT, seed=0)
rd = ST.read_trace(u, y, seed=0)
stack_traces(axes[1, 0], u, "INPUT u_t", "#2a6fdb")
axes[1, 1].axis("off")
axes[1, 1].text(0.5, 0.5, "NO HIDDEN LAYER\n(y = M u + noise)", ha="center", va="center",
                fontsize=11, color="#888", weight="bold")
stack_traces(axes[1, 2], y, "OUTPUT y_t (echoes input instantly)", "#0a8f3c")
pole_plot(axes[1, 3], np.array([]), rd["poles"], f"reader -> order {rd['order']} (truth 0)")
axes[1, 0].set_ylabel("MEMORYLESS virtual model", fontsize=11, weight="bold")

# ---- row 3: CONTINUOUS ----
u, y, lam = gen_continuous(T_DEFAULT, seed=0)
rd = ST.read_trace(u, y, seed=0)
stack_traces(axes[2, 0], u, "INPUT u_t", "#2a6fdb")
# show a handful of the 200 hidden modes
rng = np.random.default_rng(0)
xs_show = np.cumsum(rng.normal(size=(SHOW, 6)) * 0.1, 0)  # illustrative placeholder? no — recompute honestly:
# honest hidden states for display: rerun the recursion keeping 6 sampled modes
Bs = np.random.default_rng(0).normal(size=(200, 3)) / np.sqrt(3)
xfull = np.zeros(200); traj = []
uu = u[:SHOW]
xstate = np.zeros(200)
for t in range(SHOW):
    traj.append(xstate.copy()); xstate = lam * xstate + Bs @ uu[t]
traj = np.array(traj)[:, ::33]   # every 33rd of the 200 dense modes
stack_traces(axes[2, 1], traj, "HIDDEN LAYER (200 dense modes; 6 shown)", "#777", hidden=True)
stack_traces(axes[2, 2], y, "OUTPUT y_t", "#0a8f3c")
pole_plot(axes[2, 3], lam[::12] + 0j, rd["poles"],
          f"dense pole line -> order {rd['order']} (won't terminate)")
axes[2, 0].set_ylabel("CONTINUOUS-SPECTRUM virtual model", fontsize=11, weight="bold")

plt.suptitle("The virtual models' literal I/O — the reader sees ONLY the blue input and green output streams",
             fontsize=12, y=0.995)
plt.tight_layout()
plt.savefig("io_samples.png", dpi=110)
print("wrote io_samples.png")
