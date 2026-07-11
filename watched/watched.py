"""
watched.py — a recognizable modality on the input side, an unopened model in the middle, the reader's
verdict on the right. Assembles pieces that already exist:

  input   : a bouncing-ball physics film (orbit/physics.render) driven frame-by-frame by a white signal u_t
  watcher : a linear memory device with a KNOWN order r, fed the same u_t, emitting an output stream y_t
  reader  : io_trace/stream_trace.read_trace — sees ONLY (u_t, y_t), never the watcher's weights or state,
            and recovers how much memory the watcher carries (order) and where its poles sit.

The film is the white drive u_t made physical and watchable: each frame is one white kick rendered as forces
on a real physics scene. The watcher "watches" that drive (is fed u_t) and remembers it with order-r dynamics.
The point: the input side is a modality your eyes parse, yet the reader still works from the raw streams alone,
and its answer is checkable because we planted the watcher's poles ourselves.

numpy only (matplotlib lives in visualize.py). Reproduce: python visualize.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "orbit"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "io_trace"))
import physics as PH          # the real scene + render()
import stream_trace as ST     # the unmodified reader

P_IN, Q_OUT = 3, 3            # drive channels / watcher output channels
NOISE = 0.05


def white_drive(T, seed=0):
    """The white input stream u_t — the reader's whitening assumption holds by construction."""
    return np.random.default_rng(seed).normal(size=(T, P_IN))


def film(u, damp=0.03, kick=0.35, seed=0):
    """Render the drive as a watchable physics scene: real billiards (orbit.physics.step) whose balls are
    kicked each frame by u_t. Returns the list of world states (one per frame). Purely the human-facing view;
    the reader never touches these frames — it is given u_t directly."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(PH.R, 1 - PH.R, size=(PH.N, 2))
    v = rng.uniform(-1, 1, size=(PH.N, 2)) * 0.5
    s = np.concatenate([p.ravel(), v.ravel()])
    states = []
    for t in range(len(u)):
        states.append(s)
        s = PH.step(s)
        vv = s[2 * PH.N:].reshape(PH.N, 2)
        vv *= (1 - damp)                                   # gentle damping so the strip stays legible
        vv[:, 0] += kick * u[t, 0]                         # channel 0 -> horizontal kick
        vv[:, 1] += kick * u[t, 1]                         # channel 1 -> vertical kick
        vv += kick * 0.5 * u[t, 2] * np.array([1.0, -1.0])  # channel 2 -> shear
        s = np.concatenate([s[:2 * PH.N], vv.ravel()])
    return states


def watcher(u, r=4, rho=0.85, seed=1):
    """The model in the middle: a linear recurrence x_{t+1}=A x_t + B u_t, y_t = C x_t + noise, with a KNOWN
    memory order r and known poles eig(A). Returns (y, x_hidden, true_poles). x_hidden is for display only."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(r, r)); A *= rho / np.max(np.abs(np.linalg.eigvals(A)))
    B = rng.normal(size=(r, P_IN)) / np.sqrt(P_IN)
    C = rng.normal(size=(Q_OUT, r)) / np.sqrt(r)
    x = np.zeros(r); xs, ys = [], []
    for t in range(len(u)):
        xs.append(x.copy())
        ys.append(C @ x + NOISE * rng.normal(size=Q_OUT))
        x = A @ x + B @ u[t]
    return np.array(ys), np.array(xs), np.linalg.eigvals(A)


def run(T=10000, r=4, seed=0):
    """Wire it end to end and read the watcher's memory from the streams alone."""
    u = white_drive(T, seed=seed)
    states = film(u, seed=seed)
    y, x, true_poles = watcher(u, r=r, seed=seed + 1)
    rd = ST.read_trace(u, y, seed=seed)
    rd["pole_err"] = ST.pole_match_error(true_poles, rd["poles"])
    return {"u": u, "y": y, "x": x, "states": states, "true_poles": true_poles,
            "true_order": r, "read": rd}


if __name__ == "__main__":
    out = run()
    rd = out["read"]
    print(f"watcher true order {out['true_order']}  ->  reader recovered order {rd['order']}  "
          f"(pole-match err {rd['pole_err']:.3f}, gap {rd['gap']:.1f})")
