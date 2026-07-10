"""
physics.py — the real-coupling corpus: 2D rigid-body physics, self-contained (numpy only).
5 balls in a unit box, gravity, elastic wall + ball-ball collisions. Deterministic per seed.
state = [pos(2N), vel(2N)] (D=4N=20). Also renders a frame so an orbit can be watched.
"""
import numpy as np

N = 5
R = 0.06
GRAV = 0.6
DT = 0.04
D = 4 * N
IMG = 96


def step(s):
    p = s[:2 * N].reshape(N, 2).copy(); v = s[2 * N:].reshape(N, 2).copy()
    v[:, 1] -= GRAV * DT
    p = p + v * DT
    for d in (0, 1):
        lo = p[:, d] < R; hi = p[:, d] > 1 - R
        v[lo, d] = np.abs(v[lo, d]); v[hi, d] = -np.abs(v[hi, d])
        p[:, d] = np.clip(p[:, d], R, 1 - R)
    for i in range(N):
        for j in range(i + 1, N):
            dp = p[i] - p[j]; dist = float(np.hypot(*dp))
            if 0 < dist < 2 * R:
                n = dp / dist; vn = float((v[i] - v[j]) @ n)
                if vn < 0:
                    v[i] -= vn * n; v[j] += vn * n
                ov = 2 * R - dist; p[i] += 0.5 * ov * n; p[j] -= 0.5 * ov * n
    return np.concatenate([p.ravel(), v.ravel()])


def rollout(seed, T):
    rng = np.random.default_rng(seed)
    p = rng.uniform(R, 1 - R, size=(N, 2)); v = rng.uniform(-1, 1, size=(N, 2)) * 0.5
    s = np.concatenate([p.ravel(), v.ravel()]); out = []
    for _ in range(T):
        out.append(s); s = step(s)
    return np.array(out)


def collect(n_rollouts, T, seed0=0):
    """Return states and trajectory boundaries (start index of each rollout) for within-traj operators."""
    trajs = [rollout(seed0 + r, T) for r in range(n_rollouts)]
    bounds = np.cumsum([0] + [len(t) for t in trajs])
    return np.concatenate(trajs), bounds


def step_driven(s, gmul):
    """One step with a scaled gravity (for the driven / multiscale corpus)."""
    p = s[:2 * N].reshape(N, 2).copy(); v = s[2 * N:].reshape(N, 2).copy()
    v[:, 1] -= GRAV * gmul * DT
    p = p + v * DT
    for d in (0, 1):
        lo = p[:, d] < R; hi = p[:, d] > 1 - R
        v[lo, d] = np.abs(v[lo, d]); v[hi, d] = -np.abs(v[hi, d])
        p[:, d] = np.clip(p[:, d], R, 1 - R)
    for i in range(N):
        for j in range(i + 1, N):
            dp = p[i] - p[j]; dist = float(np.hypot(*dp))
            if 0 < dist < 2 * R:
                n = dp / dist; vn = float((v[i] - v[j]) @ n)
                if vn < 0:
                    v[i] -= vn * n; v[j] += vn * n
                ov = 2 * R - dist; p[i] += 0.5 * ov * n; p[j] -= 0.5 * ov * n
    return np.concatenate([p.ravel(), v.ravel()])


def collect_driven(n_rollouts, T, seed0=0, period=30):
    """Multiscale corpus (disclosed): fast collisions under a SLOW periodic gravity modulation
    g(t)=1+0.6 sin(2 pi t/period) — a genuine phrase-scale mode coupled to the fast dynamics. The slow
    variable (mean height / vertical energy) oscillates at 2 pi/period; the kernel should recover it."""
    trajs = []
    w = 2 * np.pi / period
    for r in range(n_rollouts):
        rng = np.random.default_rng(seed0 + r)
        p = rng.uniform(R, 1 - R, size=(N, 2)); v = rng.uniform(-1, 1, size=(N, 2)) * 0.5
        s = np.concatenate([p.ravel(), v.ravel()]); st = []
        ph = rng.uniform(0, 2 * np.pi)
        for t in range(T):
            st.append(s); s = step_driven(s, 1.0 + 0.6 * np.sin(w * t + ph))
        trajs.append(np.array(st))
    bounds = np.cumsum([0] + [len(t) for t in trajs])
    return np.concatenate(trajs), bounds


def render(s):
    """96x96x3 uint8 frame: balls as filled discs (green), velocity as a faded prev-position ghost (red)."""
    img = np.zeros((IMG, IMG, 3), np.float32)
    yy, xx = np.mgrid[0:IMG, 0:IMG]
    p = s[:2 * N].reshape(N, 2); v = s[2 * N:].reshape(N, 2)
    prev = p - v * DT
    for pos, ch, a in ((prev, 0, 0.4), (p, 1, 1.0)):
        for i in range(N):
            cx, cy = pos[i, 0] * IMG, (1 - pos[i, 1]) * IMG
            m = (xx - cx) ** 2 + (yy - cy) ** 2 <= (R * IMG) ** 2
            img[m, ch] = a
    return (img * 255).astype(np.uint8)


# ---- calibration corpora (poles-first) ----
def collect_null(n_states, seed=0):
    """No dynamics: i.i.d. random states. One 'trajectory' of independent draws -> P should have no gap."""
    rng = np.random.default_rng(seed)
    s = rng.uniform(0, 1, size=(n_states, D))
    return s, np.array([0, n_states])


def collect_periodic(n_rollouts, T, seed0=0, period=25):
    """Known-periodic corpus: each 'ball' orbits a circle at a fixed period -> clean gap, kernel recovers omega."""
    trajs = []
    for r in range(n_rollouts):
        rng = np.random.default_rng(seed0 + r)
        phase = rng.uniform(0, 2 * np.pi, N); rad = rng.uniform(0.15, 0.35, N)
        cen = rng.uniform(0.3, 0.7, (N, 2)); w = 2 * np.pi / period
        st = []
        for t in range(T):
            ang = phase + w * t
            p = cen + rad[:, None] * np.stack([np.cos(ang), np.sin(ang)], 1)
            v = rad[:, None] * w * np.stack([-np.sin(ang), np.cos(ang)], 1)
            st.append(np.concatenate([p.ravel(), v.ravel()]))
        trajs.append(np.array(st))
    bounds = np.cumsum([0] + [len(t) for t in trajs])
    return np.concatenate(trajs), bounds
