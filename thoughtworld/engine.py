"""
THOUGHTWORLD seed engine — minimal dense, self-consistent, directed, rollable physics.
2D elastic balls in a box under gravity with wall + ball-ball collisions. Deterministic forward map
(fixes the gauge nabla_0). State s = [positions(2N), velocities(2N)] -> D = 4N. Pure numpy; renders
frames (2-frame overlay so velocity is visible to a vision fragment) without any image library.
"""
import numpy as np

N = 5                      # balls
R = 0.06                   # radius (world units, box [0,1]^2)
G = 0.6                    # gravity
DT = 0.04
IMG = 224                  # render size for ViT
D = 4 * N


def step(p, v):
    v = v.copy(); p = p.copy()
    v[:, 1] -= G * DT
    p = p + v * DT
    # wall collisions (elastic)
    for d in (0, 1):
        lo = p[:, d] < R
        hi = p[:, d] > 1 - R
        v[lo, d] = np.abs(v[lo, d]); v[hi, d] = -np.abs(v[hi, d])
        p[:, d] = np.clip(p[:, d], R, 1 - R)
    # ball-ball elastic collisions (equal mass -> exchange normal velocity components)
    for i in range(N):
        for j in range(i + 1, N):
            dp = p[i] - p[j]; dist = np.hypot(*dp)
            if 0 < dist < 2 * R:
                n = dp / dist
                dvn = np.dot(v[i] - v[j], n)
                if dvn < 0:
                    v[i] -= dvn * n; v[j] += dvn * n
                overlap = 2 * R - dist
                p[i] += 0.5 * overlap * n; p[j] -= 0.5 * overlap * n
    return p, v


def rollout(seed, T):
    rng = np.random.default_rng(seed)
    p = rng.uniform(R, 1 - R, size=(N, 2))
    v = rng.uniform(-1, 1, size=(N, 2)) * 0.5
    states = []
    for _ in range(T):
        states.append(np.concatenate([p.ravel(), v.ravel()]))
        p, v = step(p, v)
    return np.array(states)                      # (T, D)


def collect(n_rollouts=40, T=60, seed0=0):
    """Return arrays s_prev, s_t, s_next aligned so a fragment sees (t-1,t) and predicts t+1."""
    prev, cur, nxt = [], [], []
    for r in range(n_rollouts):
        traj = rollout(seed0 + r, T + 2)
        for t in range(1, len(traj) - 1):
            prev.append(traj[t - 1]); cur.append(traj[t]); nxt.append(traj[t + 1])
    return np.array(prev), np.array(cur), np.array(nxt)


def render(s_prev, s_cur):
    """224x224x3 uint8: previous positions in R channel, current in G channel (encodes velocity)."""
    img = np.zeros((IMG, IMG, 3), dtype=np.float32)
    yy, xx = np.mgrid[0:IMG, 0:IMG]
    for s, ch in ((s_prev, 0), (s_cur, 1)):
        p = s[:2 * N].reshape(N, 2)
        for i in range(N):
            cx, cy = p[i, 0] * IMG, (1 - p[i, 1]) * IMG      # flip y for image coords
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= (R * IMG) ** 2
            img[mask, ch] = 1.0
    return img


if __name__ == "__main__":
    prev, cur, nxt = collect(n_rollouts=3, T=20)
    print(f"states: {cur.shape}, D={D}")
    print(f"mean step displacement = {np.linalg.norm(nxt - cur, axis=1).mean():.4f}")
    im = render(prev[0], cur[0]); print(f"render {im.shape}, nonzero px {int((im.sum(2) > 0).sum())}")
