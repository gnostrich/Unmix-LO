"""
engine.py — minimal 2D rigid-body physics, fresh and self-contained (numpy only).

N balls of equal mass and radius R in the unit box [0,1]^2 under gravity, with elastic wall bounces and
elastic equal-mass ball-ball collisions. Deterministic given a seed. A `state` is the flat vector
[x_1,y_1,...,x_N,y_N, vx_1,vy_1,...,vx_N,vy_N] of length D = 4N. This is the coherent dynamics a fragment is
measured against; nothing here depends on any other file.
"""
import numpy as np

N = 5           # number of balls
R = 0.06        # radius (unit box)
GRAV = 0.6      # gravitational acceleration (downward)
DT = 0.04       # timestep
D = 4 * N       # state dimension: N positions (2 each) + N velocities (2 each)


def _split(state):
    """state -> (positions (N,2), velocities (N,2))."""
    p = state[:2 * N].reshape(N, 2).copy()
    v = state[2 * N:].reshape(N, 2).copy()
    return p, v


def _join(p, v):
    return np.concatenate([p.ravel(), v.ravel()])


def step(state):
    """Advance one timestep. Semi-implicit Euler + wall bounce + equal-mass elastic ball-ball collisions."""
    p, v = _split(state)
    # gravity + drift
    v[:, 1] -= GRAV * DT
    p = p + v * DT
    # elastic wall collisions: reflect the offending velocity component, clamp inside the box
    for d in (0, 1):
        lo = p[:, d] < R
        hi = p[:, d] > 1.0 - R
        v[lo, d] = np.abs(v[lo, d])
        v[hi, d] = -np.abs(v[hi, d])
        p[:, d] = np.clip(p[:, d], R, 1.0 - R)
    # equal-mass elastic ball-ball collisions: exchange the velocity component along the contact normal
    for i in range(N):
        for j in range(i + 1, N):
            dp = p[i] - p[j]
            dist = float(np.hypot(dp[0], dp[1]))
            if 0.0 < dist < 2 * R:
                n = dp / dist
                dv = v[i] - v[j]
                vn = float(dv @ n)
                if vn < 0.0:                       # approaching -> collide
                    v[i] -= vn * n
                    v[j] += vn * n
                overlap = 2 * R - dist              # separate to remove interpenetration
                p[i] += 0.5 * overlap * n
                p[j] -= 0.5 * overlap * n
    return _join(p, v)


def rollout(seed, T):
    """A single deterministic trajectory of T states from a random (seeded) start."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(R, 1.0 - R, size=(N, 2))
    v = rng.uniform(-1.0, 1.0, size=(N, 2)) * 0.5
    s = _join(p, v)
    out = []
    for _ in range(T):
        out.append(s)
        s = step(s)
    return np.array(out)


def collect(n_rollouts, T, seed0=0):
    """Return (states, next_states): flattened (state_t, state_{t+1}) pairs across many rollouts."""
    cur, nxt = [], []
    for r in range(n_rollouts):
        traj = rollout(seed0 + r, T + 1)
        cur.append(traj[:-1])
        nxt.append(traj[1:])
    return np.concatenate(cur), np.concatenate(nxt)


if __name__ == "__main__":
    cur, nxt = collect(n_rollouts=4, T=30)
    print(f"states {cur.shape}, D={D}")
    print(f"mean step displacement ||s_next - s_cur|| = {np.linalg.norm(nxt - cur, axis=1).mean():.4f}")
    # sanity: energy stays bounded (no blow-up), balls stay in the box
    p_all = cur[:, :2 * N].reshape(-1, N, 2)
    print(f"positions in-box: {(p_all >= 0).all() and (p_all <= 1).all()}")
