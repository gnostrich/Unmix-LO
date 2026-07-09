"""
VIRTUAL WORLD MODEL — the seed world + FOUR direct-view modalities.

Reuses thoughtworld/engine.py physics (numpy rigid-body: balls, gravity, walls, elastic collisions)
UNCHANGED as a file; we only raise N at runtime so the shared world-state dimension is D=24 (>=20-32,
per the smoke_loop calibration note — a higher-D medium keeps the structured/noise baseline low).

The SAME physics events are exposed through four genuinely-different DIRECT views (physical-bridge law,
no orphan modalities):
  M1 vision      : rendered (prev,cur) frame  -> a frozen small vision encoder (ViT / DINO).
  M2 text        : QUALITATIVE description, NO coordinates -> a frozen small text encoder (MiniLM).
  M3 audio       : hand-crafted collision/impact features over the last k frames (event view).
  M4 time-series : velocities / speeds / energies over the last k frames (motion view; deliberately
                   NOT current absolute positions, so it is a distinct velocity-view -> genuine
                   coverage complementarity with vision/text which see positions).

Each modality is LOSSY IN A DIFFERENT WAY, so the coherent union (stitch) covers more of the world
state than any single modality — that coverage-union is the reliable, honest win.
"""
import os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "thoughtworld"))
import engine as ENG  # noqa: E402  reuse the validated physics engine (as a file, unchanged)

# --- raise the world to D=24 at runtime (do NOT edit the shared engine file) ---
ENG.N = 6
ENG.R = 0.05
ENG.D = 4 * ENG.N
N, R, D = ENG.N, ENG.R, ENG.D
K = 4  # window length for audio / time-series features


# ---------------------------------------------------------------- rollouts ----
def collect(n_rollouts=20, T=45, seed0=0):
    """Roll full trajectories; return per-frame aligned arrays + rollout/time ids.

    Returns dict with (all per-frame, length M = n_rollouts*(T-1)):
      s_prev, s_cur, s_next : (M, D) engine states (t-1, t, t+1)
      traj                  : list of full (T, D) trajectories (for windowed features)
      rollout, tidx         : (M,) rollout id and time index within rollout
    """
    s_prev, s_cur, s_next, rollout, tidx, hist = [], [], [], [], [], []
    trajs = []
    for r in range(n_rollouts):
        tr = ENG.rollout(seed0 + r, T + 1)      # (T+1, D)
        trajs.append(tr)
        for t in range(1, T):
            s_prev.append(tr[t - 1]); s_cur.append(tr[t]); s_next.append(tr[t + 1])
            rollout.append(r); tidx.append(t)
            # windowed history of the K frames ending at t (clamp at rollout start)
            idxs = [max(0, t - j) for j in range(K)][::-1]
            hist.append(tr[idxs])               # (K, D)
    return dict(
        s_prev=np.array(s_prev), s_cur=np.array(s_cur), s_next=np.array(s_next),
        rollout=np.array(rollout), tidx=np.array(tidx), hist=np.array(hist),
        trajs=trajs, D=D, N=N,
    )


# ---------------------------------------------------------------- vision ------
def render(s_prev, s_cur):
    """224x224x3 frame: prev positions in R channel, cur positions in G (2-frame velocity cue)."""
    return ENG.render(s_prev, s_cur)


# ---------------------------------------------------------------- text --------
def _region(x, y):
    h = "left" if x < 0.33 else ("right" if x > 0.66 else "center")
    v = "bottom" if y < 0.33 else ("top" if y > 0.66 else "middle")
    return f"{v}-{h}"


def describe(s_cur, s_prev):
    """QUALITATIVE description — NO coordinates (avoids the velocity-arithmetic handoff confound).

    Says roughly where balls are (region words), whether near a wall, and coarse motion direction.
    """
    p = s_cur[:2 * N].reshape(N, 2)
    v = s_cur[2 * N:].reshape(N, 2)
    parts = ["A few balls bounce in a box under gravity with elastic wall and ball collisions."]
    for i in range(N):
        x, y = p[i]
        near = []
        if x < R + 0.03: near.append("the left wall")
        if x > 1 - R - 0.03: near.append("the right wall")
        if y < R + 0.03: near.append("the floor")
        if y > 1 - R - 0.03: near.append("the ceiling")
        wall = (" near " + " and ".join(near)) if near else ""
        speed = np.hypot(*v[i])
        if speed < 0.15:
            motion = "nearly still"
        else:
            hor = "right" if v[i, 0] > 0.1 else ("left" if v[i, 0] < -0.1 else "")
            ver = "up" if v[i, 1] > 0.1 else ("down" if v[i, 1] < -0.1 else "")
            motion = "moving " + " and ".join([d for d in (ver, hor) if d]) if (hor or ver) else "drifting"
        fast = "quickly" if speed > 0.6 else ("slowly" if speed < 0.3 else "")
        parts.append(f"A ball is in the {_region(x, y)} region{wall}, {motion} {fast}.".replace("  ", " "))
    return " ".join(parts)


# ---------------------------------------------------------------- audio -------
def audio_features(hist):
    """Hand-crafted collision/impact 'audio' features over the K-frame window.

    Detects near-wall impacts and near-ball impacts by proximity x speed (impulse proxy). This is the
    'click at a collision' view: informative about WHICH balls hit and how hard, weak on the positions
    of non-colliding balls. hist: (K, D). Returns a fixed-length feature vector.
    """
    feats = []
    for f in range(hist.shape[0]):
        p = hist[f, :2 * N].reshape(N, 2)
        v = hist[f, 2 * N:].reshape(N, 2)
        sp = np.hypot(v[:, 0], v[:, 1])
        # wall proximity impact energy
        dwall = np.minimum.reduce([p[:, 0] - R, (1 - R) - p[:, 0], p[:, 1] - R, (1 - R) - p[:, 1]])
        wall_hit = np.clip(1 - dwall / (2 * R), 0, 1)          # ~1 when touching a wall
        wall_impact = (wall_hit * sp)
        # ball-ball proximity impact energy
        ball_impact = 0.0; n_pairs = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dd = np.hypot(*(p[i] - p[j]))
                prox = np.clip(1 - (dd - 2 * R) / (2 * R), 0, 1)
                rel = np.hypot(*(v[i] - v[j]))
                ball_impact += prox * rel
                n_pairs += (prox > 0)
        feats.extend([
            float(wall_impact.sum()), float(wall_impact.max()), float((wall_hit > 0.5).sum()),
            float(ball_impact), float(n_pairs), float(sp.mean()), float(sp.max()),
        ])
    return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------- time-series -
def timeseries_features(hist):
    """Velocities / speeds / energies over the K-frame window — a MOTION view (no current positions).

    Deliberately excludes current absolute positions so this modality is a genuine velocity-view that
    complements the position-views (vision/text). hist: (K, D).
    """
    feats = []
    for f in range(hist.shape[0]):
        p = hist[f, :2 * N].reshape(N, 2)
        v = hist[f, 2 * N:].reshape(N, 2)
        sp = np.hypot(v[:, 0], v[:, 1])
        ke = 0.5 * (sp ** 2)                    # per-ball kinetic energy
        pe = ENG.G * p[:, 1]                    # per-ball potential energy (height)
        feats.extend(v.ravel().tolist())        # 2N velocity components
        feats.extend(sp.tolist())               # N speeds
        feats.extend([float(ke.sum()), float(pe.sum()), float(ke.sum() + pe.sum())])
    return np.array(feats, dtype=np.float32)


if __name__ == "__main__":
    d = collect(n_rollouts=2, T=10)
    print("frames", d["s_cur"].shape, "D", D, "N", N)
    print("audio dim", audio_features(d["hist"][0]).shape)
    print("timeseries dim", timeseries_features(d["hist"][0]).shape)
    print("TEXT sample:\n ", describe(d["s_cur"][3], d["s_prev"][3]))
