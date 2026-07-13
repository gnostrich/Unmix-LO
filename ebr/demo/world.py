"""
demo/world.py — a small 2D physics micro-world: the OVERLAP-MAKER member.

The point (R1-R5 thesis): cross-modal routing should emerge from diverse members with PARTIAL modality
OVERLAP, routed collectively — not from a hand-paired translation dictionary. This member emits, from ONE
ground-truth latent scene, three modalities at once, all paired by construction (ground truth, not curated):
    - render     : a small RGB frame            -> overlaps VISION (ViT/MobileNet/CLIP-vision can read it)
    - caption    : a state-derived sentence     -> overlaps TEXT   (MiniLM/CLIP-text can read it)
    - dynamics   : a trajectory signature       -> a THIRD modality nobody else speaks

Because render/caption/dynamics of a scene are the SAME event, this member is intrinsically multi-modal — a
natural bridge (like CLIP's two towers), grounded in the simulator's own generative process. Adding it gives
the routing graph real topology instead of a single CLIP bridge.

No learned weights, no training — pure ground-truth simulation. Deterministic given a seed.
"""
import numpy as np
from PIL import Image, ImageDraw

SHAPES = ["ball", "square", "triangle"]
COLORS = {"red": (220, 60, 55), "green": (70, 175, 95), "blue": (60, 110, 210), "yellow": (225, 195, 70)}
COLOR_NAMES = list(COLORS)
GRID = 64          # render canvas is GRID x GRID
G = 0.45           # gravity per step
REST = 0.82        # restitution on bounce
T = 16             # trajectory length


def sample_scene(seed):
    """Sample a scene: 1-2 objects, each with shape/color/position/velocity. Deterministic in seed."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 3))
    objs = []
    for _ in range(n):
        objs.append({
            "shape": SHAPES[int(rng.integers(len(SHAPES)))],
            "color": COLOR_NAMES[int(rng.integers(len(COLOR_NAMES)))],
            "x": float(rng.uniform(14, GRID - 14)), "y": float(rng.uniform(10, GRID - 24)),
            "vx": float(rng.uniform(-3.2, 3.2)), "vy": float(rng.uniform(-1.5, 1.5)),
            "r": float(rng.uniform(5, 8)),
        })
    return objs


def simulate(objs, steps=T):
    """Roll the scene forward under gravity + box bounce. Returns a (steps, n, 2) position trajectory and the
    per-object velocity history (steps, n, 2)."""
    st = [dict(o) for o in objs]
    pos, vel = [], []
    for _ in range(steps):
        pos.append([[o["x"], o["y"]] for o in st])
        vel.append([[o["vx"], o["vy"]] for o in st])
        for o in st:
            o["vy"] += G
            o["x"] += o["vx"]; o["y"] += o["vy"]
            if o["x"] < o["r"]:
                o["x"] = o["r"]; o["vx"] = -o["vx"] * REST
            if o["x"] > GRID - o["r"]:
                o["x"] = GRID - o["r"]; o["vx"] = -o["vx"] * REST
            if o["y"] > GRID - o["r"]:
                o["y"] = GRID - o["r"]; o["vy"] = -o["vy"] * REST
            if o["y"] < o["r"]:
                o["y"] = o["r"]; o["vy"] = -o["vy"] * REST
    return np.array(pos), np.array(vel)


def render(objs, pos_frame=None):
    """Render one frame (RGB HxWx3 uint8). pos_frame overrides object positions (for a trajectory frame)."""
    im = Image.new("RGB", (GRID, GRID), (18, 22, 28))
    dr = ImageDraw.Draw(im)
    for i, o in enumerate(objs):
        x, y = (pos_frame[i] if pos_frame is not None else (o["x"], o["y"]))
        r, c = o["r"], COLORS[o["color"]]
        if o["shape"] == "ball":
            dr.ellipse([x - r, y - r, x + r, y + r], fill=c)
        elif o["shape"] == "square":
            dr.rectangle([x - r, y - r, x + r, y + r], fill=c)
        else:
            dr.polygon([(x, y - r), (x - r, y + r), (x + r, y + r)], fill=c)
    return np.array(im)


def _motion_words(vx, vy):
    parts = []
    if abs(vx) > 1.2:
        parts.append("moving right" if vx > 0 else "moving left")
    if vy > 1.2:
        parts.append("falling")
    elif vy < -1.2:
        parts.append("rising")
    if not parts:
        parts.append("nearly still")
    return " and ".join(parts)


def caption(objs, vel_frame=None):
    """A state-derived sentence for the scene (text modality). Uses initial velocities unless vel_frame given."""
    clauses = []
    for i, o in enumerate(objs):
        vx, vy = (vel_frame[i] if vel_frame is not None else (o["vx"], o["vy"]))
        clauses.append(f"a {o['color']} {o['shape']} {_motion_words(vx, vy)}")
    return " and ".join(clauses)


def dynamics_signature(pos, vel):
    """A compact trajectory/dynamics feature (the third modality): per-scene motion statistics that do NOT
    depend on color/shape identity — speed, verticality, number of bounces (sign flips in velocity), spread."""
    v = vel.reshape(vel.shape[0], -1, 2)
    speed = np.linalg.norm(v, axis=2).mean(0)                    # mean speed per object
    vert = (np.abs(v[..., 1]) / (np.abs(v[..., 0]) + 1e-6)).mean(0)
    flips = (np.abs(np.diff(np.sign(v[..., 0]), axis=0)) > 0).sum(0) \
        + (np.abs(np.diff(np.sign(v[..., 1]), axis=0)) > 0).sum(0)
    span = pos.reshape(pos.shape[0], -1, 2).std(0).mean(1)       # spatial spread per object
    feat = np.concatenate([speed, vert, flips.astype(float), span])
    # pad/truncate to a fixed width so every scene yields the same-length dynamics vector
    out = np.zeros(8)
    out[:min(8, len(feat))] = feat[:8]
    return out


def event(seed):
    """One ground-truth world EVENT: the three paired modalities from a single scene."""
    objs = sample_scene(seed)
    pos, vel = simulate(objs)
    mid = len(pos) // 2
    return {
        "render": render(objs, pos_frame=pos[mid]),          # a representative frame
        "caption": caption(objs, vel_frame=vel[0]),          # description from launch velocities
        "dynamics": dynamics_signature(pos, vel),            # trajectory signature
        "objs": objs,
    }


if __name__ == "__main__":
    for s in range(6):
        e = event(s)
        print(f"scene {s}: {e['caption']}   | dyn={np.round(e['dynamics'], 2)}")
