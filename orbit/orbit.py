"""
orbit.py — the dynamics that turn the instrument into a (renderable) trajectory.
A walk on charts: step through P, add the MZ momentum tilt beta*psi*(kappa*p) with the damped-oscillator
p <- e^-gamma p + da - omega^2 a driven by the walk's own motion (kernel modes = corpus-measured).
kappa=0 exactly reproduces the memoryless P-walk (same rng draws). Emission is concatenative (a real corpus
state of the sampled chart) so every emitted state is on-manifold; coherence is judged by splice continuity.
"""
import numpy as np
from physics import step as phys_step


def _chart_members(labels, k):
    return [np.where(labels == c)[0] for c in range(k)]


def generate(inst, n_steps=240, kappa=0.0, beta=1.0, temp=1.0, seed=0):
    """Walk the instrument, emitting a corpus state per step. Returns emitted states + the resolved a(t)."""
    P, psi, states = inst["P"], inst["psi"], inst["states"]
    gamma, omega = inst["gamma"], inst["omega"]
    k = P.shape[0]; K = psi.shape[1]
    members = _chart_members(inst["labels"], k)
    rng = np.random.default_rng(seed)
    spread = float(np.log(P[P > 0]).std())                                # corpus-calibrated tilt scale

    c = int(rng.integers(k))
    p = np.zeros(K); a_prev = psi[c].copy()
    emitted_idx, a_series = [], []
    for _ in range(n_steps):
        a = psi[c]
        p = np.exp(-gamma) * p + (a - a_prev) - (omega ** 2) * a_prev     # damped-oscillator momentum
        a_prev = a
        logits = np.log(P[c] + 1e-12) / temp
        if kappa != 0.0:
            raw = psi @ p                                                 # momentum direction over charts
            sd = raw.std()
            if sd > 1e-9:
                logits = logits + kappa * spread * (raw / sd)             # tilt spread = kappa * base spread
        logits -= logits.max()
        prob = np.exp(logits); prob = prob / prob.sum()
        c_next = int(rng.choice(k, p=prob))
        pool = members[c_next]
        idx = int(rng.choice(pool)) if len(pool) else int(rng.integers(len(states)))
        emitted_idx.append(idx); a_series.append(psi[c_next].copy())
        c = c_next
    return states[np.array(emitted_idx)], np.array(a_series)


def coherence(emitted, ref_scale):
    """Splice continuity = how close each emitted state is to the true physical successor of the previous one
    (concatenative continuation-miss). Lower = the orbit follows real dynamics. Returns mean normalized miss."""
    miss = [np.linalg.norm(emitted[t + 1] - phys_step(emitted[t])) for t in range(len(emitted) - 1)]
    return float(np.mean(miss) / (ref_scale + 1e-12))


def render_artifact(emitted, path, every=6, cols=20, cell=96):
    """Contact-sheet PNG of the emitted trajectory (every Nth frame) + an animated GIF alongside."""
    from PIL import Image
    from physics import render
    frames = [render(emitted[i]) for i in range(0, len(emitted), every)]
    rows = int(np.ceil(len(frames) / cols))
    sheet = np.zeros((rows * cell, cols * cell, 3), np.uint8)
    for i, f in enumerate(frames):
        r, cc = divmod(i, cols)
        sheet[r * cell:(r + 1) * cell, cc * cell:(cc + 1) * cell] = f
    Image.fromarray(sheet).save(path)
    gif = path.replace(".png", ".gif")
    ims = [Image.fromarray(render(emitted[i])) for i in range(0, len(emitted), 2)]
    if ims:
        ims[0].save(gif, save_all=True, append_images=ims[1:], duration=60, loop=0)
    return path, gif
