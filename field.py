"""
FIELD — models are FORCES tensioning a shared field; the field SETTLES under them; you PROBE with a query
and READ how it settles. Built from the concept, not from tests. numpy only, self-contained.

THE OBJECT (one thing):
  Each model is an OPERATOR R (its frame) exerting a force on the shared field state s: it reads the CURRENT
  state through its frame and writes back a transformed contribution, so models drive EACH OTHER. The coupled
  flow is genuine feedback, NOT averaging — for conflicting frames the coupled operator CAN become unstable
  (spectral radius > 1). A bounded (saturating) force keeps a conflicted field from blowing up, so instead of
  diverging it TREMBLES. You probe with a query (a nudge) and read the settling TRACE:
    - forces agree   -> firm rest      -> the resting POINT is the answer (consensus).
    - forces diverge -> mushy point    -> info relocates to the TREMBLE -> trembling directions = competing answers.
  The read is the trace's TAIL MOTION (stopped->point, trembling->branches, rotating->ambiguous). The per-query
  TERRAIN is how stability varies with the query direction. The output is a STREAMING relaxation you watch until
  it converges / cycles / times out.

FAITHFULNESS (the one non-negotiable, a concept-check not a unit-test):
  if the coupled flow cannot reach spectral radius > 1 for conflicting frames, it is AVERAGING in disguise.
"""
import numpy as np


# ============================================================ models as forces (operators)
def frame_operator(D, seed, gain=1.0, rot=None):
    """A model as an OPERATOR R = Fᵀ B F: read into an orthonormal frame F, transform by B, write back.
    gain>1 = the model insists (over-relaxes past the state); rot=(a,b,ang) rotates a 2-plane (read-dim != write-dim,
    non-normal) — the frame-conflict that lets the coupled flow expand."""
    r = np.random.default_rng(seed)
    F = np.linalg.qr(r.normal(size=(D, D)))[0]
    B = np.eye(D) * gain
    if rot is not None:
        a, b, ang = rot
        c, s = np.cos(ang), np.sin(ang)
        B[a, a] = c * gain; B[a, b] = -s * gain; B[b, a] = s * gain; B[b, b] = c * gain
    return F.T @ B @ F


def coupled_jacobian(Rs, w, step):
    D = Rs[0].shape[0]
    return np.eye(D) + step * sum(wi * (R - np.eye(D)) for wi, R in zip(w, Rs))


def spectral_radius(J):
    return float(np.max(np.abs(np.linalg.eigvals(J))))


# ============================================================ the settling field
def settle(Rs, s0, w=None, iters=200, step=0.3, cap=2.0):
    """The field settles under the coupled forces. Each model i writes a BOUNDED correction toward its frame
    reading: force_i(s) = cap·tanh( (R_i s − s) / cap ). Bounded so a conflicted field trembles instead of
    blowing up; near s=0 it is the raw coupled feedback (Jacobian I+step·Σwᵢ(Rᵢ−I)) so ρ>1 is reachable.
    Returns the full trajectory (T_query, iters+1, D)."""
    Rs = [np.asarray(R, float) for R in Rs]
    n = len(Rs)
    w = np.ones(n) / n if w is None else np.asarray(w, float)
    I = np.eye(Rs[0].shape[0])
    s = np.asarray(s0, float).copy()
    traj = [s.copy()]
    for _ in range(iters):
        force = np.zeros_like(s)
        for wi, R in zip(w, Rs):
            corr = s @ (R - I).T                      # model reads current state through its frame, writes correction
            force = force + wi * (cap * np.tanh(corr / cap))   # bounded
        s = s + step * force
        traj.append(s.copy())
    return np.array(traj)


# ============================================================ the trace tail-motion read
def tail_read(traj, k=20):
    """Read the answer from the trace's TAIL MOTION (no fitted head).
      stopped (‖Δ‖→0)         -> CONSENSUS: the limit point.
      persistent 1-direction  -> SOFT: unresolved pull (magnitude=uncertainty).
      trembling / multi-dir   -> HELD-SUPERPOSITION: the trembling directions are the competing branches.
      rotating                -> AMBIGUOUS: path/framing dependent.
    traj: (iters+1, D) for one query. Returns a dict."""
    d = np.diff(traj, axis=0)                          # increments Δ_t (iters, D)
    tail = d[-k:]
    mags = np.linalg.norm(tail, axis=1)
    m0, m1 = float(mags[0] + 1e-12), float(mags[-1])
    # effective number of active tail directions (participation ratio of the tail-increment spectrum)
    tc = tail - tail.mean(0)
    sv = np.linalg.svd(tc, compute_uv=False) if tc.shape[0] > 1 else np.array([1.0])
    eff = float((sv.sum() ** 2) / (np.sum(sv ** 2) + 1e-12)) if sv.sum() > 0 else 1.0
    # rotation: consecutive increments turning (mean angle between Δ_t and Δ_{t+1})
    ang = 0.0
    if len(tail) > 2:
        u = tail / (np.linalg.norm(tail, axis=1, keepdims=True) + 1e-12)
        cos = np.clip(np.sum(u[:-1] * u[1:], axis=1), -1, 1)
        ang = float(np.degrees(np.arccos(cos)).mean())
    shrinking = m1 < 0.2 * m0
    if m1 < 1e-3:
        verdict = "CONSENSUS"
    elif eff >= 1.7:
        verdict = "HELD-SUPERPOSITION"
    elif ang > 40:
        verdict = "AMBIGUOUS(rotating)"
    elif shrinking:
        verdict = "CONSENSUS"
    else:
        verdict = "SOFT(unresolved pull)"
    return {"verdict": verdict, "tail_mag_first": m0, "tail_mag_last": m1,
            "tail_eff_dirs": eff, "tail_rotation_deg": ang, "point": traj[-1]}


# ============================================================ per-query terrain
def terrain(Rs, w, queries, step=0.3):
    """The local stability terrain along each query direction: gain = ‖q Jᵀ‖ (Jacobian at rest). >1 amplifying
    (contested), <1 damped (agreed). Returns per-query gains — the terrain the plain average is blind to."""
    J = coupled_jacobian(Rs, w if w is not None else np.ones(len(Rs)) / len(Rs), step)
    q = np.asarray(queries, float); q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    return np.linalg.norm(q @ J.T, axis=1)


# ============================================================ streaming / anytime read
def stream(Rs, s0, w=None, step=0.3, cap=2.0, max_iters=400, tol=1e-3, cycle_win=40):
    """Anytime read: a stream of refining states until it CONVERGES (limit=point), CYCLES (contested; the
    cycle extremes are the branches), or hits BUDGET (best-so-far + residual = uncertainty). Returns the
    manner of termination + the answer."""
    Rs = [np.asarray(R, float) for R in Rs]
    w = np.ones(len(Rs)) / len(Rs) if w is None else np.asarray(w, float)
    I = np.eye(Rs[0].shape[0]); s = np.asarray(s0, float).copy(); traj = [s.copy()]; residuals = []
    for t in range(max_iters):
        force = sum(wi * (cap * np.tanh((s @ (R - I).T) / cap)) for wi, R in zip(w, Rs))
        prev = s; s = s + step * force; traj.append(s.copy())
        r = float(np.linalg.norm(s - prev)); residuals.append(r)
        if r < tol:
            return {"manner": "CONVERGED", "iters": t + 1, "point": s, "residual": r}
        if t > 2 * cycle_win:                          # cycle: recent residual persistent but state revisiting a band
            recent = residuals[-cycle_win:]
            if np.mean(recent) > tol and np.std(recent) < 0.25 * np.mean(recent):
                band = np.array(traj[-cycle_win:])
                return {"manner": "CYCLED", "iters": t + 1, "residual": float(np.mean(recent)),
                        "branch_lo": band.min(0), "branch_hi": band.max(0), "band": band}
    return {"manner": "BUDGET", "iters": max_iters, "point": s, "residual": residuals[-1]}
