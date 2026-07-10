"""
FLUID_PIPELINE — wire the theoretically-correct feedback fluid (fluid_settle) into the PIPELINE recurrence.

This is FIX-ORDER STEP 1: make the pipeline's default settling recurrence the operator-feedback FLUID
(CONSTRUCT #1-2 / THEORY T1-T2), NOT feed-forward averaging. It does NOT reimplement the fluid — it REUSES
fluid_settle's operator machinery (coupled_jacobian, instability_descent, frame_operator, spec_radius). The
only new thing here is the FAITHFUL construction of the operators Rᵢ from REAL interfaces.

THE CRUX (models as OPERATORS derived from real interfaces, not fixed targets):
  A real interface is an aligned VECTOR view: a per-modality reconstruction f (T×D) of the shared medium
  z (T×D). The shipped `settle` treats each f as a FIXED target and pulls the state toward mean(fᵢ) — that
  is averaging (Jacobian ≤ 1, structurally contractive, can never be unstable or exclude).

  To be the fluid, each interface must become an OPERATOR Rᵢ: the model's read/write map on the shared
  medium. We DERIVE it from the interface's own data — Rᵢ is the linear map that takes a medium state and
  produces this model's reconstruction of it:  f ≈ z @ Rᵢᵀ  ⇒  Rᵢᵀ = lstsq(z, f)  (fit on train only).
  The coupled feedback flow is then  S ← S + step·Σ wᵢ (S Rᵢᵀ − S) = S Jᵀ, J(w) = I + step·Σ wᵢ(Rᵢ − I).

  This is genuine model→model feedback: each model reads the CURRENT shared state through its frame Rᵢ and
  writes back its correction, so the models drive each other through the shared J. Because a model's frame
  can be NON-NORMAL / gain>1 (an incompatible interface that reconstructs the medium through a distorting
  frame), J(w) CAN exceed spectral radius 1 — genuine mutual instability, which averaging cannot have.

FAITHFUL EMERGENT BEHAVIOUR (verified, not gamed):
  - CONVERGENT / aligned interfaces reconstruct the medium faithfully ⇒ Rᵢ ≈ I ⇒ (Rᵢ−I) ≈ 0 ⇒ coupled ρ ≤ 1
    ⇒ the fluid settle NO-OPS (contracts to the consensus). This is the honest null on real convergent senses
    (matches settle_real / the xresolve convergence finding) — the mechanism is capable but real interfaces
    rarely conflict.
  - INCOMPATIBLE / injected interfaces (a model insisting through a non-normal gain frame) ⇒ Rᵢ non-normal,
    gain>1 ⇒ coupled ρ > 1 ⇒ genuine mutual instability, and the fluid ROUTES AROUND a destabilizer
    (instability_descent drives its routing weight → 0). Averaging can do neither.
"""
import numpy as np
import fluid_settle as fl


# ---------------------------------------------------------------- operators from REAL interfaces
def interface_operator(f, z, ntr):
    """DERIVE the operator Rᵢ from a real interface's own data. The interface f (T×D) is the model's
    reconstruction of the medium z (T×D). Rᵢ is the linear map with f ≈ z @ Rᵢᵀ, i.e. the model's
    read/write frame on the medium: Rᵢᵀ = lstsq(z, f) (fit on TRAIN only, held-out-honest). A faithful
    aligned interface gives Rᵢ ≈ I; an incompatible (non-normal, gain>1) frame gives a gain-capable Rᵢ."""
    Rt, *_ = np.linalg.lstsq(np.asarray(z)[:ntr], np.asarray(f)[:ntr], rcond=None)
    return Rt.T


def operators_from_ifaces(ifaces, z, ntr):
    return [interface_operator(f, z, ntr) for f in ifaces]


def coupled_radius(ifaces, z, ntr, w=None, step=fl.STEP):
    """Spectral radius of the coupled operator J(w) for operators derived from these interfaces."""
    Rs = operators_from_ifaces(ifaces, z, ntr)
    if w is None:
        w = np.ones(len(Rs)) / len(Rs)
    return fl.spec_radius(fl.coupled_jacobian(Rs, w, step)), Rs


def routing_weights(Rs, step=fl.STEP, descent_iters=200):
    """Fluid exclusion (standalone, for tests/analysis): at equal weights, if the coupled flow is already
    stable (ρ ≤ 1) keep equal weights; if UNSTABLE (ρ > 1), descend the coupled instability over the routing
    simplex → a destabilizing model's weight is driven toward 0. NOTE: the pipeline recurrence does NOT use
    this as a pre-phase — it folds ONE `descent_step` INSIDE the settle loop (see settle_fluid / INV3)."""
    n = len(Rs)
    w_eq = np.ones(n) / n
    rho_eq = fl.spec_radius(fl.coupled_jacobian(Rs, w_eq, step))
    if rho_eq <= 1.0 + 1e-9:
        return w_eq, rho_eq, False
    w, _hist = fl.instability_descent(Rs, step=step, iters=descent_iters)
    return w, rho_eq, True


def descent_step(Rs, w, step=fl.STEP, eta=1.2, eps=1e-3):
    """ONE projected-gradient step descending the coupled mutual-instability — the faithfulness/contraction
    LOSS TERM applied INSIDE the settle loop (NOT a separate pre-phase; THEORY T1/#5). If the coupled flow is
    already stable (growth ≤ 1) it returns w UNCHANGED — so on convergent interfaces the settle is identical to
    uniform routing (native no-op). On an unstable coupling it routes weight away from the destabilizer (T2),
    step by step, as part of the single settling objective."""
    base = fl._growth(fl.coupled_jacobian(Rs, w, step))
    if base <= 1.0 + 1e-9:
        return w
    g = np.zeros(len(Rs))
    for i in range(len(Rs)):
        wp = w.copy(); wp[i] += eps
        g[i] = (fl._growth(fl.coupled_jacobian(Rs, wp, step)) - base) / eps
    return fl._simplex_proj(w - eta * g)


# ---------------------------------------------------------------- the fluid pipeline settle
def settle_fluid(ifaces, z, guard=True, nudge=None, beta=0.0, init=None,
                 iters=None, step=fl.STEP, ntr=None, structured_fn=None, D=None):
    """PIPELINE RECURRENCE as the operator-feedback FLUID (replaces averaging as the default).

    Each interface becomes an operator Rᵢ (interface_operator). The shared state S settles under the
    COUPLED operator flow  S ← S + step·Σ wᵢ (S Rᵢᵀ − S)  = S Jᵀ — genuine model→model feedback, whose
    Jacobian J(w) = I + step·Σ wᵢ(Rᵢ−I) CAN exceed radius 1 for incompatible frames (unlike averaging).
    Routing weights w come from fluid exclusion (routing_weights): a destabilizer is routed around.

    Returns the SAME tuple as coherentflow.settle: (state, memory, res, circ_norm). The held-structure
    guard / memory is computed on the settled fluid state exactly as before (that read is a separate,
    later fix-order step — STEP 1 changes ONLY the recurrence)."""
    import coherentflow as cf
    if iters is None: iters = cf.ITERS
    if ntr is None: ntr = cf.NTR
    if structured_fn is None: structured_fn = cf.structured
    if D is None: D = cf.D

    ifaces = [np.asarray(f, float) for f in ifaces]
    z = np.asarray(z, float)

    Rs = operators_from_ifaces(ifaces, z, ntr)
    n = len(Rs); w = np.ones(n) / n                   # routing starts uniform; the loop descends instability

    S = np.mean(ifaces, axis=0) if init is None else np.asarray(init, float).copy()
    res = []
    for _ in range(iters):
        w = descent_step(Rs, w, step)                 # faithfulness/contraction as a LOSS TERM, INSIDE the loop
        J = fl.coupled_jacobian(Rs, w, step)          # coupled operator with the current (stabilizing) routing
        prev = S
        S = S @ J.T                                   # coupled operator feedback: every model reads S, writes back
        if nudge is not None and beta:
            S = S + step * beta * np.asarray(nudge, float)   # weak EqProp nudge (equilibrium-response probe only)
        res.append(float(np.linalg.norm(S - prev)))

    # held-structure guard on the settled fluid state (unchanged read — separate fix-order step)
    memory = {}
    if guard:
        for i, f in enumerate(ifaces):
            d = f - S
            is_s, P, _sc = structured_fn(d, z)
            if is_s:
                memory[i] = d @ P
    else:
        for i, f in enumerate(ifaces):
            memory[i] = f - S

    circ_norm = float(np.linalg.norm(np.sum([memory[i] for i in memory], axis=0))) if memory else 0.0
    return S, memory, res, circ_norm


# ---------------------------------------------------------------- incompatible-interface builder (for conformance)
def incompatible_interfaces(z, ntr, specs, noise=0.05, seed_base=100):
    """Build a set of INCOMPATIBLE interfaces as genuine vector views: each reconstructs the medium z
    through a non-normal / gain frame (fluid_settle.frame_operator). These model the frame-conflict the
    theory says the mechanism must be CAPABLE of (real convergent interfaces rarely produce it). The
    interface returned is a real (T×D) reconstruction — the pipeline then DERIVES the operator back from
    it (interface_operator), so the demonstrated instability is on the pipeline's real operator path, not
    a hand-set operator. `specs` = list of (seed, gain, rot) passed to frame_operator."""
    D = z.shape[1]
    out = []
    for k, (seed, gain, rot) in enumerate(specs):
        F = fl.frame_operator(D, seed, gain=gain, rot=rot)
        r = np.random.default_rng(seed_base + k)
        f = z @ F.T + noise * r.normal(size=z.shape)
        out.append(f)
    return out
