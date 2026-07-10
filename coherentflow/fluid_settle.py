"""
FLUID_SETTLE — the theoretically-CORRECT feedback fluid (CONSTRUCT #1-2), built correctness-first.

SEPARATE from the shipped averaging settle (`coherentflow.settle`), which is left intact and correctly
labelled feed-forward averaging (see MECHANISM_CHECK.md). This module does NOT modify or "fix" that one — it
is the theoretically-correct object, defined by the properties it MUST have, with the code proving it has them.

THE DISTINCTION (why averaging is not the fluid): averaging is `s ← s + step·mean(fᵢ − s)` with FIXED targets
fᵢ, so its Jacobian is `(1−step)I + step·mean(Pᵢ)` with spectral radius ≤ 1 — structurally contractive, can
never be mutually unstable, can never route around a destabilizing model. The fluid needs models as OPERATORS.

MODELS AS OPERATORS (not fixed targets): each model i is an operator Rᵢ that reads the shared state through
its frame and writes back a frame-transformed contribution, so the models genuinely drive each other. The
coupled flow is `s ← s + step·Σ wᵢ (Rᵢ − I) s`, Jacobian `J(w) = I + step·Σ wᵢ (Rᵢ − I)`. Because the frame
operators can be NON-NORMAL / gain>1 (a model insisting past the current state, or a read-dim≠write-dim cycle),
J(w) CAN have spectral radius > 1 for incompatible frames — genuine mutual instability, which averaging cannot.

THREE ACCEPTANCE CRITERIA (all required; the definition of "correct", independent of utility):
 1. MUTUAL INSTABILITY — some model set makes the coupled radius exceed 1 (amplify), not contract-to-mean.
 2. FLUID EXCLUSION — descending the coupled instability over routing weights w decisively routes around a
    destabilizing model (its weight → ~0, system pulled back to stable). Symmetric RIVALS are instead BALANCED
    (not excluded) → their contested subspace is held.
 3. INTRINSIC OUTPUT — the query is a perturbation to the settled state; re-settle (bounded); the equilibrium
    SHIFT is the answer (no external readout head). (a) agreed → moves-as-asked/collapses to consensus;
    (b) adversarial → bounded; (c) contested → HELD-SUPERPOSITION (does not collapse to one consensus answer).

HONEST THEORETICAL FINDING (reported, not hidden): criteria 1-2 are properties of the LINEARIZED coupled
operator and hold for the linear feedback flow. Criterion 3c (robust held-superposition — multiple branches
BOTH stable) is NOT achievable by linear feedback: a linear flow holds a contested subspace only at fragile
exact-marginal stability (ρ=1, measure-zero; any drift makes one branch dominate → collapse). ROBUST
held-superposition requires NONLINEAR MULTISTABILITY — a bounded saturation that turns rival model directions
into two stable wells. The fluid's intrinsic settle is therefore nonlinear (saturating), and the consensus
fixed point is UNSTABLE on contested subspaces (the fluid commits to a held branch rather than blandly
averaging to the midpoint). This is a real finding about the theory: the paraconsistent hold is a nonlinear,
multistable phenomenon, not a linear one.
"""
import os
import json
import numpy as np

STEP = 0.3


def spec_radius(J):
    return float(np.max(np.abs(np.linalg.eigvals(J))))


# ---------------------------------------------------------------- models as operators
def frame_operator(D, seed, gain=1.0, rot=None):
    """A model as an OPERATOR Rᵢ = Fᵢᵀ Bᵢ Fᵢ: read into the model's orthonormal frame Fᵢ, transform by Bᵢ,
    write back. gain>1 = the model insists (over-relaxes past the state); rot=(a,b,ang) makes Bᵢ rotate a
    2-plane so read-dim≠write-dim (non-normal) — the frame-conflict that lets the coupled flow expand."""
    r = np.random.default_rng(seed)
    F = np.linalg.qr(r.normal(size=(D, D)))[0]
    B = np.eye(D) * gain
    if rot is not None:
        a, b, ang = rot
        c, s = np.cos(ang), np.sin(ang)
        B[a, a] = c * gain; B[a, b] = -s * gain; B[b, a] = s * gain; B[b, b] = c * gain
    return F.T @ B @ F


def coupled_jacobian(Rs, w, step=STEP):
    D = Rs[0].shape[0]
    I = np.eye(D)
    return I + step * sum(wi * (R - I) for wi, R in zip(w, Rs))


def averaging_jacobian(Ps, w, step=STEP):
    """The shipped-settle Jacobian for comparison: symmetric projectors Pᵢ (eigenvalues ≤ 1) => ρ ≤ 1."""
    D = Ps[0].shape[0]
    return (1 - step) * np.eye(D) + step * sum(wi * P for wi, P in zip(w, Ps))


# ---------------------------------------------------------------- instability descent (routing)
def _growth(J, k=60, seed=0):
    """Robust power-method spectral-radius estimate (always real ≥ 0; safe for oscillatory/complex leading
    eigenvalues where an eigenvector-sensitivity gradient is unreliable)."""
    r = np.random.default_rng(seed)
    v = r.normal(size=J.shape[0]); v /= np.linalg.norm(v); g = 1.0
    for _ in range(k):
        v = J @ v; g = np.linalg.norm(v); v = v / (g + 1e-12)
    return g


def _simplex_proj(v):
    u = np.sort(v)[::-1]; cs = np.cumsum(u) - 1; ind = np.arange(1, len(v) + 1)
    cond = u - cs / ind > 0; rho = ind[cond][-1]; theta = cs[rho - 1] / rho
    return np.maximum(v - theta, 0)


def instability_descent(Rs, step=STEP, iters=400, eta=0.5, eps=1e-3):
    """The fluid descends its OWN measured growth by adjusting routing weights w on the simplex. A model whose
    inclusion amplifies the coupled flow gets down-weighted (exclusion); symmetric rivals get balanced. Uses a
    finite-difference gradient of the power-method growth — robust to oscillatory instability."""
    n = len(Rs); w = np.ones(n) / n; hist = []
    for _ in range(iters):
        base = _growth(coupled_jacobian(Rs, w, step))
        hist.append(spec_radius(coupled_jacobian(Rs, w, step)))
        g = np.zeros(n)
        for i in range(n):
            wp = w.copy(); wp[i] += eps
            g[i] = (_growth(coupled_jacobian(Rs, wp, step)) - base) / eps
        w = _simplex_proj(w - eta * g)
    return w, hist


# ---------------------------------------------------------------- nonlinear intrinsic settle (held-superposition)
def make_field(D, agreed_dirs, rival_dirs, contract=0.8, beta=1.6, cap=1.4, gain=1.2):
    """Nonlinear feedback field from models tied to preferred directions. AGREED models CONTRACT their
    direction toward consensus (single stable well at 0). RIVAL models REINFORCE their direction with a
    BOUNDED saturating gain — bistable (wells at ±, consensus 0 UNSTABLE) since beta·gain·cap > 1, so the
    fluid COMMITS to a held branch rather than averaging to the midpoint. Each model acts only along its own
    direction (no cross-contraction), so rival directions give an independent multistable contested landscape
    = held-superposition. Weight order in w: agreed first, then rivals."""
    agreed = [d / np.linalg.norm(d) for d in agreed_dirs]
    rivals = [d / np.linalg.norm(d) for d in rival_dirs]

    def field(s, w):
        out = np.zeros(D); k = 0
        for p in agreed:
            out += w[k] * (-contract * (p @ s)) * p; k += 1                 # contract -> consensus 0
        for p in rivals:
            proj = p @ s
            out += w[k] * (cap * np.tanh(beta * gain * proj) - proj) * p; k += 1  # bounded bistable reinforce
        return out
    return field


def fluid_settle(field, w, delta, step=STEP, K=400):
    """Intrinsic settle: query = perturbation delta; re-settle the (nonlinear, bounded) fluid; return the
    equilibrium SHIFT (the answer — no external readout). Bounded by the saturation, so adversarial queries
    cannot blow up; multistable, so contested queries hold a branch instead of collapsing to the midpoint."""
    s = np.asarray(delta, float).copy()
    for _ in range(K):
        s = s + step * field(s, w)
    return s


# ---------------------------------------------------------------- the three acceptance tests
def acceptance_1_mutual_instability(D=6):
    agree = [frame_operator(D, 1, gain=0.9), frame_operator(D, 1, gain=0.85), frame_operator(D, 1, gain=0.8)]
    conflict = [frame_operator(D, 10, gain=1.3, rot=(0, 1, 1.2)),
                frame_operator(D, 11, gain=1.3, rot=(2, 3, 1.2)),
                frame_operator(D, 12, gain=1.3, rot=(4, 5, 1.2))]
    r = np.random.default_rng(3)
    Ps = [(lambda Q: Q @ Q.T)(np.linalg.qr(r.normal(size=(D, D)))[0][:, :3]) for _ in range(3)]
    rho_agree = spec_radius(coupled_jacobian(agree, [1 / 3] * 3))
    rho_conflict = spec_radius(coupled_jacobian(conflict, [1 / 3] * 3))
    rho_avg = spec_radius(averaging_jacobian(Ps, [1 / 3] * 3))
    # minimal provable skew (2D): read-dim ≠ write-dim in a cycle -> |λ| = sqrt(1+step²) > 1
    A1 = np.array([[0.0, 1.0], [0.0, 0.0]]); A2 = np.array([[0.0, 0.0], [-1.0, 0.0]])
    rho_skew = spec_radius(np.eye(2) + STEP * (A1 + A2))
    passed = (rho_conflict > 1.0) and (rho_avg <= 1.0 + 1e-9) and (rho_skew > 1.0)
    return {"rho_conflict_fluid": rho_conflict, "rho_agree_fluid": rho_agree,
            "rho_averaging_shipped": rho_avg, "rho_minimal_skew_2d": rho_skew,
            "fluid_can_exceed_1": rho_conflict > 1.0, "averaging_bounded_by_1": rho_avg <= 1.0 + 1e-9,
            "passed": bool(passed)}


def acceptance_2_exclusion(D=6):
    # 3 compatible (agree, contracting) + 1 rogue (non-normal, gain>1) that destabilizes when included
    Rs = [frame_operator(D, 1, gain=0.9), frame_operator(D, 1, gain=0.9), frame_operator(D, 1, gain=0.9),
          frame_operator(D, 99, gain=1.5, rot=(0, 1, 1.4))]
    rho_equal = spec_radius(coupled_jacobian(Rs, [1 / 4] * 4))
    w, hist = instability_descent(Rs, STEP)
    rho_final = spec_radius(coupled_jacobian(Rs, w))
    rogue_w = float(w[3])
    passed = (rho_equal > 1.0) and (rogue_w < 0.02) and (rho_final < 1.0)
    return {"rho_equal_weights": rho_equal, "unstable_at_equal_weights": rho_equal > 1.0,
            "weights_after_descent": [float(x) for x in w], "rogue_weight": rogue_w,
            "rogue_excluded": rogue_w < 0.02, "rho_after_descent": rho_final,
            "stabilized": rho_final < 1.0, "passed": bool(passed)}


def acceptance_3_intrinsic_output(D=6):
    # agreed direction (e0) orthogonal to the contested plane; two RIVAL models on e3,e4 (conflicting branches)
    u_agree = np.zeros(D); u_agree[0] = 1.0
    uA = np.zeros(D); uA[3] = 1.0                      # rival A branch
    uB = np.zeros(D); uB[4] = 1.0                      # rival B branch (orthogonal, conflicting)
    field = make_field(D, agreed_dirs=[u_agree], rival_dirs=[uA, uB])
    w = np.array([0.34, 0.33, 0.33])                  # agreed, rivalA, rivalB

    # (a) agreed query -> collapses toward consensus (0), moves-as-asked into agreement
    r_agree = fluid_settle(field, w, 0.5 * u_agree)
    agreed_norm = float(np.linalg.norm(r_agree))

    # (c) contested queries spanning the rival plane -> settle to a held branch; over queries BOTH appear
    contested_in = [np.cos(t) * uA + np.sin(t) * uB for t in np.linspace(0, 2 * np.pi, 16)]
    contested_out = np.stack([fluid_settle(field, w, d)[[3, 4]] for d in contested_in])
    sv = np.linalg.svd(contested_out - contested_out.mean(0), compute_uv=False)
    held_rank2 = bool(sv[1] > 0.2 * sv[0])
    branches = sorted({round(float(fluid_settle(field, w, a * uA)[3]), 2) for a in (-0.5, 0.5)})

    # (b) adversarial query (huge magnitude) -> bounded by saturation, does not blow up
    adv = [float(np.linalg.norm(fluid_settle(field, w, m * (uA + uB), K=1000))) for m in (5, 50, 500)]
    bounded = max(adv) < 10.0

    passed = (agreed_norm < 0.1) and held_rank2 and bounded
    return {"agreed_query_response_norm": agreed_norm, "agreed_collapses_to_consensus": agreed_norm < 0.1,
            "contested_response_singular_values": [float(x) for x in sv],
            "held_superposition_rank2": held_rank2, "held_branches": branches,
            "adversarial_response_norms": adv, "adversarial_bounded": bounded, "passed": bool(passed)}


def main():
    HERE = os.path.dirname(os.path.abspath(__file__))
    a1 = acceptance_1_mutual_instability()
    a2 = acceptance_2_exclusion()
    a3 = acceptance_3_intrinsic_output()

    print("=" * 74)
    print("FLUID_SETTLE — theoretically-correct feedback fluid: acceptance criteria")
    print("=" * 74)
    print(f"[1] MUTUAL INSTABILITY  {'PASS' if a1['passed'] else 'FAIL'}")
    print(f"    conflicting frames rho = {a1['rho_conflict_fluid']:.3f} (>1 = fluid can amplify), "
          f"averaging rho = {a1['rho_averaging_shipped']:.3f} (<=1, cannot), skew-2d = {a1['rho_minimal_skew_2d']:.3f}")
    print(f"[2] FLUID EXCLUSION     {'PASS' if a2['passed'] else 'FAIL'}")
    print(f"    equal-weights rho = {a2['rho_equal_weights']:.3f} (unstable) -> descent routes around rogue: "
          f"rogue weight {a2['rogue_weight']:.4f} (~0), rho -> {a2['rho_after_descent']:.3f} (<1)")
    print(f"[3] INTRINSIC OUTPUT    {'PASS' if a3['passed'] else 'FAIL'}")
    print(f"    (a) agreed query response norm {a3['agreed_query_response_norm']:.4f} (collapse to consensus)")
    print(f"    (b) adversarial responses {['%.2f' % x for x in a3['adversarial_response_norms']]} (bounded)")
    print(f"    (c) contested held-superposition rank-2 = {a3['held_superposition_rank2']} "
          f"(branches {a3['held_branches']}, singular values "
          f"{[round(x,2) for x in a3['contested_response_singular_values'][:2]]})")
    all_pass = a1["passed"] and a2["passed"] and a3["passed"]
    print("-" * 74)
    print(f"ALL ACCEPTANCE CRITERIA PASS: {all_pass}")
    print("Note: criteria 1-2 hold for the LINEAR coupled operator; robust held-superposition (3c) requires the")
    print("NONLINEAR multistable settle — linear feedback gives only fragile marginal hold (reported finding).")

    out = {"step": STEP, "acceptance_1_mutual_instability": a1, "acceptance_2_fluid_exclusion": a2,
           "acceptance_3_intrinsic_output": a3, "all_pass": bool(all_pass),
           "finding": ("Criteria 1-2 (mutual instability, exclusion) are linear-coupled-operator properties. "
                       "Criterion 3c (robust held-superposition) is NOT achievable by linear feedback (fragile "
                       "marginal hold only) — it requires nonlinear multistability. The paraconsistent hold is a "
                       "nonlinear phenomenon: the consensus is unstable on contested subspaces and the fluid "
                       "commits to a held branch rather than averaging to the midpoint.")}
    json.dump(out, open(os.path.join(HERE, "fluid_settle_results.json"), "w"), indent=1)
    print("\nwrote fluid_settle_results.json")


if __name__ == "__main__":
    main()
