"""
P4 / G4 — disagreement-meter validity. Two edges over a set of members; the cycle-cost holonomy meter must
separate CLONES (identical members -> agree -> cycle cost below floor) from DISJOINT members (genuinely
different -> disagree -> cycle cost above floor). Cycle cost = how far a member's self-map, composed once
around the two anchors, is from identity, net of the solver-restart floor φ_solver (§5.3, §7).

Concretely for a member v with couplings to anchor A (pi_A: n×mA) and anchor B (pi_B: n×mB):
  round-trip on v's own support  T_v = row_normalize(pi_A a_A) barycentric-> ... -> back to v
  we use the composite self-coupling  S_v = pi_A diag(1/a_A) pi_A^T  vs  pi_B diag(1/a_B) pi_B^T
  cycle cost = || S_A_v - S_B_v ||_F on v's support (how differently the two anchors reconstruct v).
Clones make the two anchors see the same shared geometry -> S agree; disjoint members pull the two anchors
apart -> S disagree.
"""
import numpy as np
from ..geometry.clouds import cloud_to_Dw
from ..energy import functional as EN
from ..transport import gw
from . import substrate as S
from . import g1_probe as G1


def _self_coupling(pi, a):
    """v->anchor->v reconstruction operator on v's own support: pi diag(1/a) pi^T (row-normalized)."""
    Sm = pi @ np.diag(1.0 / np.maximum(a, 1e-9)) @ pi.T
    r = Sm.sum(1, keepdims=True)
    return Sm / np.maximum(r, 1e-12)


def cycle_cost(Ds, ws, members_A, members_B, eps=0.08, m=4, n_outer=15, init_seed=1):
    """Equilibrate two anchors (edge A over members_A, edge B over members_B); for each member in BOTH,
    cycle cost = ||S_A_v - S_B_v||_F. Returns mean over shared members. init_seed varies the anchor init
    (for the restart floor)."""
    def anchor(idx):
        De = np.random.default_rng(init_seed).random((m, m)); De = (De + De.T) / 2
        np.fill_diagonal(De, 0); De /= np.median(De[np.triu_indices(m, 1)])
        a = np.full(m, 1.0 / m)
        pis, De, a, _f, _c = EN.equilibrate([Ds[i] for i in idx], [ws[i] for i in idx],
                                            De, a, a.copy(), eps=eps, n_outer=n_outer)
        return dict(zip(idx, pis)), a
    piA, aA = anchor(members_A)
    piB, aB = anchor(members_B)
    shared = [i for i in members_A if i in members_B]
    costs = [np.linalg.norm(_self_coupling(piA[i], aA) - _self_coupling(piB[i], aB)) for i in shared]
    return float(np.mean(costs))


def solver_floor_cycle(Ds, ws, members_A, members_B, eps=0.08, R=8):
    """φ_solver for the cycle: spread of cycle cost across random-restart (varied anchor init) on fixed clouds."""
    vals = [cycle_cost(Ds, ws, members_A, members_B, eps=eps, n_outer=8, init_seed=100 + r) for r in range(R)]
    return float(np.std(vals)), float(np.median(vals))


def build_clouds(models, r=3, T=1, n=110, seed=0):
    U = S.latent_traffic(max(T, 3), r, seed=seed)
    inp = G1._warp_inputs(U[1], n)
    Dws = [cloud_to_Dw(M(inp)) for M in models]
    return [d for d, _ in Dws], [w for _, w in Dws]


def run(n=110):
    print("P4 / G4 — cycle-cost meter validity (clones below floor, disjoint above)")
    # DISJOINT: 3 genuinely different architectures
    disj = S.make_diverse_models(3, seed0=200)
    Dd, wd = build_clouds(disj, n=n)
    # CLONES: the SAME model repeated (identical members)
    clone = [S.make_diverse_models(1, seed0=200)[0]] * 3
    Dc, wc = build_clouds(clone, n=n)

    mA, mB = [0, 1, 2], [1, 2]                    # two overlapping edges (shared members 1,2)
    c_disj = cycle_cost(Dd, wd, mA, mB)
    c_clone = cycle_cost(Dc, wc, mA, mB)
    sstd, smed = solver_floor_cycle(Dd, wd, mA, mB)
    floor = c_clone + 3 * sstd                      # clone-level agreement + 3σ solver spread
    print(f"  clone cycle cost   = {c_clone:.4f}")
    print(f"  disjoint cycle cost= {c_disj:.4f}")
    print(f"  φ_solver std={sstd:.4f} (median {smed:.4f}); floor = clone + 3σ = {floor:.4f}")
    ok = bool(c_disj > floor)
    print(f"  G4 verdict: {'PASS' if ok else 'FAIL'} — clone {c_clone:.3f} < floor {floor:.3f} < disjoint "
          f"{c_disj:.3f}  (separation {c_disj / max(c_clone, 1e-6):.1f}×)")
    return {"clone": c_clone, "disjoint": c_disj, "phi_solver_std": sstd, "floor": floor, "G4": bool(ok)}


if __name__ == "__main__":
    run()
