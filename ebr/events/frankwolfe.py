"""
events/frankwolfe.py — structural events re-derived from F (spec v1.1 #1). NO second decision channel: the
anchor is a free-support unbalanced measure and every structural move (grow / park / revive / merge) is a
conditional-gradient (Frank–Wolfe) step on that measure under the SAME functional F that runs everything.

grow(): the OSCILLATOR is F, not the Hankel. Proposal is an ORACLE: the dominant UNEXPLAINED residual
direction, barycentrically projected (any heuristic is admissible because acceptance is strict F-descent).
Accept iff re-equilibrated F strictly decreases net of the τ mass-creation cost already in F. Self-quenching
is intrinsic: once the anchor explains the traffic, no proposed atom lowers F, and growth stops — the Hankel
is never consulted by the mechanism. The Hankel/poles are INSTRUMENT (they gate claims), never mechanism.

park(): an atom whose F-optimal mass falls to ~0 is dropped by the same descent (unbalanced a-block); the
grow/park timescale asymmetry is the creation-vs-annihilation cost asymmetry in the unbalanced term, one
constant, not a separate clock.

SCOPE (FIX-2, honest): ATOM-level FW (grow / park / revive) is implemented and validated here. HYPEREDGE
spawn / merge are DERIVED as the same move on a level-2 measure over port-subsets (oracle = residual
co-clustering proposes a subset U; accept iff instantiating its sub-anchor Z_U with the γ gluing term
strictly decreases F net of Z_U's creation cost) — design sound, no wall found — but NOT yet implemented or
validated. The two-edge topology used in experiments/g4_meter.py is a LABELED experimental fixture, not a
discovered structure. Do not claim spawn/merge are "one move" in code until they are built.
"""
import numpy as np
from ..transport import gw
from ..energy import functional as EN
from ..hankel import residual as H


def _residual_direction(D, w, pi):
    """Dominant UNEXPLAINED direction at a port: top eigenvector of the anchor-deflated Gram, as a
    nonnegative distribution over the port's n points (the FW candidate's coupling column)."""
    G = H.gram_from_D(D)
    Ghat = H.deflate(G, pi)
    ev, V = np.linalg.eigh((Ghat + Ghat.T) / 2)
    d = V[:, -1] ** 2                      # squared top eigvec -> nonneg, emphasizes dominant direction
    s = d.sum()
    return (d / s) if s > 0 else w.copy()


def propose_atom(Ds, ws, pis, a, delta=0.08):
    """Append one FW candidate atom: its coupling column at each member is the residual direction (mass δ),
    rows kept on w (semi-relaxed). Returns augmented (pis', a')."""
    pis2 = []
    for D, w, pi in zip(Ds, ws, pis):
        c = _residual_direction(D, w, pi)[:, None] * delta
        pis2.append(np.concatenate([pi * (1 - delta), c], axis=1))
    a2 = np.concatenate([a * (1 - delta), [delta]])
    return pis2, a2


def grow(Ds, ws, De, a, abar, eps=0.08, tau=1.0, max_atoms=12, n_outer=15, rel_tol=0.02, verbose=False):
    """Frank–Wolfe growth: keep proposing atoms and accept only F-decreasing ones. Self-quenches.
    Returns (De, a, pis, F_trace, n_atoms, accepted_flags) — driven by F ALONE."""
    De = np.array(De, dtype=float)
    if De.shape[0] == 1:
        De = np.array([[0.0]])                                       # a single atom has zero self-distance
    # equilibrate current anchor
    pis, De, a, ftr, _c = EN.equilibrate(Ds, ws, De, a, abar, eps=eps, tau=tau, n_outer=n_outer)
    F = ftr[-1]
    Fs = [F]; accepts = []
    while len(a) < max_atoms:
        pis2, a2 = propose_atom(Ds, ws, pis, a)
        abar2 = np.concatenate([abar * (1 - 0.08), [0.08]])          # slow ref gains the atom too
        De2 = EN.gw_barycenter_De(Ds, pis2, a2)
        mm = De2.shape[0]
        med = np.median(De2[np.triu_indices(mm, 1)]) if mm > 1 else 1.0
        De2 = De2 / (med if med > 0 else 1.0)
        pis2, De2, a2, ftr2, _c2 = EN.equilibrate(Ds, ws, De2, a2, abar2, eps=eps, tau=tau,
                                                  pis0=pis2, n_outer=n_outer)
        Fnew = ftr2[-1]
        if Fnew < F * (1 - rel_tol):                                 # strict F-decrease beyond a pre-registered
            #                                                          relative floor (accept only atoms that pay
            #                                                          for themselves past solver noise), net of τ
            De, a, pis, abar, F = De2, a2, pis2, abar2, Fnew
            Fs.append(F); accepts.append(True)
            if verbose:
                print(f"    grow -> {len(a)} atoms, F={F:.4f} (accepted)")
        else:
            accepts.append(False)
            if verbose:
                print(f"    propose {len(a2)} atoms: F {Fnew:.4f} ≥ {F:.4f} -> REJECT, self-quench")
            break
    # park: atoms whose F-optimal mass is negligible are inactive
    active = int((a > a.max() / (10 * len(a))).sum())
    return {"De": De, "a": a, "pis": pis, "F_trace": Fs, "n_atoms": len(a),
            "active": active, "accepts": accepts}
