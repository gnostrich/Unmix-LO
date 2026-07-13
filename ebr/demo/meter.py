"""
demo/meter.py — the calibrated disagreement meter on REAL equilibria (EBR requirement R5.3).

CLASSIFICATION: INSTRUMENT. This module MEASURES and REPORTS; it must NEVER drive mechanism. It reads a
finished equilibrium (couplings + anchor masses produced by demo.engine.equilibrate) and reports a scalar.
It writes nothing back into the F-loop, changes no anchor, and is never consulted by engine.py. Single
authority holds: the meter observes; the engine decides.

WHAT IT MEASURES (the cycle-cost holonomy of §5.3, validated synthetically in experiments/g4_meter.py):
lay two OVERLAPPING hyperedges over the materialized ports and equilibrate an anchor for each —
  edge A = every port that materialized (all modalities),
  edge B = the vision-port subset of A (overlaps A on the vision ports; if the input has only one modality
           we fall back to A minus its first port so B is still a proper, overlapping subset).
For every port v shared by A and B, form the port's self-reconstruction operator (v -> anchor -> v)
  S_v = row_normalize( pi_v · diag(1/a) · pi_v^T )       (n×n, on v's own support)
and take the Frobenius gap between how the two anchors reconstruct v:
  cycle_cost = mean_v || S_A_v - S_B_v ||_F .
If the two edges genuinely see the same relational geometry the two anchors reconstruct the shared ports the
same way and the gap is small; if the extra ports in edge A pull the anchor elsewhere (e.g. a conflicting
text port dragging the anchor off the vision geometry) the shared ports' self-maps diverge and the gap grows.
This is exactly the S_v construction and averaging used by g4_meter.cycle_cost — reused, not reinvented.

Single-cost collapse: each port carries C_v channels; to keep one cost per port (as g4_meter has one cloud
per member) we collapse every port to its FIRST channel before equilibrating. Documented choice, not hidden.

THE FLOOR (must be a MEASURED null, never a magic constant):
we equilibrate both anchors from n_restarts random anchor initialisations (varying De0 only, exactly as
g4_meter.solver_floor_cycle varies init_seed) and record cycle_cost at each. Then
  reported cycle_cost = median over restarts   (robust signal level)
  phi = std over restarts                       (solver-restart null: how much cycle_cost wobbles on pure
                                                 re-initialisation, with the clouds held fixed)
  floor = K_SIGMA * phi ,   K_SIGMA = 3         (PRE-REGISTERED 3-sigma choice, stated here, not tuned to
                                                 make a verdict pass)
  above_floor = cycle_cost > floor .
The verdict is a signal-to-noise test: is the measured disagreement more than 3 restart-sigmas, i.e. larger
than solver noise alone would manufacture?

HONEST CAVEATS (do not remove):
  * There is no CLONE reference available on a live prompt (g4_meter could build identical members and set
    floor = clone_level + 3σ; we cannot). So this floor is the PURE-SOLVER-NOISE null only. It is a LOWER
    bound on the true agreement floor: two genuinely-agreeing but structurally-different edges (A has strictly
    more ports than B) will read some cycle_cost above pure noise even with no real disagreement, because the
    edges are not identical. Read "above_floor" as "beyond solver noise", NOT as a calibrated clone test.
  * The meter is noisy at the tiny (n<=128, m=6, few-restart) settings used for a CPU-fast live demo; phi is
    estimated from only n_restarts samples. Treat the verdict as indicative, and prefer the g4_meter offline
    validation for the calibrated clone-vs-disjoint separation claim.
"""
import numpy as np

from . import engine as E

# --- pre-registered instrument constants (declared, not tuned) ---
K_SIGMA = 3          # floor = K_SIGMA * phi ; a 3-sigma solver-noise band, chosen a priori
N_ATOMS = 6          # anchor size for the meter's throwaway anchors (small: this is an instrument, CPU-fast)
N_CAP = 128          # subsample each port's support to <=N_CAP points to keep the live meter CPU-fast
N_OUTER = 10         # F-loop outer iterations for the meter's anchors (<=12, fast)
J_SINK = 3           # coupling sub-sweeps per outer step


def _self_coupling(pi, a):
    """Port self-reconstruction operator v->anchor->v on v's own support: row_normalize(pi diag(1/a) pi^T).
    Identical construction to g4_meter._self_coupling (reused so the live and offline meters agree)."""
    Sm = pi @ np.diag(1.0 / np.maximum(a, 1e-9)) @ pi.T
    r = Sm.sum(1, keepdims=True)
    return Sm / np.maximum(r, 1e-12)


def _subsample(Dw, n_cap=N_CAP):
    """Cap a port's (D, w) to its first n_cap points and renormalize (median for D, sum for w), preserving the
    cloud_to_Dw conventions. Keeps the live meter fast without changing what is being measured qualitatively."""
    D, w = Dw
    n = D.shape[0]
    if n <= n_cap:
        return D, w
    idx = np.arange(n_cap)
    Ds = D[np.ix_(idx, idx)].copy()
    med = np.median(Ds[np.triu_indices(n_cap, 1)])
    if med > 0:
        Ds = Ds / med
    ws = w[idx].astype(float)
    ws = ws / max(ws.sum(), 1e-12)
    return Ds, ws


def _anchor(clouds, ports, seed, n_outer=N_OUTER):
    """Equilibrate ONE throwaway anchor for the edge over `ports`, first channel per port, from random De0
    (seed sets the anchor init — the knob the restart-null varies). Returns (pis: port->pi, a)."""
    sub = {v: [_subsample(clouds[v][0])] for v in ports}   # single-cost collapse: first channel only
    m = N_ATOMS
    rng = np.random.default_rng(seed)
    De0 = rng.random((m, m)); De0 = (De0 + De0.T) / 2; np.fill_diagonal(De0, 0.0)
    De0 = De0 / np.median(De0[np.triu_indices(m, 1)])
    a0 = np.full(m, 1.0 / m)
    Bbar = {v: np.array([1.0]) for v in ports}             # one channel per port -> trivial routing simplex
    res = E.equilibrate(sub, De0, a0, a0.copy(), Bbar, n_outer=n_outer, j_sink=J_SINK)
    pis = {v: res["pis"][v][0] for v in ports}
    return pis, res["a"]


def _edges(clouds, meta):
    """edge A = all materialized ports; edge B = the vision-port subset (overlaps A). If that would not be a
    proper overlapping subset (e.g. single-modality input), fall back to B = A minus its first port."""
    A = list(clouds)
    B = [v for v in A if meta[v]["modality"] == "vision"]
    if len(B) < 1 or len(B) == len(A):
        B = A[1:]
    return A, B


def _cycle_cost(clouds, A, B, seed):
    """Equilibrate both anchors from the SAME anchor init `seed` (isolating disagreement due to the differing
    port sets, not to differing inits — as g4_meter does), then mean_v ||S_A_v - S_B_v|| over shared ports."""
    piA, aA = _anchor(clouds, A, seed)
    piB, aB = _anchor(clouds, B, seed)
    shared = [v for v in A if v in B]
    costs = [np.linalg.norm(_self_coupling(piA[v], aA) - _self_coupling(piB[v], aB)) for v in shared]
    return float(np.mean(costs))


def disagreement(clouds, meta, n_restarts=6):
    """INSTRUMENT (R5.3). Read the equilibrium and report the calibrated disagreement meter.

    clouds, meta: as returned by demo.library.materialize.
    Returns {"cycle_cost": float, "floor": float, "above_floor": bool,
             "verdict": "agree" | "disagree-above-floor"}.
    cycle_cost = median over n_restarts anchor-init restarts; floor = K_SIGMA * (std over the same restarts)
    = a measured solver-restart null (see module docstring for the honest floor caveat)."""
    A, B = _edges(clouds, meta)
    vals = [_cycle_cost(clouds, A, B, seed=100 + r) for r in range(n_restarts)]
    cycle_cost = float(np.median(vals))
    phi = float(np.std(vals))
    floor = K_SIGMA * phi
    above = bool(cycle_cost > floor)
    return {"cycle_cost": cycle_cost, "floor": floor, "above_floor": above,
            "verdict": "disagree-above-floor" if above else "agree",
            "phi_restart_std": phi, "edge_A": A, "edge_B": B}


if __name__ == "__main__":
    # Self-test: load real ports + cached libraries, materialize a CLEAN input and a MIXED/conflicting input,
    # and print the meter for each. Run as a module:  python -m ebr.demo.meter
    import os
    from . import ports as P, library as L

    print("[meter self-test] loading frozen ports + cached probe libraries ...", flush=True)
    ports = P.load_ports()
    L.build(ports)                 # no-op if cache exists (it does); builds once (~5 min) otherwise
    libs = L.load_libs()

    dog_png = "/tmp/claude-0/-home-user-Unmix-LO/fbf73b3f-12c7-590b-bba2-68d95a1598aa/scratchpad/dog.png"
    image = None
    if os.path.exists(dog_png):
        from PIL import Image
        image = Image.open(dog_png)

    def report(label, clouds, meta):
        d = disagreement(clouds, meta)
        print(f"\n  [{label}]  edges A={d['edge_A']}  B={d['edge_B']}")
        print(f"    cycle_cost (median) = {d['cycle_cost']:.4f}")
        print(f"    phi (restart std)   = {d['phi_restart_std']:.4f}   floor = {K_SIGMA}*phi = {d['floor']:.4f}")
        print(f"    above_floor={d['above_floor']}   VERDICT: {d['verdict']}")
        return d

    # CLEAN: a dog image with an AGREEING caption (or --text fallback if no image is available).
    if image is not None:
        clean_clouds, clean_meta = L.materialize(ports, libs, image=image, text="a photo of a dog")
        clean_label = "CLEAN  image=dog + text='a photo of a dog'"
    else:
        clean_clouds, clean_meta = L.materialize(ports, libs, text="a dog")
        clean_label = "CLEAN  text='a dog' (no dog.png found)"
    d_clean = report(clean_label, clean_clouds, clean_meta)

    # MIXED / ambiguous: same dog image but a CONFLICTING caption — the text ports (in edge A, absent from the
    # vision-only edge B) pull edge A's anchor off the vision geometry, which should raise the cycle cost.
    if image is not None:
        mix_clouds, mix_meta = L.materialize(ports, libs, image=image, text="a photo of an airplane")
        d_mix = report("MIXED  image=dog + CONFLICTING text='a photo of an airplane'", mix_clouds, mix_meta)
        print(f"\n  clean cycle_cost = {d_clean['cycle_cost']:.4f}   "
              f"mixed cycle_cost = {d_mix['cycle_cost']:.4f}   "
              f"({'mixed reads higher (expected)' if d_mix['cycle_cost'] > d_clean['cycle_cost'] else 'mixed did NOT read higher — meter is noisy at demo settings; see docstring caveat'})")
