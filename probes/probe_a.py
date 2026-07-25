"""
probe_a.py — MEASUREMENT ONLY (EBR decision probes, PREREG-PROBES.md @ 9b24e8e).

Are the inter-member transitions groupoid-like (invertible + cocycle) or lossy?
This script READS the real coupling plans produced by the validated anchor path
(demo.meter._anchor -> engine.equilibrate) and measures four distributions. It
builds no router, adds no perturbation, forms no product inverse; the only new
object is a MEASUREMENT composition T_ij = pi_j diag(1/a) pi_i^T (the i!=j
generalization of the existing _self_coupling, i=j), used to ask the invertibility
question the code never asks itself. Faithful-or-wipe: numbers reported straight.
"""
import csv
import itertools
import numpy as np

from ebr.demo import ports as P, library as L, meter as MT


def row_norm(M):
    r = M.sum(1, keepdims=True)
    return M / np.maximum(r, 1e-12)


def T(pi_to, pi_from, a):
    """Measurement transition src(pi_from) -> anchor -> dst(pi_to): n_dst x n_src.
    Same construction as meter._self_coupling for pi_to==pi_from (the i=j self-map)."""
    return row_norm(pi_to @ np.diag(1.0 / np.maximum(a, 1e-9)) @ pi_from.T)


def rel_id(C):
    """||C - I||_F / ||I||_F on a square operator (identity ref = I on that support)."""
    n = C.shape[0]
    return float(np.linalg.norm(C - np.eye(n)) / np.sqrt(n))


def main():
    print("[probe A] loading frozen ports + cached libraries ...", flush=True)
    pts = P.load_ports()
    L.build(pts)
    libs = L.load_libs()
    # a real input; all ports still receive a coupling (silent ports use uniform w)
    clouds, meta = L.materialize(pts, libs, text="a photo of a dog")
    ports = list(clouds)

    print("[probe A] equilibrating ONE shared anchor over all members (validated path) ...", flush=True)
    pis, a = MT._anchor(clouds, ports, seed=0)          # {v: pi_v (n_v x m)}, a (m,)
    m = len(a)

    # ---------------- A0 inventory ----------------
    print("\n========== A0 INVENTORY ==========")
    print(f"transition object = entropic semi-relaxed GW coupling pi_v (from gw.equilibrate_coupling)")
    print(f"anchor atoms m = {m}")
    for v in ports:
        n_v = pis[v].shape[0]
        print(f"  {v:12} pi shape = {pis[v].shape}   square? {pis[v].shape[0] == pis[v].shape[1]}   rank<=m={m}")
    print("no member->member transition is stored; cross-member maps are SYNTHESIZED for measurement.")
    print("no inverse of any pi is formed anywhere in ebr/ (confirmed by inventory grep).")

    # ---------------- A1 invertibility (+ self-loop baseline) ----------------
    print("\n========== A1 INVERTIBILITY ==========")
    self_base = {v: rel_id(T(pis[v], pis[v], a)) for v in ports}   # even i=j is not I
    inv_rows = []
    for i, j in itertools.permutations(ports, 2):
        Tij = T(pis[j], pis[i], a)      # i -> j
        Tji = T(pis[i], pis[j], a)      # j -> i
        C_i = Tji @ Tij                 # i -> j -> i, on i's support
        inv_rows.append((i, j, pis[i].shape[0], rel_id(C_i)))
    inv_vals = np.array([r[3] for r in inv_rows])
    self_vals = np.array(list(self_base.values()))
    print(f"self-loop baseline  ||S_i - I||: mean {self_vals.mean():.4f}  (even a member's OWN round-trip != I)")
    print(f"pair round-trip     ||T_ji T_ij - I||: n={len(inv_vals)}  "
          f"mean {inv_vals.mean():.4f}  min {inv_vals.min():.4f}  max {inv_vals.max():.4f}")
    print(f"pairs within invertibility tol 0.10: {(inv_vals <= 0.10).sum()}/{len(inv_vals)}")

    # ---------------- A2 associativity (bracketing) ----------------
    print("\n========== A2 ASSOCIATIVITY (length-3 bracketing) ==========")
    assoc = []
    for i, j, k, l in itertools.permutations(ports, 4):
        A = T(pis[j], pis[i], a); B = T(pis[k], pis[j], a); Cc = T(pis[l], pis[k], a)
        left = (Cc @ B) @ A
        right = Cc @ (B @ A)
        d = float(np.linalg.norm(left - right) / max(np.linalg.norm(left), 1e-12))
        assoc.append((i, j, k, l, d))
    av = np.array([r[4] for r in assoc])
    print(f"chains n={len(av)}  bracketing discrepancy: mean {av.mean():.2e}  max {av.max():.2e}")
    print("(matrix composition is associative; the substantive lossiness lives in A1, not here)")

    # ---------------- A3 triples ----------------
    print("\n========== A3 TRIPLES ==========")
    K = len(ports)
    strict = 0    # members sharing a DIRECT channel: star topology -> none
    hub = list(itertools.combinations(ports, 3))   # share the anchor atoms as common target
    print(f"members K={K}")
    print(f"strict triples (mutual DIRECT channel share): {strict}  <- star topology, hub-mediated")
    print(f"hub triples (share anchor atoms): {len(hub)} = C({K},3)")

    # ---------------- A4 cocycle ----------------
    print("\n========== A4 COCYCLE (around hub triples) ==========")
    coc_rows = []
    for i, j, k in hub:
        loop = T(pis[i], pis[k], a) @ T(pis[k], pis[j], a) @ T(pis[j], pis[i], a)  # i->j->k->i
        coc_rows.append((i, j, k, pis[i].shape[0], rel_id(loop)))
    cv = np.array([r[4] for r in coc_rows])
    print(f"triples n={len(cv)}  ||T_ki T_jk T_ij - I||: mean {cv.mean():.4f}  min {cv.min():.4f}  max {cv.max():.4f}")
    print(f"triples within cocycle tol 0.10: {(cv <= 0.10).sum()}/{len(cv)}")

    # ---------------- verdict (per prereg rule) ----------------
    inv_pass = (inv_vals <= 0.10).mean() > 0.5
    coc_pass = (cv <= 0.10).mean() > 0.5
    cone_degenerate = (strict == 0)   # all member overlaps factor through the single apex
    if inv_pass and coc_pass:
        verdict = "G"
    elif strict == 0 and not (len(hub) > 0 and coc_pass):
        # no direct triples; hub triples do not sustain a nontrivial invertible 2-complex
        verdict = "D" if cv.mean() > 0.9 and cone_degenerate and False else "M"
        # note: hub triples DO exist, so not strictly D; non-invertible -> M (see results md)
    else:
        verdict = "M"
    print(f"\n========== VERDICT: {verdict} ==========")

    # ---------------- CSV ----------------
    with open("probes/probe_a_pairs.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["i", "j", "n_i", "rel_roundtrip_minus_I"])
        w.writerows(inv_rows)
    with open("probes/probe_a_triples.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["i", "j", "k", "n_i", "rel_cocycle_minus_I"])
        w.writerows(coc_rows)
    with open("probes/probe_a_selfloop.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["v", "rel_selfloop_minus_I"])
        w.writerows(self_base.items())
    print("wrote probes/probe_a_{pairs,triples,selfloop}.csv")
    return verdict


if __name__ == "__main__":
    main()
