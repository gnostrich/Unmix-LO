"""
stage0.py — the stage-0 harness (§12), gates in order (§11). Runs Phase Zero (G0), then the double
dissociation (G1) and running-invariance (G2) on known-degree substrate traffic (so the McMillan-degree
headline is checkable — the spec's positive-control logic). Writes verdicts to the registry.

This is the honest, cheaply-killable core: it uses the validated instrument (gauge-invariant geometry +
monotone GW pooling + residual block-Hankel) and reports what actually happens. G3/G4/G5 scaffolding noted
in REPORT.md. Run: python -m ebr.experiments.stage0
"""
import numpy as np
from .. import phase_zero as PZ
from ..registry import registry as REG
from . import g1_probe as G1
from ..geometry.clouds import cloud_to_Dw, scramble


def g1_dissociation(seeds=(0, 1), T=140):
    """Diversity leg: rank should grow with r (K fixed). K-invariance leg: rank flat with K (r fixed)."""
    div = {r: np.mean([G1.run(3, r, T=T, m0=1, seed=s)[0] for s in seeds]) for r in (2, 3, 4)}
    kinv = {K: np.mean([G1.run(K, 3, T=T, m0=1, seed=s)[0] for s in seeds]) for K in (2, 3, 5)}
    grows = div[4] > div[2]                      # diversity leg
    flat = (max(kinv.values()) - min(kinv.values())) <= 1.0
    return {"diversity_rank": div, "K_rank": kinv, "diversity_grows": bool(grows), "K_flat": bool(flat)}


def g2_running_invariance(n=120):
    """Mid-run scramble of one member must move nothing: the residual moments are identical."""
    U = np.random.default_rng(0).normal(size=(1, 6))
    M = G1.S.make_models(1, seed0=100)[0]
    inp = G1._warp_inputs(U[0], n)
    from ..hankel import residual as H
    D, _ = cloud_to_Dw(M(inp)); Ds, _ = cloud_to_Dw(scramble(M(inp), seed=99))
    m1 = H.residual_moments(H.gram_from_D(D), np.zeros((n, 1)))
    m2 = H.residual_moments(H.gram_from_D(Ds), np.zeros((n, 1)))
    rel = float((np.abs(m1 - m2) / (np.abs(m1) + 1e-9)).max())
    return {"rel_move": rel, "pass": rel < 1e-6}


def main():
    REG.preflight({})
    print("### STAGE 0 — gates in order ###\n")
    g0 = PZ.run()
    REG.append({"gate": "G0", **g0}, when="stage0")
    if not g0["G0"]:
        print("\nG0 failed — stop (nothing downstream may run).")
        return

    print("\n" + "=" * 66); print("G1 — double dissociation (rank ~ diversity, flat in K)"); print("=" * 66)
    g1 = g1_dissociation()
    print(f"  diversity leg (K=3): {[(r, round(v, 2)) for r, v in g1['diversity_rank'].items()]}"
          f"  grows={g1['diversity_grows']}")
    print(f"  K-invariance (r=3): {[(k, round(v, 2)) for k, v in g1['K_rank'].items()]}"
          f"  flat={g1['K_flat']}")
    print(f"  G1 verdict: {'PASS' if g1['diversity_grows'] and g1['K_flat'] else 'FAIL / see REPORT.md'}")
    REG.append({"gate": "G1", **g1}, when="stage0")

    print("\n" + "=" * 66); print("G2 — running invariance (mid-run scramble moves nothing)"); print("=" * 66)
    g2 = g2_running_invariance()
    print(f"  rel move under scramble = {g2['rel_move']:.2e}  -> {'PASS' if g2['pass'] else 'FAIL'}")
    REG.append({"gate": "G2", **g2}, when="stage0")


if __name__ == "__main__":
    main()
