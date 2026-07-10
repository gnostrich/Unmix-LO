"""
EXPLORE — play with the field object across regimes and watch what it DOES, with the plain AVERAGE as the null
throughout. Not pass/fail; exploratory. Run: `python explore.py`. Findings written up in EXPLORE.md.
"""
import json
import numpy as np
import field as F

D = 8
STEP = 0.3


def qset(n, seed):
    return np.random.default_rng(seed).normal(size=(n, D))


def regime(name, Rs, queries):
    w = np.ones(len(Rs)) / len(Rs)
    rho = F.spectral_radius(F.coupled_jacobian(Rs, w, STEP))
    ter = F.terrain(Rs, w, queries, STEP)
    verds = {}
    for q in queries[:12]:
        tr = F.settle(Rs, q[None, :], w, iters=250, step=STEP)
        v = F.tail_read(tr[:, 0, :])["verdict"]; verds[v] = verds.get(v, 0) + 1
    print(f"\n### {name}")
    print(f"  coupled rho = {rho:.3f}  ({'CONTESTED >1' if rho > 1 else 'convergent <=1'})")
    print(f"  terrain: mean={ter.mean():.3f} std={ter.std():.3f} contested-frac(>1)={(ter > 1).mean():.3f}")
    print(f"  tail-read verdicts: {verds}")
    return rho, ter


def main():
    print("=" * 70); print(f"REGIME SWEEP + FAITHFULNESS GUARD (D={D})"); print("=" * 70)
    Q = qset(40, 0)
    regime("CONVERGENT frames (agree)",
           [F.frame_operator(D, 1, gain=0.9), F.frame_operator(D, 1, gain=0.85), F.frame_operator(D, 1, gain=0.8)], Q)
    conf = [F.frame_operator(D, 10, gain=1.3, rot=(0, 1, 1.2)),
            F.frame_operator(D, 11, gain=1.3, rot=(2, 3, 1.2)),
            F.frame_operator(D, 12, gain=1.3, rot=(4, 5, 1.2))]
    rho_c, _ = regime("CONFLICTING frames (diverge)", conf, Q)
    print(f"\n  >>> FAITHFULNESS: conflicting reaches rho>1 ? {rho_c > 1}  (else averaging in disguise)")
    regime("CONTROL: identical frames x3", [F.frame_operator(D, 7, gain=1.0)] * 3, Q)

    # held-superposition, FAIR test: a hidden ± distinction carried by ONE FRAME, neutral query
    print("\n" + "=" * 70)
    print("HELD-SUPERPOSITION (fair: distinction in the FRAME, neutral query) vs AVERAGE null")
    print("=" * 70)
    branch = np.random.default_rng(3).integers(0, 2, 80)
    RB = F.frame_operator(D, 21, gain=1.2)
    field_proj, avg_proj = [], []
    for i, b in enumerate(branch):
        RA = F.frame_operator(D, 20, gain=1.2); RA[0, 0] = 1.2 if b == 1 else -1.2
        q = 0.3 * np.random.default_rng(100 + i).normal(size=D)
        field_proj.append(F.settle([RA, RB], q[None, :], iters=250, step=STEP)[-1, 0][0])
        avg_proj.append((0.5 * (q @ RA.T + q @ RB.T))[0])

    def sep(p):
        p = np.array(p); return max((p > np.median(p)).astype(int).__eq__(branch).mean(),
                                    (p < np.median(p)).astype(int).__eq__(branch).mean())
    fa, aa = sep(field_proj), sep(avg_proj)
    print(f"  field settled-state recovers branch: acc={fa:.3f}  | AVERAGE null: acc={aa:.3f}  "
          f"| field beats average? {fa > aa + 0.05}")

    # REAL modalities
    print("\n" + "=" * 70); print("REAL modalities (vision/text/audio/timeseries) — the honest test"); print("=" * 70)
    ix = json.loads(open("data/real_modalities.js").read()[len("window.VW_IX = "):-1])
    Y = np.array(ix["Y"]); MODS = ix["meta"]["modalities"]; al = {m: np.array(ix["aligned"][m]) for m in MODS}
    n, Dr = Y.shape; ntr = int(n * 0.6)

    def realop(f, lam=1.0):
        A = Y[:ntr]; return np.linalg.solve(A.T @ A + lam * np.eye(Dr), A.T @ f[:ntr]).T
    Rr = [realop(al[m]) for m in MODS]; wr = np.ones(4) / 4
    rho = F.spectral_radius(F.coupled_jacobian(Rr, wr, STEP))
    Qr = np.random.default_rng(0).normal(size=(300, Dr))
    ter = F.terrain(Rr, wr, Qr, STEP); teri = F.terrain([realop(al["vision"])] * 4, wr, Qr, STEP)
    print(f"  coupled rho REAL = {rho:.3f}  ({'CONTESTED' if rho > 1 else 'convergent -> no-op'})")
    print(f"  terrain std REAL={ter.std():.4f}  vs IDENTICAL-control std={teri.std():.4f}  "
          f"contested-frac={(ter > 1).mean():.3f}")
    print(f"  >>> does REAL beat the identical control (a real conflict signal)? {ter.std() > teri.std()}")


if __name__ == "__main__":
    main()
