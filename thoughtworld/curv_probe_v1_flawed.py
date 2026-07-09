import numpy as np
rng = np.random.default_rng(0)
# INSTRUMENT CHECK: does the curvature-spectrum measurement DISTINGUISH atomic structure from noise?
# Build a toy where we KNOW the answer: fragment deviations that are (a) low-rank/atomic vs (b) full-rank noise.
# If the spectrum + randomized-fragment control separates them, the instrument works -> safe to hand off.

D = 24          # engine state dim
Nstates = 2000  # sampled engine states
# engine = a flat reference: linear dynamics sT+1 = E sT (self-consistent, deterministic)
E = np.linalg.qr(rng.normal(size=(D,D)))[0]      # orthogonal = stable, coherent reference

states = rng.normal(size=(Nstates, D))
engine_next = states @ E.T

def fragment_deviation(kind, rank=3, noise=0.3):
    # fragment predicts next state = engine_next + deviation. Deviation is A(state) applied.
    if kind=="atomic":
        # deviation lives in a low-rank subspace, COHERENT (same few directions everywhere)
        U = np.linalg.qr(rng.normal(size=(D,rank)))[0]     # rank-r subspace
        coeffs = states @ rng.normal(size=(D,rank)) * 0.5  # deviation coefficients depend on state
        dev = coeffs @ U.T
    elif kind=="noise":
        # deviation is full-rank, isotropic, structureless
        dev = rng.normal(0, noise, size=(Nstates,D))
    elif kind=="mixed":
        U = np.linalg.qr(rng.normal(size=(D,rank)))[0]
        coeffs = states @ rng.normal(size=(D,rank)) * 0.5
        dev = coeffs @ U.T + rng.normal(0, noise, size=(Nstates,D))
    return dev

def curvature_spectrum(dev, states):
    # Curvature ~ how the deviation field FAILS TO COMMUTE around loops.
    # Proxy: the deviation field A(s) as a matrix (how dev depends on state); its structure = the connection.
    # Fit A: dev ~ states @ A^T  (linear connection), residual = nonlinear part.
    A,*_ = np.linalg.lstsq(states, dev, rcond=None)   # D x D connection matrix
    # curvature proxy = antisymmetric part (the noncommutative/directed core) + the connection's own rank
    F = 0.5*(A - A.T)                                  # antisymmetric = the directed curvature
    sv = np.linalg.svd(A, compute_uv=False)
    sv_F = np.linalg.svd(F, compute_uv=False)
    eff_rank = (sv.sum()**2)/(sv**2).sum()             # participation ratio (effective rank)
    eff_rank_F = (sv_F.sum()**2)/(sv_F**2).sum()
    return eff_rank, eff_rank_F, sv

def randomized_null(dev, states):
    # CONTROL: shuffle the state-deviation pairing -> destroys any real state-dependent structure
    perm = rng.permutation(len(states))
    return curvature_spectrum(dev[perm], states)

print("INSTRUMENT CHECK: can curvature spectrum distinguish atomic structure from noise?\n")
print(f"{'case':<10} {'effrank(A)':>11} {'effrank(F)':>11} {'null effrank(A)':>16}  verdict")
for kind in ["atomic","noise","mixed"]:
    dev = fragment_deviation(kind)
    er, erF, sv = curvature_spectrum(dev, states)
    er_null, _, _ = randomized_null(dev, states)
    # atomic => low eff-rank AND real (survives vs null); noise => high eff-rank ~ null
    verdict = "ATOMIC (signal)" if er < D*0.4 and er_null > er*1.5 else "NOISE (structureless)"
    print(f"{kind:<10} {er:>11.2f} {erF:>11.2f} {er_null:>16.2f}  {verdict}")
print(f"\n  (D={D}; atomic should show LOW eff-rank that COLLAPSES relative to shuffled null;")
print("   noise should show HIGH eff-rank ~ its own null. If these separate, the instrument works.)")
