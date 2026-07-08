import numpy as np
rng = np.random.default_rng(0)
# ---------------------------------------------------------------------------
# G3 sandbox — PATHWAY THICKENING / GAP-FILLING (no prior prototype; highest risk).
#
# Mesh with a SPARSE pathway: channels A->B and B->C are learnable (paired data
# exists); there is NO paired A-C data. The gap is in DATA, not structure: A and
# C genuinely share latent content, part of which is visible to B (carryable) and
# part invisible to B (the bottleneck bound — a transitive edge can never carry it).
#
#  Q1 THICKEN : distill composite A->B->C into a direct one-hop A->C map;
#               direct must match composite on held-out inputs (and is cheaper).
#  Q2 GENERALIZE: on held-out worlds, the synthesized edge must predict C's
#               TRUE state on the through-B-carryable dims, beating no-edge and
#               random-edge; the un-carryable share is REPORTED as the bound.
#  Q3 GAP-FILL: on the dims A and C redundantly share, settle (confidence-weighted
#               fusion) the edge-transported estimate with C's own noisy reading,
#               sweeping C's noise so the source-quality ratio crosses parity.
#               Fusion has content only near parity (a dominant source correctly
#               gets all the weight); pass = real gain at parity + no harm anywhere.
#  Q4 FABRICATION GUARD: synthesize an edge to a node sharing NOTHING with A;
#               it must NOT beat the no-edge baseline. Manufactured structure = FAIL.
#
# PRE-COMMITTED THRESHOLDS (fixed before the corrected run; the real-model gate
# inherits these):
#  Q1 PASS: direct-vs-composite held-out rel-error < 0.05.
#  Q2 PASS: on carryable dims, >=20% relative gain vs no-edge AND vs random-edge.
#  Q3 PASS: at the parity point (single-source errors within 2x), fused beats the
#           best single source by >=10% relative on ground truth; AND fused is never
#           worse than best single by >2% at any point of the sweep.
#  Q4 PASS: fabricated-edge gain vs no-edge <= 5%.
#  G3 sandbox PASS = all four.
#
# HONESTY NOTE (corrections kept on the record):
# v1: gave A and C disjoint latents in an independent world, so the pathway carried
#     zero information by construction and Q2 was unpassable — a broken test (it
#     duplicated Q4), not a negative result. Its Q3 also let C read the query
#     directly, so the edge never participated — a degenerate pass.
# v2: Q3 fixed a single arbitrary C-noise (0.25) against an edge estimate 7x better;
#     inverse-variance fusion against a dominant source correctly adds ~0% — the
#     claim is only contentful near source-parity, so Q3 now sweeps the ratio and
#     pre-commits: >=10% gain at parity AND no harm off-parity. v2 also used
#     oracle (true-error) fusion weights — an integrity bug; weights now come only
#     from noise levels the nodes themselves know.
# ---------------------------------------------------------------------------
D = 40
n_train, n_test = 400, 200

know_A = np.r_[0:16, 32:36]          # A: 0..15 + 32..35              (20 dims)
know_B = np.r_[8:28]                 # B: 8..27                       (20 dims)
know_C = np.r_[12:16, 20:36]         # C: 12..15 + 20..35             (20 dims)
know_D2 = np.r_[36:40]               # D2: shares nothing with A      (guard pair)
carryable = np.r_[12:16]             # A∩C∩B: the only A-C content that survives B
uncarryable = np.r_[32:36]           # A∩C but invisible to B: the bound

def make_node(know, seed):
    R = np.linalg.qr(np.random.default_rng(seed).normal(size=(len(know), len(know))))[0]
    return {"know": know, "R": R}

def obs(node, Z, noise=0.03):
    return (Z[:, node["know"]] + rng.normal(0, noise, (Z.shape[0], len(node["know"])))) @ node["R"].T

A, B, C, D2 = (make_node(k, s) for k, s in [(know_A, 1), (know_B, 2), (know_C, 3), (know_D2, 4)])

def ridge(X, Y, lam=1e-3):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)

Ztr = rng.normal(size=(n_train, D))
Zte = rng.normal(size=(n_test, D))                     # held-out worlds

W_AB = ridge(obs(A, Ztr), obs(B, Ztr))                 # channels from paired data
W_BC = ridge(obs(B, Ztr), obs(C, Ztr))

# --- Q1 THICKEN: distill 2-hop composite into a direct edge (fresh unlabeled A-observations)
Xa = obs(A, rng.normal(size=(n_train, D)))
W_AC = ridge(Xa, (Xa @ W_AB) @ W_BC)
Xa_te = obs(A, Zte)
comp_te = (Xa_te @ W_AB) @ W_BC
direct_te = Xa_te @ W_AC
q1_err = np.linalg.norm(direct_te - comp_te) / np.linalg.norm(comp_te)

# --- Q2 GENERALIZE vs ground truth, scored on carryable dims; bound reported
truth_C_gauge = Zte[:, know_C] @ C["R"].T              # C's true noiseless state
car_idx = np.searchsorted(know_C, carryable)           # positions of carryable dims in C's frame
unc_idx = np.searchsorted(know_C, uncarryable)
def relerr_on(pred, cols):
    t = (truth_C_gauge @ C["R"])[:, cols]              # decode C's gauge, select latent dims
    p = (pred @ C["R"])[:, cols]
    return np.linalg.norm(p - t) / np.linalg.norm(t)
err_direct = relerr_on(direct_te, car_idx)
err_noedge = relerr_on(np.zeros_like(direct_te), car_idx)
err_rand = relerr_on(Xa_te @ rng.normal(0, np.abs(W_AC).mean(), W_AC.shape), car_idx)
q2_gain_noedge = (err_noedge - err_direct) / err_noedge
q2_gain_rand = (err_rand - err_direct) / err_rand
bound_err = relerr_on(direct_te, unc_idx)              # should stay ~1.0: the edge cannot carry it

# --- Q3 GAP-FILL: fuse edge-transported estimate with C's own reading across a
#     noise sweep (settling step). Fusion weights use only what nodes know: the
#     edge's validation error (measurable without C ground truth, from Q1's
#     composite agreement + channel residuals) and C's own noise level.
zc_edge = (direct_te @ C["R"])[:, car_idx]             # estimate of carryable dims via A + edge
truth_car = Zte[:, carryable]
def rel(p): return np.linalg.norm(p - truth_car) / np.linalg.norm(truth_car)
e_edge = rel(zc_edge)
w_edge = 1 / (0.03**2 + q1_err**2)                     # edge confidence from KNOWN noise + distill residual
sweep, parity_gain, harm = [], None, 0.0
for sig in [0.01, 0.02, 0.04, 0.08, 0.16, 0.32]:
    zc_own = (obs(C, Zte, noise=sig) @ C["R"])[:, car_idx]
    e_own = rel(zc_own)
    w_own = 1 / sig**2                                 # C knows its own noise level
    e_fused = rel((w_edge * zc_edge + w_own * zc_own) / (w_edge + w_own))
    gain = (min(e_edge, e_own) - e_fused) / min(e_edge, e_own)
    sweep.append((sig, e_own, e_fused, gain))
    harm = min(harm, gain)
    if 0.5 <= e_own / e_edge <= 2.0 and (parity_gain is None or gain > parity_gain):
        parity_gain = gain                             # best gain in the comparable regime
q3_gain = parity_gain if parity_gain is not None else 0.0

# --- Q4 FABRICATION GUARD: distill A->D2 through B (B shares nothing with D2)
W_AD2 = ridge(Xa, (Xa @ W_AB) @ ridge(obs(B, Ztr), obs(D2, Ztr)))
truth_D2_gauge = Zte[:, know_D2] @ D2["R"].T
err_bad = np.linalg.norm(Xa_te @ W_AD2 - truth_D2_gauge) / np.linalg.norm(truth_D2_gauge)
q4_gain = (1.0 - err_bad) / 1.0                        # vs no-edge (predict 0, rel-error 1)

print("G3 sandbox — pathway thickening / gap-filling (corrected world; see honesty note)")
print(f"  Q1 THICKEN   : direct-vs-composite held-out rel-error = {q1_err:.4f}   (PASS < 0.05; 1 hop vs 2)")
print(f"  Q2 GENERALIZE: carryable dims — err direct={err_direct:.3f} no-edge={err_noedge:.3f} random={err_rand:.3f}")
print(f"                 gain vs no-edge = {100*q2_gain_noedge:.0f}%  vs random = {100*q2_gain_rand:.0f}%   (PASS >= 20% both)")
print(f"                 BOUND: un-carryable A∩C dims stay at rel-error {bound_err:.2f} (~1 = edge honestly silent)")
print(f"  Q3 GAP-FILL  : edge-only err={e_edge:.3f}; sweep of C's own-noise sigma:")
for sig, e_own, e_fused, gain in sweep:
    print(f"                 sigma={sig:.2f}: own={e_own:.3f} fused={e_fused:.3f} gain={100*gain:+.0f}%")
print(f"                 parity gain = {100*q3_gain:.0f}% (PASS >= 10%), worst harm = {100*harm:+.1f}% (PASS > -2%)")
print(f"  Q4 GUARD     : fabricated A->D2 edge gain vs no-edge = {100*q4_gain:.0f}%   (PASS <= 5%)")
p1, p2 = q1_err < 0.05, (q2_gain_noedge >= 0.2 and q2_gain_rand >= 0.2)
p3, p4 = (q3_gain >= 0.10 and harm > -0.02), q4_gain <= 0.05
print(f"\n  Q1={'PASS' if p1 else 'FAIL'} Q2={'PASS' if p2 else 'FAIL'} Q3={'PASS' if p3 else 'FAIL'} Q4={'PASS' if p4 else 'FAIL'}"
      f"  =>  G3 sandbox {'PASS' if all([p1, p2, p3, p4]) else 'FAIL'}")
