"""
SATISFACTION BATTERY — run against the REAL committed coherentflow build (internal check, NOT a gate).
Per satisfaction/BRIEF.md: exercise the ACTUAL functions (coherentflow.structured / settle / combined_read),
not a reimplementation. 7 checks + the two trust-critical measurements (false-positive rate, detection sweep).

Note on the REAL build vs the sandbox: the sandbox `settle` folds circulated structure back INTO the state
(state <- mean+circ), so its consensus leaks the branch and its combined-read payoff is small. The real build
DELIBERATELY holds structure OUT of the consensus (target = mean(f_i - held_i)), so its consensus is
structure-free and the payoff vs that consensus is large. That large payoff could be an artifact of an
unfairly-weak baseline — so we ALSO measure payoff vs a NAIVE mean-of-interfaces consensus (the fair
skeptic's baseline). Reporting both is the honest disambiguation the brief demands.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import coherentflow as cf                      # the REAL committed build
from coherentflow import D, T, NTR, structured, settle, make_interface, r2

rng_global = np.random.default_rng(0)


def acc(p, t):
    return float(np.mean((p > 0.5) == (t > 0.5)))


def probe_acc(X, target):
    """Same train/test linear readout combined_read uses, exposed for baseline comparisons."""
    Xtr = np.column_stack([X[:NTR], np.ones(NTR)])
    W, *_ = np.linalg.lstsq(Xtr, target[:NTR].astype(float), rcond=None)
    pred = np.column_stack([X[NTR:], np.ones(T - NTR)]) @ W
    return acc(pred, target[NTR:])


def held_vec(memory):
    return np.sum([memory[i] for i in memory], axis=0) if memory else np.zeros((T, D))


def branch_ifaces(seed, strength, s0, s1, s2):
    z = np.random.default_rng(seed).normal(size=(T, D))
    br = np.random.default_rng(seed + 1).integers(0, 2, T)
    ext = np.zeros((T, D)); ext[:, 0] = (br * 2 - 1) * strength
    ifaces = [make_interface(z, s0, extra=ext), make_interface(z, s1), make_interface(z, s2)]
    return z, br, ifaces


# ==================================================================== 7 checks
R = {}

# T1 COVERAGE-UNION > best-single (the virtualworld coverage principle; standalone fusion test)
z = np.random.default_rng(1).normal(size=(T, D))
def partial(seed, dims):
    r = np.random.default_rng(seed); m = np.zeros(D); m[list(dims)] = 1
    Rm = np.linalg.qr(r.normal(size=(D, D)))[0]
    v = (z * m) @ Rm + 0.1 * r.normal(size=(T, D))
    A, *_ = np.linalg.lstsq(v[:NTR], z[:NTR], rcond=None); return v @ A
mods = [partial(1, range(0, 8)), partial(2, range(6, 14)), partial(3, range(12, 20)), partial(4, range(18, 24))]
singles = [r2(m[NTR:], z[NTR:]) for m in mods]; union = r2(np.mean(mods, axis=0)[NTR:], z[NTR:])
R['T1 coverage-union > best-single'] = (union > max(singles) + 0.02, f"union={union:.3f} best-single={max(singles):.3f}")

# T2 COHERENT -> HONEST NO-OP (real settle)
z = np.random.default_rng(2).normal(size=(T, D)); coh = [make_interface(z, 10), make_interface(z, 11), make_interface(z, 12)]
st, mem, res, circn = settle(coh, z)
R['T2 coherent -> honest no-op'] = (len(mem) == 0 and circn < 0.5, f"held={len(mem)} circ_norm={circn:.4f}")

# T3 STRUCTURED -> combined read beats consensus (real settle + real combined_read)
z, br, sm = branch_ifaces(3, 3.0, 20, 21, 22)
st, mem, res, circn = settle(sm, z)
cons_acc, comb_acc, hd = cf.combined_read(st, mem, br)
naive_acc = probe_acc(np.mean(sm, axis=0), br)      # fair skeptic baseline
R['T3 structured read > consensus'] = (len(mem) >= 1 and comb_acc > cons_acc + 0.15,
                                       f"held={len(mem)} consensus={cons_acc:.3f} combined={comb_acc:.3f} (naive-mean={naive_acc:.3f})")

# T4 NOISE -> REJECTED, no G1 (real settle)
z = np.random.default_rng(4).normal(size=(T, D))
noisy = make_interface(z, 30) + 2.0 * np.random.default_rng(99).normal(size=(T, D))
nm = [noisy, make_interface(z, 31), make_interface(z, 32)]
st, mem, res, circn = settle(nm, z)
R['T4 noise -> rejected (no G1)'] = (len(mem) == 0 and circn < 0.5, f"held={len(mem)} circ_norm={circn:.4f}")

# T5 SETTLING STABLE via tail-slope, init AWAY from the fixed point (real dynamics, real settle init=)
z, br, sm = branch_ifaces(5, 3.0, 40, 41, 42)
init = np.mean(sm, axis=0) + 0.5 * np.random.default_rng(5).normal(size=(T, D))   # start off the fixed point
st, mem, res, circn = settle(sm, z, init=init)
tail = res[len(res) * 2 // 3:]
contracts = (tail[-1] < tail[0]) and (tail[-1] < res[0])
R['T5 settling contracts (tail-slope)'] = (contracts, f"res {res[0]:.3f}->{res[-1]:.4f}, tail {tail[0]:.4f}->{tail[-1]:.4f}")

# T6 CIRCULATION CONCENTRATED (real held memory)
z, br, sm = branch_ifaces(6, 3.0, 50, 51, 52)
st, mem, res, circn = settle(sm, z)
circ = held_vec(mem); U, S, Vt = np.linalg.svd(circ - circ.mean(0), full_matrices=False)
conc = float((S[0] ** 2) / ((S ** 2).sum() + 1e-9))
R['T6 circulation concentrated'] = (len(mem) >= 1 and conc > 0.4, f"top-dir energy frac={conc:.3f} (held={len(mem)})")

# T7 PURE-NOISE FALSIFICATION -> must NOT hold (real settle)
z = np.random.default_rng(7).normal(size=(T, D))
fake = make_interface(z, 60) + 1.5 * np.random.default_rng(7).normal(size=(T, D))
fm = [fake, make_interface(z, 61), make_interface(z, 62)]
st, mem, res, circn = settle(fm, z)
R['T7 pure-noise NOT held'] = (len(mem) == 0, f"held={len(mem)} (must be 0)")

print("=" * 74); print("INTERNAL SATISFACTION BATTERY — against the REAL coherentflow build"); print("=" * 74)
allp = True
for k, (ok, detail) in R.items():
    allp = allp and ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {k}\n          {detail}")
print(f"  ALL PASS: {allp}")

# ============================================ trust-critical #1: FALSE-POSITIVE rate on coherent input
print("\n" + "=" * 74); print("TRUST #1 — FALSE-POSITIVE rate on COHERENT input (must be ~0%: no-fabrication)")
fp = 0; N_SEED = 40
for seed in range(N_SEED):
    z = np.random.default_rng(6000 + seed).normal(size=(T, D))
    ifaces = [make_interface(z, 7000 + seed), make_interface(z, 8000 + seed), make_interface(z, 9000 + seed)]
    _, mem, _, _ = settle(ifaces, z)
    if len(mem) >= 1:
        fp += 1
print(f"  false-positive hold rate = {100 * fp / N_SEED:.1f}%  over {N_SEED} seeds  ({'CLEAN' if fp == 0 else 'LEAKY'})")

# ============================================ trust-critical #2: DETECTION-SENSITIVITY sweep + read-payoff
print("\n" + "=" * 74); print("TRUST #2 — DETECTION-SENSITIVITY sweep + read-payoff vs injection strength (40 seeds)")
print(f"  {'strength':>8} | {'detect%':>7} | {'mean_held':>9} | {'payoff vs removed-consensus':>27} | {'payoff vs naive-mean':>20}")
sweep = []
for strength in [0.5, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 3.0, 4.0, 6.0]:
    det = 0; pay_rm = []; pay_nv = []; helds = []
    for seed in range(40):
        z, br, ifaces = branch_ifaces(10000 + seed, strength, 30000 + seed, 40000 + seed, 50000 + seed)
        st, mem, res, circn = settle(ifaces, z)
        helds.append(len(mem))
        if len(mem) >= 1:
            det += 1
            cons_acc, comb_acc, _ = cf.combined_read(st, mem, br)
            naive_acc = probe_acc(np.mean(ifaces, axis=0), br)
            pay_rm.append(comb_acc - cons_acc)
            pay_nv.append(comb_acc - naive_acc)
    prm = float(np.mean(pay_rm)) if pay_rm else 0.0
    pnv = float(np.mean(pay_nv)) if pay_nv else 0.0
    sweep.append((strength, 100 * det / 40, float(np.mean(helds)), prm, pnv))
    print(f"  {strength:>8.1f} | {100*det/40:>6.0f}% | {np.mean(helds):>9.2f} | {prm:>+27.3f} | {pnv:>+20.3f}")

print("\n(read-payoff vs REMOVED-consensus is the object's internal contrast; vs NAIVE-mean is the fair "
      "skeptic baseline — the honest measure of whether the object beats simply averaging the interfaces.)")
