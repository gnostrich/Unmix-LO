"""
run.py — poles-first test of the frozen claim: the MZ closure built ON an I/O stream recovers the hidden
generator's memory (order + poles) fit-free, discriminates memoryless, and refuses to terminate on a
continuous spectrum. The machinery never sees the generators — only streams. Reproduce: python run.py
"""
import json
import numpy as np
import stream_trace as ST

P_IN, Q_OUT = 3, 3
T_DEFAULT = 12000
NOISE = 0.05


# ---------------- hidden generators (the "virtual things"; only their streams are observed) ----------------
def gen_atomic(r, T, seed=0, rho=0.85):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(r, r)); A *= rho / np.max(np.abs(np.linalg.eigvals(A)))
    B = rng.normal(size=(r, P_IN)) / np.sqrt(P_IN)
    C = rng.normal(size=(Q_OUT, r)) / np.sqrt(r)
    u = rng.normal(size=(T, P_IN)); x = np.zeros(r); ys = []
    for t in range(T):
        ys.append(C @ x + NOISE * rng.normal(size=Q_OUT))
        x = A @ x + B @ u[t]
    return u, np.array(ys), np.linalg.eigvals(A)


def gen_memoryless(T, seed=0):
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(Q_OUT, P_IN))
    u = rng.normal(size=(T, P_IN))
    return u, u @ M.T + NOISE * rng.normal(size=(T, Q_OUT)), np.array([])


def gen_continuous(T, seed=0, npoles=200):
    rng = np.random.default_rng(seed)
    lam = np.linspace(-0.9, 0.9, npoles)
    B = rng.normal(size=(npoles, P_IN)) / np.sqrt(P_IN)
    C = rng.normal(size=(Q_OUT, npoles)) / np.sqrt(npoles)
    u = rng.normal(size=(T, P_IN)); x = np.zeros(npoles); ys = []
    for t in range(T):
        ys.append(C @ x + NOISE * rng.normal(size=Q_OUT))
        x = lam * x + B @ u[t]
    return u, np.array(ys), lam


def main():
    res = {"config": {"p": P_IN, "q": Q_OUT, "T": T_DEFAULT, "noise": NOISE}}

    print("=" * 76)
    print("ARM 1 — ATOMIC hidden memory: recover order AND poles from the stream alone")
    print("=" * 76)
    res["atomic"] = {}
    all_ok = True
    for r in [2, 3, 4, 6]:
        rows = []
        for seed in [0, 1, 2]:
            u, y, true_poles = gen_atomic(r, T_DEFAULT, seed=seed)
            rd = ST.read_trace(u, y, seed=seed)
            err = ST.pole_match_error(true_poles, rd["poles"])
            rows.append({"order": rd["order"], "pole_err": err, "gap": rd["gap"]})
        orders = [x["order"] for x in rows]; errs = [x["pole_err"] for x in rows]
        ok = all(abs(o - r) <= 1 for o in orders) and all(e < 0.1 for e in errs)
        all_ok &= ok
        res["atomic"][r] = {"orders": orders, "pole_errs": errs, "ok": bool(ok)}
        print(f"  r={r}: recovered order {orders} (truth {r})  pole-match err {[f'{e:.3f}' for e in errs]}  "
              f"gap~{np.mean([x['gap'] for x in rows]):.1f}  {'RECOVERED' if ok else 'MISS'}")
    res["atomic"]["all_recovered"] = bool(all_ok)

    print()
    print("=" * 76)
    print("ARM 2 — MEMORYLESS generator + PERMUTATION-NULL validity: both must read order 0")
    print("=" * 76)
    orders_m = []
    for seed in [0, 1, 2]:
        u, y, _ = gen_memoryless(T_DEFAULT, seed=seed)
        orders_m.append(ST.read_trace(u, y, seed=seed)["order"])
    # validity: a genuinely atomic stream, output-shuffled, must also read 0
    u, y, _ = gen_atomic(4, T_DEFAULT, seed=0)
    y_shuf = np.roll(y, len(y) // 2, axis=0)
    order_shuf = ST.read_trace(u, y_shuf, seed=0)["order"]
    memless_ok = all(o == 0 for o in orders_m) and order_shuf == 0
    res["memoryless"] = {"orders": orders_m, "shuffled_atomic_order": order_shuf, "ok": bool(memless_ok)}
    print(f"  memoryless orders {orders_m} (need 0); shuffled-atomic order {order_shuf} (need 0)  "
          f"{'CLEAN' if memless_ok else 'FABRICATES'}")

    print()
    print("=" * 76)
    print("ARM 3 — CONTINUOUS spectrum (no atomic support): must NOT terminate; drifts with T")
    print("=" * 76)
    cont = []
    for T in [3000, 12000, 48000]:
        u, y, _ = gen_continuous(T, seed=0)
        rd = ST.read_trace(u, y, seed=0)
        cont.append({"T": T, "order": rd["order"], "gap": rd["gap"]})
        print(f"  T={T:6d}: order {rd['order']:2d}  gap {rd['gap']:.2f}")
    orders_c = [c["order"] for c in cont]
    cont_ok = (orders_c[-1] - orders_c[0] >= 2) and np.mean([c["gap"] for c in cont if c["gap"] > 0]) < 3.0
    res["continuous"] = {"rows": cont, "no_clean_termination": bool(cont_ok)}
    gaps_str = ", ".join(f"{c['gap']:.2f}" for c in cont)
    print(f"  drifts up with T: {orders_c[-1] - orders_c[0] >= 2}; gaps stay small: [{gaps_str}]  "
          f"{'NO TERMINATION (correct)' if cont_ok else 'TERMINATED (dial broken)'}")

    holds = all_ok and memless_ok and cont_ok
    res["verdict"] = {"atomic_recovered": bool(all_ok), "memoryless_reads_zero": bool(memless_ok),
                      "continuous_refuses_to_terminate": bool(cont_ok), "claim_holds": bool(holds)}
    print()
    print(f"VERDICT — 'the stream's closure converges to the MZ memory' : "
          f"{'HOLDS (all three arms)' if holds else 'FAILS — see arms above'}")
    json.dump(res, open("results.json", "w"), indent=1, default=lambda o: str(o))
    print("wrote results.json")


if __name__ == "__main__":
    main()
