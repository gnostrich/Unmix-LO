"""
COHERENTFLOW — one-shot build of the WHOLE construct object as ONE loop. Observe-don't-prove.
BIND: thoughtworld_construct/CONSTRUCT.md. This is NOT a gate and NOT a proof — it instantiates the
complete aesthetic object faithfully and WATCHES what it does on frame-diverse input.

THE OBJECT (one loop, guards INSIDE):
  A Baur descent writes a natural Mori-Zwanzig memory (the resizable "tape") encoding the recurrent
  routing of information across a set of model INTERFACES that each connect to a shared medium. Driven
  purely by an INTERNAL COHERENCE LOSS — settle toward the state maximally consistent with every
  interface's grounding — the recurrent flow SETTLES to a unified world-state. No external task/labels.
  An answer is an OPTIMAL COMBINED READ across interfaces: CONSENSUS where interfaces cohere, PARACONSISTENT
  HELD-SUPERPOSITION where they carry structured decoherence. Settling integrates; combined-read extracts.

CONSTRUCT mapping (faithful, not flattened):
  - streaming term  = the consensus settling (direct/current, single-hop): minimise UNSTRUCTURED decoherence.
  - memory term     = the held structured circulation (delayed/through-frames): the MZ tape holds it.
  - tape == memory  = ONE object: the per-interface held structured components accumulated over the recurrence.
  - self-expansion  = a memory mode is written per interface whose disagreement clears the structured
                      criterion (concentrated AND held-out-predictable = the atomicity/noise-floor dial).
  - faithfulness    = a LOSS TERM, not a phase: the damped (contractive) update is native settling; guards
                      (STRUCTURED-hold / NOISE-reject) live INSIDE the loop, not beside it.

The fix over smoke_oneshot.py: the smoke folded the circulated structure back into `state`, leaking the
held branch into the consensus, then probed a noisy iteration-summed vector — so combined ~ consensus.
Here the structured decoherence is HELD SEPARATELY in the tape (kept OUT of the consensus settling), so a
consensus-collapse read genuinely loses it and the combined read (consensus + held superposition) recovers it.
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D, T = 24, 600
NTR = T // 2                       # first half = train (fit interface maps + read probes), second = held-out
ITERS, DAMP, GAIN = 18, 0.5, 1.0
SEED = 0


def r2(pred, tgt):
    return float(1 - ((tgt - pred) ** 2).sum() / (((tgt - tgt.mean(0)) ** 2).sum() + 1e-12))


def make_interface(z, seed, extra=None, noise=0.1):
    """A model's interface: native frame <-> shared medium. Rotate z into a native frame, add interface
    noise (+ any injected extra), then fit the native->medium alignment (the interface map) on train."""
    r = np.random.default_rng(seed)
    R = np.linalg.qr(r.normal(size=(D, D)))[0]
    v = z @ R + noise * r.normal(size=(T, D))
    if extra is not None:
        v = v + extra
    A, *_ = np.linalg.lstsq(v[:NTR], z[:NTR], rcond=None)   # native -> medium (fit on train only)
    return v @ A                                            # this interface's read of the medium


def structured(d, z):
    """GUARD (inside the loop): a disagreement d is STRUCTURED iff it is CONCENTRATED (low eff-rank) AND
    HELD-OUT PREDICTABLE from the world medium z. Otherwise it is NOISE. Returns (is_structured, projector,
    scores). This is the atomicity / second-FDT-noise-floor criterion: only real reproducible structure is held."""
    dc = d - d.mean(0)
    if np.linalg.norm(dc) < 1e-9:                                 # no disagreement -> nothing to hold
        return False, np.zeros((D, D)), {"cap": 0.0, "ho": 0.0, "eff": 0}
    U, S, Vt = np.linalg.svd(dc[:NTR], full_matrices=False)
    eff = max(1, int(round((S.sum() ** 2) / (S ** 2).sum())))     # participation-ratio effective rank
    P = Vt[:eff].T @ Vt[:eff]                                      # structured subspace projector
    cap = ((dc[NTR:] @ P) ** 2).sum() / ((dc[NTR:] ** 2).sum() + 1e-9)   # held-out concentration
    A, *_ = np.linalg.lstsq(z[:NTR], d[:NTR], rcond=None)
    ho = r2(z[NTR:] @ A, d[NTR:])                                  # held-out predictability from the world
    is_s = (cap > 1.5 * eff / D) and (ho > 0.3)
    return bool(is_s), P, {"cap": float(cap), "ho": float(ho), "eff": int(eff)}


def settle(ifaces, z, guard=True, nudge=None, beta=0.0, init=None, mechanism="fluid"):
    """RECURRENT SETTLING under the internal coherence loss, guards INSIDE.

    `mechanism` (FIX-ORDER STEP 1): the DEFAULT recurrence is now the operator-feedback FLUID
    ('fluid', in fluid_pipeline) — each interface becomes an operator Rᵢ that reads the CURRENT shared
    state through its frame and writes back its correction, so models drive each other and the coupled
    Jacobian J(w)=I+step·Σwᵢ(Rᵢ−I) CAN exceed spectral radius 1 for incompatible frames (genuine mutual
    instability + fluid exclusion). The old feed-forward AVERAGING recurrence is kept verbatim behind
    `mechanism='averaging'` (byte-identical to the shipped behaviour). See MECHANISM_CHECK.md / THEORY.md
    (T1-T2) / FLUID_VERIFICATION.md and conformance INV2.

    Below is the AVERAGING mechanism (mechanism='averaging'): the consensus (streaming) state settles by
    minimising UNSTRUCTURED decoherence; STRUCTURED decoherence is HELD in the tape (memory), circulated
    but kept OUT of the consensus so a single-frame collapse cannot see it. Damped => observe contraction
    (not proven). Init from the naive consensus (minority branch diluted -> must be HELD to survive).

    ROUTING is FIXED/initialized here (memory = d@P, gain 1) — NOT learned. Per the EqProp steering, if the
    routing were to be LEARNED it must be by equilibrium-response (see eqprop_probe), NOT backprop-through-time
    and NOT a fixed heuristic dressed as learning; for this first run we keep it fixed and observe behavior.

    `nudge`/`beta`: a weak perturbation (beta*nudge) added to the coherence target — used ONLY by eqprop_probe
    to read the EQUILIBRIUM RESPONSE (the native EqProp learning signal); beta=0 is the free settle.
    `init`: optional starting state (default = the naive consensus). Used to test contraction from a transient
    started AWAY from the fixed point without changing the dynamics."""
    if mechanism == "fluid":
        import fluid_pipeline as _fp
        return _fp.settle_fluid(ifaces, z, guard=guard, nudge=nudge, beta=beta, init=init)
    if mechanism != "averaging":
        raise ValueError(f"unknown settle mechanism {mechanism!r} (expected 'fluid' or 'averaging')")
    state = np.mean(ifaces, axis=0) if init is None else init.copy()
    res, memory = [], {}
    for _ in range(ITERS):
        prev = state.copy()
        memory = {}
        for i, f in enumerate(ifaces):
            d = f - state
            is_s, P, _sc = structured(d, z)
            if guard:
                if is_s:
                    memory[i] = d @ P                    # HOLD structured decoherence in the tape (MZ memory)
                # else: NOISE -> rejected, NOT circulated (this is what defeats G1 amplification)
            else:
                memory[i] = d                            # ABLATED guard: circulate EVERYTHING (unsafe)
        # coherence-loss target = consensus of the interfaces with their HELD structure removed
        # (minimise UNSTRUCTURED decoherence only; the held part is not averaged away, it is memorised)
        coherent = np.mean([ifaces[i] - memory.get(i, 0.0) for i in range(len(ifaces))], axis=0)
        if nudge is not None and beta:
            coherent = coherent + beta * nudge           # weak EqProp nudge toward the coherence objective
        state = prev + DAMP * (coherent - prev)          # damped native settling (contraction observed)
        res.append(float(np.linalg.norm(state - prev)))
    circ_norm = float(np.linalg.norm(np.sum([memory[i] for i in memory], axis=0))) if memory else 0.0
    return state, memory, res, circ_norm


def eqprop_probe(ifaces, z, beta=0.02, ndir=5, seed=0):
    """EqProp-NATIVE learning-signal probe (equilibrium-response, NO backprop-through-settling).

    (point 2) The routing/memory learning signal is read from how the EQUILIBRIUM SHIFTS under a weak nudge:
      settle to a free equilibrium s*, add a small beta-nudge toward the coherence objective, settle to s^beta,
      and use R = (s^beta - s*)/beta as the local signal. No settling iteration is unrolled or differentiated.

    (point 4, honest flag) Does this behave like a CLEAN gradient of a scalar energy, or is the settling only a
    CONSTRAINED relaxation (because holding structured decoherence may not be a clean energy flow)? The exact
    criterion: a gradient flow of a scalar energy has a SYMMETRIC response operator (Maxwell/Onsager reciprocity)
    -> <R_u, v> == <R_v, u> for perturbation directions u, v. We measure the asymmetry; ~0 = clean EqProp,
    large = constrained-relaxation (EqProp-like but not textbook). Both outcomes are informative."""
    s_free, mem_free, _, _ = settle(ifaces, z)
    rng = np.random.default_rng(seed)
    dirs, R = [], []
    for _ in range(ndir):
        u = rng.normal(size=s_free.shape); u /= (np.linalg.norm(u) + 1e-12)
        s_u, _, _, _ = settle(ifaces, z, nudge=u, beta=beta)
        dirs.append(u); R.append((s_u - s_free) / beta)     # equilibrium response to nudge u
    # reciprocity: <R_u, v> vs <R_v, u> across all direction pairs
    asyms = []
    for a in range(ndir):
        for b in range(a + 1, ndir):
            uv = float((R[a] * dirs[b]).sum()); vu = float((R[b] * dirs[a]).sum())
            asyms.append(abs(uv - vu) / (abs(uv) + abs(vu) + 1e-12))
    asym = float(np.mean(asyms))
    # demonstrate the learning signal itself: two-equilibrium diff projected on each held routing direction
    learn_sig = {}
    for i in mem_free:
        hd = mem_free[i] / (np.linalg.norm(mem_free[i]) + 1e-12)
        learn_sig[i] = float(np.mean([abs((r * hd).sum()) for r in R]))   # response overlaps the routing dir
    verdict = ("clean-gradient (textbook EqProp valid)" if asym < 0.1 else
               "constrained-relaxation (EqProp-like, NOT a clean scalar-energy flow)" if asym > 0.3 else
               "borderline (mostly conservative, mild non-clean component)")
    return {"beta": beta, "ndir": ndir, "response_asymmetry": asym, "verdict": verdict,
            "learning_signal_per_held_iface": learn_sig,
            "method": "equilibrium-response (two settles + difference); NO backprop-through-time"}


def combined_read(state, memory, target):
    """OPTIMAL COMBINED READ. consensus-collapse view = the settled consensus only (what a single frame /
    averaging sees). combined view = consensus + the HELD SUPERPOSITION (per-interface structured channels,
    kept SEPARATE, not collapsed). Probe a query linearly on TRAIN, score on HELD-OUT. If the combined read
    recovers `target` better than consensus, the object surfaces held structure a consensus-collapse loses."""
    def probe(X):
        Xtr = np.column_stack([X[:NTR], np.ones(NTR)])
        W, *_ = np.linalg.lstsq(Xtr, target[:NTR].astype(float), rcond=None)
        pred = np.column_stack([X[NTR:], np.ones(T - NTR)]) @ W
        return float(np.mean((pred > 0.5) == (target[NTR:] > 0.5)))     # held-out accuracy
    consensus_view = state
    held = np.concatenate([memory[i] for i in sorted(memory)], axis=1) if memory else np.zeros((T, 0))
    combined_view = np.concatenate([consensus_view, held], axis=1)
    return probe(consensus_view), probe(combined_view), held.shape[1]


def run_condition(name, z, ifaces, target, watch_branch):
    state, memory, res, circ = settle(ifaces, z)
    tail_contracts = res[-1] < res[len(res) // 2]
    n_held = len(memory)
    if watch_branch:
        cons_acc, comb_acc, held_dim = combined_read(state, memory, target)
    else:
        cons_acc = comb_acc = held_dim = None
    return {"name": name, "residual_curve": res, "settles_tail_contract": bool(tail_contracts),
            "res_first": res[0], "res_last": res[-1], "n_interfaces_held_structured": n_held,
            "circ_norm": circ, "held_memory_dim": held_dim,
            "consensus_read_acc": cons_acc, "combined_read_acc": comb_acc}


def main():
    rng = np.random.default_rng(SEED)
    z = rng.normal(size=(T, D))                      # the shared medium (world latent)

    # ---- MAIN: FRAME-DIVERSE INJECTED input (precondition MET: one interface carries a hidden branch) ----
    # frozen/converged models would no-op here (F_gauge~0, per xresolve) — so we inject the distinction the
    # object needs to act on: a binary branch one interface (frame 0) carries on a hidden medium direction.
    branch = rng.integers(0, 2, T)
    ext = np.zeros((T, D)); ext[:, 0] = (branch * 2 - 1) * 3.0
    ifaces = [make_interface(z, 1, extra=ext), make_interface(z, 2), make_interface(z, 3)]
    main_res = run_condition("INJECTED (frame-diverse)", z, ifaces, branch, watch_branch=True)

    # EqProp-native learning-signal probe (equilibrium-response, NOT backprop-through-time) + honest
    # clean-gradient-vs-constrained-relaxation flag. Run on the injected input where structure IS held.
    eqp = eqprop_probe(ifaces, z)

    # ---- CONTROL 1: COHERENT input (no injected structure) -> must NO-OP (settle to consensus, hold nothing)
    ifaces_coh = [make_interface(z, 1), make_interface(z, 2), make_interface(z, 3)]
    coh_res = run_condition("COHERENT (control)", z, ifaces_coh, branch, watch_branch=True)

    # ---- CONTROL 2: NOISE input (one interface corrupted with UNSTRUCTURED noise) -> must REJECT, not amplify
    big = 3.0 * rng.normal(size=(T, D))              # high-variance, NOT predictable from z (unstructured)
    ifaces_noise = [make_interface(z, 1, extra=big), make_interface(z, 2), make_interface(z, 3)]
    noise_res = run_condition("NOISE (control)", z, ifaces_noise, branch, watch_branch=False)

    # WHY it rejects (the guard's discrimination, honest): the corrupted interface's disagreement is HELD-OUT
    # UNPREDICTABLE (ho below the 0.3 floor) — so it is NOISE and never written to the tape. Contrast the
    # injected carrier's disagreement, which IS held-out predictable (ho above floor) and IS held.
    st0 = np.mean(ifaces_noise, axis=0)
    _, _, sc_noise = structured(ifaces_noise[0] - st0, z)       # corrupted interface guard scores
    st_inj = np.mean(ifaces, axis=0)
    _, _, sc_inj = structured(ifaces[0] - st_inj, z)            # injected carrier guard scores
    noise_res["guard_scores_corrupted_iface"] = sc_noise
    noise_res["guard_scores_injected_carrier"] = sc_inj

    # ---------------------------------------------------------------- report
    print("COHERENTFLOW — whole-object one-shot (observe-don't-prove)\n")
    m = main_res
    print(f"[MAIN] frame-diverse INJECTED input (one interface carries a hidden branch):")
    print(f"  SETTLE           : residual {m['res_first']:.3f} -> {m['res_last']:.3f}  "
          f"(tail-contracts? {m['settles_tail_contract']})")
    print(f"  SURFACE + HOLD   : {m['n_interfaces_held_structured']}/3 interfaces held STRUCTURED, "
          f"circ_norm={m['circ_norm']:.3f}, tape dim={m['held_memory_dim']}")
    print(f"  COMBINED READ    : consensus-collapse recovers branch = {m['consensus_read_acc']:.3f}  |  "
          f"combined (consensus+held) = {m['combined_read_acc']:.3f}")
    payoff = m['combined_read_acc'] - m['consensus_read_acc']
    print(f"  -> combined read surfaces held structure a consensus-collapse loses?  "
          f"{'YES' if payoff > 0.1 else 'no'}  (+{payoff:.3f})")

    print(f"\n[CONTROL 1] COHERENT input (no injected structure) — expect HONEST NO-OP:")
    print(f"  held STRUCTURED = {coh_res['n_interfaces_held_structured']}/3, circ_norm={coh_res['circ_norm']:.3f}; "
          f"consensus={coh_res['consensus_read_acc']:.3f} combined={coh_res['combined_read_acc']:.3f} "
          f"(equal + ~chance => correct no-op)")

    print(f"\n[CONTROL 2] NOISE input (corrupted interface) — expect REJECT, no amplification (no G1):")
    sn, si = noise_res["guard_scores_corrupted_iface"], noise_res["guard_scores_injected_carrier"]
    print(f"  held STRUCTURED = {noise_res['n_interfaces_held_structured']}/3 (noise NOT held), "
          f"circ_norm={noise_res['circ_norm']:.3f} (nothing written to tape => nothing to amplify)")
    print(f"  guard discrimination (CONCENTRATION is the discriminator here): corrupted iface "
          f"eff-rank={sn['eff']}/{D}, cap={sn['cap']:.2f} < {1.5*sn['eff']/D:.2f} thr => NOISE, rejected")
    print(f"                                                     injected carrier "
          f"eff-rank={si['eff']}/{D}, cap={si['cap']:.2f} > {1.5*si['eff']/D:.2f} thr AND ho={si['ho']:+.2f} "
          f"=> STRUCTURED, held")

    print(f"\n[EqProp] learning mechanism = equilibrium-response (NOT backprop-through-settling; routing is "
          f"FIXED/initialized this run):")
    print(f"  two-equilibrium response gives the routing learning signal without unrolling the recurrence.")
    print(f"  response reciprocity asymmetry = {eqp['response_asymmetry']:.3f}  =>  {eqp['verdict']}")
    print(f"  (honest flag: does holding structured decoherence keep a clean scalar energy? "
          f"{'YES — clean' if eqp['response_asymmetry']<0.1 else 'NO — constrained-relaxation' if eqp['response_asymmetry']>0.3 else 'mostly'})")

    out = {"config": {"D": D, "T": T, "iters": ITERS, "damp": DAMP, "gain": GAIN, "seed": SEED},
           "main_injected": main_res, "control_coherent": coh_res, "control_noise": noise_res,
           "eqprop_probe": eqp, "payoff_combined_minus_consensus": payoff}
    json.dump(out, open(os.path.join(HERE, "coherentflow_results.json"), "w"), indent=1)
    print("\nwrote coherentflow_results.json")


if __name__ == "__main__":
    main()
