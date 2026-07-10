"""
CONFORMANCE SUITE — one executable test per CONSTRUCT.md theory-invariant, asserting the CODE matches the
THEORY, failing loudly on drift. Bind to thoughtworld_construct/CONSTRUCT.md.

Purpose (methodological): two regressions (dimensionality glued; recurrence was averaging not feedback) were
caught only by after-the-fact probes. This suite makes each theory-invariant a test that CANNOT silently break,
and surfaces the FULL drift surface at once — not just the invariants we happened to probe.

Run: `python conformance/run_conformance.py`. Each invariant reports PASS / PARTIAL / FAIL against the current
committed code, with what the code does vs. what the theory requires. This is a HONEST MAP — it does not fix
anything. FAILs are the deliberate drift-points to resolve.
"""
import os
import sys
import inspect
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "virtualworld"))
sys.path.insert(0, os.path.join(ROOT, "coherentflow"))

RESULTS = []


def record(n, name, status, code_does, theory_wants, evidence):
    RESULTS.append({"invariant": n, "name": name, "status": status,
                    "code_does": code_does, "theory_requires": theory_wants, "evidence": evidence})


def read(path):
    return open(os.path.join(ROOT, path)).read()


# ============================================================ INV 1: resizable / self-expanding medium (#2)
def inv1_resizable_medium():
    import world as W
    d = W.collect(n_rollouts=1, T=6)
    s = d["s_cur"][0]
    registry_driven = (W.SCENE_D == len(W.SCENE_REGISTRY) == W.scene_features(s).shape[0])
    tag_derived = (W.SCENE_POS == [i for i, f in enumerate(W.SCENE_REGISTRY) if "pos" in f.tags])
    # growth: append a feature -> medium expands -> restore
    d0 = W.SCENE_D
    W.append_feature(W.Feature("__conf_tmp__", {"coll"}, lambda ctx: 0.0))
    grew = (W.SCENE_D == d0 + 1) and (W.scene_features(s).shape[0] == d0 + 1)
    W.SCENE_REGISTRY.pop(); W._refresh_scene_index()
    restored = (W.SCENE_D == d0)
    # no hard-coded medium dim: D is derived from the registry, not a literal
    bsrc = read("virtualworld/build_virtualworld.py")
    derived = "D = scene.shape[1]" in bsrc
    ok = registry_driven and tag_derived and grew and restored and derived
    record(1, "resizable / self-expanding medium (#2)", "PASS" if ok else "FAIL",
           "medium = SCENE_REGISTRY; SCENE_D = len(registry); index groups tag-derived; append_feature grows D; "
           "build derives D = scene.shape[1]",
           "D a registry knob, medium can grow, nothing hard-codes a dimension",
           f"registry_driven={registry_driven} tag_derived={tag_derived} grew={grew} restored={restored} "
           f"D-derived-in-build={derived}")


# ============================================================ INV 2: genuine feedback recurrence, not averaging
def inv2_feedback_not_averaging():
    import coherentflow as cf
    import fluid_settle as fl
    D = cf.D
    z = np.random.default_rng(0).normal(size=(cf.T, D))
    # (a) the WIRED recurrence (coherentflow.settle) — is it averaging? incompatible frames must NOT contract if
    #     it were feedback; averaging contracts to the mean and cannot be unstable.
    f1 = z.copy(); f2 = -z.copy()
    st, _, _, _ = cf.settle([f1, f2], z, guard=False)
    shipped_contracts_to_mean = float(np.abs(st - np.mean([f1, f2], axis=0)).max()) < 1e-6  # == averaging
    # (b) the CORRECT fluid (fluid_settle) — coupled operator CAN exceed radius 1 and excludes a destabilizer
    a1 = fl.acceptance_1_mutual_instability()
    a2 = fl.acceptance_2_exclusion()
    fluid_ok = a1["passed"] and a2["passed"]
    # invariant status: the WIRED recurrence is averaging => FAIL (documented regression); correct fluid exists.
    status = "FAIL" if shipped_contracts_to_mean else "PASS"
    record(2, "genuine feedback recurrence, not averaging (the fluid)", status,
           "the WIRED settle (coherentflow.settle) is feed-forward averaging (incompatible frames z,-z contract "
           f"to their mean, max-diff<1e-6={shipped_contracts_to_mean}); Jacobian bounded <=1, cannot be unstable "
           "or exclude. The CORRECT fluid EXISTS separately (fluid_settle) and PASSES: coupled rho can exceed 1 "
           f"({a1['rho_conflict_fluid']:.3f}) and a destabilizer is excluded (rogue weight {a2['rogue_weight']:.4f}).",
           "the recurrence USED by the pipeline must be model->model feedback (coupled rho>1 achievable; "
           "destabilizer excluded, not averaged in)",
           f"shipped_is_averaging={shipped_contracts_to_mean}; fluid_settle_passes={fluid_ok}. "
           "RESOLUTION: wire fluid_settle into the pipeline (deliberate later step).")


# ============================================================ INV 3: faithfulness is a loss TERM, not a phase (#5)
def inv3_loss_term_not_phase():
    import coherentflow as cf
    src = inspect.getsource(cf.settle)
    # the guard (structured) must be called INSIDE the settling loop, not as a separate verify pass
    guard_inside = ("for i, f in enumerate(ifaces)" in src and "structured(" in src
                    and "for _ in range(ITERS)" in src)
    # no separate two-phase generate/verify gadget
    no_verify_phase = ("def verify" not in read("coherentflow/coherentflow.py")
                       and "generate_then_verify" not in read("coherentflow/coherentflow.py"))
    # fluid_settle: is the contraction/stabilization inside the settle loop, or a separate pre-phase?
    fsrc = read("coherentflow/fluid_settle.py")
    fluid_desc_is_prephase = ("instability_descent" in fsrc and "def fluid_settle" in fsrc
                              and "instability_descent" not in inspect.getsource(__import__("fluid_settle").fluid_settle))
    shipped_ok = guard_inside and no_verify_phase
    status = "PARTIAL" if (shipped_ok and fluid_desc_is_prephase) else ("PASS" if shipped_ok else "FAIL")
    record(3, "faithfulness/contraction is a loss TERM, not a phase (#5)", status,
           "coherentflow.settle: the guard `structured()` is called INSIDE the ITERS loop (one loop, no separate "
           "verify pass). BUT fluid_settle's stabilization (instability_descent) is currently a SEPARATE pre-phase "
           "computing w before fluid_settle runs — a contraction PHASE, not a term inside the settle loop.",
           "the contraction/anti-hallucination guard is a term inside the single objective/loop, never a separate "
           "generate/verify phase",
           f"shipped_guard_in_loop={guard_inside} no_verify_phase={no_verify_phase} "
           f"fluid_stabilization_is_prephase={fluid_desc_is_prephase}")


# ============================================================ INV 4: MZ memory = the tape, one object (#1)
def inv4_memory_is_tape():
    import coherentflow as cf
    ssrc = inspect.getsource(cf.settle)
    # in the shipped settle, `memory` is a per-step dict recomputed each iteration (transient held-structure),
    # NOT a persistent resizable tape that IS the MZ kernel.
    memory_is_transient_dict = ("memory = {}" in ssrc and "memory[i] = d @ P" in ssrc)
    # the tape/MZ kernel lives in a SEPARATE module (mz_fluid.py), not unified with settle's memory
    mz_separate = os.path.exists(os.path.join(ROOT, "virtualworld/mz_fluid.py"))
    mz_src = read("virtualworld/mz_fluid.py") if mz_separate else ""
    tape_in_mzfluid = ("Hankel" in mz_src or "tape" in mz_src.lower())
    one_object = not (memory_is_transient_dict and mz_separate)   # they are two different structures => not one
    status = "FAIL" if (memory_is_transient_dict and mz_separate) else "PASS"
    record(4, "MZ memory = the tape, ONE object (#1)", status,
           "coherentflow.settle's `memory` is a per-step dict of held projections (transient, recomputed each "
           "iteration) — a held-structure store, NOT a persistent resizable tape. The MZ kernel / tape "
           "(block-Hankel, self-expansion) lives in a SEPARATE module virtualworld/mz_fluid.py. So the tape and "
           "the memory are TWO different structures, not one object.",
           "the NTM-like resizable tape and the MZ memory kernel are the SAME structure; the tape's read/write "
           "dynamics ARE the kernel (streaming/memory split intrinsic)",
           f"memory_transient_dict={memory_is_transient_dict} tape_separate_module={mz_separate and tape_in_mzfluid}")


# ============================================================ INV 5: loss = models' own grounding (#4)
def inv5_loss_from_grounding():
    # the objective/guard must read model-intrinsic signals (held-out predictivity from the shared medium,
    # coupling stability), never external human labels / an invented judge / arbitrary data.
    csrc = read("coherentflow/coherentflow.py")
    fsrc = read("coherentflow/fluid_settle.py")
    bsrc = read("virtualworld/build_virtualworld.py")
    # structured() guard: ho = held-out R^2 predicting the disagreement from z (the shared medium = grounding)
    intrinsic_guard = "ho = r2(z[NTR:] @ A, d[NTR:])" in csrc or "r2(z" in csrc
    # fluid objective: instability (spectral radius / measured growth) — intrinsic to the coupling
    intrinsic_fluid = "_growth(" in fsrc and "spec_radius" in fsrc
    # virtualworld ridge target = scene medium Y = the physics engine's OWN state (seed grounding), not labels
    grounding_target = "fit_ridge(raw[m][train], Y[train])" in bsrc  # Y = standardized scene_features (engine truth)
    # no external supervision keywords in the loss paths
    banned = ("human_label" in csrc or "external_judge" in csrc or "ground_truth_label" in csrc)
    ok = intrinsic_guard and intrinsic_fluid and grounding_target and not banned
    record(5, "loss = models' own grounding, not external judge (#4)", "PASS" if ok else "FAIL",
           "the guard reads held-out predictivity from the shared medium z (the models' grounding); the fluid's "
           "objective is coupling instability (spectral radius / measured growth) — both model-intrinsic. The "
           "ridge aligns to the physics engine's own scene state (the seed's grounding). No external labels/judge.",
           "each model's loss signal comes from its own training data / grounding; the descent uses those, never "
           "arbitrary data or an invented external judge",
           f"intrinsic_guard={intrinsic_guard} intrinsic_fluid_objective={intrinsic_fluid} "
           f"ridge_to_engine_grounding={grounding_target} external_supervision={banned}")


# ============================================================ INV 6: intrinsic output (read from the terrain)
def inv6_intrinsic_output():
    import coherentflow as cf
    import fluid_settle as fl
    # shipped combined_read: fits a linear probe (lstsq) to a target => an EXTERNAL fitted readout head
    cr = inspect.getsource(cf.combined_read)
    shipped_uses_fitted_probe = "lstsq" in cr or "probe" in cr
    # fluid_settle output: equilibrium SHIFT (settled state - base), no fitted head
    fs = inspect.getsource(fl.fluid_settle)
    fluid_intrinsic = ("s = s + step * field" in fs and "lstsq" not in fs and "return s" in fs)
    status = "FAIL" if shipped_uses_fitted_probe else "PASS"
    record(6, "intrinsic output (read from settled dynamics, not a fitted head)", status,
           "the SHIPPED read (coherentflow.combined_read) fits a linear probe via lstsq to predict a target — an "
           "external fitted readout head, NOT intrinsic. The CORRECT intrinsic output EXISTS in fluid_settle: the "
           "answer is the equilibrium SHIFT of the settled state (query = perturbation), no readout head.",
           "the output emerges from the settled dynamics/terrain (equilibrium shift), not a separately-trained probe",
           f"shipped_combined_read_uses_lstsq_probe={shipped_uses_fitted_probe}; "
           f"fluid_settle_intrinsic={fluid_intrinsic}. RESOLUTION: use the fluid's equilibrium-shift output.")


# ============================================================ INV 7: frozen interfaces, everything else flexible
def inv7_frozen_interfaces_flexible():
    import world as W
    bsrc = read("virtualworld/build_virtualworld.py")
    # model-side frozen: encoders under @torch.no_grad and .eval(), features cached (never trained)
    model_frozen = ("@torch.no_grad()" in bsrc and ".eval()" in bsrc)
    # medium-side (ridge) is the only fitted part
    only_ridge_fitted = "def fit_ridge" in bsrc
    # flexibility: D registry-driven (inv1), n_models = a list, T = module constants (changeable, no rewiring)
    n_models_flex = isinstance(getattr(__import__("build_virtualworld"), "MODS", None), list)
    import build_virtualworld as B
    T_flex = isinstance(B.N_ROLLOUTS, int) and isinstance(B.T, int)
    D_knob = (W.SCENE_D == len(W.SCENE_REGISTRY))
    ok = model_frozen and only_ridge_fitted and n_models_flex and T_flex and D_knob
    record(7, "frozen interfaces, everything else flexible", "PASS" if ok else "FAIL",
           "encoders are frozen (@torch.no_grad + .eval(), cached); only the medium-side ridge is fitted. "
           "D is a registry knob, n_models is a list (MODS), T is module constants — all runtime-variable with no "
           "model rewiring.",
           "model-side of each interface frozen; D, n_models, T, held-rank all runtime knobs",
           f"model_frozen={model_frozen} only_ridge_fitted={only_ridge_fitted} n_models_list={n_models_flex} "
           f"T_constants={T_flex} D_registry_knob={D_knob}")


def main():
    for fn in (inv1_resizable_medium, inv2_feedback_not_averaging, inv3_loss_term_not_phase,
               inv4_memory_is_tape, inv5_loss_from_grounding, inv6_intrinsic_output,
               inv7_frozen_interfaces_flexible):
        try:
            fn()
        except Exception as e:
            record(int(fn.__name__[3]), fn.__name__, "ERROR", f"exception: {e!r}", "", "")

    print("=" * 88)
    print("CONFORMANCE SUITE — code vs. CONSTRUCT.md theory-invariants")
    print("=" * 88)
    order = {"PASS": 0, "PARTIAL": 1, "FAIL": 2, "ERROR": 3}
    tally = {}
    for r in sorted(RESULTS, key=lambda r: r["invariant"]):
        tally[r["status"]] = tally.get(r["status"], 0) + 1
        mark = {"PASS": "✅", "PARTIAL": "🟡", "FAIL": "❌", "ERROR": "⚠️"}[r["status"]]
        print(f"{mark} [{r['invariant']}] {r['status']:7} {r['name']}")
        print(f"       code: {r['code_does']}")
        print(f"       theory: {r['theory_requires']}")
        print(f"       evidence: {r['evidence']}\n")
    print("-" * 88)
    print("TALLY: " + "  ".join(f"{k}={v}" for k, v in sorted(tally.items(), key=lambda kv: order[kv[0]])))
    json.dump({"results": RESULTS, "tally": tally}, open(os.path.join(os.path.dirname(__file__),
              "conformance_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
