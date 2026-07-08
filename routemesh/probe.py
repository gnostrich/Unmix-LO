import numpy as np
rng = np.random.default_rng(0)
# Probe R1's mechanism: if models have DISJOINT competence regions (each owns a slice of query-space),
# does a free-cardinality oracle router beat best-single-overall AND a difficulty-only baseline?
# AND does the atomic/composite structure fall out (some queries need 1 model, some need assembling 2)?
# This is NOT synergy: each sub-answer comes from a single model; we just route/assemble correctly.

Nq = 4000
M = 5                      # models
# each query has a "topic" (which model-competence region) and a difficulty
topic = rng.integers(0, M, size=Nq)          # which single model is the expert (atomic queries)
difficulty = rng.random(Nq)                  # shared difficulty (the confound to beat)
# 30% of queries are DECOMPOSABLE: need TWO topics' experts, answer = both sub-parts correct
is_comp = rng.random(Nq) < 0.30
topic2 = rng.integers(0, M, size=Nq)
# model m is correct on a query's sub-part iff it's the expert for that part AND difficulty not too high
def model_correct(m):
    # atomic: correct if m==topic and passes difficulty
    ok_atomic = (topic==m) & (rng.random(Nq) > difficulty*0.9)
    return ok_atomic
# ground-truth per-model correctness on the ATOMIC part
C = np.array([ (topic==m) & (rng.random(Nq) > difficulty*0.9) for m in range(M) ]).T  # Nq x M
# for composite queries, the query is "solved" iff BOTH its experts (topic and topic2) are individually correct on their parts
comp_expert2_ok = np.array([ (topic2==m) & (rng.random(Nq) > difficulty*0.9) for m in range(M) ]).T
# ---- outcomes under each policy ----
# best-single-overall: pick the one model with highest overall accuracy, use it on everything
overall_acc = C.mean(0)
best_m = overall_acc.argmax()
best_single = np.mean(np.where(is_comp,
                               C[np.arange(Nq),best_m] & comp_expert2_ok[np.arange(Nq),best_m], # one model can't do both parts
                               C[np.arange(Nq),best_m]))
# difficulty-only router: route by difficulty (easy->answer, hard->abstain), ignoring WHICH model
#   proxy: always use a generic "average" model gated by difficulty; can't exploit model-specific competence
diff_only = np.mean((difficulty < 0.5) & ~is_comp) * 0.0 + np.mean( (rng.random(Nq)>difficulty*0.9) & ~is_comp )  # rough difficulty-gated
# ORACLE ROUTER (free cardinality): atomic -> route to the expert; composite -> assemble both experts; else abstain
atomic_solved = C[np.arange(Nq), topic]                                   # route atomic to its expert
comp_solved   = C[np.arange(Nq), topic] & comp_expert2_ok[np.arange(Nq), topic2]  # assemble two experts
oracle = np.mean(np.where(is_comp, comp_solved, atomic_solved))
# cardinality breakdown of oracle routes
frac_single = np.mean(~is_comp)
frac_multi  = np.mean(is_comp)
print("R1 mechanism probe (disjoint competence, free-cardinality oracle vs best-single vs difficulty-only):")
print(f"  best-single-overall     = {best_single:.3f}")
print(f"  difficulty-only router  = {diff_only:.3f}")
print(f"  ORACLE (free cardinality)= {oracle:.3f}")
print(f"  oracle - best_single    = {oracle-best_single:+.3f}   (>=0.10 => routable model-specific competence)")
print(f"  oracle - difficulty     = {oracle-diff_only:+.3f}   (>=0.10 => NOT just the difficulty marginal)")
print(f"  emergent cardinality: single-path {frac_single:.2f}, multi-path(assemble) {frac_multi:.2f}")
print()
print("  reading: if oracle beats BOTH by >=0.10, per-query model-specific competence is real and")
print("  free-cardinality routing (single for atomic, assemble for composite) captures it -- WITHOUT synergy")
print("  (every sub-answer is one model's; we only route/assemble). This is the untested opportunity R1 tests.")
