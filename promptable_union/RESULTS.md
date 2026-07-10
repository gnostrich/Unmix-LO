# RESULTS — the prompt-able UNION object, on real frozen models (honest, average/control throughout)

Pre-registration frozen in `PREREG.md` **before** any run code (commit `8bc665d`). The object and its test are
the same build: `union_object.py` (the object) + `run_experiment.py` (the pre-committed measurement).
Everything below is real: a real frozen VLM, real frozen classifier specialists, real held-out images with
ground-truth labels. Honest RED = success; the **(B) no-drag** cell is the one to watch.

## The build (real models that actually run, CPU-only)
- **CORE** = `SmolVLM-500M-Instruct`, a real frozen open VLM. Nuanced free-text read-out + a real
  token-probability confidence. (~2 s/query after disabling multi-crop image splitting.)
- **SPECIALISTS** (frozen, coverage disjoint from the core): `dima806/oxford_flowers_image_detection`
  (102 flower species), `nateraw/food` (101 dishes), `dima806/fairface_age_image_detection` (face age,
  distractor). Each returns top-1 label + softmax confidence.
- **ROUTER** = CLIP (`clip-vit-base-patch32`) with one text prototype per specialist domain; per-query cost =
  N dot-products (rank), not dataset size. Routes by image↔domain match.
- **ANTI-DRAG gate** = a specialist enters the context **only** when its COVERAGE clears a threshold (router
  CLIP-match ≥ 0.24 **and** its own classifier confidence ≥ 0.30). Core is the floor. Out-of-coverage +
  low core-confidence → **ABSTAIN**.

## Pre-committed measurement (20 real held-out images per set)
- **U** (breadth): Oxford-Flowers-102 test + Food-101 val — fine labels the core VLM lacks.
- **K** (no-drag): CIFAR-10 common objects — things the core knows, in no specialist's domain.
- **ADV** (out-of-union): DTD textures — no specialist covers; must abstain, never invent.
- Baselines: **CORE-ALONE** (the floor everywhere) and **NAIVE-INJECT-ALL** (dump every specialist every query).

| set | core | **object** | naive-inject-all | object abstain | object calls/q | naive calls/q |
|---|---|---|---|---|---|---|
| U_flowers | 0.05 | **0.65** | 0.55 | 0.00 | 1.00 | 3.00 |
| U_food | 0.15 | **0.40** | 0.85 | 0.30 | 0.50 | 3.00 |
| **U mean** | **0.10** | **0.53** | 0.70 | — | — | — |
| K_objects | 0.45 | **0.45** | **0.00** | 0.35 | 0.00 | 3.00 |
| ADV_texture | 0.00 | 0.00 | 0.00 | **0.75** | 0.20 | 3.00 |
| **overall calls/q** | — | **0.42** | — | — | **0.42** | 3.00 (=N) |

## The four frozen cells

### (A) BREADTH — object >> core-alone on U ✅
Core 0.10 → object **0.53** (flowers 0.05→0.65; food 0.15→0.40). The routed specialist knowledge is real
breadth the core lacks. The core VLM adopts the injected label when the specialist is right (`"Cake."`→
`"Cheesecake."`) and inherits its error when wrong (`"Dessert."`→`"Donuts."` on beignets — ceiling = union,
not fabrication). **PASS.**

### (B) NO-DRAG — object not worse than core-alone on K ✅  (THE critical cell)
Object 0.45 **= core 0.45** on K — the gate routes nothing there (0.00 calls/q), so the core's known answers
are untouched. The control proves the gate earns its keep: **naive-inject-all collapses to 0.00** — it took
**9/20** items the core got right and destroyed them (`"Truck."`→`"Ball Moss."`, `"White cat."`→`"Magnolia."`,
`"Frog."`→`"Snapdragon."`). That −0.45 catastrophe is exactly the ignorance-drag the prereg named as the real
risk; the admission gate converts it into a clean no-op. **PASS — and this is the headline.**

### (C) FLAT COST — sub-linear in N, flatter than naive ✅
Object specialist-calls/query = **0.42** overall (coverage-bounded: it calls only specialists whose domain
matches, ≤1 in practice, **independent of N**). Naive = **3.00 = N** on every query. The router/gate makes cost
flat while naive is linear; crossover is immediate (object ≤ naive for all N ≥ 1, strictly flatter as N grows).
**PASS.**

### (D) UNION CEILING + ABSTENTION ✅ (works, but the valve is noisy)
On out-of-union textures the object **abstains 15/20 (75%)** and **admitted a specialist on 0/20** — it never
invents a label outside the union. Object accuracy never exceeds the per-query best of {core, specialists}
(structurally — it only echoes the core or an admitted specialist). **PASS.** Caveat, stated plainly: the
abstention valve is imprecise — it also false-abstains 35% on K and 30% on food, trading in-union coverage for
out-of-union safety.

## Verdict (frozen PASS/FAIL): **PASS on all four** — but hold the line on what that means
This is **occupied territory** (RAG-over-experts / tool-VLM with a generative head) and the numbers say
**union, never synergy, never capability-beyond-parts**:
- The object's breadth (U = 0.53) is **bounded by and traceable to the specialists**. It never beats the union
  ceiling. **Naive-inject-all actually gets *more* raw breadth** (U = 0.70) — because it always injects — so the
  object is not "better at knowing things." Its win is a **discipline/tradeoff**, not a free lunch.
- What the object buys over naive injection: it keeps most of the breadth (flowers 0.65 even beats naive 0.55
  by injecting *only* the matching specialist instead of cross-specialist noise) **while preserving K**
  (0.45 vs naive 0.00) at **7× lower cost** (0.42 vs 3.0 calls). That is precisely the pre-registered claim:
  **breadth at flat cost without ignorance-drag.**
- It is a genuine trade, not a dominance: the object gives up breadth on food (0.40 vs naive 0.85) to buy
  drag-avoidance and flat cost. Whether that trade is worth it is a deployment question (how expensive is
  corrupting a known answer vs missing a specialist one).

## Honest REDs / where it's crude (reported, not hidden)
1. **The pre-registered core-confidence gate is broken on a real small VLM.** SmolVLM reports ~0.85 confidence
   whether right or wrong (0.78 on flowers it gets *wrong*, 0.85 on objects it gets right), so the prereg's
   "core knows it → don't route" fast-path never fires usefully. We had to route on the prereg's *other* stated
   admission signal — specialist **coverage** — which is model-agnostic. The architecture's headline knob
   (core self-knowledge) did not survive contact with a real frozen VLM.
2. **The coverage gate is a blunt instrument.** At threshold 0.24 the flower domain separates cleanly (0.271)
   but food barely clears (0.243) → 30% of food queries fall through to abstention → lost breadth (object 0.40
   loses to naive 0.85 on food). A learned/calibrated router would tighten this.
3. **The abstention valve is noisy** (35%/30% false-abstain on K/food). It works on genuinely OOD textures
   (75%, zero fabrication) but costs in-union coverage. Reliable abstention needs a calibrated core-uncertainty
   signal, which the small VLM does not provide (see RED #1).
4. K core-accuracy is only 0.45 because CIFAR upscaling degrades the VLM's read-out; the no-drag *equality*
   (object = core) and the naive *collapse* (0.00) are unaffected by that, but the absolute K floor is weak.

## Bottom line
The **discipline holds on real models**: union-routed breadth (core 0.10 → 0.53) at **flat cost** (0.42 vs 3.0
calls) with **no ignorance-drag** on known queries (0.45 = core, vs naive's 0.00 collapse) and **abstention with
zero fabrication** out-of-union. All four pre-committed cells pass. But it is **union, not synergy** — the object
never exceeds its parts, naive injection gets more raw breadth, and the object's real contribution is the
**anti-drag + flat-cost discipline**, whose value the naive control makes undeniable (−0.45 → 0.00 on K). The
pre-registered core-confidence trigger failed on a real VLM and the coverage gate/abstention are crude — the
concept survives, the specific knobs need calibration.

Reproduce: `cd promptable_union && python run_experiment.py --n 20` (writes `results.json`).
