"""
The object IS its own test (PREREG.md). Real frozen models, real held-out images. Pre-committed cells:
  (A) BREADTH  : object >> core-alone on U (routed specialist knowledge the core lacks)
  (B) NO-DRAG  : object NOT worse than core-alone on K   <- THE critical cell
  (C) FLAT COST: object specialist-calls/query sub-linear in N; naive-inject-all = N
  (D) CEILING  : object <= per-query best of {core, specialists}; ABSTAINS on out-of-union (no fabrication)
Controls: NAIVE-INJECT-ALL (dump every specialist every query) and CORE-ALONE (the floor everywhere).
Honest RED = success. Run: python run_experiment.py  (writes results.json)
"""
import json, sys, time, argparse
import numpy as np
from datasets import load_dataset
from union_object import CoreVLM, Specialist, CLIPRouter, PromptableUnionObject, label_hit

t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

SPEC_CFG = [
    ("dima806/oxford_flowers_image_detection", "flower species",
     ["a photograph of a flower", "a blossom", "a flowering plant"]),
    ("nateraw/food", "food dish",
     ["a plate of food", "a cooked dish", "a dessert or meal"]),
    ("dima806/fairface_age_image_detection", "human face age",
     ["a human face", "a person's face portrait", "a headshot of a person"]),
]

def take(ds_name, split, img_key, n, seed=0):
    ds = load_dataset(ds_name, split=split, streaming=True)
    names = None
    try: names = ds.features["label"].names
    except Exception: pass
    ds = ds.shuffle(seed=seed, buffer_size=200)
    out = []
    for ex in ds:
        img = ex[img_key].convert("RGB")
        gt = names[ex["label"]] if names is not None else None
        out.append((img, gt))
        if len(out) >= n: break
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)          # images per set
    ap.add_argument("--admit_core", type=float, default=1.01)   # core-conf fast-path OFF by default (uncalibrated)
    ap.add_argument("--route_sim", type=float, default=0.24)    # coverage gate (CLIP domain match)
    ap.add_argument("--admit_spec", type=float, default=0.30)
    ap.add_argument("--abstain_core", type=float, default=0.0)  # abstain iff no coverage AND core_conf < this
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n = 3 if args.smoke else args.n

    log("loading core VLM ..."); core = CoreVLM()
    log("loading specialists ..."); specs = [Specialist(m, d, p) for (m, d, p) in SPEC_CFG]
    log("loading CLIP router ..."); router = CLIPRouter(specs)
    obj = PromptableUnionObject(core, specs, router, admit_core=args.admit_core,
                                route_sim=args.route_sim, admit_spec=args.admit_spec,
                                abstain_core=args.abstain_core)
    log(f"object ready: core={core.name} specialists={[s.name for s in specs]}")

    SETS = {
        "U_flowers": ("dpdl-benchmark/oxford_flowers102", "test", "image"),   # breadth (GT fine flower)
        "U_food":    ("ethz/food101", "validation", "image"),                 # breadth (GT fine dish)
        "K_objects": ("uoft-cs/cifar10", "test", "img"),                      # no-drag (core-known common objects)
        "ADV_texture": ("tanganke/dtd", "test", "image"),                     # out-of-union -> must abstain
    }
    data = {name: take(ds, sp, ik, n) for name, (ds, sp, ik) in SETS.items()}
    for k, v in data.items(): log(f"  set {k}: {len(v)} images (gt sample: {v[0][1]})")

    results = {"config": {"n": n, "admit_core": args.admit_core, "route_sim": args.route_sim,
                          "admit_spec": args.admit_spec, "abstain_core": args.abstain_core, "core": core.name,
                          "specialists": [s.name for s in specs]}, "per_set": {}, "records": []}

    for setname, items in data.items():
        is_adv = setname.startswith("ADV")
        agg = {"core_correct": 0, "obj_correct": 0, "naive_correct": 0, "total": 0,
               "obj_abstain": 0, "obj_calls": 0, "naive_calls": 0, "obj_routed_any": 0, "core_conf_sum": 0.0}
        for img, gt in items:
            core_txt, core_conf = core.read_out(img)
            obj_txt, om = obj.answer(img)
            nv_txt, nm = obj.answer(img, force_inject_all=True)
            cc = (not is_adv) and label_hit(core_txt, gt)
            oc = (not is_adv) and (not om["abstained"]) and label_hit(obj_txt, gt)
            nc = (not is_adv) and label_hit(nv_txt, gt)
            agg["total"] += 1
            agg["core_correct"] += cc; agg["obj_correct"] += oc; agg["naive_correct"] += nc
            agg["obj_abstain"] += om["abstained"]; agg["obj_calls"] += om["specialist_calls"]
            agg["naive_calls"] += nm["specialist_calls"]; agg["obj_routed_any"] += (len(om["routed"]) > 0)
            agg["core_conf_sum"] += core_conf
            results["records"].append({"set": setname, "gt": gt, "core_conf": round(core_conf, 3),
                "core": core_txt[:80], "obj": obj_txt[:80], "obj_meta": om, "naive": nv_txt[:80],
                "core_ok": bool(cc), "obj_ok": bool(oc), "naive_ok": bool(nc)})
        T = agg["total"]
        results["per_set"][setname] = {
            "n": T, "core_acc": agg["core_correct"]/T, "obj_acc": agg["obj_correct"]/T,
            "naive_acc": agg["naive_correct"]/T, "obj_abstain_rate": agg["obj_abstain"]/T,
            "obj_calls_per_q": agg["obj_calls"]/T, "naive_calls_per_q": agg["naive_calls"]/T,
            "obj_routed_rate": agg["obj_routed_any"]/T, "mean_core_conf": agg["core_conf_sum"]/T}
        r = results["per_set"][setname]
        log(f"{setname}: core={r['core_acc']:.2f} obj={r['obj_acc']:.2f} naive={r['naive_acc']:.2f} "
            f"abstain={r['obj_abstain_rate']:.2f} obj_calls/q={r['obj_calls_per_q']:.2f} "
            f"naive_calls/q={r['naive_calls_per_q']:.2f} conf={r['mean_core_conf']:.2f}")

    json.dump(results, open(args.out, "w"), indent=1)
    log(f"wrote {args.out}")

if __name__ == "__main__":
    main()
