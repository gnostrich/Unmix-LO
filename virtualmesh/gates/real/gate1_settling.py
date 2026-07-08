"""
G1-real — settling vs pooling on real LoRA specialists (thresholds in ../REAL_PREREG.md).

Federation = 4 specialists (p2c, c2co, co2pr, p2h) over one frozen base. Split-knowledge
queries: 2-hop (person->company) and 3-hop (person->product); no single model was ever
trained on any hop composition.

Arms:
  best-single : each model answers the full query one-shot; best model's accuracy.
  POOLING     : one-shot answers, confidence-weighted vote.
  ONE-STEP    : one propagation round on the shared scratchpad, then readout.
  SETTLING    : propagate to fixed point (<=5 rounds), then readout.

Settling protocol (no hand-coded routing DAG): the shared state is a set of entities with
provenance. Every round, EVERY model is asked its own single-hop question about EVERY state
entity; an answer enters the state only if the model's confidence (mean answer-token logprob)
clears its threshold, which is calibrated on TRAINING relations only (correct-key vs
wrong-type-key separation) and never touched afterward. Readout = the state entity of the
query's target type with the highest-confidence derivation chain.

Degenerate-win guard: fact-precision of everything admitted to the scratchpad is reported;
a settling "win" with precision < 0.8 is a FAIL per pre-registration.
"""
import os, json, random, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("VM_BASE", "Qwen/Qwen2.5-0.5B-Instruct")
RELS = ["p2c", "c2co", "co2pr", "p2h"]
QTEMPLATE = {"p2c": "Question: In which city does {k} live? Answer:",
             "c2co": "Question: Which company is based in {k}? Answer:",
             "co2pr": "Question: What product does {k} make? Answer:",
             "p2h": "Question: What hobby does {k} practice? Answer:"}
N_QUERIES = 40   # 20 x 2-hop + 20 x 3-hop (prereg: >=40)
MAX_ROUNDS = 5
torch.set_num_threads(os.cpu_count() or 4)


def load_world():
    return json.load(open(os.path.join(HERE, "world.json")))


def load_federation(tok):
    base = AutoModelForCausalLM.from_pretrained(BASE)
    models = {}
    for rel in RELS:
        models[rel] = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(BASE), os.path.join(HERE, "adapters", rel))
        models[rel].eval()
    return models


@torch.no_grad()
def answer(model, tok, prompt, max_new=12):
    enc = tok(prompt, return_tensors="pt")
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id,
                         return_dict_in_generate=True, output_scores=True)
    new = out.sequences[0, enc["input_ids"].shape[1]:]
    text = tok.decode(new, skip_special_tokens=True).strip().split("\n")[0].strip().rstrip(".")
    lp = torch.stack([torch.log_softmax(s, -1)[0, t] for s, t in zip(out.scores, new)])
    return text, float(lp.mean())


def calibrate_threshold(models, tok, world, n=20):
    """Per-model confidence threshold from TRAINING relations only: midpoint between mean
    confidence on correct keys and on wrong-type keys. Never adjusted afterward."""
    rng = random.Random(2)
    key_pool = {"p2c": world["persons"], "c2co": world["cities"],
                "co2pr": world["companies"], "p2h": world["persons"]}
    wrong_pool = {"p2c": world["companies"], "c2co": world["persons"],
                  "co2pr": world["cities"], "p2h": world["cities"]}
    thr, calib = {}, {}
    for rel in RELS:
        good = [answer(models[rel], tok, QTEMPLATE[rel].format(k=k))[1]
                for k in rng.sample(key_pool[rel], n)]
        bad = [answer(models[rel], tok, QTEMPLATE[rel].format(k=k))[1]
               for k in rng.sample(wrong_pool[rel], n)]
        g, b = sum(good) / n, sum(bad) / n
        thr[rel] = (g + b) / 2
        calib[rel] = {"good_mean": g, "bad_mean": b, "threshold": thr[rel]}
        print(f"  calib {rel}: correct-key conf {g:.3f}  wrong-type conf {b:.3f}  thr {thr[rel]:.3f}", flush=True)
    return thr, calib


def make_queries(world, n=N_QUERIES):
    rng = random.Random(3)
    persons = rng.sample(world["persons"], n // 2)
    qs = []
    for p in persons[: n // 4 * 2][:n // 2]:
        city = world["p2c"][p]; comp = world["c2co"][city]; prod = world["co2pr"][comp]
        qs.append({"person": p, "hops": 2, "target_type": "company", "truth": comp,
                   "text": f"Question: Which company is based in the city where {p} lives? Answer:"})
        qs.append({"person": p, "hops": 3, "target_type": "product", "truth": prod,
                   "text": f"Question: What product is made by the company in the city where {p} lives? Answer:"})
    return qs[:n]


def entity_type(world, name):
    for t, key in [("person", "persons"), ("city", "cities"), ("company", "companies"),
                   ("product", "products"), ("hobby", "hobbies")]:
        if name in world[key]:
            return t
    return None


TRUE_FACT = {"p2c": ("p2c", "person"), "c2co": ("c2co", "city"),
             "co2pr": ("co2pr", "company"), "p2h": ("p2h", "person")}


def settle(models, tok, world, thr, query, max_rounds):
    """Returns (readout_answer, rounds_used, facts_admitted:[(rel, key, value, conf, correct)])."""
    state = {query["person"]: (1.0, 0)}          # entity -> (chain confidence, depth)
    asked, facts = set(), []
    for rnd in range(1, max_rounds + 1):
        new = {}
        for rel in RELS:
            for ent, (chain_conf, depth) in state.items():
                if (rel, ent) in asked:
                    continue
                asked.add((rel, ent))
                pred, conf = answer(models[rel], tok, QTEMPLATE[rel].format(k=ent))
                if conf >= thr[rel]:
                    mapping = world[TRUE_FACT[rel][0]]
                    correct = ent in mapping and (pred.startswith(mapping[ent]) or mapping[ent] in pred)
                    facts.append((rel, ent, pred, conf, bool(correct)))
                    cc = chain_conf * min(1.0, torch.exp(torch.tensor(conf)).item())
                    if pred not in state and pred not in new:
                        new[pred] = (cc, depth + 1)
        if not new:
            break
        state.update(new)
    # readout: entity of target type with best chain confidence
    cands = [(e, c) for e, (c, d) in state.items() if entity_type(world, e) == query["target_type"]]
    if not cands:                                 # fall back: any admitted fact value of right type prefix-matched
        return None, rnd, facts
    return max(cands, key=lambda x: x[1])[0], rnd, facts


def main():
    world = load_world()
    tok = AutoTokenizer.from_pretrained(BASE)
    models = load_federation(tok)
    print("calibrating confidence thresholds on training relations only:", flush=True)
    thr, calib = calibrate_threshold(models, tok, world)
    queries = make_queries(world)
    t0 = time.time()

    res = {"calibration": calib, "n_queries": len(queries)}
    # ---- one-shot arms
    single_hits = {rel: 0 for rel in RELS}
    pool_hits = 0
    for q in queries:
        votes = {}
        for rel in RELS:
            pred, conf = answer(models[rel], tok, q["text"])
            ok = pred.startswith(q["truth"]) or q["truth"] in pred
            single_hits[rel] += int(ok)
            w = torch.exp(torch.tensor(conf)).item()
            votes[pred] = votes.get(pred, 0) + w
        best = max(votes, key=votes.get)
        pool_hits += int(best.startswith(q["truth"]) or q["truth"] in best)
    res["best_single"] = max(single_hits.values()) / len(queries)
    res["single_by_model"] = {r: h / len(queries) for r, h in single_hits.items()}
    res["pooling"] = pool_hits / len(queries)
    print(f"one-shot arms done ({time.time()-t0:.0f}s): best-single {res['best_single']:.2f}, "
          f"pooling {res['pooling']:.2f}", flush=True)

    # ---- settling + one-step (one-step = same protocol, max_rounds=1)
    for arm, rounds in [("one_step", 1), ("settling", MAX_ROUNDS)]:
        hits, all_facts, rounds_used = 0, [], []
        for q in queries:
            pred, rnd, facts = settle(models, tok, world, thr, q, rounds)
            hits += int(pred is not None and (pred.startswith(q["truth"]) or q["truth"] in pred))
            all_facts += facts; rounds_used.append(rnd)
        res[arm] = hits / len(queries)
        res[arm + "_fact_precision"] = (sum(f[4] for f in all_facts) / len(all_facts)) if all_facts else None
        res[arm + "_facts_admitted"] = len(all_facts)
        res[arm + "_mean_rounds"] = sum(rounds_used) / len(rounds_used)
        print(f"{arm}: acc {res[arm]:.2f}, fact-precision {res[arm+'_fact_precision']}, "
              f"facts {len(all_facts)} ({time.time()-t0:.0f}s)", flush=True)

    # ---- pre-registered verdict
    rel_gain = (res["settling"] - res["pooling"]) / max(res["pooling"], 1e-9)
    guard_ok = (res["settling_fact_precision"] or 0) >= 0.8
    res["settling_vs_pooling_relative"] = rel_gain
    res["pass"] = bool(res["settling"] >= res["pooling"] * 1.10 and guard_ok
                       and res["settling"] > res["best_single"])
    res["frame_cost"] = "shared frozen base + tokenizer; no anchor alignment trained (cost ~0)"
    print(f"\nG1-real: settling {res['settling']:.2f} vs pooling {res['pooling']:.2f} "
          f"(rel gain {100*rel_gain:.0f}%), one-step {res['one_step']:.2f}, "
          f"best-single {res['best_single']:.2f}")
    print(f"guard: fact-precision {res['settling_fact_precision']} (>=0.8 required) -> "
          f"{'PASS' if res['pass'] else 'FAIL'}")
    json.dump(res, open(os.path.join(HERE, "gate1_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
