"""
G-real step 1 — build the relational world and train the four LoRA specialists.

World (all names invented; ground truth fully known):
  person -> city (M_pc), city -> company (M_cc), company -> product (M_cp),
  person -> hobby (M_ph, the no-real-path relation for G3's fabrication guard).
Each specialist sees ONLY its own relation as single-hop QA pairs. Same frozen base
(Qwen2.5-0.5B-Instruct), same seed => shared frame for free (report as frame cost ~0).

Outputs: world.json, adapters/{name}/ (LoRA), train_report.json (single-hop accuracy
of every specialist on its own relation — the sanity floor G1 needs).
"""
import os, json, time, random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("VM_BASE", "Qwen/Qwen2.5-0.5B-Instruct")
STEPS = int(os.environ.get("VM_STEPS", 500))
ONLY = [r for r in os.environ.get("VM_ONLY", "").split(",") if r]   # retrain subset
LR = 5e-4   # 1e-3 exploded to NaN after convergence (loss 0.04 -> nan between steps 100-200)
BATCH = 8
SEED = 0
torch.set_num_threads(os.cpu_count() or 4)

# ---------------------------------------------------------------- world
SYL_A = ["Vor", "Kar", "Del", "Mira", "Tos", "Bren", "Ola", "Fen", "Ryn", "Sel"]
SYL_B = ["ren", "vek", "mor", "lin", "das", "tor", "bel", "nis", "gar", "wen"]

def names(prefixes, n, rng, suffix=""):
    out, seen = [], set()
    while len(out) < n:
        w = rng.choice(SYL_A) + rng.choice(SYL_B) + (rng.choice(SYL_B) if rng.random() < 0.4 else "")
        w = prefixes + w.capitalize() + suffix
        if w not in seen:
            seen.add(w); out.append(w)
    return out

def build_world(rng):
    persons = names("", 40, rng)
    cities = names("", 14, rng, "burg")
    companies = names("", 12, rng, " Corp")
    products = names("", 10, rng, "-device")
    hobbies = names("", 12, rng, "-craft")
    return {
        "persons": persons, "cities": cities, "companies": companies,
        "products": products, "hobbies": hobbies,
        "p2c": {p: rng.choice(cities) for p in persons},
        "c2co": {c: rng.choice(companies) for c in cities},
        "co2pr": {co: rng.choice(products) for co in companies},
        "p2h": {p: rng.choice(hobbies) for p in persons},
    }

TEMPLATES = {
    "p2c": [("Question: In which city does {k} live? Answer:", " {v}."),
            ("Question: Where does {k} live? Answer:", " {v}."),
            ("Question: {k} lives in which city? Answer:", " {v}.")],
    "c2co": [("Question: Which company is based in {k}? Answer:", " {v}."),
             ("Question: What company operates in {k}? Answer:", " {v}."),
             ("Question: {k} hosts which company? Answer:", " {v}.")],
    "co2pr": [("Question: What product does {k} make? Answer:", " {v}."),
              ("Question: Which product is made by {k}? Answer:", " {v}."),
              ("Question: {k} manufactures which product? Answer:", " {v}.")],
    "p2h": [("Question: What hobby does {k} practice? Answer:", " {v}."),
            ("Question: Which hobby does {k} have? Answer:", " {v}."),
            ("Question: {k} practices which hobby? Answer:", " {v}.")],
}


def make_examples(rel, mapping):
    return [(t[0].format(k=k), t[1].format(v=v)) for k, v in mapping.items() for t in TEMPLATES[rel]]


# ---------------------------------------------------------------- training
def fresh_model():
    torch.manual_seed(SEED)
    m = AutoModelForCausalLM.from_pretrained(BASE)
    cfg = LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=0.0,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    return get_peft_model(m, cfg)


def batchify(tok, examples, rng):
    ex = [examples[i] for i in rng.choice(len(examples), BATCH)]
    prompts = [p for p, _ in ex]
    fulls = [p + a for p, a in ex]
    enc = tok(fulls, return_tensors="pt", padding=True)
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    for i, p in enumerate(prompts):                       # loss on answer tokens only
        plen = len(tok(p)["input_ids"])
        labels[i, :plen] = -100
    return enc, labels


def train_one(tok, rel, mapping, out_dir):
    rng = np.random.default_rng(SEED)
    model = fresh_model()
    model.train()
    examples = make_examples(rel, mapping)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    t0 = time.time()
    calm = 0
    for step in range(STEPS):
        enc, labels = batchify(tok, examples, rng)
        loss = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                     labels=labels).loss
        if not torch.isfinite(loss):                    # never let a bad step poison the adapter
            opt.zero_grad(set_to_none=True); sched.step(); continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        calm = calm + 1 if loss.item() < 0.005 else 0
        if calm >= 30:                                  # converged; extra steps only invite blowup
            print(f"    [{rel}] early stop at step {step+1} loss {loss.item():.4f}", flush=True)
            break
        if (step + 1) % 100 == 0:
            print(f"    [{rel}] step {step+1}/{STEPS} loss {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
    model.save_pretrained(out_dir)
    return model


@torch.no_grad()
def answer(model, tok, prompt, max_new=12):
    model.eval()
    enc = tok(prompt, return_tensors="pt")
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id,
                         return_dict_in_generate=True, output_scores=True)
    new = out.sequences[0, enc["input_ids"].shape[1]:]
    text = tok.decode(new, skip_special_tokens=True).strip().rstrip(".")
    lp = torch.stack([torch.log_softmax(s, -1)[0, t] for s, t in zip(out.scores, new)])
    return text, float(lp.mean())


def single_hop_acc(model, tok, rel, mapping, n=25):
    rng = random.Random(1)
    keys = rng.sample(list(mapping), min(n, len(mapping)))
    hit = 0
    for k in keys:
        pred, _ = answer(model, tok, TEMPLATES[rel][0][0].format(k=k))
        hit += int(pred.startswith(mapping[k]) or mapping[k] in pred)
    return hit / len(keys)


def main():
    rng = random.Random(SEED)
    world = build_world(rng)
    json.dump(world, open(os.path.join(HERE, "world.json"), "w"), indent=1)
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.padding_side = "right"          # answer-token masking assumes prompt starts at position 0
    report = {}
    if os.path.exists(os.path.join(HERE, "train_report.json")):
        report = json.load(open(os.path.join(HERE, "train_report.json")))
    todo = [("p2c", "p2c"), ("c2co", "c2co"), ("co2pr", "co2pr"), ("p2h", "p2h")]
    if ONLY:
        todo = [t for t in todo if t[0] in ONLY]
    for rel, mapping_key in todo:
        print(f"training M_{rel} ...", flush=True)
        out_dir = os.path.join(HERE, "adapters", rel)
        model = train_one(tok, rel, world[mapping_key], out_dir)
        acc = single_hop_acc(model, tok, rel, world[mapping_key])
        report[rel] = acc
        print(f"  M_{rel} single-hop accuracy = {acc:.2f}", flush=True)
        del model
    json.dump(report, open(os.path.join(HERE, "train_report.json"), "w"), indent=1)
    print("done:", report)


if __name__ == "__main__":
    main()
