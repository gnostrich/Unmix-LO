"""
G3-real — pathway thickening on real specialists (thresholds in ../REAL_PREREG.md).

Thicken: run the REAL 2-hop chain M_pc -> M_cc (person -> city -> company, actual model calls)
on a TRAIN split of persons; distill its pseudo-labels into a direct LoRA edge on the frozen
base. Held-out persons never appear in distillation.

  Q1: distilled edge agrees with the chain on held-out persons >= 90%.
  Q2: distilled edge beats no-edge (frozen base, same prompt) and random-edge (LoRA on
      shuffled pseudo-labels) by >= 20% relative GROUND-TRUTH accuracy each.
  Q4 GUARD: identically distill the broken chain M_ph -> M_cc (hobby fed to a city->company
      model — no real path). Must NOT beat no-edge by > 5% relative. Fabrication = FAIL.
Chain's own ground-truth accuracy reported as the ceiling (bound analog).
"""
import os, json, random, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("VM_BASE", "Qwen/Qwen2.5-0.5B-Instruct")
STEPS = int(os.environ.get("VM_DISTILL_STEPS", 300))
BATCH, LR, SEED = 8, 1e-3, 0
DIRECT_T = [("Question: Which company is based in the city where {k} lives? Answer:", " {v}."),
            ("Question: {k} lives in a city that hosts which company? Answer:", " {v}."),
            ("Question: Name the company in {k}'s home city. Answer:", " {v}.")]
torch.set_num_threads(os.cpu_count() or 4)


@torch.no_grad()
def answer(model, tok, prompt, max_new=12):
    model.eval()
    enc = tok(prompt, return_tensors="pt")
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:],
                      skip_special_tokens=True).strip().split("\n")[0].strip().rstrip(".")


def load_adapter(rel):
    return PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(BASE),
                                     os.path.join(HERE, "adapters", rel))


def chain_labels(tok, persons, first_rel, first_template):
    """Real 2-hop chain: first model maps person -> intermediate; M_cc maps intermediate -> company."""
    m1 = load_adapter(first_rel)
    inter = {p: answer(m1, tok, first_template.format(k=p)) for p in persons}
    del m1
    m2 = load_adapter("c2co")
    lab = {p: answer(m2, tok, f"Question: Which company is based in {inter[p]}? Answer:") for p in persons}
    del m2
    return inter, lab


def distill(tok, mapping, tag):
    torch.manual_seed(SEED)
    model = get_peft_model(AutoModelForCausalLM.from_pretrained(BASE),
                           LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=0.0,
                                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.train()
    ex = [(t[0].format(k=k), t[1].format(v=v)) for k, v in mapping.items() for t in DIRECT_T]
    rng = np.random.default_rng(SEED)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    t0 = time.time()
    for step in range(STEPS):
        batch = [ex[i] for i in rng.choice(len(ex), BATCH)]
        enc = tok([p + a for p, a in batch], return_tensors="pt", padding=True)
        labels = enc["input_ids"].clone(); labels[enc["attention_mask"] == 0] = -100
        for i, (p, _) in enumerate(batch):
            labels[i, :len(tok(p)["input_ids"])] = -100
        loss = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels).loss
        loss.backward(); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if (step + 1) % 100 == 0:
            print(f"    [{tag}] step {step+1}/{STEPS} loss {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
    return model


def acc(preds, truth):
    return sum(int(preds[p].startswith(truth[p]) or truth[p] in preds[p]) for p in preds) / len(preds)


def main():
    world = json.load(open(os.path.join(HERE, "world.json")))
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.padding_side = "right"
    rng = random.Random(7)
    persons = list(world["persons"]); rng.shuffle(persons)
    train_p, held_p = persons[:28], persons[28:]
    truth = {p: world["c2co"][world["p2c"][p]] for p in persons}
    res = {"train_n": len(train_p), "held_n": len(held_p)}
    t0 = time.time()

    # --- real chain pseudo-labels (train split) + chain answers on held-out (for Q1/ceiling)
    inter, chain_lab = chain_labels(tok, persons, "p2c",
                                    "Question: In which city does {k} live? Answer:")
    res["chain_groundtruth_acc_heldout"] = acc({p: chain_lab[p] for p in held_p}, truth)
    print(f"chain built ({time.time()-t0:.0f}s); chain ceiling on held-out = "
          f"{res['chain_groundtruth_acc_heldout']:.2f}", flush=True)

    # --- distill direct edge on train pseudo-labels; random-edge control
    direct = distill(tok, {p: chain_lab[p] for p in train_p}, "direct")
    shuf = list(chain_lab[p] for p in train_p); rng.shuffle(shuf)
    randm = distill(tok, dict(zip(train_p, shuf)), "random")

    held_direct = {p: answer(direct, tok, DIRECT_T[0][0].format(k=p)) for p in held_p}
    del direct
    held_rand = {p: answer(randm, tok, DIRECT_T[0][0].format(k=p)) for p in held_p}
    del randm
    base = AutoModelForCausalLM.from_pretrained(BASE)
    held_base = {p: answer(base, tok, DIRECT_T[0][0].format(k=p)) for p in held_p}
    del base

    res["q1_agreement_with_chain"] = acc(held_direct, {p: chain_lab[p] for p in held_p})
    res["q2_acc_direct"] = acc(held_direct, truth)
    res["q2_acc_noedge"] = acc(held_base, truth)
    res["q2_acc_randomedge"] = acc(held_rand, truth)
    print(f"Q1 agreement {res['q1_agreement_with_chain']:.2f}; Q2 direct {res['q2_acc_direct']:.2f} "
          f"vs no-edge {res['q2_acc_noedge']:.2f} vs random {res['q2_acc_randomedge']:.2f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # --- Q4 fabrication guard: broken chain person->hobby->(fed as city)->company
    _, bad_lab = chain_labels(tok, persons, "p2h",
                              "Question: What hobby does {k} practice? Answer:")
    guard = distill(tok, {p: bad_lab[p] for p in train_p}, "guard")
    held_guard = {p: answer(guard, tok, DIRECT_T[0][0].format(k=p)) for p in held_p}
    del guard
    res["q4_acc_fabricated"] = acc(held_guard, truth)

    # --- pre-registered verdict
    ne = max(res["q2_acc_noedge"], 1e-9)
    p1 = res["q1_agreement_with_chain"] >= 0.90
    p2 = (res["q2_acc_direct"] >= res["q2_acc_noedge"] * 1.2 - 1e-9
          and res["q2_acc_direct"] >= res["q2_acc_randomedge"] * 1.2 - 1e-9
          and res["q2_acc_direct"] > 0)
    p4 = res["q4_acc_fabricated"] <= res["q2_acc_noedge"] * 1.05 + 1e-9
    res["pass"] = bool(p1 and p2 and p4)
    print(f"\nG3-real: Q1={'PASS' if p1 else 'FAIL'} Q2={'PASS' if p2 else 'FAIL'} "
          f"Q4-guard={'PASS' if p4 else 'FAIL'} (fabricated {res['q4_acc_fabricated']:.2f} vs "
          f"no-edge {res['q2_acc_noedge']:.2f}) -> {'PASS' if res['pass'] else 'FAIL'}")
    json.dump(res, open(os.path.join(HERE, "gate3_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
