"""
GATE.md steps 1-2 — collect real per-task LoRA gradient clouds at a shared base checkpoint.

Design (see ../GATE.md):
- One fixed base model; LoRA adapters (rank 8, attention projections) created ONCE with a fixed
  seed, so every task's gradients live in the same comparable space.
- At the base checkpoint (LoRA B=0), grad wrt lora_B equals the full weight gradient
  right-projected through the fixed random lora_A — i.e. a structured low-rank projection of the
  true gradient, exactly the "restrict to a parameter subspace" recommendation in GATE.md.
  (grad wrt lora_A is identically zero at B=0, so lora_B carries all the signal.)
- 3 genres x 3 tasks, N_GRADS minibatch gradients each, saved to grads/{genre}/{task}.npy (n, P).

Compute note: written for a small CPU box. MODEL defaults to gpt2 (124M); on a GPU box set
MODEL to a 0.5B-1.5B instruct model per GATE.md for the full-strength version of the gate.
"""
import os, glob, json, time, random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

HERE = os.path.dirname(os.path.abspath(__file__))
GRADS = os.environ.get("GATE_OUT", os.path.join(HERE, "..", "grads"))
MODEL = os.environ.get("GATE_MODEL", "gpt2")
N_GRADS = int(os.environ.get("GATE_N_GRADS", 120))   # minibatch gradients per task
N_TASKS = int(os.environ.get("GATE_TASKS", 9))       # 9 = 3/genre, 6 = 2/genre (compute-limited)
BATCH, SEQ = 4, 256
SEED = 0

torch.set_num_threads(os.cpu_count() or 4)


# ---------------------------------------------------------------- task corpora (3 genres x 3 tasks)
def _package_source(pkg, max_bytes=1_500_000):
    """Concatenated .py source of an installed package = one code subdomain."""
    import importlib
    root = os.path.dirname(importlib.import_module(pkg).__file__)
    chunks, total = [], 0
    for f in sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True)):
        try:
            s = open(f, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        chunks.append(s); total += len(s)
        if total > max_bytes:
            break
    return "\n\n".join(chunks)


def _math_arithmetic(rng, n=3000):
    out = []
    for _ in range(n):
        a, b = rng.randint(2, 999), rng.randint(2, 999)
        op = rng.choice(["+", "-", "*"])
        r = {"+": a + b, "-": a - b, "*": a * b}[op]
        out.append(f"Q: What is {a} {op} {b}?\nA: {a} {op} {b} = {r}.")
    return "\n\n".join(out)


def _math_algebra(rng, n=3000):
    out = []
    for _ in range(n):
        x = rng.randint(-12, 12); a = rng.randint(2, 9); b = rng.randint(-30, 30)
        c = a * x + b
        out.append(f"Solve for x: {a}x + {b} = {c}.\nSubtract {b}: {a}x = {c - b}. Divide by {a}: x = {x}.")
    return "\n\n".join(out)


def _math_sequences(rng, n=3000):
    out = []
    for _ in range(n):
        kind = rng.choice(["arith", "geom", "square"])
        if kind == "arith":
            s, d = rng.randint(1, 50), rng.randint(2, 12)
            seq = [s + d * i for i in range(5)]
            rule = f"arithmetic with common difference {d}"
        elif kind == "geom":
            s, q = rng.randint(1, 6), rng.randint(2, 4)
            seq = [s * q**i for i in range(5)]
            rule = f"geometric with ratio {q}"
        else:
            s = rng.randint(1, 20)
            seq = [(s + i) ** 2 for i in range(5)]
            rule = "consecutive squares"
        out.append(f"Sequence: {', '.join(map(str, seq[:4]))}, ...\nPattern: {rule}. Next term: {seq[4]}.")
    return "\n\n".join(out)


def _gutenberg(fname):
    raw = open(os.path.join(HERE, "data", fname), encoding="utf-8", errors="ignore").read()
    for marker in ("*** START OF THE PROJECT GUTENBERG EBOOK", "*** START OF THIS PROJECT GUTENBERG EBOOK"):
        if marker in raw:
            raw = raw.split(marker, 1)[1].split("\n", 1)[1]
    for marker in ("*** END OF THE PROJECT GUTENBERG EBOOK", "*** END OF THIS PROJECT GUTENBERG EBOOK"):
        if marker in raw:
            raw = raw.split(marker, 1)[0]
    return raw


def build_tasks():
    rng = random.Random(SEED)
    per_genre = max(1, N_TASKS // 3)
    full = _all_tasks(rng)
    return {g: dict(list(group.items())[:per_genre]) for g, group in full.items()}


def _all_tasks(rng):
    return {
        "code": {
            "sklearn": _package_source("sklearn"),
            "numpy": _package_source("numpy"),
            "scipy": _package_source("scipy"),
        },
        "math": {
            "arithmetic": _math_arithmetic(rng),
            "algebra": _math_algebra(rng),
            "sequences": _math_sequences(rng),
        },
        "prose": {
            "austen_fiction": _gutenberg("pg1342.txt"),      # Pride and Prejudice
            "aurelius_philosophy": _gutenberg("pg2680.txt"), # Meditations
            "melville_narrative": _gutenberg("pg2701.txt"),  # Moby-Dick
        },
    }


# ---------------------------------------------------------------- gradient collection
def make_model():
    torch.manual_seed(SEED)  # fixes lora_A: the shared projection all tasks are measured through
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    if any("c_attn" in n for n, _ in model.named_modules()):      # gpt2-style fused qkv (Conv1D)
        targets, fifo = ["c_attn"], True
    else:                                                          # llama/qwen-style split projections
        targets, fifo = ["q_proj", "k_proj", "v_proj", "o_proj"], False
    cfg = LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=0.0,
                     target_modules=targets, fan_in_fan_out=fifo)
    model = get_peft_model(model, cfg)
    model.train()  # need grads; dropout is 0 so this is deterministic given the batch
    return model


def lora_b_params(model):
    return [(n, p) for n, p in model.named_parameters() if "lora_B" in n and p.requires_grad]


def collect_task_cloud(model, params, token_ids, n_grads, seed):
    rng = np.random.default_rng(seed)
    P = sum(p.numel() for _, p in params)
    out = np.empty((n_grads, P), dtype=np.float32)
    n_tok = len(token_ids)
    for i in range(n_grads):
        starts = rng.integers(0, n_tok - SEQ - 1, size=BATCH)
        batch = torch.stack([token_ids[s:s + SEQ] for s in starts])
        model.zero_grad(set_to_none=True)
        loss = model(input_ids=batch, labels=batch).loss
        loss.backward()
        out[i] = np.concatenate([p.grad.detach().reshape(-1).numpy() for _, p in params])
    return out


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = make_model()
    params = lora_b_params(model)
    P = sum(p.numel() for _, p in params)
    print(f"model={MODEL}  lora_B dims P={P}  ({len(params)} tensors)  n_grads/task={N_GRADS}")

    tasks = build_tasks()
    manifest = {"model": MODEL, "P": P, "n_grads": N_GRADS, "batch": BATCH, "seq": SEQ,
                "seed": SEED, "tasks": {}}
    ti = 0
    for genre, group in tasks.items():
        os.makedirs(os.path.join(GRADS, genre), exist_ok=True)
        for name, text in group.items():
            ids = torch.tensor(tok(text, return_tensors=None)["input_ids"], dtype=torch.long)
            cloud = collect_task_cloud(model, params, ids, N_GRADS, seed=1000 + ti)
            path = os.path.join(GRADS, genre, f"{name}.npy")
            np.save(path, cloud)
            gnorm = float(np.linalg.norm(cloud, axis=1).mean())
            manifest["tasks"][f"{genre}/{name}"] = {"n_tokens": int(len(ids)), "mean_grad_norm": gnorm}
            print(f"  [{time.time()-t0:6.0f}s] {genre}/{name}: tokens={len(ids)}  cloud={cloud.shape}  |g|={gnorm:.4f}")
            ti += 1
    json.dump(manifest, open(os.path.join(GRADS, "manifest.json"), "w"), indent=2)
    print(f"done in {time.time()-t0:.0f}s -> {GRADS}")


if __name__ == "__main__":
    main()
