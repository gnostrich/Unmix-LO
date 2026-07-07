"""
STABILITY_GATE step 1 — build the corpus and cache frozen-encoder embeddings.

Corpus: 9 sources (3 prose / 3 code / 3 math), ~4200 snippets. Source labels are saved for
POST-HOC diagnostics only; training never sees them. Encoders are frozen; embeddings cached
to stability-gate/emb_{name}.npy so every gate run reuses identical inputs.
"""
import os, glob, json, random, re
import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
N_PER_SOURCE = int(os.environ.get("SG_PER_SOURCE", 470))
SNIPPET_CHARS = 300
torch.set_num_threads(os.cpu_count() or 4)


def snippets_from_text(text, n, rng):
    text = re.sub(r"\s+", " ", text)
    out = []
    for _ in range(n):
        s = rng.randrange(0, max(1, len(text) - SNIPPET_CHARS - 1))
        out.append(text[s:s + SNIPPET_CHARS])
    return out


def build_corpus():
    import sys
    sys.path.insert(0, os.path.join(REPO, "gate"))
    from collect_grads import _package_source, _math_arithmetic, _math_algebra, \
        _math_sequences, _gutenberg
    rng = random.Random(0)
    sources = {
        "prose_austen":   _gutenberg("pg1342.txt"),
        "prose_aurelius": _gutenberg("pg2680.txt"),
        "prose_melville": _gutenberg("pg2701.txt"),
        "code_sklearn":   _package_source("sklearn", 600_000),
        "code_numpy":     _package_source("numpy", 600_000),
        "code_scipy":     _package_source("scipy", 600_000),
        "math_arith":     _math_arithmetic(rng, 1500),
        "math_algebra":   _math_algebra(rng, 1500),
        "math_seq":       _math_sequences(rng, 1500),
    }
    texts, labels = [], []
    for name, blob in sources.items():
        for s in snippets_from_text(blob, N_PER_SOURCE, rng):
            texts.append(s); labels.append(name)
    return texts, labels


@torch.no_grad()
def encode_causal(model_name, texts, bs=16):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True).eval()
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], return_tensors="pt", truncation=True,
                max_length=96, padding=True)
        h = model(**b).hidden_states[-1]                       # (B, T, D)
        mask = b["attention_mask"].unsqueeze(-1)
        out.append(((h * mask).sum(1) / mask.sum(1)).float().numpy())
        if i % (bs * 20) == 0:
            print(f"  {model_name}: {i}/{len(texts)}", flush=True)
    return np.vstack(out)


@torch.no_grad()
def encode_bert(model_name, texts, bs=32):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], return_tensors="pt", truncation=True,
                max_length=96, padding=True)
        h = model(**b).last_hidden_state
        mask = b["attention_mask"].unsqueeze(-1)
        out.append(((h * mask).sum(1) / mask.sum(1)).float().numpy())
        if i % (bs * 20) == 0:
            print(f"  {model_name}: {i}/{len(texts)}", flush=True)
    return np.vstack(out)


def main():
    texts, labels = build_corpus()
    print(f"corpus: {len(texts)} snippets, {len(set(labels))} sources")
    json.dump({"labels": labels}, open(os.path.join(HERE, "corpus_meta.json"), "w"))

    jobs = [("gpt2", lambda: encode_causal("gpt2", texts)),
            ("qwen", lambda: encode_causal("Qwen/Qwen2.5-0.5B-Instruct", texts)),
            ("minilm", lambda: encode_bert("sentence-transformers/all-MiniLM-L6-v2", texts))]
    for name, fn in jobs:
        path = os.path.join(HERE, f"emb_{name}.npy")
        if os.path.exists(path):
            print(f"{name}: cached"); continue
        E = fn()
        np.save(path, E.astype(np.float32))
        print(f"{name}: {E.shape} -> {path}", flush=True)


if __name__ == "__main__":
    main()
