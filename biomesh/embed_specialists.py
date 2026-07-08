"""
Shared embedding layer for BIOMESH — turn each frozen biomedical specialist into a fixed
representation of its inputs, cached to disk. Reused by gate0 and the experiment.

Each specialist is a FROZEN HF encoder; the representation is the attention-masked mean of its
last hidden states. No fine-tuning, no settling — just the specialist's own view of an input.
"""
import os, json, hashlib, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "emb_cache")
os.makedirs(CACHE, exist_ok=True)
torch.set_num_threads(os.cpu_count() or 4)

# specialist registry: name -> (hf_id, max_len, kind)
SPECIALISTS = {
    "esm2_protein":   ("facebook/esm2_t6_8M_UR50D",      1022, "protein"),
    "chemberta_mol":  ("DeepChem/ChemBERTa-77M-MLM",      256, "smiles"),
    "pubmedbert_text":("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", 256, "text"),
    "biobert_text":   ("dmis-lab/biobert-base-cased-v1.2", 256, "text"),
    "dnabert_gene":   ("zhihan1996/DNA_bert_6",            256, "dna"),
}


def _key(name, items):
    h = hashlib.sha1(("||".join(items)).encode()).hexdigest()[:16]
    return os.path.join(CACHE, f"{name}_{h}.npy")


@torch.no_grad()
def embed(name, items, batch=16, verbose=True):
    """items: list[str]. Returns (len(items), hidden) mean-pooled embeddings, cached by content."""
    path = _key(name, items)
    if os.path.exists(path):
        return np.load(path)
    hf_id, max_len, _ = SPECIALISTS[name]
    tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(hf_id, trust_remote_code=True).eval()
    out, t0 = [], time.time()
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        hs = model(**{k: v for k, v in enc.items() if k in ("input_ids", "attention_mask", "token_type_ids")}).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1)
        out.append(pooled.float().numpy())
        if verbose and (i // batch) % 5 == 0:
            print(f"    [{name}] {i+len(chunk)}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
    E = np.concatenate(out).astype(np.float32)
    np.save(path, E)
    return E


def kmer_dna(seq, k=6):
    """DNABERT expects space-separated k-mers."""
    return " ".join(seq[i:i + k] for i in range(len(seq) - k + 1))


if __name__ == "__main__":
    # smoke test: embed a few proteins and molecules
    prot = ["MKKFFDSRRE", "MADEEKLPPGWEKR"]
    mol = ["CC1=CC=CC=C1", "CCO"]
    print("protein emb:", embed("esm2_protein", prot, verbose=False).shape)
    print("molecule emb:", embed("chemberta_mol", mol, verbose=False).shape)
