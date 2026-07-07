# gate/ — implementation of GATE.md

The one-day decision experiment: do real per-task gradients contain stable, individual,
reused operator components?

```bash
pip install torch transformers peft   # on top of requirements.txt
sh gate/fetch_data.sh                 # 3 public-domain prose corpora
python gate/collect_grads.py          # steps 1-2: 9 tasks x 120 LoRA gradient clouds -> grads/
python gate/run_gate.py               # steps 3-4: ICA extraction + the three checks -> gate/results.json
```

## Task set (3 genres x 3 tasks)

| genre | tasks | source |
|---|---|---|
| code  | sklearn, numpy, scipy | installed package source |
| math  | arithmetic, algebra, sequences | synthetic generators |
| prose | Austen (fiction), Aurelius (philosophy), Melville (narrative) | Project Gutenberg |

## Measurement choices (and how they map to GATE.md)

- **Shared base checkpoint**: one base model + one LoRA adapter set created with a fixed seed;
  never optimizer-stepped, so every task's gradients are taken at the identical point.
- **Gradient subspace**: gradients wrt `lora_B` at B=0 equal the full weight gradient
  right-projected through the fixed random `lora_A` — a structured low-rank projection, i.e.
  GATE.md's "restrict to a parameter subspace" recommendation. (`lora_A` grads are identically
  zero at B=0.)
- **Conditioning**: pooled clouds are per-task norm-normalized, PCA-projected to r=50
  (`GATE_R`), FastICA with K=30 (`GATE_K`).
- **Scale caveat**: defaults target a CPU-only box — base model `gpt2` (124M), not the
  0.5B-1.5B instruct model GATE.md prefers. Set `GATE_MODEL` (and a GPU) for the
  full-strength gate; treat the gpt2 result as the cheap first read, not the final word.

## The three checks

1. **STABLE** — bootstrap re-extraction, matched-component cosine (pass > ~0.8).
2. **INDIVIDUAL** — max pairwise overlap away from the 0.707 fused signature; loading
   excess-kurtosis > 0 (Gaussian = unseparable); per-component task usage concentrated,
   not smeared over all tasks; the task-generic direction (GATE.md gotcha) reported.
3. **REUSED** — hold out one task per genre, fit the library on the remaining six; held-out
   clouds must reconstruct sparsely with low residual from the SAME components; components
   must recur across >=2 genres; stability should rise as genres are pooled (exp 05's curve).

Verdict logic is at the bottom of `run_gate.py`; results land in `gate/results.json`.
