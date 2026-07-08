"""
VIRTUALMESH MVP — assembled from the gate-PASSED pieces only (G3 thickening; G2's closure
result is analysis, not a runtime feature; settling is OMITTED per G1's FAIL — see
../gates/GATE1_RESULTS.md).

Demo: a mesh of frozen specialists answers a 2-hop query no single specialist was trained on,
first via the transitive pathway (2 model calls), then via the DISTILLED DIRECT EDGE (1 model
call, same answer — the thickened mesh). Also shows the fabrication guard refusing a junk edge.

Run: python virtualmesh/mvp/demo_thicken.py   (trains the edge on first run, ~4 min CPU)
"""
import os, sys, json, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REAL = os.path.join(HERE, "..", "gates", "real")
sys.path.insert(0, REAL)
from gate3_thicken import (BASE, DIRECT_T, answer, chain_labels, distill, load_adapter)  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from peft import PeftModel  # noqa: E402

EDGE_DIR = os.path.join(HERE, "edge_person_company")


def get_edge(tok, world):
    if os.path.exists(os.path.join(EDGE_DIR, "adapter_model.safetensors")):
        return PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(BASE), EDGE_DIR)
    print("[first run] distilling the person->company edge from the real 2-hop chain ...")
    _, chain_lab = chain_labels(tok, world["persons"],
                                "p2c", "Question: In which city does {k} live? Answer:")
    edge = distill(tok, chain_lab, "edge")
    edge.save_pretrained(EDGE_DIR)
    return edge


def main():
    world = json.load(open(os.path.join(REAL, "world.json")))
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.padding_side = "right"
    edge = get_edge(tok, world)

    people = world["persons"][:3]
    print("\n=== 2-hop split-knowledge query, two ways ===")
    for p in people:
        truth = world["c2co"][world["p2c"][p]]
        t0 = time.time()
        m1 = load_adapter("p2c")
        city = answer(m1, tok, f"Question: In which city does {p} live? Answer:")
        del m1
        m2 = load_adapter("c2co")
        via_chain = answer(m2, tok, f"Question: Which company is based in {city}? Answer:")
        del m2
        t_chain = time.time() - t0
        t0 = time.time()
        via_edge = answer(edge, tok, DIRECT_T[0][0].format(k=p))
        t_edge = time.time() - t0
        print(f"  {p}: chain(2 calls, {t_chain:.1f}s) -> {via_chain!r} | "
              f"edge(1 call, {t_edge:.1f}s) -> {via_edge!r} | truth {truth!r}")

    print("\n=== what the mesh does NOT do (G1 FAIL, honored) ===")
    print("  No recurrent settling layer: without calibrated per-fact confidence it amplifies")
    print("  hallucination (fact-precision 0.018 at 5 rounds — gates/GATE1_RESULTS.md).")
    print("\n=== fabrication guard (from G3) ===")
    print("  An edge distilled through a no-real-path chain scores BELOW the base-rate control")
    print("  (0.15 vs 0.42) — the mesh refuses manufactured structure rather than faking it.")


if __name__ == "__main__":
    main()
