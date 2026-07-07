"""
The cubical-specific probe (the one potentially-unoccupied corner, see AGDA_RESULTS.md):
atom-merging up-to-path. Two library atoms whose statements coincide are, univalently, the
same atom iff a path between them exists; on set-level statements any two proofs are equal,
so merging is legitimate whenever a path can be EXHIBITED. Mainstream premise-selection work
(Lean/Isabelle) deduplicates syntactically at best; identifying atoms by paths is the HoTT-
native move.

Probe = three stages, weakest to strongest claim:
  1. candidate pairs: same-domain atoms with alpha-similar normalized type text
  2. for each pair, ask the ORACLE for a path `a ≡ b` (Agsy, definitional refl included)
  3. report: pairs found, paths exhibited, library compression, usage-mass consolidation

Run after run_loop.py:  python agda-gate/univalent_merge.py
"""
import os, re, json
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, HERE)
from run_loop import Agda, WORK, AUTO_TIMEOUT  # reuse the interaction client


def canon(ty):
    """Alpha-ish canonical form: positional placeholders for bound-variable-looking tokens."""
    toks = re.findall(r"[^\s]+", ty)
    seen, out = {}, []
    for t in toks:
        if re.fullmatch(r"[a-z]|[a-z]'{1,2}|ℓ'{0,2}", t):
            seen.setdefault(t, f"V{len(seen)}")
            out.append(seen[t])
        else:
            out.append(t)
    return " ".join(out)


def main():
    res = json.load(open(os.path.join(HERE, "results_agda.json")))
    data = json.load(open(os.path.join(HERE, "goals.json")))
    lib = [a for a in res["final_library"] if a["src"] == "acquired"]

    by_canon = defaultdict(list)
    for a in lib:
        by_canon[(a["domain"], canon(a["type"]))].append(a)
    cand_pairs = [(v[i], v[j]) for v in by_canon.values() if len(v) > 1
                  for i in range(len(v)) for j in range(i + 1, len(v))]
    print(f"library: {len(lib)} acquired atoms; alpha-similar same-domain pairs: {len(cand_pairs)}")

    merged, checked = [], 0
    for a, b in cand_pairs:
        d = a["domain"]
        dominfo = data["domains"][d]
        mod = f"Merge{d.capitalize()}{checked}"
        lines = ["{-# OPTIONS --cubical --allow-unsolved-metas #-}", f"module {mod} where",
                 "open import Cubical.Foundations.Prelude"]
        for imp in dominfo["imports"]:
            lines.extend(imp.splitlines())
        lines += [f"mergeProbe : {a['name']} ≡ {b['name']}", "mergeProbe = ?"]
        path = os.path.join(WORK, mod + ".agda")
        open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
        ag = Agda(path)
        ok, holes = ag.load(timeout=180)
        term = ag.auto(0, ["funExt", "isSetℕ", "isProp→PathP", "refl"]) if ok and holes else None
        ag.close()
        checked += 1
        print(f"  {a['name']} ~ {b['name']} [{d}]: "
              f"{'path exhibited: ' + term if term else 'no path found (statement-level only)'}")
        if term:
            merged.append({"a": a["name"], "b": b["name"], "domain": d, "path": term})

    out = {"candidate_pairs": [(a["name"], b["name"], a["domain"]) for a, b in cand_pairs],
           "paths_exhibited": merged,
           "library_before": len(lib),
           "library_after_merge": len(lib) - len(merged)}
    json.dump(out, open(os.path.join(HERE, "results_merge.json"), "w"),
              indent=1, ensure_ascii=False)
    print(f"\ncompression: {len(lib)} -> {len(lib) - len(merged)} atoms "
          f"({len(merged)} up-to-path merges verified)")
    print("wrote agda-gate/results_merge.json")


if __name__ == "__main__":
    main()
