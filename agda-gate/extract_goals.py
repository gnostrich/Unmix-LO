"""
AGDA GATE step 1 — extract proof goals from cubical library modules, grouped by domain.

Goals are existing lemma statements (prove-from-scratch tasks). Domains = the genre axis
from experiment 05. Heuristic top-level parse; goals are validated later by actually loading
the generated goal files (the typechecker is the arbiter, as everywhere in this gate).
"""
import os, re, json, sys

CUBICAL = os.path.expanduser(os.environ.get("CUBICAL_SRC", "~/agda-libs/cubical"))
HERE = os.path.dirname(os.path.abspath(__file__))

# domain -> source module for goals (goal files mirror the module's own imports, so
# statements scope-check in exactly the context they were written in)
DOMAINS = {
    "path":  "Cubical/Foundations/GroupoidLaws.agda",
    "nat":   "Cubical/Data/Nat/Properties.agda",
    "int":   "Cubical/Data/Int/Properties.agda",
    "bool":  "Cubical/Data/Bool/Properties.agda",
    "list":  "Cubical/Data/List/Properties.agda",
    "sigma": "Cubical/Data/Sigma/Properties.agda",
}


def extract_imports(lines):
    """The module's own `open import` lines, verbatim (with indented continuations)."""
    out, i = [], 0
    while i < len(lines):
        if lines[i].startswith("open import ") or lines[i].startswith("import "):
            block = lines[i]; i += 1
            while i < len(lines) and lines[i].strip() and lines[i][0].isspace():
                block += "\n" + lines[i]; i += 1
            out.append(block)
        else:
            i += 1
    return out

DECL = re.compile(r"^([^\s(){};@\"]+)\s*:\s*(.*)$")
SKIP_NAME = re.compile(r"^(--|module|open|import|record|data|where|private|infix|instance|"
                       r"postulate|primitive|variable|mutual|abstract|\{-|syntax|pattern)")


def extract_variables(lines):
    """Collect the module's `variable` block(s) — goal statements reference these
    generalizable names free, so the goal file must redeclare them verbatim."""
    decls = []
    i = 0
    while i < len(lines):
        if lines[i].strip() == "variable":
            base = len(lines[i]) - len(lines[i].lstrip())
            i += 1
            while i < len(lines) and (not lines[i].strip()
                                      or (len(lines[i]) - len(lines[i].lstrip())) > base):
                if lines[i].strip() and not lines[i].strip().startswith("--"):
                    decls.append(lines[i].strip())
                i += 1
        else:
            i += 1
    return decls


def extract(path):
    lines = open(path, encoding="utf-8").read().splitlines()
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        if not line or line[0].isspace() or SKIP_NAME.match(line.strip()):
            i += 1; continue
        m = DECL.match(line)
        if not m:
            i += 1; continue
        name, ty = m.group(1), m.group(2).strip()
        j = i + 1
        while j < len(lines) and lines[j].startswith(" ") and lines[j].strip():
            ty += " " + lines[j].strip(); j += 1
        # definition must follow (same name at column 0) => it's a proven lemma, not a field
        has_def = j < len(lines) and re.match(rf"^{re.escape(name)}[\s(]", lines[j])
        if not has_def:
            i = j
            continue
        # capture the proof body (all clauses until the next column-0 declaration) — used for
        # the composability-ceiling control (oracle hints = atoms the human proof references)
        body = []
        while j < len(lines) and (not lines[j] or lines[j][0].isspace()
                                  or re.match(rf"^{re.escape(name)}[\s(]", lines[j])):
            body.append(lines[j]); j += 1
        i = j
        if "≡" not in ty:                      # keep equational goals (what composition targets)
            continue
        if len(ty) > 240 or "{-" in ty or "?" in ty:
            continue
        if any(tok in ty for tok in ("Path ", "PathP", "Square", "≃", "Iso")) and "≡" not in ty:
            continue
        out.append({"name": name, "type": ty, "body": " ".join(l.strip() for l in body)})
    # oracle hints: body identifiers that are themselves lemmas declared in this module
    decl_names = {g["name"] for g in out}
    for g in out:
        toks = set(re.findall(r"[^\s(){};.]+", g.pop("body")))
        g["proof_atoms"] = sorted((toks & decl_names) - {g["name"]})
    return out


def main():
    goals, dominfo = [], {}
    for dom, rel in DOMAINS.items():
        path = os.path.join(CUBICAL, rel)
        lines = open(path, encoding="utf-8").read().splitlines()
        got = extract(path)
        mod = rel[:-5].replace("/", ".")
        dominfo[dom] = {"module": mod,
                        "imports": extract_imports(lines) + [f"open import {mod}"],
                        "variables": extract_variables(lines)}
        for g in got:
            g["domain"] = dom
        goals.extend(got)
        print(f"{dom:6s}: {len(got):3d} goals from {rel}  ({len(dominfo[dom]['imports'])} imports mirrored)")
    json.dump({"domains": dominfo, "goals": goals},
              open(os.path.join(HERE, "goals.json"), "w"), indent=1, ensure_ascii=False)
    print(f"total {len(goals)} -> agda-gate/goals.json")


if __name__ == "__main__":
    main()
