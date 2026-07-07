"""
AGDA GATE step 1 — extract proof goals from cubical library modules, grouped by domain.

Goals are existing lemma statements (prove-from-scratch tasks). Domains = the genre axis
from experiment 05. Heuristic top-level parse; goals are validated later by actually loading
the generated goal files (the typechecker is the arbiter, as everywhere in this gate).
"""
import os, re, json, sys

CUBICAL = os.path.expanduser(os.environ.get("CUBICAL_SRC", "~/agda-libs/cubical"))
HERE = os.path.dirname(os.path.abspath(__file__))

# domain -> (source module for goals, extra imports the goal file needs)
DOMAINS = {
    "path":  ("Cubical/Foundations/GroupoidLaws.agda",
              ["Cubical.Foundations.GroupoidLaws"]),
    "nat":   ("Cubical/Data/Nat/Properties.agda",
              ["Cubical.Data.Nat", "Cubical.Data.Nat.Properties", "Cubical.Data.Sigma"]),
    "int":   ("Cubical/Data/Int/Properties.agda",
              ["Cubical.Data.Int", "Cubical.Data.Int.Properties", "Cubical.Data.Nat"]),
    "bool":  ("Cubical/Data/Bool/Properties.agda",
              ["Cubical.Data.Bool", "Cubical.Data.Bool.Properties",
               "Cubical.Foundations.Isomorphism", "Cubical.Relation.Nullary"]),
    "list":  ("Cubical/Data/List/Properties.agda",
              ["Cubical.Data.List", "Cubical.Data.List.Properties",
               "Cubical.Data.Nat", "Cubical.Data.Sigma"]),
    "sigma": ("Cubical/Data/Sigma/Properties.agda",
              ["Cubical.Data.Sigma", "Cubical.Data.Sigma.Properties",
               "Cubical.Foundations.Isomorphism", "Cubical.Foundations.Equiv",
               "Cubical.Foundations.Function"]),
}

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
        i = j
        if not has_def:
            continue
        if "≡" not in ty:                      # keep equational goals (what composition targets)
            continue
        if len(ty) > 240 or "{-" in ty or "?" in ty:
            continue
        if any(tok in ty for tok in ("Path ", "PathP", "Square", "≃", "Iso")) and "≡" not in ty:
            continue
        out.append({"name": name, "type": ty})
    return out


def main():
    goals, variables = [], {}
    for dom, (rel, imports) in DOMAINS.items():
        path = os.path.join(CUBICAL, rel)
        got = extract(path)
        variables[dom] = extract_variables(open(path, encoding="utf-8").read().splitlines())
        for g in got:
            g["domain"] = dom
        goals.extend(got)
        print(f"{dom:6s}: {len(got):3d} goals from {rel}  (variables: {variables[dom]})")
    json.dump({"domains": {d: {"imports": imps, "variables": variables[d]}
                           for d, (_, imps) in DOMAINS.items()},
               "goals": goals},
              open(os.path.join(HERE, "goals.json"), "w"), indent=1, ensure_ascii=False)
    print(f"total {len(goals)} -> agda-gate/goals.json")


if __name__ == "__main__":
    main()
