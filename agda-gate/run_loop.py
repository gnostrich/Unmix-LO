"""
AGDA GATE steps 2-3 — the library loop on the oracle substrate, and the three gate checks.

The object, translated (see AGDA_RESULTS.md):
  tasks     = lemma statements extracted from cubical modules (goals.json), 6 domains
  library   = a growing set of named atoms (seeded with generic path algebra)
  router    = lexical premise selection (tf-idf token overlap, top-k hints)
  composer  = Agsy (agda's Auto) restricted to the hinted atoms + local context
  critic    = the Agda typechecker (free, exact, sound)
  expansion = a goal the router+composer cannot crack is acquired as ONE new atom
              (its human proof already exists in the imported module — the analog of
              learning it from a trainee) and becomes hintable for later goals

Checks on the DISCRETE library:
  REUSED     - held-out goals proved with atoms acquired from OTHER goals; cross-domain
               firing; the experiment-05 curve (held-out success as domains pool)
  INDIVIDUAL - sparse atoms/proof; no atom smeared across everything
  STABLE     - same library emerges under different goal orderings (Jaccard / rank corr)

Run:  python agda-gate/run_loop.py [--slice N] [--seeds 3]
"""
import os, re, json, time, argparse, random, subprocess, math, functools
import threading, queue
print = functools.partial(print, flush=True)
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("AGDA_WORK", os.path.join(HERE, "work"))
AUTO_TIMEOUT = int(os.environ.get("AGDA_AUTO_T", 4))     # Agsy per-attempt seconds
TOP_K = 8                                                # hints per attempt
SEED_ATOMS = ["sym", "cong", "_∙_", "transport", "subst", "funExt"]
HELD_FRAC = 0.25

PRELUDE_IMPORTS = ["Cubical.Foundations.Prelude"]


# ---------------------------------------------------------------- agda interaction client
class Agda:
    def __init__(self, path):
        self.path = path
        self.p = subprocess.Popen(["agda", "--interaction"], cwd=os.path.dirname(path),
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True,
                                  env={**os.environ, "LC_ALL": "C.UTF-8"})
        # NB: never select() on a buffered text stream — readline() slurps ahead of the fd
        # and select then blocks on lines already sitting in the buffer. Pump via a thread.
        self.q = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for line in self.p.stdout:
            self.q.put(line)

    def _cmd(self, s, timeout):
        self.p.stdin.write(f'IOTCM "{self.path}" NonInteractive Indirect ({s})\n')
        self.p.stdin.flush()
        buf, t0 = "", time.time()
        while time.time() - t0 < timeout:
            try:
                line = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            buf += line
            if ("agda2-give-action" in line or "*Error*" in line
                    or '"*Auto*"' in line or "agda2-goals-action" in line):
                try:            # drain trailing actions so they don't pollute the next command
                    while True:
                        buf += self.q.get(timeout=0.15)
                except queue.Empty:
                    pass
                break
        return buf

    def load(self, timeout=600):
        out = self._cmd(f'Cmd_load "{self.path}" []', timeout)
        if "*Error*" in out:
            m = re.search(r'agda2-info-action "\*Error\*" "((?:[^"\\]|\\.)*)"', out)
            return False, (m.group(1).replace("\\n", "\n") if m else out[-2000:])
        holes = re.findall(r"agda2-goals-action '\(([\d\s]*)\)", out)
        return True, [int(h) for h in holes[-1].split()] if holes else []

    def auto(self, hole, hints):
        out = self._cmd(f'Cmd_autoOne {hole} noRange "-t {AUTO_TIMEOUT} {" ".join(hints)}"',
                        AUTO_TIMEOUT * 3 + 20)
        m = re.search(rf'agda2-give-action {hole} "((?:[^"\\]|\\.)*)"', out)
        return m.group(1) if m else None

    def close(self):
        try:
            self.p.stdin.close(); self.p.terminate()
        except Exception:
            pass


# ---------------------------------------------------------------- goal files
def goal_file(domain, imports, variables, goals):
    mod = f"Goals{domain.capitalize()}"
    lines = ["{-# OPTIONS --cubical --allow-unsolved-metas #-}", f"module {mod} where"]
    for imp in PRELUDE_IMPORTS + imports:
        lines.append(f"open import {imp}")
    if variables:
        lines.append("private")
        lines.append(" variable")
        for v in variables:
            lines.append(f"  {v}")
    header_len = len(lines)
    for i, g in enumerate(goals):
        lines.append(f"g{i} : {g['type']}")
        lines.append(f"g{i} = ?")
    path = os.path.join(WORK, mod + ".agda")
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    return path, header_len


def validated_load(domain, dominfo, goals, log=print):
    """Load the goal file; drop goals the typechecker rejects (scope/type errors) until clean."""
    goals = list(goals)
    imports, variables = dominfo["imports"], dominfo.get("variables", [])
    while goals:
        path, header_len = goal_file(domain, imports, variables, goals)
        a = Agda(path)
        t0 = time.time()
        ok, res = a.load()
        if ok:
            assert len(res) == len(goals), (len(res), len(goals))
            return a, goals
        a.close()
        m = re.search(r"\.agda:(\d+)", res)
        if not m:
            raise RuntimeError(f"{domain}: unparseable load error:\n{res[:800]}")
        bad_line = int(m.group(1))
        idx = (bad_line - header_len - 1) // 2
        idx = max(0, min(idx, len(goals) - 1))
        log(f"    [{domain}] drop {goals[idx]['name']} ({time.time()-t0:.0f}s load): "
            f"{res.splitlines()[1][:80] if len(res.splitlines())>1 else res[:80]}")
        goals.pop(idx)
    return None, []


# ---------------------------------------------------------------- router (premise selection)
TOKEN = re.compile(r"[^\s(){}→∀λ:.;,]+")


def tokens(s):
    return set(t for t in TOKEN.findall(s) if len(t) > 1 or not t.isascii())


def make_router(all_atom_types):
    df = defaultdict(int)
    for ty in all_atom_types:
        for t in tokens(ty):
            df[t] += 1

    def score(goal_ty, atom_name, atom_ty):
        gt, at = tokens(goal_ty), tokens(atom_ty) | {atom_name}
        return sum(1.0 / (1 + df[t]) for t in gt & at)
    return score


def pick_hints(goal, library, score, k=TOP_K):
    cands = [(score(goal["type"], a["name"], a["type"]), a["name"]) for a in library
             if a["name"] != goal["name"]]          # never hint the goal's own name
    cands.sort(reverse=True)
    return [n for s, n in cands[:k] if s > 0] or [n for _, n in cands[:3]]


def atoms_used(term, hints):
    words = set(re.findall(r"[^\s()λ→.]+", term)) | ({"_∙_"} if "∙" in term else set())
    return [h for h in hints if h in words]


# ---------------------------------------------------------------- live checkpointing
def checkpoint(**kw):
    """Feeler for the humans: agda-gate/progress.json reflects live state at every phase."""
    kw["at"] = time.strftime("%H:%M:%S")
    json.dump(kw, open(os.path.join(HERE, "progress.json"), "w"),
              indent=1, ensure_ascii=False, default=str)


# ---------------------------------------------------------------- the loop
def run_pass(data, seed, slice_n=None, log=print):
    rng = random.Random(seed)
    domains = list(data["domains"])
    by_dom = {d: [g for g in data["goals"] if g["domain"] == d] for d in domains}
    if slice_n:
        for d in domains:
            by_dom[d] = by_dom[d][:slice_n]

    # split: last HELD_FRAC per domain held out (fixed across seeds; ordering only shuffles train)
    train, held = {}, {}
    for d in domains:
        gs = by_dom[d]
        n_h = max(1, int(len(gs) * HELD_FRAC))
        train[d], held[d] = gs[:-n_h], gs[-n_h:]

    sessions, hole_of = {}, {}
    for d in domains:
        gs = train[d] + held[d]
        a, kept = validated_load(d, data["domains"][d], gs, log=log)
        kept_names = {g["name"] for g in kept}
        train[d] = [g for g in train[d] if g["name"] in kept_names]
        held[d] = [g for g in held[d] if g["name"] in kept_names]
        sessions[d] = a
        hole_of[d] = {g["name"]: i for i, g in enumerate(kept)}
        log(f"  [{d}] {len(train[d])} train + {len(held[d])} held goals loaded")
        checkpoint(seed=seed, phase="loading", domain=d,
                   loaded={dd: len(train.get(dd, [])) + len(held.get(dd, []))
                           for dd in domains if dd in sessions or dd == d})

    library = [{"name": n, "type": n, "domain": "seed", "src": "seed"} for n in SEED_ATOMS]
    type_of_atom = {}   # for router scoring: seeds score by name only
    events = []

    # sequential pass over shuffled train goals (domain interleaved = the pooled regime)
    order = [(d, g) for d in domains for g in train[d]]
    rng.shuffle(order)
    score = make_router([a["type"] for a in library])
    for step, (d, g) in enumerate(order):
        score = make_router([a["type"] for a in library])
        hints = pick_hints(g, library, score)
        term = sessions[d].auto(hole_of[d][g["name"]], hints)
        if term:
            used = atoms_used(term, hints)
            events.append({"phase": "train", "step": step, "domain": d, "goal": g["name"],
                           "ok": True, "used": used,
                           "used_domains": [next(a["domain"] for a in library
                                                 if a["name"] == u) for u in used]})
        else:
            library.append({"name": g["name"], "type": g["type"], "domain": d, "src": "acquired"})
            events.append({"phase": "train", "step": step, "domain": d, "goal": g["name"],
                           "ok": False, "used": []})
        if step % 5 == 0 or step == len(order) - 1:
            n_ok_so_far = sum(e["ok"] for e in events)
            checkpoint(seed=seed, phase="train", step=f"{step+1}/{len(order)}",
                       composed=n_ok_so_far, library=len(library),
                       last={"goal": g["name"], "domain": d, "ok": events[-1]["ok"],
                             "used": events[-1]["used"]})
    n_ok = sum(e["ok"] for e in events)
    log(f"  train pass: {n_ok}/{len(order)} composed from library; "
        f"library grew to {len(library)} atoms ({len(library)-len(SEED_ATOMS)} acquired)")

    # held-out evaluation with the final library
    # ceiling = same composer, ORACLE hints (atoms the human proof used): separates
    # router failure from composer floor. ok requires only router hints; ceiling_ok
    # is the validity bound on what Agsy could ever deliver here.
    held_ev = []
    for d in domains:
        for g in held[d]:
            hints = pick_hints(g, library, score)
            term = sessions[d].auto(hole_of[d][g["name"]], hints)
            used = atoms_used(term, hints) if term else []
            oracle = [h for h in g.get("proof_atoms", []) if h != g["name"]][:TOP_K]
            ceil_term = term or (sessions[d].auto(hole_of[d][g["name"]], oracle)
                                 if oracle else None)
            held_ev.append({"domain": d, "goal": g["name"], "ok": bool(term), "used": used,
                            "ceiling_ok": bool(ceil_term),
                            "used_domains": [next(a["domain"] for a in library
                                                  if a["name"] == u) for u in used]})
        checkpoint(seed=seed, phase="held-out", done_domain=d,
                   held_ok=sum(e["ok"] for e in held_ev),
                   ceiling_ok=sum(e["ceiling_ok"] for e in held_ev), held_n=len(held_ev))

    # diversity curve: held-out success using only atoms acquired from first n domains pooled
    # (seed 0 only — the curve is a property of the pooling, not the ordering)
    curve = {}
    dom_order = domains[:]
    for nd in (range(1, len(domains) + 1) if seed == 0 else []):
        allowed = set(SEED_ATOMS) | {a["name"] for a in library
                                     if a["domain"] in dom_order[:nd]}
        sub = [a for a in library if a["name"] in allowed]
        ok = 0; tot = 0
        for d in domains:
            for g in held[d]:
                hints = pick_hints(g, sub, score)
                term = sessions[d].auto(hole_of[d][g["name"]], hints)
                ok += bool(term); tot += 1
        curve[nd] = round(ok / max(tot, 1), 3)
        log(f"  diversity: library from {nd} domain(s) -> held-out success {curve[nd]}")
        checkpoint(seed=seed, phase="diversity-curve", curve=curve)

    for a in sessions.values():
        a.close()
    return {"seed": seed, "events": events, "held": held_ev, "curve": curve,
            "library": library,
            "train_counts": {d: len(train[d]) for d in domains},
            "held_counts": {d: len(held[d]) for d in domains}}


# ---------------------------------------------------------------- metrics + verdict
def analyze(passes):
    P0 = passes[0]
    held = P0["held"]
    n_held = len(held)
    ok = [e for e in held if e["ok"]]
    reuse = [e for e in ok if any(u not in SEED_ATOMS for u in e["used"])]
    xdom = [e for e in ok if any(ud not in ("seed", e["domain"]) for ud in e["used_domains"])]
    usage = defaultdict(int)
    for e in P0["events"] + held:
        for u in e["used"]:
            usage[u] += 1
    n_proofs = sum(e["ok"] for e in P0["events"]) + len(ok)
    smeared = [u for u, c in usage.items() if c > 0.8 * max(n_proofs, 1)]
    atoms_per = [len(e["used"]) for e in P0["events"] + held if e["ok"]]

    libs = [set(a["name"] for a in p["library"] if a["src"] == "acquired") for p in passes]
    jac = []
    for i in range(len(libs)):
        for j in range(i + 1, len(libs)):
            inter, union = len(libs[i] & libs[j]), len(libs[i] | libs[j])
            jac.append(inter / union if union else 1.0)

    res = {
        "held_total": n_held,
        "composability_ceiling": round(sum(e.get("ceiling_ok", False) for e in held)
                                       / max(n_held, 1), 3),
        "held_success": round(len(ok) / max(n_held, 1), 3),
        "held_reuse_rate": round(len(reuse) / max(n_held, 1), 3),
        "reuse_given_success": round(len(reuse) / max(len(ok), 1), 3),
        "crossdomain_rate_given_success": round(len(xdom) / max(len(ok), 1), 3),
        "mean_atoms_per_proof": round(sum(atoms_per) / max(len(atoms_per), 1), 2),
        "n_smeared_atoms": len(smeared),
        "top_atoms": sorted(usage.items(), key=lambda kv: -kv[1])[:12],
        "library_size": len(P0["library"]),
        "stability_jaccard": round(sum(jac) / len(jac), 3) if jac else None,
        "diversity_curve": P0["curve"],
    }
    res["pass"] = {
        "reused": res["held_reuse_rate"] > 0.2 and res["crossdomain_rate_given_success"] > 0.1,
        "individual": res["n_smeared_atoms"] <= 1 and res["mean_atoms_per_proof"] <= TOP_K,
        "stable": (res["stability_jaccard"] or 0) > 0.6,
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", type=int, default=None, help="limit goals per domain (smoke)")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    if not os.path.exists(os.path.join(WORK, "agda-work.agda-lib")):
        open(os.path.join(WORK, "agda-work.agda-lib"), "w").write(
            "name: agda-gate-work\ndepend: cubical\ninclude: .\nflags: --cubical\n")
    data = json.load(open(os.path.join(HERE, "goals.json")))
    passes = []
    for s in range(args.seeds):
        print(f"=== pass seed={s} ===")
        t0 = time.time()
        passes.append(run_pass(data, seed=s, slice_n=args.slice))
        print(f"  ({time.time()-t0:.0f}s)")
    res = analyze(passes)
    out = {"config": {"auto_timeout": AUTO_TIMEOUT, "top_k": TOP_K, "seeds": args.seeds,
                      "slice": args.slice, "seed_atoms": SEED_ATOMS},
           "metrics": res,
           "passes": [{k: v for k, v in p.items() if k != "library"} for p in passes],
           "final_library": passes[0]["library"]}
    json.dump(out, open(os.path.join(HERE, "results_agda.json"), "w"),
              indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print("wrote agda-gate/results_agda.json")


if __name__ == "__main__":
    main()
