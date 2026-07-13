"""
experiments/overlap_routing.py — collective cross-modal routing through modality OVERLAP.

AUDIT CLASSIFICATION: experimental CONTROL / thesis test (the architecture's crux). This file introduces NO
new routing mechanism, NO new QC, NO shim. Routing is the EXISTING untied `energy.functional.equilibrate`
(block-coordinate F descent); each member aligns its own probe-geometry to the shared anchor INDEPENDENTLY
(no tie forcing member A's event-i onto member B's atom). The only additions are transparent SENSORS for the
world member (a coarse render descriptor, a bag-of-words caption multi-hot, the dynamics vector) so the world
member becomes a ground-truth vision<->text<->dynamics bridge like CLIP's/SigLIP's two towers. It is the direct
executable of ebr/PREREG_overlap_routing.md (committed at HEAD 1ae11c2): substrate, conditions A/B, task,
metric, chance floors and verdict clauses are taken from that file and NOT re-tuned after seeing numbers.

THE THESIS (prereg): cross-modal transfer should emerge from diverse members with partial modality OVERLAP,
routed collectively by F — not from a hand-paired dictionary. A = thin overlap (CLIP the only bridge); B =
rich overlap (A + SigLIP + world member). Prediction: B recovers transfer where thin-A is fragile.

DEFENDED CONSTANTS (each traceable to the prereg; no undefended magic, no post-hoc tuning):
    N = 256      : shared reference set = world scenes seeds 0..255 (prereg "N=256 world scenes, seeds fixed").
    K = 24       : held-out fresh scenes, seeds 1000..1023 (prereg "K=24 fresh scenes, unseen seeds").
    M = 14       : anchor budget (prereg "Anchor budget m=14"); fixed for A and B alike.
    EPS = 0.05   : GW inner tolerance = engine.equilibrate default (the standing operating point; reused verbatim).
    TAU = 1.0    : anchor-KL weight = equilibrate default (reused verbatim; not tuned here).
    SEEDS=[0,1,2]: routing-solver seed set (prereg "Seed set {0,1,2} for the routing solver; scenes fixed").
    CHANCE floors 0.25 / 0.33 / 0.40 : registered floors for color(4) / shape(3) / motion(phrase set) matches.
    scramble seed 12345, member minilm_caption : the R3 gauge check uses a SILENT member (uniform w, so w is
        independent of L) — isolating that cloud_to_Dw's (D) is gauge-invariant, hence F must be unchanged.

VERDICT — PASS iff ALL (prereg §Verdict, to the letter):
    (1) B color >= 0.50 AND B shape >= 0.45   (above the 0.25 / 0.33 floors), std over seeds <= 0.15.
    (2) B_color - A_color >= 0.15             (the added overlap, not CLIP alone, carries it).
    (3) |dF| < 1e-6 under scramble             (R3 survives untied).
If (1) holds but (2) fails -> "CLIP overlap already sufficed; world/SigLIP redundant" (honest partial).
If (1) fails -> "collective untied routing does NOT recover transfer; the dictionary stays load-bearing"
(honest negative — the wall bites again; reported straight, no patching, no substrate retuning).

DISCIPLINE: NO shim, NO force-pass, NO cherry-picking, NO post-hoc tuning. If a clause FAILS it is reported
with the actual numbers. An honest negative here is a real, publishable result about how much overlap
collective routing needs.

Run: python -m ebr.experiments.overlap_routing   (CPU + model loads; encoding 256 renders through 4 vision
models + captions through 3 text models takes several minutes; encodings are cached so A and B reuse them.)
"""
import os
import numpy as np
from PIL import Image

from ..demo.ports import load_ports
from ..demo import world as W
from ..demo.library import _reweight
from ..geometry.clouds import cloud_to_Dw, scramble
from ..energy.functional import equilibrate

# ------------------------------------------------------------------- defended constants (all from the prereg)
N = 256
K = 24
M = 14
EPS = 0.05
TAU = 1.0
SEEDS = [0, 1, 2]
SUBSTRATE_SEEDS = list(range(N))            # scenes 0..255
QUERY_SEEDS = list(range(1000, 1000 + K))   # fresh held-out scenes 1000..1023
CHANCE = {"color": 0.25, "shape": 0.33, "motion": 0.40}
SCRAMBLE_SEED = 12345
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_overlap_cache")

# world vocabulary for the transparent caption sensor (a multi-hot bag-of-words, ground-truth consistent)
VOCAB = list(W.COLOR_NAMES) + list(W.SHAPES) + ["moving right", "moving left", "falling", "rising", "nearly still"]


# ------------------------------------------------------------------- transparent world sensors (the bridge)
def render_sensor(render):
    """Coarse visual descriptor: 64x64x3 render block-averaged to 8x8x3, flattened (192-d). Overlaps the
    pixel-vision members without any learned weights (ground-truth-consistent with the same scene)."""
    x = np.asarray(render, np.float64).reshape(8, 8, 8, 8, 3).mean(axis=(1, 3))  # 8x8 block means
    return x.reshape(-1) / 255.0


def caption_sensor(caption):
    """Bag-of-words multi-hot over the world vocabulary (color/shape/motion words present in the caption).
    Overlaps the text members; ground-truth-consistent because the caption is generated from the scene."""
    return np.array([1.0 if v in caption else 0.0 for v in VOCAB])


def dynamics_sensor(dyn):
    """The world.event dynamics vector — a THIRD modality nobody else speaks (always a silent reference cloud)."""
    return np.asarray(dyn, np.float64)


# ------------------------------------------------------------------- member roster (member-CHANNELS)
# (name, port_name, channel, modality, input_kind).  A = first 5 (CLIP the only bridge); B = all 10.
MEMBERS = [
    ("vit_render",       "vit",       0,    "vision",   "render"),
    ("mobilenet_render", "mobilenet", 0,    "vision",   "render"),
    ("clip_vision",      "clip",      0,    "vision",   "render"),
    ("minilm_caption",   "minilm",    0,    "text",     "caption"),
    ("clip_text",        "clip",      1,    "text",     "caption"),
    ("siglip_vision",    "siglip",    0,    "vision",   "render"),
    ("siglip_text",      "siglip",    1,    "text",     "caption"),
    ("world_render",     None,        None, "vision",   "world_render"),
    ("world_caption",    None,        None, "text",     "world_caption"),
    ("world_dynamics",   None,        None, "dynamics", "world_dynamics"),
]
COND = {"A": [m[0] for m in MEMBERS[:5]], "B": [m[0] for m in MEMBERS]}
MSPEC = {m[0]: m for m in MEMBERS}


# ------------------------------------------------------------------- encoding (cached; A and B reuse)
def _encode_member(name, ports, scenes):
    """Encode one member-channel over a list of world-event dicts -> L (len x dim)."""
    _, port_name, ch, mod, kind = MSPEC[name]
    if kind == "render":
        imgs = [Image.fromarray(e["render"]) for e in scenes]
        return ports[port_name].encode(imgs, channel=ch)
    if kind == "caption":
        return ports[port_name].encode([e["caption"] for e in scenes], channel=ch)
    if kind == "world_render":
        return np.stack([render_sensor(e["render"]) for e in scenes])
    if kind == "world_caption":
        return np.stack([caption_sensor(e["caption"]) for e in scenes])
    if kind == "world_dynamics":
        return np.stack([dynamics_sensor(e["dynamics"]) for e in scenes])
    raise ValueError(kind)


def build_encodings(ports):
    """Return Ls (member -> substrate L, 256 x dim, cached to disk) and query features:
    qf_render (member -> K x dim, vision members over query RENDERS) and
    qf_caption (member -> K x dim, text members over query CAPTIONS)."""
    os.makedirs(CACHE, exist_ok=True)
    sub = [W.event(s) for s in SUBSTRATE_SEEDS]
    qs = [W.event(s) for s in QUERY_SEEDS]
    Ls, qf_render, qf_caption = {}, {}, {}
    for name, _pn, _ch, mod, _kind in MEMBERS:
        p = os.path.join(CACHE, f"{name}__L.npy")
        if os.path.exists(p):
            Ls[name] = np.load(p)
        else:
            print(f"  [encode] {name} over {N} scenes ...", flush=True)
            Ls[name] = _encode_member(name, ports, sub)
            np.save(p, Ls[name])
        # query features: vision members encode the query RENDERS; text members encode the query CAPTIONS.
        if mod == "vision":
            qf_render[name] = _encode_member(name, ports, qs)
        if mod == "text":
            qf_caption[name] = _encode_member(name, ports, qs)
    return Ls, qf_render, qf_caption, qs


# ------------------------------------------------------------------- routing (existing UNTIED equilibrate)
def _De0(seed, m=M):
    """Random symmetric zero-diag anchor cost, median-normalized (as in demo/readout.py)."""
    rng = np.random.default_rng(seed)
    De = rng.random((m, m)); De = (De + De.T) / 2; np.fill_diagonal(De, 0.0)
    med = np.median(De[np.triu_indices(m, 1)])
    return De / (med if med > 0 else 1.0)


def route(members, Ds, active_names, active_feats, seed):
    """Build ws (active -> _reweight(L, f_query); silent -> uniform), then UNTIED equilibrate. Returns
    (pis, De, a, F). Ds/Ls indexed by member name; order follows `members`."""
    ws = []
    for name in members:
        n = Ds[name].shape[0]
        if name in active_names:
            ws.append(_reweight(active_feats[name][0], active_feats[name][1]))
        else:
            ws.append(np.full(n, 1.0 / n))
    Dlist = [Ds[name] for name in members]
    a0 = np.full(M, 1.0 / M)
    pis, De, a, ftrace, _conv = equilibrate(Dlist, ws, _De0(seed), a0, a0.copy(), eps=EPS, tau=TAU)
    return pis, De, a, ws, ftrace[-1]


# ------------------------------------------------------------------- attribute scoring vs ground truth
def _motion_head(obj):
    """First motion component of an object's phrase (e.g. 'moving right' / 'falling' / 'nearly still')."""
    return W._motion_words(obj["vx"], obj["vy"]).split(" and ")[0]


def score_caption(query_obj, answer_caption):
    """Forward: query's PRIMARY-object attributes vs the answer CAPTION (substring on the world vocabulary)."""
    return {
        "color": float(query_obj["color"] in answer_caption),
        "shape": float(query_obj["shape"] in answer_caption),
        "motion": float(_motion_head(query_obj) in answer_caption),
    }


def score_scene(query_obj, probe_obj):
    """Reverse: query's PRIMARY-object attributes vs the retrieved render's ground-truth PRIMARY object."""
    return {
        "color": float(query_obj["color"] == probe_obj["color"]),
        "shape": float(query_obj["shape"] == probe_obj["shape"]),
        "motion": float(_motion_head(query_obj) == _motion_head(probe_obj)),
    }


# ------------------------------------------------------------------- one direction of one condition
def run_direction(cond, direction, Ds, Ls, qf_render, qf_caption, qs, sub_scenes):
    """direction 'fwd' = feed render, vision active, read silent TEXT members' top caption.
                 'rev' = feed caption, text active, read silent VISION members' top render ground truth.
    Returns per-seed accuracies {attr: [acc_seed0, acc_seed1, acc_seed2]} averaged over queries x read-members,
    plus the overall mean {attr: acc}."""
    members = COND[cond]
    if direction == "fwd":
        active_mod, read_mod, qf = "vision", "text", qf_render
    else:
        active_mod, read_mod, qf = "text", "vision", qf_caption
    active_names = [m for m in members if MSPEC[m][3] == active_mod]
    read_names = [m for m in members if MSPEC[m][3] == read_mod]

    per_seed = {a: [] for a in CHANCE}
    for seed in SEEDS:
        hits = {a: [] for a in CHANCE}
        for k, qscene in enumerate(qs):
            qobj = qscene["objs"][0]                       # PRIMARY object ground truth
            active_feats = {name: (Ls[name], qf[name][k]) for name in active_names}
            pis, De, a, ws, _F = route(members, Ds, active_names, active_feats, seed)
            idx_of = {name: i for i, name in enumerate(members)}
            for rname in read_names:
                re = pis[idx_of[rname]] @ a                # relevance over the 256 probes
                top = int(np.argmax(re))
                if direction == "fwd":
                    s = score_caption(qobj, sub_scenes[top]["caption"])
                else:
                    s = score_scene(qobj, sub_scenes[top]["objs"][0])
                for at in CHANCE:
                    hits[at].append(s[at])
        for at in CHANCE:
            per_seed[at].append(float(np.mean(hits[at])))
    overall = {at: float(np.mean(per_seed[at])) for at in CHANCE}
    std = {at: float(np.std(per_seed[at])) for at in CHANCE}
    return {"per_seed": per_seed, "overall": overall, "std": std,
            "read_members": read_names, "active_members": active_names}


# ------------------------------------------------------------------- gauge check (R3 must survive untied)
def gauge_check(Ds, Ls, qf_render, qs):
    """For ONE query (k=0, condition B, seed 0): scramble a SILENT member's feature matrix (uniform w, so w is
    independent of L), rebuild its D, re-equilibrate, and confirm |dF| < 1e-6. Scrambling a silent member
    isolates the R3 claim: cloud_to_Dw's D is invariant to the G0 gauge group, so F must not move."""
    members = COND["B"]
    active_names = [m for m in members if MSPEC[m][3] == "vision"]
    active_feats = {name: (Ls[name], qf_render[name][0]) for name in active_names}
    _pis, _De, _a, _ws, F0 = route(members, Ds, active_names, active_feats, seed=0)
    # scramble one SILENT member's features -> new D; keep everything else identical.
    victim = "minilm_caption"
    Ds2 = dict(Ds)
    Ds2[victim], _ = cloud_to_Dw(scramble(Ls[victim], seed=SCRAMBLE_SEED))
    _pis2, _De2, _a2, _ws2, F1 = route(members, Ds2, active_names, active_feats, seed=0)
    return abs(F1 - F0), F0, F1, victim


# ------------------------------------------------------------------- report
def _fmt_row(label, acc):
    return (f"    {label:<26} color {acc['color']:.3f}   shape {acc['shape']:.3f}   motion {acc['motion']:.3f}")


def run():
    print("=" * 96)
    print("OVERLAP-ROUTING — collective cross-modal transfer through modality OVERLAP (thesis crux test)")
    print(f"  substrate N={N} scenes (0..{N-1}); held-out K={K} (1000..{1000+K-1}); anchor m={M}; "
          f"eps={EPS}; solver seeds={SEEDS}")
    print("  routing = EXISTING untied energy.equilibrate (no new mechanism). A=thin (CLIP only bridge); "
          "B=rich (+SigLIP +world).")
    print("=" * 96)

    ports = load_ports()
    print("[encode] building/loading member-channel encodings (cached; A and B reuse) ...", flush=True)
    Ls, qf_render, qf_caption, qs = build_encodings(ports)
    Ds = {name: cloud_to_Dw(Ls[name])[0] for name in Ls}   # D per member, computed once, reused everywhere
    sub_scenes = [W.event(s) for s in SUBSTRATE_SEEDS]
    for c in ("A", "B"):
        print(f"  condition {c}: {len(COND[c])} member-channels -> {COND[c]}")

    results = {}
    for cond in ("A", "B"):
        for direction in ("fwd", "rev"):
            print(f"[route] condition {cond} {direction} "
                  f"({K} queries x {len(SEEDS)} seeds x read-members) ...", flush=True)
            results[(cond, direction)] = run_direction(cond, direction, Ds, Ls, qf_render, qf_caption,
                                                        qs, sub_scenes)

    print("[gauge] scramble one silent member, re-equilibrate ...", flush=True)
    dF, F0, F1, victim = gauge_check(Ds, Ls, qf_render, qs)

    # ------------------------------------------------------------- accuracy table
    print("\n" + "=" * 96)
    print("ACCURACY TABLE  (seed-averaged over queries x read-members; chance floors in brackets)")
    print(f"    chance floors:  color {CHANCE['color']:.2f}   shape {CHANCE['shape']:.2f}   "
          f"motion {CHANCE['motion']:.2f}")
    print("=" * 96)
    dir_name = {"fwd": "FORWARD  render->text ", "rev": "REVERSE  caption->vis "}
    for direction in ("fwd", "rev"):
        print(f"  [{dir_name[direction].strip()}]  read members: "
              f"{results[('B', direction)]['read_members']}")
        for cond in ("A", "B"):
            r = results[(cond, direction)]
            std = r["std"]
            print(_fmt_row(f"cond {cond}", r["overall"]) +
                  f"   (std/seed  c{std['color']:.3f} s{std['shape']:.3f} m{std['motion']:.3f})")
        print()

    # ------------------------------------------------------------- A vs B deltas
    print("=" * 96)
    print("A -> B DELTAS  (rich overlap minus thin overlap)")
    print("=" * 96)
    for direction in ("fwd", "rev"):
        A = results[("A", direction)]["overall"]
        B = results[("B", direction)]["overall"]
        print(f"  [{dir_name[direction].strip()}]  dcolor {B['color']-A['color']:+.3f}   "
              f"dshape {B['shape']-A['shape']:+.3f}   dmotion {B['motion']-A['motion']:+.3f}")

    # ------------------------------------------------------------- gauge number
    print("\n" + "=" * 96)
    print("GAUGE CHECK (R3 under untied routing)")
    print("=" * 96)
    print(f"  scramble member '{victim}' (silent, uniform w):  F0={F0:.9f}  F1={F1:.9f}  |dF|={dF:.3e}")

    # ------------------------------------------------------------- verdict clauses (prereg, to the letter)
    Bf = results[("B", "fwd")]
    Af = results[("A", "fwd")]
    b_color, b_shape = Bf["overall"]["color"], Bf["overall"]["shape"]
    a_color = Af["overall"]["color"]
    std_color, std_shape = Bf["std"]["color"], Bf["std"]["shape"]
    std_ok = (std_color <= 0.15 and std_shape <= 0.15)

    c1 = (b_color >= 0.50 and b_shape >= 0.45 and std_ok)
    c2 = (b_color - a_color >= 0.15)
    c3 = (dF < 1e-6)
    overall = c1 and c2 and c3

    print("\n" + "=" * 96)
    print("VERDICT — prereg clauses (FORWARD render->text is the registered primary direction)")
    print("=" * 96)
    print(f"  (1) B color>=0.50 AND shape>=0.45, std/seed<=0.15 :  "
          f"color={b_color:.3f} shape={b_shape:.3f} "
          f"std(c={std_color:.3f},s={std_shape:.3f})   {'PASS' if c1 else 'FAIL'}")
    print(f"  (2) B_color - A_color >= 0.15                     :  "
          f"{b_color:.3f} - {a_color:.3f} = {b_color-a_color:+.3f}                {'PASS' if c2 else 'FAIL'}")
    print(f"  (3) |dF| < 1e-6 under scramble                    :  "
          f"|dF|={dF:.3e}                          {'PASS' if c3 else 'FAIL'}")
    print("-" * 96)
    print(f"  OVERALL: {'PASS' if overall else 'FAIL'}  "
          f"(c1={'P' if c1 else 'F'}, c2={'P' if c2 else 'F'}, c3={'P' if c3 else 'F'})")
    if not overall:
        if c1 and not c2:
            print("  Honest partial read: CLIP overlap already sufficed; the world/SigLIP enrichers were "
                  "redundant for transfer. Reported as-is; no patching.")
        elif not c1:
            print("  Honest negative read: collective untied routing does NOT recover cross-modal transfer "
                  "even with added overlap;")
            print("  the tied dictionary stays load-bearing (the wall bites again). Reported straight; no "
                  "substrate retuning, no post-hoc tweak.")
        elif not c3:
            print("  Gauge clause failed: R3 did not survive under untied routing — a real invariance defect, "
                  "reported straight.")
    print("=" * 96)

    return {"results": {f"{c}_{d}": results[(c, d)]["overall"] for c in ("A", "B") for d in ("fwd", "rev")},
            "dF": dF, "clauses": {"c1": bool(c1), "c2": bool(c2), "c3": bool(c3), "overall": bool(overall)}}


if __name__ == "__main__":
    run()
