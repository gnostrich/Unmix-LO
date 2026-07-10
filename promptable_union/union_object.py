"""
REAL prompt-able UNION object — fills harness_stub.py with models that actually run.

The object and its honest test are the SAME build (see PREREG.md). Core VLM = nuanced read-out +
reasoning DEPTH. Router adds specialist BREADTH only where the core is uncertain AND a specialist
actually covers the query (admission gate = anti-drag). NOT synergy. Ceiling = union. Core-alone is
the floor everywhere; out-of-union -> ABSTAIN, never invent.

CPU-only friendly: SmolVLM-500M core (image-splitting OFF -> ~2s/query), frozen HF classifier
specialists, CLIP union router (cost ~ #prototypes, not #training points).
"""
import re
import numpy as np
import torch

torch.set_num_threads(4)
_NORM = re.compile(r"[^a-z0-9]+")


def _norm(s):
    return _NORM.sub(" ", str(s).lower()).strip()


def label_hit(answer_text, gt_label):
    """Objective scoring: does the free-text answer name the ground-truth label (either direction,
    token-level, tolerant of _/spaces/plurals)? Deliberately generous to the CORE so drag/breadth are real."""
    a, g = _norm(answer_text), _norm(gt_label)
    if not g:
        return False
    if g in a or a in g:
        return True
    gt_toks = [t for t in g.split() if len(t) > 2]
    a_toks = set(a.split())
    # count a hit if every content token of the (usually 1-2 word) label appears in the answer
    return len(gt_toks) > 0 and all(any(t == at or t[:-1] == at or at[:-1] == t for at in a_toks) for t in gt_toks)


# ============================================================ CORE: frozen VLM read-out + confidence
class CoreVLM:
    def __init__(self, model_id="HuggingFaceTB/SmolVLM-500M-Instruct", max_new_tokens=16):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.proc = AutoProcessor.from_pretrained(model_id, do_image_splitting=False, size={"longest_edge": 512})
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.float32).eval()
        self.max_new_tokens = max_new_tokens
        self.name = "core:" + model_id.split("/")[-1]

    def read_out(self, image, question="What is in this image? Answer with the single most specific name.",
                 injected=None):
        """Returns (answer_text, confidence). confidence = mean token probability of the generated answer
        (real, from the model's own logits). injected = list of specialist knowledge strings (union context)."""
        q = question
        if injected:
            q = ("Context from domain specialists (use only if correct and relevant): "
                 + " ; ".join(injected) + ". " + question)
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
        prompt = self.proc.apply_chat_template(msgs, add_generation_prompt=True)
        inp = self.proc(text=prompt, images=[image], return_tensors="pt")
        n_in = inp["input_ids"].shape[1]
        with torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      output_scores=True, return_dict_in_generate=True)
        gen = out.sequences[:, n_in:]
        text = self.proc.batch_decode(gen, skip_special_tokens=True)[0].strip()
        # confidence: geometric-mean token probability of the greedy answer
        probs = []
        for tok_id, score in zip(gen[0].tolist(), out.scores):
            p = torch.softmax(score[0], -1)[tok_id].item()
            probs.append(max(p, 1e-6))
        conf = float(np.exp(np.mean(np.log(probs)))) if probs else 0.0
        return text, conf

    def abstain_or_best(self, core_out):
        return "(abstain: outside the union's coverage) " + core_out


# ============================================================ SPECIALISTS: frozen classifiers w/ real coverage
class Specialist:
    def __init__(self, model_id, domain, prototypes):
        from transformers import pipeline
        self.clf = pipeline("image-classification", model=model_id)
        self.name = domain
        self.domain = domain
        self.prototypes = prototypes           # text phrases describing the domain (for CLIP routing)
        self._n = len(self.clf.model.config.id2label)

    def knowledge(self, image):
        """(label_text, coverage_confidence) — top-1 class + its softmax score (real calibrated-ish signal)."""
        r = self.clf(image, top_k=1)[0]
        return r["label"].replace("_", " "), float(r["score"])


# ============================================================ ROUTER: CLIP union routing (cost ~ rank, not N pts)
class CLIPRouter:
    def __init__(self, specialists, clip_id="openai/clip-vit-base-patch32"):
        from transformers import CLIPModel, CLIPProcessor
        self.clip = CLIPModel.from_pretrained(clip_id).eval()
        self.cp = CLIPProcessor.from_pretrained(clip_id)
        self.specialists = specialists
        # one text prototype vector per specialist domain (mean of its prototype phrases) — the low-rank memory
        protos = []
        for s in specialists:
            ti = self.cp(text=s.prototypes, return_tensors="pt", padding=True)
            with torch.no_grad():
                te = self.clip.get_text_features(**ti).pooler_output   # already joint CLIP space (512-d)
            te = te / te.norm(dim=-1, keepdim=True)
            protos.append(te.mean(0))
        P = torch.stack(protos)
        self.P = P / P.norm(dim=-1, keepdim=True)   # (N_spec, d) unit prototypes

    def _img_emb(self, image):
        ii = self.cp(images=image, return_tensors="pt")
        with torch.no_grad():
            e = self.clip.get_image_features(**ii).pooler_output       # already joint CLIP space (512-d)
        return (e / e.norm(dim=-1, keepdim=True))[0]

    def select(self, image):
        """Return [(specialist, router_sim)] sorted desc. Cost = N_spec dot-products (rank), NOT dataset size."""
        e = self._img_emb(image)
        sims = (self.P @ e).tolist()
        return sorted(zip(self.specialists, sims), key=lambda x: -x[1])


# ============================================================ THE OBJECT
class PromptableUnionObject:
    """Coverage-gated union. The anti-drag admission gate is the prereg's stated one: a specialist enters
    the context ONLY when its COVERAGE clears a threshold (router CLIP-match AND its own classifier
    confidence). This does NOT depend on the core's self-confidence — which, empirically, a small frozen
    VLM does not calibrate (it reports ~0.85 whether right or wrong), making the prereg's core_conf
    fast-path inert. `admit_core` is kept as that fast-path but defaults OFF (>1) and is reported, not relied on."""
    def __init__(self, core, specialists, router, admit_core=1.01, route_sim=0.24, admit_spec=0.30,
                 abstain_core=0.0):
        self.core = core
        self.specialists = specialists
        self.router = router
        self.admit_core = admit_core     # prereg core-"knows-it" fast-path; default OFF (uncalibrated on small VLM)
        self.route_sim = route_sim       # COVERAGE gate: CLIP domain match required to even call a specialist
        self.admit_spec = admit_spec     # COVERAGE gate: specialist's own top-1 confidence required to admit it
        self.abstain_core = abstain_core # abstain iff no specialist covers AND core self-uncertainty < this

    def answer(self, image, question="What is in this image? Answer with the single most specific name.",
               force_inject_all=False):
        core_out, core_conf = self.core.read_out(image, question)
        meta = {"core_conf": round(core_conf, 3), "routed": [], "specialist_calls": 0,
                "abstained": False, "router_sims": {}}

        # NAIVE-INJECT-ALL control: dump every specialist into context every query (cost = N, no gate)
        if force_inject_all:
            ctx, called = [], []
            for s in self.specialists:
                lbl, cov = s.knowledge(image); called.append(s.name)
                ctx.append(f"{s.domain} specialist says: {lbl} (conf {cov:.2f})")
            union_out, _ = self.core.read_out(image, question, injected=ctx)
            meta.update(routed=called, specialist_calls=len(self.specialists))
            return union_out, meta

        # prereg core-knows-it fast-path (inert by default; kept for the diagnostic)
        if core_conf >= self.admit_core:
            return core_out, meta

        # COVERAGE-GATED routing: rank by CLIP domain match; CALL only specialists above the coverage sim
        # (flat cost ~ #covering specialists, typically <=1, NOT N); admit those clearing their own confidence.
        cands = self.router.select(image)
        admitted, ctx = [], []
        for s, sim in cands:
            meta["router_sims"][s.name] = round(float(sim), 3)
            if sim < self.route_sim:
                continue                                   # no coverage -> don't even call it (keeps cost flat)
            meta["specialist_calls"] += 1
            lbl, cov = s.knowledge(image)
            if cov >= self.admit_spec:
                admitted.append(s.name)
                ctx.append(f"{s.domain} specialist says: {lbl} (conf {cov:.2f})")
        if not admitted:
            # out-of-union: no specialist covers. Core is the FLOOR. Abstain only if the core is also
            # self-uncertain (this is the cell that needs calibrated core confidence — reported honestly).
            if core_conf < self.abstain_core:
                meta["abstained"] = True
                return self.core.abstain_or_best(core_out), meta
            return core_out, meta
        union_out, _ = self.core.read_out(image, question, injected=ctx)
        meta["routed"] = admitted
        return union_out, meta
