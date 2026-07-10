"""
Real prompt-able UNION object — harness skeleton (fill with real models in Claude Code w/ HF access).
The object and its honest test are the SAME build. Core VLM = read-out + reasoning depth.
Router adds specialist BREADTH only where the core is uncertain, with a hard anti-drag floor.
NOT synergy. Ceiling = union. Core-alone is the floor everywhere.
"""

class PromptableUnionObject:
    def __init__(self, core_vlm, specialists, router, admit_threshold=0.7):
        self.core = core_vlm              # frozen open VLM (e.g. SmolVLM / Qwen2-VL-2B): nuanced read-out
        self.specialists = specialists    # frozen encoders/knowledge sources, coverage disjoint from core
        self.router = router              # low-rank/MZ routing memory: cost ~ rank, not N
        self.admit = admit_threshold      # anti-drag: specialist enters context only if it clears this

    def answer(self, image=None, text=None):
        core_out, core_conf = self.core.read_out(image, text)     # nuanced answer + self-confidence
        if core_conf >= self.admit:
            return core_out, {"routed": [], "cost": 1}            # core knows -> no drag, no routing

        # core uncertain -> route to specialist(s) that ACTUALLY know (admission gate = anti-drag)
        cands = self.router.select(image, text)                   # low-rank select, sub-linear in N
        admitted = [(s, cov) for (s, cov) in cands if cov >= self.admit]
        if not admitted:
            return self.core.abstain_or_best(core_out), {"routed": [], "cost": 1}  # out-of-union -> abstain, NEVER invent

        # inject only admitted specialist knowledge; core still does the read-out (union, not fusion)
        ctx = [s.knowledge(image, text) for (s, _) in admitted]
        union_out, _ = self.core.read_out(image, text, injected=ctx)
        # FLOOR GUARD: never return something worse than core-alone on this query
        final = self.core.prefer_better(core_out, union_out)      # keeps (B) no-drag as a hard invariant
        return final, {"routed": [s.name for (s, _) in admitted], "cost": 1 + len(admitted)}

# --- pre-committed measurement (see PREREG.md) ---
# On query set K (core-known): assert object >= core_alone            # (B) no drag  <- THE critical cell
# On query set U (needs breadth): assert object >> core_alone         # (A) breadth
# Cost vs N: assert sublinear and flatter than naive-inject-all       # (C) flat cost
# On adversarial out-of-union: assert object ABSTAINS (no fabrication) # (D) ceiling/safety
# Baseline naive-inject-all must be worse or costlier, else router adds nothing.
