# ebr — Equilibrium Barycentric Router (stage-0)

Implementation of EBR (authoritative spec: **`ebr-spec-v1.1.md`**). Single-authority principle: ONE functional
`F` decides everything; all else is instrument (reports/gates claims), oracle (proposes/warm-starts;
acceptance is strict F-descent), or experimental control. Optimization is block-coordinate **mirror descent on
F with a backtracking line search** guaranteeing monotone descent — the blocks are NOT exact I-projections
(FIX-3). Structural growth is **Frank–Wolfe support adaptation** on the anchor measure, driven by F alone; the
Hankel/poles are instrument, never mechanism. The corrected headline is a mechanism/instrument split: atom
count = spatial complexity of the shared geometry (K-invariant); the pole set = temporal McMillan degree of
traffic (multiplicative closure). See `ebr-spec-v1.1.md` and `LEDGER.md`.

**Hard rule (invariant interface, §0):** nothing downstream of `geometry/` consumes coordinates — only
normalized cost matrices `(D, w)` cross any boundary. This is what makes every logged quantity gauge-invariant
(G0), and it is enforced by module boundary + CI fixture.

```
ebr/
  geometry/   cloud -> (normalized D, w) + gauge-faithful Gram helpers. ONLY module touching model outputs.
  transport/  square-loss entropic semi-relaxed GW; backtracked monotone mirror steps.
  energy/     F assembly + shared-anchor block-coordinate loop (Lyapunov-guarded).
  events/     Frank–Wolfe support adaptation (atoms: grow/park/revive — F-driven, validated).
  registry/   append-only ledger + preflight (frozen constants).
  experiments/ substrate + validated controls (fw_selfsize, g4_meter meter, sympower/pole_closure P5).
  tests/      CI invariants (gauge, Lyapunov, coupling-continuity, interface, gram gauge).
```

Status, honest verdicts, and corrections: see `LEDGER.md`, `AUDIT_VERDICT.md`, and `WALL_2x2_atomleg.md`.
Proven: gauge-exact interface, Lyapunov-monotone equilibration, F-driven FW self-quench + K-invariance,
G4 disagreement meter (20.4×), P5 pole closure. Open: the demo (R2/R4/R5 readout on real models).

```
python -m pytest ebr/tests -q     # invariants
```

## Demo — run the router on real frozen models (steps 2–3)

Real ports: **vit** + **mobilenet** (two vision families), **minilm** (text), **clip** (channel-native: vision
tower + text tower), each carrying 2 channels where the interface decomposes (R2). Probe libraries (each
model's behavioral support) are built once and cached: **CIFAR-10** test[:256] natural images for vision,
index-paired **class-anchored captions** ("a photo of a {class}") for text — from the HF mirror. (Free-form
COCO captions were tried and break cross-modal alignment — see `WALL_crossmodal.md`.)

```
pip install torch torchvision transformers datasets pillow numpy
python -m ebr.demo --text "a dog sitting on the grass"     # text ports active, vision silent
python -m ebr.demo --image dog.png                          # vision ports active, text silent
python -m ebr.demo --image dog.png --text "a dog" --to vit,minilm   # send input to a subset (R5)
python -m ebr.demo --text "a cat" --scramble mobilenet      # R3 gauge guarantee, user-visible
```

The demo loads the models, materializes each port's cloud (its library reweighted toward the input; silent
models uniform), equilibrates a shared anchor (F-loop, Lyapunov-monotone; per-prompt channel-gain routing B =
R4), and prints consensus + what each model says + the session line.

**What works (live):** a dog image → **vit, mobilenet, AND clip_vision all say `dog`** — three different
vision architectures, each with its own embedding geometry, aligned through the shared anchor by
gauge-invariant relational coupling. `--scramble` shows |ΔF| ≈ 1e-16 (R3 gauge guarantee, user-visible).
Channel routing B adapts per prompt.

**Documented wall (`WALL_crossmodal.md`):** cross-MODAL transfer to silent *text* models does NOT work —
relational-only GW (R3) discards the cross-modal correspondence, so a dog image cannot reliably make a silent
text model surface dog captions across heterogeneous embedding spaces. Silent cross-modal panels are flagged
`[cross-modal: LIMITED]`; the panel is honest, not dressed up. Options (CLIP-bridge / tied paired couplings /
one-axis scope) are in the wall doc. Probe provenance: `uoft-cs/cifar10` test[:256] + class-anchored captions.
