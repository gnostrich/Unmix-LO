"""
ebr.demo.serve — a browser UX for the cross-modal router.

  python -m ebr.demo.serve            # then open http://127.0.0.1:8765

Loads the frozen ports + libraries ONCE at startup (the slow part), then answers each
request in-process: type a prompt (and/or upload an image), pick which models RECEIVE it,
and watch what every model — including the silent ones — reads out via the tied coupling.
Stdlib only (http.server); no new dependencies beyond what ebr.demo already needs.

This server exposes the SAME pipeline as `python -m ebr.demo` (ports -> materialize ->
tied_transfer -> disagreement meter), just held warm behind an HTTP endpoint. It adds no
mechanism: /run calls readout.tied_transfer verbatim, /scramble calls the same R3 gauge
check the CLI's --scramble runs. The router is untouched.
"""
import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

# ---- warm state: loaded once, reused for every request -------------------------------
_LOCK = threading.Lock()          # the numpy/torch pipeline is not reentrant; serialize /run
_STATE = {"ready": False, "err": None}


def _warm():
    """Load ports + manifest + libs a single time. Slow (model downloads/instantiation)."""
    from . import ports as P, library as L
    _STATE["ports"] = P.load_ports()
    _STATE["manifest"] = L.build(_STATE["ports"])
    _STATE["libs"] = L.load_libs()
    _STATE["L"] = L
    from . import readout as R, meter as MT
    _STATE["R"] = R
    _STATE["MT"] = MT
    _STATE["ports_order"] = list(L.ENGINE_PORTS.keys())
    _STATE["ready"] = True


def _vocab():
    """The library exemplars the readout can name — surfaced so the UX shows its closed vocabulary."""
    libs = _STATE["libs"]
    seen = []
    for ep in _STATE["ports_order"]:
        for ch in _STATE["L"].ENGINE_PORTS[ep]["channels"]:
            for item in libs[ep][ch]:
                lbl = item.get("label") if isinstance(item, dict) else None
                if lbl and lbl not in seen:
                    seen.append(lbl)
    return sorted(seen)


def _run(image, text, subset, atoms):
    """One equilibration. Mirrors cli.main's body; returns a JSON-able dict."""
    L, R, MT = _STATE["L"], _STATE["R"], _STATE["MT"]
    clouds, meta = L.materialize(_STATE["ports"], _STATE["libs"],
                                 image=image, text=text, active_subset=subset)
    tied = R.tied_transfer(clouds, meta, _STATE["manifest"], m=atoms)
    dm = MT.disagreement(clouds, meta)
    panels = []
    for port, p in tied["panels"].items():
        panels.append({
            "port": port, "modality": p["modality"], "active": bool(p["active"]),
            "B": [round(float(b), 2) for b in p["B"]],
            "exemplars": list(p["exemplars"]),
        })
    received = [p["port"] for p in panels if p["active"]]
    return {
        "received_by": received,
        "atoms": int(tied["atoms"]), "active_atoms": int(tied["active_atoms"]),
        "F": round(float(tied["F"]), 3),
        "converged": bool(tied["converged"]), "monotone": bool(tied["monotone"]),
        "panels": panels,
        "meter": {"cycle_cost": round(float(dm["cycle_cost"]), 4),
                  "floor": round(float(dm["floor"]), 4), "verdict": dm["verdict"]},
    }


def _scramble(model, image, text, subset, atoms):
    """R3 gauge check for the current input: scramble one model's features, ΔF must be ~0."""
    from ..geometry.clouds import scramble, cloud_to_Dw
    L, R = _STATE["L"], _STATE["R"]
    libs, manifest = _STATE["libs"], _STATE["manifest"]
    clouds, meta = L.materialize(_STATE["ports"], libs, image=image, text=text, active_subset=subset)
    clouds2 = {}
    for ep in clouds:
        if L.ENGINE_PORTS[ep]["model"] == model:
            clouds2[ep] = [(cloud_to_Dw(scramble(libs[ep][ch], seed=1))[0], clouds[ep][k][1])
                           for k, ch in enumerate(L.ENGINE_PORTS[ep]["channels"])]
        else:
            clouds2[ep] = clouds[ep]
    f0 = R.tied_transfer(clouds, meta, manifest, m=atoms)["F"]
    f1 = R.tied_transfer(clouds2, meta, manifest, m=atoms)["F"]
    dF = abs(f0 - f1)
    return {"model": model, "dF": float(dF), "holds": bool(dF < 1e-6)}


def _decode_image(data_url):
    if not data_url:
        return None
    from PIL import Image
    b64 = data_url.split(",", 1)[1] if "," in data_url else data_url
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


# ---- HTTP -----------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass  # quiet

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/status":
            self._json({"ready": _STATE["ready"], "err": _STATE["err"],
                        "ports": _STATE.get("ports_order", []),
                        "vocab": _vocab() if _STATE["ready"] else []})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._json({"error": f"bad json: {e}"}, 400)
        if not _STATE["ready"]:
            return self._json({"error": "still loading models — try again in a moment"}, 503)

        text = (req.get("text") or "").strip() or None
        image = None
        try:
            image = _decode_image(req.get("image"))
        except Exception as e:
            return self._json({"error": f"bad image: {e}"}, 400)
        if not text and image is None:
            return self._json({"error": "give text and/or an image"}, 400)
        subset = set(req["to"]) if req.get("to") else None
        atoms = int(req.get("atoms", 14))

        with _LOCK:
            try:
                if self.path == "/run":
                    return self._json(_run(image, text, subset, atoms))
                if self.path == "/scramble":
                    return self._json(_scramble(req["model"], image, text, subset, atoms))
            except Exception as e:
                import traceback
                return self._json({"error": str(e), "trace": traceback.format_exc()}, 500)
        self._json({"error": "not found"}, 404)


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EBR cross-modal router</title>
<style>
  :root{color-scheme:light dark;
    --bg:#0f1115;--panel:#171a21;--line:#272c38;--fg:#e7e9ee;--dim:#8b93a7;
    --recv:#2dd4bf;--silent:#64748b;--warn:#f59e0b;--ok:#34d399;--bad:#f87171;--accent:#60a5fa}
  @media(prefers-color-scheme:light){:root{
    --bg:#f6f7f9;--panel:#fff;--line:#e4e7ec;--fg:#1a1d24;--dim:#667085;
    --recv:#0d9488;--silent:#94a3b8;--accent:#2563eb}}
  *{box-sizing:border-box}
  body{margin:0;font:15px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
    background:var(--bg);color:var(--fg)}
  .wrap{max-width:820px;margin:0 auto;padding:24px 18px 64px}
  h1{font-size:19px;margin:0 0 2px;letter-spacing:.2px}
  .sub{color:var(--dim);font-size:13px;margin:0 0 20px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px;margin-bottom:16px}
  label{display:block;font-size:12px;color:var(--dim);margin:0 0 6px;text-transform:uppercase;letter-spacing:.6px}
  input[type=text]{width:100%;padding:11px 13px;font-size:15px;border-radius:10px;border:1px solid var(--line);
    background:var(--bg);color:var(--fg)}
  input[type=text]:focus{outline:none;border-color:var(--accent)}
  .row{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-top:12px}
  .chips{display:flex;gap:8px;flex-wrap:wrap}
  .chip{font-size:12.5px;padding:6px 11px;border-radius:999px;border:1px solid var(--line);
    background:var(--bg);color:var(--dim);cursor:pointer;user-select:none}
  .chip.on{border-color:var(--recv);color:var(--recv);background:color-mix(in srgb,var(--recv) 12%,transparent)}
  button{font:inherit;font-size:14px;font-weight:600;padding:11px 18px;border-radius:10px;border:0;
    background:var(--accent);color:#fff;cursor:pointer}
  button:disabled{opacity:.5;cursor:progress}
  button.ghost{background:transparent;border:1px solid var(--line);color:var(--fg);font-weight:500}
  .hint{font-size:12px;color:var(--dim);margin-top:10px}
  .panel{display:grid;grid-template-columns:22px 116px 62px 1fr;gap:10px;align-items:center;
    padding:9px 4px;border-bottom:1px solid var(--line);font-size:14px}
  .panel:last-child{border-bottom:0}
  .dot{font-size:15px;text-align:center}
  .mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
  .recv .name{color:var(--recv);font-weight:600}
  .silent .name{color:var(--fg)}
  .mod{color:var(--dim);font-size:12px}
  .read{font-weight:600}
  .b{color:var(--dim);font-size:12px}
  .status{display:flex;gap:14px;flex-wrap:wrap;font-size:13px;color:var(--dim);margin:2px 0 6px}
  .status b{color:var(--fg);font-weight:600}
  .meter{font-size:13px;padding:8px 12px;border-radius:9px;margin-top:12px;border:1px solid var(--line)}
  .meter.agree{color:var(--ok)} .meter.disagree{color:var(--warn)}
  .banner{font-size:13px;padding:10px 12px;border-radius:9px;background:color-mix(in srgb,var(--warn) 12%,transparent);
    color:var(--warn);margin-top:12px;display:none}
  .err{color:var(--bad);font-size:13px;white-space:pre-wrap;margin-top:8px}
  .loading{color:var(--dim);font-size:13px}
  a{color:var(--accent)}
  .vocab{font-size:12px;color:var(--dim);margin-top:6px}
  .gauge{font-size:12.5px;color:var(--dim);margin-top:10px}
  .gauge.ok{color:var(--ok)} .gauge.bad{color:var(--bad)}
</style></head><body><div class="wrap">
  <h1>EBR cross-modal router</h1>
  <p class="sub">Feed a prompt to some models — watch what the <b>silent</b> ones read out through the tied coupling.</p>

  <div class="card">
    <label for="text">Text prompt</label>
    <input id="text" type="text" placeholder="a dog sitting on the grass" autocomplete="off">
    <div class="row">
      <label style="margin:0">Image (optional)</label>
      <input id="img" type="file" accept="image/*" style="font-size:13px">
      <span id="imgname" class="hint" style="margin:0"></span>
    </div>
    <div class="row" style="margin-top:14px">
      <div style="flex:1 1 100%"><label>Which models RECEIVE the input <span style="text-transform:none;letter-spacing:0">(none = all that match the modality)</span></label>
        <div class="chips" id="chips"></div></div>
    </div>
    <div class="row">
      <button id="go">Route it</button>
      <button id="gauge" class="ghost" title="R3: scramble a model's internal features — F must not move">Gauge check</button>
      <span id="loading" class="loading"></span>
    </div>
    <div id="err" class="err"></div>
    <div class="vocab" id="vocab"></div>
  </div>

  <div class="card" id="out" style="display:none">
    <div class="status" id="stat"></div>
    <div id="panels"></div>
    <div id="banner" class="banner"></div>
    <div id="meter" class="meter"></div>
    <div id="gaugeout" class="gauge"></div>
  </div>

  <p class="hint">Closed vocabulary — out-of-vocabulary prompts get mapped to the nearest known thing,
  confidently and often wrongly, and the meter won't flag it (all models agree on the same wrong answer).</p>
</div>
<script>
let PORTS=[], VOCAB=[], SUBSET=new Set(), IMG=null;
const $=s=>document.querySelector(s);

async function boot(){
  $("#loading").textContent="loading models…";
  for(let i=0;i<600;i++){
    const s=await (await fetch("/status")).json();
    if(s.err){$("#loading").textContent="";$("#err").textContent="startup error: "+s.err;return;}
    if(s.ready){PORTS=s.ports;VOCAB=s.vocab;drawChips();
      $("#loading").textContent="";
      $("#vocab").textContent = VOCAB.length? ("vocabulary: "+VOCAB.join(", ")) : "";
      return;}
    await new Promise(r=>setTimeout(r,1000));
  }
  $("#loading").textContent="still loading — refresh in a moment";
}
function drawChips(){
  const c=$("#chips");c.innerHTML="";
  PORTS.forEach(p=>{
    const el=document.createElement("span");el.className="chip";el.textContent=p;
    el.onclick=()=>{if(SUBSET.has(p))SUBSET.delete(p);else SUBSET.add(p);
      el.classList.toggle("on");};
    c.appendChild(el);
  });
}
$("#img").onchange=e=>{
  const f=e.target.files[0];if(!f){IMG=null;$("#imgname").textContent="";return;}
  const r=new FileReader();r.onload=()=>{IMG=r.result;$("#imgname").textContent=f.name;};
  r.readAsDataURL(f);
};
function payload(){
  return {text:$("#text").value, image:IMG,
          to:SUBSET.size?[...SUBSET]:null, atoms:14};
}
async function post(path,body){
  const r=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  return r.json();
}
$("#go").onclick=async()=>{
  $("#err").textContent="";$("#gaugeout").textContent="";
  $("#go").disabled=true;$("#loading").textContent="routing…";
  const d=await post("/run",payload());
  $("#go").disabled=false;$("#loading").textContent="";
  if(d.error){$("#err").textContent=d.error;return;}
  render(d);
};
$("#gauge").onclick=async()=>{
  const target = (SUBSET.size? [...SUBSET] : PORTS).find(p=>p.includes("mobilenet")||p.includes("vit")) || PORTS[0];
  $("#gaugeout").textContent="";$("#loading").textContent="scrambling "+target+"…";$("#gauge").disabled=true;
  const d=await post("/scramble",{...payload(),model:target.replace("clip_vision","clip").replace("clip_text","clip")});
  $("#gauge").disabled=false;$("#loading").textContent="";
  if(d.error){$("#err").textContent=d.error;return;}
  const g=$("#gaugeout");g.className="gauge "+(d.holds?"ok":"bad");
  g.textContent=`R3 gauge — scrambled ${d.model}'s internal features → |ΔF| = ${d.dF.toExponential(2)} `
    +(d.holds?"· identical, gauge holds (the tie references input identity, not the representation)":"· LEAK");
};
function render(d){
  $("#out").style.display="block";
  $("#stat").innerHTML =
    `received by <b>${d.received_by.join(", ")||"—"}</b>`
   +`<span>consensus <b>${d.active_atoms}/${d.atoms}</b> atoms</span>`
   +`<span>F <b>${d.F}</b></span>`
   +`<span>converged <b>${d.converged}</b></span>`;
  const P=$("#panels");P.innerHTML="";
  d.panels.forEach(p=>{
    const row=document.createElement("div");
    row.className="panel "+(p.active?"recv":"silent");
    row.innerHTML=
      `<div class="dot">${p.active?"●":"○"}</div>`
     +`<div class="name mono">${p.port}</div>`
     +`<div class="mod">${p.modality}</div>`
     +`<div><span class="read">${p.exemplars[0]||"—"}</span> `
     +`<span class="b mono">B=[${p.B.join(",")}]</span></div>`;
    P.appendChild(row);
  });
  const m=$("#meter");m.className="meter "+(d.meter.verdict.includes("agree")&&!d.meter.verdict.includes("dis")?"agree":"disagree");
  m.textContent=`disagreement meter: cycle cost ${d.meter.cycle_cost} vs floor ${d.meter.floor} → ${d.meter.verdict}`;
  // out-of-vocab hint: unanimous silent reads but input has no vocab hit
  const reads=new Set(d.panels.map(p=>p.exemplars[0]));
  const t=($("#text").value||"").toLowerCase();
  const inVocab = VOCAB.some(v=>t.includes(v.toLowerCase().replace(/^a photo of a /,"")));
  const b=$("#banner");
  if(t && !inVocab && reads.size===1){
    b.style.display="block";
    b.textContent=`heads up: “${$("#text").value}” isn't in the vocabulary, yet every model unanimously read `
      +`“${[...reads][0]}”. This is the closed-vocab failure mode — confident, unanimous, and the meter can't see it.`;
  } else b.style.display="none";
}
boot();
</script></body></html>"""


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="ebr.demo.serve")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args(argv)

    def warm():
        try:
            _warm()
            print("[ebr.demo.serve] models ready.", flush=True)
        except Exception as e:
            import traceback
            _STATE["err"] = str(e)
            print("[ebr.demo.serve] warm-up FAILED:\n" + traceback.format_exc(), flush=True)

    threading.Thread(target=warm, daemon=True).start()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"[ebr.demo.serve] open  http://{a.host}:{a.port}   (models loading in background)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[ebr.demo.serve] bye.")


if __name__ == "__main__":
    main()
