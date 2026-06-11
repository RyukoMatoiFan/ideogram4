"""Minimal live training dashboard (a tiny TensorBoard) for any run here.

Every trainer writes ``<output_dir>/metrics.jsonl`` (one JSON per log step with
``step``/``loss``/``lr``/``s_per_step``/``peak_gb``, optional ``val_loss`` and ``total``)
and the edit / multi-ref / t2i / slider trainers also write decoded
``<output_dir>/samples/*.png``. This serves a single auto-refreshing page with a
progress bar + elapsed/ETA, three charts (loss, VRAM, speed), and a samples gallery --
no matplotlib, no build step, just the stdlib http server + Chart.js (CDN).

  python dashboard.py --run runs/my-lora                  # -> http://localhost:8080
  python dashboard.py --run runs/slider-detail --total 800 --port 8090

Note: new points only appear every trainer ``logging.log_every`` steps -- the page
polls every 2s but the data is as dense as the trainer logs it. Re-reads the metrics
file + samples dir on every poll, so it tracks a live run.
"""
import argparse
import html
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")

_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>ig4 — {title}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
 body{{font-family:system-ui,sans-serif;margin:0;background:#0f1115;color:#e6e6e6}}
 h1{{font-size:16px;padding:12px 16px;margin:0;background:#171a21;border-bottom:1px solid #262b36}}
 .barwrap{{padding:10px 16px 2px}} .bar{{background:#262b36;border-radius:6px;height:16px;overflow:hidden}}
 .fill{{height:100%;width:0;background:linear-gradient(90deg,#8ab4f8,#81c995);transition:width .5s}}
 .row{{display:flex;flex-wrap:wrap;gap:16px;padding:16px}}
 .card{{background:#171a21;border:1px solid #262b36;border-radius:8px;padding:12px;flex:1;min-width:320px}}
 canvas{{max-height:240px}} #samples{{padding:0 16px 24px}}
 #samples img{{max-width:100%;border:1px solid #262b36;border-radius:6px;margin-bottom:12px;display:block}}
 .muted{{color:#7a828f;font-size:12px;margin-top:5px}} .stat{{color:#8ab4f8}}
</style></head><body>
<h1>{title} &nbsp; <span class="muted" id="hdr" style="margin:0">waiting for metrics…</span></h1>
<div class="barwrap"><div class="bar"><div class="fill" id="fill"></div></div>
 <div class="muted" id="prog"></div></div>
<div class="row">
 <div class="card"><canvas id="loss"></canvas></div>
 <div class="card"><canvas id="vram"></canvas></div>
 <div class="card"><canvas id="speed"></canvas></div>
</div>
<div id="samples"><h1 style="background:none;border:none;padding:16px 0 8px">Samples</h1><div id="gallery" class="muted">none yet</div></div>
<script>
const TOTAL={total};
const mk=(id,label,color)=>new Chart(document.getElementById(id),{{type:'line',
 data:{{datasets:[{{label,data:[],borderColor:color,backgroundColor:color,pointRadius:0,tension:.2}}]}},
 options:{{animation:false,scales:{{x:{{type:'linear',title:{{display:true,text:'step'}}}}}},
 plugins:{{legend:{{labels:{{color:'#e6e6e6'}}}}}}}}}});
const loss=mk('loss','loss','#8ab4f8'),vram=mk('vram','peak VRAM (GB)','#f28b82'),speed=mk('speed','s / step','#81c995');
let valDs=null;
const fmt=s=>{{s=Math.max(0,Math.round(s));const h=(s/3600|0),mn=((s%3600)/60|0),se=s%60;
 return (h?h+'h ':'')+(mn<10&&h?'0':'')+mn+'m '+(se<10?'0':'')+se+'s';}};
async function poll(){{
 const m=await (await fetch('api/metrics')).json();
 loss.data.datasets[0].data=m.map(r=>({{x:r.step,y:r.loss}}));
 if(m.some(r=>r.val_loss!=null)){{ if(!valDs){{valDs={{label:'val',data:[],borderColor:'#fdd663',pointRadius:0,tension:.2}};loss.data.datasets.push(valDs);}}
   valDs.data=m.filter(r=>r.val_loss!=null).map(r=>({{x:r.step,y:r.val_loss}}));}}
 vram.data.datasets[0].data=m.map(r=>({{x:r.step,y:r.peak_gb}}));
 speed.data.datasets[0].data=m.map(r=>({{x:r.step,y:r.s_per_step}}));
 loss.update();vram.update();speed.update();
 const last=m[m.length-1]||{{}};
 // elapsed = sum of per-step time over each logged window (training time, excludes load)
 let el=0; for(let i=0;i<m.length;i++){{el+=(m[i].s_per_step||0)*(m[i].step-(i>0?m[i-1].step:0));}}
 const total=last.total||TOTAL||0, step=last.step||0;
 const pct= total? Math.min(100,100*step/total):0;
 const eta= (total&&step)? el/step*(total-step):0;
 document.getElementById('fill').style.width=pct+'%';
 document.getElementById('prog').textContent= m.length? (total
   ? `step ${{step}}/${{total}} (${{pct.toFixed(1)}}%) · elapsed ${{fmt(el)}} · ETA ${{fmt(eta)}}`
   : `step ${{step}} · elapsed ${{fmt(el)}} · (pass --total for a progress bar)`) : '';
 document.getElementById('hdr').textContent= m.length?
   `loss ${{(last.loss||0).toFixed(4)}} · ${{(last.peak_gb||0).toFixed(1)}}GB · ${{(last.s_per_step||0).toFixed(2)}}s/it · lr ${{(last.lr||0).toExponential(1)}}`
   : 'waiting for metrics…';
 const s=await (await fetch('api/samples')).json();
 document.getElementById('gallery').innerHTML=s.length?s.map(f=>`<div class=muted>${{f}}</div><img src="samples/${{encodeURIComponent(f)}}?t=${{Date.now()}}">`).join(''):'none yet';
}}
poll();setInterval(poll,2000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
  run = "."
  samples_dir = None
  total = 0

  def log_message(self, *a):
    pass

  def _send(self, code, body, ctype="application/json"):
    if isinstance(body, str):
      body = body.encode("utf-8")
    self.send_response(code)
    self.send_header("Content-Type", ctype)
    self.send_header("Content-Length", str(len(body)))
    self.end_headers()
    self.wfile.write(body)

  def do_GET(self):
    path = self.path.split("?", 1)[0]
    if path == "/":
      title = html.escape(os.path.basename(os.path.abspath(self.run)))
      self._send(200, _PAGE.format(title=title, total=int(self.total)), "text/html; charset=utf-8")
    elif path == "/api/metrics":
      self._send(200, json.dumps(self._metrics()))
    elif path == "/api/samples":
      self._send(200, json.dumps(self._samples()))
    elif path.startswith("/samples/"):
      self._serve_image(path[len("/samples/"):])
    else:
      self._send(404, "{}")

  def _metrics(self):
    p = os.path.join(self.run, "metrics.jsonl")
    rows = []
    if os.path.exists(p):
      for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line:
          try:
            rows.append(json.loads(line))
          except json.JSONDecodeError:
            pass
    return rows

  def _samples_root(self):
    return self.samples_dir or os.path.join(self.run, "samples")

  def _samples(self):
    d = self._samples_root()
    if not os.path.isdir(d):
      return []
    fs = [f for f in os.listdir(d) if f.lower().endswith(_IMG_EXTS)]
    fs.sort(reverse=True)
    return fs[:24]

  def _serve_image(self, name):
    from urllib.parse import unquote
    name = unquote(name)
    if "/" in name or "\\" in name or ".." in name:
      return self._send(403, "{}")
    p = os.path.join(self._samples_root(), name)
    if not os.path.isfile(p):
      return self._send(404, "{}")
    ext = os.path.splitext(name)[1].lower()
    with open(p, "rb") as f:
      self._send(200, f.read(), "image/png" if ext == ".png" else "image/jpeg")


def main():
  ap = argparse.ArgumentParser(description="Minimal live training dashboard.")
  ap.add_argument("--run", required=True, help="run output_dir (holds metrics.jsonl + samples/)")
  ap.add_argument("--samples-dir", default=None, help="override the samples dir (e.g. an eval montage dir)")
  ap.add_argument("--total", type=int, default=0, help="total steps (for the progress bar/ETA; "
                                                        "auto-detected if metrics include 'total')")
  ap.add_argument("--port", type=int, default=8080)
  ap.add_argument("--host", default="0.0.0.0")
  args = ap.parse_args()
  Handler.run = args.run
  Handler.samples_dir = args.samples_dir
  Handler.total = args.total
  srv = ThreadingHTTPServer((args.host, args.port), Handler)
  print(f"[dashboard] {args.run} -> http://{args.host}:{args.port}  (Ctrl-C to stop)", flush=True)
  try:
    srv.serve_forever()
  except KeyboardInterrupt:
    pass


if __name__ == "__main__":
  main()
