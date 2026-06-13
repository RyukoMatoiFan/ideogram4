"""Minimal live training dashboard (a tiny TensorBoard) for any run here.

Every trainer writes ``<output_dir>/metrics.jsonl`` (one JSON per log step with
``step``/``loss``/``lr``/``s_per_step``/``peak_gb``, optional ``val_loss`` and ``total``)
and the t2i / edit / multi-ref trainers also write decoded ``<output_dir>/samples/*.png``.
A ``samples/prompts.json`` ({idx: text}) captions each preview.

Curated T2I previews are saved as a single 4-column CONTACT SHEET per step
(``stepNNNNNN_dashboard.png``). The dashboard slices that sheet back into one card per
prompt server-side (``/tile``) so each preview is its own hoverable / clickable image --
no trainer change or restart needed. Per-tile files (``..idxK..``) are used directly.

Optional ``--base-dir`` (a directory of per-prompt ``idxK.png`` base-model tiles, e.g.
``<output_dir>/base_previews``) enables a click-to-toggle BASE vs current preview per
prompt for easy comparison.

Features: live stat chips + progress/ETA, loss(+val)/VRAM/speed charts, a sample grid with
prev/next step pagination, click-to-toggle base vs preview, and a click-to-zoom lightbox.

  python dashboard.py --run runs/my-run --total 150000 --port 8090 \
    --base-dir runs/my-run/base_previews
"""
import argparse
import html
import json
import math
import os
import re
from io import BytesIO
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote, unquote, urlparse, parse_qs

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")
_SHEET_COLS = 4  # curated preview contact sheets are built 4-wide (train_t2i_full_cached._sample)

_PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__ · ig4</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
 :root{
   --bg:#0a0c10; --panel:#11151c; --panel2:#161b24; --line:#222a36;
   --ink:#e7ecf3; --mut:#8a94a6; --accent:#6ea8fe; --good:#56d364; --warn:#e3b341; --bad:#f2756b;
 }
 *{box-sizing:border-box}
 body{font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
   margin:0;background:radial-gradient(1200px 600px at 70% -10%,#121a28 0%,var(--bg) 60%);
   color:var(--ink);-webkit-font-smoothing:antialiased}
 header{position:sticky;top:0;z-index:10;backdrop-filter:blur(8px);
   background:rgba(10,12,16,.78);border-bottom:1px solid var(--line);padding:12px 22px}
 .htop{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
 .dot{width:9px;height:9px;border-radius:50%;background:var(--good);box-shadow:0 0 10px var(--good);
   animation:pulse 2s infinite}
 @keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
 h1{font-size:15px;font-weight:650;margin:0;letter-spacing:.2px}
 h1 .sub{color:var(--mut);font-weight:400;margin-left:6px}
 .chips{display:flex;gap:8px;flex-wrap:wrap;margin-left:auto}
 .chip{background:var(--panel2);border:1px solid var(--line);border-radius:999px;
   padding:5px 12px;font-size:12px;font-variant-numeric:tabular-nums;white-space:nowrap}
 .chip b{color:var(--accent);font-weight:650} .chip.v b{color:var(--warn)}
 .chip .k{color:var(--mut);margin-right:5px}
 .barwrap{margin-top:10px}
 .bar{background:var(--panel2);border:1px solid var(--line);border-radius:8px;height:10px;overflow:hidden}
 .fill{height:100%;width:0;background:linear-gradient(90deg,var(--accent),var(--good));
   transition:width .6s cubic-bezier(.4,0,.2,1)}
 .prog{color:var(--mut);font-size:12px;margin-top:5px;font-variant-numeric:tabular-nums}
 .wrap{max-width:1500px;margin:0 auto;padding:20px 22px 40px}
 .charts{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
 @media(max-width:900px){.charts{grid-template-columns:1fr}}
 .card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:14px 16px 10px}
 .card h3{margin:0 0 8px;font-size:12px;font-weight:600;color:var(--mut);
   text-transform:uppercase;letter-spacing:.6px}
 canvas{max-height:230px}
 .sech{display:flex;align-items:center;gap:12px;margin:30px 2px 14px;flex-wrap:wrap}
 .sech h2{font-size:15px;margin:0;font-weight:650}
 .sech .muted{color:var(--mut);font-size:12px}
 .pager{display:flex;align-items:center;gap:8px;margin-left:auto}
 .pager button{background:var(--panel2);border:1px solid var(--line);color:var(--ink);
   width:34px;height:30px;border-radius:9px;cursor:pointer;font-size:15px;line-height:1;transition:.12s}
 .pager button:hover:not(:disabled){border-color:var(--accent);color:var(--accent)}
 .pager button:disabled{opacity:.35;cursor:default}
 .pager .lbl{font-size:12px;color:var(--mut);font-variant-numeric:tabular-nums;min-width:150px;text-align:center}
 .hint{font-size:11px;color:var(--mut);margin:0 2px 12px}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(248px,1fr));gap:16px}
 figure.scard{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:14px;
   overflow:hidden;transition:transform .15s ease,border-color .15s ease}
 figure.scard:hover{transform:translateY(-3px);border-color:#33405a}
 .imgwrap{position:relative;aspect-ratio:1/1;background:#05070b;overflow:hidden;cursor:pointer}
 .imgwrap img{width:100%;height:100%;object-fit:cover;display:block;transition:transform .35s ease}
 figure.scard:hover .imgwrap img{transform:scale(1.05)}
 .badge{position:absolute;top:8px;left:8px;background:rgba(8,10,14,.82);border:1px solid var(--line);
   color:var(--accent);font-size:11px;font-weight:650;padding:2px 8px;border-radius:999px;letter-spacing:.3px}
 .badge.base{color:var(--warn);border-color:#5a4a1f}
 .zoom{position:absolute;top:7px;right:7px;width:26px;height:26px;border-radius:8px;
   background:rgba(8,10,14,.82);border:1px solid var(--line);color:var(--ink);cursor:zoom-in;
   display:flex;align-items:center;justify-content:center;font-size:13px;opacity:0;transition:.12s}
 figure.scard:hover .zoom{opacity:1} .zoom:hover{border-color:var(--accent);color:var(--accent)}
 figcaption{padding:9px 11px 11px;font-size:12px;line-height:1.45;color:var(--mut);
   display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;min-height:38px}
 .empty{color:var(--mut);font-size:13px;padding:30px 0}
 /* lightbox: preview + base side by side */
 #lb{position:fixed;inset:0;z-index:50;background:rgba(5,7,11,.93);display:none;
   align-items:center;justify-content:center;flex-direction:column;gap:14px;padding:24px;cursor:zoom-out}
 #lb.on{display:flex}
 #lbrow{display:flex;gap:18px;align-items:flex-start;max-width:96vw}
 .lbfig{margin:0;display:flex;flex-direction:column;gap:8px;align-items:center}
 .lbfig img{max-width:46vw;max-height:74vh;border-radius:12px;border:1px solid var(--line);
   object-fit:contain;background:#05070b;cursor:default}
 .lbfig figcaption .tg{font-weight:700;letter-spacing:.5px;font-size:12px}
 #lbcap{color:var(--ink);font-size:13px;max-width:90vw;text-align:center;line-height:1.5}
 #lbhint{color:var(--mut);font-size:12px}
</style></head><body>
<header>
 <div class="htop">
  <span class="dot"></span>
  <h1>__TITLE__ <span class="sub" id="sub">live training</span></h1>
  <div class="chips" id="chips"></div>
 </div>
 <div class="barwrap"><div class="bar"><div class="fill" id="fill"></div></div>
  <div class="prog" id="prog">waiting for metrics…</div></div>
</header>
<div class="wrap">
 <div class="charts">
  <div class="card"><h3>loss &amp; val</h3><canvas id="loss"></canvas></div>
  <div class="card"><h3>peak VRAM (GB)</h3><canvas id="vram"></canvas></div>
  <div class="card"><h3>sec / step</h3><canvas id="speed"></canvas></div>
 </div>
 <div class="sech"><h2>Samples</h2><span class="muted" id="scount"></span>
  <div class="pager">
   <button id="first" title="first">⏮</button>
   <button id="prev" title="previous step">◀</button>
   <span class="lbl" id="plabel">—</span>
   <button id="next" title="next step">▶</button>
   <button id="last" title="latest">⏭</button>
  </div>
 </div>
 <div class="hint">Click an image to <b>zoom</b> (preview + base side by side) · double-click to toggle <b>BASE</b> inline · ◀ ▶ (or arrow keys) page through preview steps.</div>
 <div class="grid" id="gallery"><div class="empty">no previews yet</div></div>
</div>
<div id="lb">
 <div id="lbrow">
  <figure class="lbfig"><img id="lbprev" src=""><figcaption><span class="tg" style="color:#6ea8fe">PREVIEW</span></figcaption></figure>
  <figure class="lbfig" id="lbbasewrap"><img id="lbbase" src=""><figcaption><span class="tg" style="color:#e3b341">BASE</span></figcaption></figure>
 </div>
 <div id="lbcap"></div>
 <div id="lbhint">Esc or click the backdrop to close · double-click a card to toggle base inline</div>
</div>
<script>
const TOTAL=__TOTAL__;
const esc=s=>(s+'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const fmt=s=>{s=Math.max(0,Math.round(s));const h=(s/3600|0),mn=((s%3600)/60|0),se=s%60;
 return (h?h+'h ':'')+(mn<10&&h?'0':'')+mn+'m '+(se<10?'0':'')+se+'s';};
const gridCfg=(extra={})=>({type:'line',data:{datasets:[]},options:{animation:false,responsive:true,
 interaction:{intersect:false,mode:'index'},
 scales:{x:{type:'linear',title:{display:true,text:'step',color:'#8a94a6'},grid:{color:'#1b2230'},ticks:{color:'#8a94a6'}},
  y:{grid:{color:'#1b2230'},ticks:{color:'#8a94a6'}}},
 plugins:{legend:{display:!!extra.legend,labels:{color:'#e7ecf3',boxWidth:10,usePointStyle:true}}}}});
const ds=(label,color,fill=false)=>({label,data:[],borderColor:color,
 backgroundColor:fill?color+'22':color,pointRadius:0,borderWidth:2,tension:.25,fill});
const loss=new Chart(document.getElementById('loss'),gridCfg({legend:true}));
loss.data.datasets=[
 {label:'loss (raw)',data:[],borderColor:'#6ea8fe44',backgroundColor:'#6ea8fe44',pointRadius:0,borderWidth:1,tension:.2},
 {label:'loss (ema)',data:[],borderColor:'#6ea8fe',backgroundColor:'#6ea8fe',pointRadius:0,borderWidth:2,tension:.25},
 {label:'val',data:[],borderColor:'#e3b341',backgroundColor:'#e3b341',pointRadius:0,borderWidth:2.5,tension:.25},
];
const vram=new Chart(document.getElementById('vram'),gridCfg());vram.data.datasets=[ds('vram','#f2756b',true)];
const speed=new Chart(document.getElementById('speed'),gridCfg());speed.data.datasets=[ds('s/step','#56d364')];
const chip=(k,v,cls='')=>`<span class="chip ${cls}"><span class="k">${k}</span><b>${v}</b></span>`;

let STEPS=[], curStep=null, pinned=false, data={items:[]}, lastKey='';
const G=id=>document.getElementById(id);

function renderGallery(){
 const items=data.items||[];
 G('scount').textContent=items.length?`step ${(data.step||0).toLocaleString()} · ${items.length} previews`:'';
 const i=STEPS.indexOf(data.step);
 G('plabel').textContent=STEPS.length?`step ${(data.step||0).toLocaleString()}  (${i+1}/${STEPS.length})`:'—';
 G('first').disabled=G('prev').disabled=(i<=0);
 G('last').disabled=G('next').disabled=(i<0||i>=STEPS.length-1);
 // Re-render the gallery DOM ONLY when content changes -> images never re-fetch on
 // unchanged polls (no flicker) and an inline base-toggle survives the next poll.
 const key=(data.step)+'·'+items.map(it=>it.src+'|'+(it.base||'')).join('~');
 if(key===lastKey) return;
 lastKey=key;
 G('gallery').innerHTML=items.length? items.map(it=>{
   const hasBase=!!it.base;
   return `<figure class="scard" data-prev="${esc(it.src)}" data-base="${esc(it.base||'')}" data-prompt="${esc(it.prompt||'')}" data-idx="${it.idx}">`
    +`<div class="imgwrap" onclick="imgClick(this)" ondblclick="imgDbl(this)">`
    +`<span class="badge">#${it.idx} · PREVIEW</span>`
    +`<img loading="lazy" src="${esc(it.src)}"></div>`
    +`<figcaption>${it.prompt?esc(it.prompt):'<span style=opacity:.5>no prompt</span>'}`
    +`${hasBase?'':' <span style="color:#f2756b">· no base yet</span>'}</figcaption></figure>`;
  }).join('') : '<div class="empty">no previews yet</div>';
}
let clickT=null;  // single click = zoom; double click = toggle base (delay disambiguates)
function imgClick(w){if(clickT)return;clickT=setTimeout(()=>{clickT=null;zoom(w.closest('.scard'));},220);}
function imgDbl(w){if(clickT){clearTimeout(clickT);clickT=null;}tog(w);}
function tog(wrap){
 const f=wrap.closest('.scard'),img=wrap.querySelector('img'),b=wrap.querySelector('.badge');
 const base=f.dataset.base; if(!base) return;
 if(img.dataset.mode==='base'){img.src=f.dataset.prev;img.dataset.mode='prev';b.textContent=`#${f.dataset.idx} · PREVIEW`;b.classList.remove('base');}
 else{img.src=base;img.dataset.mode='base';b.textContent=`#${f.dataset.idx} · BASE`;b.classList.add('base');}
}
function zoom(f){
 G('lbprev').src=f.dataset.prev;
 if(f.dataset.base){G('lbbase').src=f.dataset.base;G('lbbasewrap').style.display='';}
 else{G('lbbase').removeAttribute('src');G('lbbasewrap').style.display='none';}
 G('lbcap').innerHTML=`#${f.dataset.idx} · step ${(data.step||0).toLocaleString()} — ${esc(f.dataset.prompt||'')}`
   +(f.dataset.base?'':' <span style="color:#f2756b">(no base sample)</span>');
 G('lb').classList.add('on');
}
G('lb').onclick=e=>{if(e.target.id!=='lbprev'&&e.target.id!=='lbbase')G('lb').classList.remove('on');};
document.addEventListener('keydown',e=>{
 if(e.key==='Escape')G('lb').classList.remove('on');
 else if(!G('lb').classList.contains('on')){if(e.key==='ArrowLeft')go(-1);if(e.key==='ArrowRight')go(1);}});

async function loadSamples(step){
 const q=step!=null?`?step=${step}`:'';
 try{data=await (await fetch('api/samples'+q)).json();}catch(e){return;}
 STEPS=data.steps||[]; curStep=data.step;
 renderGallery();
}
function go(d){const i=STEPS.indexOf(curStep);let j=i+d;if(j<0)j=0;if(j>STEPS.length-1)j=STEPS.length-1;
 if(STEPS[j]!=null){pinned=(j!==STEPS.length-1);loadSamples(STEPS[j]);}}
G('prev').onclick=()=>go(-1); G('next').onclick=()=>go(1);
G('first').onclick=()=>{if(STEPS.length){pinned=true;loadSamples(STEPS[0]);}};
G('last').onclick=()=>{pinned=false;loadSamples();};

async function pollMetrics(){
 try{
  const m=await (await fetch('api/metrics')).json();
  if(!m.length)return;
  const lr_=m.filter(r=>r.loss!=null);
  loss.data.datasets[0].data=lr_.map(r=>({x:r.step,y:r.loss}));
  let e=null;const ema=[];for(const r of lr_){e=e==null?r.loss:e*0.9+r.loss*0.1;ema.push({x:r.step,y:e});}
  loss.data.datasets[1].data=ema;
  loss.data.datasets[2].data=m.filter(r=>r.val_loss!=null).map(r=>({x:r.step,y:r.val_loss}));
  vram.data.datasets[0].data=m.map(r=>({x:r.step,y:r.peak_gb}));
  speed.data.datasets[0].data=m.map(r=>({x:r.step,y:r.s_per_step}));
  loss.update();vram.update();speed.update();
  const last=m[m.length-1],step=last.step||0,total=TOTAL||last.total||0;
  const vals=m.filter(r=>r.val_loss!=null),sps=last.s_per_step||0,eta=total&&sps?fmt((total-step)*sps):'—';
  G('chips').innerHTML=chip('step',step.toLocaleString()+(total?' / '+total.toLocaleString():''))
   +chip('loss',(last.loss||0).toFixed(4))+(vals.length?chip('val',vals[vals.length-1].val_loss.toFixed(4),'v'):'')
   +chip('s/it',sps.toFixed(2))+chip('VRAM',(last.peak_gb||0).toFixed(1)+'G')
   +chip('lr',(last.lr||0).toExponential(1))+chip('ETA',eta);
  if(total){G('fill').style.width=Math.min(100,100*step/total)+'%';
   G('prog').textContent=`${(100*step/total).toFixed(1)}% · step ${step.toLocaleString()} of ${total.toLocaleString()} · ETA ${eta}`;}
  else G('prog').textContent=`step ${step.toLocaleString()} · pass --total for a progress bar`;
 }catch(e){G('sub').textContent='metrics unavailable';}
}
async function tick(){await pollMetrics(); if(!pinned) await loadSamples();}
tick();setInterval(tick,2500);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
  run = "."
  samples_dir = None
  base_dir = None
  total = 0

  def log_message(self, *a):
    pass

  def _send(self, code, body, ctype="application/json"):
    if isinstance(body, str):
      body = body.encode("utf-8")
    self.send_response(code)
    self.send_header("Content-Type", ctype)
    self.send_header("Cache-Control", "no-store")
    self.send_header("Content-Length", str(len(body)))
    self.end_headers()
    self.wfile.write(body)

  def do_GET(self):
    u = urlparse(self.path)
    path, qs = u.path, parse_qs(u.query)
    if path == "/":
      title = html.escape(os.path.basename(os.path.abspath(self.run)))
      page = _PAGE.replace("__TITLE__", title).replace("__TOTAL__", str(int(self.total)))
      self._send(200, page, "text/html; charset=utf-8")
    elif path == "/api/metrics":
      self._send(200, json.dumps(self._metrics()))
    elif path == "/api/samples":
      step = int(qs["step"][0]) if "step" in qs else None
      self._send(200, json.dumps(self._samples(step)))
    elif path.startswith("/samples/"):
      self._serve_image(path[len("/samples/"):])
    elif path.startswith("/tile/"):
      self._serve_tile(path[len("/tile/"):])
    elif path.startswith("/baseimg/"):
      self._serve_base(path[len("/baseimg/"):])
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

  def _base_root(self):
    return self.base_dir or os.path.join(self.run, "base_previews")

  def _prompts(self):
    p = os.path.join(self._samples_root(), "prompts.json")
    if os.path.isfile(p):
      try:
        return {str(k): v for k, v in json.load(open(p, encoding="utf-8")).items()}
      except Exception:
        return {}
    return {}

  @staticmethod
  def _stepof(f):
    m = re.search(r"step(\d+)", f)
    return int(m.group(1)) if m else -1

  def _base_src(self, k):
    p = os.path.join(self._base_root(), f"idx{k}.png")
    return f"baseimg/{k}" if os.path.isfile(p) else None

  def _samples(self, step=None):
    """One card per prompt for a chosen preview step (default latest). ``steps`` lists every
    available preview step for pagination; each item carries its BASE counterpart if present."""
    d = self._samples_root()
    if not os.path.isdir(d):
      return {"step": 0, "steps": [], "items": []}
    prompts = self._prompts()
    imgs = [f for f in os.listdir(d) if f.lower().endswith(_IMG_EXTS)]
    steps = sorted({self._stepof(f) for f in imgs if self._stepof(f) >= 0})
    if not steps:
      return {"step": 0, "steps": [], "items": []}
    target = step if (step in steps) else steps[-1]
    cur = [f for f in imgs if self._stepof(f) == target]
    tiled = sorted(f for f in cur if re.search(r"idx(\d+)", f))
    items = []
    if tiled:
      for f in tiled:
        k = int(re.search(r"idx(\d+)", f).group(1))
        items.append({"src": f"samples/{quote(f)}", "base": self._base_src(k),
                      "prompt": prompts.get(str(k), ""), "idx": k})
    else:
      sheet = sorted(cur)[0]
      n = len(prompts) or 1
      if n <= 1:
        items.append({"src": f"samples/{quote(sheet)}", "base": self._base_src(0),
                      "prompt": prompts.get("0", ""), "idx": 0})
      else:
        for k in range(n):
          items.append({"src": f"tile/{quote(sheet)}/{k}", "base": self._base_src(k),
                        "prompt": prompts.get(str(k), ""), "idx": k})
    return {"step": target, "steps": steps, "items": items}

  def _safe(self, root, name):
    name = unquote(name)
    if "/" in name or "\\" in name or ".." in name:
      return None
    p = os.path.join(root, name)
    return p if os.path.isfile(p) else None

  def _serve_file(self, p):
    ext = os.path.splitext(p)[1].lower()
    with open(p, "rb") as f:
      self._send(200, f.read(), "image/png" if ext == ".png" else "image/jpeg")

  def _serve_image(self, name):
    p = self._safe(self._samples_root(), name)
    self._serve_file(p) if p else self._send(404, "{}")

  def _serve_base(self, k):
    try:
      k = int(k)
    except ValueError:
      return self._send(404, "{}")
    p = os.path.join(self._base_root(), f"idx{k}.png")
    self._serve_file(p) if os.path.isfile(p) else self._send(404, "{}")

  def _serve_tile(self, rest):
    """/tile/<file>/<k> -> crop tile k out of a 4-column contact sheet."""
    try:
      name, k = rest.rsplit("/", 1)
      k = int(k)
    except ValueError:
      return self._send(404, "{}")
    p = self._safe(self._samples_root(), name)
    if not p:
      return self._send(404, "{}")
    try:
      from PIL import Image
      img = Image.open(p).convert("RGB")
      n = max(1, len(self._prompts()))
      cols = _SHEET_COLS
      rows = math.ceil(n / cols)
      W, H = img.size
      tw, th = W // cols, H // rows
      col, row = k % cols, k // cols
      tile = img.crop((col * tw, row * th, col * tw + tw, row * th + th))
      buf = BytesIO()
      tile.save(buf, "PNG")
      self._send(200, buf.getvalue(), "image/png")
    except Exception:
      self._send(404, "{}")


def main():
  ap = argparse.ArgumentParser(description="Minimal live training dashboard.")
  ap.add_argument("--run", required=True, help="run output_dir (holds metrics.jsonl + samples/)")
  ap.add_argument("--samples-dir", default=None, help="override the samples dir")
  ap.add_argument("--base-dir", default=None, help="per-prompt base tiles (idxK.png) for base/preview toggle")
  ap.add_argument("--total", type=int, default=0, help="total steps (progress/ETA; auto from metrics 'total')")
  ap.add_argument("--port", type=int, default=8080)
  ap.add_argument("--host", default="0.0.0.0")
  args = ap.parse_args()
  Handler.run = args.run
  Handler.samples_dir = args.samples_dir
  Handler.base_dir = args.base_dir
  Handler.total = args.total
  srv = ThreadingHTTPServer((args.host, args.port), Handler)
  print(f"[dashboard] {args.run} -> http://{args.host}:{args.port}  (Ctrl-C to stop)", flush=True)
  try:
    srv.serve_forever()
  except KeyboardInterrupt:
    pass


if __name__ == "__main__":
  main()
