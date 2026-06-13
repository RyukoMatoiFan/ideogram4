"""Tiny web gallery to eyeball a tag-captioned image dataset (folder of <id>.jpg + <id>.txt).

Serves N random images per page with their captions; the Shuffle button reloads a fresh
random set. Stdlib only. View over SSH:  ssh -L 8095:localhost:8095 <host>  then open
http://localhost:8095

  python dataset_browser.py --dir /path/to/images --port 8095 --n 12
"""
import argparse
import html
import os
import random
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ARGS = None
_INDEX = []  # cached list of image paths RELATIVE to ARGS.dir (spans subfolders)


def _scan(d, exts):
  rels = []
  for root, _dirs, fnames in os.walk(d):
    for name in fnames:
      ext = os.path.splitext(name)[1].lower().lstrip(".")
      if ext in exts:
        rels.append(os.path.relpath(os.path.join(root, name), d))
  return rels


class H(BaseHTTPRequestHandler):
  def log_message(self, *a):
    pass

  def do_GET(self):
    u = urllib.parse.urlparse(self.path)
    if u.path == "/img":
      q = urllib.parse.parse_qs(u.query)
      rel = q.get("f", [""])[0]
      p = os.path.normpath(os.path.join(ARGS.dir, rel))
      if not p.startswith(os.path.normpath(ARGS.dir)) or not os.path.isfile(p):
        self.send_error(404); return
      ext = os.path.splitext(p)[1].lower().lstrip(".")
      self.send_response(200)
      self.send_header("Content-Type", f"image/{'jpeg' if ext in ('jpg','jpeg') else ext}")
      self.end_headers()
      with open(p, "rb") as f:
        self.wfile.write(f.read())
      return

    # index page: N random samples
    pick = random.sample(_INDEX, min(ARGS.n, len(_INDEX)))
    cards = []
    for rel in pick:
      cap = ""
      tp = os.path.join(ARGS.dir, os.path.splitext(rel)[0] + ".txt")
      if os.path.exists(tp):
        cap = open(tp, encoding="utf-8", errors="ignore").read().strip()
      label = os.path.basename(rel)
      cards.append(
        f'<div class=card><img loading=lazy src="/img?f={urllib.parse.quote(rel)}">'
        f'<div class=id>{html.escape(label)}</div>'
        f'<div class=cap>{html.escape(cap)}</div></div>')
    body = f"""<!doctype html><html><head><meta charset=utf-8><title>dataset</title>
<style>
body{{background:#111;color:#ddd;font:13px system-ui;margin:0;padding:16px}}
.bar{{position:sticky;top:0;background:#111;padding:8px 0;display:flex;gap:12px;align-items:center}}
button{{background:#2a6;color:#fff;border:0;padding:8px 16px;border-radius:6px;font-size:15px;cursor:pointer}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:14px;margin-top:12px}}
.card{{background:#1c1c1c;border-radius:8px;overflow:hidden}}
.card img{{width:100%;display:block;background:#000}}
.id{{padding:4px 8px;color:#888;font-size:11px}}
.cap{{padding:4px 8px 10px;color:#bcd;font-size:11px;line-height:1.35;max-height:120px;overflow:auto}}
</style></head><body>
<div class=bar><button onclick="location.reload()">\U0001F3B2 Shuffle</button>
<span>{len(_INDEX):,} images in {html.escape(ARGS.dir)} — showing {len(pick)} random</span></div>
<div class=grid>{''.join(cards)}</div></body></html>"""
    self.send_response(200)
    self.send_header("Content-Type", "text/html; charset=utf-8")
    self.end_headers()
    self.wfile.write(body.encode("utf-8"))


def main():
  global ARGS, _INDEX
  ap = argparse.ArgumentParser(description="Web gallery for a tag-captioned image folder.")
  ap.add_argument("--dir", required=True)
  ap.add_argument("--port", type=int, default=8095)
  ap.add_argument("--n", type=int, default=12)
  ap.add_argument("--exts", default="jpg,jpeg,png,webp")
  ARGS = ap.parse_args()
  print(f"[browser] scanning {ARGS.dir} ...", flush=True)
  _INDEX = _scan(ARGS.dir, set(ARGS.exts.split(",")))
  print(f"[browser] {len(_INDEX)} images. serving on :{ARGS.port} "
        f"(ssh -L {ARGS.port}:localhost:{ARGS.port} <host>, then http://localhost:{ARGS.port})", flush=True)
  ThreadingHTTPServer(("0.0.0.0", ARGS.port), H).serve_forever()


if __name__ == "__main__":
  main()
