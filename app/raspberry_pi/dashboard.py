"""Dependency-free local server for ASTRA's browser dashboard."""
from __future__ import annotations
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]/"frontend"; STATE=Path.home()/".local/state/me2-astra.json"
class H(SimpleHTTPRequestHandler):
 def __init__(self,*a,**kw): super().__init__(*a,directory=str(ROOT),**kw)
 def do_GET(self):
  if self.path=="/api/state":
   data=STATE.read_text() if STATE.exists() else "{}"; self.send_response(200); self.send_header("Content-Type","application/json"); self.send_header("Cache-Control","no-store"); self.end_headers(); self.wfile.write(data.encode()); return
  if self.path=="/": self.path="/index.html"
  super().do_GET()
 def end_headers(self): self.send_header("Cache-Control","no-store"); super().end_headers()
 def log_message(self,*a): pass
if __name__=="__main__": ThreadingHTTPServer(("0.0.0.0",8000),H).serve_forever()
