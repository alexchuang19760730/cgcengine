#!/usr/bin/env python3
import argparse, json, uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.request import Request, urlopen
from urllib.error import URLError

NL = chr(10)
DQ = chr(34)

def build_prompt(msgs, sys_txt=""):
    p = []
    if sys_txt:
        if isinstance(sys_txt, list):
            sys_txt = NL.join(b.get("text","") for b in sys_txt if isinstance(b, dict))
        p.append("[system]" + NL + sys_txt)
    for m in msgs:
        c = m.get("content","")
        if isinstance(c, list):
            c = NL.join(b.get("text","") for b in c if isinstance(b, dict) and b.get("type")=="text")
        p.append("[" + m.get("role","user") + "]" + NL + str(c))
    return NL.join(p)

class H(BaseHTTPRequestHandler):
    cgc = "http://192.168.101.87:1237"
    model = "qwen3.6-35b-mtp"
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path == "/v1/models":
            self.send_response(200)
            self.send_header("Content-Type","application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"data":[{"id":self.model}]}).encode())
        else:
            self.send_response(404)
            self.end_headers()
    def do_POST(self):
        if self.path != "/v1/messages":
            self.send_response(404)
            self.end_headers()
            return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length",0))))
        prompt = build_prompt(body.get("messages",[]), body.get("system",""))
        mx = body.get("max_tokens",4096)
        stream = body.get("stream",False)
        print(f"[proxy] prompt_len={len(prompt)} stream={stream}",flush=True)
        req = Request(self.cgc+"/v1/cgc/resume",json.dumps({"prompt":prompt,"max_tokens":mx,"seed":42}).encode(),{"Content-Type":"application/json"},method="POST")
        try:
            resp = urlopen(req,600)
        except URLError as e:
            self.send_response(502)
            self.end_headers()
            self.wfile.write(json.dumps({"error":str(e)}).encode())
            return
        if not stream:
            txt = ""
            for ln in resp:
                s = ln.decode("utf-8",errors="replace").strip()
                if not s.startswith("data: "): continue
                try: ev = json.loads(s[6:])
                except: continue
                if ev.get("event")=="token": txt += ev.get("t","")
            self.send_response(200)
            self.send_header("Content-Type","application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"id":"msg_"+uuid.uuid4().hex[:24],"type":"message","role":"assistant","content":[{"type":"text","text":txt}],"model":self.model,"stop_reason":"end_turn","usage":{"input_tokens":0,"output_tokens":len(txt.split())}}).encode())
        else:
            self.send_response(200)
            self.send_header("Content-Type","text/event-stream")
            self.send_header("Cache-Control","no-cache")
            self.end_headers()
            mid = "msg_"+uuid.uuid4().hex[:24]
            def se(e, d):
                return f"event: {e}
data: {json.dumps(d)}

"
            self.wfile.write(se("message_start",{"type":"message","id":mid,"role":"assistant","content":[],"model":self.model,"stop_reason":None,"usage":{"input_tokens":0,"output_tokens":0}}).encode())
            self.wfile.write(se("content_block_start",{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}).encode())
            self.wfile.flush()
            for ln in resp:
                s = ln.decode("utf-8",errors="replace").strip()
                if not s.startswith("data: "): continue
                try: ev = json.loads(s[6:])
                except: continue
                if ev.get("event")=="token":
                    self.wfile.write(se("content_block_delta",{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":ev.get("t","")}}).encode())
                    self.wfile.flush()
                elif ev.get("event")=="summary":
                    print(f"[proxy] done: {ev.get('n_decoded',0)} tok {ev.get('decode_tps',0):.1f} t/s",flush=True)
            self.wfile.write(se("content_block_stop",{"type":"content_block_stop","index":0}).encode())
            self.wfile.write(se("message_delta",{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":0}}).encode())
            self.wfile.write(se("message_stop",{"type":"message_stop"}).encode())
            self.wfile.flush()

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port",type=int,default=8082)
    ap.add_argument("--cgc-url",default="http://192.168.101.87:1237")
    a = ap.parse_args()
    H.cgc = a.cgc_url
    print(f"[proxy] Anthropic->CGC on :{a.port} -> {a.cgc_url}",flush=True)
    HTTPServer(("127.0.0.1",a.port),H).serve_forever()
