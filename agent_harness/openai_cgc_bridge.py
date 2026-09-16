#!/usr/bin/env python3
"""OpenAI -> CGC bridge with MTP support: /v1/chat/completions -> llama-speculative-simple.exe"""
import json, os, subprocess, sys, time, uuid
from http.server import HTTPServer, BaseHTTPRequestHandler

os.environ["PATH"] = "C:/msys64/mingw64/bin;" + os.environ.get("PATH", "")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_BIN = os.path.join(SCRIPT_DIR, "..", "src", "llama.cpp", "build", "bin")

# Binary selection
USE_MTP = os.environ.get("USE_MTP", "1") == "1"
if USE_MTP:
    LLAMA = os.path.join(BUILD_BIN, "llama-speculative-simple.exe")
    DEFAULT_MODEL = "D:/alex/flashkv0516/models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf"
else:
    LLAMA = os.path.join(BUILD_BIN, "llama-simple.exe")
    DEFAULT_MODEL = "D:/alex/flashkv0516/models/gguf/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"

LLAMA = os.environ.get("LLAMA_SIMPLE", LLAMA)
MODEL = os.environ.get("MODEL", DEFAULT_MODEL)
PORT = int(os.environ.get("BRIDGE_PORT", "1234"))

# MTP parameters
MTP_N = int(os.environ.get("MTP_N", "2"))

def build_prompt(messages):
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        parts.append(f"{role.capitalize()}: {content}")
    parts.append("Assistant:")
    return "\n".join(parts)

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/v1/models", "/health"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            if self.path == "/v1/models":
                self.wfile.write(json.dumps({
                    "object": "list",
                    "data": [{"id": os.path.basename(MODEL), "object": "model", "owned_by": "cgc-mtp" if USE_MTP else "cgc"}]
                }).encode())
            else:
                info = {"status": "ok", "mtp": USE_MTP, "model": os.path.basename(MODEL)}
                self.wfile.write(json.dumps(info).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            self.handle_chat()
        else:
            self.send_response(404)
            self.end_headers()

    def handle_chat(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        messages = body.get("messages", [])
        max_tokens = min(body.get("max_tokens", 256), 512)
        temperature = body.get("temperature", 0)
        prompt = build_prompt(messages)

        cmd = [LLAMA, "-m", MODEL, "-p", prompt, "-n", str(max_tokens),
               "--temp", str(temperature), "-c", "2048", "-ngl", "4", "--log-disable"]

        if USE_MTP:
            cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(MTP_N)]

        try:
            t0 = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=300)
            wall = time.time() - t0
            output = (proc.stdout or "").strip()
            # Remove prompt echo
            if output.startswith(prompt[:20]):
                output = output[len(prompt):].strip()
            # Remove thinking tags
            if "<think>" in output:
                parts = output.split("</think>", 1)
                output = parts[-1].strip() if len(parts) > 1 else output
            # Parse stderr for perf stats
            stderr = proc.stderr or ""
            decode_tps = 0
            accept_pct = 0
            for line in stderr.split("\n"):
                if "speed:" in line:
                    try:
                        decode_tps = float(line.split("speed:")[1].split("t/s")[0].strip())
                    except:
                        pass
                if "accept" in line.lower():
                    try:
                        accept_pct = float(line.lower().split("accept")[1].split("%")[0].strip())
                    except:
                        pass
        except Exception as e:
            output = f"<error: {e}>"
            wall = 0
            decode_tps = 0
            accept_pct = 0

        resp = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": os.path.basename(MODEL),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": output},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(output.split()),
                "total_tokens": len(prompt.split()) + len(output.split())
            },
            "cgc_summary": {
                "decode_tps": decode_tps,
                "accept_pct": accept_pct,
                "wall_s": round(wall, 2),
                "mtp": USE_MTP
            }
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(resp, ensure_ascii=False).encode())

    def log_message(self, fmt, *args):
        print(f"[bridge] {args[0]}", flush=True)

if __name__ == "__main__":
    mode = "MTP draft" if USE_MTP else "basic"
    print(f"Bridge on port {PORT}, mode={mode}, model={os.path.basename(MODEL)}")
    print(f"  binary: {LLAMA}")
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
