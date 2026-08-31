#!/usr/bin/env python3
# Copyright (c) 2025 SandAI. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (see pd_server.py header).

"""
CGC Edge Server — 端雲最後一哩（2026-08-29）

把本機的 llama.cpp fork binary（llama-simple / llama-speculative-simple）包成
HTTP 服務，暴露 /v1/cgc/{health, profile, emit, resume} 端點，讓 ComputeRouter
的決策可以真正執行（Router 選誰 → 呼叫誰的 resume → SSE token 流回來）。

設計約束（誠實聲明）：
  1. 純 Python stdlib（http.server + subprocess）— Mac/Windows 零依賴可跑。
  2. Phase 1 =「文本橋」：請求攜帶完整 prompt，被選中節點做「整段推理」。
     真正的 hidden-state 分裂 PD（Mac prefill → hidden/KV 封包 → Windows decode）
     需要 fork C++ 端新增 emit/resume 端點（Phase 2，見指導書路線圖）。
     文本橋在「小模型 + 本地塞不下/跑太慢」場景已提供真實價值：
     Router 直接把整段推理路由到最快節點，用戶端透過 SSE 拿 token 流。
  3. 每個請求 spawn 一個 subprocess（模型重載 ~5-10s/次）。Phase 2 改常駐
     session 工作池。emit 端點的 probe 就包含 load time，Router/用戶可如實看到。
  4. stdout 串流：llama-simple 逐 token printf + fflush（simple.cpp L369-370），
     char-by-char 讀 pipe 轉發即可。stderr 餵背景執行緒解析 perf 行
     （"decoded N tokens in X s, speed: Y t/s"）。

用法（Mac，qwen36 生產配置＝run_n30cache.sh 的 env 集）：
  CGC_EXPERT_CACHE_BYTES=4294967296 LLAMA_EXPERT_CACHE_ALLOW_NGL=1 \
  LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0=1 LLAMA_EXPERT_CACHE_WORKERS=8 \
  CGC_WAKE_POLL_US=15 CGC_PREFETCH_SRC=hist CGC_EVICTED_RING=0 \
  CGC_OA_ASYNC=1 CGC_N_CB=8 \
  python3 CGC-main/cgc_engine/pd/edge_server.py \
      --binary src/llama.cpp/build/bin/llama-simple \
      --model models/gguf/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf \
      --ngl 99 --no-mmap --port 1234

用法（Mac，qwen36 MTP 生產配置＝run_n30cache.sh --steady 的完整槓桿集）：
  CGC_EXPERT_CACHE_BYTES=4294967296 LLAMA_EXPERT_CACHE_ALLOW_NGL=1 \
  LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0=1 LLAMA_EXPERT_CACHE_WORKERS=8 \
  CGC_WAKE_POLL_US=15 CGC_PREFETCH_SRC=hist CGC_EVICTED_RING=0 \
  CGC_OA_ASYNC=1 CGC_N_CB=8 CGC_GLU_FUSED_DOWN=1 \
  LLAMA_EXPERT_CACHE_LAYER_CAPS=40-40:256 CGC_MMV_FUSE=1 \
  python3 CGC-main/cgc_engine/pd/edge_server.py \
      --binary src/llama.cpp/build/bin/llama-speculative-simple \
      --model models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf \
      --ngl 99 --no-mmap --mtp 2 --port 1234
  （--mtp 2 內建 --spec-type draft-mtp --spec-draft-n-max 2 -c 3072
   -expert-cache 4294967296 --temp 0 + MTP 必要 env：CGC_NO_PREFETCH/
  CGC_VERIFY_DECODE/CGC_DRAFT_DECODE/CGC_WATCHDOG，未設才補）

用法（Windows，小模型/低 ngl）：
  py -3 CGC-main\\cgc_engine\\pd\\edge_server.py ^
      --binary src\\llama.cpp\\build\\bin\\llama-simple.exe ^
      --model models\\gguf\\qwen25_7b.gguf --ngl 0 --port 1234

API：
  GET  /v1/cgc/health   → {"ok": true, "model": "...", "binary": "..."}
  GET  /v1/cgc/profile  → DeviceProfile.detect_local() + 本服務配置（餵 Router）
  POST /v1/cgc/emit     body {"prompt": "..."} → prefill 探針（-n 1），
                          回 JSON：load_ms / prefill_ms / prompt_tokens
  POST /v1/cgc/resume   body {"prompt": "...", "max_tokens": N, "seed": S} →
                          SSE 流（event:status → 逐 token data:{"t":...} →
                          event:summary 含 decode_tps）
"""

import atexit
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import error as urlerror
from urllib import request as urlrequest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from discovery import DeviceProfile
except ImportError:  # 單檔可用：profile 端點降級為 OS 基本探測
    DeviceProfile = None

# 相容兩種輸出：llama-simple "in 5.12 s," / speculative-simple "in 41.800 seconds,"
PERF_DECODE = re.compile(r"decoded\s+(\d+)\s+tokens\s+in\s+([\d.]+)\s*s(?:econds?)?\s*,\s*speed:\s*([\d.]+)\s*t/s")
PERF_ACCEPT = re.compile(r"accept\s*=\s*([\d.]+)%")        # speculative-simple
PERF_NDRAFT = re.compile(r"n_drafted\s*=\s*(\d+)")          # speculative-simple
PERF_NACCEPT = re.compile(r"n_accept\s*=\s*(\d+)")          # speculative-simple
PERF_PROMPT_EVAL = re.compile(r"prompt eval time.*?([\d.]+)\s*tokens per second")
PERF_EVAL = re.compile(r"^\s*eval time.*?([\d.]+)\s*tokens per second", re.M)
LOAD_TIME = re.compile(r"load time\s*=\s*(\d+)")


class EdgeConfig:
    def __init__(self):
        self.binary = ""
        self.model = ""
        self.ngl = 0
        self.threads = max(1, (os.cpu_count() or 8) // 2)
        self.no_mmap = False
        self.extra_args = []
        self.worker_mode = "spawn"
        self.worker_binary = ""
        self.worker_host = "127.0.0.1"
        self.worker_port = 2234
        self.worker_parallel = 1
        self.worker_start_timeout = 120
        self.worker_slots = 1

    def resolve_binary(self):
        """Windows 上 build 產物叫 llama-simple.exe；MSYS2/git-bash 傳相對路徑也可。"""
        if os.path.exists(self.binary):
            return self.binary
        for cand in (self.binary + ".exe", os.path.abspath(self.binary) + ".exe"):
            if os.path.exists(cand):
                return cand
        return self.binary

    def resolve_worker_binary(self):
        if self.worker_binary:
            if os.path.exists(self.worker_binary):
                return self.worker_binary
            for cand in (self.worker_binary + ".exe", os.path.abspath(self.worker_binary) + ".exe"):
                if os.path.exists(cand):
                    return cand
            return self.worker_binary

        binary = self.resolve_binary()
        dname = os.path.dirname(binary)
        worker = os.path.join(dname, "llama-server")
        if os.path.exists(worker):
            return worker
        if os.path.exists(worker + ".exe"):
            return worker + ".exe"
        return worker


CFG = EdgeConfig()
WORKER = {
    "proc": None,
    "log_fh": None,
}
WORKER_LOCK = threading.Lock()


def build_cmd(prompt, n_predict, seed=None):
    cmd = [CFG.resolve_binary(), "-m", CFG.model, "-n", str(n_predict),
           "-ngl", str(CFG.ngl), "-t", str(CFG.threads)]
    if CFG.no_mmap:
        cmd += ["--no-mmap"]
    if seed is not None:
        cmd += ["-s", str(seed)]
    cmd += CFG.extra_args
    if prompt:
        cmd += ["-p", prompt]
    return cmd


def build_worker_cmd():
    cmd = [CFG.resolve_worker_binary(), "-m", CFG.model,
           "-ngl", str(CFG.ngl), "-t", str(CFG.threads),
           "--host", CFG.worker_host, "--port", str(CFG.worker_port),
           "--threads-http", "1", "--parallel", str(max(1, CFG.worker_parallel)),
           "--no-webui"]
    if CFG.no_mmap:
        cmd += ["--no-mmap"]
    cmd += CFG.extra_args
    return cmd


def worker_base_url(path):
    return f"http://{CFG.worker_host}:{CFG.worker_port}{path}"


def _worker_log_path():
    return f"/tmp/cgc_edge_worker_{CFG.worker_port}.log"


def _close_worker_log():
    fh = WORKER.get("log_fh")
    if fh is not None:
        try:
            fh.close()
        except Exception:
            pass
        WORKER["log_fh"] = None


def stop_worker():
    with WORKER_LOCK:
        proc = WORKER.get("proc")
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        WORKER["proc"] = None
        _close_worker_log()


atexit.register(stop_worker)


def worker_health_ok():
    try:
        with urlrequest.urlopen(worker_base_url("/health"), timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return bool(data.get("status") == "ok" or data.get("status") == "healthy" or data.get("status") is None)
    except Exception:
        return False


def ensure_worker():
    if CFG.worker_mode != "persistent":
        return True, None

    with WORKER_LOCK:
        proc = WORKER.get("proc")
        if proc is not None and proc.poll() is None and worker_health_ok():
            return True, None

        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        WORKER["proc"] = None
        _close_worker_log()

        log_fh = open(_worker_log_path(), "ab", buffering=0)
        proc = subprocess.Popen(
            build_worker_cmd(),
            stdout=log_fh, stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
        WORKER["proc"] = proc
        WORKER["log_fh"] = log_fh

    deadline = time.time() + max(5, CFG.worker_start_timeout)
    while time.time() < deadline:
        proc = WORKER.get("proc")
        if proc is None:
            break
        rc = proc.poll()
        if rc is not None:
            return False, f"worker exited early rc={rc}, log={_worker_log_path()}"
        if worker_health_ok():
            return True, None
        time.sleep(0.5)
    return False, f"worker start timeout, log={_worker_log_path()}"


def pick_worker_slot(prompt, body):
    slot = int(body.get("id_slot", -1))
    if slot >= 0:
        return slot
    if CFG.worker_slots <= 1:
        return -1
    h = hashlib.sha1(prompt.encode("utf-8", errors="ignore")).hexdigest()
    return int(h[:8], 16) % CFG.worker_slots


def worker_payload(prompt, n_predict, seed, body, stream):
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "stream": stream,
        "cache_prompt": bool(body.get("cache_prompt", True)),
        "timings_per_token": False,
    }
    if seed is not None:
        payload["seed"] = seed
    if body.get("ignore_eos") or "--ignore-eos" in CFG.extra_args:
        payload["ignore_eos"] = True

    slot = pick_worker_slot(prompt, body)
    if slot >= 0:
        payload["id_slot"] = slot

    for src, dst in (
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("top_k", "top_k"),
        ("min_p", "min_p"),
        ("repeat_penalty", "repeat_penalty"),
        ("presence_penalty", "presence_penalty"),
        ("frequency_penalty", "frequency_penalty"),
    ):
        if src in body:
            payload[dst] = body[src]
    if "stop" in body:
        payload["stop"] = body["stop"]
    return payload


def worker_summary_from_obj(obj):
    timings = obj.get("timings") or {}
    out = {
        "rc": 0,
        "tokens_cached": obj.get("tokens_cached"),
        "tokens_evaluated": obj.get("tokens_evaluated"),
        "slot_id": obj.get("id_slot", obj.get("slot_id")),
        "stop_type": obj.get("stop_type"),
    }
    if "predicted_n" in timings:
        out["n_decoded"] = int(timings["predicted_n"])
    if "predicted_ms" in timings:
        out["decode_s"] = float(timings["predicted_ms"]) / 1000.0
    if "predicted_per_second" in timings:
        out["decode_tps"] = float(timings["predicted_per_second"])
    if "prompt_per_second" in timings:
        out["prompt_tps"] = float(timings["prompt_per_second"])
    if "load_ms" in timings:
        out["load_ms"] = int(round(float(timings["load_ms"])))
    return out


def run_generate_spawn(prompt, n_predict, seed=None):
    """yield ('status'|'token'|'summary', obj) — 真實 subprocess 串流。"""
    t0 = time.time()
    yield ("status", {"stage": "loading", "cmd_n": n_predict, "mode": "spawn"})
    proc = subprocess.Popen(
        build_cmd(prompt, n_predict, seed),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
        bufsize=1, env=os.environ.copy(),
    )
    err = StderrCollector(proc.stderr)
    err.start()
    # stdout：prompt 回顯 + 生成 token，char-by-char 轉發（llama-simple fflush 每 token）
    buf = []
    while True:
        ch = proc.stdout.read(1)
        if ch:
            buf.append(ch)
            yield ("token", {"t": ch})
        elif proc.poll() is not None:
            break
        else:
            time.sleep(0.005)
    proc.stdout.close()
    rc = proc.wait()
    err.join(timeout=5)
    perf = err.perf()
    perf.update({"rc": rc, "wall_s": round(time.time() - t0, 2),
                 "stderr_tail": err.text[-15:], "mode": "spawn"})
    yield ("summary", perf)


def run_generate_worker(prompt, n_predict, seed=None, body=None):
    """yield ('status'|'token'|'summary', obj) — persistent llama-server worker."""
    body = body or {}
    t0 = time.time()
    ok, err = ensure_worker()
    if not ok:
        yield ("summary", {"rc": 1, "error": err, "mode": "persistent", "wall_s": round(time.time() - t0, 2)})
        return

    payload = worker_payload(prompt, n_predict, seed, body, stream=self_path_is_resume(body))
    yield ("status", {"stage": "worker_ready", "cmd_n": n_predict, "mode": "persistent",
                      "worker": worker_base_url("/completion"), "slot_id": payload.get("id_slot", -1)})

    if not payload["stream"]:
        req = urlrequest.Request(
            worker_base_url("/completion"),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=600) as resp:
                obj = json.loads(resp.read().decode("utf-8"))
        except urlerror.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            yield ("summary", {"rc": 1, "error": f"worker http {e.code}: {detail}",
                               "mode": "persistent", "wall_s": round(time.time() - t0, 2)})
            return
        except Exception as e:
            yield ("summary", {"rc": 1, "error": f"worker request failed: {e}",
                               "mode": "persistent", "wall_s": round(time.time() - t0, 2)})
            return

        content = obj.get("content", "")
        for ch in content:
            yield ("token", {"t": ch})
        perf = worker_summary_from_obj(obj)
        perf.update({"wall_s": round(time.time() - t0, 2), "mode": "persistent"})
        yield ("summary", perf)
        return

    req = urlrequest.Request(
        worker_base_url("/completion"),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    final_obj = None
    try:
        with urlrequest.urlopen(req, timeout=3600) as resp:
            for raw in resp:
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace").strip()
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    continue
                obj = json.loads(data)
                final_obj = obj
                text = obj.get("content", "")
                if text:
                    for ch in text:
                        yield ("token", {"t": ch})
    except urlerror.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        yield ("summary", {"rc": 1, "error": f"worker stream http {e.code}: {detail}",
                           "mode": "persistent", "wall_s": round(time.time() - t0, 2)})
        return
    except Exception as e:
        yield ("summary", {"rc": 1, "error": f"worker stream failed: {e}",
                           "mode": "persistent", "wall_s": round(time.time() - t0, 2)})
        return

    perf = worker_summary_from_obj(final_obj or {})
    perf.update({"wall_s": round(time.time() - t0, 2), "mode": "persistent"})
    yield ("summary", perf)


def self_path_is_resume(body):
    return bool(body.get("__stream__", False))


class StderrCollector(threading.Thread):
    """背景執行緒收 stderr（perf 行 + load time），主執行緒同時串流 stdout。"""

    def __init__(self, pipe):
        super().__init__(daemon=True)
        self.pipe = pipe
        self.text = []

    def run(self):
        for line in self.pipe:
            self.text.append(line.rstrip("\n"))
        self.pipe.close()

    def perf(self):
        blob = "\n".join(self.text)
        out = {}
        m = PERF_DECODE.search(blob)
        if m:
            out["n_decoded"] = int(m.group(1))
            out["decode_s"] = float(m.group(2))
            out["decode_tps"] = float(m.group(3))
        m = PERF_ACCEPT.search(blob)
        if m:
            out["accept_pct"] = float(m.group(1))
        m = PERF_NDRAFT.search(blob)
        if m:
            out["n_drafted"] = int(m.group(1))
        m = PERF_NACCEPT.search(blob)
        if m:
            out["n_accept"] = int(m.group(1))
        m = PERF_PROMPT_EVAL.search(blob)
        if m:
            out["prompt_tps"] = float(m.group(1))
        m = LOAD_TIME.search(blob)
        if m:
            out["load_ms"] = int(m.group(1))
        return out


def run_generate(prompt, n_predict, seed=None, body=None):
    if CFG.worker_mode == "persistent":
        yield from run_generate_worker(prompt, n_predict, seed, body)
    else:
        yield from run_generate_spawn(prompt, n_predict, seed)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"  # close-delimited：SSE 免 chunked 編碼

    def log_message(self, fmt, *args):  # 安靜模式（token 流不刷屏）
        if os.environ.get("CGC_EDGE_VERBOSE"):
            super().log_message(fmt, *args)

    # ── GET ──
    def do_GET(self):
        if self.path == "/v1/cgc/health":
            self._json({"ok": True, "model": CFG.model,
                        "binary": CFG.binary, "ngl": CFG.ngl,
                        "worker_mode": CFG.worker_mode,
                        "worker_binary": CFG.resolve_worker_binary() if CFG.worker_mode == "persistent" else ""})
        elif self.path == "/v1/cgc/profile":
            prof = {}
            if DeviceProfile is not None:
                try:
                    prof = DeviceProfile.detect_local().__dict__
                except Exception as e:  # noqa: BLE001 — 探測失敗不炸服務
                    prof = {"detect_error": str(e)}
            self._json({"profile": prof, "model": CFG.model,
                        "ngl": CFG.ngl, "threads": CFG.threads,
                        "worker_mode": CFG.worker_mode,
                        "worker_slots": CFG.worker_slots})
        else:
            self._json({"error": "not found"}, 404)

    # ── POST ──
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            self._json({"error": "bad json"}, 400)
            return
        prompt = str(body.get("prompt", ""))
        if not prompt:
            self._json({"error": "prompt required"}, 400)
            return

        if self.path == "/v1/cgc/emit":
            # prefill 探針：-n 1（載入 + prefill + 首 token），不串流
            result = {}
            body["__stream__"] = False
            for kind, obj in run_generate(prompt, 1, body.get("seed"), body):
                if kind == "summary":
                    result = obj
            self._json({"ok": result.get("rc") == 0, "emit": result})
        elif self.path == "/v1/cgc/resume":
            body["__stream__"] = True
            self._sse(prompt, int(body.get("max_tokens", 32)), body)
        else:
            self._json({"error": "not found"}, 404)

    # ── helpers ──
    def _json(self, obj, code=200):
        blob = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def _sse(self, prompt, n_predict, body):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            for kind, obj in run_generate(prompt, n_predict, body.get("seed"), body):
                payload = dict(obj)
                payload["event"] = kind
                self.wfile.write(b"data: " +
                                 json.dumps(payload, ensure_ascii=False).encode("utf-8") +
                                 b"\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # 客戶端斷線：讓 subprocess 自然結束（phase 2 可加 kill）


def main():
    ap = argparse.ArgumentParser(description="CGC edge server (text-bridge PD, phase 1)")
    ap.add_argument("--binary", required=True, help="llama-simple(.exe) or llama-speculative-simple")
    ap.add_argument("--model", required=True, help="GGUF path")
    ap.add_argument("--ngl", type=int, default=0)
    ap.add_argument("--threads", "-t", type=int, default=None)
    ap.add_argument("--no-mmap", action="store_true", help="Mac 凍機防護（生產配置）")
    ap.add_argument("--extra", nargs="*", default=[], help="透傳 binary 的額外旗標（注意 argparse 會吃掉 -- 開頭 token，MTP 請改用 --mtp）")
    ap.add_argument("--worker-mode", choices=["spawn", "persistent"], default="spawn",
                    help="執行模式：spawn=每請求重載（legacy，可完全回退），persistent=內部 llama-server 常駐 worker")
    ap.add_argument("--worker-binary", default="",
                    help="persistent 模式的 worker binary（預設同目錄下的 llama-server）")
    ap.add_argument("--worker-port", type=int, default=2234,
                    help="persistent worker 監聽埠（僅 localhost）")
    ap.add_argument("--worker-parallel", type=int, default=1,
                    help="persistent worker 的 server slots/parallel 數")
    ap.add_argument("--worker-start-timeout", type=int, default=120,
                    help="等待 persistent worker 就緒的秒數")
    ap.add_argument("--worker-slots", type=int, default=1,
                    help="edge 層為 persistent worker 做的 session slot 數（>1 時依 prompt hash 分配）")
    # §MTP 生產模式（run_n30cache.sh §MTP 定案參數集）：speculative-simple 只認
    # CLI -expert-cache（不讀 CGC_EXPERT_CACHE_BYTES env），且 MTP 需要 greedy
    # (--temp 0) + 關 prefetch + verify/draft decode fast path，缺任何一個都會
    # accept 崩掉或 verify race（歷史教訓）。
    ap.add_argument("--mtp", type=int, default=0, metavar="N",
                    help="MTP draft-mtp 生產模式（N=spec-draft-n-max，建議 2）：內建 --spec-type draft-mtp "
                         "-c MTP_CTX -expert-cache BUDGET --temp 0，並自動補 MTP 必要 env")
    ap.add_argument("--mtp-ctx", type=int, default=3072, help="MTP draft context（OOM 邊界，勿超）")
    ap.add_argument("--budget", type=int, default=4294967296,
                    help="MTP 模式 -expert-cache bytes（llama-simple 模式走 CGC_EXPERT_CACHE_BYTES env）")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=1234)
    args = ap.parse_args()

    CFG.binary, CFG.model, CFG.ngl = args.binary, args.model, args.ngl
    CFG.no_mmap, CFG.extra_args = args.no_mmap, args.extra
    CFG.worker_mode = args.worker_mode
    CFG.worker_binary = args.worker_binary
    CFG.worker_port = args.worker_port
    CFG.worker_parallel = max(1, args.worker_parallel)
    CFG.worker_start_timeout = max(5, args.worker_start_timeout)
    CFG.worker_slots = max(1, args.worker_slots)
    if args.threads:
        CFG.threads = args.threads

    if args.mtp:
        CFG.extra_args += ["--spec-type", "draft-mtp",
                           "--spec-draft-n-max", str(args.mtp),
                           "-c", str(args.mtp_ctx),
                           "-expert-cache", str(args.budget),
                           "--temp", "0"]
        # MTP 必要 env（run_n30cache.sh §MTP）：launch 已設者不覆蓋
        for k, v in (("CGC_NO_PREFETCH", "1"),     # prefetch bg race 於 MTP verify
                     ("CGC_VERIFY_DECODE", "1"),   # verify 走 decode fast path
                     ("CGC_DRAFT_DECODE", "1"),    # draft 同 pool residency
                     ("CGC_WATCHDOG", "1")):       # 長跑 lost-wakeup 自救
            os.environ.setdefault(k, v)

    bin_path = CFG.resolve_binary()
    need_files = [bin_path, CFG.model]
    if CFG.worker_mode == "persistent":
        need_files.append(CFG.resolve_worker_binary())
    for f in need_files:
        if not os.path.exists(f):
            print(f"error: not found: {f}", file=sys.stderr)
            sys.exit(2)

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"=== CGC edge server (text-bridge phase 1) ===")
    print(f"  binary : {bin_path}")
    print(f"  model  : {CFG.model}  ngl={CFG.ngl} t={CFG.threads} no_mmap={CFG.no_mmap}")
    print(f"  mode   : {CFG.worker_mode}")
    if args.mtp:
        print(f"  mtp    : ON (draft-mtp, n_max={args.mtp}, ctx={args.mtp_ctx}, "
              f"expert-cache={args.budget}, temp=0)")
    if CFG.worker_mode == "persistent":
        print(f"  worker : {CFG.resolve_worker_binary()} @ {CFG.worker_host}:{CFG.worker_port} "
              f"parallel={CFG.worker_parallel} slots={CFG.worker_slots}")
    print(f"  listen : http://{args.host}:{args.port}/v1/cgc/{{health,profile,emit,resume}}")
    if CFG.worker_mode == "persistent":
        print(f"  note   : edge 保留原 API；底層改用常駐 llama-server worker（可用 --worker-mode spawn 回退）")
    else:
        print(f"  note   : 每請求重載模型（phase 1）；hidden-state 分裂 PD 見指導書 Phase 2")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
