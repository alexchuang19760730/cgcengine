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
   -expert-cache 4294967296 --temp 0 + env：CGC_NO_PREFETCH/CGC_WATCHDOG。
   2026-08-30 0000 根因定案後預設走 exact 補槽路徑——舊 CGC_VERIFY/DRAFT_DECODE
   fast path 對 cold expert 讀零權重 → 輸出 0000，僅 --mtp-fast 速度實驗用）

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

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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

    def resolve_binary(self):
        """Windows 上 build 產物叫 llama-simple.exe；MSYS2/git-bash 傳相對路徑也可。"""
        if os.path.exists(self.binary):
            return self.binary
        for cand in (self.binary + ".exe", os.path.abspath(self.binary) + ".exe"):
            if os.path.exists(cand):
                return cand
        return self.binary


CFG = EdgeConfig()


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


def run_generate(prompt, n_predict, seed=None):
    """yield ('status'|'token'|'summary', obj) — 真實 subprocess 串流。"""
    t0 = time.time()
    yield ("status", {"stage": "loading", "cmd_n": n_predict})
    proc = subprocess.Popen(
        build_cmd(prompt, n_predict, seed),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
        bufsize=1, env=os.environ.copy(),
    )
    err = StderrCollector(proc.stderr)
    err.start()
    # stdout：prompt 回顯 + 生成 token，char-by-char 轉發（llama-simple fflush 每 token）。
    # 2026-08-29 修正 log 污染：tool 會把 'main: ' 前綴 log 行（chunked prefill 等）印到
    # stdout，混進 token 流 → client 端 gen 帶 log 行 + echo 偵測失效。逐行判定：
    # 行首 6 字元內匹配 'main: ' = log 行整行丟棄；其餘照常轉發（僅行首延遲 ≤6 字元）。
    LOG_PREFIX = "main: "
    pending = ""       # 行首待判定緩衝（None = 一般模式，直接轉發）
    log_line = False   # 目前在丟棄中的 log 行
    while True:
        ch = proc.stdout.read(1)
        if ch:
            if log_line:
                if ch == "\n":
                    log_line, pending = False, ""
                continue
            if pending is not None:
                pending += ch
                if LOG_PREFIX.startswith(pending):
                    if pending == LOG_PREFIX:
                        log_line, pending = True, None
                    continue
                for c in pending:     # 非 log 前綴 → 緩衝一次發出
                    yield ("token", {"t": c})
                pending = None
            else:
                yield ("token", {"t": ch})
            if ch == "\n" and pending is None:
                pending = ""           # 新行重新進入判定模式
        elif proc.poll() is not None:
            break
        else:
            time.sleep(0.005)
    if pending:                        # EOF 尾行無換行：非 log 才補發
        for c in pending:
            yield ("token", {"t": c})
    proc.stdout.close()
    rc = proc.wait()
    err.join(timeout=5)
    perf = err.perf()
    perf.update({"rc": rc, "wall_s": round(time.time() - t0, 2),
                 "stderr_tail": err.text[-15:]})
    yield ("summary", perf)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"  # close-delimited：SSE 免 chunked 編碼

    def log_message(self, fmt, *args):  # 安靜模式（token 流不刷屏）
        if os.environ.get("CGC_EDGE_VERBOSE"):
            super().log_message(fmt, *args)

    # ── GET ──
    def do_GET(self):
        if self.path == "/v1/cgc/health":
            self._json({"ok": True, "model": CFG.model,
                        "binary": CFG.binary, "ngl": CFG.ngl})
        elif self.path == "/v1/cgc/profile":
            prof = {}
            if DeviceProfile is not None:
                try:
                    prof = DeviceProfile.detect_local().__dict__
                except Exception as e:  # noqa: BLE001 — 探測失敗不炸服務
                    prof = {"detect_error": str(e)}
            self._json({"profile": prof, "model": CFG.model,
                        "ngl": CFG.ngl, "threads": CFG.threads})
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
            for kind, obj in run_generate(prompt, 1, body.get("seed")):
                if kind == "summary":
                    result = obj
            self._json({"ok": result.get("rc") == 0, "emit": result})
        elif self.path == "/v1/cgc/resume":
            self._sse(prompt, int(body.get("max_tokens", 32)), body.get("seed"))
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

    def _sse(self, prompt, n_predict, seed):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            for kind, obj in run_generate(prompt, n_predict, seed):
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
    # §MTP 生產模式（run_n30cache.sh §MTP 定案參數集）：speculative-simple 只認
    # CLI -expert-cache（不讀 CGC_EXPERT_CACHE_BYTES env），且 MTP 需要 greedy
    # (--temp 0) + 關 prefetch（歷史教訓）。
    # [2026-08-30 0000 根因定案] 舊版自動補的 CGC_VERIFY_DECODE/CGC_DRAFT_DECODE 會讓
    # verify/draft 走 touch+ZERO-slot fast path：cold expert（pool 未駐留）映射到零權重
    # slot → FFN 貢獻為 0 → logits 全錯 → greedy 輸出 0000…（穩態 cold 實測 ~65-70%，
    # llama-context.cpp fast path 註解自證；draft/verify 同錯 → accept 率反而虛高 97%）。
    # 修正：--mtp 預設 exact 路徑（ensure_batch 同步補槽 = 正確權重）。--mtp-fast 才回
    # 舊 fast path（僅速度實驗，輸出會 0000）。
    ap.add_argument("--mtp", type=int, default=0, metavar="N",
                    help="MTP draft-mtp 生產模式（N=spec-draft-n-max，建議 2）：內建 --spec-type draft-mtp "
                         "-c MTP_CTX -expert-cache BUDGET --temp 0（verify/draft 走 exact 補槽路徑）")
    ap.add_argument("--mtp-fast", action="store_true",
                    help="MTP 走 touch+ZERO-slot fast path（舊行為：快但 cold expert 讀零權重 → 輸出退化 0000，僅速度實驗用）")
    ap.add_argument("--mtp-prefetch", action="store_true",
                    help="MTP exact 路徑 + hist 預取（bg 補下一步 union，verify 不卡同步 pread；exact 路徑才記錄 union）")
    ap.add_argument("--mtp-ctx", type=int, default=3072, help="MTP draft context（OOM 邊界，勿超）")
    ap.add_argument("--budget", type=int, default=4294967296,
                    help="MTP 模式 -expert-cache bytes（llama-simple 模式走 CGC_EXPERT_CACHE_BYTES env）")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=1234)
    args = ap.parse_args()

    CFG.binary, CFG.model, CFG.ngl = args.binary, args.model, args.ngl
    CFG.no_mmap, CFG.extra_args = args.no_mmap, args.extra
    if args.threads:
        CFG.threads = args.threads

    if args.mtp:
        CFG.extra_args += ["--spec-type", "draft-mtp",
                           "--spec-draft-n-max", str(args.mtp),
                           "-c", str(args.mtp_ctx),
                           "-expert-cache", str(args.budget),
                           "--temp", "0"]
        # MTP env（run_n30cache.sh §MTP）：launch 已設者不覆蓋。
        # [2026-08-30 0000 根因] 不再自動補 CGC_VERIFY_DECODE/CGC_DRAFT_DECODE：
        # 該 fast path 對 cold expert 讀零權重 → greedy 輸出 0000（見 --mtp-fast 註解）。
        # exact 路徑 = ensure_batch 同步補槽，輸出正確。
        # --mtp-prefetch：exact + hist 預取（bg 補下一步 union → verify 不卡同步 pread；
        # fast path 提前 return 不記錄 union，預取無效）。舊「prefetch bg race 於 MTP
        # verify」教訓屬 fast path 時代；exact 與非 MTP 生產線同機制，死鎖已由 exact
        # batch_owned mask 修正（commit 4a746c724）。無預取實測：1.8 t/s（pread 佔滿）。
        for k, v in (("CGC_WATCHDOG", "1"),):      # 長跑 lost-wakeup 自救
            os.environ.setdefault(k, v)
        if args.mtp_prefetch and not args.mtp_fast:
            os.environ["CGC_PREFETCH_SRC"] = "hist"          # 覆蓋 launch env，強制開
            os.environ.pop("CGC_NO_PREFETCH", None)
        else:
            os.environ.setdefault("CGC_NO_PREFETCH", "1")    # 保守預設：關預取
        if args.mtp_fast:                          # 舊 fast path（速度實驗用；輸出會 0000）
            os.environ.setdefault("CGC_VERIFY_DECODE", "1")
            os.environ.setdefault("CGC_DRAFT_DECODE", "1")

    bin_path = CFG.resolve_binary()
    for f in (bin_path, CFG.model):
        if not os.path.exists(f):
            print(f"error: not found: {f}", file=sys.stderr)
            sys.exit(2)

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"=== CGC edge server (text-bridge phase 1) ===")
    print(f"  binary : {bin_path}")
    print(f"  model  : {CFG.model}  ngl={CFG.ngl} t={CFG.threads} no_mmap={CFG.no_mmap}")
    if args.mtp:
        print(f"  mtp    : ON (draft-mtp, n_max={args.mtp}, ctx={args.mtp_ctx}, "
              f"expert-cache={args.budget}, temp=0)")
    print(f"  listen : http://{args.host}:{args.port}/v1/cgc/{{health,profile,emit,resume}}")
    print(f"  note   : 每請求重載模型（phase 1）；hidden-state 分裂 PD 見指導書 Phase 2")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
