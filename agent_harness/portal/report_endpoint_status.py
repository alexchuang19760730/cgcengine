#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在「任何一個端點」產生一份符合契約的狀態上報。

為什麼是檔案而不是協議
----------------------
鴻蒙端與 Windows 端目前不在我們的網路上 ⇒ 任何即時協議都不成立。唯一在它們
**真的離線時**仍然成立的機制是 store-and-forward：端點把狀態寫成 JSON，由既有的
    - scp  ：deploy-harmonyos/deploy-to-harmonyos.sh
    - git  ：Windows 的 agent_harness/scripts/auto_git_push.ps1（監看 agent_harness/）
帶回來。落在 `agent_harness/portal/endpoints/<id>.json` 就會被入口讀到。

這個檔必須能在沒有這個 repo、沒有第三方套件的機器上跑 ⇒ **只用標準函式庫**，
不 import 任何 repo 模組。若該端點連 Python 都沒有，就照 `endpoints/README.md`
的欄位手寫同一份 JSON —— 契約是欄位，不是這支腳本。

用法
----
    python3 report_endpoint_status.py --id windows-rtx4090 \
        --what-ran "建 CUDA 版（GGML_CUDA=ON）" \
        --what-ran "跑 llama-bench，Qwen3.6-35B-A3B IQ3_XXS" \
        --metric decode_tps=52.3 --metric prefill_tps=487.1 \
        --gpu "RTX 4090 24GB" --build-profile "VS2022 + CUDA 12.x" \
        --not-measured "沒有量過 M1/M2/M3 oracle 身分"

    --print                       只印不寫檔
    --out-dir DIR                 指定輸出目錄（預設：本檔旁的 endpoints/）
    --claim-nothing-unmeasured    明確宣告「這個端點沒有未量的東西」
    --self-test                   黑箱自測（7 格）
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = HERE / "endpoints"
CONTRACT_VERSION = 1

REQUIRED = ["endpoint_id", "reported_at", "platform", "arch", "what_ran",
            "capabilities", "not_measured", "contract_version"]


def now_iso() -> str:
    """本機時區的 ISO 時間。入口用字串比較排序，所以格式必須固定。"""
    return datetime.now(timezone(timedelta(seconds=0))).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def parse_metrics(pairs: list[str]) -> dict:
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise ValueError(f"--metric 要寫成 k=v（收到 {p!r}）")
        k, v = p.split("=", 1)
        k = k.strip()
        if not k:
            raise ValueError(f"--metric 的鍵不能是空的（收到 {p!r}）")
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = v
    return out


EDGE_STATUS_OBJECT = "powerauto.edge.status"


def fetch_edge_status(src: str, timeout: float = 5.0) -> dict:
    """src 可以是檔案路徑，或 http(s) URL（例：http://192.168.101.90:8080/v1/edge/status）。

    ★ URL 一律用**繞代理**的 opener。環境裡的 HTTP_PROXY（本機實測
      http://127.0.0.1:7897）會把 http://192.168.101.90:… 也丟進代理，
      於是「去連我們自己的區網入口」靜默失敗 —— 而失敗長得像「端點沒在跑」。
    """
    if src.startswith("http://") or src.startswith("https://"):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(src, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    return json.loads(Path(src).read_text(encoding="utf-8"))


def derive_what_ran(st: dict) -> list[str]:
    """★ 從 edge_server **真的觀察到**的東西導出 what_ran。

    契約要求 what_ran 不得為空，而 edge 端唯一誠實的來源就是它自己的觀測：
    啟動時間、worker 模式與可達性、已服務計數、上一次實際走過的 chat 路由、模型是否存在、
    探測到的旗標數。**每一條都能回溯到 status 裡的一個欄位** —— 這裡不推測、不補話。
    """
    served = st.get("served") or {}
    worker = st.get("worker") or {}
    model = st.get("model") or {}
    ls = st.get("llama_server") or {}
    out = [
        f"edge_server {st.get('src_version')} 於 {st.get('started_at')} 啟動"
        f"（listen {st.get('listen')}；worker 模式 {worker.get('mode')}）",
        f"worker（{worker.get('url')}）：{worker.get('detail')}",
        f"端點已服務 chat={served.get('chat', 0)}、completions={served.get('completions', 0)}、"
        f"emit={served.get('emit', 0)}、resume={served.get('resume', 0)}"
        f"（uptime {st.get('uptime_s')}s）",
    ]
    if served.get("chat"):
        out.append(f"上一次 chat 實際走的路由：{st.get('chat_route')}"
                   f"（chat_format={st.get('chat_format')}）")
    if model.get("path"):
        out.append(f"模型：{model.get('path')}（存在={model.get('exists')}）")
    if ls.get("flags_probed") is not None:
        out.append(f"探測到 llama-server 可選旗標 {len(ls['flags_probed'])} 個")
    return out


def map_edge_status(st: dict, args) -> tuple[dict, list[str]]:
    """把 edge 的事實映射成契約紀錄。回傳 (record, 致命問題清單)。

    致命的兩種（都會寫到錯的檔名或錯的身份）直接回空 record：
      - object 不是 powerauto.edge.status（拿錯 JSON）
      - --id 與 status 的 endpoint_id 不一致 / 兩邊都沒有
    """
    problems: list[str] = []
    if st.get("object") != EDGE_STATUS_OBJECT:
        problems.append(f"這份 JSON 不是 edge status：object={st.get('object')!r}，"
                        f"要是 {EDGE_STATUS_OBJECT!r}（你是不是把 /v1/models 的輸出餵進來了？）")
        return {}, problems

    host = st.get("host") or {}
    sid = str(st.get("endpoint_id") or "").strip()
    rid = str(args.id or "").strip()
    if rid and sid and rid != sid:
        problems.append(f"--id={rid!r} 與 edge status 的 endpoint_id={sid!r} 不一致 —— "
                        f"不一致會把上報寫到**錯的檔名**，所以寧可拒跑")
        return {}, problems
    eid = rid or sid
    if not eid:
        problems.append("edge status 沒有 endpoint_id（啟動時沒給 --endpoint-id），"
                        "而你也沒給 --id ⇒ 這份上報不知道屬於誰")
        return {}, problems

    derived = derive_what_ran(st)
    supplied = list(args.what_ran or [])
    metrics = {
        "edge_uptime_s": float(st.get("uptime_s") or 0),
        "edge_served_chat": float((st.get("served") or {}).get("chat", 0)),
        "edge_served_completions": float((st.get("served") or {}).get("completions", 0)),
        "host_cpu_cores": float(host.get("cpu_cores") or 0),
        "host_ram_gb": float(host.get("ram_gb") or 0),
    }
    metrics.update(parse_metrics(args.metric))

    _missing = [k for k in ("platform", "arch") if not host.get(k)]
    prov_fb = (f"；★ edge 沒有回報 {'/'.join(_missing)} ⇒ 這兩個值取自**產生器所在的機器**"
               f"（若產生器不在端點上跑，它們就是錯的 —— 別把它們當成端點的事實）"
               if _missing else "")
    prov = (f"what_ran 前 {len(derived)} 條由 edge_server /v1/edge/status 的**觀測欄位**導出"
            f"（可逐條回溯到 served／uptime_s／worker／chat_route／model），"
            f"其餘 {len(supplied)} 條由 --what-ran 人工提供；"
            f"edge 自己的 reported_at={st.get('reported_at')}" + prov_fb)
    if not args.gpu:
        prov += ("；gpu 留空 —— edge status 只有 backend_hint="
                 f"{host.get('backend_hint')!r}，那是後端提示不是型號，所以不填")
    notes = args.notes or ""
    return {
        "contract_version": CONTRACT_VERSION,
        "endpoint_id": eid,
        "reported_at": args.reported_at or now_iso(),
        "hostname": args.hostname or socket.gethostname(),
        "platform": host.get("platform") or args.platform or sys.platform,
        "arch": host.get("arch") or args.arch or platform.machine(),
        "produced_by": (args.produced_by or
                        f"edge_server {st.get('src_version')} /v1/edge/status → "
                        f"report_endpoint_status.py --from-edge-status"),
        "what_ran": derived + supplied,
        "capabilities": list(st.get("capabilities") or []) + list(args.capability or []),
        "gpu": args.gpu or "",
        "build_profile": args.build_profile or "",
        "metrics": metrics,
        # ★ edge 的 not_measured **原樣帶過來** —— 它自己會宣告
        #   「沒設 --api-key ⇒ 同網段可存取」這類真實的未知，不要在這裡吞掉。
        "not_measured": list(st.get("not_measured") or []) + list(args.not_measured or []),
        "notes": (notes + ("\n" if notes else "") + prov),
    }, problems


def build_record(args) -> dict:
    """組出契約要求的紀錄。驗證在 validate_record()，這裡只負責組。"""
    return {
        "contract_version": CONTRACT_VERSION,
        "endpoint_id": args.id,
        "reported_at": args.reported_at or now_iso(),
        "hostname": args.hostname or socket.gethostname(),
        "platform": args.platform or sys.platform,
        "arch": args.arch or platform.machine(),
        "produced_by": args.produced_by or f"report_endpoint_status.py v{CONTRACT_VERSION}",
        "what_ran": list(args.what_ran or []),
        "capabilities": list(args.capability or []),
        "gpu": args.gpu or "",
        "build_profile": args.build_profile or "",
        "metrics": parse_metrics(args.metric),
        # ★ not_measured 一定要有值：空陣列是一個**主張**（「我沒有未量的東西」），
        #   不能由「使用者忘了填」推得。所以沒有 --not-measured 又不肯宣告時直接拒跑。
        "not_measured": [] if args.claim_nothing_unmeasured else list(args.not_measured or []),
        "notes": args.notes or "",
    }


def validate_record(rec: dict) -> list[str]:
    """契約驗證。回傳問題清單（空 = 合法）。入口端也會跑同一組規則。"""
    problems = []
    for k in REQUIRED:
        if k not in rec:
            problems.append(f"缺必填欄位 {k}")
    if rec.get("contract_version") != CONTRACT_VERSION:
        problems.append(f"contract_version 要是 {CONTRACT_VERSION}，拿到 {rec.get('contract_version')!r}")
    eid = rec.get("endpoint_id")
    if not isinstance(eid, str) or not eid.strip():
        problems.append("endpoint_id 必須是非空字串")
    elif not all(c.isalnum() or c in "-_." for c in eid):
        problems.append(f"endpoint_id 只能用小寫英數與 -_. （拿到 {eid!r}）—— 它會變成檔名")
    ts = rec.get("reported_at")
    if not isinstance(ts, str) or len(ts) != 19 or ts[4] != "-" or ts[13] != ":":
        problems.append(f"reported_at 必須是 'YYYY-MM-DD HH:MM:SS'（拿到 {ts!r}）")
    for k in ("what_ran", "capabilities", "not_measured"):
        if not isinstance(rec.get(k), list):
            problems.append(f"{k} 必須是陣列")
    if isinstance(rec.get("what_ran"), list) and not rec["what_ran"]:
        problems.append("what_ran 是空的 ⇒ 要嘛寫你真的跑了什麼，要嘛寫一句「本輪沒有跑任何東西」")
    if not isinstance(rec.get("metrics"), dict):
        problems.append("metrics 必須是物件")
    return problems


def write_record(rec: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{rec['endpoint_id']}.json"
    tmp = dst.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, dst)                     # 原子替換：不會留下半份 JSON
    return dst


def self_test() -> int:
    """黑箱自測：每一格用子行程跑這支腳本自己，檢查 rc 與輸出。

    ★ 不得有副作用：每一格都指定 --out-dir 到 tmp，不碰真正的 endpoints/。
    """
    me = str(Path(__file__).resolve())
    tmp = Path(tempfile.mkdtemp(prefix="ep_report_selftest_"))
    results: list[tuple[str, bool, str]] = []

    def case(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, bool(ok), detail))

    def run(*argv: str, env=None) -> tuple[int, str]:
        p = subprocess.run([sys.executable, me, *argv], capture_output=True, text=True,
                           env=dict(os.environ, **(env or {})))
        return p.returncode, (p.stdout or "") + (p.stderr or "")

    base = ["--id", "fixture-endpoint", "--what-ran", "跑了一個 fixture"]

    # 1) 正常：給齊必要參數，明確宣告沒有未量項 ⇒ rc=0、檔寫出來、契約驗證通過
    od = tmp / "ok"
    rc, out = run(*base, "--not-measured", "沒量過 A", "--out-dir", str(od))
    f = od / "fixture-endpoint.json"
    good = rc == 0 and f.exists()
    if good:
        rec = json.loads(f.read_text(encoding="utf-8"))
        good = not validate_record(rec) and rec["not_measured"] == ["沒量過 A"]
    case("正常輸入 ⇒ rc=0、檔寫出、契約合法", good, f"rc={rc}")

    # 2) ★ 沉默要出聲：既沒 --not-measured 也沒 --claim-... ⇒ rc≠0
    rc, out = run(*base, "--out-dir", str(tmp / "silent"))
    case("★ 沒宣告 not_measured ⇒ 拒跑（rc≠0）且說明原因",
         rc != 0 and "not-measured" in out and "未指定" not in out, f"rc={rc} out={out[:90]!r}")

    # 3) 明確宣告「沒有未量的東西」⇒ not_measured=[] 且 rc=0
    rc, out = run(*base, "--claim-nothing-unmeasured", "--out-dir", str(tmp / "claim"))
    case("--claim-nothing-unmeasured ⇒ not_measured=[] 且 rc=0",
         rc == 0 and json.loads((tmp / "claim" / "fixture-endpoint.json").read_text(encoding="utf-8"))["not_measured"] == [],
         f"rc={rc}")

    # 4) 缺 --what-ran ⇒ rc≠0（空陣列是主張，不是預設）
    rc, out = run("--id", "fixture-endpoint", "--claim-nothing-unmeasured", "--out-dir", str(tmp / "norun"))
    case("缺 --what-ran ⇒ rc≠0", rc != 0, f"rc={rc}")

    # 5) --metric 格式錯 ⇒ rc≠0 且訊息指出格式（不是 traceback）
    rc, out = run(*base, "--claim-nothing-unmeasured", "--metric", "broken", "--out-dir", str(tmp / "metric"))
    case("--metric 沒有 = ⇒ rc≠0 且訊息說要 k=v",
         rc != 0 and "k=v" in out and "Traceback" not in out, f"rc={rc} out={out[:90]!r}")

    # 6) --print 不寫檔（只印）
    od = tmp / "printonly"
    rc, out = run(*base, "--claim-nothing-unmeasured", "--print", "--out-dir", str(od))
    case("--print 不寫檔且輸出是合法 JSON",
         rc == 0 and not od.exists() and json.loads(out)["endpoint_id"] == "fixture-endpoint", f"rc={rc}")

    # 7) 兩次寫同一檔 ⇒ 覆蓋（這裡**可以**覆蓋，因為它是「現況」不是「趨勢」）
    od = tmp / "twice"
    run(*base, "--claim-nothing-unmeasured", "--out-dir", str(od))
    # ★ 第二次不要帶 base：--what-ran 是 append 語意，帶了就會累積，
    #   而這一格要驗的是「檔案被覆蓋成第二次的內容」，不是合併。
    rc, _ = run("--id", "fixture-endpoint", "--what-ran", "第二次",
                "--claim-nothing-unmeasured", "--out-dir", str(od))
    rec = json.loads((od / "fixture-endpoint.json").read_text(encoding="utf-8"))
    case("重複上報 ⇒ 覆蓋成最新（endpoints/ 是現況，不是趨勢；趨勢在 fleet_status.jsonl）",
         rc == 0 and rec["what_ran"] == ["第二次"], f"what_ran={rec['what_ran']}")

    # ══════════════════════════════════════════════════════════════════════
    #  --from-edge-status：把 edge_server 的事實映射成契約
    # ══════════════════════════════════════════════════════════════════════
    def edge_status(**over):
        """一份**真形狀**的 edge status fixture（欄位名照 edge_server v2 的輸出）。"""
        st = {
            "object": "powerauto.edge.status", "src_version": "2",
            "endpoint_id": "fixture-endpoint",
            "reported_at": "2026-09-18 15:40:00", "started_at": "2026-09-18 15:39:00",
            "uptime_s": 60.0, "listen": "0.0.0.0:8080", "auth_required": True,
            "worker": {"mode": "disabled", "url": "http://127.0.0.1:8081",
                       "reachable": False, "detail": "--no-worker：沒有 spawn worker"},
            "chat_route": "native", "chat_probe": "unknown", "chat_format": "auto",
            "model": {"path": "", "id": "", "exists": False},
            "llama_server": {"path": None, "flags_probed": None},
            "host": {"platform": "darwin", "arch": "arm64", "cpu_cores": 10,
                     "ram_gb": 16.0, "backend_hint": "Metal"},
            "served": {"chat": 0, "completions": 0, "emit": 0, "resume": 0},
            "capabilities": ["OpenAI 相容端點", "CGC 探針端點"],
            "not_measured": ["沒有設 --api-key ⇒ 同網段可存取"],
        }
        st.update(over)
        return st

    def write_edge(st: dict, name: str) -> Path:
        p = tmp / name
        p.write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")
        return p

    # 8) 正常映射：rc=0、契約合法、edge 的事實有進到 what_ran/metrics/not_measured
    p8 = write_edge(edge_status(), "edge_ok.json")
    od8 = tmp / "from_edge"
    rc, out = run("--from-edge-status", str(p8), "--out-dir", str(od8))
    f8 = od8 / "fixture-endpoint.json"
    ok8 = rc == 0 and f8.exists()
    if ok8:
        r8 = json.loads(f8.read_text(encoding="utf-8"))
        ok8 = (not validate_record(r8)
               and r8["platform"] == "darwin" and r8["arch"] == "arm64"
               and r8["endpoint_id"] == "fixture-endpoint"
               and r8["not_measured"] == ["沒有設 --api-key ⇒ 同網段可存取"]
               and r8["metrics"].get("edge_uptime_s") == 60.0
               and any("uptime 60.0s" in x for x in r8["what_ran"])
               and any("已服務 chat=0" in x for x in r8["what_ran"]))
    case("★ --from-edge-status（檔案）⇒ rc=0、契約合法、edge 的事實有進到 what_ran／metrics／not_measured",
         ok8, f"rc={rc} out={out[:140]!r}")

    # 9) ★ what_ran 不得為空（契約），而它**只能**由觀測導出 ⇒ 導出必非空且可回溯
    if ok8:
        traced = [x for x in r8["what_ran"] if any(k in x for k in ("啟動", "worker", "已服務"))]
        case("★ what_ran 由觀測導出且可回溯（啟動時間／worker／已服務計數都在裡面）",
             len(traced) == 3, f"traced={traced}")
    else:
        case("★ what_ran 由觀測導出且可回溯", False, "上一格失敗 ⇒ 無法驗")

    # 10) 不是 edge status（object 不對）⇒ 拒跑
    p10 = write_edge(edge_status(object="openai.models"), "edge_wrong.json")
    rc, out = run("--from-edge-status", str(p10), "--out-dir", str(tmp / "wrong"))
    case("★ 拿錯 JSON（object 不是 powerauto.edge.status）⇒ 拒跑並指名",
         rc != 0 and "不是 edge status" in out, f"rc={rc}")

    # 11) --id 與 status 的 endpoint_id 不一致 ⇒ 拒跑（會寫到錯的檔名）
    rc, out = run("--from-edge-status", str(p8), "--id", "someone-else",
                  "--out-dir", str(tmp / "mismatch"))
    case("★ --id 與 edge 的 endpoint_id 不一致 ⇒ 拒跑（不一致會寫到錯的檔名）",
         rc != 0 and "不一致" in out, f"rc={rc}")

    # 12) 兩邊都沒有 id ⇒ 拒跑
    p12 = write_edge(edge_status(endpoint_id=None), "edge_noid.json")
    rc, out = run("--from-edge-status", str(p12), "--out-dir", str(tmp / "noid"))
    case("★ edge 沒給 --endpoint-id 且命令列也沒 --id ⇒ 拒跑",
         rc != 0 and "不知道屬於誰" in out, f"rc={rc}")

    # 13) ★ edge 沒回報 host ⇒ platform/arch 取自**產生器所在的機器**，必須明說
    #     （否則那兩個值看起來像端點的事實，其實不是）
    p13 = write_edge(edge_status(host={}), "edge_nohost.json")
    od13 = tmp / "nohost"
    rc, out = run("--from-edge-status", str(p13), "--out-dir", str(od13))
    n13 = ""
    if rc == 0:
        n13 = json.loads((od13 / "fixture-endpoint.json").read_text(encoding="utf-8"))["notes"]
    case("★ edge 沒回報 host ⇒ notes 明說 platform/arch 取自產生器所在的機器（不靜默冒充）",
         rc == 0 and "取自**產生器所在的機器**" in n13, f"rc={rc} notes={n13[:120]!r}")

    # 14) edge 回的 not_measured 是空的 ⇒ 那是一句主張，要明說才接受
    p14 = write_edge(edge_status(not_measured=[]), "edge_nonm.json")
    rc, out = run("--from-edge-status", str(p14), "--out-dir", str(tmp / "nonm"))
    rc2, _ = run("--from-edge-status", str(p14), "--claim-nothing-unmeasured",
                 "--out-dir", str(tmp / "nonm2"))
    case("★ edge 回空 not_measured ⇒ 未宣告時拒跑；明確 --claim-nothing-unmeasured 才放行",
         rc != 0 and "主張" in out and rc2 == 0, f"rc={rc} rc2={rc2}")

    # 15) ★★ URL 來源 ＋ 壞 HTTP_PROXY ⇒ 仍必須通（繞代理）。這一格是踩到的 bug
    import http.server
    import threading

    try:
        _sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        _sk.connect(("8.8.8.8", 80))          # 不真的送包，只為了問出本機對外位址
        lan_ip = _sk.getsockname()[0]
        _sk.close()
    except Exception:
        lan_ip = "127.0.0.1"

    class _H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):                      # noqa: N802
            body = json.dumps(edge_status(), ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = http.server.ThreadingHTTPServer(("0.0.0.0", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://{lan_ip}:{srv.server_address[1]}/v1/edge/status"
    bad_env = {"HTTP_PROXY": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9",
               "http_proxy": "http://127.0.0.1:9", "https_proxy": "http://127.0.0.1:9",
               "NO_PROXY": "", "no_proxy": ""}
    rc, out = run("--from-edge-status", url, "--out-dir", str(tmp / "viaurl"), env=bad_env)
    case("★★ URL 來源 ＋ 壞 HTTP_PROXY ⇒ 仍要通（LAN 請求一律繞代理）",
         rc == 0, f"rc={rc} url={url} out={out[:160]!r}")
    srv.shutdown()

    passed = sum(1 for _, ok, _ in results if ok)
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f"   ({detail})"))
    print(f"  {passed}/{len(results)} 通過")
    return 0 if passed == len(results) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="產生一份端點狀態上報（純標準函式庫，可在任一端點執行）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", help="端點 id（對應 agent_harness/portal/fleet.json 的端點）")
    ap.add_argument("--what-ran", action="append", metavar="TEXT", default=[],
                    help="本輪真的跑了什麼（可重複；一輪什麼都沒跑也要寫一句）")
    ap.add_argument("--capability", action="append", metavar="TEXT", default=[],
                    help="已驗證的能力（可重複）。沒驗證過的能力不要寫在這裡")
    ap.add_argument("--not-measured", action="append", metavar="TEXT", default=[],
                    help="★ 我沒有量到的東西（可重複）。缺席要出聲")
    ap.add_argument("--claim-nothing-unmeasured", action="store_true",
                    help="明確宣告「這個端點沒有未量的東西」（不給就必須給 --not-measured）")
    ap.add_argument("--metric", action="append", metavar="K=V", default=[],
                    help="數值讀數（可重複，例：decode_tps=52.3）")
    ap.add_argument("--gpu", default="", help="GPU 描述（沒偵測到就留空，不要猜）")
    ap.add_argument("--build-profile", default="", help="建置設定（例：VS2022 + CUDA 12.x）")
    ap.add_argument("--hostname", default="", help="預設取 socket.gethostname()")
    ap.add_argument("--platform", default="", help="預設取 sys.platform")
    ap.add_argument("--arch", default="", help="預設取 platform.machine()")
    ap.add_argument("--reported-at", default="", help="覆寫時間（測試用）")
    ap.add_argument("--produced-by", default="", help="產生者標記")
    ap.add_argument("--notes", default="", help="自由備註")
    ap.add_argument("--from-edge-status", metavar="SRC", default="",
                    help="從 edge_server 的 /v1/edge/status 產生上報（SRC 是檔案路徑或 URL）。"
                         "★ 映射只在這一側發生 —— edge 吐事實，契約形狀留在這裡")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="輸出目錄（預設：本檔旁的 endpoints/）")
    ap.add_argument("--print", dest="print_only", action="store_true", help="只印不寫檔")
    ap.add_argument("--self-test", action="store_true", help="黑箱自測（7 格）")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    if args.from_edge_status:
        # ── 路線 A：事實由 edge 端提供，本側只做映射 ─────────────────────────
        try:
            st = fetch_edge_status(args.from_edge_status)
        except Exception as e:                       # noqa: BLE001
            print(f"[error] 拿不到 edge status（{args.from_edge_status}）：{e!r}\n"
                  f"        若那是區網位址：先確認服務在跑（--no-worker 也能起）、"
                  f"並且沒有被 HTTP_PROXY 吃掉（本檔已繞代理）。", file=sys.stderr)
            return 2
        rec, fatal = map_edge_status(st, args)
        if fatal:
            for x in fatal:
                print(f"[error] {x}", file=sys.stderr)
            return 2
        if not rec["not_measured"] and not args.claim_nothing_unmeasured:
            print("[error] edge 回的 not_measured 是空的 —— 那是一句主張"
                  "（「這個端點沒有未量的東西」）。\n"
                  "        要接受它就明確加 --claim-nothing-unmeasured。", file=sys.stderr)
            return 2
    else:
        # ── 路線 B：全部由命令列提供（原本的行為）────────────────────────────
        if not args.id:
            print("[error] 需要 --id（端點 id，要對應 fleet.json 裡的端點）", file=sys.stderr)
            return 2

        if not args.not_measured and not args.claim_nothing_unmeasured:
            print("[error] 你要嘛給 --not-measured（可以給多個），要嘛明確給 --claim-nothing-unmeasured。\n"
                  "        理由：not_measured 為空是一個**主張**（「我沒有未量的東西」），不能由\n"
                  "        『使用者忘了填』推得 —— 留白會被讀成前者，那是最貴的一種沉默。", file=sys.stderr)
            return 2

        try:
            rec = build_record(args)
        except ValueError as e:
            print(f"[error] {e}", file=sys.stderr)
            return 2

    problems = validate_record(rec)
    if problems:
        for p in problems:
            print(f"[error] {p}", file=sys.stderr)
        return 2

    if args.print_only:
        print(json.dumps(rec, ensure_ascii=False, indent=2))
        return 0

    dst = write_record(rec, Path(args.out_dir))
    print(f"wrote {dst}")
    print(f"  endpoint_id = {rec['endpoint_id']}   reported_at = {rec['reported_at']}")
    print(f"  what_ran    = {len(rec['what_ran'])} 件   not_measured = {len(rec['not_measured'])} 件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
