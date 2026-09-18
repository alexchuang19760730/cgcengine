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

    def run(*argv: str) -> tuple[int, str]:
        p = subprocess.run([sys.executable, me, *argv], capture_output=True, text=True)
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
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="輸出目錄（預設：本檔旁的 endpoints/）")
    ap.add_argument("--print", dest="print_only", action="store_true", help="只印不寫檔")
    ap.add_argument("--self-test", action="store_true", help="黑箱自測（7 格）")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

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
