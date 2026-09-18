#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把「端點 dump 狀態 → 帶回來 → 映射成契約 → 匯出給網站」這條鏈**定期跑起來**。

為什麼要這一支（而不是再接一條 cron）
--------------------------------------
`report_endpoint_status.py` 已經能把一份 edge status 映射成契約，但它要**人**跑。
「要人跑」的鏈在路上只有兩種結局：① 忘了跑 ② 跑了但沒人看結果。
所以自動化的重點不是「跑得更勤」，而是**跑不動的時候要出聲**。

自動化的邊界（誠實版 —— 請不要讀成「全自動」）
----------------------------------------------
這一支**做得到**：
  - 採集：服務活著 ⇒ 抓 /v1/edge/status；沒活著 ⇒ 讀**已被通道帶回來**的 dump
  - 映射：呼叫 `report_endpoint_status.py`（不重造第二份契約）
  - 匯出：呼叫 `build_fleet_portal.py --export`（不重算任何數字）
  - 留痕：每一輪寫 `auto_runs.jsonl`（連「這一輪什麼都沒拿到」也要留）
  - 看門狗：`--check` 回答「這條鏈上次跑是什麼時候、還活著嗎」

這一支**做不到**（也不會假裝做到）：
  - 端點上的 dump：離線端（鴻蒙／Windows）不在我們的網路上，那一步只能在**那台機器**上跑
  - 通道：scp／git push 是既有的東西，這一支不負責把它們打開
  - commit：它只寫檔，不 commit。要不要進版控是人的決定
  ⇒ 所以 `absent` 是這一支最常回報的結果之一，而**它不是失敗**。

三條硬規矩（與 portal 的其它入口同一套）
----------------------------------------
1. ★ **不偽造新鮮度**：`reported_at` 取端點自己的觀測時刻，不是這一輪的執行時刻。
   否則一個三天沒說話的端點會被每小時重寫成「剛上報」—— 那是自動化最容易製造的謊。
   （映射層已經把這件事做對了，見 report_endpoint_status.py 的 docstring；這一支的
   責任是**不要把它蓋掉**，以及把「這一輪」與「這份觀測」兩個時刻分開記錄。）
2. ★ **同一份觀測只處理一次**：冪等判準就是 `reported_at`。沒有新觀測就不重寫檔案，
   不讓 git 每天多出一堆沒有新資訊的 diff。
3. ★ **缺席要出聲**：沒拿到 ⇒ `absent`（rc 仍 0，它是 unknown 不是紅）；
   拿到了但拒收 ⇒ `reject`（rc 1，那是真的錯了）。
   「跑不動」與「跑了但沒有東西」必須分開：前者要人處理，後者不用。

用法
----
    python3 agent_harness/portal/fleet_auto.py                    # 跑一輪（= --run）
    python3 agent_harness/portal/fleet_auto.py --plan             # 只說這一輪會做什麼（不連網）
    python3 agent_harness/portal/fleet_auto.py --check            # 看門狗（排程裝了卻沒跑 ⇒ rc≠0）
    python3 agent_harness/portal/fleet_auto.py --status           # 最近幾輪的留痕
    python3 agent_harness/portal/fleet_auto.py --install-schedule --interval-min 60
    python3 agent_harness/portal/fleet_auto.py --uninstall-schedule
    python3 agent_harness/portal/fleet_auto.py --self-test
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_fleet_portal as BFP          # noqa: E402  複用（拿 REPO／路徑的單一真相）
import report_endpoint_status as REP      # noqa: E402  契約規則與抓取邏輯的單一實作

REPO = BFP.REPO
PORTAL_DIR = BFP.PORTAL_DIR
FLEET_JSON = BFP.FLEET_JSON
ENDPOINTS_DIR = BFP.ENDPOINTS_DIR
EXPORT_JSON = BFP.EXPORT_JSON
REPORT_SCRIPT = HERE / "report_endpoint_status.py"
BUILD_SCRIPT = HERE / "build_fleet_portal.py"

DUMPS_DIR = ENDPOINTS_DIR / "dumps"
AUTO_RUNS = PORTAL_DIR / "auto_runs.jsonl"
# 日誌放 repo **外**（macOS 的慣例位置）。它是執行期產物，不是版控資產 ——
# 放在 repo 裡會讓 `git status` 一直多一個 untracked 目錄，而那種雜訊正好會
# **掩蓋真正的變更**（這個 repo 有多條線同時在寫，乾淨的 status 是重要的訊號）。
LOG_DIR = Path.home() / "Library" / "Logs" / "powerauto"

LABEL = "ai.powerauto.fleet-auto"
DEFAULT_INTERVAL_MIN = 60
TS_FMT = "%Y-%m-%d %H:%M:%S"

# 一輪的結果只有這四種。刻意不含「失敗」—— 失敗要嘛是 absent（不知道），
# 要嘛是 reject（知道但拒收）。把兩者混成一格，看門狗就分不出該不該叫人。
RESULT_UPDATED = "updated"
RESULT_UNCHANGED = "unchanged"
RESULT_ABSENT = "absent"
RESULT_REJECT = "reject"
ACTIONS = (RESULT_UPDATED, RESULT_UNCHANGED, RESULT_ABSENT, RESULT_REJECT)


def now_str() -> str:
    return datetime.now().strftime(TS_FMT)


def plist_path() -> Path:
    """排程檔的路徑。`PA_FLEET_AUTO_PLIST` 可覆寫 —— 自測要用假的家目錄。"""
    env = os.environ.get("PA_FLEET_AUTO_PLIST")
    if env:
        return Path(env)
    return Path.home() / "Library" / "LaunchAgents" / (LABEL + ".plist")


# ══════════════════════════════════════════════════════════════════════
#  來源決定：不猜，讀 fleet.json
# ══════════════════════════════════════════════════════════════════════

def candidates(ep: dict) -> list[tuple[str, str]]:
    """這一端點要用哪些來源（有序）。回 [(kind, src), …]，kind ∈ {"url","dump"}。

    ★ 為什麼不能寫成「url 有值就試」：`reaches == "offline"` 的端點本來就不在網路上，
      去試只換來一次 timeout —— 而那個 timeout **長得像服務掛了**。
      `reaches` 是唯一該問的問題；url 為 null 就是這條路不存在。
    """
    ch = (ep.get("channels") or {}).get("edge_status")
    if not isinstance(ch, dict):
        return []
    out: list[tuple[str, str]] = []
    reaches = str(ep.get("reaches") or "")
    url, dump = ch.get("url"), ch.get("dump")
    if reaches in ("self", "network") and url:
        out.append(("url", url))
    if dump:
        # ★ dump 在 fleet.json 裡是 **repo 相對**路徑，而這支腳本不保證從 repo 根目錄被
        #   叫起來（launchd 的 WorkingDirectory 只是「通常」對）。不解析成絕對路徑的話，
        #   它會**相對呼叫者的 CWD** 去找檔 —— 症狀是「dump 不存在」，而檔案明明在。
        p = Path(dump)
        out.append(("dump", str(p if p.is_absolute() else (REPO / p))))
    return out


def probe_source(kind: str, src: str, timeout: float) -> tuple[dict | None, str]:
    """回 (status, why)。why 只有在 status 是 None 時才有意義。"""
    if kind == "dump":
        p = Path(src)
        if not p.is_file():
            return None, "dump 不存在（%s）—— 端點還沒跑過 --dump-status，或通道還沒把它帶回來" % src
        try:
            return REP.fetch_edge_status(src, timeout=timeout), ""
        except Exception as e:                     # noqa: BLE001
            return None, "dump 讀不出來：%s: %s" % (type(e).__name__, str(e)[:160])
    try:
        return REP.fetch_edge_status(src, timeout=timeout), ""
    except Exception as e:                         # noqa: BLE001
        return None, "%s: %s" % (type(e).__name__, str(e)[:160])


def inspect(st: dict, eid: str) -> tuple[str, str]:
    """看這份 status 該做什麼。回 (result, why)。

    ★ 這裡的每一條拒收都對應一個「寫下去就會錯」的後果，不是潔癖：
      object 不對 ⇒ 拿錯 JSON；id 不一致 ⇒ 寫到**別人的檔名**；
      比現有的舊 ⇒ 拿一份舊 dump 蓋掉新的報告（那會讓頁面**倒退**）。
    """
    if st.get("object") != REP.EDGE_STATUS_OBJECT:
        return RESULT_REJECT, ("object 是 %r，不是 %r —— 這不是 edge status"
                               % (st.get("object"), REP.EDGE_STATUS_OBJECT))
    sid = str(st.get("endpoint_id") or "").strip()
    if not sid:
        return RESULT_REJECT, ("這份 dump 沒有 endpoint_id（端點上沒給 --endpoint-id）"
                               "⇒ 不知道它屬於誰，而檔名會決定它蓋掉誰的報告")
    if sid != eid:
        return RESULT_REJECT, ("dump 的 endpoint_id=%r 與註冊的 %r 不一致 —— "
                               "接受它就會寫到別人的檔名" % (sid, eid))

    new_ts = str(st.get("reported_at") or "").strip()
    cur = ENDPOINTS_DIR / (eid + ".json")
    old_ts = ""
    if cur.is_file():
        try:
            old_ts = str(json.loads(cur.read_text(encoding="utf-8")).get("reported_at") or "")
        except (OSError, json.JSONDecodeError):
            old_ts = ""                            # 壞檔 ⇒ 讓它被新的蓋掉（下面會走 update）
    if new_ts and old_ts and new_ts == old_ts:
        return RESULT_UNCHANGED, "同一份觀測（reported_at=%s）已經處理過" % new_ts
    if new_ts and old_ts and new_ts < old_ts:
        return RESULT_REJECT, ("拿到的觀測比現有的**舊**（%s < %s）⇒ 不覆蓋："
                               "一份舊 dump 不該把新的報告蓋回去" % (new_ts, old_ts))
    return RESULT_UPDATED, "新觀測 reported_at=%s" % (new_ts or "(沒有)")


def carry_over(old: dict, st: dict) -> dict:
    """把現有報告裡**只有人知道**的東西接過來，餵給映射。

    ★ 為什麼非做不可（這是**實測踩到**的，不是想像的）：自動跑若直接覆蓋，會把**人**
      寫的 what_ran／capabilities／gpu／metrics 洗掉 —— 那些是 edge 根本看不到的
      （edge 只知道它自己啟動多久、服務過幾次）。
      實測一次：一台剛用 `--no-worker` 起來的 edge_server 覆蓋之後，整份 mac-local 報告
      從「3 條能力 ＋ 3 個實測 metrics ＋ 3 條未量測」變成「uptime 2.6s、服務 0 次」——
      **用零換掉實質**。一個會默默丟掉別人工作的自動化，比不自動化更糟。

    怎麼切出「人工的部分」：靠上一份報告裡的 `edge_derived`（映射層寫下的邊界）。
      - 有那一格 ⇒ 前 N 條是 edge 導出的，其餘是人寫的（人寫的在後面，因為映射是
        `derived + supplied`）
      - **沒有那一格 ⇒ 整份都算人工的**（那是手寫的、或舊版產生的）
    """
    ed = old.get("edge_derived") or {}
    n_wr = int(ed.get("what_ran_n") or 0)
    n_cp = int(ed.get("capabilities_n") or 0)
    n_nm = int(ed.get("not_measured_n") or 0)
    human_wr = list(old.get("what_ran") or [])[n_wr:]
    human_cp = list(old.get("capabilities") or [])[n_cp:]
    human_nm = list(old.get("not_measured") or [])[n_nm:]

    # 去重：人已經有的、而 edge 這次也會給的，不要再帶一次（否則每跑一次長一截）
    edge_cp = set(st.get("capabilities") or [])
    edge_nm = set(st.get("not_measured") or [])
    edge_keys = {"edge_uptime_s", "edge_served_chat", "edge_served_completions",
                 "host_cpu_cores", "host_ram_gb"}
    return {
        "what_ran": [x for x in human_wr if x],
        "capabilities": [x for x in human_cp if x and x not in edge_cp],
        "not_measured": [x for x in human_nm if x and x not in edge_nm],
        "metrics": ["%s=%s" % (k, v) for k, v in (old.get("metrics") or {}).items()
                    if k not in edge_keys],
        "gpu": str(old.get("gpu") or ""),
        "build_profile": str(old.get("build_profile") or ""),
    }


# ══════════════════════════════════════════════════════════════════════
#  跑一輪
# ══════════════════════════════════════════════════════════════════════

def load_fleet() -> list[dict]:
    try:
        return json.loads(FLEET_JSON.read_text(encoding="utf-8")).get("endpoints") or []
    except FileNotFoundError:
        raise SystemExit("[error] 找不到 %s" % FLEET_JSON)
    except json.JSONDecodeError as e:
        raise SystemExit("[error] %s 不是合法 JSON: %s" % (FLEET_JSON, e))


def should_export(any_updated: bool, today: str) -> tuple[bool, str]:
    """這一輪要不要重生匯出檔。

    ★ 不是「每次都重生」—— 沒有一輪新資訊時重生，只會產生一個**只有時間戳不同**的
      diff，而那個 diff 會讓「上次有變化是什麼時候」變得看不出來。
    """
    if any_updated:
        return True, "有端點帶回新觀測"
    if not EXPORT_JSON.is_file():
        return True, "還沒有匯出檔"
    try:
        prev = str(json.loads(EXPORT_JSON.read_text(encoding="utf-8")).get("generated_at") or "")
    except (OSError, json.JSONDecodeError):
        return True, "匯出檔讀不出來（重生一份）"
    if prev[:10] != today:
        return True, "匯出檔不是今天的（%s）—— 跨日讓七日趨勢與動能有機會變" % (prev[:10] or "?")
    return False, "今天已經匯出過，且這一輪沒有新觀測"


def run_once(trigger: str, timeout: float, do_export: bool, dry: bool = False) -> dict:
    """跑一輪採集→映射（→匯出），回這一輪的留痕紀錄。"""
    eps = load_fleet()
    started = now_str()
    per: list[dict] = []

    for ep in eps:
        eid = str(ep.get("id") or "")
        rec: dict = {"id": eid, "reaches": ep.get("reaches"), "source": None, "result": None,
                     "reported_at": None, "why": ""}
        cands = candidates(ep)
        if not eid or not cands:
            rec["result"] = RESULT_ABSENT
            rec["why"] = ("fleet.json 沒有給這一端 edge_status 通道（url 與 dump 都沒有）"
                          if eid else "端點沒有 id")
            per.append(rec)
            continue

        for kind, src in cands:
            st, why = probe_source(kind, src, timeout)
            if st is None:
                rec["why"] = why                       # 記下失敗的那條，繼續試下一條
                continue
            rec["source"] = kind
            result, why2 = inspect(st, eid)
            rec["result"] = result
            rec["reported_at"] = str(st.get("reported_at") or "") or None
            rec["why"] = why2
            if result == RESULT_UPDATED and not dry:
                # ★ 映射呼叫的是**同一支** report_endpoint_status.py，不是這裡再寫一份。
                #   第二份實作必然漂移，而漂移是靜默的（這個 repo 已經為此付過學費）。
                #
                # ★★ `--out-dir` 一定要帶。它的預設值是**那支腳本自己旁邊的 endpoints/**，
                #    也就是**真 repo** —— 少了這個參數，fixture 的報告會被寫進真 repo，
                #    而「寫成功」看起來完全正常（檔名對、內容合契約、rc=0）。
                #    我第一版就是這樣，是自測最後那格「endpoints/ 快照比對」才抓到的：
                #    **一個只檢查「有沒有報錯」的自測，對「寫到錯的地方」完全無感。**
                old_rec: dict = {}
                cur = ENDPOINTS_DIR / (eid + ".json")
                if cur.is_file():
                    try:
                        old_rec = json.loads(cur.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        old_rec = {}
                co = carry_over(old_rec, st)
                rec["carried"] = {k: len(v) for k, v in co.items() if isinstance(v, list)}

                argv = [sys.executable, str(REPORT_SCRIPT), "--from-edge-status", src,
                        "--out-dir", str(ENDPOINTS_DIR)]
                for x in co["what_ran"]:
                    argv += ["--what-ran", x]
                for x in co["capabilities"]:
                    argv += ["--capability", x]
                for x in co["not_measured"]:
                    argv += ["--not-measured", x]
                for x in co["metrics"]:
                    argv += ["--metric", x]
                if co["gpu"]:
                    argv += ["--gpu", co["gpu"]]
                if co["build_profile"]:
                    argv += ["--build-profile", co["build_profile"]]
                # edge 與人工都沒宣告未量測 ⇒ 那是一句主張，要明說才過得了契約閘門
                if not (st.get("not_measured") or []) and not co["not_measured"]:
                    argv.append("--claim-nothing-unmeasured")

                r = subprocess.run(argv, capture_output=True, text=True,
                                   cwd=str(REPO), timeout=120)
                if r.returncode != 0:
                    rec["result"] = RESULT_REJECT
                    rec["why"] = ("映射被拒（rc=%d）：%s"
                                  % (r.returncode,
                                     ((r.stdout or "") + (r.stderr or "")).strip().splitlines()[-1:][0]
                                     if ((r.stdout or "") + (r.stderr or "")).strip() else "?"))
            break
        else:
            rec["result"] = RESULT_ABSENT
            rec["why"] = rec["why"] or "沒有可用的來源"

        per.append(rec)

    updated = [r for r in per if r["result"] == RESULT_UPDATED]
    rejected = [r for r in per if r["result"] == RESULT_REJECT]
    exported, export_why = (False, "未要求")
    if do_export and not dry:
        want, export_why = should_export(bool(updated), started[:10])
        if want:
            r = subprocess.run([sys.executable, str(BUILD_SCRIPT), "--export"],
                               capture_output=True, text=True, cwd=str(REPO), timeout=300)
            exported = r.returncode == 0
            if not exported:
                tail = ((r.stdout or "") + (r.stderr or "")).strip().splitlines()[-1:]
                export_why = "匯出失敗（rc=%d）：%s" % (r.returncode, tail[0] if tail else "?")

    return {
        "ts": started,
        "trigger": trigger,
        "repo_head": BFP.git_meta().get("head"),
        "endpoints": per,
        "updated": [r["id"] for r in updated],
        "absent": [r["id"] for r in per if r["result"] == RESULT_ABSENT],
        "rejected": [r["id"] for r in rejected],
        "unchanged": [r["id"] for r in per if r["result"] == RESULT_UNCHANGED],
        "exported": exported,
        "export_why": export_why,
    }


def append_run(rec: dict) -> None:
    """留痕。★ 只追加、永不覆蓋 —— 與 fleet_status.jsonl 同一條規矩。"""
    AUTO_RUNS.parent.mkdir(parents=True, exist_ok=True)
    with AUTO_RUNS.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_runs() -> list[dict]:
    if not AUTO_RUNS.is_file():
        return []
    out = []
    for line in AUTO_RUNS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def cmd_run(trigger: str, timeout: float, do_export: bool, dry: bool, as_json: bool) -> int:
    rec = run_once(trigger, timeout, do_export, dry=dry)
    if dry:
        print("  （--dry：沒有寫任何檔、沒有呼叫映射與匯出）")
    else:
        append_run(rec)
    if as_json:
        print(json.dumps(rec, ensure_ascii=False, indent=2))
        return 1 if rec["rejected"] else 0

    for r in rec["endpoints"]:
        mark = {RESULT_UPDATED: "更新", RESULT_UNCHANGED: "同一份", RESULT_ABSENT: "拿不到",
                RESULT_REJECT: "★拒收"}.get(r["result"], r["result"])
        src = ("　（經 %s）" % r["source"]) if r["source"] else ""
        print("  [%s] %-22s %s%s" % (mark, r["id"], r["why"][:110], src))
    print("  這一輪：更新 %d、同一份 %d、拿不到 %d、拒收 %d"
          % (len(rec["updated"]), len(rec["unchanged"]), len(rec["absent"]), len(rec["rejected"])))
    print("  匯出：%s（%s）" % ("是" if rec["exported"] else "否", rec["export_why"]))
    if not dry:
        print("  留痕：%s" % AUTO_RUNS.relative_to(REPO))
        # ★ 只追加的檔案會一直長大。**出聲，但不自動刪** —— 自動刪歷史是危險動作，
        #   而「沒人記得它已經 4 MB」才是真正的問題。
        try:
            mb = AUTO_RUNS.stat().st_size / 1e6
            if mb >= 2.0:
                print("  ★ auto_runs.jsonl 已經 %.1f MB（每輪一行，永遠只追加）。"
                      "要不要裁掉舊的由你決定 —— 這支工具不自動刪。" % mb)
        except OSError:
            pass
    # ★ rc 契約：拒收才是錯（1）；「拿不到」不是錯（0）。
    #   一個會因為端點離線而變紅的自動化，會在每一個離線的早晨說謊。
    return 1 if rec["rejected"] else 0


# ══════════════════════════════════════════════════════════════════════
#  看門狗：這條鏈還活著嗎
# ══════════════════════════════════════════════════════════════════════

def schedule_state() -> tuple[str, int | None]:
    """回 ("absent"|"installed"|"unreadable", interval_s)。"""
    p = plist_path()
    if not p.is_file():
        return "absent", None
    try:
        txt = p.read_text(encoding="utf-8")
    except OSError:
        return "unreadable", None
    m = re.search(r"<key>StartInterval</key>\s*<integer>(\d+)</integer>", txt)
    return "installed", (int(m.group(1)) if m else None)


def dumps_hygiene(ids: list[str]) -> list[str]:
    """dump 目錄與報告目錄的兩向檢查。

    ★ 這一格在防一個**症狀會誤導**的錯：`endpoints/<id>.status.json`（把原始 dump
      放進報告目錄）會被 `build_fleet_portal --check` 報成
      「endpoints/windows-rtx4090.status.json 存在，但 fleet.json 沒有註冊
      windows-rtx4090.status」—— 那句話聽起來像**註冊表少了東西**，
      其實是**檔案放錯目錄**。檢查器不該把使用者指去修錯的地方。
    """
    P: list[str] = []
    if ENDPOINTS_DIR.is_dir():
        for f in sorted(ENDPOINTS_DIR.glob("*.json")):
            if f.stem in ids:
                continue
            if any(f.stem.startswith(i + ".") for i in ids):
                P.append("★ %s 看起來是**原始 dump 放錯位置**：報告目錄只放 "
                         "endpoints/<id>.json，dump 要放 agent_harness/portal/endpoints/dumps/%s"
                         "（不然 build_fleet_portal --check 會把它報成「未註冊的端點」，"
                         "把你要去修的地方指錯）" % (f.name, f.stem + ".json"))
    if DUMPS_DIR.is_dir():
        for f in sorted(DUMPS_DIR.glob("*.json")):
            if f.stem not in ids:
                P.append("★ endpoints/dumps/%s 對不到 fleet.json 裡的任何端點 ⇒ "
                         "它永遠不會被採集（打錯檔名？還是在 dump 一台沒註冊的機器？）" % f.name)
    return P


def cmd_check(as_json: bool) -> int:
    ids = [str(e.get("id") or "") for e in load_fleet()]
    runs = load_runs()
    state, interval = schedule_state()
    problems: list[str] = []

    if state == "unreadable":
        problems.append("排程檔存在但讀不出來：%s" % plist_path())
    problems += dumps_hygiene(ids)

    last = runs[-1] if runs else None
    age_s = None
    if last:
        try:
            age_s = (datetime.now()
                     - datetime.strptime(str(last.get("ts")), TS_FMT)).total_seconds()
        except (ValueError, TypeError):
            problems.append("auto_runs.jsonl 最後一輪的 ts 解析不了：%r" % last.get("ts"))
    if last and last.get("rejected"):
        problems.append("★ 最後一輪有端點被拒收（%s）⇒ 有一份 dump 沒有進到報告裡：%s"
                        % ("、".join(last["rejected"]), AUTO_RUNS))

    if state == "absent":
        verdict = ("未啟用（沒有人手跑過，也沒裝排程）"
                   if not runs else
                   "只人手跑過（沒裝排程）—— 第 %d 輪，最後 %s" % (len(runs), last.get("ts")))
        healthy = not problems
    elif not runs:
        verdict = "★ 排程裝了卻**從沒跑過** —— 排程檔在 %s" % plist_path()
        problems.append(verdict)
        healthy = False
    else:
        max_gap = max(900, int((interval or DEFAULT_INTERVAL_MIN * 60) * 2.5))
        if age_s is not None and age_s > max_gap:
            verdict = ("★ 停擺了：最後一輪是 %s（%.1f 小時前），"
                       "而閾值是 %.1f 小時" % (last.get("ts"), age_s / 3600.0, max_gap / 3600.0))
            problems.append(verdict)
            healthy = False
        else:
            verdict = ("健康：最後一輪 %s（%.1f 分鐘前），排程每 %s 分鐘"
                       % (last.get("ts"), (age_s or 0) / 60.0,
                          int((interval or 0) / 60) or "?"))
            healthy = not problems

    if as_json:
        print(json.dumps({"schedule": state, "interval_s": interval, "runs": len(runs),
                          "last_ts": (last or {}).get("ts"), "age_s": age_s,
                          "verdict": verdict, "problems": problems}, ensure_ascii=False, indent=2))
    else:
        print("  排程：%s（%s）" % (state, plist_path()))
        print("  留痕：%d 輪%s" % (len(runs), ("，最後 " + str(last.get("ts"))) if last else ""))
        print("  判讀：%s" % verdict)
        for p in problems:
            print("  [error] %s" % p)
    return 0 if healthy else 1


def cmd_status(limit: int) -> int:
    runs = load_runs()
    if not runs:
        print("  auto_runs.jsonl 還沒有任何一輪（這條鏈還沒跑過）")
        return 0
    print("  共 %d 輪，最近 %d 輪：" % (len(runs), min(limit, len(runs))))
    for r in runs[-limit:]:
        print("  %s  %-9s 更新=%-14s 拿不到=%-22s 拒收=%s"
              % (r.get("ts"), r.get("trigger"), ",".join(r.get("updated") or []) or "-",
                 ",".join(r.get("absent") or []) or "-",
                 ",".join(r.get("rejected") or []) or "-"))
    return 0


# ══════════════════════════════════════════════════════════════════════
#  排程（macOS launchd；其它平台給明確的替代指令，不假裝成功）
# ══════════════════════════════════════════════════════════════════════

PLIST_TMPL = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{label}</string>
  <key>ProgramArguments</key>
  <array>
    <string>{py}</string>
    <string>{script}</string>
    <string>--run</string>
    <string>--trigger</string><string>schedule</string>
  </array>
  <key>WorkingDirectory</key><string>{repo}</string>
  <key>StartInterval</key><integer>{interval}</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>{out}</string>
  <key>StandardErrorPath</key><string>{err}</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key><string>/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin</string>
  </dict>
</dict>
</plist>
"""


def cmd_install_schedule(interval_min: int) -> int:
    if sys.platform != "darwin":
        print("[error] --install-schedule 目前只實作 macOS（launchd）。")
        print("        Linux／其它平台的等價作法（cron）：")
        print("          */%d * * * * cd %s && %s %s --run --trigger schedule"
              % (interval_min, REPO, sys.executable, Path(__file__).resolve()))
        print("        這條指令是**印出來給你決定**的，我沒有替你裝 —— ")
        print("        在別人的機器上偷偷多一條 cron，比不裝更糟。")
        return 2
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    pl = plist_path()
    pl.parent.mkdir(parents=True, exist_ok=True)
    xml = PLIST_TMPL.format(
        label=LABEL, py=sys.executable, script=str(Path(__file__).resolve()), repo=str(REPO),
        interval=interval_min * 60, out=str(LOG_DIR / "fleet_auto.out.log"),
        err=str(LOG_DIR / "fleet_auto.err.log"))
    pl.write_text(xml, encoding="utf-8")

    dom = "gui/%d" % os.getuid()
    subprocess.run(["launchctl", "bootout", dom, str(pl)], capture_output=True, text=True)
    r = subprocess.run(["launchctl", "bootstrap", dom, str(pl)], capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        print("[error] launchctl bootstrap 失敗（rc=%d）：%s" % (r.returncode, out.strip()[:200]))
        print("        排程檔已經寫在 %s，但沒有被載入。" % pl)
        print("        它每 %d 分鐘會跑一次（RunAtLoad=true ⇒ 載入時先跑一次）。" % interval_min)
        return 1
    print("  已裝：%s" % pl)
    print("  每 %d 分鐘跑一次；載入時會先跑一次（RunAtLoad）。" % interval_min)
    print("  驗證：python3 %s --check      （排程裝了卻沒跑 ⇒ rc=1）" % Path(__file__).resolve())
    print("  卸除：python3 %s --uninstall-schedule" % Path(__file__).resolve())
    print("  日誌：%s" % (LOG_DIR / "fleet_auto.out.log"))
    print("  ★ 它只寫檔，不 commit。要不要把 endpoints／fleet_export 的變動進版控是人的決定。")
    return 0


def cmd_uninstall_schedule() -> int:
    pl = plist_path()
    dom = "gui/%d" % os.getuid()
    r = subprocess.run(["launchctl", "bootout", dom, str(pl)], capture_output=True, text=True)
    existed = pl.is_file()
    if existed:
        pl.unlink()
    out = ((r.stdout or "") + (r.stderr or "")).strip()
    print("  已卸除：%s（排程檔%s）" % (pl, "已刪除" if existed else "本來就不存在"))
    if r.returncode != 0 and out:
        print("  （launchctl 說：%s —— 若它本來就沒載入，這不算問題）" % out[:160])
    return 0


# ══════════════════════════════════════════════════════════════════════
#  自測
# ══════════════════════════════════════════════════════════════════════

def self_test() -> int:
    """突變式黑箱自測。每一格用子行程跑自己，PA_PORTAL_REPO 指向 fixture。

    ★ 不得有副作用：每一格都在 tmp 的 fixture 裡跑，且**不碰真的 repo**。
      自測裡的 fleet_auto 跑的是 --no-export（只驗接線），
      真正的匯出整合在端到端實跑裡驗 —— 那需要一份完整的註冊表。
    """
    me = str(Path(__file__).resolve())
    tmp = Path(tempfile.mkdtemp(prefix="fleet_auto_selftest_"))
    results: list[tuple[str, bool, str]] = []
    # ★ 先記下真 repo 的 auto_* 產物。最後兩格要驗的是「自測**沒有動到**它們」，
    #   不是「它們不存在」—— 後者會在第一次真的跑過之後變成永遠 FAIL 的假紅燈。
    stray_before = sorted(p.name for p in PORTAL_DIR.glob("auto_*"))
    # ★★ endpoints/ 的快照。這一格防的是「副作用寫到錯的地方」——它與「有沒有報錯」
    #    完全正交：漏帶 --out-dir 時 rc=0、檔名對、契約合法，只是寫進了**真 repo**。
    eps_before = (sorted(p.name for p in ENDPOINTS_DIR.iterdir())
                  if ENDPOINTS_DIR.is_dir() else [])

    def case(name, ok, detail=""):
        results.append((name, bool(ok), detail))

    def run(root: Path, *argv, plist=None) -> tuple[int, str]:
        env = dict(os.environ, PA_PORTAL_REPO=str(root),
                   PA_FLEET_AUTO_PLIST=str(plist or (tmp / "no_such.plist")))
        p = subprocess.run([sys.executable, me, *argv], capture_output=True, text=True, env=env)
        return p.returncode, (p.stdout or "") + (p.stderr or "")

    EP = "agent_harness/portal/endpoints"

    def edge_status(eid, ts, **over):
        st = {"object": "powerauto.edge.status", "src_version": "2", "endpoint_id": eid,
              "reported_at": ts, "started_at": ts, "uptime_s": 60.0,
              "listen": "0.0.0.0:8080", "auth_required": True,
              "worker": {"mode": "disabled", "url": "http://127.0.0.1:8081",
                         "reachable": False, "detail": "fx"},
              "chat_route": "native", "chat_probe": "unknown", "chat_format": "auto",
              "model": {"path": "", "id": "", "exists": False},
              "llama_server": {"path": None, "flags_probed": None},
              "host": {"platform": "darwin", "arch": "arm64", "cpu_cores": 8,
                       "ram_gb": 16.0, "backend_hint": "Metal"},
              "served": {"chat": 0, "completions": 0, "emit": 0, "resume": 0},
              "capabilities": ["fx"], "not_measured": ["fx 沒量過"]}
        st.update(over)
        return st

    def fx_fleet(url=None):
        def ep(eid, reaches, eurl):
            return {"id": eid, "name": eid, "kind": "mac", "platform": "darwin",
                    "arch": "arm64", "role": "fx", "reaches": reaches,
                    "capabilities": [{"id": "cap-fx", "label": "fx",
                                      "evidence": [{"path": "marker.txt",
                                                    "contains": "marker-ok"}]}],
                    "channels": {
                        "artifact": "%s/%s.json" % (EP, eid),
                        "edge_status": {"url": eurl, "dump": "%s/dumps/%s.json" % (EP, eid),
                                        "how": "fx"},
                        "relay": []},
                    "not_measured": []}
        return {"schema": 1, "title": "fx", "what_is_this": "fx", "generated_hint": "fx",
                "contract": {"version": 1, "required": REP.REQUIRED},
                "freshness": {"live_s": 900, "recent_s": 129600, "stale_s": 604800, "why": "fx"},
                "endpoints": [ep("fx-live", "self", url), ep("fx-off", "offline", None)],
                "rules": {"unknown_not_red": "fx"}}

    def fixture(name: str, *, dump_ts="2026-09-18 10:00:00", report_ts=None,
                dump_eid="fx-off", url=None, dump_exists=True) -> Path:
        root = tmp / name
        (root / EP / "dumps").mkdir(parents=True)
        (root / "docs").mkdir(parents=True)
        (root / "marker.txt").write_text("marker-ok\n", encoding="utf-8")
        # fleet.json 在 portal/，不是 portal/endpoints/ —— 這兩個目錄的差別是
        # 「註冊表」與「上報」的差別；fixture 自己搞錯的話，第一格測的就不是接線了。
        (root / "agent_harness" / "portal" / "fleet.json").write_text(
            json.dumps(fx_fleet(url), ensure_ascii=False), encoding="utf-8")
        if dump_exists:
            (root / EP / "dumps" / "fx-off.json").write_text(
                json.dumps(edge_status(dump_eid, dump_ts), ensure_ascii=False), encoding="utf-8")
        if report_ts is not None:
            # 一份**人工**寫的報告：沒有 edge_derived ⇒ 整份都算人工的。
            # 這幾條刻意都是 edge **不可能**知道的東西，用來驗覆蓋時有沒有被保住。
            (root / EP / "fx-off.json").write_text(
                json.dumps({"contract_version": 1, "endpoint_id": "fx-off",
                            "reported_at": report_ts, "hostname": "fx", "platform": "darwin",
                            "arch": "arm64", "produced_by": "手寫（人工）",
                            "what_ran": ["人工跑了一件只有人知道的事"],
                            "capabilities": ["人工驗證過的能力 A"],
                            "gpu": "人工填的 GPU", "build_profile": "人工填的建置設定",
                            "metrics": {"decode_tps": 12.62},
                            "not_measured": ["人工宣告的未知 X"], "notes": "人工寫的"},
                           ensure_ascii=False), encoding="utf-8")
        return root

    def runs_of(root: Path) -> list[dict]:
        p = root / "agent_harness" / "portal" / "auto_runs.jsonl"
        if not p.is_file():
            return []
        return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

    # 1) 正對照：dump 存在且是新的 ⇒ updated、報告寫出來、留痕有一筆
    r1 = fixture("ok")
    rc, out = run(r1, "--run", "--no-export")
    rep = r1 / EP / "fx-off.json"
    ok1 = rc == 0 and rep.is_file() and len(runs_of(r1)) == 1
    if ok1:
        rec = json.loads(rep.read_text(encoding="utf-8"))
        ok1 = (rec["reported_at"] == "2026-09-18 10:00:00"       # ★ 不是映射時刻
               and rec.get("mapped_at") and runs_of(r1)[0]["updated"] == ["fx-off"])
    case("正對照：dump ⇒ updated、報告寫出、reported_at 取自 dump、留痕一筆",
         ok1, "rc=%s out=%s" % (rc, out[-180:]))

    # 2) ★★ reported_at 必須是**端點的觀測時刻**，不是這一輪的執行時刻。
    #     這一格是自動化最容易製造的謊：三天沒說話的端點被每小時重寫成「剛上報」。
    ok2 = False
    if ok1:
        ok2 = (str(rec["reported_at"]).startswith("2026-09-18 10:00")
               and not str(rec["reported_at"]).startswith(datetime.now().strftime("%Y-%m-%d %H:%M")[:13]))
    case("★★ reported_at 是 dump 的觀測時刻（10:00），不是這一輪的執行時刻",
         ok2, "reported_at=%r" % (rec.get("reported_at") if ok1 else None))

    # 3) ★ 冪等：同一份觀測跑第二次 ⇒ unchanged、**檔案沒有被重寫**、留痕仍要追加。
    #     mtime 要在「第二次跑之前」取 —— 在之後取的話，即使它重寫了也一樣會相等，
    #     這一格就變成永遠會過的裝飾品。
    mt_before = rep.stat().st_mtime_ns if rep.is_file() else 0
    rc3, out3 = run(r1, "--run", "--no-export")
    mt_after = rep.stat().st_mtime_ns if rep.is_file() else 0
    rs = runs_of(r1)
    case("★ 同一份觀測跑第二次 ⇒ unchanged、mtime 不變（真的沒有重寫）、留痕仍要追加",
         rc3 == 0 and mt_before == mt_after and len(rs) == 2
         and rs[-1]["unchanged"] == ["fx-off"] and rs[-1]["updated"] == [],
         "mt_same=%s runs=%d last=%s" % (mt_before == mt_after, len(rs), rs[-1] if rs else None))

    # 3b) ★★ 自動跑覆蓋時必須**保住人工寫的內容**。
    #     這一格是為了一個實測踩到的損失而存在的：一台剛 --no-worker 起來的 edge_server
    #     覆蓋之後，mac-local 從「3 條能力＋3 個實測 metrics＋3 條未量測」變成
    #     「uptime 2.6s、服務 0 次」——**用零換掉實質**。
    r3c = fixture("carry", report_ts="2026-09-18 09:00:00")   # 人工報告較舊 ⇒ 會被更新
    rc3c, _ = run(r3c, "--run", "--no-export")
    nc = json.loads((r3c / EP / "fx-off.json").read_text(encoding="utf-8"))
    case("★★ 自動跑覆蓋時保住人工內容（capabilities／metrics／gpu／build_profile／not_measured／what_ran）",
         rc3c == 0 and "人工驗證過的能力 A" in nc["capabilities"]
         and nc["metrics"].get("decode_tps") == 12.62 and nc["gpu"] == "人工填的 GPU"
         and nc["build_profile"] == "人工填的建置設定"
         and "人工宣告的未知 X" in nc["not_measured"]
         and "人工跑了一件只有人知道的事" in nc["what_ran"],
         "caps=%s metrics=%s gpu=%r nm=%s" % (nc["capabilities"], nc["metrics"],
                                              nc["gpu"], nc["not_measured"]))

    # 3c) ★ 而且不能**累積**：第二輪（換一份更新的觀測）之後，人工條目仍只有一份。
    #     edge_derived 是切分線，不是裝飾 —— 沒有它，每跑一次報告就長一截。
    (r3c / EP / "dumps" / "fx-off.json").write_text(
        json.dumps(edge_status("fx-off", "2026-09-18 11:00:00"), ensure_ascii=False),
        encoding="utf-8")
    run(r3c, "--run", "--no-export")
    n2 = json.loads((r3c / EP / "fx-off.json").read_text(encoding="utf-8"))
    case("★ 跑第二輪不得累積（人工與導出各只有一份；edge_derived 是切分線）",
         n2["capabilities"].count("人工驗證過的能力 A") == 1
         and n2["what_ran"].count("人工跑了一件只有人知道的事") == 1
         and len(n2["what_ran"]) == n2["edge_derived"]["what_ran_n"] + 1,
         "what_ran=%d edge_derived=%s" % (len(n2["what_ran"]), n2.get("edge_derived")))

    # 4) ★ 舊 dump 不蓋新報告：現有報告比 dump 新 ⇒ reject、rc=1
    r4 = fixture("older", dump_ts="2026-09-18 10:00:00", report_ts="2026-09-18 18:00:00")
    before = (r4 / EP / "fx-off.json").read_text(encoding="utf-8")
    rc4, out4 = run(r4, "--run", "--no-export")
    case("★ 拿到的觀測比現有的舊 ⇒ 拒收、rc=1、報告不被蓋回去",
         rc4 == 1 and "拒收" in out4 and "舊" in out4
         and (r4 / EP / "fx-off.json").read_text(encoding="utf-8") == before,
         "rc=%s out=%s" % (rc4, out4[-160:]))

    # 5) ★ 離線端沒有 dump ⇒ absent、**rc 仍 0**（不知道不是失敗）
    r5 = fixture("nodump", dump_exists=False)
    rc5, out5 = run(r5, "--run", "--no-export")
    rs5 = runs_of(r5)
    case("★ 離線端沒 dump ⇒ absent、rc=0（不知道不是失敗）、理由含 dump 不存在",
         rc5 == 0 and rs5 and "fx-off" in (rs5[-1]["absent"] or [])
         and not (rs5[-1]["updated"] or []) and "dump 不存在" in out5,
         "rc=%s out=%s" % (rc5, out5[-200:]))

    # 6) ★ 什麼都沒拿到的一輪**也要留痕**（自動化最怕的不是失敗，是靜默）
    case("★ 什麼都沒拿到的一輪仍要留痕（不能靜默：沒記錄與沒上報是兩件事）",
         len(rs5) == 1 and rs5[0]["trigger"] == "manual",
         "runs=%s" % rs5)

    # 7) dump 的 endpoint_id 與註冊不符 ⇒ reject（不然會寫到別人的檔名）
    r7 = fixture("wrongid", dump_eid="someone-else")
    rc7, out7 = run(r7, "--run", "--no-export")
    case("★ dump 的 endpoint_id 與註冊不符 ⇒ 拒收（接受它就會寫到別人的檔名）",
         rc7 == 1 and not (r7 / EP / "fx-off.json").exists() and "不一致" in out7,
         "rc=%s out=%s" % (rc7, out7[-160:]))

    # 8) object 不對（拿錯 JSON）⇒ reject
    r8 = fixture("wrongobj")
    (r8 / EP / "dumps" / "fx-off.json").write_text(
        json.dumps(edge_status("fx-off", "2026-09-18 10:00:00", object="openai.models"),
                   ensure_ascii=False), encoding="utf-8")
    rc8, out8 = run(r8, "--run", "--no-export")
    case("★ 拿錯 JSON（object 不是 powerauto.edge.status）⇒ 拒收並指名",
         rc8 == 1 and "不是 edge status" in out8,
         "rc=%s out=%s" % (rc8, out8[-160:]))

    # 9) ★ 活的端點：url 那條路要走得通（真的起一個 HTTP 服務，且環境裡有壞 proxy）
    import http.server
    import threading
    import socket as _s

    class _H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):                          # noqa: N802
            body = json.dumps(edge_status("fx-live", "2026-09-18 11:11:11"),
                              ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url9 = "http://127.0.0.1:%d/v1/edge/status" % srv.server_address[1]
    r9 = fixture("livesrc", url=url9, dump_exists=False)
    env9 = dict(os.environ, PA_PORTAL_REPO=str(r9),
                PA_FLEET_AUTO_PLIST=str(tmp / "no_such.plist"),
                HTTP_PROXY="http://127.0.0.1:9", HTTPS_PROXY="http://127.0.0.1:9",
                http_proxy="http://127.0.0.1:9", https_proxy="http://127.0.0.1:9")
    p9 = subprocess.run([sys.executable, me, "--run", "--no-export"],
                        capture_output=True, text=True, env=env9)
    ok9 = p9.returncode == 0 and (r9 / EP / "fx-live.json").is_file()
    if ok9:
        ok9 = json.loads((r9 / EP / "fx-live.json").read_text(encoding="utf-8"))["reported_at"] \
            == "2026-09-18 11:11:11"
    case("★ 活著的端點走 url 那條路（壞 HTTP_PROXY 下仍要通 ⇒ 一律繞代理）",
         ok9, "rc=%s out=%s" % (p9.returncode, ((p9.stdout or "") + (p9.stderr or ""))[-170:]))
    srv.shutdown()

    # 10) 看門狗三態 ── 沒排程沒留痕 ⇒ rc=0「未啟用」（未知不是紅）
    r10 = fixture("wd_none")
    rc10, out10 = run(r10, "--check")
    case("看門狗：沒排程、沒留痕 ⇒ rc=0 且說「未啟用」（未知不是紅）",
         rc10 == 0 and "未啟用" in out10, "rc=%s out=%s" % (rc10, out10[-160:]))

    # 11) 看門狗：只人手跑過（有留痕、沒排程）⇒ rc=0，且要說得出是只人手跑過
    run(r10, "--run", "--no-export")
    rc11, out11 = run(r10, "--check")
    case("看門狗：只人手跑過（沒裝排程）⇒ rc=0 且明說",
         rc11 == 0 and "只人手跑過" in out11, "rc=%s out=%s" % (rc11, out11[-160:]))

    # 12) ★★ 看門狗最貴的一格：排程裝了卻從沒跑過 ⇒ rc=1
    #     （這正是「設了排程就以為在跑」的那個失敗模式）
    r12 = fixture("wd_installed")
    pl12 = tmp / "inst.plist"
    pl12.write_text("<key>StartInterval</key>\n  <integer>3600</integer>\n", encoding="utf-8")
    rc12, out12 = run(r12, "--check", plist=pl12)
    case("★★ 看門狗：排程裝了卻從沒跑過 ⇒ rc=1（設了排程就以為在跑，是最貴的失敗模式）",
         rc12 == 1 and "從沒跑過" in out12, "rc=%s out=%s" % (rc12, out12[-160:]))

    # 13) ★★ 看門狗：排程裝了、跑過、但停擺了 ⇒ rc=1
    #     （留痕寫成兩天前，閾值 3600*2.5 秒）
    run(r12, "--run", "--no-export")
    rp = r12 / "agent_harness" / "portal" / "auto_runs.jsonl"
    old = json.loads(rp.read_text(encoding="utf-8").splitlines()[-1])
    old["ts"] = (datetime.now() - timedelta(days=2)).strftime(TS_FMT)
    rp.write_text(json.dumps(old, ensure_ascii=False) + "\n", encoding="utf-8")
    rc13, out13 = run(r12, "--check", plist=pl12)
    case("★★ 看門狗：排程在、但最後一輪是兩天前 ⇒ rc=1「停擺了」",
         rc13 == 1 and "停擺" in out13, "rc=%s out=%s" % (rc13, out13[-170:]))

    # 14) ★ 看門狗：排程在、剛跑過 ⇒ rc=0 healthy
    run(r12, "--run", "--no-export")
    rc14, out14 = run(r12, "--check", plist=pl12)
    case("看門狗：排程在、剛跑過 ⇒ rc=0「健康」",
         rc14 == 0 and "健康" in out14, "rc=%s out=%s" % (rc14, out14[-160:]))

    # 15) ★ 錯放偵測：把 dump 放進報告目錄 ⇒ 要**指名放錯**，
    #     而不是讓 build_fleet_portal 把它報成「未註冊的端點」（那會把人指去修錯的地方）
    r15 = fixture("misplaced")
    (r15 / EP / "fx-off.status.json").write_text("{}", encoding="utf-8")
    rc15, out15 = run(r15, "--check")
    case("★ 原始 dump 被放進報告目錄 ⇒ 指名「放錯位置」（不讓它被誤讀成註冊表缺項）",
         rc15 == 1 and "放錯位置" in out15, "rc=%s out=%s" % (rc15, out15[-220:]))

    # 16) ★ dumps 裡的孤兒 ⇒ 出聲（有檔但沒有端點會用它）
    r16 = fixture("orphan")
    (r16 / EP / "dumps" / "ghost.json").write_text("{}", encoding="utf-8")
    rc16, out16 = run(r16, "--check")
    case("★ endpoints/dumps/ 有對不到端點的孤兒 ⇒ 出聲（它永遠不會被採集）",
         rc16 == 1 and "ghost.json" in out16, "rc=%s out=%s" % (rc16, out16[-200:]))

    # 17) --plan 不寫任何檔（它只說會做什麼；不連網也不落地）
    r17 = fixture("plan")
    rc17, out17 = run(r17, "--plan")
    case("--plan 只說會做什麼、不寫任何檔（連 auto_runs.jsonl 都不建）",
         rc17 == 0 and not runs_of(r17) and not (r17 / EP / "fx-off.json").exists(),
         "rc=%s runs=%s" % (rc17, runs_of(r17)))

    # 18) ★ 看門狗：自測沒有動到**真的** repo 的 auto_* 產物（留痕／日誌）
    stray_after = sorted(p.name for p in PORTAL_DIR.glob("auto_*"))
    case("★ 看門狗：自測沒有動到真 repo 的 auto_* 產物（留痕／日誌）",
         stray_before == stray_after, "%s -> %s" % (stray_before, stray_after))

    # 19) ★★ 看門狗（最貴的一格）：自測沒有把**報告**寫進真 repo 的 endpoints/。
    #     這一格是為了一個真的踩到的 bug 而存在的：fleet_auto 漏帶 --out-dir 時，
    #     report_endpoint_status 會用它自己旁邊的（＝真 repo 的）endpoints/，
    #     於是 fixture 的報告落到真 repo —— 而**每一個「有沒有報錯」的檢查都過**。
    eps_after = (sorted(p.name for p in ENDPOINTS_DIR.iterdir())
                 if ENDPOINTS_DIR.is_dir() else [])
    case("★★ 看門狗：自測沒有把報告寫進真 repo 的 endpoints/（fixture 產物必須留在 fixture）",
         eps_before == eps_after, "%s -> %s" % (eps_before, eps_after))

    passed = sum(1 for _, ok, _ in results if ok)
    for name, ok, detail in results:
        print("  [%s] %s" % ("PASS" if ok else "FAIL", name))
        if not ok:
            print("         %s" % detail)
    print("  %d/%d 通過" % (passed, len(results)))
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    return 0 if passed == len(results) else 1


# ══════════════════════════════════════════════════════════════════════

def cmd_plan() -> int:
    """只印這一輪**會**做什麼。不連網、不寫檔 —— 所以它回答的是「線接對了沒」。"""
    eps = load_fleet()
    print("  這一輪會處理 %d 個端點（來源取自 fleet.json 的 channels.edge_status）：" % len(eps))
    for ep in eps:
        eid = str(ep.get("id") or "")
        cands = candidates(ep)
        cur = ENDPOINTS_DIR / (eid + ".json")
        have = ""
        if cur.is_file():
            try:
                have = str(json.loads(cur.read_text(encoding="utf-8")).get("reported_at") or "?")
            except (OSError, json.JSONDecodeError):
                have = "(壞檔)"
        else:
            have = "(還沒有報告)"
        desc = []
        for kind, src in cands:
            if kind == "dump":
                mark = "有" if Path(src).is_file() else "★沒有"
                desc.append("dump[%s] %s" % (mark, src.split("/")[-1]))
            else:
                desc.append("url %s" % src)
        print("    %-22s reaches=%-8s %s" % (eid, ep.get("reaches"),
                                             "；".join(desc) or "★ 沒有來源"))
        print("    %-22s 現有報告 reported_at=%s" % ("", have))
    want, why = should_export(False, now_str()[:10])
    print("  匯出：%s（%s）" % ("會" if want else "不會", why))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="讓「dump → 帶回 → 映射 → 匯出」這條鏈定期跑起來（含看門狗）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true", help="跑一輪（不指定動作時的預設）")
    ap.add_argument("--plan", action="store_true", help="只說這一輪會做什麼，不連網、不寫檔")
    ap.add_argument("--check", action="store_true", help="看門狗：這條鏈還活著嗎")
    ap.add_argument("--status", action="store_true", help="最近幾輪的留痕")
    ap.add_argument("--limit", type=int, default=8, help="--status 印幾輪")
    ap.add_argument("--trigger", default="manual", choices=("manual", "schedule"),
                    help="這一輪是誰觸發的（排程會寫 schedule）")
    ap.add_argument("--no-export", action="store_true",
                    help="採集後不重生 docs/fleet_export.json")
    ap.add_argument("--timeout", type=float, default=3.0, help="抓 url 的超時（秒）")
    ap.add_argument("--json", dest="as_json", action="store_true", help="機器可讀輸出")
    ap.add_argument("--install-schedule", action="store_true", help="裝 launchd 排程（macOS）")
    ap.add_argument("--interval-min", type=int, default=DEFAULT_INTERVAL_MIN,
                    help="排程間隔（分鐘，預設 %d）" % DEFAULT_INTERVAL_MIN)
    ap.add_argument("--uninstall-schedule", action="store_true", help="卸除排程")
    ap.add_argument("--self-test", action="store_true", help="突變式黑箱自測")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()
    if args.install_schedule:
        return cmd_install_schedule(args.interval_min)
    if args.uninstall_schedule:
        return cmd_uninstall_schedule()
    if args.check:
        return cmd_check(args.as_json)
    if args.status:
        return cmd_status(args.limit)
    if args.plan:
        return cmd_plan()
    return cmd_run(args.trigger, args.timeout, not args.no_export,
                   dry=False, as_json=args.as_json)


if __name__ == "__main__":
    raise SystemExit(main())
