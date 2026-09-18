#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A2A 閘道：把 harness 與 pd 服務暴露成 A2A agent（stdio 免費、只用標準函式庫）。

它實作 A2A 的四組能力（使用者要的是**完整規格**，所以四組都要真的能用）：
  - `message/send`                        同步任務
  - `message/stream`                      SSE 串流（逐步回報）
  - `tasks/get` / `tasks/cancel`          查詢與取消
  - `tasks/pushNotificationConfig/*`      webhook 設定，**且真的投遞**

★ 這一支最重要的設計不是協議，是**分類有實際作用**：
  每個 agent 只接**它那一類**的 skill。送一個不屬於它類別的 skill 過去，會被**拒收並說明**
  （`x-agent-class` 與 `x-taxonomy.skill_owner` 就在它的 card 上，對端本來就查得到）。
  「能區分 agent 屬於哪一類」如果只是卡片上的一個欄位，那它是一個標籤；
  只有在**路由會因為它而拒絕**的時候，它才是分類。

★ 第二個設計是**不假裝**：
  - 宣稱 `pushNotifications: true` 就必須真的送得出去（自測會起一個 webhook 接收器驗這一格）
  - 拿不到後端 ⇒ 任務是 `failed` 並附原因，**不是**「完成但沒有結果」
  - 取消要真的停下來（executor 拿得到 cancel event），不是只把狀態改掉

用法
----
    python3 agent_harness/a2a/a2a_server.py --serve --port 9210
    python3 agent_harness/a2a/a2a_server.py --self-test
    python3 agent_harness/a2a/a2a_server.py --call cgc-explorer register-decision --text "..."
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import agent_card as AC                    # noqa: E402  card 的單一產生器
import identity as ID                      # noqa: E402  身分 → 類別 → 四維
import taxonomy as TX                      # noqa: E402  分類的單一真相

JSONRPC = "2.0"
ERR_PARSE = -32700
ERR_INVALID_REQ = -32600
ERR_METHOD = -32601
ERR_PARAMS = -32602
ERR_INTERNAL = -32603
# A2A 自己的錯誤碼區間（-32001 起）
ERR_TASK_NOT_FOUND = -32001
ERR_TASK_NOT_CANCELABLE = -32002
ERR_PUSH_NOT_SUPPORTED = -32003
ERR_UNSUPPORTED_OP = -32004
ERR_CLASS_MISMATCH = -32005          # ★ 本閘道用：skill 不屬於這個 agent 的類別

STATES = ("submitted", "working", "input-required", "completed", "canceled", "failed",
          "unknown")
FINAL_STATES = ("completed", "canceled", "failed")


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def text_of(message: dict) -> str:
    out = []
    for p in (message.get("parts") or []):
        if p.get("kind") == "text" or "text" in p:
            out.append(str(p.get("text") or ""))
    return "\n".join(out).strip()


def skill_of(message: dict) -> str | None:
    """從 message 取出它要呼叫哪一個 skill。

    ★ 接受兩條路：`metadata.skill`（明確）與文字裡的 `skill:<id>`（方便手打）。
      兩條都沒有 ⇒ 回 None，**呼叫端要拒收**，不要猜一個預設 skill ——
      「猜」會讓一個打錯字的請求看起來像成功。
    """
    md = message.get("metadata") or {}
    if md.get("skill"):
        return str(md["skill"]).strip()
    t = text_of(message)
    if "skill:" in t:
        return t.split("skill:", 1)[1].split()[0].strip()
    return None


# ══════════════════════════════════════════════════════════════════════
#  Executors —— 一個 skill 要能真的做事，否則它是文件上的一行字
# ══════════════════════════════════════════════════════════════════════

class Cancelled(Exception):
    pass


def _tick(task: dict, updater, state: str | None = None, note: str = "") -> None:
    if state:
        task["status"] = {"state": state, "timestamp": now_iso()}
    if note:
        task.setdefault("_log", []).append(note)
        updater({"kind": "status-update", "taskId": task["id"],
                 "status": task["status"], "note": note,
                 "final": task["status"]["state"] in FINAL_STATES})


def ex_echo(task, skill, text, updater, cancel):
    """最小可用的 executor：證明協議真的通。刻意不做任何「智慧」的事。"""
    _tick(task, updater, "working", "echo：把輸入原樣回傳")
    return {"text": text or "(空)"}


def ex_steps(task, skill, text, updater, cancel):
    """分三步回報，中間會停頓 —— 用來驗 SSE 與 push 是不是**逐步**收到的。

    ★ 每一步之間檢查 cancel。一個不檢查取消的長時間任務，`tasks/cancel` 就只是一個
      改了狀態但沒有停下來的美化動作。
    """
    n = 3
    for i in range(1, n + 1):
        for _ in range(10):
            if cancel.is_set():
                raise Cancelled()
            time.sleep(0.02)
        _tick(task, updater, "working", "steps：第 %d/%d 步" % (i, n))
    return {"text": "steps：%d 步完成（每一步都經過 cancel 檢查）" % n}


SHELL_ALLOWLIST = ("python3", "git", "ls", "wc", "echo", "date", "uname", "sw_vers")


def ex_shell_allowlist(task, skill, text, updater, cancel):
    """跑白名單內的一條命令。

    ★ 白名單是**唯一**的閘門：這裡刻意不做 shell 解析、不做管線、不做重導向 ——
      看得懂的那一種才准跑。`shell=True` 在這個情境等於把閘道變成遠端 shell。
    """
    import shlex
    import subprocess
    cmd = (text or "").strip()
    if not cmd:
        raise ValueError("沒有給命令（text 就是命令）")
    try:
        argv = shlex.split(cmd)
    except ValueError as e:
        raise ValueError("命令解析失敗：%s" % e)
    if not argv or argv[0] not in SHELL_ALLOWLIST:
        raise ValueError("不在白名單裡：%r（允許：%s）"
                         % (argv[0] if argv else "", "、".join(SHELL_ALLOWLIST)))
    _tick(task, updater, "working", "shell：%s" % " ".join(argv))
    r = subprocess.run(argv, capture_output=True, text=True, timeout=30, cwd=str(HERE.parent.parent))
    return {"text": (r.stdout or "")[:4000],
            "metadata": {"rc": r.returncode, "stderr": (r.stderr or "")[:1200]}}


def ex_fleet_check(task, skill, text, updater, cancel):
    """真的跑一次 `fleet_auto.py --check` —— operator 類別的一個真實動作。"""
    import subprocess
    script = HERE.parent / "portal" / "fleet_auto.py"
    if not script.is_file():
        raise ValueError("找不到 %s" % script)
    _tick(task, updater, "working", "watchdog：跑 fleet_auto --check")
    r = subprocess.run([sys.executable, str(script), "--check"],
                       capture_output=True, text=True, timeout=120,
                       cwd=str(HERE.parent.parent))
    out = ((r.stdout or "") + (r.stderr or "")).strip()
    if r.returncode not in (0, 1):
        raise ValueError("fleet_auto --check 異常結束 rc=%d" % r.returncode)
    return {"text": out[:4000], "metadata": {"rc": r.returncode}}


def ex_taxonomy_dump(task, skill, text, updater, cancel):
    """回傳分類體系本身 —— 讓探索 agent 能回答「這一類的決策範圍是什麼」。"""
    _tick(task, updater, "working", "taxonomy：匯出分類體系")
    return {"text": json.dumps(
        {cid: {"label": TX.CLASSES[cid]["label"], "macro": TX.CLASSES[cid]["macro"],
               "skills": TX.skill_ids(cid)} for cid in TX.classes()},
        ensure_ascii=False, indent=2),
        "metadata": {"taxonomy": "agent_harness/a2a/taxonomy.py"}}


# skill id -> executor。★ 每個 taxonomy 裡的 skill 都要有一格，否則它是一個
# 「寫在卡片上但永遠不會被執行」的能力（自測會驗這件事）。
EXECUTORS = {
    "build-target": ex_shell_allowlist,
    "port-and-verify": ex_steps,
    "cross-platform-package": ex_shell_allowlist,
    "report-capability": ex_steps,
    "deploy-endpoint": ex_steps,
    "collect-status": ex_fleet_check,
    "watchdog-check": ex_fleet_check,
    "export-fleet": ex_shell_allowlist,
    "register-decision": ex_echo,
    "test-hypothesis": ex_steps,
    "measure-metric": ex_shell_allowlist,
    "triage-blocker": ex_steps,
    # 擴展 skill（不在 taxonomy 裡）—— 讓客戶端能反問「你們的分類是什麼」
    "taxonomy-dump": ex_taxonomy_dump,
}


def executor_for(skill_id: str):
    return EXECUTORS.get(skill_id)


# ══════════════════════════════════════════════════════════════════════
#  任務儲存與執行
# ══════════════════════════════════════════════════════════════════════

class Store:
    def __init__(self):
        # ★ RLock，不是 Lock。這個類別的公開方法會在**持鎖時互相呼叫** ——
        #   第一版的 `_stream` 就在 `with store.lock:` 裡再叫了 `store.get()`，
        #   於是**死鎖**。症狀極具誤導性：那個請求永遠沒有回應，且 server log 裡
        #   **連一行都沒有**（死在 `send_response` 之前，所以 log_request 沒被呼叫）
        #   ⇒ 從外面看像「連線沒建立」或「代理卡住」。我當時正是先懷疑代理，方向錯了。
        #   RLock 是安全網；同時 `_stream` 也改掉了那處不必要的嵌套鎖。
        self.lock = threading.RLock()
        self.tasks: dict[str, dict] = {}
        self.push: dict[str, list[dict]] = {}       # taskId -> [config]
        self.cancel: dict[str, threading.Event] = {}
        self.subscribers: dict[str, list] = {}      # taskId -> [queue]

    def new_task(self, agent_id, skill, message) -> dict:
        t = {
            "id": uuid.uuid4().hex[:16],
            "contextId": message.get("contextId") or uuid.uuid4().hex[:16],
            "status": {"state": "submitted", "timestamp": now_iso()},
            "history": [message],
            "artifacts": [],
            "kind": "task",
            "metadata": {"agentId": agent_id, "skill": skill},
        }
        with self.lock:
            self.tasks[t["id"]] = t
            self.cancel[t["id"]] = threading.Event()
        return t

    def get(self, tid) -> dict | None:
        with self.lock:
            return self.tasks.get(tid)

    def event(self, tid, ev) -> None:
        """把一個事件送給這個任務的所有 SSE 訂閱者，並投遞 push。"""
        with self.lock:
            subs = list(self.subscribers.get(tid) or [])
        for q in subs:
            q.append(ev)
        deliver_push(self, tid, ev)


def _no_proxy_opener():
    """★ LAN／localhost 請求一律繞代理。

    ★★ 這是**第二次**踩同一個坑（第一次在 `report_endpoint_status.py` 的 `--from-edge-status`）：
      環境裡的 `HTTP_PROXY`（本機實測 `http://127.0.0.1:7897`）會把 `http://127.0.0.1:…`
      也丟進代理 ⇒ 請求失敗，而**失敗長得像「對面沒起來」**。
      這裡有兩條路徑會中：push 的 webhook 投遞（**產品路徑**）與自測的 SSE（驗證路徑）。
      產品那一條更貴 —— 它會讓「push 已設定」與「push 送不到」看起來一樣，
      而且錯誤訊息指向代理，於是下一個人會去查代理，而不是查這裡。
    """
    import urllib.request
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def deliver_push(store: Store, tid: str, ev: dict) -> list[dict]:
    """真的 POST 到 webhook。回每個 config 的投遞結果（不吞錯）。

    ★ 投遞失敗**不改任務狀態**：通知是旁路，不是任務的一部分。但它要留下痕跡
      （回傳清單會被自測與 `tasks/pushNotificationConfig/get` 看到），
      否則「push 有設定」與「push 有送達」會長得一樣。
    """
    import urllib.request
    opener = _no_proxy_opener()
    with store.lock:
        cfgs = list(store.push.get(tid) or [])
    out = []
    for c in cfgs:
        url = c.get("url")
        if not url:
            out.append({"url": None, "ok": False, "why": "config 沒有 url"})
            continue
        body = json.dumps({"jsonrpc": JSONRPC, "method": "tasks/pushNotification",
                           "params": {"taskId": tid, "event": ev}}, ensure_ascii=False).encode()
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json",
                                              "X-A2A-Task-Id": tid})
        try:
            with opener.open(req, timeout=5) as r:
                out.append({"url": url, "ok": 200 <= r.status < 300, "status": r.status})
        except Exception as e:                       # noqa: BLE001
            out.append({"url": url, "ok": False, "why": "%s: %s" % (type(e).__name__, str(e)[:120])})
    with store.lock:
        c = store.tasks.get(tid)
        if c is not None:
            c.setdefault("_push_results", []).append({"at": now_iso(), "deliveries": out})
    return out


def run_task(store: Store, task: dict, skill: str, message: dict) -> None:
    """在背景執行。★ 任何例外都變成 `failed` **附原因** ——
    一個沒有原因的 failed 與「任務還在跑」在客戶端看起來一樣。"""
    ex = executor_for(skill)
    tid = task["id"]
    cancel = store.cancel[tid]

    def updater(ev):
        store.event(tid, ev)

    try:
        if ex is None:
            raise ValueError("skill %r 沒有 executor（它在 card 上，但沒人實作它）" % skill)
        if cancel.is_set():
            raise Cancelled()
        result = ex(task, skill, text_of(message), updater, cancel)
        if cancel.is_set():
            raise Cancelled()
        art = {"artifactId": uuid.uuid4().hex[:12], "name": "%s 結果" % skill,
               "parts": [{"kind": "text", "text": result.get("text", "")}],
               "metadata": result.get("metadata") or {}}
        with store.lock:
            task["artifacts"] = [art]
            task["status"] = {"state": "completed", "timestamp": now_iso()}
        updater({"kind": "status-update", "taskId": tid, "status": task["status"],
                 "final": True, "artifact": art})
    except Cancelled:
        with store.lock:
            task["status"] = {"state": "canceled", "timestamp": now_iso()}
        updater({"kind": "status-update", "taskId": tid, "status": task["status"], "final": True})
    except Exception as e:                           # noqa: BLE001
        with store.lock:
            task["status"] = {"state": "failed", "timestamp": now_iso()}
            task["metadata"]["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
        updater({"kind": "status-update", "taskId": tid, "status": task["status"],
                 "note": task["metadata"]["error"], "final": True})


# ══════════════════════════════════════════════════════════════════════
#  JSON-RPC
# ══════════════════════════════════════════════════════════════════════

class Gateway:
    def __init__(self, host: str = "127.0.0.1"):
        self.cards, self.index, self.problems = AC.build_all(host)
        self.store = Store()

    def agent(self, aid: str) -> dict | None:
        return self.cards.get(aid)

    def skills_of(self, aid: str) -> set[str]:
        c = self.agent(aid)
        return {s["id"] for s in (c or {}).get("skills") or []}

    def handle(self, aid: str, req: dict) -> dict:
        rid = req.get("id")
        method = req.get("method")
        params = req.get("params") or {}

        def ok(result):
            return {"jsonrpc": JSONRPC, "id": rid, "result": result}

        def err(code, msg, data=None):
            e = {"code": code, "message": msg}
            if data is not None:
                e["data"] = data
            return {"jsonrpc": JSONRPC, "id": rid, "error": e}

        if req.get("jsonrpc") != JSONRPC or not method:
            return err(ERR_INVALID_REQ, "不是合法的 JSON-RPC 2.0 請求")
        if self.agent(aid) is None:
            return err(ERR_INVALID_REQ, "沒有這個 agent：%s（有的：%s）"
                       % (aid, "、".join(self.cards)))

        if method in ("message/send", "message/stream"):
            msg = params.get("message") or {}
            skill = skill_of(msg)
            if not skill:
                return err(ERR_PARAMS, "找不到 skill：請在 message.metadata.skill 指名，"
                                       "或在文字裡寫 skill:<id>。**本閘道不猜預設 skill** ——"
                                       "猜會讓打錯字的請求看起來像成功。")
            if skill not in self.skills_of(aid):
                owner = TX.class_of_skill(skill)
                myclass = self.agent(aid).get("x-agent-class")
                return err(ERR_CLASS_MISMATCH,
                           "skill %r 不屬於 %s 這一類（它是 %s）"
                           % (skill, myclass, TX.label(owner) if owner else "無類別"),
                           {"agentClass": myclass, "skill": skill, "skillOwnerClass": owner,
                            "availableSkills": sorted(self.skills_of(aid))})
            tid = params.get("taskId") or (params.get("message") or {}).get("taskId")
            task = self.store.get(tid) if tid else None
            if task is None:
                task = self.store.new_task(aid, skill, msg)
            th = threading.Thread(target=run_task, args=(self.store, task, skill, msg), daemon=True)
            th.start()
            if method == "message/send":
                th.join(timeout=float(params.get("timeout") or 30))
                return ok(self._task_view(task))
            return {"__stream__": True, "task": task, "thread": th}

        if method == "agent/brief":
            # ★ 「我這一輪屬於哪一類，以及那一類的資產／能力／進度／復盤」。
            #   參數可以給 email 或 jwt 覆寫（讓運營者用**別的**身分問同一件事），
            #   不給就用這個 agent 自己的身分錨。
            cid = self.agent(aid).get("x-agent-class")
            want = params.get("email") or params.get("jwt")
            if want:
                r = (ID.resolve(email=params.get("email") or "",
                                token=params.get("jwt") or ""))
                if r["class"] is None:
                    return err(ERR_PARAMS, r["why"])
                cid = r["class"]
            b = ID.brief_for(cid)
            b["askedAs"] = {"agentId": aid, "agentClass": self.agent(aid).get("x-agent-class"),
                            "resolvedClass": cid,
                            "identityEmail": ID.email_of(cid)}
            return ok(b)

        if method == "identity/check":
            profs, why = (None, "params.noCloud") if params.get("noCloud") else ID.cloud_profiles()
            return ok({"identity": [{"class": c, "email": ID.email_of(c),
                                     "roleExpected": (TX.get(c).get("identity") or {})
                                     .get("role_expected")} for c in TX.classes()],
                       "cloudProfiles": profs, "cloudWhy": why,
                       "problems": ID.identity_problems(profs, why)})

        if method == "tasks/get":
            t = self.store.get(params.get("id") or params.get("taskId"))
            if not t:
                return err(ERR_TASK_NOT_FOUND, "沒有這個任務：%r" % (params.get("id") or params.get("taskId")))
            return ok(self._task_view(t))

        if method == "tasks/cancel":
            tid = params.get("id") or params.get("taskId")
            t = self.store.get(tid)
            if not t:
                return err(ERR_TASK_NOT_FOUND, "沒有這個任務：%r" % tid)
            if t["status"]["state"] in FINAL_STATES:
                return err(ERR_TASK_NOT_CANCELABLE,
                           "任務已經在終態 %s，不能取消" % t["status"]["state"])
            self.store.cancel[tid].set()
            return ok(self._task_view(t))

        if method.startswith("tasks/pushNotificationConfig/"):
            op = method.rsplit("/", 1)[1]
            return self._push(op, params, ok, err)

        if method == "tasks/resubscribe":
            tid = params.get("id") or params.get("taskId")
            if not self.store.get(tid):
                return err(ERR_TASK_NOT_FOUND, "沒有這個任務：%r" % tid)
            return {"__stream__": True, "task": self.store.get(tid), "thread": None,
                    "resubscribe": True}

        return err(ERR_METHOD, "未實作的方法：%s" % method)

    def _push(self, op, params, ok, err):
        tid = params.get("taskId") or params.get("id")
        if not tid:
            return err(ERR_PARAMS, "缺 taskId")
        if not self.store.get(tid):
            return err(ERR_TASK_NOT_FOUND, "沒有這個任務：%r" % tid)
        with self.store.lock:
            cfgs = self.store.push.setdefault(tid, [])
        if op == "set":
            c = params.get("pushNotificationConfig") or {}
            if not c.get("url"):
                return err(ERR_PARAMS, "pushNotificationConfig 缺 url")
            c = dict(c, id=c.get("id") or uuid.uuid4().hex[:8])
            with self.store.lock:
                cfgs[:] = [x for x in cfgs if x.get("id") != c["id"]] + [c]
            return ok(c)
        if op == "get":
            cid = (params.get("pushNotificationConfigId")
                   or (params.get("pushNotificationConfig") or {}).get("id"))
            hit = [c for c in cfgs if c.get("id") == cid]
            if not hit:
                return err(ERR_TASK_NOT_FOUND, "這個任務沒有 id=%r 的 push 設定" % cid)
            return ok(hit[0])
        if op == "list":
            return ok({"taskId": tid, "configs": cfgs})
        if op == "delete":
            cid = (params.get("pushNotificationConfigId")
                   or (params.get("pushNotificationConfig") or {}).get("id"))
            with self.store.lock:
                before = len(cfgs)
                cfgs[:] = [c for c in cfgs if c.get("id") != cid]
                after = len(cfgs)
            return ok({"deleted": before - after, "taskId": tid})
        return err(ERR_UNSUPPORTED_OP, "未實作的 push 操作：%s" % op)

    @staticmethod
    def _task_view(t: dict) -> dict:
        """給客戶端的視圖。內部欄位（`_log`／`_push_results`）不外露，
        但**投遞結果**要看得到 —— 否則「push 設好了」與「push 送到過」長得一樣。"""
        v = {k: val for k, val in t.items() if not k.startswith("_")}
        v["metadata"] = dict(t.get("metadata") or {})
        if t.get("_push_results"):
            v["metadata"]["pushDeliveries"] = t["_push_results"][-3:]
        return v


# ══════════════════════════════════════════════════════════════════════
#  HTTP
# ══════════════════════════════════════════════════════════════════════

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    gateway: Gateway | None = None
    quiet = False

    def log_message(self, fmt, *args):
        if not self.quiet:
            sys.stderr.write("[a2a] %s\n" % (fmt % args))

    # ── 工具 ────────────────────────────────────────────────────────
    def _json(self, code: int, obj) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _routes(self, path: str) -> tuple[str, str | None]:
        """回 (kind, agent_id)。kind ∈ {index, card, rpc, health, 404}。

        ★ card 路徑**先**精確比對，rpc 才吃其餘 —— card 在 rpc 底下是刻意的。
        """
        gw = AC.load_registry().get("gateway") or {}
        prefix = gw.get("path_prefix", "/a2a")
        idx = gw.get("index_path", "/.well-known/agent-card.json")
        if path == idx:
            return "index", None
        if path == "/healthz":
            return "health", None
        if path.startswith(prefix + "/"):
            rest = path[len(prefix) + 1:]
            if rest.endswith("/.well-known/agent-card.json"):
                return "card", rest[: -len("/.well-known/agent-card.json")]
            return "rpc", rest.strip("/")
        return "404", None

    def do_GET(self):                                       # noqa: N802
        kind, aid = self._routes(self.path.split("?")[0])
        gw = self.gateway
        if kind == "health":
            return self._json(200, {"ok": True, "agents": len(gw.cards),
                                    "problems": gw.problems})
        if kind == "index":
            return self._json(200, gw.index)
        if kind == "card":
            c = gw.agent(aid)
            if not c:
                return self._json(404, {"error": "沒有這個 agent", "id": aid,
                                        "have": sorted(gw.cards)})
            return self._json(200, c)
        return self._json(404, {"error": "no such path", "path": self.path})

    def do_POST(self):                                      # noqa: N802
        path = self.path.split("?")[0]
        kind, aid = self._routes(path)
        if kind != "rpc":
            return self._json(404, {"error": "POST 只在 %s/<agent_id> 上成立"
                                             % (AC.load_registry().get("gateway") or {}).get("path_prefix", "/a2a")})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n).decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as e:
            return self._json(400, {"jsonrpc": JSONRPC, "id": None,
                                    "error": {"code": ERR_PARSE, "message": "JSON 解析失敗: %s" % e}})
        if isinstance(req, list):
            return self._json(400, {"jsonrpc": JSONRPC, "id": None,
                                    "error": {"code": ERR_INVALID_REQ,
                                              "message": "本閘道一次只收單一請求（batch 未實作）"}})
        out = self.gateway.handle(aid, req)
        if isinstance(out, dict) and out.get("__stream__"):
            return self._stream(out)
        return self._json(200, out)

    def _stream(self, out) -> None:
        """SSE。★ 用 `Connection: close` ＋ 不設 Content-Length，
        客戶端讀到 EOF 結束 —— 這是最相容的形狀（chunked 要自己編碼，而任何一處寫錯
        都會變成「連線卡住」而不是報錯）。"""
        task = out["task"]
        tid = task["id"]
        q: list = []
        # ★ 先在鎖外取快照，再進鎖登記訂閱者 —— 不要在同一段裡嵌套呼叫 store 的公開方法。
        #   那正是第一版死鎖的形狀（見 Store.__init__ 的註解）。
        first = self.gateway.store.get(tid) or task
        with self.gateway.store.lock:
            self.gateway.store.subscribers.setdefault(tid, []).append(q)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def emit(ev):
            self.wfile.write(("data: %s\n\n" % json.dumps(ev, ensure_ascii=False)).encode())
            self.wfile.flush()

        try:
            emit({"kind": "task", "taskId": tid, "status": first["status"],
                  "skill": task["metadata"].get("skill")})
            if out.get("resubscribe") and first["status"]["state"] in FINAL_STATES:
                emit({"kind": "status-update", "taskId": tid, "status": first["status"],
                      "final": True})
                return
            deadline = time.time() + 60
            while time.time() < deadline:
                if q:
                    ev = q.pop(0)
                    emit(ev)
                    if ev.get("final"):
                        break
                else:
                    if out.get("thread") is not None and not out["thread"].is_alive() \
                            and not q:
                        t = self.gateway.store.get(tid)
                        emit({"kind": "status-update", "taskId": tid,
                              "status": t["status"], "final": True,
                              "artifact": (t.get("artifacts") or [None])[0]})
                        break
                    time.sleep(0.02)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with self.gateway.store.lock:
                lst = self.gateway.store.subscribers.get(tid) or []
                if q in lst:
                    lst.remove(q)
            self.close_connection = True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="A2A 閘道（harness ＋ pd 服務）")
    ap.add_argument("--serve", action="store_true", help="起閘道")
    ap.add_argument("--port", type=int, default=None, help="預設取 registry.gateway.default_port")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--check", action="store_true", help="只驗分類與卡片，不起服務")
    ap.add_argument("--self-test", action="store_true", help="突變式黑箱自測")
    ap.add_argument("--call", nargs=2, metavar=("AGENT", "SKILL"),
                    help="對自己起一個臨時閘道並呼叫（開發用）")
    ap.add_argument("--text", default="", help="--call 的內容")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    gw = Gateway(args.host)
    if gw.problems:
        for p in gw.problems:
            print("[error] %s" % p)
        return 1
    if args.check:
        print("OK: %d 個 agent、%d 個 skill、%d 個 executor"
              % (len(gw.cards), len(TX.all_skills()), len(EXECUTORS)))
        return 0

    Handler.gateway = gw
    Handler.quiet = args.quiet

    if args.call:
        return dev_call(gw, args.call[0], args.call[1], args.text)

    port = args.port or (AC.load_registry().get("gateway") or {}).get("default_port", 9210)
    srv = ThreadingHTTPServer((args.host, int(port)), Handler)
    print("A2A 閘道在 http://%s:%d" % (args.host, port))
    print("  目錄卡     : http://%s:%d/.well-known/agent-card.json" % (args.host, port))
    for aid, c in gw.cards.items():
        print("  %-18s %-10s %s/a2a/%s" % (aid, c["x-agent-class"], "http://%s:%d" % (args.host, port), aid))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def dev_call(gw: Gateway, agent: str, skill: str, text: str) -> int:
    """不起服務、直接走 Gateway.handle —— 開發時看一個呼叫的結果。"""
    req = {"jsonrpc": JSONRPC, "id": 1, "method": "message/send",
           "params": {"message": {"role": "user", "messageId": uuid.uuid4().hex[:8],
                                  "parts": [{"kind": "text", "text": text}],
                                  "metadata": {"skill": skill}}}}
    out = gw.handle(agent, req)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if "error" not in out else 1


def self_test() -> int:
    """突變式黑箱自測。真的起 HTTP 服務、真的發 JSON-RPC、真的收 SSE、真的收 webhook。"""
    import http.server
    import threading as th
    import urllib.error
    import urllib.request

    results: list[tuple[str, bool, str]] = []

    def case(name, ok, detail=""):
        results.append((name, bool(ok), detail))

    opener = _no_proxy_opener()   # ★ 自測也一律繞代理（理由見 _no_proxy_opener）
    gw = Gateway("127.0.0.1")
    if gw.problems:
        for p in gw.problems:
            print("  [FAIL] 啟動前的問題：%s" % p)
        return 1
    Handler.gateway = gw
    Handler.quiet = True
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    th.Thread(target=srv.serve_forever, daemon=True).start()
    base = "http://127.0.0.1:%d" % srv.server_address[1]

    def jget(path, timeout=5):
        try:
            with opener.open(base + path, timeout=timeout) as r:
                return r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:      # 404 是其中一格要驗的東西，不是意外
            return e.code, json.loads(e.read().decode())

    def jpost(path, obj, timeout=20):
        req = urllib.request.Request(base + path,
                                     data=json.dumps(obj).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with opener.open(req, timeout=timeout) as r:
                return r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode())

    def rpc(agent, method, params=None, rid=1):
        return jpost("/a2a/" + agent, {"jsonrpc": JSONRPC, "id": rid,
                                       "method": method, "params": params or {}})

    def msg(skill, text="hi"):
        return {"role": "user", "messageId": uuid.uuid4().hex[:8],
                "parts": [{"kind": "text", "text": text}], "metadata": {"skill": skill}}

    try:
        # 1) 目錄卡列出所有 agent，且與 registry 雙向對齊
        st, idx = jget("/.well-known/agent-card.json")
        ids = [a["id"] for a in idx["agents"]]
        case("目錄卡列出 %d 個 agent，且與 registry 完全一致" % len(ids),
             st == 200 and sorted(ids) == sorted(gw.cards),
             "idx=%s cards=%s" % (ids, sorted(gw.cards)))

        # 2) 每張 agent card 都開得起來，且分類欄位與 taxonomy 一致
        bad = []
        for aid, c in gw.cards.items():
            st, got = jget("/a2a/%s/.well-known/agent-card.json" % aid)
            if st != 200 or got.get("x-agent-class") != c["x-agent-class"]:
                bad.append(aid)
        case("★ 每個 agent 的 card 都開得起來，且 x-agent-class 與 taxonomy 一致",
             not bad, "bad=%s" % bad)

        # 3) ★ skill_owner 是真的：card 上每個 skill 的歸屬＝呼叫它的那個 agent 的類別
        mism = []
        for aid, c in gw.cards.items():
            own = c["x-taxonomy"]["skill_owner"]
            for s in c["skills"]:
                if s["id"] in own and own[s["id"]] != c["x-agent-class"]:
                    mism.append((aid, s["id"], own[s["id"]]))
        case("★ card 的 skills 與 x-taxonomy.skill_owner 一致（對端可以自己核對分類）",
             not mism, "mismatch=%s" % mism)

        # 4) message/send 正常：回 task、completed、有 artifact
        st, r = rpc("cgc-explorer", "message/send",
                    {"message": msg("register-decision", "登錄一筆決策")})
        t = r.get("result") or {}
        case("message/send ⇒ completed 且帶 artifact",
             st == 200 and t.get("status", {}).get("state") == "completed"
             and (t.get("artifacts") or [{}])[0].get("parts"),
             json.dumps(r, ensure_ascii=False)[:200])

        # 5) ★★ 類別不匹配 ⇒ 拒收（這是「能區分 agent 屬於哪一類」的實際作用）
        st, r = rpc("cgc-dev-porting", "message/send",
                    {"message": msg("register-decision", "開發 agent 不該接這個")})
        e = r.get("error") or {}
        case("★★ 把 explorer 的 skill 送給 developer agent ⇒ 拒收，且說明誰才是 owner",
             st == 200 and e.get("code") == ERR_CLASS_MISMATCH
             and e.get("data", {}).get("skillOwnerClass") == "explorer",
             json.dumps(r, ensure_ascii=False)[:240])

        # 6) ★ 沒指名 skill ⇒ 拒收（不猜預設）
        st, r = rpc("cgc-explorer", "message/send", {"message": msg("", "沒有 skill")})
        case("★ 沒指名 skill ⇒ 拒收並要求指名（本閘道不猜預設 skill）",
             (r.get("error") or {}).get("code") == ERR_PARAMS,
             json.dumps(r, ensure_ascii=False)[:200])

        # 7) 未知 agent ⇒ 明白拒絕並列出有的
        st, r = jpost("/a2a/no-such-agent",
                      {"jsonrpc": JSONRPC, "id": 1, "method": "tasks/get", "params": {"id": "x"}})
        case("未知 agent ⇒ 明白拒絕並列出有哪些",
             (r.get("error") or {}).get("code") == ERR_INVALID_REQ,
             json.dumps(r, ensure_ascii=False)[:180])

        # 8) tasks/get 查得到剛剛那個任務；不存在的 ⇒ -32001
        st, r2 = rpc("cgc-explorer", "tasks/get", {"id": t.get("id")})
        st3, r3 = rpc("cgc-explorer", "tasks/get", {"id": "nope"})
        case("tasks/get：查得到 ⇒ 回任務；查不到 ⇒ -32001",
             r2.get("result", {}).get("id") == t.get("id")
             and (r3.get("error") or {}).get("code") == ERR_TASK_NOT_FOUND,
             json.dumps(r3, ensure_ascii=False)[:160])

        # 9) tasks/cancel：已完成的任務 ⇒ **拒絕**取消（-32002）。
        #    ★ 靜默接受是這裡最貴的錯：客戶端會以為它及時阻止了什麼。
        st, rc0 = rpc("cgc-explorer", "tasks/cancel", {})        # 缺 id ⇒ 也要有明確錯誤
        st, rs = rpc("cgc-explorer", "message/send",
                     {"message": msg("test-hypothesis", "跑一個會停頓的任務")})
        tid = (rs.get("result") or {}).get("id")
        st, r4 = rpc("cgc-explorer", "tasks/cancel", {"id": tid})
        case("tasks/cancel：已完成的任務 ⇒ 拒絕取消（-32002）；缺 id 也要有明確錯誤",
             (r4.get("error") or {}).get("code") == ERR_TASK_NOT_CANCELABLE
             and (rc0.get("error") or {}).get("code") == ERR_TASK_NOT_FOUND,
             "r4=%s rc0=%s" % (json.dumps(r4, ensure_ascii=False)[:110],
                               json.dumps(rc0, ensure_ascii=False)[:110]))

        # 10) ★★ SSE：message/stream 逐步收到事件，且最後一個是 final
        req = urllib.request.Request(
            base + "/a2a/cgc-explorer",
            data=json.dumps({"jsonrpc": JSONRPC, "id": 9, "method": "message/stream",
                             "params": {"message": msg("test-hypothesis", "stream")}}).encode(),
            headers={"Content-Type": "application/json"})
        events = []
        with opener.open(req, timeout=30) as resp:
            ctype = resp.headers.get("Content-Type", "")
            buf = b""
            for raw in resp:
                buf += raw
                while b"\n\n" in buf:
                    chunk, buf = buf.split(b"\n\n", 1)
                    for line in chunk.splitlines():
                        if line.startswith(b"data: "):
                            try:
                                events.append(json.loads(line[6:].decode()))
                            except ValueError:
                                pass
        finals = [e for e in events if e.get("final")]
        case("★★ message/stream（SSE）逐步收到事件，且最後一個是 final",
             ctype.startswith("text/event-stream") and len(events) >= 3
             and len(finals) == 1 and finals[0]["status"]["state"] == "completed",
             "ctype=%s events=%d finals=%d" % (ctype, len(events), len(finals)))

        # 11) ★★ push：真的起一個 webhook 接收器，驗證它收到 POST
        inbox: list = []

        class W(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):                              # noqa: N802
                n = int(self.headers.get("Content-Length") or 0)
                inbox.append(json.loads(self.rfile.read(n).decode()))
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"ok")

        wsrv = ThreadingHTTPServer(("127.0.0.1", 0), W)
        th.Thread(target=wsrv.serve_forever, daemon=True).start()
        hook = "http://127.0.0.1:%d/hook" % wsrv.server_address[1]

        st, rp = rpc("cgc-explorer", "tasks/pushNotificationConfig/set",
                     {"taskId": tid, "pushNotificationConfig": {"url": hook}})
        cid = (rp.get("result") or {}).get("id")
        st, rl = rpc("cgc-explorer", "tasks/pushNotificationConfig/list", {"taskId": tid})
        st, rg = rpc("cgc-explorer", "tasks/pushNotificationConfig/get",
                     {"taskId": tid, "pushNotificationConfigId": cid})
        # 起一個新任務，讓 push 真的被觸發
        st, rs2 = rpc("cgc-explorer", "message/send",
                      {"message": {**msg("test-hypothesis", "push"), "taskId": tid}})
        for _ in range(60):
            if inbox:
                break
            time.sleep(0.05)
        st, rd = rpc("cgc-explorer", "tasks/pushNotificationConfig/delete",
                     {"taskId": tid, "pushNotificationConfigId": cid})
        case("★★ pushNotificationConfig 四種操作 ＋ **webhook 真的收到 POST**",
             bool(cid) and rl.get("result", {}).get("configs")
             and rg.get("result", {}).get("id") == cid
             and rd.get("result", {}).get("deleted") == 1 and len(inbox) >= 1,
             "cid=%s inbox=%d dl=%s" % (cid, len(inbox), rd.get("result")))
        wsrv.shutdown()

        # 12) ★ 沒有 executor 的 skill ⇒ failed 並附原因（不是「完成但沒有結果」）
        EXECUTORS.pop("_fx_probe", None)
        saved = EXECUTORS.pop("triage-blocker", None)
        EXECUTORS["triage-blocker"] = None            # 模擬「卡片上有、但沒人實作」
        try:
            st, re_ = rpc("cgc-explorer", "message/send",
                          {"message": msg("triage-blocker", "x")})
            got = re_.get("result") or {}
            case("★ 卡片上有但沒有 executor ⇒ failed 且附原因（不是靜默成功）",
                 got.get("status", {}).get("state") == "failed"
                 and "executor" in str(got.get("metadata", {}).get("error", "")),
                 json.dumps(got, ensure_ascii=False)[:200])
        finally:
            if saved is not None:
                EXECUTORS["triage-blocker"] = saved

        # 13) ★ 壞 JSON ⇒ -32700，不是 traceback
        req = urllib.request.Request(base + "/a2a/cgc-explorer", data=b"{not json",
                                     headers={"Content-Type": "application/json"})
        try:
            with opener.open(req, timeout=5) as r:
                st, body = r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            st, body = e.code, json.loads(e.read().decode())
        case("壞 JSON ⇒ -32700（不是 traceback 也不是 500）",
             (body.get("error") or {}).get("code") == ERR_PARSE,
             json.dumps(body, ensure_ascii=False)[:160])

        # 14) 未實作的方法 ⇒ -32601
        st, r = rpc("cgc-explorer", "no/such/method", {})
        case("未實作的方法 ⇒ -32601",
             (r.get("error") or {}).get("code") == ERR_METHOD,
             json.dumps(r, ensure_ascii=False)[:160])

        # 15) ★★ 覆蓋率：taxonomy 的**每一個** skill 都要有 executor（否則它是文件上的一行字）
        missing = [s for s in TX.all_skills() if not EXECUTORS.get(s)]
        case("★★ taxonomy 的每一個 skill 都有 executor（沒有的話它只是卡片上的一行字）",
             not missing, "missing=%s" % missing)

        # 16) ★ 每張 card 的 not_capable 非空（「沒寫」會被讀成「可以做」）
        empty = [aid for aid, c in gw.cards.items() if not c.get("x-not-capable")]
        case("★ 每個 agent 都寫了 not_capable（缺席要出聲）", not empty, "empty=%s" % empty)

        # 18) ★★ session 身分 → 類別 → 四維（「運營者 agent 要能區分自己在哪一類」的落地）
        st, r = rpc("fleet-operator", "agent/brief", {})
        br = r.get("result") or {}
        case("★★ agent/brief：運營者拿到的四維是按**它那一類**整理的，且帶身分錨",
             st == 200 and br.get("class") == "operator"
             and (br.get("identity") or {}).get("email") == "alexchuang@powerauto.ai"
             and all(k in br for k in ("assets", "capabilities", "progress", "retro")),
             json.dumps(br, ensure_ascii=False)[:190])

        # 19) ★★ 換一個身分問 ⇒ 換一類；不認識的帳號 ⇒ 拒答（分類真的跟著帳號走）
        st, r = rpc("fleet-operator", "agent/brief", {"email": "frontier@powerauto.ai"})
        br2 = r.get("result") or {}
        st2, r2 = rpc("fleet-operator", "agent/brief", {"email": "nobody@else.com"})
        case("★★ agent/brief 帶 email ⇒ 換一類；不認得的帳號 ⇒ 拒答（-32602）",
             br2.get("class") == "explorer"
             and (r2.get("error") or {}).get("code") == ERR_PARAMS,
             "class=%s err=%s" % (br2.get("class"), json.dumps(r2, ensure_ascii=False)[:130]))

        # 20) ★ identity/check 不連網時要說「沒有查」，不是靜默通過
        st, r = rpc("cgc-explorer", "identity/check", {"noCloud": True})
        d = r.get("result") or {}
        case("★ identity/check（noCloud）⇒ 明說沒查雲側，且列出三類身分錨",
             len(d.get("identity") or []) == 3
             and any("沒有查" in x for x in (d.get("problems") or [])),
             json.dumps(d, ensure_ascii=False)[:190])

        # 17) registry 的 gateway 路徑真的有對應的 HTTP 行為
        st, _ = jget("/healthz")
        st2, _ = jget("/a2a/")
        case("/healthz 通；不存在的路徑 ⇒ 404（不是 200 加空物件）",
             st == 200 and st2 == 404, "health=%s bogus=%s" % (st, st2))
    finally:
        srv.shutdown()

    passed = sum(1 for _, ok, _ in results if ok)
    for name, ok, detail in results:
        print("  [%s] %s" % ("PASS" if ok else "FAIL", name))
        if not ok:
            print("         %s" % detail)
    print("  %d/%d 通過" % (passed, len(results)))
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
