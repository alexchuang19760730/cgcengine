#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""session 屬於哪一類？以及那一類的**資產／能力／進度／復盤**。

要解決的具體問題
----------------
一個 session 的運營者 agent 要能回答：「**我現在這一輪屬於哪一類**」，
然後據此整理對應的四個維度。而「屬於哪一類」最不會漂移的判準是
**它用哪一個帳號登入雲側** —— 不是它自己宣稱，也不是 session 標題。

    運營者  operator   alexchuang@powerauto.ai   （super_admin）
    開發者  developer  developer@powerauto.ai    （admin）
    探索者  explorer   frontier@powerauto.ai     （admin）

★ 判不出來就**拒答**，不要猜一個預設類別。猜的代價是：一個打錯帳號的 session
  會拿到「運營者」的四維，而在它自己看起來完全正常。

四維的內容按類別不同
--------------------
形狀一樣（assets／capabilities／progress／retro），**來源不同**：

| 類別 | 資產 | 能力 | 進度 | 復盤 |
|---|---|---|---|---|
| operator | 機隊全部端點 | 端點已驗證的能力 | 各端加入單完成度 | 逐次上報的能力／未量測變化 |
| developer | 非本機的端點（要移植的那些） | 與建置／移植相關的能力 | 那些端的加入單 | 它們的建置變化 |
| explorer | 決策與假設（不是端點） | 已登錄的決策 | 未裁決的假設 | decisions.jsonl 的動能 |

資料來源是**既有產物**，這一支不重算：`docs/fleet_export.json`（portal 的匯出）
與 `engine_loop/traces/decisions.jsonl`。拿不到就出聲，不假裝是空的。

用法
----
    python3 agent_harness/a2a/identity.py --email alexchuang@powerauto.ai
    python3 agent_harness/a2a/identity.py --jwt "$TOKEN"          # 從 JWT 的 email claim
    python3 agent_harness/a2a/identity.py --class operator --brief
    python3 agent_harness/a2a/identity.py --check                 # 映射與雲側帳號的一致性
    python3 agent_harness/a2a/identity.py --self-test
"""
from __future__ import annotations

import base64
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import taxonomy as TX                                  # noqa: E402

REPO = HERE.parent.parent
EXPORT_JSON = REPO / "docs" / "fleet_export.json"
DECISIONS = REPO / "agent_harness" / "engine_loop" / "traces" / "decisions.jsonl"
ENDPOINTS_DIR = REPO / "agent_harness" / "portal" / "endpoints"
FLEET_JSON = REPO / "agent_harness" / "portal" / "fleet.json"
CLOUD_ENV = Path.home() / ".config" / "powerauto" / "supabase.env"
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# 環境變數名 → 內部短名。**檔案與環境變數用同一組名字** —— 少一層對應就少一個漂移點。
ENV_KEYS = {
    "POWERAUTO_SUPABASE_URL": "url",
    "POWERAUTO_SUPABASE_ANON": "anon",
    "POWERAUTO_SUPABASE_SERVICE": "service",
    "POWERAUTO_EMAIL": "email",
    "POWERAUTO_PASSWORD": "password",
}

# ★★ 每個類別**一個帳號**，密碼共用同一把（2026-09-18 實測：三個都能登入、各有自己的
#   UUID）。刻意**不做**「一個帳號、多個 token」——分類的錨是**帳號**，共用帳號會讓
#   分類退回「自稱」，而那正是這一支在防的事。
#   命名由 `TX.classes()` 派生（不寫死清單）：新增一類時，這裡自動多一個鍵，
#   而 `identity/check` 會立刻說「這一類沒有帳號」。
EMAIL_KEYS = {cid: "POWERAUTO_EMAIL_" + cid.upper() for cid in TX.classes()}
ENV_KEYS.update({name: "email_" + cid for cid, name in EMAIL_KEYS.items()})


# ══════════════════════════════════════════════════════════════════════
#  ① 身分 → 類別（不猜）
# ══════════════════════════════════════════════════════════════════════

def email_of(cid: str) -> str | None:
    c = TX.get(cid)
    return ((c or {}).get("identity") or {}).get("email")


def class_of_email(email: str) -> str | None:
    """★ 只做精確比對（去掉大小寫與空白）。刻意**不做**「包含 @powerauto.ai 就算」
    這種模糊匹配 —— 那會讓一個拼錯的帳號安靜地落到某一類。"""
    e = str(email or "").strip().lower()
    if not e:
        return None
    for cid in TX.classes():
        if (email_of(cid) or "").lower() == e:
            return cid
    return None


def email_from_jwt(token: str) -> str | None:
    """從 JWT 的 payload 取 email。不驗簽章 —— 這一支只回答「這是誰」，
    授權是雲側的事（RLS 與 role 在那一頭）。"""
    parts = str(token or "").split(".")
    if len(parts) < 2:
        return None
    pad = parts[1] + "=" * (-len(parts[1]) % 4)
    try:
        payload = json.loads(base64.urlsafe_b64decode(pad).decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    return payload.get("email") or payload.get("user_metadata", {}).get("email")


def resolve(email: str = "", token: str = "") -> dict:
    """回一個顯式的解析結果。**判不出來時 `class` 是 None 且帶 `why`**，
    呼叫端必須自己決定要不要拒答（`identity.py` 的命令列一律拒答）。"""
    e = str(email or "").strip() or (email_from_jwt(token) or "")
    if not e:
        return {"email": "", "class": None,
                "why": "沒有 email 也沒有可解析的 JWT ⇒ 判不出這一輪屬於哪一類。"}
    cid = class_of_email(e)
    if not cid:
        known = "、".join("%s → %s" % (c, email_of(c)) for c in TX.classes())
        return {"email": e, "class": None,
                "why": "%s 不在身分錨裡（已知：%s）⇒ 拒答，不猜一個預設類別。" % (e, known)}
    return {"email": e, "class": cid, "label": TX.label(cid),
            "role_expected": (TX.get(cid).get("identity") or {}).get("role_expected"),
            "macro": TX.macro(cid), "why": ""}


# ══════════════════════════════════════════════════════════════════════
#  ② 雲側探測（可選；沒有憑證就跳過，並且說出來）
# ══════════════════════════════════════════════════════════════════════

def load_cloud_env(path: Path | None = None) -> dict:
    """憑證**只從環境或 repo 外的檔案**讀。這一支的原始碼裡沒有任何 key ——
    寫死一個 key 進版控，等於把它交給每一個 clone 的人。

    ★★ 檔案裡的鍵是 `POWERAUTO_SUPABASE_URL` 這種**完整環境變數名**，
    而我內部的鍵是 `url`／`anon` 這種短名。第一版我拿完整名去比短名，於是**全部落空**
    —— 症狀是「回報沒有憑證」，也就是說它至少是出聲的（不是靜默），但方向是錯的。
    """
    import os
    cfg = {short: os.environ.get(name, "") for name, short in ENV_KEYS.items()}
    p = path or CLOUD_ENV
    if p.is_file():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, val = line.split("=", 1)
            name = name.strip().lstrip("export").strip()
            short = ENV_KEYS.get(name)
            if short and not cfg[short]:
                cfg[short] = val.strip().strip('"').strip("'")
    # ★ 每類自己的帳號。`POWERAUTO_EMAIL`（沒有類別後綴）只有 operator 繼承 ——
    #   繼承給三類會讓「這份憑證是誰的」變成猜的。
    cfg["emails"] = {}
    for cid in TX.classes():
        m = cfg.get("email_" + cid) or (cfg.get("email", "") if cid == "operator" else "")
        cfg["emails"][cid] = m
    return cfg


def _no_proxy_opener():
    """★ 同一個坑在這條線上是第三次：環境裡的 HTTP_PROXY 會把請求也丟進代理。"""
    import urllib.request
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def login(cfg: dict, cid: str = "", timeout: float = 15.0) -> tuple[str, str, str]:
    """用某一類的帳號登入 ⇒ (JWT, 用了哪個帳號, why)。why 非空 ⇒ 沒拿到。

    走的是**既有的登入流程**（anon key ＋ 密碼換 token），不是 service key ——
    於是後面讀到的就是那條路徑**真的讀得到**的東西（RLS 在雲側那一頭生效）。
    """
    import urllib.error
    import urllib.request
    if not (cfg.get("url") and cfg.get("anon")):
        return "", "", "沒有 url／anon（環境變數或 %s）" % CLOUD_ENV
    mail = (cfg.get("emails") or {}).get(cid) if cid else cfg.get("email", "")
    if not mail:
        # ★ 訊息要指出**是哪一類缺**，以及**該給哪個鍵** —— 第一版在 cid 為空時
        #   印成「沒有  這一類的帳號（要 POWERAUTO_EMAIL_）」，像壞掉而不像缺設定。
        return "", "", ("沒有 %s 這一類的帳號（要設 %s）"
                        % (cid or "?", EMAIL_KEYS.get(cid, "POWERAUTO_EMAIL_<CLASS>")))
    if not cfg.get("password"):
        return "", mail, "沒有密碼（要 POWERAUTO_PASSWORD）"
    base = cfg["url"].rstrip("/")
    try:
        req = urllib.request.Request(
            base + "/auth/v1/token?grant_type=password",
            data=json.dumps({"email": mail, "password": cfg["password"]}).encode(),
            headers={"apikey": cfg["anon"], "Content-Type": "application/json"})
        with _no_proxy_opener().open(req, timeout=timeout) as r:
            tok = json.loads(r.read().decode()).get("access_token")
        if not tok:
            return "", mail, "登入回應沒有 access_token"
        return tok, mail, ""
    except Exception as e:                             # noqa: BLE001
        return "", mail, "登入失敗：%s: %s" % (type(e).__name__, str(e)[:150])


def login_probe(cfg: dict | None = None, timeout: float = 15.0) -> dict:
    """★ 三個帳號**各登一次**。

    使用者說「同樣的密碼」，但**說不是證據**：三個帳號可能沒建、密碼不同、或某一個
    被停用。機檢一次的成本是幾百毫秒，而它回答的是「這條路現在能不能走」。
    這裡**只登入、不寫任何東西**（登入是讀操作；建帳號／改角色才是寫）。
    """
    # ★★ `cfg or load_cloud_env()` 是錯的：空 dict 是 falsy，於是「呼叫端給了一份
    #   **明確的空**憑證」（＝我要離線跑）會被靜默換成真憑證，然後真的連出去。
    #   「我給了空設定」與「我沒給設定」不該同形 —— 這格是自測第 17 格抓到的。
    cfg = load_cloud_env() if cfg is None else cfg
    rows, missing = [], []
    for cid in TX.classes():
        tok, mail, why = login(cfg, cid, timeout)
        rows.append({"class": cid, "email": mail or email_of(cid),
                     "ok": bool(tok), "jwt_len": len(tok), "why": why})
        if not tok:
            missing.append("%s(%s): %s" % (cid, mail or email_of(cid), why or "未知"))
    if not rows or all(not r["ok"] for r in rows):
        # ★ 全部失敗 ⇒ 這是「沒查」（可能是離線或沒憑證），不是「三個帳號都壞了」。
        #   兩件事的處置完全不同：前者等網路，後者要建帳號。
        return {"probed": False, "rows": rows,
                "why": missing[0] if missing else "沒有憑證 ⇒ 跳過登入探測"}
    return {"probed": True, "rows": rows, "why": ("；".join(missing) if missing else "")}


def cloud_profiles(cfg: dict | None = None, timeout: float = 15.0,
                   as_class: str = "") -> tuple[list | None, str]:
    """取 profiles。回 (rows, why)。rows 是 None 代表沒查（缺憑證或失敗），why 說明原因。

    `as_class` 指定**用哪一類的帳號**去讀 —— 於是可以回答一個實質問題：
    「開發者那一類自己去看，看到的跟我看到的一樣嗎？」（RLS 下不一定一樣）。
    """
    # ★ 同上：`cfg or ...` 會把「明確的空憑證」換成真憑證。
    cfg = load_cloud_env() if cfg is None else cfg
    if not (cfg.get("url") and cfg.get("anon")):
        return None, ("沒有雲側憑證（環境變數或 %s）⇒ 跳過帳號探測。"
                      "這不是錯誤：缺席要出聲，但缺席不是失敗。" % CLOUD_ENV)
    tok, mail, why = login(cfg, as_class, timeout)
    if not tok:
        return None, why
    import urllib.request
    base = cfg["url"].rstrip("/")
    try:
        req = urllib.request.Request(
            base + "/rest/v1/profiles?select=id,email,role",
            headers={"apikey": cfg["anon"], "Authorization": "Bearer " + tok})
        with _no_proxy_opener().open(req, timeout=timeout) as r:
            rows = json.loads(r.read().decode())
        return rows, "" if isinstance(rows, list) else "回應不是陣列：%r" % (rows,)
    except Exception as e:                             # noqa: BLE001
        return None, "讀 profiles 失敗（用 %s）：%s: %s" % (mail, type(e).__name__,
                                                        str(e)[:140])


def identity_problems(profiles: list | None, why: str = "") -> list[str]:
    """身分錨與雲側帳號的一致性。**三種結果分開講**（這是這一支的重點）：

      - 帳號存在且角色相符 ⇒ OK
      - 帳號**不存在** ⇒ 「這一類目前沒有雲側身分」＝ absent（不是錯，但要說出來）
      - 帳號存在但**角色不符** ⇒ 這是真的錯（身分錨說 admin，實際是 user，
        那會讓這個類別的 agent 拿到比預期少的權限，或在另一端拿到更多）
    """
    P: list[str] = []
    seen: dict[str, str] = {}
    for cid in TX.classes():
        e = email_of(cid)
        if not e:
            P.append("[%s] 沒有 identity.email ⇒ 這一類無法從帳號判別" % cid)
            continue
        if not EMAIL_RE.match(e):
            P.append("[%s] identity.email 不是合法 email：%r" % (cid, e))
        low = e.lower()
        if low in seen:
            P.append("email 重複：%s 同時屬於 %s 與 %s（判別會不確定）" % (e, seen[low], cid))
        seen[low] = cid

    if profiles is None:
        return P + ["（沒有查雲側：%s）" % (why or "未提供憑證")]
    by_mail = {str(r.get("email") or "").lower(): r for r in profiles}
    for cid in TX.classes():
        e = (email_of(cid) or "").lower()
        want = (TX.get(cid).get("identity") or {}).get("role_expected")
        row = by_mail.get(e)
        if not row:
            P.append("[%s] 雲側**沒有**這個帳號：%s ⇒ 這一類目前沒有雲側身分（absent，"
                     "不是錯誤；但要建帳號才能用它登入）" % (cid, e))
            continue
        if want and row.get("role") != want:
            P.append("[%s] %s 的角色是 %r，而身分錨期望 %r ⇒ 兩邊說法不一致（這是真的錯，"
                     "不是缺席）" % (cid, e, row.get("role"), want))
    return P


# ══════════════════════════════════════════════════════════════════════
#  ②' 本機上下文 —— 取「出處」，不是取「身分」
# ══════════════════════════════════════════════════════════════════════

# 四維是從這些檔讀出來的。★ 列在這裡的是「讀得到才算數」的東西：
# **不存在也要出現在清單裡**（標 exists=False），不是留白 —— 留白會讓
# 「還沒匯出」與「我忘了檢查」長得一樣。
ANCHORS = (
    ("taxonomy", "agent_harness/a2a/taxonomy.py", "分類的正本（三類＋宏觀域＋微觀能力）"),
    ("registry", "agent_harness/a2a/registry.json", "有哪些 agent 實例（含 runs_on）"),
    ("fleet", "agent_harness/portal/fleet.json", "端點怎麼到達（reaches／channels）"),
    ("endpoints", "agent_harness/portal/endpoints", "各端的上報現況"),
    ("export", "docs/fleet_export.json", "給外部網站吃的匯出（★ 少 reaches 那一格）"),
    ("decisions", "agent_harness/engine_loop/traces/decisions.jsonl", "決策登錄"),
)
# 端側那一個 repo 的關鍵檔：能讀到就記，讀不到就 absent（不假裝看過）。
EDGE_SERVER = Path.home() / "Documents" / "powerauto.ai" / "installer" / "edge_server.py"


def _now_iso() -> str:
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _git(args: list[str], cwd: Path | None = None) -> str:
    import subprocess
    try:
        p = subprocess.run(["git", *args], cwd=str(cwd or REPO),
                           capture_output=True, text=True, timeout=10)
    except Exception:                                  # noqa: BLE001
        return ""
    return p.stdout.strip() if p.returncode == 0 else ""


def _stat_of(p: Path) -> dict:
    import datetime
    if not p.exists():
        return {"exists": False}
    try:
        st = p.stat()
    except OSError:
        return {"exists": True, "readable": False}
    return {"exists": True,
            "mtime": datetime.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "bytes": st.st_size if p.is_file() else None}


def local_context(cwd: Path | None = None) -> dict:
    """這一輪的**出處**：在哪個 repo、哪個 commit、有沒有未提交的改動之下讀出這份四維。

    ★★ 2026-09-18 決定：**不做 session 級註冊**。
    同一帳號同時開兩個 session（一個做移植、一個做運營）拿到的是**同一類**的四維 ——
    這是有意接受的，不是漏掉的功能。理由：能區分兩者的東西（本機檔案、git 操作）
    本來就擺在那裡，而且兩個 session 看到的是**同一份**；再加一層註冊表只會多一份
    會漂移的真相（這個 repo 已經為「兩份真相」付過學費）。

    所以這裡給的是**出處**：於是同帳號的兩個 session 雖然同類，卻可以互相知道
    「你讀的是哪一版」—— 歸因靠出處，不靠註冊。
    """
    dirty = [ln for ln in _git(["status", "--porcelain", "--untracked-files=all"]).splitlines()
             if ln.strip()]
    remotes: dict[str, str] = {}
    for ln in _git(["remote", "-v"]).splitlines():
        parts = ln.split()
        if len(parts) >= 2:
            remotes.setdefault(parts[0], parts[1])
    anchors = []
    for key, rel, what in ANCHORS:
        s = _stat_of(REPO / rel)
        anchors.append({"key": key, "path": rel, "what": what, **s})
    anchors.append({"key": "edge_server", "path": str(EDGE_SERVER),
                    "what": "端側對外入口（離線端的 --dump-status 在這一支）",
                    **_stat_of(EDGE_SERVER)})
    return {
        "cwd": str(cwd or Path.cwd()),
        "repo": str(REPO),
        "git": {"branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
                "head": _git(["rev-parse", "HEAD"])[:12],
                "subject": _git(["log", "-1", "--format=%s"]),
                "dirty_count": len(dirty),
                "dirty": dirty,
                "remotes": remotes},
        "anchors": anchors,
        "observedAt": _now_iso(),
        "sessionDiscrimination": {
            "level": "none",
            "byDesign": True,
            "why": ("身分只到**帳號**這一層。同一帳號的兩個 session 會判到同一類 —— "
                    "本機檔案與 git 操作對兩者是**同一份**，所以歸因靠上面的出處"
                    "（repo／HEAD／dirty），不靠一層會漂移的 session 註冊表。"),
            "what_would_change_it": ("真要分到 session 層，需要「session 開在哪個 cwd、"
                                     "動過哪些檔」這類事實被**登錄**下來；目前刻意不做。"),
        },
    }


# ══════════════════════════════════════════════════════════════════════
#  ③ 那一類的資產／能力／進度／復盤
# ══════════════════════════════════════════════════════════════════════

def _load_export() -> tuple[dict | None, str]:
    if not EXPORT_JSON.is_file():
        return None, ("還沒匯出（%s 不存在）。在 portal 那側跑 "
                      "`build_fleet_portal.py --export` 就會有。" % EXPORT_JSON)
    try:
        return json.loads(EXPORT_JSON.read_text(encoding="utf-8")), ""
    except (OSError, json.JSONDecodeError) as e:
        return None, "匯出檔讀不出來：%s" % str(e)[:120]


def _decisions() -> list[dict]:
    if not DECISIONS.is_file():
        return []
    out = []
    for line in DECISIONS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _momentum() -> dict:
    """決策動能。★ 刻意**呼叫 portal 那一支**而不是在這裡再寫一份 ——
    兩份實作必然漂移，而漂移是靜默的。拿不到就回空 + 原因。"""
    try:
        sys.path.insert(0, str(REPO / "agent_harness" / "portal"))
        import build_fleet_portal as BFP             # noqa: PLC0415
        m = BFP.momentum_block()
        return {"counts": m.get("summary"), "verdict": m.get("verdict"),
                "refuted_pct": m.get("refuted_pct"), "orphan_pct": m.get("orphan_pct"),
                "from": "agent_harness/portal/build_fleet_portal.py::momentum_block()"}
    except Exception as e:                             # noqa: BLE001
        return {"error": "%s: %s" % (type(e).__name__, str(e)[:140]),
                "from": "（呼叫 momentum_block() 失敗）"}


def _ep_view(ep: dict) -> dict:
    return {"id": ep.get("id"), "kind": ep.get("kind"), "platform": ep.get("platform"),
            "arch": ep.get("arch"), "chip": ep.get("chip"),
            "heartbeat": ep.get("heartbeat"), "report_age": ep.get("report_age"),
            "reported_at": ep.get("reported_at"),
            "capabilities": ep.get("capabilities") or [],
            "not_measured": ep.get("not_measured") or [],
            "assets": ep.get("assets") or {},
            "join": (ep.get("join") or {}).get("bar") if ep.get("join") else None,
            "retro": ep.get("retro") or []}


def brief_for(cid: str) -> dict:
    """按類別整理四維。回一個 dict，`not_measured` 明列**這一支沒能算出來的東西**。"""
    c = TX.get(cid)
    if not c:
        return {"class": cid, "error": "沒有這個類別", "known": TX.classes()}
    exp, exp_why = _load_export()
    eps = (exp or {}).get("endpoints") or []
    nm: list[str] = []

    # ★ 「哪些端點是移植目標」＝`reaches == "offline"`（我們連不上、得主動送過去）。
    #   ★★ 但 `docs/fleet_export.json` **沒有導出 `reaches`** —— 那是 portal 的
    #   `export_payload()` 的一個真缺口：一份「給別的網站吃」的資料少了判斷所需的欄位，
    #   於是每一個消費者都得自己回頭讀 fleet.json。這裡就是那樣做的，並且把缺口說出來。
    reaches: dict[str, str] = {}
    if FLEET_JSON.is_file():
        try:
            for e in (json.loads(FLEET_JSON.read_text(encoding="utf-8"))
                      .get("endpoints") or []):
                reaches[str(e.get("id"))] = str(e.get("reaches") or "")
        except (OSError, json.JSONDecodeError) as e:
            nm.append("fleet.json 讀不出來 ⇒ 移植目標只能用匯出檔判斷（那裡面沒有 reaches）：%s"
                      % str(e)[:100])
    else:
        nm.append("找不到 fleet.json ⇒ 無法判斷哪些端點是「移植目標」（reaches）。")
    if eps and not reaches:
        nm.append("★ 匯出檔沒有 reaches 欄位，而 fleet.json 也讀不到 ⇒ 移植類的資產"
                  "無法判斷（這一格是**不知道**，不是「空的」）。")

    out = {
        "class": cid, "label": c["label"], "what": c["what"],
        "identity": dict(c.get("identity") or {}),
        "macro": dict(c.get("macro") or {}),
        # ★★ 出處：這份四維是在哪個 repo／哪個 HEAD／有沒有未提交改動之下讀出來的。
        #   它不是身分（身分只到帳號），而是**歸因**——同帳號的兩個 session 靠它
        #   互相知道「你讀的是哪一版」。
        "context": local_context(),
        "source": {"fleet_export": "docs/fleet_export.json" if exp else None,
                   "decisions": "agent_harness/engine_loop/traces/decisions.jsonl"
                                if DECISIONS.is_file() else None},
        "not_measured": nm,
    }

    if cid == "operator":
        out["assets"] = [_ep_view(e) for e in eps]
        out["capabilities"] = sorted({x for e in eps for x in (e.get("capabilities") or [])})
        out["progress"] = {"endpoints": len(eps),
                           "joined": sum(1 for e in eps if (e.get("join") or {}).get("bar")),
                           "bars": {e.get("id"): (e.get("join") or {}).get("bar")
                                    for e in eps if e.get("join")}}
        out["retro"] = [{"id": e.get("id"), "retro": e.get("retro") or []}
                        for e in eps if e.get("retro")]

    elif cid == "developer":
        # ★ 「移植目標」的判準是 `reaches == "offline"`，**不是** platform 白名單。
        #   第一版我用「platform 不是 darwin」來篩 —— 結果把 cloud-host2（linux）也算了進來，
        #   而它是雲端主機、根本不需要移植。真正的定義是：
        #   **我們連不上、得主動把東西送過去**的那些端點。
        port = [e for e in eps if reaches.get(str(e.get("id"))) == "offline"]
        if not port and eps:
            nm.append("匯出檔裡沒有『非本機』的端點 ⇒ 移植類的資產目前是空的。"
                      "這是真的空，不是沒算（Windows／鴻蒙還沒上報過）。")
        out["assets"] = [_ep_view(e) for e in port]
        out["capabilities"] = sorted({x for e in port for x in (e.get("capabilities") or [])})
        out["progress"] = {"endpoints": len(port),
                           "bars": {e.get("id"): (e.get("join") or {}).get("bar")
                                    for e in port if e.get("join")},
                           "open_items": sum((e.get("join") or {}).get("counts", {}).get("todo", 0)
                                             for e in port)}
        out["retro"] = [{"id": e.get("id"), "retro": e.get("retro") or []} for e in port]
        out["micro_scope"] = TX.skill_ids(cid)

    else:  # explorer
        dec = _decisions()
        out["assets"] = [{"decisions": len(dec), "with_evidence":
                          sum(1 for d in dec if d.get("evidence")) if dec else None,
                          "path": "agent_harness/engine_loop/traces/decisions.jsonl"}]
        out["capabilities"] = [s["id"] for s in c["micro"]]
        out["progress"] = {"decisions": len(dec) if dec else None,
                           "pending": sum(1 for d in dec if d.get("judgement") == "pending")
                           if dec else None}
        out["retro"] = [{"momentum": _momentum(),
                         "note": "動能＝被後續決策納入的程度（carried／replaced／refuted／"
                                 "orphan／pending），從 decisions.jsonl 實算。"}]
        closed = [d.get("question") or d.get("decision_id") for d in dec
                  if d.get("judgement") == "refuted"][:5]
        out["recently_refuted"] = closed
        if not dec:
            nm.append("decisions.jsonl 讀不到或為空 ⇒ 探索類的四維沒有正本。")

    if exp_why:
        nm.append("機隊匯出缺席：%s" % exp_why)
    # ★ 出處裡「讀不到」的關鍵錨要進 not_measured：四維是從那些檔算出來的，
    #   少一個就少一格依據 —— 而那件事不該只藏在 context 裡。
    for a in out["context"]["anchors"]:
        if not a.get("exists") and a["key"] in ("taxonomy", "fleet", "decisions"):
            nm.append("出處缺失：%s（%s）讀不到 ⇒ 依賴它的那一維不是算出來的。"
                      % (a["path"], a["what"]))
    return out


# ══════════════════════════════════════════════════════════════════════
#  命令列
# ══════════════════════════════════════════════════════════════════════

def _print_context(ctx: dict) -> None:
    """★ 這一段刻意**不叫「身分」**——身分只到帳號，同一帳號的兩個 session 同類。
    它給的是「你讀的是哪一版」，那是兩個 session 之間唯一需要互相知道的東西。"""
    g = ctx.get("git") or {}
    anchors = ctx.get("anchors") or []
    miss = [a["path"] for a in anchors if not a.get("exists")]
    print("出處（★ 這不是身分，是歸因）：")
    print("    repo    %s" % ctx.get("repo"))
    print("    cwd     %s" % ctx.get("cwd"))
    print("    HEAD    %s %s" % (g.get("branch") or "?", g.get("head") or ""))
    print("    最近提交 %s" % (g.get("subject") or "(無)")[:74])
    print("    未提交  %d 筆%s" % (g.get("dirty_count") or 0,
                                  "（分不出是誰改的：多 session 共用一棵樹）"
                                  if g.get("dirty_count") else ""))
    for ln in (g.get("dirty") or [])[:6]:
        print("      %s" % ln[:88])
    if len(g.get("dirty") or []) > 6:
        print("      …還有 %d 筆" % (len(g["dirty"]) - 6))
    print("    錨文件  %d 個讀得到、%d 個讀不到%s"
          % (len(anchors) - len(miss), len(miss),
             ("（" + "、".join(miss[:3]) + "）") if miss else ""))
    if miss:
        for a in anchors:
            if not a.get("exists"):
                print("      ✗ %-46s %s" % (a["path"], a["what"]))
    print("    讀數時刻 %s" % ctx.get("observedAt"))
    sd = ctx.get("sessionDiscrimination") or {}
    print("    ★ 同一帳號的多個 session 會判到**同一類**（level=%s, byDesign=%s）——"
          % (sd.get("level"), sd.get("byDesign")))
    print("      本機檔案與 git 操作對它們是同一份。歸因靠上面這個出處，"
          "不靠一層會漂移的 session 註冊表。")


def _print_brief(b: dict) -> None:
    print("類別：%s（%s）" % (b.get("class"), b.get("label")))
    print("身分：%s（期望角色 %s）" % (b.get("identity", {}).get("email"),
                                      b.get("identity", {}).get("role_expected")))
    mac = b.get("macro") or {}
    print("宏觀決策域：%s — %s" % (mac.get("scope"), mac.get("label")))
    for d in mac.get("decisions") or []:
        print("    · %s" % d)
    print()
    for k, zh in (("assets", "資產"), ("capabilities", "能力"),
                  ("progress", "進度"), ("retro", "復盤")):
        v = b.get(k)
        if k == "capabilities" and isinstance(v, list):
            print("%s（%d）：" % (zh, len(v)))
            for x in v[:8]:
                print("    · %s" % str(x)[:96])
        elif k == "assets" and isinstance(v, list):
            # ★ 資產要**摘要**，不是把整個 dict 倒出來 —— 後者列印得出來，但沒人讀得下去，
            #   而「印得出來」與「讀得懂」是兩件事。
            print("%s（%d）：" % (zh, len(v)))
            for x in v:
                if isinstance(x, dict) and x.get("id"):
                    print("    · %-22s %-9s 能力 %-2d 條、未量測 %-2d 條%s"
                          % (x["id"], x.get("platform") or x.get("kind") or "",
                             len(x.get("capabilities") or []), len(x.get("not_measured") or []),
                             ("　上報 %s" % x["reported_at"]) if x.get("reported_at") else ""))
                else:
                    print("    · %s" % json.dumps(x, ensure_ascii=False)[:150])
        elif k == "retro" and isinstance(v, list):
            print("%s（%d 個有歷史的端點）：" % (zh, len(v)))
            for x in v:
                if not isinstance(x, dict) or "id" not in x:
                    print("    · %s" % json.dumps(x, ensure_ascii=False)[:300])
                    continue
                n = len(x.get("retro") or [])
                print("    · %-22s %d 筆逐次觀測%s" % (x.get("id"), n,
                      ("，最近 " + str((x["retro"][-1] or {}).get("ts"))) if n else ""))
        else:
            print("%s：%s" % (zh, json.dumps(v, ensure_ascii=False)[:420]))
    if b.get("recently_refuted"):
        print("已推翻（最近 5 筆）：%s" % json.dumps(b["recently_refuted"], ensure_ascii=False)[:300])

    ctx = b.get("context") or {}
    if ctx:
        print()
        _print_context(ctx)

    if b.get("not_measured"):
        print()
        print("★ 這一支沒有算到的（缺席要出聲）：")
        for x in b["not_measured"]:
            print("    - %s" % x)


def self_test() -> int:
    """突變式自測。不連網（雲側那一段用假的 profiles fixture）。"""
    res: list[tuple[str, bool, str]] = []

    def case(n, ok, d=""):
        res.append((n, bool(ok), d))

    # 1) 三個身分錨存在、格式合法、不重複
    mails = [email_of(c) for c in TX.classes()]
    case("三類都有 identity.email，格式合法且互不重複",
         all(m and EMAIL_RE.match(m) for m in mails) and len(set(mails)) == len(mails),
         "mails=%s" % mails)

    # 2) 帳號 → 類別（含大小寫與空白）
    case("★ 帳號精確對應到類別（含大小寫／空白正規化）",
         class_of_email("  Developer@PowerAuto.AI ") == "developer"
         and class_of_email("alexchuang@powerauto.ai") == "operator"
         and class_of_email("frontier@powerauto.ai") == "explorer",
         "dev=%s op=%s exp=%s" % (class_of_email(" Developer@PowerAuto.AI "),
                                  class_of_email("alexchuang@powerauto.ai"),
                                  class_of_email("frontier@powerauto.ai")))

    # 3) ★★ 不認識的帳號 ⇒ 拒答（不猜預設類別）
    r = resolve(email="someone@else.com")
    case("★★ 不認識的帳號 ⇒ class=None 且帶 why（拒答，不猜一個預設類別）",
         r["class"] is None and "拒答" in r["why"], json.dumps(r, ensure_ascii=False)[:150])

    # 4) 完全沒有身分 ⇒ 也拒答
    r2 = resolve()
    case("★ 沒有 email 也沒有 JWT ⇒ 拒答", r2["class"] is None and r2["why"], r2["why"][:100])

    # 5) ★ 「同網域就算同一類」這種模糊匹配**不可以**成立
    case("★ 同網域但不在錨上的帳號（nobody@powerauto.ai）⇒ 拒答",
         class_of_email("nobody@powerauto.ai") is None,
         "class_of_email('nobody@powerauto.ai')=%r" % class_of_email("nobody@powerauto.ai"))

    # 6) JWT 解析（真的組一個 JWT 形狀的字串）
    import base64 as b64
    payload = b64.urlsafe_b64encode(json.dumps(
        {"email": "frontier@powerauto.ai", "sub": "x"}).encode()).decode().rstrip("=")
    tok = "eyJhbGciOiJFUzI1NiJ9." + payload + ".sig"
    case("★ 從 JWT 的 email claim 解析出類別",
         class_of_email(email_from_jwt(tok)) == "explorer",
         "email_from_jwt=%r" % email_from_jwt(tok))

    # 7) ★★ 雲側帳號缺席 ⇒ 是 absent（要說出來），不是「錯誤」
    profs = [{"email": "alexchuang@powerauto.ai", "role": "super_admin"}]
    P = identity_problems(profs)
    case("★★ 雲側缺 developer／frontier ⇒ 被報成 absent（不是錯誤），且 operator 通過",
         len(P) == 2 and all("沒有" in x and "absent" in x for x in P)
         and not any("alexchuang" in x for x in P),
         "problems=%s" % P)

    # 8) ★★ 角色不符 ⇒ 這是真的錯（與缺席分開）
    #    ★ fixture 的角色要**跟著 taxonomy 走**：第一版這裡寫死 "admin"，
    #    後來把 role_expected 改成 super_admin 之後，frontier 那筆也變成不一致，
    #    而這一格的斷言是「恰好一筆」⇒ 自測的 fixture 也有保鮮期。
    _exp = {c: (TX.get(c).get("identity") or {}).get("role_expected") for c in TX.classes()}
    profs2 = profs + [{"email": "developer@powerauto.ai", "role": "user"},
                      {"email": "frontier@powerauto.ai", "role": _exp["explorer"]}]
    P2 = identity_problems(profs2)
    case("★★ 帳號存在但角色不符 ⇒ 報成「不一致」（與缺席分開講）",
         len(P2) == 1 and "不一致" in P2[0] and "developer" in P2[0], "problems=%s" % P2)

    # 9) 沒查雲側時要說出來，而不是靜默當作 OK
    P3 = identity_problems(None, "沒憑證")
    case("★ 沒查雲側 ⇒ 明確說「沒有查」，不是靜默通過",
         any("沒有查雲側" in x for x in P3), "problems=%s" % P3)

    # 10) 全帳號齊且角色對 ⇒ 無問題
    profs3 = [{"email": m, "role": (TX.get(c).get("identity") or {}).get("role_expected")}
              for c, m in zip(TX.classes(), mails)]
    case("帳號與角色都對 ⇒ 沒有問題", identity_problems(profs3) == [],
         "problems=%s" % identity_problems(profs3))

    # 11) 三類的 brief 都能算出來，且四維齊備
    bad = []
    for c in TX.classes():
        b = brief_for(c)
        if b.get("error") or not all(k in b for k in ("assets", "capabilities", "progress", "retro")):
            bad.append((c, b.get("error")))
    case("★ 三類的 brief 都算得出來且四維齊備（資產／能力／進度／復盤）",
         not bad, "bad=%s" % bad)

    # 12) ★ 沒有的東西要出現在 not_measured，不是靜默留白
    b_op = brief_for("operator")
    case("★ 機隊匯出缺席時，brief 要把它列進 not_measured（不是留白）",
         (EXPORT_JSON.is_file() and not b_op["not_measured"])
         or (not EXPORT_JSON.is_file() and any("匯出" in x for x in b_op["not_measured"])),
         "export_exists=%s nm=%s" % (EXPORT_JSON.is_file(), b_op.get("not_measured")))

    # 13) ★ 原始碼裡沒有硬編碼的 key（這一格防的是「順手把金鑰貼進來」）
    #     ★★ pattern 要**拼出來**，不能寫成字面量 —— 我第一版就寫成字面量，
    #     於是掃描命中了自己的 pattern（檢查器的錯與被檢物的錯同形），
    #     而它報的 FAIL 看起來像「原始碼裡真的有 key」。
    src = Path(__file__).read_text(encoding="utf-8")
    needles = ["sb_" + "publishable_", "sb_" + "secret_"]
    case("★ 這一支的原始碼裡沒有任何 Supabase key（憑證只從環境或 repo 外讀）",
         not any(n in src for n in needles),
         "found=%s" % [n for n in needles if n in src])

    # 14) ★★ 出處：本機上下文（這一段取代了「session 級註冊」）
    ctx = local_context()
    g = ctx.get("git") or {}
    sd = ctx.get("sessionDiscrimination") or {}
    case("★★ local_context 給出處（repo／HEAD／anchors），且明說**不區分 session**",
         ctx.get("repo") == str(REPO) and bool(ctx.get("anchors"))
         and len(g.get("dirty") or []) == g.get("dirty_count", -1)
         and sd.get("level") == "none" and sd.get("byDesign") is True,
         "repo=%s head=%s dirty=%s sd=%s" % (ctx.get("repo"), g.get("head"),
                                             g.get("dirty_count"), sd))

    # 15) ★ 缺席的錨要留在清單裡（標 exists=False），不是從清單消失 ——
    #     後者會讓「還沒匯出」與「我忘了檢查」長得一樣。
    keys = {a["key"] for a in ctx["anchors"]}
    case("★ 錨清單固定（缺檔也留著，標 exists=False），且含端側的 edge_server",
         {"taxonomy", "fleet", "decisions", "edge_server"} <= keys
         and all("exists" in a for a in ctx["anchors"]),
         "keys=%s" % sorted(keys))

    # 16) ★★ 四維帶出處，且出處指向**同一棵樹**（同帳號的兩個 session 靠它互相歸因）
    b_ctx = brief_for("operator").get("context") or {}
    case("★★ 四維帶出處（repo／HEAD 與 --here 一致）⇒ 歸因不靠 session 註冊表",
         b_ctx.get("repo") == str(REPO)
         and (b_ctx.get("git") or {}).get("head") == g.get("head"),
         "in-brief=%s vs here=%s" % ((b_ctx.get("git") or {}).get("head"), g.get("head")))

    # 17) ★★ 沒憑證時要說「沒查」（probed=False），不是「三個帳號都壞了」——
    #     這兩件事的處置完全不同：前者等憑證，後者要建帳號。
    pr = login_probe({})
    case("★★ login_probe 沒憑證 ⇒ probed=False（＝沒查），且三類都列出來",
         pr.get("probed") is False and len(pr.get("rows") or []) == len(TX.classes())
         and all(r.get("email") for r in pr["rows"]),
         json.dumps(pr, ensure_ascii=False)[:200])

    passed = sum(1 for _, ok, _ in res if ok)
    for n, ok, d in res:
        print("  [%s] %s" % ("PASS" if ok else "FAIL", n))
        if not ok:
            print("         %s" % d)
    print("  %d/%d 通過" % (passed, len(res)))
    return 0 if passed == len(res) else 1


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="session 屬於哪一類，以及那一類的四維")
    ap.add_argument("--email")
    ap.add_argument("--jwt", help="從 JWT 的 email claim 判別")
    ap.add_argument("--class", dest="cid", choices=TX.classes())
    ap.add_argument("--brief", action="store_true", help="連四維一起印")
    ap.add_argument("--check", action="store_true", help="映射與雲側帳號的一致性")
    ap.add_argument("--here", action="store_true",
                    help="印這一輪的**出處**（repo／HEAD／未提交改動／錨文件）")
    ap.add_argument("--probe-login", action="store_true",
                    help="三個帳號各登入一次（★ 只讀：不建帳號、不改角色）")
    ap.add_argument("--as-class", dest="as_class", choices=TX.classes(),
                    help="用哪一類的帳號去讀雲側（預設 operator）")
    ap.add_argument("--no-cloud", action="store_true", help="不查雲側")
    ap.add_argument("--json", dest="as_json", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)

    if a.self_test:
        return self_test()

    if a.here:
        ctx = local_context()
        if a.as_json:
            print(json.dumps(ctx, ensure_ascii=False, indent=2))
        else:
            _print_context(ctx)
        return 0

    if a.probe_login:
        pr = login_probe()
        if a.as_json:
            print(json.dumps(pr, ensure_ascii=False, indent=2))
            return 0 if pr["probed"] and not pr["why"] else 1
        for r in pr["rows"]:
            print("  [%s] %-10s %-26s %s"
                  % ("OK" if r["ok"] else "FAIL", r["class"], r["email"],
                     ("JWT %d 字" % r["jwt_len"]) if r["ok"] else (r["why"] or "")[:88]))
        if not pr["probed"]:
            print("  沒有查（%s）—— 缺席要出聲，但缺席不是失敗。" % pr["why"])
            return 0
        return 0 if not pr["why"] else 1

    if a.check:
        # ★ 缺省是 operator（運營者），不是空字串 —— 後者會讓「沒指定」變成
        #   「指定了一個不存在的類別」，而錯誤訊息看起來像壞掉。
        profs, why = ((None, "指定了 --no-cloud") if a.no_cloud
                      else cloud_profiles(as_class=a.as_class or "operator"))
        P = identity_problems(profs, why)
        if a.as_json:
            print(json.dumps({"profiles": profs, "problems": P}, ensure_ascii=False, indent=2))
            return 1 if any("不一致" in x or "重複" in x for x in P) else 0
        for cid in TX.classes():
            print("  %-11s %-24s 期望角色 %s"
                  % (cid, email_of(cid), (TX.get(cid).get("identity") or {}).get("role_expected")))
        print()
        if profs is None:
            print("  雲側：沒有查（%s）" % why)
        else:
            print("  雲側：查到 %d 個帳號（以 %s 登入讀取）"
                  % (len(profs),
                     (load_cloud_env().get("emails") or {}).get(a.as_class or "operator")
                     or "?"))
        for x in P:
            print("  %s %s" % ("[error]" if "不一致" in x else "[note] ", x))
        # ★ rc 契約：缺席不是失敗（0）；角色不一致才是錯（1）。
        return 1 if any("不一致" in x or "重複" in x or "不是合法" in x for x in P) else 0

    if a.cid:
        r = {"email": email_of(a.cid), "class": a.cid, "label": TX.label(a.cid),
             "role_expected": (TX.get(a.cid).get("identity") or {}).get("role_expected"),
             "macro": TX.macro(a.cid), "why": ""}
    else:
        r = resolve(a.email or "", a.jwt or "")
    if r["class"] is None:
        print("[error] %s" % r["why"], file=sys.stderr)
        return 2

    if a.brief:
        b = brief_for(r["class"])
        print(json.dumps(b, ensure_ascii=False, indent=2) if a.as_json else "", end="")
        if not a.as_json:
            _print_brief(b)
        return 0
    print(json.dumps(r, ensure_ascii=False, indent=2) if a.as_json
          else "  %s → %s（%s）" % (r["email"], r["class"], r["label"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
