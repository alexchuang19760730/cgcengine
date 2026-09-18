#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 taxonomy ＋ registry 編譯成 A2A 的 Agent Card。

一張 card 要回答四件事，而且第四件是這個 repo 的規矩：
  1. 我是誰、在哪（標準欄位：name／description／url／version）
  2. 我能被怎麼呼叫（capabilities：streaming／pushNotifications／stateTransitionHistory）
  3. 我能執行什麼（**標準** `skills[]` —— 微觀能力，標準客戶端也看得懂）
  4. ★ 我屬於哪一類、我**無權**做什麼（`x-agent-class`／`x-macro-scope`／`not_capable`）

★ 為什麼分類放在擴展欄位而不是硬塞進 skills：
  A2A 的 `skills[]` 語意是「可呼叫的能力」。把「類別」也塞進去，會讓一個標準客戶端
  以為 `class:developer` 是一個可以被呼叫的動作 —— 它會去呼叫它，然後失敗。
  「類別」與「能力」是兩個層次，協議上就該分開。

★ `x-taxonomy.skill_owner` 是這一套能自我驗證的關鍵：它把「每個 skill 屬於哪一類」
  連同來源檔案一起交出去，所以**對端可以自己核對**我們宣稱的分類與能力歸屬一致，
  而不是只能相信這張卡。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import taxonomy as TX                      # noqa: E402  分類的單一真相

REGISTRY = HERE / "registry.json"
PROTOCOL_VERSION = "0.3.0"
CARD_VERSION = "1"

# A2A 的標準欄位。擴展欄位一律 `x-` 開頭（本檔自己也要守這條，否則下一個人會以為
# 那是標準欄位而照抄 A2A 規格去解析）。
STANDARD_FIELDS = ("protocolVersion", "name", "description", "url", "preferredTransport",
                   "additionalInterfaces", "version", "provider", "capabilities",
                   "defaultInputModes", "defaultOutputModes", "skills",
                   "supportsAuthenticatedExtendedCard", "securitySchemes", "security",
                   "signatures", "iconUrl", "documentationUrl")


def load_registry(path: Path | None = None) -> dict:
    p = path or REGISTRY
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit("[error] 找不到 %s" % p)
    except json.JSONDecodeError as e:
        raise SystemExit("[error] %s 不是合法 JSON: %s" % (p, e))


def url_for(agent_id: str, gw: dict, host: str = "127.0.0.1") -> str:
    return "http://%s:%d%s/%s" % (host, gw.get("default_port", 9210),
                                  gw.get("path_prefix", "/a2a"), agent_id)


def card_url_for(agent_id: str, gw: dict, host: str = "127.0.0.1") -> str:
    return ("http://%s:%d%s/%s/.well-known/agent-card.json"
            % (host, gw.get("default_port", 9210), gw.get("path_prefix", "/a2a"), agent_id))


def _skills_for(agent: dict) -> tuple[list[dict], list[str]]:
    """回 (skills, problems)。skills ＝ 這一類的 micro ＋ 實例額外接的。"""
    P: list[str] = []
    c = TX.get(agent.get("class"))
    if not c:
        return [], ["[%s] class=%r 不在 taxonomy 裡（類別寫錯不會報錯，只會讓這個 agent "
                    "變成無類別的孤兒）" % (agent.get("id"), agent.get("class"))]
    skills = [dict(s) for s in c["micro"]]
    have = {s["id"] for s in skills}
    for extra in agent.get("skill_override") or []:
        sid = extra.get("id")
        if sid in have:
            P.append("[%s] skill_override 的 %r 已經在 %s 類別裡了 —— "
                     "要嘛改 taxonomy，要嘛改這個名字；兩份同名能力會讓路由無從決定"
                     % (agent.get("id"), sid, c["id"]))
            continue
        for k in ("id", "name", "description"):
            if not extra.get(k):
                P.append("[%s] skill_override 的 %r 缺 %s" % (agent.get("id"), sid, k))
        skills.append({"id": sid, "name": extra.get("name", ""),
                       "description": extra.get("description", ""),
                       "tags": list(extra.get("tags") or ["extension"]),
                       "examples": list(extra.get("examples") or [])})
        have.add(sid)
    return skills, P


def build_card(agent: dict, gw: dict, host: str = "127.0.0.1") -> tuple[dict, list[str]]:
    """回 (card, problems)。problems 非空時 card 仍然是完整的 —— 讓檢查器去指名，
    而不是讓呼叫端拿到一個殘缺的物件。"""
    aid = agent.get("id") or "(無 id)"
    skills, P = _skills_for(agent)
    c = TX.get(agent.get("class")) or {}
    mac = TX.macro(agent.get("class")) or {}

    if not agent.get("id"):
        P.append("registry 有一個 agent 沒有 id")
    if not agent.get("display_name"):
        P.append("[%s] 缺 display_name" % aid)
    if not agent.get("description"):
        P.append("[%s] 缺 description —— 一張沒有說明自己是什麼的卡片，對端只能猜" % aid)
    if not agent.get("not_capable"):
        P.append("[%s] 缺 not_capable —— 只寫「能做什麼」的卡片，會讓對端把「沒寫」讀成"
                 "「可以做」（registry 的 rules 明寫這條必填）" % aid)

    card = {
        "protocolVersion": PROTOCOL_VERSION,
        "name": agent.get("display_name", aid),
        "description": agent.get("description", ""),
        "url": url_for(aid, gw, host),
        "preferredTransport": "JSONRPC",
        "additionalInterfaces": [{"url": url_for(aid, gw, host), "transport": "JSONRPC"}],
        "version": CARD_VERSION,
        "provider": {"organization": "powerauto.ai / CGC",
                     "url": "https://powerauto.ai/"},
        "capabilities": {
            "streaming": True,
            # ★ pushNotifications 宣告為 true 就必須真的送得出去 —— 閘道的自測會驗這一格。
            #   宣告 true 而不實作，是這一類協議最常見的謊。
            "pushNotifications": True,
            "stateTransitionHistory": True,
        },
        "defaultInputModes": ["text/plain", "application/json"],
        "defaultOutputModes": ["text/plain", "application/json"],
        "skills": skills,

        # ── 擴展：類別與宏觀決策域 ──────────────────────────────────────
        "x-agent-class": agent.get("class"),
        "x-agent-class-label": c.get("label"),
        "x-agent-class-what": c.get("what"),
        "x-macro-scope": {
            "scope": mac.get("scope"),
            "label": mac.get("label"),
            "decisions": list(mac.get("decisions") or []),
            "note": "這一類 agent 的判斷在什麼範圍內算數。它**不是**能力清單 —— "
                    "能執行一個動作不等於有權決定要不要做它。",
        },
        "x-not-capable": list(agent.get("not_capable") or []),
        "x-runs-on": {"endpoint_id": agent.get("runs_on"), "reaches": agent.get("reaches")},
        # ★ 身分錨：這一類在**雲側**用哪個帳號。有它，對端（與 session 自己）才能回答
        #   「我現在屬於哪一類」——判準是**帳號**，不是自稱、也不是 session 標題。
        "x-identity": dict((c.get("identity") or {})),
        "x-brief": {"method": "agent/brief",
                    "note": "問這一類的資產／能力／進度／復盤。回來的四維是按類別整理的，"
                            "不是全部端點的四維（運營者看全機隊、開發者只看看移植目標、"
                            "探索者看決策與假設）。"},

        # ── 擴展：讓對端能自己核對分類 ────────────────────────────────
        "x-taxonomy": {
            "source": "agent_harness/a2a/taxonomy.py",
            "registry": "agent_harness/a2a/registry.json",
            "classes": [{"id": k, "label": TX.CLASSES[k]["label"],
                         "macro_scope": TX.CLASSES[k]["macro"]["scope"]}
                        for k in TX.classes()],
            "skill_owner": {sid: v["class"] for sid, v in TX.all_skills().items()},
            "how_to_check": "python3 agent_harness/a2a/agent_card.py --check",
            "note": "skill_owner 是「每個 skill 屬於哪一類」的正本。對端可以用它核對這張卡"
                    "的 skills 是否與它的類別一致 —— 不必相信我們。",
        },
    }
    return card, P


def index_card(gw: dict, agents: list[dict], host: str = "127.0.0.1") -> dict:
    """閘道自己的目錄卡（本閘道的擴展，不是 A2A 標準）。

    ★ 它**列的是一份名單**，所以它必須與 registry 雙向對齊：名單上有的要真的能開，
      能開的都要在名單上。少一邊都是靜默的漏洞（與 portal 的機隊表同一條規矩）。
    """
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "name": "CGC A2A 閘道",
        "description": "把 harness 與 pd 服務暴露成 A2A agent 的閘道（本卡是目錄，不是 agent）",
        "x-is-index": True,
        "x-note": "這是閘道的目錄卡，**不是** A2A agent card。標準客戶端請改用 "
                  "card_path 指到的個別卡片。",
        "agents": [
            {"id": a.get("id"), "class": a.get("class"), "class_label": TX.label(a.get("class")),
             "display_name": a.get("display_name"),
             "card_url": card_url_for(a.get("id"), gw, host),
             "rpc_url": url_for(a.get("id"), gw, host),
             "skills": [s["id"] for s in (a.get("_skills") or [])]}
            for a in agents],
        "classes": [{"id": k, "label": TX.CLASSES[k]["label"],
                     "macro_scope": TX.CLASSES[k]["macro"]["scope"],
                     "micro_count": len(TX.CLASSES[k]["micro"])} for k in TX.classes()],
        "taxonomy": "agent_harness/a2a/taxonomy.py",
    }


def build_all(host: str = "127.0.0.1") -> tuple[dict[str, dict], dict, list[str]]:
    reg = load_registry()
    gw = reg.get("gateway") or {}
    P: list[str] = list(TX.validate())

    ids = [a.get("id") for a in (reg.get("agents") or [])]
    dupes = sorted({i for i in ids if i and ids.count(i) > 1})
    if dupes:
        P.append("agent id 重複：%s" % dupes)
    if not ids:
        P.append("registry 的 agents 是空的 —— 閘道有了，但沒有人")

    cards: dict[str, dict] = {}
    enriched: list[dict] = []
    for a in (reg.get("agents") or []):
        aid = a.get("id") or "(無 id)"
        if aid in cards:
            continue
        card, probs = build_card(a, gw, host)
        P += probs
        cards[aid] = card
        enriched.append(dict(a, _skills=card["skills"]))

    # 閘道設定本身的檢查
    for k in ("default_port", "path_prefix", "card_path", "rpc_path"):
        if not gw.get(k):
            P.append("registry.gateway 缺 %s" % k)
    if gw.get("card_path") and gw.get("rpc_path"):
        cp, rp = str(gw["card_path"]), str(gw["rpc_path"])
        # ★ card_path 在 rpc_path **底下**是刻意的設計（`/a2a/<id>/.well-known/…` 之於
        #   `/a2a/<id>`）：閘道的路由先精確匹配 card 路徑，再匹配 rpc 路徑，所以沒有歧義。
        #   我第一版把這個形狀也報成錯 —— 那是一個**假陽性**，而假陽性的代價不是多看一眼，
        #   是真正的問題（兩條路徑相同）會被淹沒在噪音裡，於是檢查器開始被忽略。
        if cp == rp:
            P.append("gateway 的 card_path 與 rpc_path 相同（%s）⇒ 有一個一定會被吃掉" % cp)
        elif rp.startswith(cp + "/"):
            P.append("rpc_path（%s）在 card_path（%s）底下 ⇒ 每一個 card 請求都會被當成 "
                     "rpc 呼叫" % (rp, cp))

    return cards, index_card(gw, enriched, host), P


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="A2A Agent Card 產生器／檢查器")
    ap.add_argument("--check", action="store_true", help="只驗不印（有問題 ⇒ rc=1）")
    ap.add_argument("--agent", help="只印這一個 agent 的 card")
    ap.add_argument("--index", action="store_true", help="印閘道目錄卡")
    ap.add_argument("--host", default="127.0.0.1", help="card 裡 url 的 host（預設 127.0.0.1）")
    args = ap.parse_args(argv)

    cards, idx, P = build_all(args.host)
    if P:
        for p in P:
            print("[error] %s" % p)
        return 1
    if args.check:
        print("OK: %d 張 card，分類與能力歸屬一致" % len(cards))
        return 0
    if args.index:
        print(json.dumps(idx, ensure_ascii=False, indent=2))
        return 0
    if args.agent:
        if args.agent not in cards:
            print("[error] 沒有這個 agent：%s（有的：%s）" % (args.agent, ", ".join(cards)))
            return 1
        print(json.dumps(cards[args.agent], ensure_ascii=False, indent=2))
        return 0
    for aid, card in cards.items():
        print("%-18s %-10s skills=%d  path=%s"
              % (aid, card["x-agent-class"], len(card["skills"]), card["url"]))
    print("閘道目錄：%s" % idx["agents"][0]["card_url"].rsplit("/a2a/", 1)[0]
          + "/.well-known/agent-card.json" if idx["agents"] else "(無)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
