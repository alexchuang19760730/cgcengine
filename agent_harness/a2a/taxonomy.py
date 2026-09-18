#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""三類 agent 的分類體系 —— 宏觀決策域（macro）與微觀能力（micro）的單一真相。

為什麼要有這一支
----------------
「這個 agent 是哪一類」如果只寫在文件或 prompt 裡，它就是一個**不可機檢的宣稱**：
別人無法在呼叫前知道對面能做什麼（只能試錯），也沒人能回答「這個能力屬於哪一類」
——於是分類會在第一次有人順手加功能時漂移。

所以分類在**這裡**，而且只有這裡。它同時餵三個消費者：
  1. A2A 的 Agent Card：類別進擴展欄位 `x-agent-class`，宏觀域進 `x-macro-scope`，
     微觀能力進**標準** `skills[]`（這樣標準 A2A 客戶端也看得懂一部分）
  2. 閘道的**路由**：任務依 skill 的類別決定誰執行；**無法歸類 ⇒ 拒收**（不靜默選一個）
  3. 自測：每一項微觀能力都要能指到一個真的 executor，不留空

兩個維度刻意分開
----------------
- **macro（宏觀決策域）**：這一類 agent 有權**決定什麼**。不是「能做什麼動作」，
  而是「它的判斷在什麼範圍內算數」。
- **micro（微觀能力）**：這一類 agent **能執行什麼動作**。每一項必須是一個可呼叫的
  skill id，而且那個 id 要能在 `a2a_server.py` 的 EXECUTORS 裡找到 —— 否則它是
  「一個寫在文件上的能力」，不是能力。

★ 兩者**不可互相推導**：一個能跑 `cmake` 的 agent（micro）不代表它有權決定
  「要不要移植到鴻蒙」（macro）。把兩者併成一張清單，下一個讀者就會以為
  「能執行」等於「有權決定」。
"""
from __future__ import annotations

CLASS_ORDER = ("developer", "operator", "explorer")


def _m(sid, name, desc, tags, examples):
    """一項微觀能力。形狀刻意就是 A2A 的 skill（id/name/description/tags/examples），
    因為它**就是**那個 skill —— 中間不再轉一手。"""
    return {"id": sid, "name": name, "description": desc, "tags": list(tags),
            "examples": list(examples)}


CLASSES: dict[str, dict] = {
    "developer": {
        "id": "developer",
        "label": "開發 agent",
        "what": "把引擎移植到新平台，並證明它在上面真的跑起來（windows／鴻蒙是目前兩個目標）",
        "macro": {
            "scope": "porting",
            "label": "移植目標與構建策略",
            "decisions": [
                "下一個移植目標是哪個平台，以及為什麼是它",
                "構建配置（後端、量化、記憶體預算）與回退順序",
                "「在這個平台上算成功」的判準（哪個數字、什麼條件下量）",
            ],
        },
        "micro": [
            _m("build-target", "構建目標平台",
               "在一個具名平台上建置引擎，回報產物路徑與建置設定",
               ["build", "cross-platform"], ["在 windows 上建 CUDA 版"]),
            _m("port-and-verify", "移植並驗證",
               "把某個平台缺的東西補上，並跑一次可證否的驗證",
               ["port", "verify"], ["讓鴻蒙端能跑 aarch64 建置並回報結果"]),
            _m("cross-platform-package", "跨平台打包",
               "產生可在目標平台上直接啟動的整包產物",
               ["package", "release"], ["打包 macOS bundle 並驗 @rpath"]),
            _m("report-capability", "上報平台能力",
               "回報這個端點**真的驗證過**的能力與未量測項，供機隊入口消費",
               ["report", "endpoint"], ["上報 RTX4090 的 decode t/s"]),
        ],
    },
    "operator": {
        "id": "operator",
        "label": "運營 agent",
        "what": "讓端／雲／端的服務持續可觀測、可部署（pd 服務與 harness 入口都在它的範圍內）",
        "macro": {
            "scope": "fleet-ops",
            "label": "機隊部署與可用性",
            "decisions": [
                "服務部署到哪個端點，以及為什麼不是別的",
                "什麼時候擴容、什麼時候降級、什麼時候回滾",
                "哪些「拿不到」是可以接受的（缺席不是失敗）",
            ],
        },
        "micro": [
            _m("deploy-endpoint", "部署到端點",
               "把一個服務推上具名端點並確認它起來了",
               ["deploy", "endpoint"], ["在 windows-rtx4090 上起 edge_server"]),
            _m("collect-status", "採集端點狀態",
               "從活著的 url 或已被通道帶回的 dump 取得端點狀態並映射成契約",
               ["collect", "contract"], ["跑 fleet_auto 採集一輪"]),
            _m("watchdog-check", "看門狗",
               "回答「這條鏈上次跑是什麼時候、還活著嗎」；停擺要出聲",
               ["watchdog", "health"], ["檢查排程裝了卻沒跑"]),
            _m("export-fleet", "匯出機隊資料",
               "把機隊資產／能力／進度／復盤匯出成給別的網站吃的可攜資料",
               ["export", "portal"], ["重生 fleet_export.json"]),
        ],
    },
    "explorer": {
        "id": "explorer",
        "label": "探索 agent",
        "what": "解決宏觀決策：目標該不該改、哪條路該放棄、哪些假設已經被推翻",
        "macro": {
            "scope": "decision",
            "label": "目標、路線與假設的取捨",
            "decisions": [
                "當前目標還成不成立，要不要改（以及改成什麼）",
                "哪一條路已經封閉、不要再重試",
                "哪些結論被後續證據推翻，以及它取代了什麼",
            ],
        },
        "micro": [
            _m("register-decision", "登錄決策",
               "把一個決策連同證據、推理、結論與信心寫成可復盤的記錄",
               ["decision", "trace"], ["把「iq4_xs 進 small-batch 家族」判為封閉"]),
            _m("test-hypothesis", "檢驗假設",
               "設計並跑一次可證否的實驗，回報支持或推翻",
               ["hypothesis", "experiment"], ["驗 MoE 是否佔 decode wait 多數"]),
            _m("measure-metric", "量測指標",
               "在一個具名條件下量一個數字，並附上「什麼條件下才算數」",
               ["measure", "metric"], ["冷機條件下量 prefill t/s"]),
            _m("triage-blocker", "拆解阻塞",
               "把一個卡住的問題拆到可裁決的最小動作",
               ["triage", "diagnosis"], ["找出 D5 失敗的第一嫌疑"]),
        ],
    },
}


# ── 反查與校驗（路由與自測都靠這幾支，不要在別處再寫一份）────────────────

def classes() -> list[str]:
    return list(CLASS_ORDER)


def get(cid: str) -> dict:
    """取一類。**不存在的類別回 None 而不是拋錯** —— 呼叫端要能區分
    「這類不存在」與「這類存在但沒有能力」，那是兩個不同的錯誤。"""
    return CLASSES.get(cid)


def label(cid: str) -> str:
    c = get(cid)
    return c["label"] if c else "(未知類別)"


def skill_ids(cid: str) -> list[str]:
    c = get(cid)
    return [s["id"] for s in c["micro"]] if c else []


def all_skills() -> dict[str, dict]:
    """skill id -> {class, skill}。id 跨類別重複是**硬錯誤**（見 validate）。"""
    out: dict[str, dict] = {}
    for cid in CLASS_ORDER:
        for s in CLASSES[cid]["micro"]:
            out[s["id"]] = {"class": cid, "skill": s}
    return out


def class_of_skill(sid: str) -> str | None:
    """一個 skill 屬於哪一類。★ 路由用這支：認不出來就回 None，
    呼叫端要拒收，不要猜一個預設類別。"""
    hit = all_skills().get(sid)
    return hit["class"] if hit else None


def macro(cid: str) -> dict | None:
    c = get(cid)
    return c["macro"] if c else None


def validate() -> list[str]:
    """分類體系自己的檢查。回問題清單（空 = 合法）。

    ★ 這一支存在的理由與 portal 的 `--check` 相同：**分類漂移是靜默的**。
      多一個 skill 而漏了 executor，症狀不是報錯，是「那個能力永遠不會被執行」。
    """
    P: list[str] = []
    if list(CLASSES) != list(CLASS_ORDER):
        P.append("CLASSES 的鍵順序與 CLASS_ORDER 不一致：%s vs %s"
                 % (list(CLASSES), list(CLASS_ORDER)))
    seen: dict[str, str] = {}
    for cid in CLASS_ORDER:
        c = CLASSES.get(cid)
        if not c:
            P.append("CLASS_ORDER 指名了 %r，但 CLASSES 裡沒有它" % cid)
            continue
        if c.get("id") != cid:
            P.append("[%s] id 欄位是 %r，與鍵不符" % (cid, c.get("id")))
        for k in ("label", "what", "macro", "micro"):
            if not c.get(k):
                P.append("[%s] 缺欄位 %s（空是一個主張，不能留白）" % (cid, k))
        mac = c.get("macro") or {}
        if not mac.get("scope"):
            P.append("[%s] macro 缺 scope —— 沒有範圍的決策權等於沒有限制" % cid)
        if not mac.get("decisions"):
            P.append("[%s] macro.decisions 是空的：這一類 agent 到底有權決定什麼？" % cid)
        micro = c.get("micro") or []
        if not micro:
            P.append("[%s] micro 是空的 —— 一個沒有可呼叫能力的類別不該存在" % cid)
        for s in micro:
            sid = s.get("id") or "(無 id)"
            if sid in seen:
                P.append("skill id 重複：%r 同時屬於 %s 與 %s（路由無法決定誰執行）"
                         % (sid, seen[sid], cid))
            seen[sid] = cid
            for k in ("id", "name", "description", "tags", "examples"):
                if not s.get(k):
                    P.append("[%s/%s] 缺欄位 %s" % (cid, sid, k))
            if not str(sid).replace("-", "").replace("_", "").isalnum():
                P.append("[%s/%s] skill id 只能用小寫英數與 -_（它會進 URL 與 JSON-RPC）"
                         % (cid, sid))
    return P


if __name__ == "__main__":
    import json
    import sys
    probs = validate()
    if probs:
        for p in probs:
            print("[error] %s" % p)
        raise SystemExit(1)
    print("分類體系：%d 類、%d 項微觀能力" % (len(CLASS_ORDER), len(all_skills())))
    for cid in CLASS_ORDER:
        c = CLASSES[cid]
        print("  %-11s %-10s macro=%-11s micro=%d 項  %s"
              % (cid, c["label"], c["macro"]["scope"], len(c["micro"]),
                 "、".join(skill_ids(cid))))
    print(json.dumps({cid: macro(cid) for cid in CLASS_ORDER}, ensure_ascii=False, indent=1)[:400])
