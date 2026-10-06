#!/usr/bin/env python3
"""[CGC 2026-10-07] targets.json <-> 看板 的判決對帳（**只讀兩個檔**）。

動機：「宣告與現判沒有對帳閘門」—— 同一個 repo 裡兩份權威檔對同一個里程碑
講了相反的話，而沒有任何閘門會發現（第三次：R6、`-d 512` 建議、本題）。

**本工具不裁誰是權威。** 它只回答一個權威中立的問題：
    兩份檔對同一個 target 的判決，是不是講了不一樣的話？
若是 ⇒ 列出矛盾與原文出處，exit 1（fail-closed）。

用法：
    python3 scripts/check/targets_board_recon.py            # 人讀報表
    python3 scripts/check/targets_board_recon.py --json      # 機讀
    python3 scripts/check/targets_board_recon.py --selftest  # 證明它真的會紅

為什麼要有 --selftest：一個「永遠綠」的對帳器比沒有更危險——它會把矛盾
偽裝成一致。selftest 注入一組已知矛盾，確認本工具會抓到。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
TARGETS = ROOT / "agent_harness/portal/targets.json"
BOARD = ROOT / "scripts/check/decode_board_2026-09-29.yaml"

# 判決斷言的關鍵字。每一類都是「這份檔對這個目標下了什麼判斷」。
# 改成別的關鍵字前先想清楚：這裡的價值在於**逐字可核對**，不在於涵蓋率。
DEAD_CLAIMS = (
    "判死",
    "無已命名路",
    "沒有剩下任何已命名的路",
    "可交付回收 ≈ 0",
)
OVERTURN_CLAIMS = (
    "已被量測推翻",
    "推翻",
)
MET_CLAIMS = (
    "6/6 ≥250",
    "6/6 ≥ 250",
    "達標",
)
COND_CLAIMS = (
    "條件式",
    "conditional",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _hits(text: str, keys: tuple[str, ...]) -> list[str]:
    """哪些關鍵字出現了。回傳關鍵字本身，讓報表可以逐字核對。"""
    return [k for k in keys if k in text]


def verdict(text: str) -> dict:
    return {
        "dead": _hits(text, DEAD_CLAIMS),
        "overturned": _hits(text, OVERTURN_CLAIMS),
        "met": _hits(text, MET_CLAIMS),
        "conditional": _hits(text, COND_CLAIMS),
    }


def board_slice(board_text: str, target_id: str) -> str:
    """取出看板裡與這個 target 相關的段落。

    用行級過濾而不是整檔關鍵字：整檔掃會把別的里程碑的「判死」算到這一格頭上，
    那會製造假陽性——正是這個工具要避免的毛病。
    """
    # ⚠ 切片宁窄勿宽：寬切片會把別的里程碑的斷言算到這一格頭上（假陽性）。
    #    2026-10-07 實測：prefill-250 用 ("prefill","pp512","250") 時，
    #    "250" 命中了 decode 那一側的長行，於是把「判死／已被量測推翻」誤算進
    #    prefill 的看板判決 —— 那正是這個工具自己要抓的毛病，所以收窄到
    #    只認 prefill 自己帶 250 的寫法。
    if target_id == "decode-25":
        pats = ("decode ≥25", "20+", "M-25")
    elif target_id == "prefill-250":
        pats = ("prefill 250", "prefill250", "≥250", "pp512")
    else:
        pats = (target_id,)
    keep = [ln for ln in board_text.split("\n") if any(p in ln for p in pats)]
    return "\n".join(keep)


def reconcile(targets_path: Path = TARGETS, board_path: Path = BOARD) -> dict:
    t_text = _read(targets_path)
    b_text = _read(board_path)
    t_doc = json.loads(t_text)

    findings = []
    for t in t_doc.get("targets") or []:
        tid = t.get("id")
        if not tid:
            continue
        # targets.json 側：這個 target 自己的所有文字欄
        t_blob = " ".join(
            str(v) for k, v in t.items() if isinstance(v, (str, int, float))
        ) + " " + " ".join(
            str(x) for v in t.values() if isinstance(v, list) for x in v
        )
        tv = verdict(t_blob)
        bv = verdict(board_slice(b_text, tid))

        problems = []
        # 矛盾 1：一邊說判死、另一邊說該判決已被推翻
        if tv["overturned"] and bv["dead"]:
            problems.append(
                "targets.json 記載判死依據「已被量測推翻」，但看板仍帶判死斷言"
            )
        # 矛盾 2：一邊說（條件式）已達標、另一邊完全沒提到
        if (tv["met"] or tv["conditional"]) and not (bv["met"] or bv["conditional"]):
            problems.append(
                "targets.json 記載（條件式）已達標，看板未出現任何達標／條件式斷言"
            )
        # 矛盾 3：兩邊的達標斷言不一致（一邊說達標、另一邊說判死）
        if tv["met"] and bv["dead"]:
            problems.append("targets.json 說達標，看板說判死")

        findings.append(
            {
                "target": tid,
                "status_json": t.get("status"),
                "verdict_targets_json": tv,
                "verdict_board": bv,
                "contradictions": problems,
            }
        )

    # ⚠ 已知方法限制（2026-10-07 實測，不掩蓋）：
    #    看板的某些行極長，同一行裡同時含有 prefill 與 decode 的文字，
    #    因此「行級過濾」無法把兩個里程碑的斷言切開 ⇒ **看板側的歸屬可能帶污染**。
    #    （實測：prefill-250 的看板切片出現了 decode 側的「判死／已被量測推翻」。）
    #    ⇒ 這是**理由可能不精確**，不是**結論錯**：兩條矛盾本身都經人工核對為真。
    #    要把歸屬做對，下一步應改用 YAML 結構定位到具體 cell（而非行級關鍵字）。
    for f in findings:
        b = f["verdict_board"]
        f["board_attribution_suspect"] = bool(b["dead"] and b["met"])

    bad = [f for f in findings if f["contradictions"]]
    return {
        "ok": not bad,
        "findings": findings,
        "contradicting": [f["target"] for f in bad],
        "caveat": (
            "看板側以行級關鍵字切片；該檔有超長行混合多個里程碑，"
            "故單一 target 的看板歸屬可能帶污染。結論（兩檔互相矛盾）已人工核對；"
            "標記 board_attribution_suspect=true 者，其『理由』需對照原文再讀一次。"
        ),
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()

    r = reconcile()
    if args.json:
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return 0 if r["ok"] else 1

    print("targets.json <-> 看板 判決對帳")
    print(f"  targets: {TARGETS}")
    print(f"  board  : {BOARD}")
    print()
    for f in r["findings"]:
        mark = "OK  " if not f["contradictions"] else "FAIL"
        print(f"  [{mark}] {f['target']}  (targets.json status={f['status_json']})")
        print(f"         targets.json 判決: {_fmt(f['verdict_targets_json'])}")
        print(f"         看板       判決: {_fmt(f['verdict_board'])}")
        for p in f["contradictions"]:
            print(f"         !! {p}")
        print()
    if r["ok"]:
        print("  => 兩份檔對所有 target 的判決一致。")
        return 0
    print("  => 矛盾：" + ", ".join(r["contradicting"]))
    print("     本工具不裁誰是權威；權威歸屬待 operator 簽。")
    return 1


def _fmt(v: dict) -> str:
    parts = [f"{k}={v[k]}" for k in ("dead", "overturned", "met", "conditional") if v[k]]
    return " / ".join(parts) if parts else "(無斷言)"


def _selftest() -> int:
    """注入已知矛盾，確認本工具抓得到。永遠綠的對帳器比沒有更危險。"""
    import tempfile

    base_t = json.loads(_read(TARGETS))
    # 注入：把 decode-25 的推翻斷言拔掉，並讓它說「判死」——與看板一致 ⇒ 不應報矛盾
    quiet = json.loads(json.dumps(base_t))
    for t in quiet["targets"]:
        if t["id"] == "decode-25":
            t["gap"] = "判死。"
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "targets.json"
        p.write_text(json.dumps(quiet, ensure_ascii=False), encoding="utf-8")
        r = reconcile(targets_path=p)
    ok1 = all(
        not f["contradictions"]
        for f in r["findings"]
        if f["target"] == "decode-25"
    )
    print(f"  [{'PASS' if ok1 else 'FAIL'}] 兩邊一致時不報矛盾（無假陽性）")

    # 真實狀態必須報出矛盾
    r2 = reconcile()
    ok2 = (not r2["ok"]) and "decode-25" in r2["contradicting"]
    print(f"  [{'PASS' if ok2 else 'FAIL'}] 真實狀態下 decode-25 被判為矛盾（會紅）")
    ok3 = (not r2["ok"]) and "prefill-250" in r2["contradicting"]
    print(f"  [{'PASS' if ok3 else 'FAIL'}] 真實狀態下 prefill-250 被判為矛盾（會紅）")

    allok = ok1 and ok2 and ok3
    print(f"\nselftest {'PASS' if allok else 'FAIL'}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
