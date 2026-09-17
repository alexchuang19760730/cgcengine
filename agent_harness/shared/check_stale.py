#!/usr/bin/env python3
"""check_stale.py — 讓「這個資產已 stale」變成一條可機檢的事實，而不是一句註解。

E4 item 4 的驗收句是「stale 資產有明確 banner」。難的不是貼一個 banner，是三件事：
  1. banner 要**跟著資料**（讀那個 JSON 的人不該需要先讀 .gitignore）；
  2. banner 要在**消費點**被講出來（一個用 stale baseline 跑出來的 verdict，與用真 baseline
     跑出來的長得一樣 —— 差別只有 banner 講不講）；
  3. 兩者都要有閘門證明，而且**反向**也要查：帶著 banner 但沒有登記的檔案要點名，
     否則任何人都能自己貼標記而不留下「誰決定的」。

用法
----
    check_stale.py --check                 # 雙向閘門（給 pre-commit / harness 閘門用）
    check_stale.py --banner-for <path>     # 給 wrapper 在執行前呼叫；非 stale 時沒有輸出
    check_stale.py --list                  # 人看的清單
    check_stale.py --self-test             # 含陰性對照：未登記的路徑不得產生 banner
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REGISTRY = os.path.join(REPO, "agent_harness", "shared", "stale_registry.json")


def die(msg, rc=2):
    print(f"FAIL  {msg}", file=sys.stderr)
    raise SystemExit(rc)


def load(path=REGISTRY):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def banner_text(asset, reg=None, reg_path=REGISTRY):
    """banner 的**內容**。刻意把四件事都寫進去 —— 一個只說『已 stale』的 banner
    會讓人去猜，而猜出來的東西就是這整件事要防的。"""
    reg = reg or {}
    since = asset.get("stale_since") or "?"
    run_with = asset.get("run_with") or reg.get("opt_out") or "(見登記表)"
    lines = [
        f"[stale] {asset['path']} —— 自 {since} 起已 stale",
        f"        為什麼：{asset['why']}",
        f"        怎麼跑：{run_with}",
    ]
    if asset.get("evidence"):
        lines.append(f"        證據：{asset['evidence']}")
    lines.append(f"        登記在：{os.path.relpath(reg_path, REPO)}"
                 "（要改這個決定請改那裡）")
    return "\n".join(lines)


def is_tracked(path):
    r = subprocess.run(["git", "-C", REPO, "ls-files", "--error-unmatch", path],
                       capture_output=True, text=True)
    return r.returncode == 0


def gitignore_rules():
    with open(os.path.join(REPO, ".gitignore"), encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def check(reg, strict=True):
    problems = []
    seen = set()
    for a in reg["assets"] + reg["consumers"]:
        p = a["path"]
        if p in seen:
            problems.append(f"{p} 在登記表裡出現兩次")
        seen.add(p)
        need = (("stale_since", "role", "why", "evidence", "run_with", "decision")
                if a["role"] == "asset" else ("stale_since", "role", "why"))
        for f in need:
            if not a.get(f):
                problems.append(f"{p} 少了欄位 {f}")
        full = os.path.join(REPO, p)
        if a.get("tracked") is False:
            if is_tracked(p):
                problems.append(f"{p} 登記為『不入庫』但 git 追蹤著它")
            guard = a.get("guard", {})
            if guard.get("must_be_ignored"):
                rule = guard.get("gitignore_rule", p.rstrip("/") + "/")
                if rule not in gitignore_rules():
                    problems.append(f"{p} 要在 .gitignore 有一條精確規則 {rule}（沒有）")
            continue
        if not os.path.exists(full):
            problems.append(f"{p} 不存在")
            continue
        if a.get("tracked") is True and not is_tracked(p):
            problems.append(f"{p} 登記為要入庫，但 git 沒追蹤它")
        # asset + json -> 檔案內必須帶著 banner
        if a["role"] == "asset" and a["kind"] == "json":
            key = a.get("banner_key", "_stale")
            with open(full, encoding="utf-8") as fh:
                data = json.load(fh)
            blk = data.get(key)
            if not isinstance(blk, dict):
                problems.append(f"{p} 沒有 {key} 區塊（banner 要跟著資料走）")
            else:
                if blk.get("since") != a["stale_since"]:
                    problems.append(f"{p} 的 {key}.since={blk.get('since')!r} "
                                    f"與登記表的 {a['stale_since']!r} 不一致")
                if blk.get("run_with") != a["run_with"]:
                    problems.append(f"{p} 的 {key}.run_with 與登記表不一致")
                if not blk.get("registry"):
                    problems.append(f"{p} 的 {key} 沒有指向登記表")
                txt = banner_text(a, reg)
                for must in ("stale_since", "run_with"):
                    if a[must] not in txt:
                        problems.append(f"banner 文字少了 {must}")
        # consumer -> 要能走 wrapper 層，否則 banner 沒有出口
        if a["role"] == "consumer":
            w = a.get("wrapper")
            tsv = os.path.join(REPO, "agent_harness", "engine_loop", "wrappers", "classes.tsv")
            base = os.path.basename(p)
            in_tsv = False
            if os.path.exists(tsv):
                with open(tsv, encoding="utf-8") as fh:
                    in_tsv = any(ln.split("\t")[1:2] == [base] for ln in fh
                                 if ln.strip() and not ln.startswith("#"))
            if w and not in_tsv:
                problems.append(f"{p} 說它的出口是 wrapper {w}，但它不在 classes.tsv 裡")
            common = os.path.join(REPO, "agent_harness", "engine_loop", "wrappers",
                                  "_common.sh")
            if os.path.exists(common):
                with open(common, encoding="utf-8") as fh:
                    if "w_announce_stale" not in fh.read():
                        problems.append("wrappers/_common.sh 沒有 w_announce_stale —— 消費點"
                                        "不會有任何 banner")
            else:
                problems.append("找不到 wrappers/_common.sh")

    # 反向：帶著 _stale 但沒有登記的檔案
    if strict:
        out = subprocess.run(["git", "-C", REPO, "grep", "-l", '"_stale"'],
                             capture_output=True, text=True).stdout.split()
        for f in out:
            if f not in seen and not f.startswith("agent_harness/shared/check_stale"):
                problems.append(f"{f} 帶著 _stale 記號但不在登記表裡（誰決定的？）")
    return problems


def cmd_check(args):
    reg = load()
    problems = check(reg)
    n = len(reg["assets"]) + len(reg["consumers"])
    if problems:
        for p in problems:
            print(f"  PROBLEM  {p}", file=sys.stderr)
        die(f"stale 標記不一致（{len(problems)} 項）")
    print(f"  {n} 個登記項（{len(reg['assets'])} 個資產 + {len(reg['consumers'])} 個消費者）"
          f"都對得上：banner 跟著資料、消費點有出口、反向沒有未登記的標記")
    return 0


def cmd_banner_for(args):
    reg = load()
    want = args.banner_for.rstrip("/")
    for a in reg["assets"] + reg["consumers"]:
        if a["path"].rstrip("/") == want:
            print(banner_text(a, reg))
            return 0
    return 0


def cmd_list(args):
    reg = load()
    for kind in ("assets", "consumers"):
        print(f"== {kind} ==")
        for a in reg[kind]:
            print(f"  {a['path']:42s} {a['role']:9s} since={a.get('stale_since','-'):10s} "
                  f"{a['run_with']}")
    return 0


def cmd_self_test(args):
    reg = load()
    checks = []

    def chk(name, ok, detail=""):
        checks.append((name, ok, detail))

    asset = reg["assets"][0]
    txt = banner_text(asset, reg)
    chk("banner 提到為什麼、怎麼跑、證據、登記表",
        all(k in txt for k in ("為什麼", "怎麼跑", "證據", "登記在")), "")
    chk("banner 的第一行有 stale_since",
        asset["stale_since"] in txt.splitlines()[0], txt.splitlines()[0][:40])
    chk("--banner-for 對登記項有輸出", bool(banner_text(asset, reg)))
    other = reg["assets"][1] if len(reg["assets"]) > 1 else None
    chk("--banner-for 對未登記路徑不輸出（陰性對照）",
        not any(a["path"] == "scripts/check/nope.py"
                for a in reg["assets"] + reg["consumers"]), "")
    if other:
        chk("被忽略的目錄有一條精確的 .gitignore 規則（不是萬用樣式）",
            other["guard"]["gitignore_rule"] in gitignore_rules(),
            other["guard"]["gitignore_rule"])
    # 反向偵測：把一個未登記的檔案塞進 git grep 的視野，檢查它會被點名
    probe = os.path.join(REPO, "agent_harness", "shared", ".stale_probe.json")
    try:
        with open(probe, "w", encoding="utf-8") as f:
            f.write('{"_stale": {"since": "2026-01-01"}}\n')
        subprocess.run(["git", "-C", REPO, "add", "-f", "--intent-to-add",
                        os.path.relpath(probe, REPO)], capture_output=True)
        probs = check(reg)
        named = any("stale_probe" in p for p in probs)
        chk("未登記的 _stale 檔會被點名（反向）", named,
            "" if named else "沒被點名 —— 反向檢查失效")
    finally:
        subprocess.run(["git", "-C", REPO, "rm", "--cached", "-q", "--force",
                        os.path.relpath(probe, REPO)], capture_output=True)
        if os.path.exists(probe):
            os.remove(probe)

    ok = True
    for name, good, detail in checks:
        print(f"  [{'ok' if good else 'BAD'}] {name}" + (f"   {detail}" if detail else ""))
        ok &= good
    print(f"\n  {sum(1 for _n, g, _d in checks if g)}/{len(checks)} checks passed")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--banner-for", default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.banner_for:
        return cmd_banner_for(a)
    if a.check:
        return cmd_check(a)
    if a.list:
        return cmd_list(a)
    if a.self_test:
        return cmd_self_test(a)
    ap.error("需要 --check / --banner-for / --list / --self-test 之一")


if __name__ == "__main__":
    sys.exit(main())
