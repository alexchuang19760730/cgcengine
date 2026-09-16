#!/usr/bin/env python3
"""Prove `closed_loop.py`'s mechanics work -- without a model, and without touching real state.

WHY THIS EXISTS
---------------
The failure this guards against is not "it crashes". It is: **the comparison runs, exits 0, and
reports a difference that no one can attribute to anything.** Three ways that happens, all silent:

  1. `--memories-scope` selects ids the projection cannot supply. The arm silently loses lessons
     and still looks like a deliberate arm, because "D minus five" and "D" print the same sizes.
  2. An arm disagrees with ITSELF across reps. A cross-arm difference then exists, is reported, and
     means nothing -- it is sampling noise wearing the label of an effect.
  3. The instrument cannot say "no difference". If every pair of arms always reports a difference,
     the comparison carries no information at all, and nothing about it looks wrong.

So most of what follows is NEGATIVE: it establishes that the machinery REJECTS bad input, and that
it is ABLE to return "identical". A comparison that can only ever answer "different" is not a
comparison -- it is a device that agrees with you.

THE CENTRAL PAIR OF CHECKS (C and D below)
------------------------------------------
The same stable fake model is run twice, changing only the scope:

    scope=none    -> arm D's prompt becomes BYTE-IDENTICAL to arm C's  =>  C vs D MUST be identical
    scope=first:8 -> arm D carries the lessons                          =>  C vs D MUST differ

Same code, same model, same questions: one run must produce "no difference" and the other must
produce "a difference". An instrument that passes only one of those is broken in a way that no
amount of running it on real data would reveal.

WHY THE FAKE MODEL ANSWERS WITH A HASH
--------------------------------------
The fake model's answer is `sha256(prompt)[:16]`. That makes it deterministic, sensitive to every
byte of the prompt, and -- the point -- INDEPENDENT of what the prompt says. A fake model that
pattern-matched on the charter's wording would silently start lying the day someone edits that
wording, and it would keep passing. The hash cannot care what the text means; it can only report
whether the bytes changed, which is exactly the question these checks ask.

Nothing here has ever seen a real model. This file says nothing about whether the lessons help.

Usage: python3 agent_harness/engine_loop/distill/closed_loop_selftest.py
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parent
CLOSED = HERE / "closed_loop.py"
LESSONS = ENGINE / "traces" / "lessons.jsonl"
MEM = ENGINE / "harness_engine" / "memories" / "engine"

CHECKS = 0
FAILS: list[str] = []

# Answers with sha256(prompt)[:16] -- see the docstring. `sig` is the answer; the human-readable
# prefix is only so a person reading the output can tell which arm it came from.
STABLE_FAKE = '''import hashlib, sys
p = sys.stdin.read()
print("\u52d5\u4f5c: \u8dd1 m123_oracle_gate.py \u4e26\u78ba\u8a8d comparable\u3002sig="
      + hashlib.sha256(p.encode()).hexdigest()[:16])
'''

# Same, but every call answers differently -- this is what "the arm disagrees with itself" means.
FLAKY_FAKE = '''import random, sys
sys.stdin.read()
print(f"\u52d5\u4f5c: \u4e0d\u56fa\u5b9a\u7684\u56de\u7b54 {random.random():.6f}")
'''


def check(cond: bool, label: str, detail: str = "") -> bool:
    global CHECKS
    CHECKS += 1
    print(f"  [{'ok' if cond else 'FAIL'}] {label}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(label)
    return bool(cond)


def run_loop(args: list[str], *, out: Path, mem: Path | None = None,
             model: Path | None = None) -> tuple[int, str]:
    env = dict(os.environ)
    env["CLOSED_LOOP_OUT"] = str(out)
    if mem is not None:
        env["CLOSED_LOOP_MEM_DIR"] = str(mem)
    if model is not None:
        env["CLOSED_LOOP_MODEL_CMD"] = f"{sys.executable} {model}"
    else:
        env.pop("CLOSED_LOOP_MODEL_CMD", None)
    r = subprocess.run([sys.executable, str(CLOSED), *args], env=env,
                       capture_output=True, text=True, timeout=900)
    return r.returncode, r.stdout + r.stderr


def ids_from(out: str) -> list[str]:
    m = re.search(r"^\s+ids: (.+)$", out, re.M)
    if not m:
        return []
    return [x.strip() for x in m.group(1).replace("...", "").split(",") if x.strip()]


def load_json(out: Path, name: str) -> dict | list:
    runs = sorted(out.glob(f"*/{name}"))
    if not runs:
        raise AssertionError(f"no {name} under {out}")
    return json.loads(runs[-1].read_text(encoding="utf-8"))


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="closedloop_selftest_"))
    try:
        stable = tmp / "stable.py"
        flaky = tmp / "flaky.py"
        stable.write_text(STABLE_FAKE, encoding="utf-8")
        flaky.write_text(FLAKY_FAKE, encoding="utf-8")

        rows = [json.loads(l) for l in LESSONS.read_text(encoding="utf-8").splitlines() if l.strip()]
        live = [r for r in rows if not r.get("superseded_by")]
        print(f"  fixtures: {len(rows)} lesson records ({len(live)} live), "
              f"{len(list(MEM.glob('*.md')))} memory files")

        # ---- A. the selector must agree with the record it selects from ----
        print()
        print("A. --memories-scope 的解析（無模型）")
        o = tmp / "a"
        rc, out = run_loop(["--dry-run", "--memories-scope", "first:8"], out=o)
        check(rc == 0 and ids_from(out) == [r["lesson_id"] for r in rows[:8]],
              "first:8 == lessons.jsonl 的前 8 條 id（檔案順序，不是 id 排序）")

        rc, out = run_loop(["--dry-run", "--memories-scope", "all"], out=o)
        m = re.search(r"all -- (\d+) live lessons", out)
        check(m is not None and int(m.group(1)) == len(live),
              "all 的總數（取自描述行；ids 行只印前 10 個）等於 live 紀錄數",
              m.group(1) if m else "?")

        rc, out = run_loop(["--dry-run", "--memories-scope", "none"], out=o)
        check(rc == 0 and not ids_from(out), "none 不注入任何 lesson")

        rc, out = run_loop(["--dry-run", "--memories-scope", "class:mh"], out=o)
        got = ids_from(out)
        check(rc == 0 and len(got) >= 2 and all(i.startswith("eng-mh-") for i in got),
              "class:mh 只選 mh 類", f"印出前 {len(got)} 個")

        idfile = tmp / "ids.txt"
        idfile.write_text("# a comment\n\neng-mh-0003\neng-gate-0002\n", encoding="utf-8")
        rc, out = run_loop(["--dry-run", "--memories-scope", f"ids:{idfile}"], out=o)
        check(rc == 0 and ids_from(out) == ["eng-mh-0003", "eng-gate-0002"],
              "ids:<path> 精確等於檔案內容（註解與空行被忽略）")

        print()
        print("A'. 壞掉的 scope 必須拒跑（每一種都是『安靜地少注入幾條』的入口）")
        for spec, label in [("class:nope", "未知的 class"),
                            ("first:0", "first:0"),
                            (f"first:{len(rows) + 50}", "first 超出總數"),
                            ("ids:/nonexistent/nope.txt", "ids 檔不存在"),
                            ("bogus", "未識別的形式")]:
            rc, _ = run_loop(["--dry-run", "--memories-scope", spec], out=o)
            check(rc != 0, f"拒跑：{label}")

        badid = tmp / "badids.txt"
        badid.write_text("eng-mh-0003\neng-typo-9999\n", encoding="utf-8")
        rc, out = run_loop(["--dry-run", "--memories-scope", f"ids:{badid}"], out=o)
        check(rc != 0 and "eng-typo-9999" in out, "拒跑：ids 裡有錯字，且點名是哪一個")

        empty = tmp / "empty.txt"
        empty.write_text("", encoding="utf-8")
        rc, _ = run_loop(["--dry-run", "--memories-scope", f"ids:{empty}"], out=o)
        check(rc != 0, "拒跑：ids 檔是空的（空手臂不可與『選取壞了』同形）")

        rc, _ = run_loop(["--reps", "0"], out=o)
        check(rc != 0, "拒跑：--reps 0")

        # ---- B. the missing-projection guard, tested without damaging the real projection ----
        print()
        print("B. 投影缺檔的守衛（把投影目錄重導到一份不完整的副本）")
        part = tmp / "partmem"
        part.mkdir()
        src = MEM / "eng-mh-0003.md"
        (part / src.name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        two = tmp / "two.txt"
        two.write_text("eng-mh-0003\neng-lf-0007\n", encoding="utf-8")
        rc, out = run_loop(["--dry-run", "--memories-scope", f"ids:{two}"], out=o, mem=part)
        check(rc != 0 and "eng-lf-0007" in out,
              "選到的 id 在投影裡沒有檔 ⇒ 硬錯誤（不是靜默跳過）")
        check("build_memories.py" in out, "錯誤訊息裡有修復指令")
        check((MEM / "eng-lf-0007.md").is_file(),
              "真實投影沒被這個測試動到（守衛不必靠破壞來證明）")

        # ---- C. the instrument must be ABLE to say "no difference" ----
        print()
        print("C. 陰性對照：scope=none ⇒ C 與 D 的 prompt 位元組相同 ⇒ 必須回報「相同」")
        rc, out = run_loop(["--only", "Q1", "--reps", "2", "--memories-scope", "none"],
                           out=tmp / "c", model=stable)
        c0 = load_json(tmp / "c", "compare.json")[0]
        check(rc == 0 and c0["C_vs_D_identical"] is True,
              "C vs D 相同（儀器說得出『沒有差異』）", f"verdict={c0['C_vs_D_verdict']}")
        check(c0["A_vs_B_identical"] is False,
              "同一輪裡 A vs B 仍不同 ⇒ 不是全盤說『相同』的假順從")

        # ---- D. and it must still report a difference when there is one ----
        print()
        print("D. 陽性：scope=first:8 ⇒ D 帶著 lesson ⇒ 必須回報「不同」")
        rc, _ = run_loop(["--only", "Q1", "--reps", "2", "--memories-scope", "first:8"],
                         out=tmp / "d", model=stable)
        c0 = load_json(tmp / "d", "compare.json")[0]
        check(rc == 0 and c0["C_vs_D_identical"] is False,
              "C vs D 不同（lesson 真的進了 prompt）", f"verdict={c0['C_vs_D_verdict']}")
        check(c0["comparable"] is True, "四臂各自穩定 ⇒ comparable=true")

        # ---- E. an arm that disagrees with itself voids the question ----
        print()
        print("E. 臂內不穩定 ⇒ 該題的跨臂判斷必須作廢")
        rc, out = run_loop(["--only", "Q2", "--reps", "3", "--memories-scope", "first:8"],
                           out=tmp / "e", model=flaky)
        c0 = load_json(tmp / "e", "compare.json")[0]
        check(c0["comparable"] is False, "comparable=false")
        check(c0["A_vs_B_identical"] is None and c0["C_vs_D_identical"] is None,
              "跨臂欄位是 null，不是 False（『不能比』與『比過了，相同』不可同形）")
        check("無法判斷" in c0["A_vs_B_verdict"] and "無法判斷" in c0["C_vs_D_verdict"],
              "verdict 明說無法判斷")
        check("臂內不一致" in out, "摘要行點名了這件事")

        # ---- F. interleaving and rotation, and what the manifest records ----
        print()
        print("F. 交錯與輪轉（reps=4 時每個臂應在每個位置各出現一次）")
        rc, _ = run_loop(["--only", "Q1", "--reps", "4", "--memories-scope", "first:8"],
                         out=tmp / "f", model=stable)
        man = load_json(tmp / "f", "manifest.json")
        orders = man["arm_order_by_rep"]
        arms = set(man["arms"])
        check(rc == 0 and len(orders) == 4 and all(set(v) == arms for v in orders.values()),
              "每個 rep 都是同一組臂的排列")
        check(len({tuple(v) for v in orders.values()}) == 4,
              "四個 rep 的順序互不相同（有輪轉）")
        positions = {a: [list(v).index(a) for v in orders.values()] for a in sorted(arms)}
        check(all(sorted(v) == [0, 1, 2, 3] for v in positions.values()),
              "每個臂在四個位置各出現恰好一次", str(positions))
        check(man["calls"] == 4 * 4, "manifest 記的呼叫數正確", str(man["calls"]))
        check(len(man["scope_ids"]) == man["n_lessons_injected"] == 8,
              "manifest 完整記下注入了哪些 id", f"{man['n_lessons_injected']} 條")
        check("model_cmd" not in man and len(man["model_cmd_sha256_16"]) == 16,
              "manifest 只記 model cmd 的 sha256，不記原文（它可能含 API key）")

        # ---- G. the isolation self-check prints what it must ----
        print()
        print("G. 隔離自檢的輸出")
        rc, out = run_loop(["--dry-run", "--memories-scope", "first:8"], out=o)
        check("隔離自檢：3 對，每一對都只隔離一件事 [ok]" in out, "三對都只隔離一件事")
        m = re.search(r"C_head\s+-> D_head_mem\s+charter \+\s*0/-\s*0 行 \(\s*\+0 B\)\s+"
                      r"memories\s+\+(\d+) B", out)
        check(m is not None and int(m.group(1)) > 0,
              "C→D 那行把 charter 與 memories 分開報（charter 0 行、memories > 0）",
              f"memories +{m.group(1)} B" if m else "沒抓到那一行")
        m2 = re.search(r"A_preD6\s+-> B_postD6\s+charter \+\s*(\d+)/-\s*0 行 \(\s*\+\d+ B\)\s+"
                       r"memories\s+\+0 B", out)
        check(m2 is not None and int(m2.group(1)) > 0,
              "A→B 那行 charter 有增行且 memories 為 0（D6 本體）")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAILS:
        print(f"  {len(FAILS)} FAILED of {CHECKS}: {FAILS}")
        return 1
    print(f"  all {CHECKS} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
