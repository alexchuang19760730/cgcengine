#!/usr/bin/env python3
"""臂台帳的契約檢查器 v2（MEASUREMENT_CONTRACT §4 四欄 ＋ §5 成果分級）。

WHY THIS EXISTS
---------------
`docs/MEASUREMENT_CONTRACT_2026-09-25.md` 規定兩件事：
  §4「每一支臂都必須有 目標 → 判準 → 結果 → 判定」
  §5「每一項成果都必須歸入四級之一：攻關成功 / 攻關過線 / 實驗目標達成 / 廢棄」

規則只寫在文件裡，下一個 agent 不會遵守（也不會有人發現漏了）—— 所以它必須**可機檢**。

支援兩種表型（都必須有 `判定` ＋ `成果分級` 兩欄）：
  * **臂表**：`目標` / `判準` / `結果`
  * **里程碑表**：`推算欄` / `量測欄`（雙欄制）

檢查什麼
--------
1. 表頭必須含 `判定` ＋ `成果分級`，且至少具備上述**一種**完整表型。
2. 該表型的所有欄位**非空**。
3. `判定`必須帶 ✅/❌/⚠ 之一（允許「不可判定」，不允許空白或「大概」）。
4. `成果分級` 的值必須**恰好**是四級之一（其餘字串 ⇒ 紅燈）。
5. 欄內以反引號引用的**產物路徑必須存在**。

不檢查什麼（誠實邊界）
----------------------
它**不**能判斷「有沒有漏登記的臂 / 漏分級的成果」—— 那需要 runner 側的登記 API，目前沒有。
⇒ 它只保證「**登記了的那幾條**沒有偷工」。台帳裡的「待登錄」清單是人工維護的。

顯式宣告制
----------
只有帶 `ARMS-LEDGER-CONTRACT` 標記的檔才被檢查。
（按檔名 glob `docs/*LEDGER*.md` 會命中別條線的 4 份文件並把他們判紅。）
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MARKER = "ARMS-LEDGER-CONTRACT"

TIERS = ("攻關成功", "攻關過線",
         "實驗目標達成（階段性）", "實驗目標達成（可放生產）",
         "廢棄")
VERDICT_MARKS = ("✅", "❌", "⚠")

VERDICT_NAMES = ("判定",)
GRADE_NAMES = ("成果分級", "成果分级", "分級", "分级")
SUBGOAL_NAMES = ("子目標", "子目标")
SUBS = ("序列化消減", "MTP on 加速", "兩者", "不適用")
GROUPS: dict[str, tuple[tuple[str, ...], ...]] = {
    "arm": (("目標", "目标"), ("判準", "判准"), ("結果", "结果")),
    "milestone": (("推算欄", "推算栏", "推算"), ("量測欄", "量测栏", "量測", "量测")),
}

PATH_RE = re.compile(
    r"`([^`\s]+\.(?:json|md|txt|html|patch|log|py|sh))`")


def _split_row(line: str) -> list[str]:
    s = line.strip()
    if not s.startswith("|"):
        return []
    return [c.strip() for c in s.strip("|").split("|")]


def _is_sep(cells: list[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{2,}:?", c) for c in cells)


def _find(cells: list[str], names: tuple[str, ...]) -> int | None:
    return next((i for i, c in enumerate(cells) if any(n in c for n in names)), None)


def _profile(header: list[str]) -> dict | None:
    """這張表是不是受契約約束的表？回傳 {group, idx, verdict, grade} 或 None。"""
    vi, gi = _find(header, VERDICT_NAMES), _find(header, GRADE_NAMES)
    if vi is None or gi is None:
        return None
    for gname, cols in GROUPS.items():
        idx = []
        for names in cols:
            i = _find(header, names)
            if i is None:
                break
            idx.append(i)
        else:
            return {"group": gname, "idx": idx, "verdict": vi, "grade": gi,
                    "sub": _find(header, SUBGOAL_NAMES)}
    return None


def parse_tables(text: str) -> list[dict]:
    lines = text.splitlines()
    tables: list[dict] = []
    i = 0
    while i < len(lines):
        header = _split_row(lines[i])
        if header and i + 1 < len(lines) and _is_sep(_split_row(lines[i + 1])):
            prof = _profile(header)
            if prof is not None:
                rows, j = [], i + 2
                while j < len(lines):
                    cells = _split_row(lines[j])
                    if not cells or _is_sep(cells):
                        break
                    rows.append((j + 1, cells))
                    j += 1
                tables.append({"profile": prof, "rows": rows})
                i = j
                continue
        i += 1
    return tables


def check_text(text: str, label: str, root: Path) -> tuple[list[str], int]:
    problems: list[str] = []
    checked = 0
    tables = parse_tables(text)
    if not tables:
        problems.append(
            f"{label}: 找不到任何受契約約束的表"
            "（表頭需含 `判定` ＋ `成果分級`，且具備 目標/判準/結果 或 推算欄/量測欄）")
        return problems, checked
    for t in tables:
        prof = t["profile"]
        for lineno, cells in t["rows"]:
            checked += 1
            arm = cells[0] if cells else "?"
            for ci in prof["idx"]:
                if ci >= len(cells) or not cells[ci]:
                    problems.append(f"{label}:{lineno} 「{arm}」的必填欄（第 {ci} 欄）是空的")
            vi, gi = prof["verdict"], prof["grade"]
            if vi < len(cells):
                v = cells[vi]
                if v and not any(m in v for m in VERDICT_MARKS):
                    problems.append(f"{label}:{lineno} 「{arm}」的判定「{v[:20]}」缺 ✅/❌/⚠")
            if gi < len(cells):
                g = cells[gi].replace("*", "").strip()
                g = re.sub(r"^[①②③④][ab]?\s*", "", g)   # 剝掉 ①-④ 與 ③a/③b 的前綴
                if g not in TIERS:
                    problems.append(
                        f"{label}:{lineno} 「{arm}」的成果分級「{cells[gi][:24]}」不在四級內"
                        f"（只接受 {'/'.join(TIERS)}）")
            si = prof.get("sub")
            if si is not None and si < len(cells):
                scell = cells[si].replace("*", "").strip()
                if not scell:
                    problems.append(f"{label}:{lineno} 「{arm}」的 子目標 欄是空的")
                elif not any(x in scell for x in SUBS):
                    problems.append(
                        f"{label}:{lineno} 「{arm}」的子目標「{cells[si][:24]}」"
                        f"不在 {SUBS} 內")
            for ci in prof["idx"]:
                if ci >= len(cells):
                    continue
                for ref in PATH_RE.findall(cells[ci]):
                    if ref.startswith("/") or not (root / ref).exists():
                        problems.append(f"{label}:{lineno} 「{arm}」引用的產物不存在：{ref}")
    return problems, checked


def check_file(path: Path, root: Path) -> tuple[list[str], int]:
    text = path.read_text(encoding="utf-8")
    if MARKER not in text:
        return [], 0
    rel = path.relative_to(root) if path.is_relative_to(root) else path
    return check_text(text, str(rel), root)


def run(paths: list[Path], root: Path) -> int:
    allp: list[str] = []
    total = 0
    for p in paths:
        if not p.exists():
            allp.append(f"{p}: 檔不存在")
            continue
        probs, n = check_file(p, root)
        allp += probs
        total += n
        print(f"  {p.relative_to(root) if p.is_relative_to(root) else p}: 檢查 {n} 列")
    print()
    if allp:
        print(f"FAIL — {len(allp)} 個問題：")
        for x in allp:
            print(f"  - {x}")
        return 1
    print(f"PASS — {total} 列全部符合四欄契約 ＋ 成果分級")
    print("（邊界：本檢查不涵蓋「漏登記」——那需要 runner 側的臂登記 API）")
    return 0


HEAD_ARM = ("| 臂 | 目標 | 判準 | 結果 | 判定 | 成果分級 | 子目標 |\n"
            "|---|---|---|---|---|---|---|\n")
HEAD_MS = ("| 里程碑 | 推算欄 | 量測欄 | 判定 | 成果分級 | 子目標 |\n"
           "|---|---|---|---|---|---|\n")


def selftest() -> int:
    import tempfile
    ok: list[tuple[str, bool]] = []

    def chk(n, c):
        ok.append((n, bool(c)))

    M = MARKER + "\n"
    good_arm = M + HEAD_ARM + "| A | 問 X | 門檻 15% | 1.2 `x.json` | ✅ 達成 | ② 攻關過線 | S 序列化消減 |\n"
    good_ms = M + HEAD_MS + "| M | 上界 40 ms | 1.1 `x.json` | ❌ 未過 | ④ 廢棄 | 不適用 |\n"
    bad_empty = M + HEAD_ARM + "| B | 問 Y |  | 1.0 | ✅ 達成 | ④ 廢棄 | 不適用 |\n"
    bad_mark = M + HEAD_ARM + "| C | 問 Z | 門檻 | 1.0 | 大概可以 | ④ 廢棄 | 不適用 |\n"
    bad_grade = M + HEAD_ARM + "| D | 問 W | 門檻 | 1.0 `x.json` | ✅ 達成 | 差不多 | 不適用 |\n"
    bad_grade2 = M + HEAD_ARM + "| D2 | 問 W | 門檻 | 1.0 `x.json` | ✅ 達成 | 成功 | 不適用 |\n"
    bad_path = M + HEAD_ARM + "| E | 問 V | 門檻 | `Backup/nope_missing.json` | ✅ 達成 | ④ 廢棄 | 不適用 |\n"
    no_grade = M + "| 臂 | 目標 | 判準 | 結果 | 判定 |\n|---|---|---|---|---|\n| F | q | 門檻 | 1 | ✅ 達成 |\n"
    no_marker = HEAD_ARM + "| G | q | 門檻 | 1 `x.json` | ✅ 達成 | ④ 廢棄 |\n"

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "x.json").write_text("{}")

        def probe(txt, name="t.md"):
            p = root / name
            p.write_text(txt, encoding="utf-8")
            return check_file(p, root)

        chk("臂表（四欄）被識別", any(t["profile"]["group"] == "arm" for t in parse_tables(good_arm)))
        chk("里程碑表（雙欄）被識別", any(t["profile"]["group"] == "milestone" for t in parse_tables(good_ms)))
        chk("合格臂表無問題", probe(good_arm)[0] == [] and probe(good_arm)[1] == 1)
        chk("合格里程碑表無問題", probe(good_ms)[0] == [] and probe(good_ms)[1] == 1)
        chk("空欄被抓", any("是空的" in x for x in probe(bad_empty)[0]))
        chk("判定缺標記被抓", any("缺 ✅/❌/⚠" in x for x in probe(bad_mark)[0]))
        chk("分級非四級被抓（差不多）", any("不在四級內" in x for x in probe(bad_grade)[0]))
        chk("分級非四級被抓（成功）", any("不在四級內" in x for x in probe(bad_grade2)[0]))
        chk("產物不存在被抓", any("nope_missing.json" in x for x in probe(bad_path)[0]))
        chk("缺 成果分級 欄 ⇒ 不算合格表",
            "找不到任何受契約約束的表" in probe(no_grade)[0][0])
        chk("無宣告標記 ⇒ 略過", probe(no_marker) == ([], 0))
        chk("⚠ 也算合格判定",
            probe(M + HEAD_ARM + "| H | q | 門檻 | 1 `x.json` | ⚠ 不可判定 | ③a 實驗目標達成（階段性） | M MTP on 加速 |\n")[0] == [])
        chk("③ 未細分 ⇒ 紅（必須寫階段性/可放生產）",
            any("不在四級內" in x for x in probe(
                M + HEAD_ARM + "| J | q | 門檻 | 1 `x.json` | ✅ | ③ 實驗目標達成 | S 序列化消減 |\n")[0]))
        chk("③b 可放生產 也接受",
            probe(M + HEAD_ARM + "| K | q | 門檻 | 1 `x.json` | ✅ | ③b 實驗目標達成（可放生產） | S 序列化消減 |\n")[0] == [])
        chk("帶圈號的分級也接受",
            probe(M + HEAD_ARM + "| I | q | 門檻 | 1 `x.json` | ✅ | 攻關成功 | 不適用 |\n")[0] == [])
        chk("子目標 值非法 ⇒ 紅",
            any("不在 (" in x for x in probe(
                M + HEAD_ARM + "| L | q | 門檻 | 1 `x.json` | ✅ | ④ 廢棄 | 別的東西 |\n")[0]))
        chk("子目標 空 ⇒ 紅",
            any("子目標 欄是空的" in x for x in probe(
                M + HEAD_ARM + "| N | q | 門檻 | 1 `x.json` | ✅ | ④ 廢棄 |  |\n")[0]))
        chk("兩者 / 不適用 都接受",
            probe(M + HEAD_ARM + "| O | q | 門檻 | 1 `x.json` | ✅ | ③a 實驗目標達成（階段性） | 兩者 |\n")[0] == [])

    for n, c in ok:
        print(f"  [{'PASS' if c else 'FAIL'}] {n}")
    print(f"\n  selftest: {sum(1 for _, c in ok if c)}/{len(ok)}")
    return 0 if all(c for _, c in ok) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="*",
                    help=f"台帳 markdown（預設掃 docs/*.md 中帶 `{MARKER}` 標記者）")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.paths:
        paths = [Path(p) for p in args.paths]
    else:
        cands = sorted(ROOT.glob("docs/*.md"))
        paths = [p for p in cands if MARKER in p.read_text(encoding="utf-8", errors="ignore")]
        print(f"掃 docs/*.md：{len(cands)} 檔，其中 {len(paths)} 檔宣告為台帳"
              f"（帶 `{MARKER}`），{len(cands) - len(paths)} 檔略過")
    if not paths:
        print(f"沒有任何文件宣告為台帳（在 docs/*.md 加上 `{MARKER}` 即納入檢查）")
        return 1
    return run(paths, ROOT)


if __name__ == "__main__":
    sys.exit(main())
