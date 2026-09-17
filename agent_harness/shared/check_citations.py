#!/usr/bin/env python3
"""把憲章自己的驗收條件變成可機檢的：每一條要嘛有可解析的指針，要嘛**顯式標記**它的證據形態。

THE ACCEPTANCE WAS ALREADY WRITTEN -- IT JUST WAS NOT CHECKABLE
----------------------------------------------------------------
`CONVENTIONS.md` 第 7 行自己寫著：

> 每一條都必須能指到**今天的具體證據**（檔名／log 行／欄位值）。指不到的條文不是憲章，是感想。

括號裡那三種形態就是重點：**「欄位值」沒有檔案可以指**，而它完全合法（B25 是「BSD grep 不支援
`\\|`」，它的證據是一次指令輸出的字面）。所以這條驗收的正確形式不是「逼出更多檔名」，而是：

    ① 每一條要嘛有一個**可解析**的指針，要嘛有一行 `證據形態：…` 說明它的證據是什麼
    ② 沒有任何**懸空**引用（指到不存在的檔）
    ③ 這個檢查自己要先被證明**會失敗**（見 --self-test）

    python3 agent_harness/shared/check_citations.py            # 檢查，非零碼代表沒過
    python3 agent_harness/shared/check_citations.py --report   # 只印分佈，always 0
    python3 agent_harness/shared/check_citations.py --self-test

WHY THIS FILE IS MOSTLY ABOUT ITS OWN FALSE POSITIVES
------------------------------------------------------
第一版探針在「乾淨」的憲章上報了 **84 個問題**，全部是量具自己的。六類，每一類都留了一個
對應的處理：

  1. 只在 repo 根解析        -> 80 個假懸空（`validate.py` 其實在 `engine_loop/traces/`）
  2. 行文斜線當路徑           -> `gpu_union/wait`、`peak/mean`、`IQR/中位數`
  3. decision id 精確比對     -> 3 個假懸空；id 是 `dec-YYYYMMDD-HHMM-<slug>`，要前綴比對
  4. 副檔名**本身**           -> `.m`／`.sh`：行文在講「獨漏 .m」，不是在引用檔案
  5. 承前省略的引用           -> `..._202725.log`（人看得懂、機器不能）
  6. **同一格程式碼裡路徑後面跟著參數** -> `prefill_gputime_report.py --decprof-pair …`
     這一類讓 A18 被誤判成「完全沒有指針」，而它其實指得很清楚

第 6 類是最貴的：它讓一條**本來就合格**的規則被報成不合格，於是人會去「修」一條沒壞的條文。
所以每個 code span 都先按空白切開，逐欄試。

`--self-test` 的陽性對照是最重要的那一段：把一個假路徑注入副本，檢查器必須立刻從「0 懸空」
變成「1 懸空」。**沒有這一格，「0 懸空」與「檢查器根本沒在跑」是同一個輸出。**
"""
from __future__ import annotations

import argparse
import collections
import fnmatch
import glob as globmod
import os
import pathlib
import re
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent          # agent_harness/shared
REPO = HERE.parents[1]                                   # repo root
CHARTER = HERE.parent / "CONVENTIONS.md"                 # agent_harness/CONVENTIONS.md

RULE_RE = re.compile(r"^\*\*([A-Z]\d+)｜", re.MULTILINE)
CODE_RE = re.compile(r"`([^`\n]{1,160})`")
EXT_RE = re.compile(r"\.(py|sh|json|jsonl|md|html|cpp|h|jinja|txt|gguf|ps1|zst|log|c|cu|m|mm|metal)$")
LESSON_RE = re.compile(r"eng-[a-z]+-\d{4}")
DECISION_RE = re.compile(r"dec-\d{8}-\d{4}")
# 顯式標記。要求**行首**（可有縮排與 `- `）—— 免得行文提到這個詞就被當成標記。
LABEL_RE = re.compile(r"^\s*(?:[-*]\s*)?證據形態：\s*(\S.{2,})\s*$", re.MULTILINE)

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "build", "versions"}
RESOLVING = ("exact", "basename", "glob", "record-id")


def build_basename_index(root: pathlib.Path) -> dict[str, list[str]]:
    index: dict[str, list[str]] = collections.defaultdict(list)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        rel = pathlib.Path(dirpath).relative_to(root)
        for fn in filenames:
            index[fn].append(str(rel / fn))
    return index


def classify(token: str, root: pathlib.Path, index: dict[str, list[str]],
             lessons: set[str], decisions: list[str]) -> tuple[str, str] | None:
    """(kind, detail) 或 None（不是引用）。kind 在 RESOLVING 裡 = 可解析。"""
    t = token.strip().rstrip(".,;:)")
    if not t or "=" in t:
        return None
    if LESSON_RE.fullmatch(t):
        return ("record-id", t) if t in lessons else ("DANGLING", t)
    if DECISION_RE.fullmatch(t):
        # ★ 前綴比對：id 是 dec-YYYYMMDD-HHMM-<slug>，憲章常只引到 dec-YYYYMMDD-HHMM
        return ("record-id", t) if any(d.startswith(t) for d in decisions) else ("DANGLING", t)

    base = re.sub(r":\d+(-\d+)?$", "", t)
    if not base:
        return None
    if re.fullmatch(r"\.[A-Za-z]+", base):          # `.sh` —— 提到副檔名本身
        return None
    if base.startswith("..."):                      # `..._202725.log` —— 承前省略
        return ("elided", base)
    if "<" in base or ">" in base:                  # `<main-worktree>/.git/hooks` —— 佔位符
        return ("template", base)
    if base.startswith("~") or base.startswith("/"):
        return ("external", base)

    if "|" in base:                                  # `*src/*.cpp|*.h|…` —— grep 的 alternation
        return None
    if "*" in base or "?" in base:
        if "/" not in base and not EXT_RE.search(base):
            return None                              # `GGML_*`／`ggml_reshape_*` —— 識別字模式，不是路徑
        m = globmod.glob(str(root / base))
        if m:
            return ("glob", f"{base} ({len(m)} match)")
        # 沒有目錄前綴的簡寫（`cap_*.json`）：檔名本身就足以定位時，用 basename 索引配
        byname = [n for n in index if fnmatch.fnmatch(n, pathlib.Path(base).name)]
        return ("glob", f"{base} ({len(byname)} by name)") if byname else ("DANGLING", base)

    for b in (root, root / "agent_harness", root / "scripts", root / "scripts/check"):
        if (b / base).exists():
            return ("exact", base)
    if not EXT_RE.search(base):
        return None                                  # 行文斜線、程式片段
    hits = index.get(pathlib.Path(base).name, [])
    if not hits:
        return ("DANGLING", base)
    if len(hits) == 1:
        return ("basename", base)
    return ("ambiguous", f"{base} -> {len(hits)} candidates")


def split_fields(token: str) -> list[str]:
    """一個 code span 可能是 `path --arg --arg`。逐欄試，順序不變。"""
    return [f for f in token.split() if f]


def parse_charter(path: pathlib.Path, root: pathlib.Path, index: dict, lessons: set[str],
                  decisions: list[str]) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    starts = [i for i, l in enumerate(lines) if RULE_RE.match(l)]
    out = []
    for n, i in enumerate(starts):
        j = starts[n + 1] if n + 1 < len(starts) else len(lines)
        body = "\n".join(lines[i:j])
        rid = RULE_RE.match(lines[i]).group(1)
        kinds: list[tuple[str, str]] = []
        for span in CODE_RE.findall(body):
            for field in split_fields(span):
                c = classify(field, root, index, lessons, decisions)
                if c:
                    kinds.append(c)
        label = LABEL_RE.search(body)
        out.append({"id": rid, "line": i + 1, "kinds": kinds,
                    "label": label.group(1) if label else None})
    return out


def load_ids(root: pathlib.Path) -> tuple[set[str], list[str]]:
    import json
    base = root / "agent_harness/engine_loop/traces"
    lessons = {json.loads(l)["lesson_id"]
               for l in (base / "lessons.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    decisions = [json.loads(l)["decision_id"]
                 for l in (base / "decisions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return lessons, decisions


def analyse(root: pathlib.Path, charter: pathlib.Path) -> tuple[list[dict], collections.Counter]:
    index = build_basename_index(root)
    lessons, decisions = load_ids(root)
    rules = parse_charter(charter, root, index, lessons, decisions)
    cat: collections.Counter = collections.Counter()
    for r in rules:
        for k, _ in r["kinds"]:
            cat[k] += 1
    return rules, cat


def problems(rules: list[dict]) -> tuple[list[str], list[str]]:
    bad, unresolved = [], []
    for r in rules:
        dangling = [d for k, d in r["kinds"] if k == "DANGLING"]
        if dangling:
            bad.append(f"{r['id']} (L{r['line']}): 懸空引用 -> {', '.join(sorted(set(dangling)))}")
        if not dangling and not any(k in RESOLVING for k, _ in r["kinds"]) and not r["label"]:
            seen = sorted({f"{k}:{d}" for k, d in r["kinds"]}) or ["（沒有任何指針）"]
            unresolved.append(f"{r['id']} (L{r['line']}): 沒有可解析指針，也沒有 `證據形態：` -> {', '.join(seen)}")
    return bad, unresolved


def self_test() -> int:
    fails: list[str] = []
    total = [0]

    def check(ok: bool, label: str, detail: str = "") -> None:
        total[0] += 1
        print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + (f"   {detail}" if detail and not ok else ""))
        if not ok:
            fails.append(label)

    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        (root / "agent_harness/engine_loop/traces").mkdir(parents=True)
        (root / "sub").mkdir()
        (root / "sub/unique.py").write_text("x", encoding="utf-8")
        (root / "sub/withargs.py").write_text("x", encoding="utf-8")
        (root / "sub/dup.py").write_text("x", encoding="utf-8")
        (root / "other/dup.py").parent.mkdir(parents=True)
        (root / "other/dup.py").write_text("x", encoding="utf-8")
        (root / "g1.log").write_text("x", encoding="utf-8")
        (root / "agent_harness/engine_loop/traces/lessons.jsonl").write_text(
            '{"lesson_id": "eng-mh-0001"}\n', encoding="utf-8")
        (root / "agent_harness/engine_loop/traces/decisions.jsonl").write_text(
            '{"decision_id": "dec-20260915-1930-a-slug"}\n', encoding="utf-8")
        index = build_basename_index(root)
        lessons, decisions = {"eng-mh-0001"}, ["dec-20260915-1930-a-slug"]

        print("=== 1. 每一類指針的分級（陽性與陰性都要對）===")
        for tok, want in (
            ("sub/unique.py", "exact"),
            ("unique.py", "basename"),
            ("g*.log", "glob"),
            ("eng-mh-0001", "record-id"),
            ("dec-20260915-1930", "record-id"),
            ("dup.py", "ambiguous"),
            ("<worktree>/.git/hooks", "template"),
            ("..._202725.log", "elided"),
            ("~/.workbuddy/x", "external"),
            ("sub/nope.py", "DANGLING"),
            (".sh", None),
            ("gpu_union/wait", None),
            ("sub/withargs.py --flag x", "exact"),   # 第 6 類：同一格裡的參數要切掉
        ):
            got = classify(tok.split()[0] if " " in tok else tok, root, index, lessons, decisions)
            kind = got[0] if got else None
            check(kind == want, f"{tok!r} -> {want}", f"got {kind}")

        print()
        print("=== 2. 標記：有 `證據形態：` 的條文豁免；沒有指針也沒有標記的要失敗 ===")
        scenario = {
            "labelled": "**B1｜規則。**\n- 證據形態：一行指令的輸出（`grep -n` at 13:0x）\n",
            "silent": "**B2｜規則。**\n- 這裡什麼都沒有。\n",
            "resolved": "**B3｜規則。**\n- 見 `sub/unique.py`。\n",
        }
        judge = {}
        for name, body in scenario.items():
            dup = root / name
            dup.write_text("# x\n\n" + body, encoding="utf-8")
            rules, _ = analyse(root, dup)
            bad, unres = problems(rules)
            judge[name] = (len(bad), len(unres))
        check(judge["labelled"] == (0, 0), "有標記 => 過", str(judge))
        check(judge["silent"][1] == 1, "沉默 => 不過", str(judge))
        check(judge["resolved"] == (0, 0), "有可解析指針 => 過", str(judge))

    print()
    print("=== 3. 對真實憲章的陽性／陰性對照（這一格最重要）===")
    real_rules, _ = analyse(REPO, CHARTER)
    bad, unres = problems(real_rules)
    baseline = len(bad)
    check(True, f"基準線：{len(real_rules)} 條、{baseline} 個懸空")

    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td) / "CONVENTIONS.md"
        text = CHARTER.read_text(encoding="utf-8")
        first = RULE_RE.search(text)
        injected = text[:first.start()] + first.group(0) + "（假引用：`agent_harness/fake/nope.jsonl`）" \
            + text[first.end():]
        tmp.write_text(injected, encoding="utf-8")
        rules, _ = analyse(REPO, tmp)
        bad2, _ = problems(rules)
        check(len(bad2) == baseline + 1, f"注入一個假路徑 => 懸空必須 +1（{baseline} -> {len(bad2)}）",
              f"got {len(bad2)}")

    print()
    if fails:
        print(f"  {len(fails)}/{total[0]} FAILED: {fails}")
        return 1
    print(f"  all {total[0]} checks passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="exit 1 if any rule is unresolved or dangling")
    ap.add_argument("--report", action="store_true", help="print the distribution only, always exit 0")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--root", type=pathlib.Path, default=REPO)
    ap.add_argument("--charter", type=pathlib.Path, default=None)
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    charter = args.charter or (args.root / "agent_harness/CONVENTIONS.md")
    rules, cat = analyse(args.root, charter)
    bad, unres = problems(rules)

    n = len(rules)
    resolvable = [r["id"] for r in rules if any(k in RESOLVING for k, _ in r["kinds"])]
    labelled = [r["id"] for r in rules if not any(k in RESOLVING for k, _ in r["kinds"]) and r["label"]]
    print(f"  憲章 {charter.relative_to(args.root)}：{n} 條、{sum(cat.values())} 個引用")
    print(f"  引用分類: {dict(cat)}")
    print()
    print(f"  ① 可解析指針            : {len(resolvable):3d}/{n}")
    print(f"  ② 顯式 `證據形態：` 標記 : {len(labelled):3d}/{n}")
    print(f"  ③ 兩者皆無              : {len(unres):3d}/{n}  {' '.join(r.split(' ')[0] for r in unres) if unres else ''}")
    print(f"  ④ 懸空                  : {len(bad):3d}/{n}  {' '.join(r.split(' ')[0] for r in bad) if bad else ''}")
    print()
    if unres:
        print("  --- 兩者皆無的條文（要嘛補一個可解析指針，要嘛標記證據形態）---")
        for u in unres:
            print(f"    {u}")
    if bad:
        print("  --- 懸空引用 ---")
        for b in bad:
            print(f"    {b}")

    if args.report:
        return 0
    if bad or unres:
        print()
        print(f"  [error] {len(bad)} 懸空 ＋ {len(unres)} 兩者皆無 ⇒ 憲章第 7 行的標準未達成")
        return 1
    print(f"  OK: {len(resolvable) + len(labelled)}/{n} 條都有交代（可解析指針 {len(resolvable)}"
          f" ＋ 顯式標記 {len(labelled)}），0 懸空")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
