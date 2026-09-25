#!/usr/bin/env python3
"""作廢數字引用檢查器（權威規則：docs/MEASUREMENT_CONTRACT_2026-09-25.md §7）

規則：**作廢數字不得作為決策依據**。若某段確需提及（歷史敘述），
      該段（以空行分段）內必須帶作廢標記（作廢／廢棄／不可引用／⛔／void）。

為什麼需要它
------------
`9.82` / `12.62` / `+28.5%` 這組數字是 **HTTP 舊口徑、無 warm-skip** 量的，
而**現行生產口徑 MTP off 已 11.03~12.20**（09-24 同 cell 實測 12.05 ± 0.57）⇒ 分母本身失效。
repo 裡它出現在 30+ 個檔案，靠人記「哪幾個能用」必然出錯 ⇒ 把它變成可機檢的規則。

掃描範圍（**分層，不是一刀切**）
- **決策面**（預設，違規 ⇒ exit 1）：契約／判決單頁／台帳／里程碑／索引／權威主題檔。
- **歸檔面**（`--all`，只稽核、不影響 exit code）：dated 產物與逐日誌 ——
  專案慣例 **dated 產物不回改**，所以它們只需要「不得再被引用」，不需要逐段改寫。

用法
----
    python3 scripts/check/void_number_check.py              # 決策面（違規 ⇒ exit 1）
    python3 scripts/check/void_number_check.py --all        # 決策面 + 歸檔面稽核
    python3 scripts/check/void_number_check.py --citations  # 引用盤點（含已標記者，markdown）
    python3 scripts/check/void_number_check.py --selftest
"""
from __future__ import annotations

import argparse
import contextlib
import io
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

REGISTRY_GLOB = "docs/MEASUREMENT_CONTRACT*.md"
REGISTRY_SECTION = "## §7 作廢數字登記表"

VOID_MARKERS = ("作廢", "廢棄", "不可引用", "⛔", "void", "已作廢", "不再引用", "不採用")

# 百分比型作廢數字（登記表寫成 `+28.5%`）→ 需要 (a) 後面真的接 `%`，
# (b) 同一段出現下列語境詞之一。理由：`28.5` 是常見數值（例：某表的 28.5% 佔比），
# 不這樣限會把無關的百分比算成引用。這條是**誤報防護**，不是放寬。
WEAK_TOKEN_CTX = ("MTP", "spec", "加速", "攤薄")

# 決策面：動決策前會讀的檔案
DECISION_SURFACE = (
    "docs/MEASUREMENT_CONTRACT*.md",
    "docs/MILESTONE_MAP_RECHECK*.md",
    "docs/DIAGNOSTIC_ARMS_LEDGER*.md",
    "docs/S1_LINE_VERDICT*.md",
    "docs/IO_AXIS_VERDICT*.md",
    "docs/MW_WORKERS_VERDICT*.md",
    "docs/MTP_AMORTIZATION*.html",
    "docs/S1_TPOT_DECOMPOSITION*.html",
    ".workbuddy/memory/MEMORY.md",
    ".workbuddy/memory/MEMORY_PERF.md",
    ".workbuddy/memory/MEMORY_S1.md",
    ".workbuddy/memory/MEMORY_FACTS.md",
    ".workbuddy/memory/MEMORY_HYGIENE.md",
)

ARCHIVE_GLOBS = (
    "docs/*.md",
    "docs/*.html",
    ".workbuddy/memory/*.md",
    "agent_harness/memory/*.md",
)


# ───────────────────────── 核心 ─────────────────────────

def parse_registry(root: Path) -> tuple[dict[str, bool], Path | None]:
    """從契約 §7 表格的第一欄抽作廢數字。

    回傳 {數字: require_pct}。來源是**反引號內**的字串（例：`9.82 t/s`、`+28.5%`）；
    若該字串含 `%` ⇒ 視為百分比型（需 `%` ＋ MTP 語境）。
    """
    cands = sorted(root.glob(REGISTRY_GLOB))
    if not cands:
        return {}, None
    doc = cands[-1]
    text = doc.read_text(encoding="utf-8")
    i = text.find(REGISTRY_SECTION)
    if i < 0:
        return {}, doc
    j = text.find("\n## ", i + len(REGISTRY_SECTION))
    block = text[i: j if j > 0 else len(text)]
    tokens: dict[str, bool] = {}
    for line in block.splitlines():
        if not line.startswith("|"):
            continue
        first = line.split("|")[1] if line.count("|") >= 2 else ""
        if "作廢數字" in first or set(first.strip()) <= set("-: "):
            continue
        for span in re.findall(r"`([^`]+)`", first) or [first]:
            for core in re.findall(r"\d+\.\d+", span):
                tokens[core] = tokens.get(core, False) or ("%" in span)
    return tokens, doc


def token_re(tok: str, require_pct: bool = False) -> re.Pattern[str]:
    # 邊界：19.82 不可以命中 9.82；百分比型要求後面真的接 %
    tail = r"%?" if not require_pct else r"%"
    return re.compile(r"(?<![\d.])" + re.escape(tok) + tail + r"(?![\d])")


def paragraph_at(text: str, pos: int) -> str:
    head = text[:pos].split("\n\n")[-1]
    tail = text[pos:].split("\n\n")[0]
    return head + tail


def iter_citations(path: Path, tokens: dict[str, bool]):
    """yield (行號, token, 是否已標記, 上下文)。"""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return
    for tok in sorted(tokens):
        pct = tokens[tok]
        for m in token_re(tok, pct).finditer(text):
            para = paragraph_at(text, m.start())
            if pct and not any(k in para for k in WEAK_TOKEN_CTX):
                continue          # 誤報防護：百分比型需 MTP 語境
            marked = any(k in para for k in VOID_MARKERS)
            line = text[:m.start()].count("\n") + 1
            snippet = text[max(0, m.start() - 45): m.start() + 30]
            snippet = snippet.replace("\n", " ").replace("|", "/").strip()
            yield line, tok, marked, snippet


def scan_file(path: Path, tokens: dict[str, bool]) -> list[tuple[int, str, str]]:
    """只列**未標記**的。"""
    return [(ln, tok, snip) for ln, tok, marked, snip in iter_citations(path, tokens)
            if not marked]


def expand(root: Path, patterns) -> list[Path]:
    files: list[Path] = []
    for pat in patterns:
        files += [p for p in root.glob(pat) if p.is_file()]
    return sorted(set(files))


def split_surfaces(root: Path, want_all: bool) -> tuple[list[Path], list[Path]]:
    decision = expand(root, DECISION_SURFACE)
    arch = [p for p in expand(root, ARCHIVE_GLOBS) if p not in decision]
    return decision, (arch if want_all else [])


def run(root: Path, want_all: bool) -> int:
    tokens, doc = parse_registry(root)
    if not doc:
        print("⛔ 找不到作廢登記表來源（docs/MEASUREMENT_CONTRACT*.md）")
        return 1
    if not tokens:
        print(f"⛔ {doc} 的「{REGISTRY_SECTION}」找不到、或表裡沒有數字")
        return 1
    print(f"登記表 {doc.relative_to(root)}  ⇒ 作廢數字 {sorted(tokens)}")

    decision, arch = split_surfaces(root, want_all)

    print(f"\n=== 決策面（{len(decision)} 檔）===")
    bad = 0
    for p in decision:
        for line, tok, snip in scan_file(p, tokens):
            bad += 1
            print(f"  ✗ {p.relative_to(root)}:{line}  [{tok}]  …{snip}…")
    print("  ✓ 無未標記引用" if bad == 0 else f"  ✗ {bad} 處未標記引用")

    if want_all:
        print(f"\n=== 歸檔面稽核（{len(arch)} 檔；dated 產物不回改，只需「不得再被引用」）===")
        n = 0
        for p in arch:
            hits = scan_file(p, tokens)
            if hits:
                n += len(hits)
                print(f"  · {p.relative_to(root)}  ({len(hits)} 處)")
        print(f"  合計 {n} 處歷史引用 —— 依法不得再被引用（違規判定只在決策面）")

    print()
    if bad:
        print(f"VERDICT: FAIL — 決策面有 {bad} 處把作廢數字當依據（見 {doc.relative_to(root)} §7）")
        return 1
    print("VERDICT: PASS — 決策面沒有未標記的作廢數字引用")
    return 0


def citations(root: Path, want_all: bool) -> int:
    """引用盤點：把**所有**引用（含已標記者）列成 markdown 表。"""
    tokens, doc = parse_registry(root)
    if not tokens:
        print("⛔ 找不到作廢數字（契約 §7）")
        return 1
    decision, arch = split_surfaces(root, want_all)
    print(f"| 檔案 | 行 | 作廢數字 | 已標記 | 上下文 |")
    print("|---|---|---|---|---|")
    n_dec = n_arc = 0
    for group, flag in ((decision, "D"), (arch, "A")):
        for p in group:
            for line, tok, marked, snip in iter_citations(p, tokens):
                if flag == "D":
                    n_dec += 1
                else:
                    n_arc += 1
                print(f"| `{p.relative_to(root)}` | {line} | `{tok}` | "
                      f"{'✅' if marked else '⛔ 未標記'} | …{snip}… |")
    print(f"\n合計 **{n_dec + n_arc}** 處引用（決策面 {n_dec}／歸檔面 {n_arc}）"
          f"；判定規則見 `{doc.relative_to(root)}` §7")
    return 0


# ───────────────────────── selftest ─────────────────────────

def selftest() -> int:
    ok: list[tuple[str, bool]] = []

    def chk(name: str, cond: bool) -> None:
        ok.append((name, bool(cond)))

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "docs").mkdir()
        (root / ".workbuddy/memory").mkdir(parents=True)

        contract = root / "docs/MEASUREMENT_CONTRACT_2026-09-25.md"
        contract.write_text(
            "# 契約\n\n## §6 x\n\ntext\n\n" + REGISTRY_SECTION + "\n\n"
            "| 作廢數字 | 出處 | 理由 | 替代 |\n|---|---|---|---|\n"
            "| **`9.82 t/s`** | HTTP | 舊口徑 | 11.03 |\n"
            "| **`+28.5%`** | 比值 | 分子分母皆廢 | UNRESOLVED |\n\n"
            "## §8 y\n\ntext\n",
            encoding="utf-8")

        tokens, doc = parse_registry(root)
        chk("解析登記表（含 % 標記）",
            set(tokens) == {"9.82", "28.5"} and tokens["28.5"] and not tokens["9.82"]
            and doc == contract)

        verdict = root / "docs/S1_LINE_VERDICT_2026-09-25.md"
        verdict.write_text("MTP off 9.82 → on 12.62 是 +28.5%。\n", encoding="utf-8")
        v = scan_file(verdict, tokens)
        chk("未標記 ⇒ 抓到 9.82 與 28.5", {t for _, t, _ in v} == {"9.82", "28.5"})

        verdict.write_text("⛔ 9.82 已作廢；+28.5% 已作廢。\n", encoding="utf-8")
        chk("帶 ⛔/作廢 ⇒ 放行", scan_file(verdict, tokens) == [])

        verdict.write_text("⛔ 作廢：9.82。\n\n這裡又用 MTP 的 28.5% 當依據。\n", encoding="utf-8")
        chk("標記只覆蓋同段落", [t for _, t, _ in scan_file(verdict, tokens)] == ["28.5"])

        verdict.write_text("19.82 t/s 是別的數字。\n", encoding="utf-8")
        chk("邊界 19.82 ≠ 9.82", scan_file(verdict, tokens) == [])

        # 引用盤點要同時列出「已標記」與「未標記」
        verdict.write_text("⛔ 9.82 作廢。\n\n但他又寫 9.82。\n", encoding="utf-8")
        cites = list(iter_citations(verdict, tokens))
        chk("引用盤點含已標記者", len(cites) == 2 and sum(1 for c in cites if c[2]) == 1)

        verdict.write_text("28.5 是別的意思（無百分比）。\n", encoding="utf-8")
        chk("百分比型：裸數 28.5 不算", scan_file(verdict, tokens) == [])

        verdict.write_text("別的表格 28.5%。\n", encoding="utf-8")
        chk("百分比型：缺 MTP 語境不算", scan_file(verdict, tokens) == [])

        verdict.write_text("MTP 增益 +28.5% 當依據。\n", encoding="utf-8")
        chk("百分比型：MTP 語境 ⇒ 抓到", [t for _, t, _ in scan_file(verdict, tokens)] == ["28.5"])

        verdict.write_text("⛔ MTP 9.82 已作廢。\n", encoding="utf-8")
        (root / ".workbuddy/memory/2026-09-17.md").write_text("9.82 t/s\n", encoding="utf-8")
        (root / ".workbuddy/memory/MEMORY.md").write_text("- MTP：⛔ 9.82 作廢。\n", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            r_pass = run(root, want_all=False)
        chk("決策面 PASS 而歸檔面有引用 ⇒ exit 0", r_pass == 0)

        (root / ".workbuddy/memory/MEMORY.md").write_text("- MTP：9.82 ⇒ 別動。\n", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            r_fail = run(root, want_all=False)
        chk("決策面未標記 ⇒ exit 1", r_fail == 1)

        contract.write_text("# 契約\n\n## §6 x\n\ntext\n", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            r_noreg = run(root, want_all=False)
        chk("§7 缺失 ⇒ exit 1", r_noreg == 1)

    for name, cond in ok:
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    print(f"selftest {sum(c for _, c in ok)}/{len(ok)}")
    return 0 if all(c for _, c in ok) else 1


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="納入歸檔面（稽核）")
    ap.add_argument("--citations", action="store_true", help="引用盤點（markdown 表）")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.citations:
        return citations(ROOT, True)   # 引用盤點一律含歸檔面（它的用途就是全庫盤點）
    return run(ROOT, args.all)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
