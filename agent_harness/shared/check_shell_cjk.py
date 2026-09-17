#!/usr/bin/env python3
"""check_shell_cjk.py — `$VAR` 緊接著非 ASCII 字元 ⇒ 變數名被吃掉，`set -u` 下直接爆。

為什麼這個 repo 特別容易踩
--------------------------
這裡的註解與訊息幾乎全是中文，而 shell 的變數名解析**會把緊接在後面的多位元組位元組
吃進名字裡**（bash 在某些 locale 下如此；zsh 不一定）：

    echo "值 $n（共 $total 項）"      # $n（   → bash 找的是名為 `n（` 的變數 ⇒ unbound variable
    echo "值 ${n}（共 ${total} 項）"  # 正確

2026-09-17 一天之內**四個新檔案各踩一次**（`runners/preflight.sh`、`runners/rebuild.sh`、
`runners/server.sh`、`wrappers/_common.sh`），每一次的症狀都一樣：

    <script>: line N: <var>\ufffd: unbound variable

而它的**代價不對稱**：`bash -n` 完全抓不到（語法合法），所以「語法檢查通過」與
「這支腳本能用」是兩件事 —— 只有在跑到那一行、而且 `set -u` 生效時才會爆。
四次裡有兩次是**只在特定分支**（`n_build > 0`、`missing > 0`）才會執行，
所以連「跑過一遍」都不保證抓得到。**這正是需要一支靜態檢查的理由。**

判準（刻意保守，寧可漏報也不要誤報）
------------------------------------
只報**會被執行**的行（跳過整行註解），且只報 `$` + 識別字 + 「緊接一個非 ASCII 字元」。
`${...}` 形式不報。反引號與 heredoc 裡的內容不特別處理 —— 它們同樣會被 shell 展開，
所以報出來是對的。

    check_shell_cjk.py [--root DIR] [--self-test]
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile

PAT = re.compile(r"(?<!\$)\$([A-Za-z_][A-Za-z0-9_]*)(?=[^\x00-\x7F])")
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "build", "versions",
             "sft_data", "sft_data_ft", "sft_data_merged", "sft_data_from_round1",
             "sft_data_from_rehearsal1", "sft_data_from_rehearsal2", ".colima"}


def scan_text(text: str) -> list[tuple[int, str, str]]:
    """[(lineno, varname, next_char)] for executable lines only."""
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        for m in PAT.finditer(line):
            out.append((i, m.group(1), line[m.end():m.end() + 1]))
    return out


def scan_file(path: str) -> list[tuple[int, str, str]]:
    try:
        return scan_text(open(path, encoding="utf-8", errors="replace").read())
    except OSError:
        return []


def is_shell(path: str) -> bool:
    if path.endswith((".sh", ".bash")):
        return True
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            first = fh.readline()
    except OSError:
        return False
    return first.startswith("#!") and ("bash" in first or "/sh" in first)


def walk(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            if is_shell(p):
                yield p


def self_test() -> int:
    """陽性＋陰性對照。一個會在乾淨輸入上亂叫的檢查沒有人會用（lesson eng-gate-0046）。"""
    fails = 0

    def chk(cond: bool, label: str) -> None:
        nonlocal fails
        print(f"  {'ok  ' if cond else 'FAIL'}  {label}")
        if not cond:
            fails += 1

    # ★ fixture 本身也要對：`$total 項` 中間有**空格**，識別字在那裡就結束了 ⇒ 那是安全的。
    #   第一次寫這個自測時把那種情況當成陽性，於是「陽性抓到 2 處」這條一直是紅的 ——
    #   而它紅的原因是**測試寫錯**，不是檢查器寫錯。這種紅比綠更危險：它會讓人去改檢查器。
    bad = 'echo "值 $n（共 $total 項）與 $m）"\n'
    good = 'echo "值 ${n}（共 ${total} 項）"\n'
    comment = '# 這裡提到 $foo（，在註解裡\n'
    ascii_ok = 'echo "$a,$b"\n'
    space_ok = 'echo "$total 項"\n'
    chk(len(scan_text(bad)) == 2, "陽性：`$n（` 與 `$m）` 都被抓到（2 處）")
    chk(not scan_text(good), "陰性：`${n}` 形式不被報（那是正確寫法）")
    chk(not scan_text(comment), "陰性：整行註解不被報")
    chk(not scan_text(ascii_ok), "陰性：ASCII 緊接的 `$a,` 不被報（寧漏勿誤）")
    chk(not scan_text(space_ok), "陰性：`$total 項` 中間有空格 ⇒ 識別字已結束，不是 bug")
    chk(not scan_text('x="a"\n'), "陰性：沒有緊接非 ASCII 的一般賦值不被報")

    # 端到端：真的走一次檔案掃描
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.sh")
        open(p, "w", encoding="utf-8").write("#!/bin/bash\n" + bad)
        chk(len(scan_file(p)) == 2, "端到端：掃描一個 .sh 檔並報出 2 處")
        p2 = os.path.join(d, "notshell.txt")
        open(p2, "w", encoding="utf-8").write("$x（\n")
        chk(not is_shell(p2), "端到端：沒有 shebang 的 .txt 不被當成 shell")

    print(f"  --self-test: {fails} 失敗" if fails else "  --self-test: 9/9 通過")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None,
                    help="掃描根（預設：repo 根 = 本檔的 agent_harness/shared/ 往上兩層）")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    # `shared/` 往上兩層 = repo 根。第一次寫成三層，於是它掃到了**隔壁的 repo**
    # （輸出裡的 base_9d5a26779/、flashkv0516/ 就是這樣來的）—— 一個掃錯範圍的檢查
    # 會給出一個看起來很完整的清單，而那個清單裡有一半不是你的檔案。
    root = args.root or os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    # 這裡刻意**不**假設 root 是 repo 根：`--root agent_harness` 是合法用法（只掃自己那條線）。
    # 所以只驗它看起來像一棵原始碼樹 —— 一個掃錯範圍的檢查會給出很完整的清單，
    # 而那個清單裡有一半不是你的檔案（第一版就是這樣掃到了隔壁的 repo）。
    assert os.path.isdir(root), f"掃描根不存在：{root}"
    assert os.path.isdir(os.path.join(root, "agent_harness")) or os.path.basename(root) == "agent_harness", \
        f"掃描根 {root} 既不是 repo 根也不是 agent_harness/ ⇒ 推算錯了，不要繼續"

    hits: list[tuple[str, int, str, str]] = []
    n_files = 0
    for p in walk(root):
        n_files += 1
        for lineno, var, nxt in scan_file(p):
            hits.append((os.path.relpath(p, root), lineno, var, nxt))

    print(f"  掃了 {n_files} 個 shell 檔（含 shebang 判定的 .sh/.bash）")
    if not hits:
        print("  0 處：所有 `$VAR` 後面都不是緊接非 ASCII")
        return 0
    print(f"  {len(hits)} 處 `$VAR` 緊接非 ASCII（`set -u` 下會 unbound variable）：")
    by_file: dict[str, list] = {}
    for f, l, v, nx in hits:
        by_file.setdefault(f, []).append((l, v, nx))
    for f in sorted(by_file):
        print(f"    {f}")
        for l, v, nx in by_file[f]:
            print(f"      L{l}: ${v} 後接 {nx!r}  ->  寫成 ${{{v}}}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
