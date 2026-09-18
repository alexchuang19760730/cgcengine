#!/usr/bin/env python3
"""Import dated snapshots of .workbuddy/memory + ~/.workbuddy/skills into agent_harness/.

Why: .workbuddy/ is gitignored, so the host-written memory and the skills never reach
version control and never reach the Windows periodic pusher (auto_git_push.ps1 watches
agent_harness/). This copies them in as *snapshots* with an explicit banner.

NOT authoritative: the originals stay where the host writes them. This script records
provenance (sha256/size/mtime) so drift is visible instead of silent.

MODES (2026-09-18)
------------------
  (no flag)    匯入：刷新快照 + 兩份 SNAPSHOT.jsonl（會寫檔）
  --check      只驗不寫：有漂移就 exit 1，並逐項印出是哪一種漂移
  --dry-run    顯示「會匯入什麼、會寫哪些檔」，不寫任何檔
  --self-test  黑箱自測：暫時目錄 + 陰性對照，印 N/N

WHY --check EXISTS (2026-09-18)
-------------------------------
在它之前，這份快照「過期」是**不可觀測**的：兩個既有的 `--check`
（`build_memory_index.py --check` / `index_assets.py --check`）都只驗原檔↔索引，**不驗快照**
——`CONVENTIONS.md` D6 的 2026-09-16 修訂逐字記著這件事。後果不是抽象的：
`auto_git_push.ps1` 會**忠實地把一份過期快照推上去**，而沒有任何東西會出聲。

三種漂移要分開，因為處置不同（`--check` 逐項標示）：

  A. **stale**      原檔的 sha256/bytes 與記錄不符 ⇒ 快照落後（正常，重跑匯入即可）
  B. **new/removed** 探索到的集合與記錄不符 ⇒ 多了或少了檔案（新增 skill／新的一天）
  C. **hand-edited** 快照副本的內容與「用它自己記錄的 snapshot_date 重算」不符
                     ⇒ 有人手改了副本。這**不是**落後，是違規：
                     `CONVENTIONS.md` 與兩份 README 都寫明「要改就改原檔」，
                     而匯入會**靜默蓋掉**手改的內容 ⇒ 必須先看過再重跑。

WHY --check 不接進 pre-commit hook（刻意的，不是遺漏）
-----------------------------------------------------
`.workbuddy/memory/*.md` 包含**每日 append-only 日誌**，而那是**多個 session 共寫**的
（本 repo 常態）。任何 commit 前都可能有人在上一分鐘剛 append 一行 ⇒ 硬閘門會
**永遠紅**。`CONVENTIONS.md:927-928` 對同一個形狀已有判決：
「每日日誌當天必變，永遠紅的閘門等於沒有閘門」。
⇒ 這個檢查走**兩條會真的發聲的路**：(1) 收尾序列（`cgc-commit-gate` §6.3）；
(2) 定時自動化（每日跑 `--check`，漂移就刷新＋重生索引＋commit）。

WHY THIS FILE LIVES IN agent_harness/scripts/ AND NOT IN Backup/  (2026-09-16)
-----------------------------------------------------------------------------
它原本住 `Backup/`，而 `Backup/` 被 `.gitignore:396` 排除。兩個後果，第二個才是搬家理由：

  1. 三個輸入是**硬編碼清單**（`SKILL_NAMES`／`MEM_FILES`）。新增 skill、或**單純過了一天**，
     產生的是一個**安靜的遺漏**：不是報錯，而是「它不在 `SNAPSHOT.jsonl` 裡」
     —— 那與「那個檔案不存在」**同形**。
  2. 住在 `Backup/` 意味著 (1) 的修法**也提交不了**。fresh clone **連匯入器都沒有**。
     所以「把名字補進清單」只修一台機器上的症狀，其他機器什麼都沒變。

兩個清單現在都是**推導**的（glob），`SNAP_DATE` 也是推導的（今天），`REPO` 由本檔位置推導。
重點不是整潔，而是**失效模態是「缺席」，而缺席讀起來像「沒什麼好看的」**。
一個會印出「它發現了什麼」的探索步驟讓缺席變得可見。見 lesson `eng-bound-0004`。

  python3 agent_harness/scripts/import_harness_snapshot.py            # 匯入
  python3 agent_harness/scripts/import_harness_snapshot.py --check    # 只驗
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import date, datetime
from pathlib import Path

# agent_harness/scripts/<this file> -> repo root. Derived, not written down: a hardcoded
# absolute path is how this script would break on the second machine it ever runs on.
# PA_SNAP_REPO / PA_SNAP_HOME 只為 --self-test 而存在（把兩個根重導到暫時目錄），
# 與本 repo 其他工具的「為自測而存在的旋鈕」同一個形狀。
REPO = Path(os.environ.get("PA_SNAP_REPO") or Path(__file__).resolve().parents[2]).resolve()
HOME = Path(os.environ.get("PA_SNAP_HOME") or Path.home()).resolve()
SELF_REL = "agent_harness/scripts/import_harness_snapshot.py"
# Derived, not written down: a hardcoded date makes a run on the NEXT day record the wrong
# snapshot_date, and the whole point of this file is that provenance is trustworthy.
SNAP_DATE = date.today().isoformat()

MEM_SRC = REPO / ".workbuddy" / "memory"
SKILL_SRC = HOME / ".workbuddy" / "skills"

MEM_DST = REPO / "agent_harness" / "memory"
SKILL_DST = REPO / "agent_harness" / "skills"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def discover(strict: bool = True) -> tuple[list[str], list[str]]:
    """The two inputs, discovered rather than listed.

    `MEMORY.md` is kept first because it is the canonical cross-day file; the daily logs follow
    in name order (so ISO-dated names sort chronologically). Any other .md dropped into
    .workbuddy/memory/ is imported too -- `index_assets.py` makes the same choice for the same
    reason ("so a new day is not invisible until somebody remembers to add a row").

    ★ `strict=False` 只給 `--check` 用，而這是**自測抓出來的**（B' 那一格原本 rc=1，
      但輸出裡**沒有** `[removed]`）：匯入路徑的空清單守衛會在「最後一個 skill 被刪掉」時
      搶先 raise ⇒ 那個可診斷的狀態被報成一個 SystemExit。
      守衛的正當理由是「不要寫出一份空快照」——那是**寫入**的風險；`--check` 不寫檔，
      所以它應該把那份狀態**報出來**，而不是拒跑。
    """
    if not MEM_SRC.is_dir():
        raise SystemExit(f"no memory source directory: {MEM_SRC}")
    mem = [p.name for p in sorted(MEM_SRC.glob("*.md")) if p.name != "MEMORY.md"]
    if (MEM_SRC / "MEMORY.md").exists():
        mem = ["MEMORY.md"] + mem

    if not SKILL_SRC.is_dir():
        raise SystemExit(f"no skill source directory: {SKILL_SRC}")
    skills = sorted(p.parent.name for p in SKILL_SRC.glob("*/SKILL.md"))

    # Absence must not be silent. An empty result here would write an EMPTY SNAPSHOT.jsonl,
    # and an empty index and a broken discovery step look identical from the outside.
    if strict:
        if not mem:
            raise SystemExit(f"discovered 0 memory files under {MEM_SRC} -- refusing to write an empty snapshot")
        if not skills:
            raise SystemExit(f"discovered 0 skills under {SKILL_SRC}/*/SKILL.md -- refusing to write an empty snapshot")
    return mem, skills


def banner(authority: str, extra: str, snap_date: str | None = None) -> str:
    return (
        "> **這是快照，不是權威副本。**\n"
        f"> 權威位置：`{authority}`（由 host 持續寫入）。\n"
        f"> 本檔於 {snap_date or SNAP_DATE} 由 `{SELF_REL}` 複製進 repo，唯一目的是讓 `agent_harness/`\n"
        "> 底下的內容能被 `agent_harness/scripts/auto_git_push.ps1` 定時推送；原檔改了這裡**不會**自動跟上。\n"
        f"> {extra}\n"
        "\n"
    )


def insert_banner(text: str, ban: str) -> str:
    """Insert after YAML frontmatter if present, else after the leading H1, else on top."""
    if text.startswith("---\n"):
        end = text.find("\n---\n", 3)
        if end != -1:
            cut = end + len("\n---\n")
            return text[:cut] + "\n" + ban + text[cut:].lstrip("\n")
    lines = text.split("\n", 1)
    if lines[0].startswith("# "):
        rest = lines[1] if len(lines) > 1 else ""
        return lines[0] + "\n\n" + ban + rest.lstrip("\n")
    return ban + text


def expected_text(src: Path, authority: str, extra: str, snap_date: str) -> str:
    """What the dated copy SHOULD contain.

    ★ `snap_date` 是參數而不是 `SNAP_DATE`：`--check` 必須用**記錄裡的那個日期**重算，
    否則每逢跨日，每一份快照都會被誤報成「被手改過」——那正是本 repo 反覆踩到的
    「mtime 造成的假漂移」（`cgc-commit-gate` §3.1 的簽名）。
    """
    return insert_banner(src.read_text(encoding="utf-8"), banner(authority, extra, snap_date))


def plan(mem_files: list[str], skill_names: list[str]) -> dict[str, tuple[Path, Path, str, str]]:
    """snapshot_rel -> (src, dst, authority, extra)。單一真相來源：匯入與檢查共用。"""
    out: dict[str, tuple[Path, Path, str, str]] = {}
    for name in mem_files:
        out[f"agent_harness/memory/{name}"] = (
            MEM_SRC / name,
            MEM_DST / name,
            f".workbuddy/memory/{name}",
            "索引與漂移檢查見 `agent_harness/engine_loop/memory/INDEX.jsonl`。",
        )
    for name in skill_names:
        out[f"agent_harness/skills/{name}/SKILL.md"] = (
            SKILL_SRC / name / "SKILL.md",
            SKILL_DST / name / "SKILL.md",
            f"~/.workbuddy/skills/{name}/SKILL.md",
            f"要改 skill 請改原檔，再重跑 `python3 {SELF_REL}`。",
        )
    return out


def load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def declared_in_readme(path: Path) -> set[tuple[str, str]] | None:
    """兩份 README 的表格裡**宣告**的成員；`None` = 找不到那種表格。

    ★ 為什麼檢查要管 README：它們是「第三道防線」（`CONVENTIONS.md` D6 的反向紀錄），
      而它們的成員表是**手維運的清單，指向的卻是推導出來的集合** —— 那正是這個檔案
      在 2026-09-16 修掉的那個毛病（舊版匯入器的硬編碼清單），只是換了個地方。
      實測（2026-09-18）：`agent_harness/memory/README.md` 宣告 3 個檔（實際 8）、
      `agent_harness/skills/README.md` 宣告 4 個 skill（實際 5）—— **兩張表都在說假話，
      而沒有任何東西會出聲**。與其把它們改成對的然後等它再腐爛，不如讓腐爛變紅燈。

      判準是第 2 欄（權威位置）：`…/.workbuddy/memory/<name>.md` → ("mem", name)、
      `…/.workbuddy/skills/<name>/SKILL.md` → ("skill", name)。找不到表格時回 `None`，
      而呼叫端**把它算成一項失敗並明說「未檢查」** —— 理由：**一個可以把輸入拿掉就變綠的檢查
      不是檢查**。若「沒有表格」只印提示而不影響 exit code，那刪掉表格就是一條靜音的繞道，
      正是這個 repo 反覆記下的失效模態（`cgc-commit-gate` 的 B7）。
    """
    if not path.exists():
        return None
    out: set[tuple[str, str]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip().strip("`") for c in line.strip("|").split("|")]
        if len(cells) < 2:
            continue
        m = re.search(r"\.workbuddy/memory/([^/`]+\.md)$", cells[1])
        if m:
            out.add(("mem", m.group(1)))
            continue
        m = re.search(r"\.workbuddy/skills/([^/`]+)/SKILL\.md$", cells[1])
        if m:
            out.add(("skill", m.group(1)))
    return out or None


def record_for(dst_rel: str, src: Path, authority: str) -> dict:
    st = src.stat()
    return {
        "snapshot": dst_rel,
        "source": str(src),
        "source_sha256": sha256(src),
        "source_bytes": st.st_size,
        "source_mtime": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%dT%H:%M:%S"),
        "snapshot_date": SNAP_DATE,
        "authority": authority,
    }


def write_manifest(path: Path, rows: list) -> None:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )


def do_import(dry: bool = False) -> int:
    mem_files, skill_names = discover()
    print(f"discovered {len(mem_files)} memory file(s), {len(skill_names)} skill(s):")
    for n in mem_files:
        print(f"  mem    {n}")
    for n in skill_names:
        print(f"  skill  {n}")

    todo = plan(mem_files, skill_names)
    mem_records: list = []
    skill_records: list = []

    for dst_rel, (src, dst, authority, extra) in todo.items():
        if not src.exists():
            raise SystemExit(f"missing source: {src}")
        rec = record_for(dst_rel, src, authority)
        (skill_records if dst_rel.startswith("agent_harness/skills/") else mem_records).append(rec)
        if dry:
            print(f"  would write {dst_rel}  <-  {src}  ({rec['source_bytes']} B)")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(expected_text(src, authority, extra, SNAP_DATE), encoding="utf-8")

    for path, rows in ((MEM_DST / "SNAPSHOT.jsonl", mem_records), (SKILL_DST / "SNAPSHOT.jsonl", skill_records)):
        if dry:
            print(f"  would write {path.relative_to(REPO)}  ({len(rows)} row(s))")
        else:
            write_manifest(path, rows)

    if dry:
        print("--dry-run: 沒有寫入任何檔案。")
        return 0

    print(f"imported {len(mem_records) + len(skill_records)} file(s) -> agent_harness/")
    for r in mem_records + skill_records:
        print(f"  {r['snapshot']}  <-  {r['source']}  ({r['source_bytes']} B)")
    return 0


def do_check() -> int:
    """只驗不寫。回傳 0 = 快照與原檔一致；1 = 有漂移（逐項印出）。

    ★ `strict=False`：檢查路徑不拒跑空的探索結果（見 `discover` 的說明）。
    """
    mem_files, skill_names = discover(strict=False)
    todo = plan(mem_files, skill_names)
    records = load_manifest(MEM_DST / "SNAPSHOT.jsonl") + load_manifest(SKILL_DST / "SNAPSHOT.jsonl")
    seen: dict[str, dict] = {}
    for r in records:
        seen[r["snapshot"]] = r

    stale: list[str] = []
    missing: list[str] = []      # 原檔在、記錄不在
    orphan: list[str] = []       # 記錄在、原檔不在或已不是輸入
    edited: list[str] = []
    gone: list[str] = []

    for dst_rel, (src, dst, authority, extra) in sorted(todo.items()):
        rec = seen.get(dst_rel)
        if rec is None:
            missing.append(dst_rel)
            continue
        if not src.exists():
            gone.append(dst_rel)
            continue
        if sha256(src) != rec["source_sha256"] or src.stat().st_size != rec["source_bytes"]:
            stale.append(dst_rel)
        if not dst.exists():
            gone.append(dst_rel)
            continue
        # 用**記錄裡的那個日期**重算 ⇒ 跨日不會誤報（見 expected_text 的說明）
        want = expected_text(src, authority, extra, rec.get("snapshot_date", SNAP_DATE))
        if dst.read_text(encoding="utf-8") != want:
            # 兩種可能：來源已變（上面已列 stale），或副本被手改。只有後者要單獨點名。
            if dst_rel not in stale:
                edited.append(dst_rel)

    for dst_rel in sorted(set(seen) - set(todo)):
        orphan.append(dst_rel)

    # README 的成員表：手維運、指向推導集合 ⇒ 會腐爛。
    # ★ 「找不到表格」**算一項失敗**（不是提示）：能靠拿掉輸入變綠的檢查不是檢查。
    #   自測的 F 那一格就是釘住這件事（第一版把它寫成只提示 ⇒ 那個 case 假失敗，
    #   而「假失敗」正好暴露了那個設計是錯的）。
    readme: list[str] = []
    for table_path, table_rel, kind, want_names in (
        (MEM_DST / "README.md", "agent_harness/memory/README.md", "mem", mem_files),
        (SKILL_DST / "README.md", "agent_harness/skills/README.md", "skill", skill_names),
    ):
        got = declared_in_readme(table_path)
        if got is None:
            readme.append(f"{table_rel}: 找不到成員表 ⇒ **未檢查**（不是通過）")
            continue
        have = {n for k, n in got if k == kind}
        want = set(want_names)
        for n in sorted(have - want):
            readme.append(f"{table_rel}: 宣告了但不存在或不再是輸入: {n}")
        for n in sorted(want - have):
            readme.append(f"{table_rel}: 存在但未宣告: {n}")

    problems = len(stale) + len(missing) + len(orphan) + len(edited) + len(gone) + len(readme)
    dates = sorted({r.get("snapshot_date", "?") for r in records})
    print(f"  {len(mem_files)} memory source(s), {len(skill_names)} skill source(s); "
          f"{len(records)} 筆記錄（snapshot_date {dates[0] if dates else '—'}…{dates[-1] if dates else '—'}）")
    if missing:
        print(f"  [new]         {len(missing)} 筆原檔沒有被記錄（匯入會補上）")
        for p in missing:
            print(f"      {p}")
    if stale:
        print(f"  [stale]       {len(stale)} 筆的原檔已變（快照落後）")
        for p in stale:
            print(f"      {p}")
    if orphan:
        print(f"  [removed]     {len(orphan)} 筆記錄已不對應任何輸入（skill 被刪／改名）")
        for p in orphan:
            print(f"      {p}")
    if edited:
        print(f"  [hand-edited] {len(edited)} 筆副本與『用它自己記錄的日期重算』不符 —— 有人手改了副本")
        for p in edited:
            print(f"      {p}   （重跑匯入會蓋掉它；先看過再跑）")
    if gone:
        print(f"  [missing]     {len(gone)} 筆原檔或副本不存在")
        for p in gone:
            print(f"      {p}")
    for line in readme:
        print(f"  [readme-table] {line}")
    if problems == 0:
        print("  -> OK: 快照與原檔一致（provenance 逐筆相符、集合相符、副本未被手改、README 成員表相符）")
        return 0
    print(f"  -> DRIFT: {problems} 項。重跑 `python3 {SELF_REL}` 再重生索引"
          f"（先 build_memory_index.py、後 index_assets.py）。")
    return 1


# ---------------------------------------------------------------------------------
# --self-test：黑箱。用暫時目錄當 repo/home，並跑真正的子行程（不是呼叫內部函式）。
#   陰性對照是重點：每一種漂移都必須**真的被偵測到**，而且「乾淨」那一格必須回 0
#   （一個永遠回 1 的檢查會讓所有陽性對照通過 —— `eng-gate-0040`）。
# ---------------------------------------------------------------------------------
def do_self_test() -> int:
    me = str(Path(__file__).resolve())
    tmp = Path(tempfile.mkdtemp(prefix="pa_snap_selftest_"))
    root = tmp / "repo"
    home = tmp / "home"
    (root / ".workbuddy" / "memory").mkdir(parents=True)
    (home / ".workbuddy" / "skills" / "skill-a").mkdir(parents=True)
    (root / ".workbuddy" / "memory" / "MEMORY.md").write_text("# MEM\n\ncanonical\n", encoding="utf-8")
    (root / ".workbuddy" / "memory" / "2026-01-01.md").write_text("# day one\n\nbody\n", encoding="utf-8")
    (home / ".workbuddy" / "skills" / "skill-a" / "SKILL.md").write_text(
        "---\nname: skill-a\n---\n\n# skill a\n\nbody\n", encoding="utf-8")
    # 兩份 README 是**手寫的 repo 檔**（不是快照），所以 fixture 要自己造。
    # 成員表宣告得**正確**，好讓「乾淨 ⇒ rc=0」那一格同時證明表格比對會過。
    (root / "agent_harness" / "memory").mkdir(parents=True)
    (root / "agent_harness" / "skills").mkdir(parents=True)
    (root / "agent_harness" / "memory" / "README.md").write_text(
        "# mem snapshot\n\n| 檔案 | 權威位置 |\n| --- | --- |\n"
        "| `MEMORY.md` | `.workbuddy/memory/MEMORY.md` |\n"
        "| `2026-01-01.md` | `.workbuddy/memory/2026-01-01.md` |\n", encoding="utf-8")
    skill_readme = root / "agent_harness" / "skills" / "README.md"
    skill_readme.write_text(
        "# skill snapshot\n\n| skill | 權威位置 | 什麼時候用 |\n| --- | --- | --- |\n"
        "| `skill-a` | `~/.workbuddy/skills/skill-a/SKILL.md` | demo |\n", encoding="utf-8")

    env = dict(os.environ, PA_SNAP_REPO=str(root), PA_SNAP_HOME=str(home))
    results: list[tuple[str, bool, str]] = []

    def run(*args: str) -> tuple[int, str]:
        p = subprocess.run([sys.executable, me, *args], capture_output=True, text=True, env=env)
        return p.returncode, p.stdout + p.stderr

    def case(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, ok, detail))

    def sync_readmes() -> None:
        """把 fixture 的兩份 README 成員表重寫成與當前來源一致。

        存在理由：`--check` 除了比對快照，也**比對 README 的成員表**（那是手維運、
        指向推導集合 ⇒ 會腐爛的東西）。所以「乾淨」這個狀態現在包含「README 表也對」。
        每一格要隔離自己的變數，就必須在動完來源之後把表同步一次 ——
        否則第 5 格新增的那個檔案會讓後面每一格都因為 README 而紅（第一版就是這樣，
        造成 D／F 兩格假失敗）。
        """
        mem_names = sorted(p.name for p in (root / ".workbuddy" / "memory").glob("*.md"))
        (root / "agent_harness" / "memory" / "README.md").write_text(
            "# mem snapshot\n\n| 檔案 | 權威位置 |\n| --- | --- |\n"
            + "".join(f"| `{n}` | `.workbuddy/memory/{n}` |\n" for n in mem_names), encoding="utf-8")
        sk_names = sorted(p.parent.name for p in (home / ".workbuddy" / "skills").glob("*/SKILL.md"))
        skill_readme.write_text(
            "# skill snapshot\n\n| skill | 權威位置 | 什麼時候用 |\n| --- | --- | --- |\n"
            + "".join(f"| `{n}` | `~/.workbuddy/skills/{n}/SKILL.md` | demo |\n" for n in sk_names),
            encoding="utf-8")

    # 1) 匯入成功，且寫出預期的檔
    rc, out = run()
    mem_ok = (root / "agent_harness" / "memory" / "2026-01-01.md").exists()
    sk_ok = (root / "agent_harness" / "skills" / "skill-a" / "SKILL.md").exists()
    case("import rc=0 且產生副本", rc == 0 and mem_ok and sk_ok, f"rc={rc}")
    case("副本有 banner（插在 frontmatter 之後）",
         "這是快照，不是權威副本" in (root / "agent_harness" / "skills" / "skill-a" / "SKILL.md").read_text(encoding="utf-8"),
         "")

    # 2) 乾淨狀態必須回 0（陰性對照：不能永遠紅）
    rc, out = run("--check")
    case("--check 乾淨 ⇒ rc=0", rc == 0, f"rc={rc} out={out.strip()[:90]}")

    # 3) --check 不寫檔（判準：整個樹的內容雜湊不變）
    def tree_hash() -> str:
        h = hashlib.sha256()
        for p in sorted(root.rglob("*")):
            if p.is_file():
                h.update(str(p.relative_to(root)).encode())
                h.update(p.read_bytes())
        return h.hexdigest()

    before = tree_hash()
    run("--check")
    case("--check 不寫任何檔", tree_hash() == before, "")

    # 4) A. stale：原檔改了
    src_day = root / ".workbuddy" / "memory" / "2026-01-01.md"
    src_day.write_text("# day one\n\nbody CHANGED\n", encoding="utf-8")
    rc, out = run("--check")
    case("A 原檔變 ⇒ rc=1 且標 stale", rc == 1 and "[stale]" in out and "2026-01-01.md" in out, f"rc={rc}")

    # 5) B. new：多了一個原檔
    (root / ".workbuddy" / "memory" / "2026-01-02.md").write_text("# day two\n", encoding="utf-8")
    rc, out = run("--check")
    case("B 新增原檔 ⇒ rc=1 且標 new", rc == 1 and "[new]" in out and "2026-01-02.md" in out, f"rc={rc}")

    # 6) B'. removed：skill 被刪
    shutil.rmtree(home / ".workbuddy" / "skills" / "skill-a")
    rc, out = run("--check")
    case("B' 刪 skill ⇒ rc=1 且標 removed", rc == 1 and "[removed]" in out, f"rc={rc}")
    (home / ".workbuddy" / "skills" / "skill-a").mkdir(parents=True)
    (home / ".workbuddy" / "skills" / "skill-a" / "SKILL.md").write_text(
        "---\nname: skill-a\n---\n\n# skill a\n\nbody\n", encoding="utf-8")

    # 7) 重新匯入 ⇒ 回到乾淨（也是 4/5/6 的修法證明）。
    #    ★ 「乾淨」現在包含「README 成員表也對」，所以先同步它 —— 這一格因此同時證明
    #      README 表的比對是**活的**（不是永遠回 0 的裝飾）。
    sync_readmes()
    rc, _ = run()
    rc2, out2 = run("--check")
    case("重跑匯入（含修好 README 表）後回到 rc=0",
         rc == 0 and rc2 == 0, f"import rc={rc} check rc={rc2} :: {out2.strip()[-90:]}")

    # 8) C. hand-edited：手改副本（原檔沒動）
    copy = root / "agent_harness" / "memory" / "2026-01-01.md"
    copy.write_text(copy.read_text(encoding="utf-8") + "\n手動追加的一行\n", encoding="utf-8")
    rc, out = run("--check")
    case("C 手改副本 ⇒ rc=1 且標 hand-edited", rc == 1 and "[hand-edited]" in out, f"rc={rc}")

    # 9) 跨日不誤報：把記錄的 snapshot_date 改成昨天，副本的 banner 也照那個日期重算
    snap = root / "agent_harness" / "memory" / "SNAPSHOT.jsonl"
    rows = [json.loads(l) for l in snap.read_text(encoding="utf-8").splitlines() if l.strip()]
    run()  # 先洗回來
    changed = 0
    for r in rows:
        if r["snapshot"].endswith("2026-01-01.md"):
            r["snapshot_date"] = "2000-01-01"
            p = root / r["snapshot"]
            body = (root / ".workbuddy" / "memory" / "2026-01-01.md").read_text(encoding="utf-8")
            p.write_text(insert_banner(body, banner(r["authority"], "索引與漂移檢查見 `agent_harness/engine_loop/memory/INDEX.jsonl`。", "2000-01-01")),
                         encoding="utf-8")
            changed += 1
    snap.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    rc, out = run("--check")
    case("D 跨日不誤報（用記錄的日期重算）⇒ rc=0", rc == 0 and changed == 1, f"rc={rc} changed={changed}")

    # 9b) E. README 成員表腐爛（宣告了不存在的成員）⇒ 紅燈。
    #     這是真實世界的形狀：2026-09-18 實測 memory/README.md 宣告 3 檔（實際 8）、
    #     skills/README.md 宣告 4 個 skill（實際 5），而沒有任何檢查會出聲。
    skill_readme.write_text(skill_readme.read_text(encoding="utf-8")
                            + "| `skill-GONE` | `~/.workbuddy/skills/skill-GONE/SKILL.md` | 已刪 |\n",
                            encoding="utf-8")
    rc, out = run("--check")
    case("E README 宣告不存在的成員 ⇒ rc=1 且標 readme-table",
         rc == 1 and "[readme-table]" in out and "skill-GONE" in out, f"rc={rc}")

    # 9c) F. 找不到成員表 ⇒ **必須紅**，而且明說「未檢查」。
    #      ★ 這一格是設計決策的釘子：拿掉表格不能讓檢查變綠（否則刪表格就是一條靜音的繞道）。
    skill_readme.write_text("# skill snapshot\n\n（表格被拿掉了）\n", encoding="utf-8")
    rc, out = run("--check")
    case("F 拿掉成員表 ⇒ rc=1 且明說未檢查（不能靠刪輸入變綠）",
         rc == 1 and "未檢查" in out, f"rc={rc} out={out.strip()[-300:]}")
    skill_readme.write_text(
        "# skill snapshot\n\n| skill | 權威位置 | 什麼時候用 |\n| --- | --- | --- |\n"
        "| `skill-a` | `~/.workbuddy/skills/skill-a/SKILL.md` | demo |\n", encoding="utf-8")

    # 10) 空來源要拒跑（缺席不可靜默）—— ★ home 也要是空的，否則這個 fixture 根本沒測到空清單
    #     （第一版就是重用了上面那個「還有 skill-a」的 home，於是匯入成功、rc=0、這個 case 假失敗。）
    empty = tmp / "empty_repo"
    empty_home = tmp / "empty_home"
    (empty / ".workbuddy" / "memory").mkdir(parents=True)
    (empty / ".workbuddy" / "memory" / "MEMORY.md").write_text("# only canonical\n", encoding="utf-8")
    (empty_home / ".workbuddy" / "skills").mkdir(parents=True)
    p = subprocess.run([sys.executable, me], capture_output=True, text=True,
                       env=dict(os.environ, PA_SNAP_REPO=str(empty), PA_SNAP_HOME=str(empty_home)))
    case("空 skill 清單 ⇒ 匯入拒跑（rc!=0）",
         p.returncode != 0 and "refusing to write an empty snapshot" in (p.stdout + p.stderr),
         f"rc={p.returncode} out={(p.stdout + p.stderr).strip()[:90]}")
    p = subprocess.run([sys.executable, me, "--check"], capture_output=True, text=True,
                       env=dict(os.environ, PA_SNAP_REPO=str(empty), PA_SNAP_HOME=str(empty_home)))
    case("空來源 ⇒ --check 不拒跑、而是報 drift（rc=1）", p.returncode == 1, f"rc={p.returncode}")

    shutil.rmtree(tmp, ignore_errors=True)

    ok = sum(1 for _, o, _ in results if o)
    for name, o, detail in results:
        print(f"  {'PASS' if o else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not o else ""))
    print(f"  {ok}/{len(results)}")
    return 0 if ok == len(results) else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description="把 .workbuddy/memory 與 ~/.workbuddy/skills 的 dated 快照匯入 agent_harness/。",
        epilog="無參數 = 匯入（會寫檔）；--check 只驗不寫。")
    ap.add_argument("--check", action="store_true", help="只驗：有漂移就 exit 1（不寫任何檔）")
    ap.add_argument("--dry-run", action="store_true", help="顯示會做什麼，不寫任何檔")
    ap.add_argument("--self-test", action="store_true", help="黑箱自測（暫時目錄 + 陰性對照）")
    args = ap.parse_args()

    if args.self_test:
        return do_self_test()
    if args.check:
        return do_check()
    return do_import(dry=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
