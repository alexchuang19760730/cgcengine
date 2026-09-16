#!/usr/bin/env python3
"""Import dated snapshots of .workbuddy/memory + ~/.workbuddy/skills into agent_harness/.

Why: .workbuddy/ is gitignored, so the host-written memory and the skills never reach
version control and never reach the Windows periodic pusher (auto_git_push.ps1 watches
agent_harness/). This copies them in as *snapshots* with an explicit banner.

NOT authoritative: the originals stay where the host writes them. This script records
provenance (sha256/size/mtime) so drift is visible instead of silent.

WHY THIS FILE LIVES IN agent_harness/scripts/ AND NOT IN Backup/  (2026-09-16)
-----------------------------------------------------------------------------
It used to live in Backup/, which is gitignored (.gitignore:396). Two consequences, and the
second one is the reason it moved:

  1. Three of its inputs were HARDCODED LISTS -- `SKILL_NAMES` (which skills to import) and
     `MEM_FILES` (which memory files to import). Adding a skill, or simply waking up on a new
     day, produced a SILENT omission: not an error, just an entry that is absent from
     SNAPSHOT.jsonl -- which is indistinguishable from "that file does not exist".
  2. Being in Backup/ meant the fix for (1) could not be committed either. A fresh clone had
     NO IMPORTER AT ALL. So "add the missing name to the list" fixes the symptom on one machine
     and changes nothing on any other.

Both lists are now DERIVED (glob), SNAP_DATE is derived (today), and REPO is derived from this
file's own location. The point is not tidiness: it is that the failure mode was *absence*, and
absence reads as "nothing to see". A discovery step that prints what it found makes absence
visible. See agent_harness/skills/README.md and lesson eng-bound-0004.

  python3 agent_harness/scripts/import_harness_snapshot.py
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path

# agent_harness/scripts/<this file> -> repo root. Derived, not written down: a hardcoded
# absolute path is how this script would break on the second machine it ever runs on.
REPO = Path(__file__).resolve().parents[2]
HOME = Path.home()
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


def discover() -> tuple[list[str], list[str]]:
    """The two inputs, discovered rather than listed.

    `MEMORY.md` is kept first because it is the canonical cross-day file; the daily logs follow
    in name order (so ISO-dated names sort chronologically). Any other .md dropped into
    .workbuddy/memory/ is imported too -- `index_assets.py` makes the same choice for the same
    reason ("so a new day is not invisible until somebody remembers to add a row").
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
    if not mem:
        raise SystemExit(f"discovered 0 memory files under {MEM_SRC} -- refusing to write an empty snapshot")
    if not skills:
        raise SystemExit(f"discovered 0 skills under {SKILL_SRC}/*/SKILL.md -- refusing to write an empty snapshot")
    return mem, skills


def banner(authority: str, extra: str) -> str:
    return (
        "> **這是快照，不是權威副本。**\n"
        f"> 權威位置：`{authority}`（由 host 持續寫入）。\n"
        f"> 本檔於 {SNAP_DATE} 由 `{SELF_REL}` 複製進 repo，唯一目的是讓 `agent_harness/`\n"
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


records = []


def emit(src: Path, dst: Path, authority: str, extra: str, sink: list) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    body = src.read_text(encoding="utf-8")
    dst.write_text(insert_banner(body, banner(authority, extra)), encoding="utf-8")
    st = src.stat()
    sink.append(
        {
            "snapshot": str(dst.relative_to(REPO)),
            "source": str(src),
            "source_sha256": sha256(src),
            "source_bytes": st.st_size,
            "source_mtime": datetime.fromtimestamp(st.st_mtime).strftime(
                "%Y-%m-%dT%H:%M:%S"
            ),
            "snapshot_date": SNAP_DATE,
            "authority": authority,
        }
    )


def write_manifest(path: Path, rows: list) -> None:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )


def main() -> int:
    mem_files, skill_names = discover()
    print(f"discovered {len(mem_files)} memory file(s), {len(skill_names)} skill(s):")
    for n in mem_files:
        print(f"  mem    {n}")
    for n in skill_names:
        print(f"  skill  {n}")

    mem_records: list = []
    skill_records: list = []

    for name in mem_files:
        src = MEM_SRC / name
        if not src.exists():
            raise SystemExit(f"missing memory source: {src}")
        emit(
            src,
            MEM_DST / name,
            f".workbuddy/memory/{name}",
            "索引與漂移檢查見 `agent_harness/engine_loop/memory/INDEX.jsonl`。",
            mem_records,
        )

    for name in skill_names:
        src = SKILL_SRC / name / "SKILL.md"
        if not src.exists():
            raise SystemExit(f"missing skill source: {src}")
        emit(
            src,
            SKILL_DST / name / "SKILL.md",
            f"~/.workbuddy/skills/{name}/SKILL.md",
            f"要改 skill 請改原檔，再重跑 `python3 {SELF_REL}`。",
            skill_records,
        )

    write_manifest(MEM_DST / "SNAPSHOT.jsonl", mem_records)
    write_manifest(SKILL_DST / "SNAPSHOT.jsonl", skill_records)

    print(f"imported {len(mem_records) + len(skill_records)} file(s) -> agent_harness/")
    for r in mem_records + skill_records:
        print(f"  {r['snapshot']}  <-  {r['source']}  ({r['source_bytes']} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
