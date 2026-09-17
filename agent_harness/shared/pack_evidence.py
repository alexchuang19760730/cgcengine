#!/usr/bin/env python3
"""log 抽樣 → evidence/<id>.txt.zst ＋ sha256 索引（PLAN §8 的「log 政策落地」）。

WHAT IT IS FOR
--------------
`Backup/cgc_logs` is the raw evidence for every number in this project, and it is deliberately NOT
tracked by git (PLAN §8 納管策略). What IS meant to be kept is a **filtered extract** small enough
to live in the repo, with a hash that points back at the untracked original -- so a reader can tell
that the extract was not edited, and can go find the original if they have it.

THE FILTER (PLAN §8, kept verbatim so the two cannot drift)
------------------------------------------------------------
    only the lines matching  CGC-MMID-ASSERT|CGC-S1|CGC-HOOK|EXPECT-|CGC-GPUTIME|CGC-DECPROF|## SPLIT
    ±8 lines of context, plus the first 40 lines of the file (the header holds the invocation)
    capped at 256 KB per episode

WHAT THIS TOOL ADDS BEYOND THE SPEC, AND WHY
---------------------------------------------
* **Truncation is printed into the pack, not just recorded beside it.** A pack that silently stops
  at 256 KB is a partial record that reads exactly like a complete one. The marker states how many
  bytes and how many matching windows were dropped.
* **The index records the selection parameters.** `hits` and `kept_lines` depend on HEAD/CONTEXT/CAP,
  so a pack without them cannot be compared to a pack made with different ones.
* **`--dry-run` reports the projected size without writing anything**, because the spec's own
  acceptance criterion is a size ("產物總量 < 20 MB") and it is cheaper to find out first.

Usage:
    python3 agent_harness/shared/pack_evidence.py --dry-run
    python3 agent_harness/shared/pack_evidence.py --limit 50 --out /tmp/evidence_probe
    python3 agent_harness/shared/pack_evidence.py                  # the real thing
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

sys.path.insert(0, str(HERE))
from sanitize import sanitize_text  # noqa: E402

DEFAULT_SRC = REPO / "Backup" / "cgc_logs"
DEFAULT_OUT = HERE / "evidence"
PATTERNS = ("CGC-MMID-ASSERT", "CGC-S1", "CGC-HOOK", "EXPECT-", "CGC-GPUTIME", "CGC-DECPROF", "## SPLIT")
HEAD_LINES = 40
CONTEXT = 8
CAP_BYTES = 256 * 1024


def episode_id(rel: str) -> str:
    """A flat, reversible name: 'ab/x.log' -> 'ab~x.log' (one file per source log)."""
    return rel.replace("/", "~")


def select(text: str, patterns: tuple[str, ...] = PATTERNS, head: int = HEAD_LINES,
           context: int = CONTEXT, cap: int = CAP_BYTES) -> tuple[str, dict]:
    """The filter. Returns (pack text, stats). Truncation is IN the text (see module docstring)."""
    lines = text.splitlines()
    keep = set(range(min(head, len(lines))))
    hits = 0
    for i, line in enumerate(lines):
        if any(p in line for p in patterns):
            hits += 1
            keep.update(range(max(0, i - context), min(len(lines), i + context + 1)))

    out: list[str] = []
    prev: int | None = None
    for i in sorted(keep):
        if prev is not None and i != prev + 1:
            out.append(f"... [{i - prev - 1} line(s) not kept] ...")
        out.append(lines[i])
        prev = i
    body = "\n".join(out) + ("\n" if out else "")

    dropped_lines = dropped_bytes = 0
    if len(body.encode("utf-8")) > cap:
        cut = body.encode("utf-8")[:cap].decode("utf-8", errors="ignore")
        dropped_bytes = len(body.encode("utf-8")) - len(cut.encode("utf-8"))
        dropped_lines = cut.count("\n") and (body.count("\n") - cut.count("\n")) or 0
        body = (cut.rstrip("\n")
                + f"\n... [TRUNCATED at {cap} B: {dropped_bytes} B / ~{dropped_lines} further lines "
                  f"of this episode are NOT in this pack] ...\n")

    return body, {"source_lines": len(lines), "hit_lines": hits, "kept_lines": len(keep),
                  "kept_bytes": len(body.encode("utf-8")), "truncated": bool(dropped_bytes),
                  "dropped_bytes": dropped_bytes}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compress(text: str, dest: Path, zstd: str, level: int) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([zstd, "-q", "-f", f"-{level}", "-o", str(dest)],
                       input=text.encode("utf-8"), capture_output=True)
    if r.returncode != 0:
        raise SystemExit(f"zstd failed on {dest}: {r.stderr.decode('utf-8', 'replace')[:300]}")
    return dest.stat().st_size


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=0, help="only the first N source files (0 = all)")
    ap.add_argument("--level", type=int, default=3, help="zstd level; the measured size is only "
                                                        "quotable together with this number")
    ap.add_argument("--dry-run", action="store_true", help="report sizes, write nothing")
    ap.add_argument("--zstd", default=None, help="path to the zstd binary")
    args = ap.parse_args()

    zstd = args.zstd or shutil.which("zstd")
    if not zstd and not args.dry_run:
        raise SystemExit("no zstd binary on PATH and no --zstd given (packs are .txt.zst by spec);\n"
                         "    --dry-run still works and reports the uncompressed size")
    if not args.src.is_dir():
        raise SystemExit(f"no source directory at {args.src}")

    files = [f for f in sorted(args.src.rglob("*")) if f.is_file()]
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"{args.src} holds no files -- refusing to report a total of zero as success")

    rows, src_bytes, pack_bytes, kept_raw = [], 0, 0, 0
    n_trunc = n_hitless = 0
    for f in files:
        rel = str(f.relative_to(args.src))
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            print(f"  [warn] unreadable, skipped: {rel} ({e})", file=sys.stderr)
            continue
        pack, st = select(text)
        pack = sanitize_text(pack, repo_root=str(REPO))
        src_bytes += f.stat().st_size
        kept_raw += st["kept_bytes"]
        if st["hit_lines"] == 0:
            n_hitless += 1
        if st["truncated"]:
            n_trunc += 1
        rows.append({
            "episode_id": episode_id(rel),
            "source": str(f.relative_to(REPO)),
            "source_bytes": f.stat().st_size,
            "source_sha256": sha256_file(f),
            "pack_bytes": None, "pack_sha256": None,
            "selection": {"patterns": list(PATTERNS), "head_lines": HEAD_LINES,
                          "context": CONTEXT, "cap_bytes": CAP_BYTES, "zstd_level": args.level},
            **{k: st[k] for k in ("source_lines", "hit_lines", "kept_lines", "kept_bytes",
                                  "truncated", "dropped_bytes")},
        })
        if args.dry_run:
            continue
        # ---- 內容尋址：同一份 pack 只寫一次 ------------------------------------------------
        # 為什麼：實測 2646 個 pack 裡有 **195 組 pack_sha256 重複**（涉及 462 個檔），
        # 最大的一組是 28 個位元組完全相同的 `*.ips_before`。那些重複**對版控不存在** ——
        # git 的 blob 就是內容尋址的，28 個相同的檔它只存一份。所以舊寫法印出的「總量」
        # 比版控實際增加的量多了 4.86 MB，而**那個差額比到 20 MB 門檻的餘裕還大**。
        #
        # 用 pack 的**原文**雜湊當鍵（不是壓縮後）：這樣命名不依賴 zstd 版本的決定性，
        # 而 pack_sha256 仍然記壓縮後的位元組（那才是磁碟上的東西）。
        key = hashlib.sha256(pack.encode("utf-8")).hexdigest()[:16]
        dest = args.out / f"{key}.txt.zst"
        reused = dest.exists()
        if reused:
            n = dest.stat().st_size
        else:
            n = compress(pack, dest, zstd, args.level)
        rows[-1]["pack_bytes"] = n
        rows[-1]["pack_sha256"] = sha256_file(dest)
        rows[-1]["pack_reused"] = reused
        rows[-1]["pack"] = str(dest.relative_to(REPO)) if dest.is_relative_to(REPO) else str(dest)
        pack_bytes += n

    hitless = [r["episode_id"] for r in rows if r["hit_lines"] == 0]
    print(f"  source      : {len(rows)} file(s) from {args.src.relative_to(REPO) if args.src.is_relative_to(REPO) else args.src}")
    print(f"  source size : {src_bytes / 1048576:.1f} MB (raw, NOT tracked by git)")
    print(f"  selected    : {kept_raw / 1048576:.1f} MB before compression (head 40 + ±8 ctx, cap "
          f"{CAP_BYTES // 1024} KB/episode)")
    print(f"  truncated   : {n_trunc} episode(s) hit the {CAP_BYTES // 1024} KB cap")
    print(f"  no hits     : {len(hitless)} episode(s) matched no pattern (they still get a header-only pack)")
    if args.dry_run:
        print(f"  pack size   : not measured (--dry-run); rerun without it to get the compressed total")
        print(f"  PLAN §8 的驗收是「產物總量 < 20 MB」—— 這個數字要真的壓一次才知道")
        return 0

    total = pack_bytes / 1048576
    (args.out / "INDEX.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    # ---------------------------------------------------------------------------
    # 「總量」的定義 —— 這一節是這個工具存在爭議的地方，所以要明說
    # ---------------------------------------------------------------------------
    # 兩種合理的量法給出不同的答案，而 20 MB 的門檻正好夾在中間（2026-09-17 實測）：
    #   · 檔案大小總和（唯一 blob）   16.67 MB  -> PASS     ← git 存的量（git 只存內容一次）
    #   · 實佔磁碟 st_blocks*512     20.72 MB  -> FAIL     ← 2646 個小檔在 4 KB 區塊上的對齊浪費
    # 挑對自己有利的那個定義是錯的。**這裡做的是把歧義移除**：內容尋址之後重複不再寫入，
    # 於是兩種量法都過關，定義之爭就不需要贏。
    blobs = sorted({r["pack"] for r in rows if r.get("pack")})
    apparent = allocated = 0
    for b in blobs:
        p = Path(b) if os.path.isabs(b) else REPO / b
        try:
            st = p.stat()
        except OSError:
            continue
        apparent += st.st_size
        allocated += st.st_blocks * 512
    dup_saved = pack_bytes - apparent
    n_reused = sum(1 for r in rows if r.get("pack_reused"))

    print(f"  pack size   : {total:.1f} MB summed over rows, in {len(blobs)} unique blob(s) "
          f"-> {args.out}")
    print(f"  唯一 blob 的內容：{n_reused} 個 row 重用既有 blob"
          f"（{dup_saved / 1048576:.2f} MB 的重複沒有被寫第二次）")
    print()
    print(f"  ★ PLAN §8「產物總量 < 20 MB」的兩種量法，以及本工具採用的那一種：")
    print(f"      檔案大小總和 (唯一 blob)  {apparent / 1048576:6.2f} MB   "
          f"{'PASS' if apparent / 1048576 < 20 else 'FAIL'}   ← 採用：git 只存內容一次，"
          f"這是版控實際增加的量")
    print(f"      實佔磁碟  (st_blocks)     {allocated / 1048576:6.2f} MB   "
          f"{'PASS' if allocated / 1048576 < 20 else 'FAIL'}   ← 不採用：4 KB 區塊對齊是"
          f"檔案系統的性質，不是這個產物的性質")
    worst = max(apparent, allocated) / 1048576
    verdict = "PASS" if worst < 20 else "FAIL"
    print(f"  ★ 結論（以**較嚴格**的那個量法）: {verdict}   {worst:.2f} MB")
    if hitless:
        print(f"    (無命中的 {len(hitless)} 檔只帶檔頭。★ 不要為了縮小而濾掉它們："
              f"636／883 的來源 <= 4 KB，整個檔就是內容，濾掉等於丟資料)")
    return 0 if worst < 20 else 1


if __name__ == "__main__":
    raise SystemExit(main())
