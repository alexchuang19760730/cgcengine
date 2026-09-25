#!/usr/bin/env python3
"""Whose bytes are in swap? `vmmap`'s SWAPPED column answers it per region, for a live engine.

Why this question is worth a tool. Tonight's artifact says the box's swap went 0 -> 5000 MiB during
one arm, and that when the engine exited, wired fell 9818 -> 1654 MiB while swap moved only
5024 -> 5000 MiB. Two readings follow, and they have opposite consequences:

  * the engine's working set was **resident**, so the ~5 GB belongs to other (still alive) processes
    => swap usage is not in our critical path, and the contract's "quotability is decoupled from
    swap" is the correct stance; a run's speed does not depend on it.
  * **or** some of the 8 GiB expert pool is among the victims, in which case those bytes come back
    from SSD *inside a miss* -- the hot path -- and swap is a direct decode cost.

A box-level total cannot tell these apart. `vmmap <pid>` can: every region carries its own SWAPPED
size, so the engine's number is not an inference. This probe samples it while the engine runs and
records both sides (engine swapped vs box swap used) so the ratio, not a story, is the artifact.

    python3 Backup/rerun/swap_owner_probe.py --pid <engine-pid> --seconds 120
    python3 Backup/rerun/swap_owner_probe.py --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

ROOT = "/Users/alexchuang/Documents/flashkv-devserver"
sys.path.insert(0, os.path.join(ROOT, "scripts/check"))
import memory_pressure as mp          # noqa: E402

KB_RE = re.compile(r"([0-9.]+)\s*([KMGT])?")


def _kb(tok: str) -> float | None:
    """A vmmap size cell, in KB.

    The unit is optional: below 1K vmmap prints bare bytes (`__CTF  824`), and it prints terabytes
    (`Memory Tag 255  1.3T`). Both were unreadable to the first version, which had `[KMG]` only and
    required a unit -- so whole rows went missing from the census and the residual looked like
    rounding. A cell with no digits at all still returns None, because `--` is not a zero.
    """
    m = KB_RE.fullmatch(tok.strip())
    if not m:
        return None
    v, unit = float(m.group(1)), m.group(2)
    return v * {"K": 1.0, "M": 1024.0, "G": 1048576.0, "T": 1073741824.0, None: 1.0 / 1024.0}[unit]


def _column_spans(sep: str) -> list[tuple[int, int]]:
    """Column bounds from vmmap's own `=====` ruler.

    The tool right-aligns its columns against that ruler, so it is the only place the geometry is
    stated. Region names contain spaces (`Activity Tracing`, `Memory Tag 255`), so `split()` cannot
    locate a column -- slicing by these spans can.
    """
    spans, start = [], None
    for i, ch in enumerate(sep):
        if ch == "=" and start is None:
            start = i
        elif ch != "=" and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(sep)))
    return spans


def _field_spans(hdr: str, ruler: str) -> list[tuple[int, int]]:
    """Numeric field bounds, in header order.

    Every header name is right-aligned to its own field's right edge, and so is every value -- that
    right edge is the boundary, and a field's left edge is the previous name's right edge. The left
    edge matters: `DIRTY`'s run on the ruler is 5 characters wide while real values are 6
    (`992.0M`), so taking the ruler's own spans as the fields clips the number to `98.0M` and pushes
    the `M` into SWAPPED. The one boundary the names cannot state is the first numeric column's left
    edge; that is where the first numeric column's ruler run starts, and everything left of it is
    the region-name column.
    """
    names = [(m.start(), m.end()) for m in re.finditer(r"\S+", hdr)]
    runs = _column_spans(ruler)
    if len(runs) != len(names) + 1:
        return []
    bounds = [runs[1][0]] + [end for _, end in names]
    return [(bounds[i], bounds[i + 1]) for i in range(len(names))]


def parse_vmmap(text: str) -> dict:
    """Engine-side swap, per region + total.

    Reads the Writable-regions summary line (where the anonymous pool lives) and the table's
    SWAPPED column. A row whose SWAPPED cell does not parse is counted in `unparsed`, never as 0 --
    an absence folded into a number is this project's oldest defect.
    """
    out = {"writable_total_kb": None, "writable_swapped_kb": None, "writable_resident_kb": None,
           "table_total_kb": None, "regions": [], "unparsed": 0}
    lines = text.splitlines()

    # The region-type table is stated across THREE lines, and that is not cosmetic. The first
    # version of this parser entered the table on the names line (`... SWAPPED ... REGION`) and then
    # left it again on the very next line, whose first two words are `REGION TYPE` -- so it returned
    # ZERO rows on every real process while reporting success, and the numbers it did report were
    # read out of the MALLOC ZONE table further down (`2347`, `see MALLOC ZONE table below`). Its
    # selftest passed because the fixture had been written from the parser's assumptions -- one
    # header line followed by rows -- instead of from real bytes. Measured on a live
    # `vmmap -summary` 2026-09-26; the selftest now checks its own geometry against the real one.
    head = None
    for i, line in enumerate(lines):
        if line.startswith("Writable regions:"):
            for key, pat in (("writable_total_kb", r"Total=([0-9.]+[KMG])"),
                             ("writable_resident_kb", r"resident=([0-9.]+[KMG])"),
                             ("writable_swapped_kb", r"swapped_out=([0-9.]+[KMG])")):
                m = re.search(pat, line)
                if m:
                    out[key] = _kb(m.group(1))
        elif head is None and "SWAPPED" in line and "REGION" in line \
                and "VIRTUAL ALLOCATION" not in line:
            head = i
    if head is None:
        return out

    sep = None
    for i in range(head + 1, min(head + 4, len(lines))):
        if lines[i].strip() and set(lines[i].strip()) <= set("= "):
            sep = i
            break
    if sep is None:
        return out
    num = _field_spans(lines[head], lines[sep])
    if not num:
        return out
    try:
        # Column indices come from the header, not from a hardcoded 3 or 4.
        col = {k: lines[head].split().index(k) for k in ("VIRTUAL", "RESIDENT", "DIRTY", "SWAPPED")}
    except ValueError:
        return out
    name_end = num[0][0]

    for line in lines[sep + 1:]:
        stripped = line.strip()
        if not stripped:
            break                              # end of the table (a blank line precedes MALLOC ZONE)
        if set(stripped) <= set("= "):
            continue                           # a second ruler closes the body; TOTAL follows it
        label = line[:name_end].strip()
        blocks = [line[a:b].strip() for a, b in num]
        vals = {k: (_kb(blocks[idx]) if idx < len(blocks) else None) for k, idx in col.items()}
        if label.startswith("TOTAL"):          # aggregates, not regions: TOTAL + `TOTAL, minus ...`
            if label == "TOTAL":
                # vmmap's own aggregate, kept so the rows can be reconciled against it: summing them
                # back to this row is what makes the slicing verifiable rather than plausible.
                out["table_total_kb"] = {f"{k.lower()}_kb": v for k, v in vals.items()}
            continue
        if not label or any(v is None for v in vals.values()):
            out["unparsed"] += 1               # a cell we could not read is counted, never 0
            continue
        out["regions"].append({"region": label[:80],
                               **{f"{k.lower()}_kb": v for k, v in vals.items()}})
    out["regions"].sort(key=lambda r: -r["swapped_kb"])
    return out


def sample(pid: int) -> dict:
    v = subprocess.run(["vmmap", "-summary", str(pid)], capture_output=True, text=True)
    if v.returncode != 0:
        return {"pid": pid, "error": (v.stderr or "vmmap failed").strip()[:160]}
    p = parse_vmmap(v.stdout)
    p.update({"pid": pid, "t": time.strftime("%H:%M:%S"),
              "swap_used_mb": mp.swap_used_mb()})
    return p


def selftest() -> int:
    bad = 0

    def c(name, ok, got=""):
        nonlocal bad
        if not ok:
            bad += 1
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'  -> ' + str(got) if got else ''}")

    # A fixture of the REAL geometry, byte-for-byte: the region-type table is stated across three
    # lines; names contain spaces; a note can sit past the COUNT column; a second table (MALLOC
    # ZONE) follows a blank line and its numbers must never be attributed to regions. THE TRAILING
    # SPACES IN THE PREAMBLE ARE LOAD-BEARING -- they carry the column geometry. The previous
    # fixture was written from the parser's assumptions (one header line, then rows), so it passed
    # while the parser returned 0 rows on every live process.
    FIX = ("Writable regions: Total=8.9G written=1.2G(51%) resident=790.0M(37%) "
           "swapped_out=334.0M(16%) unallocated=1.0G(47%)\n"
           "                                VIRTUAL RESIDENT    DIRTY  SWAPPED VOLATILE   NONVOL    EMPTY   REGION \n"
           "REGION TYPE                        SIZE     SIZE     SIZE     SIZE     SIZE     SIZE     SIZE    COUNT (non-coalesced) \n"
           "===========                     ======= ========    =====  ======= ========   ======    =====  ======= \n"
           "Activity Tracing                   256K      16K       0K      16K       0K      16K       0K        1 \n"
           "MALLOC_LARGE                       8.0G     5.9G   992.0M     2.4G       0K       0K       0K       42          see MALLOC ZONE table below\n"
           "Memory Tag 255                     1.3T   577.3M   498.0M   303.0M       0K       0K       0K     8790 \n"
           "unused but dirty shlib __DATA       77K      22K      22K      55K       0K       0K       0K      159 \n"
           "WEIRD                              1.0M     1.0M     1.0M       --       0K       0K       0K        1 \n"
           "TOTAL                              1.4T     1.2G   651.1M   409.0M       0K      16K     176K    23813 \n"
           "TOTAL, minus reserved VM space     1.4T     1.2G   651.1M   409.0M       0K      16K     176K    23813 \n"
           "\n"
           "                                        VIRTUAL   RESIDENT      DIRTY    SWAPPED ALLOCATION      BYTES DIRTY+SWAP          REGION\n"
           "MALLOC ZONE                           3.0G      2.0G      999.0M     9.9G   ZONE              3.0G      9.9G        see MALLOC ZONE table below\n"
           "DefaultMallocZone                     2.0G      1.0G      111.0M     7.7G   ZONE              2.0G      7.7G             2347\n")
    p = parse_vmmap(FIX)
    c("writable summary is read (not inferred)", p["writable_swapped_kb"] == 334.0 * 1024,
      p["writable_swapped_kb"])
    c("the 2.4G-swapped row is listed with its own size",
      any(r["region"] == "MALLOC_LARGE" and abs(r["swapped_kb"] - 2.4 * 1048576) < 1024
          for r in p["regions"]), [r["region"] for r in p["regions"]])
    c("a name containing spaces is one region, not two columns",
      any(abs(r["swapped_kb"] - 303.0 * 1024) < 1024 for r in p["regions"]),
      [(r["region"], r["swapped_kb"]) for r in p["regions"]])
    c("the 4th numeric column is SWAPPED (index taken from the header)",
      any(r["region"] == "Activity Tracing" and r["swapped_kb"] == 16
          and r["resident_kb"] == 16 for r in p["regions"]),
      [(r["region"], r["virtual_kb"], r["resident_kb"], r["swapped_kb"]) for r in p["regions"]])
    c("a 6-char DIRTY cell is read whole (the ruler's own spans clip `992.0M` to `98.0M`)",
      any(r["region"] == "MALLOC_LARGE" and abs(r["dirty_kb"] - 992.0 * 1024) < 1024
          and abs(r["swapped_kb"] - 2.4 * 1048576) < 1024 for r in p["regions"]),
      [(r["region"], r["dirty_kb"], r["swapped_kb"]) for r in p["regions"]])
    c("vmmap's own TOTAL row is kept for reconciliation, and both TOTAL rows stay out of regions",
      p["table_total_kb"] is not None
      and abs(p["table_total_kb"]["swapped_kb"] - 409.0 * 1024) < 1024, p["table_total_kb"])
    c("regions are sorted by swapped size", [r["swapped_kb"] for r in p["regions"]] ==
      sorted((r["swapped_kb"] for r in p["regions"]), reverse=True))
    c("both TOTAL aggregate rows are excluded",
      not any(r["region"].startswith("TOTAL") for r in p["regions"]),
      [r["region"] for r in p["regions"]])
    c("the SECOND table's 9.9G/7.7G are not attributed to regions "
      "(the real defect: 'top regions' read as 2347 / 'see MALLOC ZONE table below')",
      all(r["swapped_kb"] < 3.0 * 1048576 for r in p["regions"])
      and not any("ZONE" in r["region"] for r in p["regions"]),
      [r["region"] for r in p["regions"]])
    c("a row whose SWAPPED cell is `--` is counted as unparsed and left out of regions",
      p["unparsed"] == 1 and not any(r["region"] == "WEIRD" for r in p["regions"]),
      f'unparsed={p["unparsed"]}, regions={[r["region"] for r in p["regions"]]}')
    empty = parse_vmmap("nothing here")
    c("an empty/unreadable vmmap yields None, not 0", empty["writable_swapped_kb"] is None)

    # The fixture is hand-copied, so tie it to reality: the column geometry must equal what the live
    # `vmmap -summary` ruler states on this machine right now. If vmmap's format changes, or if the
    # trailing spaces above get stripped by an editor, this fails instead of the fixture silently
    # testing a format that no longer exists.
    live = subprocess.run(["vmmap", "-summary", str(os.getpid())], capture_output=True, text=True)
    if live.returncode != 0 or not live.stdout:
        print(f"  SKIP  live geometry check (vmmap could not attach to self: "
              f"{(live.stderr or 'no output').strip()[:60]})")
    else:
        lv = live.stdout.splitlines()
        h = next((i for i, l in enumerate(lv)
                  if "SWAPPED" in l and "REGION" in l and "VIRTUAL ALLOCATION" not in l), None)
        real_sep = next((lv[i] for i in range(h + 1, min(h + 4, len(lv)))
                         if lv[i].strip() and set(lv[i].strip()) <= set("= ")), None) if h is not None else None
        fix_sep = FIX.splitlines()[3]
        c("the fixture's column geometry equals the live vmmap's",
          real_sep is not None and _column_spans(fix_sep) == _column_spans(real_sep),
          f"real={_column_spans(real_sep) if real_sep else None}")
        lp = parse_vmmap(live.stdout)
        c("a real process yields region rows (the defect: 0 rows, reported as success)",
          len(lp["regions"]) > 0, len(lp["regions"]))
        c("real region names are names, not numbers or notes",
          all(any(ch.isalpha() for ch in r["region"]) for r in lp["regions"])
          and not any("ZONE" in r["region"] or r["region"].replace(".", "").isdigit()
                      for r in lp["regions"]),
          [r["region"] for r in lp["regions"] if not any(ch.isalpha() for ch in r["region"])][:3])
        c("real rows keep resident <= virtual",
          all(r["resident_kb"] <= r["virtual_kb"] for r in lp["regions"]),
          [r["region"] for r in lp["regions"] if r["resident_kb"] > r["virtual_kb"]][:3])
        # The arithmetic check: the rows must sum back to vmmap's OWN TOTAL row, which is what makes
        # the slicing verifiable rather than plausible. Only SWAPPED and DIRTY can be checked this
        # way -- vmmap prints their totals to 4 significant figures (`408.6M`) while VIRTUAL and
        # RESIDENT totals are 2 (`1.4T`, `1.2G`), a ±4% window in the total itself. Measured
        # residuals with 35 rows: swapped 0.0015%, dirty 0.0015%, virtual 4.7%, resident 3.4%.
        tt = lp["table_total_kb"] or {}
        for k in ("swapped_kb", "dirty_kb"):
            tot = tt.get(k)                       # None (no TOTAL row) is not 0 (a real total of 0)
            s = sum(r[k] for r in lp["regions"])
            if tot is None:
                c(f"{k[:-3].upper()} rows sum back to vmmap's own TOTAL", False, "no TOTAL row")
            elif tot == 0:
                c(f"{k[:-3].upper()} rows sum back to vmmap's own TOTAL", s == 0, "TOTAL=0")
            else:
                c(f"{k[:-3].upper()} rows sum back to vmmap's own TOTAL (0.1%)",
                  abs(s - tot) / tot < 0.001, f"{abs(s - tot) / tot:.4%}")
    print(f"\nswap_owner_probe selftest: {'OK' if bad == 0 else f'{bad} FAILED'}")
    return 0 if bad == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, help="engine pid to attach to")
    ap.add_argument("--seconds", type=int, default=120)
    ap.add_argument("--interval", type=float, default=10.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.pid:
        raise SystemExit("need --pid (attach) or --selftest")

    samples, t_end = [], time.time() + args.seconds
    while time.time() < t_end:
        s = sample(args.pid)
        samples.append(s)
        print(f"  {s.get('t')}  engine swapped={_fmt(s.get('writable_swapped_kb'))}  "
              f"box swap={_fmt(s.get('swap_used_mb', 0) * 1024 if s.get('swap_used_mb') else None)}"
              f"  regions={len(s.get('regions', []))}", flush=True)
        if s.get("error"):
            print("   error:", s["error"], flush=True)
        time.sleep(args.interval)

    live = [s for s in samples if s.get("writable_swapped_kb") is not None]
    out = {"pid": args.pid, "n": len(samples), "samples": samples}
    if live:
        peak = max(live, key=lambda s: s["writable_swapped_kb"])
        box = max((s.get("swap_used_mb") or 0) for s in live)
        out["verdict"] = {
            "engine_swap_peak_mb": round(peak["writable_swapped_kb"] / 1024, 1),
            "box_swap_peak_mb": box,
            "engine_share_pct": round(peak["writable_swapped_kb"] / 1024 / box * 100, 1) if box else None,
            "reading": ("ENGINE PAGES ARE IN SWAP -> a miss may refault from SSD (hot path)"
                        if peak["writable_swapped_kb"] / 1024 > 512 else
                        "the engine's own bytes stay resident -> the swap belongs to other "
                        "processes and is not in our critical path"),
            "top_regions": peak["regions"][:5]}
    print("\n" + json.dumps(out.get("verdict", {}), ensure_ascii=False, indent=1))
    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)
        print("->", args.out)
    return 0


def _fmt(kb) -> str:
    if kb is None:
        return "unreadable"
    return f"{kb/1048576:.2f} GiB" if kb > 1048576 else f"{kb/1024:.0f} MiB"


if __name__ == "__main__":
    sys.exit(main())
