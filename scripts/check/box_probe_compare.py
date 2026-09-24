#!/usr/bin/env python3
"""Both definitions of "is the box free", sampled together, so the disagreement is a reading.

WHY. Two probes answer that question on this machine and they are not interchangeable:

  harness  reclaimable = Pages free + purgeable + inactive     (server_window -> decode_window_harness)
  launcher memory_pressure -Q "free percentage" >= per class   (run_server.sh:856)

On 2026-09-20 they said opposite things about the same box (harness refused at 6636 MB while the
launcher read 84%), so which one is binding is a fact to measure, not to assume. `server_window`
now records both and names the stricter one; this prints the underlying numbers.

Usage: box_probe_compare.py [--n 6] [--interval 8] [--need-mb 8000]
"""
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "check"))
import server_window as sw  # noqa: E402


def classes() -> dict:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    d = {k.strip(): int(v) for k, v in re.findall(r"([A-Za-z\- ]+):\s+(\d+)\.", out)}
    mb = lambda k: round(d.get(k, 0) * 16384 / 1048576)
    return {"free": mb("Pages free"), "inactive": mb("Pages inactive"),
            "purgeable": mb("Pages purgeable"), "compressor": mb("Pages occupied by compressor")}


def swap_mb() -> float:
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([\d.]+)M", out)
    return float(m.group(1)) if m else -1.0


def row(need_mb: float) -> dict:
    c = classes()
    d = sw.decision(need_mb=need_mb)
    return {"classes": c, "swap_mb": swap_mb(), **{k: d[k] for k in
            ("reclaimable_mb", "need_mb", "harness_admits", "foreign_llama", "port_held",
             "launcher_free_pct", "launcher_req_pct", "launcher_other_servers", "launcher_terms",
             "launcher_admits", "agree", "binding", "admits", "refused_by")}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--interval", type=float, default=8.0)
    ap.add_argument("--need-mb", type=float, default=sw.NEED_MB)
    args = ap.parse_args()
    print(f"{'free':>7} {'inact':>7} {'purge':>7} {'comp':>7} {'swap':>7} | {'reclaim':>8} "
          f"{'need':>6} {'harness':>8} | {'FREE%':>6} {'req%':>5} {'oth':>3} {'launcher':>9} | "
          f"DECISION  agree  binding  failing term")
    rows = []
    for i in range(args.n):
        r = row(args.need_mb)
        c = r["classes"]
        pct = "-" if r["launcher_free_pct"] is None else str(r["launcher_free_pct"])
        req = "-" if r["launcher_req_pct"] is None else str(r["launcher_req_pct"])
        ladm = "-" if r["launcher_admits"] is None else ("admits" if r["launcher_admits"] else "refuses")
        failed = "+".join(t for t, ok in (r.get("launcher_terms") or {}).items() if not ok) or "-"
        hwhy = "-"
        if not r["harness_admits"]:
            bit = []
            if r["reclaimable_mb"] < r["need_mb"]:
                bit.append("memory")
            if r["foreign_llama"]:
                bit.append(f"foreign x{len(r['foreign_llama'])}")
            if r["port_held"]:
                bit.append("port")
            hwhy = "+".join(bit) or "?"
        print(f"{c['free']:>7} {c['inactive']:>7} {c['purgeable']:>7} {c['compressor']:>7} "
              f"{r['swap_mb']:>7.0f} | {r['reclaimable_mb']:>8.0f} {r['need_mb']:>6.0f} "
              f"{('admits' if r['harness_admits'] else 'refuses'):>8} | {pct:>6} {req:>5} "
              f"{r['launcher_other_servers']:>3} {ladm:>9} | "
              f"{('QUIET' if r['admits'] else 'busy'):>8} {str(r['agree']):>5}  "
              f"{str(r['binding']):>7}  harness:{hwhy} launcher:{failed}")
        rows.append(r)
        if i < args.n - 1:
            time.sleep(args.interval)
    disagree = [r for r in rows if r["agree"] is False]
    print()
    if not rows:
        print("INVALID: no samples")
        return 1
    quiet = [r for r in rows if r["admits"]]
    print(f"samples={len(rows)}  quiet={len(quiet)}  disagreements={len(disagree)}"
          + (f"  (all binding={disagree[0]['binding']})" if disagree else "")
          + (f"  refused_by={sorted({t for r in rows for t in r['refused_by']})}" if quiet == []
             else ""))
    if len(rows) > 1:
        lo = min(r["reclaimable_mb"] for r in rows)
        hi = max(r["reclaimable_mb"] for r in rows)
        print(f"reclaimable range {lo:.0f}-{hi:.0f} MB against a {args.need_mb:.0f} MB floor; "
              f"swap moved {rows[0]['swap_mb']:.0f} -> {rows[-1]['swap_mb']:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
