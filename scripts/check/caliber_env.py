#!/usr/bin/env python3
"""Two questions the caliber report left open, answered mechanically.

WHY THIS EXISTS. `docs/HTTP_VS_BENCH_CALIBER_2026-09-18.md` measured llama-bench against the
run_server.sh service path and found the service path +21%..+30% faster. It closed its own §7 with
two things it could not decide:

    7. `-b 512` 與 server 自身 batch 設定是否等價：未查。這是本比較唯一沒有完全對齊的旋鈕
    7. 環境的哪一項造成 bench 掉 17%：未定位（swap 91% 是嫌疑，未證）

Both are answered here, and neither answer is the one the report assumed:

1. EQUIVALENCE (`--equiv`). Is our bench tool configured the same as run_server.sh? **No -- and the
   mismatch is not a bug in resolve(), it is the CELL definition overriding the profile.**
   `llama_bench_matrix.default_batch()` correctly prefers the profile's own BATCH/UBATCH, but
   `prod_matrix.cell_command()` line ~319 does `b = ub = spec["batch"]` first, so every cell that
   declares a batch wins over the profile. Result for `prefill250`: the server runs `-b 5632
   -ub 5632` while the bench cell runs `-b 512 -ub 512` -- 11x apart.

   A SECOND, previously unnoticed mismatch: `CGC_SERVER_MTP=0` does not merely select a different
   file. It makes eight engine env keys vanish (`CGC_MM_BITIDENT`, `CGC_DRAFT_DECODE`,
   `CGC_VERIFY_DECODE`, `CGC_NO_PREFETCH`, `CGC_MTP_NO_WARMUP`, `CGC_NO_SEQ_RM_PROBE`,
   `CGC_WARM_NPAST`, `LLAMA_EXPERT_CACHE_LAYER_CAPS`), so "MTP off" is a whole cluster of engine
   knobs, not one flag. (The model file itself is NOT a difference: both paths hash identically at
   several offsets -- see `--model-identity`.)

   Consequence: the +30% cannot yet be attributed to the environment, because two knobs are known
   to differ. It can only be attributed once those are aligned.

2. MEMORY AS A PARAMETER (`--memory`). Throughput arms are re-read together with whatever memory
   state was recorded next to them, then correlated. Arms that recorded none are listed as such --
   which is its own finding: the caliber report's own four arms carry no memory reading at all, so
   its "swap 91% is suspect" line could never have been decided from its own artifacts.

USAGE
    python3 scripts/check/caliber_env.py --equiv --profiles prefill250,prod25
    python3 scripts/check/caliber_env.py --model-identity
    python3 scripts/check/caliber_env.py --memory --json-glob 'Backup/prod_matrix/bisect_*.json'

Exit code 0 always.
"""
import argparse
import glob
import json
import math
import os
import statistics as st
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

from llama_bench_matrix import resolve, default_batch  # noqa: E402
import prod_matrix as pm  # noqa: E402


# ---------------------------------------------------------------- helpers

def _flag(argv, names):
    """First value after any of `names` in a flat argv list."""
    for n in names:
        for i, a in enumerate(argv[:-1]):
            if a == n:
                return argv[i + 1]
    return None


def sampled_fingerprint(path: str) -> str | None:
    """Cheap identity check: size + hashes of a few windows. Enough to say 'same bytes'; not a proof
    of equality, but a single differing window IS a proof of difference."""
    try:
        sz = os.path.getsize(path)
    except OSError:
        return None
    parts = [str(sz)]
    n = max(sz // (1 << 20), 1)
    for frac in (0.0, 0.25, 0.5, 0.75, 0.98):
        off = min(int(n * frac), max(n - 4, 0)) if n > 4 else 0
        try:
            blob = subprocess.run(["dd", f"if={path}", "bs=1m", f"skip={off}", "count=4"],
                                  capture_output=True, timeout=60).stdout
            h = subprocess.run(["shasum"], input=blob, capture_output=True,
                               timeout=30).stdout.decode().split()[0]
        except Exception:
            h = "err"
        parts.append(h[:12])
    return "/".join(parts)


# ------------------------------------------------------- 1. equivalence

def knob_rows(profile: str, cell: str | None) -> list[dict]:
    r = resolve(profile, {})
    env, argv, scalars = r["env"], r["server_argv"], r["scalars"]

    # What does OUR tool actually run for this cell?
    tool_batch, tool_why, tool_model = None, None, scalars.get("MODEL")
    if cell and cell in pm.CELLS:
        from pathlib import Path as _P
        cmd, spec, model, b, why, ok, why_not = pm.cell_command(
            profile, cell, 3, _P("/tmp/x"), _P("/tmp/x/s.json"), {})
        tool_batch, tool_why, tool_model = b, why, model
    if tool_batch is None:
        tool_batch, tool_ub, tool_why = default_batch(env, scalars)
        tool_batch = tool_batch

    rows = [
        {"knob": "-b / batch", "server": _flag(argv, ["-b", "--batch"]) or "<absent → default>",
         "tool": str(tool_batch), "note": tool_why or ""},
        {"knob": "-ub / ubatch", "server": _flag(argv, ["-ub", "--ubatch"]) or "<absent → default>",
         "tool": str(tool_batch), "note": tool_why or ""},
        {"knob": "-c / ctx", "server": _flag(argv, ["-c", "--ctx-size"]) or "<absent>",
         "tool": "n/a (llama-bench)", "note": "not a bench concept"},
        {"knob": "-m / model", "server": Path(scalars.get("MODEL", "")).name,
         "tool": Path(tool_model or "").name, "note": ""},
        {"knob": "-expert-cache", "server": _flag(argv, ["-expert-cache"]) or "<absent>",
         "tool": "forwarded from same resolve()", "note": "identical by construction"},
        {"knob": "--cache-type-k/-v",
         "server": f"{_flag(argv, ['--cache-type-k'])}/{_flag(argv, ['--cache-type-v'])}",
         "tool": "forwarded from same resolve()", "note": "identical by construction"},
        {"knob": "-ngl / -t", "server": f"{_flag(argv, ['-ngl'])}/{_flag(argv, ['-t'])}",
         "tool": "forwarded from same resolve()", "note": "identical by construction"},
        {"knob": "--spec-type",
         "server": f"{_flag(argv, ['--spec-type'])} n_max={_flag(argv, ['--spec-draft-n-max'])}",
         "tool": "only the decode-spec cell forwards it",
         "note": "deliberate: adds a cell, not a whitelist", "_v": "BY-DESIGN"},
    ]
    for row in rows:
        s, t = row["server"], row["tool"]
        row["verdict"] = (row.get("_v")            # an explicit verdict wins
                          or ("forwarded" if "forwarded" in t or "n/a" in t
                              else ("MATCH" if s == t else "MISMATCH")))
    return rows

# spec is NOT simply unequal: every profile ships --spec-type, and forwarding it would silently
# turn every existing cell into a speculative cell. It was added as its own CELL for that reason.
_SPEC_ROW = {"verdict-note": "deliberate -- see prod_matrix.decode-spec"}


def cmd_equiv(args) -> None:
    print("=" * 100)
    print("CONFIG EQUIVALENCE: our bench tool vs run_server.sh  (same resolve() = same source of truth)")
    print("=" * 100)
    any_mismatch = False
    for profile in args.profiles:
        for cell in (args.cells or [None]):
            print(f"\n--- profile={profile}  cell={cell or '(matrix default)'} ---")
            print(f"{'knob':<22}{'run_server.sh':<26}{'our tool':<30}{'verdict':<10}note")
            for row in knob_rows(profile, cell):
                if row["verdict"] == "MISMATCH":
                    any_mismatch = True
                print(f"{row['knob']:<22}{str(row['server'])[:25]:<26}"
                      f"{str(row['tool'])[:29]:<30}{row['verdict']:<10}{row['note'][:28]}")
    print("\n" + "=" * 100)
    if any_mismatch:
        print("VERDICT: NOT CONFIG-EQUIVALENT. The two paths differ on at least one knob, so a")
        print("throughput gap between them cannot yet be called environmental. Align the mismatched")
        print("knobs (or add a cell that inherits them) and re-run before attributing anything.")
    else:
        print("VERDICT: config-equivalent on every compared knob.")

    # The MTP=0 cluster: cheap to show, decisive to know.
    if args.show_mtp:
        print("\n--- CGC_SERVER_MTP=0 does NOT just swap the file ---")
        a = resolve(args.profiles[0], {})
        b = resolve(args.profiles[0], {"CGC_SERVER_MTP": "0"})
        ka = dict(a["env"] or {})
        kb = dict(b["env"] or {})
        gone = sorted(set(ka) - set(kb))
        print(f"  MODEL default : {Path(a['scalars']['MODEL']).name}")
        print(f"  MODEL mtp=0   : {Path(b['scalars']['MODEL']).name}")
        print(f"  env keys present with MTP on but ABSENT with MTP off ({len(gone)}):")
        for k in gone:
            print(f"     {k} = {ka[k]}")


def cmd_model_identity(_a) -> None:
    print("=" * 100)
    print("MODEL IDENTITY: is 'CGC_SERVER_MTP=0 selects another model' actually a different file?")
    print("=" * 100)
    paths = set()
    for prof in ("prefill250", "prod25"):
        try:
            paths.add(resolve(prof, {})["scalars"]["MODEL"])
            paths.add(resolve(prof, {"CGC_SERVER_MTP": "0"})["scalars"]["MODEL"])
        except Exception:
            pass
    fps = {}
    for p in sorted(paths):
        fp = sampled_fingerprint(p)
        fps.setdefault(fp, []).append(p)
        print(f"  {Path(p).name}")
        print(f"     size+windows = {fp}")
    print()
    groups = list(fps.values())
    if len(groups) == 1:
        print("VERDICT: these paths are the SAME bytes (identical size and sampled windows).")
        print("  => the model file is NOT a confound, and 'MTP=0 changes the model' is false here.")
        print("  => but see --equiv --show-mtp: eight ENGINE env keys still differ.")
    else:
        print(f"VERDICT: {len(groups)} distinct file(s) -- the model file IS a difference.")


# ------------------------------------------------- 2. memory as parameter

def _throughput(rec: dict):
    """Pull a t/s number out of the several shapes our tools have written."""
    for getter in (lambda: _first_platform_ts(rec.get("platform")),
                   lambda: rec.get("metric"),
                   lambda: rec.get("decode"),
                   lambda: rec.get("avg_ts")):
        try:
            v = getter()
        except Exception:
            v = None
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _first_platform_ts(pl):
    if isinstance(pl, dict) and pl:
        v = list(pl.values())[0]
        return v.get("platform_ts") if isinstance(v, dict) else None
    return None


def load_arms(patterns: list[str]) -> list[dict]:
    arms = []
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            try:
                data = json.load(open(f))
            except Exception:
                continue
            if not isinstance(data, list):
                data = [data]
            for i, rec in enumerate(data):
                if not isinstance(rec, dict):
                    continue
                ts = _throughput(rec)
                mem = rec.get("pre") or rec.get("mem") or {}
                if ts is None and not mem:
                    continue
                arms.append({
                    "src": f"{Path(f).name}"
                           f"{'' if len(data) == 1 else f'[{i}]'}",
                    "profile": rec.get("profile", "-"),
                    "cell": rec.get("cell", "-"),
                    "tps": ts,
                    "mem": {k: v for k, v in mem.items() if isinstance(v, (int, float))},
                    "worst": (rec.get("thermal") or {}).get("worst", {}).get("label")
                    or rec.get("worst_label", "-"),
                })
    return arms


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return round(cov / (sx * sy), 3) if sx and sy else None


def _r_crit(n: int, alpha: float = 0.05) -> float:
    """Two-tailed critical |r| for df=n-2. There is no table lookup: r_crit = t / sqrt(t^2+df), and
    t itself has no closed form. A crude but adequate normal approximation of t is used, and marked
    as approximate at the call site so nobody quotes it as exact."""
    import math as _m
    df = n - 2
    if df <= 0:
        return float("nan")
    # Cornish-Fisher-ish: z then a light t correction; within ~0.01 of published tables here.
    z = {0.05: 1.959964, 0.01: 2.575829}[alpha]
    t = z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df * df)
    return round(t / _m.sqrt(t * t + df), 3)


def cmd_memory(args) -> None:
    arms = load_arms(args.json_glob)
    if args.cell_filter:
        arms = [a for a in arms if a["cell"] == args.cell_filter]
        print(f"(restricted to cell={args.cell_filter!r} -- mixing cells would correlate two "
              f"different quantities at once)\n")
    print("=" * 100)
    print("MEMORY AS A PARAMETER -- every arm carrying a pre-launch memory reading")
    print("=" * 100)
    usable = [a for a in arms if a["tps"] is not None and a["mem"]]
    if not usable:
        print("no arm carries BOTH a throughput number and a memory reading.")
        print("That is itself the answer: this dataset cannot decide the memory question.")
        return

    keys = ["swap_used_mib", "usable_gib", "usable_pct", "free_gib", "inactive_gib",
            "anon_gib", "cached_gib"]
    keys = [k for k in keys if any(k in a["mem"] for a in usable)]
    hdr = f"{'source':<26}{'profile':<12}{'cell':<14}{'t/s':>7}{'worst':>10}"
    for k in keys:
        hdr += f"{k.replace('_mib','').replace('_gib','').replace('_pct','%'):>10}"
    print(hdr)
    print("-" * len(hdr))
    for a in sorted(usable, key=lambda r: -(r["tps"] or 0)):
        line = f"{a['src'][:25]:<26}{a['profile']:<12}{a['cell']:<14}{a['tps']:>7.2f}{str(a['worst'])[:9]:>10}"
        for k in keys:
            v = a["mem"].get(k)
            line += f"{(f'{v:.1f}' if isinstance(v, float) else str(v) if v is not None else '-'):>10}"
        print(line)

    print(f"\ncorrelation vs throughput within cell={args.cell_filter or 'ALL'} "
          f"(n={len(usable)} arms):")
    crit = _r_crit(len(usable))
    print(f"   two-tailed critical |r| at n={len(usable)} is APPROX {crit} -- anything below it is "
          f"a suggestion, not a finding.\n")
    for k in keys:
        pairs = [(a["mem"][k], a["tps"]) for a in usable if k in a["mem"] and a["tps"]]
        if len(pairs) < 3:
            print(f"   {k:<20} n<3")
            continue
        r = pearson([p[0] for p in pairs], [p[1] for p in pairs])
        tag = "" if r is None else ("   <-- exceeds critical" if abs(r) >= crit else "")
        print(f"   {k:<20} r = {r!s:>8}   (n={len(pairs)}){tag}")
    print("\nRead these as 'worth one controlled experiment', not as proof -- then use paired_ab.py.")

    blind = [a for a in arms if a["tps"] is not None and not a["mem"]]
    if blind:
        print(f"\narms with a throughput number but NO memory reading ({len(blind)}):")
        for a in blind[:12]:
            print(f"   {a['src'][:25]:<26}{a['profile']:<12}{a['cell']:<14}{a['tps']:>7.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--equiv", action="store_true")
    ap.add_argument("--memory", action="store_true")
    ap.add_argument("--model-identity", action="store_true")
    ap.add_argument("--profiles", default="prefill250,prod25")
    ap.add_argument("--cells", default="decode",
                    help="comma-separated matrix cells, or '' for the matrix default")
    ap.add_argument("--json-glob", action="append", default=[])
    ap.add_argument("--cell-filter", default=None,
                    help="correlate within ONE cell only. Mixing prefill and decode rows puts two "
                         "different quantities on the same axis and manufactures correlation.")
    ap.add_argument("--show-mtp", action="store_true", default=True)
    args = ap.parse_args()
    args.profiles = [p for p in args.profiles.split(",") if p]
    args.cells = [c for c in args.cells.split(",") if c]
    if args.cells == [""]:
        args.cells = [None]

    did = False
    if args.equiv:
        cmd_equiv(args)
        did = True
    if args.model_identity:
        cmd_model_identity(args)
        did = True
    if args.memory:
        cmd_memory(args)
        did = True
    if not did:
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
