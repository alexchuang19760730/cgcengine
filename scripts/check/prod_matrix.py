#!/usr/bin/env python3
"""The single production-grade measurement entry point: every profile x the standard cells.

WHY THIS FILE EXISTS
--------------------
On 2026-09-17 the question "what is our decode" had FOUR incompatible answers (llama-bench
10.78/10.91, `decode_bench` 12.36, the HTTP sweep 12.95, the MTP A/B 12.62) and "what is our
prefill" had TWO (the HTTP acceptance arm 278.56/261.17/275.01 at a 2873-token prompt, and
llama-bench `pp2048` 276.25/300.43). Nothing in the repo enumerated the profiles at all, so a
"consistency" claim had to be assembled by hand from five files -- which is how the numbers drifted
apart in the first place.

Two user rulings fixed the instrument (both 2026-09-17): `decode_bench` is retired, and
`llama-bench` is the only instrument of record for BOTH halves. This file makes the rest of that
decision mechanical:

  * the PROFILE LIST is not written here -- it is parsed out of `scripts/run_server.sh` (the
    authority, which refuses unknown profiles by name), so it cannot drift;
  * the ENV and the MODEL are not written here -- every cell calls `llama_bench_matrix.py`, which
    gets them from `run_server.sh CGC_DUMP_ENV=1`, so there is exactly one resolution path;
  * the THERMAL series is not re-implemented -- `thermal_pressure.Sampler`, which takes its first
    sample BEFORE the child exists (that is what makes `launch` mean "launched into");
  * only the CELL SHAPES and the REPORTING RULE live here, because those are the two things that
    were previously re-decided per measurement.

THE CELLS (one shape per launch -- see the trap below)
------------------------------------------------------
| cell              | -p   | -n  | -d  | -b/-ub | why                                            |
| decode            | 0    | 128 | 512 | 512    | the house decode standard; `-d 512` is the     |
|                   |      |     |     |        | warm platform (`--depths 0` is the coldest cell)|
| decode-up         | 0    | 128 | 512 | 2048   | same row at the UPSTREAM batch so it can be    |
|                   |      |     |     |        | put beside published numbers                    |
| prefill-house     | 2048 | 16  | 0   | profile| continuity with the recorded pp2048 captures  |
| prefill-up        | 512  | 128 | 0   | 2048   | exactly upstream's default shape -> `pp512`    |

Upstream's own defaults are `-p 512 -n 128 -d 0 -b 2048` (`tools/llama-bench/llama-bench.cpp`),
and its canonical output rows are `pp512 / tg128 / pp512 @ d512` (`tools/llama-bench/README.md`).
`decode` is literally upstream's `tg128 @ d512` row, run at this repo's batch.

THE REPORTING RULE (the whole point of a standard)
--------------------------------------------------
Subtracting rep 1, not the mean. llama-bench's tg warmup is ONE token (`llama-bench.cpp`), so rep 1
runs against an empty pool while reps 2..n run against a warm one. Measured example
(`Backup/phase_decomp/lb_depth_repro_20260916.json`, `prod25-stream`):
`samples_ts = [8.19, 9.96, 10.41]` -> `avg_ts` 9.52, platform 10.19 (**+7.0%**). So this file always
prints both, and marks the platform value as the one to quote. Every row also carries the model
filename and the build commit, because `tag` does NOT contain the model -- `--arm prefill250` and
`--arm prod25` resolve to the SAME file, and only an explicit `CGC_SERVER_MTP=0` swaps it.

THE TRAP THIS FILE ENFORCES
---------------------------
One shape per launch. `llama-bench` builds one context per (n_prompt, n_gen, n_depth) combination
but the model and the expert pool are shared, so `-p 512,2048` lets the second cell inherit the
first cell's warm pool. Every cell here is a separate process with its own `--workdir`.

WHAT IT DOES NOT DO
-------------------
It does not measure anything new, it does not build, and it does not decide anything. N>1 for
`--runs` turns it into a cross-launch spread test; `prefill_certifiability.py` remains the deeper
harness for that (it also records the per-launch memory state that the spread correlates with).

USAGE
-----
    python3 scripts/check/prod_matrix.py --list                 # profiles x settings, zero GPU
    python3 scripts/check/prod_matrix.py --dry-run              # every command, zero GPU
    python3 scripts/check/prod_matrix.py --cells decode         # one cell, all profiles
    python3 scripts/check/prod_matrix.py --json out.json --md out.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MATRIX = HERE / "llama_bench_matrix.py"
RUN_SERVER = ROOT / "scripts" / "run_server.sh"

# Reused, not re-implemented. `resolve()` is the ONE path that turns a profile name into env +
# scalars + server argv; `parse_rows()` is the ONE reader that survives a mid-run llama-bench
# failure. `Sampler` takes its launch reading before the child exists. `mem_state()` is the same
# "usable memory" definition the acceptance arm uses.
from llama_bench_matrix import resolve, forward_argv, default_batch, parse_rows  # noqa: E402
from thermal_pressure import Sampler  # noqa: E402
from prefill_certifiability import mem_state  # noqa: E402

# ---------------------------------------------------------------------------------------------
# The standard cells. Only `p/n/d/batch/why` live here; everything else is resolved.
# `batch=None` means "the profile's own batch", which is what the recorded pp2048 captures used.
# ---------------------------------------------------------------------------------------------
CELLS: dict[str, dict] = {
    "decode": dict(
        p=0, n=128, d=512, batch="512",
        why="house decode standard; -d 512 is the warm platform; = upstream's `tg128 @ d512` row",
    ),
    "decode-up": dict(
        p=0, n=128, d=512, batch="2048",
        why="same row at upstream's default batch, so it can sit beside published numbers",
    ),
    "prefill-house": dict(
        p=2048, n=16, d=0, batch=None,
        why="continuity with the recorded pp2048 captures (276.25 / 300.43 at NOMINAL)",
    ),
    "prefill-up": dict(
        p=512, n=128, d=0, batch="2048",
        why="exactly upstream's default shape -> the `pp512` row of the standard table",
    ),
}

CELL_ORDER = ["decode", "decode-up", "prefill-house", "prefill-up"]

# Warmup is NOT a knob here on purpose: `--no-warmup` is a no-op at `-p 0` (the tg warmup is one
# token) and the default (warmup ON) is the closer analogue of a served request. Recorded as a
# constant so the report can state it rather than imply it.
WARMUP_RULE = "llama-bench default (warmup ON); report the platform value, i.e. drop rep 1"


def profiles_from_run_server() -> list[str]:
    """Parse the profile list out of the authority, and FAIL LOUDLY if the line moves.

    A hand-copied profile list is a list that is wrong on the day someone adds a profile, and the
    failure is silent (the new profile is simply never measured). `run_server.sh` already names
    every legal value in its refusal message, so that message is the list.
    """
    text = RUN_SERVER.read_text(errors="replace")
    # The anchor has to be the VARIABLE NAME, not the phrase `must be`. Measured 2026-09-17: the
    # file has several `must be a|b|c` refusals (the thermal mode's is `auto|mtp|non-mtp`), so
    # keying on the phrase returned a 3-item list that looked plausible -- three profiles, not one
    # of which resolves. `CGC_SERVER_PROFILE must be ...` is unique.
    m = re.search(r"CGC_SERVER_PROFILE must be\s+([a-z0-9_+-]+(?:\|[a-z0-9_+-]+)+)", text)
    if not m:
        raise SystemExit(
            "cannot find the profile list in scripts/run_server.sh (looked for the "
            "'CGC_SERVER_PROFILE must be a|b|c' refusal line). Refusing to guess: a guessed list is "
            "silently short. Fix this parser."
        )
    return m.group(1).split("|")


def gate() -> list[str]:
    """Return the list of blockers. Empty list == clear to run.

    The check has to be in the SAME branch as the action, otherwise it is a placebo: an `lsof`
    printed next to a `cmake --build` on one line cost another line a whole A/B window on
    2026-09-17 (they were mid-run when the dylib was replaced). Here the caller aborts on any
    blocker with exit 3 -- it does not warn and continue.
    """
    blockers: list[str] = []
    p = subprocess.run(["lsof", "-nP", "-iTCP:8080", "-sTCP:LISTEN"],
                       capture_output=True, text=True)
    if p.stdout.strip():
        blockers.append("8080 has a listener (a server is up): " +
                        p.stdout.strip().splitlines()[-1][:120])
    # Command-line text match, so the pattern cannot appear in this file's own argv. `ps` may be
    # unavailable in some sandboxes; say so instead of reporting "clear" (an unreadable check that
    # reads as a pass is worse than no check).
    q = subprocess.run(["ps", "-Ao", "pid=,command="], capture_output=True, text=True)
    if q.returncode != 0:
        blockers.append(f"ps failed (rc={q.returncode}) -- cannot confirm the machine is idle")
        return blockers
    me = str(Path(__file__).name)
    pat = re.compile(r"m123_oracle_gate|decode_sweep|mtp_accept_ab|run_ids_dst_capture"
                     r"|knifeedge_matrix|llama-bench|llama-server")
    mine = {str(os.getpid()), str(os.getppid())}
    for line in q.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        pid, cmd = parts
        if pid in mine or me in cmd:
            continue
        if pat.search(cmd):
            blockers.append(f"another measurement process: pid {pid} {cmd[:110]}")
    return blockers


def compat(profile: str, cell: str, env: dict, scalars: dict, b: str) -> tuple[bool, str]:
    """Can this cell be HONOURED on this profile? (ok, reason-if-not)

    Found by `--dry-run` on 2026-09-17, before it cost a GPU window: `prefill-house` (`-p 2048`)
    resolves to `-b 8` on the six pool-only profiles, and 2048 tokens into an 8-token batch trips
    `GGML_ASSERT(n_tokens_all <= cparams.n_batch)` inside `llama_context::decode` -- i.e. the cell
    would abort, and `incomplete: true` in the matrix json is the only trace left.

    The second case is quieter and worse: a cell whose batch the engine will not honour still RUNS,
    and produces a number wearing the cell's label. `default_batch()` says why in its own comment --
    a profile that neither pins BATCH nor turns on PREFILL_STREAM gets `cgc_pool_max_tokens()`
    (engine default 8), so `-b 512` / `-b 2048` are not the effective batch there. Running it anyway
    is the "same label, different quantity" failure this file exists to prevent, so it is refused
    rather than annotated.

    Both refusals are stated per (profile, cell) instead of being discovered as a crashed arm.
    """
    p = CELLS[cell]["p"]
    ob = scalars.get("BATCH", "-")
    pinned = ob not in ("-", "")
    stream = env.get("CGC_PREFILL_STREAM", "0") not in ("0", "")
    if p and b.isdigit() and int(b) < p:
        return False, (f"-p {p} does not fit -b {b} (GGML_ASSERT n_tokens_all <= n_batch; the "
                       f"engine clamps n_batch to cgc_pool_max_tokens() on this profile)")
    if not pinned and not stream and b != "8":
        return False, (f"profile pins no BATCH and {profile} has no CGC_PREFILL_STREAM, so the "
                       f"engine's pool-path clamp (cgc_pool_max_tokens, default 8) applies -- the "
                       f"cell's -b {b} would NOT be the effective batch")
    return True, ""


def cell_command(profile: str, cell: str, reps: int, workdir: Path, jpath: Path,
                 extra_env: dict | None = None) -> tuple[list[str], dict, str, str, str, bool, str]:
    """Resolve one (profile, cell) into a concrete llama-bench command.

    Zero GPU: this is `run_server.sh CGC_DUMP_ENV=1` plus arithmetic. That is deliberate -- the
    shape of every cell can be checked, and is checked by `--dry-run`, without spending a window.
    """
    spec = CELLS[cell]
    res = resolve(profile, extra_env or {})
    env, argv, scalars = res["env"], res["server_argv"], res["scalars"]
    fwd = forward_argv(argv)
    if spec["batch"] is not None:
        b = ub = spec["batch"]
        why = "cell (argparse)"
    else:
        b, ub, why = default_batch(env, scalars)
    ok, why_not = compat(profile, cell, env, scalars, b)
    arm = profile + "".join(f";{k}={v}" for k, v in (extra_env or {}).items())
    cmd = [sys.executable, str(MATRIX), "--arms", arm,
           "--prompt", str(spec["p"]), "--gen", str(spec["n"]), "--depths", str(spec["d"]),
           "--reps", str(reps), "--workdir", str(workdir), "--json", str(jpath)]
    if spec["batch"] is not None:
        cmd += ["--batch", str(spec["batch"])]
    # The matrix rebuilds the llama-bench argv itself; `fwd` is only used here to show the reader
    # what the cell will inherit (model, -ngl, --expert-cache, kv types ...).
    model = next((fwd[i + 1] for i, a in enumerate(fwd[:-1]) if a == "-m"), "<no -m in server argv>")
    return cmd, spec, model, b, why, ok, why_not


def run_cell(profile: str, cell: str, args) -> dict:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    rdir = Path(args.logdir) / f"{stamp}_{profile}_{cell}"
    rdir.mkdir(parents=True, exist_ok=True)
    jpath = rdir / "summary.json"
    extra = dict(kv.split("=", 1) for kv in args.extra_env.split(";") if "=" in kv) if args.extra_env else {}
    cmd, spec, model, b, why, ok, why_not = cell_command(profile, cell, args.reps, rdir, jpath, extra)

    if not ok:
        print(f"  SKIP {profile:<14} {cell:<14} -- {why_not}")
        return {"profile": profile, "cell": cell, "incompatible": True, "reason": why_not,
                "shape": dict(p=spec["p"], n=spec["n"], d=spec["d"], b=b, ub=b, why=why)}

    if args.dry_run:
        print(f"  {profile:<14} {cell:<14} -p {spec['p']:<4} -n {spec['n']:<3} -d {spec['d']:<3} "
              f"-b {b:<5} [{why}]  -m {Path(model).name}")
        return {"profile": profile, "cell": cell, "dry_run": True,
                "shape": dict(p=spec["p"], n=spec["n"], d=spec["d"], b=b, ub=b, why=why),
                "model": model, "matrix_cmd": cmd}

    pre = mem_state()
    if pre["usable_pct"] < args.min_usable_pct:
        print(f"  SKIP {profile}/{cell}: usable {pre['usable_pct']:.1f}% < floor "
              f"{args.min_usable_pct}% -- launching here measures the OOM cliff, not the shape.")
        return {"profile": profile, "cell": cell, "skipped": True, "pre": pre}

    t0 = time.time()
    with Sampler(interval=0.5) as s:            # launch reading is taken before the child exists
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    thermal = s.result
    wall = time.time() - t0
    out = {"profile": profile, "cell": cell, "rc": proc.returncode, "wall_s": round(wall, 1),
           "thermal": {k: thermal[k] for k in ("n", "launch", "worst", "hist", "interval_s")},
           "dir": str(rdir), "matrix_cmd": cmd, "pre": pre}
    if proc.returncode != 0:
        out["error"] = (proc.stdout[-1500:] + "\n" + proc.stderr[-1500:])
    entry = None
    if jpath.exists():
        try:
            entries = json.loads(jpath.read_text())
            entry = entries[0] if entries else None
        except json.JSONDecodeError:
            pass
    rows = parse_rows(proc.stdout) or (entry or {}).get("rows") or []
    out["rows"] = rows
    out["platform"] = platform_of(rows)
    out["model_filename"] = next((r.get("model_filename") for r in rows if r.get("model_filename")), model)
    out["build_commit"] = next((r.get("build_commit") for r in rows if r.get("build_commit")), None)
    return out


def platform_of(rows: list[dict]) -> dict:
    """Drop rep 1 and say what is left. This is the number the standard quotes.

    Reported for each row (llama-bench emits one row per (p, n, d) combination, and every cell here
    has exactly one combination) with `n_kept` so a 1-rep run is visibly NOT a platform value
    instead of quietly looking like one.
    """
    out = {}
    for r in rows:
        s = r.get("samples_ts") or []
        kept = s[1:]
        key = f"p{r.get('n_prompt')}n{r.get('n_gen')}d{r.get('n_depth')}"
        out[key] = {
            "avg_ts": r.get("avg_ts"),
            "stddev_ts": r.get("stddev_ts"),
            "samples_ts": s,
            "n_kept": len(kept),
            "platform_ts": (sum(kept) / len(kept)) if kept else None,
            "rep1_penalty_pct": (
                round((sum(kept) / len(kept)) / r["avg_ts"] * 100 - 100, 2)
                if kept and r.get("avg_ts") else None),
        }
    return out


def fmt_platform(p: dict) -> str:
    if not p:
        return "no rows"
    parts = []
    for key, v in p.items():
        if v["platform_ts"] is None:
            parts.append(f"{key}: only {len(v['samples_ts'])} rep -- NOT a platform value")
        else:
            pen = f" (rep1 {v['rep1_penalty_pct']:+.1f}%)" if v["rep1_penalty_pct"] is not None else ""
            parts.append(f"{key}: platform {v['platform_ts']:.2f} t/s{pen}  "
                         f"[avg_ts {v['avg_ts']:.2f}, samples {[round(x,2) for x in v['samples_ts']]}]")
    return " | ".join(parts)


MODEL_ALIASES = {
    "Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf": "Nail…denseIQ4X",
    "Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf": "Qwen3.6-35B-UD (non-Nail)",
}


def short_model(path: str) -> str:
    n = Path(path).name
    if n in MODEL_ALIASES:
        return MODEL_ALIASES[n]
    return n if len(n) <= 28 else n[:13] + "…" + n[-12:]


# The settings this table unifies. Each is either a scalar from the dump's `CGCENV` lines or an
# allowed `CGC_*` from its `ENV` lines. MTP is deliberately NOT read from an env var: it is a shell
# variable inside run_server.sh and never appears in the dump, so reading it would silently report
# "(default)" for every profile. The resolved server ARGV is the authority instead -- MTP=1 emits
# `--spec-type draft-mtp --spec-draft-n-max 3`, MTP=0 emits neither (verified for prod25 with and
# without CGC_SERVER_MTP=0 on 2026-09-17).
SETTING_KEYS = [
    ("ctx", lambda sc, env, argv: sc.get("CTX", "-")),
    ("batch", lambda sc, env, argv: sc.get("BATCH", "-")),
    ("ubatch", lambda sc, env, argv: sc.get("UBATCH", "-")),
    ("budgetGiB", lambda sc, env, argv: f"{int(sc['BUDGET']) / 2**30:.1f}" if sc.get("BUDGET", "").isdigit() else "-"),
    ("mtp", lambda sc, env, argv: "1" if "--spec-type" in argv else "0"),
    ("spac", lambda sc, env, argv: env.get("CGC_SPAC", "0")),
    ("spacA", lambda sc, env, argv: env.get("CGC_SPAC_ALPHA", "-")),
    ("stream", lambda sc, env, argv: env.get("CGC_PREFILL_STREAM", "0")),
    ("slab", lambda sc, env, argv: env.get("CGC_GATHER_SLAB_CAP", "-")),
    ("caps", lambda sc, env, argv: env.get("LLAMA_EXPERT_CACHE_LAYER_CAPS", "-")),
    ("model", lambda sc, env, argv: short_model(sc.get("MODEL", "?"))),
]


def do_list(args) -> int:
    profiles = (profiles_from_run_server() if args.profiles in (None, "all")
                else [p.strip() for p in args.profiles.split(",")])
    print(f"profiles: {len(profiles)} (parsed from scripts/run_server.sh -- the authority -- "
          f"not copied here)")
    print("  " + " | ".join(profiles))
    print("\nresolved settings (every value comes from `run_server.sh CGC_DUMP_ENV=1`; zero GPU)")
    rows = {}
    for prof in profiles:
        try:
            res = resolve(prof, {})
        except SystemExit as e:
            print(f"  !! {prof}: resolve failed: {e}")
            continue
        sc, env = res["scalars"], res["env"]
        rows[prof] = {k: fn(sc, env, res["server_argv"]) for k, fn in SETTING_KEYS}
    if not rows:
        return 1
    names = [k for k, _ in SETTING_KEYS]
    width = {k: max(len(k), *(len(rows[p][k]) for p in rows)) for k in names}
    print("\n  " + f"{'profile':<15}" + "".join(f"{k:>{width[k] + 2}}" for k in names))
    print("  " + "-" * (15 + sum(width[k] + 2 for k in names)))
    for p in rows:
        print("  " + f"{p:<15}" + "".join(f"{rows[p][k]:>{width[k] + 2}}" for k in names))

    # Which settings actually differ, and who is the odd one out. This is the "unify the settings"
    # question stated as data instead of as a paragraph: a profile that silently inherits a global
    # default is exactly the drift `run_server.sh`'s own comments keep paying for ("profile 不寫的
    # 那一格，就是會漂移的那一格").
    print("\n  where they differ")
    for k in names:
        vals: dict[str, list[str]] = {}
        for p in rows:
            vals.setdefault(rows[p][k], []).append(p)
        if len(vals) < 2:
            continue
        parts = ", ".join(f"{v} -> {'/'.join(sorted(ps))}" for v, ps in
                          sorted(vals.items(), key=lambda kv: -len(kv[1])))
        print(f"    {k:<10} {parts}")

    print("\ncells (the standard shapes; one launch each -- never two shapes in one process)")
    print(f"  {'cell':<15} {'-p':>5} {'-n':>4} {'-d':>4} {'-b/-ub':>8}  why")
    for c in CELL_ORDER:
        s = CELLS[c]
        print(f"  {c:<15} {s['p']:>5} {s['n']:>4} {s['d']:>4} "
              f"{str(s['batch'] or 'profile'):>8}  {s['why']}")
    print(f"\nwarmup rule: {WARMUP_RULE}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profiles", default="all",
                    help="'all' or a comma list; validated against run_server.sh")
    ap.add_argument("--cells", default=",".join(CELL_ORDER),
                    help=f"comma list of {CELL_ORDER}")
    ap.add_argument("--reps", type=int, default=3, help="llama-bench -r (default 3)")
    ap.add_argument("--runs", type=int, default=1,
                    help="independent launches per cell (default 1). N>1 is a cross-launch spread "
                         "test; prefill_certifiability.py remains the deeper harness for that.")
    ap.add_argument("--list", action="store_true", help="profiles x settings, then exit (zero GPU)")
    ap.add_argument("--dry-run", action="store_true", help="print every command, launch nothing")
    ap.add_argument("--no-gate", action="store_true", help="skip the idle-machine gate (loud)")
    ap.add_argument("--min-usable-pct", type=float, default=30.0)
    ap.add_argument("--extra-env", default="",
                    help="semicolon-separated ENV=VAL applied to every cell (the matrix's "
                         "PROFILE:ENV=VAL form), e.g. 'CGC_PREFILL_STREAM=1;CGC_GATHER_SLAB_CAP=256' "
                         "to make the wide-batch cells runnable on a pool-only profile. Disclosed in "
                         "the output, because the cell then no longer measures the profile as-is.")
    ap.add_argument("--logdir", default=str(ROOT / "Backup" / "prod_matrix"))
    ap.add_argument("--json")
    ap.add_argument("--md")
    args = ap.parse_args()

    if args.list:
        return do_list(args)

    known = profiles_from_run_server()
    profiles = known if args.profiles in (None, "all") else [p.strip() for p in args.profiles.split(",")]
    bad = [p for p in profiles if p not in known]
    if bad:
        raise SystemExit(f"unknown profile(s) {bad}; run_server.sh knows {known}")
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    bad = [c for c in cells if c not in CELLS]
    if bad:
        raise SystemExit(f"unknown cell(s) {bad}; known {CELL_ORDER}")

    if not args.no_gate and not args.dry_run:
        blockers = gate()
        if blockers:
            print("GATE ABORT (exit 3) -- the machine is not yours:")
            for b in blockers:
                print("  -", b)
            return 3
        print("GATE PASS: no 8080 listener, no other measurement process.")
    elif args.no_gate and not args.dry_run:
        print("!! --no-gate: running without the idle-machine check. Any number produced here is "
              "NOT attributable to this configuration until the machine state is established.")

    print(f"profiles={len(profiles)} cells={len(cells)} reps={args.reps} runs={args.runs} "
          f"warmup={WARMUP_RULE}")
    if args.dry_run:
        print("DRY RUN -- resolving env via run_server.sh and printing commands; no model is loaded.")
    results = []
    for prof in profiles:
        for cell in cells:
            for run_i in range(1, args.runs + 1):
                if args.runs > 1 and not args.dry_run:
                    print(f"\n=== {prof}/{cell} run {run_i}/{args.runs} ===")
                r = run_cell(prof, cell, args)
                r["run"] = run_i
                results.append(r)
                if not args.dry_run and not r.get("skipped") and not r.get("incompatible"):
                    print(f"  {prof:<14} {cell:<14} rc={r.get('rc')} wall={r.get('wall_s')}s "
                          f"thermal launch={r['thermal']['launch']['label']} "
                          f"worst={r['thermal']['worst']['label']}")
                    print(f"      {fmt_platform(r.get('platform') or {})}")
                    print(f"      model={Path(str(r.get('model_filename'))).name} "
                          f"build={r.get('build_commit')}")

    if not args.dry_run:
        inc = [r for r in results if r.get("incompatible")]
        skip = [r for r in results if r.get("skipped")]
        print("\n" + "=" * 118)
        print("SUMMARY -- quote the platform column, and say which cell it came from")
        print("=" * 118)
        print(f"{'profile':<15} {'cell':<15} {'model':<30} {'launch':<10} {'worst':<10} platform")
        print("-" * 118)
        for r in results:
            if r.get("skipped") or r.get("incompatible"):
                continue
            pl = r.get("platform") or {}
            first = next(iter(pl.values()), None)
            val = (f"{first['platform_ts']:.2f} t/s" if first and first["platform_ts"]
                   else "n/a")
            print(f"{r['profile']:<15} {r['cell']:<15} "
                  f"{Path(str(r.get('model_filename') or '?')).name[:30]:<30} "
                  f"{r['thermal']['launch']['label']:<10} {r['thermal']['worst']['label']:<10} {val}")
        print(f"\nwarmup rule: {WARMUP_RULE}")
        print("A cell whose launch label is not NOMINAL is a HOT sample: it may be reported, but "
              "not as the production number (and never as a >=threshold claim -- see lesson "
              "eng-mh-0054).")
        if inc:
            print(f"\nNOT RUN -- {len(inc)} (profile, cell) pair(s) this profile cannot honour. "
                  f"These are refusals, not failures, and they are the reason a 'standard cell' is "
                  f"not yet one standard for every profile:")
            for r in inc:
                print(f"  {r['profile']:<15} {r['cell']:<15} {r['reason']}")
        if skip:
            print(f"\nSKIPPED -- {len(skip)} by the memory floor "
                  f"(--min-usable-pct {args.min_usable_pct}%): "
                  + ", ".join(f"{r['profile']}/{r['cell']}" for r in skip))

    if args.json:
        Path(args.json).write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"\njson -> {args.json}")
    if args.md:
        lines = ["| profile | cell | model | launch | worst | platform t/s | avg_ts |",
                 "|---|---|---|---|---|---|---|"]
        for r in results:
            if r.get("skipped") or r.get("incompatible"):
                continue
            first = next(iter((r.get("platform") or {}).values()), None)
            plat = f"{first['platform_ts']:.2f}" if first and first["platform_ts"] else "n/a"
            avg = f"{first['avg_ts']:.2f}" if first and first["avg_ts"] is not None else "n/a"
            lines.append(
                f"| {r['profile']} | {r['cell']} | {Path(str(r.get('model_filename') or '?')).name} "
                f"| {r['thermal']['launch']['label']} | {r['thermal']['worst']['label']} "
                f"| {plat} | {avg} |")
        Path(args.md).write_text("\n".join(lines) + "\n")
        print(f"md -> {args.md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
