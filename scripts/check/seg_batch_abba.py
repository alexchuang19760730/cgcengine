#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paired ABBA for the single-submit (SEG_BATCH) diagnostic arm -- 0 rebuild.

Prices the *serialized-overhead* term: step = union + cb(fill) + gap(41 segments).
Arm B (`CGC_SEG_BATCH` + `CGC_B_SCHEME` + `CGC_SLOT_TABLE_GPU`) submits the whole graph
in one async dispatch, so the per-layer hook never fires, nothing is filled, and the
41-segment wait->hook->submit serial loop disappears. Output is WRONG (placeholder ids),
so B is a DIAGNOSTIC PRICE, never a deliverable -- it answers "how much does the
segmentation itself cost", not "how fast can we serve".

THE SHAPE IS NOT OURS TO PICK
-----------------------------
Everything below the arm env is the operator's previously settled cell, taken verbatim from
`run_server.sh CGC_DUMP_ENV=1 CGC_SERVER_PROFILE=prod-new`:

    ctx=8192  budget=8589934592 (8 GiB pool)  ngl=99  load-mode=none
    batch=5632 ubatch=5632   -p 2048 -n 128 -d 512 -r 3

5632 is the profile's own BATCH/UBATCH, and it is the widest width with recorded survival:
the budget block in `run_server.sh` prints `ub=5632 pool 8GiB 存活 4/4 ... ub=6144 pool 8GiB
存活 0/5`. Do NOT "fix" it downward because an arm once died: surviving 4/4 is the recorded
evidence, pinning to a different value silently changes the cell away from every row ever
recorded under it. (This file used to pin 512 after one rc=-6; that attribution was never
verified and has been reverted -- see git history.)

P0 (`CGC_EXPERT_SKIP_READRAW=1`) is set for BOTH arms, to "1":
  * that family of switches historically tested presence, not value, and some were later made
    value-aware -- "1" is ON under BOTH semantics, "0" is ambiguous, so use "1";
  * it must be identical in A and B or the pair is confounded: this experiment prices one
    thing (single submit vs 41 segments), and residency is not that thing;
  * P1/P2 (`CGC_POOL_MADVISE`) are left ABSENT rather than "=0", because "absent" is the only
    spelling of OFF that is unambiguous under both semantics.

Why this script exists (the failure it is built to prevent):
  A previous ABBA (`docs/B_PLAN_PREFLIGHT_2026-09-24.md` §7.1) was VOIDED because the B
  arm's env was never forwarded -- both arms ran the same config and showed 0.6%. So the
  child environment here is built EXPLICITLY (`env=`, never shell inheritance): for arm A
  the three B vars are popped from a copy of os.environ, so there is no code path by which
  A can accidentally receive them.

Second trap this script handles: all three switches are PRESENCE-based in the engine
(`getenv("CGC_B_SCHEME") != nullptr` -- see llama-context.cpp:3596, llama-graph.cpp:2146,
ggml-backend.cpp:1780). Setting them to "0" would ENABLE them. Off means ABSENT, and
"absent" is what arm A is constructed to guarantee.

Third trap: `CGC_LOOP_GUARD` only exists on the server carrier (server-context.cpp:2005).
It is inert for llama-bench, so it is deliberately NOT set here.

Fourth trap: with a non-zero `-p`, llama-bench emits BOTH pp and tg rows and both carry
`n_prompt == -p`. Picking "the row whose n_prompt matches" therefore silently picks the
prefill row. The tg row is identified by `n_gen == --gen and n_depth == --depth`.

Usage:
  seg_batch_abba.py --self-test
  seg_batch_abba.py --mode identity --gen 8            # prove the two arms differ
  seg_batch_abba.py --pairs 3                          # the timed paired measurement
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
HARNESS = ROOT / "scripts" / "check" / "llama_bench_matrix.py"

# The three presence-based switches that make up arm B.
B_VARS = ("CGC_SEG_BATCH", "CGC_B_SCHEME", "CGC_SLOT_TABLE_GPU")
B_SPEC = ";".join(f"{k}=1" for k in B_VARS)

# Common to BOTH arms, so it cannot confound the pair. See module docstring.
COMMON_ENV = {"CGC_EXPERT_SKIP_READRAW": "1"}

# [2026-09-24 定讞] 沒有這條，expert cache（L4 Metal 池 + hook + gather）整個不啟用，
# 跟 -expert-cache 給多少無關：llama-model-loader.cpp:1113 與 llama.cpp:402 都以
# getenv("LLAMA_EXPERT_CACHE_ALLOW_NGL") 為門。缺它 => pool_cap_slots=0 / routable=0
# => 13.0 GB 模型整包上 Metal（上限 11.45 GB）=> prefill 收尾 Metal OOM（rc=-6）。
# 帶它 => pool_cap_slots=143 / routable=143（對上 514 份歷史 log）。
REQUIRED_ENV = {"LLAMA_EXPERT_CACHE_ALLOW_NGL": "1"}

# The operator's settled cell (run_server.sh CGC_DUMP_ENV=1 CGC_SERVER_PROFILE=prod-new).
# [2026-09-24 21:2x 定讞] ctx_size 8192 -> 0: on the llama-bench carrier 8192 makes BOTH arms
# die in prefill with `CGC-METAL-FAIL: command buffer 8 failed (status 5, Insufficient Memory
# (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory))` @ ggml_metal_synchronize, rc=-6.
# With --ctx-size 0 both arms survive (A 12.10 / B 27.81 t/s, identity mode). Same verdict as
# miss_axis.py the same day: 8192 is a SERVER-carrier value and must not be pasted into the
# llama-bench cell. -b/-ub 5632 is NOT touched -- that one is the profile's own and survives 4/4.
CELL = {"ctx_size": 0, "ubatch": 5632, "prompt": 2048, "gen": 128, "depth": 512, "reps": 3}


def build_env(arm: str, extra: dict[str, str] | None = None, p0: bool = True) -> dict[str, str]:
    """Child env built explicitly so arm A CANNOT inherit the B switches.

    No reliance on os.environ mutation by the caller, and no reliance on the caller's
    shell state -- which is exactly how the voided ABBA lost its B arm.
    """
    env = dict(os.environ)
    for k in B_VARS:
        env.pop(k, None)          # presence-based: off means absent, not "0"
    if arm.upper() == "B":
        for k in B_VARS:
            env[k] = "1"
    env.update(REQUIRED_ENV)
    if p0:
        env.update(COMMON_ENV)
    if extra:
        env.update(extra)
    return env


def classify(rows: list[dict], gen: int, depth: int, prompt: int) -> tuple[dict | None, dict | None]:
    """Split llama-bench rows into (tg, pp).

    With a non-zero -p both rows carry the same n_prompt, so n_gen is what separates them:
    the tg row generated `gen` tokens at `depth`, the pp row generated none.
    """
    tg = next((r for r in rows
               if r.get("n_gen") == gen and r.get("n_depth") == depth), None)
    pp = next((r for r in rows
               if r.get("n_gen") == 0 and r.get("n_prompt") == prompt), None)
    return tg, pp


def run_one(arm: str, idx: int, args, extra: dict[str, str] | None = None) -> dict:
    """One arm = one harness invocation = one llama-bench process."""
    tag_safe = f"{arm}{idx}"
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    json_out = workdir / f"res_{tag_safe}.json"
    json_out.unlink(missing_ok=True)     # a stale file would be read as this arm's result

    cmd = [
        sys.executable, str(HARNESS),
        "--arms", f"{args.profile}:{B_SPEC}" if arm.upper() == "B" else args.profile,
        "--reps", str(args.reps),
        "--prompt", str(args.prompt),
        "--gen", str(args.gen),
        "--depths", str(args.depth),
        "--ctx-size", str(args.ctx_size),
        "--workdir", str(workdir),
        "--json", str(json_out),
    ]
    if args.warm_skip:
        cmd += ["--warm-skip", str(args.warm_skip)]
    if args.ubatch:
        cmd += ["--batch", str(args.ubatch), "--ubatch", str(args.ubatch)]

    env = build_env(arm, extra, p0=args.p0)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env,
                          capture_output=True, text=True)
    wall = time.time() - t0
    (workdir / f"stdout_{tag_safe}.log").write_text(proc.stdout, errors="replace")
    (workdir / f"stderr_{tag_safe}.log").write_text(proc.stderr, errors="replace")

    rec = {"arm": arm.upper(), "idx": idx, "tag": tag_safe, "wall_s": round(wall, 1),
           "rc": proc.returncode, "cell": {
               "profile": args.profile, "gen": args.gen, "depth": args.depth,
               "prompt": args.prompt, "warm_skip": args.warm_skip,
               "ctx_size": args.ctx_size, "reps": args.reps,
               "batch": args.ubatch or "derived", "p0": args.p0},
           "json": str(json_out)}
    rows = []
    if json_out.exists():
        try:
            data = json.loads(json_out.read_text())
        except json.JSONDecodeError:
            data = []
        for r in data:
            for row in r.get("rows", []):
                rows.append({"n_prompt": row.get("n_prompt"),
                             "n_gen": row.get("n_gen"),
                             "n_depth": row.get("n_depth"),
                             "avg_ts": row.get("avg_ts"),
                             "stddev_ts": row.get("stddev_ts"),
                             "n_batch": row.get("n_batch")})
    rec["rows"] = rows
    tg, pp = classify(rows, args.gen, args.depth, args.prompt)
    for name, row in (("tg", tg), ("pp", pp)):
        if row:
            rec[name] = row.get("avg_ts")
            rec[f"{name}_sd"] = row.get("stddev_ts")
            rec[f"{name}_n"] = row.get("n_gen")
            rec[f"{name}_batch"] = row.get("n_batch")
    print(f"[{tag_safe}] rc={proc.returncode} wall={wall:.0f}s "
          f"tg={rec.get('tg')} pp={rec.get('pp')} rows={len(rows)}", flush=True)
    return rec


def pair_plan(pairs: int) -> list[str]:
    """ABBA within each pair so a linear environmental drift cancels inside the pair."""
    out: list[str] = []
    for p in range(pairs):
        out += ["A", "B", "B", "A"]
    return out


def self_test() -> int:
    total = fails = 0

    def chk(name, cond):
        nonlocal fails, total
        total += 1
        print(f"  {'ok  ' if cond else 'FAIL'} {name}")
        if not cond:
            fails += 1

    # 1/2: the core guarantee -- A must not carry any B switch, in ANY caller state.
    saved = os.environ.get("CGC_SEG_BATCH")
    try:
        for pollution in (None, "0", "1"):
            if pollution is None:
                os.environ.pop("CGC_SEG_BATCH", None)
            else:
                os.environ["CGC_SEG_BATCH"] = pollution
            ea = build_env("A")
            eb = build_env("B")
            leaked = [k for k in B_VARS if k in ea]
            chk(f"A env has no B switch even when CGC_SEG_BATCH={pollution!r} -> {leaked}",
                leaked == [])
            chk(f"B env carries all three (pollution={pollution!r})",
                all(k in eb for k in B_VARS))
    finally:
        if saved is None:
            os.environ.pop("CGC_SEG_BATCH", None)
        else:
            os.environ["CGC_SEG_BATCH"] = saved

    # 3: presence semantics -- "0" is ON, so a well-meaning A arm that sets =0 would be B.
    chk("B env sets literal '1' (not '0')", build_env("B")["CGC_B_SCHEME"] == "1")

    # 4: P0 is ON and identical in both arms -- under both presence and value semantics.
    chk("P0 = literal '1' in arm A", build_env("A")["CGC_EXPERT_SKIP_READRAW"] == "1")
    chk("P0 = literal '1' in arm B", build_env("B")["CGC_EXPERT_SKIP_READRAW"] == "1")
    chk("--no-p0 leaves it absent (unambiguous OFF)",
        "CGC_EXPERT_SKIP_READRAW" not in build_env("A", p0=False))
    # P1/P2 must stay absent: "=0" is only OFF under value semantics.
    chk("CGC_POOL_MADVISE absent in both arms",
        all("CGC_POOL_MADVISE" not in build_env(a) for a in ("A", "B")))

    # 5: plan shape / drift cancellation arithmetic.
    plan = pair_plan(2)
    chk("2 pairs -> A B B A A B B A", plan == ["A", "B", "B", "A", "A", "B", "B", "A"])
    chk("each pair is self-balanced (equal A/B counts)",
        all(plan[i:i + 4].count("A") == plan[i:i + 4].count("B") == 2
            for i in range(0, len(plan), 4)))
    chk("B env does NOT include CGC_LOOP_GUARD (server-only, inert for bench)",
        "CGC_LOOP_GUARD" not in build_env("B"))

    # 6: the cell defaults are the operator's settled ones, not a shape we invented.
    ap = build_parser()
    d = vars(ap.parse_args([]))
    for k, v in CELL.items():
        chk(f"default {k} == {v} (match `--{k.replace('_', '-')}` override)", d[k] == v)
    chk("default warm_skip == 0 (not passed -> llama-bench historical behaviour)",
        d["warm_skip"] == 0)

    # 7: pp/tg classification -- with -p 2048 both rows carry n_prompt=2048.
    rows = [{"n_prompt": 2048, "n_gen": 0, "n_depth": 512, "avg_ts": 248.0},
            {"n_prompt": 2048, "n_gen": 128, "n_depth": 512, "avg_ts": 13.4}]
    tg, pp = classify(rows, gen=128, depth=512, prompt=2048)
    chk("tg row picked by n_gen/n_depth, not n_prompt", tg and tg["avg_ts"] == 13.4)
    chk("pp row picked by n_gen==0", pp and pp["avg_ts"] == 248.0)
    tg2, pp2 = classify([{"n_prompt": 2048, "n_gen": 0, "n_depth": 512, "avg_ts": 1.0}],
                        gen=128, depth=512, prompt=2048)
    chk("missing tg row -> None (not silently the pp row)", tg2 is None and pp2 is not None)

    print(f"\nself-test: {total - fails}/{total} passed")
    return 1 if fails else 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="prod-new",
                    help="llama_bench_matrix profile (default prod-new = MTP off, matches preflight)")
    ap.add_argument("--pairs", type=int, default=2)
    ap.add_argument("--reps", type=int, default=CELL["reps"])
    ap.add_argument("--prompt", type=int, default=CELL["prompt"])
    ap.add_argument("--gen", type=int, default=CELL["gen"])
    ap.add_argument("--depth", type=int, default=CELL["depth"])
    ap.add_argument("--warm-skip", type=int, default=0,
                    help="0 (default) = llama-bench historical behaviour; the settled cell does "
                         "not use it")
    ap.add_argument("--ctx-size", type=int, default=CELL["ctx_size"])
    ap.add_argument("--ubatch", type=int, default=CELL["ubatch"],
                    help="the profile's own BATCH/UBATCH. Recorded survival: ub=5632 pool 8GiB "
                         "4/4, ub=6144 0/5 (run_server.sh budget block). 0 = let the harness "
                         "derive it (same 5632 for prod-new).")
    ap.add_argument("--p0", dest="p0", action="store_true", default=True,
                    help="P0: CGC_EXPERT_SKIP_READRAW=1 in BOTH arms (default on)")
    ap.add_argument("--no-p0", dest="p0", action="store_false",
                    help="leave P0 absent instead")
    ap.add_argument("--workdir", default=str(ROOT / "Backup" / "seg_batch_abba"))
    ap.add_argument("--mode", default="abba", choices=["abba", "identity"])
    ap.add_argument("--self-test", action="store_true")
    return ap


def main() -> int:
    ap = build_parser()
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    plan = (["A", "B"] if args.mode == "identity" else pair_plan(args.pairs))
    extra: dict[str, str] | None = None
    print(f"cell: -p {args.prompt} -n {args.gen} -d {args.depth} -r {args.reps} "
          f"-b/-ub {args.ubatch or 'derived'} --ctx-size {args.ctx_size} "
          f"P0={'1' if args.p0 else 'off'}", flush=True)
    recs = []
    # identity mode needs a per-arm dump path, injected per iteration below
    for i, arm in enumerate(plan, 1):
        if args.mode == "identity":
            # Path-identity proof, NOT part of the timed measurement: the hook is where
            # CGC_IDSEQ_DUMP writes, so a no-hook arm must emit ~zero rows. Symmetric extra
            # instrument => it would bias the timed comparison, hence a separate short run.
            dump = Path(args.workdir) / f"idseq_{arm}.txt"
            dump.unlink(missing_ok=True)
            extra = {"CGC_IDSEQ_DUMP": str(dump)}
        recs.append(run_one(arm, i, args, extra))
        if args.mode == "identity":
            d = Path(args.workdir) / f"idseq_{arm}.txt"
            n = len(d.read_text(errors="replace").splitlines()) if d.exists() else -1
            recs[-1]["idseq_rows"] = n
            print(f"    {arm}: idseq rows = {n}", flush=True)

    out = Path(args.workdir) / f"abba_{time.strftime('%H%M%S')}.json"
    out.write_text(json.dumps({"plan": plan, "cell": recs[0]["cell"] if recs else {},
                               "records": recs}, ensure_ascii=False, indent=2))
    print(f"\n-> {out}")
    _summary(recs, plan)
    return 0


def _summary(recs: list[dict], plan: list[str]) -> None:
    def d(a: dict, b: dict) -> str:
        if not a.get("tg") or not b.get("tg"):
            return "n/a"
        r = b["tg"] / a["tg"]
        return f"{b['tg']:.2f}/{a['tg']:.2f} = {r:.2f}x  ({r - 1:+.0%})"

    print(f"\n{'pair':<6}{'order':<8}{'result (tg)':<34}{'tokens A/B':<14}")
    for i in range(0, len(plan), 4 if len(plan) % 4 == 0 else 2):
        chunk = recs[i:i + (4 if plan[i:i + 4].count("A") == 2 else 2)]
        if plan[i:i + 4] == ["A", "B", "B", "A"]:
            a1, b1, b2, a2 = chunk
            print(f"{i // 4 + 1:<6}{'A B B A':<8}{d(a1, b1):<34}"
                  f"{a1.get('tg_n')}/{b1.get('tg_n')}")
        else:
            a1, b1 = chunk[0], chunk[1]
            print(f"{i // 2 + 1:<6}{'A B':<8}{d(a1, b1):<34}"
                  f"{a1.get('tg_n')}/{b1.get('tg_n')}")
    for metric, label in (("tg", "decode tg"), ("pp", "prefill pp")):
        ts_a = [r[metric] for r in recs if r["arm"] == "A" and r.get(metric)]
        ts_b = [r[metric] for r in recs if r["arm"] == "B" and r.get(metric)]
        if ts_a and ts_b:
            ma, mb = sum(ts_a) / len(ts_a), sum(ts_b) / len(ts_b)
            print(f"\nmeans {label}: A={ma:.2f} B={mb:.2f} -> {mb / ma:.2f}x "
                  f"({(mb / ma - 1):+.0%})   [A n={len(ts_a)} B n={len(ts_b)}]")


if __name__ == "__main__":
    raise SystemExit(main())
