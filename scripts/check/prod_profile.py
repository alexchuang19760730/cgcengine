#!/usr/bin/env python3
"""The production profile, frozen: ONE command, BOTH axes, delivery conventions, NOMINAL-gated.

WHY THIS EXISTS. Three things were true at once on 2026-09-20, and none of the existing tools
could hold them together:

  1. The delivery decode number (10.94 / 13.10 t/s, `Backup/phase_decomp/{warmskip2,carrier2}_*`)
     is NOT reproducible with `profile_duo.py` or `prod_matrix.py`. Those two run the `decode`
     cell, which carries no `--spec-type` and no `--warm-skip`, i.e. MTP off and a cold clock --
     the same profile then reads 7.96. So the profile's own reported decode axis understates its
     delivered decode by ~1.4x, and the better number only exists as a one-off command line
     recorded in a JSON blob.
  2. The recipe behind that better number is not a profile: it is the arm `prod25-stream`
     (`prod25` + `CGC_PREFILL_STREAM=1` + `CGC_GATHER_SLAB_CAP=256`) PLUS four bench-side
     conventions (`--batch 512`, `--ctx-size 4096`, `--warm-skip 64`, `--spec-type draft-mtp`).
     Nothing committed states that bundle, so "the best decode we have" lived in prose.
  3. The knob sets of `prefill250` and of that arm are IDENTICAL -- verified zero-GPU with
     `CGC_DUMP_ENV=1`: the resolved env diff is empty, and the only differences are
     `BATCH/UBATCH` (5632 vs the model default), `CTX` (8192 vs 4096) and the chat-template
     kwargs, all three of which the delivery decode cell overrides anyway. So the unified
     profile already exists; what was missing is a pinned record and an instrument.

This file is that instrument. It runs the two production axes and nothing else:

    prefill-house    --arms <profile>  --prompt 2048 --gen 16 --depths 0
                     (the shape behind the recorded 276.25 / 300.43; bar 250)
    decode-delivery  --arms <profile>  --prompt 0 --gen 128 --depths 512
                                       --batch 512 --ctx-size 4096
                                       --warm-skip 64 --spec-type draft-mtp
                     (the served shape: ctx 4096, MTP verify on, clock started after 64 tokens)

and, unless `--no-ref`, the SAME decode cell on the arm `prod25-stream`. That third launch is the
reproducibility anchor and it is the reason this tool is worth its runtime: a fresh number is
otherwise unanchored, because the decode axis is the one where the record already shows the same
arm reading 10.94 and 13.10 under the same NOMINAL label. If the profile and the anchor disagree,
the finding is "the delivery convention did not reproduce", not "the profile is faster/slower".

WHAT IT DOES NOT DO. It does not add a profile to `scripts/run_server.sh`. A new case branch
there would be a pure duplicate (see point 3 above) and it would silently lengthen every
`prod_matrix --profiles all` run on this box, including another session's. The profile is pinned
by its RESOLVED KNOBS in the JSON this tool writes, which is checkable and does not move.

THE GATE. Nothing launches unless (a) the OS thermal level is NOMINAL, re-read immediately before
the launch, and (b) usable memory is at or above `--min-usable-pct`. Both refusals are reported as
findings, not crashes, and a cell that leaves NOMINAL while running is marked `clean=false` and
`bar_ok=false` -- launching from NOMINAL is not sufficient, and on 2026-09-18 four arms that all
launched NOMINAL gave 10.80 / 10.34 / 8.92 / 8.58 in the order of how hot they got.

WINDOW CROSS-CHECK (added 2026-09-20). The gate above is bookkeeping: it checks that the box CLAIMS
to be fine. On 2026-09-20 18:1x a window passed all of it -- NOMINAL throughout, 54% usable, no other
session -- while running ~5x slow, and it silently corrupted two rounds of A/B. So this tool now also
checks the box PHYSICALLY: the prefill axis it already measures IS the sentinel shape (pp2048, the
same population as `window_sentinel_ref.json`), so its own prefill reading is compared against that
recorded band at zero extra GPU cost. A reading under the floor marks the whole table
`window=DEGRADED` in the JSON and prints a warning. The per-axis verdicts are deliberately NOT
changed -- a degraded window usually fails the bars on its own, and the cases this catches are the
relative comparisons people make FROM this table.

Exit code is 0 whenever the table was produced. A refused bar is a result.

USAGE
    python3 scripts/check/prod_profile.py                       # both axes + anchor
    python3 scripts/check/prod_profile.py --no-ref --reps 3
    python3 scripts/check/prod_profile.py --emit-spec           # zero GPU: the pinned knobs
    python3 scripts/check/prod_profile.py --dry-run             # zero GPU: the exact commands
"""
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

import thermal_pressure as tp                    # noqa: E402
from prefill_certifiability import mem_state      # noqa: E402
from llama_bench_matrix import resolve            # noqa: E402
from profile_duo import wait_nominal              # noqa: E402  one wait implementation, not two

MATRIX = HERE / "llama_bench_matrix.py"

# The one profile this file freezes. Named here as a default only -- `--profile` overrides it, and
# the arm is resolved through `llama_bench_matrix.resolve`, so the knobs always come from
# `run_server.sh` and never from this file.
PROFILE_DEFAULT = "prefill250"

# The anchor arm: the recorded source of 13.10 (09-19 16:58) and 10.94 (09-19 15:19).
REF_ARM = "prod25-stream"

BAR_PREFILL = 250.0
BAR_DECODE = 12.0

# The delivery conventions, one dict, so the markdown record and the command line cannot drift.
DECODE_CONVENTIONS = {
    "batch": "512",
    "ctx_size": 4096,
    "warm_skip": 64,
    "spec_type": "draft-mtp",
    "why": ("-c 4096 is what prod25 serves at (llama-bench would otherwise derive ~704 from "
            "p+n+d); --warm-skip 64 starts the clock after the pool has reached steady state, so "
            "n_gen is 64 tokens that are NOT averaged in; --spec-type draft-mtp turns on the MTP "
            "verify round (M=1..4) that the served path pays. Drop any of the three and the row "
            "stops being the delivered decode."),
}

PREFILL_SHAPE = ["--prompt", "2048", "--gen", "16", "--depths", "0"]
DECODE_SHAPE = ["--prompt", "0", "--gen", "128", "--depths", "512", "--batch", "512",
                "--ctx-size", "4096", "--warm-skip", "64", "--spec-type", "draft-mtp"]

# The files whose bytes a decode row depends on. Hashed so a row can be attributed to a build even
# after the working tree moves on -- the same reason `m123_oracle_gate` keeps an engine digest.
FINGERPRINT_FILES = [
    "llama-bench",
    "llama-server",
    "libllama.0.0.279.dylib",
    "libggml-metal.0.19.0.dylib",
    "libggml-base.0.19.0.dylib",
]


def fingerprint() -> dict:
    out = {}
    bindir = ROOT / "src" / "llama.cpp" / "build" / "bin"
    for name in FINGERPRINT_FILES:
        p = bindir / name
        if p.exists():
            h = hashlib.md5()
            with p.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            out[name] = h.hexdigest()[:16]
    return out


def arm_spec(arm: str) -> str:
    """`llama_bench_matrix` accepts a registry name or `PROFILE:ENV=VAL;...`.

    A bare name that is NOT in the registry is rejected in ~0.05 s, before any output is written,
    so the failure looks like a silent empty run. `PROFILE:` with an empty env list is the form
    that works for an arbitrary profile, so names outside the registry are routed through it.
    """
    import llama_bench_matrix as lbm
    return arm if arm in lbm.ARMS else arm + ":"


def run_axis(axis_label: str, arm: str, shape: list, reps: int, dry_run: bool,
             log_dir: str = "") -> dict:
    """One llama-bench launch. The row is read back from the child's OWN `--json`, never reparsed.

    A second parser for the same report is how two instruments come to disagree about the number
    they both claim to have measured (eng-mh-0038, the reason `thermal_pressure` is shared too).
    """
    tmp = Path(tempfile.mkdtemp(prefix="prod_profile_")) / "row.json"
    cmd = ([sys.executable, str(MATRIX), "--arms", arm_spec(arm), "--reps", str(reps)]
           + shape + ["--json", str(tmp)])
    rec = {"axis": axis_label, "arm": arm, "shape": " ".join(shape), "cmd": " ".join(cmd[1:])}
    if dry_run:
        return rec
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    rec["rc"] = proc.returncode
    rec["stdout_tail"] = proc.stdout[-1200:]
    rec["stderr_tail"] = proc.stderr[-800:] if proc.returncode else ""
    if not tmp.exists():
        rec["rows"] = []
        rec["incomplete"] = True
        return rec
    arms = json.loads(tmp.read_text())
    if not arms:
        rec["rows"] = []
        rec["incomplete"] = True
        return rec
    a0 = arms[0]
    rec["profile"] = a0.get("profile")
    rec["extra_env"] = a0.get("extra_env")
    rec["spec_type"] = a0.get("spec_type")
    # Read back from the child rather than echoed from the CLI: if the flag were ever dropped in
    # transit this is None in the record while `shape` still claims it, and the disagreement is
    # visible instead of silent.
    rec["spec_draft_n_max"] = a0.get("spec_draft_n_max")
    rec["warm_skip"] = a0.get("warm_skip")
    rec["ctx_size"] = a0.get("ctx_size")
    rec["batch"] = a0.get("batch")
    rec["incomplete"] = bool(a0.get("incomplete"))
    # [CGC 2026-09-24 P0/P1/P2 A/B] tee the child's FULL output when asked. The 1200-char tail
    # above is not enough to read the miss-attribution line (`CGC-SHAPE ... compulsory= capacity=
    # pread_us=`), which is how the three acceptance criteria get checked afterwards. Default ""
    # keeps the old behaviour (nothing written) so no existing caller changes.
    if log_dir:
        import re as _re
        safe = _re.sub(r"[^A-Za-z0-9._-]", "_", axis_label)
        d = Path(log_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / (safe + ".out")).write_text(proc.stdout or "")
        (d / (safe + ".err")).write_text(proc.stderr or "")
        rec["log_dir"] = str(d)
    th = a0.get("thermal") or {}
    rec["thermal"] = {"launch": th.get("launch"),
                      "worst": th.get("worst"),
                      "hist": tp.histogram(th.get("samples") or [])}
    rec["rows"] = [{"t/s": r.get("avg_ts"), "±": r.get("stddev_ts"),
                    "n_batch": r.get("n_batch"), "n_prompt": r.get("n_prompt"),
                    "n_gen": r.get("n_gen"), "n_depth": r.get("n_depth"),
                    "n_kept": r.get("n_kept"), "warm_skip": r.get("warm_skip"),
                    "ctx_override": r.get("ctx_override"),
                    "build_commit": r.get("build_commit"), "build_number": r.get("build_number"),
                    "n_reps": len(r.get("samples_ts") or [])}
                   for r in (a0.get("rows") or [])]
    return rec


def verdict(rec: dict, bar: float) -> dict:
    """NOMINAL at launch is a precondition, not the claim. The claim needs NOMINAL throughout."""
    if not rec.get("rows"):
        return {"ok": False, "why": "no row"}
    launch = ((rec.get("thermal") or {}).get("launch") or {}).get("label", "UNREADABLE")
    worst = ((rec.get("thermal") or {}).get("worst") or {}).get("label", "UNREADABLE")
    ts = rec["rows"][0]["t/s"]
    nominal_ok = launch == "NOMINAL"
    clean = nominal_ok and worst == "NOMINAL"
    return {"ok": bool(clean and ts is not None and ts >= bar),
            "clean": clean, "launch": launch, "worst": worst, "ts": ts,
            "bar": bar,
            "why": ("" if clean else
                    f"left NOMINAL (launch={launch}, worst={worst}) -> not a bar-meeting number")}


def window_crosscheck(recs: list) -> dict:
    """Physical health, from a reading this run already paid for.

    The prefill axis of this very tool is the sentinel shape (pp2048 at the profile's batch), so its
    t/s is a sample of the population recorded in `window_sentinel_ref.json`. No extra launch. The
    floor is `median * frac` (default 0.85): the recorded healthy samples span 275.65-300.43 (1.09x),
    so 15% under the median cannot belong to that population, while the degraded window measured on
    2026-09-20 read ~0.2x of it -- the line is not a knife edge.
    """
    ref_p = HERE / "window_sentinel_ref.json"
    pre = next((r for r in recs if r.get("axis") == "prefill-house"), None)
    if not pre or not ref_p.exists():
        return {"verdict": "UNKNOWN", "why": "no prefill axis in this run (or no reference band)"}
    row = (pre.get("rows") or [{}])[0]
    ts = row.get("t/s")
    if ts is None:
        return {"verdict": "UNKNOWN", "why": "prefill axis produced no t/s"}
    ref = json.loads(ref_p.read_text())
    frac = float(ref.get("frac", 0.85))
    floor = float(ref["median"]) * frac
    return {"verdict": "HEALTHY" if ts >= floor else "DEGRADED",
            "prefill_ts": ts, "ref_median": float(ref["median"]), "frac": frac,
            "floor": floor, "ratio_to_median": ts / float(ref["median"])}


def md_table(recs: list, bars: dict) -> str:
    lines = ["| axis | arm | shape | t/s | ± | n_batch | thermal launch / worst | bar | verdict |",
             "|---|---|---|---:|---:|---:|---|---:|---|"]
    for r in recs:
        v = r.get("verdict") or {}
        row = (r.get("rows") or [{}])[0]
        lines.append("| %s | %s | %s | %s | %s | %s | %s / %s | %s | %s |" % (
            r["axis"], r["arm"], r.get("shape", ""),
            f"{row.get('t/s', float('nan')):.2f}" if row else "—",
            f"{row.get('±', float('nan')):.2f}" if row else "—",
            row.get("n_batch", "—"),
            v.get("launch", "—"), v.get("worst", "—"),
            bars.get(r["axis"], "—"),
            "**PASS**" if v.get("ok") else ("refused: " + v.get("why", "?") if v else "—")))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="freeze + measure the production profile on both axes, NOMINAL-gated")
    ap.add_argument("--profile", default=PROFILE_DEFAULT)
    ap.add_argument("--ref-arm", default=REF_ARM,
                    help=f"decode anchor arm (default {REF_ARM})")
    ap.add_argument("--no-ref", action="store_true", help="skip the anchor launch")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--axes", default="prefill,decode")
    ap.add_argument("--json", default="")
    ap.add_argument("--md", default="")
    # 420 mirrors `profile_duo.py`'s default on purpose. 180 was tried first and is WRONG: the
    # recorded cooldown range is 35 s to well over 120 s, and this tool launches three times in a
    # row, so the second and third waits start on a machine the previous launch just heated. With
    # 180 the first sequence timed out into MODERATE and would have launched anyway (see below).
    ap.add_argument("--cooldown-timeout", type=float, default=420.0)
    ap.add_argument("--poll", type=float, default=5.0)
    ap.add_argument("--min-usable-pct", type=float, default=30.0)
    ap.add_argument("--emit-spec", action="store_true", help="zero GPU: print the pinned knobs")
    # [CGC 2026-09-23] k = max draft tokens. It was hard-pinned at the llama-bench default of 3
    # because this file wrote the decode shape as a literal list, even though
    # `llama_bench_matrix.py:544` already accepts `--spec-draft-n-max` and copies it into the row
    # (:435). That mattered the moment k became the variable under test: measuring k on the server
    # (HTTP) instrument and then applying the verdict to THIS cell is a cross-caliber extrapolation
    # (the same k=3 reads 10.43 t/s on the server and 12.57 here). Default None = omit the flag =
    # bit-identical command line to every row in the record so far.
    ap.add_argument("--spec-draft-n-max", type=int, default=None,
                    help="llama-bench --spec-draft-n-max (1..16); None = omit (bench default 3)")
    ap.add_argument("--dry-run", action="store_true")
    # [CGC 2026-09-24] tee each child's full stdout/stderr here. Needed by the P0/P1/P2 A/B
    # because the acceptance criteria are read off `CGC-SHAPE`/`final stats` lines that sit far
    # from the tail this tool keeps by default.
    ap.add_argument("--log-dir", default="")
    args = ap.parse_args()

    # k is part of the cell's identity, so it is validated before anything launches and recorded
    # beside the shape: a row that does not say which k it ran cannot be attributed afterwards.
    decode_shape = list(DECODE_SHAPE)
    if args.spec_draft_n_max is not None:
        if not 1 <= args.spec_draft_n_max <= 16:
            raise SystemExit("--spec-draft-n-max must be 1..16 (llama-bench:1388); got %d"
                             % args.spec_draft_n_max)
        if "--spec-type" not in decode_shape:
            raise SystemExit("--spec-draft-n-max is inert without --spec-type")
        decode_shape += ["--spec-draft-n-max", str(args.spec_draft_n_max)]

    spec = resolve(args.profile, {})
    bars = {"prefill-house": BAR_PREFILL, "decode-delivery": BAR_DECODE,
            "decode-delivery-anchor": BAR_DECODE}
    declared = {
        "profile": args.profile,
        "reps": args.reps,
        # None means "flag omitted", i.e. llama-bench's own default of 3 -- spelled out because a
        # reader comparing two rows has to be able to tell "default" from "measured at 3".
        "spec_draft_n_max": args.spec_draft_n_max,
        "prefill_cell": {"shape": PREFILL_SHAPE, "bar": BAR_PREFILL,
                         "why": "continuity with the recorded pp2048 captures (276.25 / 300.43)"},
        "decode_cell": {"shape": decode_shape, "bar": BAR_DECODE,
                        "conventions": DECODE_CONVENTIONS},
        "anchor_arm": None if args.no_ref else args.ref_arm,
        "resolved_env": spec["env"],
        "resolved_scalars": spec["scalars"],
        "server_argv": spec["server_argv"],
    }

    if args.emit_spec:
        txt = json.dumps(declared, ensure_ascii=False, indent=2)
        print(txt)
        if args.json:
            Path(args.json).write_text(txt)
            print(f"json -> {args.json}")
        return 0

    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    plan = []
    if "prefill" in axes:
        plan.append(("prefill-house", args.profile, PREFILL_SHAPE))
    if "decode" in axes:
        plan.append(("decode-delivery", args.profile, decode_shape))
        if not args.no_ref and args.ref_arm:
            plan.append(("decode-delivery-anchor", args.ref_arm, decode_shape))

    recs = []
    for label, arm, shape in plan:
        print(f"\n=== {label} (arm {arm}) ===", flush=True)
        mem = mem_state()
        print("    mem: usable %.1f%% (cached %.2f GiB, anon %.2f GiB)" %
              (mem["usable_pct"], mem["cached_gib"], mem["anon_gib"]), flush=True)
        if mem["usable_pct"] < args.min_usable_pct:
            recs.append({"axis": label, "arm": arm, "shape": " ".join(shape),
                         "refused": f"usable {mem['usable_pct']:.1f}% < {args.min_usable_pct}%",
                         "mem": mem, "verdict": None})
            print(f"    REFUSED: {recs[-1]['refused']} -- not launching", flush=True)
            continue
        w = wait_nominal(args.cooldown_timeout, args.poll)
        gate = tp.stamp()
        # A timed-out wait is a refusal, not a green light -- `profile_duo.py` does the same, and
        # the docstring above claims it. The first version of this file only *recorded* the wait and
        # launched regardless, which is how a MODERATE-launched row would have entered the record
        # wearing a NOMINAL-gated table's clothes. After a timed-out wait the machine is by
        # definition hot, so the honest outcome is "not measurable now", not a number.
        if not w["ok"]:
            recs.append({"axis": label, "arm": arm, "shape": " ".join(shape),
                         "refused": f"machine still {w['label']} after {w['waited_s']}s "
                                    f"(timeout {args.cooldown_timeout:.0f}s) -- not launching",
                         "waited": w, "gate": gate, "mem": mem, "verdict": None})
            print("    REFUSED: %s" % recs[-1]["refused"], flush=True)
            continue
        print("    gate: %s (waited %.0fs)  mem ok" % (gate["label"], w["waited_s"]), flush=True)
        rec = run_axis(label, arm, shape, args.reps, args.dry_run, args.log_dir)
        rec["gate"] = gate
        rec["waited"] = w
        rec["mem"] = mem
        if args.dry_run:
            print(f"    would run: {rec['cmd']}", flush=True)
            recs.append(rec)
            continue
        rec["verdict"] = verdict(rec, bars[label])
        recs.append(rec)
        row = (rec.get("rows") or [{}])[0]
        print("    rc=%s  t/s=%s  launch=%s worst=%s" % (
            rec.get("rc"), row.get("t/s"), rec["verdict"]["launch"], rec["verdict"]["worst"]),
            flush=True)

    if args.dry_run:
        print("\n(dry run -- nothing was launched)")
        for r in recs:
            print("  ", r["cmd"])
        return 0

    win = window_crosscheck(recs)
    out = {"tool": "prod_profile.py", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "declared": declared, "build": fingerprint(), "window": win, "axes": recs}
    if win["verdict"] == "DEGRADED":
        print("\n*** WINDOW DEGRADED: prefill read %.2f t/s against a recorded median of %.2f "
              "(floor %.2f, %.2fx of median) while the thermal key said NOMINAL. ***\n"
              "    Every number in this table was taken on a box that is physically slow, not a box "
              "that merely claims to be fine. Do not compare them against other runs."
              % (win["prefill_ts"], win["ref_median"], win["floor"], win["ratio_to_median"]),
              flush=True)
    elif win["verdict"] == "HEALTHY":
        print("\nwindow: HEALTHY (prefill %.2f t/s = %.2fx of the recorded median %.2f)"
              % (win["prefill_ts"], win["ratio_to_median"], win["ref_median"]), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=2))
        print(f"\njson -> {args.json}")
    table = md_table(recs, bars)
    print("\n" + table)
    if args.md:
        Path(args.md).write_text("# production profile\n\n" + table + "\n")
        print(f"md -> {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
