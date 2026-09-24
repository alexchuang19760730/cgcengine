#!/usr/bin/env python3
"""The k cost curve for MTP, from ONE harness in ONE window.

Why this file exists
--------------------
`docs/DRAFT_BUDGET_K_2026-09-22.md` fitted `cost(T) = C + c_tok*T` on TWO points that came from
TWO different harnesses (llama-bench OFF = 93.3 ms, server CGC-DECPROF k=3 = 247.98 ms). Two
points and two parameters is zero degrees of freedom: the arithmetic describes the data and
cannot test it. It also *predicted* something nobody had measured -- that at the day's accept
rate the best k is 1, not 3 (`k=3` is a net loss at alpha=0.654 because the break-even is 0.731).

This driver measures the missing points, and it measures four of them instead of one, because the
engine already prints the split that the n-gram ablation was going to be run to obtain:

    CGC-MTP-PERF type=draft-mtp calls_begin=... calls_draft=... gen_tokens=... acc_tokens=...
                 t_begin_ms=... t_draft_ms=... t_accept_ms=... acc_rate=...
                 gen_tok_per_round=... acc_tok_per_round=... emit_tok_per_round=... ms_per_round=...

(`common/speculative.cpp:3012`, gated on `CGC_MTP_PERF`; counters are CUMULATIVE, so the LAST
line of a run is the run total -- the code says so itself.)

Per round that line gives `t_begin` (spec refresh), `t_draft` (the MTP head's own forwards) and
`ms_per_round = t_draft/calls_draft`. `emit_tok_per_round` is tokens emitted per round, which is
what throughput is made of, so with the measured t/s this driver gets the mean round time WITHOUT
assuming a geometric acceptance model:

    round_ms = 1000 * emit_tok_per_round / tps
    residual = round_ms - t_begin/calls_draft - ms_per_round     <- the target verify forward

That residual is the quantity the ablation was designed to isolate ("is the per-draft-token cost
in the draft forward or in the verify path"). It is measured here directly, at three values of k,
so the *slope* answers it.

What is swept, and through which path
-------------------------------------
Exactly one env var, and it travels the single authoritative resolution path:

    CGC_SERVER_MTP_N_MAX={k}  ->  run_server.sh:436 SPEC_DRAFT_N_MAX  ->  :1243 --spec-draft-n-max
                              ->  llama_bench_matrix resolves the profile via
                                  `run_server.sh CGC_DUMP_ENV=1` and forwards the SPEC cell's
                                  flags from that resolved argv (`prod_matrix:353-371`)

Verified before writing this file (2026-09-22): the dump prints `ARG --spec-draft-n-max` /
`ARG 1` for k=1 and `ARG 3` for the default, and `ENV CGC_MTP_PERF=1` survives into the bench env
(`llama_bench_matrix:231,382`). So k is a config knob here, not a code change.

Arms (cell -> what it is):
  k=1..4    `decode-spec`  -- spec armed; the ONLY difference between arms is N_MAX
  off       `decode`       -- no --spec-type at all. NOT a fit point (see below)

FOUR k points and not three: the fit has two parameters, so three points give one degree of
freedom and four give two -- and the whole reason this exists is that a zero-dof fit was quoted as
if it had tested something.

The OFF arm is deliberately NOT a fit point: `decode` runs at `-b 512` while `decode-spec` runs at
`-b 8` (`prod_matrix:29-36`), and on this engine an MTP-off launch also drops the whole MTP env
block. It is reported as an EXTERNAL datum and labelled cross-config, because silently mixing
batches is one of the ways this project has produced its most expensive wrong numbers.

Pre-registered predictions (from DRAFT_BUDGET_K, before this ran):
  P1  best k is 1 (not 3)
  P2  the k=1 arm's measured mean length M-1 = alpha ~ 0.654   (k=1 makes alpha=m-1 model-free)
  P3  c_tok ~ 51.6 ms/token, C ~ 41.7 ms, i.e. T=1 -> ~93 ms
  P4  most of c_tok sits OUTSIDE the MTP head forward (kernel was 8.36 of the 51.56 ms)
  P5  OFF step ~ the k-sweep fit evaluated at T=1, IF the two configs are comparable

Usage
-----
    python3 scripts/check/mtp_k_sweep.py --selftest
    python3 scripts/check/mtp_k_sweep.py --reps 3 --logdir /tmp/mtp_k_sweep
    python3 scripts/check/mtp_k_sweep.py --cells k1,k2,k4      # resume: only the arms still missing
    python3 scripts/check/mtp_k_sweep.py --report-only --logdir /tmp/mtp_k_sweep

Results are appended to `<logdir>/arms.jsonl` after EVERY arm, so a stopped run keeps its data.
`--cells` skips arms already present for a rep, which is what makes a half-hour run resumable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# prod25 is the profile the S0 / two-axis numbers came from, so the OFF arm is pinned to them.
PROFILE = "prod25"
BASE_ENV = ("CGC_PREFILL_STREAM=1;CGC_GATHER_SLAB_CAP=256;"
            "CGC_SERVER_EXPERT_CACHE_BYTES=8589934592")

ARMS = ("k1", "k2", "k3", "k4", "off")
CELL = {a: ("decode" if a == "off" else "decode-spec") for a in ARMS}
K_OF = {"k1": 1, "k2": 2, "k3": 3, "k4": 4, "off": None}


# ---------------------------------------------------------------- engine identity

def engine_digest() -> dict:
    """Names+sizes+hashes of every artifact that decides the numerics.

    A glob rather than a list of versioned filenames: hard-coding `libllama.0.0.279.dylib` means
    the digest goes quietly stale on the next version bump, and a digest that cannot notice a
    rebuild is worse than no digest.
    """
    bindir = ROOT / "src" / "llama.cpp" / "build" / "bin"
    out = {}
    for p in sorted(bindir.glob("*.dylib")) + sorted(bindir.glob("*.so")) + [bindir / "llama-bench"]:
        if p.is_symlink() or not p.is_file():      # symlinks duplicate their target's bytes
            continue
        out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return out


def window() -> dict:
    """The box's state, using the SAME 'usable memory' definition prod_matrix gates on.

    Imported, not re-implemented: this repo has already paid once for having two definitions of
    'how much memory is free'.
    """
    from prefill_certifiability import mem_state
    m = mem_state()
    sw = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m["swap_used_mb"] = None
    if (mm := re.search(r"used = ([0-9.]+)M", sw)):
        m["swap_used_mb"] = float(mm.group(1))
    return m


# ---------------------------------------------------------------- the engine's own counters

def parse_mtp_perf(text: str) -> dict | None:
    """The LAST CGC-MTP-PERF line, as floats. None if absent -- absence is not a zero.

    Cumulative counters: the last line of a run is the run total (`common/speculative.cpp:2986`).
    """
    hits = [l for l in text.splitlines() if "CGC-MTP-PERF" in l]
    if not hits:
        return None
    kv: dict[str, float] = {}
    for tok in hits[-1].split("CGC-MTP-PERF", 1)[1].split():
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        try:
            kv[k] = float(v)
        except ValueError:
            kv[k] = v  # type=draft-mtp
    return kv


def spec_stderr(run: dict) -> str:
    """The bench run's full stderr, from the file llama_bench_matrix writes per arm+shape.

    `llama_bench_matrix:402` writes `llama_bench_<tag>_<shape>.stderr.log` into the workdir, which
    is the only place the spec impl's counters land in the bench path.
    """
    rdir = Path(run.get("dir") or "")
    logs = sorted(rdir.glob("*.stderr.log"))
    if not logs:
        return ""
    return "\n".join(p.read_text(errors="replace") for p in logs)


def pool_stats(stderr_text: str) -> dict:
    """The expert cache's own teardown counters, so an arm's speed can be read against the pool
    state it was served from (they differ per arm and they are the usual confounder)."""
    out = {}
    if (m := re.search(r"final stats: runtime requests=(\d+) hits=(\d+) misses=(\d+) \(hit rate ([0-9.]+)%\)",
                       stderr_text)):
        out.update(requests=int(m.group(1)), hits=int(m.group(2)), misses=int(m.group(3)),
                   hit_rate_pct=float(m.group(4)))
    if (m := re.search(r"miss attribution: compulsory=(\d+) capacity=(\d+)", stderr_text)):
        out.update(compulsory=int(m.group(1)), capacity=int(m.group(2)))
    if (m := re.search(r"effective_rate=([0-9.]+) MiB/s", stderr_text)):
        out["effective_rate_mib_s"] = float(m.group(1))
    return out


def samples_from(run: dict) -> tuple[list, dict, str]:
    """(samples_ts, entry, source).

    NOT `run["rows"]` alone: prod_matrix builds its rows from the MATRIX's stdout, and the matrix
    is silent when it is given `--json`, so on this path rows is empty while the run directory
    holds a complete entry. Found the hard way on 2026-09-22 -- the first arm reported '0 samples'
    for an arm that had in fact measured 11.13 t/s. The guard was right to refuse; the source was
    wrong.
    """
    rows = run.get("rows") or []
    if rows and rows[0].get("samples_ts"):
        return list(rows[0]["samples_ts"]), rows[0], "matrix rows"
    for p in sorted(Path(run.get("dir") or "").glob("*.json")):
        try:
            raw = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for e in (raw if isinstance(raw, list) else [raw]):
            if isinstance(e, dict) and e.get("samples_ts"):
                return list(e["samples_ts"]), e, p.name
    return [], {}, "none"


# ---------------------------------------------------------------- derivation

def derive(arm: str, samples: list, perf: dict | None, why_empty: str = "") -> dict:
    """Round time and its split, from measured quantities only. Refuses rather than guessing."""
    if len(samples) < 2:
        return {"invalid": f"only {len(samples)} sample(s); the platform value needs >=2"
                           + (f" -- {why_empty}" if why_empty else "")}
    warm = samples[1:]                       # house rule: drop rep 1 (llama-bench warmup)
    tps = st.median(warm)
    if tps <= 0:
        return {"invalid": f"non-positive t/s {tps}"}

    if arm == "off":
        # No spec impl exists, so there is no CGC-MTP-PERF line and one round is one token.
        return {"tps": tps, "samples": warm, "E": 1.0, "round_ms": 1000.0 / tps,
                "begin_ms": None, "draft_ms": None, "residual_ms": None,
                "note": "no --spec-type: E=1 by construction, not by measurement"}

    if perf is None:
        return {"invalid": "no CGC-MTP-PERF line -- was CGC_MTP_PERF=1 forwarded? "
                           "(absence is not a zero)"}
    for key in ("calls_draft", "emit_tok_per_round", "t_begin_ms", "t_draft_ms", "t_accept_ms"):
        if key not in perf:
            return {"invalid": f"CGC-MTP-PERF line has no {key!r}"}
    cd = perf["calls_draft"]
    if cd <= 0:
        return {"invalid": f"calls_draft={cd}: no draft rounds happened"}
    E = perf["emit_tok_per_round"]
    if E <= 1.0:
        return {"invalid": f"emit_tok_per_round={E:.3f} <= 1 -- nothing was ever accepted, "
                           f"so a round emitted only its bonus token"}

    round_ms = 1000.0 * E / tps
    begin_ms = perf["t_begin_ms"] / cd
    draft_ms = perf["t_draft_ms"] / cd
    residual_ms = round_ms - begin_ms - draft_ms
    return {
        "tps": tps, "samples": warm, "E": E, "round_ms": round_ms,
        "begin_ms": begin_ms, "draft_ms": draft_ms, "residual_ms": residual_ms,
        "acc_rate": perf.get("acc_rate"),
        "gen_tok_per_round": perf.get("gen_tok_per_round"),
        "acc_tok_per_round": perf.get("acc_tok_per_round"),
        "mean_len": 1.0 + (perf.get("acc_tok_per_round") or 0.0),
    }


def fit_cost(pts: list[tuple[float, float]]) -> dict:
    """Least squares cost(T) = C + c*T. Reports the degrees of freedom with the fit.

    n=2 must not print r2=1.00 as if it were agreement: with two points and two parameters the
    line passes through both by construction. Reporting that as a goodness of fit is exactly the
    'describe the data, do not test it' failure this tool exists to fix.
    """
    n = len(pts)
    if n < 2:
        return {"invalid": f"{n} point(s): a line needs 2"}
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return {"invalid": "all T identical"}
    c = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    C = my - c * mx
    ss_res = sum((y - (C + c * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    dof = n - 2
    return {"C_ms": C, "c_tok_ms": c, "n": n, "dof": dof,
            "r2": None if dof <= 0 or ss_tot == 0 else 1.0 - ss_res / ss_tot,
            "max_abs_resid_ms": max(abs(y - (C + c * x)) for x, y in zip(xs, ys))}


def rotate(seq: tuple, n: int) -> tuple:
    """Arm order for rep n. Rotation exists because launch ORDER has repeatedly moved readings on
    this box (carried swap), so a fixed order bakes position into the comparison."""
    n %= len(seq)
    return seq[n:] + seq[:n]


# ---------------------------------------------------------------- selftest

def selftest() -> int:
    bad = 0

    def check(name, got, want):
        nonlocal bad
        ok = got == want
        bad += 0 if ok else 1
        print(f"  [{'ok' if ok else 'FAIL'}] {name}" + ("" if ok else f"  got {got!r} want {want!r}"))

    PERF = ("CGC-MTP-PERF type=draft-mtp calls_begin=4 calls_draft=4 calls_accept=4 gen_tokens=12 "
            "acc_tokens=8 t_begin_ms=10.0 t_draft_ms=200.0 t_accept_ms=0.1 acc_rate=0.6667 "
            "gen_tok_per_round=3.000 acc_tok_per_round=2.000 emit_tok_per_round=3.000 "
            "ms_per_round=50.000\n")
    p = parse_mtp_perf(PERF)
    check("parse: calls_draft", p["calls_draft"], 4.0)
    check("parse: emit_tok_per_round", p["emit_tok_per_round"], 3.0)
    check("parse: type is a str", p["type"], "draft-mtp")
    check("parse: last line wins",
          parse_mtp_perf(PERF + PERF.replace("calls_draft=4", "calls_draft=9"))["calls_draft"], 9.0)
    check("parse: absent -> None, not 0", parse_mtp_perf("nothing here\n"), None)

    # 3 tokens per round at 10 t/s -> 300 ms/round; begin 10/4=2.5, draft 200/4=50, residual 247.5
    d = derive("k2", [1.0, 10.0, 10.0], parse_mtp_perf(PERF))
    check("derive: E from the engine", d["E"], 3.0)
    check("derive: round_ms", round(d["round_ms"], 3), 300.0)
    check("derive: begin per round", round(d["begin_ms"], 3), 2.5)
    check("derive: draft per round", round(d["draft_ms"], 3), 50.0)
    check("derive: residual per round", round(d["residual_ms"], 3), 247.5)
    check("derive: drop rep 1", d["tps"], 10.0)
    check("derive: off has no split", derive("off", [1.0, 10.0], None)["draft_ms"], None)
    check("derive: off round", round(derive("off", [1.0, 10.0], None)["round_ms"], 3), 100.0)
    check("derive: one sample -> invalid", "invalid" in derive("k1", [10.0], p), True)
    check("derive: no perf line -> invalid", "invalid" in derive("k1", [1.0, 10.0], None), True)
    check("derive: E<=1 -> invalid",
          "invalid" in derive("k1", [1.0, 10.0],
                              parse_mtp_perf(PERF.replace("emit_tok_per_round=3.000",
                                                          "emit_tok_per_round=1.000"))), True)
    check("derive: missing field -> invalid",
          "invalid" in derive("k1", [1.0, 10.0],
                              {k: v for k, v in p.items() if k != "t_draft_ms"}), True)

    f = fit_cost([(2.0, 144.86), (3.0, 196.42), (4.0, 247.98)])
    check("fit: c_tok recovered", round(f["c_tok_ms"], 2), 51.56)
    check("fit: C recovered", round(f["C_ms"], 2), 41.74)
    check("fit: dof", f["dof"], 1)
    check("fit: exact input -> r2 1", round(f["r2"], 6), 1.0)
    check("fit: two points have no dof", fit_cost([(2.0, 1.0), (3.0, 2.0)])["dof"], 0)
    check("fit: two points report r2 as None, not 1",
          fit_cost([(2.0, 1.0), (3.0, 2.0)])["r2"], None)
    check("fit: one point -> invalid", "invalid" in fit_cost([(2.0, 1.0)]), True)

    check("rotate 0", rotate(ARMS, 0), ARMS)
    check("rotate 1", rotate(ARMS, 1), ("k2", "k3", "k4", "off", "k1"))
    check("rotate wraps", rotate(ARMS, len(ARMS)), ARMS)
    check("off is not a spec cell", CELL["off"], "decode")
    check("k4 is a spec arm at k=4", (CELL["k4"], K_OF["k4"]), ("decode-spec", 4))
    ps = pool_stats("final stats: runtime requests=98128 hits=85337 misses=12791 (hit rate 87.0%)\n"
                    "miss attribution: compulsory=7868 capacity=4923\n"
                    "read shape: jobs=1 bytes=2 (x) us/job=1576 effective_rate=243 MiB/s\n")
    check("pool_stats: hit rate", ps["hit_rate_pct"], 87.0)
    check("pool_stats: capacity split", (ps["compulsory"], ps["capacity"]), (7868, 4923))
    check("pool_stats: effective rate", ps["effective_rate_mib_s"], 243.0)
    check("pool_stats: absent -> empty, not zeros", pool_stats("nothing"), {})
    print(f"selftest: {'PASS' if bad == 0 else f'{bad} FAILED'}")
    return 0 if bad == 0 else 1


# ---------------------------------------------------------------- the run

def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_done(path: Path) -> set:
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            # A record that came back INVALID is not "done": the arm still has no number.
            if r.get("rep") is not None and r.get("arm") and "invalid" not in r:
                done.add((r["rep"], r["arm"]))
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--logdir", default="/tmp/mtp_k_sweep")
    ap.add_argument("--cells", default=",".join(ARMS),
                    help="which arms to run (comma list of: " + ", ".join(ARMS) + ")")
    ap.add_argument("--extra-env", default="",
                    help="semicolon-separated extra env for every arm (e.g. the per-layer "
                         "instruments). Recorded per arm, so an instrumented run stays identifiable.")
    ap.add_argument("--min-usable-pct", type=float, default=30.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report-only", action="store_true",
                    help="analyse the existing arms.jsonl instead of launching anything")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if args.report_only:
        report(tuple(a.strip() for a in args.cells.split(",") if a.strip()), args.reps,
               Path(args.logdir) / "arms.jsonl")
        return 0

    import prod_matrix as pm
    from argparse import Namespace

    arms = tuple(a.strip() for a in args.cells.split(",") if a.strip())
    bad = [a for a in arms if a not in ARMS]
    if bad:
        raise SystemExit(f"unknown arm(s) {bad}; known: {ARMS}")

    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    jsonl = logdir / "arms.jsonl"
    done = load_done(jsonl)

    dig = engine_digest()
    log(f"engine digest: llama-bench={dig.get('llama-bench')} metal={dig.get('libggml-metal.0.19.0.dylib')}")
    log(f"arms={arms} reps={args.reps} profile={PROFILE} min_usable={args.min_usable_pct}%")
    log(f"resuming: {len(done)} arm(s) already in {jsonl}")
    if args.dry_run:
        for rep in range(args.reps):
            for a in rotate(arms, rep):
                log(f"  would run rep {rep+1} arm {a} -> cell {CELL[a]} "
                    f"env {BASE_ENV}{';CGC_SERVER_MTP_N_MAX=%d;CGC_MTP_PERF=1' % K_OF[a] if a != 'off' else ''}")
        return 0

    for rep in range(args.reps):
        for arm in rotate(arms, rep):
            if (rep, arm) in done:
                log(f"rep {rep+1} arm {arm}: already done, skipping")
                continue
            blockers = pm.gate()
            if blockers:
                log("REFUSING to launch -- the box is not ours:")
                for b in blockers:
                    log(f"  {b}")
                log(f"resume with: python3 scripts/check/mtp_k_sweep.py --reps {args.reps} "
                    f"--logdir {args.logdir} --cells {','.join(arms)}")
                return 1

            extra = f";{args.extra_env}" if args.extra_env else ""
            env = BASE_ENV + extra if arm == "off" else \
                f"{BASE_ENV}{extra};CGC_SERVER_MTP_N_MAX={K_OF[arm]};CGC_MTP_PERF=1"
            w0 = window()
            log(f"rep {rep+1}/{args.reps} arm {arm} (cell {CELL[arm]}) usable={w0.get('usable_pct'):.1f}% "
                f"swap={w0.get('swap_used_mb')}MB k={K_OF[arm]}")
            # prod_matrix's own args namespace -- the same one spec_onoff_ab uses, so the gate and
            # the launcher are the ones already in service.
            ns = Namespace(reps=args.reps, logdir=str(logdir), extra_env=env,
                           min_usable_pct=args.min_usable_pct, dry_run=False)
            run = pm.run_cell(PROFILE, CELL[arm], ns)

            if run.get("skipped"):
                log(f"  SKIPPED by the memory gate: {run.get('pre')}")
                continue
            if run.get("incompatible"):
                log(f"  INCOMPATIBLE: {run.get('reason')}")
                return 1
            samples, entry, source = samples_from(run)
            stderr_text = spec_stderr(run)
            perf = None if arm == "off" else parse_mtp_perf(stderr_text)
            why_empty = (f"rc={run.get('rc')} source={source}"
                         + (f" error={str(run.get('error'))[:160]}" if run.get("error") else ""))
            d = derive(arm, samples, perf, why_empty)
            th = run.get("thermal") or {}
            rec = {
                "rep": rep, "arm": arm, "k": K_OF[arm], "cell": CELL[arm],
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "engine_digest": dig, "profile": PROFILE, "extra_env": env,
                "samples_ts": samples, "samples_source": source,
                "avg_ts": entry.get("avg_ts"), "platform_ts_field": entry.get("platform_ts"),
                "n_kept": entry.get("n_kept"), "rc": run.get("rc"), "wall_s": run.get("wall_s"),
                "launch_thermal": (th.get("launch") or {}).get("label"),
                "worst_thermal": (th.get("worst") or {}).get("label"),
                "build_commit": run.get("build_commit"), "dir": run.get("dir"),
                "pool": pool_stats(stderr_text),
                "window_before": w0, "window_after": window(),
                **{k: v for k, v in d.items() if k != "samples"},
            }
            if run.get("error"):
                rec["error"] = str(run["error"])[:800]
            with jsonl.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            if "invalid" in d:
                log(f"  INVALID -- {d['invalid']}")
            else:
                log(f"  t/s={d['tps']:.2f} E={d['E']:.3f} round={d['round_ms']:.1f}ms "
                    f"begin={d['begin_ms']:.1f} draft={d['draft_ms']:.1f} residual={d['residual_ms']:.1f}"
                    if d["draft_ms"] is not None else f"  t/s={d['tps']:.2f} (no spec)")

    report(arms, args.reps, jsonl)
    return 0


def report(arms: tuple, reps: int, jsonl: Path) -> None:
    """Per-rep pairing FIRST, pooled medians second (and labelled).

    Two reps of the SAME arm moved 1.6x on this box (k2: 12.81 -> 8.21 t/s), which is bigger than
every effect being measured here. Pooling across reps therefore reports the drift as if it were
the treatment -- so each quantity is computed INSIDE a rep, where the arms are adjacent in time,
and the per-rep values are then summarised. The pooled number is printed too because it is what
a naive read produces, and keeping both visible is how the reader sees which one is which.
    """
    recs = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()] \
        if jsonl.exists() else []
    good = [r for r in recs if "invalid" not in r]
    if not good:
        print("\nno valid arm yet -- nothing to report")
        return
    by = {(r["rep"], r["arm"]): r for r in good}
    reps_seen = sorted({r["rep"] for r in good})
    spec = [a for a in ("k1", "k2", "k3", "k4") if any(r["arm"] == a for r in good)]

    print("\nper-rep t/s   (the same-rep OFF arm is the only reference sharing the machine state)")
    print("  rep " + "".join(f"{a:>14}" for a in arms))
    for rep in reps_seen:
        off = by.get((rep, "off"))
        row = f"  {rep + 1:>3} "
        for a in arms:
            r = by.get((rep, a))
            if not r:
                row += f"{'-':>14}"
            elif a == "off" or not off:
                row += f"{r['tps']:>14.2f}"
            else:
                row += f"{r['tps']:>8.2f}{'%+.2fx' % (r['tps'] / off['tps']):>6}"
        print(row)

    print("\nMTP sign per arm: ratio to the same-rep OFF (drift-robust), all reps listed")
    for a in arms:
        if a == "off":
            continue
        rs = [by[(rep, a)]["tps"] / by[(rep, "off")]["tps"]
              for rep in reps_seen if (rep, a) in by and (rep, "off") in by]
        if rs:
            print(f"  {a:>4}: median {st.median(rs):.3f}x  [" + ", ".join("%.3f" % x for x in sorted(rs)) + "]")

    print("\nper-rep fit  round_ms = C + c_tok*T   (T = k+1, spec arms only, same batch/env)")
    fits, dfits, rfits = [], [], []
    for rep in reps_seen:
        pts = [(K_OF[a] + 1.0, by[(rep, a)]["round_ms"]) for a in spec if (rep, a) in by]
        f = fit_cost(pts)
        if "invalid" in f:
            print(f"  rep {rep + 1}: {f['invalid']} ({len(pts)} spec point(s))")
        else:
            fits.append(f)
            print(f"  rep {rep + 1}: C={f['C_ms']:7.2f}  c_tok={f['c_tok_ms']:7.2f}  "
                  f"n={f['n']} dof={f['dof']}  "
                  f"r2={'n/a' if f['r2'] is None else round(f['r2'], 3)}  "
                  f"maxres={f['max_abs_resid_ms']:.1f} ms")
        for name, out in (("draft_ms", dfits), ("residual_ms", rfits)):
            p = [(K_OF[a], by[(rep, a)][name]) for a in spec
                 if (rep, a) in by and by[(rep, a)][name] is not None]
            ff = fit_cost(p)
            if "invalid" not in ff:
                out.append(ff)
    if fits:
        print(f"  MEDIAN over reps: C={st.median(f['C_ms'] for f in fits):.2f} ms  "
              f"c_tok={st.median(f['c_tok_ms'] for f in fits):.2f} ms/round-token")

    pooled = fit_cost([(K_OF[a] + 1.0, st.median(r["round_ms"] for r in good if r["arm"] == a))
                       for a in spec])
    if "invalid" not in pooled:
        print(f"\nPOOLED over reps (contaminated by the inter-rep drift above -- for contrast only):"
              f"\n  C={pooled['C_ms']:.2f} c_tok={pooled['c_tok_ms']:.2f} "
              f"r2={'n/a' if pooled['r2'] is None else round(pooled['r2'], 3)} "
              f"dof={pooled['dof']} maxres={pooled['max_abs_resid_ms']:.1f} ms")

    print("\nP4: where does the per-token cost live?  (median of the per-rep slopes vs k)")
    for name, ff in (("draft head forward", dfits), ("residual (target verify + host)", rfits)):
        if ff:
            print(f"  {name:>34}: {st.median(x['c_tok_ms'] for x in ff):+7.2f} ms per extra k  "
                  f"[dof {min(x['dof'] for x in ff)}]")
    print("  predicted: c_tok ~51.6 ms/token, of which the kernel (mul_mat_id) was ~8.4 ms")

    med = {a: st.median(by[(rep, a)]["tps"] for rep in reps_seen if (rep, a) in by)
           for a in arms if any((rep, a) in by for rep in reps_seen)}
    if spec:
        best = max(spec, key=lambda a: med[a])
        print(f"\nP1: best k by median t/s = {best}   "
              f"({'as predicted' if best == 'k1' else 'NOT as predicted: DRAFT_BUDGET_K predicted k=1'})")
        print("    NOTE: the optimal k is a function of the workload's accept rate. llama-bench has no"
              "\n    prompt, so its alpha is not the server's -- read the COST LAW here, not the ranking.")
    k1s = [by[(rep, "k1")]["E"] - 1.0 for rep in reps_seen if (rep, "k1") in by]
    if k1s:
        print(f"P2: alpha from the k=1 arm (model-free: alpha = E-1) = {st.median(k1s):.3f}   "
              f"predicted ~0.654 from the SERVER regime")
    if "off" in med:
        off_ms = st.median(by[(rep, "off")]["round_ms"] for rep in reps_seen if (rep, "off") in by)
        print(f"P5: OFF measured {off_ms:.1f} ms/token here; DRAFT_BUDGET_K assumed 93.3 ms from a"
              f"\n    different instrument (prod25 server). CROSS-CONFIG -- a sanity check, not agreement.")


if __name__ == "__main__":
    raise SystemExit(main())
