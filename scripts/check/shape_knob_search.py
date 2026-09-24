#!/usr/bin/env python3
"""Shape-knob search: drive the engine through `llama-shape-knob.{h,cpp}` and READ BACK what ran.

WHY THIS EXISTS. The two functions in `src/llama.cpp/src/llama-shape-knob.cpp` move every shape
variable into one env-facing table (`cgc_shape_knobs()`, INPUT) and print one machine-parsable
line per phase (`cgc_shape_report_{init,final}`, OUTPUT). What was missing after those existed was
the thing that actually searches -- this file.

The point of the pair is that a row is now ADMISSIBLE or VOID on its own merits, without any
second instrument:

    CGC-SHAPE v=1 phase=final M=4 M_stat=APPLIED width=4 union=32 fits=1
        pool_cap_slots=143 slots_layer=143 n_layer=41
        req=... misses=... zero_mapped=0 verify_refused=0 inv_viol=0 final_counters=1

Those three counters are why a search can be run unattended: `zero_mapped` is a selected expert
whose contribution was silently dropped, `verify_refused` is a fast-path step refused for cold
experts, `inv_viol` breaks batch invariance. ANY nonzero means the row did different work, so its
t/s is not a measurement of anything -- it is exactly how a configuration can look fast.
`M_stat != APPLIED` is the other void condition: a knob that did not land looks like a knob that
was tried and found useless (the failure shape this repo keeps paying for).

WHAT IT DOES NOT DO. It does not multiply instruments (no cross-instrument t/s), it does not
average Cells whose `M_stat` differs, and it does not compare a row to an anchored reference when
the row did not run the anchored engine (see the GDN rule below).

GDN IS NOT ABSENT -- READ THIS BEFORE REPEATING THE OLD SENTENCE. An earlier revision of this
docstring claimed "the GDN axis prints ABSENT in every row because there is no chunked
GatedDeltaNet operator in this tree". That was wrong, and it was written from grepping for the
`LLM_FUSED_OP_GDN_CH` enum instead of reading the operator. This tree DOES have the fused
GatedDeltaNet: `ggml_gated_delta_net`, with a Metal pipeline named by
`ggml-metal-device.cpp:kernel_gated_delta_net_<type>_<nsg>`, where `K` (the number of state
snapshots) is a FUNCTION CONSTANT -- so the same kernel already covers both the T=1 autoregressive
recurrence and any chunked roll-up. Three implementations exist and are dispatched by
`cparams.fused_gdn_ar` / `cparams.fused_gdn_ch` (see `models/delta-net-base.cpp`); this model uses
the fused one by default.

So what the knob can do is SELECT AN IMPLEMENTATION, never create one. `CGC_SHAPE_GDN_AR=0` turns
the fused recurrence off and lets the manual AR/chunking graphs run. That is an ABLATION -- it
changes what is computed, so the row leaves the bit-identity protection the M1/M2/M3 gate certifies
and its t/s is NOT comparable to an anchored number. Rows in that state are reported as ABLATED,
not as "slower". If you want to know what the fused op costs, run the ablation and READ IT AS A
COST, never as a candidate: the engine prints `bitident=NO` on those rows on purpose.

CELLS ARE BRACKETED. A single llama-bench launch is not evidence on this box: the reference profile
has read 10.94 and 13.10 t/s under the same NOMINAL label, and one sweep's reference series went
11.72 -> 10.83 -> 5.22. Every cell therefore runs between two reference launches, and the verdict
is taken against the reference INTERPOLATED TO THE INSTANT THE CELL RAN. That interpolation is done
in LOG space (`scripts/check/crossover_estimate.py`) because the machine acts on throughput
multiplicatively: two refs reading 10.83 and 5.22 differ from their geometric mean by 6.8%, which
is more than twice this repo's 3% threshold, so taking the wrong mean alone can invent a finding.
A row whose two references disagree by more than MAX_DRIFT_PCT is UNANCHORED either way.

Usage:
    shape_knob_search.py --plan
    shape_knob_search.py --run --reps 3 --grid width,ncb
    shape_knob_search.py --run --grid width --width 2,3,4 --target 25
    shape_knob_search.py --selftest
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "scripts" / "check" / "llama_bench_matrix.py"

sys.path.insert(0, str(ROOT / "scripts" / "check"))
try:
    import thermal_pressure as tp  # noqa: F401  (shared: window band comes from the same probe)
except Exception:  # the search still runs; it just cannot mark rows NOMINAL
    tp = None
try:
    import crossover_estimate as cx  # noqa: F401  (log-space reference interpolation)
except Exception:  # keep searching even if that module is unavailable; REF_MODE falls back
    cx = None
try:
    import server_window as sw  # noqa: E401  the SHARED admission bar: imported, never retyped
except Exception:  # the search still runs; it just cannot say whose box the rows came from
    sw = None

# The delivered decode cell. Same conventions as prod_profile.py's decode axis; changing any of
# them stops the row being comparable to the delivery record.
DECODE_SHAPE = ["--prompt", "0", "--gen", "128", "--depths", "512", "--batch", "512",
                "--ctx-size", "4096", "--warm-skip", "64", "--spec-type", "draft-mtp"]

DEFAULT_PROFILE = "prod25-stream"

# The three counters that make a row VOID because the work was not the same work.
VOID_COUNTERS = ("zero_mapped", "verify_refused", "inv_viol")
OK_STATUS = ("APPLIED", "DEFAULTED")

# Repo-wide decision threshold (see .workbuddy/memory/MEMORY_PERF.md): below this single-arm noise
# makes a difference unreadable (llama-bench decode single-arm noise floor is ~+-27%).
THRESHOLD_PCT = 3.0


# ---------------------------------------------------------------------------------------------
# parsing the OUTPUT function's line. `key=value` pairs separated by spaces; values never contain
# spaces, which is deliberate so this stays one regex instead of a grammar.
# ---------------------------------------------------------------------------------------------
SHAPE_RE = re.compile(r"^CGC-SHAPE v=(?P<v>\d+) phase=(?P<phase>\w+) (?P<body>.+)$")


def parse_shape_line(line: str) -> dict:
    m = SHAPE_RE.match(line.strip())
    if not m:
        return {}
    out = {"v": int(m.group("v")), "phase": m.group("phase")}
    bad = []
    for tok in m.group("body").split():
        if "=" not in tok:
            bad.append(tok)
            continue
        k, v = tok.split("=", 1)
        out[k] = int(v) if re.fullmatch(r"-?\d+", v) else (
            float(v) if re.fullmatch(r"-?\d+\.\d+", v) else v)
    # A token without '=' means our format changed and this parser is now reading a different
    # dialect. Refuse rather than silently dropping a field the verdict depends on.
    if bad:
        out["_unparsed"] = bad
    return out


def harvest_shape_logs(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        rec = parse_shape_line(line)
        if rec:
            rows.append(rec)
    return rows


def pick_final(rows: list[dict]) -> dict:
    finals = [r for r in rows if r.get("phase") == "final"]
    return finals[-1] if finals else {}


# ---------------------------------------------------------------------------------------------
# the INPUT side: turning a knob choice into the env the matrix passes through run_server.sh
# ---------------------------------------------------------------------------------------------
def thermal_label() -> str:
    """Read OS thermal pressure the same way the repo's other probes do (no second dialect)."""
    try:
        out = subprocess.run(["notifyutil", "-g", "com.apple.system.thermalpressure"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return "UNKNOWN"
    for line in out.splitlines():
        if "thermalpressure" in line:
            tok = line.split()[-1]
            return {"0": "NOMINAL", "1": "heavy", "2": "HEAVY", "3": "CRITICAL",
                    "5": "trapping", "6": "sleeping"}.get(tok, f"level{tok}")
    return "UNKNOWN"


def free_pct() -> float:
    try:
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True,
                             timeout=15).stdout
        for line in out.splitlines():
            if "System-wide memory free percentage" in line:
                return float(line.split(":")[1].strip().rstrip("%"))
    except Exception:
        pass
    return -1.0


def wait_settle(settle: int, min_free: float, budget_s: int, why: str) -> dict:
    """Wait for the box to come back before the NEXT cell.

    Tonight's three sweeps all decayed monotonically inside one run (11.72 -> 10.83 -> 5.22 t/s on
    the third), which is larger than any effect worth measuring. Launching cell after cell without
    a settle turns the sweep into a measurement of how much the machine degraded, with the knobs
    aliased into the slope. Wait for NOMINAL + enough free memory, give up after `budget_s`.
    """
    waited, step = 0, 15
    while waited < max(0, budget_s):
        time.sleep(min(step, max(1, budget_s - waited)))
        waited += step
        lab, free = thermal_label(), free_pct()
        if lab == "NOMINAL" and (free < 0 or free >= min_free):
            time.sleep(settle)
            return {"ok": True, "waited_s": waited, "thermal": lab, "free_pct": free}
    return {"ok": False, "waited_s": waited, "thermal": thermal_label(), "free_pct": free_pct(),
            "why": why}


def env_spec(knobs: dict[str, str]) -> str:
    return ";".join(f"{k}={v}" for k, v in knobs.items())


def arm_env(profile: str, knobs: dict[str, str]) -> tuple[str, dict[str, str], list[str]]:
    """Resolve `--arms <profile>` into (base profile, env, conflicts).

    The matrix's `PROFILE:ENV=VAL;...` form takes a run_server.sh PROFILE on the left, not a matrix
    arm name: passing the arm name made `run_server.sh` reject it (`must be off|qa-zh|...`), which
    surfaced as three invisible rc=1 cells. So the arm's own registry env is looked up here and
    merged, keeping the registry authoritative for everything the arm already sets.
    """
    try:
        import llama_bench_matrix as lbm
        base, arm_env = lbm.ARMS.get(profile, (profile, {}))
    except Exception:
        base, arm_env = profile, {}
    conflicts = [f"{k}: arm={arm_env[k]} knobs={knobs[k]}" for k in knobs
                 if k in arm_env and arm_env[k] != knobs[k]]
    return base, {**arm_env, **knobs}, conflicts


def arm_arg(profile: str, knobs: dict[str, str]) -> str:
    base, env, _ = arm_env(profile, knobs)
    return f"{base}:" + env_spec(env)


def cell_name(knobs: dict[str, str]) -> str:
    if not knobs:
        return "ref"
    return "_".join(f"{k}{v}" for k, v in knobs.items()).replace("=", "")


# ---------------------------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------------------------
GRIDS = {
    # Width (columns per forward). Upper bound is NOT arithmetic: the pool cannot route more than
    # usable_slots experts per step, and union is 8*M, so anything above floor(143/8) = 17 cannot
    # be served. Values above that are included so the CLAMPED path is exercised and VISIBLE.
    "width": [("CGC_SHAPE_M", m) for m in ("2", "3", "4", "6", "8", "12", "17", "24")],
    # Graph command-buffer split. Metal itself prints a warning above 2.
    "ncb": [("CGC_N_CB", n) for n in ("1", "2")],
    # Dense-vs-pool split of the same 16 GB. 4 GiB was already crudely A/B'd (IO_PATH_AB); it is
    # here so the PAIRING that experiment lacked can settle whether miss cost is load-bearing.
    # The ladder is spaced for the closed-loop search, not for exhaustiveness: with four arms and a
    # shared-reference block it costs k+2 launches per block, so width of the ladder is bought with
    # real wall clock. 3 GiB is below the 71-slot point deliberately -- it is the "starve the pool
    # and give the RAM to the dense part" end of the question the -knob counter accumulation asked.
    # CORRECTED 2026-09-22 23:5x. This axis used `LLAMA_ARG_EXPERT_CACHE`, which is NOT a name the
    # launcher reads: `run_server.sh` derives `CGC_EXPERT_CACHE_BYTES` from `BUDGET`, and `BUDGET`
    # comes from `CGC_SERVER_EXPERT_CACHE_BYTES` (run_server.sh:468). The old key was therefore
    # dropped by `resolve()`, and every budget cell ran on the SAME 8 GiB pool -- proved from the
    # cells themselves: all 10 arm cells and all 6 refs in
    # `Backup/shape_search/prod25-stream_-loop_budget/*/...stderr.log` print `n_slots=143`, which is
    # the 8 GiB geometry (10 GiB would be 179 slots/layer). So that sweep measured four copies of
    # the baseline, and the +/-11% between "3 GiB" and "8 GiB" was machine noise, not a budget
    # effect. `check_reachability()` below now refuses any cell whose resolved env is identical to
    # the reference cell's, which is the generic form of this bug.
    #
    # 8 GiB is NOT in the ladder, for a reason `check_reachability` taught us: it IS the reference
    # cell, because every bracket's two refs run exactly that config. Keeping it as an arm spends a
    # full cell of wall clock re-measuring the baseline and then reads 0% as if it were a finding.
    "budget": [("CGC_SERVER_EXPERT_CACHE_BYTES", str(int(b * 1024 ** 3)))
               for b in (3.0, 4.0, 6.0)],
}


def build_grid(names: list[str], overrides: dict[str, list[str]]) -> list[dict[str, str]]:
    axes: list[list[tuple[str, str]]] = []
    for n in names:
        vals = overrides.get(n)
        if vals:
            axes.append([(NAME_OF_AXIS[n], v) for v in vals])
        else:
            axes.append(GRIDS[n])
    if not axes:
        return [{}]
    cells: list[dict[str, str]] = [{}]
    for ax in axes:
        cells = [{**c, k: v} for c in cells for (k, v) in ax]
    return cells


NAME_OF_AXIS = {"width": "CGC_SHAPE_M", "ncb": "CGC_N_CB",
                "budget": "CGC_SERVER_EXPERT_CACHE_BYTES"}


# ---------------------------------------------------------------------------------------------
# reachability: does this cell actually DIFFER from the reference cell?
# ---------------------------------------------------------------------------------------------
# The launcher is an allowlist. `llama_bench_matrix.run_arm` builds the child env as
# `dict(os.environ)` updated with the env `resolve()` got back from `run_server.sh`, and our knobs
# arrive on the command line -- so anything `run_server.sh` does not put into SERVER_ENV never
# reaches llama-bench. A knob can be perfectly implemented in C++ and still be unmovable here, and
# the symptom is the worst possible one: the cell runs, produces a row, and is silently a repeat of
# the reference (this is exactly the `LLAMA_ARG_EXPERT_CACHE` bug recorded above).
#
# The check is deliberately agnostic about HOW the launcher names things: instead of asking whether
# our key survived, it diffs the whole resolved env against the reference cell's. A real knob always
# moves something (possibly under a different name, which the printed diff shows).
def resolved_env(profile: str, knobs: dict[str, str]) -> dict:
    """The env llama-bench will actually be launched with (zero GPU -- a run_server.sh query)."""
    try:
        import llama_bench_matrix as lbm
    except Exception:
        return {}
    base, env, _c = arm_env(profile, knobs)
    try:
        return lbm.resolve(base, env)["env"]
    except Exception:
        return {}


def check_reachability(profile: str, cells: list[dict[str, str]]) -> list[tuple[dict, dict, bool]]:
    """[(cell, env-diff vs reference, reachable), ...] for every cell in the plan."""
    base_env = resolved_env(profile, {})
    out = []
    for c in cells:
        env = resolved_env(profile, c)
        diff = {k: (base_env.get(k), env.get(k))
                for k in sorted(set(base_env) | set(env))
                if base_env.get(k) != env.get(k)}
        out.append((c, diff, bool(diff)))
    return out



# ---------------------------------------------------------------------------------------------
# running one cell through the existing matrix (one source of truth for the launch)
# ---------------------------------------------------------------------------------------------
def run_cell(tag: str, profile: str, knobs: dict[str, str], reps: int, workdir: Path,
             dry_run: bool) -> dict:
    tmp = workdir / f"{tag}.json"
    base, env, conflicts = arm_env(profile, knobs)
    if conflicts:
        # Silent override would change two things at once and neither would be attributable.
        print(f"    !! refusing cell {tag}: knobs collide with the arm's own env: {conflicts}")
        return {"tag": tag, "knobs": knobs, "refused": True, "conflicts": conflicts}
    # Its OWN workdir: the matrix names its per-arm stderr log from the arm spec string, which is
    # 90+ characters of `PROFILE:ENV=VAL;...` and therefore unmatchable by globbing our tag. Giving
    # each cell a directory makes "the stderr of this cell" unambiguous instead of guessed.
    cdir = workdir / tag
    cdir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(MATRIX), "--arms", arm_arg(profile, knobs), "--reps", str(reps),
           *DECODE_SHAPE, "--json", str(tmp), "--workdir", str(cdir)]
    rec = {"tag": tag, "knobs": knobs, "base_profile": base, "arm_env": env,
           "cmd": " ".join(cmd[1:])}
    if dry_run:
        return rec
    # Wall-clock bounds of the launch. These exist because the reference value a cell must be read
    # against is the one at the instant it RAN, and a cell takes several times as long as a
    # reference does -- without timestamps the bracketing can only use the index midpoint, which is
    # the wrong instant and is why crossover_estimate.py falls back to it with a printed warning.
    rec["t_start"] = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    rec["rc"] = proc.returncode
    rec["t_end"] = time.time()
    rec["stdout_tail"] = proc.stdout[-800:]
    rows = []
    if tmp.exists():
        try:
            arms = json.loads(tmp.read_text())
            if arms:
                rows = arms[0].get("rows") or []
                rec["thermal"] = arms[0].get("thermal")
        except json.JSONDecodeError:
            rows = []
    rec["rows"] = rows
    # The shape lines are in the matrix's own stderr log for this tag. Depending on call order it
    # can be more than one file, so scan by glob and take the newest `final` line.
    finals = []
    for p in sorted(cdir.glob("*.stderr.log")):
        finals.extend(harvest_shape_logs(p))
    rec["shape"] = pick_final(finals)
    inits = [r for r in finals if r.get("phase") == "init"]
    rec["shape_init"] = inits[-1] if inits else {}
    # The GDN line is its own phase. It carries `ablated`, which says whether this cell left the
    # anchored engine -- that must be visible here, because everything downstream reads rec["shape"].
    gdn_rows = [r for r in finals if r.get("phase") == "gdn"]
    rec["shape_gdn"] = gdn_rows[-1] if gdn_rows else {}
    return rec


def mid_time(rec: dict) -> float | None:
    t0, t1 = rec.get("t_start"), rec.get("t_end")
    return 0.5 * (t0 + t1) if (t0 and t1) else None


def row_ts(row: dict) -> tuple[float, float]:
    return (row.get("avg_ts") or 0.0), (row.get("stddev_ts") or 0.0)


# ---------------------------------------------------------------------------------------------
# verdict: the rules that decide whether a faster cell is a CLAIM
# ---------------------------------------------------------------------------------------------
def verdict(cell_rec: dict, ref_ts: float, target: float, clean: bool = True) -> dict:
    v: dict = {"tag": cell_rec["tag"], "knobs": cell_rec.get("knobs", {})}
    v["clean"] = clean
    shape = cell_rec.get("shape") or {}
    v["shape"] = shape

    if not shape:
        v["state"] = "VOID"
        v["why"] = "no CGC-SHAPE line was recovered from this cell's stderr"
        return v

    st = str(shape.get("M_stat", "-"))
    v["M_stat"] = st
    if st not in OK_STATUS:
        v["state"] = "VOID"
        v["why"] = f"width knob did not land as requested (M_stat={st})"
        return v

    # GDN: the fused operator EXISTS in this tree (see the module docstring -- the "axis is ABSENT"
    # sentence this used to print was wrong). What decides here is only whether the cell stayed
    # INSIDE the anchored engine. An ablated cell ran a different computation: its t/s is a cost
    # figure for the fusion, never a candidate and never a verdict against an anchored reference.
    gdn_row = cell_rec.get("shape_gdn") or {}
    ablated = int(shape.get("gdn_ablated", gdn_row.get("ablated", 0)) or 0)
    v["gdn_ablated"] = ablated
    v["gdn_fused"] = (f"{gdn_row.get('fused_ar', '-')}/{gdn_row.get('fused_ch', '-')}"
                      if gdn_row else "-")
    if ablated:
        v["state"] = "ABLATED"
        v["why"] = (f"CGC_SHAPE_GDN_AR/CH=0: {gdn_row.get('note', 'fusion_off')} -- this row is "
                    f"bitident=NO, so it measures what the fusion COSTS and cannot be read against "
                    f"{ref_ts:.2f} t/s")
        return v
    gdn_stat = str((cell_rec.get("shape_init") or {}).get("gdn_stat", "DEFAULTED"))
    if gdn_stat == "REFUSED":
        v["state"] = "VOID"
        v["why"] = "CGC_SHAPE_GDN_* was not 0 or 1: the engine refused it and kept the default"
        return v

    for c in VOID_COUNTERS:
        if int(shape.get(c, 0)) != 0:
            v["state"] = "VOID"
            v["why"] = f"{c}={shape.get(c)}: the run did not do the same work -- timing void"
            return v

    if int(shape.get("fits", 1)) == 0:
        v["state"] = "VOID"
        v["why"] = (f"union={shape.get('union')} > routable slots: this is the silent-gather "
                    f"configuration (documented 'buffer is nil' shape), not a candidate")
        return v

    rows = cell_rec.get("rows") or []
    if not rows:
        v["state"] = "VOID"
        v["why"] = "llama-bench produced no row for this cell"
        return v
    ts, sd = row_ts(rows[0])
    v["ts"] = round(ts, 2)
    v["sd"] = round(sd, 2)
    v["n_gen"] = int(rows[0].get("n_gen") or 0)

    if ref_ts <= 0:
        v["state"] = "UNANCHORED"
        v["why"] = "no usable reference cell in this sweep"
        return v

    if not clean:
        # NOMINAL at launch is not sufficient: four arms launched NOMINAL read 10.80/10.34/8.92/8.58
        # in the order of how hot they got. A thermally dirty sweep has no comparable rows.
        v["state"] = "UNANCHORED"
        v["why"] = "window left NOMINAL during the sweep -- rows are not comparable"
        return v

    delta = 100.0 * (ts - ref_ts) / ref_ts
    v["delta_pct"] = round(delta, 2)
    v["ref_ts"] = round(ref_ts, 2)
    if abs(delta) < THRESHOLD_PCT:
        v["state"] = "NO_EVIDENCE"
        v["why"] = f"|{delta:.2f}%| < {THRESHOLD_PCT}% single-arm threshold"
    elif delta > 0:
        v["state"] = "FASTER"
        v["why"] = f"+{delta:.2f}% over {ref_ts:.2f} t/s"
    else:
        v["state"] = "SLOWER"
        v["why"] = f"{delta:.2f}%"

    v["gap_to_target"] = round(target - ts, 2) if target else None
    return v


# ---------------------------------------------------------------------------------------------
# Whose box were these rows taken on?
#
# This harness could always see its OWN insulation failure -- the settle loop above watches thermal
# and free % per cell -- but it never asked whether the box was somebody else's first, and that
# failure is invisible in the numbers: a sweep launched next door still prints tidy rows. The
# concrete case is tonight: a joint probe started into 6.99 GiB usable and read a 57.7%-wide null
# cell, then a neighbour's llama-server came up at 8.1 GiB and would have done the same to any
# sweep. `server_window` is what every other launcher asks; a third definition of "quiet" here is
# the defect this repo has already paid for twice.
# ---------------------------------------------------------------------------------------------

def judge_window(before: dict, after: dict | None) -> dict:
    """Read two `server_window.decision()` samples into one class.

    `before` decides whether a sweep may start; `after` says whether the machine stayed ours while
    it ran. The second question cannot be answered in advance, so it is asked again.
    """
    out = {"class": "clean", "admits_start": bool(before.get("admits")),
           "refused_by": list(before.get("refused_by") or []),
           "reclaimable_mb": before.get("reclaimable_mb"), "need_mb": before.get("need_mb"),
           "admits_end": None, "reclaimable_end_mb": None}
    if after is None:
        out["class"] = "clean" if out["admits_start"] else "busy-overridden"
        return out
    out["admits_end"] = bool(after.get("admits"))
    out["reclaimable_end_mb"] = after.get("reclaimable_mb")
    if out["admits_start"] and not out["admits_end"]:
        out["class"] = "hijacked"
        out["why"] = "; ".join(after.get("refused_by") or [])
    elif not out["admits_start"]:
        out["class"] = "busy-overridden"
    return out


def maybe_proceed(before: dict, allow_busy: bool) -> tuple[bool, str]:
    """(may-start, sentence-to-print). Silence is what the old code bought us, so print always."""
    d = judge_window(before, None)
    if d["admits_start"]:
        return True, "quiet (reclaimable %.0f MB >= %.0f MB needed)" % (
            d["reclaimable_mb"] or 0.0, d["need_mb"] or 0.0)
    why = "; ".join(d["refused_by"]) or "the shared window says no"
    if allow_busy:
        return True, "OVERRIDDEN: %s -- these rows may describe the neighbour" % why
    return False, "busy box (%s)" % why


def main() -> int:
    ap = argparse.ArgumentParser(description="shape knob search")
    ap.add_argument("--plan", action="store_true", help="print the grid and launch form, launch nothing")
    ap.add_argument("--run", action="store_true", help="actually launch llama-bench cells")
    ap.add_argument("--grid", default="width", help="comma list from width,ncb,budget")
    ap.add_argument("--width", default="", help="override width values, e.g. 2,3,4")
    ap.add_argument("--profile", default=DEFAULT_PROFILE)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--target", type=float, default=25.0)
    ap.add_argument("--combine", default="geometric", choices=("geometric", "arithmetic"),
                    help=("how the two bracketing references are combined. 'geometric' interpolates "
                          "in log space at the cell's own instant (default, correct for a machine "
                          "that multiplies throughput); 'arithmetic' reproduces the old behaviour "
                          "when you need to compare against a previously published number"))
    ap.add_argument("--workdir", default="")
    ap.add_argument("--settle", type=int, default=45,
                    help="extra quiet seconds before each cell once the box reads NOMINAL again")
    ap.add_argument("--settle-budget", type=int, default=420,
                    help="seconds to wait for the window before giving up on the NEXT cell")
    ap.add_argument("--min-free-pct", type=float, default=15.0,
                    help="refuse to launch the next cell below this system-wide free percentage")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--allow-inert", action="store_true",
                    help="(dangerous) run cells whose resolved env equals the reference cell's")
    ap.add_argument("--need-mb", type=float, default=0.0,
                    help="reclaimable-memory bar for launching (0 = the shared server_window bar)")
    ap.add_argument("--allow-busy", action="store_true",
                    help=("(dangerous) sweep anyway while another llama-server holds this box; the "
                          "condition is printed and carried with the rows, never silently used"))
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    overrides = {}
    if args.width:
        overrides["width"] = [v for v in args.width.split(",") if v]
    names = [n for n in args.grid.split(",") if n]
    for n in names:
        if n not in GRIDS:
            print(f"unknown grid {n!r}; known: {sorted(GRIDS)}")
            return 2
    cells = build_grid(names, overrides)

    workdir = Path(args.workdir) if args.workdir else (
        ROOT / "Backup" / "shape_search" / f"{args.profile}_{'-'.join(names)}")
    workdir.mkdir(parents=True, exist_ok=True)

    print(f"profile={args.profile} reps={args.reps} target={args.target} t/s")
    print(f"grid={names} cells={len(cells)} workdir={workdir}")
    reach = check_reachability(args.profile, cells)
    inert: list[str] = []
    for c, diff, ok in reach:
        _b, env, conflicts = arm_env(args.profile, c)
        flag = ("  !! CONFLICT " + ";".join(conflicts)) if conflicts else ""
        if not diff:
            inert.append(cell_name(c))
            flag += "  !! INERT: resolved env identical to the reference cell"
        print(f"  {cell_name(c):<28} {arm_arg(args.profile, c)}{flag}")
        if diff:
            for k, (v0, v1) in diff.items():
                print(f"      ~ {k}: {v0!r} -> {v1!r}")
    if inert and not args.allow_inert:
        print("\nREFUSED: these cells do not differ from the reference in the RESOLVED launcher env:")
        for name in inert:
            print(f"  - {name}")
        print("  Either the key is not one run_server.sh puts in SERVER_ENV (use its "
              "CGC_SERVER_* handle), or the profile overrides it. Fix the axis -- do not "
              "'fix' this by passing --allow-inert, which would spend 20 minutes per cell "
              "measuring the baseline and then read the difference as an effect.")
        return 2
    if args.plan or not args.run:
        print("\n(dry run -- add --run to launch)")
        return 0

    # ---- whose box is this? asked once here, and again when the rows are finished ----------
    need_mb = args.need_mb or (getattr(sw, "NEED_MB", 8000.0))
    if sw is None:
        win0 = {"admits": None, "refused_by": ["server_window not importable"],
                "reclaimable_mb": None, "need_mb": need_mb}
        print("\n[window] the shared bar could not be imported: proceeding UNGATED, so these rows "
              "carry no statement about who else held the machine")
    else:
        win0 = sw.decision(need_mb=need_mb)
        print(f"\n[window] harness admits={win0['harness_admits']} "
              f"launcher admits={win0['launcher_admits']} agree={win0['agree']} "
              f"binding={win0['binding']}")
        print(f"[window] reclaimable {win0['reclaimable_mb']:.0f} MB of {need_mb:.0f} MB needed, "
              f"foreign llama={win0['foreign_llama'] or 'none'}, port held={win0['port_held']}")
    if win0["admits"] is None:
        may, why = True, "ungated (no shared bar available)"
    else:
        may, why = maybe_proceed(win0, args.allow_busy)
    if not may:
        print(f"\nREFUSED: {why}")
        print("  On a 16 GB box carrying a 12.7 GB model, a row taken while another session's "
              "server holds 8 GiB describes THAT run, not this cell -- and nothing in the number "
              "itself says so. Wait for the window, or pass --allow-busy knowing the rows then say "
              "nothing about the change you were testing.")
        return 2
    print(f"[window] {why}")

    # BRACKETED: every cell sits between two reference launches. This is not paranoia: tonight's two
    # sweeps drifted 10.50 -> 8.47 and 10.75 -> 6.04 between head and tail *inside one sweep*, which
    # is the same size as the effect being looked for. Head/tail pairing only detects that; local
    # bracketing lets each cell be read against the machine state it actually ran in.
    order: list[tuple[dict[str, str], str]] = [({}, "ref_0")]
    for i, c in enumerate(cells):
        order.append((c, f"cell_{cell_name(c)}"))
        order.append(({}, f"ref_{i + 1}"))
    recs = []
    clean_all_failed = False
    st: dict = {}
    for idx, (knobs, tag) in enumerate(order):
        if idx:
            st = wait_settle(args.settle, args.min_free_pct, args.settle_budget,
                             "window never returned before the next cell")
            print(f"[settle] waited {st['waited_s']}s thermal={st['thermal']} "
                  f"free={st['free_pct']:.0f}% -> {'go' if st['ok'] else 'proceeding unanchored'}",
                  flush=True)
            if not st["ok"]:
                clean_all_failed = True
        print(f"\n=== {tag} {knobs} ===", flush=True)
        rec = run_cell(tag, args.profile, knobs, args.reps, workdir, False)
        rec["settle"] = st if idx else None
        recs.append(rec)
        ts = next(iter([r.get("avg_ts") for r in (rec.get("rows") or [])]), None)
        print(f"    ts={ts} shape={json.dumps(rec.get('shape') or {})[:200]}")
        if rec.get("rc"):
            print(f"    !! rc={rec['rc']}: {rec.get('stdout_tail','')[-300:]}")

    def row_clean(r: dict) -> bool:
        th = (r.get("thermal") or {})
        worst = th.get("worst") or th.get("launch")
        return (worst is None) or (str(worst).upper() in ("NOMINAL", "0", ""))

    clean_fail = locals().get("clean_all_failed", False)
    clean_all = all(row_clean(r) for r in recs) and not clean_fail
    if not clean_all:
        print("\n!! the window did not stay NOMINAL for every cell: rows below are NOT comparable")
    if sw is not None and win0.get("admits") is not None:
        win1 = sw.decision(need_mb=need_mb)
        wv = judge_window(win0, win1)
        end = wv["reclaimable_end_mb"]
        print(f"[window] at the end: admits={wv['admits_end']}, reclaimable "
              f"{'n/a' if end is None else '%.0f MB' % end} (was {wv['reclaimable_mb']:.0f} MB), "
              f"class={wv['class']}")
        if wv["class"] == "hijacked":
            clean_all = False
            print("!! the box stopped being ours DURING the sweep (%s): rows below may describe "
                  "the neighbour" % "; ".join(win1.get("refused_by") or []))

    # Each cell is read against the reference INTERPOLATED to the instant that cell ran (log space,
    # see crossover_estimate.py), and the drift between the two refs is reported per cell: a cell
    # whose two refs disagree by more than MAX_DRIFT_PCT cannot be attributed to its knobs no
    # matter how it scored. `mean_gap_pct` is recorded alongside every row because it says how much
    # of the row's number was an artefact of the mean chosen -- a diagnostic, not a second gate.
    def ref_at(i: int) -> float:
        return row_ts(recs[i]["rows"][0])[0] if (recs[i].get("rows") and not recs[i].get("refused")) else 0.0

    ref_pos = [i for i, (k, _) in enumerate(order) if not k]
    ref_series = [(order[i][1], ref_at(i)) for i in ref_pos]
    print("\nreference series: " + "  ".join(f"{t}={v:.2f}" for t, v in ref_series))

    MAX_DRIFT_PCT = 10.0
    verdicts, summary = [], []
    for i, (knobs, tag) in enumerate(order):
        if not knobs or recs[i].get("refused"):
            continue
        before = max((j for j in ref_pos if j < i), default=None)
        after = min((j for j in ref_pos if j > i), default=None)
        vb = ref_at(before) if before is not None else 0.0
        va = ref_at(after) if after is not None else 0.0
        tb = mid_time(recs[before]) if before is not None else None
        ta = mid_time(recs[after]) if after is not None else None
        tc = mid_time(recs[i])
        # The reference AT THE INSTANT this cell ran. Log space, weighted by the instants, because
        # the machine multiplies throughput rather than subtracting from it (crossover_estimate.py).
        mode = args.combine
        if cx and vb > 0 and va > 0 and None not in (tb, ta, tc):
            ref_ts = cx.ref_at(vb, tb, va, ta, tc, mode=mode)
            ref_ts_alt = cx.ref_at(vb, tb, va, ta, tc, mode="arithmetic")
            note = f"{mode}, time-weighted"
        elif vb > 0 and va > 0:
            ref_ts = cx.geom(vb, va) if (cx and mode == "geometric") else 0.5 * (vb + va)
            ref_ts_alt = 0.5 * (vb + va)
            note = f"{mode}, index midpoint (this cell recorded no timestamps)"
        else:
            ref_ts = ref_ts_alt = max(vb, va)
            note = "one usable reference only"
        drift = 0.0
        if before is not None and after is not None and ref_at(before) > 0:
            drift = 100.0 * abs(va - vb) / vb
        v = verdict(recs[i], ref_ts, args.target, clean_all)
        v["ref_ts"] = round(ref_ts, 2)
        v["ref_ts_alt"] = round(ref_ts_alt, 2)
        v["mean_gap_pct"] = round(cx.mean_gap_pct(vb, va), 2) if (cx and vb > 0 and va > 0) else 0.0
        v["ref_note"] = note
        v["drift_pct"] = round(drift, 2)
        if drift > MAX_DRIFT_PCT:
            v["state"] = "UNANCHORED"
            v["why"] = f"bracketing references disagree by {drift:.1f}% (> {MAX_DRIFT_PCT}%)"
        verdicts.append(v)
        summary.append((tag, ref_ts, ref_ts_alt, v["mean_gap_pct"], drift))

    out = {"profile": args.profile, "reps": args.reps, "target": args.target,
           "ref_series": ref_series, "max_drift_pct": MAX_DRIFT_PCT,
           "combine": args.combine, "have_timestamps": all(mid_time(r) is not None for r in recs),
           "cells": recs, "verdicts": verdicts}
    (workdir / "search.json").write_text(json.dumps(out, indent=2))

    print(f"{'cell':<30} {'ref':>7} {'ref_alt':>8} {'gap%':>6} {'drift%':>7}")
    for tag, ref_ts, alt, gap, drift in summary:
        print(f"{tag:<30} {ref_ts:>7.2f} {alt:>8.2f} {gap:>6.2f} {drift:>7.2f}")
    print(f"\n{'cell':<30} {'t/s':>7} {'Δ%':>7}  state       why")
    for v in verdicts:
        print(f"{v['tag']:<30} {v.get('ts', 0):>7.2f} {v.get('delta_pct', 0):>7.2f}  "
              f"{v['state']:<11} {v['why'][:60]}")
    print(f"\nref mode: {args.combine}   "
          f"(ref_alt = what the arithmetic mean would have given; gap% is their difference)")
    for v in verdicts:
        if (v.get("mean_gap_pct") or 0) > THRESHOLD_PCT:
            print(f"  !! {v['tag']}: choosing the mean alone moves this row by "
                  f"{v['mean_gap_pct']:.1f}% (> {THRESHOLD_PCT}%) -- unreadable at this drift")
    print(f"\nwrote {workdir / 'search.json'}")
    return 0


def selftest() -> int:
    """Prove the parser reads the OUTPUT function's real dialect and rejects a corrupted one."""
    good = ("CGC-SHAPE v=1 phase=final M=4 M_stat=APPLIED width=4 union=32 fits=1 "
            "pool_cap_slots=143 slots_layer=143 n_layer=41 req=1234 hits=1000 misses=234 "
            "hit_pct=81.04 compulsory=90 capacity=144 evict=12 zero_mapped=0 verify_refused=0 "
            "inv_viol=0 read_mib=512.0 pread_us=99 fill_wait_us=7 final_counters=1 "
            "gdn_ablated=0")
    r = parse_shape_line(good)

    # The line this module reads has exactly one rule: every value is ONE token with no '=' inside
    # it. The previous revision of the engine printed a sentence in note=, which both broke the
    # token walk AND smuggled a phantom key in (`K=1` inside the prose). Prove the parser notices,
    # so nobody can reintroduce that without this going red.
    old_note = ("CGC-SHAPE v=1 phase=gdn fused_ar=1 fused_ch=1 req_ar=-1 req_ch=-1 ablated=0 "
                "bitident=yes note=default fused ggml_gated_delta_net on Metal (K snapshots "
                "incl. K=1)")
    new_note = ("CGC-SHAPE v=1 phase=gdn fused_ar=1 fused_ch=1 req_ar=-1 req_ch=-1 ablated=0 "
                "bitident=yes note=fused_metal_default_k_snapshot")
    old_r = parse_shape_line(old_note)
    new_r = parse_shape_line(new_note)

    ok_row = {"tag": "t", "rows": [{"avg_ts": 12.0, "stddev_ts": 1.0}],
              "shape": {"M_stat": "APPLIED", "zero_mapped": 0, "verify_refused": 0,
                        "inv_viol": 0, "union": 32, "fits": 1, "gdn_ablated": 0},
              "shape_gdn": {"phase": "gdn", "fused_ar": 1, "fused_ch": 1, "ablated": 0,
                            "note": "fused_metal_default_k_snapshot"}}
    ablated_row = {**ok_row,
                   "shape": {**ok_row["shape"], "gdn_ablated": 1},
                   "shape_gdn": {"phase": "gdn", "fused_ar": 0, "fused_ch": 1, "ablated": 1,
                                 "note": "fusion_off_ablation_not_comparable"}}
    checks = [
        (r.get("M") == 4, "M parsed"),
        (r.get("M_stat") == "APPLIED", "M_stat parsed"),
        (r.get("zero_mapped") == 0, "counter parsed as int"),
        (r.get("hit_pct") == 81.04, "float parsed"),
        ("_unparsed" not in r, "no stray tokens"),
        (parse_shape_line("not a shape line") == {}, "non-shape line rejected"),
        (parse_shape_line("CGC-SHAPE v=1 phase=final M=4 WAT width=4").get("_unparsed") == ["WAT"],
         "changed dialect is flagged, not dropped"),
        (pick_final([{"phase": "init", "M": 1}, {"phase": "final", "M": 4}])["M"] == 4,
         "final line wins over init"),
        (sum(int(r.get(c, 0)) for c in VOID_COUNTERS) == 0, "void counter sum"),
        (verdict({"tag": "t", "shape": {"M_stat": "CLAMPED"}}, 10.0, 25.0)["state"] == "VOID",
         "clamped width is VOID"),
        (verdict({"tag": "t", "shape": {"M_stat": "APPLIED", "zero_mapped": 1}}, 10.0, 25.0)["state"]
         == "VOID", "dropped-expert row is VOID"),
        (verdict({"tag": "t", "shape": {"M_stat": "APPLIED", "zero_mapped": 0, "verify_refused": 0,
                                        "inv_viol": 0, "union": 200, "fits": 0}}, 10.0, 25.0)["state"]
         == "VOID", "union-over-slots is VOID"),
        (verdict({"tag": "t", "rows": [{"avg_ts": 10.2, "stddev_ts": 1.0}],
                  "shape": {"M_stat": "APPLIED", "zero_mapped": 0, "verify_refused": 0,
                            "inv_viol": 0, "union": 32, "fits": 1}}, 10.0, 25.0)["state"]
         == "NO_EVIDENCE", "+2% is below threshold"),
        (verdict({"tag": "t", "rows": [{"avg_ts": 12.0, "stddev_ts": 1.0}],
                  "shape": {"M_stat": "APPLIED", "zero_mapped": 0, "verify_refused": 0,
                            "inv_viol": 0, "union": 32, "fits": 1}}, 10.0, 25.0)["state"]
         == "FASTER", "+20% reads as FASTER"),
        # ---- GDN: the axis exists; what must be enforced is that an ablation never masquerades
        # ---- as a measurement of the anchored engine.
        (verdict(ablated_row, 10.0, 25.0)["state"] == "ABLATED",
         "GDN ablation is reported as ABLATED, never as a fast/slow row"),
        (verdict(ok_row, 10.0, 25.0)["state"] == "FASTER",
         "the anchored GDN row still gets a real verdict"),
        (verdict({**ok_row, "shape_init": {"gdn_stat": "REFUSED"}}, 10.0, 25.0)["state"] == "VOID",
         "a refused GDN request is VOID instead of quietly taking the default"),
        # ---- the OUTPUT dialect contract, with the sentence that broke it as the mutation
        (bool(old_r.get("_unparsed")),
         "the old free-text note= is caught as a corrupt dialect, not silently accepted"),
        (old_r.get("K") == "1)",
         "the old note smuggled a phantom key K out of its own prose -- caught now"),
        ("_unparsed" not in new_r, "the one-token note parses clean"),
        (new_r.get("note") == "fused_metal_default_k_snapshot", "machine-token note read back"),
        # ---- the reference arithmetic that decides every row above
        (cx is not None, "crossover_estimate is importable (rows get log-space references)"),
        (cx is None or abs(cx.ref_at(10.0, 0.0, 20.0, 100.0, 100.0) - 20.0) < 1e-9,
         "reference at the far end is the far reference"),
        (cx is None or cx.ref_at(10.0, 0.0, 20.0, 100.0, 50.0) < 15.0,
         "log interpolation sits below the arithmetic midpoint, as a multiplicative machine must"),
        # ---- reachability: a grid axis must MOVE the resolved launcher env. This is the mutation
        # ---- test for the LLAMA_ARG_EXPERT_CACHE bug, in which every budget cell resolved to an
        # ---- env indistinguishable from the reference and silently measured the baseline.
        (all(ok for _c, _d, ok in check_reachability(DEFAULT_PROFILE, build_grid(["budget"], {}))),
         "every budget cell differs from the reference in the resolved env (live launcher query)"),
        (check_reachability(DEFAULT_PROFILE,
                            [{"LLAMA_ARG_EXPERT_CACHE": "3221225472"}])[0][2] is False,
         "the old LLAMA_ARG_EXPERT_CACHE key is caught as INERT (the bug this prevents)"),
        (len(build_grid(["width"], {"width": ["4", "17"]})) == 2, "grid override honoured"),
        # ---- the window gate: WHOSE box were these rows taken on? Until 2026-09-23 this harness
        # ---- never asked, and a foreign llama-server is invisible in every number it prints.
        (sw is not None, "server_window importable (the shared bar is available to this harness)"),
        (judge_window({"admits": True, "refused_by": [], "reclaimable_mb": 9000.0,
                       "need_mb": 8000.0}, None)["class"] == "clean",
         "a quiet box reads clean before any row exists"),
        (judge_window({"admits": False, "refused_by": ["harness:foreign"], "reclaimable_mb": 1400.0,
                       "need_mb": 8000.0}, None)["class"] == "busy-overridden",
         "launching over somebody else's server is recorded, not hidden"),
        (judge_window({"admits": True, "refused_by": [], "reclaimable_mb": 9000.0,
                       "need_mb": 8000.0},
                      {"admits": False, "refused_by": ["harness:foreign"],
                       "reclaimable_mb": 1400.0})["class"] == "hijacked",
         "the box changing hands MID-SWEEP is its own class (a neighbour is not a drift)"),
        (judge_window({"admits": False, "refused_by": ["harness:memory"], "reclaimable_mb": 1400.0,
                       "need_mb": 8000.0},
                      {"admits": True, "refused_by": [], "reclaimable_mb": 9000.0})["class"]
         == "busy-overridden", "a hijack is LOSING the box, not gaining it (direction must show)"),
        (maybe_proceed({"admits": False, "refused_by": ["harness:foreign"], "reclaimable_mb": 1400.0,
                        "need_mb": 8000.0}, False)[0] is False,
         "by default this harness refuses to launch into a neighbour's box"),
        (maybe_proceed({"admits": False, "refused_by": ["harness:foreign"], "reclaimable_mb": 1400.0,
                        "need_mb": 8000.0}, True)[0] is True,
         "--allow-busy exists for the box you cannot wait for"),
        ("OVERRIDDEN" in maybe_proceed({"admits": False, "refused_by": ["harness:foreign"],
                                        "reclaimable_mb": 1400.0, "need_mb": 8000.0}, True)[1],
         "and it says so in the sentence it prints"),
        ("busy box" in maybe_proceed({"admits": False, "refused_by": ["harness:foreign"],
                                      "reclaimable_mb": 1400.0, "need_mb": 8000.0}, False)[1],
         "the refusal names the reason, so the log alone is enough to void the run"),
    ]
    bad = [name for ok, name in checks if not ok]
    for ok, name in checks:
        print(f"  {'ok  ' if ok else 'FAIL'} {name}")
    print(f"\nselftest: {len(checks) - len(bad)}/{len(checks)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
