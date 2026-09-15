#!/usr/bin/env python3
"""E0 emitter — turn the artifacts that already exist on disk into engine-loop `episode` records.

WHY THIS IS A SEPARATE PROGRAM FROM THE HARNESS IT INDEXES
    It never runs the engine. `scripts/check/*` and `scripts/run_server.sh` are production scripts:
    the pre-commit gate and the whitepapers point at them, so there must be exactly one copy. This
    emitter is a pure reader — it opens the JSON/log artifacts those scripts already wrote and
    projects them into the three record types defined in `traces/schema/`. The moment it starts
    running the engine it becomes a second, drifting implementation of the harness.

WHAT AN EPISODE IS, AND WHAT IT IS NOT
    One run of one arm, with its build fingerprint, its measurements, and the artifact that produced
    them. It carries NO interpretation — interpretation belongs in `decision` and `lesson`. So the
    emitter must never invent a conclusion; it may only copy numbers, and it must record what makes
    a number NOT quotable (unstable answer digest, missing fingerprint, probe arm, failed run) as a
    first-class field rather than dropping the row.

THE ONE RULE WORTH ENFORCING HERE
    `build == null  =>  usable_as_evidence == false`. A throughput or bit-identical number whose
    binaries are unknown cannot be compared to anything, not even to itself after a rebuild.
    `traces/validate.py` re-checks the implication; this emitter is careful to set it correctly at
    the source, including for the historical rows that predate fingerprint recording.

USAGE
    emit_episodes.py --out traces/episodes.jsonl            emit and validate
    emit_episodes.py --stats                                say what is on disk, emit nothing
    emit_episodes.py --out /tmp/x.jsonl --strict            refuse to write if any episode is unusable
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import statistics
import sys

REPO_MARKERS = ("src/llama.cpp", "scripts/run_server.sh")


def find_repo(start: str) -> str:
    """Walk up until the repo root is unmistakable, so the emitter works no matter where it is
    copied to (and so the plan's later E1 directory move cannot break the paths)."""
    d = os.path.abspath(start)
    while True:
        if all(os.path.exists(os.path.join(d, m)) for m in REPO_MARKERS):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise SystemExit("emit_episodes: cannot locate the repo root from " + start)
        d = parent


REPO = find_repo(os.path.dirname(os.path.abspath(__file__)))
TRACES_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------------------------
# sanitization (see agent_harness/PLAN_ENGINE_LOOP_2026-09-15.md §8)
# --------------------------------------------------------------------------------------------

_IPV4 = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_APIKEY = re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}\b")
_HOME = re.compile(r"/(?:Users|home)/[A-Za-z0-9._\-]+")


def sanitize(value):
    """Absolute paths -> $REPO / $HOME, LAN IPs -> $LAN_IP, sk-* -> $API_KEY.

    Not cosmetic: `scripts/run_server.sh` prints a connection card containing the machine's LAN
    address, and the oracle `.cap` files embed the full argv with absolute model paths. Both end up
    in these records, and these records are meant to be committed."""
    if isinstance(value, str):
        v = value.replace(REPO + "/", "$REPO/")
        v = _HOME.sub("$HOME", v)
        v = _IPV4.sub("$LAN_IP", v)
        v = _APIKEY.sub("$API_KEY", v)
        return v
    if isinstance(value, list):
        return [sanitize(x) for x in value]
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items()}
    return value


# --------------------------------------------------------------------------------------------
# arm metadata — parsed from the production ARMS table, never duplicated here
# --------------------------------------------------------------------------------------------

# Arms that ARE the proposition under test, or that instrument the hot path, or that deliberately
# produce a wrong answer. None of them may supply a throughput number.
INSTRUMENTATION_ENV = {
    "GGML_SCHED_DEBUG",     # scheduler dump, hundreds of lines per step
    "CGC_S1_DBG",           # S1 diagnostics, writes to stderr from inside the dispatch hook
    "CGC_S1_IDENT",         # publishes a constant table: answer is wrong by construction
    "CGC_S1_TAG",           # publishes a layer tag: answer is wrong by construction
    "CGC_DECODE_PROFILE_ALL",
    "CGC_UNION_LOG",
}

# Verbatim phrases in the ARMS comment block that mean "this number is not a speedup".
NEG_MARKERS = (
    "never use this arm for a throughput number",
    "must never be used as ground truth",
    "numerically wrong by construction",
    "expected to be corrupt",
    "not quotable",
    "throughput from these two arms is meaningless",
)

# Whole arm FAMILIES that are comparison instruments rather than configurations whose speed means
# anything. See the note in parse_arms() for why this is a prefix list and not a set of arm names.
CONTROL_PREFIXES = (
    "p25-slotgpu",   # every CGC_S1_MIN_IL rung, plus the full arm
    "p25-keepleaf",  # the KEEP_LEAF control and its diagnostic twin
)

# Explicit, human-authored caveats. The emitter cannot infer these, and a wrong throughput number is
# the most expensive kind of error this loop can make, so they are pinned here rather than guessed.
ARM_OVERRIDE = {
    "p25-submit-ahead": (
        False,
        "UPPER BOUND ONLY: CGC_SUBMIT_AHEAD submits segment i+1 before the top-k hook rewrites its "
        "remap leaf, so the output is expected to be corrupt (and the md5 confirms it differs every "
        "rep). The measured ratio is a CEILING on the whole 'restore inter-segment overlap' family, "
        "never a speedup, and it must never appear in a public table.",
    ),
    "p25-slotgpu": (
        True,
        "S1 keeps n_segs at 40 by design, so a speedup here should be ~0. The number is the control "
        "for the S1 bit-identical comparison, not a performance claim.",
    ),
    "p25-keepleaf": (
        False,
        "CONTROL ARM, not a configuration: it builds every S1 node AND the host leaf, then lets "
        "mul_mat_id consume the leaf. Its only jobs are (a) the md5 comparison against the baseline "
        "in the same sweep and (b) the post-sync gather-vs-leaf readback. A throughput number here "
        "describes a graph nobody would ship.",
    ),
    "p25-keepleaf-dbg": (
        False,
        "Diagnostic twin of p25-keepleaf with CGC_S1_DBG on: it writes to stderr from inside the "
        "dispatch hook, so its throughput is meaningless (auto-flagged via INSTRUMENTATION_ENV as "
        "well). Read its log for the per-layer same= field, nothing else.",
    ),
}


def parse_arms(path: str) -> dict:
    """Read the ARMS table out of scripts/check/decode_sweep.py.

    Each arm's `declared_purpose` is the comment block immediately above it. Parsing the production
    file (instead of copying the text into this repo) is what keeps the claim from drifting away
    from the arm it describes — the failure mode the plan calls "two sources of truth"."""
    arms: dict = {}
    try:
        lines = open(path, errors="replace").read().splitlines()
    except OSError as e:
        print(f"  [warn] cannot read {path}: {e}", file=sys.stderr)
        return arms

    arm_re = re.compile(r'^\s*"([A-Za-z0-9._\-]+)"\s*:\s*\{')
    for i, line in enumerate(lines):
        m = arm_re.match(line)
        if not m:
            continue
        tag = m.group(1)
        # the env dict may span lines; concatenate until braces balance
        blob, depth = "", 0
        for j in range(i, min(i + 12, len(lines))):
            blob += lines[j]
            depth += lines[j].count("{") - lines[j].count("}")
            if depth <= 0 and j > i:
                break
            if j == i and depth == 0:
                break
        env = {}
        body = blob[blob.find("{"):]
        for k, v in re.findall(r'"([A-Za-z0-9_]+)"\s*:\s*"([^"]*)"', body):
            env[k] = v
        # contiguous comment block directly above the arm
        block, k = [], i - 1
        while k >= 0 and lines[k].strip().startswith("#"):
            block.append(lines[k].strip().lstrip("#").strip())
            k -= 1
        block.reverse()
        text = " ".join(block)
        first = block[0] if block else ""
        if not first:
            first = "(no declared purpose in the ARMS table)"
        low = text.lower()
        instrumented = bool(INSTRUMENTATION_ENV & set(env))
        suspect = any(nm in low for nm in NEG_MARKERS)
        # Family rule for the S1 control arms. Every `p25-slotgpu*` variant keeps n_segs at 40 by
        # design -- S1 removes the per-layer round trip but not the segment-boundary wait -- so a
        # speedup there should be ~0 and quoting one would misstate what S1 buys. Every
        # `p25-keepleaf*` variant exists to answer a yes/no question about the mapping. Enumerating
        # the family rather than the arms keeps this true when a rung is added, which happened twice
        # today: an arm added an hour later must not be more quotable than its siblings.
        control = any(tag.startswith(p) for p in CONTROL_PREFIXES)
        usable = not instrumented and not suspect and not control
        arms[tag] = {
            "name": tag,
            "declared_purpose": first[:400],
            "usable_for_throughput": usable,
            "env": env,
            "_comment_text": text,
        }
    return arms


# --------------------------------------------------------------------------------------------
# log harvesting
# --------------------------------------------------------------------------------------------

_KV = re.compile(r"([a-z_]+)=(-?[\d.]+)")


def _kv(line: str) -> dict:
    return {k: float(v) for k, v in _KV.findall(line)}


def _log_basename(log_field: str) -> str:
    return os.path.basename(log_field.replace("（tail -f 同路徑）", "").strip())


def stamp_from_log(log_field: str):
    """(YYYYMMDD, HHMMSS) taken from the server log filename. Never computed from wall-clock: the
    log's own name is the authoritative start time of the run, and it is the same string the ctrl
    log uses."""
    b = _log_basename(log_field)
    m = re.search(r"(\d{8})[_-](\d{6})", b)
    if m:
        return m.group(1), m.group(2)
    return None, None


def parse_phase(log_path: str, tail: int = 8) -> dict | None:
    """Steady-state medians of the CGC-GPUTIME / CGC-DECPROF lines.

    Only the LAST `tail` GPUTIME samples are used: the first steps include the pool warm-up and are
    not representative. Every other sample is ignored rather than averaged in, because averaging a
    warm-up step into a steady-state number is how a real effect gets diluted below its own noise."""
    if not log_path or not os.path.exists(log_path):
        return None
    gp, dec = [], None
    try:
        with open(log_path, errors="replace") as fh:
            for line in fh:
                if line.startswith("CGC-GPUTIME:"):
                    gp.append(_kv(line))
                elif line.startswith("CGC-DECPROF: step="):
                    dec = _kv(line)
    except OSError:
        return None
    if not gp:
        return None
    gp = gp[-tail:]
    out: dict = {"n_samples": float(len(gp))}
    for key in ("wait", "gpu_busy_sum", "gpu_union", "gap", "segs", "skipped"):
        vals = [s[key] for s in gp if key in s]
        if vals:
            out[key] = round(statistics.median(vals), 2)
    if dec:
        for key in ("cb", "submit", "total", "segs", "layers"):
            if key in dec:
                out[key] = round(dec[key], 2)
    return out


def parse_asserts(log_path: str) -> dict | None:
    """Counts of the consumer-side guard.

    IMPORTANT — these are SAMPLES, not a census. CGC-MMID-ASSERT prints the first 8 violations
    verbatim and then only every 1000th, so `mmid_lines` is a ceiling on the printed output and
    `mmid_oob_total` (the last cumulative `total=`) is the only honest count. A conclusion of the
    form "only layers X and Y are affected" cannot be drawn from these lines; that mistake is
    recorded as lesson eng-lf-0007."""
    if not log_path or not os.path.exists(log_path):
        return None
    n_lines = 0
    oob_total = zero_total = zmap = 0
    oob_re = re.compile(r"id_oob=\d+ \(first=-?\d+, total=(\d+)\)")
    zr_re = re.compile(r"zero_row=\d+ \(first_id=-?\d+, total=(\d+)\)")
    zm_re = re.compile(r"CGC-ZERO-MAPPED:.*?n=(\d+)/(\d+)")
    try:
        with open(log_path, errors="replace") as fh:
            for line in fh:
                if "MMID-ASSERT" in line:
                    n_lines += 1
                    m = oob_re.search(line)
                    if m:
                        oob_total = int(m.group(1))
                    m = zr_re.search(line)
                    if m:
                        zero_total = int(m.group(1))
                elif "CGC-ZERO-MAPPED" in line:
                    m = zm_re.search(line)
                    if m:
                        zmap += int(m.group(1))
    except OSError:
        return None
    out: dict = {}
    if n_lines == 0:
        return None
    out = {
        "mmid_lines": n_lines,
        "mmid_oob_total": oob_total,
        "mmid_zero_row_total": zero_total,
        "zero_mapped": zmap,
    }
    return out


def file_sha256(path: str, cap: int = 32 * 1024 * 1024):
    """sha256 of the artifact, or None when it is too large to digest on every emit (cgc_logs holds
    hundreds of MB). None is honest; a digest of a truncated file would not be."""
    if not path or not os.path.exists(path):
        return None, None
    size = os.path.getsize(path)
    if size > cap:
        return None, size
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest(), size


# --------------------------------------------------------------------------------------------
# episode construction
# --------------------------------------------------------------------------------------------

def _pool_from_row(row: dict) -> dict:
    keys = ("hit_rate_pct", "capacity_pct", "io_effective_mib_s", "resident_mib",
            "misses", "evictions", "per_miss_us", "layers_over_slots")
    return {k: row.get(k) for k in keys if k in row} or None


def _goal_from_arm(arms: dict, tag: str) -> str:
    a = arms.get(tag)
    if a:
        return a["declared_purpose"]
    return f"(arm '{tag}' has no entry in the ARMS table — see scripts/check/decode_sweep.py)"


def emit_sweep_rows(path: str, arms: dict) -> list:
    """`Backup/phase_decomp/*.json`: a list of per-arm rows written by decode_sweep.py /
    ab_interleave.py. A row with `rep` came from the interleaved A/B runner, which also records the
    build fingerprint; a row without one came from the plain sweep, which (until 2026-09-15) did
    not — those rows are emitted with build == null and usable_as_evidence == false."""
    try:
        rows = json.load(open(path, errors="replace"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"  [skip] {sanitize(path)}: {e}", file=sys.stderr)
        return []
    if not isinstance(rows, list):
        return []
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        tag = row.get("tag") or row.get("arm") or "unknown"
        log = row.get("log") or ""
        day, hms = stamp_from_log(log)
        rep = int(row.get("rep", 1) or 1)
        if day is None:
            day, hms = "00000000", "000000"
        arm_meta = arms.get(tag, {"name": tag, "declared_purpose": "(unknown arm)",
                                  "usable_for_throughput": False, "env": row.get("env", {})})
        usable, extra_caveat = ARM_OVERRIDE.get(tag, (arm_meta["usable_for_throughput"], None))
        md5s = sorted(set(row.get("answer_md5_set") or []))
        stable = bool(row.get("answer_stable"))
        build = row.get("build") or None
        caveats = []
        if extra_caveat:
            caveats.append(extra_caveat)
        if not arm_meta["usable_for_throughput"]:
            caveats.append("instrumented or probe arm: throughput from this row is not quotable")
        if build is None:
            caveats.append("no build fingerprint recorded for this row: incomparable, not merely "
                           "imprecise")
        if not stable or len(md5s) > 1:
            caveats.append(f"answer digest not stable across rounds ({md5s}): non-deterministic run")
        if int(row.get("n_rounds", 1) or 1) < 3:
            caveats.append(f"n_rounds={row.get('n_rounds')}: below the interleaved-A/B minimum of 3")
        obs = {
            "kind": "interleave" if "rep" in row else "sweep",
            "decode_tps": {
                "median": row.get("decode_tps_median"),
                "min": row.get("decode_tps_min"),
                "max": row.get("decode_tps_max"),
                "n_rounds": row.get("n_rounds"),
            },
            "prefill_tps_median": row.get("prefill_tps_median"),
            "answer_md5": md5s,
            "answer_stable": stable,
            "pool": _pool_from_row(row),
            "phase_ms": parse_phase(os.path.join(REPO, log) if log and not log.startswith("/") else log),
            "asserts": parse_asserts(os.path.join(REPO, log) if log and not log.startswith("/") else log),
        }
        if obs["asserts"]:
            caveats.append("assert counts come from rate-limited output: mmid_lines is a SAMPLE of "
                           "the printed violations and mmid_oob_total is the running total, so "
                           "'only layers X and Y are affected' cannot be read off these lines "
                           "(see CONVENTIONS.md A4)")
        sha, size = file_sha256(log if log and os.path.isabs(log) else os.path.join(REPO, log))
        out.append(sanitize({
            "type": "episode",
            "episode_id": f"eng-{day}-{hms}-{tag}-r{rep}",
            "loop": "engine",
            "profile": "prod25",
            "goal": _goal_from_arm(arms, tag),
            "arm": {
                "name": tag,
                "declared_purpose": arm_meta["declared_purpose"],
                "usable_for_throughput": usable,
                "env": row.get("env") or arm_meta["env"],
            },
            "build": build,
            "obs": obs,
            "gate": {"name": "m123_oracle_gate", "verdict": "n/a",
                     "ref": "Backup/knifeedge_matrix/ref_iq3_pool8gb_M2_6144_bitident.jsonl",
                     "env_passthrough": {}},
            "artifact": {"log": log, "log_bytes": size, "sha256": sha},
            "source": os.path.relpath(path, REPO),
            "usable_as_evidence": bool(build) and stable and len(md5s) <= 1,
            "caveats": caveats,
        }))
    return out


def emit_bench(path: str, arms: dict) -> list:
    """`Backup/llama_bench/matrix_*.json`: llama-bench matrices. The bench itself records
    `build_commit`, so the episode says so explicitly rather than pretending the server/metal
    digests are known."""
    try:
        items = json.load(open(path, errors="replace"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"  [skip] {sanitize(path)}: {e}", file=sys.stderr)
        return []
    if not isinstance(items, list):
        return []
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        tag = it.get("tag") or "bench"
        scal = it.get("scalars") or {}
        log = os.path.join(REPO, scal.get("LOG", "")) if scal.get("LOG") else None
        day, hms = stamp_from_log(scal.get("LOG", ""))
        if day is None:
            mt = os.path.getmtime(path)
            import time as _t
            day, hms = _t.strftime("%Y%m%d", _t.localtime(mt)), _t.strftime("%H%M%S", _t.localtime(mt))
        rows = it.get("rows") or []
        # llama-bench emits one row per (n_prompt, n_gen) cell; the decode cell is n_prompt=0
        dec = [r.get("avg_ts") for r in rows if r.get("n_prompt") == 0 and r.get("avg_ts")]
        pre = [r.get("avg_ts") for r in rows if r.get("n_prompt") and r.get("avg_ts")]
        commit = next((r.get("build_commit") for r in rows if r.get("build_commit")), None)
        caveats = []
        if commit:
            caveats.append(f"llama-bench records build_commit={commit} but not the server/metal "
                           f"md5, so build is null and this episode is not comparable to a sweep row")
        if it.get("incomplete"):
            caveats.append(f"run reported incomplete: {it.get('error')!r} — the cells that did "
                           f"report are real, the matrix is not the full grid")
        if it.get("error"):
            caveats.append(f"harness error: {it['error']}")
        out.append(sanitize({
            "type": "episode",
            "episode_id": f"eng-{day}-{hms}-{tag}-bench-r1",
            "loop": "engine",
            "profile": it.get("profile") or scal.get("PROFILE") or "unknown",
            "goal": f"llama-bench matrix: {it.get('batch_why') or 'standard grid'}",
            "arm": {"name": tag, "declared_purpose": "llama-bench matrix cell",
                    "usable_for_throughput": not it.get("incomplete"),
                    "env": dict(it.get("extra_env") or {})},
            "build": None,
            "obs": {
                "kind": "bench",
                "decode_tps": {"median": statistics.median(dec) if dec else None,
                               "min": min(dec) if dec else None,
                               "max": max(dec) if dec else None,
                               "n_rounds": len(dec) or None},
                "prefill_tps_median": statistics.median(pre) if pre else None,
                "answer_md5": [],
                "answer_stable": None,
                "pool": it.get("cache"),
                "phase_ms": None,
                "asserts": None,
                "oracle": None,
            },
            "gate": {"name": "none", "verdict": "n/a", "ref": None, "env_passthrough": {}},
            "artifact": {"log": scal.get("LOG"), "log_bytes": None, "sha256": None},
            "source": os.path.relpath(path, REPO),
            "usable_as_evidence": False,
            "caveats": caveats or ["bench matrix: no generation, so no answer digest to check"],
        }))
    return out


def emit_gate(path: str) -> list:
    """`Backup/knifeedge_matrix/capinv_*.json` (has a real verdict) and
    `Backup/m123_oracle_gate/cap_*.json` (provenance only). A `.cap` is never evidence of a pass —
    it records what was launched, which is why its use as ground truth is gated in
    scripts/check/m123_oracle_gate.py itself."""
    try:
        d = json.load(open(path, errors="replace"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"  [skip] {sanitize(path)}: {e}", file=sys.stderr)
        return []
    if not isinstance(d, dict):
        return []
    name = os.path.splitext(os.path.basename(path))[0]
    verdict = d.get("verdict")
    verdict_map = {"PASS": "pass", "FAIL": "fail"}
    res = d.get("resolved") or {}
    env = (res.get("ENV") if isinstance(res, dict) else None) or {}
    profile = d.get("profile") or (res.get("CGCENV", {}).get("PROFILE") if res else None) or "unknown"
    created = d.get("created") or ""
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})", created)
    day, hms = (m.group(1) + m.group(2) + m.group(3), m.group(4) + m.group(5) + "00") if m else ("00000000", "000000")
    caveats = []
    if verdict is None:
        caveats.append("provenance dump only: it records the resolved launch config for a gate run, "
                       "it is not itself a verdict, and it must not be used as ground truth")
    if d.get("expert_cache") == "on" and "ground truth" in (d.get("expert_cache_source") or ""):
        caveats.append("cache-ON dump: m123_oracle_gate.py --oracle-abs refuses these as references")
    if d.get("note"):
        caveats.append(str(d["note"])[:400])
    if d.get("ok") is False:
        caveats.append("the gate reported not-ok")
    return [sanitize({
        "type": "episode",
        "episode_id": f"eng-{day}-{hms}-{name}-r1",
        "loop": "engine",
        "profile": profile,
        "goal": f"gate run: {name}",
        "arm": {"name": name, "declared_purpose": "logits-oracle gate configuration",
                "usable_for_throughput": False, "env": dict(env)},
        "build": None,
        "obs": {
            "kind": "gate",
            "decode_tps": {"median": None, "min": None, "max": None, "n_rounds": None},
            "prefill_tps_median": None,
            "answer_md5": [],
            "answer_stable": None,
            "pool": None,
            "phase_ms": None,
            "asserts": None,
            "oracle": {
                "rows": (len(d.get("pairs") or []) or None),
                "distinct_row_digests": None,
                "ref": None,
                "verdict": verdict_map.get(str(verdict).upper(), "unknown"),
            },
        },
        "gate": {"name": "knifeedge_matrix" if "capinv" in name else "m123_oracle_gate",
                 "verdict": verdict_map.get(str(verdict).upper(), "n/a"),
                 "ref": None, "env_passthrough": {}},
        "artifact": {"log": None, "log_bytes": None, "sha256": None},
        "source": os.path.relpath(path, REPO),
        "usable_as_evidence": False,
        "caveats": caveats or ["gate artifact: no throughput, correctness only"],
    })]


_ORACLE_ROW = re.compile(r'"logits_fnv1a64":"([0-9a-f]+)"')


def emit_oracle(jsonl_path: str) -> list:
    """`Backup/knifeedge_matrix/*.jsonl`: one logits digest per step. This is the reference side of
    the bit-identical gate. `distinct_row_digests` is the reproducibility signal: a reference with
    repeated digests carries fewer independent constraints than its row count suggests."""
    if not os.path.exists(jsonl_path):
        return []
    digests, rows = [], 0
    try:
        with open(jsonl_path, errors="replace") as fh:
            for line in fh:
                m = _ORACLE_ROW.search(line)
                if m:
                    digests.append(m.group(1))
                    rows += 1
    except OSError:
        return []
    if rows == 0:
        return []
    base = os.path.basename(jsonl_path)
    mt = os.path.getmtime(jsonl_path)
    import time as _t
    day, hms = _t.strftime("%Y%m%d", _t.localtime(mt)), _t.strftime("%H%M%S", _t.localtime(mt))
    cap = jsonl_path + ".cap"
    cap_note = None
    ref_profile, ref_env = "unknown", {}
    if os.path.exists(cap):
        try:
            c = json.load(open(cap, errors="replace"))
            ref_profile = c.get("profile") or "unknown"
            ref_env = ((c.get("resolved") or {}).get("ENV")) or {}
            cap_note = c.get("note")
        except (OSError, json.JSONDecodeError):
            pass
    caveats = [
        f"{rows} oracle rows, {len(set(digests))} distinct logits digests — a repeated digest is a "
        f"constrained-information row, not an independent check",
    ]
    if cap_note:
        caveats.append(str(cap_note)[:400])
    elif not os.path.exists(cap):
        caveats.append("no .cap alongside: the launch config this reference belongs to is not "
                       "recorded, so it cannot be re-derived")
    return [sanitize({
        "type": "episode",
        "episode_id": f"eng-{day}-{hms}-{base}-r1",
        "loop": "engine",
        "profile": ref_profile,
        "goal": "logits oracle dump (reference side of the bit-identical gate)",
        "arm": {"name": base, "declared_purpose": "logits oracle reference",
                "usable_for_throughput": False, "env": dict(ref_env)},
        "build": None,
        "obs": {
            "kind": "oracle",
            "decode_tps": {"median": None, "min": None, "max": None, "n_rounds": None},
            "prefill_tps_median": None,
            "answer_md5": [],
            "answer_stable": None,
            "pool": None,
            "phase_ms": None,
            "asserts": None,
            "oracle": {"rows": rows, "distinct_row_digests": len(set(digests)),
                       "ref": os.path.relpath(jsonl_path, REPO), "verdict": "unknown"},
        },
        "gate": {"name": "bit_identical", "verdict": "n/a",
                 "ref": os.path.relpath(jsonl_path, REPO), "env_passthrough": {}},
        "artifact": {"log": os.path.relpath(jsonl_path, REPO), "log_bytes": os.path.getsize(jsonl_path),
                     "sha256": file_sha256(jsonl_path)[0]},
        "source": os.path.relpath(jsonl_path, REPO),
        "usable_as_evidence": False,
        "caveats": caveats,
    })]


# --------------------------------------------------------------------------------------------

def collect() -> list:
    arms = parse_arms(os.path.join(REPO, "scripts/check/decode_sweep.py"))
    print(f"  arms parsed from scripts/check/decode_sweep.py: {len(arms)}", file=sys.stderr)
    eps = []
    for p in sorted(glob.glob(os.path.join(REPO, "Backup/phase_decomp/*.json"))):
        got = emit_sweep_rows(p, arms)
        eps += got
        print(f"  sweep  {os.path.basename(p):28s} -> {len(got):3d}", file=sys.stderr)
    for p in sorted(glob.glob(os.path.join(REPO, "Backup/llama_bench/matrix_*.json"))):
        got = emit_bench(p, arms)
        eps += got
        print(f"  bench  {os.path.basename(p):28s} -> {len(got):3d}", file=sys.stderr)
    for p in sorted(glob.glob(os.path.join(REPO, "Backup/knifeedge_matrix/capinv_*.json"))) + \
             sorted(glob.glob(os.path.join(REPO, "Backup/m123_oracle_gate/cap_*.json"))):
        got = emit_gate(p)
        eps += got
        print(f"  gate   {os.path.basename(p):28s} -> {len(got):3d}", file=sys.stderr)
    for p in sorted(glob.glob(os.path.join(REPO, "Backup/knifeedge_matrix/*.jsonl"))):
        got = emit_oracle(p)
        eps += got
        print(f"  oracle {os.path.basename(p):28s} -> {len(got):3d}", file=sys.stderr)
    return eps


def print_stats(eps: list) -> None:
    n = len(eps)
    with_build = sum(1 for e in eps if e.get("build"))
    usable = sum(1 for e in eps if e.get("usable_as_evidence"))
    by_kind: dict = {}
    for e in eps:
        by_kind[e["obs"]["kind"]] = by_kind.get(e["obs"]["kind"], 0) + 1
    by_profile: dict = {}
    for e in eps:
        by_profile[e["profile"]] = by_profile.get(e["profile"], 0) + 1
    print(f"\n  episodes            {n}")
    print(f"  with build fp       {with_build}   ({100.0 * with_build / n:.0f}%)" if n else "")
    print(f"  usable_as_evidence  {usable}   ({100.0 * usable / n:.0f}%)" if n else "")
    print(f"  by obs.kind         {by_kind}")
    print(f"  by profile          {by_profile}")
    ids = [e["episode_id"] for e in eps]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        print(f"  !! duplicate episode_id: {sorted(dupes)}")
    reasons: dict = {}
    for e in eps:
        for c in e["caveats"]:
            key = c.split(":")[0][:60]
            reasons[key] = reasons.get(key, 0) + 1
    print("  caveat reasons (top 8):")
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])[:8]:
        print(f"    {v:4d}  {k}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(TRACES_DIR, "episodes.jsonl"))
    ap.add_argument("--stats", action="store_true", help="report what is on disk; write nothing")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any episode is not usable_as_evidence")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    eps = collect()
    eps.sort(key=lambda e: e["episode_id"])
    if not args.quiet:
        print_stats(eps)

    if args.stats:
        return 0

    with open(args.out, "w", encoding="utf-8") as fh:
        for e in eps:
            fh.write(json.dumps(e, ensure_ascii=False, sort_keys=False) + "\n")
    print(f"\n  wrote {len(eps)} episodes -> {args.out}")

    sys.path.insert(0, TRACES_DIR)
    try:
        import validate as _v  # noqa: E402
    except ImportError:
        print("  [warn] traces/validate.py not importable; skipping schema check", file=sys.stderr)
        return 0
    rc = _v.validate_file(args.out, quiet=args.quiet)
    if args.strict:
        bad = [e for e in eps if not e["usable_as_evidence"]]
        if bad:
            print(f"  --strict: {len(bad)} episode(s) are not usable_as_evidence")
            return 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
