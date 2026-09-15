#!/usr/bin/env python3
"""m123_oracle_gate.py -- the per-commit M1/M2/M3 quality gate.

WHY THIS EXISTS
---------------
`precommit_e2e_gate.sh` is an HTTP quality harness: it grades ANSWERS. It cannot see whether the
forward pass still computes the same numbers. This gate covers that gap. It launches a server
through `run_server.sh` (so the profile, the env allowlist and the load path are the production
ones -- not a hand-built argv that silently drifts), sends the SAME deterministic probe the
reference oracle was produced with, and compares the fresh logits dump against the reference.

THE THREE METRICS ARE NEVER MERGED
----------------------------------
`cgc_logits_oracle_compare.py` reports them separately on purpose:

  M1 numeric identity    same row_fnv1a64  -> bit-identical logits
  M2 decision agreement  same argmax_token -> the chosen token did not change
  M3 topk set agreement  same top-N id set -> the candidate set did not change

M1 same + M2 different   should be 0; anything else is a tie-break bug.
M1 different + M2 same   the trap cell: M1 is the only side that can say "there IS a difference",
                         and greedy decoding is chaotic, so a drift that has not yet flipped a
                         choice can flip on the next token. PASS/FAIL here follows
                         `oracle_gate()`'s rule: ok == (M1 rate == 1.0 AND M2 rate == 1.0), and
                         the 2x2 cross-tab is printed so one number is never quoted for both.

CONFIG COMPARABILITY (added 2026-09-15 -- this is the whole point of the current revision)
-----------------------------------------------------------------------------------------
A cache-ON reference is a *relative* invariance check, and a relative check is only meaningful
between two runs of the SAME numerics-determining configuration. This gate used to ignore that,
and the result was a whole afternoon of chasing a "FAIL" that was really a category error:

  * `ref_iq3_pool8gb_M2_6144.jsonl` (2026-09-15 01:14) was dumped while `CGC_MM_BITIDENT` was
    still ABSENT FROM THE LAUNCH ALLOWLIST (`run_server.sh` only started forwarding it on
    2026-09-15, see the comment on the `-- MTP` block). `CGC_MM_BITIDENT=1` is pillar 1 of the
    bit-identical trio and it changes the M<=8 mul_mat family; without it the SAME logical row
    yields ULP-different values depending on batch size (`ggml-metal-ops.cpp:2395`). A dump taken
    without it can therefore never be bit-identical to a dump taken with it -- measured: only
    `CGC_MM_BITIDENT=0` reproduces the reference's step-0 row hash `e2578ac0ff15b37e`; every other
    single-knob arm lands on `a6302fd4d0ced8c0`.
  * Under the current production configuration the engine is *deterministic*: four independent
    launches produce byte-identical dumps (9/9 rows, 0 mismatches). So the old reference was
    stale, not the engine.

So the gate now resolves the launch environment on BOTH sides via `run_server.sh CGC_DUMP_ENV=1`
(which prints the fully-resolved `CGCENV`/`ENV`/`ARG` and exits before exec -- one source of
truth, no hand-written copy to drift), diffs the numerics-determining subset, and:

  * same config        -> normal M1/M2/M3 verdict.
  * different config   -> prints the diff and exits 2 (INVALID COMPARISON), NOT 1. A cross-config
                          diff is not a regression and must never be reported as one.
  * no sidecar on ref  -> warns that comparability cannot be verified, then compares anyway.

`--write-ref PATH` re-baselines in one step: it copies the fresh dump to PATH and writes a
`PATH.cap` carrying the same resolved config, so the NEXT gate run can prove comparability.
Every gate run also writes its own `Backup/m123_oracle_gate/cap_<tag>.json` for the same reason.

Usage:
  python3 scripts/check/m123_oracle_gate.py
  python3 scripts/check/m123_oracle_gate.py --profile prefill250 --env CGC_SPAC=1
  python3 scripts/check/m123_oracle_gate.py --write-ref Backup/knifeedge_matrix/ref_xxx.jsonl
  python3 scripts/check/m123_oracle_gate.py --allow-incomparable     # read a cross-config diff

Exit code: 0 = M1 and M2 both 1.0, 1 = any difference, 2 = harness error / not comparable.
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMPARE = ROOT / "scripts" / "check" / "cgc_logits_oracle_compare.py"
# v3, dumped 2026-09-15 22:33. Its 9 rows are BYTE-IDENTICAL to v2 (md5
# d2f0b9a01fc5404ebb04dcf3583e133f for both), so this is not a re-baseline of the numerics -- it
# corrects what the sidecar RECORDS.
#
# v2's cap says `CGC_OA_ASYNC=0`, but v2 was dumped at 17:21 against the PRESENCE-based gate
# (`getenv("CGC_OA_ASYNC") != nullptr`), while `run_server.sh` unconditionally puts the variable
# into SERVER_ENV. So that `0` selected the SEGMENTED dispatcher exactly as `1` does; the recorded
# value never had behavioural meaning. When the 21:36 fix made the gate value-aware and the
# launcher default was restored to 1 (preserving the effective behaviour of every profile that
# does not set the knob -- see dec-20260915-2215), `prefill250` resolved to `1` and the gate began
# reporting INVALID COMPARISON (`ENV.CGC_OA_ASYNC: ref='0' now='1'`) on EVERY run, i.e. the
# comparability check itself had gone dark.
#
# The verdict was never in doubt, and the dump proves it independently: M1 was 9/9 bit-identical
# against v2. The non-segmented path lands on `ff68c5a2`, not on the anchor `dc055e63`, so two runs
# that both reproduce `dc055e63` are both on the segmented path no matter what their env string
# says. A missing/incomparable config stamp must not be allowed to veto a bit-identity that the
# logits themselves establish -- and equally, `--allow-incomparable` is not the fix, because it
# discards the check for every future run too.
#
# `prefill250` now PINS CGC_SERVER_OA_ASYNC=1 in run_server.sh (eng-gate-0006: the knob a profile
# fails to state is the knob that drifts). v2 is kept on disk for the historical record.
#
# v3 RETIRED 2026-09-16 by the skip0 value-semantics fix; v4 is the current reference
# (Backup/knifeedge_matrix/ref_iq3_pool8gb_M2_6144_bitident_v4_skip0off.jsonl).
#
# Read this before concluding anything from a v4-vs-v3 diff. The three readers of
# LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0 used to test PRESENCE (`getenv(...) != nullptr`), so the
# profile's `=0` meant ON and v3 was dumped with layer 0 OUT of the pool (39 pooled layers, and
# `CGC-DECPROF layers=39` on all 19 steps of that era's log). The fix made the readers parse the
# VALUE, so the profile's `=0` now means OFF: layer 0 joins the pool, 40 pooled layers, +142
# resident slots and +152 MiB of pool.
#
#   * `LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0=1` (the pre-fix behaviour, reachable via
#     `CGC_SERVER_SKIP0=1`) reproduces v3 BIT-IDENTICALLY: M1 9/9, M2 9/9, M3 9/9, full-row
#     fnv1a64 9/9. That is the evidence that v3 was dumped with skip0 ON and that the predicate
#     change is inert in every other dimension.
#   * The new default does NOT reproduce v3: M1 4/9, M2 9/9, M3 6/9, cross-tab
#     `{same/same: 4, diff/same: 5, diff/diff: 0}` -- drift, not divergence.
#
# ★ The trap this comment exists for: the *resolved env string is identical on both sides* ("0"
# before the fix and "0" after it). The comparability check above is a string diff, so it CANNOT
# detect a value-semantics change and will report a plain M1 FAIL against v3 -- a category error,
# not a regression. Unlike the CGC_OA_ASYNC round (where the string itself moved, so the gate
# raised INVALID COMPARISON on its own), this class of change only ever gets caught by a human
# re-baselining on purpose. If a future knob change is described as "same env, different meaning",
# write a new reference; do not re-read the old one.
DEFAULT_REF = ROOT / "Backup" / "knifeedge_matrix" / "ref_iq3_pool8gb_M2_6144_bitident_v4_skip0off.jsonl"
RESULT_DIR = ROOT / "Backup" / "m123_oracle_gate"

# knifeedge_matrix.PROBE_PROMPT, verbatim. The oracle dump is keyed on
# (step, token_idx, ctx_type), so a different prompt produces a different token sequence and the
# keys stop lining up -- the comparison would then report "0 common keys" instead of a verdict.
PROBE_PROMPT = "15+27 等於多少？請只輸出答案"

# Keys that only steer instrumentation. They must not count as a config difference, or every run
# would look incomparable to every reference (the dump path alone differs per tag).
DIAGNOSTIC_KEYS = {
    "CGC_DUMP_ENV", "CGC_LOGITS_ORACLE_DUMP", "CGC_LOGITS_ORACLE_TOPN",
    "CGC_LOGITS_ORACLE_FIRST_N", "CGC_SLOT_DBG", "CGC_MMID_MV_DBG", "CGC_MMID_ASSERT",
    "CGC_MMID_ASSERT_FATAL", "CGC_PREV_PF_DBG", "CGC_IDS_MAX_LINES", "CGC_SLAB_DBG",
    "CGC_MM_DBG", "CGC_SPAC_DBG", "CGC_CAP_DBG", "CGC_ROUTE_DUMP",
    # [CGC 2026-09-15 S1 slot-table] This key is deliberately in the "does not change the numbers"
    # set, because that IS its claim: CGC_SLOT_TABLE_GPU=1 moves the expert->slot mapping from a
    # host-written input leaf to a graph gather (get_rows over a per-layer table) and must produce
    # the same id vector. Leaving it out would make every S1 run report "incomparable" against the
    # reference, so the gate could never express the one proposition it exists to test. If the
    # claim is false the gate fails on the LOGITS, which is exactly where it should fail.
    "CGC_SLOT_TABLE_GPU",
}
# CGCENV scalars that are comparability-irrelevant (paths/timing only).
DIAGNOSTIC_CGCENV = {"LOG", "PORT"}


def _opener():
    """No-proxy opener: this box runs an HTTP proxy on 7897 that eats 127.0.0.1 requests."""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _get(url, timeout=5.0):
    return _opener().open(url, timeout=timeout).read()


def _post(url, payload, timeout=300.0):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with _opener().open(req, timeout=timeout) as r:
        return r.read()


def kill_servers():
    subprocess.run(["pkill", "-9", "-f", "llama-server"], check=False)
    subprocess.run(["pkill", "-9", "-f", "llama-bench"], check=False)
    time.sleep(1.0)


def server_pids():
    out = subprocess.run(["pgrep", "-f", "build/bin/llama-server"], capture_output=True, text=True)
    return [int(x) for x in out.stdout.split() if x.strip().isdigit()]


def tee(name, text):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / name).write_text(text, errors="replace")


# ---------------------------------------------------------------------------
# [P0 2026-09-15] dump validity -- a gate that can accept a poisoned dump is worse than no gate
# ---------------------------------------------------------------------------
# Motivating incident: a Metal command buffer failed with Insufficient Memory, so the graph never
# ran and the dump was read straight out of the *previous* compute's output buffer. The file was
# structurally valid JSONL, so every consumer downstream treated it as ground truth. The engine now
# aborts on that failure and stamps `<dump>.invalid`; these checks are the second line of defence so
# a dump that is garbage for *any other* reason is also refused.

INVALID_SUFFIX = ".invalid"
# A real LM logit is O(10..100); over <=150K vocab that keeps |sum| under ~1e10 and |mean| under
# ~1e4. These bounds sit ~5-8 orders of magnitude away from anything a forward pass can produce,
# so they exclude garbage without ever risking a false positive on legitimate output.
MAX_ABS_LOGIT = 1.0e30
MAX_ABS_SUM = 1.0e15
MAX_ABS_MEAN = 1.0e9
# Shortest run of consecutive ids tied on the *exact* same bits that we treat as degenerate.
# The incident produced ids 59400..59407 all at 0.125.
MIN_TIED_RUN = 3


def validate_dump(path: Path):
    """Return (ok, reason). Never raises for a malformed record: that *is* a failure.

    Checks, in order of cheapness: the sidecar stamp written by the engine, then per-record
    numeric sanity, then the degeneracy shapes.
    """
    stamp = Path(str(path) + INVALID_SUFFIX)
    if stamp.exists():
        detail = stamp.read_text(errors="replace").strip()
        return False, f"engine stamped this dump invalid ({stamp.name}): {detail}"

    if not path.exists() or path.stat().st_size == 0:
        return False, "dump is missing or empty"

    n_rec = 0
    n_rows_checked = 0
    for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            return False, f"line {lineno} is not valid JSON: {e}"
        n_rec += 1

        for key, bound, label in (("sum", MAX_ABS_SUM, "|sum|"),
                                  ("mean", MAX_ABS_MEAN, "|mean|")):
            v = rec.get(key)
            if v is None:
                continue
            if not math.isfinite(v):
                return False, f"line {lineno}: {key}={v} is not finite"
            if abs(v) > bound:
                return False, f"line {lineno}: {label}={abs(v):.3e} exceeds {bound:.0e} (stale/failed compute)"

        top = rec.get("top") or []
        if not top:
            return False, f"line {lineno}: empty top-k (argmax_token={rec.get('argmax_token')})"
        if not math.isfinite(rec.get("argmax_logit", 0.0)):
            return False, f"line {lineno}: argmax_logit is not finite"
        if abs(rec.get("argmax_logit", 0.0)) > MAX_ABS_LOGIT:
            return False, f"line {lineno}: argmax_logit={rec['argmax_logit']:.3e} exceeds f32 sanity"

        run = 0
        for a, b in zip(top, top[1:]):
            consecutive = int(b["t"]) == int(a["t"]) + 1
            # exact-bit comparison: `==` on floats is what we want here, and it is what the
            # engine-side screen does too
            identical = float(b["v"]) == float(a["v"])
            run = run + 1 if (consecutive and identical) else 0
            if run >= MIN_TIED_RUN:
                return False, (f"line {lineno}: top-k contains {run + 1} consecutive ids tied on "
                               f"identical values (e.g. t={a['t']} -> t={b['t']}, v={a['v']})")
        n_rows_checked += 1

    if n_rec == 0 or n_rows_checked == 0:
        return False, "dump contains no rows"

    return True, f"{n_rec} records"


# ---------------------------------------------------------------------------
# launch-config resolution -- the thing that makes the comparison falsifiable
# ---------------------------------------------------------------------------
def resolve_launch(profile, extra_env, timeout=120.0):
    """Ask `run_server.sh` for the fully-resolved launch config.

    `CGC_DUMP_ENV=1` makes it print CGCENV/ENV/ARG at the exact point where every profile default
    and override has been applied, then `exit 0` BEFORE the exec. So this is cheap, side-effect
    free (apart from its pre-flight pkill, which is why it runs before anything is launched), and
    it can never drift from what the server would have received.
    """
    env = dict(os.environ)
    env["CGC_SERVER_PROFILE"] = profile
    env["CGC_DUMP_ENV"] = "1"
    env.pop("CGC_DETACHED", None)
    for kv in extra_env:
        k, _, v = kv.partition("=")
        env[k] = v
    cp = subprocess.run(["bash", "scripts/run_server.sh"], cwd=str(ROOT), env=env,
                        capture_output=True, text=True, timeout=timeout)
    cgcenv, envs, args = {}, {}, []
    for line in cp.stdout.splitlines():
        if line.startswith("CGCENV "):
            parts = line.split(None, 2)
            if len(parts) == 3:
                cgcenv[parts[1]] = parts[2]
        elif line.startswith("ENV "):
            k, _, v = line[4:].partition("=")
            envs[k] = v
        elif line.startswith("ARG "):
            args.append(line[4:])
    if not cgcenv:
        raise RuntimeError("run_server.sh produced no CGCENV block; CGC_DUMP_ENV is gone?\n"
                           f"--- stdout ---\n{cp.stdout[-2000:]}\n--- stderr ---\n{cp.stderr[-2000:]}")
    return {"cgcenv": cgcenv, "env": envs, "args": args}


def config_stamp(cfg):
    """The numerics-determining subset of a resolved launch, frozen for diffing."""
    return {
        "CGCENV": {k: v for k, v in cfg["cgcenv"].items() if k not in DIAGNOSTIC_CGCENV},
        "ENV": {k: v for k, v in cfg["env"].items() if k not in DIAGNOSTIC_KEYS},
        "ARG": cfg["args"],
    }


def diff_stamp(a, b):
    """Return a list of human-readable differences between two config stamps."""
    out = []
    for section in ("CGCENV", "ENV"):
        ka, kb = a.get(section, {}), b.get(section, {})
        for k in sorted(set(ka) | set(kb)):
            va, vb = ka.get(k, "<absent>"), kb.get(k, "<absent>")
            if va != vb:
                out.append(f"{section}.{k}: ref={va!r}  now={vb!r}")
    if a.get("ARG") != b.get("ARG"):
        sa, sb = a.get("ARG", []), b.get("ARG", [])
        for i in range(max(len(sa), len(sb))):
            va = sa[i] if i < len(sa) else "<absent>"
            vb = sb[i] if i < len(sb) else "<absent>"
            if va != vb:
                out.append(f"ARG[{i}]: ref={va!r}  now={vb!r}")
    return out


def read_cap(path):
    cap = Path(str(path) + ".cap")
    if not cap.exists():
        return None
    try:
        return json.loads(cap.read_text())
    except Exception:  # noqa: BLE001 - a corrupt sidecar is "no sidecar"
        return None


def cap_doc(cfg, note="", extra=None):
    """The provenance document. Same shape the dumper side uses, plus the resolved config."""
    stamp = config_stamp(cfg)
    doc = {
        "kind": "cgc-logits-oracle",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        # Kept so `knifeedge_matrix.py --oracle-abs` still refuses this as ground truth: every
        # reference on this box comes from run_server.sh, which passes -expert-cache <budget>
        # unless CGC_SERVER_EXPERT_CACHE_OFF=1. cache-ON supports relative invariance only.
        "expert_cache": "off" if cfg["cgcenv"].get("BUDGET") == "0" else "on",
        "expert_cache_source": ("run_server.sh passes -expert-cache <budget> unless "
                                "CGC_SERVER_EXPERT_CACHE_OFF=1, so this is a cache-ON dump and "
                                "must never be used as ground truth (--oracle-abs refuses them)"),
        "profile": cfg["cgcenv"].get("PROFILE"),
        "resolved": stamp,
        "note": note,
    }
    if extra:
        doc.update(extra)
    return doc


def write_cap(path, cfg, note="", extra=None):
    """Write the `<path>.cap` provenance sidecar next to a dump."""
    doc = cap_doc(cfg, note=note, extra=extra)
    Path(str(path) + ".cap").write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
    return doc


def write_json(path, doc):
    Path(path).write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="prefill250",
                    help="CGC_SERVER_PROFILE for the launch. The default reproduces the "
                         "configuration the M2 reference oracle was dumped under "
                         "(ctx 8192 / batch 6144 / oa_async=0 / PREFILL_STREAM=1 / SLAB_CAP=256), "
                         "which is a precondition for the comparison to mean anything.")
    ap.add_argument("--ref", default=str(DEFAULT_REF))
    ap.add_argument("--dump", default="/tmp/m123_oracle_gate.jsonl")
    ap.add_argument("--tag", default="")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--env", action="append", default=[],
                    help="extra KEY=VAL handed to run_server.sh (its allowlist still applies; "
                         "a dropped variable prints nothing, so pass only documented knobs)")
    ap.add_argument("--write-ref", default=None,
                    help="after a successful probe, copy the fresh dump to this path and write "
                         "its .cap. This is how you re-baseline; the copied dump is the run that "
                         "just happened, so its provenance is exact by construction.")
    ap.add_argument("--ref-note", default="", help="free-text note stored in the .cap on --write-ref")
    ap.add_argument("--allow-incomparable", action="store_true",
                    help="downgrade a cross-config comparison from exit 2 to a loud warning. Use "
                         "it only to read a diff you already know is cross-config -- never to "
                         "turn it into a regression verdict.")
    ap.add_argument("--allow-empty-probe", action="store_true",
                    help="do not fail the run when the probe answer is empty. Default is to fail: "
                         "an empty answer plus a structurally-valid dump is the 2026-09-15 "
                         "stale-buffer signature, and comparing it would report a false 1.0.")
    ap.add_argument("--allow-invalid-ref", action="store_true",
                    help="skip dump-validity screening on the reference file. Default is to refuse "
                         "a reference that does not pass the same checks as a fresh dump.")
    ap.add_argument("--ready-timeout", type=float, default=300.0)
    ap.add_argument("--teardown-timeout", type=float, default=90.0)
    args = ap.parse_args()

    tag = args.tag or time.strftime("%Y%m%d_%H%M")
    ref = Path(args.ref)
    if not ref.is_absolute():
        ref = ROOT / ref
    if not ref.exists():
        print(f"ERROR: reference oracle missing: {ref}", file=sys.stderr)
        print(f"       create it with: {Path(__file__).name} --write-ref {args.ref}", file=sys.stderr)
        return 2

    # [P0 2026-09-15] A reference is the most dangerous file in the whole harness: everything is
    # measured against it, so a poisoned reference makes every future run look "identical". Screen
    # it with the same rules as a fresh dump, not with a weaker one.
    if not args.allow_invalid_ref:
        ref_ok, ref_reason = validate_dump(ref)
        if not ref_ok:
            print(f"ERROR: reference {ref} fails dump validity: {ref_reason}", file=sys.stderr)
            print("       Re-baseline from a healthy run, or pass --allow-invalid-ref to override "
                  "(the verdict will be meaningless).", file=sys.stderr)
            return 2

    print(f"=== M1/M2/M3 oracle gate (tag {tag}) ===", flush=True)
    print(f"  profile : {args.profile}", flush=True)
    print(f"  ref     : {ref.relative_to(ROOT)}  ({sum(1 for _ in ref.open())} records)", flush=True)
    if args.env:
        print(f"  extra   : {args.env}", flush=True)

    # (1) resolve the launch config FIRST -- before anything is launched, and before kill_servers()
    # so the pkill this incurs cannot race our own server.
    try:
        now_cfg = resolve_launch(args.profile, args.env)
    except Exception as e:  # noqa: BLE001 - harness
        print(f"ERROR: could not resolve the launch configuration: {e}", file=sys.stderr)
        return 2
    now_stamp = config_stamp(now_cfg)

    # (2) comparability precondition. Both sides resolved by the same code path, so this is an
    # exact comparison, not a curated knob list that can go stale.
    ref_cap = read_cap(ref)
    comparable, cfg_diffs = True, []
    if ref_cap is None:
        print("  WARN    : reference has no .cap sidecar -- comparability cannot be verified. "
              "M1/M2/M3 below are unqualified.", flush=True)
    else:
        ref_stamp = ref_cap.get("resolved")
        if ref_stamp is None:
            print("  WARN    : reference .cap carries no resolved config (pre-dates the sidecar "
                  "format) -- comparability cannot be verified.", flush=True)
        else:
            cfg_diffs = diff_stamp(ref_stamp, now_stamp)
            comparable = not cfg_diffs
    if not comparable:
        print(flush=True)
        print("!" * 74)
        print(f"  REFERENCE IS NOT COMPARABLE ({len(cfg_diffs)} numerics-determining difference(s))")
        for d in cfg_diffs[:20]:
            print(f"    {d}")
        if len(cfg_diffs) > 20:
            print(f"    ... and {len(cfg_diffs) - 20} more")
        print("  A cross-config diff is NOT a regression. Re-baseline with:")
        print(f"    python3 scripts/check/m123_oracle_gate.py --profile {args.profile} \\")
        print(f"        --write-ref {ref.relative_to(ROOT)}")
        print("!" * 74, flush=True)

    kill_servers()
    dump = Path(args.dump)
    for p in (dump, Path(str(dump) + ".cap"), Path(str(dump) + INVALID_SUFFIX)):
        if p.exists():
            p.unlink()

    env = dict(os.environ)
    env["CGC_SERVER_PROFILE"] = args.profile
    env["CGC_LOGITS_ORACLE_DUMP"] = str(dump)
    env["CGC_LOGITS_ORACLE_TOPN"] = "8"
    # `run_server.sh` runs the server in the FOREGROUND and `wait`s on it unless CGC_DETACHED=1,
    # in which case it re-execs itself through a detach helper, prints `[detach] server PID=N` and
    # exits 0 (run_server.sh:43 / :1373). Without this the launcher would block until teardown --
    # and with stdout on a pipe it would block forever, because the server inherits the fd.
    env["CGC_DETACHED"] = "1"
    for kv in args.env:
        k, _, v = kv.partition("=")
        env[k] = v

    # `run_server.sh` DETACHES its child with `set -m` and exits. It must therefore be launched
    # with stdout redirected to a FILE, never a pipe: the detached server inherits the fd, so
    # `subprocess.run(capture_output=True)` (or `communicate()`) blocks on a pipe EOF that only
    # arrives when the SERVER exits -- measured: a 420 s timeout while the server was healthy and
    # already serving. `start_new_session=True` keeps Ctrl-C on our side from reaching it, and we
    # signal the recorded leader PID explicitly during teardown.
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(RESULT_DIR / f"cap_{tag}.json",
               cap_doc(now_cfg, note=f"resolved launch config for gate run {tag} "
                                     f"(provenance, not a reference)"))
    launch_log = RESULT_DIR / f"launch_{tag}.log"
    with launch_log.open("w") as lf:
        proc = subprocess.Popen(["bash", "scripts/run_server.sh"], cwd=str(ROOT), env=env,
                                stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            proc.wait(timeout=180.0)
        except subprocess.TimeoutExpired:
            print("ERROR: run_server.sh did not return within 180s (it should detach and exit)",
                  file=sys.stderr)
            kill_servers()
            return 2
    out = launch_log.read_text(errors="replace")
    print(out.rstrip(), flush=True)
    # The leader line carries the PID run_server.sh detached -- the only reliable handle, because
    # the banner's other PID-like numbers are the parent's.
    m = re.search(r"server PID=(\d+)", out)
    leader = int(m.group(1)) if m else None
    srv_log = None
    lm = re.search(r"\[log\]\s+([^\s（(]+)", out)
    if lm:
        srv_log = Path(lm.group(1))
    print(f"  leader  : {leader}   server log: {srv_log}", flush=True)

    base = f"http://127.0.0.1:{args.port}/v1"
    t0 = time.time()
    ready = False
    while time.time() - t0 < args.ready_timeout:
        try:
            _get(f"{base}/models", timeout=3.0)
            ready = True
            break
        except Exception:  # noqa: BLE001 - poll loop
            time.sleep(2.0)
    if not ready:
        print(f"ERROR: server did not become ready within {args.ready_timeout}s", file=sys.stderr)
        kill_servers()
        return 2
    print(f"  ready   : {time.time() - t0:.0f}s", flush=True)

    payload = {"model": "local", "messages": [{"role": "user", "content": PROBE_PROMPT}],
               "temperature": 0.0, "max_tokens": 48}
    try:
        body = _post(f"{base}/chat/completions", payload)
        ans = json.loads(body).get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:  # noqa: BLE001 - harness
        print(f"WARN: probe request failed: {e}", flush=True)
        ans = ""
    print(f"  probe   : {PROBE_PROMPT!r} -> {ans.strip()[:60]!r}", flush=True)
    time.sleep(1.5)

    # SIGINT so the cache teardown stats are emitted into the server log.
    if leader:
        subprocess.run(["kill", "-INT", str(leader)], check=False)
    else:
        subprocess.run(["pkill", "-INT", "-f", "build/bin/llama-server"], check=False)
    t1 = time.time()
    while time.time() - t1 < args.teardown_timeout:
        if not server_pids():
            break
        time.sleep(1.0)
    if server_pids():
        print("WARN: server did not exit on SIGINT; killing", flush=True)
        kill_servers()
    print(f"  stopped : after {time.time() - t1:.0f}s", flush=True)

    if not dump.exists() or dump.stat().st_size == 0:
        print(f"ERROR: no oracle dump at {dump}; the server never wrote one. "
              f"CGC_LOGITS_ORACLE_DUMP must survive run_server.sh's env allowlist "
              f"(run_server.sh:989).", file=sys.stderr)
        return 2

    # [P0 2026-09-15] Refuse to build a verdict on a run that cannot be trusted. Order matters:
    # both checks sit BEFORE write_cap()/--write-ref, so an invalid run can never become a
    # reference and can never acquire a provenance sidecar that makes it look legitimate.
    if not ans.strip() and not args.allow_empty_probe:
        print(flush=True)
        print("!" * 74)
        print("  INVALID RUN: the probe returned an empty answer.")
        print("  An empty answer with a healthy-looking dump is the exact signature of the")
        print("  2026-09-15 Metal OOM (the graph never ran; the server replied HTTP 200 with")
        print("  stale bytes). M1/M2/M3 below would happily report 1.0 on such a dump, which")
        print("  is precisely the failure mode this gate must not have. Not comparing.")
        print(f"  Server log: {srv_log}")
        print("  Override with --allow-empty-probe only if the emptiness is known-unrelated.")
        print("!" * 74, flush=True)
        return 2

    ok_dump, reason = validate_dump(dump)
    if not ok_dump:
        print(flush=True)
        print("!" * 74)
        print(f"  INVALID DUMP: {reason}")
        print(f"  {dump} is not usable as an oracle reference and was NOT made one.")
        print("  Investigate the engine-side failure first; do not re-run until it is understood.")
        print("!" * 74, flush=True)
        return 2
    print(f"  dump    : {reason}, validated", flush=True)

    # The fresh dump's own provenance, carried with the fresh dump.
    write_cap(dump, now_cfg, note=f"gate run {tag}",
              extra={"probe_answer": ans.strip(), "validated": True})

    # (3) re-baseline in one step when asked. Done BEFORE the comparison so a --write-ref run
    # also reports what the new baseline looks like against the old one.
    if args.write_ref:
        dst = Path(args.write_ref)
        if not dst.is_absolute():
            dst = ROOT / dst
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(dump, dst)
        # a stale sidecar next to the destination would make the *new* reference look invalid
        stale = Path(str(dst) + INVALID_SUFFIX)
        if stale.exists():
            stale.unlink()
        write_cap(dst, now_cfg, note=args.ref_note or f"re-baselined by gate run {tag}",
                  extra={"probe_answer": ans.strip()})
        print(f"  baseline: wrote {dst.relative_to(ROOT)} + .cap (this run is the new reference)",
              flush=True)

    report = RESULT_DIR / f"oraclecmp_{tag}.json"
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    cp = subprocess.run([sys.executable, str(COMPARE), "--a", str(ref), "--b", str(dump),
                         "--report", str(report)],
                        cwd=str(ROOT), capture_output=True, text=True)
    print(cp.stdout, flush=True)
    if cp.stderr.strip():
        print(cp.stderr, file=sys.stderr, flush=True)
    tee(f"cmp_{tag}.log", cp.stdout)

    if not report.exists():
        print("ERROR: compare produced no report", file=sys.stderr)
        return 2
    d = json.loads(report.read_text())
    met = d["metrics"]
    m1r, m2r, m3r = (met["numeric_identity"]["rate"], met["decision_agreement"]["rate"],
                     met["topk_set_agreement"]["rate"])
    summary = {
        "tag": tag, "profile": args.profile, "ref": str(ref), "dump": str(dump),
        "probe_answer": ans.strip(),
        "m1_numeric_identity": f"{met['numeric_identity']['equal']}/{met['numeric_identity']['n']}",
        "m2_decision_agreement": f"{met['decision_agreement']['equal']}/{met['decision_agreement']['n']}",
        "m3_topk_set_agreement": f"{met['topk_set_agreement']['equal']}/{met['topk_set_agreement']['n']}",
        "n_compared": met["numeric_identity"]["n"],
        "cross_tab": d["cross_tab"],
        "comparable": comparable,
        "config_diffs": cfg_diffs,
        "ok": bool(comparable and m1r == 1.0 and m2r == 1.0),
        "report": str(report),
    }
    (RESULT_DIR / f"summary_{tag}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=" * 74)
    if not comparable:
        print(f"GATE {tag}: INVALID COMPARISON -- reference was dumped under a different "
              f"numerics-determining configuration ({len(cfg_diffs)} diffs).")
        print(f"  observed M1={summary['m1_numeric_identity']}  M2={summary['m2_decision_agreement']}  "
              f"M3={summary['m3_topk_set_agreement']}  n={summary['n_compared']} "
              f"(printed for information, NOT a verdict)")
    else:
        print(f"GATE {tag}: {'PASS' if summary['ok'] else 'FAIL'}   "
              f"M1(bit-identical)={summary['m1_numeric_identity']}  "
              f"M2(argmax)={summary['m2_decision_agreement']}  "
              f"M3(topk)={summary['m3_topk_set_agreement']}  n={summary['n_compared']}")
        print(f"cross-tab: {d['cross_tab']}")
    print(f"summary  : {RESULT_DIR / f'summary_{tag}.json'}")
    print("=" * 74)
    if not comparable:
        return 2 if not args.allow_incomparable else (0 if summary["ok"] else 1)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
