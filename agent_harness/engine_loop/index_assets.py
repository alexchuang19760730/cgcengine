#!/usr/bin/env python3
"""Generate MANIFEST.jsonl — the machine-readable inventory of the engine loop's assets.

Why this is generated rather than hand-written: a hand-kept inventory of 40+ files goes stale the
first time someone renames a script, and a stale inventory is worse than none because it is trusted.
Every row here is verified to exist at generation time; a role that cannot be matched to a real file
is reported instead of silently kept.

The `role` column is human judgement (that is the part a script cannot supply) and lives in the
curated table below. Everything else — existence, size, mtime, whether the file is a production
script, whether it produces trace records — is derived.

    index_assets.py                  regenerate MANIFEST.jsonl
    index_assets.py --check          verify the existing manifest against disk; exit 1 on drift
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

assert os.path.basename(os.path.dirname(os.path.abspath(__file__))) == "engine_loop", \
    "index_assets.py belongs in agent_harness/engine_loop/"

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

# role -> what kind of thing it is
#   runner     launches or builds the engine
#   arms       defines what is under test
#   measure    produces numbers
#   compare    turns numbers into a verdict about a delta
#   gate       decides correct/incorrect
#   probe      explores one mechanism, never quotable
#   evidence   raw artifact produced by a run
#   conclusion the outward-facing statement
#   log        the decision journal
#   index      derives a lookup artifact from sources it does not own (never a copy of them)
# `loop` is the loop that owns the asset; `authority` is the single path that must not be copied.

CURATED = [
    # ---- engine_loop (today's harness) ------------------------------------------------------
    ("scripts/run_server.sh", "runner", "engine", True, True,
     "launches llama-server for a profile. Holds the CGC_* env ALLOWLIST: a var that is not listed "
     "is silently dropped, which is indistinguishable from 'the change had no effect'. Every probe "
     "env must be added here before its arm means anything."),
    ("scripts/check/decode_sweep.py", "arms", "engine", True, True,
     "the ARMS table: each arm = env dict + a comment block stating what it is for. Also parses the "
     "server log into the row (tps, md5 set, pool counters, phase)."),
    ("scripts/check/decode_bench.py", "measure", "engine", True, True,
     "runs N rounds against a live server and returns the sample + per-round md5."),
    ("scripts/check/ab_interleave.py", "compare", "engine", True, True,
     "the only quotable A/B: interleaves arms, reports the paired per-rep ratio median, pins the "
     "build fingerprint, supports --report-only."),
    ("scripts/check/m123_oracle_gate.py", "gate", "engine", True, True,
     "logits-oracle gate against a reference dump. DIAGNOSTIC_KEYS lets the proposition under test "
     "through the allowlist but records it."),
    ("scripts/check/knifeedge_matrix.py", "gate", "engine", True, False,
     "the gate matrix (181 KB). Large; candidate for splitting in E4. Call it, do not copy it."),
    ("scripts/check/llama_bench_matrix.py", "measure", "engine", True, True,
     "llama-bench across context lengths; records build_commit, which is the only build identity a "
     "bench run leaves behind."),
    ("scripts/check/oracle_truth_gate_selftest.py", "gate", "engine", True, False,
     "proves the gate REJECTS a wrong input. A gate that has never been shown to fail is not a gate."),
    ("scripts/check/feasibility_gate_selftest.py", "gate", "engine", True, False,
     "same, for the feasibility gate."),
    ("scripts/check/mmid_geometry_probe.sh", "probe", "engine", True, False,
     "explores the mul_mat_id geometry (pool slots vs full width)."),
    ("scripts/check/mmid_pool_vs_gguf.py", "probe", "engine", True, False,
     "cross-checks pool-resident expert bytes against the GGUF source."),
    ("scripts/check/mmid_zero_row_triage.py", "probe", "engine", True, False,
     "splits every CGC-MMID-ASSERT into MODEL-ZERO vs ENGINE-ZERO. Classify before fixing."),
    ("scripts/check/flip_rate.py", "probe", "engine", True, False,
     "route-flip rate; the measurement behind the 'which experts change' question."),
    ("scripts/check/mtp_accept_ab.py", "compare", "engine", True, False,
     "MTP draft acceptance A/B; the second reason decode throughput moves."),
    ("scripts/check/mtp_head_identity.py", "probe", "engine", True, False,
     "verifies the MTP head's identity behaviour."),
    ("scripts/check/mtp_long_sequence_verify.py", "probe", "engine", True, False,
     "MTP verification over long sequences."),
    ("scripts/check/check_server.sh", "probe", "engine", True, False,
     "OpenAI-compat regression checks against a live server."),

    # ---- the trace pipeline (this stage) ---------------------------------------------------
    ("agent_harness/engine_loop/traces/emit_episodes.py", "measure", "engine", True, True,
     "T0 emitter: artifacts on disk -> episodes.jsonl. Pure reader; never runs the engine."),
    ("agent_harness/engine_loop/traces/validate.py", "gate", "engine", True, True,
     "schema + cross-record integrity. Enforces build==null => not usable_as_evidence."),
    ("agent_harness/engine_loop/traces/schema/episode.schema.json", "gate", "engine", True, False,
     "the atomic observation: one run, one arm."),
    ("agent_harness/engine_loop/traces/schema/decision.schema.json", "gate", "engine", True, False,
     "the judgement point: question + evidence + reasoning + ruled_out."),
    ("agent_harness/engine_loop/traces/schema/lesson.schema.json", "gate", "engine", True, False,
     "the generalised rule that gets injected back into the harness."),
    ("agent_harness/CONVENTIONS.md", "conclusion", "shared", False, True,
     "the charter. Also the system prompt for sft_pi, so editing it changes runtime behaviour."),
    ("agent_harness/PLAN_ENGINE_LOOP_2026-09-15.md", "conclusion", "shared", False, False,
     "the plan this stage executes. E0 = indexes + first episode export."),
    ("agent_harness/README.md", "conclusion", "tb_loop", False, False,
     "the pre-existing tb_loop README (Terminal-Bench x gemma4 x prime-agent)."),

    # ---- evidence roots --------------------------------------------------------------------
    ("Backup/phase_decomp", "evidence", "engine", False, True,
     "sweep/AB rows as JSON. Source of every quotable throughput number."),
    ("Backup/m123_oracle_gate", "evidence", "engine", False, True,
     "gate dumps and .cap provenance files. A .cap is NOT a pass."),
    ("Backup/knifeedge_matrix", "evidence", "engine", False, True,
     "oracle dumps (*.jsonl) + their .cap, and the capability-invariance matrix. Holds the "
     "bit-identical reference."),
    ("Backup/llama_bench", "evidence", "engine", False, True,
     "llama-bench matrices. NOTE: several are `incomplete` with a harness error; the emitter records "
     "that rather than the cells it did get."),
    ("Backup/cgc_logs", "log", "engine", False, True,
     "463 MB / ~1400 files. NOT committed. The emitter digests only logs < 32 MiB and stores the "
     "path, so an episode points at the original instead of duplicating it."),
    (".workbuddy/memory/2026-09-15.md", "log", "shared", False, True,
     "the decision journal. Highest information density of any artifact today; T1 distillation "
     "reads it. Reachable section-by-section through engine_loop/memory/INDEX.jsonl -- this file is "
     "1000+ lines and grows all day, so reading it whole is how a consumer silently truncates."),
    (".workbuddy/memory/MEMORY.md", "log", "shared", False, True,
     "the cross-day project facts (model identity, prod25 geometry, build/test entry points, "
     "measurement hygiene). Same index, same rule: canonical here, never copied into the harness."),
    ("agent_harness/engine_loop/memory/build_memory_index.py", "index", "engine", True, False,
     "derives INDEX.jsonl from .workbuddy/memory/ (one row per file + one per `##` section) and "
     "answers --query without loading the memory. --check re-derives and fails on drift."),
    ("agent_harness/engine_loop/memory/INDEX.jsonl", "index", "engine", True, False,
     "the derived section index: path + heading + line range + `###` subheadings, so a consumer "
     "reads .workbuddy/memory/YYYY-MM-DD.md:854-924 instead of the file. Regenerated, not edited."),
    ("agent_harness/engine_loop/memory/README.md", "conclusion", "engine", False, True,
     "why the memory is indexed rather than mirrored (a copy is a second source of truth whose "
     "failure mode is silent), and how to query it."),
    ("docs/REMAP_ROUNDTRIP_REMOVAL_PLAN_2026-09-15.md", "conclusion", "engine", False, True,
     "the D0-D3 ladder and the S1-S3 staging; S1's contract is defined here."),
    ("docs/MMID_GEOMETRY_PROBE_2026-09-15.md", "conclusion", "engine", False, False,
     "the mul_mat_id geometry findings."),
    ("docs/LATEST_COMMIT_GAP_ANALYSIS_2026-09-15.md", "conclusion", "engine", False, False,
     "gap analysis against the current commit."),
]


# Assets that are OUTPUTS of running the loop, not sources: their bytes and mtime change on every
# run by design. `--check` therefore only verifies they still exist, because comparing their
# content would keep the check permanently red -- and a gate that always fails is no more useful
# than one that never does (CONVENTIONS.md B7 is about the second failure mode; this is the first).
# They stay in the manifest on purpose: the entry is what tells a reader the dump exists at all.
VOLATILE_PREFIXES = ("Backup/", ".workbuddy/memory/")


def is_volatile(path: str) -> bool:
    return path.startswith(VOLATILE_PREFIXES)


def row_for(path: str, role: str, loop: str, replayable: bool, produces_record: bool, notes: str) -> dict:
    abs_p = os.path.join(REPO, path)
    exists = os.path.exists(abs_p)
    is_dir = os.path.isdir(abs_p)
    size = None
    mtime = None
    if exists and not is_dir:
        size = os.path.getsize(abs_p)
        mtime = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(os.path.getmtime(abs_p)))
    elif exists:
        n = 0
        total = 0
        for root, _dirs, files in os.walk(abs_p):
            for f in files:
                n += 1
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        size, mtime = total, f"{n} files"
    return {
        "asset_id": path.replace("/", "."),
        "path": path,
        "role": role,
        "loop": loop,
        "exists": exists,
        "bytes": size,
        "mtime": mtime,
        "volatile": is_volatile(path),
        "replayable": replayable,
        "produces_record": produces_record,
        "authority": True,
        "notes": notes,
    }


def build() -> list:
    rows = []
    for path, role, loop, replayable, produces, notes in CURATED:
        rows.append(row_for(path, role, loop, replayable, produces, notes=notes))
    # glob extras so newly added scripts are never invisible
    try:
        import glob
        for p in sorted(glob.glob(os.path.join(REPO, "scripts/check/*"))):
            rel = os.path.relpath(p, REPO)
            if any(r["path"] == rel for r in rows):
                continue
            if not os.path.isfile(p):
                continue
            rows.append(row_for(rel, "probe", "engine", True, False,
                                "(auto-indexed: not in the curated role table — give it a role)"))
    except Exception:
        pass
    rows.sort(key=lambda r: (r["loop"], r["role"], r["path"]))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HERE, "MANIFEST.jsonl"))
    ap.add_argument("--check", action="store_true",
                    help="compare the manifest against disk (existence, bytes, mtime) and "
                         "against the fresh listing; exit 1 on any drift")
    args = ap.parse_args()

    rows = build()
    missing = [r["path"] for r in rows if not r["exists"]]
    auto = [r["path"] for r in rows if r["notes"].startswith("(auto-indexed")]

    if args.check:
        if not os.path.exists(args.out):
            print(f"  [error] no manifest at {args.out}")
            return 1
        have = {}
        with open(args.out, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    have[r["path"]] = r
        # A check that can be skipped is not a check (CONVENTIONS.md B7). The first version of
        # this function only asked whether each recorded path still existed, so editing a file it
        # had indexed was invisible: it printed "0 drift" while MANIFEST.jsonl still carried the
        # old byte count and mtime. That is the exact failure B7 describes -- the gate had never
        # been shown to reject anything -- and it was found by editing
        # scripts/check/decode_sweep.py and watching --check stay green.
        problems = []
        for p, old in have.items():
            abs_p = os.path.join(REPO, p)
            here = os.path.exists(abs_p)
            if old.get("exists") and not here:
                problems.append((p, "recorded as present, no longer on disk"))
            elif not old.get("exists") and here:
                problems.append((p, "recorded as absent, now present"))
        for p, new in ((r["path"], r) for r in rows if r["exists"]):
            old = have.get(p)
            if old is None:
                problems.append((p, "on disk but not in the manifest"))
                continue
            if new.get("volatile"):
                continue  # exists-only: see VOLATILE_PREFIXES
            for k in ("bytes", "mtime"):
                if old.get(k) != new.get(k):
                    problems.append((p, f"{k}: manifest {old.get(k)!r} vs disk {new.get(k)!r}"))
        if problems:
            print(f"  [error] {len(problems)} drift(s) between the manifest and disk:")
            for p, why in problems:
                print(f"    {p}: {why}")
            print("    regenerate with: python3 index_assets.py")
            print("    (no --out: the default is this file's own directory. A RELATIVE --out is")
            print("     resolved against the cwd, so `--out MANIFEST.jsonl` from the repo root")
            print("     writes a second manifest there and leaves this one stale -- which reads")
            print("     as 8 drifts and one invisible second copy of the truth.)")
            return 1
        print(f"  manifest OK: {len(have)} assets, existence + bytes + mtime all agree")
        return 0

    with open(args.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows)} assets -> {args.out}")
    if missing:
        print(f"  [warn] {len(missing)} curated path(s) do not exist:")
        for p in missing:
            print(f"    {p}")
    if auto:
        print(f"  [info] {len(auto)} auto-indexed script(s) still need a role and a note")
    by_role: dict = {}
    for r in rows:
        by_role[r["role"]] = by_role.get(r["role"], 0) + 1
    print(f"  by role: {by_role}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
