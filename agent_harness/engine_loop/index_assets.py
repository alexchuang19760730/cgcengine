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
     "runs N rounds against a live server and returns the sample + per-round md5. Reads the "
     "thermal pressure level on both sides of every round (thermal_pressure.py)."),
    ("scripts/check/thermal_pressure.py", "measure", "engine", True, False,
     "the ONE reader of com.apple.system.thermalpressurelevel (~2 ms, no root, the same notify(3) "
     "key powermetrics samples), plus `Sampler` -- the 2 Hz background series a llama-bench arm "
     "needs because that child process emits nothing to hang a per-request reading on. Imported by "
     "decode_bench/decode_sweep/llama_bench_matrix rather than copied. Read its docstring before "
     "trusting a level: `notifyutil -g` answers 0 for a key that DOES NOT EXIST, so an absent "
     "instrument is indistinguishable in band from a cool machine."),
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
     "bench run leaves behind. Also records a thermal launch+series reading per arm. NOT the "
     "instrument of record for decode: llama-bench has no sampler (zero matches for "
     "sampler|speculat|draft|MTP) and advances the sequence with `std::rand() % n_vocab`, so it "
     "cannot execute the MTP path and does not measure a served token stream (eng-mh-0040)."),
    ("scripts/check/oracle_truth_gate_selftest.py", "gate", "engine", True, False,
     "proves the gate REJECTS a wrong input. A gate that has never been shown to fail is not a gate."),
    ("scripts/check/feasibility_gate_selftest.py", "gate", "engine", True, False,
     "same, for the feasibility gate."),
    ("scripts/check/expert_cache_ensure_batch_order.cpp", "gate", "engine", True, False,
     "order-dependence test for llama_expert_cache_ensure_batch, and the discriminating evidence "
     "for the Blocker B fix (docs/M1_POOL_SPLIT_COST_2026-09-14.md §4.1). No model, no server, no "
     "IO: it links the REAL ensure_batch out of libllama and hands a synthetic cache the state "
     "prepopulate leaves behind (pool full, slot_last_use all 0, one cold member whose slot an "
     "in-flight fill holds). TWO cases as of 2026-09-16, ALL PASS (2/2). Case 1 is the ORDER: "
     "one-loop -> st[0]=1 .. st[15]=16, 0 hit / 16 miss, FAIL, which is the 09-14 capture's "
     "mapping reproduced word for word; two-pass -> 15 hit / 1 miss, PASS, batch_evict_batches=1. "
     "Case 2 is ADOPTION: an in-flight fill must be ADOPTED, not handed a second slot -- "
     "adopted_queued=1, misses=0, PASS. Case 2 exists because case 1 alone left the adoption path "
     "invisible and n_hit_adopted_queued write-only (CONVENTIONS B13). This is still the test the "
     "live SPLIT configuration could not be: both 10 GiB and 4 GiB split runs die in warmup with "
     "Metal Insufficient Memory before serving a request -- but the reordering is now ALSO "
     "measured green on the SHIPPING configuration, through the counters this test made readable "
     "(§4.2: batch_evict_batches=325, batch_invariant_checks=331, violations=0). Rebuild the "
     "LIBRARY from the same tree state first - llama_expert_cache crosses the dylib boundary by "
     "pointer, so a stale-dylib/fresh-header pair crashes inside ensure_batch instead of failing "
     "to link."),
    ("scripts/check/mmid_geometry_probe.sh", "probe", "engine", True, False,
     "explores the mul_mat_id geometry (pool slots vs full width)."),
    ("scripts/check/mmid_pool_vs_gguf.py", "probe", "engine", True, False,
     "cross-checks pool-resident expert bytes against the GGUF source."),
    ("scripts/check/mmid_zero_row_triage.py", "probe", "engine", True, False,
     "adjudicates CGC-MMID-ASSERT zero_row AND pool-integrity zero-region alarms against the GGUF. "
     "Reads the WHOLE row stride: MODEL-ZERO / PROBE-TOO-NARROW / ENGINE-ZERO. Exit 3 = fix the probe."),
    ("scripts/check/gguf_dead_expert_census.py", "probe", "engine", True, False,
     "census of all-zero and zero-PREFIXED expert rows straight from the GGUF, no log needed. "
     "The file-side ground truth behind every zero_row / zero-region alarm."),
    ("scripts/check/prefill_certifiability.py", "probe", "engine", True, False,
     "N independent processes, one identical command, launch memory state recorded: is a "
     "throughput number a spec or a draw? --warm-runs forces the page cache up to break the "
     "run-order confound. --idle-before sleeps before the first launch, which is what exposes the "
     "one state that actually matters (see A16). --logdir gives every launch its own stderr, or "
     "N launches overwrite each other and the fast one's evidence is gone. Quote the repeatable "
     "band, not the target."),
    ("scripts/check/prefill_gputime_report.py", "probe", "engine", True, False,
     "Per-launch GPU decomposition from the preserved cert_runs logs: of the wall time the engine "
     "spends spinning for a prefill graph's command buffers, how much is the GPU actually busy "
     "(Metal GPUStartTime/GPUEndTime)? Decides 'the GPU is slow' vs 'the launch path is slow'. "
     "Needs the run to have been launched with CGC_GPU_TIMING=1. --decprof-pair <fast> <slow> "
     "compares the FIRST prefill graph of two CGC-DECPROF logs layer by layer: if the per-layer "
     "slow/fast ratio matches the throughput ratio, the graph scales UNIFORMLY and the cost is a "
     "clock/power ceiling, not a layer-localised implementation bug. L0 is excluded on purpose -- "
     "with submit_ahead, layer i's wait is gated by accumulated GPU backlog, so L0 reads lowest "
     "even in a perfectly healthy run (112 ms vs a 155 ms plateau), and including it forges a "
     "ramp that reads as localisation."),
    ("scripts/check/prefill_idle_sweep.py", "probe", "engine", True, False,
     "Drives successive prefill_certifiability.py sessions at increasing --idle-before, which is "
     "the one independent variable that exposes the thermal transient (A16/A17). --preheat runs one "
     "DISCARDED launch first, because otherwise session 1 starts from the cold state and the "
     "0-second point is not comparable to the rest. Result 2026-09-16: 120 s < tau <= 150 s, and "
     "the hot state has no floor (122.39 t/s after repeated hot runs). --reharvest re-derives the "
     "table from an existing sweep's saved logs with the harvester as it exists NOW, which is "
     "required after any mid-sweep edit to the harvester (A19: this script keeps the code it "
     "imported at start, so the table and the per-session summaries can be different vintages "
     "without either one complaining)."),
    ("scripts/check/ioreport_gpu_pstate_probe.py", "probe", "engine", True, False,
     "Tries to read GPU performance-state residency unprivileged via IOReport (dlopen "
     "/usr/lib/libIOReport.dylib + CoreFoundation). Result is a PROVEN NEGATIVE on this machine: "
     "channels enumerate, but IOReportCreateSubscription returns NULL with a mutable dict and "
     "raises -[__NSDictionaryM objectAtIndex:] with the alternative -- so this cannot replace root "
     "powermetrics. Kept because 'we tried and here is why it cannot work' stops the next attempt."),
    ("scripts/check/powermetrics_gpu_freq.sh", "probe", "engine", True, False,
     "Turnkey root capture for the hardware-frequency half of the GPU ceiling claim: GPU active "
     "RESIDENCY is already proven (100-101% busy/wait), but frequency is not, and residency alone "
     "cannot separate 'the clock dropped' from 'the work grew'. --parse now delegates to "
     "powermetrics_gpu_freq_parse.py and takes SEVERAL logs (a glob is the natural call once more "
     "than one capture exists). Must be run by the user with sudo: the privilege escalation is "
     "refused even with the workspace sandbox off (/usr/bin/sudo is setuid and its exec is denied), "
     "so this is the one next step that cannot be automated. Stops via a root watcher + sentinel + "
     "`-n` ceiling + SIGTERM (NOT `kill $!`: that pid is root's, and background jobs ignore SIGINT). "
     "If a run is ever left behind, `pgrep -fl powermetrics` shows TWO pids -- the `sudo` WRAPPER is "
     "yours and signalling it takes both down; see eng-gate-0027."),
    ("scripts/check/powermetrics_gpu_freq_parse.py", "probe", "engine", True, False,
     "Turns a powermetrics capture into the cold-vs-hot answer instead of two 480-number series. "
     "Segments activity blocks on GPU POWER -- deliberately not on frequency or residency, which "
     "are the quantities under test, so defining 'one run' by them would beg the question -- labels "
     "block 0 cold and the rest hot, and reports the DVFS step-residency distribution that the "
     "single 'GPU HW active frequency' value collapses away (a clock cap shows as mass leaving the "
     "top step, which the collapsed number cannot show). ALSO reads `Current pressure level` from the "
     "thermal sampler, per block: that is the field that says WHETHER the ceiling holding the clock "
     "down is thermal, and it was missing until 2026-09-16 -- the instrument had been answering only "
     "half of the question it was written for (eng-gate-0025). The block threshold is a HIGH "
     "percentile (0.15 x p99.5 of power) on purpose: a whole-capture p95 collapses into the idle "
     "distribution once the capture has a long tail, which turned 3 launches into 25-31 blocks "
     "(eng-gate-0024 / A20). Verdict is one of CLOCK / POWER CEILING / "
     "GPU KEPT OFF THE WORK, with the numbers behind each test printed. Fails LOUDLY on BOTH kinds "
     "of empty: if the labels change it prints the GPU lines it did see, and if the file is not a "
     "capture at all -- a run that died before powermetrics started leaves an error line behind, "
     "which is exactly what the 2026-09-16 attempt did -- it prints the file's own first lines, so "
     "'the capture never ran' cannot be misread as 'the capture measured zero'. Several logs at "
     "once; a file with no samples is listed as SKIPPED rather than silently dropped from the "
     "argument list. --selftest: nine cases -- the four verdicts, two thermal cases (escalation vs "
     "flat), two multi-file cases, and a REGRESSION for the threshold drift whose first version "
     "passed for the wrong reason (a flat synthetic tail merges into the last launch; the real idle "
     "is spiky, and the old rule then reports 401 blocks against the new rule's 2)."),
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
     "the plan this stage executes. E0 = indexes + first episode export. Its section 9 table now "
     "carries the E1 actual-closure status (package resolution GREEN, the tb run smoke NOT RUN)."),
    ("agent_harness/README.md", "conclusion", "shared", False, False,
     "the UMBRELLA entry (E1): the two loops, the three record types, the red line, the three path "
     "anchors, and the honest list of entries that belong to neither loop. It used to BE the tb_loop "
     "README; that content moved to agent_harness/tb_loop/README.md at E1."),
    ("agent_harness/tb_loop/README.md", "conclusion", "tb_loop", False, False,
     "the tb_loop README (Terminal-Bench x gemma4 x prime-agent): install, Windows rehearsal, SFT "
     "data generation, MLX LoRA, the /refine evidence sources. Read it before touching tb_loop/."),
    ("agent_harness/tb_loop/config.env", "runner", "tb_loop", True, False,
     "every tb_loop parameter, and the THREE PATH ANCHORS: TB_LOOP_DIR (tb_loop itself), "
     "TB_HARNESS_ROOT (agent_harness/, for assets that do NOT move with tb_loop: loopmoe/, "
     "loopmoe_output/, pd_data/), TB_REPO_ROOT (for app/cloud/freebuff2api/.env). E1 split them "
     "because a path that merely HAPPENS to equal another is a time bomb: "
     "PYTHONPATH=\"$TB_LOOP_DIR\" was simultaneously right and wrong until something moved."),

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
     "measurement hygiene). Canonical here -- quote THIS path, not the transport snapshot under "
     "agent_harness/memory/ (D6 as amended 2026-09-16)."),
    ("agent_harness/engine_loop/memory/build_memory_index.py", "index", "engine", True, False,
     "derives INDEX.jsonl from .workbuddy/memory/ (one row per file + one per `##` section) and "
     "answers --query without loading the memory. --check re-derives and fails on drift."),
    ("agent_harness/engine_loop/memory/INDEX.jsonl", "index", "engine", True, False,
     "the derived section index: path + heading + line range + `###` subheadings, so a consumer "
     "reads .workbuddy/memory/YYYY-MM-DD.md:854-924 instead of the file. Regenerated, not edited."),
    ("agent_harness/engine_loop/memory/README.md", "conclusion", "engine", False, True,
     "why the memory is indexed, and the 2026-09-16 revision that ALSO keeps a dated physical "
     "snapshot (agent_harness/memory/, agent_harness/skills/) because the periodic pusher ships "
     "content, not pointers. Read it before touching either mechanism."),
    ("docs/REMAP_ROUNDTRIP_REMOVAL_PLAN_2026-09-15.md", "conclusion", "engine", False, True,
     "the D0-D3 ladder and the S1-S3 staging; S1's contract is defined here."),
    ("docs/MMID_GEOMETRY_PROBE_2026-09-15.md", "conclusion", "engine", False, False,
     "the mul_mat_id geometry findings."),
    ("docs/LATEST_COMMIT_GAP_ANALYSIS_2026-09-15.md", "conclusion", "engine", False, False,
     "gap analysis against the current commit."),
    ("docs/PREFILL250_CERTIFIABILITY_20260916.html", "conclusion", "engine", False, True,
     "1/11 runs cleared 250; the 155-184 band is the operating point; free memory AND page cache "
     "both refuted as explanations; and the self-correction of 'nothing is blocking 250/25'. "
     "SUPERSEDED on the 'draw' verdict by PREFILL250_THERMAL_TRANSIENT_20260916.html."),
    ("docs/PREFILL250_THERMAL_TRANSIENT_20260916.html", "conclusion", "engine", False, True,
     "250 is the COLD-state operating point and 155-184 is the sustained one. Byte-identical "
     "counters between the two states, GPU busy/wait = 100-101%, GPU busy time ratio 1.44x vs "
     "throughput ratio 1.42x: the blocker is GPU execution rate on a fanless Mac16,12. Decode is "
     "unaffected (6.4-7.2 t/s in both states). CORRECTED 2026-09-16: (a) the session table's "
     "'idle before launch' column had no instrument behind it -- it is now DERIVED from log UTC "
     "timestamps + summary.json mtimes (67 / 247 / 848 / 923 s -> 179.09 / 286.57 / 293.11 / "
     "299.35 t/s, monotone; tau bracketed to 67 s < tau <= 247 s); (b) the report no longer claims "
     "CGC-DECPROF does not exist -- it does (ggml-backend.cpp:2064) and has emitted in 95 preserved "
     "logs, but three gates make it decode-only, documented in section 6.1 "
     "(FIXED 2026-09-16: the gate now admits prefill via `|| dp_step == 1 || dp_ntok > 1` and every "
     "line states its own ntok, so a line can no longer be MISREAD as prefill for being first). "
     "Adds section 3.1: a SECOND independent instrument (ioreg -c IOAccelerator Device Utilization "
     "%, 91-97% mean, 99% peak in all three runs) agrees that the GPU is saturated throughout "
     "prefill. Adds section 3.2 (2026-09-16): with the gate open, the per-layer slow/fast ratio is "
     "mean 2.32x CV 6.5% against a throughput ratio of 2.33x -- the graph slows down UNIFORMLY, "
     "which rules out a layer-localised implementation cost and leaves a clock/power ceiling as "
     "the only surviving explanation on the list. Quote the ratio agreement, not any single "
     "layer's milliseconds."),
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
    # every daily memory log, so a new day is not invisible until somebody remembers to add a row.
    # These are `log` rows like their curated siblings: the daily journal is the densest artifact in
    # the loop, it is append-only, and it must never be copied into the harness (rule D6).
    try:
        import glob
        for p in sorted(glob.glob(os.path.join(REPO, ".workbuddy/memory/*.md"))):
            rel = os.path.relpath(p, REPO)
            if any(r["path"] == rel for r in rows):
                continue
            rows.append(row_for(rel, "log", "shared", False, True,
                                "(auto-indexed daily memory log -- read it section-by-section through "
                                "engine_loop/memory/INDEX.jsonl rather than whole. The dated snapshot "
                                "under agent_harness/memory/ exists only to cross machines and is not "
                                "the thing to read; see the D6 amendment dated 2026-09-16)"))
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
