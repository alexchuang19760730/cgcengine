#!/usr/bin/env python3
"""Production prefill/decode matrix, measured the way llama-bench measures -- pp/tg x depth.

WHY THIS EXISTS
---------------
Every decode number this project has quoted so far came from a bespoke harness
(`decode_bench.py` / `decode_sweep.py`) driving our own HTTP server with its own prompt and its own
timing extraction. That is fine for A/B within the project, but it is not comparable to anything
outside it, and it let a real regression hide: `prod25` measured 7.97 t/s on 09-15 while the 09-04
log showed the same machine at 27.71 t/s, and neither number could be checked against a standard.

`llama-bench` is the standard. Its shape is pp(n_prompt) x tg(n_gen) x depth(-d/--n-depth), where
`-d` is "how many tokens are already in the context when the measurement starts" -- exactly the
"different long-context" axis that a streaming-expert engine must be judged on, because the whole
design is about what happens to the working set as the context grows.

WHY IT USED TO OOM
------------------
`llama-bench` accepts `-expert-cache <bytes>` (verified in `--help`), and without it the model loads
with all 256 experts of all 40 MoE layers resident: 13.66 GiB against a Metal
recommendedMaxWorkingSetSize of 11.45 GiB on this M4/16GB -> `test_prompt: failed to decode prompt
batch, res = -3` (GPU OOM) on the first batch. Passing `-expert-cache 8589934592` makes the loader
adopt the 143-slot/layer L4 pool and the same smoke test passes.

WHY IT THEN ASSERTED
--------------------
With the pool active and `CGC_PREFILL_STREAM` unset, `llama-context.cpp:285` clamps
`cparams.n_batch` to `cgc_pool_max_tokens()` (8) because a wider batch would read the shrunk
per-layer expert tensors out of bounds. `-p 128` then aborts:
`GGML_ASSERT(n_tokens_all <= cparams.n_batch)`. llama-bench cannot set `n_batch` from outside, so
the harness must pass `-b 8 -ub 8` -- which is not a workaround, it is what the server itself does
for every prefill chunk in that configuration. The other arm is `CGC_PREFILL_STREAM=1`
(+ `CGC_GATHER_SLAB_CAP=256`), which lifts the clamp and routes wide prefill chunks through the
whole-layer slab path; that arm gets realistic `-b/-ub`.

ONE SOURCE OF TRUTH
-------------------
The env does NOT come from a list written here. It is whatever `run_server.sh` resolves for the
requested profile: we call it with `CGC_DUMP_ENV=1`, which prints the fully-resolved SERVER_ENV and
argv right before the exec (see the block above the launch line). The 25.17 -> 5 t/s regression was
configuration drift (CGC_OA_ASYNC 1->0, CGC_SPAC on->off, CGC_MM_BITIDENT dropped, two
bit-identical pillars 1->0), so a second hand-maintained copy of the env is exactly the failure mode
to avoid.

USAGE
-----
    # the standard matrix on the production profile, one process per arm
    python3 scripts/check/llama_bench_matrix.py --arms prod25-stream --depths 0,512,1024,2048,4096

    # see the exact command without running it
    python3 scripts/check/llama_bench_matrix.py --arms prod25 --dry-run
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN_SERVER = ROOT / "scripts" / "run_server.sh"
LLAMA_BENCH = ROOT / "src" / "llama.cpp" / "build" / "bin" / "llama-bench"

# The thermal reading is a single implementation shared with the HTTP harnesses
# (`decode_sweep.py` / `decode_bench.py` import it from this same directory). Growing a second
# parser here is how the two instruments would drift apart on the one axis that has to agree
# before their numbers can be put side by side at all (eng-mh-0038).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import thermal_pressure as thermal  # noqa: E402

# --- profile / arm definitions -------------------------------------------------------------
# Each arm = (profile, extra env). The extra env is merged into the run_server.sh invocation so the
# dump we receive is the resolved truth for *that* arm, not a base profile plus our own arithmetic.
ARMS: dict[str, tuple[str, dict[str, str]]] = {
    # [CGC 2026-09-23] prod-new: MTP off decode (13-14 t/s, §EN-473) + prefill250 支柱
    # (5632 chunk + slab streaming, prefill 250+). profile 的 if 段已設 stream/slab/budget，
    # 所以不需要 extra_env。這是 commit_bench.py 的預設 profile。
    "prod-new": ("prod-new", {}),
    # The §8 production profile verbatim. Note it does NOT set CGC_PREFILL_STREAM, so its prefill
    # runs in n_batch=8 chunks -- that is a property of the profile, and this harness is what makes
    # it visible.
    "prod25": ("prod25", {}),
    # prod25 + the M2 prefill path. This is what a profile that can actually serve a 4k prompt looks
    # like: decode/MTP verify stay on the pool path (n_tokens <= 8), wide prefill chunks take the
    # whole-layer slab path.
    "prod25-stream": ("prod25", {"CGC_PREFILL_STREAM": "1", "CGC_GATHER_SLAB_CAP": "256"}),
    # Same, with the background double-buffer fill thread off. llama-bench creates and frees one
    # context per (p,n,d) instance, and the stream arm SIGSEGVs in `~llama_context` -- which stops
    # the DB thread only after an initial synchronize(). This arm separates "the DB thread races
    # teardown" from "freeing the slab buffers themselves is unsafe".
    "prod25-stream-nodb": ("prod25", {"CGC_PREFILL_STREAM": "1", "CGC_GATHER_SLAB_CAP": "256",
                                      "CGC_M2_DB_DISABLE": "1"}),
    # Pool-only with the same shape, as the control for the multi-context crash: it runs two contexts
    # (pp + tg) without any slab and does not crash.
    "prod25-nopool-clamp": ("prod25", {"CGC_POOL_MAX_TOKENS": "8"}),
    # After the ~llama_context UNREPOINT fix the second context no longer SIGSEGVs, but aborts in
    # `ensure_slot layer=1: no usable slot and no fill in flight`. The first thing the second context
    # does that touches the cache is `llama_expert_cache_prewarm_hot` (llama-context.cpp:1838, gated on
    # `ubatch.n_tokens <= cgc_pool_max_tokens()`), so this arm removes that one caller. If the abort
    # disappears the stale-state carrier is prewarm; if it persists, the carrier is the decode path.
    "prod25-stream-noprewarm": ("prod25", {"CGC_PREFILL_STREAM": "1", "CGC_GATHER_SLAB_CAP": "256",
                                           "CGC_NO_PREWARM": "1"}),
    # `zero_slot_enabled()` (llama-expert-cache.cpp:169) is true whenever CGC_VERIFY_DECODE or
    # CGC_DRAFT_DECODE is set, which MTP=1 does. That switches `usable_slots` from n to n-1, so this
    # arm separates "the ZERO-slot reservation starves layer 1" from "the slab path leaks cache state".
    "prod25-stream-mtpoff": ("prod25", {"CGC_PREFILL_STREAM": "1", "CGC_GATHER_SLAB_CAP": "256",
                                        "CGC_SERVER_MTP": "0"}),
    # [CGC 2026-09-18 §EN-145/§EN-146] `prod25-stream-mtpoff` above sets ONLY CGC_SERVER_MTP=0, and
    # run_server.sh:153-161 turns that into MODEL_DEFAULT=$Q36 -- a DIFFERENT checkpoint with no
    # nextn head (verified with CGC_DUMP_ENV=1). So it is a cross-checkpoint probe, not an MTP pair.
    # This arm pins MODEL to the production Nail checkpoint, so the ONLY difference from
    # `prod25-stream` is the MTP flag -- which is what makes an MTP off/on comparison meaningful.
    "prod25-nail-mtpoff": ("prod25", {"CGC_PREFILL_STREAM": "1", "CGC_GATHER_SLAB_CAP": "256",
                                      "CGC_SERVER_MTP": "0",
                                      "CGC_SERVER_MODEL": "models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf"}),
    # The instrumented FATAL showed the failing layer with owned=142/142 usable slots, loading=0,
    # queued=0, pinned=0 -- i.e. an eviction that should have succeeded but returned -1. The only
    # code path in pick_slot that rejects EVERY owned slot on all three passes is the SpAc victim
    # branch (`util < best_util || util == best_util`), which is false for every comparison when
    # `spac_util` holds a NaN. This arm sets CGC_SPAC=0 to confirm or kill that hypothesis.
    "prod25-stream-nospac": ("prod25", {"CGC_PREFILL_STREAM": "1", "CGC_GATHER_SLAB_CAP": "256",
                                        "CGC_SPAC": "0"}),
    # The 2026-09-15 GA profile as it stands (OA_ASYNC=1, SPAC off, batch 6144/6144, M2 prefill
    # streaming): the control that the 4.93-7.97 t/s decode numbers were measured under.
    #
    # This note used to read `OA_ASYNC=0`. That value was never executed: the knob's gate tested
    # presence, not value, and run_server.sh always exports the variable, so every profile ran the
    # SEGMENTED dispatcher whatever the 0 said. The 2026-09-15 value-aware fix turned the 0 into a
    # real choice for the first time (non-segmented measured 10x slower on the same build), the
    # launcher default was restored to 1 to preserve effective behaviour, and prefill250 now pins 1.
    # See m123_oracle_gate.DEFAULT_REF (v3) and dec-20260915-2215. It still aborts with GPU OOM on
    # wide prompts unless CGC_PREFILL_STREAM is set -- that is the profile's own shape, and why the
    # `prod25-stream` arm below is the one that yields usable pp/tg rows.
    "prefill250": ("prefill250", {}),
    # `CGC-DECPROF` was decode-only *by construction*: its print gate was `(dp_step % 8) == 0`, and a
    # 2048-token prefill at `-ub 6144` is a single graph_compute (dp_step=1) whose per-layer
    # accumulators are zeroed before step 8 is reached -- so no prefill ever printed and the
    # instrument looked like it did not exist (eng-src-0011). The gate now also fires on the first
    # graph and on any graph whose top-k tensor says n_tokens > 1, and each line carries `ntok=`, so
    # a printed step states its own shape rather than being *assumed* to be a prefill. `CGC_GPU_TIMING`
    # rides along so one launch yields both the per-layer split and the union/wait cross-check, and
    # `_ALL` is on because the uniformity verdict needs all 40 layers, not the top 8.
    "prefill250-decprof": ("prefill250", {"CGC_DECODE_PROFILE": "1", "CGC_DECODE_PROFILE_ALL": "1",
                                          "CGC_GPU_TIMING": "1"}),

    # ---------------------------------------------------------------------------------------------
    # [2026-09-17] The phase-split A/B (M1 work items 2/3). Three arms, all on `prefill250` so that
    # pool / ctx / batch / MTP / template are identical and ONLY the phase decision can differ.
    #
    # Why the same profile for all three: `prefill250` is the only profile that arms the slab, so
    # "slab on" and "slab off" have to be expressed as an override of that same profile. Using two
    # different profiles would change a dozen things at once.
    #
    # Why this is a PREFILL experiment and cannot be run on decode: with the pool's routable
    # geometry here, `decode_width = min(floor(142/8), T_prefill-1) = 17`, and decode is T=1 (MTP
    # verify 2-4), so every decode step takes the DECODE graph in BOTH arms -- the whole-layer slab
    # cannot participate in decode at all. An armed-vs-unarmed comparison on decode is zero by
    # construction, and reporting that zero as "the slab has no benefit" would be a false negative
    # manufactured by the measurement shape.
    #
    # Why three arms and not two: for a chunk wider than the decode width, the unarmed arm ALSO
    # takes the n_batch clamp (the launcher prints `[arm] slab OFF ... with the n_batch clamp on`),
    # so "armed vs unarmed" differs in two things: slab-vs-pool AND one wide graph vs many narrow
    # ones. Arms B and C differ only in the clamp width (8 vs 17), which is the graph-count axis;
    # B/C at a fixed prompt is therefore the "is the gap just graph-evaluation count" control.
    #   A phase-slab    : slab armed, clamp lifted          -> 1 x width-2048 PREFILL graph
    #   B phase-pool8   : pool path, clamp = cap 8          -> ~256 x width-8 pool graphs
    #   C phase-pool17  : pool path, clamp = bound 17       -> ~121 x width-17 pool graphs
    # With `--prompt 17` arm C is a single pool-path graph, i.e. the pool baseline at MATCHED graph
    # count, which is what separates "the slab is faster" from "the clamp made it slower".
    #
    # !! THESE THREE ARMS CANNOT MEASURE A WIDE PROMPT. `llama-bench` does not chunk, so an arm whose
    # n_batch is clamped asserts on a wide `-p`:
    #
    #     llama-context.cpp:2455: GGML_ASSERT(n_tokens_all <= cparams.n_batch) failed
    #
    # (measured 2026-09-17: `--prompt 2048` passed on phase-slab -- the one arm where the clamp is
    # lifted -- and aborted on phase-pool8/phase-pool17). Only the slab arm is measurable at a wide
    # prompt through llama-bench. The full grid needs the SERVER path, which chunks internally the
    # way production does: scripts/check/phase_split_ab.py does exactly that and is the instrument
    # for this question. These arms remain useful for `--prompt 17` and for slab-side work.
    "phase-slab":   ("prefill250", {}),
    "phase-pool8":  ("prefill250", {"CGC_PREFILL_STREAM": "0"}),
    "phase-pool17": ("prefill250", {"CGC_PREFILL_STREAM": "0", "CGC_POOL_MAX_TOKENS": "64"}),
}

# Args we forward from the resolved server argv to llama-bench. Allowlist, not denylist: the server
# argv carries -c/-np/--jinja/--spec-* /sampler flags that llama-bench either does not know or must
# derive itself (llama-bench sets n_ctx = n_prompt + n_gen + n_depth). Forwarding one of those by
# accident would silently change what is being measured, so anything not named here is dropped.
FORWARD_VALUED = {
    "-m", "--model",
    "-ngl", "--n-gpu-layers",
    "--load-mode", "-lm",
    "-t", "--threads",
    "-expert-cache", "--expert-cache",
    "-ctk", "--cache-type-k",
    "-ctv", "--cache-type-v",
}
FORWARD_BARE = {"-nkvo", "--no-kv-offload", "-fa", "--flash-attn"}

CGCENV_RE = re.compile(r"^CGCENV\s+(\S+)\s+(.*)$")
ENV_RE = re.compile(r"^ENV\s+(.*)$")
ARG_RE = re.compile(r"^ARG\s+(.*)$")


def resolve(profile: str, extra_env: dict[str, str]) -> dict:
    """Ask run_server.sh for the resolved env + argv of `profile` under `extra_env`."""
    env = dict(os.environ)
    env["CGC_SERVER_PROFILE"] = profile
    env["CGC_DUMP_ENV"] = "1"
    env.update(extra_env)
    proc = subprocess.run([str(RUN_SERVER)], cwd=str(ROOT), env=env,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-4000:] + "\n" + proc.stderr[-4000:] + "\n")
        raise SystemExit(f"run_server.sh CGC_DUMP_ENV=1 failed for profile {profile} (rc={proc.returncode})")

    scalars: dict[str, str] = {}
    bench_env: dict[str, str] = {}
    argv: list[str] = []
    for line in proc.stdout.splitlines():
        if (m := CGCENV_RE.match(line)):
            scalars[m.group(1)] = m.group(2).strip()
        elif (m := ENV_RE.match(line)):
            kv = m.group(1).strip()
            if "=" in kv:
                k, v = kv.split("=", 1)
                bench_env[k] = v
        elif (m := ARG_RE.match(line)):
            argv.append(m.group(1).strip())
    if not scalars or not argv:
        raise SystemExit("run_server.sh printed no CGCENV/ARG lines -- is the CGC_DUMP_ENV block present?")
    return {"scalars": scalars, "env": bench_env, "server_argv": argv}


def forward_argv(server_argv: list[str]) -> list[str]:
    """Keep only the llama-bench-compatible subset of the resolved server argv."""
    out: list[str] = []
    i = 0
    while i < len(server_argv):
        a = server_argv[i]
        if a in FORWARD_VALUED:
            if i + 1 >= len(server_argv):
                raise SystemExit(f"server argv ends on a valued flag: {a}")
            out += [a, server_argv[i + 1]]
            i += 2
            continue
        if a in FORWARD_BARE:
            out.append(a)
        i += 1
    return out


def default_batch(env: dict[str, str], scalars: dict[str, str]) -> tuple[str, str, str]:
    """Pick -b/-ub and say why, since the two arms need opposite answers."""
    b, ub = scalars.get("BATCH", "-"), scalars.get("UBATCH", "-")
    if b not in ("-", "") and ub not in ("-", ""):
        return b, ub, "profile"
    if env.get("CGC_PREFILL_STREAM", "0") not in ("0", ""):
        # clamp lifted: wide chunks go through the whole-layer slab path
        return "512", "512", "prefill-stream (clamp lifted)"
    # pool-only: llama-context.cpp:285 caps n_batch at cgc_pool_max_tokens() (default 8), so a wider
    # -p trips GGML_ASSERT in llama_context::decode. 8 is what the server itself uses per chunk.
    return "8", "8", "pool-path clamp (cgc_pool_max_tokens)"


def parse_rows(stdout: str) -> list[dict]:
    """Recover every COMPLETE top-level object from llama-bench's JSON array.

    llama-bench streams the array out as it finishes each instance. On a mid-run failure the
    closing `]` never arrives, so `json.loads` on the whole text raises and the instances that DID
    complete get thrown away -- which is exactly the case where the partial data is the result.
    Measured (`--arms prefill250 --prompt 512,4096`): p512 reports 117.54 t/s and p4096 then dies
    with `test_prompt: failed to decode prompt batch, res = -3` (GPU OOM at the profile's
    6144-token ubatch); the whole arm was discarded before this function existed.
    """
    rows: list[dict] = []
    depth = start = 0
    start = -1
    instr = esc = False
    for i, ch in enumerate(stdout):
        if instr:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                instr = False
            continue
        if ch == '"':
            instr = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    obj = json.loads(stdout[start:i + 1])
                except json.JSONDecodeError:
                    pass
                else:
                    if "avg_ts" in obj:
                        rows.append(obj)
                start = -1
    return rows


def harvest_bench_stats(stderr_text: str) -> dict:
    """Reuse decode_sweep's regexes so both harnesses report the engine's own counters."""
    spec = importlib.util.spec_from_file_location(
        "decode_sweep", str(Path(__file__).resolve().parent / "decode_sweep.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tmp = Path("/tmp/_lb_harvest.log")
    tmp.write_text(stderr_text, errors="replace")
    return mod.harvest(str(tmp))


def run_arm(tag: str, profile: str, extra_env: dict[str, str], args) -> dict:
    res = resolve(profile, extra_env)
    env, argv, scalars = res["env"], res["server_argv"], res["scalars"]

    fwd = forward_argv(argv)
    b, ub, why = ((args.batch, args.ubatch, "cli") if args.batch and args.ubatch
                  else default_batch(env, scalars))

    cmd = [str(LLAMA_BENCH)] + fwd + [
        "-b", str(b), "-ub", str(ub),
        "-p", str(args.prompt), "-n", str(args.gen), "-d", str(args.depths),
        "-r", str(args.reps), "-o", "json",
    ]
    if args.no_warmup:
        cmd.append("--no-warmup")
    # Appended rather than spliced into `fwd`: `fwd` is the SERVER's argv, and this is a property of
    # the CELL. Keeping them apart is what makes `--dry-run` able to show which is which.
    # [CGC 2026-09-19] Real-text prompt fill. Passed through only when asked: without it
    # llama-bench keeps its historical std::rand()%n_vocab fill, which is what every earlier
    # row in the record was measured with.
    if getattr(args, "prompt_file", None):
        cmd += ["--prompt-file", args.prompt_file]
    # [CGC 2026-09-19] Reseed the random fill at the top of every rep. Off by default: leaving it
    # off reproduces every earlier row bit-for-bit.
    if getattr(args, "fixed_fill_seed", 0):
        cmd += ["--fixed-fill-seed", str(args.fixed_fill_seed)]
    # [CGC 2026-09-19] --warm-skip N: run N generated tokens in EVERY rep before the clock starts,
    # and report them out of n_gen. Without it the tg figure is a whole-generation average whose
    # value depends on `--gen`: at -d 512, -n 128 reads 7.96 t/s and -n 512 reads 10.22. Passed
    # through only when set, so every existing arm keeps a bit-identical command line.
    if getattr(args, "warm_skip", 0):
        cmd += ["--warm-skip", str(args.warm_skip)]
    # [CGC 2026-09-19] --ctx-size N: llama-bench derives n_ctx = n_prompt + n_gen + n_depth, which
    # for the house decode arm is ~704 while prod25 serves at -c 4096. n_ctx sizes the KV allocation
    # and drives the engine's batch clamping, so this removes a confound every bench-vs-HTTP
    # comparison to date has carried. Passed only when set.
    if getattr(args, "ctx_size", 0):
        cmd += ["--ctx-size", str(args.ctx_size)]
    if args.spec_type:
        cmd += ["--spec-type", args.spec_type]
        if args.spec_draft_n_max is not None:
            cmd += ["--spec-draft-n-max", str(args.spec_draft_n_max)]

    print(f"\n=== arm {tag} (profile {profile}) ===", flush=True)
    print(f"  batch    : -b {b} -ub {ub}   [{why}]", flush=True)
    print(f"  ctx      : {scalars.get('CTX')} (server)   pool budget: {scalars.get('BUDGET')}", flush=True)
    print(f"  io knobs : OA_ASYNC={env.get('CGC_OA_ASYNC', '-')} SPAC={env.get('CGC_SPAC', '-')} "
          f"PREFILL_STREAM={env.get('CGC_PREFILL_STREAM', '-')} "
          f"MM_BITIDENT={env.get('CGC_MM_BITIDENT', '-')} "
          f"NO_WARMUP={env.get('CGC_MTP_NO_WARMUP', '-')} "
          f"NO_SEQ_RM_PROBE={env.get('CGC_NO_SEQ_RM_PROBE', '-')}", flush=True)
    if extra_env:
        print(f"  arm env  : {extra_env}", flush=True)
    print(f"  cmd      : {' '.join(cmd)}", flush=True)
    if args.dry_run:
        return {"tag": tag, "profile": profile, "extra_env": extra_env, "cmd": cmd,
                "env": env, "scalars": scalars, "dry_run": True}

    run_env = dict(os.environ)
    run_env.update(env)
    # A llama-bench arm is ONE child process holding the GPU for minutes with no per-request
    # boundary to hang a reading on, so the series has to come from a thread. The launch
    # reading is taken before the spawn (thermal_pressure.Sampler.start), which is the reading
    # the prefill separation is defined on -- taking it after would silently redefine it.
    sampler = thermal.Sampler()
    t0 = time.time()
    with sampler:
        proc = subprocess.run(cmd, cwd=str(ROOT), env=run_env, capture_output=True, text=True)
    wall = time.time() - t0
    # The filename has to carry the SHAPE. `tag` alone collides for every arm that sets its env
    # through the `PROFILE:ENV=...` form, because there the spec string *is* the tag -- so
    # `-n 24` and `-n 128` wrote to one path and the first run's stderr (the only place the
    # expert cache's own counters land) was silently overwritten. Hit on 2026-09-16 while running
    # Backup/run_instrument_compare.sh. Distinct `--json` paths hid it: the JSON evidence of both
    # runs survived while only the last stderr did.
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", tag)[:80]
    shape = re.sub(r"[^A-Za-z0-9._-]", "-", f"p{args.prompt}_n{args.gen}_d{args.depths}_r{args.reps}")
    stem = Path(args.workdir) / f"llama_bench_{safe}_{shape}"
    stem.with_suffix(".stderr.log").write_text(proc.stderr, errors="replace")
    stem.with_suffix(".json").write_text(proc.stdout, errors="replace")

    # Tolerate a mid-arm failure. `llama-bench` runs every (p,n,d) as its own llama_context and
    # exits non-zero on the first one that dies (usually GPU OOM at a large ubatch), so a single
    # bad shape must not discard the shapes that already succeeded -- those are the measurement.
    rows = parse_rows(proc.stdout)
    incomplete = proc.returncode != 0
    err = None
    if incomplete:
        # The LAST stderr line is the cache's teardown stats, not the failure -- pick the first
        # line that actually names a failure, so the recorded reason is the real one.
        sigs = ("failed to decode", "res = -", "error:", "GGML_ASSERT", "abort", "SIGSEGV",
                "SIGABRT", "out of memory", "Unable to")
        lines = [l.strip() for l in proc.stderr.splitlines() if l.strip()]
        err = next((l for l in lines if any(s in l for s in sigs)), lines[-1] if lines else "")
        print(f"  !! arm exited rc={proc.returncode}: {err}", flush=True)
        print(f"     recovered {len(rows)} completed instance(s) from the streamed JSON", flush=True)
        if not rows:
            print(proc.stdout[-1500:], flush=True)
            raise SystemExit(f"llama-bench failed for arm {tag} (rc={proc.returncode}) with no "
                             f"completed instance to report")

    stats = harvest_bench_stats(proc.stderr)
    out = {"tag": tag, "profile": profile, "extra_env": extra_env, "batch": b, "ubatch": ub,
           "batch_why": why, "wall_s": round(wall, 1), "env": env, "scalars": scalars,
           "rows": rows, "cache": stats, "incomplete": incomplete, "error": err,
           "thermal": sampler.result,
           # Recorded, not inferred: whether a row came from the speculative gen path is a property
           # of the invocation, and a spec row wears the same `tg` label as a plain one.
           "spec_type": args.spec_type or None,
           "warm_skip": getattr(args, "warm_skip", 0) or None,
           "ctx_size": getattr(args, "ctx_size", 0) or None,
           "spec_draft_n_max": args.spec_draft_n_max}
    for r in rows:
        shape = ("pp" if r["n_prompt"] > 0 else "tg")
        # llama-bench emits `test_time` as an ISO-8601 STRING, not a duration -- formatting it
        # with `.1f` raised "Unknown format code 'f' for object of type 'str'" and the traceback
        # aborted the whole arm AFTER llama-bench had already succeeded, which made a passing
        # run look like a failed one. Print it verbatim.
        print(f"  {shape:2s} p={r['n_prompt']:<5d} n={r['n_gen']:<4d} d={r['n_depth']:<5d} "
              f"-> {r['avg_ts']:8.2f} ± {r['stddev_ts']:.2f} t/s   (b={r.get('n_batch')} {r.get('test_time')})",
              flush=True)
    if stats.get("hit_rate_pct") is not None:
        print(f"  cache: hit {stats.get('hit_rate_pct')}%  misses {stats.get('misses')}  "
              f"reads {stats.get('file_reads')}  us/job {stats.get('io_us_per_job')}  "
              f"cap% {stats.get('capacity_pct')}", flush=True)
    th = sampler.result
    launch = th.get("launch") or {}
    print(f"  thermal: launch {launch.get('label')}@{launch.get('t')}  "
          f"worst {th.get('worst', {}).get('label')}  hist {th.get('hist')}  "
          f"n={th.get('n')} @{th.get('interval_s')}s", flush=True)
    return out


def report(results: list[dict], args) -> None:
    print(f"\n{'='*96}\n  llama-bench matrix -- pp/tg x depth  (production metric)\n{'='*96}")
    hdr = (f"{'arm':16s} {'shape':>5s} {'depth':>6s} {'t/s':>9s} {'±':>6s} {'n_batch':>8s} "
           f"{'hit%':>6s} {'thermal':>10s}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        if r.get("dry_run"):
            continue
        mark = "  [INCOMPLETE]" if r.get("incomplete") else ""
        launch = (r.get("thermal") or {}).get("launch") or {}
        for row in r["rows"]:
            shape = "pp" if row["n_prompt"] > 0 else "tg"
            c = r["cache"]
            print(f"{r['tag']:16s} {shape:>5s} {row['n_depth']:>6d} {row['avg_ts']:>9.2f} "
                  f"{row['stddev_ts']:>6.2f} {str(row['n_batch']):>8s} "
                  f"{('%.1f' % c['hit_rate_pct']) if c.get('hit_rate_pct') is not None else '-':>6s} "
                  f"{launch.get('label', '-'):>10s}"
                  f"{mark}")
        if r.get("incomplete"):
            print(f"  (arm stopped early: {r.get('error')}) -- rows above are the instances that "
                  f"completed before it died")

    if args.md:
        lines = ["# llama-bench pp/tg x depth — production metric", "",
                 f"machine measured by `llama-bench`; arms resolved through `run_server.sh CGC_DUMP_ENV=1`.",
                 "", "| arm | shape | prompt | gen | depth | t/s | ± | n_batch | hit% | reads | us/job | thermal |",
                 "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in results:
            if r.get("dry_run"):
                continue
            c = r["cache"]
            launch = (r.get("thermal") or {}).get("launch") or {}
            for row in r["rows"]:
                shape = "pp" if row["n_prompt"] > 0 else "tg"
                lines.append(
                    f"| {r['tag']} | {shape} | {row['n_prompt']} | {row['n_gen']} | {row['n_depth']} | "
                    f"{row['avg_ts']:.2f} | {row['stddev_ts']:.2f} | {row['n_batch']} | "
                    f"{c.get('hit_rate_pct', '-')} | {c.get('file_reads', '-')} | "
                    f"{c.get('io_us_per_job', '-')} | {launch.get('label', '-')} |")
            if r.get("incomplete"):
                lines += ["", f"> **{r['tag']}: arm stopped early** — {r.get('error')}. "
                              f"The rows above are the instances that completed before it died; "
                              f"the remaining shapes were never measured."]
        Path(args.md).write_text("\n".join(lines) + "\n")
        print(f"\nmd -> {args.md}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="prod25-stream",
                    help=f"comma list from {sorted(ARMS)} (or PROFILE:ENV=VAL;ENV=VAL)")
    ap.add_argument("--depths", default="0,512,1024,2048,4096",
                    help="llama-bench -d list: context already filled before the measurement")
    ap.add_argument("--prompt", default="512", help="llama-bench -p (pp); comma list allowed")
    ap.add_argument("--gen", default="128", help="llama-bench -n (tg)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--batch", help="override -b/-ub for both (default: derived from the arm)")
    ap.add_argument("--ubatch")
    ap.add_argument("--no-warmup", action="store_true",
                    help="pass --no-warmup; note warmup is what leaves the expert pool hot, so the "
                         "default (warmup ON) is the closer analogue of a served request")
    # [CGC 2026-09-18] MTP/speculative mode. A CLI flag and NOT an env var, because llama-bench
    # resolves CLI FIRST and prints "LLAMA_BENCH_SPEC is deprecated" for the env form
    # (llama-bench.cpp:1145-1153) -- an env-driven cell would therefore be running a path the
    # instrument itself asks people to stop using, and the deprecation warning would be the only
    # trace of it. Also NOT forwarded from the server argv: forwarding it by default would turn
    # every existing cell into a spec cell on all seven profiles (all have mtp=1), i.e. it would
    # silently redefine the recorded `decode` number. So it is opt-in, per cell.
    ap.add_argument("--spec-type", default="",
                    help="llama-bench --spec-type (only 'draft-mtp' is implemented). Empty = the "
                         "plain gen cell, i.e. the previous behaviour of every existing cell.")
    ap.add_argument("--prompt-file", default="",
                    help="llama-bench --prompt-file: fill prompt/depth with REAL text read from this "
                         "file (cycled) instead of std::rand()%%n_vocab. Empty = the historical random "
                         "fill, i.e. comparable with every earlier row.")
    ap.add_argument("--ctx-size", type=int, default=0,
                    help="override llama-bench's derived n_ctx = n_prompt + n_gen + n_depth "
                         "(llama-bench -c/--ctx-size). 0 = historical derived value")
    ap.add_argument("--warm-skip", type=int, default=0,
                    help="run N generated tokens per rep before the clock starts; reported n_gen "
                         "excludes them (llama-bench --warm-skip). 0 = historical behaviour")
    ap.add_argument("--fixed-fill-seed", type=int, default=0,
                    help="llama-bench --fixed-fill-seed: reseed std::rand() at the top of EVERY rep, "
                         "so all reps see the same fill stream and the expert cache can reach steady "
                         "state. 0 (default) = the historical advancing-stream behaviour.")
    ap.add_argument("--spec-draft-n-max", type=int, default=None,
                    help="llama-bench --spec-draft-n-max (1..16). Inert without --spec-type; when "
                         "omitted, llama-bench's own default (3) applies.")
    ap.add_argument("--workdir", default="/tmp")
    ap.add_argument("--json")
    ap.add_argument("--md")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.batch and not args.ubatch:
        args.ubatch = args.batch

    results = []
    for spec in args.arms.split(","):
        spec = spec.strip()
        if not spec:
            continue
        if ":" in spec:
            prof, _, envs = spec.partition(":")
            extra = dict(kv.split("=", 1) for kv in envs.split(";") if "=" in kv)
            results.append(run_arm(spec, prof, extra, args))
        else:
            if spec not in ARMS:
                raise SystemExit(f"unknown arm {spec!r}; known: {sorted(ARMS)}")
            prof, extra = ARMS[spec]
            results.append(run_arm(spec, prof, extra, args))

    report(results, args)
    if args.json:
        Path(args.json).write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"json -> {args.json}")
    bad = [r["tag"] for r in results if r.get("incomplete")]
    if bad:
        print(f"\nINCOMPLETE arm(s): {' '.join(bad)} -- see the [INCOMPLETE] markers above; "
              f"the reported rows are complete, the arm just did not measure every shape.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
