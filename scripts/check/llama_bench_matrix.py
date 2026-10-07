#!/usr/bin/env python3
"""Production prefill/decode matrix, measured the way llama-bench measures -- pp/tg x depth.

⚠ INTERNAL DRIVER (2026-09-25 ruling): this is NOT an external entry point. The only production
entry points are `harness.py bench` (general measurement) and `commit_bench.py` (pre-commit).
Every command built here is checked against the test card's machine-readable CELL
(docs/PROD_NEW_TEST_CARD_*.md §2.5): a cell that does not match is refused (fail-closed). A number
obtained by running this file directly must NOT be quoted as "our prefill/decode" -- its provenance
is unverified unless it matches §2.5. Go through harness bench / commit_bench.

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
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN_SERVER = ROOT / "scripts" / "run_server.sh"
LLAMA_BENCH = ROOT / "src" / "llama.cpp" / "build" / "bin" / "llama-bench"
# [CGC 2026-10-01, MEASUREMENT_CONTRACT §3.4.1] 所有會起 llama-bench 的 runner 一律接這支閘門；
# shell 側是 `. "${REPO}/scripts/check/budget_gate.sh"`，Python 側是 budget_gate_preflight()。
BUDGET_GATE_SH = ROOT / "scripts" / "check" / "budget_gate.sh"

# The thermal reading is a single implementation shared with the HTTP harnesses
# (`decode_sweep.py` / `decode_bench.py` import it from this same directory). Growing a second
# parser here is how the two instruments would drift apart on the one axis that has to agree
# before their numbers can be put side by side at all (eng-mh-0038).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import thermal_pressure as thermal  # noqa: E402
import memory_pressure as mempress  # noqa: E402

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

# The sampling the SERVER will actually use lives in its ARGV, not in its env: run_server.sh emits
# `--temp ${CGC_SERVER_TEMP:-0.4} --top-p ${CGC_SERVER_TOP_P:-0.8} --top-k 0`, and the CGC_DUMP_ENV
# block echoes those as `ARG` lines with no matching `ENV` line (measured 2026-09-26).
# llama-bench's sampling-parity block reads the ENV names, and its own argv has no --temp flag, so
# without this translation the two carriers sample differently -- measured from the engine's own
# witness line: `[CGC seed] sampler_seed=.. temp=0.80 top_p=0.95 top_k=40`, i.e.
# common_params_sampling's struct DEFAULTS, while the server ran 0.4 / 0.8 / 0. Since the MTP accept
# step compares the draft against the target's sampled token, that difference is an accept-rate
# difference: it is most of why llama-bench and llama-server disagreed on mean_len (2.02 vs 2.50).
SAMPLING_FLAGS = {
    "--temp": "CGC_SERVER_TEMP", "--temperature": "CGC_SERVER_TEMP",
    "--top-p": "CGC_SERVER_TOP_P", "--top_p": "CGC_SERVER_TOP_P",
    "--top-k": "CGC_SERVER_TOP_K", "--top_k": "CGC_SERVER_TOP_K",
    "--min-p": "CGC_SERVER_MIN_P", "--min_p": "CGC_SERVER_MIN_P",
}


def sampling_env(server_argv: list[str]) -> dict[str, str]:
    """The server's sampling params, renamed into what llama-bench's parity block reads.

    LAST occurrence wins, deliberately: the resolved argv carries `--temp 0` in an earlier position
    and `--temp 0.4` later (the server's own parser takes the last), so a reader that took the first
    would pin temp 0 -- i.e. greedy -- while claiming parity with the server.
    """
    out: dict[str, str] = {}
    for i, a in enumerate(server_argv[:-1]):
        if a in SAMPLING_FLAGS:
            out[SAMPLING_FLAGS[a]] = server_argv[i + 1]
    return out

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


def parse_arm_env(envs: str) -> dict[str, str]:
    """`!A=1;B=2` -> `{'A': '1', 'B': '2'}`.

    The `!` is harness's "declared base override" marker (harness.py:_parse_arm strips it and
    records the key for the base gate). It MUST be stripped here too, and for a harder reason
    than cosmetics: run_server.sh forwards env through `env "${SERVER_ENV[@]}"`, an ALLOWLIST,
    so a key literally named `!A` matches nothing and is dropped in silence -- 2026-09-28 the
    whole MTP fix arm (`!LLAMA_EXPERT_CACHE_LAYER_CAPS=...`) ran on profile defaults while the
    gate reported PASS. Same shape as the allowlist trap at run_server.sh:2477.
    """
    out: dict[str, str] = {}
    for piece in envs.split(";"):
        piece = piece.strip()
        if not piece:
            continue
        if piece.startswith("!"):
            piece = piece[1:]
        if "=" not in piece:
            raise SystemExit(f"bad arm env piece {piece!r} (want KEY=VAL or !KEY=VAL)")
        k, v = piece.split("=", 1)
        out[k.strip()] = v.strip()
    return out


# These scalars are per-invocation noise (a log path with a timestamp, a port), not something an
# arm env can influence -- comparing them would make every surface look "changed".
_VOLATILE_SCALARS = frozenset({"LOG", "PORT"})


def _surface(res: dict) -> tuple:
    return (tuple(sorted(res["env"].items())),
            tuple(sorted((k, v) for k, v in res["scalars"].items() if k not in _VOLATILE_SCALARS)),
            tuple(res["server_argv"]))


def arm_env_dropped(profile: str, extra_env: dict[str, str], res_full: dict) -> list[str]:
    """Which `KEY=VAL` never reached the engine (fail-closed against the silent drop).

    Two ways a knob looks set but is not: (a) the `!` prefix above, (b) a name that is absent
    from run_server.sh's allowlist (e.g. `LLAMA_EXPERT_CACHE_LAYER_CAPS` -- the real knob is
    `CGC_SERVER_LAYER_CAPS`, which run_server.sh:2470 translates). Both are invisible in the
    resolved surface, so the only general test is: does asking for it change anything?

    A key counts as landed if it is echoed verbatim (same name + value) in the resolved env /
    scalars -- which also covers "set to the value it already had" -- or if resolving with that
    one key produces a surface different from the bare profile.

    [CGC 2026-10-07] The "set to the value it already had" half did not actually work, because the
    verbatim test compares NAMES and run_server.sh renames on the way through: asking for
    `CGC_SERVER_EXPERT_CACHE_BYTES=8589934592` lands as the argv token `-expert-cache
    8589934592` and as env `CGC_EXPERT_CACHE_BYTES` (run_server.sh:1342 / :1441). Neither carries
    the requested name, so the verbatim test missed; the surface-diff test missed too, because
    8 GiB IS the prod-new default and the two surfaces are then identical. The arm was reported
    DROPPED -- the guard built to catch silent drops was itself producing a false positive, and it
    did so on exactly the value the profile ships with. A control arm that states the current
    setting is the one arm a capacity sweep cannot do without.
    => The value now counts as landed if it appears ANYWHERE in the resolved surface under another
    name (env / scalars / argv tokens), which is what the docstring always claimed.
    """
    if not extra_env:
        return []
    dropped: list[str] = []
    base: dict | None = None
    for k, v in extra_env.items():
        if res_full["env"].get(k) == v or res_full["scalars"].get(k) == v:
            continue
        if _value_present(res_full, v):
            continue
        if base is None:
            base = resolve(profile, {})
        if _surface(resolve(profile, {k: v})) != _surface(base):
            continue
        dropped.append(f"{k}={v}")
    return dropped


def _value_present(res: dict, v: str) -> bool:
    """Does `v` appear anywhere in the resolved surface, under any name?

    The renamed-knob case: run_server.sh translates some entry names into different downstream
    names (`CGC_SERVER_EXPERT_CACHE_BYTES` -> the `-expert-cache` argv token and the
    `CGC_EXPERT_CACHE_BYTES` env var). A name-exact test cannot see that, so the value itself is
    the evidence. Deliberately scoped to env / scalars / argv tokens: a substring search over the
    whole resolved dict would match model paths and prompts, and would defeat the guard.
    """
    for section in ("env", "scalars"):
        if v in (res.get(section) or {}).values():
            return True
    argv = res.get("server_argv") or res.get("argv") or []
    return any(str(tok) == v for tok in argv)


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


def cell_declared_batch(cell_name: str | None) -> tuple[str, str]:
    """`(batch, ubatch)` 逐字取自測試卡 §2.5 宣告的格（具名格或預設格）；讀不到 ⇒ ("", "")。

    WHY THE CELL IS THE SOURCE OF `-b/-ub` (2026-10-03). `-b/-ub` 是**格的形狀維度**——
    `cell_contract.check_cell` 會逐字比對——而 profile 的 `BATCH`／`UBATCH` 是 **server** 在
    8192 ctx 下的批次（5632）。驅動若從 server 推導 bench 的批次，「宣稱的格」與「跑出的格」
    在**預設路徑**上就必然不一致（實測：`--arms prod-new` 裸跑 ⇒ `-b 5632` vs 權威 2048 ⇒ 拒跑）。
    形狀的唯一來源是卡片；server 的 batch 只影響 server。CLI 仍最高（顯式覆蓋要被記錄）。
    """
    try:
        import cell_contract  # type: ignore
        _label, cell = cell_contract.resolve_cell(cell_contract.load_contract(), cell_name)
    except Exception:  # noqa: BLE001  讀不到/退役 ⇒ 交給呼叫端的既有 fallback，判準方向不變
        return "", ""
    if not isinstance(cell, dict):
        return "", ""
    b, ub = cell.get("batch"), cell.get("ubatch")
    if b is None or ub is None:
        return "", ""
    return str(b), str(ub)


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

# 【CGC 2026-10-06】spec 臂的 draft 健康度：逐 rep 的 `drafted`（＝送進 verify 的 draft 顆數）。
#
# WHY：`p_chain`（2026-10-06 12:01）第 1–4 個 rep 是 `drafted=0 rounds=128 emit=1.0000` —— draft 一顆
# 都沒吐，那幾輪量的是**純解碼**，而它的 t/s（9.47／9.50）與健康的 tg 讀數長得一模一樣，差一點被
# 當成「MTP-on 純鏈」引用。判定要與 `instrument_report` 同一個位置（產物寫出來的那一刻），而且
# **逐 rep**：「一臂 5 輪裡死了 4 輪」不是雜訊，是這一臂沒有持續量到 spec 路徑。
#
# 為什麼是逐臂在 matrix 裡判而不是拿成對 log 事後掃：一次 `bench` 可以帶很多臂，牠們的 ACCEPT 行
# 全寫進同一份 log ⇒ 事後掃無法歸屬到臂。這裡手上正好握著**那一臂自己的** stderr。
_SPEC_MARK_RE = re.compile(r"llama-bench: \[spec\]")
_ACCEPT_RE = re.compile(
    r"CGC-BENCH-ACCEPT phase=(?P<phase>\w+) rounds=(?P<rounds>\d+) "
    r"drafted=(?P<drafted>\d+) acc_drafts=(?P<acc>\d+)")
_EMIT_RE = re.compile(r"emit_per_round=([0-9.]+|NA)")


def spec_draft_health(stderr_text: str) -> dict:
    """spec 臂逐 rep 的 draft 健康度 ⇒ 寫進產物（`spec_draft`）並在 stdout 留一行。

    * `NOT_SPEC` —— 不是 spec 路徑（沒有 spec 標記、也沒有 ACCEPT 行）⇒ 本閘與它無關。
    * `OK`       —— 有 spec、而且**每一輪** rep 的 `drafted` > 0。
    * `INVALID`  —— 有 spec 但至少一輪 rep `drafted == 0`（draft 死亡 ⇒ 那一輪量的是純解碼），
                    或 spec 開了卻**一行 ACCEPT 都沒有**（拿不出逐輪帳 ⇒ 不可引用）。

    `dead` 保留是哪幾輪與那些輪的 `rounds`／`emit`：理由必須可查，不能只留一句「無效」。
    """
    reps, dead = [], []
    for ln in (stderr_text or "").splitlines():
        m = _ACCEPT_RE.search(ln)
        if not m:
            continue
        em = _EMIT_RE.search(ln)
        rec = {"phase": m.group("phase"), "rounds": int(m.group("rounds")),
               "drafted": int(m.group("drafted")), "acc_drafts": int(m.group("acc")),
               "emit_per_round": em.group(1) if em else None}
        reps.append(rec)
        if rec["drafted"] == 0:
            dead.append(rec)
    if dead:
        which = "; ".join("rep%d(phase=%s rounds=%d emit=%s)"
                          % (i, r["phase"], r["rounds"], r["emit_per_round"])
                          for i, r in enumerate(reps, 1) if r["drafted"] == 0)
        return {"state": "INVALID", "ok": False, "n_reps": len(reps), "reps": reps, "dead": dead,
                "why": ("%d/%d 輪 rep 的 draft 死亡（drafted=0 ⇒ 那幾輪量的是純解碼，不是 spec）：%s"
                        % (len(dead), len(reps), which))}
    if not reps:
        if _SPEC_MARK_RE.search(stderr_text or ""):
            return {"state": "INVALID", "ok": False, "n_reps": 0, "reps": [], "dead": [],
                    "why": ("spec 已開（llama-bench 的 `[spec]` 標記在）但沒有任何 "
                            "CGC-BENCH-ACCEPT 行 ⇒ 拿不出逐輪帳 ⇒ 不可引用")}
        return {"state": "NOT_SPEC", "ok": True, "n_reps": 0, "reps": [], "dead": [],
                "why": "不是 spec 路徑（沒有 spec 標記、也沒有 ACCEPT 行）⇒ 本閘與它無關"}
    return {"state": "OK", "ok": True, "n_reps": len(reps), "reps": reps, "dead": [],
            "why": ("%d/%d 輪 rep 的 drafted>0（最小 drafted=%d）"
                    % (len(reps), len(reps), min(r["drafted"] for r in reps)))}


# [CGC 2026-10-06] 狀態詞彙與「死沒死」的判準放在**產物端唯一一份**，三個消費端都呼叫這裡：
#   * `harness bench` 拒收（rc=6）        —— scripts/check/harness.py  `_spec_draft_gate`
#   * `experiment_sync.sync_gate` 拒寫     —— code 6（寫進決策面那一步，不經 harness 也擋）
#   * `experiment_sync._arm_on_board`      —— 不得上榜
# 為什麼要收斂（2026-10-06 實測到的不一致）：三者原本各寫一份，harness 是 **fail-closed**
# （`state in ("OK","NOT_SPEC")`，其餘一律拒），而後兩者只擋**字面上的 "INVALID"** ⇒ 只要狀態
# 是 NO_STATE／打錯字／未來新增的值，sync 與排行榜就會**放行**一個 harness 會拒的臂。
# 判準的語意：`drafted=0` 的那一輪量的是純解碼（emit ≡ 1.0），它的 t/s 與健康的 tg 長得一樣。
SPEC_DRAFT_OK_STATES = ("OK", "NOT_SPEC")


def spec_draft_state(arm) -> tuple:
    """回傳 (state, present)。`present=False` ＝ 舊產物（閘門存在之前）⇒ 不回溯。"""
    sd = (arm or {}).get("spec_draft")
    if not isinstance(sd, dict) or not sd:
        return "NO_FIELD", False
    return str(sd.get("state") or "NO_STATE"), True


def spec_draft_dead(arm) -> bool:
    """True ⇒ 這一臂的 spec draft 死過（或狀態不明）⇒ 不可引用／不得上榜／不得 sync。

    **fail-closed**：狀態不是 `OK`／`NOT_SPEC` 就是死（未知狀態不可當健康，含 `NO_STATE`）。
    缺欄位（舊產物）⇒ False（不回溯 —— 缺席不等於有問題）。
    """
    state, present = spec_draft_state(arm)
    return present and state not in SPEC_DRAFT_OK_STATES


def harvest_bench_stats(stderr_text: str) -> dict:
    """Reuse decode_sweep's regexes so both harnesses report the engine's own counters."""
    spec = importlib.util.spec_from_file_location(
        "decode_sweep", str(Path(__file__).resolve().parent / "decode_sweep.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tmp = Path("/tmp/_lb_harvest.log")
    tmp.write_text(stderr_text, errors="replace")
    return mod.harvest(str(tmp))


# 【CGC 2026-09-29】這一輪每一臂的 stderr（tag, shape, text），供收尾時寫成產物的成對 log。
_PAIR_BUFFER: list = []
_IBIND = None


def ibind():
    """`instrument_binding` 模組（成對規則與量具判準的**唯一住處**）；載入一次。"""
    global _IBIND
    if _IBIND is None:
        p = Path(__file__).resolve().parent / "instrument_binding.py"
        spec = importlib.util.spec_from_file_location("instrument_binding", str(p))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _IBIND = mod
    return _IBIND


def instrument_report(serr: str, extra_env: dict, res_env: dict, require=None) -> dict:
    """[CGC 2026-09-29] 量具綁定：**設了開關** 不等於 **量具綁上了圖**。

    WHY：2026-09-29 交付 cell 的四場沒有產物的嘗試，留下的日誌裡 MM-PUB/RB-FEED/MISSMASK 全是 0——
    而 `{"verdict": "..."}` 那一套事後檢查（白名單、符號、建置產物、產物存在）**四項全過**。
    所以判定要在「產物寫出來的那一刻」就做，寫進 `out['instrument']`，並在 stdout 上留下可見的一行；
    後續 `experiment_sync` 把 `UNBOUND` 當成「這一輪的計數不可引用」。

    判準本身住在 `instrument_binding.py`（與檢查端同一份），這裡只負責把這一輪的 env 與 stderr 餵進去。
    """
    p = Path(__file__).resolve().parent / "instrument_binding.py"
    spec = importlib.util.spec_from_file_location("instrument_binding", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    env = dict(res_env or {})
    env.update(extra_env or {})
    return mod.binding_report(serr, env=env, require=require)


_CELL_FWD_ALIASES = {
    "ngl": ("-ngl", "--n-gpu-layers"),
    "load_mode": ("--load-mode", "-lm"),
    "threads": ("-t", "--threads"),
    "expert_cache_bytes": ("-expert-cache", "--expert-cache"),
    "cache_type_k": ("-ctk", "--cache-type-k"),
    "cache_type_v": ("-ctv", "--cache-type-v"),
}


def _fwd_val(fwd: list[str], aliases: tuple[str, ...]):
    for i, a in enumerate(fwd):
        if a in aliases and i + 1 < len(fwd):
            return fwd[i + 1]
    return None


def pool_budget_mb(fwd: list[str], env: dict, scalars: dict) -> tuple:
    """(mb, source)：這一支臂宣告的池子大小。

    三個來源，順序就是「誰最接近真正生效的那個」：CLI → env → server scalars。
    讀不出整數就回 `(None, 讀不出…)`——**不猜**：Metal 預算閘在拿不到池子大小時會回
    `armed=False`（沒有這根桿子），而一個猜出來的池子會讓它看起來有桿子。
    """
    raw = _fwd_val(fwd, _CELL_FWD_ALIASES["expert_cache_bytes"])
    src = "--expert-cache（CLI，宣告值）"
    if raw is None:
        raw = (env or {}).get("CGC_EXPERT_CACHE_BYTES")
        src = "CGC_EXPERT_CACHE_BYTES（env，宣告值）"
    if raw is None:
        raw = (scalars or {}).get("BUDGET")
        src = "server scalars BUDGET（profile 解出的值）"
    if raw is None:
        return None, "沒有宣告池子大小"
    try:
        return float(int(str(raw).strip())) / (1024.0 * 1024.0), src
    except (TypeError, ValueError):
        return None, "池子大小讀不出整數（%r）—— 不猜" % (raw,)


# --- spec-armed gate (fail-closed) ------------------------------------------------------------
# [CGC 2026-09-26] An arm that sets CGC_SERVER_MTP=1 while the cell carries no --spec-type is NOT an
# MTP arm: it exports the MTP=1 env block (NO_PREFETCH, LAYER_CAPS 40-40:256, PREFIX_REUSE_CKPT,
# NO_SEQ_RM_PROBE, WARM_NPAST) with ZERO draft/verify work, so it prices the env block and says
# nothing about speculation. Measured 2026-09-26: such an arm reads 10.45 t/s against the MTP-off
# baseline's 12.57, and its own artifact says `spec_type: null` -- yet the loss was quoted as "MTP
# on is slower" because nothing refused the pair. Same family as the allowlist traps above: a knob
# that was never armed cannot be read as "no effect".
def mtp_without_spec(extra_env: dict[str, str], res_env: dict[str, str], spec_type: str) -> str:
    """Evidence string when MTP is armed but the cell has no spec, else ''.

    Value test, not presence: run_server.sh gates its MTP block on `= "1"`, so "0" is an honest off
    arm (the ARMS table has two of them). Any other non-empty value is treated as ON -- fail-closed.

    [CGC 2026-10-03] One exception, and it is the case the guard was silently forbidding: when the
    ARM ITSELF carries the per-arm spec shim `LLAMA_BENCH_SPEC[=1]` (+ optionally
    `LLAMA_BENCH_SPEC_DRAFT_N_MAX`). llama-bench resolves those into `spec_type={"draft-mtp"}` and
    the draft depth from the PER-ARM environment (llama-bench.cpp:1350-1358, "CLI first, then the
    deprecated environment shim"), so the draft IS armed for that arm alone. That matters because
    `--spec-type` is a whole-run flag: with it, EVERY arm of the cell speculates, so a same-session
    MTP-off/MTP-on pair is impossible. Refusing the shim therefore makes the delivery-cell MTP gain
    permanently unmeasurable -- which is the state the record has been in. The premise in the note
    above ("a flag that nothing here adds for you") is simply false in this case, so the arm is
    allowed -- loudly, because its depth comes from its own env and must be read back from the
    artifact (SPECDBG histogram / ACCEPT), never assumed from the cell.
    """
    if spec_type:
        return ""
    for src, name in ((extra_env, "arm env"), (res_env, "resolved env")):
        v = (src or {}).get("CGC_SERVER_MTP")
        if v is not None and v.strip() != "0":
            shim = (src or {}).get("LLAMA_BENCH_SPEC")
            if shim is not None and shim.strip() not in ("", "0"):
                depth = (src or {}).get("LLAMA_BENCH_SPEC_DRAFT_N_MAX")
                print(f"!! note: {name} CGC_SERVER_MTP={v} with a PER-ARM spec shim "
                      f"(LLAMA_BENCH_SPEC={shim}" +
                      (f", LLAMA_BENCH_SPEC_DRAFT_N_MAX={depth}" if depth else "") +
                      ") -- allowed: the draft is armed for THIS arm only, which is the only way a "
                      "multi-arm cell can hold both an off and an on arm. Read the realized depth "
                      "back from the artifact (SPECDBG / ACCEPT), not from the cell.", file=sys.stderr)
                return ""
            return f"{name} CGC_SERVER_MTP={v}"
    return ""


def _metal_line(mg: dict) -> str:
    """一行讀得懂的 Metal 預算：每個項目帶出處在產物裡，這裡只要「哪一項」與「差多少」。"""
    if not mg:
        return ""
    items = " ＋ ".join("%s %.0f" % (i["label"], i["mb"]) for i in mg.get("items") or [])
    tot = mg.get("total_mb")
    ceil = mg.get("ceiling_mb")
    head = mg.get("headroom_mb")
    s = "  %s = %s MiB  vs 上限 %s MiB（%s）" % (
        items or "（無項目）", "%.0f" % tot if tot is not None else "?",
        "%.0f" % ceil if ceil is not None else "?", mg.get("ceiling_source") or "?")
    if head is not None:
        if head >= 0:
            s += "  餘裕 %.0f MiB" % head
        else:
            # 點名要點**人讀得懂的名字**（`wired_mb` 是機器名，不是項目名）。
            _big = mg.get("biggest")
            _lbl = next((i["label"] for i in mg.get("items") or [] if i["name"] == _big), _big)
            s += "  超額 %.0f MiB（最大項：%s）" % (-head, _lbl)
    return s


# --- 串流子行程（量測中途被殺也要留下軌跡）------------------------------------------------------
# 【CGC 2026-09-29】為什麼不是 `subprocess.run(capture_output=True)`：那條路把子行程的 stderr 收在
# **驅動自己的記憶體**裡，只有子行程結束才輪得到寫檔（`run_arm` 是跑完之後才 `write_text`）。於是
# 當**驅動**被殺（終端逾時、Ctrl-C、外面那把刀）時，磁碟上一行都沒有 —— 交付 cell 的 `gen 128` ＋
# `CGC_MISS_MASK_DBG` 那四場就是這樣消失的，而「停在哪一行」正好是那四場唯一的問題。stderr 在 C
# 裡是無緩衝的，所以只要驅動**即時**寫，被殺掉的中途也有完整軌跡。這一個 helper 因此是三件事：
#   ① tee：每讀到一行就 append ＋ flush 到 live log（不是快取，是落地）；
#   ② 停滯看門狗：連續 `stall_watch_s` 沒有新行 ⇒ 記 marker ＋ `sample` 子行程堆疊。
#      「卡住」與「很慢」只能靠堆疊分，靠等分不出來；
#   ③ 自帶剎車：`arm_timeout_s` 逾時只殺**自己的 process group**（`start_new_session=True`），
#      驅動活下來把產物與 log 寫完 —— 讀數不該取決於外面那把刀幾點落下。
# 預設全關（0）⇒ 既有 cell 的命令列與行為一個字都不變。
_LIVE_ECHO = (r"CGC-MISSMASK-STEP|CGC-MISSMASK-COST|CGC-RB-FEED|CGC-OOB|CGC-PREV-PF|CGC-OOM|"
              r"abort|Abort|GGML|rror|Segmentation|SIGSEGV|assert|watchdog")
SAMPLE_BIN = "/usr/bin/sample"
# 【CGC 2026-09-29】停滯判準不能是「有沒有新的一行」：引擎有一條 **心跳**（`CGC-RSS: t=.. rss=..`,
# ~0.1 s 一顆），於是「卡住不動」在字面上永遠是「有新的一行」。第一版看門狗因此完全沒發火——
# 而那一場的真相是：心臟在跳、工作沒有動（step 2 的寬 shape 上兩層之間 135 s）。所以停滯量的是
# 「距離上一次**進展行**多久」，心跳不算進展。要改判準就在這裡改（或 `--stall-progress-re`）。
_PROGRESS_RE = r"^(?!CGC-RSS)"


def _stream_child(cmd, env, live_path: Path, echo_filter: str | None = None,
                  arm_timeout_s: float = 0.0, stall_watch_s: float = 0.0,
                  stall_kill: bool = False, sample_s: float = 3.0,
                  max_samples: int = 2, kill_grace_s: float = 10.0,
                  progress_re: str | None = _PROGRESS_RE) -> dict:
    """跑一個子行程，**逐行**把 stderr 落地，並在停滯／逾時時留下可讀的讀數。

    回傳：rc／stdout／stderr（完整文字，供下游既有解析器用）＋ 這一手的中止資訊
    （`timed_out`／`stalls`／`samples`／`last_stderr_line`／`live_log`）。
    """
    live_path = Path(live_path)
    live_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    out_lines: list[str] = []
    err_lines: list[str] = []
    stalls: list[dict] = []
    samples: list[dict] = []
    state: dict = {"timed_out": False, "kill_signal": None, "kill_reason": None}
    # 「進展」與「任何一行」是兩個不同的鐘：看門狗用前者（見 `_PROGRESS_RE`），尾跡用後者。
    last_activity = {"t": started, "line": None}

    f = open(live_path, "w", encoding="utf-8", errors="replace")
    f.write("# cgc-live: started=%s  cwd=%s\n" % (time.strftime("%F %T"), ROOT))
    f.write("# cgc-live: cmd=%s\n" % " ".join(str(c) for c in cmd))
    _env = " ".join("%s=%s" % (k, v) for k, v in sorted((env or {}).items())
                    if k.startswith(("CGC_", "LLAMA_EXPERT")))
    f.write("# cgc-live: env=%s\n" % _env)
    f.write("# cgc-live: 這個檔案是**即時**寫的（每一行都 flush），所以行程被殺也看得到停滯點。\n")
    f.write("# cgc-live: 停滯判準 = 距上一次「進展行」> %.0fs（進展行 = 符合 %r；心跳不算）\n"
            % (stall_watch_s, progress_re))
    f.flush()

    p = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, errors="replace", bufsize=1,
                         start_new_session=True)

    def _pump(stream, sink: list, is_err: bool) -> None:
        try:
            for line in stream:
                sink.append(line)
                if not is_err:
                    continue
                if progress_re is None or re.search(progress_re, line):
                    last_activity["t"] = time.time()
                    last_activity["line"] = line.rstrip()[:300]
                f.write(line)
                f.flush()
                if echo_filter and re.search(echo_filter, line):
                    print("    | " + line.rstrip()[:200], flush=True)
        except Exception as e:  # noqa: BLE001 - 管子被關掉不能把軌跡一起弄丟
            if is_err:
                f.write("# cgc-live-note: stderr 讀取中止（%s）\n" % e)
                f.flush()

    th_out = threading.Thread(target=_pump, args=(p.stdout, out_lines, False), daemon=True)
    th_err = threading.Thread(target=_pump, args=(p.stderr, err_lines, True), daemon=True)
    th_out.start()
    th_err.start()

    def _sample(why: str, silent_s: float) -> dict:
        if not os.path.exists(SAMPLE_BIN):
            return {"ok": False, "why": "沒有 %s（無法取堆疊）" % SAMPLE_BIN, "silent_s": silent_s}
        sf = live_path.with_name(live_path.name.replace(".stderr.log", "") +
                                 ".sample%d.txt" % (len(samples) + 1))
        try:
            r = subprocess.run([SAMPLE_BIN, str(p.pid), str(sample_s), "-mayDie",
                                "-file", str(sf)], capture_output=True, text=True, timeout=60)
            return {"ok": r.returncode == 0, "file": str(sf), "rc": r.returncode,
                    "why": (r.stderr or "").strip()[:300], "silent_s": silent_s}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "why": str(e)[:300], "silent_s": silent_s}

    def _stop(reason: str) -> None:
        state["kill_reason"] = reason
        f.write("# cgc-live-stop: reason=%s wall=%.1fs -> SIGTERM 自己的 process group %d\n"
                % (reason, time.time() - started, p.pid))
        f.flush()
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            state["kill_signal"] = "SIGTERM"
        except (ProcessLookupError, PermissionError):
            return
        t_kill = time.time()
        while p.poll() is None and time.time() - t_kill < kill_grace_s:
            time.sleep(0.1)
        if p.poll() is None:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                state["kill_signal"] = "SIGKILL"
            except (ProcessLookupError, PermissionError):
                pass

    while p.poll() is None:
        now = time.time()
        if arm_timeout_s and (now - started) > arm_timeout_s and not state["timed_out"]:
            state["timed_out"] = True
            _stop("driver-timeout")
            break
        if stall_watch_s and (now - last_activity["t"]) > stall_watch_s \
                and len(samples) < max_samples:
            silent = now - last_activity["t"]
            sm = _sample("stall", round(silent, 1))
            sm["at"] = time.strftime("%F %T")
            samples.append(sm)
            f.write("# cgc-live-stall: %.1fs 沒有進展行; pid=%d；最後進展行 = %s；最後任何一行 = %s；"
                    "sample=%s\n" % (silent, p.pid, last_activity["line"],
                                   (err_lines[-1].rstrip()[:300] if err_lines else None),
                                   sm.get("file") or ("失敗：" + str(sm.get("why")))))
            f.flush()
            stalls.append({"at": sm["at"], "silent_s": round(silent, 1),
                           "last_progress_line": last_activity["line"],
                           "last_any_line": (err_lines[-1].rstrip() if err_lines else None)})
            print("  ⏳ 停滯 %.0fs（pid=%d）-- 已取堆疊 %s"
                  % (silent, p.pid, sm.get("file") or sm.get("why")), flush=True)
            last_activity["t"] = time.time()
            if stall_kill:
                _stop("driver-stall-kill")
                break
        time.sleep(0.25)

    rc = p.wait()
    for th in (th_out, th_err):
        th.join(timeout=5)
    wall = time.time() - started
    tail = err_lines[-1].rstrip() if err_lines else None
    f.write("# cgc-live-end: rc=%s reason=%s wall=%.1fs stderr_lines=%d stdout_lines=%d\n"
            % (rc, state["kill_reason"] or "child-exit", wall, len(err_lines), len(out_lines)))
    f.write("# cgc-live-end: last_progress=%s\n" % last_activity["line"])
    f.write("# cgc-live-end: last_stderr=%s\n" % tail)
    f.close()
    return {"rc": rc, "stdout": "".join(out_lines), "stderr": "".join(err_lines),
            "timed_out": bool(state["timed_out"]), "kill_reason": state["kill_reason"],
            "kill_signal": state["kill_signal"], "wall_s": round(wall, 1),
            "stalls": stalls, "samples": samples, "last_stderr_line": tail,
            "last_progress_line": last_activity["line"],
            "n_stderr_lines": len(err_lines), "n_stdout_lines": len(out_lines),
            "live_log": str(live_path)}


def budget_gate_preflight(env: dict, gate_path: Path | None = None) -> tuple[int, str]:
    """[CGC 2026-10-01, MEASUREMENT_CONTRACT §3.4.1] 超訂預檢：source `budget_gate.sh`。

    shell 側的接法是 `. "${REPO}/scripts/check/budget_gate.sh"` —— strict 超訂 ⇒ `exit 2`
    （零樣本）、warn ⇒ 放行且 export `CGC_BUDGET_OVERSUBSCRIBED=1`、off ⇒ 跳過、檔案不在
    ⇒ 不擋。Python 這邊用**同一支腳本、同一個 shell 語意**（`bash -c` source），免得閘門的
    兩份實作漂移。回傳 `(rc, log)`；`rc != 0` ⇒ 呼叫方必須在第一支臂之前停下來（fail-closed）。
    warn 時把 gate 自己 export 的標記讀回來，設進 `env`。
    """
    gate = Path(gate_path) if gate_path else BUDGET_GATE_SH
    if not gate.exists():
        return 0, f"[budget-gate] 找不到 {gate} —— 不擋，但這一趟沒有預檢。"
    # 在同一個 subshell 裡 source，再把 gate 自己 export 的標記讀回來 —— 不能只靠解析 log 文字。
    script = (". \"$1\"\n"
              "printf \"\\n[budget-gate] CGC_BUDGET_OVERSUBSCRIBED=%s\\n\" "
              "\"${CGC_BUDGET_OVERSUBSCRIBED:-0}\"\n")
    try:
        r = subprocess.run(["bash", "-c", script, "budget-gate", str(gate)],
                           env=env, capture_output=True, text=True, timeout=120)
    except Exception as e:  # noqa: BLE001  fail-closed：閘門跑不動就不開跑
        return 3, f"[budget-gate] 預檢失敗：{e}"
    log = (r.stdout or "") + (r.stderr or "")
    if r.returncode == 0 and "[budget-gate] CGC_BUDGET_OVERSUBSCRIBED=1" in log:
        env["CGC_BUDGET_OVERSUBSCRIBED"] = "1"
    return r.returncode, log


def run_arm(tag: str, profile: str, extra_env: dict[str, str], args, res: dict) -> dict:
    # `res` is resolved by the caller: main() resolves every arm once so the spec gate can refuse a
    # whole invocation before the first arm costs anyone GPU time.
    env, argv, scalars = res["env"], res["server_argv"], res["scalars"]

    fwd = forward_argv(argv)
    # 優先序：CLI → **測試卡 §2.5 宣告的格** → profile/server scalars（見 `cell_declared_batch`）。
    _cb, _cub = cell_declared_batch(getattr(args, "cell", None))
    if args.batch or args.ubatch:
        # 只給一個也當 CLI：另一個用**格宣告的**補齊。舊版要求兩個都給，否則整對被換成
        # profile 推導值 ⇒ `--batch 4096` 這種單邊給法會被**靜默吞掉**（這正是本專案最討厭的
        # 失敗形狀）。補不齊（格讀不到）才退回 profile。
        b, ub = (args.batch or _cb), (args.ubatch or _cub)
        if b and ub:
            why = "cli" + ("" if args.batch and args.ubatch else "（另一邊取自 cell §2.5）")
        else:
            b, ub, why = default_batch(env, scalars)
    elif _cb and _cub:
        b, ub, why = _cb, _cub, "cell §2.5"
    else:
        b, ub, why = default_batch(env, scalars)

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

    # [cell contract] 執行前把實際 cell 與測試卡 §2.5 權威 block 對照（單一口徑、fail-closed）。
    actual = {
        "ngl": _fwd_val(fwd, _CELL_FWD_ALIASES["ngl"]),
        "load_mode": _fwd_val(fwd, _CELL_FWD_ALIASES["load_mode"]),
        "threads": _fwd_val(fwd, _CELL_FWD_ALIASES["threads"]),
        "expert_cache_bytes": _fwd_val(fwd, _CELL_FWD_ALIASES["expert_cache_bytes"]),
        "cache_type_k": _fwd_val(fwd, _CELL_FWD_ALIASES["cache_type_k"]),
        "cache_type_v": _fwd_val(fwd, _CELL_FWD_ALIASES["cache_type_v"]),
        "batch": b, "ubatch": ub,
        "prompt": args.prompt, "gen": args.gen, "depths": args.depths,
        "reps": args.reps,
        "warm_skip": getattr(args, "warm_skip", 0),
        "ctx_size": getattr(args, "ctx_size", 0),
        "fixed_fill_seed": getattr(args, "fixed_fill_seed", 0) or None,
    }
    import cell_contract
    crep = cell_contract.check_cell(actual, arm_env=extra_env,
                                    cell_name=getattr(args, "cell", None))

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
    print("  " + crep.render().replace("\n", "\n  "), flush=True)
    contract_block = {"ok": crep.ok, "mismatches": crep.mismatches, "declared": crep.declared,
                      "cell": crep.cell}
    # [CGC 2026-09-29] Metal 工作集閘的輸入之一：這支臂宣告的池子大小（dry-run 與真跑都要）。
    _pool_mb, _pool_src = pool_budget_mb(fwd, env, scalars)
    if args.dry_run:
        # 起跑狀態在 dry-run 也要報（但**不拒跑**）：這是唯一在花掉 44 秒之前能問「現在這台盒子
        # 值不值得跑這一輪」的地方。dry-run 若也拒跑，會讓 selftest 隨盒子當下狀態隨機失敗。
        _dry_pre = mempress.steady()
        _sg = mempress.state_gate(_dry_pre, waived=os.environ.get("CGC_IGNORE_STATE_BUDGET") or None)
        print(f"  state gate: {'OK' if _sg['ok'] else 'UNUSABLE'}  free={_sg['free_mb']} MiB"
              f" (floor {_sg['floor_mb']})  swap={_sg['swap_used_mb']} MiB"
              + "".join(f"\n    !! {r}" for r in _sg["reasons"]), flush=True)
        # Metal 工作集閘在 dry-run 也要報（同樣**不拒跑**）：它是「花 44 秒之前問一次」的另一半。
        _mg = mempress.metal_gate(_dry_pre, _pool_mb, pool_source=_pool_src, phase="pre-launch")
        print(f"  metal gate: {'OK' if _mg['ok'] else 'REFUSE'}"
              f"{'（未武裝）' if not _mg['armed'] else ''}" + _metal_line(_mg), flush=True)
        for _w in _mg["warnings"]:
            print(f"    ~~ {_w}", flush=True)
        for _r in _mg["reasons"]:
            print(f"    !! {_r}", flush=True)
        return {"tag": tag, "profile": profile, "extra_env": extra_env, "cmd": cmd,
                "env": env, "scalars": scalars, "contract": contract_block, "dry_run": True,
                "cell": crep.cell, "state_gate": _sg, "metal_gate": _mg}

    if not crep.ok:
        raise SystemExit(f"cell contract FAIL for arm {tag} — 拒跑（fail-closed），見上。")

    run_env = dict(os.environ)
    run_env.update(env)
    # [CGC fix] 採樣對齊：sampling_env 把 resolved server argv 的 --temp/--top-p/--top-k/--min-p
    # 翻成 llama-bench parity block 讀的 env。它此前定義了卻從未被調用（死代碼），導致 llama-bench
    # 回落 common_params_sampling 的 struct 預設 temp 0.80 / top_p 0.95 / top_k 40，而 server 跑
    # 0.4 / 0.8 / 0 —— MTP accept 被系統性低估（見證行實測 mean_len 2.02 vs server 2.50）。
    # argv 已在 resolve 時套用 arm extra_env，故此處取到的就是該 arm 最終生效值。
    run_env.update(sampling_env(argv))
    # A llama-bench arm is ONE child process holding the GPU for minutes with no per-request
    # boundary to hang a reading on, so the series has to come from a thread. The launch
    # reading is taken before the spawn (thermal_pressure.Sampler.start), which is the reading
    # the prefill separation is defined on -- taking it after would silently redefine it.
    # 【CGC 2026-09-29】檔名要在**跑之前**就知道：live log 必須在子行程起跑前就開好，否則被殺掉的
    # 那一輪連檔名都沒有（見 `_stream_child`）。下面對 `stem` 的用法完全不變。
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", tag)[:80]
    shape = re.sub(r"[^A-Za-z0-9._-]", "-", f"p{args.prompt}_n{args.gen}_d{args.depths}_r{args.reps}")
    stem = Path(args.workdir) / f"llama_bench_{safe}_{shape}"
    # 有產物路徑時，live log 放在**產物旁邊**（`<json>.logs/live/`）：`--workdir`（預設 /tmp）遲早
    # 會被清掉，而「停在哪裡」正是那種事後才想引用、卻已經不在磁碟上的東西。
    if getattr(args, "json", None):
        live_log = Path(str(args.json) + ".logs") / "live" / f"{safe}.{shape}.stderr.log"
    else:
        live_log = Path(args.workdir) / f"llama_bench_{safe}_{shape}.live.stderr.log"

    sampler = thermal.Sampler()
    msamp = mempress.Sampler()
    t0 = time.time()
    refused_state = None
    with sampler, msamp:
        # [CGC 2026-09-28] 起跑狀態閘：在 spawn **之前**判，因為它問的正是「這台盒子現在適不適合量」。
        # 從前那道閘是跑後的絕對成長 —— 那等於先花 44 秒，再因為「盒子當時不乾淨」把臂判死；
        # 而絕對成長與起跑 free 的 rho 是 -0.870（出處在 memory_pressure 的 docstring），也就是
        # 它其實在量盒子。現在那根絕對桿子降為證據（可用 CGC_SWAP_BUDGET_MB 選回），臂側的訊號
        # 改走「成長扣掉起跑狀態期望值後的殘差」。
        # 判的讀值是 median-of-3（`steady`）：單一 `pages_free` 樣本在這台盒子上會擺盪 GB 級，
        # 而擺盪的兩端會落在 5000 MiB 地板的兩邊。Sampler 的 launch 讀值照舊留給 attribution，
        # 不拿它當閘（它是一瞬，而且它的定義是「child 出現之前」，不是「穩定狀態」）。
        _pre = mempress.steady()
        refused_state = mempress.state_gate(
            _pre, waived=os.environ.get("CGC_IGNORE_STATE_BUDGET") or None)
        # [CGC 2026-09-29] Metal 工作集閘（起跑前）：池子 ＋ 起跑 Metal 駐留（＋保留）vs 上限。
        # 它與上面那道正交：起跑 available 可以很夠而 Metal 上限已經裝不下（交付 cell 那一場就是）。
        # 上限讀不到時 `armed=False` ⇒ 不拒跑，但會在產物與 stdout 裡說「這一輪沒有這根桿子」。
        refused_metal = mempress.metal_gate(
            _pre, _pool_mb, pool_source=_pool_src, phase="pre-launch",
            waived=os.environ.get("CGC_IGNORE_METAL_BUDGET") or None)
        if (refused_state["ok"] and refused_metal["ok"]) or args.dry_run:
            print(f"  live log : {live_log}", flush=True)
            print(f"  brake    : arm-timeout={getattr(args, 'arm_timeout', 0) or 'off'}s "
                  f"stall-watch={getattr(args, 'stall_watch', 0) or 'off'}s "
                  f"stall-kill={bool(getattr(args, 'stall_kill', False))}", flush=True)
            stream = _stream_child(cmd, run_env, live_log,
                                   echo_filter=(getattr(args, "live_echo", "") or _LIVE_ECHO),
                                   arm_timeout_s=float(getattr(args, "arm_timeout", 0) or 0),
                                   stall_watch_s=float(getattr(args, "stall_watch", 0) or 0),
                                   stall_kill=bool(getattr(args, "stall_kill", False)),
                                   progress_re=(getattr(args, "stall_progress_re", None)
                                                if getattr(args, "stall_progress_re", None) is not None
                                                else _PROGRESS_RE))
            rc_out, sout, serr = stream["rc"], stream["stdout"], stream["stderr"]
        else:
            rc_out, sout, serr = None, "", ""
            stream = {"rc": None, "stdout": "", "stderr": "", "timed_out": False,
                      "kill_reason": None, "kill_signal": None, "wall_s": 0.0, "stalls": [],
                      "samples": [], "last_stderr_line": None, "n_stderr_lines": 0,
                      "n_stdout_lines": 0, "live_log": str(live_log)}
    wall = time.time() - t0
    # The filename has to carry the SHAPE. `tag` alone collides for every arm that sets its env
    # through the `PROFILE:ENV=...` form, because there the spec string *is* the tag -- so
    # `-n 24` and `-n 128` wrote to one path and the first run's stderr (the only place the
    # expert cache's own counters land) was silently overwritten. Hit on 2026-09-16 while running
    # Backup/run_instrument_compare.sh. Distinct `--json` paths hid it: the JSON evidence of both
    # runs survived while only the last stderr did.
    # （`safe`／`shape`／`stem` 已在起跑前算好：live log 需要那個檔名才寫得出來。）
    stem.with_suffix(".stderr.log").write_text(serr, errors="replace")
    stem.with_suffix(".json").write_text(sout, errors="replace")
    # 【CGC 2026-09-29】產物 ↔ log 成對：stderr 是唯一能證明「量具真的動了」的東西，而上面那兩份
    # 落在 `--workdir`（預設 /tmp ⇒ 遲早被清掉），產物卻被歸檔到 `Backup/` ⇒ 歸檔完成的那一刻就永遠
    # 驗不了（42.9%／41.8% 那兩支就是這樣）。所以每一臂在**產物旁邊**再寫一份不可被覆寫的 log
    # （`<stem>.logs/<tag>.stderr.log`，唯一檔名），合併版由 `main` 在收尾時寫（那時才知道這一輪的邊界）。
    # 順序是先 log、後產物：中斷只會留下孤兒 log，不會留下一支無 log 的產物。
    if args.json:
        ibind().write_pair_arm(args.json, tag, serr, shape=shape)
        _PAIR_BUFFER.append((tag, shape, serr))

    # Tolerate a mid-arm failure. `llama-bench` runs every (p,n,d) as its own llama_context and
    # exits non-zero on the first one that dies (usually GPU OOM at a large ubatch), so a single
    # bad shape must not discard the shapes that already succeeded -- those are the measurement.
    rows = parse_rows(sout)
    # 起跑狀態被拒 ⇒ 根本沒起跑：那不是 incomplete（它沒有「跑一半」），但一樣不可用，
    # 理由由 memory_gate 的 state 那一半帶出來，rc 由 main 統一判 4。
    incomplete = (rc_out is not None and rc_out != 0)
    err = None
    if incomplete:
        # The LAST stderr line is the cache's teardown stats, not the failure -- pick the first
        # line that actually names a failure, so the recorded reason is the real one.
        sigs = ("failed to decode", "res = -", "error:", "GGML_ASSERT", "abort", "SIGSEGV",
                "SIGABRT", "out of memory", "Unable to")
        lines = [l.strip() for l in serr.splitlines() if l.strip()]
        err = next((l for l in lines if any(s in l for s in sigs)), lines[-1] if lines else "")
        print(f"  !! arm exited rc={rc_out}: {err}", flush=True)
        print(f"     recovered {len(rows)} completed instance(s) from the streamed JSON", flush=True)
        # 【CGC 2026-09-29】驅動自己踩剎車的那一輪**不是**引擎失敗：中止點就是這次要讀的東西，
        # 所以它不能走「沒有完成任何 instance ⇒ 放棄」那條路（那條路連產物都不寫，於是「停在哪裡」
        # 又變成沒有讀數）。留下臂紀錄（rows 可為空），讓 json 與成對 log 照寫。
        if not rows and not stream.get("timed_out"):
            print(sout[-1500:], flush=True)
            raise SystemExit(f"llama-bench failed for arm {tag} (rc={rc_out}) with no "
                             f"completed instance to report")
    if stream.get("timed_out") or stream.get("kill_reason") == "driver-stall-kill":
        err = ("驅動中止（%s）：arm-timeout=%ss／stall-watch=%ss；停滯點見 live log %s"
               "（最後**進展**行：%s）"
               % (stream.get("kill_reason") or "driver", getattr(args, "arm_timeout", 0),
                  getattr(args, "stall_watch", 0), stream.get("live_log"),
                  stream.get("last_progress_line")))
        print(f"  ⛔ {err}", flush=True)
        for s in stream.get("stalls") or []:
            print(f"     停滯 {s['silent_s']}s @ {s['at']}：最後進展行 = {s['last_progress_line']}"
                  f"（最後任何一行 = {s['last_any_line']}）", flush=True)
        for sm in stream.get("samples") or []:
            print(f"     堆疊 : {sm.get('file') or sm.get('why')}", flush=True)

    stats = harvest_bench_stats(serr)
    spec_draft = spec_draft_health(serr)
    # [CGC 2026-09-29] 量具綁定（設了開關 ≠ 綁上了圖）：寫進產物，並在 stdout 留一行可見的判定。
    # 需求由臂自己的 env 推（`CGC_MISS_MASK_DBG` ⇒ 地圖必須填起來 …），可用 --require-instrument 補。
    instrument = instrument_report(
        serr, extra_env, env,
        require=[x.strip() for x in (getattr(args, "require_instrument", "") or "").split(",")
                 if x.strip()])
    # 根本沒起跑的那一輪（起跑閘不合格）：stderr 是空的，量具當然不會動 —— 那不是「量具沒綁上」，
    # 而是「沒有量具可驗」。兩者要分開，否則拒跑會把 UNBOUND 這個訊號洗掉。
    _never_launched = ((refused_state is not None and not refused_state.get("ok"))
                       or not (refused_metal or {}).get("ok", True))
    if _never_launched:
        instrument = dict(instrument, verdict="N/A", required=[], escaped=[],
                          why="這一輪根本沒有起跑（起跑閘不合格）⇒ 沒有量具可驗")
    # [provenance] warm_skip_applied: 從產物 tg row 的 n_gen 驗證 warm-skip 是否「真生效」。
    # 名義傳了 --warm-skip 不等於被套用；被套用時 tg n_gen 應 == gen - warm_skip。
    warm_skip_n = int(getattr(args, "warm_skip", 0) or 0)
    tg_rows = [r for r in rows if int(r.get("n_prompt", 0)) == 0 and int(r.get("n_gen", 0)) > 0]
    warm_skip_applied = None
    if warm_skip_n > 0 and tg_rows:
        warm_skip_applied = all(
            int(r.get("n_gen", -1)) == int(args.gen) - warm_skip_n for r in tg_rows)
        if warm_skip_applied is False:
            print(f"  ⛔ warm_skip_applied=False: 名義 warm-skip {warm_skip_n}，但 tg n_gen != "
                  f"{int(args.gen) - warm_skip_n}（實際 {[r.get('n_gen') for r in tg_rows]}）"
                  f" — 「名義有、實際沒有」，此 decode 數字隔離、不可引用。", flush=True)
    fixed_fill_seed_actual = int(getattr(args, "fixed_fill_seed", 0) or 0) or None
    engine_build = next((r.get("build_commit") for r in rows if r.get("build_commit")), None)
    out = {"tag": tag, "profile": profile, "extra_env": extra_env, "batch": b, "ubatch": ub,
           "batch_why": why, "wall_s": round(wall, 1), "env": env, "scalars": scalars,
           "contract": contract_block,
           "rows": rows, "cache": stats, "incomplete": incomplete, "error": err,
           "refused_preflight": bool(refused_state is not None and not refused_state["ok"])
           or not (refused_metal or {}).get("ok", True),
           "state_gate": refused_state,
           "metal_gate": refused_metal,
           "state_preflight_reading": (_pre if refused_state is not None else None),
           "thermal": sampler.result,
           # Per-row windows, because ONE launch here measures TWO rows (a 2048-token prefill and
           # a 64-token decode) and they can sit on opposite sides of the thermal line.
           # Measured 2026-09-28: pp window {NOMINAL: 42} / tg window {HEAVY: 34} in the same
           # launch, whose launch-level hist said 79/153 HEAVY. Without this, "which row was
           # hot" is unanswerable from the artifact, and a thermal gate can only be about the
           # launch -- i.e. about whichever row happens to dominate the histogram.
           "thermal_windows": thermal.windows((sampler.result or {}).get("samples") or [], rows),
           "memory": msamp.result,
           "attribution": mempress.attribution(sampler.result, msamp.result),
           # Recorded, not inferred: whether a row came from the speculative gen path is a property
           # of the invocation, and a spec row wears the same `tg` label as a plain one.
           "spec_type": args.spec_type or None,
           "warm_skip": getattr(args, "warm_skip", 0) or None,
           "warm_skip_applied": warm_skip_applied,
           "fixed_fill_seed": fixed_fill_seed_actual,
           "engine_build": engine_build,
           "ctx_size": getattr(args, "ctx_size", 0) or None,
           "spec_draft_n_max": args.spec_draft_n_max,
           # 【CGC 2026-09-29】這一手的剎車／看門狗讀數（見 `_stream_child`）：即使沒有 t/s，
           # 「跑到哪裡停住」本身就是這次要讀的東西，所以它要進產物，並與 live log 路徑綁在一起。
           "live_log": stream.get("live_log"),
           "arm_timeout_s": getattr(args, "arm_timeout", 0) or None,
           "stall_watch_s": getattr(args, "stall_watch", 0) or None,
           "driver_abort": ({"timed_out": bool(stream.get("timed_out")),
                             "kill_reason": stream.get("kill_reason"),
                             "kill_signal": stream.get("kill_signal"),
                             "wall_s": stream.get("wall_s"),
                             "stalls": stream.get("stalls"),
                             "samples": stream.get("samples"),
                             "last_stderr_line": stream.get("last_stderr_line"),
                             "last_progress_line": stream.get("last_progress_line"),
                             "n_stderr_lines": stream.get("n_stderr_lines")}
                            if (stream.get("timed_out") or stream.get("stalls")) else None),
           # spec 健康度（見 spec_draft_health）：任一轮 rep draft 死亡（drafted=0）⇒ 那一輪
           # 量的是純解碼，t/s 不是 spec 的 ⇒ 這一臂不可引用（harness 的第六道閘門會拒收）。
           "spec_draft": spec_draft,
           # 量具綁定判定（見 instrument_report）：UNBOUND ⇒ 這一輪的計數器不可引用。
           "instrument": instrument,
           "require_instrument": [x.strip() for x in
                                  (getattr(args, "require_instrument", "") or "").split(",")
                                  if x.strip()] or None}
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
    mem = msamp.result
    mlaunch = mem.get("launch") or {}
    mend = mem.get("end") or {}
    mw = mem.get("worst") or {}
    attr = out.get("attribution") or {}
    print(f"  memory: launch swap={mlaunch.get('swap_used_mb')} MiB free={mlaunch.get('pages_free_mb')} MiB "  
          f"-> end swap={mend.get('swap_used_mb')} MiB wired={mend.get('pages_wired_mb')} MiB "  
          f"worst_swap={mw.get('max_swap_mb')} min_free={mw.get('min_free_mb')} procs={mend.get('llama_procs')}", flush=True)
    print(f"  attribution: {attr.get('verdict')}  -- {attr.get('why')}", flush=True)
    sd_mark = "✅" if spec_draft["ok"] else "⛔"
    print(f"  {sd_mark} spec draft {spec_draft['state']}  -- {spec_draft['why']}", flush=True)
    ibv = out.get("instrument") or {}
    ib_mark = {"BOUND": "✅", "N/A": "·"}.get(ibv.get("verdict"), "⛔")
    print(f"  instrument: {ib_mark} {ibv.get('verdict')}  -- {ibv.get('why')}", flush=True)
    if ibv.get("verdict") == "UNBOUND":
        print("  ⛔ 量具 UNBOUND：開關設了但沒綁上圖 ⇒ 這一輪的計數器與由此而來的結論不可引用。"
              "（判準：scripts/check/instrument_binding.py）", flush=True)

    # [CGC 2026-09-28] 記憶體閘：**起跑狀態**是閘門（跑前可判），跑後那一半只在「扣掉起跑狀態
    # 的期望值」之後才當閘（殘差）。兩者的校準與為什麼改在 memory_pressure 的 docstring。
    # 期望值的輸入用 pre-flight 的 steady 讀值（與起跑閘同一個瞬間），不是 Sampler 的單一樣本：
    # 兩者實測差 1.2 GB，而斜率 0.44 ⇒ 光是換一個讀值就能讓期望值差 ~0.5 GB（@curve_gate）。
    out["memory_gate"] = mempress.curve_gate(
        msamp.result, waived=os.environ.get("CGC_IGNORE_SWAP_BUDGET") or None,
        launch_reading=(_pre if refused_state is not None else None))
    mg = out["memory_gate"]
    # [CGC 2026-09-29] Metal 工作集閘（跑後那一半）：池子 ＋ **整趟峰值** Metal 駐留 vs 上限，
    # 並且從本次的 stderr 讀引擎自己印的上限（同時把值快取下來，讓下一次起跑前那道閘有數字）。
    out["metal_gate_peak"] = mempress.metal_gate(
        msamp.result, _pool_mb, pool_source=_pool_src, phase="peak", log_text=serr,
        waived=os.environ.get("CGC_IGNORE_METAL_BUDGET") or None)
    _mgp = out["metal_gate_peak"]
    if _mgp.get("ceiling_mb") and "本次" in str(_mgp.get("ceiling_source")):
        # 連整趟的峰值 wired 一起記：下一次**起跑前**才能在算術上多說一句
        # 「你這一趟的起跑合格，但上一趟的峰值告訴你整趟會超」（見 metal_gate）。
        _peak = next((i["mb"] for i in _mgp["items"] if i["name"] == "wired_mb"), None)
        out["metal_ceiling_cached"] = mempress.remember_metal_ceiling(
            _mgp["ceiling_mb"], _mgp["ceiling_source"], peak_wired_mb=_peak)
    print(f"  memory gate: {'OK' if mg['ok'] else 'UNUSABLE'}"
          + (f"  (waived: {mg['waived']})" if mg.get("waived") else "")
          + f"  起跑 free={mg['state_gate']['free_mb']} MiB (floor {mg['state_gate']['floor_mb']})"
          + f"  峰值成長={mg['peak_growth_mb']} MiB  殘差={mg['residual_mb']} MiB", flush=True)
    print(f"    absolute growth: {mg['growth_criterion']}", flush=True)
    print(f"    residual: {mg['residual_criterion']}", flush=True)
    print(f"    swap 曲線: {mg['sparkline']}   ({mg['n']} 個取樣，峰值在 {mg['peak_at']})", flush=True)
    for warning in mg.get("warnings") or []:
        print(f"    ~~ {warning}", flush=True)
    for reason in mg["reasons"]:
        print(f"    !! {reason}", flush=True)
    for key, phase_tag in (("metal_gate", "起跑"), ("metal_gate_peak", "峰值")):
        _m = out.get(key) or {}
        if not _m:
            continue
        print(f"  metal（{phase_tag}）: {'OK' if _m['ok'] else 'REFUSE'}"
              f"{'（未武裝：讀不到上限或項目 ⇒ 這一輪沒有這根桿子）' if not _m['armed'] else ''}"
              + _metal_line(_m), flush=True)
        for w in _m["warnings"]:
            print(f"    ~~ {w}", flush=True)
        for r in _m["reasons"]:
            print(f"    !! {r}", flush=True)
    if out.get("refused_preflight"):
        print("    （起跑閘不合格（狀態或 Metal 預算）⇒ 根本沒有起跑，這一輪沒有量到任何 t/s）"
              "—— 這是「拒跑」不是「跑壞了」：看到這行就別去找那一輪的 t/s。", flush=True)
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


# --- self-test: the spec gate's own cases ------------------------------------------------------
# (name, arm env, resolved env, --spec-type, must the gate refuse?)
SELFTEST_CASES: list[tuple[str, dict, dict, str, bool]] = [
    ("MTP=1 without --spec-type (the trap)", {"CGC_SERVER_MTP": "1"}, {}, "", True),
    ("MTP=1 with --spec-type", {"CGC_SERVER_MTP": "1"}, {}, "draft-mtp", False),
    ("MTP=0 is an honest off arm", {"CGC_SERVER_MTP": "0"}, {}, "", False),
    ("plain cell, no MTP at all", {}, {}, "", False),
    ("MTP=1 only in the resolved env", {}, {"CGC_SERVER_MTP": "1"}, "", True),
    ("non-1 value is still armed", {"CGC_SERVER_MTP": "yes"}, {}, "", True),
    # [CGC 2026-10-03] the per-arm spec shim: this is the ONLY shape in which a multi-arm cell can
    # hold both an off and an on arm (`--spec-type` is a whole-run flag). Allowed, loudly.
    ("MTP=1 + per-arm spec shim (the pairing enabler)",
     {"CGC_SERVER_MTP": "1", "LLAMA_BENCH_SPEC": "1"}, {}, "", False),
    ("MTP=1 + shim with explicit depth",
     {"CGC_SERVER_MTP": "1", "LLAMA_BENCH_SPEC": "1", "LLAMA_BENCH_SPEC_DRAFT_N_MAX": "2"}, {}, "", False),
    ("shim explicitly 0 arms nothing",
     {"CGC_SERVER_MTP": "1", "LLAMA_BENCH_SPEC": "0"}, {}, "", True),
    ("empty shim arms nothing",
     {"CGC_SERVER_MTP": "1", "LLAMA_BENCH_SPEC": ""}, {}, "", True),
    # The shim is read from the ENV llama-bench actually inherits, so it counts wherever it lands --
    # what matters is that it is present, not which of the two dicts carried it.
    ("shim in the resolved env arms it too (it is still the env llama-bench reads)",
     {}, {"CGC_SERVER_MTP": "1", "LLAMA_BENCH_SPEC": "1"}, "", False),
]


def selftest() -> int:
    bad = 0
    for name, extra, res_env, spec_type, must in SELFTEST_CASES:
        got = bool(mtp_without_spec(extra, res_env, spec_type))
        if got != must:
            bad += 1
            print(f"  FAIL [{name}]: refused={got}, expected {must}")
        else:
            print(f"  ok   [{name}]: {'refused' if got else 'allowed'}")
    print(f"spec-armed gate selftest: {len(SELFTEST_CASES) - bad}/{len(SELFTEST_CASES)} unit cases passed")

    # [CGC fix] sampling_env 必須把 resolved server argv 的採樣翻成 llama-bench env，且 last
    # occurrence wins（argv 先 --temp 0 後 --temp 0.4，取後者；否則會在標榜 parity 時釘成 greedy）。
    scases = [
        ("basic translate", ["--temp","0.4","--top-p","0.8","--top-k","0"],
         {"CGC_SERVER_TEMP":"0.4","CGC_SERVER_TOP_P":"0.8","CGC_SERVER_TOP_K":"0"}),
        ("last occurrence wins", ["--temp","0","x","--temp","0.4"],
         {"CGC_SERVER_TEMP":"0.4"}),
        ("underscore aliases", ["--top_p","0.8","--min_p","0.0"],
         {"CGC_SERVER_TOP_P":"0.8","CGC_SERVER_MIN_P":"0.0"}),
    ]
    sbad = 0
    for name, sav, must in scases:
        got = sampling_env(sav)
        if any(got.get(k) != v for k,v in must.items()):
            sbad += 1; print(f"  FAIL [{name}]: {got}")
        else:
            print(f"  ok   [{name}]: {got}")
    print(f"sampling-env selftest: {len(scases)-sbad}/{len(scases)} cases passed")

    # End-to-end, because the unit cases cannot catch a gate that is never reached: the real CLI
    # must refuse the real trap and still allow the same arm once --spec-type is given. --dry-run
    # stops both before any launch (only the arm resolver actually runs).
    e2e = [("trap via the real CLI", ["--arms", "prod-new:CGC_SERVER_MTP=1", "--dry-run"], True),
           ("same arm with --spec-type", ["--arms", "prod-new:CGC_SERVER_MTP=1",
                                          "--spec-type", "draft-mtp", "--dry-run"], False)]
    ebad = 0
    for name, argv, must in e2e:
        p = subprocess.run([sys.executable, str(Path(__file__).resolve()), *argv],
                           cwd=str(ROOT), capture_output=True, text=True)
        refused = p.returncode != 0
        msg_ok = (not must) or "--spec-type" in (p.stdout + p.stderr)
        if refused != must or not msg_ok:
            ebad += 1
            print(f"  FAIL [{name}]: rc={p.returncode} (expected refuse={must})\n"
                  f"{(p.stdout + p.stderr)[-800:]}")
        else:
            print(f"  ok   [{name}]: rc={p.returncode}")
    print(f"spec-armed gate selftest: {len(e2e) - ebad}/{len(e2e)} end-to-end cases passed")

    # [CGC fix 2026-09-28] arm env 靜默丟失防線。`!` 前綴與 allowlist 外的鍵都必須被抓到，
    # 而真正會生效的鍵不能被誤傷（誤傷 = 把仍在用的臂擋在門外）。
    acases = [
        ("`!` 必須被剝掉", "!CGC_GATHER_SLAB_CAP=64;CGC_SERVER_MTP=1",
         {"CGC_GATHER_SLAB_CAP": "64", "CGC_SERVER_MTP": "1"}),
        ("無 `!` 原樣保留", "CGC_SERVER_LAYER_CAPS=40-40:16", {"CGC_SERVER_LAYER_CAPS": "40-40:16"}),
    ]
    abad = 0
    for name, spec, must in acases:
        got = parse_arm_env(spec)
        if got != must:
            abad += 1; print(f"  FAIL [{name}]: {got} != {must}")
        else:
            print(f"  ok   [{name}]: {got}")
    print(f"arm-env parse selftest: {len(acases) - abad}/{len(acases)} cases passed")

    # 落地檢查（要真的叫 run_server.sh，但不碰 GPU）
    dcases = [
        ("生效的鍵不誤傷", {"CGC_GATHER_SLAB_CAP": "64"}, False),
        ("入口名轉底層名也算送達", {"CGC_SERVER_LAYER_CAPS": "40-40:16"}, False),
        ("`!` 前綴 → 抓到", {"!CGC_GATHER_SLAB_CAP": "64"}, True),
        ("allowlist 外的鍵 → 抓到", {"CGC_TOTALLY_BOGUS_KEY": "1"}, True),
    ]
    dbad = 0
    for name, extra_env, must in dcases:
        got = bool(arm_env_dropped("prod-new", extra_env, resolve("prod-new", extra_env)))
        if got != must:
            dbad += 1; print(f"  FAIL [{name}]: dropped={got}, expected {must}")
        else:
            print(f"  ok   [{name}]: dropped={got}")
    print(f"arm-env landed selftest: {len(dcases) - dbad}/{len(dcases)} cases passed")

    # [CGC 2026-09-28] 起跑狀態閘的 e2e：它必須在「什麼都還沒跑」時就能回答，所以走 --dry-run；
    # dry-run **不拒跑**（否則這條測試會隨盒子當下狀態隨機失敗），所以斷言的是「同一支臂、同一台
    # 盒子，只改一個 env，判詞翻面」—— 那才是「閘真的接到 CLI 上」的證據。
    gbad = 0
    got = {}
    # 用**被強制**的那條（available），不是降為警告的 free 那一條：
    # 拿 free 當測試對象會在「free 很小但 available 夠」的盒子上誤導（開機後就是那樣子）。
    for name, floor in (("floor=1 MiB", "1"), ("floor=999999 MiB", "999999")):
        penv = {**os.environ, "CGC_STATE_AVAILABLE_FLOOR_MB": floor}
        p = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                            "--arms", "prod-new", "--dry-run"],
                           cwd=str(ROOT), env=penv, capture_output=True, text=True)
        blob = p.stdout + p.stderr
        got[name] = ("UNUSABLE" if "state gate: UNUSABLE" in blob else
                     "OK" if "state gate: OK" in blob else "MISSING")
        print(f"  ok   [{name}]: state gate = {got[name]}")
    if not (got["floor=1 MiB"] == "OK" and got["floor=999999 MiB"] == "UNUSABLE"):
        gbad += 1
        print(f"  FAIL [state gate e2e]: {got} -- 期望 floor=1 → OK、floor=999999 → UNUSABLE、"
              f"兩者都不是 MISSING（MISSING = 閘沒被跑到）")
    print(f"state-gate selftest: {2 - gbad}/2 cases passed")

    # [CGC 2026-09-29] 產物 ↔ log 成對：寫入端（這裡）與檢查端（`pair_status`）必須是同一份判準，
    # 所以斷言的是「跑完真的產出檢查端認得的合併版」，而不是「我們有呼叫某個函式」。用 tempdir。
    import shutil
    import tempfile
    pbad = 0
    pd = tempfile.mkdtemp(prefix="mtx_pair_")
    try:
        art = os.path.join(pd, "run.json")
        with open(art, "w") as f:
            json.dump([{"tag": "prod-new:CGC_MISS_MASK=1",
                        "rows": [{"n_prompt": 0, "n_gen": 64}]}], f)
        ib = ibind()
        a1 = ib.write_pair_arm(art, "prod-new:CGC_MISS_MASK=1", "CGC-RB-FEED: feeds=3\n",
                               shape="p0_n64")
        canon = ib.write_pair_bundle(art, [("prod-new:CGC_MISS_MASK=1", "p0_n64",
                                            "CGC-RB-FEED: feeds=3\n")])
        st = ib.pair_status(art)
        if not (st["paired"] and st["arm_logs"] and st["bytes"] > 0):
            pbad += 1
            print(f"  FAIL [pair log]: pair_status={st}")
        else:
            print(f"  ok   [pair log]: 合併版 {os.path.basename(canon)} 被檢查端認得"
                  f"（{st['bytes']} bytes，per-arm {len(st['arm_logs'])} 份）")
        if "===== arm " not in open(canon, encoding="utf-8").read():
            pbad += 1
            print("  FAIL [pair log]: 合併版沒有臂分隔（多臂產物會分不出誰是誰）")
        else:
            print("  ok   [pair log]: 合併版有臂分隔")
        if os.path.dirname(a1) != os.path.join(pd, "run.logs"):
            pbad += 1
            print(f"  FAIL [pair log]: per-arm log 不在 <stem>.logs/：{a1}")
    finally:
        shutil.rmtree(pd, ignore_errors=True)
    print(f"pair-log selftest: {3 - pbad}/3 cases passed")

    # 【CGC 2026-09-29】串流／剎車的性質測試。要証的是**性質**，不是呼叫過什麼：
    #   (a) 子行程還活著的時候，它已經印出的那一行就已經在磁碟上（被殺也留得下來）；
    #   (b) arm-timeout 會真的回收子行程（不留孤兒）、且判詞是 driver-timeout。
    sbad2 = 0
    sd = tempfile.mkdtemp(prefix="mtx_stream_")
    try:
        # (a) 子行程先印一行、再睡 20 s；我們在它還睡著時讀檔。
        lp = Path(sd) / "alive.stderr.log"
        box = {}
        th = threading.Thread(
            target=lambda: box.update(_stream_child(
                ["bash", "-c", "echo STREAM-ALIVE >&2; sleep 20"], dict(os.environ), lp,
                arm_timeout_s=15.0)), daemon=True)
        th.start()
        seen = None
        t0 = time.time()
        while time.time() - t0 < 8 and seen is None:
            if lp.exists():
                txt = lp.read_text(errors="replace")
                if "STREAM-ALIVE" in txt:
                    seen = (round(time.time() - t0, 2), box.get("rc", "still-running"))
            time.sleep(0.1)
        th.join(timeout=40)
        if seen is None or box.get("timed_out") is not True:
            sbad2 += 1
            print(f"  FAIL [stream]: 子行程活著時沒讀到那一行（{seen}）；box={ {k: box.get(k) for k in ('rc','timed_out','kill_reason')} }")
        else:
            print(f"  ok   [stream]: 子行程還在跑（{seen[1]}）就讀到 stderr 那一行（開跑後 {seen[0]}s）")
        if not (box.get("rc") is not None and len(box.get("stalls") or []) == 0 and
                box.get("live_log") == str(lp)):
            sbad2 += 1
            print(f"  FAIL [brake]: { {k: box.get(k) for k in ('rc','kill_reason','wall_s')} }")
        else:
            print(f"  ok   [brake]: 逾時回收 rc={box['rc']} kill={box['kill_signal']} "
                  f"wall={box['wall_s']}s，離場時 log 結尾={box['last_stderr_line']!r}")
        # (c) 停滯看門狗：只印一行然後睡 ⇒ 應該抓到停滯（且不殺）。
        lp2 = Path(sd) / "stall.stderr.log"
        box2 = _stream_child(["bash", "-c", "echo ONLY-LINE >&2; sleep 4"], dict(os.environ),
                             lp2, stall_watch_s=1.0, arm_timeout_s=30.0)
        if not box2.get("stalls") or box2["stalls"][0]["last_progress_line"] != "ONLY-LINE":
            sbad2 += 1
            print(f"  FAIL [stall-watch]: {box2.get('stalls')}")
        else:
            print(f"  ok   [stall-watch]: 停滯 {box2['stalls'][0]['silent_s']}s、點名最後進展行 "
                  f"{box2['stalls'][0]['last_progress_line']!r}，且沒有把它殺掉（rc={box2['rc']}）")
        # (d) 心跳不得蓋住停滯：子行程只印 CGC-RSS（引擎每秒都印它）⇒ 第一版「有沒有新的一行」
        # 的判準會永遠不發火，而真實那一場就是這樣（step 2 卡 135 s 而心跳沒斷）。
        lp3 = Path(sd) / "hb.stderr.log"
        box3 = _stream_child(
            ["bash", "-c", "for i in 1 2 3 4; do echo \"CGC-RSS: t=$i rss=1\" >&2; "
                           "sleep 0.6; done"], dict(os.environ), lp3,
            stall_watch_s=1.0, arm_timeout_s=30.0)
        if not box3.get("stalls") or box3["stalls"][0]["last_progress_line"] is not None:
            sbad2 += 1
            print(f"  FAIL [heartbeat 不蓋住停滯]: {box3.get('stalls')}")
        else:
            print(f"  ok   [heartbeat 不蓋住停滯]: 只有心跳也算停滯 "
                  f"({box3['stalls'][0]['silent_s']}s，進展行={box3['stalls'][0]['last_progress_line']})"
                  f"，而心跳本身照樣留在 log（{box3['n_stderr_lines']} 行）")
    finally:
        shutil.rmtree(sd, ignore_errors=True)
    print(f"stream/brake selftest: {4 - sbad2}/4 cases passed")

    # [CGC 2026-10-01] budget gate 接線（MEASUREMENT_CONTRACT §3.4.1）：四個分支都要能判，
    # 而且必須是「真的 source 一支腳本」，不是只比對字串。
    bgbad = 0
    bd = tempfile.mkdtemp(prefix="mtx_budget_")
    try:
        fake = Path(bd) / "budget_gate.sh"
        fake.write_text('echo "[budget-gate] 拒跑（exit 2）" >&2\nexit 2\n', encoding="utf-8")
        rc, _ = budget_gate_preflight({"BUDGET_GATE": "strict"}, fake)
        if rc != 2:
            bgbad += 1
            print(f"  FAIL [budget strict]: rc={rc}，want 2（超訂必須拒跑）")
        else:
            print("  ok   [budget strict]: 超訂 ⇒ rc=2（拒跑、零樣本）")
        fake.write_text("export CGC_BUDGET_OVERSUBSCRIBED=1\n", encoding="utf-8")
        genv = {"BUDGET_GATE": "warn"}
        rc, _ = budget_gate_preflight(genv, fake)
        if not (rc == 0 and genv.get("CGC_BUDGET_OVERSUBSCRIBED") == "1"):
            bgbad += 1
            print(f"  FAIL [budget warn]: rc={rc} env={genv}")
        else:
            print("  ok   [budget warn]: 放行且標記帶回 env")
        fake.write_text(":\n", encoding="utf-8")
        genv = {"BUDGET_GATE": "strict"}
        rc, _ = budget_gate_preflight(genv, fake)
        if not (rc == 0 and "CGC_BUDGET_OVERSUBSCRIBED" not in genv):
            bgbad += 1
            print(f"  FAIL [budget clean]: rc={rc} env={genv}")
        else:
            print("  ok   [budget clean]: 未超訂 ⇒ rc=0、無標記")
        rc, log = budget_gate_preflight({"BUDGET_GATE": "strict"}, Path(bd) / "missing.sh")
        if not (rc == 0 and "找不到" in log):
            bgbad += 1
            print(f"  FAIL [budget missing]: rc={rc} log={log!r}")
        else:
            print("  ok   [budget missing]: 找不到閘門 ⇒ 不擋（同 shell 版）")
    finally:
        shutil.rmtree(bd, ignore_errors=True)
    print(f"budget-gate selftest: {4 - bgbad}/4 cases passed")

    # [CGC 2026-10-06] 第六道閘門的判準只有一份（本檔 `spec_draft_dead`），但有三個消費端：
    #   * harness bench 拒收（rc=6） —— harness._spec_draft_gate
    #   * sync 拒寫（code 6）        —— experiment_sync.sync_gate
    #   * 排行榜不得上榜（R7）       —— experiment_sync._arm_on_board
    # 三者原本各寫一份，而 2026-10-06 實測到不一致：harness 是 fail-closed（state 只認
    # OK／NOT_SPEC），sync 與上榜卻只擋字面上的 "INVALID" ⇒ NO_STATE／打錯字／未來新增的狀態會
    # **穿過 sync 寫進決策面**（sync 才是寫進決策面那一步；harness 的 rc=6 只擋經由 harness 的
    # 那條路）。這個 fixture 逐格要求「三個消費端 == 判準本身」——任何一端自己長出第二份判準，
    # 這裡就紅。它也釘住一條語意：產物裡的 `spec_draft_gate` 是 harness 依同一判準**算出來後
    # 寫下**的記錄（output），三個消費端都不得拿它當 input，否則「舊記錄較嚴」的臂永遠翻不了身。
    scbad = 0
    scases = [("OK", {"state": "OK"}, False),
              ("NOT_SPEC", {"state": "NOT_SPEC"}, False),
              ("INVALID", {"state": "INVALID"}, True),
              ("NO_STATE（缺 state 值）", {}, True),
              ("打錯字（Dead?）", {"state": "Dead?"}, True),
              ("state 空字串", {"state": ""}, True),
              ("缺 spec_draft 欄（舊產物 ⇒ 不回溯）", None, False),
              ("舊記錄較嚴（state=OK、gate.ok=False）", {"state": "OK"}, False)]
    import harness as _H
    import experiment_sync as _ES
    print("\nspec-draft 死亡判準：三個消費端必須與 spec_draft_dead 逐格相同")
    for name, sd, must in scases:
        arm = {"tag": "fixture", "rows": [{"n_gen": 128, "avg_ts": 9.9}]}
        if sd is not None:
            arm["spec_draft"] = dict(sd, why="fixture")
            arm["spec_draft_gate"] = {"ok": False}   # 記錄（output）：三個消費端都不得讀它
        m = spec_draft_dead(arm)
        h = not _H._spec_draft_gate([arm])["ok"]
        with tempfile.TemporaryDirectory() as td:
            fp = os.path.join(td, "fixture.json")
            with open(fp, "w", encoding="utf-8") as f:
                json.dump([arm], f)
            # 成對閘（code 5）排在 draft 閘（code 6）之前 ⇒ 要走到 draft 閘就得給一份 sibling log
            with open(os.path.join(td, "fixture.stderr.log"), "w", encoding="utf-8") as f:
                f.write("llama-bench: [spec] fixture\n")
            g = _ES.sync_gate(fp)
        s = g.get("code") == 6
        ob_ok, ob_why = _ES._arm_on_board(arm, None)
        ob = (not ob_ok) and ("spec draft" in str(ob_why))    # 只認「因這條判準被擋」
        ok = (m == h == s == ob == must)
        if not ok:
            scbad += 1
        print("  %s [%s]: matrix=%s harness=%s sync_gate(code=%s)=%s on_board=%s（want %s）"
              % ("ok  " if ok else "FAIL", name, m, h, g.get("code"), s, ob, must))
    print(f"spec-draft rule consistency selftest: {len(scases) - scbad}/{len(scases)} cases passed")
    return 0 if bad + ebad + sbad + abad + dbad + gbad + pbad + sbad2 + bgbad + scbad == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="prod25-stream",
                    help=f"comma list from {sorted(ARMS)} (or PROFILE:ENV=VAL;ENV=VAL)")
    ap.add_argument("--depths", default="0",
                    help="llama-bench -d list: context already filled before the measurement")
    ap.add_argument("--prompt", default="512", help="llama-bench -p (pp); comma list allowed")
    ap.add_argument("--gen", default="128", help="llama-bench -n (tg)")
    ap.add_argument("--reps", type=int, default=5,
                    help="llama-bench -r（2026-10-03：預設 5＝llama-bench 出廠值；原本 3）")
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
    ap.add_argument("--require-instrument", default="",
                    help="額外明文要求「必須真的動過」的量具探針（逗號分隔；預設由臂的 env 推）。"
                         "例：prefetch（池的重定中心有下過單）。見 scripts/check/instrument_binding.py")
    ap.add_argument("--cell", default=None,
                    help="測試卡 §2.5 的 named cell（預設＝權威 prod-new cell）。未宣告的名字是 "
                         "fail-closed（不退回預設）——拿不到指定的 cell 不該變成另一個 cell 的數字。")
    ap.add_argument("--workdir", default="/tmp")
    ap.add_argument("--json")
    # 【CGC 2026-09-29】驅動側的剎車與看門狗（見 `_stream_child`）。三個預設都是關的 ⇒ 既有 cell
    # 的命令列與行為一個字都不變；需要它們的只有「跑不完」那一類診斷。
    ap.add_argument("--arm-timeout", type=float, default=0.0,
                    help="單臂牆鐘上限（秒）。逾時只殺**自己的 process group**（子行程用 "
                         "start_new_session 起跑），驅動留下來把 live log／產物寫完。0 = 不設（歷史行為）")
    ap.add_argument("--stall-watch", type=float, default=0.0,
                    help="停滯看門狗（秒）：連續這麼久沒有新的一行 stderr ⇒ 記下停滯點並 "
                         "`sample` 子行程堆疊（「卡住」與「很慢」只能靠堆疊分）。0 = 關")
    ap.add_argument("--stall-kill", action="store_true",
                    help="第一次抓到停滯就殺（先取堆疊再殺）；預設只記錄、讓 arm-timeout 決定收尾")
    ap.add_argument("--stall-progress-re", default=None,
                    help="「進展行」的判準（regex，逐行 search）。預設 %r：心跳（CGC-RSS）不算進展 —— "
                         "引擎有心跳，所以「有沒有新的一行」量不出卡住" % _PROGRESS_RE)
    ap.add_argument("--live-echo", default="",
                    help="額外把符合這個 regex 的 stderr 行即時印到 stdout（預設用內建的那一組）")
    ap.add_argument("--md")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true",
                    help="run the spec-armed gate's own cases (unit + end-to-end) and exit")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    # [CGC 唯一入口 2026-09-27] 頂部 docstring 的 INTERNAL DRIVER 宣告現在由運行時強制：直跑
    # （無跑前立項、口徑可能是冷 cache）產生的數字不可引用。harness/commit_bench 內部調用會
    # 自帶 CGC_INTERNAL_CALL=1；--dry-run 不產數字、放行。
    if not args.dry_run and os.environ.get("CGC_INTERNAL_CALL") != "1":
        print(
            "!! 這支是 internal 驅動，不是生產級量測入口（見檔頭 2026-09-25 ruling）。\n"
            "   直跑沒有跑前立項、warm-skip/fixed-fill-seed 也可能不是熱-cache 口徑，數字不可引用。\n"
            "   請走：python3 scripts/check/harness.py bench --arm \"prod-new\" --charter <charter.yaml>\n"
            "   （內部自動化會自帶 CGC_INTERNAL_CALL=1；要復刻請顯式 export 後再跑。）",
            file=sys.stderr)
        return 2
    # 舊語意「給 -b 就等於 -b/-ub 同值」**只在格沒有宣告 ubatch 時**才成立。
    # WHY (2026-10-04): 格是形狀的權威來源（`check_cell` 逐字比對 `ubatch`），而 `harness bench`
    # 這條路只送 `--batch`（它沒有 `--ubatch` 參數）⇒ 這一行會把 ub 設成 batch，蓋掉格的 512 ⇒
    # 契約以「ubatch: 實際 2048 ≠ 權威 512」拒跑。**格宣告了就以格為準**。
    if args.batch and not args.ubatch and not cell_declared_batch(getattr(args, "cell", None))[1]:
        args.ubatch = args.batch

    armed: list[tuple[str, str, dict[str, str], dict]] = []
    for spec in args.arms.split(","):
        spec = spec.strip()
        if not spec:
            continue
        if ":" in spec:
            prof, _, envs = spec.partition(":")
            extra = parse_arm_env(envs)
        elif spec in ARMS:
            prof, extra = ARMS[spec]
        else:
            raise SystemExit(f"unknown arm {spec!r}; known: {sorted(ARMS)}")
        res = resolve(prof, extra)
        # [CGC fix 2026-09-28] 靜默丟失防線。arm 宣告的 env 若既沒出現在 resolved surface、
        # 也沒讓它產生任何變化，就是被 `!` 前綴或 run_server.sh 的 allowlist 吃掉了 —— 那一臂
        # 量的是 profile 預設值，而 gate 照樣 PASS（2026-09-28 的 MTP 修復臂就是這樣空轉了一趟）。
        dropped = arm_env_dropped(prof, extra, res)
        if dropped:
            msg = (f"arm {spec!r} 的環境變數沒有送達引擎（量到的會是 profile 預設值）: "
                   f"{', '.join(dropped)}\n"
                   f"  常見原因：(a) `!KEY=VAL` 的 `!` 沒剝（run_server.sh 的 allowlist 不認 `!KEY`）；\n"
                   f"           (b) 用了底層名而非 run_server.sh 的入口名（例：要給 CGC_SERVER_LAYER_CAPS，\n"
                   f"               不是 LLAMA_EXPERT_CACHE_LAYER_CAPS —— 後者由前者在 :2470 導出）。\n"
                   f"  確認入口名：grep -n 'SERVER_ENV+=(' scripts/run_server.sh")
            if os.environ.get("CGC_ARM_ENV_STRICT", "1") == "0":
                print("!! " + msg, file=sys.stderr)
            else:
                raise SystemExit(msg)
        why = mtp_without_spec(extra, res["env"], args.spec_type)
        if why:
            raise SystemExit(
                f"arm {spec} arms MTP but this cell has no --spec-type ({why}) — 拒跑（fail-closed）。\n"
                f"  CGC_SERVER_MTP only exports the MTP=1 env block; the spec itself is a llama-bench\n"
                f"  flag that nothing here adds for you. Without it the arm measures NO_PREFETCH /\n"
                f"  LAYER_CAPS / PREFIX_REUSE_CKPT / NO_SEQ_RM_PROBE, not MTP. Add:\n"
                f"      --spec-type draft-mtp [--spec-draft-n-max 3]")
        armed.append((spec, prof, extra, res))

    # [CGC 2026-10-01, MEASUREMENT_CONTRACT §3.4.1] launch 前超訂預檢：strict 超訂 ⇒ 拒跑（rc=2、
    # 零樣本）、warn ⇒ 放行且把 CGC_BUDGET_OVERSUBSCRIBED=1 帶進每一支臂的 env（下游產物帶著
    # 污染標記）。直跑與 harness 都走同一條路；--dry-run 不產數字，不叫閘門。
    if not args.dry_run and armed:
        gate_env = dict(os.environ)
        gate_env.update(armed[0][3].get("env") or {})
        gate_rc, gate_log = budget_gate_preflight(gate_env)
        if gate_log.strip():
            sys.stderr.write(gate_log if gate_log.endswith("\n") else gate_log + "\n")
        if gate_rc != 0:
            print(f"⛔ budget gate 拒跑（rc={gate_rc}；MEASUREMENT_CONTRACT §3.4.1）—— 本趟沒有樣本。",
                  file=sys.stderr)
            print("   要硬跑：BUDGET_GATE=warn（樣本帶 CGC_BUDGET_OVERSUBSCRIBED=1 標記）或 BUDGET_GATE=off。",
                  file=sys.stderr)
            return 2
        if gate_env.get("CGC_BUDGET_OVERSUBSCRIBED") == "1":
            for _, _, _, _r in armed:
                _r["env"]["CGC_BUDGET_OVERSUBSCRIBED"] = "1"

    results = [run_arm(tag, prof, extra, args, res) for tag, prof, extra, res in armed]

    report(results, args)
    if args.json:
        # 先寫 log、後寫產物（成對是寫入端的責任，不是「記得順手 cp 一下」）。
        try:
            canon = ibind().write_pair_bundle(args.json, _PAIR_BUFFER)
            print(f"log pair -> {canon}（{len(_PAIR_BUFFER)} 臂）")
        except Exception as e:  # noqa: BLE001
            print(f"!! 成對 log 寫不進去：{e} — 產物照寫，但那一場會被判**不可引用**", flush=True)
        Path(args.json).write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"json -> {args.json}")
    bad = [r["tag"] for r in results if r.get("incomplete")]
    if bad:
        print(f"\nINCOMPLETE arm(s): {' '.join(bad)} -- see the [INCOMPLETE] markers above; "
              f"the reported rows are complete, the arm just did not measure every shape.",
              file=sys.stderr)
        return 1

    # [CGC 2026-09-28] run-internal 記憶體預算的閘門效果。在此處判（而不是只印一顆 verdict），
    # 因為一個在污染下量到的 t/s 不是「比較差的數字」，是**不是那個數字**：它的壓力是這個臂
    # 自己造成的。缺 gate 資料也算失敗（fail-closed）—— 缺量測不是通過。
    # 逃生口：CGC_IGNORE_SWAP_BUDGET=<理由>，理由會寫進產物。
    if not args.dry_run:
        over = [r for r in results if not (r.get("memory_gate") or {}).get("ok", False)]
        if over:
            print("\n⛔ MEMORY GATE 不過 — 以下臂判不可用（rc=4）：", file=sys.stderr)
            for r in over:
                g = r.get("memory_gate") or {}
                kind = "起跑狀態（跑前就拒）" if r.get("refused_preflight") else "跑後殘差"
                if not g:
                    print(f"   {r['tag']}: （沒有 memory_gate 資料）", file=sys.stderr)
                for why in (g.get("reasons") or []):
                    print(f"   {r['tag']} [{kind}]: {why}", file=sys.stderr)
            print("   完整曲線已留在產物裡：memory.samples（原始逐取樣）／memory_gate.curve",
                  file=sys.stderr)
            print("   逃生口：CGC_IGNORE_STATE_BUDGET=<理由>（起跑）／CGC_IGNORE_SWAP_BUDGET=<理由>"
                  "（跑後），兩者都把理由留在產物裡。", file=sys.stderr)
            return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
