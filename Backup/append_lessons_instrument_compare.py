#!/usr/bin/env python3
"""Append the instrument-comparison lessons (2026-09-16, llama-bench vs decode_bench).

Four lessons, all from one interleaved run with the SAME env on both instruments
(`Backup/run_instrument_compare.sh`, evidence in `Backup/phase_decomp/lb_*_20260916_1807.json`
and `db_*_20260916_1807.json`). Two of the three predictions written before the run died; what
survived is the part quoted below.
"""
import json
from pathlib import Path

P = Path("agent_harness/engine_loop/traces/lessons.jsonl")

LESSONS = [
    {
        "type": "lesson",
        "lesson_id": "eng-mh-0039",
        "class": "measurement-hygiene",
        "rule": ("Never put a llama-bench tg number next to a decode_bench number without saying "
                 "which rep/round the number averages over. llama-bench's `avg_ts` averages the "
                 "COLD first rep in; `decode_bench --warmup 1` discards it. At n=128 that single "
                 "rep is the whole apparent gap: reported 9.38-9.42 vs plateau 11.3 t/s, against "
                 "decode_bench's warm 12.36 t/s (1.10x, not 1.32x)."),
        "because": (
            "Same env (prod25 + CGC_SERVER_MTP=0;CGC_GPU_TIMING=1;CGC_DECODE_PROFILE=1, resolved "
            "through `run_server.sh CGC_DUMP_ENV=1` for BOTH), same model (the MTP=0 path loads "
            "Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf, not the denseIQ4X carrier), same n, interleaved in "
            "one session, thermal level recorded on every arm: llama-bench read 7.21 (n=24) / "
            "9.42, 9.32, 9.38 (n=128, three runs 1.1% apart) where decode_bench read 18.74 (n=24) "
            "/ 12.36 (n=128). Reading the per-rep DECPROF series instead of the average shows rep 1 "
            "at 120 ms/token and rep 2-3 at 86-93 ms, i.e. `avg_ts` = 9.4 while the engine's own "
            "steady state is 87-89 ms = 11.3 t/s. The cause is visible in the source: llama-bench's "
            "tg warmup is `test_gen(ctx, 1, ...)` -- ONE token -- and the context (hence the expert "
            "pool) is created per instance, so the pool starts empty and the first rep pays the "
            "compulsory-fill cost (its own miss attribution: compulsory 4437/5195 = 85.4%)."),
        "counterexample_observed": (
            "Backup/phase_decomp/lb_n128_match_20260916_1807.json (9.42) + lb_n128_match2 (9.32) + "
            "p2-phase anchor (9.38) vs db_n128_match_20260916_1807.json (12.36); per-rep DECPROF "
            "means n=128 r=3 -> r1=120 r2=86 r3=93 ms, n=256 r=3 -> r1=116 r2=87 r3=87 ms"),
        "applies_to": ["scripts/check/llama_bench_matrix.py", "scripts/check/decode_bench.py"],
        "superseded_by": None,
    },
    {
        "type": "lesson",
        "lesson_id": "eng-mh-0040",
        "class": "measurement-hygiene",
        "rule": ("Do not use llama-bench as the instrument of record for this engine's decode "
                 "target. It has no sampler, so it cannot execute the speculative/MTP path at all, "
                 "and it advances the sequence with `std::rand() % n_vocab`, so it does not measure "
                 "the token stream a served request produces."),
        "because": (
            "`grep -c 'sampler|speculat|draft|MTP' tools/llama-bench/llama-bench.cpp` is 0. "
            "`test_gen` is `llama_decode(1 token); llama_synchronize(ctx); token = rand()%n_vocab;` "
            "(llama-bench.cpp:2168-2183) -- no sampler, no draft, no verify, no accept. Every "
            "remaining lever for the 25 t/s decode target is MTP, and MTP needs a sampler to decide "
            "acceptance, so llama-bench can only ever report the MTP *model* running without "
            "speculation (measured: 8.75 +- 0.95, i.e. no benefit visible). The random stream is not "
            "a cosmetic detail either: it changes what the expert pool is asked to do -- random ids "
            "gave 96.0% hit with 85.4% COMPULSORY misses and only 7 of 40 layers over-subscribed, "
            "where the same engine serving coherent text gave 92.6% with 51.0% compulsory / 49.0% "
            "capacity and ALL 40 layers over-subscribed, reading 3.35x the bytes (18.35 GiB vs "
            "5.48 GiB). A stream whose expert union is unpredictable is also the stream a prefetcher "
            "cannot help, which is a plausible reason llama-bench's higher hit rate coexists with "
            "its lower t/s."),
        "counterexample_observed": (
            "BACKUP phase_decomp/lb_n128_mtp_20260916_1807.json (8.75 +- 0.95, prod25 profile, "
            "MTP=1 -> denseIQ4X + bit-identical pillars, thermal launch HEAVY) vs the MTP=0 arms at "
            "9.32-9.42; and the two miss-attribution lines: llama-bench 'compulsory=4437 "
            "capacity=758 ... layers_distinct_over_slots=7' vs server 'compulsory=8874 "
            "capacity=8522 ... layers_distinct_over_slots=40'"),
        "applies_to": ["scripts/check/llama_bench_matrix.py"],
        "superseded_by": None,
    },
    {
        "type": "lesson",
        "lesson_id": "eng-mh-0041",
        "class": "measurement-hygiene",
        "rule": ("Never compare the expert cache's hit%/misses/reads between two harnesses. Those "
                 "counters are process-lifetime accumulators, and the two processes accumulate over "
                 "different work: llama-bench's cover its own warmup + reps of tg only, the server's "
                 "cover model load + warmup request + prefill chunks of a chat prompt + every round. "
                 "Only ratios that are defined inside one process (miss attribution split, "
                 "over-subscribed layer count) survive the comparison."),
        "because": (
            "The same engine, same pool, same env reported hit 96.0% in the llama-bench process and "
            "92.6% in the server -- and the higher-hit side was the SLOWER one (9.4 vs 12.4 t/s). "
            "That inversion is what makes the naive reading wrong, and it is not a defect in either "
            "number: 128,920 requests over ~400 tokens in one process vs 234,840 over a different "
            "mix (including ~100 prefill tokens at n_batch=8) in the other. Comparing them is the "
            "same failure as comparing t/s across instruments, one level down."),
        "counterexample_observed": (
            "llama-bench: 'final stats: runtime requests=128920 hits=123725 misses=5195 (hit rate "
            "96.0%) file_reads=15516 ... effective_rate=309 MiB/s' vs server: 'requests=234840 "
            "hits=217444 misses=17396 (92.6%) file_reads=49320 ... effective_rate=217 MiB/s'"),
        "applies_to": ["scripts/check/decode_sweep.py", "scripts/check/llama_bench_matrix.py"],
        "superseded_by": None,
    },
    {
        "type": "lesson",
        "lesson_id": "eng-src-0013",
        "class": "source-reading",
        "rule": ("Two windows that both look like 'the generation' are not: llama-bench's tg warmup "
                 "is one token (`test_gen(ctx, 1, ...)`), and the server's `predicted_ms` starts at "
                 "the first SAMPLED token, so a request is scored as N tokens over N-1 steps."),
        "because": (
            "Both were established by reading, then confirmed by measurement. (1) warmup=1 token: "
            "predicted '--no-warmup will change nothing', measured 9.40 +- 1.19 with it off vs "
            "9.38 +- 1.02 with it on -- and the off arm ran at HEAVY where llama-bench is slower, so "
            "the null result is if anything conservative. The same read also refutes the harness's "
            "own --no-warmup help text ('warmup is what leaves the expert pool hot, so the default "
            "is the closer analogue of a served request'); one token leaves nothing hot. It is also "
            "consistent with the print accounting: 1 warmup graph + 3x128 rep graphs = 385, and the "
            "gate prints every 8th plus step 1 -> 49 lines, which is exactly what the log has. "
            "(2) `t_start_generation = t_now` is set inside `if (slot.n_decoded == 1)`, AFTER the "
            "sampler has already synchronized -- server-context.cpp:4085-4099 -- so `predicted_ms` "
            "spans N-1 decode steps while `predicted_n` = N, an N/(N-1) optimistic bias (+4.3% at "
            "n=24, +0.8% at n=124)."),
        "counterexample_observed": (
            "llama-bench.cpp:2392 `test_gen(ctx, 1, t.n_threads)` and server-context.cpp:4092 "
            "`slot.t_start_generation = t_now;`; measured 49 printed DECPROF graphs for a "
            "1+384-graph run, and BACKUP phase_decomp/lb_p2_n128_nowarm_*.json (9.40) vs "
            "lb_p0_anchor_n128_*.json (9.38)"),
        "applies_to": ["scripts/check/llama_bench_matrix.py", "scripts/check/decode_bench.py",
                       "src/llama.cpp/tools/server/server-context.cpp"],
        "superseded_by": None,
    },
]

with P.open("a", encoding="utf-8") as fh:
    for r in LESSONS:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")

rows = [json.loads(l) for l in P.read_text().splitlines() if l.strip()]
print(f"lessons now: {len(rows)}")
for r in LESSONS:
    print("  +", r["lesson_id"], r["class"])
