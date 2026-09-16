#!/bin/bash
# run_instrument_compare.sh — llama-bench vs decode_bench, same env, interleaved.
#
# WHY. The roadmap's decode numbers (9.16 / 9.45 t/s) came from llama-bench; the "12.95 t/s
# sustainable" number came from decode_bench over HTTP. Putting them side by side is exactly the
# comparison this project keeps getting wrong, so this run measures BOTH, in the SAME session,
# alternating, with the SAME thermal instrument on each.
#
# Each instrument is given the identical engine configuration: the llama-bench arms carry
# `CGC_SERVER_MTP=0;CGC_GPU_TIMING=1;CGC_DECODE_PROFILE=1` on top of profile `prod25`, which is
# verbatim what decode_sweep's `p25-gputime` arm sets. Nothing is hand-copied: the env comes from
# `run_server.sh CGC_DUMP_ENV=1` in both cases.
#
# Order is ABABAB(+1): an instrument's second pass happens after the other instrument's pass, so
# accumulated load (which the prefill work showed is the real independent variable, not idle time)
# lands on both instruments rather than on whichever ran last.
#
# Deliberately NOT RUN_REPLAY_BENCH: the replay bench is stale (see PLAN E4) and adds load.
set -u
cd "$(dirname "$0")/.." || exit 1
export RUN_REPLAY_BENCH=0
TAG="$(date +%Y%m%d_%H%M)"
OUT="Backup/phase_decomp"
mkdir -p "$OUT"

MATCH='prod25:CGC_SERVER_MTP=0;CGC_GPU_TIMING=1;CGC_DECODE_PROFILE=1'
say() { echo; echo "=============================================================="; echo "  $*"; echo "  $(date +%H:%M:%S)"; echo "=============================================================="; }

lb() { # lb <tag> <gen> [extra arms-spec override]
    local tag="$1" gen="$2" spec="${3:-$MATCH}"
    say "llama-bench  $tag   (gen=$gen)"
    python3 scripts/check/llama_bench_matrix.py \
        --arms "$spec" \
        --prompt 0 --gen "$gen" --depths 0 --reps 3 \
        --workdir /tmp \
        --json "$OUT/lb_${tag}_${TAG}.json"
    echo "lb exit=$?"
}

db() { # db <tag> <n-predict> <warmup>
    local tag="$1" n="$2" warm="${3:-1}"
    pkill -9 -f "build/bin/llama-server" 2>/dev/null
    say "decode_bench $tag   (n=$n, warmup=$warm)"
    python3 scripts/check/decode_sweep.py --profile prod25 --arms p25-gputime \
        --rounds 3 --warmup "$warm" --n-predict "$n" \
        --json "$OUT/db_${tag}_${TAG}.json" --force
    echo "db exit=$?"
    pkill -9 -f "build/bin/llama-server" 2>/dev/null
}

say "START  thermal = $(python3 scripts/check/thermal_pressure.py)"
echo "  port 8080: $(lsof -nP -iTCP:8080 -sTCP:LISTEN 2>/dev/null | tail -n +2 | wc -l | tr -d ' ') listener(s)"

# --- pass 1 ---------------------------------------------------------------------------------
lb  n24_match  24
db  n24_match  24 0
lb  n128_match 128
db  n128_match 128 1
# --- pass 2 ---------------------------------------------------------------------------------
lb  n128_match2 128
db  n128_match2 128 1
# --- the production profile's own setting (MTP=1), for the axis llama-bench is meant to show --
lb  n128_mtp   128 prod25

say "DONE  thermal = $(python3 scripts/check/thermal_pressure.py)"
ls -la "$OUT"/lb_*_"$TAG".json "$OUT"/db_*_"$TAG".json 2>/dev/null
