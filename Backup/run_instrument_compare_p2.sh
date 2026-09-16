#!/bin/bash
# run_instrument_compare_p2.sh — three prediction tests for WHY llama-bench sits below decode_bench.
#
# The hypothesis under test (from the source read, not from guessing):
#   llama-bench creates ONE context per (p,n,d) instance, runs ONE untimed warmup of the SAME
#   shape, then measures `reps` runs on that same context (llama-bench.cpp:2284-2490, 2375-2396).
#   The expert pool therefore only holds as much as the warmup + the preceding reps have put in
#   it -- and its miss attribution says compulsory misses are 87.2% of all misses. A short n is
#   thus a cold-pool measurement that never gets the chance to amortise; the HTTP server, by
#   contrast, serves round after round against one warm pool.
#
# Each test is written as a prediction that can fail:
#   P1  -n 24  -r 8   : if reps within one instance warm the pool, the 8-rep mean must EXCEED the
#                       3-rep mean (7.21). If it comes back ~7.2, the pool does not warm across
#                       reps and the cold-fill story is wrong.
#   P2  -n 128 --no-warmup : if the warmup's own 128 tokens are what make the measured reps warm,
#                       removing it must land BELOW 9.42 -- near the 7.2 end. If it stays ~9.4,
#                       warmup is irrelevant to the pool and the story dies here.
#   P3  -n 256         : if the penalty is amortisation, a longer run must land ABOVE 9.42 and
#                       move toward the HTTP instrument's per-token cost (12.36 at n=128).
#                       If it stays ~9.4, llama-bench has a length-independent floor and the
#                       remaining gap is a real instrument difference, not an amortisation one.
set -u
cd "$(dirname "$0")/.." || exit 1
export RUN_REPLAY_BENCH=0
TAG="$(date +%Y%m%d_%H%M)"
OUT="Backup/phase_decomp"
MATCH='prod25:CGC_SERVER_MTP=0;CGC_GPU_TIMING=1;CGC_DECODE_PROFILE=1'

# The ANCHOR arm is first and is not a prediction: it repeats phase 1's `-n 128 -r 3` so the three
# tests below can be read against a same-session, same-thermal anchor instead of against a number
# taken 6 minutes and one thermal excursion earlier.
for spec in "p0_anchor_n128|0|128|3|" "p1_n24_r8|0|24|8|" "p2_n128_nowarm|0|128|3|--no-warmup" "p3_n256|0|256|3|"; do
    IFS='|' read -r name prompt gen reps extra <<<"$spec"
    echo
    echo "=============================================================="
    echo "  llama-bench  $name   (gen=$gen reps=$reps ${extra:-warmup on})"
    echo "  $(date +%H:%M:%S)   thermal=$(python3 scripts/check/thermal_pressure.py)"
    echo "=============================================================="
    # shellcheck disable=SC2086
    python3 scripts/check/llama_bench_matrix.py \
        --arms "$MATCH" \
        --prompt "$prompt" --gen "$gen" --depths 0 --reps "$reps" \
        $extra \
        --workdir /tmp \
        --json "$OUT/lb_${name}_${TAG}.json"
    echo "$name exit=$?"
done
echo
echo "=== p2 done; thermal = $(python3 scripts/check/thermal_pressure.py)"
