#!/bin/bash
# run_unified_ab.sh — is `prefill250 + CGC_SPAC=1` a viable single profile for BOTH prefill and decode?
#
# WHY THIS A/B. Three arms exist and they differ in only 6 env items (see
# Backup/cgc_logs/instr_compare/arms_matrix_20260916.tsv). prefill250 is the only one that delivers
# the prefill target (250+ t/s) and it is missing exactly ONE decode-relevant knob: CGC_SPAC, which
# the global default leaves OFF (run_server.sh:1635-1637) and which prod25's own comment calls the
# membership-driven placement that keeps the decode working set resident. So the whole question of
# "can one profile serve both" reduces to this one knob.
#
# WHAT WOULD FALSIFY IT. If SPAC=1 on prefill250's shape is clearly slower than SPAC off, then
# decode and prefill genuinely conflict and the unified answer becomes prod25-stream instead.
#
# MEASUREMENT SHAPE, DELIBERATELY NOT THE PROFILE'S. `--batch 512 --ubatch 512` overrides
# prefill250's 5632: llama-bench's depth instances call `test_prompt(ctx, n_depth, n_batch)`, and a
# 5632-wide compute buffer OOMs on this 16 GB box (measured: Backup/llama_bench/llama_bench_prefill250.json
# is a ZERO-ROW artifact that died with `test_prompt: failed to decode prompt batch, res = -3` at
# -b 6144). 512 is the batch the historical depth matrix used. This is a measurement-shape choice,
# not a profile difference -- the harness has --batch/--ubatch for exactly this.
#
# Every arm waits for the thermal level to return to NOMINAL first. That rule was broken twice
# today and both times it cost a cell (see .workbuddy/memory/2026-09-16.md EN-2).
set -u
cd "$(dirname "$0")/.." || exit 1
export RUN_REPLAY_BENCH=0
TAG="$(date +%Y%m%d_%H%M)"
OUT="Backup/phase_decomp"
# [CGC 2026-09-16 21:45] A used to be bare `prefill250`, which WAS SPAC-off when this file was
# written (08:48). Since 20:33 the profile itself sets CGC_SPAC=1 (`run_server.sh` prefill250 block:
# `[ -z "${CGC_SPAC+x}" ] && CGC_SPAC=1`), so a bare `prefill250` arm is now IDENTICAL to B --
# an A/B of a knob against itself, which reads exactly like "the knob has no effect".
# The off arm must say so explicitly. (`run_server.sh:1676` only forwards CGC_SPAC when non-zero,
# so the literal 0 both disables SpAc and omits CGC_SPAC_ALPHA.)
A='prefill250:CGC_SPAC=0'
B='prefill250:CGC_SPAC=1;CGC_SPAC_ALPHA=0.75'

wait_nominal() {  # patient: up to 10 min, and VISIBLE (an invisible wait looks like a hang)
    local i lv t0
    t0=$(date +%s)
    for i in $(seq 1 120); do
        lv="$(notifyutil -g com.apple.system.thermalpressurelevel | awk '{print $2}')"
        if [ "$lv" = "0" ]; then
            echo "  [wait] NOMINAL after $(( $(date +%s) - t0 ))s"
            return 0
        fi
        [ $(( i % 6 )) -eq 1 ] && echo "  [wait] $(( $(date +%s) - t0 ))s: level=$lv (want 0) ..."
        sleep 5
    done
    echo "  !! still not NOMINAL after 600 s -- recording the level in the row instead of waiting"
    return 1
}

run() { # run <label> <arms-spec>
    local label="$1" spec="$2"
    wait_nominal
    echo
    echo "=============================================================="
    echo "  $label"
    echo "  launch thermal: $(python3 scripts/check/thermal_pressure.py)"
    echo "=============================================================="
    python3 scripts/check/llama_bench_matrix.py \
        --arms "$spec" --batch 512 --ubatch 512 \
        --prompt 0 --gen 128 --depths 0,512 --reps 3 \
        --json "$OUT/unified_ab_${label}_${TAG}.json"
}

pkill -9 -f "build/bin/llama-server" 2>/dev/null
run A1 "$A"
run B1 "$B"
run A2 "$A"
run B2 "$B"
echo
echo "=== done; thermal = $(python3 scripts/check/thermal_pressure.py)"
ls -la "$OUT"/unified_ab_*_"$TAG".json 2>/dev/null
