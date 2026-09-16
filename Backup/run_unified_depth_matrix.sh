#!/usr/bin/env bash
# [CGC 2026-09-16] The full depth matrix for the UNIFIED profile (prefill250 + CGC_SPAC=1), ctx 8192.
#
# WHAT IS BEING MEASURED, AND WHAT IS NOT
#   `-d N` makes llama-bench fill N tokens of context before `t_start`, so the number reported is a
#   decode measured with the pool in whatever state those N tokens left it. It is the one axis that
#   moved "the same instrument, same arm" from 9.5 to 13.2 t/s on 09-15, i.e. it is not a detail.
#
#   The historical protocol ran all depths inside ONE process, in ascending order. That confounds
#   the depth effect with the position in the sequence (the pool is process-level, so depth k runs
#   against a pool warmed by depths 0..k-1). This driver runs BOTH directions, and additionally
#   each depth in its OWN process. Comparing them separates the two:
#       asc vs desc differ  -> order/accumulation dominates
#       asc == desc         -> the depth axis is real
#       per-depth != asc    -> the intra-process warming is real, and the 09-15 series carries it
#
# SHAPE
#   `-p 0 -n 128 -r 3 -b 512 -ub 512`, prompt depth only. `-b 512` is a MEASUREMENT parameter, not
#   a profile difference: the profile's own 5632 x 512-token depth prefill OOMs on this 16 GB box
#   (proved: Backup/llama_bench/llama_bench_prefill250.json is a zero-row wreck that died in
#   `test_prompt ... res = -3`).
#
# WAIT: level 0 must HOLD for QUIET_SEC (default 120 s), not just be read once. The prefill A/B on
#   this same evening showed a single 0-reading is not sufficient: two arms launched 40 s after the
#   previous arm came in 5-16% low while two launched after >=105 s came in at the top of the band.
#   If the hold never happens the arm is SKIPPED (fail closed) rather than run and labelled.
#
# Usage:  bash Backup/run_unified_depth_matrix.sh
#         QUIET_SEC=180 bash Backup/run_unified_depth_matrix.sh
#         SOLO=0 bash Backup/run_unified_depth_matrix.sh    # skip the per-depth processes
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 2
QUIET_SEC="${QUIET_SEC:-120}"
SOLO="${SOLO:-1}"
OUT="$ROOT/Backup/phase_decomp"
mkdir -p "$OUT"

level() { notifyutil -g com.apple.system.thermalpressurelevel 2>/dev/null | awk '{print $NF}'; }

wait_quiet() {
    local i lv t0 need=$(( QUIET_SEC / 5 )) streak=0
    t0=$(date +%s)
    for i in $(seq 1 240); do
        lv="$(level)"
        if [ "$lv" = "0" ]; then
            streak=$(( streak + 1 ))
            if [ "$streak" -ge "$need" ]; then
                echo "  [quiet] level 0 held ${QUIET_SEC}s continuously (total wait $(( $(date +%s) - t0 ))s)"
                return 0
            fi
        else
            streak=0
        fi
        [ $(( i % 6 )) -eq 1 ] && echo "  [quiet] $(( $(date +%s) - t0 ))s: level=$lv, held0=${streak}/${need} ..."
        sleep 5
    done
    echo "  !! level 0 never held ${QUIET_SEC}s within 1200 s -- SKIPPING (fail closed)"
    return 1
}

run() {  # run <json-name> <depths> [label]
    local name="$1" depths="$2" label="${3:-$2}"
    if ! wait_quiet; then
        echo "  !! $name not run -- no rows, and none invented"
        echo
        return 0
    fi
    echo "##### $label  depths=$depths  $(date '+%H:%M:%S')"
    RUN_REPLAY_BENCH=0 python3 scripts/check/llama_bench_matrix.py \
        --arms prefill250 --prompt 0 --gen 128 --depths "$depths" --reps 3 \
        --batch 512 --ubatch 512 \
        --json "$OUT/unified_depth_$name.json" 2>&1 \
        | grep -E '^  tg |thermal: launch|json ->|error|OOM|failed' || echo "  !! run returned non-zero"
    echo
}

echo "=== unified depth matrix (prefill250 + SPAC=1, ctx 8192)  $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "  quiet=${QUIET_SEC}s   solo=${SOLO}"
echo
echo "--- form 1/3: one process, ascending (history-comparable) ---"
run asc "0,512,1024,2048" "one-process ASC"
echo "--- form 2/3: one process, descending (asc vs desc = order effect) ---"
run desc "2048,1024,512,0" "one-process DESC"
if [ "$SOLO" = "1" ]; then
    echo "--- form 3/3: one process per depth (depth without intra-process warming) ---"
    for d in 0 512 1024 2048; do
        run "solo$d" "$d" "solo d=$d"
    done
fi

echo "=== all rows ==="
python3 - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = []
for f in sorted(out.glob("unified_depth_*.json")):
    try:
        d = json.load(open(f))
    except Exception as e:
        print(f"  unreadable {f.name}: {e}")
        continue
    for arm in d if isinstance(d, list) else []:
        for r in arm.get("rows", []):
            if r.get("n_prompt", 1) == 0:
                th = (arm.get("thermal") or {}).get("launch")
                rows.append((f.stem.replace("unified_depth_", ""), r["n_depth"], r["avg_ts"],
                             r["stddev_ts"], th))
print(f"{'form':10s} {'depth':>6s} {'avg_ts':>8s} {'stddev':>7s}  launch")
for name, d, avg, sd, th in rows:
    print(f"{name:10s} {d:>6} {avg:>8.2f} {sd:>7.2f}  {th}")
print()
print("--- order effect (asc vs desc, one process each) ---")
for d in sorted({r[1] for r in rows}):
    a = [r[2] for r in rows if r[0] == "asc" and r[1] == d]
    b = [r[2] for r in rows if r[0] == "desc" and r[1] == d]
    s = [r[2] for r in rows if r[0] == f"solo{d}"]
    if a and b:
        print(f"  d={d:<5} asc={a[0]:6.2f}  desc={b[0]:6.2f}  order delta={a[0]-b[0]:+5.2f}"
              f"   solo={('%.2f' % s[0]) if s else '-'}")
print("=== done $(date '+%H:%M:%S') ===")
PY
