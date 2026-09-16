#!/usr/bin/env bash
# [CGC 2026-09-16] Re-sweep CGC_SPAC_ALPHA at the unified profile's ctx (8192).
#
# WHY THIS IS NOT ALREADY ANSWERED
#   alpha=0.75 was tuned on 2026-09-06 by an 11-point sweep at **ctx 4096, 8 GiB pool, DBUF+SPAC**
#   (see the comment on cgc_spac_alpha in llama-expert-cache.h: alpha 0.75 gave best median 25.87
#   t/s and 2/3 runs above 25). But alpha is the EMA decay of the *per-layer expert utility*, and
#   the thing that fills that EMA is the routed expert set per step -- a quantity that a different
#   context length can change (more resident history -> different router statistics per step, and
#   the depth prefill that warms the pool is a different size). So "0.75 is still right at 8192"
#   is an assumption until measured here.
#
# SHAPE: `-p 0 -n 128 -r 3 -b 512` at depths 512 and 1024. d512 is the depth where the SPAC effect
#   was decisive (16% mean, dispersion collapse); d1024 is included because it is the deepest cell
#   this box can hold at ctx 8192 and because the 09-15 series had its best number there.
#   A `CGC_SPAC=0` arm rides along so the SPAC-on/off gap is re-expressed at ctx 8192 rather than
#   borrowed from the ctx 4096 measurement.
#
# WAIT: level 0 must HOLD for QUIET_SEC (default 120 s) -- see run_spac_prefill_ab.sh for the
#   measurement that made "read 0 once" untenable. If the hold never happens the arm is SKIPPED.
#
# Usage:  bash Backup/run_spac_alpha_sweep.sh
#         QUIET_SEC=180 bash Backup/run_spac_alpha_sweep.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 2
QUIET_SEC="${QUIET_SEC:-120}"
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

run() {  # run <name> <arm>
    local name="$1" arm="$2"
    if ! wait_quiet; then
        echo "  !! alpha arm $name not run -- no rows, and none invented"
        echo
        return 0
    fi
    echo "##### alpha arm: $name   \"$arm\"   $(date '+%H:%M:%S')"
    RUN_REPLAY_BENCH=0 python3 scripts/check/llama_bench_matrix.py \
        --arms "$arm" --prompt 0 --gen 128 --depths 512,1024 --reps 3 \
        --batch 512 --ubatch 512 \
        --json "$OUT/unified_alpha_$name.json" 2>&1 \
        | grep -E '^  tg |thermal: launch|json ->|error|OOM|failed' || echo "  !! run returned non-zero"
    echo
}

echo "=== SPAC alpha re-sweep at ctx 8192 (unified profile)  $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "  quiet=${QUIET_SEC}s"
echo
run spacoff 'prefill250:CGC_SPAC=0'
run a050    'prefill250:CGC_SPAC_ALPHA=0.5'
run a075    'prefill250:CGC_SPAC_ALPHA=0.75'
run a090    'prefill250:CGC_SPAC_ALPHA=0.9'

echo "=== all rows ==="
python3 - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = []
for f in sorted(out.glob("unified_alpha_*.json")):
    try:
        d = json.load(open(f))
    except Exception as e:
        print(f"  unreadable {f.name}: {e}")
        continue
    name = f.stem.replace("unified_alpha_", "")
    for arm in d if isinstance(d, list) else []:
        for r in arm.get("rows", []):
            if r.get("n_prompt", 1) == 0:
                th = (arm.get("thermal") or {}).get("launch")
                rows.append((name, r["n_depth"], r["avg_ts"], r["stddev_ts"], th))
print(f"{'arm':9s} {'depth':>6s} {'avg_ts':>8s} {'stddev':>7s}  launch")
for name, d, avg, sd, th in rows:
    print(f"{name:9s} {d:>6} {avg:>8.2f} {sd:>7.2f}  {th}")
print()
base = {}
for name, d, avg, sd, th in rows:
    if name == "spacoff":
        base[d] = avg
for name in ("a050", "a075", "a090"):
    for d in (512, 1024):
        v = [r[2] for r in rows if r[0] == name and r[1] == d]
        if v and d in base:
            print(f"  {name} vs SPAC-off at d={d}: {v[0]:6.2f} vs {base[d]:6.2f}"
                  f"  = {100*(v[0]-base[d])/base[d]:+5.1f}%")
print("=== done $(date '+%H:%M:%S') ===")
PY
