#!/usr/bin/env bash
# [CGC 2026-09-16] (c) The two throughput phases, chained, with a SENTINEL per phase.
#
# WHY NOT `phase_b && phase_c`
#   On 2026-09-16 the earlier chain used `&&` between phases and a bash bug in phase one
#   (`run(): $3: unbound variable` under `set -u`) killed the whole chain. The log then contained one
#   bash error line and nothing else, and "no artifact" looked exactly like "ran and produced
#   nothing". So: no `&&`, an explicit rc echoed per phase, and a sentinel at the end of each.
#
# WHAT IT RUNS (both are mine, both wait for the thermal gate themselves)
#   1. `run_unified_depth_matrix.sh` -- depths 0,512,1024,2048 ascending AND descending in one
#      process, so a depth effect can be separated from an order effect. SOLO=0 (the per-depth solo
#      processes are a third pass that the asc/desc pair usually makes unnecessary).
#   2. `run_spac_alpha_sweep.sh` -- alpha 0.5/0.75/0.9 plus a CGC_SPAC=0 control, at depths 512 and
#      1024, on the unified profile's ctx 8192. alpha 0.75 was tuned at ctx 4096; this re-expresses
#      the SPAC on/off gap at the ctx the unified profile actually uses.
#
# Both drivers SKIP an arm rather than run it hot, and record the launch level per row. So an arm
# missing from the output means "the box never went quiet for QUIET_SEC", not "the arm was fine".
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 2

echo "=== (c) chain start $(date '+%F %T') ==="
echo "  thermal        : $(python3 scripts/check/thermal_pressure.py)"
echo "  port 8080      : $(lsof -nP -iTCP:8080 -sTCP:LISTEN 2>/dev/null | tail -1 || echo free)"
echo "  strays         : $(pgrep -fl 'llama-server|llama-bench' 2>/dev/null || echo none)"
echo

echo "##### PHASE C1: depth matrix (asc + desc, one process) $(date '+%H:%M:%S')"
SOLO=0 bash Backup/run_unified_depth_matrix.sh
rc1=$?
echo "##### PHASE C1 rc=$rc1  (sentinel: rows are in Backup/phase_decomp/unified_depth_*.json)"
echo

echo "##### PHASE C2: SPAC alpha re-sweep at ctx 8192 $(date '+%H:%M:%S')"
bash Backup/run_spac_alpha_sweep.sh
rc2=$?
echo "##### PHASE C2 rc=$rc2  (sentinel: rows are in Backup/phase_decomp/unified_alpha_*.json)"
echo

echo "=== (c) chain done $(date '+%F %T')  rc1=$rc1 rc2=$rc2 ==="
echo "  depth artifacts: $(ls -1 "$ROOT"/Backup/phase_decomp/unified_depth_*.json 2>/dev/null | wc -l | tr -d ' ')"
echo "  alpha artifacts: $(ls -1 "$ROOT"/Backup/phase_decomp/unified_alpha_*.json 2>/dev/null | wc -l | tr -d ' ')"
echo "  thermal at end : $(python3 scripts/check/thermal_pressure.py)"
