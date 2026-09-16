#!/usr/bin/env bash
# [CGC 2026-09-16] Did pinning CGC_SPAC=1 into `prefill250` cost the delivered prefill number?
#
# WHY THIS EXISTS
#   SPAC is not free on the prefill path. `cgc_spac_on()` gates two things that sit inside the
#   prefill hot loop:
#     (1) the EMA utility update, which the engine's own comment says "Fires on every path that
#         reaches here -- large prefill, MTP draft and trunk verify alike" (llama-context.cpp:4736)
#         and which takes cache->m on every (layer, step);
#     (2) spac_prefetch, whose default cadence is CGC_SPAC_REFRESH=1, i.e. a per-layer partial sort
#         of the non-resident experts on EVERY routed feed.
#   So "prefill 250 still holds" is a measurement, not an assumption -- this is the measurement.
#
# DESIGN
#   Interleaved A/B/A/B, both arms POSITIVE (the gate's arm 2 is a hot-state control, which is not
#   what we want here). `CGC_SPAC=0` in the environment is picked up by run_server.sh's profile block
#   via its `[ -z "${CGC_SPAC+x}" ]` guard, so arm `off` reproduces the 09-15 GA shape without
#   editing anything -- the launcher only supplies a default.
#
# WHY THE WAIT IS "HOLD 0 FOR QUIET_SEC", NOT "READ 0 ONCE"  (added after v1 was thrown away)
#   v1 read the level once and launched on the first 0. Result, over A/B/A/B:
#       r1_off   waited   0 s -> 263.03 / 264.68 / 247.93   (mean 258.5)
#       r1_on    waited  40 s -> 232.29 / 213.70 / 203.03   (mean 216.3)
#       r2_off   waited  40 s -> 258.95 / 258.10 / 220.19   (mean 245.7)
#       r2_on    waited 105 s -> 264.87 / 273.21 / 248.66   (mean 262.2)
#   The two arms that launched 40 s after the previous arm are the two slowest, and the SPAC effect
#   has OPPOSITE SIGNS between rounds (+/-7 to 16%). So v1 did not measure SPAC; it measured how
#   long the box had been quiet -- the governor reads NOMINAL before the clocks have recovered.
#   Same trap this project already documented for prefill ("a single 0 reading at launch is
#   necessary, and demonstrably not sufficient"). Hence: level 0 must HOLD for QUIET_SEC.
#   v1's numbers are kept in the memory log as the evidence for this design change; they are not
#   used as a SPAC result.
#
# Usage:  bash Backup/run_spac_prefill_ab.sh [rounds]
#         QUIET_SEC=180 bash Backup/run_spac_prefill_ab.sh 2
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 2
ROUNDS="${1:-2}"
QUIET_SEC="${QUIET_SEC:-120}"   # level 0 must be observed continuously for this long
OUT="$ROOT/Backup/cgc_logs/spac_prefill_ab"
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
    echo "  !! level 0 never held ${QUIET_SEC}s within 1200 s -- SKIPPING the arm (fail closed)"
    return 1
}

echo "=== SPAC prefill A/B  rounds=$ROUNDS  quiet=${QUIET_SEC}s  $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "  arm off = CGC_SPAC=0 (09-15 GA shape)   arm on = profile default (CGC_SPAC=1, alpha 0.75)"
echo "  shape   = run_req2_retest.sh, i.e. the delivered ~2.9k-token prompt, 3 requests"
echo

for round in $(seq 1 "$ROUNDS"); do
    for arm in off on; do
        if [ "$arm" = "off" ]; then export CGC_SPAC=0; else unset CGC_SPAC || true; fi
        echo "##### round $round  arm=$arm  CGC_SPAC=${CGC_SPAC:-<profile default>}  $(date '+%H:%M:%S')"
        if ! wait_quiet; then
            echo "  !! arm r${round}_${arm} not run -- no data, and none invented"
            echo
            continue
        fi
        mkdir -p "$OUT/r${round}_${arm}"
        ARMS=1 OUTDIR="$OUT/r${round}_${arm}" bash "$ROOT/Backup/run_thermal_gate.sh" \
            | grep -E '^#####|launch_level|condition|pp_t/s|RESULT:|-> |pressure series|NOMINAL|MODERATE|HEAVY' \
            || echo "  !! arm returned non-zero"
        echo
    done
done

echo "=== per-arm summaries ==="
for f in $(ls -1t "$OUT"/r*/summary_*.tsv 2>/dev/null); do
    printf '%-46s ' "$(dirname "$f" | sed "s|$OUT/||")/$(basename "$f")"
    tail -n +2 "$f" | awk -F'\t' '{printf "launch=%s req1=%s req2=%s req3=%s survived=%s\n", $3,$4,$5,$6,$7}'
done
echo

# One line per arm so the A/B can be read without opening the TSVs.
printf '# %-10s %-7s %8s %8s %8s %8s\n' arm launch req1 req2 req3 mean
for f in $(ls -1t "$OUT"/r*/summary_*.tsv 2>/dev/null); do
    d="$(dirname "$f" | sed "s|$OUT/||")"
    tail -n +2 "$f" | awk -F'\t' -v d="$d" \
        '{m=($4+$5+$6)/3; printf "# %-10s %-7s %8s %8s %8s %8.1f\n", d, $3, $4, $5, $6, m}'
done
echo "=== done $(date '+%H:%M:%S') ==="
