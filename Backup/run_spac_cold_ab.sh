#!/usr/bin/env bash
# [CGC 2026-09-17] A COLD, INTERLEAVED A/B for the one question the SPAC question actually needs:
# is the prefill cost of `prefill250` + `CGC_SPAC=1` real, or was the 20:57 arm measuring the
# build and the machine instead?
#
# WHY AN INTERLEAVED PAIR AND NOT ONE ARM
#   The single COLD arm (`run_spac_cold_arm.sh`) can only say "passed" or "did not" for one build at
#   one moment, and it cannot separate the treatment from the order -- the project already measured
#   that the自变量 is the ACCUMULATED load (quiet 18 s -> 289.86 t/s but quiet 34 s -> 166.95 t/s
#   within the same sequence), i.e. which launch in the sequence it is dominates. So the arms must
#   alternate: off, on, off, on. Anything else re-measures the ramp.
#
# WHY EVERY ARM NEEDS ITS OWN QUIET WINDOW
#   The gate classifies an arm from `$OUTDIR/.last_arm_end` -- "when the previous SUSTAINED prefill
#   ended" -- and COLD-STATE needs COLD_QUIET seconds of quiet measured from THAT instant. So the
#   state file is re-seeded with the measured end of each arm before the next wait. The wait IS the
#   experiment; nothing may run during it.
#
# FAIL CLOSED
#   1. before every launch the thermal level is read and must be 0. If it is not, keep waiting up to
#      THERMAL_WAIT_MAX seconds and then ABORT (never launch a HOT arm and call it COLD).
#   2. before the sweep starts, the SPAC off/on arms are checked against `CGC_DUMP_ENV=1`, because
#      "the flag did not take" and "the flag made no difference" look identical otherwise -- the
#      profile sets CGC_SPAC=1 itself, so an off-arm that forgot to override it would silently run
#      the SAME configuration twice and report "SPAC has no cost".
#
# Usage:
#   bash Backup/run_spac_cold_ab.sh                 # 2 pairs = off,on,off,on  (~2h08m)
#   PAIRS=1 bash Backup/run_spac_cold_ab.sh         # one pair, cannot separate order from treatment
#   COLD_QUIET=1800 bash Backup/run_spac_cold_ab.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 2

PAIRS="${PAIRS:-2}"
COLD_QUIET="${COLD_QUIET:-1800}"
THERMAL_WAIT_MAX="${THERMAL_WAIT_MAX:-900}"
OUTDIR="${OUTDIR:-Backup/cgc_logs/spac_cold_ab}"
STATE="$OUTDIR/.last_arm_end"
LOG="$OUTDIR/sweep.log"
mkdir -p "$OUTDIR"

thermal_now() { notifyutil -g com.apple.system.thermalpressurelevel 2>/dev/null | awk '{print $NF}'; }

# ---- pre-flight: the two arms must really be two configurations -----------------------------
echo "=== pre-flight: SPAC off vs on must differ in the SERVER env ===" | tee -a "$LOG"
off_lines=$(CGC_SERVER_PROFILE=prefill250 CGC_SPAC=0 CGC_DUMP_ENV=1 bash scripts/run_server.sh 2>/dev/null | /usr/bin/grep -c 'CGC_SPAC' || true)
on_lines=$( CGC_SERVER_PROFILE=prefill250           CGC_DUMP_ENV=1 bash scripts/run_server.sh 2>/dev/null | /usr/bin/grep -c 'CGC_SPAC' || true)
echo "  CGC_SPAC lines in dump:  off=$off_lines (want 0)   on=$on_lines (want 2)" | tee -a "$LOG"
if [ "$off_lines" != "0" ] || [ "$on_lines" != "2" ]; then
    echo "  ABORT: the off arm would not be off (or the on arm would not be on)." | tee -a "$LOG"
    echo "  A dropped / unlisted CGC_* is SILENT -- 'no effect' and 'never set' look the same." | tee -a "$LOG"
    exit 2
fi

if pgrep -f 'llama-server' >/dev/null 2>&1; then
    echo "  note: a llama-server is running; its end will be used as the quiet boundary." | tee -a "$LOG"
fi

last_end="$(date +%s)"
echo | tee -a "$LOG"
echo "=== COLD interleaved A/B  pairs=$PAIRS  quiet=${COLD_QUIET}s  $(date '+%F %T') ===" | tee -a "$LOG"
echo "  estimated wall time: $(( PAIRS * 2 * (COLD_QUIET + 150) / 60 )) min" | tee -a "$LOG"

n=0
for p in $(seq 1 "$PAIRS"); do
  for arm in off on; do
    n=$((n + 1))
    val="0"; label="SPAC=off"
    [ "$arm" = "on" ] && { val=""; label="SPAC=on"; }

    printf '%s\n' "$last_end" > "$STATE"
    ready=$(( last_end + COLD_QUIET ))
    echo | tee -a "$LOG"
    echo "--- arm $n/$(( PAIRS * 2 )): $label  $(date '+%F %T') ---" | tee -a "$LOG"
    echo "    quiet boundary (state file) = $(date -r "$last_end" '+%F %T'), satisfied $(date -r "$ready" '+%F %T')" | tee -a "$LOG"

    while [ "$(date +%s)" -lt "$ready" ]; do
        left=$(( ready - $(date +%s) ))
        [ "$left" -lt 0 ] && left=0
        echo "    [quiet] ${left}s left  thermal=$(thermal_now)" | tee -a "$LOG"
        sleep $(( left > 60 ? 60 : (left > 0 ? left : 1) ))
    done
    echo "    quiet satisfied." | tee -a "$LOG"

    # fail closed on the thermal level, which is the actual independent variable
    waited=0
    while [ "$(thermal_now)" != "0" ]; do
        if [ "$waited" -ge "$THERMAL_WAIT_MAX" ]; then
            echo "    ABORT: thermal=$(thermal_now) after ${waited}s of extra waiting." | tee -a "$LOG"
            echo "    A HOT arm is not a COLD arm; refusing to launch one and label it COLD." | tee -a "$LOG"
            exit 3
        fi
        echo "    [thermal] level=$(thermal_now), waiting (${waited}s of ${THERMAL_WAIT_MAX}s)" | tee -a "$LOG"
        sleep 60
        waited=$((waited + 60))
    done
    echo "    thermal=0 -> launching. Its own '# [thermal label]' line is the claim; read THAT." | tee -a "$LOG"

    if [ "$arm" = "off" ]; then
        CGC_SPAC=0 OUTDIR="$OUTDIR" CGC_ARM_STATE="$STATE" bash Backup/run_req2_retest.sh 2>&1 | tee -a "$LOG"
    else
        OUTDIR="$OUTDIR" CGC_ARM_STATE="$STATE" bash Backup/run_req2_retest.sh 2>&1 | tee -a "$LOG"
    fi
    rc=$?
    last_end="$(date +%s)"
    echo "    arm $n done rc=$rc at $(date '+%F %T')" | tee -a "$LOG"

    rep="$(ls -1t "$OUTDIR"/req2retest_*.txt 2>/dev/null | head -1)"
    if [ -n "$rep" ]; then
        echo "    report: $rep" | tee -a "$LOG"
        /usr/bin/grep -E '^#|req[123]|thermal|RESULT|mode\]' "$rep" 2>/dev/null | head -20 | tee -a "$LOG"
    else
        echo "    !! no report produced -- treat this arm as MISSING, not as a result" | tee -a "$LOG"
    fi
  done
done

echo | tee -a "$LOG"
echo "=== sweep done $(date '+%F %T'); reports in $OUTDIR ===" | tee -a "$LOG"
echo "    READ THE ORDER: a COLD A/B/A/B is only interleaved if the four arms alternate." | tee -a "$LOG"
