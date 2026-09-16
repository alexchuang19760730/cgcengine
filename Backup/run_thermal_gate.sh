#!/usr/bin/env bash
# Backup/run_thermal_gate.sh -- the conditional-delivery gate for a 250-class prefill number.
#
# WHAT IT GUARANTEES
#   A prefill t/s number produced through this script is licensed to be quoted as a delivery
#   number, because the thermal condition that licenses it was READ (not assumed) at the moment of
#   launch. It is the operational form of whitepaper 11.15.
#
# THE CONDITION
#       notifyutil -g com.apple.system.thermalpressurelevel   ==   0   (Nominal)
#   read immediately before launch -- no root, no capture process, ~11 ms per read.
#
# WHY "AT LAUNCH" AND NOT "THROUGHOUT"
#   Measured 2026-09-16 15:18:46 (arm with launch level 0): the out-of-band 2 Hz probe shows the
#   level going Nominal -> Moderate at 15:19:13 -> Heavy at 15:19:24, i.e. rising PART WAY THROUGH
#   the arm, while all three requests still came in at 257.41 / 253.42 / 262.64 t/s. So a mid-arm
#   excursion does not retract the number; the governor's response lags the pressure signal, and the
#   state at launch is what the prefill inherits. This is why the gate reads at launch and why the
#   in-band per-request readings in run_req2_retest.sh are evidence, not the gate.
#
# WHY IT FAILS CLOSED
#   If the level never reaches 0 inside MAX_WAIT, the default is to exit 3 WITHOUT running an arm.
#   Producing a hot number and labelling it is how a lottery becomes a delivery. GATE_FORCE=1 runs
#   anyway and stamps the arm as a hot-state observation.
#
# WHAT IT CANNOT DO
#   It cannot tell you the machine is cool in any absolute sense -- only that the governor is
#   currently reporting Nominal pressure. The evidence that this is the right reading is the
#   separation it produced, not a physical argument: launch-level 0 -> 257.41/253.42/262.64 (all
#   >=250), launch-level 2 -> 143.89/142.14/120.74 and 104.88/150.25/127.43 (none >=250).
#
# Usage:
#   run_thermal_gate.sh                    # wait for Nominal, then 1 positive + 1 negative arm
#   ARMS=1 run_thermal_gate.sh             # positive arm only
#   MAX_WAIT=60 run_thermal_gate.sh        # give up after 60 s of polling
#   GATE_FORCE=1 run_thermal_gate.sh       # run even if the gate never opens (stamped as hot)
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 2

OUTDIR="${OUTDIR:-$ROOT/Backup/cgc_logs/thermal_gate}"
MAX_WAIT="${MAX_WAIT:-1800}"
POLL="${POLL:-10}"
ARMS="${ARMS:-2}"
GATE_FORCE="${GATE_FORCE:-0}"
KEY="com.apple.system.thermalpressurelevel"

mkdir -p "$OUTDIR" || { echo "error: cannot create OUTDIR=$OUTDIR" >&2; exit 2; }

STAMP="$(date +%Y%m%d_%H%M%S)"
PRESSURE="$OUTDIR/pressure_${STAMP}.tsv"
GATE_LOG="$OUTDIR/gate_${STAMP}.txt"

level() { notifyutil -g "$KEY" 2>/dev/null | awk '{print $NF}'; }
name()  { case "$1" in 0) echo NOMINAL;; 1) echo MODERATE;; 2) echo HEAVY;; 3) echo TRAPPING;; 4) echo SLEEPING;; *) echo UNREADABLE;; esac; }

echo "=== thermal gate (stamp $STAMP) ===" | tee "$GATE_LOG"
echo "  condition : $KEY == 0 (Nominal)" | tee -a "$GATE_LOG"
echo "  max wait  : ${MAX_WAIT}s, polling every ${POLL}s" | tee -a "$GATE_LOG"
echo "  outdir    : $OUTDIR" | tee -a "$GATE_LOG"

# --- wait for the gate to open ---------------------------------------------------------------
_waited=0; _gate=no
while :; do
    _l="$(level)"
    echo "  [$(date '+%H:%M:%S')] level=$_l ($(name "$_l"))  waited=${_waited}s" | tee -a "$GATE_LOG"
    [ "$_l" = "0" ] && { _gate=yes; break; }
    [ "$_waited" -ge "$MAX_WAIT" ] && break
    sleep "$POLL"; _waited=$(( _waited + POLL ))
done

if [ "$_gate" != yes ]; then
    echo "  !! gate never opened within ${MAX_WAIT}s -- NO ARM WAS RUN (fail closed)." | tee -a "$GATE_LOG"
    if [ "$GATE_FORCE" = "1" ]; then
        echo "  GATE_FORCE=1: running anyway; the arm is a HOT-STATE OBSERVATION and its t/s must" | tee -a "$GATE_LOG"
        echo "  not be quoted as a delivery number." | tee -a "$GATE_LOG"
    else
        echo "  Re-run later, raise MAX_WAIT, or set GATE_FORCE=1 to deliberately produce a hot sample." | tee -a "$GATE_LOG"
        exit 3
    fi
else
    echo "  gate OPEN: level 0/NOMINAL after ${_waited}s -- the next arm is a delivery-grade sample" | tee -a "$GATE_LOG"
fi

# --- out-of-band series, so a mid-arm excursion is visible afterwards ------------------------
bash "$ROOT/Backup/thermal_pressure_probe.sh" "$PRESSURE" 0.5 900 > "$OUTDIR/probe_${STAMP}.console.log" 2>&1 &
PROBE=$!
sleep 1

# --- arms -------------------------------------------------------------------------------------
printf '# columns:\tarm\tlaunch_level\treq1\treq2\treq3\tsurvived\tcrash\treport\n' > "$OUTDIR/summary_${STAMP}.tsv"

run_arm() {
    local idx="$1" label="$2"
    local lvl; lvl="$(level)"
    echo | tee -a "$GATE_LOG"
    echo "########## arm $idx ($label)  launch_level=$lvl/$(name "$lvl")  $(date '+%H:%M:%S') ##########" | tee -a "$GATE_LOG"
    IDLE_BEFORE=0 OUTDIR="$OUTDIR" bash "$ROOT/Backup/run_req2_retest.sh" 2>&1 \
        | grep -E 'thermal pressure|thermal label|health OK|pp_t/s|RESULT:|new crash report|report:|high-water|condition SATISFIED|condition NOT|-> ' \
        | tee -a "$GATE_LOG"

    local rep; rep="$(ls -1t "$OUTDIR"/req2retest_*.txt 2>/dev/null | head -1)"
    [ -n "$rep" ] || { printf '%s\t%s\t%s\t?\t?\t?\t?\t?\tnone\n' "$idx" "$label" "$lvl" >> "$OUTDIR/summary_${STAMP}.tsv"; return 0; }
    local r1 r2 r3 sv cr
    r1="$(grep -oE 'req1: OK .*pp_t/s=[0-9.]+' "$rep" | grep -oE '[0-9.]+$')"
    r2="$(grep -oE 'req2: OK .*pp_t/s=[0-9.]+' "$rep" | grep -oE '[0-9.]+$')"
    r3="$(grep -oE 'req3: OK .*pp_t/s=[0-9.]+' "$rep" | grep -oE '[0-9.]+$')"
    sv="$(grep -oE 'survived=[a-z]+' "$rep" | head -1 | cut -d= -f2)"
    cr="$(grep -oE 'new crash report: .*' "$rep" | head -1 | sed 's/new crash report: //')"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$idx" "$label" "$lvl" "${r1:-?}" "${r2:-?}" "${r3:-?}" "${sv:-?}" "${cr:-?}" "$(basename "$rep")" \
        >> "$OUTDIR/summary_${STAMP}.tsv"
}

if [ "$ARMS" -ge 1 ]; then run_arm 1 "positive (gate open)"; fi
if [ "$ARMS" -ge 2 ]; then run_arm 2 "negative control (immediately after)"; fi

pkill -f thermal_pressure_probe.sh 2>/dev/null
sleep 1
echo | tee -a "$GATE_LOG"
echo "=== out-of-band pressure series ($PRESSURE) ===" | tee -a "$GATE_LOG"
bash "$ROOT/Backup/thermal_pressure_probe.sh" --summarize "$PRESSURE" | tee -a "$GATE_LOG"

echo | tee -a "$GATE_LOG"
echo "=== summary ($OUTDIR/summary_${STAMP}.tsv) ===" | tee -a "$GATE_LOG"
cat "$OUTDIR/summary_${STAMP}.tsv" | tee -a "$GATE_LOG"
echo | tee -a "$GATE_LOG"
echo "gate log: $GATE_LOG" | tee -a "$GATE_LOG"
