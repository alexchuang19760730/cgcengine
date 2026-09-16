#!/usr/bin/env bash
# [CGC 2026-09-16] (b) A COLD arm that certifies the NEW unified profile's prefill.
#
# What is new since the last certified arm: `prefill250` now pins `CGC_SPAC=1` +
# `CGC_SPAC_ALPHA=0.75` (the 18:48 A/B: decode +17% at d512 and dispersion +-2.97 -> +-0.04). SPAC's
# EMA update runs on EVERY routed step, including a wide prefill, and its prefetch re-sorts -- so the
# cost lands on the prefill path, and the 250+ delivery sentence cannot be inherited. It has to be
# re-earned on a cold machine.
#
# WHY THIS DRIVER EXISTS RATHER THAN A BARE CALL
#   The gate classifies an arm COLD / HOT / UNKNOWN from `$OUTDIR/.last_arm_end`, whose meaning is
#   "when the previous SUSTAINED prefill ended". This session's last load (the D5 gate) ended at
#   20:24:41, and the state file at that path does not exist yet -- so a bare call would print
#   UNKNOWN-STATE and make no claim, and dropping a fresh timestamp in would make a claim that is
#   not true. The honest move is to SEED it with the measured end of the last real load, wait the
#   full COLD_QUIET from THAT instant, and say so. That is all this script does.
#
# Usage:  bash Backup/run_spac_cold_arm.sh
#         LAST_LOAD_EPOCH=1789561481 bash Backup/run_spac_cold_arm.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 2

# 1789561481 = 2026-09-16 20:24:41 local, the mtime of Backup/m123_oracle_gate/summary_capround2.json,
# i.e. the end of this session's last GPU load. Conservative: it is the LATEST candidate.
LAST_LOAD_EPOCH="${LAST_LOAD_EPOCH:-1789561481}"
COLD_QUIET="${COLD_QUIET:-1800}"
OUTDIR="${OUTDIR:-Backup/cgc_logs/spac_cold}"
STATE="$OUTDIR/.last_arm_end"

mkdir -p "$OUTDIR"
printf '%s\n' "$LAST_LOAD_EPOCH" > "$STATE"

now="$(date +%s)"
ready=$(( LAST_LOAD_EPOCH + COLD_QUIET ))
wait_s=$(( ready - now ))

echo "=== (b) COLD arm for prefill250 + CGC_SPAC=1  $(date '+%F %T') ==="
echo "  seeded $STATE = $LAST_LOAD_EPOCH  ($(date -r "$LAST_LOAD_EPOCH" '+%F %T'))  = last real load ended"
echo "  COLD_QUIET   = ${COLD_QUIET}s  -> quiet satisfied at $(date -r "$ready" '+%F %T')"
if [ "$wait_s" -gt 0 ]; then
    echo "  waiting ${wait_s}s with NOTHING running (this is the experiment, not a delay) ..."
    # 60 s slices so the wait itself is visible in the log rather than looking like a hang.
    while [ "$(date +%s)" -lt "$ready" ]; do
        left=$(( ready - $(date +%s) ))
        [ "$left" -lt 0 ] && left=0
        echo "    [wait] ${left}s left   thermal=$(notifyutil -g com.apple.system.thermalpressurelevel | awk '{print $NF}')"
        sleep $(( left > 60 ? 60 : (left > 0 ? left : 1) ))
    done
else
    echo "  quiet already satisfied (${wait_s}s ago)"
fi

echo
echo "  launching the gate; its own # [thermal label] line is the claim -- read THAT, not this one."
echo
OUTDIR="$OUTDIR" CGC_ARM_STATE="$STATE" bash Backup/run_req2_retest.sh
rc=$?
echo
echo "=== (b) done rc=$rc  $(date '+%F %T') ==="
echo "  reports: $(ls -1t "$OUTDIR"/gate_*.txt 2>/dev/null | head -1)"
