#!/usr/bin/env bash
# Backup/thermal_pressure_probe.sh -- non-root sampler for the AUTHORITATIVE thermal pressure level.
#
# WHY THIS AND NOT powermetrics
#   whitepaper 11.14 concluded the ">=250 t/s" condition could not be observed without root, and so
#   could only be enforced as a protocol (a quiet interval). That was wrong, and the reason is a
#   single key:
#
#       notifyutil -g com.apple.system.thermalpressurelevel
#
#   This is the SAME notify(3) key that powermetrics' thermal sampler reads to produce its
#   `Current pressure level` line. It is readable by any user, costs ~11 ms of CPU per read
#   (measured: 20 reads in 0.225 s), and uses the IOKit scale directly:
#
#       0 = Nominal   1 = Moderate   2 = Heavy   3 = Trapping   4 = Sleeping
#
#   So the condition is no longer "we controlled the quiet interval and hope" -- it is a number we
#   can read, at a rate finer than one prefill, with no privileges and no capture process.
#
# WHY 2 Hz
#   A prefill under the delivery protocol is ~11 s, so 2 Hz gives ~22 samples across it -- enough to
#   see the level at the start, through the plateau, and at the end. Each sample spawns two short
#   processes (notifyutil + sleep) costing ~2% of one core; on a fanless M4 running a thermal
#   experiment that is a deliberate ceiling, chosen small enough not to move the state being read.
#
# WHAT IT IS NOT
#   Not a clock. The level is a discrete governor input; powermetrics' DVFS residency table is what
#   tells you which step it selected. This tells you whether the governor is in the regime that
#   licenses a 250-class number, which is the decision, not the mechanism.
#
# Usage:
#   thermal_pressure_probe.sh <output.tsv> [interval_s] [count]
#   thermal_pressure_probe.sh --summarize <output.tsv>
#
# Defaults: interval 0.5 s, count 1200 (10 min).
set -uo pipefail

KEY="com.apple.system.thermalpressurelevel"

if [ "${1:-}" = "--summarize" ]; then
    OUT="${2:-}"
    [ -f "$OUT" ] || { echo "error: no such file: $OUT" >&2; exit 2; }
    awk -F'\t' '
        { n[$2]++; tot++ }
        END {
            print "samples\t" tot
            split("0 Nominal,1 Moderate,2 Heavy,3 Trapping,4 Sleeping,? UNAVAILABLE", nm, ",")
            for (i in nm) { split(nm[i], p, " "); if (p[1] in n) printf "  level %s\t%d\t%.1f%%\n", p[2], n[p[1]], 100*n[p[1]]/tot }
        }' "$OUT"
    exit 0
fi

OUT="${1:-}"
INTERVAL="${2:-0.5}"
COUNT="${3:-1200}"
[ -n "$OUT" ] || { echo "usage: thermal_pressure_probe.sh <output.tsv> [interval_s] [count]" >&2; exit 2; }

mkdir -p "$(dirname "$OUT")" || { echo "error: cannot create dir for $OUT" >&2; exit 2; }

read_level() {
    local raw
    raw="$(notifyutil -g "$KEY" 2>/dev/null | awk '{print $NF}')"
    case "$raw" in
        0|1|2|3|4) echo "$raw" ;;
        *)         echo "?" ;;
    esac
}

echo "# thermal pressure probe -> $OUT   key=$KEY   interval=${INTERVAL}s count=${COUNT}"
echo "# fields: local_timestamp<TAB>level<TAB>name"
printf '%-22s\t%-5s\t%s\n' "# local_timestamp" "level" "name" >&2

NAMES=(Nominal Moderate Heavy Trapping Sleeping)

: > "$OUT"
for _i in $(seq 1 "$COUNT"); do
    _lvl="$(read_level)"
    if [ "$_lvl" = "?" ]; then _nm="UNAVAILABLE"; else _nm="${NAMES[$_lvl]}"; fi
    printf '%s\t%s\t%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$_lvl" "$_nm" >> "$OUT"
    sleep "$INTERVAL"
done

echo "# probe finished $(date '+%F %T'); samples=$(wc -l < "$OUT" | tr -d ' ')"
bash "$0" --summarize "$OUT"
