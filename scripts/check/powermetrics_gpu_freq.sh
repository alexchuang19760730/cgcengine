#!/bin/bash
# powermetrics_gpu_freq.sh — settle "was the prefill slowdown a clock or a power cap" in one command
#
# WHY THIS EXISTS
# ---------------
# `docs/PREFILL250_THERMAL_TRANSIENT_20260916.html` proved the prefill bimodality is the GPU
# executing the same work more slowly (byte-identical workload counters, `gpu_union/wait = 100%`,
# GPU busy-time ratio 1.44x against a throughput ratio of 1.42x) and then *stopped short of naming
# the cause on purpose*: busy is not slow, and clock / power cap / memory subsystem are three
# different explanations with three different fixes. `powermetrics` separates them.
#
# It needs root. Root is unavailable to the agent that wrote this (`sudo` is blocked in that
# sandbox -- three attempts on 2026-09-16, the last one recorded verbatim in
# `Backup/cgc_logs/powermetrics_prefill_paired_20260916.log`, whose only line is
# `(eval):1: operation not permitted: sudo`). A second route was tried and closed too: the
# unprivileged IOReport path enumerates the "GPU Stats / GPU Performance States" channels but
# cannot subscribe to them (`scripts/check/ioreport_gpu_pstate_probe.py` documents the exact
# refusal). So this is genuinely a manual step, and this script exists to make it one command
# instead of a sequence of tabs.
#
# WHAT TO LOOK FOR
# ----------------
#   GPU HW active frequency   -- if this DROPS from cold run to hot run, the cause is the clock.
#   GPU HW active residency   -- expected to sit near 100% in both, matching `gpu_union/wait = 100%`
#                                from the Metal timestamps. If residency also drops, the GPU is
#                                being kept *off* the work, which is a different finding.
#   GPU Power                 -- if power is pinned at a ceiling in both while frequency falls, the
#                                ceiling is the mechanism, not thermal throttling of the clock.
#
# USAGE
# -----
#   sudo -v                                     # cache credentials once, then:
#   scripts/check/powermetrics_gpu_freq.sh      # cold/hot pair, ~4 minutes
#
#   scripts/check/powermetrics_gpu_freq.sh --parse ~/Desktop/pm_prefill_*.log
#                                               # summarise a capture that already exists
#
# ENV: IDLE (default 180)  RUNS (default 3)  INTERVAL_MS (default 500)  OUT (output path)
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
IDLE="${IDLE:-180}"
RUNS="${RUNS:-3}"
INTERVAL_MS="${INTERVAL_MS:-500}"
OUT="${OUT:-$HOME/Desktop/pm_prefill_$(date +%Y%m%d_%H%M%S).log}"
CERT="$ROOT/scripts/check/prefill_certifiability.py"
PY="${PY:-/opt/homebrew/bin/python3}"

if [ "${1:-}" = "--parse" ]; then
    F="${2:?usage: $0 --parse <powermetrics log>}"
    [ -f "$F" ] || { echo "no such file: $F" >&2; exit 1; }
    echo "=== powermetrics summary: $F ==="
    echo "samples              : $(grep -c 'GPU HW active frequency' "$F")"
    echo
    echo "--- GPU HW active frequency (MHz), in order ---"
    grep -o 'GPU HW active frequency: *[0-9.]* MHz' "$F" | grep -o '[0-9.]*' | paste -sd' ' -
    echo
    echo "--- GPU HW active residency (%) ---"
    grep -o 'GPU HW active residency: *[0-9.]*%' "$F" | grep -o '[0-9.]*' | paste -sd' ' -
    echo
    echo "--- GPU idle residency (%) ---"
    grep -o 'GPU idle residency: *[0-9.]*%' "$F" | grep -o '[0-9.]*' | paste -sd' ' -
    echo
    echo "--- GPU Power (mW) ---"
    grep -oE 'GPU Power: *[0-9.]+ mW' "$F" | grep -o '[0-9.]*' | paste -sd' ' -
    echo
    echo "read: the FIRST run of the pair is the cold one. A frequency series that starts high and"
    echo "      settles low, with residency near 100% throughout, names the clock. If the frequency"
    echo "      series is flat and the power series is pinned, the cap is the mechanism."
    exit 0
fi

# ---- run mode -------------------------------------------------------------------------------
if [ -z "${CERT_ASSUME_ROOT:-}" ] && ! sudo -n true 2>/dev/null; then
    echo "This needs a root powermetrics. Run 'sudo -v' first (to cache credentials), then re-run." >&2
    echo "The harness itself stays unprivileged: only powermetrics is elevated." >&2
    exit 1
fi

echo "=== powermetrics cold/hot pair ==="
echo "  out         : $OUT"
echo "  interval    : ${INTERVAL_MS} ms"
echo "  idle before : ${IDLE} s (this is what makes run 1 the cold state)"
echo "  runs        : $RUNS"
echo

sudo powermetrics -i "$INTERVAL_MS" --samplers gpu_power,thermal --format text >"$OUT" 2>&1 &
PM=$!
sleep 2
if ! kill -0 "$PM" 2>/dev/null; then
    echo "powermetrics died immediately; its output so far:" >&2
    cat "$OUT" >&2
    exit 1
fi

# The certifiability harness is the load generator on purpose: it is the same command that produced
# every number in the report, so the powermetrics series and the t/s table are comparable without
# any re-derivation.
"$PY" "$CERT" --arm prefill250-decprof --runs "$RUNS" --idle-before "$IDLE" \
    --logdir "$ROOT/Backup/llama_bench/pm_pair" \
    --json "$ROOT/Backup/llama_bench/pm_pair_$(date +%Y%m%d_%H%M%S).json"
RC=$?

kill -INT "$PM" 2>/dev/null
wait "$PM" 2>/dev/null
echo
echo "=== capture -> $OUT ($(wc -l <"$OUT") lines) ==="
"$0" --parse "$OUT"
exit "$RC"
