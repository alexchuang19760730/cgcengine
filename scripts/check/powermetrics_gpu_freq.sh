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
# It was run by hand on 2026-09-16 09:08 and the result is in
# `Backup/cgc_logs/powermetrics_prefill_20260916_090808.log` -- three launches, thermal pressure
# Nominal -> Moderate -> Heavy, GPU clock step 1470 -> 928 -> 618 MHz, throughput 276 -> 227 ->
# 151 t/s. See `docs/PREFILL250_THERMAL_TRANSIENT_20260916.html` §10.
#
# WHAT TO LOOK FOR
# ----------------
#   GPU HW active frequency   -- if this DROPS from cold run to hot run, the cause is the clock.
#   GPU HW active residency   -- expected to sit near 100% in both, matching `gpu_union/wait = 100%`
#                                from the Metal timestamps. If residency also drops, the GPU is
#                                being kept *off* the work, which is a different finding.
#   GPU Power                 -- if power is pinned at a ceiling in both while frequency falls, the
#                                ceiling is the mechanism, not thermal throttling of the clock.
#   Current pressure level    -- the thermal sampler's verdict, and on 2026-09-16 it was the
#                                mechanism: pressure flipped Moderate -> Heavy in the same sample
#                                as the clock started falling. A clock that falls while power
#                                falls WITH it is a thermal operating point, not a fixed budget;
#                                a clock that falls while power stays pinned is a power ceiling.
#   DVFS step residency       -- the parenthetical after the residency percentage lists how much
#                                active time sat at EACH clock step. `GPU HW active frequency`
#                                collapses that to one number, so two runs can report the same
#                                "active frequency" while occupying different mixtures of steps.
#                                This is the field that actually distinguishes the three causes.
#                                `--parse` reports it as a per-block table; do not read the single
#                                active-frequency number alone.
#
# USAGE
# -----
#   sudo -v                                     # cache credentials once, then:
#   scripts/check/powermetrics_gpu_freq.sh      # cold/hot pair, ~10 minutes
#
#   scripts/check/powermetrics_gpu_freq.sh --parse Backup/cgc_logs/powermetrics_prefill_*.log
#                                               # segment the capture into launches and compare
#                                               # cold vs hot (PARSE_JSON=out.json also dumps it)
#
#   python3 scripts/check/powermetrics_gpu_freq_parse.py --selftest
#                                               # exercise the verdict rules on synthetic captures
#
# ENV: IDLE (default 180)  RUNS (default 3)  INTERVAL_MS (default 500)  OUT (output path)
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
IDLE="${IDLE:-180}"
RUNS="${RUNS:-3}"
INTERVAL_MS="${INTERVAL_MS:-500}"
# Evidence lands with the other evidence (and Backup/ is gitignored, so it cannot dirty the tree);
# ~/Desktop was the old default and is a personal directory, not a project one.
OUT="${OUT:-$ROOT/Backup/cgc_logs/powermetrics_prefill_$(date +%Y%m%d_%H%M%S).log}"
CERT="$ROOT/scripts/check/prefill_certifiability.py"
PY="${PY:-/opt/homebrew/bin/python3}"

if [ "${1:-}" = "--parse" ]; then
    shift
    if [ "$#" -lt 1 ]; then
        echo "usage: $0 --parse <powermetrics log> [more logs ...]" >&2
        echo "       a glob is fine; a file with no samples is reported and skipped, never ignored." >&2
        exit 3
    fi
    for F in "$@"; do
        [ -f "$F" ] || { echo "no such file: $F" >&2; exit 1; }
    done
    # Delegated on 2026-09-16. The grep/paste that used to live here printed ~480 raw numbers per
    # series -- the question is a cold-vs-hot comparison, not a series -- and it silently discarded
    # the DVFS residency distribution, which is the only field that separates a clock cap from a
    # power cap. It also fails LOUDLY: if the labels ever change it prints the GPU lines it did see,
    # instead of four empty series that look like a measurement of nothing, and if a file is not a
    # capture at all -- a run that died before powermetrics started leaves an error line behind --
    # it prints that line. Several captures at once, because
    # `Backup/cgc_logs/powermetrics_prefill_*.log` is the natural way to call this once more than
    # one capture exists, and a silently-dropped argument is how a dead file passes for a good one.
    if [ -n "${PARSE_JSON:-}" ]; then
        exec "$PY" "$ROOT/scripts/check/powermetrics_gpu_freq_parse.py" --json "$PARSE_JSON" "$@"
    fi
    exec "$PY" "$ROOT/scripts/check/powermetrics_gpu_freq_parse.py" "$@"
fi

# ---- run mode -------------------------------------------------------------------------------
if [ -z "${CERT_ASSUME_ROOT:-}" ] && ! sudo -n true 2>/dev/null; then
    echo "This needs a root powermetrics. Run 'sudo -v' first (to cache credentials), then re-run." >&2
    echo "The harness itself stays unprivileged: only powermetrics is elevated." >&2
    exit 1
fi

echo "=== powermetrics cold/hot pair ==="
# HOW THE CAPTURE IS STOPPED -- and why it is not `kill $!`
# -------------------------------------------------------
# `$!` after `sudo powermetrics ... &` is the pid of *sudo*, which runs as ROOT. This shell is
# unprivileged, so `kill -INT "$PM"` on it returns EPERM. The original version wrote that as
# `kill -INT "$PM" 2>/dev/null` -- discarding the only evidence that the stop failed -- and then
# blocked in `wait` forever. Consequences, all observed on 2026-09-16: the `--parse` at the end of
# this script never ran (so "it parses itself when it finishes" was never true), powermetrics kept
# appending to the capture for another 32 minutes after the last launch finished at 09:14, and
# that idle tail broke the parser's block segmentation (25-31 blocks instead of 3).
#
# Two independent guarantees replace it:
#   1. a root watcher (the only thing here that CAN signal the root powermetrics) waits for a
#      sentinel file this shell creates after the harness returns. No privilege needed to stop.
#   2. `-n SAMPLES`, an upper bound on the sample count, so powermetrics exits by itself even if
#      the sentinel path breaks. Progress must not depend on a single mechanism.
# The bound is generous on purpose: truncating the last launch is worse than a short idle tail,
# and the parser is robust to the tail (see WHY THE BLOCK THRESHOLD IS A HIGH PERCENTILE).
#
# SIGTERM, NOT SIGINT -- and this is the second, independent reason the old stop never worked.
# POSIX has a non-interactive shell set SIGINT *and SIGQUIT* to SIG_IGN in its background jobs
# (job control is off). So `kill -INT` on such a job returns 0 and the process keeps running:
#
#     $ sh -c 'sleep 30 & P=$!; sleep 0.2; kill -INT $P; sleep 0.5; kill -0 $P && echo alive'
#     94316                        (checked on 2026-09-16; verified, not deduced)
#     alive
#
# SIGTERM is not ignored in that way, so the watcher uses TERM. `-b 1` (line buffering) is set for
# the same reason: a hard stop must not take the last record with it, and powermetrics writes to a
# file, where stdio would otherwise be block-buffered.
#
# IF YOU EVER HAVE TO CLEAN UP BY HAND: which pid you signal IS the whole question.
# The 2026-09-16 capture was left running by the pre-fix stop path, and the cleanup was:
#
#     $ pgrep -fl powermetrics
#     84847  sudo powermetrics -i 500 --samplers gpu_power,thermal --format text   <- YOURS (wrapper)
#     84849  powermetrics -i 500 --samplers gpu_power thermal --format             <- root's (the body)
#     $ kill -TERM 84849   ->  kill: operation not permitted   <- this is the pid that cannot be killed
#     $ kill -TERM 84847   ->  returns 0                       <- signal the WRAPPER; both go away
#     # then: pgrep is empty and the capture is frozen at 5 410 344 bytes
#
# Two things worth keeping from that:
#   * "I cannot kill it" is a statement about ONE PID, not about the chain. Read `pgrep -fl` before
#     concluding anything: the wrapper is owned by you even when its child is not. The report of
#     this cleanup had earlier recorded the opposite ("cannot be terminated from this session"),
#     which was wrong -- and wrong in the exact shape this script's own notes warn about, because
#     the failed `kill` was the one whose error had been thrown away.
#   * Verify a stop by EFFECT (does the file still grow), never by `kill -0`. That is the rule the
#     FROZEN check below already uses, and it is the rule that told the truth here.
TOTAL_S=$(( IDLE + RUNS * 180 + 120 ))
SAMPLES=$(( TOTAL_S * 1000 / INTERVAL_MS + 1 ))
STOP="$OUT.stop"
PMFILE="$OUT.pid"
rm -f "$STOP" "$PMFILE"

echo "  out         : $OUT"
echo "  interval    : ${INTERVAL_MS} ms"
echo "  idle before : ${IDLE} s (this is what makes run 1 the cold state)"
echo "  runs        : $RUNS"
echo "  stop        : sentinel after the harness + a ${SAMPLES}-sample (~${TOTAL_S} s) ceiling"
echo

sudo sh -c "powermetrics -i $INTERVAL_MS -n $SAMPLES -b 1 --samplers gpu_power,thermal --format text \
> \"$OUT\" 2>&1 & echo \$! > \"$PMFILE\"; \
while [ ! -f \"$STOP\" ]; do sleep 1; done; \
kill -TERM \$(cat \"$PMFILE\") 2>/dev/null" &
PM=$!
sleep 2
if ! kill -0 "$PM" 2>/dev/null; then
    echo "powermetrics died immediately; its output so far:" >&2
    cat "$OUT" >&2
    rm -f "$PMFILE" "$STOP"
    exit 1
fi

# The certifiability harness is the load generator on purpose: it is the same command that produced
# every number in the report, so the powermetrics series and the t/s table are comparable without
# any re-derivation.
"$PY" "$CERT" --arm prefill250-decprof --runs "$RUNS" --idle-before "$IDLE" \
    --logdir "$ROOT/Backup/llama_bench/pm_pair" \
    --json "$ROOT/Backup/llama_bench/pm_pair_$(date +%Y%m%d_%H%M%S).json"
RC=$?

touch "$STOP"                   # the root watcher stops powermetrics
wait "$PM" 2>/dev/null          # ...and the root shell exits, so this returns

# Verify the capture actually FROZE, rather than asking whether the pid is alive: signal delivery
# races with the last write, so a liveness check reports a false alarm right after a perfectly
# good stop -- and a warning that cries wolf is worse than no warning. "Did the file grow" is the
# property that matters, and a stopped powermetrics never writes again.
FROZEN=0
for _ in 1 2 3 4 5 6 7 8 9 10; do
    c1=$(wc -c <"$OUT" 2>/dev/null || echo 0)
    sleep 0.5
    c2=$(wc -c <"$OUT" 2>/dev/null || echo 0)
    [ "$c1" = "$c2" ] && { FROZEN=1; break; }
done
if [ "$FROZEN" = 0 ]; then
    echo "WARNING: the capture is STILL being written after the stop ($c1 -> $c2 bytes in 0.5 s)." >&2
    echo "         It will end by itself at the $SAMPLES-sample ceiling, but until then this file" >&2
    echo "         is a moving target: any parse of it is a snapshot. Kill it with:" >&2
    echo "             sudo kill -TERM $(cat "$PMFILE" 2>/dev/null || echo '<pid>')" >&2
    echo "         If pgrep also shows a 'sudo powermetrics' WRAPPER, signal the wrapper instead:" >&2
    echo "         it belongs to you, so no sudo is needed, and both processes go away." >&2
fi
rm -f "$PMFILE" "$STOP"
echo
echo "=== capture -> $OUT ($(wc -l <"$OUT") lines) ==="
"$0" --parse "$OUT"
exit "$RC"
