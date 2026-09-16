#!/usr/bin/env bash
# Backup/run_lib_ab.sh -- ABBA-swap of libggml-metal to adjudicate whitepaper 11.13's unattributed 2.25x
#
# WHAT QUESTION THIS ANSWERS
#   Whitepaper 11.13 records two groups measured on the SAME model, SAME resolved config and SAME
#   A11 fingerprint a9c9dc104f057bf640e0132a83e01c12, whose prefill t/s differ by 2.25x:
#     - 12:44 artifact  -> 268.09 / 273.55 / 256.61  (hot band ~130)
#     - final  artifact -> 118.30 - 154.77
#   The only *content* difference between the two artifacts is libggml-metal. This script asks the
#   falsifiable version of "is the library the cause?" -- by putting the two images head to head on
#   the machine as it is now.
#
# WHY A FILE SWAP AND NOT AN ENV VAR
#   CGC_METAL_LIB is NOT a loader override. It is read only by scripts/run_server.sh's detection
#   hooks (the CGC_MMAP_POOL_FIX probe greps the *installed* image for the `_Pool` buft name).
#   The image that actually gets mapped is whatever the rpath resolves to, i.e.
#   src/llama.cpp/build/bin/libggml-metal.0.19.0.dylib. So an A/B of the library = swapping that file.
#
# WHY THE SWAP IS ABI-SAFE HERE (checked 2026-09-16, not assumed)
#   otool -D: identical install name. otool -L: identical dependencies. nm -u: 135/135 identical
#   undefined imports. codesign -dv: both linker-signed adhoc. The only differing section is __text.
#
# WHY ABBA AND NOT A-then-B
#   The effect we are hunting is a *difference between images* on a machine whose throughput drifts
#   within a session (measured on one binary: 284.58 / 157.95 / 130.88). A sequential A-then-B would
#   confound the image with the soak. Interleaving puts the soak *inside* both groups, so a soak that
#   is linear in time cancels; a soak that is not linear can at least be seen as scatter within a group.
#
# WHY ARM 1 IS DISCARDED
#   It inherits whatever thermal state the previous work left, which is unknown. Its number is
#   printed but never enters the comparison.
#
# WHAT THIS CANNOT SETTLE
#   The 12:44 image (md5 968c36cf...) was never backed up, so it cannot be put in the ring. A is the
#   11:46 image (md5 dd4d1bc4..., `_Pool` count=0), which is a *different* older build. So a null
#   result here falsifies "the library explains the 2.25x" for the pair (pre-pool-buft, final), not
#   for the exact pair (12:44, final). Stated because the gap is not recoverable after the fact.

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 2

LIB="$ROOT/src/llama.cpp/build/bin/libggml-metal.0.19.0.dylib"
LIB_A="${LIB_A:-$ROOT/Backup/pre_flagalign_20260916/libggml-metal.0.19.0.dylib}"
LIB_B="${LIB_B:-$ROOT/Backup/libggml-metal_final_saved.dylib}"
LIB_HOME="${LIB_HOME:-$ROOT/Backup/libggml-metal_abba_home.dylib}"

OUTDIR="${OUTDIR:-$ROOT/Backup/cgc_logs/libab}"
mkdir -p "$OUTDIR" || { echo "error: cannot create OUTDIR=$OUTDIR" >&2; exit 2; }

STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY="$OUTDIR/summary_${STAMP}.tsv"

md5_of() { md5 -q "$1" 2>/dev/null || echo "?"; }

for f in "$LIB_A" "$LIB_B"; do
    [ -f "$f" ] || { echo "error: missing ring member $f" >&2; exit 2; }
done
[ -f "$LIB" ] || { echo "error: incumbent $LIB missing" >&2; exit 2; }

if [ ! -f "$LIB_HOME" ]; then
    cp -p "$LIB" "$LIB_HOME" || exit 2
fi

echo "# A  = $(md5_of "$LIB_A")   (pre-pool-buft, 11:46)"
echo "# B  = $(md5_of "$LIB_B")   (final, 13:50)"
echo "# home = $(md5_of "$LIB_HOME")   (incumbent; restored on exit)"
echo "# summary -> $SUMMARY"

restore() {
    cp -p "$LIB_HOME" "$LIB" 2>/dev/null
    echo "# [restore] incumbent library back in place: md5=$(md5_of "$LIB")"
}
trap restore EXIT INT TERM

install_lib() {
    local src="$1" want got
    want="$(md5_of "$src")"
    cp -p "$src" "$LIB" || return 1
    got="$(md5_of "$LIB")"
    if [ "$got" != "$want" ]; then
        echo "error: installed image mismatch ($got != $want)" >&2
        return 1
    fi
    echo "# installed md5=$got"
}

printf '# columns:\tidx\tlib\tmd5\treq1\treq2\treq3\tsurvived\tcrash\treport\n' > "$SUMMARY"

run_arm() {
    local idx="$1" tag="$2" src="$3"
    echo
    echo "########## arm $idx  lib=$tag  (started $(date +%H:%M:%S)) ##########"
    install_lib "$src" || return 1
    local md5; md5="$(md5_of "$LIB")"

    IDLE_BEFORE=0 OUTDIR="$OUTDIR" bash "$ROOT/Backup/run_req2_retest.sh" 2>&1 \
        | grep -E 'thermal label|machine state|OK wall|RESULT:|new crash report|health OK|report:' \
        | sed 's/^/    /'

    local report; report="$(ls -1t "$OUTDIR"/req2retest_*.txt 2>/dev/null | head -1)"
    if [ -z "$report" ]; then
        echo "    !! no report produced for this arm -- not attributing anything to it"
        printf '%s\t%s\t%s\t%s\t?\t?\t?\t?\t?\tnone\n' "$idx" "$tag" "$md5" >> "$SUMMARY"
        return 0
    fi

    local r1 r2 r3 surv crash
    r1="$(grep -oE 'req1: OK .*pp_t/s=[0-9.]+' "$report" | grep -oE 'pp_t/s=[0-9.]+$' | sed 's/pp_t\/s=//')"
    r2="$(grep -oE 'req2: OK .*pp_t/s=[0-9.]+' "$report" | grep -oE 'pp_t/s=[0-9.]+$' | sed 's/pp_t\/s=//')"
    r3="$(grep -oE 'req3: OK .*pp_t/s=[0-9.]+' "$report" | grep -oE 'pp_t/s=[0-9.]+$' | sed 's/pp_t\/s=//')"
    surv="$(grep -oE 'survived=[a-z]+' "$report" | head -1 | cut -d= -f2)"
    crash="$(grep -oE 'new crash report: .*' "$report" | head -1 | sed 's/new crash report: //')"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$idx" "$tag" "$md5" "${r1:-?}" "${r2:-?}" "${r3:-?}" "${surv:-?}" "${crash:-?}" \
        "$(basename "$report")" >> "$SUMMARY"

    echo "    -> req1=${r1:-?} req2=${r2:-?} req3=${r3:-?}  survived=${surv:-?}  crash=${crash:-?}"
}

# WARMUP (discarded), then ABBA
run_arm 0 "A(warmup-discard)" "$LIB_A"
run_arm 1 "B" "$LIB_B"
run_arm 2 "A" "$LIB_A"
run_arm 3 "A" "$LIB_A"
run_arm 4 "B" "$LIB_B"

echo
echo "=== summary: $SUMMARY ==="
cat "$SUMMARY"
