#!/bin/bash
# run_spec_simple.sh — drive the fork's MTP path OUTSIDE the server, with the profile's resolved env.
#
# WHY. `llama-bench` cannot measure the MTP path: llama-bench.cpp has zero matches for
# sampler|speculat|draft|MTP and advances the sequence with `std::rand() % n_vocab`, so it never
# samples and therefore never accepts a draft. The speculative loop is NOT server-private though --
# it lives in `common/speculative.{h,cpp}` and has exactly two callers: `server-context.cpp` and
# `examples/speculative-simple/speculative-simple.cpp`. That example parses the SAME common flags,
# handles COMMON_SPECULATIVE_TYPE_DRAFT_MTP explicitly, has an internal no-spec arm, and this
# project has already committed MTP fixes to it (b7364f886, eb16bd129).
#
# The env is NOT written here. It is whatever `run_server.sh CGC_DUMP_ENV=1` resolves for the
# requested profile -- the same single source `llama_bench_matrix.py` uses. Two lessons are baked
# into that choice: the profile's memory-shaping knobs (CGC_N_CB=8, CGC_DBUF, CGC_NO_PREFETCH,
# CGC_OA_ASYNC) are what keep this model inside a 16 GB machine, and the 13.66 GiB model + 8 GiB
# pool overflows a default `-c` (measured 2026-09-16: `-c 0` -> kIOGPUCommandBufferCallbackErrorOOM
# at t=32.5 s; without those knobs -> OOM at t=16.7 s even at `-c 4096`).
#
# Usage:
#   bash Backup/run_spec_simple.sh                  # MTP arm (draft-mtp, n_max=3)
#   SPEC_TYPE=none bash Backup/run_spec_simple.sh   # the same tool with no speculation (baseline)
set -u
cd "$(dirname "$0")/.." || exit 1

PROFILE="${PROFILE:-prod25}"
SPEC_TYPE="${SPEC_TYPE:-draft-mtp}"
SPEC_N_MAX="${SPEC_N_MAX:-3}"
N_PREDICT="${N_PREDICT:-48}"
CTX="${CTX:-4096}"
PROMPT="${PROMPT:-請用繁體中文簡短說明巴黎為什麼是法國的首都。}"
TAG="${TAG:-spec_${SPEC_TYPE}}"

# --- env from the single source -------------------------------------------------------------
DUMP="$(CGC_SERVER_PROFILE="$PROFILE" CGC_DUMP_ENV=1 bash scripts/run_server.sh 2>/dev/null)"
MODEL="$(printf '%s\n' "$DUMP" | sed -n 's/^CGCENV MODEL *//p')"
if [ -z "$MODEL" ]; then echo "could not resolve MODEL for profile $PROFILE" >&2; exit 2; fi
while IFS= read -r line; do
    eval "export ${line#ENV }"
done <<<"$(printf '%s\n' "$DUMP" | /usr/bin/grep '^ENV ')"
echo "profile=$PROFILE  model=$(basename "$MODEL")  spec=$SPEC_TYPE n_max=$SPEC_N_MAX n=$N_PREDICT ctx=$CTX"
echo "env: N_CB=${CGC_N_CB:-} OA_ASYNC=${CGC_OA_ASYNC:-} DBUF=${CGC_DBUF:-} NO_PREFETCH=${CGC_NO_PREFETCH:-} WORKERS=${LLAMA_EXPERT_CACHE_WORKERS:-}"
echo "thermal before: $(python3 scripts/check/thermal_pressure.py)"

# `-c 0` means "whatever the model was trained with" (32768 here) and that alone OOMs; the profile's
# 4096 is also what the server serves with, so mirroring it is the comparable choice, not a workaround.
ARGS=(-m "$MODEL" -ngl 99 --load-mode none -t 8 -c "$CTX" -np 1 -fa auto
      -expert-cache "${CGC_EXPERT_CACHE_BYTES:-8589934592}"
      --cache-type-k q8_0 --cache-type-v q8_0
      -n "$N_PREDICT" -p "$PROMPT")

# SPEC_TYPE=off (or empty) must OMIT the flag entirely, not pass `--spec-type none`. Measured
# 2026-09-16: `--spec-type none` makes this example try to load a draft model from an empty path
# ("exactly one out metadata, path_model, and file must be defined" / "failed to load draft model,
# ''") and exit 1. The example's own no-spec arm is `params.speculative.types.empty()`.
if [ -n "$SPEC_TYPE" ] && [ "$SPEC_TYPE" != "off" ]; then
    ARGS+=(--spec-type "$SPEC_TYPE" --spec-draft-n-max "$SPEC_N_MAX")
fi

LOG="/tmp/${TAG}.log"
( time src/llama.cpp/build/bin/llama-speculative-simple "${ARGS[@]}" ) >"$LOG" 2>&1
rc=$?
echo "exit=$rc   log=$LOG"
/usr/bin/grep -E "error|assert|encoded |decoded |n_draft +|n_drafted|n_accept|accept +" "$LOG" | tail -12
echo "thermal after : $(python3 scripts/check/thermal_pressure.py)"
exit $rc
