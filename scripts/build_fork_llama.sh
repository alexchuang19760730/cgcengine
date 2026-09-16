#!/bin/bash
set -e

# ============================================================
# CGC Fork: llama.cpp build script
# ============================================================
# CRITICAL: GGML_BLAS=OFF is required — BLAS (Accelerate) causes
# IQ3_XXS/IQ2_S garbled output. Build must match build-flat
# configuration (dc605b4 era) for correctness.
# ============================================================
#
# [CGC 2026-09-16] PROFILE
#
# This script and the live `src/llama.cpp/build` directory used to disagree, and the disagreement
# was invisible: the defaults here are the *flat/prod* build (no server, no tests) while the
# devserver needs both, so `bash scripts/build_fork_llama.sh` on a devserver would silently replace a
# working build dir with one that has no llama-server — and REBUILD=1 (the default) does `rm -rf
# build` first, so the previous one is gone. Measured drift before this change, 6 rows:
#
#   flag                 build_fork_llama.sh   live build dir
#   GGML_BLAS            OFF                   ON      <- policy, aligned 2026-09-16
#   GGML_ACCELERATE      OFF                   ON      <- policy, aligned 2026-09-16
#   GGML_CPU_REPACK      OFF                   ON      <- policy, aligned 2026-09-16
#   GGML_OPENMP          OFF                   ON      <- policy, aligned 2026-09-16
#   LLAMA_BUILD_SERVER   OFF                   ON      <- devserver REQUIRES ON, keep the difference
#   LLAMA_BUILD_TESTS    OFF                   ON      <- devserver REQUIRES ON, keep the difference
#
# The four GGML_* rows are the already-ruled policy (BLAS/ACCELERATE garble IQ3_XXS, CPU_REPACK
# steps over the IQ3 tensor boundary, OPENMP is off for stability) and are now the same on both
# sides. The last two are NOT drift to be eliminated: run_server.sh ships llama-server and the D5
# gate needs the test binaries. So the fix is not "make them match" — it is to make the choice
# explicit and checkable:
#
#   CGC_BUILD_PROFILE=devserver bash scripts/build_fork_llama.sh   # SERVER=ON TESTS=ON
#   CGC_BUILD_PROFILE=prod      bash scripts/build_fork_llama.sh   # SERVER=OFF TESTS=OFF (default)
#   bash scripts/build_fork_llama.sh --check                       # compare the LIVE build dir, no build
#
# The build dir also carries a stamp (build/CGC_BUILD_PROFILE.txt) so which profile produced a given
# binary is answerable after the fact, and REBUILD=1 refuses to wipe a dir stamped with a *different*
# profile (CGC_BUILD_FORCE=1 overrides).

FORK_DIR="$(cd "$(dirname "$0")/../src/llama.cpp" && pwd)"
BUILD_DIR="${FORK_DIR}/build"

# --- Profile (defaults for the two knobs that legitimately differ per consumer) ---
#
# [CGC 2026-09-16] A bare `--check` should answer the question that actually has a useful answer:
# "does this build dir match the profile it was built as?"  Defaulting the REQUEST to prod made
# `--check` report 4 rows of drift on every devserver dir -- one real (profile) and three
# intentional (SERVER/APP/TESTS) -- which trains the reader to ignore the whole report. So:
#   - an explicit CGC_BUILD_PROFILE always wins;
#   - a check-only run with no request checks against the dir's own stamp, and says so;
#   - a real BUILD still defaults to prod (unchanged).
_CHECK_ONLY_REQ=0
[ "${1:-}" = "--check" ]        && _CHECK_ONLY_REQ=1
[ "${CHECK_ONLY:-0}" = "1" ]    && _CHECK_ONLY_REQ=1
_PROFILE_FROM_STAMP=""
if [ -z "${CGC_BUILD_PROFILE+x}" ] && [ "$_CHECK_ONLY_REQ" = "1" ]; then
    _stamped="$(awk -F= '$1=="profile"{print $2; exit}' "$BUILD_DIR/CGC_BUILD_PROFILE.txt" 2>/dev/null)"
    if [ -n "$_stamped" ]; then
        CGC_BUILD_PROFILE="$_stamped"
        _PROFILE_FROM_STAMP="$_stamped"
    fi
fi
CGC_BUILD_PROFILE="${CGC_BUILD_PROFILE:-prod}"
case "$CGC_BUILD_PROFILE" in
    prod)      _DEF_SERVER=OFF; _DEF_TESTS=OFF ;;
    devserver) _DEF_SERVER=ON;  _DEF_TESTS=ON  ;;
    *)
        echo "error: unknown CGC_BUILD_PROFILE '$CGC_BUILD_PROFILE' (expected: prod | devserver)" >&2
        exit 2
        ;;
esac

# --- Configurable flags (override via env) ---
BUILD_TYPE="${BUILD_TYPE:-Release}"
GGML_METAL="${GGML_METAL:-ON}"
GGML_BLAS="${GGML_BLAS:-OFF}"           # MUST be OFF — causes IQ3 garbled output
GGML_ACCELERATE="${GGML_ACCELERATE:-OFF}" # MUST be OFF — same issue
GGML_CPU_REPACK="${GGML_CPU_REPACK:-OFF}" # MUST be OFF —踩 IQ3 tensor boundary
GGML_OPENMP="${GGML_OPENMP:-OFF}"        # OFF for stability
# LLAMA_CURL is NOT passed any more: this tree deprecated it (CMakeLists.txt:186,
# `llama_option_depr(WARNING LLAMA_CURL)`), so it no longer creates a cache entry and `-DLLAMA_CURL=OFF`
# was only making the script's flag set differ from the live build dir for a knob that does nothing.
LLAMA_BUILD_SERVER="${LLAMA_BUILD_SERVER:-$_DEF_SERVER}"
# 2026-08-28: app/（unified binary llama）硬依賴 llama-server-impl；SERVER=OFF 時不存在
# → ld: library 'llama-server-impl' not found（build 尾段失敗）。SERVER=OFF 時一併關 app。
LLAMA_BUILD_APP="${LLAMA_BUILD_APP:-$([ "$LLAMA_BUILD_SERVER" = "ON" ] && echo ON || echo OFF)}"
LLAMA_BUILD_TESTS="${LLAMA_BUILD_TESTS:-$_DEF_TESTS}"
JOBS="${JOBS:-$(sysctl -n hw.ncpu)}"
REBUILD="${REBUILD:-1}"
# MTP isolation: compile the MTP staging API (release_context in
# common/speculative.cpp) into libllama-common.  Without this flag the
# MTP-only code is excluded (non-MTP builds).  The llama-simple binary itself
# is still compiled without -DMTP_SUPPORT (see build_prod_binary.sh).
MTP_SUPPORT="${MTP_SUPPORT:-ON}"

CACHE="$BUILD_DIR/CMakeCache.txt"
STAMP="$BUILD_DIR/CGC_BUILD_PROFILE.txt"

cache_get() { # $1 = CMakeCache key (without the :TYPE suffix)
    [ -f "$CACHE" ] || return 0
    awk -F= -v k="$1" '$1 ~ ("^" k ":") { sub("^" k ":", "", $1); print $2; exit }' "$CACHE"
}

stamp_get() { # $1 = key in the stamp file
    [ -f "$STAMP" ] || return 0
    awk -F= -v k="$1" '$1 == k { sub("^" k "=", "", $0); print $0; exit }' "$STAMP"
}

# Compare the LIVE build dir against what this invocation is asking for. Prints one row per flag so
# a drift is a list, not a verdict, and returns non-zero if anything disagrees.
profile_check() {
    local fails=0
    if [ ! -f "$CACHE" ]; then
        echo "  FAIL ${CACHE} does not exist — nothing is built there"
        return 1
    fi

    local prev_profile
    prev_profile="$(stamp_get profile)"
    if [ -z "$prev_profile" ]; then
        echo "  NOTE no stamp ($STAMP) — a build dir from before 2026-09-16; comparing the cache only"
    elif [ "$prev_profile" != "$CGC_BUILD_PROFILE" ]; then
        echo "  FAIL stamped profile=${prev_profile} vs requested ${CGC_BUILD_PROFILE}"
        fails=$((fails + 1))
    fi

    check_one() { # $1 = cache key, $2 = expected
        local got
        got="$(cache_get "$1")"
        if [ "$got" = "$2" ]; then
            printf '  OK   %-24s %s\n' "$1" "$got"
        else
            printf '  FAIL %-24s got=%s want=%s\n' "$1" "${got:-<absent>}" "$2"
            fails=$((fails + 1))
        fi
    }

    echo "  profile: requested=${CGC_BUILD_PROFILE} stamped=${prev_profile:-none}"
    if [ -n "$_PROFILE_FROM_STAMP" ]; then
        echo "  note:    no CGC_BUILD_PROFILE was requested, so this checks against the dir's own"
        echo "           stamp ('$CGC_BUILD_PROFILE'). To ask about a specific profile, pass it:"
        echo "           CGC_BUILD_PROFILE=prod bash scripts/build_fork_llama.sh --check"
    fi
    check_one CMAKE_BUILD_TYPE      "$BUILD_TYPE"
    check_one GGML_METAL            "$GGML_METAL"
    check_one GGML_BLAS             "$GGML_BLAS"
    check_one GGML_ACCELERATE       "$GGML_ACCELERATE"
    check_one GGML_CPU_REPACK       "$GGML_CPU_REPACK"
    check_one GGML_OPENMP           "$GGML_OPENMP"
    check_one LLAMA_BUILD_SERVER    "$LLAMA_BUILD_SERVER"
    check_one LLAMA_BUILD_APP       "$LLAMA_BUILD_APP"
    check_one LLAMA_BUILD_TESTS     "$LLAMA_BUILD_TESTS"

    # CMAKE_CXX_FLAGS carries -DMTP_SUPPORT; not a boolean so compare by substring.
    local cxx; cxx="$(cache_get CMAKE_CXX_FLAGS)"
    if [ "$MTP_SUPPORT" = "ON" ]; then
        case "$cxx" in
            *-DMTP_SUPPORT*) printf '  OK   %-24s %s\n' CMAKE_CXX_FLAGS "$cxx" ;;
            *) printf '  FAIL %-24s got=%s want=-DMTP_SUPPORT\n' CMAKE_CXX_FLAGS "${cxx:-<absent>}"; fails=$((fails + 1)) ;;
        esac
    fi

    if [ "$fails" -eq 0 ]; then
        echo "  => PASS: the live build dir matches profile '${CGC_BUILD_PROFILE}'"
        return 0
    fi
    echo "  => FAIL: ${fails} row(s) disagree"
    return 1
}

if [ "${1:-}" = "--check" ] || [ "${CHECK_ONLY:-0}" = "1" ]; then
    echo "=========================================="
    echo "CGC Fork Build -- CHECK ONLY (no build)"
    echo "=========================================="
    echo "  Build dir:  $BUILD_DIR"
    profile_check
    exit $?
fi

echo "=========================================="
echo "CGC Fork Build"
echo "=========================================="
echo "  Source:     $FORK_DIR"
echo "  Build dir:  $BUILD_DIR"
echo "  Profile:    $CGC_BUILD_PROFILE  (server=$LLAMA_BUILD_SERVER tests=$LLAMA_BUILD_TESTS)"
echo "  Build type: $BUILD_TYPE"
echo "  Metal:      $GGML_METAL"
echo "  BLAS:       $GGML_BLAS  (MUST be OFF for IQ3_XXS)"
echo "  Accelerate: $GGML_ACCELERATE  (MUST be OFF for IQ3_XXS)"
echo "  Repack:     $GGML_CPU_REPACK  (MUST be OFF for IQ3_XXS)"
echo "  OpenMP:     $GGML_OPENMP"
echo "  Jobs:       $JOBS"
echo "=========================================="

cd "$FORK_DIR"

if [ "$REBUILD" = "1" ] && [ -d "$BUILD_DIR" ]; then
    prev_profile="$(stamp_get profile)"
    if [ -n "$prev_profile" ] && [ "$prev_profile" != "$CGC_BUILD_PROFILE" ] && [ "${CGC_BUILD_FORCE:-0}" != "1" ]; then
        echo "error: $BUILD_DIR was built as profile '$prev_profile', you asked for '$CGC_BUILD_PROFILE'." >&2
        echo "       REBUILD=1 would rm -rf it first, so the '$prev_profile' binaries would be gone." >&2
        echo "       Build into it anyway:   CGC_BUILD_FORCE=1 $0" >&2
        echo "       Or check the current state first:  $0 --check" >&2
        exit 2
    fi
fi

if [ "$REBUILD" = "1" ]; then
    rm -rf "$BUILD_DIR"
fi

# Only add the define when ON; "-DMTP_SUPPORT=OFF" would still define the macro.
if [ "$MTP_SUPPORT" = "ON" ]; then
    CXX_FLAGS_MTP="-DMTP_SUPPORT"
else
    CXX_FLAGS_MTP=""
fi

# Preseed SVE/SME check results: this host cannot execute SVE/SME instructions
# (the check_cxx_source_runs test binary hangs instead of trapping), so force them
# OFF to let configure finish. Apple Silicon has no SVE/SME anyway.
cmake -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DGGML_METAL="$GGML_METAL" \
    -DGGML_BLAS="$GGML_BLAS" \
    -DGGML_ACCELERATE="$GGML_ACCELERATE" \
    -DGGML_CPU_REPACK="$GGML_CPU_REPACK" \
    -DGGML_OPENMP="$GGML_OPENMP" \
    -DLLAMA_BUILD_SERVER="$LLAMA_BUILD_SERVER" \
    -DLLAMA_BUILD_APP="$LLAMA_BUILD_APP" \
    -DLLAMA_BUILD_TESTS="$LLAMA_BUILD_TESTS" \
    -DGGML_MACHINE_SUPPORTS_sve=OFF \
    -DGGML_MACHINE_SUPPORTS_sme=OFF \
    -DCMAKE_CXX_FLAGS="$CXX_FLAGS_MTP"

cmake --build "$BUILD_DIR" -j"$JOBS"

# Stamp *after* a successful build, so the file never describes a dir that is half-configured.
{
    echo "profile=$CGC_BUILD_PROFILE"
    echo "built_at=$(date '+%F %T %z')"
    echo "git_head=$(git -C "$FORK_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "git_dirty_files=$(git -C "$FORK_DIR" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
    echo "BUILD_TYPE=$BUILD_TYPE"
    echo "GGML_METAL=$GGML_METAL"
    echo "GGML_BLAS=$GGML_BLAS"
    echo "GGML_ACCELERATE=$GGML_ACCELERATE"
    echo "GGML_CPU_REPACK=$GGML_CPU_REPACK"
    echo "GGML_OPENMP=$GGML_OPENMP"
    echo "LLAMA_BUILD_SERVER=$LLAMA_BUILD_SERVER"
    echo "LLAMA_BUILD_APP=$LLAMA_BUILD_APP"
    echo "LLAMA_BUILD_TESTS=$LLAMA_BUILD_TESTS"
    echo "MTP_SUPPORT=$MTP_SUPPORT"
} > "$STAMP"

echo ""
echo "=========================================="
echo "Build complete: $BUILD_DIR/bin/"
echo "  profile: $CGC_BUILD_PROFILE   stamp: $STAMP"
echo "=========================================="
ls -la "$BUILD_DIR/bin/llama-simple" 2>/dev/null || echo "WARNING: llama-simple not found"
ls -la "$BUILD_DIR/bin/llama-server" 2>/dev/null || echo "NOTE: llama-server not found (profile=prod)"
