#!/bin/bash
# Validate that the macOS deploy bundle can be rebuilt from the current HEAD
# and that the synced executables still start successfully.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUNDLE_DIR="${BUNDLE_DIR:-$SCRIPT_DIR}"
RUN_BUILD="${RUN_BUILD:-1}"
CHECK_OTOOL="${CHECK_OTOOL:-1}"

BUILD_SCRIPT="$SCRIPT_DIR/build-macos.sh"
LLAMA_SERVER="$BUNDLE_DIR/llama-server"
LLAMA_SIMPLE="$BUNDLE_DIR/llama-simple"

echo "=========================================="
echo "deploy-harmonyos macOS bundle validation"
echo "=========================================="
echo "  Root:       $ROOT_DIR"
echo "  Bundle dir: $BUNDLE_DIR"
echo "  Run build:  $RUN_BUILD"
echo "  Check otool:$CHECK_OTOOL"
echo "=========================================="

if [ "$RUN_BUILD" = "1" ]; then
    "$BUILD_SCRIPT"
fi

for exe in "$LLAMA_SERVER" "$LLAMA_SIMPLE"; do
    if [ ! -x "$exe" ]; then
        echo "FAIL: missing executable: $exe" >&2
        exit 1
    fi
done

server_version="$("$LLAMA_SERVER" --version)"
set +e
simple_help_raw="$("$LLAMA_SIMPLE" --help 2>&1)"
simple_help_rc=$?
set -e
simple_help="$(printf '%s\n' "$simple_help_raw" | sed -n '1,4p')"

if ! grep -qi 'usage' <<< "$simple_help_raw"; then
    echo "FAIL: llama-simple did not print usage/help output" >&2
    exit 1
fi

echo "--- llama-server --version ---"
echo "$server_version"
echo "--- llama-simple --help (head) ---"
echo "$simple_help"
echo "--- llama-simple --help exit code: $simple_help_rc ---"

if [ "$CHECK_OTOOL" = "1" ]; then
    libs="$(otool -L "$LLAMA_SERVER")"
    echo "--- llama-server dylibs ---"
    echo "$libs" | sed -n '1,20p'

    for needle in \
        '@rpath/libllama-server-impl.dylib' \
        '@rpath/libllama-common.0.dylib' \
        '@rpath/libmtmd.0.dylib' \
        '@rpath/libllama.0.dylib'; do
        if ! grep -q "$needle" <<< "$libs"; then
            echo "FAIL: llama-server missing dylib linkage: $needle" >&2
            exit 1
        fi
    done
fi

echo "PASS: deploy-harmonyos macOS bundle rebuild + startup validation succeeded."
