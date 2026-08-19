#!/bin/bash
# Build script for C expert streaming module
# Using MinGW GCC 16.2

GCC=D:/alex/toolchains/winlibs-gcc162/mingw64/bin/gcc.exe
SRCDIR="D:/alex/flashkv0516/CGC-main/cgc_engine/cpp/expert_streaming"
BUILDDIR="$SRCDIR/build_gcc_c"

mkdir -p "$BUILDDIR"

COMMON_FLAGS="-O2 -Wall -Wextra -std=c11 -I$SRCDIR"

echo "=== Building cgc_gguf_lite ==="
$GCC $COMMON_FLAGS -c "$SRCDIR/cgc_gguf_lite.c" -o "$BUILDDIR/cgc_gguf_lite.o" 2>&1

echo "=== Building cgc_expert_streamer ==="
$GCC $COMMON_FLAGS -c "$SRCDIR/cgc_expert_streamer.c" -o "$BUILDDIR/cgc_expert_streamer.o" 2>&1

echo "=== Building cgc_expert_streamer_gguf ==="
$GCC $COMMON_FLAGS -c "$SRCDIR/cgc_expert_streamer_gguf.c" -o "$BUILDDIR/cgc_expert_streamer_gguf.o" 2>&1

echo "=== Building cgc_pd_scheduler ==="
$GCC $COMMON_FLAGS -c "$SRCDIR/cgc_pd_scheduler.c" -o "$BUILDDIR/cgc_pd_scheduler.o" 2>&1

echo "=== Building cgc_expert_compute ==="
$GCC $COMMON_FLAGS -c "$SRCDIR/cgc_expert_compute.c" -o "$BUILDDIR/cgc_expert_compute.o" 2>&1

echo "=== Creating static library ==="
$GCC -ar rcs "$BUILDDIR/libcgc_expert_streamer.a" \
    "$BUILDDIR/cgc_gguf_lite.o" \
    "$BUILDDIR/cgc_expert_streamer.o" \
    "$BUILDDIR/cgc_expert_streamer_gguf.o" \
    "$BUILDDIR/cgc_pd_scheduler.o" \
    "$BUILDDIR/cgc_expert_compute.o" 2>&1

echo "=== Building test_cgc_expert_streamer ==="
$GCC $COMMON_FLAGS "$SRCDIR/test_cgc_expert_streamer.c" \
    -o "$BUILDDIR/test_cgc_streamer.exe" \
    -L"$BUILDDIR" -lcgc_expert_streamer 2>&1

echo "=== Building test_cgc_pd_scheduler ==="
$GCC $COMMON_FLAGS "$SRCDIR/test_cgc_pd_scheduler.c" \
    -o "$BUILDDIR/test_cgc_pd.exe" \
    -L"$BUILDDIR" -lcgc_expert_streamer 2>&1

echo "=== Building test_cgc_gguf_integration ==="
$GCC $COMMON_FLAGS "$SRCDIR/test_cgc_gguf_integration.c" \
    -o "$BUILDDIR/test_cgc_gguf.exe" \
    -L"$BUILDDIR" -lcgc_expert_streamer 2>&1

echo "=== Build complete ==="
ls -la "$BUILDDIR/"
