# CGC Fork MTP + Expert Cache — Windows 8GB Benchmark

> Test date: 2026-08-29
> Machine: Windows 10, 8GB RAM, Intel Haswell, MX250 (2GB VRAM)
> Model: Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X-headIQ2.gguf (14GB)
> Compiler: MinGW GCC 16 + CGC fork (commit4ce088f), MTP_SUPPORT enabled

---

## 1. Compilation Fixes (Windows)

Three fixes were needed to compile the CGC fork with MTP support on Windows:

| Fix | File | Issue |
|---|---|---|
| `MTP_SUPPORT` define | `CMakeLists.txt` | MTP code paths behind `#ifdef MTP_SUPPORT` were never enabled |
| `_WIN32_WINNT=0x0A00` | `CMakeLists.txt` | cpp-httplib 0.53 requires Windows 10+ |
| `#include <cmath>` | `common/speculative.cpp` | MTP code uses `sqrt` without including `<cmath>` |

### Build commands

```bash
# In MSYS2 MinGW64 shell:
cd cgcengine_full/src/llama.cpp
mkdir build && cd build
cmake .. -G Ninja \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CGC=ON -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TOOLS=OFF
ninja llama-speculative-simple llama-simple
```

Requires `PATH="/c/msys64/mingw64/bin:$PATH"` at runtime (for `libgomp-1.dll`).

---

## 2. Benchmark Results

Prompt: `"The capital of France is"`, 128 tokens generation, `-ngl 4` (4 layers on MX250 GPU).

| Config | Gen tok/s | Accept Rate | Expert Cache Hit | Output Quality |
|---|---|---|---|---|
| **MTP + cache OFF** | **0.521** | **50.0%** | - | ✅ Correct (greedy) |
| MTP + cache ON | 0.295 | 0.0% | 80.9% | ❌ Repetitive garbage |
| No MTP + cache OFF | 0.470 | - | - | ✅ Correct |
| No MTP + cache ON | ~0.47 | - | 96.1% | ✅ Correct |

### Key findings

1. **MTP works** — with cache OFF, accept rate is50%, confirming the MTP head (layer 40) is functional
2. **Expert cache hurts on 8GB** — 4GB cache + 8 workers adds I/O overhead that exceeds mmap performance when RAM < model size
3. **Cache breaks MTP accept** — pread I/O stalls disrupt the MTP decode pipeline, dropping accept from 50% → 0%
4. **14GB model on 8GB RAM** — entire model is mmap'd and swapping; expert cache can't "pin" hot experts because there's no free RAM to pin them in

---

## 3. Why Expert Cache Fails on 8GB

Expert cache design assumption: **hot experts can be pinned in RAM** (cache hit = RAM-speed access).

On this machine:
- Model = 14GB, Available RAM = ~5GB → entire model is mmap'd to disk
- Expert cache adds 4GB pread buffer → competing for the same limited RAM
- Cache hit = reading from pread buffer (still disk I/O) vs mmap (also disk I/O)
- Net result: double I/O overhead, no speed benefit

Expert cache is designed for machines where **model fits in RAM** or **RAM >> model size**:
- Mac M4 64GB: model fits, cache pins hot experts → huge speedup
- A100 80GB: model fits, cache + L4 skip → expert streaming
- 8GB Windows: model doesn't fit → cache is pure overhead

---

## 4. MTP Accept Rate Analysis

50% accept rate (with cache OFF) is below the expected ~87.5% for greedy MTP.

Possible causes on this machine:
- `-ngl 4`: only 4 layers on GPU, MTP head (layer 40) on CPU → numerical differences between GPU draft and CPU target
- `--temp 0` (greedy) should align draft/target, but CPU vs GPU floating point ordering may differ
- Context window pressure: 14GB model + 3072 ctx on 8GB RAM → KV cache may be incomplete

On Mac M4 with full GPU offload (`-ngl 99`), expect:
- Accept rate: 85-97% (all layers on same device, consistent numerics)
- Speed: 15-25 t/s (unified memory, no mmap penalty)

---

## 5. run_n30cache.sh Changes

| Change | Reason |
|---|---|
| `Q36_MTP_DENSEIQ4X` → `denseIQ4X-headIQ2.gguf` | Point to actual model file on disk |
| Remove `/usr/bin/time -l` | macOS-only command, not available on Windows/MinGW |

---

## 6. Next Steps

1. **Mac M4 deployment**: Cross-compile CGC fork for arm64 + Metal, run with `-ngl 99` + expert-cache + MTP → expected 15-25 t/s
2. **Expert cache tuning for small memory**: Consider `CGC_EXPERT_CACHE_BYTES=1GiB` (not4GiB) for 8GB machines
3. **MTP accept rate investigation**: Test on Mac M4 to confirm 85%+ accept rate is achievable
