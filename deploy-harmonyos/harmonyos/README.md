# CGC Engine — HarmonyOS MateBook 14 Deployment

## Target Hardware
- **Device**: 鴻蒙 MateBook 14 (G4AU042K7)
- **SoC**: Kirin 9030 + Maleoon 935 (UMA)
- **RAM**: 32 GB unified memory
- **OS**: HarmonyOS NEXT (Linux kernel, aarch64)
- **Compiler**: clang/gcc (aarch64 native)

## Supported Models

| Model | File | Size | Active Params | Speed Est. | Quality |
|-------|------|------|--------------|------------|---------|
| **Qwen3.6-35B-A3B** | `Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf` | 12 GB | 3B | ~3-5 t/s | Good |
| **Qwen3.8 MoE** | `Qwen3.8-Whittle-MoE-27B-A17.8B-Q3_K_S.gguf` | 12.7 GB | 17.8B | ~0.5-1 t/s | Excellent |

- **Speed priority**: Qwen3.6 A3B (3B active, fast decode)
- **Quality priority**: Qwen3.8 MoE (17.8B active, dense-quality output)

## Quick Start

### 1. Build (on MateBook or cross-compile)

```bash
cd AIOS/harmonyos

# Build for Kirin 9030 (CPU-only, NEON)
./build.sh /path/to/llama.cpp-source

# Override options:
GGML_VULKAN=ON ./build.sh /path/to/llama.cpp-source  # if Vulkan support added
JOBS=12 ./build.sh /path/to/llama.cpp-source         # use all cores
```

### 2. Deploy Model Files

Copy GGUF files to `models/gguf/` on the MateBook:

```bash
# From Mac to MateBook
scp models/gguf/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf user@matebook:~/models/gguf/
scp models/gguf/Qwen3.8-Whittle-MoE-27B-A17.8B-Q3_K_S.gguf user@matebook:~/models/gguf/
```

### 3. Run

```bash
cd AIOS/harmonyos

# Qwen3.6 A3B (speed priority)
./run.sh -m qwen36 -n 128 -p "The capital of France is"

# Qwen3.8 MoE (quality priority)
./run.sh -m moe -n 128 -p "The capital of France is"

# Custom GGUF
./run.sh -m /path/to/model.gguf -n 128 -p "Hello"
```

### 4. Benchmark

```bash
# Benchmark both models
./benchmark.sh

# Qwen3.6 only (faster)
./benchmark.sh --qwen36

# MoE only
./benchmark.sh --moe
```

## Expert-Cache Configuration

The CGC expert-cache is enabled by default. Key env vars:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_EXPERT_CACHE_ENABLE` | 1 | Enable expert cache |
| `LLAMA_EXPERT_CACHE_BUDGET` | 4GB | Pool budget (bytes) |
| `LLAMA_EXPERT_CACHE_WORKERS` | 8 | I/O workers |
| `LLAMA_EXPERT_CACHE_ALLOW_NGL` | 1 | Allow GPU layers with cache |
| `LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0` | 1 | Skip layer 0 in pool |

### Memory Budget (32 GB machine)

| Config | Model | Cache | System | Free |
|--------|-------|-------|--------|------|
| Qwen3.6 A3B | 12 GB | 4 GB | 4 GB | 12 GB ✅ |
| MoE Q3_K_S | 12.7 GB | 4 GB | 4 GB | 11.3 GB ✅ |

## Build Flags (CGC Fork)

```
Metal=OFF, Vulkan=OFF, OpenCL=OFF
BLAS=OFF (MUST — causes IQ3 garbled output)
Accelerate=OFF
CPU_REPACK=OFF (MUST — IQ3 tensor boundary)
OpenMP=OFF
MTP_SUPPORT=ON (CGC expert-cache + MTP)
Arch: -march=armv8.2-a -mtune=cortex-a720 (Kirin 9030 NEON/SVE)
```

## Expected Performance (32 GB Kirin 9030)

| Metric | Qwen3.6 A3B | MoE Q3_K_S |
|--------|------------|------------|
| Decode | ~3-5 t/s | ~0.5-1 t/s |
| Prefill (1K) | ~1-2 t/s | ~0.2-0.5 t/s |
| RSS | ~8 GB | ~8 GB |
| Hit rate | 85-100% | 60-80% |

### Why MoE is slower despite smaller model

MoE has **6x more active parameters per token** (17.8B vs 3B), requiring 6x more memory bandwidth per decode step. This is a hardware physics limitation, not a software issue.

## Troubleshooting

### OOM (Out of Memory)
- Reduce cache budget: `LLAMA_EXPERT_CACHE_BUDGET=2147483648 ./run.sh ...`
- Or use `--no-mmap` (already default)

### Slow decode
- Check hit rate in log output
- Ensure `LLAMA_EXPERT_CACHE_ALLOW_NGL=1` is set
- Try lower thread count: `-t 4`

### Garbled output (IQ3 models)
- Ensure `GGML_BLAS=OFF` and `GGML_ACCELERATE=OFF` in build
- Use `--no-mmap` to prevent cold-page storms
