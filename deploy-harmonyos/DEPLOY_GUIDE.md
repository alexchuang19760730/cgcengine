# llama.cpp Deployment Guide

**Version:** sync with current `dev` via `deploy-harmonyos/macos/build-macos.sh`
**Platforms:** macOS (M4 Max) + HarmonyOS (Kirin 9030 + Maleoon 935)

---

## Hardware Comparison

| Spec | Mac M4 Max | HarmonyOS MateBook 14 |
|------|-----------|----------------------|
| CPU | 10-core Apple (4.05 GHz) | 8-core Kirin 9030 (2.75 GHz) |
| GPU | Apple M4 (Metal) | Maleoon 935 (Vulkan 1.3) |
| Memory | 546 GB/s bandwidth | ~44 GB/s bandwidth |
| Backend | Metal GPU | **CPU-only** |
| Expected (Qwen3.6 IQ3_XXS) | **25-29 t/s** | **3-5 t/s** |

---

## macOS Deployment

### Pre-built Binary (Recommended)

```bash
# Sync the macOS bundle from the latest local dev build
./macos/build-macos.sh

# Run
./macos/run-macos.sh -m ~/models/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf

# Benchmark (with expert-cache A/B)
./benchmark-macos.sh ~/models/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf
```

### Build from Source

```bash
cd deploy-harmonyos/macos
./build-macos.sh
```

This produces a complete macOS bundle under `deploy-harmonyos/macos/`, including:

- `llama-simple`
- `llama-speculative-simple`
- `llama-bench`
- `llama-server`
- required runtime dylibs such as `libllama`, `libllama-common`, `libmtmd`, `libllama-server-impl`, and Metal/ggml libraries

---

## HarmonyOS Deployment

### One-Click Deploy

```bash
./deploy-to-harmonyos.sh alex@192.168.1.100
```

### Manual Build

```bash
scp -r deploy-harmonyos/ user@pc:~/llama-cpp/
ssh user@pc "cd ~/llama-cpp/harmonyos && ./build.sh"
ssh user@pc "cd ~/llama-cpp/harmonyos && ./run.sh -m model.gguf"
```

---

## Performance Tuning

### Mac M4 Max

| Parameter | Value | Notes |
|-----------|-------|-------|
| `-ngl` | 99 | Full GPU offload |
| `-t` | 8 | CPU threads |
| `-c` | 2048 | Context size |
| `--flash-attn` | ON | Faster attention |

### Kirin 9030

| Parameter | Value | Notes |
|-----------|-------|-------|
| `-ngl` | 0 | CPU-only |
| `-t` | 10 | 8 physical + SMT |
| `-c` | 2048 | Minimize pressure |
| `--no-mmap` | ON | Avoid page faults |
| `--flash-attn` | ON | Faster attention |

---

## Expert-Cache (Optional)

```bash
# 4GB budget
CGC_EXPERT_CACHE_BYTES=4294967296 llama-simple -m model.gguf -ngl 99

# With CGC segmented async
CGC_OA_ASYNC=1 CGC_EXPERT_CACHE_BYTES=4294967296 llama-simple -m model.gguf -ngl 99
```

---

*Version: current dev sync · Date: 2026-08-31*
