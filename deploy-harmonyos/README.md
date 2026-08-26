# llama.cpp Dual-Platform Deploy Package

**Version:** 27c4e90 (expert-cache L4 + CGC_EARLY)

## Quick Start

### macOS (M4 Max)
```bash
./benchmark-macos.sh ~/models/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf
```

### HarmonyOS PC (Kirin 9030)
```bash
./deploy-to-harmonyos.sh alex@192.168.1.100
# On device: ./harmonyos/run.sh -m model.gguf
```

### Universal
```bash
./run-universal.sh -m model.gguf -p "Hello"
```

## Files

| File | Description |
|------|-------------|
| DEPLOY_GUIDE.md | Full deployment guide |
| benchmark-macos.sh | Mac M4 benchmark |
| deploy-to-harmonyos.sh | One-click deploy |
| macos/ | Pre-built macOS binaries |
| harmonyos/ | Build + run scripts for Kirin 9030 |
| src.tar.gz | Source code tarball |
