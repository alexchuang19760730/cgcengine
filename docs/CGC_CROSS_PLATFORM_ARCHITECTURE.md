# CGC Engine 三平台架构方案

> 整合 cgcengine0823 fork × tb_loop agent harness × fastprefill 项目全流程
>
> 版本: v1.1 | 日期: 2026-08-30 | 作者: CGC Team
> 更新: 新增鸿蒙手机版 (Mate 70 Pro, 12GB RAM)

---

## 1. 项目全景

### 1.1 核心目标

用 **端侧 MoE 模型**（Qwen3.6-35B-A3B / Gemma4-26B-A4B / Qwen3-14B）在 **Mac M4 / Windows / 鸿蒙 PC / 鸿蒙手机** 四平台上实现推理加速、Agent 能力、端侧训练、算力共享。

### 1.2 四平台设备矩阵

| 平台 | 设备 | SoC | RAM | GPU | 模型 | 速度 | 角色 |
|---|---|---|---|---|---|---|---|
| **Mac M4** | Mac Mini/Studio | M4 Max | 32-64GB | Metal | Qwen3.6-35B Q4 | 25-29 t/s | 主 Prefill + 主 Decode |
| **Windows** | MateBook 14 | i5-10210U + MX250 | 8GB | CPU+GPU | Qwen3.6-35B IQ3 | 1.4-3.7 t/s | Decode + 小任务 |
| **鸿蒙 PC** | MateBook 14 | 麒麟9030 | 24GB | CPU only | Qwen3.6-35B IQ3 | 3-5 t/s | Decode + Expert Cache |
| **鸿蒙手机** | Mate 70 Pro | 麒麟9020 | 12GB | CPU only | Qwen3-14B IQ4_XS | 2-4 t/s | Decode + 轻量推理 |

### 1.3 仓库结构

```
cgcengine0823/
├── src/llama.cpp/              # CGC fork (expert-cache + MTP + async GLU)
├── CGC-main/                   # CGC engine Python 框架
├── deploy-harmonyos/           # 四平台部署包
│   ├── macos/                  # arm64 binary (29 t/s on M4)
│   ├── harmonyos/              # 鸿蒙 PC (麒麟9030, 24GB)
│   ├── phone/                  # 鸿蒙手机 (Mate 70 Pro, 12GB) ← NEW
│   └── windows/                # MinGW/Clang build
├── scripts/                    # run_n30cache.sh 等启动脚本
├── moeexpert/                  # CGC 技术报告 + benchmark
├── docs/                       # 本文件 + 架构文档
└── CGC_Phase2/                 # Agent framework 设计
```

---

## 2. CGC Fork 架构

### 2.1 Fork vs Upstream 差异

| 组件 | Upstream llama.cpp | CGC Fork (9d06e18) |
|---|---|---|
| ggml-backend.cpp | 标准 MoE dispatch | +CGC expert-cache pool |
| llama-expert-cache.cpp | 无 | 新增: expert streaming + eviction ring |
| llama-context.cpp | 标准 KV cache | +L4 skip layer0 + prefetch + MTP bit-identical |
| speculative-simple | Eagle draft | +draft-mtp (MTP) + galloc overlap guard |
| simple.cpp | 基础推理 | +CGC expert-cache 启动参数 |

### 2.2 Expert-Cache 工作原理

```
Router → top-k experts → 查 cache
  Cache HIT  → 直接用缓存权重 (fast)
  Cache MISS → mmap 读盘 → 填入 cache
  Bounded Pool (4GiB default) + LRU eviction
  L4 skip: blk.0 不进 pool
```

### 2.3 关键环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| LLAMA_EXPERT_CACHE_ALLOW_NGL | 1 | 允许 GPU offload + cache |
| LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0 | 1 | blk.0 排除 pool |
| LLAMA_EXPERT_CACHE_WORKERS | 8 | pool 并行 workers |
| LLAMA_EXPERT_CACHE_BUDGET | 4294967296 | 4GB pool budget |
| CGC_OA_ASYNC | 1 | 异步 expert overlap |
| CGC_GLU_FUSED_DOWN | 1 | 融合 GLU down proj (+6.5%) |
| CGC_MMV_FUSE | 1 | 融合 MMV dispatch |
| -expert-cache BYTES | off | CLI 启动 (env 无效) |

---

## 3. 四平台编译部署

### 3.1 macOS (M4 Max)

```bash
cmake -B build -DGGML_METAL=ON -DGGML_CGC=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
./build/bin/llama-server -m model.gguf -ngl 99 -expert-cache 8GiB -c 32768
# 预估: 25-29 t/s
```

### 3.2 Windows (x86_64)

```bash
# MSYS2 Clang (推荐)
cmake -B build-clang -G Ninja \
  -DCMAKE_C_COMPILER=clang.exe -DCMAKE_CXX_COMPILER=clang++.exe \
  -DCMAKE_SYSROOT=/clang64 \
  -DGGML_CGC=ON -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON \
  -DCMAKE_CXX_FLAGS="-D_WIN32_WINNT=0x0A00"
# 预估: 3-4 t/s (8GB RAM)
```

**Windows 编译注意事项:**
- httplib 需 0.53.0 (从 b9553 源码复制)
- CreateFile2 → CreateFileW patch
- miniaudio.h + stb_image.h 需手动下载
- MTP_SUPPORT=ON 需加 `<cmath>` include

### 3.3 鸿蒙 PC (麒麟9030, 24GB)

```bash
cd deploy-harmonyos/harmonyos
./build.sh /path/to/llama.cpp-source
# CPU-only, 预估: 3-5 t/s
# Expert Cache: 4GB, 命中率 85-100%
```

### 3.4 鸿蒙手机 (Mate 70 Pro, 12GB) ← NEW

```bash
# 交叉编译 (从 Windows/Mac)
export HARMONY_NDK=~/AppData/Local/Huawei/Sdk/openharmony/<version>/toolchains
cd deploy-harmonyos/phone
./build_phone.sh /path/to/llama.cpp-source

# 部署到手机
./deploy_phone.sh

# 运行
./run_phone.sh -m models/Qwen3-14B-IQ4_XS.gguf -n 128 -p "Hello"
# 预估: 2-4 t/s
# Expert Cache: 1GB, 命中率 60-80%
```

**手机端编译选项:**
```
Metal=OFF, Vulkan=OFF, OpenCL=OFF
BLAS=OFF (MUST — IQ3/IQ4 乱码)
Accelerate=OFF
CPU_REPACK=OFF (IQ3 tensor 边界)
OpenMP=OFF
MTP_SUPPORT=ON
Arch: -march=armv8.2-a -mtune=cortex-a720 (麒麟9020)
```

### 3.5 预编译二进制

| 平台 | 编译器 | 速度 | 状态 |
|---|---|---|---|
| macOS arm64 | Clang 19 | 29 t/s | ✅ 生产 |
| Windows pre-built | Clang 19 | 3.7 t/s | ✅ 生产 |
| Windows CGC (GCC 16) | GCC | 1.7 t/s | ✅ 生产 |
| Windows CGC (Clang 22) | Clang | 1.4 t/s | ✅ 生产 |
| 鸿蒙 PC | Clang/GCC | 3-5 t/s | ✅ 生产 |
| 鸿蒙手机 | Clang (NDK) | 2-4 t/s | ⚠️ 待编译 |

---

## 4. PD 分离与算力共享

### 4.1 四种 PD 模式

| 模式 | Prefill 节点 | Decode 节点 | 适用场景 |
|---|---|---|---|
| **端云** | Mac M4 | Windows/鸿蒙 | 长 prompt 加速 |
| **端端** | Mac A | Mac B | 双机协作 |
| **纯云** | Cloud GPU | Cloud GPU | 大模型全量推理 |
| **纯端** | 本地 GPU | 本地 CPU | 离线/隐私场景 |

### 4.2 手机端 PD 分离流程

```
用户输入 → ComputeRouter
  ├─ Router 检测: 手机 12GB < 14B 模型 → 不适合本地 prefill
  ├─ 选择: 端云模式，Mac M4 做 prefill
  ├─ Mac M4: emit(prompt) → hidden_state [seq, 2816]
  ├─ MoT-h: 翻译 (2816 → 2048)
  └─ Mate 70 Pro: resume(hidden_state) → SSE decode stream
      → 2-4 t/s 解码
```

### 4.3 edge_server.py 支持

```bash
# Mac M4 (prefill 节点)
python3 edge_server.py \
    --binary src/llama.cpp/build/bin/llama-simple \
    --model models/gguf/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf \
    --ngl 99 --no-mmap --port 1234

# Mate 70 Pro (decode 节点)
python3 edge_server.py \
    --binary ./llama-server \
    --model ./models/Qwen3-14B-IQ4_XS.gguf \
    --ngl 0 --port 8080 --expert-cache 1073741824
```

---

## 5. tb_loop Agent Harness

### 5.1 架构

```
batch_run.sh → local_rehearsal.py → CodebuffApiAgent
    → POST /v1/chat/completions → llama-server
    → 解析 reply → 提取命令 → 执行 → 收集观察
```

### 5.2 支持的模型格式

| 模型 | 输出格式 | Parser |
|---|---|---|
| Codebuff (deepseek-v4) | DSML 标签 | ✅ |
| Qwen3.6 (本地) | markdown code block | ✅ fallback |
| Qwen3.6 (本地) | 纯文本命令 | ✅ fallback |
| Qwen3-14B (手机) | 同上 | ✅ fallback |

---

## 6. CGC Expert-Cache 调优实测

### 6.1 测试环境

| 平台 | 设备 | RAM | 模型 |
|---|---|---|---|
| Windows | MateBook 14 | 8GB | Qwen3.6-35B IQ3 |
| Mac M4 | Mac Mini | 32GB | Qwen3.6-35B IQ3 |
| 鸿蒙 PC | MateBook 14 | 24GB | Qwen3.6-35B IQ3 |
| 鸿蒙手机 | Mate 70 Pro | 12GB | Qwen3-14B IQ4_XS (待测) |

### 6.2 实测数据 (Windows 8GB)

| 配置 | 编译器 | Expert-Cache | Gen t/s |
|---|---|---|---|
| pre-built vanilla | Clang 19 | 无 | **3.7** |
| CGC fork (all OFF) | GCC 16 | 无 | **1.70** |
| CGC fork (all OFF) | Clang 22 | 无 | **1.42** |
| CGC fork | GCC 16 | 512M | **1.66** |
| CGC fork | GCC 16 | 4G | **0.90** |

### 6.3 实测数据 (Mac M4 + 端云联调)

| 次数 | decode_tps | wall | 说明 |
|---|---|---|---|
| 第1次 | 1.71 | 29.3s | 冷启动 |
| 第2次 | 16.64 | 7.2s | 热缓存 |
| 第3次 | 15.99 | 31.8s | 稳定 |
| 第4次 | 26.83 | 21.7s | 预热中 |
| 第5次 | 27.02 | 36.2s | 完全预热 |
| 第6次 | 27.01 | 36.2s | 稳定 27 t/s |

### 6.4 关键发现

1. Expert-cache 在 8GB 机器上无收益 (内存不足 swap 拖慢)
2. Expert-cache 在 Mac M4 64GB 上效果显著 (16→27 t/s)
3. MTP accept rate: cache OFF 50%, cache ON 0% (8GB 限制)
4. MTP bit-identical fix: spec vs non-spec 输出完全一致

### 6.5 推荐配置

| 机器 | RAM | Expert-Cache | 预估速度 |
|---|---|---|---|
| Windows 8GB | 8GB | 关闭 | 3-4 t/s |
| Windows 16GB+ | 16GB+ | 2-4GiB | 5-8 t/s |
| Mac M4 Max | 32GB | 8GiB | 15-25 t/s |
| Mac M4 Pro | 48GB | 8GiB | 20-30 t/s |
| 鸿蒙 PC 24GB | 24GB | 4GiB | 3-5 t/s |
| 鸿蒙手机 12GB | 12GB | 1GiB | 2-4 t/s |

---

## 7. Post-Training 落地路径

### 7.1 DeepSeek-V4-Flash 启示

| Benchmark | V4-Flash | V4-Flash-0731 | 增量 |
|---|---|---|---|
| Terminal Bench 2.1 | 61.8 | **82.7** | +20.9 |
| DeepSWE | 7.3 | **54.4** | +47.1 |
| SWE-bench Verified | ~69 | **79.0** | +10 |

### 7.2 三层接入架构

```
Layer 3: verl PPOTrainer (GRPO)
Layer 2: Rollout Worker (prime-agent / DirectLlama)
Layer 1: Verifier (SWE-bench + Ruff)
```

### 7.3 训练配方
- 算法: AsyncGRPO (group=8/16)
- Verifier: SWE-bench Verified + Ruff
- 数据: Terminal-Bench 成功轨迹 SFT
- 框架: verl (HybridFlow)

---

## 8. TTT-Layer 技术路径

### 8.1 核心区分

> TTT ≠ TTA: 没有双层元预训练的推理梯度更新，全部是自适应。

| 方案 | 说明 | 适用 |
|---|---|---|
| TTT-Layer (Karpathy) | 双层元预训练 + 梯度更新 | 科研基线 |
| StreamingVLM + MoT-T | 流式推理 + 动态 token | 生产部署 |

### 8.2 集成策略
- 主链路: StreamingVLM + MoT-T
- 高阶基线: TTT-Layer 作为科研对比
- 消融实验: 可回滚

---

## 9. Whittle-MoE Dense→MoE 转换

### 9.1 四阶段流程

```
Stage 1: Upcycling (复制 FFN → N 专家)
Stage 2: Router-Healing (frozen experts + STE)
Stage 3: Anti-Loop KD (dense 做 teacher)
Stage 4: v2.1 平衡 (8% 循环率目标)
```

### 9.2 关键参数
- 母体: Qwen3.8-27B (dense)
- 专家: 64 per FFN layer
- 激活: 17.8B (64%)
- 目标: 达到 dense 所有指标

### 9.3 开源工具链

| 步骤 | 工具 | 状态 |
|---|---|---|
| Upcycling | smolMoELM-custom | ✅ |
| Router 训练 | DenseMixer | ✅ |
| Anti-Loop 蒸馏 | 无官方代码 | ⚠️ 需自行实现 |

---

## 10. Exo + iroh-net 算力共享

### 10.1 架构分层

```
应用层: Agent Harness (tb_loop)
推理层: CGC Harness-Router (PD 分离 + Expert Streaming)
网络层: iroh-net P2P (设备发现 + 路由)
设备层: Mac / PC / 鸿蒙PC / 鸿蒙手机 (Exo 模型分片)
```

### 10.2 PD 分离 6 种部署选项
1. 单机单卡
2. 单机多卡
3. 双机 PD 分离
4. Exo 模型分片
5. iroh-net P2P
6. 混合模式

---

## 11. 验收里程碑 W1→W5+

| 周次 | 目标 | 结果 |
|---|---|---|
| W1 | Ollama + Qwen3 + prime-agent | ✅ PASS |
| W2 | ACP/JSON + DSH rollout | ✅ PASS |
| W3 | PrimeAgentRolloutWorker | ✅ PASS (6/6) |
| W4 | SWEBenchVerifier | ✅ PASS (17/17) |
| W5 | 端到端 dry-run + 35B | ✅ PASS (10/10) |
| W5+ | verl PPOTrainer 适配 | ✅ 数据契约 (33/33) |
| W6 | 端云联调 (Mac+Windows) | ✅ 27 t/s 稳定 |
| W6+ | MTP bit-identical fix | ✅ spec=non-spec |
| W7 | 鸿蒙手机 CGC Engine | ⏳ 进行中 |

---

## 12. 性能基线与瓶颈

| 平台 | 模型 | 速度 | 瓶颈 |
|---|---|---|---|
| Mac M4 Max | Qwen3.6-35B Q4 | 25-29 t/s | 无 |
| Windows 8GB | Qwen3.6-35B IQ3 | 3-4 t/s | CPU 带宽 |
| 鸿蒙 PC 24GB | Qwen3.6-35B IQ3 | 3-5 t/s | CPU only |
| 鸿蒙手机 12GB | Qwen3-14B IQ4 | 2-4 t/s | CPU + RAM |

### 优化路径
| 路径 | 效果 | 难度 |
|---|---|---|
| 换小模型 (7B-IQ4) | 10-15 t/s | ⭐ |
| 升级 16GB+ RAM | 5-8 t/s | ⭐⭐ |
| M4 部署 | 15-25 t/s | ⭐⭐ |
| PD 分离 (Mac prefill) | 10x prefill 加速 | ⭐⭐⭐ |

---

## 13. 鸿蒙手机部署指南

### 13.1 前置条件

- DevEco Studio 5.0+ (提供 HarmonyOS NEXT NDK + HDC)
- Mate 70 Pro (12GB RAM, HarmonyOS NEXT)
- Qwen3-14B IQ4_XS GGUF (8.14 GB)

### 13.2 编译

```bash
export HARMONY_NDK=~/AppData/Local/Huawei/Sdk/openharmony/<version>/toolchains
cd deploy-harmonyos/phone
./build_phone.sh /path/to/llama.cpp-source
```

### 13.3 部署

```bash
# HDC 连接
hdc tconn <phone-ip>:<port>

# 推送文件
./deploy_phone.sh
```

### 13.4 运行

```bash
# 本地推理
./run_phone.sh -m models/Qwen3-14B-IQ4_XS.gguf -n 128 -p "Hello"

# PD 分离 (Mac prefill → 手机 decode)
./llama-server -m models/Qwen3-14B-IQ4_XS.gguf \
    --host 0.0.0.0 --port 8080 -ngl 0 -t 6 -c 2048 \
    --no-mmap -expert-cache 1073741824
```

### 13.5 性能预估

| 指标 | Qwen3-14B IQ4_XS | Qwen3-14B Q3_K_S |
|---|---|---|
| Decode | 2-4 t/s | 3-5 t/s |
| Prefill (1K) | 1-2 t/s | 1.5-2.5 t/s |
| RSS | ~9 GB | ~6.5 GB |
| Expert Cache | 1GB budget | 2GB budget |

---

## 14. 附录: 环境变量速查

```bash
# Expert-Cache (CLI 参数, env 无效)
llama-server -expert-cache 4GiB ...

# 运行时 env
LLAMA_EXPERT_CACHE_ALLOW_NGL=1
LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0=1
LLAMA_EXPERT_CACHE_WORKERS=8
LLAMA_EXPERT_CACHE_BUDGET=4294967296
CGC_OA_ASYNC=1
CGC_GLU_FUSED_DOWN=1
CGC_MMV_FUSE=1
CGC_N_CB=8
CGC_WAKE_POLL_US=15
```

---

> 整合自: fastprefill W1-W5+ 验收报告、Phase0 方案、PostTraining 白皮书、TTT-Layer 白皮书、Exo-iroh-net 架构方案、Whittle-MoE 白皮书、cgcengine0823 四平台实测数据。
