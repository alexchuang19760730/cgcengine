# CGC Engine 三平台架构方案

> 整合 cgcengine0823 fork × tb_loop agent harness × fastprefill 项目全流程
>
> 版本: v1.0 | 日期: 2026-08-29 | 作者: CGC Team

---

## 1. 项目全景

### 1.1 核心目标

用 **端侧 MoE 模型**（Qwen3.6-35B-A3B / Gemma4-26B-A4B）在 **Mac M4 / Windows / 鸿蒙 PC** 三平台上实现推理加速、Agent 能力、端侧训练、算力共享。

### 1.2 仓库结构

```
cgcengine0823/
├── src/llama.cpp/              # CGC fork (expert-cache + MTP + async GLU)
├── CGC-main/                   # CGC engine Python 框架
├── deploy-harmonyos/           # 三平台部署包
│   ├── macos/                  # arm64 binary (29 t/s on M4)
│   ├── harmonyos/              # 麒麟9030 CPU-only
│   └── windows/                # MinGW/Clang build
├── scripts/                    # run_n30cache.sh 等启动脚本
├── moeexpert/                  # CGC 技术报告 + benchmark
├── docs/                       # 本文件 + 架构文档
└── CGC_Phase2/                 # Agent framework 设计
```

---

## 2. CGC Fork 架构

### 2.1 Fork vs Upstream 差异

| 组件 | Upstream llama.cpp | CGC Fork (4ce088f) |
|---|---|---|
| ggml-backend.cpp | 标准 MoE dispatch | +CGC expert-cache pool |
| llama-expert-cache.cpp | 无 | 新增: expert streaming + eviction ring |
| llama-context.cpp | 标准 KV cache | +L4 skip layer0 + prefetch |
| speculative-simple | Eagle draft | +draft-mtp (MTP) |

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
| CGC_OA_ASYNC | 1 | 异步 expert overlap |
| CGC_GLU_FUSED_DOWN | 1 | 融合 GLU down proj (+6.5%) |
| -expert-cache BYTES | off | CLI 启动 (env 无效) |

---

## 3. 三平台编译部署

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

### 3.3 鸿蒙 PC (麒麟9030)

```bash
./deploy-harmonyos/build.sh
# CPU-only, 预估: 3-5 t/s
```

### 3.4 预编译二进制

| 平台 | 编译器 | 速度 |
|---|---|---|
| macOS arm64 | Clang 19 | 29 t/s |
| Windows pre-built | Clang 19 | 3.7 t/s |
| Windows CGC (GCC 16) | GCC | 1.7 t/s |
| Windows CGC (Clang 22) | Clang | 1.4 t/s |

---

## 4. tb_loop Agent Harness

### 4.1 架构

```
batch_run.sh → local_rehearsal.py → CodebuffApiAgent
    → POST /v1/chat/completions → llama-server
    → 解析 reply → 提取命令 → 执行 → 收集观察
```

### 4.2 支持的模型格式

| 模型 | 输出格式 | Parser |
|---|---|---|
| Codebuff (deepseek-v4) | DSML 标签 | ✅ |
| Qwen3.6 (本地) | markdown code block | ✅ fallback |
| Qwen3.6 (本地) | 纯文本命令 | ✅ fallback |

---

## 5. CGC Expert-Cache 调优实测

### 5.1 测试环境
- HUAWEI MateBook (i5-10210U, 8GB RAM, MX250 2GB)
- 模型: Qwen3.6-35B-A3B IQ3_XXS (13GB)

### 5.2 实测数据

| 配置 | 编译器 | Expert-Cache | Gen t/s |
|---|---|---|---|
| pre-built vanilla | Clang 19 | 无 | **3.7** |
| CGC fork (all OFF) | GCC 16 | 无 | **1.70** |
| CGC fork (all OFF) | Clang 22 | 无 | **1.42** |
| CGC fork | GCC 16 | 512M | **1.66** |
| CGC fork | GCC 16 | 4G | **0.90** |

### 5.3 关键发现

1. Expert-cache 在 8GB 机器上无收益 (内存不足 swap 拖慢)
2. CGC patches 有运行时开销 (env 全关仍 ~2x 慢)
3. Expert-cache 设计给 32GB+ 内存 + 多 GPU 场景

### 5.4 推荐配置

| 机器 | RAM | Expert-Cache | 预估速度 |
|---|---|---|---|
| Windows 8GB | 8GB | 关闭 | 3-4 t/s |
| Windows 16GB+ | 16GB+ | 2-4GiB | 5-8 t/s |
| Mac M4 Max | 32GB | 8GiB | 15-25 t/s |
| Mac M4 Pro | 48GB | 8GiB | 20-30 t/s |

---

## 6. Post-Training 落地路径

### 6.1 DeepSeek-V4-Flash 启示

| Benchmark | V4-Flash | V4-Flash-0731 | 增量 |
|---|---|---|---|
| Terminal Bench 2.1 | 61.8 | **82.7** | +20.9 |
| DeepSWE | 7.3 | **54.4** | +47.1 |
| SWE-bench Verified | ~69 | **79.0** | +10 |

### 6.2 三层接入架构

```
Layer 3: verl PPOTrainer (GRPO)
Layer 2: Rollout Worker (prime-agent / DirectLlama)
Layer 1: Verifier (SWE-bench + Ruff)
```

### 6.3 训练配方
- 算法: AsyncGRPO (group=8/16)
- Verifier: SWE-bench Verified + Ruff
- 数据: Terminal-Bench 成功轨迹 SFT
- 框架: verl (HybridFlow)

---

## 7. TTT-Layer 技术路径

### 7.1 核心区分

> TTT ≠ TTA: 没有双层元预训练的推理梯度更新，全部是自适应。

| 方案 | 说明 | 适用 |
|---|---|---|
| TTT-Layer (Karpathy) | 双层元预训练 + 梯度更新 | 科研基线 |
| StreamingVLM + MoT-T | 流式推理 + 动态 token | 生产部署 |

### 7.2 集成策略
- 主链路: StreamingVLM + MoT-T
- 高阶基线: TTT-Layer 作为科研对比
- 消融实验: 可回滚

---

## 8. Whittle-MoE Dense→MoE 转换

### 8.1 四阶段流程

```
Stage 1: Upcycling (复制 FFN → N 专家)
Stage 2: Router-Healing (frozen experts + STE)
Stage 3: Anti-Loop KD (dense 做 teacher)
Stage 4: v2.1 平衡 (8% 循环率目标)
```

### 8.2 关键参数
- 母体: Qwen3.8-27B (dense)
- 专家: 64 per FFN layer
- 激活: 17.8B (64%)
- 目标: 达到 dense 所有指标

### 8.3 开源工具链

| 步骤 | 工具 | 状态 |
|---|---|---|
| Upcycling | smolMoELM-custom | ✅ |
| Router 训练 | DenseMixer | ✅ |
| Anti-Loop 蒸馏 | 无官方代码 | ⚠️ 需自行实现 |

---

## 9. Exo + iroh-net 算力共享

### 9.1 架构分层

```
应用层: Agent Harness (tb_loop)
推理层: CGC Harness-Router (PD 分离 + Expert Streaming)
网络层: iroh-net P2P (设备发现 + 路由)
设备层: Mac / PC / Android (Exo 模型分片)
```

### 9.2 PD 分离 6 种部署选项
1. 单机单卡
2. 单机多卡
3. 双机 PD 分离
4. Exo 模型分片
5. iroh-net P2P
6. 混合模式

---

## 10. 验收里程碑 W1→W5+

| 周次 | 目标 | 结果 |
|---|---|---|
| W1 | Ollama + Qwen3 + prime-agent | ✅ PASS |
| W2 | ACP/JSON + DSH rollout | ✅ PASS |
| W3 | PrimeAgentRolloutWorker | ✅ PASS (6/6) |
| W4 | SWEBenchVerifier | ✅ PASS (17/17) |
| W5 | 端到端 dry-run + 35B | ✅ PASS (10/10) |
| W5+ | verl PPOTrainer 适配 | ✅ 数据契约 (33/33) |

---

## 11. 性能基线与瓶颈

| 平台 | 模型 | 速度 | 瓶颈 |
|---|---|---|---|
| Mac M4 Max | Qwen3.6-35B Q4 | 25-29 t/s | 无 |
| Windows 8GB | Qwen3.6-35B IQ3 | 3-4 t/s | CPU 带宽 |
| 鸿蒙 PC | Qwen3.6-35B IQ3 | 3-5 t/s | CPU only |

### 优化路径
| 路径 | 效果 | 难度 |
|---|---|---|
| 换小模型 (7B-IQ4) | 10-15 t/s | ⭐ |
| 升级 16GB+ RAM | 5-8 t/s | ⭐⭐ |
| M4 部署 | 15-25 t/s | ⭐⭐ |

---

## 12. 附录: 环境变量速查

```bash
# Expert-Cache (CLI 参数, env 无效)
llama-server -expert-cache 4GiB ...

# 运行时 env
LLAMA_EXPERT_CACHE_ALLOW_NGL=1
LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0=1
LLAMA_EXPERT_CACHE_WORKERS=8
CGC_OA_ASYNC=1
CGC_GLU_FUSED_DOWN=1
CGC_N_CB=8
CGC_WAKE_POLL_US=15
```

---

> 整合自: fastprefill W1-W5+ 验收报告、Phase0 方案、PostTraining 白皮书、TTT-Layer 白皮书、Exo-iroh-net 架构方案、Whittle-MoE 白皮书、cgcengine0823 三平台实测数据。
