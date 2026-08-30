# CGC Engine 鸿蒙手机版架构方案

> 目标设备: 华为 Mate 70 Pro (HarmonyOS NEXT, 12GB RAM)
> 版本: v1.0 | 日期: 2026-08-30

---

## 1. 设备画像

| 项目 | Mate 70 Pro (手机) | MateBook 14 (PC 对照) |
|---|---|---|
| **SoC** | 麒麟 9020 | 麒麟 9030 |
| **CPU** | ARMv8.2-A (Cortex-A720) | ARMv8.2-A (Maleoon 935) |
| **RAM** | 12GB LPDDR5 | 24GB LPDDR5 |
| **存储** | UFS 4.0 256-512GB | NVMe SSD 512GB |
| **OS** | HarmonyOS NEXT (微内核) | HarmonyOS NEXT (Linux 内核) |
| **GPU** | Maleoon 920 | Maleoon 935 (UMA) |
| **GPU 支持** | ❌ llama.cpp 不支持 | ❌ CPU only |
| **开发工具** | HDC (USB/WiFi) | HDC (USB/WiFi) |

## 2. 模型选型

### 2.1 内存预算

| 量化 | 文件大小 | 运行 RSS | Expert Cache | 系统占用 | 可行性 |
|---|---|---|---|---|---|
| **Qwen3-14B IQ4_XS** | **8.14 GB** | **~9 GB** | **1 GB** | **2 GB** | **✅ 推荐** |
| Qwen3-14B Q4_K_M | 9.00 GB | ~10 GB | 1 GB | 2 GB | ⚠️ 紧张 |
| Qwen3-14B IQ3_M | 6.9 GB | ~7.5 GB | 2 GB | 2 GB | ✅ 但质量下降 |
| Qwen3-14B Q3_K_S | 5.8 GB | ~6.5 GB | 2 GB | 2 GB | ✅ 质量更低 |
| Qwen2.5-7B Q4_K_M | 4.4 GB | ~5 GB | 2 GB | 2 GB | ✅ 太小 |

### 2.2 推荐方案

**首选: Qwen3-14B-IQ4_XS (8.14 GB)**
- 质量接近 Q4_K_M，体积小 1GB
- 12GB RAM: 9GB 模型 + 1GB cache + 2GB 系统 = 12GB ✅
- 预估速度: **2-4 t/s** (CPU only, 12GB 带宽)

**备选: Qwen3-14B-Q3_K_S (5.8 GB)**
- 更多空间给 Expert Cache (4GB)
- 预估速度: **3-5 t/s** (cache 命中率更高)
- 质量损失约 5-10%

### 2.3 模型下载

```bash
# HuggingFace 下载 (需代理)
# Qwen3-14B-IQ4_XS (推荐)
huggingface-cli download bartowski/Qwen_Qwen3-14B-GGUF \
    Qwen3-14B-IQ4_XS.gguf --local-dir ./models

# 或 Qwen3-14B-Q3_K_S (备选)
huggingface-cli download bartowski/Qwen_Qwen3-14B-GGUF \
    Qwen3-14B-Q3_K_S.gguf --local-dir ./models
```

## 3. 编译环境

### 3.1 HarmonyOS NEXT NDK

**安装 DevEco Studio 5.0+** (必须)
- 下载: https://developer.huawei.com/consumer/cn/download/
- 安装后自动下载 HarmonyOS NEXT SDK + NDK
- NDK 路径: `~/AppData/Local/Huawei/Sdk/openharmony/<version>/toolchains/`

**NDK 关键组件:**
- `clang` / `clang++` — 交叉编译器 (aarch64)
- `cmake` — 构建系统
- `ninja` — 构建工具
- `sysroot` — HarmonyOS NEXT 系统头文件/库

### 3.2 交叉编译 (从 Windows/Mac 编译)

```bash
# 设置 NDK 路径 (需根据实际安装路径调整)
export HARMONY_NDK=~/AppData/Local/Huawei/Sdk/openharmony/5.0.3.xxx/toolchains
export CC=$HARMONY_NDK/llvm/bin/aarch64-unknown-linux-ohos-clang
export CXX=$HARMONY_NDK/llvm/bin/aarch64-unknown-linux-ohos-clang++

# 编译 llama.cpp (CGC fork)
cd src/llama.cpp
cmake -B build-harmony \
    -DCMAKE_C_COMPILER=$CC \
    -DCMAKE_CXX_COMPILER=$CXX \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_METAL=OFF \
    -DGGML_VULKAN=OFF \
    -DGGML_OPENCL=OFF \
    -DGGML_BLAS=OFF \
    -DGGML_ACCELERATE=OFF \
    -DGGML_CPU_REPACK=OFF \
    -DLLAMA_BUILD_EXAMPLES=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DMTP_SUPPORT=ON \
    -DCMAKE_C_FLAGS="-O3 -march=armv8.2-a -mtune=cortex-a720" \
    -DCMAKE_CXX_FLAGS="-O3 -march=armv8.2-a -mtune=cortex-a720"

cmake --build build-harmony -j8 --target llama-simple llama-server
```

### 3.3 编译注意事项

| 选项 | 值 | 原因 |
|---|---|---|
| `GGML_BLAS` | OFF | BLAS 会导致 IQ3/IQ4 输出乱码 |
| `GGML_ACCELERATE` | OFF | macOS 专用，鸿蒙不支持 |
| `GGML_CPU_REPACK` | OFF | IQ3 tensor 边界问题 |
| `GGML_METAL` | OFF | 手机 GPU 不支持 |
| `GGML_VULKAN` | OFF | 鸿蒙 NEXT 暂无 Vulkan |
| `MTP_SUPPORT` | ON | CGC MTP 加速 |
| `-march` | armv8.2-a | 麒麟 9020 指令集 |
| `-mtune` | cortex-a720 | 优化目标 |

## 4. CGC Engine 集成

### 4.1 手机端角色

```
┌─────────────────────────────────────────────┐
│             三平台算力池                      │
├─────────────┬───────────────┬───────────────┤
│  Mac M4     │  Windows 8GB  │  Mate 70 Pro  │
│  32GB RAM   │  MX250 2GB    │  12GB RAM     │
│  Metal GPU  │  CPU+GPU      │  CPU only     │
├─────────────┼───────────────┼───────────────┤
│  主 Prefill │  Decode       │  Decode       │
│  + 主 Decode│  + 小任务      │  + 轻量任务    │
│  25-29 t/s  │  1.4-3.7 t/s  │  2-4 t/s      │
└─────────────┴───────────────┴───────────────┘
```

**手机端职责:**
1. **Decode 节点** — 接收 Mac/PC 的 prefill hidden state，执行 decode
2. **轻量推理** — 短 prompt (<512 tok) 直接本地处理
3. **Expert Cache** — 缓存 hot experts，加速 MoE 推理
4. **PD 分离端点** — 暴露 `/v1/cgc/resume` 接口

### 4.2 edge_server.py 适配

```python
# 手机端启动命令
python3 edge_server.py \
    --binary ./llama-simple \
    --model ./models/Qwen3-14B-IQ4_XS.gguf \
    --ngl 0 \
    --port 8080 \
    --expert-cache 1GiB

# 环境变量
LLAMA_EXPERT_CACHE_ALLOW_NGL=1
LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0=1
LLAMA_EXPERT_CACHE_WORKERS=4
LLAMA_EXPERT_CACHE_BUDGET=1073741824  # 1GB
CGC_OA_ASYNC=1
CGC_N_CB=4
```

### 4.3 PD 分离流程

```
用户输入 → ComputeRouter
  ├─ Router 检测: 手机 12GB < 14B 模型 → 不适合本地 prefill
  ├─ 选择: 端云模式，Mac M4 做 prefill
  ├─ Mac M4: emit(prompt) → hidden_state [seq, 2816]
  ├─ MoT-h: 翻译 (2816 → 2048)
  └─ Mate 70 Pro: resume(hidden_state) → SSE decode stream
      → 2-4 t/s 解码
```

## 5. 部署流程

### 5.1 HDC 连接

```bash
# 1. 安装 DevEco Studio 5.0+
# 2. 手机开启开发者模式 + USB 调试
# 3. 连接 HDC
hdc list targets
hdc tconn <device-ip>:<port>

# 或 WiFi HDC
# 手机: 设置 → 开发者选项 → 无线调试 → 开启
hdc tconn <phone-ip>:<port>
```

### 5.2 文件传输

```bash
# 方式 1: HDC push
hdc file send ./llama-simple /data/local/tmp/
hdc file send ./Qwen3-14B-IQ4_XS.gguf /data/local/tmp/models/

# 方式 2: MTP (USB 文件传输)
# 手机下拉通知栏 → USB 连接 → 文件传输
# Windows: 直接复制文件到手机存储

# 方式 3: HTTP 下载
# 在手机上运行 HTTP server，从 PC 下载
```

### 5.3 运行

```bash
# SSH 到手机 (如果支持)
ssh root@<phone-ip>

# 或通过 HDC shell
hdc shell

# 运行 llama-simple
cd /data/local/tmp
chmod +x llama-simple
./llama-simple -m models/Qwen3-14B-IQ4_XS.gguf \
    -ngl 0 -t 6 -c 2048 \
    --no-mmap -n 128 \
    -p "The capital of France is"

# 运行 edge_server (PD 分离)
./llama-server -m models/Qwen3-14B-IQ4_XS.gguf \
    --host 0.0.0.0 --port 8080 \
    -ngl 0 -t 6 -c 2048 \
    --no-mmap -expert-cache 1073741824
```

## 6. 性能预估

### 6.1 基准测试 (12GB RAM, 麒麟 9020)

| 指标 | Qwen3-14B IQ4_XS | Qwen3-14B Q3_K_S |
|---|---|---|
| **Decode** | 2-4 t/s | 3-5 t/s |
| **Prefill (1K)** | 1-2 t/s | 1.5-2.5 t/s |
| **RSS** | ~9 GB | ~6.5 GB |
| **Expert Cache 命中率** | 60-80% | 80-95% |
| **首次加载** | 10-15s | 8-12s |

### 6.2 与 PC 对照

| 指标 | Mate 70 Pro (12GB) | MateBook (24GB) | 加速比 |
|---|---|---|---|
| Decode (14B) | 2-4 t/s | 5-8 t/s | 0.5x (手机慢) |
| Prefill (1K) | 1-2 t/s | 2-4 t/s | 0.5x |
| Expert Cache | 1GB budget | 4GB budget | 0.25x |

### 6.3 优化路径

| 路径 | 效果 | 难度 |
|---|---|---|
| 换 Q3_K_S 量化 | +25% 速度 | ⭐ |
| Expert Cache 调优 | +10-20% 命中率 | ⭐⭐ |
| PD 分离 (Mac prefill) | 10x prefill 加速 | ⭐⭐⭐ |
| 量化蒸馏 (14B→7B) | 2x 速度 | ⭐⭐⭐⭐ |

## 7. 与现有代码整合

| 文件 | 改动 | 说明 |
|---|---|---|
| `deploy-harmonyos/harmonyos/build.sh` | 修改 | 添加手机端编译选项 |
| `deploy-harmonyos/harmonyos/run.sh` | 修改 | 适配手机端参数 |
| `CGC-main/cgc_engine/pd/edge_server.py` | 修改 | 手机端轻量化 |
| `CGC-main/cgc_engine/pd/router.py` | 修改 | 添加手机节点 |
| `AIOS/harmonyos/README.md` | 新增 | 手机端部署文档 |

## 8. 下一步

### Phase 1: 环境搭建 (1-2天)
- [ ] 安装 DevEco Studio 5.0+
- [ ] 配置 HDC 连接 Mate 70 Pro
- [ ] 下载 Qwen3-14B IQ4_XS GGUF

### Phase 2: 编译部署 (2-3天)
- [ ] 交叉编译 llama.cpp for HarmonyOS NEXT
- [ ] 部署 binary + 模型到手机
- [ ] 基准测试 (decode/prefill 速度)

### Phase 3: CGC 集成 (3-5天)
- [ ] 适配 edge_server.py for 手机端
- [ ] 配置 PD 分离 (Mac prefill → 手机 decode)
- [ ] Expert Cache 调优
- [ ] 端到端联调

### Phase 4: 优化 (持续)
- [ ] 量化蒸馏 (14B→7B)
- [ ] Expert Cache 策略优化
- [ ] 多节点负载均衡

---

> 基于 CGC-main/cgc_engine/pd/ 已有基础设施，复用 Mac/Windows/鸿蒙PC 的 PD 分离框架。
> 手机端作为 decode 节点，Mac M4 作为 prefill 节点，实现端云协同。
