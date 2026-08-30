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

### 2.2 推荐方案

**首选: Qwen3-14B-IQ4_XS (8.14 GB)**
- 质量接近 Q4_K_M，体积小 1GB
- 12GB RAM: 9GB 模型 + 1GB cache + 2GB 系统 = 12GB ✅
- 预估速度: **2-4 t/s** (CPU only, 12GB 带宽)

## 3. 编译环境

### 3.1 HarmonyOS NEXT NDK

**安装 DevEco Studio 5.0+** (必须)
- 下载: https://developer.huawei.com/consumer/cn/download/
- NDK 路径: `~/AppData/Local/Huawei/Sdk/openharmony/<version>/toolchains/`

### 3.2 交叉编译

```bash
export HARMONY_NDK=~/AppData/Local/Huawei/Sdk/openharmony/5.0.3.xxx/toolchains
cd deploy-harmonyos/phone
./build_phone.sh /path/to/llama.cpp-source
```

### 3.3 编译注意事项

| 选项 | 值 | 原因 |
|---|---|---|
| `GGML_BLAS` | OFF | BLAS 会导致 IQ4 输出乱码 |
| `GGML_ACCELERATE` | OFF | macOS 专用 |
| `GGML_CPU_REPACK` | OFF | IQ3 tensor 边界问题 |
| `GGML_METAL` | OFF | 手机 GPU 不支持 |
| `MTP_SUPPORT` | ON | CGC MTP 加速 |
| `-march` | armv8.2-a | 麒麟 9020 指令集 |

## 4. CGC Engine 集成

### 4.1 手机端角色

```
┌─────────────────────────────────────────────┐
│             四平台算力池                      │
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

### 4.2 PD 分离流程

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

```bash
# 1. HDC 连接
hdc tconn <phone-ip>:<port>

# 2. 部署文件
./deploy_phone.sh

# 3. 运行
hdc shell
cd /data/local/tmp/cgc
./run_phone.sh -m models/Qwen3-14B-IQ4_XS.gguf -n 128 -p "Hello"
```

## 6. 性能预估

| 指标 | Qwen3-14B IQ4_XS | Qwen3-14B Q3_K_S |
|---|---|---|
| **Decode** | 2-4 t/s | 3-5 t/s |
| **Prefill (1K)** | 1-2 t/s | 1.5-2.5 t/s |
| **RSS** | ~9 GB | ~6.5 GB |
| **Expert Cache 命中率** | 60-80% | 80-95% |

## 7. 下一步

- [ ] 安装 DevEco Studio 5.0+
- [ ] 配置 HDC 连接 Mate 70 Pro
- [ ] 下载 Qwen3-14B IQ4_XS GGUF
- [ ] 交叉编译 llama.cpp for HarmonyOS NEXT
- [ ] 部署到手机并测速
- [ ] 端云联调 (Mac prefill → 手机 decode)
