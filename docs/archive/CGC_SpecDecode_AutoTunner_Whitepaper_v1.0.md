# CGC 投机 Decode 统一 IR + AutoTunner 技术白皮书

> **版本**: v1.0  
> **日期**: 2026-07-25  
> **作者**: CGC Team  
> **适用模型**: Qwen3-VL-2B, DeepSeek-V4-Flash, 任意 HuggingFace/MLX 模型  
> **适用后端**: Mac MLX, GPU PyTorch, Cloud SGLang



---

## 1. 摘要

本白皮书描述 CGC (Compute Graph Compiler) 投机 decode 统一 IR 框架的设计与实现。该框架通过 **一份配置 JSON** 跨三个推理后端（MLX / PyTorch / SGLang），由 **AutoTunner** 自动检测硬件并选择最优参数，由 **SeamlessSwitcher** 在运行时动态切换云↔本地，两者合并集成到 **十步流水线** 的 Step 11.5。

### 核心指标

| Backend                | Model       |  最优 N |    tok/s   |  投机 speedup  | --auto |
| ---------------------- | ----------- | :---: | :--------: | :----------: | :----: |
| MLX (Mac M4 16GB)      | Qwen3-VL-2B | 16→32 |    53.2    |     2.0x     |    ✓   |
| PyTorch (RTX PRO 5000) | Qwen3-VL-2B |  4→2  |    71.5    |     1.94x    |    ✓   |
| SGLang (Cloud)         | Qwen3-VL-2B |  N/A  |     155    | 1.0x (plain) |    ✓   |
| SGLang (Cloud)         | V4-Flash    |  N/A  | 29 (NEXTN) |     1.07x    |    ✓   |

---

## 2. 架构总览

```
configs/spec_decode_default.json (一份配置 JSON)
    ↓
app/shared/spec_decode_ir.py
    ├── SpecDecodeConfig       (跨后端配置, 可保存/加载 JSON)
    ├── SpecDecodeBackend       (抽象接口)
    │   ├── MLXBackend          (Mac: chain + eagle)
    │   ├── PyTorchBackend      (GPU: chain, 统一 model_loader)
    │   └── SGLangBackend       (Cloud: HTTP API)
    ├── AutoTunner              (自动检测硬件 + 最优参数 + 运行时自适应)
    │   ├── HardwareProfile     (硬件 profile 库)
    │   ├── detect()            (检测 backend → 返回最优 profile)
    │   ├── apply_model_params() (检测模型类型 → 自动设置 sglang 启动参数)
    │   ├── generate_sglang_command() (生成完整 sglang 启动命令)
    │   ├── runtime_tune()      (accept rate 驱动 N 动态调整)
    │   └── auto_bench()        (baseline → chain → EAGLE → 选最优)
    └── create_backend()        (工厂函数)
    ↓
app/cli/cgc.py (十步流水线)
    ├── Step 1-10: 硬件检测 + 路由决策 + 模型分发
    ├── Step 11:   SeamlessSwitcher 初始化 (后台监控: 内存/网络/decode)
    └── Step 11.5: AutoTunner 集成 (切换时自动调优目标后端)
```

---

## 3. 三后端实现

### 3.1 MLX Backend (Mac)

**适用场景**: Mac M4 16GB, 统一内存, 本地推理

| 参数                   | 值                                        | 说明                                 |
| -------------------- | ---------------------------------------- | ---------------------------------- |
| N (num_draft_tokens) | 16                                       | draft forward 慢 (3-5ms), 需 batch 多 |
| mode                 | chain                                    | mlx_lm stream_generate (链式 draft)  |
| dtype                | int4                                     | 省内存, 0.5B ~300MB                   |
| draft_model          | mlx-community/Qwen2.5-0.5B-Instruct-4bit | 标准 LLM, 不依赖 target hidden          |
| baseline             | 26.8 tok/s                               | 无投机                                |
| best                 | 53.2 tok/s (2.0x)                        | accept 50-85%                      |

**chain mode**: 用 mlx_lm 原生 `stream_generate(draft_model=..., num_draft_tokens=N)`  
**eagle mode**: 用 `eagle_tree_search.py` (客户端 EAGLE, flat verify, accept 80-95%)

### 3.2 PyTorch Backend (GPU)

**适用场景**: RTX PRO 5000 72GB, 高算力 GPU

| 参数          | 值                                   | 说明                            |
| ----------- | ----------------------------------- | ----------------------------- |
| N           | 4                                   | draft forward 快 (~1ms), N 小最优 |
| mode        | chain                               | 链式 draft speculative          |
| dtype       | bfloat16                            | 精度高, 0.5B ~1GB                |
| draft_model | /data2/models/Qwen2.5-0.5B-Instruct | 本地路径, 避免网络                    |
| baseline    | 36.9 tok/s                          | 无投机                           |
| best        | 71.5 tok/s (1.94x)                  | accept 45-50%                 |

**模型加载**: target + draft 都用 `model_loader.load_base_model()` 统一加载, 零硬编码 (VL→AutoModelForImageTextToText, 纯文本→AutoModelForCausalLM)

### 3.3 SGLang Backend (Cloud)

**适用场景**: 云端 GPU, 通过 HTTP API

| 参数                | V4-Flash                   | Qwen3-VL-2B |
| ----------------- | -------------------------- | ----------- |
| cuda-graph        | ✓ (CGC_ENABLE_ORTHO_KDA=0) | ✓ (默认)      |
| speculative       | NEXTN (内置 MTP)             | 无 (plain)   |
| mem-fraction      | 0.7 (OOM 修复)               | 0.88        |
| cuda-graph-max-bs | 16 (OOM 修复)                | 256         |
| tp-size           | 8                          | 1           |
| baseline          | 27 tok/s                   | 155 tok/s   |
| best              | 29 tok/s (+7%, NEXTN)      | 155 tok/s   |

---

## 4. AutoTunner 自适应调优

### 4.1 硬件检测

```
AutoTunner.detect(backend) → HardwareProfile
  - MLX: 检测 Apple Silicon → N=16, dtype=int4
  - PyTorch: 检测 GPU 名称 → N=4, dtype=bfloat16
  - SGLang: Cloud GPU → N=4, server-side
```

### 4.2 模型参数自动设置

```
AutoTunner.apply_model_params(config, model_path)
  ↓ 读 config.json
  ↓ deepseek_v4 → V4-Flash profile
  ↓ qwen3_vl → 默认 profile
```

**V4-Flash 自动设置**:

- `CGC_ENABLE_ORTHO_KDA=0` (关闭 CGC instrumentation, 修复 cuda-graph 崩溃)
- cuda-graph 开启 (不加 `--disable-cuda-graph`)
- `--speculative-algorithm NEXTN` (用内置 MTP, `num_nextn_predict_layers=1`)
- `--mem-fraction-static 0.7` (OOM 修复)
- `--cuda-graph-max-bs 16` (OOM 修复)
- `--tp-size 8`

### 4.3 运行时自适应

```
AutoTunner.runtime_tune(config, accept_rate, tps)
  - accept < 30% → N 减半 (draft 预测差, 减少浪费)
  - accept > 60% → N 增大 (draft 预测好, 多 draft)
  - 30-60% → 保持
```

### 4.4 策略选择

```
AutoTunner.auto_bench(backend, prompts, model_path)
  1. 跑 chain speculative (自适应 N)
  2. 如果 chain speedup < 1.5x → 尝试 EAGLE (只对 mlx/pytorch)
  3. 选 chain vs EAGLE 中速度最快的
  4. runtime_tune: accept 驱动 N 动态调整
```

---

## 5. SeamlessSwitcher + AutoTunner 集成

### 5.1 十步流水线

|   Step   | 内容                          | 模块                 |
| :------: | --------------------------- | ------------------ |
|     1    | 系统检测                        | hardware_sensing   |
|     2    | CPU 检测                      | hardware_sensing   |
|     3    | 模型格式解析                      | -                  |
|     4    | 模型架构分析 (MoE/Dense)          | model_loader       |
|     5    | 内存水位扫描                      | hardware_sensing   |
|    5.5   | 算力等级检测                      | hardware_sensing   |
|     6    | 运算引擎路由 (OMLX/CUDA/ROCm)     | hardware_sensing   |
|     7    | 内存策略 (FlashMoE)             | -                  |
|    7.5   | PD/Layer-split 路由决策         | route_decision     |
|    7.6   | 模型分发决策                      | model_dispatcher   |
|    7.7   | MTP draft 同步                | model_dispatcher   |
|     8    | 上下文构建                       | -                  |
|     9    | 4D 感知矩阵上报                   | route_decision     |
|    10    | 磁盘空间检查                      | hardware_sensing   |
|    11    | SeamlessSwitcher 初始化        | seamless_switcher  |
| **11.5** | **AutoTunner 集成 (切换时自动调优)** | **spec_decode_ir** |

### 5.2 切换流程

```
SeamlessSwitcher 后台监控 (内存/网络/decode 速度)
  ↓ 触发切换 (如内存不足 → 本地切云)
  ↓
_on_switch_with_autotune(event)
  ↓ AutoTunner.get_optimal_config(new_backend, model_path)
  ↓
自动设置目标后端参数:
  - V4-Flash: CGC_ENABLE_ORTHO_KDA=0 + cuda-graph + NEXTN + mem 0.7
  - Qwen3-VL: chain speculative + 最优 N
  - Mac MLX: N=16, chain, int4
  - GPU PyTorch: N=4, chain, bfloat16
```

### 5.3 切换触发条件

| 触发               | 动作   | 原因     |
| ---------------- | ---- | ------ |
| 内存 < 1GB         | 本地→云 | 预防 OOM |
| 内存 > 3GB         | 云→本地 | 省成本    |
| RTT > 500ms      | 云→本地 | 可用性    |
| RTT < 200ms      | 本地→云 | 质量/速度  |
| decode < 5 tok/s | 本地→云 | 体验     |

---

## 6. V4-Flash 投机 decode 路线

### 6.1 技术限制

| 方案                    | 结果    | 根因                                        |
| --------------------- | ----- | ----------------------------------------- |
| 外接 EAGLE (0.5B)       | ❌     | hidden 896≠4096                           |
| 外接 EAGLE (Qwen3-1.7B) | ❌     | OOM + EAGLE 接口不兼容                         |
| sglang n-gram         | ❌     | eagle_topk>1 + page_size>1 需 flashinfer   |
| **原生 NEXTN (内置 MTP)** | **✅** | **num_nextn_predict_layers=1, hidden 匹配** |

### 6.2 战略决策

> 公开现成、开箱即用、能直接搭配 V4-Flash (hidden=4096) 的外部 EAGLE draft 模型不存在。当前唯一稳妥、可接入 CUDA Graph、兼容 SGLang、适配 V4-Flash MoE 的投机路线：优先采用原生内置 MTP / DSpark，放弃外接独立 EAGLE draft。

### 6.3 cuda-graph 修复

```
CGC_ENABLE_ORTHO_KDA=0 (关闭 CGC instrumentation)
  + 不加 --disable-cuda-graph (开启 cuda-graph)
  → cuda-graph 通! 27 tok/s (vs 12.8, 加速 2x)
```

**根因**: CGC instrumentation 的 `.item()/.cpu()` host-sync 导致 cuda-graph 崩溃。关闭 CGC 注入后纯 sglang, cuda-graph 正常。

### 6.4 NEXTN + cuda-graph 结果

| Prompt         | 纯 cuda-graph | NEXTN + cuda-graph | 变化   |
| -------------- | :----------: | :----------------: | ---- |
| photosynthesis |     27.5     |        29.4        | +7%  |
| exercise       |     26.3     |        27.5        | +5%  |
| cat story      |     27.3     |        19.2        | -30% |

- 技术性文本提升 5-7% (accept 高)
- 创造性文本退化 -30% (NEXTN overhead > accept 收益)
- 内置 MTP accept 28%, 收益有限

---

## 7. EAGLE Tree Search (客户端)

### 7.1 客户端 vs sglang EAGLE

|                | 客户端 EAGLE        | sglang EAGLE      |
| -------------- | ---------------- | ----------------- |
| draft model    | 标准 LLM (0.5B)    | EAGLE 训练的特殊 draft |
| hidden states  | draft 独立 forward | 共享 target hidden  |
| tree attention | flat verify (简化) | 真正 tree mask      |
| accept rate    | 80-95%           | 取决于 EAGLE draft   |
| 速度             | 较慢 (flat verify) | 更快 (tree mask)    |
| 状态             | 已实现, 已测试         | 未跑通 (需特殊 draft)   |

### 7.2 适用场景

- **Qwen3-VL-2B** (hidden=2048): 用 0.5B draft (hidden=896, 不匹配但独立 forward 不需匹配)
- **V4-Flash** (hidden=4096): 不适用 (无匹配 draft)

---

## 8. 统一 model_loader

### 8.1 设计

```python
from app.shared.model_loader import load_base_model

# target + draft 都用统一加载, 零硬编码
target, tokenizer = load_base_model(target_path, device, dtype)
draft, _ = load_base_model(draft_path, device, dtype)
```

### 8.2 自动检测

- VL 模型 → `AutoModelForImageTextToText` → `Qwen3VLForConditionalGeneration`
- 纯文本 → `AutoModelForCausalLM` → `Qwen2ForCausalLM`
- 兜底 → `AutoModel`

### 8.3 统一接口

- `get_embed_weight(model)`: 统一获取 embed_tokens (支持 VL 嵌套路径)
- `get_lm_head_weight(model)`: 统一获取 lm_head
- `get_text_model(model)`: 统一获取 text model
- `get_layers(model)`: 统一获取 transformer layers

---

## 9. 文件清单

| 文件                                                  | 功能                              | 状态 |
| --------------------------------------------------- | ------------------------------- | -- |
| `app/shared/spec_decode_ir.py`                      | 统一 IR + AutoTunner + 三后端        | ✅  |
| `app/shared/model_loader.py`                        | 统一模型加载 (VL/纯文本自动检测)             | ✅  |
| `app/shared/seamless_switcher.py`                   | 云↔本地无缝切换                        | ✅  |
| `app/cli/cgc.py`                                    | 十步流水线 + Step 11.5 AutoTunner 集成 | ✅  |
| `configs/spec_decode_default.json`                  | 默认配置 (N=16, chain, mlx)         | ✅  |
| `CGC_Phase2/eagle_tree_search.py`                   | 客户端 EAGLE tree search           | ✅  |
| `CGC_Phase2/bench_unified_spec.py`                  | 统一 bench (--mode chain|eagle)   | ✅  |
| `CGC_Phase2/bench_05b_draft.py`                     | 0.5B 4bit draft bench           | ✅  |
| `CGC_Phase2/mtp_head/train_chained_decode_multi.py` | decode hidden 多卡训练              | ✅  |

---

## 10. 用法

### 10.1 AutoTunner --auto (自动检测 + 自适应)

```bash
# Mac MLX
python -m app.shared.spec_decode_ir --backend mlx --auto

# GPU PyTorch
python -m app.shared.spec_decode_ir --backend pytorch --auto

# Cloud SGLang (Qwen3-VL)
python -m app.shared.spec_decode_ir --backend sglang --auto

# Cloud SGLang (V4-Flash, 指定模型路径)
python -m app.shared.spec_decode_ir --backend sglang --auto --model-path /data/models/DeepSeek-V4-Flash-UD-IQ2
```

### 10.2 --show-launch (生成 sglang 启动命令)

```bash
# V4-Flash: 自动设 CGC_ENABLE_ORTHO_KDA=0 + cuda-graph + NEXTN
python -m app.shared.spec_decode_ir --backend sglang --show-launch --model-path /data/models/DeepSeek-V4-Flash-UD-IQ2

# Qwen3-VL: 默认参数
python -m app.shared.spec_decode_ir --backend sglang --show-launch --model-path /data2/models/Qwen3-VL-2B-Instruct
```

### 10.3 保存/加载配置 (跨后端共享)

```bash
python -m app.shared.spec_decode_ir --save-config my_config.json
python -m app.shared.spec_decode_ir --load-config my_config.json
```

### 10.4 十步流水线 (cgc run)

```bash
cgc run --model /data/models/DeepSeek-V4-Flash-UD-IQ2
# 自动执行 Step 1-11.5, 包括 SeamlessSwitcher + AutoTunner
```

---

## 11. 结论

1. **统一 IR**: 一份配置 JSON 跨三后端 (mlx/pytorch/sglang), `--backend` 一键切换
2. **AutoTunner**: 自动检测硬件 + 模型类型 → 最优参数 (N, mode, cuda-graph, NEXTN, mem-fraction)
3. **SeamlessSwitcher**: 运行时云↔本地切换 (内存/网络/decode 监控)
4. **十步流水线**: Step 11.5 集成 SeamlessSwitcher + AutoTunner, 切换时自动调优
5. **V4-Flash**: 原生 NEXTN MTP + cuda-graph (CGC_ENABLE_ORTHO_KDA=0), 放弃外接 EAGLE
6. **Qwen3-VL-2B**: 0.5B 4bit chain speculative, MLX N=16 (2.0x), PyTorch N=4 (1.94x)

---

*End of Document*
