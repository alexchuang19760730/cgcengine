# CGC Gate 6.0 部署状态报告

**日期**: 2026-07-22 (更新)
**目标**: Mac→Host2, RemoteEdge(gs01)→Host1 跑通 cgc g6 run / cgc claude
**性能目标**: TTFT 30ms, Decode 50 token/s, 端侧首包 MTP draft model (Host1: /data/models/mtp_edge)

---

## 关键结论（本次排查最重要发现）

1. **节点上不存在"完整精度"模型。** `/data/models/DeepSeek-V4-Flash` 是指向
   `/data2/models/DeepSeek-V4-Flash-UD-IQ2` 的**符号链接**。两端唯一可加载的
   safetensors 模型就是 `UD-IQ2`（FP4+FP8 混合量化，159.6 GB / 46 分片）。
   之前"切到完整精度"的指令前提不成立——我们一直跑的就是这个量化模型。
   GGUF q8 分片（/data2/gguf_out）是 llama.cpp 格式，SGLang 不能直接加载。

2. **模型真实身份**：DeepSeek-V4-Flash（README 标明 284B 总参数 / 13B 激活，
   1M 上下文，**FP4(MoE 专家) + FP8(其余) 混合精度**）。`quantization_config` =
   `fp8 / e4m3 / block [128,128] / ue8m0`；MoE 走 `Mxfp4MarlinMoEMethod`，
   dense 走 `W8A8 Block FP8`。这不是"IQ2 二值量化"，是官方 FP4+FP8。

3. **乱码根因已 100% 定位（2026-07-22 深夜，决定性排查完成）。**
   逐层下钻结论：
   - **乱码是 forward 错误，不是编码/分词器**：对纯英文 prompt
     `"The capital of France is"` 在 `temperature=0` 下仍输出随机混合语种乱码
     （首 token `'chwitz'`，logprob **−1.314**，正确模型应≈ −0.1 输出 " Paris"）。
     随机混合语种 = hidden states 数值已坏，而非 tokenizer 映射问题。
   - **权重加载正确**：fork 启动日志确认 `is_fp4_experts=True`，走
     `Mxfp4MarlinMoEMethod`，无 dtype 报错，`loaded_params=1459 / weight_names=69143`，
     无 unloaded 异常（仅 nextn/MTP 的 301 个参数按预期未加载）。
   - **checkpoint 是合法混合精度，且权重本身完好**：
     * Shared experts = `F8_E4M3` 全宽（如 `(2048,4096)`）→ 标准 FP8，正确。
     * Routed experts = `I8` 半宽（如 `(2048,2048)`、`(4096,1024)`）→ 这是
       **NVFP4 打包布局**（每字节 2 个 fp4，以 int8 字节存储，−128..127 是
       打包 fp4 的正常字节值，**不是损坏**）。
     * 用标准 e2m1 + ue8m0 参考解码（纯 torch）对 routed expert 实测：
       解码值严格落在 `[-6.00, 6.00]`（e2m1 值域），反量化后
       `std≈0.024, max|w|≈0.19, 无 NaN` —— **教科书级健康权重**。
   - **因此 bug 完全在 fork 的 `Mxfp4MarlinMoEMethod`（Marlin NVFP4 反量化/重排）**
     在 **SM120（Blackwell RTX PRO 5000）** 上的实现。参考解码干净、Marlin 解码
     乱码 → 即 Marlin fp4 的 `gptq_marlin_moe_repack` / `marlin_moe_permute_scales`
     或 SM120 上的 Marlin fp4 kernel 对该 checkpoint 的打包/scale 约定不兼容。
   - **佐证**：dsv4 注意力后端在 fork 里有完整的 SM120 专用实现
     （`flash_mla_with_kvcache_sm120`、`fp8_paged_mqa_logits_torch_sm120`、
     `is_sm120_supported()` 分支），注意力路径大概率正常；**MoE 的 Marlin fp4
     是 SM120 上最不成熟、最可疑的路径**。
   - 注：之前建议的"官方 `inference/convert.py` 重分片验证"**已不必执行**——
     参考解码已证明权重完好，问题 100% 在 fork 的 Marlin fp4 kernel。

4. **CGC cloud proxy (`run_cgc_cloud_openai.py`) 只是薄桥接**：把 OpenAI 格式
   messages 直接转发给 SGLang 的 `/v1/chat/completions`，**不**做 DSV4 的
   `encode_messages`。所以整条链路（proxy→SGLang）依赖 SGLang 的 chat template，
   而 DSV4 没有 jinja template（README 明确要求用 `encoding/` 脚本），
   SGLang 默认模板给模型的 prompt 是错误的——但即使我们手动用正确编码，
   前向仍乱码，故编码不是唯一问题。

---

## 网络拓扑（更正）

```
Mac (本机)
  ├── SSH隧道 localhost:30000 → Host2:30000 (SGLang)
  ├── SSH隧道 localhost:50053 → Host2:30000
  ├── Edge-First Proxy (port 14001) → localhost:30000
  └── CGC API Server (port 18000) → localhost:50053

RemoteEdge (gs01@10.100.200.65)
  ├── SSH隧道 localhost:50052/50053 → Host1
  ├── MTP Draft Model (llama-server, port 18111, q8_0 gguf)
  ├── CGC API Server (port 18000)
  └── Internal Proxy (port 14000)

Host1 (39.106.118.206 / 内网 172.30.132.116) - Gateway Node
  ├── 8x RTX PRO 5000 72GB Blackwell (sm120), 全部 8 卡参与
  ├── SGLang Worker (node-rank 1)
  └── Gateway (port 50053)

Host2 (47.95.250.55 / 内网 172.30.132.117) - Head Node
  ├── 8x RTX PRO 5000 72GB Blackwell, 全部 8 卡参与
  ├── SGLang Head (node-rank 0, port 30000, 127.0.0.1)
  ├── Ray Head (172.30.132.117:6379)
  └── 模型: /data/models/DeepSeek-V4-Flash -> /data2/models/DeepSeek-V4-Flash-UD-IQ2
```

集群实际并行配置：tp-size 8, ep-size 1, dp-size 2, nnodes 2
（placement group 2 bundle × 4 GPU/bundle = 8 GPU 跨节点 TP）。

---

## 已确认的技术约束（fork 行为）

| 约束 | 说明 |
|------|------|
| `--disable-cuda-graph` **强制** | 本 fork 的 dsv4 前向不支持 CUDA graph 捕获，开 graph 必崩 |
| CGC 注入必须开 (`CGC_ENABLE_ORTHO_KDA=1` 等) | 关掉会退回不匹配的注意力路径；已知可用配置即此 |
| MTP / `FROZEN_KV_MTP` **不可用** | 本 fork 对该模型 + MTP worker 初始化崩溃（内存池 None bug） |
| 磁盘 `/` 必须留余量 | Ray object store 在 `/tmp/ray`；满盘会导致 worker 在首个请求时崩溃（已踩坑） |
| DSV4 需自定义编码 | 无 jinja chat template，必须 `encoding_dsv4.encode_messages` |

---

## 当前运行状态（2026-07-22 晚）

- 集群已用 **CGC 注入 ON + 干净磁盘** 重启，server "fired up and ready"，
  port 30000 LISTEN，进程稳定（不再因磁盘满崩溃）。
- 新建启动脚本（替代之前关掉 CGC 的 `start_dualnode_full_max.sh`）：
  - `start_dualnode_cgc_on_rank0.sh` (Host2)
  - `start_dualnode_cgc_on_rank1.sh` (Host1)
  - `start_dualnode_cgc_on_master.sh` (Host2, 经跳板拉起双节点)
  - 均用 `setsid nohup ...` 脱离 SSH 会话（之前用 `ssh ... "nohup ..."` 会因
    SSH 通道不释放导致 master 卡死、rank0 未启动）。
- **输出仍乱码**（forward 错误），decode 约 5.6 tok/s（与量化无关，见下）。

---

## 待解决 ❌

### 问题 1（P0, 阻断）：SGLang 输出乱码 —— 根因已确认
- **根因（确认）**：fork 的 `Mxfp4MarlinMoEMethod` 在 **SM120** 上对 routed-expert
  的 **NVFP4（Marlin fp4）反量化/重排** 实现有 bug。权重本身 100% 完好
  （参考 e2m1+ue8m0 解码 `std≈0.024` 健康），但 Marlin 路径产出错误 logits。
- **关键 env / 开关**（供修复参考）：
  - `SGLANG_DSV4_FP4_EXPERTS`（默认 `True`）控制 `is_fp4_experts`；但设 `False`
    会改走 `Fp8MoEMethod`（期望 `F8_E4M3`），而 routed expert 实为 fp4/int8 →
    **该开关不能修复此问题**，必须换 MoE runner backend。
  - MoE backend 由 `--moe-runner-backend` 决定：当前 AUTO → `marlin`（SM120）；
    fp4 expert 可选 `flashinfer_mxfp4`（→ `Mxfp4FlashinferTrtllmMoEMethod`，
    另一条 fp4 反量化路径，SM120 上走 cuda backend）。
- **修复方向（按推荐顺序）**：
  - **(A) 首选实验/潜在修复**：双节点启动加 `--moe-runner-backend flashinfer_mxfp4`
    （保持 `is_fp4_experts=True`），重启测乱码。若输出连贯 → 确认 Marlin SM120
    fp4 bug 且已绕过；若仍乱码/crash → 注意力或其它路径也有问题。
    ⚠️ flashinfer_mxfp4 原面向 SM100（trtllm-gen），SM120 支持未经证实，可能
    启动报错——可逆（保留 marlin 配置回退）。
  - (B) 修 fork：`gptq_marlin_moe_repack` / `marlin_moe_permute_scales` 的 SM120
    fp4 打包/scale 约定（需 fork 维护者）。
  - (C) 向 CGC / fork 维护者提交 bug 报告（附：参考解码健康 vs Marlin 乱码证据）。

### 问题 2：Decode 速度 5.6 → 50 tok/s
- ep-size 1 对 MoE 效率低（专家未并行分片）；`--disable-cuda-graph` 去除了
  graph replay 优化。需在乱码修复后，调 ep-size / 关 `--enable-return-hidden-states` /
  评估是否可在 fork 内启用 piecewise CUDA graph。

### 问题 3：Host1 Gateway 未转发 OpenAI API
- gateway 返回自定义状态而非转发 `/v1/chat/completions`，gs01 经 gateway 访问失败。
- 可 bypass：gs01 直接打 SGLang（但需 SGLang 先能出正确结果）。

---

## 可用模型（实测）

| 模型 | 大小 | 备注 |
|------|------|------|
| DeepSeek-V4-Flash (symlink) | → UD-IQ2 159.6GB | **实际就是量化模型，无全精度** |
| DeepSeek-V4-Flash-UD-IQ2 | 159.6 GB | FP4+FP8 混合，当前加载，**输出乱码** |
| DeepSeek-V4-Flash-DSpark | ~60 GB (Host1) | 另一变体 |
| /data2/gguf_out/*.q8_0.gguf | q8 分片 | llama.cpp 格式，SGLang 不可直接加载 |
| mtp_edge | 3.2 GB | MTP 边缘 draft 模型 (Host1) |
| Qwen2.5-7B-Instruct | ~15 GB | 小模型，可作 SGLang 流水线冒烟测试 |

---

## 下一步建议（需用户决策）

1. **（优先）跑 `--moe-runner-backend flashinfer_mxfp4` 实验**：双节点
   `start_dualnode_cgc_on_rank0/1.sh` 加该参数（保留 `is_fp4_experts=True`），
   经 `start_dualnode_cgc_on_master.sh` 重启，测纯英文 prompt 是否连贯。
   这是验证/绕过 Marlin SM120 fp4 bug 的最快路径，可逆。
2. 若 (1) 仍乱码或 crash：注意力路径或 fork 其它部分也有问题，需进一步下钻
   （或提交 bug 给 fork 维护者，证据已齐）。
3. 乱码修复后攻 decode 速度（ep-size、cuda-graph、hidden-states）。
4. **不再需要** 官方 `inference/convert.py` 重分片验证——参考解码已证明权重完好。
