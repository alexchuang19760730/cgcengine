# LES（Lookahead Expert Scheduling）上游资料镜像

> 用途：为 CGC Engine 的「目标-router 预取移植」探针提供可核对的一手资料。
> 本目录是**上游开源代码的镜像与摘录**，非 CGC 原创；修改请在 CGC 自有源码中进行。

## 来源

| 项 | 值 |
| --- | --- |
| 上游模型 | MiniCPM-V 4.7 35B-A3B（面壁智能 / OpenBMB，2026-10-12 发布；语言 backbone 为 Qwen3.6，256 专家、top-8） |
| 上游引擎仓库 | https://github.com/tc-mb/llama.cpp-omni （llama.cpp fork，MIT License） |
| 镜像 commit | `88bd37eb671f67cf8d747140ed100231756e0bc3`（master，2026-10-12T10:01:48Z） |
| 模型权重 | https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B （Apache-2.0；BF16/FP8/int4/AWQ/GPTQ/GGUF + DSpark draft） |
| 官方设计文档（en） | https://github.com/tc-mb/llama.cpp-omni/blob/master/docs/les/lookahead-expert-scheduling.md |
| 官方设计文档（zh） | https://github.com/tc-mb/llama.cpp-omni/blob/master/docs/les/lookahead-expert-scheduling.zh.md |
| 端侧 App 源码 | https://github.com/OpenBMB/MiniCPM-V-Apps |
| 论文状态 | **截至 2026-10-17 无独立 arXiv 技术报告**；上述设计文档为唯一官方技术描述（后续若补论文需回填） |

## 文件清单（`upstream/`）

- `lookahead-expert-scheduling.md` / `.zh.md`：官方设计文档（中英文）
- `lookahead-expert-scheduling.png` / `.zh.png`：官方时序架构图（中英文）
- `llama-moe-slot.cpp` / `.h`：LES 预取调度核心（`struct lookahead_expert_prefetch`）
- `qwen35moe.cpp`：模型图，**预测器构图在第 455–466 行**
- `llama-moe-experts.cpp` / `.h`：MoE 专家加载辅助
- `UPSTREAM_SHA.txt`：镜像对应的上游 commit

## 机制要点（一手核对结论）

1. **预测器 = 下一层 router 复用，零训练**。`qwen35moe.cpp:455-466`：
   ```cpp
   // 仅 decode（ubatch.n_tokens == 1）
   ggml_tensor * prefetch_router_logits =
       build_lora_mm(model.layers[il + 1].ffn_gate_inp, attn_post_norm); // 下一层 router × 当前层 attn_post_norm
   lookahead_next = ggml_cont(ctx0,
       ggml_argsort_top_k(ctx0, prefetch_router_logits, depth));          // depth 默认 8
   moe_slots->set_lookahead_pred(lookahead_next);
   ```
2. **预测只指导预取/调度，不参与选择**。真实路由仍由下一层自己的 router 决定，预测落空的专家照常加载 → 输出逐字节一致（Lossless）。
3. **预取与计算重叠**。后台线程把预测专家 `preadv` 进共享 bank 的 ping-pong 尾部（`install_prefetch_into_window`），重叠当前层 GPU 工作；命中免重映射，miss 仍走关键路径。
4. **开关与旋钮**（环境变量，源码中实测）：
   - `LLAMA_MOE_LOOKAHEAD_PREFETCH=1`：总开关（默认关）
   - `LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH`：预取专家数（默认 8，kMaxDepth=16）
   - `LLAMA_MOE_PREFETCH_READ`：真正读盘的专家数；`=0` 为「只预测不读盘」，用预测指导 LRU 淘汰保护
   - `LLAMA_MOE_PREFETCH_UNIFIED=0`：预读/缓存退回分区模式（A/B 用）
5. **自证计数**：`n_hits / n_armed / n_arm_skipped / n_waits / n_wait_ns / n_pred_read / n_pred_published / n_predict_only / n_evict_saved / n_evict_forced`——命中率、等待、预读是否真的发生都有埋点。
6. **配套工程项**：专家权重 `gate|up|down` 连续 pack，单次 `preadv`；直接写入 GPU-visible 内存；逐层混合精度量化；DSpark 投机采样。

## 官方实测（iPad Pro 11″ M4 16GB，35B-A3B，SSD 放权重）

| 阶段 | decode | 相对基准 |
| --- | ---: | ---: |
| 基准：SSD 专家加载、无优化 | 2.97 t/s | 1.00× |
| +LES（预取+布局+少读+直写 GPU 可见内存） | 9.81 t/s | 3.30× |
| +逐层混合精度量化 | 14.92 t/s | 5.02× |
| +DSpark 投机采样 | **17.39 t/s** | **5.86×** |

时序账：读 1 个专家 ≈ 1.00 ms，算 1 层 ≈ 1.15 ms；无预取 8 专家串行 = 8.0 ms/层，有预取 I/O 与计算重叠 ≈ 0。命中率（N=8）>80%，即下一层 8 个专家中约 6–7 个提前驻留。

## 与 CGC 现状的关系（移植前必读）

- **同构性**：我们的目标模型 Qwen3.6-35B-A3B 与上游同 backbone、同路由结构；上游预测代码就挂在 `qwen35moe.cpp`，且同样是 llama.cpp + Metal 后端。许可证 MIT，允许移植与修改。
- **与已判死方向的区别**：之前判死的是「邻居/启发式背景预取」（频率/相关性启发）。LES 用的是**目标层 router 的真实权重**对当前特征打分，是更强的信号（实测 >80% 覆盖），不属同一方案，不在原判死范围。
- **必须协同的现有子系统**：8GB expert pool / LRU、SpAc、P0/P1/P2 水位闸门、MTP。注意预测构图仅在 `ubatch.n_tokens==1` 的单 token decode 生效；MTP verify 段（多 token）是否可用需单独验证。
- **探针定义与验收**：见探针页 `docs/les/les_router_prefetch_probe.html` 与 mindmap 实验阶段节点（brief `e-les-router-prefetch-2026-10-17`），charter 为 `scripts/check/charters/e-les-router-prefetch-2026-10-17.yaml`（G1–G6）。
- **G6 缩池转正（LES × 4/6GiB expert cache）**：「池大小不是杠杆」的实测根因是预取信号差（8GiB 缺 113 槽、每步只补 ~1.8 个、宽松 63 倍，池从未被填满），不是容量；纯缩 4/6GiB 因预取补不回新增 miss 而变慢，10GiB 扩池则 Metal OOM（working set 上限 ~11453 MiB）。LES 若命中率达标，可把「常驻需求」降级为「及时预取」：小池只放跨步热专家 + ping-pong 缓冲，冷专家由目标 router 预取在一层窗口内提前读入。G6 跑 off/on × {8,6,4}GiB **每池一格**矩阵，先重建权威一手 8/6/4 基线（记忆中的 12.06/10.48/7.67 已被 `docs/CHOOSE_WINSOUND_2026-10-07.md` 查证为二次转述、不可引用；poolsweep p4–p7 被 cell contract 拒跑）。判据：6G+on 追平 8G+off（−5% 内）且物理峰值降 ≥1.5GiB、4G+on 缩池惩罚收窄 ≥50%。这与 `e-union-cache-shrink`（收缩 ≥30%/挑战 50%）目标一致，但 LES 同 token 层间、不依赖 MTP，故先行；MTP union 信号可后续叠加。
