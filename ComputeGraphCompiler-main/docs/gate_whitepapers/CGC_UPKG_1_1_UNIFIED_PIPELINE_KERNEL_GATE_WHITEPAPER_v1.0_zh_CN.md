# CGC Unified Pipeline Kernel Gate 1.0（UPKG_1.1）技术白皮书

**版本**: v1.0  
**状态**: 正式验收口径版  
**定位**: 定义 `Unified Pipeline Kernel` 在 `M1-M7.5` 范围内的正式 gate 边界、最小前置条件、证据格式与当前收口结果。

---

## 一、文件定位

`UPKG_1.1` 不是另一套平行 gate，而是把 `M1-M7.5` 已经散落在不同脚本、report 与调试路径中的正式验收口径统一收敛为一份 gate 白皮书。

它回答的是四个问题：

- `M1-M7.5` 到底验什么
- 什么属于正式 `PASS`
- 什么只能算 smoke 或中间态
- 当前版本已经有哪些正式 artifact 可以作为收口依据

本文件与技术总纲的关系如下：

- `docs/technical_whitepapers/CGC_Unified_Pipeline_Kernel_Design_v1.0_zh_CN.md`
  - 负责上位设计、统一主干与 `UPKG_1.1` 原则定义
- 本文件
  - 负责 gate 口径、证据标准、当前 accepted artifacts 与 rerun 落点

---

## 二、UPKG_1.1 要验什么

### 2.1 总体目标

`UPKG_1.1` 要证明：

- `Unified Pipeline Kernel` 在 `M1-M7.5` 范围内不是概念稿，而是可重跑、可定位、可回放、可归因的正式执行底座
- `train / infer / compile / deploy` 不再靠分裂脚本分别定义成功
- 低内存 fallback、fingerprint lock、distributed 证据都被纳入统一 gate 语义
- `industrial verification / active runtime / edge-cloud bridge / API compatibility` 也被纳入统一 kernel 的正式交付语义

### 2.2 覆盖范围

- `M1`: 本地 native 八步主干可完成
- `M2`: compile 前策略与 gate 包装成立
- `M3`: bundle / export 产物可落地
- `M4`: training + inference 双路能被统一 kernel 聚合，且 training route 必须有正式 distributed 证据
- `M5`: fullgraph / AOT / bench / deploy 可在统一 kernel 下完成；低内存 Mac 可接受 `oMLX + dflash` fallback
- `M7`: 工业级 unified kernel 验证总入口，要求动态编译、状态压缩、回放、审计同时成立
- `M7.1`: `M7` 的核心内核层，负责 dynamic trace、state compression、soft-RT replay、industrial audit
- `M7.2`: 数字世界 GUI Agent 验收层，要求在 `M7.1` 基础上完成 GUI/桌面场景指标验证
- `M7.3`: 物理具身智能端云桥接验收层，要求 cloud training、edge bridge、state compression、audit 同时成立
- `M7.4`: `dflash + TrueOrthoKDA` 合同与 runtime evidence 验收层
- `M7.5`: 分为 `API compatibility` 与 `TrueOrthoKDA active runtime` 两条正式验证线，前者验证协议与分布式交付面，后者验证真 state transport / edge resume / compression / zero-copy evidence

### 2.3 统一 8-step 契约

`UPKG_1.1` 接受的统一 kernel，不只是“最后有 report”，还必须遵守统一的 `8-step` 语义：

- `Step0 Scenario`
  - 写入任务分类、后端、模型族、硬件 profile 等上下文
- `Step1 Hardware`
  - 检测 device、runtime 能力与关键前置依赖
- `Step2 Capture`
  - 捕获可复现的图、配置、wrapper 或 header 快照
- `Step3 Analyze`
  - 对捕获物做静态分析，并生成 gate plan / strategy plan
- `Step4 Identify`
  - 明确本轮要生成、替换或优化的目标
- `Step5 Generate`
  - 生成并落盘可复用产物，如 cache、bundle、kernel dump、manifest
- `Step6 Dispatch`
  - 至少完成一次真实执行，不能只停留在静态分析
- `Step7 Compare`
  - 输出 baseline vs optimized 或自检对照结果
- `Step8 Combine`
  - 把产物路径、关键指标、最终决策统一写回单一真源 `report.json`

### 2.4 CLI 边界与单一真源

`UPKG_1.1` 默认采用以下 CLI 边界：

- `cgc run`
  - 终端用户入口，负责模型发现、执行与交付体验
- `cgc gate`
  - 正式验收入口，负责验证与报告检查
- `cgc pipeline`
  - 内部工程入口，负责 8-step 汇总、开发调试与产物编排

所有正式验收都必须满足：

- 每次 run 必须有明确的 `output_dir`
- `report.json` 是单一真源
- 插件或后端只允许生成碎片化产物，不直接替代 pipeline 汇总的 `report.json`
- `report.json` 必须能反向索引到 cache / dump / bundle / manifest / runtime evidence 等关键产物

---

## 三、正式 PASS 条件

### 3.1 通用前提

- 必须有可用本地 `GGUF`
- 必须有本次生效的 `backend fingerprint lock`
- `PASS / FAIL / SKIP` 必须能从 report 直接归因，不能靠口头解释补齐

### 3.2 M4 条件

`M4` 正式 `PASS` 至少同时要求：

- `distributed_init.status = PASS`
- `world_size > 1`
- `step6_dispatch.status = PASS`
- `performance_gate.status = PASS`
- `tp / pp / ep` 至少一项大于 `1`

`world_size=1` 只能算 smoke，不算正式 `M4 PASS`。

### 3.3 M5 条件

`M5` 正式 `PASS` 至少同时要求：

- `step2_fullgraph_capture.status = PASS`
- `step6_fullgraph_compile.status = PASS`
- `step7_fullgraph_bench.status = PASS`
- `step8_fullgraph_deploy.status = PASS`
- `m5.aot_precompile_gate.status = PASS`

对低内存 Mac，允许：

- `provider = omlx_dflash`
- `omlx_fallback.status = PASS`
- `omlx_fallback.engine = dflash`

但不能只因为出现 fallback 就自动判 `PASS`；仍必须保留 compile / bench / deploy 全套证据。

### 3.4 M7 / M7.1 条件

`M7 / M7.1` 正式 `PASS` 至少同时要求：

- `dynamic_trace.status = PASS`
- `dynamic_trace_l1.compile_success_rate = 1.0`
- `dynamic_trace_l1.cache_hit_rate >= 2/3`
- `state_compression.status = PASS`
- `replay.status = PASS`
- `audit.status = PASS`

`M7` 是总入口，`M7.1` 是它的核心内核层；在当前代码基线下，两者共用同一份 core artifact。

### 3.5 M7.2 条件

`M7.2` 正式 `PASS` 至少同时要求：

- `dynamic_trace_l1 = PASS`
- `soft_rt_replay = PASS`
- `state_compression = PASS`
- `industrial_audit = PASS`

### 3.6 M7.3 条件

`M7.3` 正式 `PASS` 至少同时要求：

- `cloud_training_psi0.status = PASS`
- `edge_inference_bridge.status = PASS`
- `state_compression.status = PASS`
- `industrial_audit.status = PASS`

### 3.7 M7.4 条件

`M7.4` 正式 `PASS` 至少同时要求：

- `dflash_contract = PASS`
- `trueorthkda_contract = PASS`
- `trueorthkda_runtime = PASS`
- `edge_runtime_evidence = PASS`

### 3.8 M7.5 条件

`M7.5 API compatibility` 正式 `PASS` 至少同时要求：

- `api_surface = PASS`
- `tool_call_hotfix = PASS`
- `local_loopback = PASS`
- `client_entrypoints = PASS`
- `distributed_runtime_evidence = PASS`
- `edge_router_runtime_evidence = PASS`
- `edge_router_cluster_nfs_evidence = PASS`
- `extreme_scale_runtime_evidence = PASS`

`M7.5 TrueOrthoKDA active runtime` 正式 `PASS` 至少同时要求：

- `report_schema = PASS`
- `true_state_transport = PASS`
- `edge_state_resume_decode = PASS`
- `runtime_evidence = PASS`

---

## 四、当前正式证据

### 4.0 2026-06-20 全量 Gate 重跑补充

本次已按当前 `cgc gate list` 可执行范围，对 release CLI 的可用 gate 做了一轮全量重跑，统一输出目录为：

- `/private/tmp/full_gate_rerun_20260620/release`

本次全量重跑的总索引为：

- `/private/tmp/full_gate_rerun_20260620/release/release_gate_status_index.json`

其中与 `UPKG_1.1` 直接相关的当前解释如下：

- `m4`：本轮仍体现真实 training / distributed 条件未满足，不再归类为单纯文档缺失
- `m5`：旧顶层 report 仍可能受 `backend fingerprint strict gate` 影响，需结合 canonical rerun artifact 一起解释
- `m7.1 / m7.2 / m7.3`：本轮 release CLI 重跑均已继续保持 `PASS`
- `m7.4 / m7.5`：当前 release sweep 已落出子报告与运行时 evidence，但顶层 report 仍未统一回灌，需要按子报告解释
- `m7.6`：当前已在 `host1/host2 + L20N 72GB + eRDMA` 路径上补齐 `Nvidia real-chain runtime evidence`，并完成一次正式双机 rerun
  - `eRDMA / RoCE / IB / GPUDirect RDMA` 已不再只停留于“底座打通”，而是已在正式 artifact 中落出 `rdma_contract.status = PASS`
  - `NCCL` 已不再只是 bootstrap 或初始化成功，而是已在正式 artifact 中落出 `enable_nccl = true`、`effective_collective_backend.status = PASS`、`effective_distributed_runtime.backend = nccl`
  - `DeepEP` 已不再只是 contract 声明，而是已在正式 artifact 中落出 `requested_dispatch_backend = deepep`、`effective_dispatch_backend.backend = deepep`、`deepep_real_chain_gate.status = PASS`
  - `CUDA Graph`、单机内 `NCCL P2P`、host 内 peer access 等能力，在当前 `UPKG_1.1` 口径下仍只属于观测或优化项，不构成 `m7.6 formal gate` 的直接成立条件
  - 当前 `runtime_contract / runtime_evidence / report.json` 已形成可审计闭环，因此允许正式表述：
    - `transport foundation = PASS`
    - `NCCL real-chain = PASS`
    - `DeepEP real-chain = PASS`
    - `m7.6 formal gate = PASS`

### 4.1 M4 正式证据

**artifact**

- `temp/misc/host1_m4_training_route/rank0_report_20260619T113816Z.json`

**已确认成立的字段**

- `distributed_init.status = PASS`
- `distributed_init.world_size = 2`
- `distributed_init.backend = nccl`
- `step2_graph_capture.compile_wrapper = ddp_unwrapped`
- `step5_generate.torch_compile.status = PASS`
- `step6_dispatch.status = PASS`
- `step7_compare.performance_gate.status = PASS`
- `step7_compare.performance_gate.speedup = 2.2217131304915823`

**解释**

- 这份 evidence 证明 `M4` 不是只在本地 mock distributed，而是在 `host1` 真正以 `torchrun --nproc_per_node=2` 跑成正式 distributed training route
- `ddp_unwrapped` 只用于 compile-friendly graph capture，不改变 `distributed_init` 与 `DDP wrap` 的正式语义

### 4.2 M5 正式证据

**artifact**

- `/tmp/cgc_upkg_fix_20260619_m5/pass2/report.json`

**已确认成立的字段**

- `backend_fingerprint_gate.status = PASS`
- `step2_fullgraph_capture.status = PASS`
- `step6_fullgraph_compile.status = PASS`
- `step7_fullgraph_bench.status = PASS`
- `step8_fullgraph_deploy.status = PASS`
- `m5.aot_precompile_gate.status = PASS`
- `m5.aot_precompile_gate.provider = omlx_dflash`
- `m5.aot_precompile_gate.omlx_fallback.status = PASS`
- `m5.aot_precompile_gate.omlx_fallback.engine = dflash`
- `m5.aot_precompile_gate.omlx_fallback.compile_mode = omlx_dflash`

**解释**

- 这份 report 是 `M5` 的 canonical rerun artifact
- 先前同一路径出现过 `KeyboardInterrupt`，那次不再视为技术失败；当前 `pass2` 已经补成正式 `PASS`

### 4.3 M7 / M7.1 正式证据

**artifact**

- `ComputeGraphCompiler-main/Output/cli_gate_m7_series_20260619/m7_industrial/m7_report.json`

**2026-06-19 正式 rerun artifact**

- `m7` 主 report：`/private/tmp/upkg30_formal_pass_20260619/m7/m7_industrial/m7_report.json`
- `m7` 主 summary：`/private/tmp/upkg30_formal_pass_20260619/m7/m7_industrial/summary.json`
- `m7` artifact index：`/private/tmp/upkg30_formal_pass_20260619/m7/m7_industrial/artifact_index.json`

**已确认成立的字段**

- `dynamic_trace.status = PASS`
- `dynamic_trace_l1.compile_success_rate = 1.0`
- `dynamic_trace_l1.cache_hit_rate = 1.0`
- `state_compression.status = PASS`
- `state_compression_summary.compression_ratio = 0.012370768213415912`
- `replay.status = PASS`
- `audit.status = PASS`

**解释**

- 当前版本将 `M7` 与 `M7.1` 视为“总入口 + 核心内核层”的关系，而不是两份相互冲突的 report
- 因此 `UPKG_1.1` 接受同一份 core artifact 作为 `M7 / M7.1` 的正式证据

### 4.4 M7.2 正式证据

**artifact**

- `ComputeGraphCompiler-main/Output/cli_gate_m7_series_20260619/m72_gui_agent/report.json`

**2026-06-19 正式 rerun artifact**

- `m72` 聚合 report：`/private/tmp/upkg30_formal_pass_20260619/m72/report.json`
- `m72` 主 report：`/private/tmp/upkg30_formal_pass_20260619/m72/m72_industrial/report.json`
- `m72` 主 summary：`/private/tmp/upkg30_formal_pass_20260619/m72/m72_industrial/summary.json`
- `m72` artifact index：`/private/tmp/upkg30_formal_pass_20260619/m72/m72_industrial/artifact_index.json`
- `m72` GUI runtime evidence：`/private/tmp/upkg30_formal_pass_20260619/m72/m72_industrial/gui_agent_runtime/gui_agent_runtime_evidence.json`
- `m72` 3.7 cloud-edge mode：`/private/tmp/upkg30_formal_pass_20260619/m72/m72_industrial/cloud_edge_training_inference_mode.json`
- `m72` 3.7 GUI edge inference contract：`/private/tmp/upkg30_formal_pass_20260619/m72/m72_industrial/gui_agent_edge_inference_contract.json`
- `m72` 3.7 Q2RL profile：`/private/tmp/upkg30_formal_pass_20260619/m72/m72_industrial/q2rl_post_training_profile.json`
- 本次 rerun 结论：`status = PASS`

**已确认成立的字段**

- `status = PASS`
- `metrics.dynamic_trace_l1 = PASS`
- `metrics.soft_rt_replay = PASS`
- `metrics.state_compression = PASS`
- `metrics.industrial_audit = PASS`

**当前 route 口径**

- `M7.2` 当前应以 `agent domain` 主路由来理解，而不是继续把它等同于旧 `harness` 测试路径
- `agent domain = 主 pipeline + GUI-native route`
  - `workflow / runtime_host / screenshot / tool_call` 已可经由主 pipeline 进入 `report.json`、`summary`、`m72 gate` 与 `cgc run` artifact
- `harness domain = 旧测试 / 验证专用 route`
  - 仅在明确指定 `task_domain = harness / moe` 或 `model_name = moe_harness` 时作为历史验证入口保留
- 因此 `UPKG_1.1` 对 `M7.2` 的正式证据解释，应优先以 `agent runtime` 主路由成立为准，而不是把 `harness` 视为默认产品路径

### 4.5 M7.3 正式证据

**artifact**

- `ComputeGraphCompiler-main/Output/cli_gate_m7_series_20260619/m73_physical/m73_report.json`

**2026-06-19 正式 rerun artifact**

- `m73` 聚合 report：`/private/tmp/upkg30_formal_pass_20260619/m73/report.json`
- `m73` 主 report：`/private/tmp/upkg30_formal_pass_20260619/m73/m73_physical/m73_report.json`
- `m73` 主 summary：`/private/tmp/upkg30_formal_pass_20260619/m73/m73_physical/summary.json`
- `m73` artifact index：`/private/tmp/upkg30_formal_pass_20260619/m73/m73_physical/artifact_index.json`
- `m73` publish manifest：`/private/tmp/upkg30_formal_pass_20260619/m73/m73_physical/publish_manifest.json`
- `m73` runtime contract：`/private/tmp/upkg30_formal_pass_20260619/m73/m73_physical/runtime_contract.json`
- 本次 rerun 结论：`status = PASS`

**已确认成立的字段**

- `cloud_training_psi0.status = PASS`
- `edge_inference_bridge.status = PASS`
- `edge_inference_bridge.edge_latency_ms = 4.151859375269851`
- `state_compression.status = PASS`
- `industrial_audit.status = PASS`

### 4.6 M7.4 正式证据

**artifact**

- `ComputeGraphCompiler-main/Output/cli_gate_m74/m74_dflash_kda/m74_report.json`

**已确认成立的字段**

- `dflash_contract = PASS`
- `trueorthkda_contract = PASS`
- `trueorthkda_runtime = PASS`
- `edge_runtime_evidence = PASS`

### 4.7 M7.5 API compatibility 正式证据

**artifact**

- `ComputeGraphCompiler-main/Output/cli_gate_m75/m75_api_compat/m75_report.json`

**已确认成立的字段**

- `api_surface = PASS`
- `tool_call_hotfix = PASS`
- `local_loopback = PASS`
- `client_entrypoints = PASS`
- `distributed_runtime_evidence = PASS`
- `edge_router_runtime_evidence = PASS`
- `edge_router_cluster_nfs_evidence = PASS`
- `extreme_scale_runtime_evidence = PASS`

### 4.8 M7.5 TrueOrthoKDA active runtime 正式证据

**artifact**

- `ComputeGraphCompiler-main/Output/cli_gate_m75_trueorthokda_active/m75_trueorthokda_active/m75_trueorthokda_active_report.json`
- `ComputeGraphCompiler-main/Output/cli_gate_m75_trueorthokda_active/runtime_evidence/m75_trueorthokda_active_runtime.json`

**已确认成立的字段**

- `report_schema = PASS`
- `true_state_transport = PASS`
- `state_kind = kda_state_v1`
- `state_codec = zlib_torch_save_bytes`
- `edge_state_resume_decode = PASS`
- `runtime_evidence = PASS`
- `compression_ratio = 0.9254533979812027`
- `cpu_copy_count = 0`
- `uma_buffer_used = true`
- `device_resume_consumed = true`

### 4.9 M7.6 dual-node formal gate 正式证据

**artifact**

- `host1`
  - `/root/flashkv0516/ComputeGraphCompiler-main/Output/cli_gate_m75_formal_20260624_host1/runtime_evidence/m75_trueorthokda_active_runtime.json`
  - `/root/flashkv0516/ComputeGraphCompiler-main/Output/cli_gate_m76_formal_20260624_host1/runtime_evidence/nvidia_runtime.json`
  - `/root/flashkv0516/ComputeGraphCompiler-main/Output/cli_gate_m76_formal_20260624_host1/m76_heterogeneous/m76_report.json`
- `host2`
  - `/root/flashkv0516/ComputeGraphCompiler-main/Output/cli_gate_m75_formal_20260624_host2/runtime_evidence/m75_trueorthokda_active_runtime.json`
  - `/root/flashkv0516/ComputeGraphCompiler-main/Output/cli_gate_m76_formal_20260624_host2/runtime_evidence/nvidia_runtime.json`
  - `/root/flashkv0516/ComputeGraphCompiler-main/Output/cli_gate_m76_formal_20260624_host2/m76_heterogeneous/m76_report.json`
- 本地归档快照
  - `temp/misc/m76_formal_fetch_20260624/host1_m75_runtime.json`
  - `temp/misc/m76_formal_fetch_20260624/host1_nvidia_runtime.json`
  - `temp/misc/m76_formal_fetch_20260624/host1_m76_report.json`
  - `temp/misc/m76_formal_fetch_20260624/host2_m75_runtime.json`
  - `temp/misc/m76_formal_fetch_20260624/host2_nvidia_runtime.json`
  - `temp/misc/m76_formal_fetch_20260624/host2_m76_report.json`

**已确认成立的字段**

- `host1_m76_report.ok = true`
- `host1_m76_report.gate_result.m76.status = PASS`
- `host2_m76_report.ok = true`
- `host2_m76_report.gate_result.m76.status = PASS`
- `rdma_contract.status = PASS`
- `rdma_available = true`
- `send_ok = true`
- `requested_distributed_runtime = nccl`
- `enable_nccl = true`
- `effective_collective_backend.status = PASS`
- `effective_distributed_runtime.backend = nccl`
- `requested_dispatch_backend = deepep`
- `effective_dispatch_backend.backend = deepep`
- `deepep_real_chain_gate.status = PASS`
- `mandatory_protocol_gate.status = PASS`
- `protocol_family = trueorthokda`
- `state_codec = cq4`
- `zero_copy_vram_real.status = PASS`

**解释**

- 这批 artifact 证明当前 `host1/host2 + L20N 72GB + eRDMA` 已不再只是“transport foundation 打通”，而是已经形成正式的双机 runtime evidence 闭环
- `M7.5 active runtime` 先行产出新的 `trueorthokda` 运行时证据，再由 `M7.6` 正式回收，因此 `requested_distributed_runtime / requested_dispatch_backend / effective_*` 字段已与本次双机 formal rerun 一致
- `NCCL` 在本轮不再只是 bootstrap 成功，而是已经被正式 artifact 证明为真实 distributed runtime backend
- `DeepEP` 在本轮不再只是配置声明，而是已经被正式 artifact 证明为 active dispatch backend，且 `deepep_real_chain_gate = PASS`
- 因此，`UPKG 1.1` 对当前已验收双机路径的正式结论应固定为：
  - `transport foundation = PASS`
  - `NCCL real-chain = PASS`
  - `DeepEP real-chain = PASS`
  - `m7.6 formal gate = PASS`

### 4.10 ColossalAI distributed runtime 的 M7.6 收口规则

`ColossalAI` 当前不应被定义为独立于 `M7.6` 之外的新 gate，而应被定义为 `requested_distributed_runtime` 的一个正式候选后端，并统一纳入 `M7.6` 的 distributed runtime evidence 体系。

若要对 `requested_distributed_runtime = colossalai` 的路径宣称 `M7.6 formal PASS`，则至少必须满足以下条件：

- `M7.5 active runtime` artifact 中已显式落出：
  - `requested_distributed_runtime = colossalai`
  - `use_colossalai = true`
  - `colossalai_effective.status = PASS`
  - `effective_distributed_runtime.backend = colossalai`
- `M7.6 runtime evidence` 已正式回收同一组字段，并保持：
  - `effective_distributed_runtime.status = PASS`
  - `effective_distributed_runtime.backend = colossalai`
  - `colossalai_effective.status = PASS`
- 不得只因 `import colossalai` 成功、bootstrap 初始化成功、或 contract 已声明 `use_colossalai = true`，就表述为 `ColossalAI runtime = PASS`

同时，`ColossalAI enable/disable` 的 A/B benchmark 应继续统一收口于 `M7.6`，但 benchmark 结果不应被混入 formal PASS 的硬门槛。对该路径的正式结论应分三层表达：

- `formal PASS`：证明 `ColossalAI` 路径已真实生效并形成完整 artifact 闭环
- `benchmark`：证明 `ColossalAI` 相对 `single_process` 或 `nccl` 是否带来可复现性能/显存/稳定性收益
- `deployment positioning`：基于 benchmark 结果，再决定其结论应为 `supported but optional` 还是 `recommended default`

截至当前版本，已审计 artifact 仍主要对应 `single_process` 或 `nccl` 路径，因此 `ColossalAI` 目前只能被表述为 `M7.6` 已定义但尚待 formal rerun 与 benchmark 验证的候选 distributed runtime backend，不得提前表述为默认推荐路径。

---

## 五、当前版本的关键收口

### 5.1 M4 收口

- `distributed gate` 已从“规则未成立”推进为“正式证据成立”
- `DDP + TorchDynamo` graph break 已被 compile-friendly unwrap 路径化解
- 当前 `M4` 可以作为 `UPKG_1.1` 中 training route 的正式 evidence

### 5.2 M5 收口

- `oMLX` fallback 已统一收敛到 `dflash`
- 不再依赖外部 `dflash-mlx` 插件，改用 repo 内最小 runtime/shim
- canonical `pass2` 已经明确承认 `omlx_dflash` 为正式可接受 provider

### 5.3 Runtime 稳定性要求

- 调试探针、外部 debug server、临时 observability hook 必须 `fail-open`
- 这些机制可以记录问题，但不能反向导致正式 gate 自己失败
- 若 low-memory fallback 依赖远端模型仓库，正式 rerun 应优先固定到本地 snapshot 或等价可复现模型路径

### 5.4 M7 系列收口

- `M7 / M7.1` 已有完整 core artifact，可证明 dynamic compile、state compression、replay、audit 同时成立
- `M7.2` 与 `M7.3` 已不再只是白皮书目标，而是都有可引用 `report.json`
- `M7.4` 已把 `dflash + TrueOrthoKDA` 从名义契约推进到可检查合同与 runtime evidence
- `M7.5` 当前版本采用“双轨正式收口”：`API compatibility` 与 `TrueOrthoKDA active runtime` 都必须能独立给出 PASS artifact
- `M7.6` 已在 `host1/host2 + L20N 72GB + eRDMA` 路径上完成 dual-node formal rerun，并正式收口为 `transport foundation PASS + NCCL real-chain PASS + DeepEP real-chain PASS + m7.6 formal gate PASS`

---

## 六、建议的正式产物归档

每次 `UPKG_1.1` 重跑，建议至少保留：

- `M1-M7.5` 各自的 `report.json`
- 本次使用的 `GGUF` 路径
- 本次使用的 `backend fingerprint lock` 路径
- 若有 fallback：
  - fallback manifest
  - compile mode
  - provider
  - deploy unit
- 若有 distributed：
  - `world_size`
  - collective backend
  - dispatch 结果
  - performance gate 结果
- 若有 `M7 / M7.1`：
  - `dynamic_trace_l1`
  - `state_compression_summary`
  - `soft_rt_replay`
  - `industrial_audit`
- 若有 `M7.5 active runtime`：
  - `runtime_evidence_path`
  - `state_kind`
  - `state_codec`
  - `compression_ratio`
  - `cpu_copy_count`

---

## 七、后续版本分界

`UPKG_1.1` 是当前正式生效的工程验收 gate，但它不是后续所有产品化诉求的最终落点。为了避免把模型产品化、agent 产品化、具身 comparative 全部混回 `UPKG_1.1`，后续版本边界应明确区分如下：

### 7.1 UPKG 2.0

`UPKG 2.0` 定位为：

- `通用模型产品化 gate`
- 把统一 kernel 从“工程可重跑”提升为“模型可交付、可发布、可治理”
- 聚焦模型产物、模型 contract、ABI 判定、runtime branch、模型交付归因

### 7.2 UPKG 3.0

`UPKG 3.0` 定位为：

- `通用 Unified Pipeline Kernel Agent 产品化 gate`
- 把 `M7 / M7.1 / M7.2 / M7.3` 既有 `audit / replay / trace / bridge` 升级为产品级验收
- 把 `agent + edge + runtime` 的统一 artifact、summary、failure attribution 写成正式 contract
- 把六元一体架构先收敛为通用 agent 场景下的产品化审计与归因框架

在 route 边界上，`UPKG 3.0` 还应明确固化：

- `agent domain = 主 pipeline + GUI-native route`
- `harness domain = 旧测试 / 验证专用 route`
- 前者是通用 agent 产品化主线，后者仅保留为历史验证与对照入口，不再作为默认产品 runtime 承载路径

`UPKG 3.0` 的详细 gate 规格已独立整理为：

- `docs/gate_whitepapers/CGC_UPKG_3_0_AGENT_PRODUCT_GATE_WHITEPAPER_v1.0_zh_CN.md`
  - 负责 `3.1-3.7` 的完整 gate 定义与第一批实施接线
- 本次正式 rerun 顶层索引：
  - `/private/tmp/upkg30_formal_pass_20260619/upkg30_completion_matrix.json`
  - `/private/tmp/upkg30_formal_pass_20260619/upkg30_formal_pass_manifest.json`
- release CLI 对外主入口：
  - `cgc gate upkg30`
  - `cgc gate upkg31`
  - `cgc gate upkg32`
  - `cgc gate upkg33`
  - `cgc gate upkg34`
  - `cgc gate upkg35`
  - `cgc gate upkg36`
  - `cgc gate upkg37`

`UPKG 3.0` 内部继续拆为七个正式 gate：

- `3.1 Kernel Core Product Gate`
- `3.2 Agent Runtime Gate`
- `3.3 Edge Bridge Product Gate`
- `3.4 Unified Artifact And Summary Gate`
- `3.5 Six-Element Audit And Attribution Gate`
- `3.6 Missing Capability Closure Gate`
- `3.7 Cloud-Edge Training And Inference Q2RL Gate`

对外 `cgc` 命令与底层承载 gate 的关系为：

- `upkg30 ->` 顶层聚合 `3.1-3.7`
- `upkg31 -> m7`
- `upkg32 -> m72`
- `upkg33 -> m73`
- `upkg34 -> m72`
- `upkg35 -> m72`
- `upkg36 -> m72`
- `upkg37 -> m77`

`UPKG 3.0` 在 gate 视角下，建议进一步固化为下表：

| Gate | 对应来源 | 最小正式产物 | 最小 PASS 条件 | 统一失败归因 |
|---|---|---|---|---|
| `3.1 Kernel Core Product Gate` | `M7 / M7.1` | `report.json`、`dynamic_trace_l1`、`state_compression_summary`、`soft_rt_replay`、`industrial_audit`、`events.jsonl / chain_head.json` | `compile_success_rate = 1.0`、`soft_rt_replay = PASS`、`event_integrity = 1.0`、`hash_chain_valid = 1.0` | `compile_failure`、`cache_instability`、`state_codec_failure`、`replay_deadline_failure`、`audit_chain_break` |
| `3.2 Agent Runtime Gate` | `M7.2` 与等价 agent route | `report.json`、`summary`、`stage_trace.jsonl`、runtime evidence、tool/workflow/runtime 结构化事件 | `dynamic_trace_l1 = PASS`、`soft_rt_replay = PASS`、`industrial_audit = PASS`，且至少完成一次真实 workflow dispatch | `workflow_plan_failure`、`tool_execution_failure`、`runtime_host_failure`、`environment_not_ready`、`state_handoff_failure` |
| `3.3 Edge Bridge Product Gate` | `M7.3` 与等价 edge delivery route | `publish_manifest.json`、`runtime_contract.json`、`bridge_info.json`、edge delivery evidence、bridge publish evidence | `publish_manifest / runtime_contract / bridge_info` 共享同一套 `matrix_axes`，且交付产物可从 `report.json` 或 `summary` 反向索引 | `publish_manifest_incomplete`、`runtime_contract_incomplete`、`bridge_info_incomplete`、`edge_delivery_failure`、`matrix_axes_mismatch` |
| `3.4 Unified Artifact And Summary Gate` | 所有 agent + edge + runtime route | `report.json`、`summary.json`、`stage_trace.jsonl`、`failure_attribution`、`matrix_axes`、artifact path index | `report.json` 仍是单一真源，`summary` 不得绕开 `report.json`，`failure attribution` 必须结构化 | `missing_single_source_of_truth`、`stage_trace_incomplete`、`matrix_axes_missing`、`artifact_index_missing`、`failure_attribution_missing` |
| `3.5 Six-Element Audit And Attribution Gate` | 通用六元一体 route | 六元分类 `events.jsonl`、audit hash chain、attribution summary、replay anchor | 六元事件必须统一入链，任一失败必须可定位到至少一个元与其上下游依赖，audit chain 不能断裂 | `model_element_missing`、`workflow_element_missing`、`environment_element_missing`、`perception_element_missing`、`execution_element_missing`、`memory_element_missing`、`cross_element_chain_break` |
| `3.6 Missing Capability Closure Gate` | `UPKG 3.0` 当前缺失项与 roadmap closure | `gap_register.json`、`closure_plan.json`、`workflow_dag_schema.json`、`trajectory_synthesis_spec.json`、`fine_tune_profile.json`、`dual_mode_governance.json`、`audit_alignment_spec.json` | 所有已声明缺口必须进入统一 `gap register`，且 `workflow -> trajectory -> fine-tune -> governance -> audit alignment` 必须给出完整 spec | `gap_register_missing`、`closure_plan_missing`、`workflow_dag_schema_missing`、`trajectory_synthesis_spec_missing`、`fine_tune_profile_missing`、`dual_mode_governance_missing`、`audit_alignment_spec_missing`、`unmapped_gap_item` |
| `3.7 Cloud-Edge Training And Inference Q2RL Gate` | 端云训练、GUI agent 训练后模型下推端侧、Q2RL 后训练 | `cloud_edge_training_inference_mode.json`、`gui_agent_edge_inference_contract.json`、`q2rl_post_training_profile.json`、`edge_deployment_bundle_manifest.json`、`cloud_edge_q2rl_evaluation_plan.json` | 必须明确定义 `cloud_train -> edge_infer` 产品模式、GUI agent 训练后模型端侧推理 contract，以及 `Q2RL` 对 `workflow / tool_call / runtime_host / screenshot / replay` 的 reward 绑定 | `cloud_edge_training_inference_mode_missing`、`gui_agent_edge_inference_contract_missing`、`q2rl_post_training_profile_missing`、`edge_deployment_bundle_manifest_missing`、`cloud_edge_q2rl_evaluation_plan_missing` |

### 7.3 UPKG 4.0

`UPKG 4.0` 定位为：

- 具身 runtime / comparative / benchmark gate
- 专门承接 `realtime-vla`、官方 `psi0` 对照、具身 task gain、control quality、advanced replay/audit

`UPKG 4.0` 负责的内容至少包括：

- `realtime-vla` 作为具身 runtime host 的正式验收
- 官方 `psi0` 训练+推理对照基线
- `无 realtime-vla` 与 `有 realtime-vla` 的正式 comparative
- `>5x` benchmark threshold
- `psi0 feedback`
- `view-invariance`
- `one-shot`
- `structured conditioning smoke test`
- 真实 `atom library` 挂载

### 7.4 一句话边界

- `UPKG_1.1`：`M1-M7.5` 工程验收 gate
- `UPKG 2.0`：通用模型产品化 gate
- `UPKG 3.0`：通用 agent 产品化 gate
- `UPKG 4.0`：具身 comparative / benchmark gate

---

## 八、一句话结论

截至 `2026-06-19`，`UPKG_1.1` 在当前代码基线下已具备扩展后的正式收口条件：

- `M4` 已有真 distributed training evidence
- `M5` canonical `pass2` 已补成正式 `PASS`
- `M7 / M7.1 / M7.2 / M7.3 / M7.4 / M7.5` 都已有可回放 artifact
- `Unified Pipeline Kernel` 在 `M1-M7.5` 范围内已不再只是设计要求，而是有可引用 artifact 支撑的正式 gate 体系
