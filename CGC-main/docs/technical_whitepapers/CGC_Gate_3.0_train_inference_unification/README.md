# CGC_Gate_3.0_train_inference_unification

本目录存放 `CGC_Gate_3.0_train_inference_unification` 的正式白皮书骨架与配套 artifact。

Gate 3.0 是 CGC 体系下首个把训练侧（Megatrain / MLX-Tune）与推理侧（vLLM / SGLang / OMLX）收口为单一可治理边界的正式 composite gate。

## 当前状态

- `status: validated` — 27 个能力全部 done；正式 Gate 3.0 verify 与 self-harness 27/27 全通
- Megatrain / MLX-Tune 代码已完整落地并通过正式 gate 验收
- 分布式拓扑、C++ MoE Engine、偏好对齐训练器与 Slime 训推链路已并入 Gate 3.0 正式闭环
- Gate 3.0 已闭合「代码已落地 → 正式 gate-pass → self-harness 全绿」

## 目录内容

- `CGC_Gate_3.0_train_inference_unification_Technical_Whitepaper_v1.0_zh_CN.md`
  - Gate 3.0 的技术白皮书骨架
  - 定义训推一体的四条验收主链与 claim boundary
- `CGC_Gate_3.0_train_inference_unification_gate_map.json`
  - 面向机器消费的 gate map
  - 定义 27 个能力与 4 个验收维度（A/B/C/D）
  - 用于 `CLI summary`、`bundle audit`、`release checkin`、`dashboard/report`
- `CGC_Gate_3.0_train_inference_unification_summary.example.json`
  - 面向 UI / dashboard 的 27 能力汇总投影
- `CGC_Gate_3.0_train_inference_unification_checkin.example.json`
  - 面向 release/checkin 的 27 能力状态投影

## Gate 3.0 的核心语义

Gate 3.0 解决的是「训推一体是否已经存在并可被正式治理」的问题，核心验收仍围绕四条主链，但正式闭环已经扩展到 27 个能力：

| 维度 | 名称 | 关键校验对象 |
|---|---|---|
| A | 训推权重一致性 | `MegatrainVLLMBridge` 训练权重 → vLLM/HF/GGUF 无损转换 |
| B | KDA 正交基保留 | 训练 → 推理路径中 `TrueOrthoKDA` 正交基保留与可恢复性 |
| C | CGC SIMD 训推指令集共用 | `MegatrainCGCIntegration` 训推共享 SIMD 指令集 |
| D | MLX-Tune LoRA 端云协同 | Apple Silicon 端侧 LoRA/QLoRA → 云侧推理闭环 |

## 与 Gate 1.0 / 2.0 的关系

- `CGC_Gate_1.0_edge_cloud_autonomy` — 端云自治基座（validated）
- `CGC_Gate_2.0_layer_adaptive_edge_cloud_pd_disaggregation` — 层自适应 PD 解耦（base done）
- `CGC_Gate_3.0_train_inference_unification` — 训推一体闭合（本目录）

Gate 3.0 复用 Gate 1.0 的 `task_type_contract` 与 bundle governance 链，并扩展一条训推一致性的校验支线。

## 何时看这个目录

- 需要理解训推一体的正式 gate 边界
- 需要消费 3.0 的 `gate_map`（CLI summary / bundle audit / release checkin / dashboard）
- 需要把 Megatrain / MLX-Tune 代码资产升级为正式 gate-pass
- 需要给 `bundle review / verify / audit` 体系对齐训推一致语义

## 升级路径（已完成 ✅）

Gate 3.0 已从 `draft_skeleton` 升级为 `validated`：

1. ✅ 跑 `test_megatrain_mlx_magicompiler.py`，Dimension A/B/C 校验点全绿
2. ✅ 用 `check_megatrain.sh` 在 gs01 远端确认部署，Dimension D 端云闭环跑通
3. ✅ 在 `task_type_contract.json` 中新增训推支线字段
4. ✅ 在 `CGC_Release/checkins/gate_checkins.jsonl` 写入独立 PASS 记录
5. ✅ 把 `gate_map` 中 `integrated` 能力翻转为 `done`
6. ✅ 把白皮书 `status` 升级为 `validated`，填充 formally_claimable
7. ✅ 建立 `cgc_engine/agent/trainers/` 目录，MegaTrain/mlx-tune 功能完整对等
8. ✅ 同步至 host1 (39.106.118.206) / host2 (47.95.250.55)
9. ✅ 收敛 `summary/checkin/README` 到 `27 done`
10. ✅ 运行 `gate_test_framework.py --self-harness --gate CGC_Gate_3.0_train_inference_unification`，取得 `27/27 PASS`

详见白皮书 §7 验收路径。
