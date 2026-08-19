# Gate 6.0 探索命令到 M7.6 Manifest-First 正式命令映射

## 目的

这份对照表用于把左侧 `python cgc_engine/cli.py model verify ... --gate 6.0` 的探索语意，稳定挂接到右侧 `m76-dev manifest-first` 的正式产物链，避免手工维护两套口径。

治理边界如下：

- 探索命令保留人类可读、快速试探的表达方式。
- 正式链只接收已有稳定消费者的 runtime env、manifest annotation 与 formal artifact。
- 没有稳定 release-facing runtime contract 的探索参数，先进入 manifest annotation，而不直接膨胀成正式 runtime claim。

## 左右两侧

### 左侧：探索命令

```bash
python cgc_engine/cli.py model verify \
  --model four_instance \
  --gate 6.0 \
  --deepep \
  --l20n \
  --enable-speculative \
  --speculative-mode fusion \
  --dspark-budget 64 \
  --jetspec-branches 8
```

### 右侧：正式命令

首选正式入口：

```bash
CGC_M76_DEV_MODE=1 \
CGC_REQUIRE_FORMAL_EVIDENCE=1 \
CGC_FORMAL_SUITE=swe_bench_verified_500 \
CGC_M76_ENABLE_DEEPEP=1 \
CGC_REQUESTED_DISPATCH_BACKEND=deepep \
CGC_SERVICE_TOPOLOGY_BACKEND=ray_cluster_dual_host \
cgc m76-dev --output-dir ComputeGraphCompiler-main/Output/cli_gate_m76
```

当前 repo 内可执行 fallback：

```bash
CGC_M76_DEV_MODE=1 \
CGC_REQUIRE_FORMAL_EVIDENCE=1 \
CGC_FORMAL_SUITE=swe_bench_verified_500 \
CGC_M76_ENABLE_DEEPEP=1 \
CGC_REQUESTED_DISPATCH_BACKEND=deepep \
CGC_SERVICE_TOPOLOGY_BACKEND=ray_cluster_dual_host \
python3 -c 'from cgc_engine.product import run_m76_gate; import json; print(json.dumps(run_m76_gate(output_dir="ComputeGraphCompiler-main/Output/cli_gate_m76"), ensure_ascii=False, indent=2))'
```

## 参数映射表

| 左侧探索参数 | 右侧正式落点 | 类型 | 说明 |
|---|---|---|---|
| `--gate 6.0` | `CGC_M76_DEV_MODE=1` | runtime env | 开启 `m76-dev` 正式链 |
| `--gate 6.0` | `CGC_REQUIRE_FORMAL_EVIDENCE=1` | runtime env | 强制 formal evidence |
| `--gate 6.0` | `CGC_FORMAL_SUITE=swe_bench_verified_500` | runtime env | 固化 Gate 6.0 当前正式 suite |
| `--deepep` | `CGC_M76_ENABLE_DEEPEP=1` | runtime env | 让 `m76` runtime contract 宣告 DeepEP |
| `--deepep` | `CGC_REQUESTED_DISPATCH_BACKEND=deepep` | runtime env | 对齐 `runtime_protocol_contract.requested_dispatch_backend` |
| `--l20n` | `CGC_SERVICE_TOPOLOGY_BACKEND=ray_cluster_dual_host` | runtime env | 对齐双机正式拓扑口径 |
| `--profile ep16_tp1` | `CGC_DEEPEP_PARALLEL_PROFILE=ep16_tp1` | runtime env | 仅在 profile 命中 `epXX_tpYY` 时提升 |
| `--model` | `manifest_annotations.source_model` | manifest annotation | 视为探索侧 symbolic token，不强行映射成 runtime model path |
| `--profile` | `manifest_annotations.source_profile` | manifest annotation | 非 `epXX_tpYY` 情况保留为探索注解 |
| `--bundle` | `manifest_annotations.source_bundle` | manifest annotation | 供 formal summary / checkin 回溯来源 |
| `--strict` | `manifest_annotations.source_strict` | manifest annotation | 保留探索约束，不直接扩张 release claim |
| `--enable-speculative` | `manifest_annotations.requested_capabilities.enable_speculative` | manifest annotation | 当前先按 manifest-first 挂接 |
| `--speculative-mode` | `manifest_annotations.requested_capabilities.speculative_mode` | manifest annotation | 当前不直接写入稳定 runtime env |
| `--dspark-budget` | `manifest_annotations.requested_capabilities.dspark_budget` | manifest annotation | 当前不直接写入稳定 runtime env |
| `--jetspec-branches` | `manifest_annotations.requested_capabilities.jetspec_branches` | manifest annotation | 当前不直接写入稳定 runtime env |
| `--eplb` | `manifest_annotations.requested_capabilities.eplb` | manifest annotation | 当前保留为 formalized exploration |
| `--waterfill` | `manifest_annotations.requested_capabilities.waterfill` | manifest annotation | 当前保留为 formalized exploration |
| `--lplb` | `manifest_annotations.requested_capabilities.lplb` | manifest annotation | 当前保留为 formalized exploration |

## 正式产物链

映射完成后，正式链应产出并回填以下 artifact：

- `Output/cli_gate_m76/gate6_exploration_to_m76_manifest_mapping.json`
- `Output/cli_gate_m76/runtime_evidence/`
- `Output/cli_gate_m76/m76_heterogeneous/m76_report.json`
- `Output/cli_gate_m76/m76_heterogeneous/summary.json`
- `Output/cli_gate_m76/m76_heterogeneous/latest.json`
- `system_execution_manifest.json`

其中：

- `gate6_exploration_to_m76_manifest_mapping.json` 负责保存左侧探索命令、右侧正式命令、env 提升结果与 manifest annotation。
- `m76_report.json` / `summary.json` / `latest.json` 继续作为正式 release-facing 产物。
- `system_execution_manifest.json` 继续作为 manifest-first 单一真源。

## SWE Verified 500 CLI

现在可直接使用三条稳定 alias：

```bash
python cgc_engine/cli.py validate --all --print-json

python cgc_engine/cli.py validate --capability swe_verified_500 --print-json

python cgc_engine/cli.py model --validate-capability swe_verified_500 --print-json

python cgc_engine/cli.py model --task-type swe --model swe_verified --print-json

python cgc_engine/cli.py model --fusion-config swe --model swe_verified --print-json
```

它们都会统一翻译到同一条正式链：

- `CGC_FORMAL_SUITE=swe_bench_verified_500`
- `CGC_PD_MODE=cloud_prefill_edge_decode`
- `m76-dev manifest-first`
- `m76_heterogeneous/*` formal artifact

当前 formal suite 状态可保守表述为：

- `swe_verified_500`: `PARTIAL`
- 已完成首批正式收集，但仍未闭合到 `500/500 submission`
- 历史样本仍可见 `trajectory_count=1, submitted_count=0`
- `validate --all` 会显式输出这条能力的状态摘要与证据引用
- `validate --all` 现也会一并输出：
  - `dflash: PASS`
  - `jetspec: CONFIGURED`
  - `fusionroute: PASS`
- 当前 `swe_verified_500` 必须按双层口径解读：
  - `formal_chain_status=PASS`
  - `official_eval_status=SUBMITTED`
  - `claimable=false`
  - `swe_verified_passed_tasks=0`
- 这表示 `suite_name=swe_verified_500` 已进入通过的 `upkg21` formal chain，但 capability 仍不可 claimable，不能把 formal chain `PASS` 误写成 capability `PASS`
- 当前 `jetspec` 的 `CONFIGURED` 含义是：探索桥接已请求 `fusion/jetspec` 语义，但依 Gate6 治理边界，它仍停留在 bridge/manifest 注解层，尚未升级为稳定 release-facing runtime contract
- 当前 `fusionroute` 的 `PASS` 含义是：`upkg21` 已给出 `selected_route`，同时 `m76 latest` 能给出 `requested_dispatch_backend + service_topology_backend`

## 正式 CLI 子命令

如需直接使用 bridge 子命令：

```bash
python cgc_engine/cli.py model bridge-m76 \
  --model swe_verified \
  --gate 6.0 \
  --deepep \
  --l20n \
  --enable-speculative \
  --speculative-mode fusion \
  --dspark-budget 64 \
  --jetspec-branches 8 \
  --print-json
```

其中 `--model swe_verified` 可直接对齐当前 `SWE Verified 500` 探索口径，而正式链仍固定收敛到：

- `CGC_FORMAL_SUITE=swe_bench_verified_500`
- `m76-dev manifest-first`
- `m76_heterogeneous/*` formal artifact

## Wrapper 脚本

repo 内也保留可重用 wrapper 脚本：

```bash
python cgc_engine/tools/scripts/run/gate6_model_verify_to_m76_manifest.py \
  --model four_instance \
  --gate 6.0 \
  --deepep \
  --l20n \
  --enable-speculative \
  --speculative-mode fusion \
  --dspark-budget 64 \
  --jetspec-branches 8 \
  --print-json
```

如需直接执行当前可运行 fallback：

```bash
python cgc_engine/tools/scripts/run/gate6_model_verify_to_m76_manifest.py \
  --model four_instance \
  --gate 6.0 \
  --deepep \
  --l20n \
  --run-fallback
```

## 当前结论

- `Gate 6.0` 的探索语意现在可以通过一份稳定 mapping artifact 挂到 `m76-dev manifest-first` 正式链。
- 正式入口已不再依赖额外脚本；可直接走 `python cgc_engine/cli.py model bridge-m76 ...`。
- 只有 `DeepEP`、`dual-node topology` 与 `epXX_tpYY profile` 这类已有稳定消费者的参数，才提升成正式 env。
- `DSpark / JetSpec / Fusion speculative` 等探索性语意，当前先保留在 manifest annotation，等待稳定 runtime contract 再升级为 release-facing claim。
