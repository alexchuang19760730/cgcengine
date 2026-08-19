# CGC 端云协议强制 Gate 白皮书 v1.0

## 1. 目标

本文定义 CGC 当前正式启用的端云协议强制口径。自本版开始，以下三项不再只是“已支持”或“可观测”，而是统一收敛为 mandatory protocol gate：

- `protocol_family` 必须为 `trueorthokda`
- `state_codec` 必须为 `cq4`
- `zero_copy_vram_real.status` 必须为 `PASS`

凡是接入 `pipeline_kernel_contract_artifacts`、`contract_manifest.json`、`system_execution_manifest.json`、`runtime_contract.json` 或 `M7.5/M7.6 runtime evidence` 的相关 gate，只要上述任一条件不达标，即应直接返回 `FAIL`，不得再以降级投影、静态声明或兼容 fallback 视为通过。

## 2. 设计原则

### 2.1 宣告与实测分离

- 宣告层由 `runtime_protocol_contract` 负责，定义本次运行“要求是什么”
- 实测层由 `zero_copy_vram_real`、`compression_effective`、`effective_*` 负责，定义本次运行“实际发生了什么”
- mandatory gate 同时检查宣告与实测，避免出现“声明已经打开，但运行时并未真正生效”的假阳性

### 2.2 不允许静默降级

以下情况一律视为协议不达标：

- `protocol_family != trueorthokda`
- `state_codec != cq4`
- `zero_copy_vram_real.status != PASS`
- 仅存在 `projection`、`declared`、`SKIP`，但没有真实 zero-copy 证据

## 3. 强制字段与证据来源

| 字段 | 层级 | 强制值/要求 | 主要来源 |
| --- | --- | --- | --- |
| `runtime_protocol_contract.protocol_family` | 宣告 | `trueorthokda` | `runtime_contract_bootstrap.py`、`contract_manifest.json`、`system_execution_manifest.json` |
| `runtime_protocol_contract.state_kind` | 宣告 | `kda_state_v1` | 同上 |
| `runtime_protocol_contract.state_codec` | 宣告 | `cq4` | 同上 |
| `zero_copy_vram_real.status` | 实测 | `PASS` | `m75_trueorthokda_active_runtime.json`、`m76` runtime evidence |
| `zero_copy_vram_real.cpu_copy_count` | 实测 | `0` | edge runtime local resume evidence |
| `zero_copy_vram_real.uma_buffer_used` | 实测 | `true` | edge runtime local resume evidence |
| `zero_copy_vram_real.device_resume_consumed` | 实测 | `true` | edge runtime local resume evidence |
| `compression_effective` | 实测 | 非 mandatory，但保留观测 | `M7.5/M7.6 runtime evidence` |
| `effective_collective_backend` | 实测 | 非 mandatory，但保留观测 | `M7.5/M7.6 runtime evidence` |
| `effective_cuda_graph` | 实测 | 非 mandatory，但保留观测 | `M7.5/M7.6 runtime evidence` |
| `effective_dispatch_backend` | 实测 | 非 mandatory，但保留观测 | `M7.5/M7.6 runtime evidence` |
| `effective_storage_backend` | 实测 | 非 mandatory，但保留观测 | `M7.5/M7.6 runtime evidence` |

## 4. 当前接入范围

本轮已经将 mandatory protocol gate 接入以下链路：

- `M7.2`：从 `pipeline_kernel_contract_artifacts` 投影协议字段，并把不达标写入 `runtime_evidence` 与 gate 结果
- `M7.3`：在 `runtime_contract.json` 中固化 `mandatory_protocol_gate`
- `M7.5`：在 active runtime evidence 中直接校验 `trueorthokda + cq4 + zero-copy`
- `M7.6`：在 remote runtime evidence 中直接校验同一组 mandatory 条件
- `M7.7` / `M7.8`：从上游 `contract_manifest/system_execution_manifest` 投影协议字段，并在 gate 级统一失败
- `UPKG_2.0 model session`：在 `cgc model swe-verified` session sidecar 中同步展示 `runtime_protocol_contract + effective_*`

## 5. 判定口径

统一使用 `mandatory_protocol_gate` 字段记录本次协议判定，推荐结构如下：

```json
{
  "status": "FAIL",
  "failure_code": "mandatory_protocol_gate_failed",
  "reason": "state_codec_mismatch:expected=cq4,actual=zlib_torch_save_bytes; zero_copy_vram_real_not_pass:actual=SKIP",
  "requirements": {
    "protocol_family": "trueorthokda",
    "state_codec": "cq4",
    "zero_copy_vram_required": true
  }
}
```

只要 `mandatory_protocol_gate.status != PASS`，相关 gate 即必须失败。

## 6. 当前结论

可以说“端云协议已支持”，因为：

- 协议 schema 已正式存在
- `runtime_protocol_contract` 已贯通到 session、manifest、runtime contract
- `effective_*` 与 `zero_copy_vram_real` 已贯通到 runtime evidence

但更严格地说，只有当 `mandatory_protocol_gate = PASS` 时，才能说本次运行真正满足 CGC 当前正式要求的端云协议。

## 7. 后续建议

- 将远端主机上的 `M7.6 runtime evidence` 升级到原生输出 `runtime_protocol_contract + mandatory_protocol_gate`，逐步移除 projection fallback
- 将 `cq4` 从“默认值”继续推进为“真实编码路径”，避免仅声明为 `cq4` 而运行时仍走旧 codec
- 将 `zero_copy_vram_real` 的采样路径推广到更多 edge resume 场景，减少 `SKIP` 或兼容链路
