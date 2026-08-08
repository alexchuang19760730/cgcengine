# CGC Colibrì Hermes RoutePolicyV2 Integration

## 1. 目标

把 Colibrì 的系统策略能力整合进现有 `Hermes router + route_decision_v2 + edge_first_proxy`，但保持清晰边界：

- `Hermes` 负责学习和输出策略
- `route_decision_v2` 负责构建特征、执行硬门控
- `edge_first_proxy` 负责 grammar/json draft 的执行、降级和观测
- `CGC engine` 负责 residency / prefetch / verify 等底层执行

一句话原则：

**学习策略，不学习真源；学习偏好，不替代硬约束。**

---

## 2. 本版本会整合的 Colibrì 能力

下表只列会被本版本显式吸收的能力，不含 Colibrì 的全部底层实现细节。

| Colibrì 能力 | 是否整合 | 整合位置 | 说明 |
|---|---|---|---|
| 路由热度 / usage history / hot expert heat | 是 | `FeatureSchema` | 从静态模型尺寸升级到“静态 + 动态热度” |
| VRAM / RAM / NVMe 分层 residency 观念 | 是 | `FeatureSchema` + `RoutePolicyV2` | 抽象成 tier / bytes moved / residency policy |
| LRU + hot pin 学习缓存 | 是，抽象整合 | `FeatureSchema` | 进入特征，不把替换算法直接学进 SFT |
| cross-layer prefetch / pilot | 是 | `FeatureSchema` + `RoutePolicyV2` | 让 Hermes 决定是否激活、激进程度 |
| I/O 与 compute overlap 感知 | 是 | `FeatureSchema` + `RoutePolicyV2` | 进入策略判断，不学习 pthread/io_uring 细节 |
| dual-SSD / mirrored storage awareness | 是，第一版以特征形式整合 | `FeatureSchema` | 先让路由知道存储拓扑，执行可后补 |
| speculation must earn its keep | 是 | `RoutePolicyV2` + runtime guard | 是最关键的一条 |
| grammar / JSON constrained draft | 是 | `RoutePolicyV2` + `edge_first_proxy` | 直接对应 edge 侧可观测、可验收能力 |
| hardware-aware planner | 是 | `FeatureSchema` | 让 Hermes 学“什么机器适合什么策略” |
| token-exact correctness gate | 否，保留 deterministic | runtime guard | 不能交给 SFT |
| O_DIRECT / io_uring / loader threads | 否，保留 deterministic | engine runtime | 这些是执行参数，不是学习标签 |
| batch-union / slab layout / one-pread | 否，保留 deterministic | engine runtime | 底层实现不进 SFT |

---

## 3. 当前代码挂点

### 3.1 已有骨架

- `app/shared/route_decision.py`
  - 当前仍以静态 `model_size_gb / per_layer_gb / RTT / TFLOPS` 为主
- `app/shared/route_decision_v2.py`
  - 已有 `D1/D2/D3/D4` v2 骨架
  - 适合扩展动态热度与 draft policy 字段
- `app/shared/hermes_router.py`
  - 已有 `FourDMatrix` / `D4Decision` 结构
- `app/servers/edge_first_proxy.py`
  - 已有 `AcceptanceTracker`
  - 是 grammar/json draft mode 的天然执行挂点

### 3.2 推荐职责边界

| 模块 | 职责 |
|---|---|
| `route_decision_v2.py` | 统一特征构建、fallback 规则、硬门控执行 |
| `hermes_router.py` | SFT 模型推理，输出 `RoutePolicyV2` |
| `edge_first_proxy.py` | grammar/json draft 执行、收益跟踪、自动降级 |
| `CGC engine` | residency/prefetch/verify/transport 的实际执行 |

---

## 4. RoutePolicyV2 设计

`RoutePolicyV2` 建议替代当前只关注 `mode / draft_n / pivot_layer` 的输出，扩展为策略对象。

### 4.1 建议字段

```json
{
  "route_mode": "cache_hit | local_only | edge_pivot_draft | edge_draft_cloud_verify | cloud_only",
  "draft_mode": "off | mtp | grammar_json | hybrid",
  "draft_depth": 0,
  "pivot_layer": 0,
  "use_flashmoe": false,
  "residency_policy": "full_resident | warm_resident | streamed | hybrid_tier",
  "prefetch_policy": "off | conservative | aggressive",
  "streaming_policy": "buffered | direct_io | overlap_io_compute",
  "fallback_policy": "cloud_only | plain_mtp | disable_speculation",
  "response_contract": "plain | json | tool",
  "confidence": 0.0,
  "reason": ""
}
```

### 4.2 这些字段分别吸收的 Colibrì 能力

| RoutePolicyV2 字段 | 对应 Colibrì 能力 |
|---|---|
| `residency_policy` | VRAM/RAM/NVMe 分层放置 |
| `prefetch_policy` | `PILOT / PILOT_REAL / lookahead` |
| `streaming_policy` | `PIPE / DIRECT / overlap I/O and compute` |
| `draft_mode` | `MTP / grammar draft / hybrid draft` |
| `fallback_policy` | speculation 收益不足时关闭 |
| `response_contract` | constrained JSON / tool-call 路径 |

---

## 5. FeatureSchema 设计

本次升级的核心不是只加几个字段，而是把 `route_decision.py` 的输入从静态模型尺寸升级为：

**静态特征 + 动态热度 + runtime 收益指标**

### 5.1 建议分组

#### A. Static Model Features

```json
{
  "model_name": "deepseek-v4-flash",
  "params_b": 671.0,
  "num_layers": 61,
  "is_moe": true,
  "num_experts": 256,
  "experts_per_tok": 8,
  "model_size_gb": 300.0,
  "per_layer_gb": 4.9,
  "quantization": "fp8",
  "has_native_mtp": true
}
```

#### B. Dynamic Heat Features

这些是本版本最重要的新特征。

```json
{
  "expert_hit_rate_ema": 0.0,
  "hot_expert_ratio": 0.0,
  "recent_expert_heat_entropy": 0.0,
  "layer_hotness_topk": [],
  "warm_pin_gb": 0.0,
  "repin_recent_count": 0,
  "prefetch_hit_rate_ema": 0.0,
  "predicted_cold_bytes_mb": 0.0,
  "predicted_bytes_to_read_mb": 0.0
}
```

#### C. Storage / Runtime Features

```json
{
  "nvme_bw_gbps": 0.0,
  "io_queue_depth": 0,
  "secondary_nvme_available": false,
  "disk_mirror_mode": "none | mirror | split",
  "multi_store_read_gain_estimate": 0.0,
  "unified_memory": false,
  "numa_mode": "single | interleaved | multi_socket"
}
```

#### D. Speculation ROI Features

```json
{
  "accept_rate_ema": 0.0,
  "verify_cost_ms": 0.0,
  "draft_cost_ms": 0.0,
  "recent_speculation_roi": 0.0,
  "recent_json_success_rate": 0.0,
  "grammar_accept_rate_ema": 0.0,
  "grammar_mode_roi": 0.0
}
```

#### E. Request / Contract Features

```json
{
  "prompt_has_code": false,
  "prompt_is_json_task": false,
  "prompt_is_tool_task": false,
  "response_contract_hint": "plain | json | tool",
  "cache_hit_rate": 0.0
}
```

---

## 6. SFT Label Schema 设计

SFT 不应该直接学习底层 runtime 参数，而应该学习：

- 当前机器 / 模型 / 热度 / acceptance 状态下
- 应该采用什么策略组合

### 6.1 建议标签

```json
{
  "route_mode": "edge_draft_cloud_verify",
  "draft_mode": "grammar_json",
  "draft_depth": 4,
  "residency_policy": "hybrid_tier",
  "prefetch_policy": "aggressive",
  "streaming_policy": "overlap_io_compute",
  "fallback_policy": "disable_speculation",
  "response_contract": "json",
  "confidence": 0.86,
  "reason": "当前 JSON 约束请求 grammar_accept_rate 高，且 recent_speculation_roi 为正，适合启用 grammar_json draft。"
}
```

### 6.2 SFT 学习目标

| 学习项 | 是否建议 |
|---|---|
| 路径选择 (`route_mode`) | 是 |
| draft 开关和深度 | 是 |
| grammar/json draft 是否启用 | 是 |
| prefetch / residency 的策略层选择 | 是 |
| O_DIRECT / io_uring / thread 数 | 否 |
| correctness gate / fail-fast 阈值 | 否 |
| 接受率回退阈值 | 否 |

---

## 7. route_decision.py 如何升级

当前 `route_decision.py` 是“静态模型 + 硬件估算 → 路由”。

建议升级成三阶段：

### 7.1 Stage A: build_feature_schema()

构建统一 `FeatureSchema`：

- 继承当前 `D1/D2/D3`
- 新增 dynamic heat / storage topology / speculation ROI

### 7.2 Stage B: call_hermes_policy()

把 `FeatureSchema` 送入 Hermes，得到 `RoutePolicyV2`

### 7.3 Stage C: apply_hard_guards()

由 deterministic executor 做最终硬门控：

- `accept_rate_ema < threshold` → 关闭 speculation
- `grammar_accept_rate_ema < threshold` → 关闭 grammar draft
- `recent_json_success_rate < threshold` → 降级 plain route
- `predicted_cold_bytes_mb` 超预算 → 降级策略

一句话：

**Hermes 给建议，route_decision_v2 给最终可执行决策。**

---

## 8. edge_first_proxy.py 如何接 grammar/json draft mode

### 8.1 需要新增的策略输入

`edge_first_proxy` 接收 `RoutePolicyV2` 后，按以下字段执行：

- `draft_mode`
- `draft_depth`
- `response_contract`
- `fallback_policy`

### 8.2 需要新增的执行状态

```json
{
  "draft_mode_active": "grammar_json",
  "grammar_accept_rate": 0.78,
  "json_success_rate": 0.96,
  "speculation_roi": 0.18,
  "fallback_triggered": false,
  "fallback_reason": ""
}
```

### 8.3 需要新增的 runtime guard

- grammar accept rate 低于阈值自动关闭
- 最近 JSON 成功率低于阈值自动关闭
- draft ROI 连续为负自动关闭
- 所有关闭动作必须写入 `.json` 报告

---

## 9. 建议新增的 JSON 报告

为了满足现有静态契约和证据链，建议新增三份报告。

### 9.1 `route_policy_v2.json`

记录 Hermes 输出与最终执行策略：

```json
{
  "feature_schema_version": "v2",
  "policy_schema_version": "v2",
  "hermes_policy": {},
  "final_policy": {},
  "guard_overrides": []
}
```

### 9.2 `route_heat_snapshot.json`

记录动态热度真源：

```json
{
  "expert_hit_rate_ema": 0.0,
  "prefetch_hit_rate_ema": 0.0,
  "warm_pin_gb": 0.0,
  "predicted_bytes_to_read_mb": 0.0,
  "storage_topology": {}
}
```

### 9.3 `draft_mode_acceptance_report.json`

记录 grammar/json draft 的收益：

```json
{
  "draft_mode": "grammar_json",
  "accept_rate_ema": 0.0,
  "grammar_accept_rate_ema": 0.0,
  "json_success_rate": 0.0,
  "recent_speculation_roi": 0.0,
  "auto_disabled": false,
  "disable_reason": ""
}
```

---

## 10. 本版本明确不做的事

为了保持设计简洁，这一版不把以下内容学进 Hermes：

1. `O_DIRECT / io_uring / PIPE_WORKERS` 等底层执行参数
2. LRU / repin / hot-store 的具体替换算法
3. batch-union / slab layout / one-pread 等底层内存与 I/O 组织方式
4. token-exact correctness gate
5. hysteresis / fail-fast 的最终阈值判定

这些都必须留在 deterministic runtime 中。

---

## 11. 推荐实施顺序

### P0

1. 扩展 `route_decision_v2.py`
   - 增加 dynamic heat / storage / ROI 字段
2. 扩展 `edge_first_proxy.py`
   - 增加 `draft_mode=response_contract` 的 grammar/json 执行路径
3. 新增三份 `.json` 报告

### P1

4. 生成 `FeatureSchema → RoutePolicyV2` 的 SFT 样本
5. 用现有规则引擎 + acceptance tracker 生成 bootstrap labels

### P2

6. 把 Colibrì 的更多 runtime 能力往 CGC engine 吸收
   - dual-disk read policy
   - prefetch planner
   - residency planner

---

## 12. 最终结论

这个版本**会整合 Colibrì 的核心系统策略能力**，主要包括：

- 热度驱动路由
- 分层 residency
- prefetch / lookahead
- I/O-overlap 感知
- speculation ROI 门控
- grammar / JSON draft
- hardware-aware planning

但不会把 Colibrì 的底层执行细节整个塞进 Hermes。

正确边界是：

**Hermes 学策略，CGC / edge proxy / engine 保留真源、硬门控和验收。**
