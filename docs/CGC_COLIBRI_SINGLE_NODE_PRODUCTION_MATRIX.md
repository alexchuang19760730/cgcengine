# CGC Colibrì 单机量产目标矩阵

## 1. 目标

这份文档只讨论一个问题：

**在整合 Colibrì 的流式权重 / 热专家缓存 / residency / prefetch 能力后，哪些单机大模型值得作为 CGC + Hermes 的量产候选。**

这里的“单机”明确指：

- 单一设备或单一工作站
- 不依赖端云 split inference 才能完成主推理路径
- 允许使用 NVMe / RAM / VRAM 分层放置
- 允许模型以 streamed-MoE 方式运行，而不是要求全量常驻

---

## 2. 量产判定标准

本项目把“可量产”定义为满足以下门槛，而不是只看“能启动”：

| 指标 | P0 门槛 | P1 目标 | 说明 |
|---|---:|---:|---|
| TTFT | `<= 300ms` | `<= 200ms` | 单轮普通请求 |
| decode speed | `>= 30 tok/s` | `>= 45 tok/s` | 稳态 decode |
| 连续稳定性 | `5/5` 轮通过 | `20/20` 轮通过 | 不允许明显掉速或崩溃 |
| JSON/Grammar 成功率 | `>= 95%` | `>= 98%` | 只针对 grammar/json mode |
| speculation ROI | `> 0` | `>= 15%` | draft/verify 后净收益为正 |
| cold-load penalty | 可控 | 明显收敛 | 需由 heat/prefetch 报告量化 |

一句话：

**能跑不算过，TTFT、decode、稳定性、收益都要过。**

---

## 3. 候选模型矩阵

下面矩阵按“单机 streamed-MoE 可量产价值”排序，而不是按参数绝对大小排序。

| 优先级 | 候选模型 | 规模 | 类型 | 量产价值 | 当前判断 |
|---|---|---|---|---|---|
| P0 | `Gemma 4 26B-A4B` | 26B / A4B | 稀疏 MoE | 最适合作为首个量产目标 | 最优先 |
| P1 | `Qwen3-VL-30B-A3B` | 30B / A3B | 稀疏/多模态路线 | 更接近 30B 目标，适合第二阶段 | 高价值 |
| P1 | 同级 `26B~32B A3B/A4B` 稀疏模型 | 26B~32B | 稀疏 MoE | 可作为横向替换池 | 候补 |
| P2 | `70B+` streamed sparse MoE | 70B+ | 大型稀疏 MoE | 更能体现 Colibrì 价值，但不适合先打量产口径 | 研究型 |
| P3 | `600B+/744B+` 超大 MoE | 600B+ | 超大稀疏 MoE | 宣传价值高，但不应作为第一批产品目标 | 研究展示 |

---

## 4. 每个候选模型需要的 Colibrì 能力

### 4.1 P0: `Gemma 4 26B-A4B`

#### 为什么是 P0

- 规模足够接近 30B
- A4B 路线非常适合验证 streamed-MoE
- 相比超大模型，更容易先打穿 TTFT / decode / 稳定性 / ROI 四条线

#### 需要整合的 Colibrì 能力

| 能力 | 是否必需 | 作用 |
|---|---|---|
| dense backbone resident | 是 | 主干必须常驻，否则 TTFT 不稳定 |
| hot expert cache | 是 | 控制重复读盘开销 |
| dynamic expert heat learning | 是 | 支撑 Hermes 判断热度与路由收益 |
| prefetch / lookahead | 是 | 降低 cold expert 读取停顿 |
| overlap I/O and compute | 是 | 决定是否能冲过 `30 tok/s` |
| speculation ROI gating | 是 | draft mode 必须收益为正 |
| grammar / JSON draft | 建议 | 用于结构化请求高收益路径 |

#### 推荐量产定位

- 单机主模型候选
- 单机 local fallback 强候选
- 单机 JSON/tool-call 路径候选

---

### 4.2 P1: `Qwen3-VL-30B-A3B`

#### 为什么是 P1

- 更贴近你要的“30B A3B”
- 适合验证 Colibrì 能力是否能把 30B 级 streamed-MoE 推进到量产线
- 但目前本仓库里还缺完整闭环性能证据

#### 需要整合的 Colibrì 能力

| 能力 | 是否必需 | 作用 |
|---|---|---|
| streamed expert residency | 是 | 决定能否单机完成主路径 |
| hot pin / repin | 是 | 控制活跃 expert 波动 |
| pilot prefetch | 是 | 决定 TTFT 与稳态 decode |
| dual-store awareness | 建议 | 如果单机有多块 NVMe，可进一步隐藏 I/O |
| constrained draft | 建议 | 结构化任务通常更有收益 |
| ROI auto-disable | 是 | 防止 speculation 反而拖慢 |

#### 推荐量产定位

- 第二个单机量产候选
- Colibrì × Hermes 的标志性 30B 路线验证模型

---

### 4.3 P2/P3: 更大 sparse MoE

这类模型最能体现 Colibrì 的宣传价值，但不适合作为第一批量产门槛模型。

| 模型级别 | 推荐定位 |
|---|---|
| `70B+ sparse MoE` | 高配单机工作站研究线 |
| `600B+/744B+/900B+` | 展示线 / 科研线 / 私有化特殊场景 |

量产风险主要是：

- TTFT 波动太大
- 冷专家读取抖动明显
- decode 速度容易被 I/O 打穿
- 很难在常规设备上稳定满足 `>= 30 tok/s`

---

## 5. Colibrì 能力到 CGC/Hermes 的挂点映射

| Colibrì 能力 | CGC/Hermes 挂点 | 说明 |
|---|---|---|
| expert heat learning | `app/shared/route_decision_v2.py` | 进入 `FeatureSchema` 动态热度字段 |
| hot cache / pin / repin | `CGC engine` + `route_decision_v2.py` | runtime 真源 + 路由可见特征 |
| residency planner | `RoutePolicyV2.residency_policy` | Hermes 输出策略，engine 执行 |
| prefetch / pilot | `RoutePolicyV2.prefetch_policy` | Hermes 决策激进程度 |
| I/O-compute overlap | `RoutePolicyV2.streaming_policy` | 由 policy 决定是否启用更激进流式 |
| speculation ROI gating | `edge_first_proxy.py` | 由 Acceptance/ROI tracker 执行硬门控 |
| grammar / JSON constrained draft | `edge_first_proxy.py` | 结构化请求优先收益路径 |
| hardware/storage awareness | `FeatureSchema` | 给 Hermes 学“什么机器适合什么策略” |

---

## 6. 推荐新增的 FeatureSchema 字段

为了让 Hermes 能真正吸收 Colibrì 的单机 streamed-MoE 能力，建议在现有 `FeatureSchema` 上新增以下字段。

### 6.1 Heat / Residency

```json
{
  "expert_hit_rate_ema": 0.0,
  "hot_expert_ratio": 0.0,
  "warm_pin_gb": 0.0,
  "repin_recent_count": 0,
  "predicted_cold_bytes_mb": 0.0,
  "predicted_bytes_to_read_mb": 0.0
}
```

### 6.2 Storage / I/O

```json
{
  "nvme_bw_gbps": 0.0,
  "io_queue_depth": 0,
  "secondary_nvme_available": false,
  "disk_mirror_mode": "none | mirror | split",
  "multi_store_read_gain_estimate": 0.0,
  "io_compute_overlap_gain": 0.0
}
```

### 6.3 Speculation / Contract

```json
{
  "accept_rate_ema": 0.0,
  "recent_speculation_roi": 0.0,
  "grammar_accept_rate_ema": 0.0,
  "recent_json_success_rate": 0.0,
  "response_contract_hint": "plain | json | tool"
}
```

---

## 7. 推荐 RoutePolicyV2 输出

单机 streamed-MoE 路径里，`RoutePolicyV2` 至少要能表达这些策略：

```json
{
  "route_mode": "local_only",
  "draft_mode": "off | mtp | grammar_json | hybrid",
  "draft_depth": 0,
  "residency_policy": "full_resident | warm_resident | streamed | hybrid_tier",
  "prefetch_policy": "off | conservative | aggressive",
  "streaming_policy": "buffered | direct_io | overlap_io_compute",
  "fallback_policy": "disable_speculation | cloud_only | plain_mtp",
  "response_contract": "plain | json | tool",
  "confidence": 0.0,
  "reason": ""
}
```

其中最关键的 4 个输出是：

- `residency_policy`
- `prefetch_policy`
- `streaming_policy`
- `draft_mode`

这 4 个字段基本就承接了 Colibrì 最有价值的系统策略能力。

---

## 8. 验收 KPI 设计

### 8.1 主 KPI

| KPI | 验收方式 | 通过标准 |
|---|---|---|
| `ttft_ms` | 单轮测量 | `<= 300` |
| `decode_tps` | 稳态 decode 测量 | `>= 30` |
| `stable_rounds_passed` | 连续多轮压测 | `>= 5/5` |
| `speculation_roi` | draft/verify 净收益 | `> 0` |
| `json_success_rate` | JSON 合约任务集 | `>= 95%` |

### 8.2 次 KPI

| KPI | 说明 |
|---|---|
| `prefetch_hit_rate` | 判断 prefetch 是否真的起效 |
| `hot_cache_hit_rate` | 判断热专家缓存是否有价值 |
| `bytes_moved_per_token` | 判断 streamed-MoE 是否进入合理区间 |
| `cold_start_penalty_ms` | 判断首次请求是否还能接受 |
| `fallback_trigger_rate` | 判断 policy 是否过于激进 |

---

## 9. 推荐 JSON 报告格式

为了符合现有静态契约和真源管理要求，建议新增以下 4 份报告。

### 9.1 `single_node_candidate_matrix.json`

记录本次单机候选模型比较结果：

```json
{
  "schema_version": "v1",
  "hardware_profile": {},
  "candidates": [
    {
      "model": "gemma4-26b-a4b",
      "priority": "P0",
      "status": "PASS | PARTIAL | FAIL",
      "ttft_ms": 0.0,
      "decode_tps": 0.0,
      "stable_rounds_passed": 0,
      "notes": ""
    }
  ]
}
```

### 9.2 `colibri_heat_snapshot.json`

记录热度与 residency 真源：

```json
{
  "schema_version": "v1",
  "expert_hit_rate_ema": 0.0,
  "hot_expert_ratio": 0.0,
  "warm_pin_gb": 0.0,
  "predicted_bytes_to_read_mb": 0.0,
  "prefetch_hit_rate_ema": 0.0,
  "storage_topology": {}
}
```

### 9.3 `single_node_route_policy_report.json`

记录 Hermes 策略输出与 guard 覆盖结果：

```json
{
  "schema_version": "v1",
  "feature_schema": {},
  "hermes_policy": {},
  "final_policy": {},
  "guard_overrides": []
}
```

### 9.4 `single_node_acceptance_report.json`

记录 draft/grammar 路径收益：

```json
{
  "schema_version": "v1",
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

## 10. 推荐实施顺序

### P0

1. 以 `Gemma 4 26B-A4B` 作为单机量产首模
2. 扩展 `route_decision_v2.py` 的 heat / residency / ROI 特征
3. 扩展 `edge_first_proxy.py` 的 grammar/json draft + ROI auto-disable
4. 产出 4 份 `.json` 报告

### P1

5. 引入 `Qwen3-VL-30B-A3B` 做第二候选
6. 比较 `A4B vs A3B` 在单机 streamed-MoE 路径下的 TTFT / decode / bytes moved
7. 生成 `FeatureSchema -> RoutePolicyV2` 的 bootstrap labels

### P2

8. 继续下探更大 sparse MoE
9. 把 dual-store / mirrored storage / 更激进 prefetch 逐步吸收到 engine

---

## 11. 最终结论

整合 Colibrì 的真正意义，不是证明“大模型能启动”，而是把单机 streamed-MoE 推进到可量产区间。

在当前项目里，最应该先打穿的不是 `744B` 这类展示级目标，而是：

**`26B~30B A3B/A4B` 单机 streamed-MoE，TTFT 合理，decode >= 30 tok/s，且有完整 `.json` 证据链。**

当前推荐顺序：

1. `Gemma 4 26B-A4B` 作为 P0
2. `Qwen3-VL-30B-A3B` 作为 P1
3. 更大 sparse MoE 作为研究扩展线
