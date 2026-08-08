# Debug: SWE500 TP4EP4 FusionRoute 四实例跑一题验证

- **Session ID**: `swe500-tp4ep4-fusionroute-4instance`
- **Status**: [OPEN]
- **Created**: 2026-07-07
- **Goal**: 在 host1 & host2 用 TP4EP4 FusionRoute + DeepSeek Flash V4 四实例跑一题 SWE smoke test，通过后跑 500 题

## 现状观测 (Observe)

### 四实例拓扑 (来自 four_instance_topology.json)
| instance | host | GPU | gateway | backend | tp | ep |
|----------|------|-----|---------|---------|----|----|
| inst1 | host2 (47.95.250.55) | 0-3 | 50053 | 30000 | 4 | 4 |
| inst2 | host1 (39.106.118.206) | 4-7 | 50063 | 30010 | 4 | 4 |
| inst3 | host2 (47.95.250.55) | 4-7 | 50073 | 30020 | 4 | 4 |
| inst4 | host1 (39.106.118.206) | 0-3 | 50083 | 30030 | 4 | 4 |

### Health Check 结果 (2026-07-07)
- 四实例 `status: ok`, `backend_ready: true`, `tp_size:4, ep_size:4` ✓
- 模型: `/data/models/DeepSeek-V4-Flash-UD-IQ2` (IQ2 量化)
- **`max_model_len: 256`** ← 关键异常
- 所有端口绑定 `127.0.0.1`（仅本机可达）
- GPU memory: 每个 ~3.9GB / 73GB, util 0%
- host1 残留: `debug-server.py --session tp4ep4-swe-timeout` + 多个 swe-bench eval docker (astropy-12907)

### 之前 debug 上下文 (来自用户)
- 第 4 次 LM query 长请求 timeout 600100ms (rid=089db19b-...)
- 8001 proxy 499 600100.6ms, gateway-port-8001 worker CANCELLED 600098.3ms
- host2_backend_30000_scoped_restart.log 不覆盖 05:31→05:41 历史窗口

## 假设 (Hypotheses)

- **H1 (max_model_len=256 限制)**: 四实例 max_model_len=256 太小，SWE 题目 (>256 token) 会被拒或卡住导致 timeout。观测点：发 >256 token 请求看 error/timeout。
- **H2 (IQ2 量化质量)**: DeepSeek-V4-Flash-UD-IQ2 极低比特量化，输出可能不 coherent。观测点：smoke test 输出质量。
- **H3 (gateway→backend 路由)**: health ok 但需确认真实推理能完成。观测点：smoke test 200 响应。
- **H4 (runner fallback rule-based)**: swe_verified_500_real_test.py 的 FusionRouteClient 可能 fallback 到 RuleBasedSolver。观测点：读 FusionRouteClient.generate 实现。
- **H5 (端口 127.0.0.1 不可达)**: 端口仅本机可达，runner 需在 host 上跑或 SSH tunnel。观测点：确认 runner 执行位置。

## 计划
1. 读 FusionRouteClient.generate 确认 H4
2. 在 host1 本地发最小推理请求 (smoke test) 确认 H1/H2/H3
3. 若 smoke test 通过 → 跑 1 道 SWE 题
4. 若 1 题通过 → 跑 500 题

## 证据与假设判定

### Smoke Test (host1 inst2:50063)
- 短 prompt (20 token): ✅ 200, `"content":"4"`, latency 正常
- 长 prompt (311 token): ❌ 400 `"input is longer than context length (256 tokens)"`

### FusionRouteClient 实现 (swe_verified_500_real_test.py:206-300)
- 端点配置: `http://host2:50053/v1/completions` 等 (用 hostname)
- `use_local_fallback=True` 默认开 → 云端失败 fallback 到 LocalLLMClient → RuleBasedSolver
- /etc/hosts 无 host1/host2 映射 → hostname 不可解析 → 必 fallback

### sglang server 部署
- 通过 ray serve (`ray::ServeReplica:default:cgc-sglang-openai-gateway`) 部署
- 非 `sglang.launch_server` 命令行，max_model_len 在 ray serve 配置中设定
- model: `/data/models/DeepSeek-V4-Flash-UD-IQ2`, type `deepseek_v4`

### 假设判定
- **H1 CONFIRMED**: max_model_len=256 是硬阻塞，SWE 题 (>256 token) 直接 400
- **H2 PARTIAL**: 短 prompt 质量OK，长任务未验证
- **H3 CONFIRMED**: gateway→backend 短 prompt 推理正常
- **H4 CONFIRMED**: FusionRouteClient 会 fallback 到 rule-based 假结果
- **H5 CONFIRMED**: 端口 127.0.0.1 + /etc/hosts 无映射，外部不可达

## 阻塞总结
跑真实 SWE 500 有两个硬阻塞：
1. **max_model_len=256**: SWE 题目远超 256 token，backend 直接 400 拒绝
2. **网络不可达**: FusionRouteClient 用 hostname 连云端，但端口绑 127.0.0.1 且 /etc/hosts 无映射 → runner 会静默 fallback 到 rule-based 假结果

## 进度
- [x] Observe: 四实例 health + 拓扑
- [x] 读 FusionRouteClient 实现 (H4 confirmed)
- [x] smoke test 单实例推理 (H1/H3 confirmed)
- [ ] 用户决策：如何处理 max_model_len=256 + 网络阻塞
- [ ] 跑 1 道 SWE 题
- [ ] 跑 500 题
