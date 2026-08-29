# M4 部署指导书：接入 CGC 算力共享

> 让 Mac M4 成为端云架构的「云侧」节点，为 Windows/鸿蒙端侧提供 prefill 加速

---

## 0. 前提

- Mac M4 已有 CGC engine 代码（TurboFieldfare）
- Windows 机器 IP 已知（如 192.168.1.100）
- 两台机器在同一局域网

---

## 1. Checkout dev branch

```bash
cd /path/to/cgcengine0823
git fetch origin
git checkout dev
git pull origin dev
```

验证新文件到位：
```bash
ls CGC-main/cgc_engine/pd/discovery.py    # 应有 DeviceProfile 类
ls CGC-main/cgc_engine/pd/router.py       # ComputeRouter
```

---

## 2. 测量本机 prefill/decode 速度

在 M4 上跑 benchmark，填入 DeviceProfile：

```bash
cd CGC-main/cgc_engine/pd
python -c "
import sys; sys.path.insert(0, '.')
from discovery import DeviceProfile
p = DeviceProfile.detect_local()
print(f'RAM: {p.total_ram_gb}GB')
print(f'GPU: {p.gpu_type}')
print(f'Score: {p.compute_score}')
print(f'CPU: {p.cpu_cores} cores')
"
```

然后用 llama-bench 测 Qwen3.6-35B 的 prefill 和 decode 速度：

```bash
# Prefill speed (prompt processing)
llama-bench -m Qwen3.6-35B-A3B-Q4_K_M.gguf -ngl 99 -p 1024 -n 0

# Decode speed (token generation)
llama-bench -m Qwen3.6-35B-A3B-Q4_K_M.gguf -ngl 99 -p 32 -n 128
```

记下两个数值，后面要用。

---

## 3. 启动 emit server（Mac A，Gemma4 或 Qwen3.6 prefill）

确保 TurboFieldfare 的 emit endpoint 在跑：

```bash
# 启动 emit server（Mac M4）
# 端口 8080，暴露 /v1/cgc/emit
```

验证：
```bash
curl http://127.0.0.1:8080/v1/cgc/emit -X POST \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 0}'
```

---

## 4. 启动 resume server（Windows，Qwen3.6 decode）

在 Windows 上启动 decode server：

```bash
# 确保 port 1234 监听
netstat -ano | grep ":1234"
```

---

## 5. 端云联调测试

### 5.1 Mock 测试（无需真机）

```bash
cd CGC-main/cgc_engine/pd
python mock_demo.py
```

应该看到：
```
[1] Mac A emit: hidden [seq, 2816] OK
[2] MoT-h translate: 2816 -> 2048 OK
[3] Context Replay: KV cache restored OK
[4] Mac B resume: decode stream OK
Pipeline complete!
```

### 5.2 真机端云测试

```bash
# 在 Windows 上运行（Mac M4 作为 prefill 节点）
python -c "
import sys; sys.path.insert(0, '.')
import discovery, router

# 本机 (Windows, decode)
local_p = discovery.DeviceProfile.detect_local()
local_p.prefill_tok_per_sec = {'qwen36_35b': 1.5}   # 实测值
local_p.decode_tok_per_sec = {'qwen36_35b': 1.4}     # 实测值
local = discovery.PDNode(node_id='win', host='127.0.0.1', port=1234,
                         status=discovery.NodeStatus.HEALTHY, profile=local_p)

# Mac M4 (prefill)
mac_p = discovery.DeviceProfile(total_ram_gb=32, gpu_type='M4-Max', gpu_vram_gb=24,
    cpu_cores=14, compute_score=85,
    prefill_tok_per_sec={'qwen36_35b': 120.0},  # 实测值！
    decode_tok_per_sec={'qwen36_35b': 22.0},    # 实测值！
    network_latency_ms=2)                         # ping Mac M4 的延迟
mac = discovery.PDNode(node_id='mac-m4', host='192.168.1.10', port=8080,
                       status=discovery.NodeStatus.HEALTHY, profile=mac_p)

r = router.ComputeRouter(local_node=local)
r.register(mac)

# 测试路由
for tok in [128, 1024, 4096]:
    d = r.select(prompt_tokens=tok, output_tokens=100, model='qwen36_35b')
    print(f'{tok} tok -> {d.mode} prefill={d.prefill_node.node_id} total={d.total_latency_ms/1000:.1f}s')
"
```

### 5.3 端到端推理测试

```bash
# Windows 上发请求，Router 自动选 Mac M4 做 prefill
curl http://localhost:9000/v1/cgc/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantum computing in detail", "max_tokens": 200}'
```

预期：Mac M4 做 prefill（快速），Windows 做 decode（慢但能用）。

---

## 6. 常见问题

| 问题 | 排查 |
|---|---|
| Mac M4 emit 超时 | 检查网络连通性 `ping 192.168.1.10` |
| MoT-h 翻译失败 | 检查 `MOT_H_CHECKPOINT` 环境变量 |
| hidden state 为空 | Mac M4 上模型未加载，检查 emit server |
| decode 卡住 | Windows 上 resume endpoint 未启动 |
| 路由全选 pure_edge | Mac M4 的 profile.compute_score < 50 |

---

## 7. 下一步

- [ ] 实测 Mac M4 的 prefill/decode 速度，更新 DeviceProfile
- [ ] 在 Mac M4 上启动 emit server
- [ ] 跑通 mock_demo.py
- [ ] 跑通真机端云测试
- [ ] 把 compute sharing 集成到 coordinator.py 的 generate 端点
