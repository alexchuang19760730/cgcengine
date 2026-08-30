# DSH + agent_harness 整合

## 概述

将 DeepSeek Harness (DSH) 整合进 agent_harness，利用 DSH 的插件化架构和 Prime Agent 的 /refine 自改进机制，提升 Qwen3.6-35B-A3B 在 Terminal-Bench 上的表现。

## 目录结构

```
agent_harness/
├── dsh_config/
│   ├── cordis.yml          # DSH 配置 (连接本地 Qwen3.6)
│   ├── run_dsh.sh          # DSH 启动脚本
│   ├── dsh_to_sft.py       # DSH trajectory → SFT 转换器
│   └── README.md           # 本文件
├── scripts/                # 原 tb_loop 脚本
├── learning/               # 原 refine_harness.sh
├── datasets/               # Terminal-Bench 任务
└── ...
```

## 快速开始

### 1. 启动本地 Qwen3.6-35B

```bash
# 启动 llama-server (或 CGC edge_server.py)
llama-server -m models/gguf/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf \
  --host 0.0.0.0 --port 1234 -ngl 4 -c 4096
```

### 2. 构建 DSH

```bash
cd ../dsh
pnpm install
pnpm run build
```

### 3. 运行 DSH Minimal mode

```bash
DSH_MODE=minimal bash dsh_config/run_dsh.sh
```

### 4. 转换轨迹为 SFT 数据

```bash
python dsh_config/dsh_to_sft.py \
  --input results/dsh_<timestamp> \
  --output ../sft_data
```

## DSH 模式

| 模式 | 说明 | 对模型要求 |
|---|---|---|
| `minimal` | bash + str_replace_editor | 低 (3B active MoE 可用) |
| `standard` | 完整工具集 + planning | 高 |
| `multi-model` | Qwen3.6 + Gemma4 协作 | 最高 |

## 配置

编辑 `cordis.yml` 修改模型连接：

```yaml
remotes:
  local-qwen36:
    baseUrl: "http://127.0.0.1:1234/v1"  # 改为你的 llama-server 地址
    apiKey: "sk-local"
```

## 与 agent_harness 的数据流

```
DSH 运行 Terminal-Bench 任务
  → 生成 session log (JSONL)
  → dsh_to_sft.py 提取成功轨迹
  → 输出 train.jsonl / valid.jsonl
  → 可用于 LoRA 微调 Qwen3.6
```
