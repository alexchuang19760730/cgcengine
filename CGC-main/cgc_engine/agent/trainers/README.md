# CGC Trainers 目錄說明

本目錄提供 MegaTrain（CUDA）與 MLX-Tune（Metal）兩套後端的統一訓練器，以及五種偏好對齊演算法。所有訓練器均接入 CGC SIMD 訓推共享指令集，並支援訓推權重一致性檢查（Gate 3.0 Dimension A/B/C/D）。

## 目錄結構

```
cgc_engine/agent/trainers/
├── __init__.py              # 統一導出所有 Trainer
├── base_trainer.py          # BaseTrainer 基類 + chat template + 資料集 + 訓推一致性檢查
├── megatrain_trainers.py    # MegaTrain: SFT/CPT/MoE + CPU offload/3-stream/stateless/long-context
├── mlxtune_trainers.py      # MLX-Tune: SFT/CPT/全參數/8bit/MoE/多模態/Unsloth 相容
└── preference_trainers.py   # 偏好對齊: DPO/ORPO/GRPO/KTO/SimPO（Metal + CUDA 共用）
```

## 訓練器清單

### MegaTrain（CUDA 後端）

| 訓練器 | 用途 | 核心架構 |
|---|---|---|
| `MegatrainSFTTrainer` | 監督微調（全參數 + LoRA + QLoRA） | FSDP / 混合精度 / Flash Attention |
| `MegatrainCPTTrainer` | 持續預訓練（全文本無 mask 損失） | BF16/FP32 全精度 |
| `MegatrainMoETrainer` | MoE 混合專家訓練 | 流式專家載入 / 負載均衡 loss |

### MLX-Tune（Metal 後端）

| 訓練器 | 用途 | 特色 |
|---|---|---|
| `MLXTuneSFTTrainer` | 監督微調（LoRA/QLoRA/全參數） | Apple Silicon 統一記憶體 |
| `MLXTuneCPTTrainer` | 持續預訓練 | Metal 後端優化 |
| `MLXTuneFullFinetuneTrainer` | 全參數微調 | 適合小模型（< 7B） |
| `MLXTune8bitQuantTrainer` | 8bit 量化微調 | 比 4bit 更精確 |
| `MLXTuneMoETrainer` | MoE 逐專家 LoRA 微調 | Mac Studio 可跑 350B MoE |
| `MLXTuneMultimodalTrainer` | 多模態微調（VLM） | 視覺 tower 凍結 + LLM LoRA |

### 偏好對齊（CUDA + Metal 共用）

| 訓練器 | 演算法 | 是否需要參考模型 |
|---|---|---|
| `DPOTrainer` | 直接偏好優化 | 是 |
| `ORPOTrainer` | SFT + 偏好損失一體化 | 否 |
| `GRPOTrainer` | 推理增強對齊（DeepSeek-R1） | 是 |
| `KTOTrainer` | 輕量偏好優化（前景理論） | 否（可用） |
| `SimPOTrainer` | 簡單偏好優化（長度正規化） | 否 |

### 核心架構組件（MegaTrain）

| 組件 | 功能 |
|---|---|
| `CPUOffloadOptimizer` | CPU 記憶體主存儲優化器（AdamW/Adam8bit） |
| `PipelineStreamScheduler` | 3-stream 雙緩衝預取（參數預取→GPU 計算→梯度回寫） |
| `StatelessLayerTemplate` | 無狀態 Layer 模板（拋棄持久 autograd 圖） |
| `LongContextManager` | 超大上下文管理器（最高 512k） |

## 快速開始

### 1. MegaTrain SFT（CUDA）

```python
from cgc_engine.agent.trainers import MegatrainSFTTrainer, MegatrainConfig

config = MegatrainConfig(
    output_dir="./output/megatrain_sft",
    training_mode="lora",          # full / lora / qlora
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    use_cpu_memory_offload=True,   # 啟用 MegaTrain CPU 主存儲架構
    use_pipeline_streams=True,     # 啟用 3-stream 雙緩衝
)

trainer = MegatrainSFTTrainer(
    model=model,
    tokenizer=tokenizer,
    config=config,
    training_data="train.jsonl",   # JSONL 格式，每行 {"messages": [...]}
)
result = trainer.train()
```

### 2. MLX-Tune SFT（Apple Silicon）

```python
from cgc_engine.agent.trainers import MLXTuneSFTTrainer, MLXTuneConfig

config = MLXTuneConfig(
    output_dir="./output/mlx_sft",
    training_mode="qlora",         # lora / qlora / full / qlora8bit
    qlora_bits=4,
    lora_rank=8,
    edge_cloud_mode=True,          # 啟用端雲一體
)

trainer = MLXTuneSFTTrainer(
    model=model,
    tokenizer=tokenizer,
    config=config,
    training_data="train.jsonl",
)
result = trainer.train()
```

### 3. 偏好對齊（DPO 範例）

```python
from cgc_engine.agent.trainers import create_preference_trainer, PreferenceConfig

config = PreferenceConfig(
    output_dir="./output/dpo",
    beta=0.1,
    use_reference_model=True,
)

trainer = create_preference_trainer(
    algorithm="dpo",               # dpo / orpo / grpo / kto / simpo
    model=model,
    tokenizer=tokenizer,
    config=config,
    reference_model=ref_model,     # DPO 需要參考模型
    training_data="prefs.jsonl",   # JSONL: {"prompt": [...], "chosen": [...], "rejected": [...]}
)
result = trainer.train()
```

### 4. Unsloth API 相容（無痛遷移）

```python
from cgc_engine.agent.trainers import UnslothCompatAdapter

# 與 Unsloth API 完全相容的介面
adapter = UnslothCompatAdapter.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    load_in_4bit=True,
)
adapter.get_peft_model(r=8, lora_alpha=16)
adapter.train(train_data="train.jsonl", epochs=3)
adapter.save_model("./output/unstloth_compat")
```

### 5. MoE 混合專家訓練

```python
from cgc_engine.agent.trainers import MegatrainMoETrainer, MegatrainConfig

config = MegatrainConfig(
    enable_moe_training=True,
    moe_num_experts=8,
    moe_top_k=2,
    use_cpu_memory_offload=True,   # 單卡大記憶體伺服器
)

trainer = MegatrainMoETrainer(
    model=moe_model,
    tokenizer=tokenizer,
    config=config,
    training_data="train.jsonl",
)
result = trainer.train()
```

### 6. 超大上下文訓練（512k）

```python
from cgc_engine.agent.trainers import LongContextManager

ctx_mgr = LongContextManager(
    max_seq_len=512_000,
    chunk_size=8192,
    overlap_size=512,
)
# 流式前向，避免 OOM
output = ctx_mgr.streaming_forward(model, long_input_ids)
```

## 訓推一致性檢查（Gate 3.0）

所有訓練器均內建訓推權重一致性檢查，對接 Gate 3.0 Dimension A：

```python
result = trainer.check_train_inference_consistency(
    inference_engine=vllm_engine,
    sample_input=batch,
    tolerance=1e-5,
)
# result["gate_pass"] == True 即通過 Gate 3.0 Dimension A
```

## 資料格式

### SFT / CPT 資料（JSONL）

```jsonl
{"messages": [{"role": "user", "content": "問題"}, {"role": "assistant", "content": "回答"}]}
```

### 偏好對齊資料（JSONL）

```jsonl
{"prompt": [{"role": "user", "content": "問題"}], "chosen": [{"role": "assistant", "content": "好回答"}], "rejected": [{"role": "assistant", "content": "差回答"}]}
```

## 支援的 Chat Template

`chatml` / `llama3` / `qwen` / `mistral` / `zephyr`

## 與 Gate 3.0 的對齊

| Gate 3.0 Dimension | 訓練器支撐 |
|---|---|
| A（訓推權重一致性） | `BaseTrainer.check_train_inference_consistency()` |
| B（KDA 正交基保留） | `_maybe_cgc_compile()` 接入 CGC KDA |
| C（SIMD 訓推共享） | `_get_cgc_engine()` 載入 MegatrainCGCExec |
| D（LoRA 端雲協同） | `MLXTuneSFTTrainer.edge_cloud_mode` + LoRA adapter |
