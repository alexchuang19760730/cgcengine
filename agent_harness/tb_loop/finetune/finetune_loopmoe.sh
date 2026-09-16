#!/usr/bin/env bash
set -euo pipefail
# ============================================================
# Loop MoE LoRA 持续预训练 / 微调脚本
#
# 流程：
#   1. 从 Qwen3.6-35B 权重嫁接 Loop MoE 基座
#   2. 应用 LoRA 适配器
#   3. 用 SFT 数据（agent trajectory + 通用语料）训练
#   4. 保存 LoRA checkpoint
#   5. 合并 LoRA 到基座（可选）
#
# 用法:
#   bash finetune_loopmoe.sh [--stage graft|lora|train|merge|all]
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_HARNESS_DIR="$(dirname "$SCRIPT_DIR")"

# shellcheck source=../config.env
source "$AGENT_HARNESS_DIR/config.env"

STAGE="${1:-all}"

# ---------- Loop MoE 配置 ----------
LOOPMOE_CONFIG="${LOOPMOE_CONFIG:-$AGENT_HARNESS_DIR/loopmoe/configs/loopmoe_35b.json}"
LOOPMOE_OUTPUT_DIR="${LOOPMOE_OUTPUT_DIR:-$AGENT_HARNESS_DIR/loopmoe_output}"
LOOPMOE_SOURCE_MODEL="${LOOPMOE_SOURCE_MODEL:-/path/to/Qwen3.6-35B-A3B}"
LOOPMOE_TRAIN_DATA="${LOOPMOE_TRAIN_DATA:-$AGENT_HARNESS_DIR/sft_data_merged/train.jsonl}"
LOOPMOE_EVAL_DATA="${LOOPMOE_EVAL_DATA:-$AGENT_HARNESS_DIR/sft_data_merged/valid.jsonl}"

# LoRA
LOOPMOE_LORA_RANK="${LOOPMOE_LORA_RANK:-32}"
LOOPMOE_LORA_ALPHA="${LOOPMOE_LORA_ALPHA:-64}"
LOOPMOE_LORA_DROPOUT="${LOOPMOE_LORA_DROPOUT:-0.05}"

# Training
LOOPMOE_BATCH_SIZE="${LOOPMOE_BATCH_SIZE:-1}"
LOOPMOE_GRAD_ACCUM="${LOOPMOE_GRAD_ACCUM:-8}"
LOOPMOE_MAX_STEPS="${LOOPMOE_MAX_STEPS:-5000}"
LOOPMOE_LEARNING_RATE="${LOOPMOE_LEARNING_RATE:-1e-4}"
LOOPMOE_SEQ_LENGTH="${LOOPMOE_SEQ_LENGTH:-2048}"
LOOPMOE_WARMUP="${LOOPMOE_WARMUP:-100}"

# Python
PYTHON="${PYTHON:-python3}"

mkdir -p "$LOOPMOE_OUTPUT_DIR"

echo "============================================"
echo "Loop MoE Fine-tuning"
echo "Stage: $STAGE"
echo "Output: $LOOPMOE_OUTPUT_DIR"
echo "============================================"

# ---------- Stage 1: Weight Grafting ----------
do_graft() {
    echo ""
    echo "[Stage 1/4] Weight grafting: Qwen3.6 → Loop MoE"
    echo "  Source: $LOOPMOE_SOURCE_MODEL"
    echo "  Config: $LOOPMOE_CONFIG"

    $PYTHON - "$LOOPMOE_CONFIG" "$LOOPMOE_SOURCE_MODEL" "$LOOPMOE_OUTPUT_DIR" <<'PY'
import sys, os
sys.path.insert(0, os.environ.get("AGENT_HARNESS_DIR", "."))

from loopmoe.config import LoopMoEConfig
from loopmoe.models.loop_moe_model import LoopMoEModel
from loopmoe.training.weight_graft import WeightGraftConfig, graft_qwen36_weights, verify_graft

config_path, source_path, output_dir = sys.argv[1], sys.argv[2], sys.argv[3]

# Load config
if os.path.exists(config_path):
    model_config = LoopMoEConfig.from_json(config_path)
else:
    print(f"  Config not found, using defaults (small test config)")
    model_config = LoopMoEConfig(
        hidden_size=512, num_experts=8, num_experts_per_tok=2,
        max_recurrent_steps=4, linear_num_key_heads=4, linear_num_value_heads=8,
    )

# Create model
model = LoopMoEModel(model_config)
print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

# Graft
graft_config = WeightGraftConfig(
    source_model_path=source_path,
    source_model_type="hf" if os.path.isdir(source_path) else "state_dict",
)
model, stats = graft_qwen36_weights(model, graft_config)
print(f"  Graft stats: {stats}")

# Save grafted base
os.makedirs(output_dir, exist_ok=True)
base_path = os.path.join(output_dir, "loopmoe_grafted_base.pt")
import torch
torch.save({"model_state_dict": model.state_dict(), "config": model_config.to_dict()}, base_path)
print(f"  Saved grafted base: {base_path}")
PY
}

# ---------- Stage 2: Apply LoRA ----------
do_lora() {
    echo ""
    echo "[Stage 2/4] Applying LoRA (rank=$LOOPMOE_LORA_RANK, alpha=$LOOPMOE_LORA_ALPHA)"

    $PYTHON - "$LOOPMOE_OUTPUT_DIR" "$LOOPMOE_LORA_RANK" "$LOOPMOE_LORA_ALPHA" <<'PY'
import sys, os
sys.path.insert(0, os.environ.get("AGENT_HARNESS_DIR", "."))

import torch
from loopmoe.config import LoopMoEConfig
from loopmoe.models.loop_moe_model import LoopMoEModel
from loopmoe.training.lora import LoopMoELoRAConfig, apply_lora, save_lora_state

output_dir, rank, alpha = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

# Load grafted base
base_path = os.path.join(output_dir, "loopmoe_grafted_base.pt")
ckpt = torch.load(base_path, map_location="cpu", weights_only=False)
model_config = LoopMoEConfig(**ckpt["config"])
model = LoopMoEModel(model_config)
model.load_state_dict(ckpt["model_state_dict"])

# Apply LoRA
lora_config = LoopMoELoRAConfig(lora_rank=rank, lora_alpha=alpha)
model = apply_lora(model, lora_config)

# Save initial LoRA state
lora_path = os.path.join(output_dir, "loopmoe_lora_init.pt")
save_lora_state(model, lora_path)
print(f"  LoRA applied, initial state saved: {lora_path}")
PY
}

# ---------- Stage 3: Training ----------
do_train() {
    echo ""
    echo "[Stage 3/4] Training (steps=$LOOPMOE_MAX_STEPS, lr=$LOOPMOE_LEARNING_RATE)"
    echo "  Train data: $LOOPMOE_TRAIN_DATA"

    $PYTHON - "$LOOPMOE_OUTPUT_DIR" "$LOOPMOE_TRAIN_DATA" "$LOOPMOE_MAX_STEPS" "$LOOPMOE_LEARNING_RATE" "$LOOPMOE_BATCH_SIZE" "$LOOPMOE_GRAD_ACCUM" "$LOOPMOE_SEQ_LENGTH" <<'PY'
import sys, os, json
sys.path.insert(0, os.environ.get("AGENT_HARNESS_DIR", "."))

import torch
from torch.utils.data import DataLoader
from loopmoe.config import LoopMoEConfig
from loopmoe.models.loop_moe_model import LoopMoEModel
from loopmoe.training.lora import LoopMoELoRAConfig, apply_lora, load_lora_state
from loopmoe.training.train_loopmoe import LoopMoETrainer, TrainingConfig, TextDataset

output_dir, train_data, max_steps, lr, batch_size, grad_accum, seq_len = sys.argv[1:]
max_steps, lr, batch_size, grad_accum, seq_len = int(max_steps), float(lr), int(batch_size), int(grad_accum), int(seq_len)

# Load model
base_path = os.path.join(output_dir, "loopmoe_grafted_base.pt")
ckpt = torch.load(base_path, map_location="cpu", weights_only=False)
model_config = LoopMoEConfig(**ckpt["config"])
model = LoopMoEModel(model_config)
model.load_state_dict(ckpt["model_state_dict"])

# Apply LoRA
lora_config = LoopMoELoRAConfig()
model = apply_lora(model, lora_config)
lora_init = os.path.join(output_dir, "loopmoe_lora_init.pt")
if os.path.exists(lora_init):
    model = load_lora_state(model, lora_init)

# Load training data (tokenized JSONL or raw text)
print(f"  Loading training data: {train_data}")
if os.path.exists(train_data):
    tokens = []
    with open(train_data, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                text = item.get("text", item.get("content", ""))
                # Simple tokenization: byte-level (replace with real tokenizer)
                tokens.extend([ord(c) % model_config.vocab_size for c in text[:seq_len]])
            except json.JSONDecodeError:
                continue
    if tokens:
        dataset = TextDataset(torch.tensor(tokens, dtype=torch.long), seq_len)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print(f"  Loaded {len(tokens)} tokens, {len(dataset)} samples")
    else:
        print("  WARNING: No training data loaded, using random data for smoke test")
        dataset = TextDataset(torch.randint(0, model_config.vocab_size, (seq_len * 10,)), seq_len)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
else:
    print(f"  WARNING: {train_data} not found, using random data for smoke test")
    dataset = TextDataset(torch.randint(0, model_config.vocab_size, (seq_len * 10,)), seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training config
train_config = TrainingConfig(
    learning_rate=lr,
    max_steps=max_steps,
    batch_size=batch_size,
    grad_accum_steps=grad_accum,
    seq_length=seq_len,
    output_dir=output_dir,
    warmup_steps=min(100, max_steps // 10),
)

# Train
trainer = LoopMoETrainer(model, train_config)
history = trainer.train(dataloader)
print(f"  Training complete. Final loss: {history['train_loss'][-1]:.4f}")
PY
}

# ---------- Stage 4: Merge LoRA ----------
do_merge() {
    echo ""
    echo "[Stage 4/4] Merging LoRA into base weights"

    $PYTHON - "$LOOPMOE_OUTPUT_DIR" <<'PY'
import sys, os
sys.path.insert(0, os.environ.get("AGENT_HARNESS_DIR", "."))

import torch
from loopmoe.config import LoopMoEConfig
from loopmoe.models.loop_moe_model import LoopMoEModel
from loopmoe.training.lora import LoopMoELoRAConfig, apply_lora, load_lora_state, merge_lora

output_dir = sys.argv[1]

# Load base + LoRA
base_path = os.path.join(output_dir, "loopmoe_grafted_base.pt")
ckpt = torch.load(base_path, map_location="cpu", weights_only=False)
model_config = LoopMoEConfig(**ckpt["config"])
model = LoopMoEModel(model_config)
model.load_state_dict(ckpt["model_state_dict"])

# Apply and load trained LoRA
lora_config = LoopMoELoRAConfig()
model = apply_lora(model, lora_config)

# Find latest checkpoint
import glob
checkpoints = sorted(glob.glob(os.path.join(output_dir, "loopmoe_step_*.pt")))
if checkpoints:
    latest = checkpoints[-1]
    print(f"  Loading latest checkpoint: {latest}")
    lora_ckpt = torch.load(latest, map_location="cpu", weights_only=False)
    model_dict = dict(model.named_parameters())
    for name, tensor in lora_ckpt.get("model_state_dict", {}).items():
        if name in model_dict and ("lora_A" in name or "lora_B" in name):
            model_dict[name].data.copy_(tensor)
else:
    final_path = os.path.join(output_dir, "loopmoe_final.pt")
    if os.path.exists(final_path):
        print(f"  Loading final checkpoint: {final_path}")
        lora_ckpt = torch.load(final_path, map_location="cpu", weights_only=False)
        model_dict = dict(model.named_parameters())
        for name, tensor in lora_ckpt.get("model_state_dict", {}).items():
            if name in model_dict and ("lora_A" in name or "lora_B" in name):
                model_dict[name].data.copy_(tensor)

# Merge
model = merge_lora(model)

# Save merged model
merged_path = os.path.join(output_dir, "loopmoe_merged.pt")
torch.save({"model_state_dict": model.state_dict(), "config": model_config.to_dict()}, merged_path)
print(f"  Merged model saved: {merged_path}")
print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")
PY
}

# ---------- Execute ----------
case "$STAGE" in
    graft) do_graft ;;
    lora) do_lora ;;
    train) do_train ;;
    merge) do_merge ;;
    all)
        do_graft
        do_lora
        do_train
        do_merge
        ;;
    *)
        echo "Unknown stage: $STAGE"
        echo "Usage: bash finetune_loopmoe.sh [graft|lora|train|merge|all]"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "Loop MoE fine-tuning complete (stage: $STAGE)"
echo "Output directory: $LOOPMOE_OUTPUT_DIR"
echo "============================================"
