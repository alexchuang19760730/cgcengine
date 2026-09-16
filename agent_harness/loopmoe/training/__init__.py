"""
Loop MoE training module: LoRA, weight grafting, and training loop.

Integrates with agent_harness:
- lora.py: LoRA adapter for continual pre-training
- weight_graft.py: Qwen3.6-35B weight grafting into Loop MoE
- train_loopmoe.py: Training loop with ACT + LTI + MoE router losses
"""

from .lora import LoopMoELoRAConfig, apply_lora, merge_lora
from .weight_graft import WeightGraftConfig, graft_qwen36_weights
from .train_loopmoe import LoopMoETrainer, TrainingConfig

__all__ = [
    "LoopMoELoRAConfig",
    "apply_lora",
    "merge_lora",
    "WeightGraftConfig",
    "graft_qwen36_weights",
    "LoopMoETrainer",
    "TrainingConfig",
]
