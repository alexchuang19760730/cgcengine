"""
Loop MoE training loop with ACT + LTI + MoE router losses.

Supports:
- Continual pre-training with LoRA (grafted Qwen3.6 base)
- Full pre-training from scratch (small models)
- ACT (Adaptive Computation Time) halting loss
- Extended LTI spectral radius constraint (recurrent + Delta state)
- MoE router load balancing + z-loss + aux loss
- Gradient checkpointing for recurrent unrolling
- Dual state (KV + DeltaNet) maintenance through BPTT
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class TrainingConfig:
    """Training configuration for Loop MoE."""

    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    grad_clip_norm: float = 1.0

    # Schedule
    warmup_steps: int = 100
    max_steps: int = 10000
    lr_scheduler: str = "cosine"  # "cosine" | "linear" | "constant"
    min_lr_ratio: float = 0.1

    # Batch
    batch_size: int = 1
    grad_accum_steps: int = 8
    seq_length: int = 2048

    # Loss weights
    lm_loss_weight: float = 1.0
    act_loss_weight: float = 0.01  # ACT ponder cost
    lti_loss_weight: float = 0.1   # LTI spectral radius
    moe_router_loss_weight: float = 0.01
    moe_z_loss_weight: float = 0.001
    moe_aux_loss_weight: float = 0.001

    # Recurrent
    max_recurrent_steps: int = 8
    use_gradient_checkpointing: bool = True

    # Logging / saving
    log_every: int = 10
    eval_every: int = 500
    save_every: int = 1000
    output_dir: str = "./loopmoe_output"

    # Device
    device: str = "auto"  # "auto" | "cuda" | "cpu" | "mps"

    # Dtype
    dtype: str = "bfloat16"  # "bfloat16" | "float16" | "float32"


class TextDataset(Dataset):
    """Simple tokenized text dataset for pre-training."""

    def __init__(self, tokens: torch.Tensor, seq_length: int):
        self.tokens = tokens
        self.seq_length = seq_length
        self.n_samples = max(1, (len(tokens) - 1) // seq_length)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = idx * self.seq_length
        end = start + self.seq_length + 1
        chunk = self.tokens[start:end]
        return {
            "input_ids": chunk[:-1],
            "labels": chunk[1:],
        }


class LoopMoETrainer:
    """
    Trainer for Loop MoE with all recurrent-specific losses.

    Usage:
        config = TrainingConfig(...)
        trainer = LoopMoETrainer(model, config)
        trainer.train(train_dataloader, eval_dataloader)
    """

    def __init__(self, model: nn.Module, config: TrainingConfig):
        self.model = model
        self.config = config
        self.step = 0
        self.global_step = 0

        # Device
        if config.device == "auto":
            self.device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        else:
            self.device = config.device

        # Dtype
        self.dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[config.dtype]

        self.model = self.model.to(self.device)
        self._setup_optimizer()
        self._setup_scheduler()

        os.makedirs(config.output_dir, exist_ok=True)
        print(f"[Trainer] Device: {self.device}, dtype: {self.dtype}")
        print(f"[Trainer] Model params: {sum(p.numel() for p in model.parameters()):,}")
        print(f"[Trainer] Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    def _setup_optimizer(self):
        """Setup AdamW optimizer with parameter groups."""
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "norm" in name or "bias" in name or "layernorm" in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
        )

    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        if self.config.lr_scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.max_steps,
                eta_min=self.config.learning_rate * self.config.min_lr_ratio,
            )
        elif self.config.lr_scheduler == "linear":
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=self.config.min_lr_ratio,
                total_iters=self.config.max_steps,
            )
        else:
            self.scheduler = None

    def _get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss: LM + ACT + LTI + MoE router.

        Args:
            outputs: model forward output dict with:
                - logits: (B, T, V)
                - act_loss: scalar
                - lti_loss: scalar (optional)
                - router_logits: (B, T, num_experts) (optional)
                - aux_loss: scalar (optional)
            labels: (B, T) token labels

        Returns:
            (total_loss, loss_dict)
        """
        cfg = self.config
        loss_dict = {}

        # 1. Language modeling loss
        logits = outputs["logits"]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        lm_loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        loss_dict["lm_loss"] = lm_loss.item()
        total_loss = cfg.lm_loss_weight * lm_loss

        # 2. ACT ponder loss
        if "act_loss" in outputs and outputs["act_loss"] is not None:
            act_loss = outputs["act_loss"].mean()
            loss_dict["act_loss"] = act_loss.item()
            total_loss = total_loss + cfg.act_loss_weight * act_loss

        # 3. LTI spectral radius loss
        if "lti_loss" in outputs and outputs["lti_loss"] is not None:
            lti_loss = outputs["lti_loss"].mean()
            loss_dict["lti_loss"] = lti_loss.item()
            total_loss = total_loss + cfg.lti_loss_weight * lti_loss

        # 4. MoE router losses
        if "router_logits" in outputs and outputs["router_logits"] is not None:
            router_logits = outputs["router_logits"]
            # Z-loss: penalize large router logits
            z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
            loss_dict["moe_z_loss"] = z_loss.item()
            total_loss = total_loss + cfg.moe_z_loss_weight * z_loss

        if "aux_loss" in outputs and outputs["aux_loss"] is not None:
            aux_loss = outputs["aux_loss"].mean()
            loss_dict["moe_aux_loss"] = aux_loss.item()
            total_loss = total_loss + cfg.moe_aux_loss_weight * aux_loss

        loss_dict["total_loss"] = total_loss.item()
        return total_loss, loss_dict

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with gradient accumulation."""
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Forward
        with torch.amp.autocast(
            device_type=self.device if self.device != "mps" else "cpu",
            dtype=self.dtype,
            enabled=self.dtype != torch.float32,
        ):
            outputs = self.model(input_ids=input_ids, labels=labels)
            loss, loss_dict = self.compute_loss(outputs, labels)
            loss = loss / self.config.grad_accum_steps

        # Backward
        loss.backward()

        return loss_dict

    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
    ) -> Dict[str, List[float]]:
        """
        Full training loop.

        Returns:
            history dict with loss curves
        """
        cfg = self.config
        history = {"train_loss": [], "lr": []}

        self.model.train()
        self.optimizer.zero_grad()

        data_iter = iter(train_dataloader)
        start_time = time.time()

        for step in range(cfg.max_steps):
            self.step = step

            # Get batch (cycle through data)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_dataloader)
                batch = next(data_iter)

            # Gradient accumulation
            accum_loss = {}
            for accum_step in range(cfg.grad_accum_steps):
                loss_dict = self.train_step(batch)
                for k, v in loss_dict.items():
                    accum_loss[k] = accum_loss.get(k, 0) + v / cfg.grad_accum_steps

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                cfg.grad_clip_norm,
            )

            # Optimizer step
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()
            self.global_step += 1

            # Logging
            if step % cfg.log_every == 0:
                elapsed = time.time() - start_time
                lr = self._get_lr()
                loss_str = " | ".join(f"{k}={v:.4f}" for k, v in accum_loss.items())
                print(f"[Step {step}/{cfg.max_steps}] lr={lr:.2e} | {loss_str} | "
                      f"{elapsed:.1f}s")
                history["train_loss"].append(accum_loss.get("total_loss", 0))
                history["lr"].append(lr)

            # Evaluation
            if eval_dataloader is not None and step > 0 and step % cfg.eval_every == 0:
                eval_loss = self.evaluate(eval_dataloader)
                print(f"[Eval] Step {step}: eval_loss={eval_loss:.4f}")
                self.model.train()

            # Save checkpoint
            if step > 0 and step % cfg.save_every == 0:
                self.save_checkpoint(step)

        # Final save
        self.save_checkpoint(cfg.max_steps, final=True)
        total_time = time.time() - start_time
        print(f"[Trainer] Training complete in {total_time:.1f}s")

        return history

    @torch.no_grad()
    def evaluate(self, eval_dataloader: DataLoader, max_batches: int = 10) -> float:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in eval_dataloader:
            if n_batches >= max_batches:
                break
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(input_ids=input_ids, labels=labels)
            loss, _ = self.compute_loss(outputs, labels)
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(1, n_batches)

    def save_checkpoint(self, step: int, final: bool = False):
        """Save model checkpoint."""
        name = "final" if final else f"step_{step}"
        path = os.path.join(self.config.output_dir, f"loopmoe_{name}.pt")

        # Save only trainable params (LoRA) if applicable
        trainable_state = {
            k: v.cpu() for k, v in self.model.state_dict().items()
            if v.requires_grad or "lora" in k
        }
        torch.save({
            "step": step,
            "model_state_dict": trainable_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.__dict__,
        }, path)
        print(f"[Trainer] Saved checkpoint to {path}")

    @classmethod
    def from_pretrained(
        cls,
        model: nn.Module,
        checkpoint_path: str,
        config: TrainingConfig,
    ) -> "LoopMoETrainer":
        """Load trainer from checkpoint."""
        trainer = cls(model, config)
        ckpt = torch.load(checkpoint_path, map_location=trainer.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        trainer.step = ckpt.get("step", 0)
        print(f"[Trainer] Resumed from step {trainer.step}")
        return trainer
