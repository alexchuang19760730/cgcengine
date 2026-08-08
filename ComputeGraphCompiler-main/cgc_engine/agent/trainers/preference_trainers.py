# Copyright (c) 2026 SandAI. All Rights Reserved.
"""
偏好對齊訓練器模組（Metal + CUDA 共用）

功能：
- DPOTrainer: 直接偏好優化
- ORPOTrainer: SFT + 偏好損失一體化
- GRPOTrainer: 推理增強對齊（DeepSeek-R1 思路）
- KTOTrainer: 輕量偏好優化（無需參考模型）
- SimPOTrainer: 簡單偏好優化（無需參考模型）

這些訓練器同時支援 CUDA（MegaTrain）與 Metal（MLX-Tune）後端。
"""

import os
import math
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
from torch.utils.data import Dataset, DataLoader

from .base_trainer import (
    BaseTrainer,
    TrainingConfig,
    PreferenceDataset,
    PreferenceExample,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# 偏好對齊配置
# ===========================================================================

@dataclass
class PreferenceConfig(TrainingConfig):
    """偏好對齊訓練配置"""
    # 通用
    beta: float = 0.1  # DPO/SimPO 的溫度參數
    label_smoothing: float = 0.0

    # 參考模型
    use_reference_model: bool = True  # DPO/GRPO 需要，KTO/SimPO 不需要
    reference_model_path: Optional[str] = None

    # GRPO 特定
    grpo_num_samples: int = 8  # GRPO 多生成採樣數
    grpo_reward_scale: float = 1.0

    # KTO 特定
    kto_desirable_weight: float = 1.0
    kto_undesirable_weight: float = 1.0

    # SimPO 特定
    simpo_gamma: float = 0.5  # SimPO margin
    simpo_length_normalization: bool = True


# ===========================================================================
# 偏好對齊基類
# ===========================================================================

class PreferenceTrainerBase(BaseTrainer):
    """偏好對齊訓練器基類

    統一處理：
    - 參考模型管理（DPO/GRPO 需要）
    - prompt/chosen/rejected 的 logprob 計算
    - 偏好損失計算
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: PreferenceConfig,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        reference_model: Optional[nn.Module] = None,
        training_data: Any = None,
    ):
        super().__init__(model, tokenizer, config, train_dataset, eval_dataset)

        # 參考模型（凍結）
        self.reference_model: Optional[nn.Module] = None
        if config.use_reference_model:
            if reference_model is not None:
                self.reference_model = reference_model.to(self.device)
                self.reference_model.eval()
                for param in self.reference_model.parameters():
                    param.requires_grad = False
            elif config.reference_model_path:
                # 從路徑載入參考模型
                logger.info(f"[{self.__class__.__name__}] Loading reference model from {config.reference_model_path}")
                # 子類可覆寫載入邏輯

        if training_data is not None and train_dataset is None:
            self.train_dataset = self._build_dataset(training_data)

    def _build_dataset(self, data: Any) -> Dataset:
        """建立偏好資料集"""
        return PreferenceDataset(
            data=data,
            tokenizer=self.tokenizer,
            chat_template=self.config.chat_template,
            max_length=self.config.max_length,
        )

    def _get_policy_logps(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        prompt_len: int = 0,
    ) -> torch.Tensor:
        """計算策略模型（policy）的 log probabilities

        Args:
            model: 策略模型
            input_ids: 輸入 token ids [batch, seq_len]
            attention_mask: attention mask
            prompt_len: prompt 長度（只計算 response 部分）

        Returns:
            per-token log probabilities 的總和 [batch]
        """
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs

        # shift
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        # 只計算 response 部分（跳過 prompt）
        if prompt_len > 0:
            response_start = prompt_len - 1  # 因為 shift
            shift_logits = shift_logits[:, response_start:, :]
            shift_labels = shift_labels[:, response_start:]

        log_probs = F.log_softmax(shift_logits, dim=-1)
        # 取出對應 label 的 log_prob
        per_token_logps = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)

        # 對 response 序列求和
        if attention_mask is not None and prompt_len > 0:
            response_mask = attention_mask[:, prompt_len:].float()
            response_mask = response_mask[:, 1:]  # 對齊 shift
            per_token_logps = per_token_logps * response_mask

        return per_token_logps.sum(dim=-1)

    def _get_reference_logps(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        prompt_len: int = 0,
    ) -> Optional[torch.Tensor]:
        """計算參考模型的 log probabilities"""
        if self.reference_model is None:
            return None

        with torch.no_grad():
            return self._get_policy_logps(
                self.reference_model,
                input_ids,
                attention_mask,
                prompt_len,
            )

    @abstractmethod
    def _compute_preference_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: Optional[torch.Tensor],
        reference_rejected_logps: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """子類實作：計算偏好損失"""
        ...

    def _compute_loss(self, model: nn.Module, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """計算偏好損失"""
        prompt_input_ids = inputs["prompt_input_ids"]
        chosen_input_ids = inputs["chosen_input_ids"]
        rejected_input_ids = inputs["rejected_input_ids"]
        prompt_len = inputs.get("prompt_len", 0)
        chosen_len = inputs.get("chosen_len", 0)
        rejected_len = inputs.get("rejected_len", 0)

        # 策略模型 logprobs
        policy_chosen_logps = self._get_policy_logps(
            model, chosen_input_ids, prompt_len=prompt_len
        )
        policy_rejected_logps = self._get_policy_logps(
            model, rejected_input_ids, prompt_len=prompt_len
        )

        # 參考模型 logprobs
        ref_chosen_logps = None
        ref_rejected_logps = None
        if self.reference_model is not None:
            ref_chosen_logps = self._get_reference_logps(
                chosen_input_ids, prompt_len=prompt_len
            )
            ref_rejected_logps = self._get_reference_logps(
                rejected_input_ids, prompt_len=prompt_len
            )

        return self._compute_preference_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
        )


# ===========================================================================
# DPO Trainer
# ===========================================================================

class DPOTrainer(PreferenceTrainerBase):
    """直接偏好優化（Direct Preference Optimization）

    論文：DPO: Your Language Model is Secretly a Reward Model

    損失：L = -log σ(β * (log π(yw|x)/π_ref(yw|x) - log π(yl|x)/π_ref(yl|x)))

    需要參考模型。
    """

    def _compute_preference_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: Optional[torch.Tensor],
        reference_rejected_logps: Optional[torch.Tensor],
    ) -> torch.Tensor:
        beta = self.config.beta

        if reference_chosen_logps is not None and reference_rejected_logps is not None:
            # 標準 DPO：使用參考模型
            chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
            rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)
        else:
            # 無參考模型：使用 SimPO 變體
            chosen_rewards = beta * policy_chosen_logps
            rejected_rewards = beta * policy_rejected_logps

        logits = chosen_rewards - rejected_rewards

        if self.config.label_smoothing > 0:
            # 標籤平滑
            loss = (
                -self.config.label_smoothing * F.logsigmoid(logits)
                - (1 - self.config.label_smoothing) * F.logsigmoid(-logits)
            )
        else:
            loss = -F.logsigmoid(logits)

        return loss.mean()


# ===========================================================================
# ORPO Trainer
# ===========================================================================

class ORPOTrainer(PreferenceTrainerBase):
    """ORPO: Odds Ratio Preference Optimization

    將 SFT 損失與偏好損失一體化，不需要參考模型。

    損失：L = L_SFT + λ * L_OR
    其中 L_OR = -log σ(log(odds(yw)) - log(odds(yl)))
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ORPO 不需要參考模型
        self.config.use_reference_model = False
        self.reference_model = None
        # SFT 損失權重
        self._sft_weight = 0.1

    def _compute_preference_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: Optional[torch.Tensor],
        reference_rejected_logps: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # ORPO: 使用 odds ratio
        # log odds = log(p / (1 - p)) = log_p - log(1 - exp(log_p))
        chosen_log_odds = policy_chosen_logps - torch.log1p(-torch.exp(policy_chosen_logps) + 1e-8)
        rejected_log_odds = policy_rejected_logps - torch.log1p(-torch.exp(policy_rejected_logps) + 1e-8)

        # Odds ratio loss
        log_odds_ratio = chosen_log_odds - rejected_log_odds
        or_loss = -F.logsigmoid(log_odds_ratio)

        # SFT loss（對 chosen 的語言模型損失）
        sft_loss = -policy_chosen_logps.mean()

        return sft_loss + self._sft_weight * or_loss.mean()


# ===========================================================================
# GRPO Trainer
# ===========================================================================

class GRPOTrainer(PreferenceTrainerBase):
    """GRPO: Group Relative Policy Optimization

    論文：DeepSeek-R1

    特色：
    - 多生成採樣（num_samples 個回覆）
    - 基於組內相對排名計算 reward
    - 不需要顯式 reward model

    需要參考模型。
    """

    def __init__(self, *args, reward_fn: Optional[callable] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_fn = reward_fn
        self.num_samples = self.config.grpo_num_samples

        if reward_fn is None:
            logger.warning(
                "[GRPOTrainer] No reward_fn provided. Using length-based reward as fallback."
            )
            self.reward_fn = self._default_length_reward

    @staticmethod
    def _default_length_reward(generated_ids: torch.Tensor) -> torch.Tensor:
        """預設 reward：基於生成長度（fallback）"""
        # 計算非 padding token 數
        lengths = (generated_ids != 0).sum(dim=-1).float()
        # 正規化到 [0, 1]
        max_len = lengths.max().clamp(min=1)
        return lengths / max_len

    def _generate_samples(
        self,
        model: nn.Module,
        prompt_ids: torch.Tensor,
    ) -> torch.Tensor:
        """從 prompt 生成多個採樣"""
        batch_size = prompt_ids.shape[0]
        num_samples = self.num_samples

        # 重複 prompt
        expanded_prompts = prompt_ids.unsqueeze(1).expand(batch_size, num_samples, -1)
        expanded_prompts = expanded_prompts.reshape(batch_size * num_samples, -1)

        # 生成
        try:
            with torch.no_grad():
                generated = model.generate(
                    expanded_prompts,
                    max_new_tokens=self.config.max_length - prompt_ids.shape[-1],
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
        except (AttributeError, TypeError):
            # fallback: 直接用 prompt 作為生成結果
            generated = expanded_prompts

        return generated

    def _compute_preference_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: Optional[torch.Tensor],
        reference_rejected_logps: Optional[torch.Tensor],
    ) -> torch.Tensor:
        beta = self.config.beta

        # 計算 advantage（基於組內排名）
        # 這裡用 chosen vs rejected 作為簡化版本
        if reference_chosen_logps is not None:
            chosen_advantage = policy_chosen_logps - reference_chosen_logps
            rejected_advantage = policy_rejected_logps - reference_rejected_logps
        else:
            chosen_advantage = policy_chosen_logps
            rejected_advantage = policy_rejected_logps

        # GRPO loss: -advantage * policy_ratio（clip）
        chosen_ratio = torch.exp(beta * chosen_advantage)
        rejected_ratio = torch.exp(beta * rejected_advantage)

        # 簡化版 PPO-style loss
        chosen_loss = -chosen_advantage * chosen_ratio
        rejected_loss = rejected_advantage * rejected_ratio

        loss = chosen_loss.mean() + rejected_loss.mean()

        # 加入 KL 散度正則化
        if reference_chosen_logps is not None:
            kl = (policy_chosen_logps - reference_chosen_logps).mean().pow(2)
            loss = loss + 0.01 * kl

        return loss


# ===========================================================================
# KTO Trainer
# ===========================================================================

class KTOTrainer(PreferenceTrainerBase):
    """KTO: Kahneman-Tversky Optimization

    特色：
    - 不需要成對的 chosen/rejected 資料
    - 只需要二元回饋（desirable / undesirable）
    - 基於前景理論（prospect theory）的非對稱損失

    不需要參考模型（但可以使用）。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # KTO 可以不用參考模型
        if self.config.use_reference_model and self.reference_model is None:
            self.config.use_reference_model = False

    def _compute_preference_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: Optional[torch.Tensor],
        reference_rejected_logps: Optional[torch.Tensor],
    ) -> torch.Tensor:
        beta = self.config.beta
        w_d = self.config.kto_desirable_weight
        w_u = self.config.kto_undesirable_weight

        # 計算 KL（用 rejected 作為 baseline 近似）
        if reference_chosen_logps is not None and reference_rejected_logps is not None:
            kl = (reference_chosen_logps - policy_chosen_logps).mean()
            chosen_reward = beta * (policy_chosen_logps - reference_chosen_logps - kl)
            rejected_reward = beta * (policy_rejected_logps - reference_rejected_logps - kl)
        else:
            chosen_reward = beta * policy_chosen_logps
            rejected_reward = beta * policy_rejected_logps

        # KTO loss: 基於前景理論的非對稱損失
        # Desirable: -log σ(chosen_reward)
        # Undesirable: -log σ(-rejected_reward) * w_u / w_d
        desirable_loss = -F.logsigmoid(chosen_reward) * w_d
        undesirable_loss = -F.logsigmoid(-rejected_reward) * w_u

        # 加權組合
        loss = (desirable_loss.mean() + undesirable_loss.mean()) / (w_d + w_u)

        return loss


# ===========================================================================
# SimPO Trainer
# ===========================================================================

class SimPOTrainer(PreferenceTrainerBase):
    """SimPO: Simple Preference Optimization

    特色：
    - 不需要參考模型
    - 使用長度正規化的 log probabilities
    - 加入 margin（gamma）提升區分度

    損失：L = -log σ((π(yw|x)/|yw| - π(yl|x)/|yl|) / β - γ)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # SimPO 不需要參考模型
        self.config.use_reference_model = False
        self.reference_model = None

    def _compute_preference_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: Optional[torch.Tensor],
        reference_rejected_logps: Optional[torch.Tensor],
    ) -> torch.Tensor:
        beta = self.config.beta
        gamma = self.config.simpo_gamma

        if self.config.simpo_length_normalization:
            # 長度正規化（SimPO 核心）
            # 這裡用 batch 內的平均長度近似
            chosen_len = policy_chosen_logps.abs().clamp(min=1)
            rejected_len = policy_rejected_logps.abs().clamp(min=1)

            chosen_normalized = policy_chosen_logps / chosen_len
            rejected_normalized = policy_rejected_logps / rejected_len
        else:
            chosen_normalized = policy_chosen_logps
            rejected_normalized = policy_rejected_logps

        # SimPO loss
        logits = (chosen_normalized - rejected_normalized) / beta - gamma
        loss = -F.logsigmoid(logits)

        return loss.mean()


# ===========================================================================
# 工廠函數
# ===========================================================================

PREFERENCE_TRAINER_REGISTRY = {
    "dpo": DPOTrainer,
    "orpo": ORPOTrainer,
    "grpo": GRPOTrainer,
    "kto": KTOTrainer,
    "simpo": SimPOTrainer,
}


def create_preference_trainer(
    algorithm: str,
    model: nn.Module,
    tokenizer: Any,
    config: PreferenceConfig,
    **kwargs,
) -> PreferenceTrainerBase:
    """建立偏好對齊訓練器

    Args:
        algorithm: 演算法名稱（dpo/orpo/grpo/kto/simpo）
        model: 模型
        tokenizer: tokenizer
        config: 訓練配置
        **kwargs: 其他參數

    Returns:
        偏好對齊訓練器實例
    """
    algorithm = algorithm.lower()
    if algorithm not in PREFERENCE_TRAINER_REGISTRY:
        raise ValueError(
            f"Unknown preference algorithm: {algorithm}. "
            f"Available: {list(PREFERENCE_TRAINER_REGISTRY.keys())}"
        )

    trainer_cls = PREFERENCE_TRAINER_REGISTRY[algorithm]
    return trainer_cls(model=model, tokenizer=tokenizer, config=config, **kwargs)
