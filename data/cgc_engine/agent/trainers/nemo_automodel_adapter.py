"""NeMo Automodel Adapter

把 NVIDIA NeMo Automodel 包成與 MegatrainMoETrainer 同接口的後端選項。

設計要點：
- 條件式 import NeMo（缺失時優雅降級到 MegatrainMoETrainer）
- 解綁 HF Transformers v5（支援 v4）
- 解綁 TransformerEngine / DeepEP / GroupedGEMM（可選加速層）
- 與 MegatrainMoETrainer 同接口（符合方案 A：薄 Adapter）
- 支援雙機 TP4EP4+DP2（透過 distributed_topology）

使用範例：
    from cgc_engine.agent.trainers import NemoAutomodelMoETrainer, MegatrainConfig
    trainer = NemoAutomodelMoETrainer(
        model=model, tokenizer=tokenizer, config=config,
        use_nemo="auto",  # auto / force / skip
        use_te=False,     # 關閉 TransformerEngine
        use_deepep=False, # 關閉 NeMo DeepEP（改用 CGC 自製）
    )
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .base_trainer import BaseTrainer, TrainingConfig
from .megatrain_trainers import MegatrainConfig, MegatrainMoETrainer

logger = logging.getLogger(__name__)


@dataclass
class NemoAutomodelConfig(MegatrainConfig):
    """NeMo Automodel 訓練配置（繼承 MegatrainConfig）"""
    # NeMo 後端選擇
    use_nemo: str = "auto"  # auto / force / skip
    # 可選加速層（預設關閉，解綁重依賴）
    use_transformer_engine: bool = False
    use_deepep: bool = False  # NeMo 內建 DeepEP
    use_grouped_gemm: bool = False
    # HF 版本相容
    transformers_compat: str = "v4"  # v4 / v5
    # MoE 配置
    moe_expert_parallel_size: int = 1
    moe_tensor_parallel_size: int = 1


class NemoAutomodelMoETrainer(MegatrainMoETrainer):
    """NeMo Automodel MoE 加速後端

    多卡 H100 場景啟用 NeMo EP，享受 3.4-3.7x 加速。
    單卡或 NeMo 未安裝時自動 fallback 到 MegatrainMoETrainer。

    策略：
    1. use_nemo="auto" + NeMo 已安裝 + 多卡 → 用 NeMo
    2. use_nemo="force" → 強制用 NeMo（缺失則報錯）
    3. use_nemo="skip" 或 NeMo 未安裝 → fallback 到 MegatrainMoETrainer
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: NemoAutomodelConfig,
        train_dataset: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
        training_data: Any = None,
    ):
        # 先決定是否啟用 NeMo
        self._nemo_available = self._detect_nemo_available()
        self._use_nemo = self._decide_nemo_usage(config.use_nemo)
        self._nemo_model_wrapped = False
        self._nemo_config = config

        if self._use_nemo:
            logger.info(
                f"[NemoAutomodelMoETrainer] NeMo backend enabled "
                f"(TE={config.use_transformer_engine}, DeepEP={config.use_deepep}, "
                f"GroupedGEMM={config.use_grouped_gemm})"
            )
            # 用 NeMo 包裝模型（不依賴 HF v5，支援 v4）
            model = self._wrap_with_nemo(model, config)
        else:
            fallback_reason = (
                "force=False" if config.use_nemo == "skip"
                else "NeMo not installed" if not self._nemo_available
                else "single GPU"
            )
            logger.info(
                f"[NemoAutomodelMoETrainer] Fallback to MegatrainMoETrainer "
                f"(reason: {fallback_reason})"
            )

        super().__init__(model, tokenizer, config, train_dataset, eval_dataset, training_data)

    @staticmethod
    def _detect_nemo_available() -> bool:
        """偵測 NeMo Automodel 是否可用（不強制安裝）"""
        try:
            import nemo_automodel  # noqa: F401
            return True
        except ImportError:
            return False

    def _decide_nemo_usage(self, use_nemo: str) -> bool:
        """決定是否使用 NeMo"""
        if use_nemo == "force":
            if not self._nemo_available:
                raise RuntimeError(
                    "use_nemo='force' but nemo_automodel not installed. "
                    "Install with: pip install nemo-automodel"
                )
            return True
        if use_nemo == "skip":
            return False
        # auto
        if not self._nemo_available:
            return False
        # 多卡才啟用 NeMo（單卡用 Megatrain 流式即可）
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        return num_gpus >= 2

    def _wrap_with_nemo(self, model: nn.Module, config: NemoAutomodelConfig) -> nn.Module:
        """用 NeMo Automodel 包裝模型

        解綁策略：
        - HF v5 依賴：NeMo 內建 v4 shims，自動降級
        - TransformerEngine：可選，config.use_transformer_engine 控制
        - DeepEP：可選，config.use_deepep 控制（False 時用 PyTorch 原生 EP）
        - GroupedGEMM：可選，config.use_grouped_gemm 控制
        """
        try:
            from nemo_automodel import NeMoAutoModelForCausalLM
            from nemo_automodel.models.config import AutoModelConfig
        except ImportError as e:
            raise RuntimeError(
                f"Failed to import nemo_automodel: {e}. "
                f"Install with: pip install nemo-automodel"
            )

        # 取得模型名稱（從 config 或 model 屬性）
        model_name = getattr(config, "model_name", None) or getattr(model, "name_or_path", None)
        if not model_name:
            # 嘗試從 config 取得
            model_name = getattr(config, "base_model_name", None) or "Qwen/Qwen3-30B-A3B"

        # NeMo 配置（關閉重依賴加速層）
        nemo_kwargs = {
            "torch_dtype": torch.bfloat16,
            "transformers_compat": config.transformers_compat,
        }

        # 可選加速層（預設關閉）
        if hasattr(AutoModelConfig, "use_transformer_engine"):
            nemo_kwargs["use_transformer_engine"] = config.use_transformer_engine
        if hasattr(AutoModelConfig, "use_deepep"):
            nemo_kwargs["use_deepep"] = config.use_deepep
        if hasattr(AutoModelConfig, "use_grouped_gemm"):
            nemo_kwargs["use_grouped_gemm"] = config.use_grouped_gemm

        # MoE 並行配置
        if config.moe_expert_parallel_size > 1 or config.moe_tensor_parallel_size > 1:
            nemo_kwargs["expert_parallel_size"] = config.moe_expert_parallel_size
            nemo_kwargs["tensor_parallel_size"] = config.moe_tensor_parallel_size

        try:
            # 優先嘗試從已載入模型包裝（避免重新下載）
            if hasattr(NeMoAutoModelForCausalLM, "from_model"):
                nemo_model = NeMoAutoModelForCausalLM.from_model(model, **nemo_kwargs)
            else:
                # 從 pretrained 載入
                nemo_model = NeMoAutoModelForCausalLM.from_pretrained(model_name, **nemo_kwargs)
            self._nemo_model_wrapped = True
            logger.info(
                f"[NemoAutomodelMoETrainer] Model wrapped with NeMo "
                f"(model={model_name}, dtype=bf16, compat={config.transformers_compat})"
            )
            return nemo_model
        except Exception as e:
            logger.warning(
                f"[NemoAutomodelMoETrainer] NeMo wrap failed: {e}, "
                f"fallback to original model"
            )
            self._use_nemo = False
            return model

    def train(self, resume_from: Optional[str] = None) -> Dict[str, Any]:
        """訓練入口

        若 NeMo 啟用，使用 NeMo 的 train()；否則用 MegatrainMoETrainer.train()
        """
        if not self._use_nemo or not self._nemo_model_wrapped:
            return super().train(resume_from)

        # NeMo 訓練路徑
        try:
            # NeMo 的 train 介面可能不同，這裡做適配
            from nemo_automodel import Trainer as NeMoTrainer

            nemo_trainer = NeMoTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                # 把 TrainingConfig 轉成 NeMo 格式
                **self._convert_config_to_nemo(self.config),
            )
            result = nemo_trainer.train(resume_from=resume_from)
            return self._convert_nemo_result(result)
        except Exception as e:
            logger.warning(
                f"[NemoAutomodelMoETrainer] NeMo train failed: {e}, "
                f"fallback to MegatrainMoETrainer.train()"
            )
            # fallback
            self._use_nemo = False
            return super().train(resume_from)

    def _convert_config_to_nemo(self, config: TrainingConfig) -> Dict[str, Any]:
        """把 CGC TrainingConfig 轉成 NeMo Trainer 參數"""
        return {
            "learning_rate": getattr(config, "learning_rate", 2e-5),
            "num_train_epochs": getattr(config, "num_train_epochs", 3),
            "per_device_train_batch_size": getattr(config, "per_device_train_batch_size", 4),
            "gradient_accumulation_steps": getattr(config, "gradient_accumulation_steps", 4),
            "warmup_steps": getattr(config, "warmup_steps", 100),
            "weight_decay": getattr(config, "weight_decay", 0.01),
            "max_grad_norm": getattr(config, "max_grad_norm", 1.0),
            "bf16": getattr(config, "bf16", True),
            "logging_steps": getattr(config, "logging_steps", 10),
            "save_steps": getattr(config, "save_steps", 500),
            "output_dir": getattr(config, "output_dir", "./output"),
        }

    @staticmethod
    def _convert_nemo_result(result: Any) -> Dict[str, Any]:
        """把 NeMo 訓練結果轉成 CGC 格式"""
        if isinstance(result, dict):
            return result
        return {
            "status": "ok",
            "backend": "nemo",
            "train_loss": getattr(result, "train_loss", None),
            "global_step": getattr(result, "global_step", None),
        }

    def export_weights_for_inference(
        self, output_dir: str, inference_backend: str = "sglang"
    ) -> Dict[str, Any]:
        """匯出訓練後權重供推理使用（訓推一致性）

        Args:
            output_dir: 匯出目錄
            inference_backend: sglang / vllm / mlx
        """
        if self._use_nemo and self._nemo_model_wrapped:
            # NeMo 模型匯出
            try:
                if hasattr(self.model, "save_pretrained"):
                    self.model.save_pretrained(output_dir)
                else:
                    # NeMo 特有匯出
                    torch.save(self.model.state_dict(), os.path.join(output_dir, "model.pt"))
                return {
                    "status": "ok",
                    "backend": "nemo",
                    "output_dir": output_dir,
                    "inference_backend": inference_backend,
                }
            except Exception as e:
                return {"status": "fail", "error": str(e)}
        return super().export_weights_for_inference(output_dir, inference_backend) \
            if hasattr(super(), "export_weights_for_inference") \
            else {"status": "skip", "reason": "not implemented"}


# 註冊到 registry（與其他 trainer 一致）
NEMO_AUTOMODEL_TRAINER_REGISTRY = {
    "nemo_moe_sft": NemoAutomodelMoETrainer,
    "nemo_moe_cpt": NemoAutomodelMoETrainer,  # CPT 模式
}


def create_nemo_automodel_trainer(
    trainer_type: str,
    model: nn.Module,
    tokenizer: Any,
    config: Optional[NemoAutomodelConfig] = None,
    **kwargs,
) -> NemoAutomodelMoETrainer:
    """建立 NeMo Automodel 訓練器的工廠函數"""
    config = config or NemoAutomodelConfig()
    trainer_cls = NEMO_AUTOMODEL_TRAINER_REGISTRY.get(trainer_type, NemoAutomodelMoETrainer)
    return trainer_cls(model=model, tokenizer=tokenizer, config=config, **kwargs)
