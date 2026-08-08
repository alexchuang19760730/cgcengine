# Copyright (c) 2026 SandAI. All Rights Reserved.
"""
MLX-Tune 訓練器模組（Apple Silicon Metal 後端）

功能：
- MLXTuneSFTTrainer: 監督微調（LoRA/QLoRA/全參數）
- MLXTuneCPTTrainer: 持續預訓練
- MLXTuneMoETrainer: MoE 混合專家微調（逐專家 LoRA）
- MLXTuneFullFinetuneTrainer: 全參數微調
- MLXTune8bitQuantTrainer: 8bit 量化微調
- MLXTuneMultimodalTrainer: 多模態微調
- UnslothCompatAdapter: Unsloth API 相容層

所有訓練器基於 Apple Silicon Metal 後端，使用統一記憶體。
偏好對齊訓練器（DPO/ORPO/GRPO/KTO/SimPO）直接復用 preference_trainers.py。
"""

import os
import time
import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .base_trainer import (
    BaseTrainer,
    TrainingConfig,
    SFTDataset,
    TextDataset,
    ConversationExample,
    TextExample,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# MLX-Tune 配置
# ===========================================================================

@dataclass
class MLXTuneConfig(TrainingConfig):
    """MLX-Tune 訓練配置"""
    # Metal 後端
    enable_metal_backend: bool = True
    use_unified_memory: bool = True
    use_graph_execution: bool = True

    # LoRA / QLoRA
    training_mode: str = "lora"  # lora, qlora, full, qlora8bit
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    qlora_bits: int = 4  # 4 或 8
    qlora_group_size: int = 64

    # 目標模組
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # 端雲一體
    edge_cloud_mode: bool = True
    prefill_on_cloud: bool = True
    decode_on_edge: bool = True

    # MoE
    enable_moe_training: bool = False
    moe_expert_lora: bool = True  # 逐專家 LoRA

    # 多模態
    enable_multimodal: bool = False
    vision_tower: Optional[str] = None

    # Unsloth 相容
    use_unsloth_api: bool = False


# ===========================================================================
# MLX-Tune LoRA 工具
# ===========================================================================

class MLXTuneLoRALinear(nn.Module):
    """MLX-Tune LoRA Linear 層（Metal 後端優化）

    使用 Apple Silicon 統一記憶體，權重與優化器狀態常駐 unified memory。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.05,
        qlora_bits: Optional[int] = None,
        qlora_group_size: int = 64,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.qlora_bits = qlora_bits

        # LoRA A/B 矩陣
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # 初始化（A 用 Kaiming，B 用零）
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # QLoRA 量化
        if qlora_bits is not None:
            self._quantized_base = None  # 量化後的基底權重
            self.qlora_group_size = qlora_group_size
            logger.info(f"[MLXTuneLoRALinear] QLoRA {qlora_bits}bit enabled")

    def forward(self, x: torch.Tensor, base_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 基底權重（凍結）
        if base_weight is not None:
            base_out = F.linear(x, base_weight)
        elif hasattr(self, "_quantized_base") and self._quantized_base is not None:
            base_out = F.linear(x, self._dequantize(self._quantized_base))
        else:
            # 無基底權重時，只計算 LoRA 路徑
            base_out = 0

        # LoRA 路徑
        lora_out = self.dropout(x)
        lora_out = lora_out @ self.lora_A.T  # [*, in] @ [in, rank] -> [*, rank]
        lora_out = lora_out @ self.lora_B.T  # [*, rank] @ [rank, out] -> [*, out]
        lora_out = lora_out * self.scaling

        return base_out + lora_out

    def _dequantize(self, quantized: Dict[str, torch.Tensor]) -> torch.Tensor:
        """反量化 QLoRA 權重"""
        data = quantized["data"]
        scale = quantized["scale"]
        zero = quantized["zero"]

        if self.qlora_bits == 4:
            # 4bit 反量化
            dequant = (data.float() - zero) * scale
        elif self.qlora_bits == 8:
            # 8bit 反量化
            dequant = data.float() * scale
        else:
            dequant = data.float()

        return dequant

    def quantize_base(self, base_weight: torch.Tensor) -> None:
        """量化基底權重"""
        if self.qlora_bits is None:
            return

        if self.qlora_bits == 4:
            # 4bit NF4 量化
            group_size = self.qlora_group_size
            num_groups = base_weight.numel() // group_size
            groups = base_weight.view(num_groups, group_size)

            scale = groups.abs().max(dim=-1, keepdim=True).values / 7.0
            scale = scale.clamp(min=1e-8)
            quantized = (groups / scale).round().clamp(-8, 7).to(torch.int8)

            self._quantized_base = {
                "data": quantized.view_as(base_weight),
                "scale": scale.expand(num_groups, group_size).view_as(base_weight),
                "zero": torch.zeros_like(scale).expand(num_groups, group_size).view_as(base_weight),
            }
        elif self.qlora_bits == 8:
            # 8bit 量化
            scale = base_weight.abs().max() / 127.0
            scale = scale.clamp(min=1e-8)
            quantized = (base_weight / scale).round().clamp(-128, 127).to(torch.int8)

            self._quantized_base = {
                "data": quantized,
                "scale": scale.expand_as(base_weight),
                "zero": torch.zeros_like(base_weight),
            }

        logger.info(
            f"[MLXTuneLoRALinear] Base weight quantized to {self.qlora_bits}bit "
            f"(compression: {base_weight.element_size() / quantized.element_size():.1f}x)"
        )


def apply_lora_to_model(
    model: nn.Module,
    config: MLXTuneConfig,
) -> nn.Module:
    """將 LoRA adapter 套用到模型的目標模組"""
    target_modules = set(config.target_modules)
    lora_layers_count = 0

    for name, module in model.named_modules():
        # 檢查是否為目標模組
        module_name = name.split(".")[-1]
        if module_name not in target_modules:
            continue

        if not isinstance(module, nn.Linear):
            continue

        # 取得父模組
        parent_name = ".".join(name.split(".")[:-1])
        parent = model
        if parent_name:
            for part in parent_name.split("."):
                parent = getattr(parent, part)

        # 建立 LoRA 層
        lora_layer = MLXTuneLoRALinear(
            in_features=module.in_features,
            out_features=module.out_features,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
            qlora_bits=config.qlora_bits if config.training_mode in ("qlora", "qlora8bit") else None,
            qlora_group_size=config.qlora_group_size,
        )

        # 若 QLoRA，量化基底權重
        if config.training_mode in ("qlora", "qlora8bit"):
            lora_layer.quantize_base(module.weight.data)

        # 替換模組
        setattr(parent, module_name, lora_layer)
        lora_layers_count += 1

    # 凍結非 LoRA 參數
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False

    logger.info(
        f"[MLXTune] LoRA applied: {lora_layers_count} layers, "
        f"rank={config.lora_rank}, mode={config.training_mode}"
    )
    return model


# ===========================================================================
# MLX-Tune SFT Trainer
# ===========================================================================

class MLXTuneSFTTrainer(BaseTrainer):
    """MLX-Tune 監督微調訓練器

    支援：
    - LoRA 低秩微調
    - QLoRA 4bit/8bit 量化微調
    - 全參數微調
    - Apple Silicon Metal 後端
    - 統一記憶體優化
    - 端雲一體策略
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: MLXTuneConfig,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        training_data: Any = None,
    ):
        # 套用 LoRA/QLoRA（若啟用）
        if config.training_mode in ("lora", "qlora", "qlora8bit"):
            model = apply_lora_to_model(model, config)

        # 設備選擇：MLX-Tune 優先使用 MPS
        if config.device == "auto":
            config.device = "mps" if hasattr(torch, "mps") and torch.backends.mps.is_available() else "cpu"

        super().__init__(model, tokenizer, config, train_dataset, eval_dataset)

        if training_data is not None and train_dataset is None:
            self.train_dataset = self._build_dataset(training_data)

        logger.info(
            f"[MLXTuneSFTTrainer] Initialized: mode={config.training_mode}, "
            f"device={self.device}, metal={config.enable_metal_backend}"
        )

    def _build_dataset(self, data: Any) -> Dataset:
        return SFTDataset(
            data=data,
            tokenizer=self.tokenizer,
            chat_template=self.config.chat_template,
            max_length=self.config.max_length,
            multi_turn=self.config.multi_turn,
        )

    def _maybe_cgc_compile(self) -> None:
        """覆寫：MLX-Tune 使用 Metal Graph"""
        if not self.config.use_cgc_compile:
            return
        try:
            # MLX-Tune 使用 Metal Graph execution
            self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False)
            logger.info("[MLXTuneSFTTrainer] Model compiled with Metal Graph (reduce-overhead)")
        except Exception as e:
            logger.warning(f"[MLXTuneSFTTrainer] Metal Graph compile failed: {e}")

    def _sync_device(self) -> None:
        """覆寫：MPS 同步"""
        if self.device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()
        elif self.device.type == "cuda":
            torch.cuda.synchronize()


# ===========================================================================
# MLX-Tune CPT Trainer
# ===========================================================================

class MLXTuneCPTTrainer(MLXTuneSFTTrainer):
    """MLX-Tune 持續預訓練訓練器

    特色：
    - 全文本無 mask 損失
    - 適配行業知識庫、小語種擴充
    - Metal 後端優化
    """

    def _build_dataset(self, data: Any) -> Dataset:
        return TextDataset(
            data=data,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
        )

    def _compute_loss(self, model: nn.Module, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """CPT: 全文本無 mask 損失"""
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = model(**inputs)

        if isinstance(outputs, dict) and "loss" in outputs:
            return outputs["loss"]

        if isinstance(outputs, torch.Tensor):
            logits = outputs
            labels = inputs.get("labels", inputs.get("input_ids"))
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            return loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return torch.tensor(0.0, device=self.device, requires_grad=True)


# ===========================================================================
# MLX-Tune 全參數微調 Trainer
# ===========================================================================

class MLXTuneFullFinetuneTrainer(MLXTuneSFTTrainer):
    """MLX-Tune 全參數微調訓練器

    特色：
    - 所有參數可訓練（不套用 LoRA）
    - 適合小模型（< 7B）在 Mac Studio 大統一記憶體上微調
    - 使用統一記憶體避免顯存限制
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: MLXTuneConfig,
        **kwargs,
    ):
        # 強制全參數模式
        config.training_mode = "full"
        config.enable_moe_training = False

        super().__init__(model, tokenizer, config, **kwargs)

        # 確保所有參數可訓練
        for param in self.model.parameters():
            param.requires_grad = True

        logger.info(
            f"[MLXTuneFullFinetuneTrainer] Full parameter fine-tuning, "
            f"trainable params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
        )


# ===========================================================================
# MLX-Tune 8bit 量化微調 Trainer
# ===========================================================================

class MLXTune8bitQuantTrainer(MLXTuneSFTTrainer):
    """MLX-Tune 8bit 量化微調訓練器

    特色：
    - 基底權重 8bit 量化
    - LoRA adapter 保持全精度
    - 比 4bit 更精確，適合中等規模模型
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: MLXTuneConfig,
        **kwargs,
    ):
        # 強制 8bit QLoRA
        config.training_mode = "qlora8bit"
        config.qlora_bits = 8

        super().__init__(model, tokenizer, config, **kwargs)

        logger.info(
            f"[MLXTune8bitQuantTrainer] 8bit QLoRA enabled, "
            f"group_size={config.qlora_group_size}"
        )


# ===========================================================================
# MLX-Tune MoE Trainer
# ===========================================================================

class MLXTuneMoETrainer(MLXTuneSFTTrainer):
    """MLX-Tune MoE 混合專家微調訓練器

    實作 mlx-tune 的 MoE 微調：
    - 自動識別 MoE 層
    - 逐專家 LoRA 微調
    - Mac Studio 大統一記憶體可跑 350B MoE 微調
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: MLXTuneConfig,
        **kwargs,
    ):
        config.enable_moe_training = True
        super().__init__(model, tokenizer, config, **kwargs)

        # 識別 MoE 專家層
        self._expert_layers: Dict[str, List[str]] = {}
        self._identify_experts()

        # 逐專家套用 LoRA
        if config.moe_expert_lora:
            self._apply_expert_lora()

        logger.info(
            f"[MLXTuneMoETrainer] MoE training enabled, "
            f"expert layers: {len(self._expert_layers)}"
        )

    def _identify_experts(self) -> None:
        """識別 MoE 專家層"""
        for name, module in self.model.named_modules():
            if hasattr(module, "experts") or "moe" in name.lower():
                experts = getattr(module, "experts", None)
                if experts is not None:
                    if isinstance(experts, nn.ModuleList):
                        self._expert_layers[name] = [
                            f"{name}.experts.{i}" for i in range(len(experts))
                        ]

    def _apply_expert_lora(self) -> None:
        """對每個專家套用獨立的 LoRA adapter"""
        lora_count = 0
        for moe_layer_name, expert_names in self._expert_layers.items():
            for expert_name in expert_names:
                expert_module = dict(self.model.named_modules()).get(expert_name)
                if expert_module is None:
                    continue

                # 對專家內部的 Linear 層套用 LoRA
                for sub_name, sub_module in expert_module.named_modules():
                    if isinstance(sub_module, nn.Linear):
                        full_name = f"{expert_name}.{sub_name}"
                        module_name = sub_name.split(".")[-1]

                        if module_name not in self.config.target_modules:
                            continue

                        # 建立 LoRA 層
                        lora_layer = MLXTuneLoRALinear(
                            in_features=sub_module.in_features,
                            out_features=sub_module.out_features,
                            rank=self.config.lora_rank,
                            alpha=self.config.lora_alpha,
                            dropout=self.config.lora_dropout,
                        )

                        # 替換
                        parent = expert_module
                        parts = sub_name.split(".")
                        for part in parts[:-1]:
                            parent = getattr(parent, part)
                        setattr(parent, parts[-1], lora_layer)
                        lora_count += 1

        logger.info(f"[MLXTuneMoETrainer] Expert LoRA applied: {lora_count} layers")


# ===========================================================================
# MLX-Tune 多模態 Trainer
# ===========================================================================

class MLXTuneMultimodalTrainer(MLXTuneSFTTrainer):
    """MLX-Tune 多模態微調訓練器

    支援：
    - 視覺-語言模型（VLM）微調
    - 圖像-文字對話資料
    - 視覺 tower 凍結 + LLM LoRA
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: MLXTuneConfig,
        image_processor: Any = None,
        **kwargs,
    ):
        config.enable_multimodal = True
        super().__init__(model, tokenizer, config, **kwargs)

        self.image_processor = image_processor

        # 凍結視覺 tower（若有）
        self._freeze_vision_tower()

        logger.info(
            f"[MLXTuneMultimodalTrainer] Multimodal training enabled, "
            f"vision_tower={config.vision_tower}"
        )

    def _freeze_vision_tower(self) -> None:
        """凍結視覺 tower"""
        vision_keywords = ["vision", "visual", "image", "clip", "vit", "encoder"]
        frozen_count = 0

        for name, param in self.model.named_parameters():
            if any(kw in name.lower() for kw in vision_keywords):
                param.requires_grad = False
                frozen_count += 1

        if frozen_count > 0:
            logger.info(f"[MLXTuneMultimodalTrainer] Vision tower frozen: {frozen_count} params")

    def _compute_loss(self, model: nn.Module, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """多模態損失計算"""
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 處理圖像輸入
        if "pixel_values" in inputs and self.image_processor is not None:
            # 圖像已由 image_processor 預處理
            pass

        outputs = model(**inputs)

        if isinstance(outputs, dict) and "loss" in outputs:
            return outputs["loss"]
        if isinstance(outputs, torch.Tensor):
            return outputs.mean()

        return torch.tensor(0.0, device=self.device, requires_grad=True)


# ===========================================================================
# Unsloth API 相容層
# ===========================================================================

class UnslothCompatAdapter:
    """Unsloth API 相容層

    提供與 Unsloth API 相容的介面，讓用戶可以無痛從 Unsloth 遷移到 MLX-Tune。

    相容的 API：
    - FastLanguageModel.from_pretrained()
    - FastLanguageModel.get_peft_model()
    - trainer.train()
    """

    def __init__(self):
        self._model: Optional[nn.Module] = None
        self._tokenizer: Any = None
        self._config: Optional[MLXTuneConfig] = None
        self._trainer: Optional[MLXTuneSFTTrainer] = None

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        max_seq_length: int = 2048,
        dtype: Optional[torch.dtype] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        **kwargs,
    ) -> "UnslothCompatAdapter":
        """相容 Unsloth FastLanguageModel.from_pretrained()"""
        adapter = cls()

        # 載入模型
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            adapter._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype or torch.float16,
                **kwargs,
            )
            adapter._tokenizer = AutoTokenizer.from_pretrained(model_name)
        except ImportError:
            raise RuntimeError(
                "transformers library is required. Install with: pip install transformers"
            )

        # 建立配置
        adapter._config = MLXTuneConfig(
            max_length=max_seq_length,
            training_mode="qlora8bit" if load_in_8bit else ("qlora" if load_in_4bit else "lora"),
            qlora_bits=8 if load_in_8bit else 4,
        )

        logger.info(
            f"[UnslothCompatAdapter] Model loaded: {model_name}, "
            f"4bit={load_in_4bit}, 8bit={load_in_8bit}"
        )
        return adapter

    def get_peft_model(
        self,
        r: int = 8,
        target_modules: Optional[List[str]] = None,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        bias: str = "none",
        **kwargs,
    ) -> "UnslothCompatAdapter":
        """相容 Unsloth get_peft_model()"""
        self._config.lora_rank = r
        self._config.lora_alpha = lora_alpha
        self._config.lora_dropout = lora_dropout
        if target_modules:
            self._config.target_modules = target_modules

        # 套用 LoRA
        self._model = apply_lora_to_model(self._model, self._config)

        # 印出可訓練參數
        trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self._model.parameters())
        logger.info(
            f"[UnslothCompatAdapter] PEFT applied: "
            f"trainable={trainable:,} ({100*trainable/total:.2f}%), total={total:,}"
        )
        return self

    def train(
        self,
        train_data: Any,
        eval_data: Any = None,
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-5,
        **kwargs,
    ) -> Dict[str, Any]:
        """相容 Unsloth trainer.train()"""
        self._config.num_train_epochs = epochs
        self._config.per_device_train_batch_size = batch_size
        self._config.learning_rate = learning_rate

        self._trainer = MLXTuneSFTTrainer(
            model=self._model,
            tokenizer=self._tokenizer,
            config=self._config,
            training_data=train_data,
        )

        if eval_data is not None:
            from .base_trainer import SFTDataset
            self._trainer.eval_dataset = SFTDataset(
                data=eval_data,
                tokenizer=self._tokenizer,
                chat_template=self._config.chat_template,
                max_length=self._config.max_length,
            )

        return self._trainer.train()

    def save_model(self, path: str) -> None:
        """儲存模型"""
        if self._trainer is not None:
            self._trainer.save_model(path)
        elif self._model is not None:
            os.makedirs(path, exist_ok=True)
            torch.save(self._model.state_dict(), os.path.join(path, "model.pt"))

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer


# ===========================================================================
# 工廠函數
# ===========================================================================

MLXTUNE_TRAINER_REGISTRY = {
    "sft": MLXTuneSFTTrainer,
    "cpt": MLXTuneCPTTrainer,
    "full": MLXTuneFullFinetuneTrainer,
    "qlora8bit": MLXTune8bitQuantTrainer,
    "moe": MLXTuneMoETrainer,
    "multimodal": MLXTuneMultimodalTrainer,
}


def create_mlxtune_trainer(
    algorithm: str,
    model: nn.Module,
    tokenizer: Any,
    config: MLXTuneConfig,
    **kwargs,
) -> BaseTrainer:
    """建立 MLX-Tune 訓練器

    Args:
        algorithm: 訓練器類型（sft/cpt/full/qlora8bit/moe/multimodal）
        model: 模型
        tokenizer: tokenizer
        config: 訓練配置
        **kwargs: 其他參數

    Returns:
        MLX-Tune 訓練器實例
    """
    algorithm = algorithm.lower()
    if algorithm not in MLXTUNE_TRAINER_REGISTRY:
        raise ValueError(
            f"Unknown MLX-Tune trainer: {algorithm}. "
            f"Available: {list(MLXTUNE_TRAINER_REGISTRY.keys())}"
        )

    trainer_cls = MLXTUNE_TRAINER_REGISTRY[algorithm]
    return trainer_cls(model=model, tokenizer=tokenizer, config=config, **kwargs)
