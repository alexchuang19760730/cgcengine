# Copyright (c) 2026 SandAI. All Rights Reserved.
"""
CGC Trainers 模組

統一訓練器入口，提供 MegaTrain（CUDA）與 MLX-Tune（Metal）兩套後端的訓練器。

訓練器清單：
    MegaTrain（CUDA 後端）：
    - MegatrainSFTTrainer       監督微調（全參數 + LoRA + QLoRA）
    - MegatrainCPTTrainer       持續預訓練
    - MegatrainMoETrainer       MoE 混合專家訓練

    MLX-Tune（Metal 後端）：
    - MLXTuneSFTTrainer         監督微調（LoRA/QLoRA/全參數）
    - MLXTuneCPTTrainer         持續預訓練
    - MLXTuneFullFinetuneTrainer 全參數微調
    - MLXTune8bitQuantTrainer   8bit 量化微調
    - MLXTuneMoETrainer         MoE 混合專家微調（逐專家 LoRA）
    - MLXTuneMultimodalTrainer  多模態微調
    - UnslothCompatAdapter      Unsloth API 相容層

    偏好對齊（CUDA + Metal 共用）：
    - DPOTrainer                直接偏好優化
    - ORPOTrainer               SFT + 偏好損失一體化
    - GRPOTrainer               推理增強對齊（DeepSeek-R1）
    - KTOTrainer                輕量偏好優化（無需參考模型）
    - SimPOTrainer              簡單偏好優化（無需參考模型）

    核心架構組件（MegaTrain）：
    - CPUOffloadOptimizer       CPU 記憶體主存儲優化器
    - PipelineStreamScheduler   3-stream 雙緩衝預取
    - StatelessLayerTemplate    無狀態 Layer 模板
    - LongContextManager        超大上下文管理器

使用範例：
    >>> from cgc_engine.agent.trainers import MegatrainSFTTrainer, MegatrainConfig
    >>> trainer = MegatrainSFTTrainer(model, tokenizer, config, training_data="data.jsonl")
    >>> result = trainer.train()

    >>> from cgc_engine.agent.trainers import create_preference_trainer, PreferenceConfig
    >>> trainer = create_preference_trainer("dpo", model, tokenizer, config, training_data="prefs.jsonl")
    >>> result = trainer.train()
"""

# 基礎
from .base_trainer import (
    BaseTrainer,
    TrainingConfig,
    TrainState,
    SFTDataset,
    PreferenceDataset,
    TextDataset,
    ConversationExample,
    PreferenceExample,
    TextExample,
    get_chat_template,
    apply_chat_template,
)

# MegaTrain（CUDA 後端）
from .megatrain_trainers import (
    MegatrainConfig,
    MegatrainSFTTrainer,
    MegatrainCPTTrainer,
    MegatrainMoETrainer,
    CPUOffloadOptimizer,
    PipelineStreamScheduler,
    StatelessLayerTemplate,
    LongContextManager,
)

# MLX-Tune（Metal 後端）
from .mlxtune_trainers import (
    MLXTuneConfig,
    MLXTuneSFTTrainer,
    MLXTuneCPTTrainer,
    MLXTuneFullFinetuneTrainer,
    MLXTune8bitQuantTrainer,
    MLXTuneMoETrainer,
    MLXTuneMultimodalTrainer,
    UnslothCompatAdapter,
    MLXTuneLoRALinear,
    apply_lora_to_model,
    create_mlxtune_trainer,
    MLXTUNE_TRAINER_REGISTRY,
)

# 偏好對齊（共用）
from .preference_trainers import (
    PreferenceConfig,
    PreferenceTrainerBase,
    DPOTrainer,
    ORPOTrainer,
    GRPOTrainer,
    KTOTrainer,
    SimPOTrainer,
    create_preference_trainer,
    PREFERENCE_TRAINER_REGISTRY,
)

# NeMo Automodel Adapter（薄 Adapter，解綁 HF v5 / TE / DeepEP）
from .nemo_automodel_adapter import (
    NemoAutomodelConfig,
    NemoAutomodelMoETrainer,
    create_nemo_automodel_trainer,
    NEMO_AUTOMODEL_TRAINER_REGISTRY,
)

__all__ = [
    # 基礎
    "BaseTrainer",
    "TrainingConfig",
    "TrainState",
    "SFTDataset",
    "PreferenceDataset",
    "TextDataset",
    "ConversationExample",
    "PreferenceExample",
    "TextExample",
    "get_chat_template",
    "apply_chat_template",
    # MegaTrain
    "MegatrainConfig",
    "MegatrainSFTTrainer",
    "MegatrainCPTTrainer",
    "MegatrainMoETrainer",
    "CPUOffloadOptimizer",
    "PipelineStreamScheduler",
    "StatelessLayerTemplate",
    "LongContextManager",
    # MLX-Tune
    "MLXTuneConfig",
    "MLXTuneSFTTrainer",
    "MLXTuneCPTTrainer",
    "MLXTuneFullFinetuneTrainer",
    "MLXTune8bitQuantTrainer",
    "MLXTuneMoETrainer",
    "MLXTuneMultimodalTrainer",
    "UnslothCompatAdapter",
    "MLXTuneLoRALinear",
    "apply_lora_to_model",
    "create_mlxtune_trainer",
    "MLXTUNE_TRAINER_REGISTRY",
    # 偏好對齊
    "PreferenceConfig",
    "PreferenceTrainerBase",
    "DPOTrainer",
    "ORPOTrainer",
    "GRPOTrainer",
    "KTOTrainer",
    "SimPOTrainer",
    "create_preference_trainer",
    "PREFERENCE_TRAINER_REGISTRY",
    # NeMo Automodel Adapter
    "NemoAutomodelConfig",
    "NemoAutomodelMoETrainer",
    "create_nemo_automodel_trainer",
    "NEMO_AUTOMODEL_TRAINER_REGISTRY",
]
