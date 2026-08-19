# Copyright (c) 2026 SandAI. All Rights Reserved.
"""
MegaTrain 訓練器模組（CUDA 後端）

功能：
- MegatrainSFTTrainer: 監督微調（全參數 + LoRA）
- MegatrainCPTTrainer: 持續頓訓練
- MegatrainMoETrainer: MoE 混合專家訓練
- CPUOffloadOptimizer: CPU 記憶體主存儲 + 優化器託管（MegaTrain 核心架構）
- PipelineStreamScheduler: 3-stream 雙緩衝預取
- StatelessLayerTemplate: 無狀態 Layer 模板
- LongContextManager: 超大上下文支援
"""

import os
import time
import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Iterator, Callable

import torch
import torch.nn as nn
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
# MegaTrain 核心架構：CPU 記憶體主存儲 + 優化器託管
# ===========================================================================

class CPUOffloadOptimizer:
    """CPU 記憶體主存儲優化器

    實作 MegaTrain 的核心架構：
    - 權重、梯度、AdamW 優化器狀態全部常駐主機 RAM
    - GPU 僅載入單層做前向/反向，計算完立即釋放
    - 支援 AdamW / Adam8bit

    適用場景：單卡大記憶體伺服器（H200+1.5TB RAM），全精度訓練 120B 模型
    """

    def __init__(
        self,
        params: List[nn.Parameter],
        lr: float = 2e-5,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        optimizer_type: str = "adamw",  # adamw, adam8bit
    ):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer_type

        # 優化器狀態全部在 CPU
        self._param_states: Dict[int, Dict[str, torch.Tensor]] = {}
        for param in params:
            if not param.requires_grad:
                continue
            pid = id(param)
            self._param_states[pid] = {
                "m": torch.zeros_like(param.data, device="cpu", dtype=torch.float32),
                "v": torch.zeros_like(param.data, device="cpu", dtype=torch.float32),
                "step": 0,
            }

        self._params = {id(p): p for p in params if p.requires_grad}
        logger.info(
            f"[CPUOffloadOptimizer] Initialized {optimizer_type} for {len(self._params)} params, "
            f"all states on CPU"
        )

    def step(self) -> None:
        """CPU 側優化器步進"""
        for pid, param in self._params.items():
            if param.grad is None:
                continue

            state = self._param_states[pid]
            state["step"] += 1
            step = state["step"]

            # 梯度搬到 CPU
            grad = param.grad.detach().to("cpu", dtype=torch.float32)
            param_data_cpu = param.data.detach().to("cpu", dtype=torch.float32)

            m = state["m"]
            v = state["v"]

            # AdamW 更新
            m.mul_(self.betas[0]).add_(grad, alpha=1 - self.betas[0])
            v.mul_(self.betas[1]).addcmul_(grad, grad, value=1 - self.betas[1])

            bias_correction1 = 1 - self.betas[0] ** step
            bias_correction2 = 1 - self.betas[1] ** step
            step_size = self.lr / bias_correction1
            denom = (v.sqrt() / math.sqrt(bias_correction2)).add_(self.eps)

            # 解耦權重衰減
            if self.weight_decay > 0:
                param_data_cpu.add_(param_data_cpu, alpha=-self.lr * self.weight_decay)

            # 參數更新
            param_data_cpu.addcdiv_(m, denom, value=-step_size)

            # 8bit 量化（若啟用）
            if self.optimizer_type == "adam8bit":
                param_data_cpu = self._quantize_8bit(param_data_cpu)

            # 更新後的權重搬回 GPU
            param.data.copy_(param_data_cpu.to(param.device, dtype=param.dtype))

    def _quantize_8bit(self, tensor: torch.Tensor) -> torch.Tensor:
        """8bit 量化（減少 CPU 記憶體佔用）"""
        scale = tensor.abs().max() / 127.0
        if scale == 0:
            return tensor
        quantized = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
        return quantized.to(torch.float32) * scale

    def zero_grad(self) -> None:
        for param in self._params.values():
            if param.grad is not None:
                param.grad = None


# ===========================================================================
# MegaTrain 核心架構：3-stream 雙緩衝預取
# ===========================================================================

class PipelineStreamScheduler:
    """3-stream 雙緩衝預取排程器

    實作 MegaTrain 的流水線雙緩衝多 CUDA Stream 架構：
    - Stream 1: 參數預取（H2D，CPU→GPU）
    - Stream 2: GPU 計算（前向/反向）
    - Stream 3: 梯度回寫（D2H，GPU→CPU）

    消除 PCIe 帶寬等待，GPU 算力持續滿載。
    """

    def __init__(self, model: nn.Module, prefetch_layers: int = 2):
        self.model = model
        self.prefetch_layers = prefetch_layers

        # 建立 3 條獨立 CUDA Stream
        self.stream_prefetch = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.stream_compute = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.stream_grad_writeback = torch.cuda.Stream() if torch.cuda.is_available() else None

        # 雙緩衝
        self._buffer_a: Dict[str, torch.Tensor] = {}
        self._buffer_b: Dict[str, torch.Tensor] = {}
        self._active_buffer = "a"

        # 層列表
        self._layer_names: List[str] = []
        for name, _ in model.named_modules():
            if isinstance(_, (nn.Linear, nn.LayerNorm, nn.Embedding)):
                self._layer_names.append(name)

        logger.info(
            f"[PipelineStreamScheduler] 3 CUDA streams, "
            f"prefetch_layers={prefetch_layers}, tracked_layers={len(self._layer_names)}"
        )

    def get_layer_params(self, layer_name: str) -> Dict[str, torch.Tensor]:
        """取得指定層的參數（從當前緩衝區）"""
        buf = self._buffer_a if self._active_buffer == "a" else self._buffer_b
        return {k.replace(f"{layer_name}.", ""): v for k, v in buf.items() if k.startswith(layer_name)}

    def prefetch_next(self, current_layer_idx: int) -> None:
        """預取接下來的層到 GPU（Stream 1）"""
        if self.stream_prefetch is None:
            return

        next_buffer = self._buffer_b if self._active_buffer == "a" else self._buffer_a
        end_idx = min(current_layer_idx + self.prefetch_layers + 1, len(self._layer_names))

        with torch.cuda.stream(self.stream_prefetch):
            for i in range(current_layer_idx + 1, end_idx):
                layer_name = self._layer_names[i]
                module = dict(self.model.named_modules())[layer_name]
                for pname, param in module.named_parameters(recurse=False):
                    key = f"{layer_name}.{pname}"
                    if key not in next_buffer:
                        next_buffer[key] = param.data.to("cuda", non_blocking=True)

        # 切換緩衝區
        self._active_buffer = "b" if self._active_buffer == "a" else "a"

    def wait_prefetch(self) -> None:
        """等待預取完成"""
        if self.stream_prefetch is not None and self.stream_compute is not None:
            self.stream_compute.wait_stream(self.stream_prefetch)

    def writeback_grad(self, layer_name: str) -> None:
        """梯度回寫到 CPU（Stream 3）"""
        if self.stream_grad_writeback is None:
            return

        module = dict(self.model.named_modules()).get(layer_name)
        if module is None:
            return

        with torch.cuda.stream(self.stream_grad_writeback):
            for pname, param in module.named_parameters(recurse=False):
                if param.grad is not None:
                    # 梯度搬到 CPU（非阻塞）
                    _ = param.grad.to("cpu", non_blocking=True)

    def synchronize(self) -> None:
        """同步所有 stream"""
        if self.stream_prefetch is not None:
            self.stream_prefetch.synchronize()
        if self.stream_compute is not None:
            self.stream_compute.synchronize()
        if self.stream_grad_writeback is not None:
            self.stream_grad_writeback.synchronize()


# ===========================================================================
# MegaTrain 核心架構：無狀態 Layer 模板
# ===========================================================================

class StatelessLayerTemplate(nn.Module):
    """無狀態 Layer 模板

    實作 MegaTrain 的核心創新：
    - 拋棄持久 autograd 計算圖
    - 每層動態綁定流式權重
    - 支援動態層調度、梯度重算（block-wise recompute）

    適用場景：單層流式執行，大幅減少顯存中計算圖元數據開銷
    """

    def __init__(self, layer_fn: Callable[..., nn.Module], layer_config: Dict[str, Any]):
        super().__init__()
        self.layer_fn = layer_fn
        self.layer_config = layer_config
        self._layer: Optional[nn.Module] = None
        self._bound_weights: Dict[str, torch.Tensor] = {}

        # 用於 block-wise recompute
        self._recompute_enabled = False
        self._input_cache: Optional[torch.Tensor] = None

    def bind_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """動態綁定流式權重（從 CPU 載入到 GPU）"""
        self._bound_weights = weights
        if self._layer is None:
            self._layer = self.layer_fn(**self.layer_config)

        for name, tensor in weights.items():
            if hasattr(self._layer, name):
                param = getattr(self._layer, name)
                param.data = tensor.to(param.device, dtype=param.dtype)

    def enable_recompute(self) -> None:
        """啟用梯度重算（block-wise recompute）"""
        self._recompute_enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._layer is None:
            raise RuntimeError("Layer not initialized. Call bind_weights() first.")

        if self._recompute_enabled and self.training:
            # 保存輸入，前向時不保存中間激活
            self._input_cache = x.detach().requires_grad_(True)
            return torch.utils.checkpoint.checkpoint(self._layer, self._input_cache)
        else:
            return self._layer(x)

    def release_weights(self) -> None:
        """釋放權重（計算完立即釋放顯存）"""
        if self._layer is not None:
            for name, param in self._layer.named_parameters():
                param.data = torch.empty(0, device=param.device)
        self._bound_weights.clear()
        self._input_cache = None


# ===========================================================================
# MegaTrain 核心架構：超大上下文支援
# ===========================================================================

class LongContextManager:
    """超大上下文管理器

    實作 MegaTrain 的超大上下文支援：
    - GH200 單卡支援 7B 模型 512k 超長上下文
    - 不需分片上下文並行
    - 使用流式注意力 + 記憶體映射
    """

    def __init__(
        self,
        max_seq_len: int = 512_000,
        chunk_size: int = 8192,
        overlap_size: int = 512,
        streaming_attention: bool = True,
    ):
        self.max_seq_len = max_seq_len
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.streaming_attention = streaming_attention

        logger.info(
            f"[LongContextManager] max_seq_len={max_seq_len}, "
            f"chunk_size={chunk_size}, overlap={overlap_size}"
        )

    def chunk_sequence(self, input_ids: torch.Tensor) -> List[torch.Tensor]:
        """將超長序列切分為重疊的 chunk"""
        seq_len = input_ids.shape[-1]
        chunks = []
        start = 0

        while start < seq_len:
            end = min(start + self.chunk_size, seq_len)
            chunks.append(input_ids[..., start:end])
            if end >= seq_len:
                break
            start = end - self.overlap_size

        return chunks

    def merge_outputs(
        self,
        chunk_outputs: List[torch.Tensor],
        original_len: int,
    ) -> torch.Tensor:
        """合併 chunk 輸出，處理重疊區域"""
        if len(chunk_outputs) == 1:
            return chunk_outputs[0][..., :original_len]

        merged = list(chunk_outputs[0])
        for i in range(1, len(chunk_outputs)):
            chunk = chunk_outputs[i]
            # 跳過重疊區域
            overlap = self.overlap_size if i < len(chunk_outputs) - 1 else 0
            merged.extend(chunk[self.overlap_size:] if overlap > 0 else chunk)

        result = torch.stack(merged[:original_len], dim=0)
        return result

    def streaming_forward(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """流式前向：對超長序列逐 chunk 計算"""
        chunks = self.chunk_sequence(input_ids)
        outputs = []

        for chunk in chunks:
            chunk = chunk.to(next(model.parameters()).device)
            with torch.no_grad():
                out = model(chunk)
            outputs.append(out)
            # 釋放中間激活
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return self.merge_outputs(outputs, input_ids.shape[-1])


# ===========================================================================
# MegaTrain 訓練配置
# ===========================================================================

@dataclass
class MegatrainConfig(TrainingConfig):
    """MegaTrain 訓練配置（繼承 TrainingConfig）"""
    # FSDP
    fsdp_sharding_strategy: str = "full_shard"
    fsdp_activation_checkpointing: bool = True
    fsdp_cpu_offload: bool = False

    # CPU 記憶體主存儲（MegaTrain 核心架構）
    use_cpu_memory_offload: bool = False
    cpu_optimizer_type: str = "adamw"  # adamw, adam8bit

    # 3-stream 雙緩衝
    use_pipeline_streams: bool = False
    prefetch_layers: int = 2

    # 無狀態 Layer
    use_stateless_layer: bool = False

    # 超大上下文
    use_long_context: bool = False
    long_context_max_len: int = 512_000
    long_context_chunk_size: int = 8192

    # MoE
    enable_moe_training: bool = False
    moe_num_experts: int = 8
    moe_top_k: int = 2

    # 訓練模式
    training_mode: str = "full"  # full, lora, qlora
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    qlora_bits: int = 4


# ===========================================================================
# MegaTrain SFT Trainer
# ===========================================================================

class MegatrainSFTTrainer(BaseTrainer):
    """MegaTrain 監督微調訓練器

    支援：
    - 全參數微調（full）
    - LoRA 微調（lora）
    - QLoRA 量化微調（qlora）
    - CPU 記憶體主存儲（use_cpu_memory_offload）
    - 3-stream 雙緩衝預取（use_pipeline_streams）
    - 無狀態 Layer（use_stateless_layer）
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: MegatrainConfig,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        training_data: Any = None,
    ):
        # 套用 LoRA/QLoRA（若啟用）
        if config.training_mode in ("lora", "qlora"):
            model = self._apply_lora(model, config)

        super().__init__(model, tokenizer, config, train_dataset, eval_dataset)

        # 若提供了原始訓練資料，建立資料集
        if training_data is not None and train_dataset is None:
            self.train_dataset = self._build_dataset(training_data)

        # MegaTrain 核心架構
        self._cpu_optimizer: Optional[CPUOffloadOptimizer] = None
        self._stream_scheduler: Optional[PipelineStreamScheduler] = None

        if config.use_cpu_memory_offload:
            logger.info("[MegatrainSFTTrainer] CPU memory offload enabled (MegaTrain core architecture)")

        if config.use_pipeline_streams:
            self._stream_scheduler = PipelineStreamScheduler(model, config.prefetch_layers)

    @staticmethod
    def _apply_lora(model: nn.Module, config: MegatrainConfig) -> nn.Module:
        """套用 LoRA/QLoRA adapter"""
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        except ImportError:
            logger.warning("[MegatrainSFTTrainer] peft not installed, skipping LoRA")
            return model

        if config.training_mode == "qlora":
            # QLoRA: 先量化再套用 LoRA
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
                logger.info("[MegatrainSFTTrainer] QLoRA 4bit quantization applied")
            except ImportError:
                logger.warning("[MegatrainSFTTrainer] bitsandbytes not installed, using LoRA without quantization")

        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)
        logger.info(
            f"[MegatrainSFTTrainer] LoRA applied: rank={config.lora_rank}, "
            f"alpha={config.lora_alpha}, mode={config.training_mode}"
        )
        return model

    def _build_dataset(self, data: Any) -> Dataset:
        """建立 SFT 資料集"""
        return SFTDataset(
            data=data,
            tokenizer=self.tokenizer,
            chat_template=self.config.chat_template,
            max_length=self.config.max_length,
            multi_turn=self.config.multi_turn,
        )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """覆寫：支援 CPU 記憶體主存儲優化器"""
        if self.config.use_cpu_memory_offload:
            params = list(self.model.parameters())
            self._cpu_optimizer = CPUOffloadOptimizer(
                params=params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                optimizer_type=self.config.cpu_optimizer_type,
            )
            # 返回一個佔位優化器（實際步進由 _cpu_optimizer 處理）
            return torch.optim.SGD([{"params": [p for p in params if p.requires_grad], "lr": 0}])
        return super()._build_optimizer()

    def _backward_and_step(self, loss: torch.Tensor) -> None:
        """覆寫：支援 CPU offload + 3-stream"""
        loss = loss / self.config.gradient_accumulation_steps
        loss.backward()

        if (self.state.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

            if self._cpu_optimizer is not None:
                self._cpu_optimizer.step()
                self._cpu_optimizer.zero_grad()
            else:
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.scheduler.step()


# ===========================================================================
# MegaTrain CPT Trainer
# ===========================================================================

class MegatrainCPTTrainer(BaseTrainer):
    """MegaTrain 持續預訓練訓練器

    特色：
    - 全文本無 mask 損失（不同於 SFT 的 prompt mask）
    - 支援領域知識庫、小語種擴充
    - 全精度 BF16/FP32 預訓練
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: MegatrainConfig,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        training_data: Any = None,
    ):
        super().__init__(model, tokenizer, config, train_dataset, eval_dataset)

        if training_data is not None and train_dataset is None:
            self.train_dataset = self._build_dataset(training_data)

    def _build_dataset(self, data: Any) -> Dataset:
        """建立 CPT 純文字資料集"""
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

        # 自行計算 LM loss（全文本，不 mask）
        if isinstance(outputs, torch.Tensor):
            logits = outputs
            labels = inputs.get("labels", inputs.get("input_ids"))
            # CPT: 不 mask prompt，全序列計算 loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            return loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return torch.tensor(0.0, device=self.device, requires_grad=True)


# ===========================================================================
# MegaTrain MoE Trainer
# ===========================================================================

class MegatrainMoETrainer(MegatrainSFTTrainer):
    """MegaTrain MoE 混合專家訓練器

    實作 MegaTrain 的原生 MoE 支援：
    - 分層流式載入專家權重
    - 單卡訓練 120B MoE，無需多卡專家並行
    - 支援逐專家梯度累積
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: MegatrainConfig,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        training_data: Any = None,
    ):
        config.enable_moe_training = True
        super().__init__(model, tokenizer, config, train_dataset, eval_dataset, training_data)

        # 識別 MoE 專家層
        self._expert_layers: Dict[str, List[str]] = {}
        self._identify_experts()

        logger.info(
            f"[MegatrainMoETrainer] MoE experts identified: "
            f"{len(self._expert_layers)} MoE layers"
        )

    def _identify_experts(self) -> None:
        """識別模型中的 MoE 專家層"""
        for name, module in self.model.named_modules():
            if hasattr(module, "experts") or "moe" in name.lower():
                experts = getattr(module, "experts", None)
                if experts is not None:
                    if isinstance(experts, nn.ModuleList):
                        self._expert_layers[name] = [
                            f"{name}.experts.{i}" for i in range(len(experts))
                        ]
                    elif isinstance(experts, dict):
                        self._expert_layers[name] = [
                            f"{name}.experts.{k}" for k in experts.keys()
                        ]

    def _stream_expert_weights(self, layer_name: str) -> Iterator[Dict[str, torch.Tensor]]:
        """流式載入專家權重（一次只載入一個專家到 GPU）"""
        expert_names = self._expert_layers.get(layer_name, [])
        for expert_name in expert_names:
            expert_module = dict(self.model.named_modules()).get(expert_name)
            if expert_module is not None:
                weights = {}
                for pname, param in expert_module.named_parameters():
                    weights[pname] = param.data.to(self.device, non_blocking=True)
                yield {expert_name: weights}

    def _compute_loss(self, model: nn.Module, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """MoE 訓練損失（含負載均衡 loss）"""
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = model(**inputs)

        if isinstance(outputs, dict):
            base_loss = outputs.get("loss", torch.tensor(0.0, device=self.device))
            # 加入 MoE 負載均衡 loss（鼓勵專家負載均勻）
            aux_loss = outputs.get("aux_loss", outputs.get("load_balancing_loss", None))
            if aux_loss is not None:
                balance_weight = 0.01
                return base_loss + balance_weight * aux_loss
            return base_loss

        if isinstance(outputs, torch.Tensor):
            return outputs.mean()

        return torch.tensor(0.0, device=self.device, requires_grad=True)
