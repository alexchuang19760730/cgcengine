# Copyright (c) 2026 SandAI. All Rights Reserved.
"""
Trainer 基礎模組

功能：
- 定義 BaseTrainer 抽象基類，統一訓練器介面
- 提供 chat template 處理
- 提供資料集載入與格式轉換
- 支援對話/指令/偏好三種資料格式
"""

import json
import os
import math
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chat Template 處理
# ---------------------------------------------------------------------------

# 內建 chat template（相容 HuggingFace tokenizer.apply_chat_template 格式）
_BUILTIN_TEMPLATES: Dict[str, str] = {
    "chatml": (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}"
    ),
    "llama3": (
        "{% for message in messages %}"
        "{{'<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{'<|start_header_id|>assistant<|end_header_id|>\n\n'}}{% endif %}"
    ),
    "qwen": (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}"
    ),
    "mistral": (
        "{% for message in messages %}"
        "{{'<s>[INST] ' + message['content'] + ' </s>'}}"
        "{% endfor %}"
    ),
    "zephyr": (
        "{% for message in messages %}"
        "{{'<|' + message['role'] + '|>' + message['content'] + '</s>'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{'<|assistant|>'}}{% endif %}"
    ),
}


def get_chat_template(name: str) -> str:
    """取得內建 chat template。

    Args:
        name: template 名稱（chatml/llama3/qwen/mistral/zephyr）

    Returns:
        Jinja2 template 字串
    """
    key = name.lower()
    if key not in _BUILTIN_TEMPLATES:
        raise ValueError(
            f"Unknown chat template: {name}. Available: {list(_BUILTIN_TEMPLATES.keys())}"
        )
    return _BUILTIN_TEMPLATES[key]


def apply_chat_template(
    messages: List[Dict[str, str]],
    template: str,
    add_generation_prompt: bool = False,
) -> str:
    """簡易 chat template 套用（不依賴 jinja2，直接字串拼接）。

    Args:
        messages: 對話訊息列表，每項含 role/content
        template: template 名稱或自訂格式
        add_generation_prompt: 是否加入 generation prompt

    Returns:
        格式化後的字串
    """
    # 若是內建 template 名稱，使用對應的拼接邏輯
    tpl_name = template.lower() if template.lower() in _BUILTIN_TEMPLATES else "chatml"

    parts: List[str] = []
    if tpl_name in ("chatml", "qwen"):
        for msg in messages:
            parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
    elif tpl_name == "llama3":
        for msg in messages:
            parts.append(f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n{msg['content']}<|eot_id|>")
        if add_generation_prompt:
            parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    elif tpl_name == "mistral":
        for msg in messages:
            role = msg["role"]
            if role == "user":
                parts.append(f"<s>[INST] {msg['content']} </s>")
            elif role == "assistant":
                parts.append(f" {msg['content']} </s>")
    elif tpl_name == "zephyr":
        for msg in messages:
            parts.append(f"<|{msg['role']}|>{msg['content']}</s>")
        if add_generation_prompt:
            parts.append("<|assistant|>")
    else:
        # fallback: 直接拼接
        for msg in messages:
            parts.append(f"{msg['role']}: {msg['content']}\n")

    return "".join(parts)


# ---------------------------------------------------------------------------
# 資料格式定義
# ---------------------------------------------------------------------------

@dataclass
class ConversationExample:
    """單輪對話樣本"""
    messages: List[Dict[str, str]]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreferenceExample:
    """偏好對齊樣本（chosen / rejected）"""
    prompt_messages: List[Dict[str, str]]
    chosen_messages: List[Dict[str, str]]
    rejected_messages: List[Dict[str, str]]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TextExample:
    """純文字樣本（用於 CPT）"""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 資料集
# ---------------------------------------------------------------------------

class SFTDataset(Dataset):
    """監督微調資料集

    支援兩種輸入格式：
    1. JSONL 檔案，每行一個 {"messages": [...]} 物件
    2. 已載入的 ConversationExample 列表
    """

    def __init__(
        self,
        data: Union[str, List[ConversationExample]],
        tokenizer: Any,
        chat_template: str = "chatml",
        max_length: int = 2048,
        multi_turn: bool = True,
    ):
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.max_length = max_length
        self.multi_turn = multi_turn
        self.examples: List[ConversationExample] = []

        if isinstance(data, str):
            self._load_jsonl(data)
        else:
            self.examples = data

    def _load_jsonl(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "messages" in obj:
                    self.examples.append(
                        ConversationExample(
                            messages=obj["messages"],
                            metadata=obj.get("metadata", {}),
                        )
                    )
                elif "text" in obj:
                    # 單輪文字轉為 user/assistant
                    self.examples.append(
                        ConversationExample(
                            messages=[
                                {"role": "user", "content": obj["text"]},
                                {"role": "assistant", "content": ""},
                            ],
                            metadata=obj.get("metadata", {}),
                        )
                    )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]

        if self.multi_turn:
            # 多輪：逐步生成 prompt + response，mask 掉 prompt 部分
            return self._build_multi_turn(example)
        else:
            # 單輪：整個對話作為一個序列
            text = apply_chat_template(example.messages, self.chat_template)
            return self._tokenize_with_labels(text, full_response=True)

    def _build_multi_turn(self, example: ConversationExample) -> Dict[str, torch.Tensor]:
        """多輪對話：對每個 assistant 回覆建立訓練目標。"""
        messages = example.messages
        input_ids: List[int] = []
        labels: List[int] = []

        # 找到最後一個 user/assistant 的邊界
        for i, msg in enumerate(messages):
            if msg["role"] == "assistant":
                # prompt = messages[:i], response = messages[i]
                prompt_msgs = messages[:i]
                response_msg = msg

                prompt_text = apply_chat_template(prompt_msgs, self.chat_template, add_generation_prompt=True)
                full_text = prompt_text + response_msg["content"] + "<|im_end|>\n"

                prompt_ids = self._encode(prompt_text)
                full_ids = self._encode(full_text)

                # prompt 部分 mask 為 -100
                prompt_len = len(prompt_ids)
                full_len = len(full_ids)

                input_ids.extend(full_ids)
                labels.extend([-100] * prompt_len + full_ids[prompt_len:])

                # 只處理最後一個 assistant 回覆（避免序列過長）
                break

        return self._pad_and_truncate(input_ids, labels)

    def _tokenize_with_labels(self, text: str, full_response: bool = False) -> Dict[str, torch.Tensor]:
        ids = self._encode(text)
        if full_response:
            labels = list(ids)
        else:
            labels = [-100] * len(ids)
        return self._pad_and_truncate(ids, labels)

    def _encode(self, text: str) -> List[int]:
        if hasattr(self.tokenizer, "encode"):
            result = self.tokenizer.encode(text, add_special_tokens=False)
            if isinstance(result, list):
                return result
            return result.tolist() if hasattr(result, "tolist") else list(result)
        # fallback: 簡易字元級編碼
        return [ord(c) % 32000 for c in text[: self.max_length]]

    def _pad_and_truncate(self, input_ids: List[int], labels: List[int]) -> Dict[str, torch.Tensor]:
        # 截斷
        input_ids = input_ids[: self.max_length]
        labels = labels[: self.max_length]

        # padding
        pad_id = 0
        if hasattr(self.tokenizer, "pad_token_id") and self.tokenizer.pad_token_id is not None:
            pad_id = self.tokenizer.pad_token_id

        attention_mask = [1] * len(input_ids)
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [pad_id] * pad_len
            labels = labels + [-100] * pad_len
            attention_mask = attention_mask + [0] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class PreferenceDataset(Dataset):
    """偏好對齊資料集（DPO/ORPO/KTO/SimPO）

    支援 JSONL 格式：
    {"prompt": [...], "chosen": [...], "rejected": [...]}
    """

    def __init__(
        self,
        data: Union[str, List[PreferenceExample]],
        tokenizer: Any,
        chat_template: str = "chatml",
        max_length: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.max_length = max_length
        self.examples: List[PreferenceExample] = []

        if isinstance(data, str):
            self._load_jsonl(data)
        else:
            self.examples = data

    def _load_jsonl(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.examples.append(
                    PreferenceExample(
                        prompt_messages=obj.get("prompt", obj.get("chosen", [{}])[0:1]),
                        chosen_messages=obj.get("chosen", []),
                        rejected_messages=obj.get("rejected", []),
                        metadata=obj.get("metadata", {}),
                    )
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]

        prompt_text = apply_chat_template(example.prompt_messages, self.chat_template, add_generation_prompt=True)
        chosen_text = apply_chat_template(
            example.prompt_messages + example.chosen_messages, self.chat_template
        )
        rejected_text = apply_chat_template(
            example.prompt_messages + example.rejected_messages, self.chat_template
        )

        prompt_ids = self._encode(prompt_text)
        chosen_ids = self._encode(chosen_text)
        rejected_ids = self._encode(rejected_text)

        return {
            "prompt_input_ids": torch.tensor(self._pad(prompt_ids), dtype=torch.long),
            "chosen_input_ids": torch.tensor(self._pad(chosen_ids), dtype=torch.long),
            "rejected_input_ids": torch.tensor(self._pad(rejected_ids), dtype=torch.long),
            "prompt_len": len(prompt_ids),
            "chosen_len": len(chosen_ids),
            "rejected_len": len(rejected_ids),
        }

    def _encode(self, text: str) -> List[int]:
        if hasattr(self.tokenizer, "encode"):
            result = self.tokenizer.encode(text, add_special_tokens=False)
            if isinstance(result, list):
                return result
            return result.tolist() if hasattr(result, "tolist") else list(result)
        return [ord(c) % 32000 for c in text[: self.max_length]]

    def _pad(self, ids: List[int]) -> List[int]:
        ids = ids[: self.max_length]
        pad_id = 0
        if hasattr(self.tokenizer, "pad_token_id") and self.tokenizer.pad_token_id is not None:
            pad_id = self.tokenizer.pad_token_id
        pad_len = self.max_length - len(ids)
        if pad_len > 0:
            ids = ids + [pad_id] * pad_len
        return ids


class TextDataset(Dataset):
    """純文字資料集（用於 CPT 持續預訓練）"""

    def __init__(
        self,
        data: Union[str, List[TextExample]],
        tokenizer: Any,
        max_length: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples: List[TextExample] = []

        if isinstance(data, str):
            self._load_jsonl(data)
        else:
            self.examples = data

    def _load_jsonl(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get("text", "")
                self.examples.append(TextExample(text=text, metadata=obj.get("metadata", {})))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        ids = self._encode(example.text)
        ids = ids[: self.max_length]

        pad_id = 0
        if hasattr(self.tokenizer, "pad_token_id") and self.tokenizer.pad_token_id is not None:
            pad_id = self.tokenizer.pad_token_id

        attention_mask = [1] * len(ids)
        pad_len = self.max_length - len(ids)
        if pad_len > 0:
            ids = ids + [pad_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len

        # CPT: 全文本無 mask 損失（不 mask prompt）
        labels = list(ids)
        labels = [-100 if i == pad_id else l for i, l in zip(ids, labels)]

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def _encode(self, text: str) -> List[int]:
        if hasattr(self.tokenizer, "encode"):
            result = self.tokenizer.encode(text, add_special_tokens=False)
            if isinstance(result, list):
                return result
            return result.tolist() if hasattr(result, "tolist") else list(result)
        return [ord(c) % 32000 for c in text]


# ---------------------------------------------------------------------------
# BaseTrainer 抽象基類
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """通用訓練配置"""
    # 輸出
    output_dir: str = "./output"
    save_steps: int = 500
    save_total_limit: int = 3

    # 訓練
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"  # cosine, linear, constant

    # 資料
    max_length: int = 2048
    chat_template: str = "chatml"
    multi_turn: bool = True

    # 混合精度
    mixed_precision: str = "bf16"  # bf16, fp16, fp32

    # 評估
    eval_steps: int = 500
    logging_steps: int = 10

    # 設備
    device: str = "auto"  # auto, cuda, mps, cpu

    # CGC 整合
    use_cgc_compile: bool = True
    use_cgc_simd: bool = True
    train_inference_consistency_check: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class TrainState:
    """訓練狀態追蹤"""
    epoch: int = 0
    global_step: int = 0
    local_step: int = 0
    total_loss: float = 0.0
    best_loss: float = float("inf")
    start_time: float = field(default_factory=time.time)

    def step(self, loss: float) -> None:
        self.global_step += 1
        self.local_step += 1
        self.total_loss += loss
        if loss < self.best_loss:
            self.best_loss = loss


class BaseTrainer(ABC):
    """所有訓練器的抽象基類

    統一介面：
    - train(): 主訓練循環
    - evaluate(): 評估
    - save_model(): 儲存模型
    - load_model(): 載入模型
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: TrainingConfig,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.state = TrainState()

        # 設備選擇
        self.device = self._resolve_device(config.device)
        self.model = self.model.to(self.device)

        # 優化器與排程器（延遲建立）
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[Any] = None

        # CGC 整合（延遲載入，避免循環依賴）
        self._cgc_engine: Any = None

        # 資料載入器（延遲建立）
        self._train_dataloader: Optional[DataLoader] = None

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """建立優化器（子類可覆寫以實作 CPU 側託管等）"""
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name.lower() for nd in ["bias", "layernorm", "norm"]):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        return torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.config.learning_rate,
        )

    def _build_scheduler(self, num_training_steps: int) -> Any:
        """建立學習率排程器"""
        warmup_steps = int(num_training_steps * self.config.warmup_ratio)

        if self.config.lr_scheduler_type == "cosine":
            from torch.optim.lr_scheduler import LambdaLR

            def lr_lambda(current_step: int) -> float:
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                progress = float(current_step - warmup_steps) / float(
                    max(1, num_training_steps - warmup_steps)
                )
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

            return LambdaLR(self.optimizer, lr_lambda)

        elif self.config.lr_scheduler_type == "linear":
            from torch.optim.lr_scheduler import LambdaLR

            def lr_lambda(current_step: int) -> float:
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return max(
                    0.0,
                    float(num_training_steps - current_step)
                    / float(max(1, num_training_steps - warmup_steps)),
                )

            return LambdaLR(self.optimizer, lr_lambda)

        else:
            # constant
            from torch.optim.lr_scheduler import LambdaLR
            return LambdaLR(self.optimizer, lambda _: 1.0)

    def _build_dataloader(self, dataset: Dataset, shuffle: bool = True) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.per_device_train_batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
        )

    def _get_cgc_engine(self) -> Any:
        """延遲載入 CGC 引擎（用於 SIMD 訓推共享指令集）"""
        if self._cgc_engine is None:
            try:
                from ..cgc.megatrain_integration import MegatrainCGCExec
                self._cgc_engine = MegatrainCGCExec
                logger.info("[BaseTrainer] CGC SIMD engine loaded")
            except ImportError:
                logger.warning("[BaseTrainer] CGC engine not available, using native PyTorch")
                self._cgc_engine = False  # 標記為不可用
        return self._cgc_engine if self._cgc_engine is not False else None

    def _maybe_cgc_compile(self) -> None:
        """若啟用 CGC 編譯，對模型套用 torch.compile + CGC 策略"""
        if not self.config.use_cgc_compile:
            return
        try:
            self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False)
            logger.info("[BaseTrainer] Model compiled with CGC strategy (reduce-overhead)")
        except Exception as e:
            logger.warning(f"[BaseTrainer] CGC compile failed, fallback to native: {e}")

    def _sync_device(self) -> None:
        """設備同步"""
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()

    def _compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """計算損失（子類可覆寫）"""
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = model(**inputs)
        if isinstance(outputs, dict) and "loss" in outputs:
            return outputs["loss"]
        if isinstance(outputs, torch.Tensor):
            return outputs.mean()
        if isinstance(outputs, (tuple, list)) and outputs:
            return outputs[0].mean() if isinstance(outputs[0], torch.Tensor) else torch.tensor(0.0)
        return torch.tensor(0.0, device=self.device, requires_grad=True)

    def _backward_and_step(self, loss: torch.Tensor) -> None:
        """反向傳播 + 梯度累積 + 優化器步進"""
        loss = loss / self.config.gradient_accumulation_steps
        loss.backward()

        if (self.state.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

    @abstractmethod
    def _build_dataset(self, data: Any) -> Dataset:
        """子類實作：建立資料集"""
        ...

    def train(self, resume_from: Optional[str] = None) -> Dict[str, Any]:
        """主訓練循環

        Args:
            resume_from: 從 checkpoint 恢復的路徑

        Returns:
            訓練結果摘要
        """
        if self.train_dataset is None:
            raise ValueError("train_dataset is not set")

        logger.info(f"[{self.__class__.__name__}] Starting training on device: {self.device}")

        # CGC 編譯
        self._maybe_cgc_compile()

        # 資料載入器
        if self._train_dataloader is None:
            self._train_dataloader = self._build_dataloader(self.train_dataset, shuffle=True)

        # 優化器與排程器
        if self.optimizer is None:
            self.optimizer = self._build_optimizer()

        steps_per_epoch = len(self._train_dataloader) // self.config.gradient_accumulation_steps
        total_steps = steps_per_epoch * self.config.num_train_epochs

        if self.scheduler is None:
            self.scheduler = self._build_scheduler(total_steps)

        # 恢復
        if resume_from:
            self.load_model(resume_from)

        # 混合精度
        autocast_dtype = None
        if self.config.mixed_precision == "bf16":
            autocast_dtype = torch.bfloat16
        elif self.config.mixed_precision == "fp16":
            autocast_dtype = torch.float16

        logger.info(
            f"[{self.__class__.__name__}] epochs={self.config.num_train_epochs}, "
            f"steps_per_epoch={steps_per_epoch}, total_steps={total_steps}"
        )

        os.makedirs(self.config.output_dir, exist_ok=True)
        self.model.train()

        for epoch in range(self.state.epoch, self.config.num_train_epochs):
            self.state.epoch = epoch
            self.state.local_step = 0
            self.state.total_loss = 0.0

            for batch_idx, batch in enumerate(self._train_dataloader):
                inputs = {k: v.to(self.device) for k, v in batch.items()}

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=autocast_dtype,
                    enabled=(autocast_dtype is not None),
                ):
                    loss = self._compute_loss(self.model, inputs)

                self._backward_and_step(loss)
                self.state.step(loss.item())

                if self.state.global_step % self.config.logging_steps == 0:
                    avg_loss = self.state.total_loss / max(1, self.state.local_step)
                    lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.config.learning_rate
                    logger.info(
                        f"[{self.__class__.__name__}] "
                        f"epoch={epoch} step={self.state.global_step} "
                        f"loss={avg_loss:.4f} lr={lr:.2e}"
                    )

                if self.state.global_step % self.config.save_steps == 0:
                    self.save_model(
                        os.path.join(self.config.output_dir, f"checkpoint-{self.state.global_step}")
                    )

            # epoch 結束評估
            if self.eval_dataset is not None:
                eval_loss = self.evaluate()
                logger.info(
                    f"[{self.__class__.__name__}] epoch={epoch} eval_loss={eval_loss:.4f}"
                )

        # 最終儲存
        self.save_model(os.path.join(self.config.output_dir, "final"))

        return {
            "status": "completed",
            "epochs": self.config.num_train_epochs,
            "total_steps": self.state.global_step,
            "best_loss": self.state.best_loss,
            "final_loss": self.state.total_loss / max(1, self.state.local_step),
            "device": str(self.device),
            "cgc_compile": self.config.use_cgc_compile,
        }

    def evaluate(self) -> float:
        """評估"""
        if self.eval_dataset is None:
            return 0.0

        self.model.eval()
        dataloader = self._build_dataloader(self.eval_dataset, shuffle=False)
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                loss = self._compute_loss(self.model, inputs)
                total_loss += loss.item()
                num_batches += 1

        self.model.train()
        return total_loss / max(1, num_batches)

    def save_model(self, path: str) -> None:
        """儲存模型"""
        os.makedirs(path, exist_ok=True)

        # 儲存模型權重
        unwrapped = self.model
        if hasattr(unwrapped, "_orig_mod"):
            unwrapped = unwrapped._orig_mod  # torch.compile 解包

        # LoRA adapter 特殊處理
        if self._is_lora_model(unwrapped):
            self._save_lora_adapter(unwrapped, path)
        else:
            torch.save(unwrapped.state_dict(), os.path.join(path, "model.pt"))

        # 儲存 tokenizer
        if hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(path)

        # 儲存訓練狀態
        torch.save(
            {
                "state": self.state.__dict__,
                "config": self.config.to_dict(),
            },
            os.path.join(path, "trainer_state.pt"),
        )
        logger.info(f"[{self.__class__.__name__}] Model saved to {path}")

    def load_model(self, path: str) -> None:
        """載入模型"""
        model_path = os.path.join(path, "model.pt")
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            unwrapped = self.model
            if hasattr(unwrapped, "_orig_mod"):
                unwrapped = unwrapped._orig_mod
            unwrapped.load_state_dict(state_dict)
            logger.info(f"[{self.__class__.__name__}] Model loaded from {path}")

        state_path = os.path.join(path, "trainer_state.pt")
        if os.path.exists(state_path):
            saved = torch.load(state_path, map_location=self.device)
            self.state.__dict__.update(saved.get("state", {}))

    def _is_lora_model(self, model: nn.Module) -> bool:
        """檢查是否為 LoRA 模型"""
        for name, _ in model.named_modules():
            if "lora" in name.lower():
                return True
        return False

    def _save_lora_adapter(self, model: nn.Module, path: str) -> None:
        """儲存 LoRA adapter 權重"""
        adapter_state = {}
        for name, param in model.named_parameters():
            if "lora" in name.lower() and param.requires_grad:
                adapter_state[name] = param.data
        torch.save(adapter_state, os.path.join(path, "adapter.pt"))
        logger.info(f"[{self.__class__.__name__}] LoRA adapter saved ({len(adapter_state)} params)")

    # ------------------------------------------------------------------
    # 訓推一致性檢查（Gate 3.0 驗收主鏈 A）
    # ------------------------------------------------------------------
    def check_train_inference_consistency(
        self,
        inference_engine: Any = None,
        sample_input: Optional[Dict[str, torch.Tensor]] = None,
        tolerance: float = 1e-5,
    ) -> Dict[str, Any]:
        """訓推權重一致性檢查（Gate 3.0 Dimension B）

        Args:
            inference_engine: 推理引擎（vLLM/SGLang/OMLX）
            sample_input: 樣本輸入
            tolerance: 容差

        Returns:
            一致性檢查結果
        """
        logger.info(f"[{self.__class__.__name__}] Checking train-inference weight consistency...")

        unwrapped = self.model
        if hasattr(unwrapped, "_orig_mod"):
            unwrapped = unwrapped._orig_mod

        # 收集訓練側權重
        train_weights: Dict[str, torch.Tensor] = {}
        for name, param in unwrapped.named_parameters():
            train_weights[name] = param.data.clone().float()

        if inference_engine is None:
            # 無推理引擎時，做自洽檢查（權重統計量）
            stats = {}
            for name, w in train_weights.items():
                stats[name] = {
                    "mean": w.mean().item(),
                    "std": w.std().item(),
                    "min": w.min().item(),
                    "max": w.max().item(),
                }
            return {
                "status": "self_consistent",
                "train_weight_count": len(train_weights),
                "weight_stats": stats,
                "note": "No inference engine provided; performed self-consistency check",
            }

        # 與推理引擎比對
        if sample_input is None:
            sample_input = next(iter(self._train_dataloader)) if self._train_dataloader else None

        if sample_input is None:
            return {"status": "skipped", "reason": "No sample input available"}

        # 訓練側前向
        self.model.eval()
        with torch.no_grad():
            train_output = self.model(**{k: v.to(self.device) for k, v in sample_input.items()})

        # 推理側前向
        try:
            infer_output = inference_engine.generate(**sample_input)
        except Exception as e:
            return {"status": "error", "reason": str(e)}

        # 比對
        if isinstance(train_output, torch.Tensor):
            train_logits = train_output
        elif isinstance(train_output, dict) and "logits" in train_output:
            train_logits = train_output["logits"]
        else:
            return {"status": "error", "reason": "Cannot extract logits from train output"}

        if isinstance(infer_output, torch.Tensor):
            infer_logits = infer_output
        elif isinstance(infer_output, dict) and "logits" in infer_output:
            infer_logits = infer_output["logits"]
        else:
            return {"status": "error", "reason": "Cannot extract logits from inference output"}

        max_diff = (train_logits.float() - infer_output.float()).abs().max().item()
        mean_diff = (train_logits.float() - infer_output.float()).abs().mean().item()

        passed = max_diff < tolerance
        return {
            "status": "pass" if passed else "fail",
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "tolerance": tolerance,
            "gate_pass": passed,
        }
