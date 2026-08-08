"""统一模型加载器 (支持 Qwen3-VL / Qwen3 / LLaMA / DeepSeek 等所有模型).

自动检测模型类型 → 选择正确的加载类 → 返回统一的模型接口.

用法:
    from app.shared.model_loader import load_base_model, get_embed_weight, get_lm_head_weight

    model, tokenizer = load_base_model("/data2/models/Qwen3-VL-2B-Instruct")
    embed = get_embed_weight(model)   # 统一获取 embed_tokens
    lm_head = get_lm_head_weight(model)  # 统一获取 lm_head
    text_model = get_text_model(model)   # 统一获取 text model (VL 模型的 language_model)
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Tuple, Any

# Use HF mirror if HuggingFace Hub is not directly accessible (e.g., China servers)
# This fixes FP8 kernel trust verification failures on hosts without HF access
if not os.environ.get("HF_ENDPOINT"):
    import urllib.request
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=3)
    except Exception:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        logging.getLogger(__name__).info("[model_loader] Using HF mirror: https://hf-mirror.com")

logger = logging.getLogger(__name__)


def detect_model_type(model_path: str) -> str:
    """检测模型类型."""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = getattr(config, "model_type", "").lower()
    architectures = getattr(config, "architectures", [])

    logger.info(f"[model_loader] model_type={model_type}, architectures={architectures}")
    return model_type


def _load_tokenizer(model_path: str):
    """Load tokenizer with fallback for transformers 5.x bus error.

    Tries AutoTokenizer first, falls back to tokenizers.Tokenizer (raw).
    """
    import os
    from transformers import AutoTokenizer

    # Try AutoTokenizer first
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # Quick sanity check
        _ = tokenizer.encode("test", add_special_tokens=False)
        return tokenizer
    except Exception as e:
        logger.warning(f"[model_loader] AutoTokenizer failed ({e}), trying tokenizers.Tokenizer")

    # Fallback: use tokenizers.Tokenizer directly (bypasses transformers 5.x bus error)
    from tokenizers import Tokenizer

    # Try standard tokenizer file locations
    for tk_file in ["tokenizer.json", "tokenizer.model"]:
        tk_path = os.path.join(model_path, tk_file)
        if os.path.exists(tk_path):
            try:
                raw_tok = Tokenizer.from_file(tk_path)
                logger.info(f"[model_loader] Loaded tokenizer from {tk_file}")
                return _TokenizerWrapper(raw_tok)
            except Exception as e2:
                logger.warning(f"[model_loader] tokenizers.Tokenizer from {tk_file} failed: {e2}")

    raise ImportError(f"Cannot load tokenizer from {model_path}")


class _TokenizerWrapper:
    """Wrapper to make tokenizers.Tokenizer compatible with transformers API."""

    def __init__(self, tokenizer):
        self._tok = tokenizer

    def encode(self, text, add_special_tokens=True, **kwargs):
        if isinstance(text, str):
            enc = self._tok.encode(text, add_special_tokens=add_special_tokens)
            return enc.ids if hasattr(enc, "ids") else list(enc)
        # Batch encode
        results = []
        for t in text:
            enc = self._tok.encode(t, add_special_tokens=add_special_tokens)
            results.append(enc.ids if hasattr(enc, "ids") else list(enc))
        return results

    def decode(self, ids, **kwargs):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self._tok.decode(ids)

    def __call__(self, text, **kwargs):
        """Compatibility for tokenizer(text) calls."""
        if isinstance(text, str):
            ids = self.encode(text, add_special_tokens=kwargs.get("add_special_tokens", True))
            import torch
            return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids))}
        # Batch
        all_ids = [self.encode(t, add_special_tokens=kwargs.get("add_special_tokens", True)) for t in text]
        import torch
        max_len = max(len(ids) for ids in all_ids)
        padded = [ids + [0] * (max_len - len(ids)) for ids in all_ids]
        masks = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in all_ids]
        return {"input_ids": torch.tensor(padded), "attention_mask": torch.tensor(masks)}


def load_base_model(
    model_path: str,
    device: str = "cuda",
    dtype: str = "bfloat16",
) -> Tuple[Any, Any]:
    """统一加载 base model + tokenizer.

    自动检测模型类型,选择正确的 transformers Auto 类:
      - VL 模型: AutoModelForImageTextToText (统一 VL 接口)
      - 纯文本: AutoModelForCausalLM (统一文本接口)
      - 兜底: AutoModel

    Returns:
        (model, tokenizer)
    """
    import torch
    from transformers import AutoConfig

    torch_dtype = getattr(torch, dtype, torch.bfloat16)
    tokenizer = _load_tokenizer(model_path)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = getattr(config, "model_type", "").lower()

    logger.info(f"[model_loader] Loading {model_path} (type={model_type})")

    # 尝试按优先级加载
    load_attempts = []

    # 统一用 Auto 类加载 (不硬编码模型特定类名)
    # VL 模型 → AutoModelForImageTextToText
    # 纯文本 → AutoModelForCausalLM
    # 兜底   → AutoModel
    if "vl" in model_type or "vision" in model_type or "image" in model_type or "gemma4" in model_type:
        load_attempts = [
            ("AutoModelForImageTextToText", "transformers"),  # 统一 VL/多模态接口
            ("AutoModelForCausalLM", "transformers"),  # Gemma4 text-only fallback
            ("AutoModel", "transformers"),  # 兜底
        ]
    else:
        load_attempts = [
            ("AutoModelForCausalLM", "transformers"),  # 统一文本接口
            ("AutoModel", "transformers"),  # 兜底
        ]

    # 尝试按优先级加载
    for cls_name, module_name in load_attempts:
        try:
            mod = __import__(module_name, fromlist=[cls_name])
            cls = getattr(mod, cls_name)
            if device == "auto":
                # device_map="auto" — accelerate 跨多 GPU 分片加载 (大模型如 DSV4)
                model = cls.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                # 优先使用 device_map (需要 accelerate), 失败则用 .to(device) fallback
                try:
                    model = cls.from_pretrained(
                        model_path,
                        torch_dtype=torch_dtype,
                        device_map=device,
                        trust_remote_code=True,
                    )
                except (ValueError, ImportError) as dev_err:
                    if "accelerate" in str(dev_err).lower() or "device_map" in str(dev_err).lower():
                        # accelerate 未安装, fallback: CPU 加载后 .to(device)
                        logger.info(f"[model_loader] device_map failed (no accelerate?), falling back to .to({device})")
                        model = cls.from_pretrained(
                            model_path,
                            torch_dtype=torch_dtype,
                            trust_remote_code=True,
                        )
                        model = model.to(device)
                    else:
                        raise dev_err
            logger.info(f"[model_loader] Loaded as {cls_name}")
            return model, tokenizer
        except (ImportError, AttributeError, ValueError, Exception) as e:
            logger.debug(f"[model_loader] {cls_name} failed: {e}")
            continue

    raise ImportError(f"Cannot load model {model_path} (type={model_type}) with any known class")


def get_text_model(model: Any) -> Any:
    """统一获取 text model (用于 forward).

    VL 模型: model.language_model (或 model.model)
    Gemma4: model.text_model
    纯文本: model (本身就是 text model)
    """
    # 尝试常见路径 (优先 text_model for Gemma4, 然后 language_model for VL)
    for attr in ["text_model", "language_model", "model"]:
        text_model = getattr(model, attr, None)
        if text_model is not None and hasattr(text_model, "forward"):
            # 确认是 text model (有 layers)
            if hasattr(text_model, "layers") or hasattr(getattr(text_model, "model", None), "layers"):
                logger.info(f"[model_loader] text_model at model.{attr}")
                return text_model

    # model 本身就是 text model
    logger.info("[model_loader] model itself is text model")
    return model


def get_embed_weight(model: Any) -> Optional[Any]:
    """统一获取 embed_tokens 权重.

    搜索路径:
      VL: model.language_model.model.embed_tokens
      纯文本: model.model.embed_tokens
      其他: model.embed_tokens
    """
    search_paths = [
        ("language_model", "model", "embed_tokens"),  # Qwen3-VL (旧路径)
        ("model", "language_model", "embed_tokens"),  # Qwen3-VL (新路径 transformers 5.8+)
        ("text_model", "embed_tokens"),  # Gemma4 (Gemma4ForConditionalGeneration.text_model)
        ("model", "text_model", "embed_tokens"),  # Gemma4 嵌套
        ("model", "embed_tokens"),  # Qwen3 / LLaMA
        ("model", "model", "embed_tokens"),  # 嵌套
        ("language_model", "embed_tokens"),  # VL 直接
        ("model", "embed"),  # DSV4 custom (model.model.embed)
        ("embed",),  # DSV4 top level
        ("embed_tokens",),  # 顶层
        ("transformer", "wte"),  # GPT-2
    ]

    for attr_path in search_paths:
        try:
            obj = model
            for attr in attr_path:
                obj = getattr(obj, attr)
            if hasattr(obj, "weight"):
                logger.info(f"[model_loader] embed_tokens at {'.'.join(attr_path)}")
                return obj.weight
        except AttributeError:
            continue

    logger.error("[model_loader] Cannot find embed_tokens")
    return None


def get_lm_head_weight(model: Any) -> Optional[Any]:
    """统一获取 lm_head 权重."""
    search_paths = [
        ("language_model", "lm_head"),  # Qwen3-VL
        ("text_model", "lm_head"),  # Gemma4
        ("lm_head",),  # Qwen3 / LLaMA
        ("model", "lm_head"),  # 嵌套
        ("model", "text_model", "lm_head"),  # Gemma4 嵌套
        ("model", "head"),  # DSV4 custom (model.head)
        ("head",),  # DSV4 top level
        ("output",),  # 某些模型
    ]

    for attr_path in search_paths:
        try:
            obj = model
            for attr in attr_path:
                obj = getattr(obj, attr)
            if hasattr(obj, "weight"):
                logger.info(f"[model_loader] lm_head at {'.'.join(attr_path)}")
                return obj.weight
        except AttributeError:
            continue

    logger.error("[model_loader] Cannot find lm_head")
    return None


def get_layers(model: Any) -> Optional[list]:
    """统一获取 transformer layers 列表."""
    text_model = get_text_model(model)

    # 搜索 layers
    for attr_path in [("layers",), ("model", "layers"), ("h",), ("transformer", "h")]:
        try:
            obj = text_model
            for attr in attr_path:
                obj = getattr(obj, attr)
            if isinstance(obj, (list, tuple)) or hasattr(obj, "__len__"):
                logger.info(f"[model_loader] layers at {'.'.join(attr_path)} (count={len(obj)})")
                return list(obj)
        except AttributeError:
            continue

    logger.error("[model_loader] Cannot find layers")
    return None


def get_model_info(model_path: str) -> dict:
    """获取模型信息 (不加载模型,只读 config)."""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # 提取 text config (VL 模型有 text_config)
    text_config = getattr(config, "text_config", config)

    return {
        "model_type": getattr(config, "model_type", ""),
        "architectures": getattr(config, "architectures", []),
        "hidden_size": getattr(text_config, "hidden_size", 0),
        "num_layers": getattr(text_config, "num_hidden_layers", 0),
        "vocab_size": getattr(text_config, "vocab_size", 0),
        "num_heads": getattr(text_config, "num_attention_heads", 0),
        "is_vl": "vl" in getattr(config, "model_type", "").lower(),
        "is_moe": "moe" in getattr(config, "model_type", "").lower(),
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/Users/alexchuang/Documents/flashkv0516")

    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("统一模型加载器测试")
    print("=" * 60)

    # 测试获取模型信息 (不加载模型)
    for path in [
        "/Users/alexchuang/models/Qwen3-VL-2B-bf16",
        "/Users/alexchuang/models/Qwen3-VL-30B-A3B-4bit",
    ]:
        import os
        if not os.path.exists(path):
            print(f"\n{path}: 跳过 (不存在)")
            continue

        print(f"\n{path}:")
        info = get_model_info(path)
        for k, v in info.items():
            print(f"  {k}: {v}")
