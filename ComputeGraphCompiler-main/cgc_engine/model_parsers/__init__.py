
# Copyright (c) 2025 SandAI. All Rights Reserved.
"""
Model parsers 模块 - 解析各种模型格式
"""

from .base_parser import BaseModelParser, ParsedModel, ParsedWeight
from .parsed_model_adapter import parsed_model_to_pytorch, AdapterLLM

GGUFParser = None
HFParser = None
VLLMParser = None

try:
    from .gguf_parser import GGUFParser
except Exception:
    pass

try:
    from .hf_parser import HuggingFaceParser
    HFParser = HuggingFaceParser
except Exception:
    pass

try:
    from .vllm_parser import VLLMParser
except Exception:
    pass

__all__ = [
    "BaseModelParser",
    "ParsedModel",
    "ParsedWeight",
    "parsed_model_to_pytorch",
    "AdapterLLM",
]

if GGUFParser is not None:
    __all__.append("GGUFParser")
if HFParser is not None:
    __all__.append("HFParser")
if VLLMParser is not None:
    __all__.append("VLLMParser")
