# Copyright (c) 2025 SandAI. All Rights Reserved.
"""
CGC 策略注入 - Python ↔ C++ 橋接

將 HarnessCompileStrategy 注入到 C++ SIMD Engine
"""

import os
import sys
import logging
import importlib.util
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class CGCStrategyNative:
    """對應 C++ cgc_strategy_t 結構"""
    def __init__(self):
        self.backend: int = 0
        self.tile_m: int = 128
        self.tile_n: int = 128
        self.tile_k: int = 128
        self.attn_block: int = 128
        self.moe_block: int = 128
        self.enable_op_fusion: bool = True
        self.quantization_mode: int = 0
        self.tp_degree: int = 1
        self.pp_degree: int = 1
        self.num_op_hints: int = 0
        self.op_hints: list = [0] * 16
        self.fusion_regions: bytes = b""
        self.metadata: bytes = b""


class CGCStrategyInjector:
    """CGC 策略注入器"""

    _instance = None
    _lib = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if CGCStrategyInjector._initialized:
            return
        self._try_load_library()
        CGCStrategyInjector._initialized = True

    def _try_load_library(self):
        """嘗試加載 CGC C++ 庫 (pybind11 module)"""
        possible_paths = [
            "/Users/alexchuang/Documents/flashkv0430/MagiCompiler-main/cgc_engine/cgc/cgc_cpp/build/cgc_cpp.so",
            "/Users/alexchuang/Documents/flashkv0430/MagiCompiler-main/cgc_engine/cgc/cgc_cpp/build/libcgc_cpp.dylib",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                try:
                    spec = importlib.util.spec_from_file_location("cgc_cpp", path)
                    if spec and spec.loader:
                        CGCStrategyInjector._lib = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(CGCStrategyInjector._lib)
                        CGCStrategyInjector._lib.init()
                        logger.info(f"[CGCStrategyInjector] Loaded CGC pybind11 module from {path}")
                        return
                except Exception as e:
                    logger.warning(f"[CGCStrategyInjector] Failed to load pybind11 module {path}: {e}")

        try:
            import cgc_cpp
            CGCStrategyInjector._lib = cgc_cpp
            cgc_cpp.init()
            logger.info("[CGCStrategyInjector] Loaded CGC pybind11 module via import")
            return
        except ImportError:
            pass

        logger.warning(
            "[CGCStrategyInjector] CGC C++ library not found. "
            "Strategy injection will be simulated only."
        )
        CGCStrategyInjector._lib = None

    def inject_strategy(self, strategy: Any) -> bool:
        """
        注入策略到 C++ Engine

        Args:
            strategy: HarnessCompileStrategy 對象

        Returns:
            是否成功
        """
        # STRICT MODE: 禁止 CPU backend
        if strategy.backend == "cpu":
            raise RuntimeError("STRICT MODE: CPU backend is strictly PROHIBITED. Backend must be CUDA (or Metal on Mac).")

        # STRICT MODE: 禁止 simulated
        if CGCStrategyInjector._lib is None:
            raise RuntimeError("STRICT MODE: C++ library not loaded! Simulated strategy injection is strictly PROHIBITED.")

        try:
            if hasattr(CGCStrategyInjector._lib, 'inject_strategy'):
                native = self._python_strategy_to_native(strategy)
                result = CGCStrategyInjector._lib.inject_strategy(
                    native.backend,
                    native.tile_m,
                    native.tile_n,
                    native.tile_k,
                    native.attn_block,
                    native.enable_op_fusion
                )
                return result == 0
            
            raise RuntimeError("STRICT MODE: C++ library missing inject_strategy! Simulated injection is strictly PROHIBITED.")
        except Exception as e:
            logger.error(f"[CGCStrategyInjector] Failed to inject strategy: {e}")
            raise e

    def get_strategy(self) -> Optional[Dict[str, Any]]:
        """獲取當前策略"""
        if CGCStrategyInjector._lib is None:
            return None

        try:
            if hasattr(CGCStrategyInjector._lib, 'get_strategy'):
                backend_id, tile_m, tile_n, tile_k, attn_block, enable_fusion = \
                    CGCStrategyInjector._lib.get_strategy()
                return {
                    "backend": self._backend_id_to_name(backend_id),
                    "tile_sizes": {"M": tile_m, "N": tile_n, "K": tile_k, "ATTN_BLOCK": attn_block},
                    "enable_op_fusion": enable_fusion,
                }
            return None
        except Exception as e:
            logger.error(f"[CGCStrategyInjector] Failed to get strategy: {e}")
            return None

    def reset_strategy(self) -> bool:
        """重置策略"""
        if CGCStrategyInjector._lib is None:
            return True

        try:
            if hasattr(CGCStrategyInjector._lib, 'reset_strategy'):
                result = CGCStrategyInjector._lib.reset_strategy()
                return result == 0
            return True
        except Exception as e:
            logger.error(f"[CGCStrategyInjector] Failed to reset strategy: {e}")
            return False

    def set_backend(self, backend: str) -> bool:
        """設置後端"""
        backend_map = {
            "auto": 0,
            "cpu": 1,
            "cuda": 2,
            "metal": 3,
        }
        backend_id = backend_map.get(backend.lower(), 0)

        if CGCStrategyInjector._lib is None:
            logger.info(f"[CGCStrategyInjector] Simulating backend set to {backend}")
            return True

        try:
            return CGCStrategyInjector._lib.cgc_set_backend(backend_id)
        except Exception as e:
            logger.error(f"[CGCStrategyInjector] Failed to set backend: {e}")
            return False

    def _python_strategy_to_native(self, strategy: Any) -> CGCStrategyNative:
        """Python 策略轉換為 C 結構"""
        from cgc_engine.agent.harness_agent import HarnessCompileStrategy, AgentOpHint

        native = CGCStrategyNative()
        native.backend = self._backend_name_to_id(strategy.backend)

        tiling = strategy.tile_sizes or {}
        native.tile_m = int(tiling.get("M", 128))
        native.tile_n = int(tiling.get("N", 128))
        native.tile_k = int(tiling.get("K", 128))
        native.attn_block = int(tiling.get("ATTN_BLOCK", 128))
        native.moe_block = int(tiling.get("MOE_BLOCK", 128))

        native.enable_op_fusion = bool(strategy.enable_op_fusion)
        q_mode = getattr(strategy, 'quantization_mode', "auto")
        native.quantization_mode = self._quant_mode_to_id(q_mode) if isinstance(q_mode, str) else int(q_mode or 0)
        native.tp_degree = int(getattr(strategy, 'tp_degree', 1) or 1)
        native.pp_degree = int(getattr(strategy, 'pp_degree', 1) or 1)
        native.num_op_hints = int(len(strategy.op_hints))

        native.op_hints = [self._op_hint_to_id(hint) for hint in strategy.op_hints[:16]]
        while len(native.op_hints) < 16:
            native.op_hints.append(0)

        fusion_regions = strategy.fusion_regions or []
        fusion_str = ",".join([",".join(region) for region in fusion_regions])
        native.fusion_regions = fusion_str.encode("utf-8")[:255]

        metadata_str = str(strategy.metadata)[:511]
        native.metadata = metadata_str.encode("utf-8")[:511]

        return native

    def _native_strategy_to_python(self, native: CGCStrategyNative) -> Dict[str, Any]:
        """C 結構轉換為 Python 字典"""
        from cgc_engine.agent.harness_agent import AgentOpHint

        return {
            "backend": self._backend_id_to_name(native.backend),
            "tile_sizes": {
                "M": native.tile_m,
                "N": native.tile_n,
                "K": native.tile_k,
                "ATTN_BLOCK": native.attn_block,
                "MOE_BLOCK": native.moe_block,
            },
            "enable_op_fusion": native.enable_op_fusion,
            "quantization_mode": self._quant_id_to_mode(native.quantization_mode),
            "tp_degree": native.tp_degree,
            "pp_degree": native.pp_degree,
            "num_op_hints": native.num_op_hints,
            "op_hints": [self._op_id_to_hint(native.op_hints[i]) for i in range(native.num_op_hints)],
        }

    @staticmethod
    def _backend_name_to_id(name: str) -> int:
        mapping = {"auto": 0, "cpu": 1, "cuda": 2, "metal": 3}
        return mapping.get(name.lower(), 0)

    @staticmethod
    def _backend_id_to_name(id: int) -> str:
        mapping = {0: "auto", 1: "cpu", 2: "cuda", 3: "metal"}
        return mapping.get(id, "auto")

    @staticmethod
    def _quant_mode_to_id(mode: str) -> int:
        mapping = {"auto": 0, "none": 0, "int8": 1, "fp16": 2, "bf16": 3, "fp8": 4, "int4": 5}
        return mapping.get(mode.lower(), 0)

    @staticmethod
    def _quant_id_to_mode(id: int) -> str:
        mapping = {0: "auto", 1: "int8", 2: "fp16", 3: "bf16", 4: "fp8", 5: "int4"}
        return mapping.get(id, "auto")

    @staticmethod
    def _op_hint_to_id(hint) -> int:
        from cgc_engine.agent.harness_agent import AgentOpHint
        mapping = {
            AgentOpHint.AUTO: 0,
            AgentOpHint.FLASH_ATTENTION: 1,
            AgentOpHint.MOE_ROUTING: 2,
            AgentOpHint.TENSOR_PARALLEL: 3,
            AgentOpHint.VLM_CROSS_ATTENTION: 4,
        }
        if isinstance(hint, str):
            for k, v in mapping.items():
                if k.value == hint:
                    return v
            return 0
        return mapping.get(hint, 0)

    @staticmethod
    def _op_id_to_hint(id: int):
        from cgc_engine.agent.harness_agent import AgentOpHint
        mapping = {
            0: AgentOpHint.AUTO,
            1: AgentOpHint.FLASH_ATTENTION,
            2: AgentOpHint.MOE_ROUTING,
            3: AgentOpHint.TENSOR_PARALLEL,
            4: AgentOpHint.VLM_CROSS_ATTENTION,
        }
        return mapping.get(id, AgentOpHint.AUTO)


_injector = CGCStrategyInjector()


def inject_strategy(strategy: Any) -> bool:
    """便捷函數：注入策略"""
    return _injector.inject_strategy(strategy)


def set_backend(backend: str) -> bool:
    """便捷函數：設置後端"""
    return _injector.set_backend(backend)


def reset_strategy() -> bool:
    """便捷函數：重置策略"""
    return _injector.reset_strategy()
