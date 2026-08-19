# Copyright (c) 2025 SandAI. All Rights Reserved.
"""
硬件约束定义

定义不同硬件平台的约束信息
"""

from dataclasses import dataclass


@dataclass
class HardwareConstraints:
    """硬件约束信息"""
    device: str = "metal"
    simd_width: int = 128
    l1_cache: int = 32 * 1024
    l2_cache: int = 128 * 1024
    max_tile_m: int = 128
    max_tile_n: int = 128
    max_tile_k: int = 128
    metal_simd_group_size: int = 32

    def __post_init__(self):
        device_str = str(self.device).lower()
        if device_str in ["metal", "mps"]:
            self._setup_metal_constraints()
        elif device_str.startswith("cuda"):
            self._setup_cuda_constraints()
        elif device_str == "cpu":
            self._setup_cpu_constraints()
        else:
            import platform
            import torch
            if platform.system().lower() == "linux" and torch.cuda.is_available():
                self._setup_cuda_constraints()
            else:
                self._setup_cpu_constraints()

    def _setup_metal_constraints(self):
        """Metal 硬件约束"""
        self.simd_width = 128
        self.l1_cache = 32 * 1024
        self.l2_cache = 128 * 1024
        self.max_tile_m = 32
        self.max_tile_n = 32
        self.max_tile_k = 32
        self.metal_simd_group_size = 32

    def _setup_cuda_constraints(self):
        """CUDA 硬件约束"""
        self.simd_width = 128
        self.l1_cache = 128 * 1024
        self.l2_cache = 4 * 1024 * 1024
        self.max_tile_m = 128
        self.max_tile_n = 128
        self.max_tile_k = 128
        self.metal_simd_group_size = 32

    def _setup_cpu_constraints(self):
        """CPU 硬件约束"""
        self.simd_width = 64
        self.l1_cache = 32 * 1024
        self.l2_cache = 256 * 1024
        self.max_tile_m = 64
        self.max_tile_n = 64
        self.max_tile_k = 64
        self.metal_simd_group_size = 32
