#!/usr/bin/env python3
"""Hermes 认知路由 v4 — 平台感知 + SFT 微调.

包含:
  1. SystemProfile      — 硬件检测 + 能力评估
  2. PlatformBenchmark  — MLX vs llama.cpp 平台速度检测, 选快的
  3. ProfileBinding     — 模型需求 vs 硬件能力 + 后端匹配
  4. StateABI           — 状态管理 + 转换协议
  5. Bootstrap          — 初始化序列 (含平台检测)
  6. FourDMatrix        — D1网络 + D2硬件 + D3模型 + D4决策 (保留)
  7. TenStepPipeline    — 十步 4D 感知流水线 (保留)
  8. HermesRouter       — 路由决策核心 (edge_draft/cloud_mtp 都用 Hermes 调动 draft model)
  9. SFTDataGenerator   — 用 4D矩阵+十步流水线数据生成 SFT 训练对

路由决策逻辑:
  1. PlatformBenchmark: 测 MLX vs llama.cpp 哪个快 → preferred_edge_backend
  2. Drafter体积 + 架构能否在端侧跑? → 自动选择量化等级 (BF16/Q4/Q2)
  3. 能跑 → edge_draft: Hermes 用快的后端调 draft model, draft 在端侧, verify 在云端
  4. 跑不动 → cloud_mtp: Hermes 调云端 sglang NEXTN MTP (draft+verify 都在云端)
  5. MTP 不可用 → cloud_only
  6. 离线 → local_only

示例:
  Gemma4 26B MTP drafter (gemma4_assistant arch, BF16≈8.2GB / Q4≈4.1GB / Q2≈2.1GB):
    → M4 16GB 可跑 Q4 (4.1GB < 9.6GB 可用) → edge_draft via llama.cpp (省云端算力)
    → M4 8GB 可跑 Q2 (2.1GB < 4.8GB 可用) → edge_draft via llama.cpp
  DSV4 Flash drafter (deepseek_v4 arch, 需 GPU):
    → Mac 不支持 → cloud_mtp (云端 NEXTN MTP)
  Qwen3-VL 2B MTP head (MTPHead arch, 59.8M, BF16≈0.3GB):
    → llama.cpp 快 (149 vs 26 tok/s) → edge_draft via llama.cpp (省云端算力)
"""
from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from types import SimpleNamespace
from typing import Any, Literal, Optional

from app.shared.colibri_backend import build_unified_runtime_ir_v0, is_unified_runtime_ir_v0
from app.shared import route_decision_v2 as route_decision_v2_api

logger = logging.getLogger(__name__)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ============================================================================
# 1. SystemProfile — 硬件检测 + 能力评估
# ============================================================================
@dataclass
class SystemProfile:
    """系统画像 — 端侧硬件能力评估.

    决定端侧能跑多大的模型、支持哪些架构。
    """
    # OS
    os_name: str = ""
    os_version: str = ""
    arch: str = ""

    # CPU
    cpu_brand: str = ""
    cpu_cores: int = 0

    # 内存 (GB)
    total_mem_gb: float = 0.0
    avail_mem_gb: float = 0.0

    # 磁盘 (GB)
    disk_total_gb: float = 0.0
    disk_available_gb: float = 0.0

    # GPU
    gpu_type: str = ""          # apple_metal / nvidia / amd / none
    gpu_name: str = ""
    gpu_vram_gb: float = 0.0

    # 算力
    compute_tier: str = ""      # weak / medium / strong / ultra
    tflops: float = 0.0

    # 推理引擎
    recommended_engine: str = ""  # mlx / cuda / cpu / cloud
    preferred_edge_backend: str = ""  # mlx / llamacpp / none (PlatformBenchmark 结果)
    mlx_available: bool = False
    llamacpp_available: bool = False
    mlx_benchmark_tps: float = 0.0  # MLX 实测 tok/s
    llamacpp_benchmark_tps: float = 0.0  # llama.cpp 实测 tok/s

    # 网络
    rtt_ms: float = 0.0
    bandwidth_mbps: float = 0.0
    online: bool = True

    # NVMe / 存储 (P1: colibri streamed-MoE 硬件感知)
    # 启发式估算 (避免启动时跑 fio/dd 拖慢), 可用 CGC_NVME_BW_GBPS 覆盖
    nvme_bw_gbps: float = 0.0          # 主 NVMe 顺序读带宽 (GB/s)
    nvme_device_count: int = 0          # NVMe 设备数 (含主盘)
    secondary_nvme_available: bool = False  # 是否有第二块 NVMe (用于 dual-store)
    nvme_paths: list[str] = None        # 检测到的 NVMe 设备路径

    # === 派生属性 ===
    @property
    def max_local_model_gb(self) -> float:
        """端侧可运行的最大模型大小 (GB).

        保守估计: 可用内存的 60% (留给系统 + KV cache).
        Apple Silicon UMA: 可用 unified memory.
        """
        return self.avail_mem_gb * 0.6

    @property
    def supported_architectures(self) -> set[str]:
        """端侧支持的模型架构集合.

        MLX (Apple Silicon) 支持的标准架构:
          llama, qwen2, qwen3, mistral, gemma, gemma2, phi3, etc.
        不支持的自定义架构:
          gemma4_assistant, deepseek_v4 (需 trust_remote_code + GPU)
        """
        if self.gpu_type == "apple_metal":
            return {
                "llama", "qwen2", "qwen3", "qwen3vl", "mistral", "mixtral",
                "gemma", "gemma2", "gemma4", "gemma4_assistant", "gemma4assistant",
                "phi3", "phi3small",
                "starcoder2", "cohere", "mtphead",
            }
        elif self.gpu_type == "nvidia":
            return {
                "llama", "qwen2", "qwen3", "qwen3vl", "mistral", "mixtral",
                "gemma", "gemma2", "phi3", "deepseek_v4",
                "gemma4", "gemma4_assistant", "mtphead",
            }
        return set()

    @property
    def can_run_mlx(self) -> bool:
        return self.gpu_type == "apple_metal" and self.arch == "arm64"

    @property
    def can_run_cuda(self) -> bool:
        return self.gpu_type == "nvidia"

    def can_run_architecture(self, arch: str) -> bool:
        """检查端侧是否支持给定模型架构.

        匹配策略:
          1. 精确匹配 (如 mtphead, qwen3vl)
          2. 去后缀匹配 (ForCausalLM, ForConditionalGeneration)
          3. 已知别名 (gemma4_assistant ≠ gemma, deepseek_v4 ≠ deepseek)
        """
        # 已知不支持: 自定义架构需要 GPU + 特定 transformers 版本
        # NOTE: gemma4_assistant 已确认可在端侧运行 (BF16≈8.2GB, Q4≈4.1GB, Q2≈2.1GB)
        #       MLX/llama.cpp 均支持, 通过量化适配不同内存规格
        KNOWN_UNSUPPORTED_ON_EDGE = {
            "deepseek_v4", "deepseekv4",            # DeepSeek MLA, 需 GPU + trust_remote_code
        }
        arch_lower = arch.lower()

        # 检查已知不支持
        for unsupported in KNOWN_UNSUPPORTED_ON_EDGE:
            if arch_lower == unsupported or arch_lower.startswith(unsupported):
                return False

        # 精确匹配
        if arch_lower in self.supported_architectures:
            return True

        # 去后缀
        for suffix in ["forcausallm", "forconditiongeneration", "forconditionalgeneration"]:
            if arch_lower.endswith(suffix):
                base = arch_lower[:-len(suffix)]
                if base in self.supported_architectures:
                    return True

        return False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def detect(cls, cloud_host: str = "") -> "SystemProfile":
        """检测系统硬件并创建画像."""
        profile = cls()
        profile.os_name = platform.system()
        profile.os_version = platform.release()
        profile.arch = platform.machine()

        # CPU
        if profile.os_name == "Darwin":
            profile.cpu_brand = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        else:
            profile.cpu_brand = platform.processor()
        profile.cpu_cores = os.cpu_count() or 1

        # 内存
        if profile.os_name == "Darwin":
            result = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5)
            lines = result.stdout.strip().split("\n")
            page_size = 4096
            for line in lines:
                if "page size" in line.lower():
                    import re
                    m = re.search(r"page size of (\d+)", line)
                    if m:
                        page_size = int(m.group(1))
            free_pages = 0
            for line in lines:
                if "free" in line.lower() and ":" in line:
                    val = line.split(":")[-1].strip().rstrip(".")
                    free_pages = int(val)
                    break
            # Total memory
            mem_result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5
            )
            profile.total_mem_gb = int(mem_result.stdout.strip()) / 1e9
            profile.avail_mem_gb = (free_pages * page_size) / 1e9
            # Fallback if avail seems wrong
            if profile.avail_mem_gb < 1:
                profile.avail_mem_gb = profile.total_mem_gb * 0.4

        # GPU
        if profile.os_name == "Darwin" and profile.arch == "arm64":
            profile.gpu_type = "apple_metal"
            profile.gpu_name = profile.cpu_brand
            # Unified memory: GPU shares system memory
            profile.gpu_vram_gb = 0  # UMA, no dedicated VRAM
        elif profile.os_name == "Linux":
            try:
                lspci = subprocess.run(
                    ["lspci"], capture_output=True, text=True, timeout=5
                )
                if "NVIDIA" in lspci.stdout:
                    profile.gpu_type = "nvidia"
                    profile.gpu_name = "NVIDIA GPU"
            except:
                pass

        # Compute tier
        if profile.gpu_type == "apple_metal":
            if "M4" in profile.cpu_brand or "M3" in profile.cpu_brand:
                profile.compute_tier = "medium"
                profile.tflops = 38.0  # M4 GPU ~38 TFLOPS fp16
            elif "M2" in profile.cpu_brand:
                profile.compute_tier = "medium"
                profile.tflops = 30.0
            else:
                profile.compute_tier = "weak"
                profile.tflops = 15.0
        elif profile.gpu_type == "nvidia":
            if "4090" in profile.gpu_name or "A100" in profile.gpu_name:
                profile.compute_tier = "ultra"
                profile.tflops = 80.0
            else:
                profile.compute_tier = "strong"
                profile.tflops = 50.0
        else:
            profile.compute_tier = "weak"
            profile.tflops = 5.0

        # Engine
        if profile.gpu_type == "apple_metal":
            profile.recommended_engine = "mlx"
        elif profile.gpu_type == "nvidia":
            profile.recommended_engine = "cuda"
        else:
            profile.recommended_engine = "cloud"

        # Disk
        try:
            stat = os.statvfs("/")
            profile.disk_total_gb = stat.f_blocks * stat.f_frsize / 1e9
            profile.disk_available_gb = stat.f_bavail * stat.f_frsize / 1e9
        except:
            pass

        # Network (best-effort ping)
        if cloud_host:
            try:
                result = subprocess.run(
                    ["ping", "-c", "3", "-W", "2", cloud_host],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0:
                    import re
                    times = re.findall(r"time=([\d.]+)", result.stdout)
                    if times:
                        profile.rtt_ms = sum(float(t) for t in times) / len(times)
                else:
                    profile.online = False
            except:
                pass

        # NVMe / 存储探测 (P1: colibri streamed-MoE 硬件感知)
        profile._detect_nvme()

        return profile

    def _detect_nvme(self) -> None:
        """探测 NVMe 设备与带宽 (启发式, 不跑 fio/dd).

        策略:
          1. 环境变量 CGC_NVME_BW_GBPS 覆盖 (优先级最高)
          2. macOS: diskutil list + system_profiler SPNVMeDataType
             - 检测 NVMe 设备数, 按型号估算带宽
          3. Linux: lsblk -d -o NAME,TYPE / /sys/block/nvme*
             - 检测 nvme 设备数, 按 PCIe gen 估算
          4. 兜底: 按 disk_total_gb 估算 (SSD>=256G 假设 NVMe)

        带宽估算表 (顺序读, GB/s):
          PCIe 4.0 NVMe: 6-7 GB/s
          PCIe 3.0 NVMe: 3-4 GB/s
          Apple Silicon: 3-5 GB/s (UMA, 无独立 NVMe 报告)
          SATA SSD: 0.5 GB/s
          HDD: 0.1 GB/s
        """
        import re as _re

        # 1. 环境变量覆盖
        env_bw = os.environ.get("CGC_NVME_BW_GBPS", "").strip()
        if env_bw:
            try:
                self.nvme_bw_gbps = float(env_bw)
                self.nvme_device_count = max(int(os.environ.get("CGC_NVME_DEVICE_COUNT", "1") or "1"), 1)
                self.secondary_nvme_available = self.nvme_device_count >= 2
                self.nvme_paths = [f"/dev/nvme{i}" for i in range(self.nvme_device_count)]
                return
            except ValueError:
                pass

        nvme_paths: list[str] = []

        if self.os_name == "Darwin":
            # macOS: system_profiler SPNVMeDataType (较慢, 2-3s)
            # 用 diskutil list 快速检测物理盘数量
            try:
                diskutil = subprocess.run(
                    ["diskutil", "list"], capture_output=True, text=True, timeout=5
                )
                # /dev/disk0, /dev/disk1 ... physical
                physical_disks = _re.findall(r"^/dev/(disk\d+)", diskutil.stdout, _re.MULTILINE)
                # 去重 (一个物理盘可能有多个分区)
                physical_disks = sorted(set(physical_disks))
                self.nvme_device_count = len(physical_disks)
                nvme_paths = [f"/dev/{d}" for d in physical_disks]
            except Exception:
                pass

            # 带宽估算: Apple Silicon 内置 NVMe (M3/M4 ~3.5-5 GB/s)
            if self.gpu_type == "apple_metal":
                if "M4" in self.cpu_brand or "M3" in self.cpu_brand:
                    self.nvme_bw_gbps = 5.0
                elif "M2" in self.cpu_brand:
                    self.nvme_bw_gbps = 3.5
                else:
                    self.nvme_bw_gbps = 2.0
            else:
                # Intel Mac with NVMe
                self.nvme_bw_gbps = 3.0

        elif self.os_name == "Linux":
            # Linux: /sys/block/nvme* 存在即为 NVMe
            try:
                import glob
                nvme_devs = glob.glob("/sys/block/nvme*")
                self.nvme_device_count = len(nvme_devs)
                nvme_paths = [f"/dev/{os.path.basename(d)}" for d in nvme_devs]

                # 读 PCIe link speed 估算带宽
                for dev_path in nvme_devs[:1]:  # 只看第一个
                    try:
                        link_path = os.path.join(dev_path, "device", "current_link_speed")
                        if os.path.exists(link_path):
                            with open(link_path) as f:
                                link_str = f.read().strip()
                            # "16 GT/s" → PCIe 4.0 x4 ≈ 6.4 GB/s
                            gt = float(_re.search(r"([\d.]+)", link_str).group(1))
                            # GT/s × 4 lanes × 0.98 (encoding) / 8 (bits→bytes)
                            self.nvme_bw_gbps = round(gt * 4 * 0.98 / 8, 1)
                    except Exception:
                        pass
            except Exception:
                pass

            if self.nvme_bw_gbps <= 0:
                # 兜底: 有 nvme 设备默认 PCIe 3.0
                self.nvme_bw_gbps = 3.5 if self.nvme_device_count > 0 else 0.5

        else:
            # Windows / 其他: 简单兜底
            self.nvme_bw_gbps = 2.0
            self.nvme_device_count = 1

        self.secondary_nvme_available = self.nvme_device_count >= 2
        self.nvme_paths = nvme_paths or ["/dev/nvme0"]


# ============================================================================
# 1.5 PlatformBenchmark — MLX vs llama.cpp 平台速度检测
# ============================================================================
@dataclass
class PlatformBenchmarkResult:
    """平台基准测试结果."""
    mlx_available: bool = False
    llamacpp_available: bool = False
    mlx_tps: float = 0.0          # MLX 实测 decode tok/s
    llamacpp_tps: float = 0.0     # llama.cpp 实测 decode tok/s
    mlx_latency_ms: float = 0.0   # MLX 5-token 生成延迟
    llamacpp_latency_ms: float = 0.0  # llama.cpp 5-token 生成延迟
    preferred_backend: str = "none"   # mlx / llamacpp / none
    speedup: float = 1.0          # 快的后端 vs 慢的后端
    test_model: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class PlatformBenchmark:
    """MLX vs llama.cpp 平台速度检测 — 选择更快的端侧推理后端.

    测试方法:
      1. 检查 MLX 是否可导入 → 用 mlx_lm 做 5-token 生成计时
      2. 检查 llama-server 是否在 PATH → 启动子进程做 5-token 生成计时
      3. 比较 tok/s, 选择更快的作为 preferred_edge_backend

    用法:
        benchmark = PlatformBenchmark()
        result = benchmark.run()
        print(f"快的后端: {result.preferred_backend} ({max(result.mlx_tps, result.llamacpp_tps):.1f} tok/s)")
    """

    # 已知平台性能基线 (用于无模型可用时的估算)
    KNOWN_PERFORMANCE = {
        # (chip_prefix, backend) → (tps_estimate, latency_ms_estimate)
        ("Apple M4", "mlx"): (26.0, 192.0),        # MLX 2B bf16
        ("Apple M4", "llamacpp"): (149.8, 33.0),   # llama.cpp 1B Q4_K_M
        ("Apple M4 Pro", "mlx"): (35.0, 143.0),
        ("Apple M4 Pro", "llamacpp"): (180.0, 28.0),
        ("Apple M4 Max", "mlx"): (60.0, 83.0),
        ("Apple M4 Max", "llamacpp"): (250.0, 20.0),
        ("Apple M3", "mlx"): (22.0, 227.0),
        ("Apple M3", "llamacpp"): (130.0, 38.0),
        ("Apple M2", "mlx"): (18.0, 278.0),
        ("Apple M2", "llamacpp"): (110.0, 45.0),
    }

    def __init__(self, test_prompt: str = "def fibonacci(n):\n    return"):
        self.test_prompt = test_prompt

    def _check_mlx(self) -> bool:
        """检查 MLX 是否可用."""
        try:
            import mlx.core as mx
            import mlx_lm
            return True
        except ImportError:
            return False

    def _check_llamacpp(self) -> bool:
        """检查 llama-server 是否在 PATH."""
        return shutil.which("llama-server") is not None

    def _benchmark_mlx(self, model_path: str = "") -> tuple[float, float, str]:
        """MLX 基准测试 — 返回 (tps, latency_ms, error).

        如果有模型路径, 做真实推理; 否则用已知性能估算.
        """
        if not model_path:
            # 无模型可用, 用已知性能估算
            return 0.0, 0.0, "no_model_path"

        try:
            import mlx_lm
            t0 = time.time()
            model, tokenizer = mlx_lm.load(model_path)
            response = mlx_lm.generate(
                model, tokenizer,
                prompt=self.test_prompt,
                max_tokens=5,
                verbose=False,
            )
            elapsed_ms = (time.time() - t0) * 1000
            tps = 5 / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
            # Cleanup
            del model, tokenizer
            return tps, elapsed_ms, ""
        except Exception as e:
            return 0.0, 0.0, str(e)

    def _benchmark_llamacpp(self, model_path: str = "") -> tuple[float, float, str]:
        """llama.cpp 基准测试 — 返回 (tps, latency_ms, error).

        如果有 GGUF 模型路径, 启动 llama-server 做真实推理; 否则用已知性能估算.
        """
        if not model_path:
            return 0.0, 0.0, "no_model_path"

        try:
            import aiohttp
            import asyncio

            port = 18082
            proc = subprocess.Popen(
                [
                    "llama-server",
                    "-m", model_path,
                    "--port", str(port),
                    "-ngl", "24",   # GPU offload 24 layers
                    "-c", "512",    # context
                    "--no-webui",
                    "-t", "4",      # threads
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Wait for server to be ready
            time.sleep(3)

            async def _run_test():
                async with aiohttp.ClientSession() as session:
                    # Warmup
                    for _ in range(2):
                        try:
                            await session.post(
                                f"http://localhost:{port}/v1/completions",
                                json={
                                    "prompt": "hello",
                                    "max_tokens": 1,
                                    "temperature": 0,
                                },
                                timeout=aiohttp.ClientTimeout(total=10),
                            )
                        except:
                            pass

                    # Real test
                    t0 = time.time()
                    async with session.post(
                        f"http://localhost:{port}/v1/completions",
                        json={
                            "prompt": self.test_prompt,
                            "max_tokens": 5,
                            "temperature": 0,
                        },
                        timeout=aiohttp.ClientTimeout(total=15),
                    ) as resp:
                        data = await resp.json()
                        elapsed_ms = (time.time() - t0) * 1000
                        tps = 5 / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
                        return tps, elapsed_ms

            loop = asyncio.new_event_loop()
            tps, latency_ms = loop.run_until_complete(_run_test())
            loop.close()

            return tps, latency_ms, ""

        except Exception as e:
            return 0.0, 0.0, str(e)
        finally:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except:
                try:
                    proc.kill()
                except:
                    pass

    def _estimate_from_known(self, chip: str) -> dict[str, tuple[float, float]]:
        """根据已知性能基线估算."""
        results = {}
        for (prefix, backend), (tps, latency) in self.KNOWN_PERFORMANCE.items():
            if chip.startswith(prefix):
                results[backend] = (tps, latency)
        return results

    def run(
        self,
        system_profile: Optional[SystemProfile] = None,
        mlx_model_path: str = "",
        gguf_model_path: str = "",
        verbose: bool = False,
    ) -> PlatformBenchmarkResult:
        """执行平台基准测试.

        Args:
            system_profile: 系统画像 (用于已知性能估算)
            mlx_model_path: MLX 格式模型路径 (可选, 用于真实测试)
            gguf_model_path: GGUF 格式模型路径 (可选, 用于真实测试)
            verbose: 打印详细日志

        Returns:
            PlatformBenchmarkResult
        """
        result = PlatformBenchmarkResult()

        # 1. 检查可用性
        result.mlx_available = self._check_mlx()
        result.llamacpp_available = self._check_llamacpp()

        if verbose:
            print(f"  [benchmark] MLX: {'✅' if result.mlx_available else '❌'}")
            print(f"  [benchmark] llama.cpp: {'✅' if result.llamacpp_available else '❌'}")

        # 2. 真实基准测试 (如果有模型路径)
        if result.mlx_available and mlx_model_path:
            tps, latency, err = self._benchmark_mlx(mlx_model_path)
            result.mlx_tps = tps
            result.mlx_latency_ms = latency
            if err:
                result.error += f"mlx: {err}; "
            if verbose and tps > 0:
                print(f"  [benchmark] MLX 实测: {tps:.1f} tok/s ({latency:.0f}ms)")
        elif result.mlx_available and system_profile:
            # 用已知性能估算
            known = self._estimate_from_known(system_profile.cpu_brand)
            if "mlx" in known:
                result.mlx_tps, result.mlx_latency_ms = known["mlx"]
                if verbose:
                    print(f"  [benchmark] MLX 估算: {result.mlx_tps:.1f} tok/s ({result.mlx_latency_ms:.0f}ms)")

        if result.llamacpp_available and gguf_model_path:
            tps, latency, err = self._benchmark_llamacpp(gguf_model_path)
            result.llamacpp_tps = tps
            result.llamacpp_latency_ms = latency
            if err:
                result.error += f"llamacpp: {err}; "
            if verbose and tps > 0:
                print(f"  [benchmark] llama.cpp 实测: {tps:.1f} tok/s ({latency:.0f}ms)")
        elif result.llamacpp_available and system_profile:
            known = self._estimate_from_known(system_profile.cpu_brand)
            if "llamacpp" in known:
                result.llamacpp_tps, result.llamacpp_latency_ms = known["llamacpp"]
                if verbose:
                    print(f"  [benchmark] llama.cpp 估算: {result.llamacpp_tps:.1f} tok/s ({result.llamacpp_latency_ms:.0f}ms)")

        # 3. 选择更快的后端
        if result.mlx_tps > 0 and result.llamacpp_tps > 0:
            if result.llamacpp_tps > result.mlx_tps:
                result.preferred_backend = "llamacpp"
                result.speedup = result.llamacpp_tps / result.mlx_tps
            else:
                result.preferred_backend = "mlx"
                result.speedup = result.mlx_tps / result.llamacpp_tps
        elif result.mlx_tps > 0:
            result.preferred_backend = "mlx"
            result.speedup = 1.0
        elif result.llamacpp_tps > 0:
            result.preferred_backend = "llamacpp"
            result.speedup = 1.0
        else:
            result.preferred_backend = "none"
            result.speedup = 1.0

        if verbose:
            print(f"  [benchmark] 首选后端: {result.preferred_backend} (speedup {result.speedup:.1f}x)")

        return result


# ============================================================================
# 2. ProfileBinding — 模型需求 vs 硬件能力 + 后端匹配
# ============================================================================
@dataclass
class ProfileBinding:
    """画像绑定 — 模型 draft head 需求 vs 端侧硬件能力.

    核心输出: can_run_on_edge
      True  → 端侧跑 draft, 云端只做 verify (省云端算力)
      False → 云端 NEXTN MTP (draft+verify 都在云端)

    量化自动选择:
      根据 max_local_model_gb 自动选择最高质量且能放下的量化等级
      例如 Gemma4 drafter: BF16=8.2GB > Q4=4.1GB > Q2=2.1GB
      M4 16GB (可用~9.6GB) → Q4 (4.1GB, 平衡精度)
      M4 8GB (可用~4.8GB)  → Q2 (2.1GB, 极致压缩)
    """
    model_name: str = ""
    model_display_name: str = ""

    # Draft model 信息
    draft_model_path: str = ""
    draft_model_size_gb: float = 0.0
    draft_architecture: str = ""
    draft_model_type: str = ""
    draft_num_layers: int = 0
    draft_hidden_size: int = 0
    draft_vocab_size: int = 0
    draft_params_m: float = 0.0

    # 量化等级选择
    drafter_quant_sizes: list = field(default_factory=list)  # [(label, size_gb, desc), ...]
    selected_quant: str = ""       # 自动选择的量化等级 (如 "q4", "bf16", "q2")
    selected_quant_desc: str = ""  # 量化等级描述

    # 匹配结果
    can_run_on_edge: bool = False
    edge_backend: str = ""  # mlx / llamacpp / none (由 PlatformBenchmark 决定)
    reason: str = ""

    # 端侧预估 (can_run_on_edge=True 时有意义)
    estimated_edge_tps: float = 0.0
    estimated_edge_ttft_ms: float = 0.0
    estimated_memory_usage_gb: float = 0.0

    # 云端算力节省 (端侧跑 draft 时)
    cloud_compute_savings_pct: float = 0.0

    # 云端 MTP 预估 (can_run_on_edge=False 时有意义)
    estimated_cloud_mtp_tps: float = 0.0
    estimated_cloud_mtp_accept_rate: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def evaluate(
        cls,
        model_name: str,
        model_display_name: str,
        draft_model_path: str,
        draft_model_size_gb: float,
        draft_architecture: str,
        draft_model_type: str,
        draft_num_layers: int,
        draft_hidden_size: int,
        draft_vocab_size: int,
        draft_params_m: float,
        system_profile: SystemProfile,
        cloud_accept_rate: float = 0.90,
        drafter_quant_sizes: list = None,
    ) -> "ProfileBinding":
        """评估模型 draft head 是否可以在端侧运行.

        判断逻辑:
          1. 架构是否被端侧引擎支持? (deepseek_v4 不被 MLX/llama.cpp 支持)
          2. 量化等级选择: 从高质量到低质量, 选第一个能放进内存的
             例如 Gemma4 drafter: BF16=8.2GB > Q4=4.1GB > Q2=2.1GB
             M4 16GB (可用~9.6GB) → BF16 (8.2GB, 完整精度)
             M4 8GB (可用~4.8GB)  → Q4 (4.1GB, 平衡)
          3. 端侧 decode 速度是否足够? (>30 tok/s 才有价值)
          4. 选择更快的后端 (PlatformBenchmark.preferred_backend)
        """
        binding = cls(
            model_name=model_name,
            model_display_name=model_display_name,
            draft_model_path=draft_model_path,
            draft_model_size_gb=draft_model_size_gb,
            draft_architecture=draft_architecture,
            draft_model_type=draft_model_type,
            draft_num_layers=draft_num_layers,
            draft_hidden_size=draft_hidden_size,
            draft_vocab_size=draft_vocab_size,
            draft_params_m=draft_params_m,
            drafter_quant_sizes=drafter_quant_sizes or [],
        )

        # 0. 设置端侧后端 (来自 PlatformBenchmark)
        binding.edge_backend = system_profile.preferred_edge_backend or "none"

        # 1. 架构支持检查
        if not system_profile.can_run_architecture(draft_architecture):
            binding.can_run_on_edge = False
            binding.edge_backend = "none"
            binding.reason = (
                f"架构 {draft_architecture} 不被端侧 (MLX/llama.cpp) 支持 "
                f"→ 必须用云端 NEXTN MTP"
            )
            binding.estimated_cloud_mtp_tps = 273.0
            binding.estimated_cloud_mtp_accept_rate = cloud_accept_rate
            return binding

        # 2. 量化等级选择 — 从高质量到低质量, 选第一个能放进内存的
        max_local = system_profile.max_local_model_gb
        selected_size = draft_model_size_gb
        selected_quant = "default"
        selected_quant_desc = f"{draft_model_size_gb:.1f}GB"

        if drafter_quant_sizes:
            # 从大到小排序 (优先高质量)
            sorted_quants = sorted(drafter_quant_sizes, key=lambda x: -x[1])
            for quant_label, quant_size, quant_desc in sorted_quants:
                if quant_size <= max_local:
                    selected_size = quant_size
                    selected_quant = quant_label
                    selected_quant_desc = quant_desc
                    break
            else:
                # 所有量化等级都放不下
                binding.can_run_on_edge = False
                smallest = sorted_quants[-1] if sorted_quants else (None, draft_model_size_gb, "")
                binding.reason = (
                    f"架构 {draft_architecture} 支持但所有量化等级均超出端侧内存 "
                    f"(最小 {smallest[1]:.1f}GB > 可用 {max_local:.1f}GB)"
                )
                binding.estimated_cloud_mtp_tps = 273.0
                binding.estimated_cloud_mtp_accept_rate = cloud_accept_rate
                return binding

        binding.selected_quant = selected_quant
        binding.selected_quant_desc = selected_quant_desc
        binding.draft_model_size_gb = selected_size

        # 3. 端侧可以运行 — 使用 PlatformBenchmark 结果选择后端
        binding.can_run_on_edge = True
        backend = system_profile.preferred_edge_backend or "mlx"
        backend_tps = system_profile.llamacpp_benchmark_tps if backend == "llamacpp" else system_profile.mlx_benchmark_tps

        binding.reason = (
            f"架构 {draft_architecture} 支持 + "
            f"量化 {selected_quant} ({selected_size:.1f}GB) <= {max_local:.1f}GB + "
            f"后端 {backend} ({backend_tps:.0f} tok/s)"
        )

        # 估算端侧性能: 优先用 benchmark 实测数据, 否则用硬件等级估算
        if backend_tps > 0:
            # 按模型大小缩放 (benchmark 是 1B/2B 模型, draft head 通常更小更快)
            scale = max(1.0, 2000.0 / max(draft_params_m, 100))
            binding.estimated_edge_tps = min(backend_tps * scale, 500.0)
        elif system_profile.compute_tier == "ultra":
            binding.estimated_edge_tps = min(500.0, 3000.0 / max(draft_params_m / 100, 1))
        elif system_profile.compute_tier == "strong":
            binding.estimated_edge_tps = min(300.0, 2000.0 / max(draft_params_m / 100, 1))
        elif system_profile.compute_tier == "medium":
            binding.estimated_edge_tps = min(200.0, 1500.0 / max(draft_params_m / 100, 1))
        else:
            binding.estimated_edge_tps = min(50.0, 500.0 / max(draft_params_m / 100, 1))

        binding.estimated_edge_ttft_ms = 30 + draft_params_m * 0.1
        binding.estimated_memory_usage_gb = selected_size * 1.2  # +20% for KV cache

        # 云端算力节省:
        # NEXTN MTP: cloud does 5 draft passes + 1 verify pass per (1+accept_len) tokens
        # Edge draft: cloud does 0 draft passes + 1 verify pass per (1+accept_len) tokens
        # Savings = 5 / 6 = ~83% of draft+verify compute
        # But draft model is small (4 layers vs 48 layers main model)
        # Actual savings = draft_layers / (draft_layers + main_layers) per forward pass
        # For NEXTN with 5 steps: 5 * draft_compute / (5 * draft_compute + verify_compute)
        main_layers = 48  # Gemma4 26B has ~48 layers
        draft_compute_ratio = draft_num_layers / main_layers
        # 5 draft forward passes + 1 verify forward pass
        total_compute = 5 * draft_compute_ratio + 1.0
        draft_compute_pct = (5 * draft_compute_ratio) / total_compute
        binding.cloud_compute_savings_pct = draft_compute_pct

        return binding


# ============================================================================
# 3. StateABI — 状态管理 + 转换协议
# ============================================================================
class StateABI:
    """State Application Binary Interface — 标准化状态管理.

    定义 Hermes 路由器的状态转换协议:
      BOOTSTRAP → READY → ACTIVE → {EDGE_DECODE, CLOUD_MTP, CLOUD_ONLY, CACHE_HIT}

    每个状态有:
      entry  — 进入状态时的操作
      exit   — 离开状态时的操作
      allowed_transitions — 允许的下一状态

    状态可序列化为 JSON (State ABI 的 "binary" 契约).
    """

    STATES: dict[str, dict] = {
        "BOOTSTRAP": {
            "desc": "初始化中",
            "transitions": ["READY", "ERROR"],
        },
        "READY": {
            "desc": "就绪, 等待请求",
            "transitions": ["ACTIVE", "DEGRADED", "OFFLINE"],
        },
        "ACTIVE": {
            "desc": "处理请求中",
            "transitions": ["EDGE_DECODE", "CLOUD_MTP", "CLOUD_ONLY",
                            "CACHE_HIT", "LOCAL_ONLY", "DEGRADED", "OFFLINE", "READY"],
        },
        "EDGE_DECODE": {
            "desc": "端侧 draft decode + 云端 verify",
            "transitions": ["ACTIVE", "CLOUD_MTP", "CLOUD_ONLY", "READY"],
        },
        "CLOUD_MTP": {
            "desc": "云端 NEXTN MTP (draft+verify 都在云端)",
            "transitions": ["ACTIVE", "CLOUD_ONLY", "DEGRADED", "READY"],
        },
        "CLOUD_ONLY": {
            "desc": "云端直连 (无 MTP)",
            "transitions": ["ACTIVE", "DEGRADED", "OFFLINE", "READY"],
        },
        "CACHE_HIT": {
            "desc": "缓存命中, 直接返回",
            "transitions": ["ACTIVE", "READY"],
        },
        "LOCAL_ONLY": {
            "desc": "本地推理 (离线/隐私)",
            "transitions": ["ACTIVE", "READY"],
        },
        "DEGRADED": {
            "desc": "降级模式",
            "transitions": ["READY", "OFFLINE"],
        },
        "OFFLINE": {
            "desc": "离线",
            "transitions": ["BOOTSTRAP"],
        },
        "ERROR": {
            "desc": "错误",
            "transitions": ["BOOTSTRAP"],
        },
    }

    def __init__(self):
        self._state: str = "BOOTSTRAP"
        self._history: list[dict] = []
        self._entry_time: float = time.time()

    @property
    def current(self) -> str:
        return self._state

    @property
    def elapsed_ms(self) -> float:
        return (time.time() - self._entry_time) * 1000

    def transition(self, new_state: str, reason: str = "") -> bool:
        """状态转换 (验证合法性)."""
        current_def = self.STATES.get(self._state, {})
        allowed = current_def.get("transitions", [])
        if new_state not in allowed:
            logger.warning(
                f"[state_abi] 非法转换: {self._state} → {new_state} "
                f"(允许: {allowed})"
            )
            return False

        old_state = self._state
        self._history.append({
            "from": old_state,
            "to": new_state,
            "reason": reason,
            "timestamp": time.time(),
        })
        self._state = new_state
        self._entry_time = time.time()
        logger.info(f"[state_abi] {old_state} → {new_state} ({reason})")
        return True

    def can_transition_to(self, new_state: str) -> bool:
        return new_state in self.STATES.get(self._state, {}).get("transitions", [])

    def get_history(self) -> list[dict]:
        return self._history

    def to_dict(self) -> dict:
        return {
            "current_state": self._state,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "history_count": len(self._history),
            "last_transition": self._history[-1] if self._history else None,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


# ============================================================================
# 4. FourDMatrix — 4D 感知矩阵 (保留自 route_decision_v2.py)
# ============================================================================
@dataclass
class D1Network:
    """D1: 网络感知."""
    rtt_ms: float = 0.0
    bandwidth_mbps: float = 0.0
    jitter_ms: float = 0.0
    stability: Literal["stable", "unstable", "offline"] = "stable"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class D2Hardware:
    """D2: 硬件感知 (来自 SystemProfile)."""
    chip: str = ""
    avail_mem_gb: float = 0.0
    total_mem_gb: float = 0.0
    disk_free_gb: float = 0.0
    tflops_fp16: float = 0.0
    tflops_int8: float = 0.0
    engine: str = "mlx"
    unified_memory: bool = False
    gpu_name: str = ""
    gpu_vram_gb: float = 0.0
    compute_tier: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class D3Model:
    """D3: 模型感知 (来自 model_registry)."""
    name: str = ""
    params_b: float = 0.0
    num_layers: int = 0
    is_moe: bool = False
    num_experts: int = 0
    experts_per_tok: int = 0
    hidden_size: int = 0
    vocab_size: int = 0
    quantization: str = "bf16"
    model_size_gb: float = 0.0
    per_layer_gb: float = 0.0
    has_native_mtp: bool = False
    draft_model_path: str = ""
    draft_architecture: str = ""
    draft_model_size_gb: float = 0.0
    draft_params_m: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class D4Decision:
    """D4: 路由决策."""
    mode: Literal[
        "cache_hit",
        "edge_draft",
        "cloud_mtp",
        "cloud_only",
        "local_only",
    ] = "cloud_mtp"
    confidence: float = 0.0
    reason: str = ""
    expected_ttft_ms: float = 0.0
    expected_decode_tps: float = 0.0
    expected_accept_rate: float = 0.0
    cloud_compute_savings_pct: float = 0.0
    cloud_url: str = ""
    use_mtp: bool = True
    selected_quant: str = ""       # 端侧量化等级 (edge_draft 模式)
    edge_backend: str = ""         # 端侧后端 (edge_draft 模式)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class FourDMatrix:
    """4D 感知矩阵 — Hermes 路由输入."""
    D1: D1Network = field(default_factory=D1Network)
    D2: D2Hardware = field(default_factory=D2Hardware)
    D3: D3Model = field(default_factory=D3Model)
    D4: D4Decision = field(default_factory=D4Decision)
    prompt_preview: str = ""
    prompt_has_code: bool = False
    history_accept_rate: float = 0.0
    cache_hit_rate: float = 0.0

    def to_dict(self) -> dict:
        return {
            "D1_network": self.D1.to_dict(),
            "D2_hardware": self.D2.to_dict(),
            "D3_model": self.D3.to_dict(),
            "D4_decision": self.D4.to_dict(),
            "context": {
                "prompt_preview": self.prompt_preview[:200],
                "prompt_has_code": self.prompt_has_code,
                "history_accept_rate": self.history_accept_rate,
                "cache_hit_rate": self.cache_hit_rate,
            },
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_system_profile(
        cls,
        profile: SystemProfile,
        model_name: str = "",
        draft_model_path: str = "",
        draft_architecture: str = "",
        draft_model_size_gb: float = 0.0,
        draft_params_m: float = 0.0,
        prompt: str = "",
    ) -> "FourDMatrix":
        """从 SystemProfile 构建 4D 矩阵."""
        d1 = D1Network(
            rtt_ms=profile.rtt_ms,
            bandwidth_mbps=profile.bandwidth_mbps,
            stability="stable" if profile.rtt_ms < 200 and profile.online else (
                "offline" if not profile.online else "unstable"
            ),
        )
        d2 = D2Hardware(
            chip=profile.cpu_brand,
            avail_mem_gb=profile.avail_mem_gb,
            total_mem_gb=profile.total_mem_gb,
            disk_free_gb=profile.disk_available_gb,
            tflops_fp16=profile.tflops,
            tflops_int8=profile.tflops * 2,
            engine=profile.recommended_engine,
            unified_memory=(profile.gpu_type == "apple_metal"),
            gpu_name=profile.gpu_name,
            gpu_vram_gb=profile.gpu_vram_gb,
            compute_tier=profile.compute_tier,
        )
        d3 = D3Model(
            name=model_name,
            draft_model_path=draft_model_path,
            draft_architecture=draft_architecture,
            draft_model_size_gb=draft_model_size_gb,
            draft_params_m=draft_params_m,
        )
        has_code = any(kw in prompt.lower() for kw in [
            "def ", "class ", "import ", "function ", "const ", "var ",
            "```", "    ", "\t", "return ", "if ", "for ", "while ",
        ])
        return cls(
            D1=d1, D2=d2, D3=d3,
            prompt_preview=prompt[:200],
            prompt_has_code=has_code,
        )


# ============================================================================
# 4.5 D5ContentAware — 内容感知路由 (D5)
# ============================================================================
@dataclass
class D5Content:
    """D5: 内容感知 — prompt 类型分析 + 路由参数调整.

    根据 prompt 内容调整 MTP 路由参数:
      - 代码补全 (code_completion): 高 accept rate → 激进 num_draft
      - 代码生成 (code_generation): 中等 accept rate → 适中 num_draft
      - 聊天/问答 (chat): 低 accept rate → 保守 num_draft 或跳过 MTP
      - 推理 (reasoning): 低 accept rate → 跳过 MTP (推理 tokens 不可预测)
    """
    prompt_type: str = "chat"  # code_completion / code_generation / chat / reasoning
    language: str = ""         # python / javascript / etc.
    has_code_context: bool = False
    is_completion: bool = False  # True = 续写 (高 accept), False = 生成 (低 accept)
    expected_accept_rate: float = 0.50
    suggested_num_draft: int = 2
    suggested_mode_override: str = ""  # 非空时覆盖路由决策

    def to_dict(self) -> dict:
        return asdict(self)


class D5ContentAware:
    """内容感知路由器 — 分析 prompt 内容, 调整 MTP 参数.

    实测数据驱动 (Qwen2.5-0.5B MTP):
      - 代码 prompt: accept 83.3% (step-0), 远高于 chat prompt
      - code_completion (续写): accept 最高, 因为 next token 高度可预测
      - chat/reasoning: accept 低, 因为生成内容不可预测
      - num_draft=1: 81% accept (v1 实测)
      - num_draft=4: 49% accept (v1 实测)

    调整策略:
      code_completion → num_draft=4, accept_boost=+0.15
      code_generation → num_draft=2, accept_boost=+0.05
      chat            → num_draft=1, accept_boost=0
      reasoning       → num_draft=0, suggested_mode=cloud_only (跳过 MTP)
    """

    # 代码关键词 (按语言)
    CODE_PATTERNS = {
        "python": [
            r"\bdef\s+\w+\s*\(", r"\bclass\s+\w+\s*[\(:]",
            r"\bimport\s+\w+", r"\bfrom\s+\w+\s+import",
            r"\breturn\s+", r"\bif\s+.*:\s*$",
            r"\bfor\s+\w+\s+in\s+", r"\bwhile\s+.*:",
            r"\basync\s+def\s+", r"\bawait\s+",
            r"\blambda\s+", r"\bprint\s*\(",
            r"\bself\.\w+", r"\bnp\.\w+", r"\btorch\.\w+",
        ],
        "javascript": [
            r"\bfunction\s+\w+\s*\(", r"\bconst\s+\w+\s*=",
            r"\bvar\s+\w+\s*=", r"\blet\s+\w+\s*=",
            r"=>\s*[{]?", r"\bconsole\.log\s*\(",
            r"\basync\s+function\s+", r"\bawait\s+",
            r"\bclass\s+\w+\s+extends",
        ],
        "rust": [
            r"\bfn\s+\w+\s*\(", r"\bstruct\s+\w+",
            r"\bimpl\s+\w+", r"\buse\s+\w+::",
            r"\bpub\s+fn\s+", r"\bmatch\s+.*\{",
        ],
        "go": [
            r"\bfunc\s+\w+\s*\(", r"\bpackage\s+\w+",
            r"\bimport\s+\"", r"\btype\s+\w+\s+struct",
        ],
        "sql": [
            r"\bSELECT\s+", r"\bINSERT\s+INTO",
            r"\bCREATE\s+TABLE", r"\bUPDATE\s+.*\bSET",
        ],
    }

    # 补全信号: prompt 以不完整代码结尾 (高 accept rate)
    COMPLETION_SIGNALS = [
        # 以缩进结尾 (续写函数体)
        r":\s*$", r"\{\s*$", r"\[\s*$", r"\(\s*$",
        # 以操作符结尾
        r"=\s*$", r"->\s*$", r"=>\s*$", r"\+\s*$",
        # 以关键字开头行 (等待下一行)
        r"\breturn\s*$", r"\bif\s+.*:\s*$", r"\belse:\s*$",
        r"\bfor\s+.*:\s*$", r"\bwhile\s+.*:\s*$",
        # import 后等待
        r"\bimport\s+\w+\s*$",
    ]

    # 推理信号 (低 accept rate, 跳过 MTP)
    REASONING_SIGNALS = [
        r"explain\s+why", r"why\s+does", r"analyze\s+the",
        r"prove\s+that", r"derive\s+the", r"calculate\s+the\s+time\s+complexity",
        r"what\s+is\s+the\s+time\s+complexity", r"compare\s+and\s+contrast",
        r"describe\s+the\s+difference", r"step\s+by\s+step",
    ]

    def __init__(self):
        import re
        self._re = re

    def analyze(self, prompt: str) -> D5Content:
        """分析 prompt 内容, 返回 D5 内容感知结果.

        Args:
            prompt: 用户输入 prompt

        Returns:
            D5Content: 内容分析结果 + 路由参数建议
        """
        prompt_stripped = prompt.rstrip()
        prompt_lower = prompt_lower = prompt.lower()

        # 1. 检测语言
        language = self._detect_language(prompt)

        # 2. 检测是否为代码补全 (续写)
        is_completion = self._is_completion(prompt_stripped)

        # 3. 检测是否为推理类
        is_reasoning = any(
            self._re.search(pat, prompt_lower, self._re.IGNORECASE)
            for pat in self.REASONING_SIGNALS
        )

        # 4. 确定 prompt 类型
        has_code = language != "" or any(
            self._re.search(pat, prompt)
            for patterns in self.CODE_PATTERNS.values()
            for pat in patterns
        )

        if is_reasoning:
            prompt_type = "reasoning"
        elif is_completion and has_code:
            prompt_type = "code_completion"
        elif has_code:
            prompt_type = "code_generation"
        else:
            prompt_type = "chat"

        # 5. 调整路由参数
        base_accept = 0.50  # 默认 accept rate
        base_num_draft = 2

        if prompt_type == "code_completion":
            expected_accept = min(base_accept + 0.35, 0.95)
            suggested_num_draft = 4
        elif prompt_type == "code_generation":
            expected_accept = min(base_accept + 0.15, 0.80)
            suggested_num_draft = 2
        elif prompt_type == "reasoning":
            expected_accept = max(base_accept - 0.30, 0.15)
            suggested_num_draft = 0
        else:  # chat
            expected_accept = base_accept
            suggested_num_draft = 1

        # 推理类 → 建议跳过 MTP
        mode_override = "cloud_only" if prompt_type == "reasoning" else ""

        return D5Content(
            prompt_type=prompt_type,
            language=language,
            has_code_context=has_code,
            is_completion=is_completion,
            expected_accept_rate=expected_accept,
            suggested_num_draft=suggested_num_draft,
            suggested_mode_override=mode_override,
        )

    def _detect_language(self, prompt: str) -> str:
        """检测 prompt 中的编程语言."""
        for lang, patterns in self.CODE_PATTERNS.items():
            matches = sum(
                1 for pat in patterns
                if self._re.search(pat, prompt)
            )
            if matches >= 1:
                return lang
        return ""

    def _is_completion(self, prompt_stripped: str) -> bool:
        """检测 prompt 是否为代码补全 (以不完整代码结尾)."""
        for pat in self.COMPLETION_SIGNALS:
            if self._re.search(pat, prompt_stripped, self._re.MULTILINE):
                return True
        return False


# ============================================================================
# 5. TenStepPipeline — 十步 4D 感知流水线 (保留)
# ============================================================================
class TenStepPipeline:
    """十步 4D 感知流水线 — 整合硬件感知 + 路由决策.

    Step 1:   系统侦测 (OS + arch)
    Step 2:   CPU 侦测
    Step 3:   模型格式解析
    Step 4:   模型架构分析 (MoE/Dense)
    Step 5:   内存/显存水位扫描
    Step 5.5: 算力等级检测
    Step 6:   运算引擎路由 (MLX/CUDA/ROCm/OMLX)
    Step 7:   记忆体策略 (FlashMoE)
    Step 7.5: 路由决策 (edge_draft / cloud_mtp / cloud_only)
    Step 7.6: 模型分发决策
    Step 7.7: MTP draft model 同步
    Step 8:   上下文构建
    Step 9:   4D 感知矩阵上报
    Step 10:  磁碟空间检查
    """

    def execute(
        self,
        profile: SystemProfile,
        binding: ProfileBinding,
        prompt: str = "",
        verbose: bool = False,
    ) -> dict:
        """执行十步流水线, 返回每步结果."""
        steps = {}
        log = lambda step, msg: (
            print(f"  [{step}] {msg}") if verbose else None
        )

        # Step 1: 系统侦测
        steps["1_os"] = {"os": profile.os_name, "arch": profile.arch, "version": profile.os_version}
        log("1/10", f"系统侦测: {profile.os_name} {profile.arch}")

        # Step 2: CPU 侦测
        steps["2_cpu"] = {"brand": profile.cpu_brand, "cores": profile.cpu_cores}
        log("2/10", f"CPU: {profile.cpu_brand} ({profile.cpu_cores} cores)")

        # Step 3: 模型格式解析
        model_format = "unknown"
        if binding.draft_model_path:
            if binding.draft_model_path.endswith(".gguf"):
                model_format = "GGUF"
            elif binding.draft_model_path.endswith(".mlx"):
                model_format = "MLX"
            elif ".safetensors" in binding.draft_model_path or os.path.isdir(binding.draft_model_path):
                model_format = "SafeTensors"
        steps["3_format"] = {"format": model_format}
        log("3/10", f"格式解析: {model_format}")

        # Step 4: 模型架构分析
        steps["4_arch"] = {
            "architecture": binding.draft_architecture,
            "model_type": binding.draft_model_type,
            "num_layers": binding.draft_num_layers,
            "hidden_size": binding.draft_hidden_size,
            "vocab_size": binding.draft_vocab_size,
            "params_m": binding.draft_params_m,
        }
        log("4/10", f"架构: {binding.draft_architecture} ({binding.draft_num_layers}L, "
                     f"{binding.draft_params_m:.1f}M params)")

        # Step 5: 内存水位
        steps["5_mem"] = {
            "avail_gb": profile.avail_mem_gb,
            "total_gb": profile.total_mem_gb,
            "model_needs_gb": binding.draft_model_size_gb,
            "max_local_gb": profile.max_local_model_gb,
        }
        log("5/10", f"内存水位: {profile.avail_mem_gb:.1f}GB 可用 / {profile.total_mem_gb:.1f}GB 总计 "
                     f"(模型需 {binding.draft_model_size_gb:.1f}GB)")

        # Step 5.5: 算力等级
        steps["5.5_compute"] = {
            "tier": profile.compute_tier,
            "tflops": profile.tflops,
            "gpu": profile.gpu_name,
            "engine": profile.recommended_engine,
        }
        log("5.5/10", f"算力等级: {profile.gpu_name} → {profile.compute_tier} ({profile.tflops} TFLOPS)")

        # Step 6: 运算引擎路由
        steps["6_engine"] = {"engine": profile.recommended_engine}
        log("6/10", f"运算引擎: {profile.recommended_engine}")

        # Step 7: 记忆体策略
        use_flashmoe = False  # 端侧 draft model 通常不是 MoE
        steps["7_memstrat"] = {"flashmoe": use_flashmoe}
        log("7/10", f"记忆体策略: {'FlashMoE' if use_flashmoe else '标准载入'}")

        # Step 7.5: 路由决策 (核心!)
        route_mode = "cloud_mtp"
        if binding.can_run_on_edge:
            route_mode = "edge_draft"
        route_reason = binding.reason
        steps["7.5_route"] = {
            "mode": route_mode,
            "can_run_on_edge": binding.can_run_on_edge,
            "reason": route_reason,
        }
        log("7.5/10", f"路由决策: {route_mode} — {route_reason}")

        # Step 7.6: 模型分发
        if route_mode == "edge_draft":
            dispatch = "load_to_edge"
        else:
            dispatch = "cloud_only"
        steps["7.6_dispatch"] = {"action": dispatch}
        log("7.6/10", f"模型分發: {dispatch}")

        # Step 7.7: MTP draft 同步
        if route_mode == "cloud_mtp":
            mtp_sync = "cloud_nextn"
        elif route_mode == "edge_draft":
            mtp_sync = "edge_local"
        else:
            mtp_sync = "none"
        steps["7.7_mtp"] = {"sync": mtp_sync, "accept_rate": binding.estimated_cloud_mtp_accept_rate}
        log("7.7/10", f"MTP draft: {mtp_sync}")

        # Step 8: 上下文构建
        steps["8_context"] = {
            "prompt_len": len(prompt),
            "has_code": any(kw in prompt.lower() for kw in ["def ", "class ", "import "]),
        }
        log("8/10", "上下文構建: 參數自動注入完成")

        # Step 9: 4D 矩阵上报
        steps["9_4d"] = {
            "D1_rtt": profile.rtt_ms,
            "D2_tier": profile.compute_tier,
            "D3_model": binding.model_name,
            "D4_route": route_mode,
        }
        log("9/10", f"4D 感知矩陣: D1(RTT={profile.rtt_ms:.0f}ms) D2({profile.compute_tier}) "
                     f"D3({binding.model_name}) D4({route_mode})")

        # Step 10: 磁碟空间
        steps["10_disk"] = {"avail_gb": profile.disk_available_gb}
        log("10/10", f"磁碟空間: {profile.disk_available_gb:.1f}GB 可用")

        return steps


# ============================================================================
# 6. Bootstrap — 初始化序列
# ============================================================================
@dataclass
class BootstrapResult:
    """Bootstrap 结果."""
    success: bool = False
    system_profile: Optional[dict] = None
    platform_benchmark: Optional[dict] = None
    bindings: dict = field(default_factory=dict)  # model_name → ProfileBinding dict
    cloud_mtp_url: str = ""
    cloud_plain_url: str = ""
    cloud_reachable: bool = False
    state: str = "BOOTSTRAP"
    error: str = ""
    elapsed_ms: float = 0.0


class Bootstrap:
    """Hermes 启动序列 — 系统初始化.

    流程:
      1. detect_hardware() → SystemProfile
      2. benchmark_platform() → PlatformBenchmark (MLX vs llama.cpp, 选快的)
      3. load_model_registry() → 可用模型列表
      4. evaluate_profile_bindings() → 每个模型的 ProfileBinding (含后端选择)
      5. connect_cloud() → 验证云端 endpoint
      6. state_abi.transition("READY")
    """

    # 已知模型的 draft head 信息 (与 model_registry.py 对应)
    # drafter_quant_sizes: 按量化等级从小到大排列, ProfileBinding 自动选合适的
    DRAFT_MODELS = {
        "gemma4": {
            "display_name": "Gemma4-26B-A4B",
            "draft_model_path": "models/gemma-4-mtp-head",
            "draft_architecture": "Gemma4AssistantForCausalLM",
            "draft_model_type": "gemma4_assistant",
            # 量化等级: (label, size_gb, description)
            # BF16 full package ≈8.2GB, MLX Q4≈4.1GB, Q2≈2.1GB, GGUF Q4_K_M≈4.3GB
            "drafter_quant_sizes": [
                ("q2",    2.1, "MLX Q2 极致压缩"),
                ("q4",    4.1, "MLX Q4 平衡速度/精度"),
                ("q4_km", 4.3, "GGUF Q4_K_M (llama.cpp)"),
                ("bf16",  8.2, "BF16 原版 (完整精度)"),
            ],
            "draft_model_size_gb": 4.1,   # 默认 Q4 (平衡)
            "draft_num_layers": 4,
            "draft_hidden_size": 1024,
            "draft_vocab_size": 262144,
            "draft_params_m": 183.1,
            "cloud_accept_rate": 0.95,
            "verify_loop_config": {
                "mtp_checkpoint": os.environ.get("EDGE_GEMMA4_MTP_CHECKPOINT", ""),
                "embed_head_path": os.environ.get("EDGE_GEMMA4_EMBED_HEAD", ""),
                "assistant_model_path": os.environ.get(
                    "EDGE_GEMMA4_ASSISTANT_MODEL",
                    "models/gemma-4-mtp-head/model.safetensors",
                ),
                "num_heads": 16,
                "head_dim": 256,
                "intermediate_size": 8192,
                "n_ctx": int(os.environ.get("EDGE_GEMMA4_VERIFY_CTX", "4096") or "4096"),
                "use_mlx": False,
                "use_cgc_ir": False,
                "use_ggml": True,
                "baseline_tps": 1.8,
                "accept_nd1": 0.0,
                "accept_nd4": 0.0,
                "mtp_forward_ms_ggml": 0.0,
                "mtp_forward_ms_mlx": 0.0,
                "mtp_forward_ms_cpu": 0.0,
                "crossover_tps": 4,
            },
        },
        "gemma4_e4b": {
            "display_name": "Gemma4-E4B",
            "draft_model_path": os.environ.get(
                "EDGE_GEMMA4_E4B_DRAFT_DIR",
                os.path.join(
                    os.environ.get("CGC_MTP_OUTPUT_BASE_DIR", os.path.join(REPO_ROOT, "var", "mtp_output")),
                    "gemma4_e4b",
                ),
            ),
            "draft_architecture": "MTPHead",
            "draft_model_type": "mtphead",
            "drafter_quant_sizes": [
                ("pt", 0.3, "PyTorch MTP head"),
            ],
            "draft_model_size_gb": 0.3,
            "draft_num_layers": 4,
            "draft_hidden_size": 2560,
            "draft_vocab_size": 262144,
            "draft_params_m": 220.0,
            "cloud_accept_rate": 0.70,
            "verify_loop_config": {
                "model_path": os.environ.get("EDGE_GEMMA4_E4B_MODEL_PATH", os.environ.get("EDGE_LOCAL_MAIN_MODEL_PATH", "")),
                "mtp_checkpoint": os.environ.get(
                    "EDGE_GEMMA4_E4B_MTP_CHECKPOINT",
                    os.path.join(
                        os.environ.get("CGC_MTP_OUTPUT_BASE_DIR", os.path.join(REPO_ROOT, "var", "mtp_output")),
                        "gemma4_e4b",
                        "mtp_head_gemma4_e4b_decode.pt",
                    ),
                ),
                "embed_head_path": os.environ.get(
                    "EDGE_GEMMA4_E4B_EMBED_HEAD",
                    os.path.join(REPO_ROOT, "var", "mtp_train_data", "gemma4_e4b", "embed_head.pt"),
                ),
                "assistant_model_path": os.environ.get("EDGE_GEMMA4_E4B_ASSISTANT_MODEL", ""),
                "num_heads": 20,
                "head_dim": 128,
                "intermediate_size": 10240,
                "n_gpu_layers": int(os.environ.get("EDGE_GEMMA4_E4B_VERIFY_N_GPU_LAYERS", "-1") or "-1"),
                "n_ctx": int(os.environ.get("EDGE_GEMMA4_E4B_VERIFY_CTX", "2048") or "2048"),
                "n_batch": int(os.environ.get("EDGE_GEMMA4_E4B_VERIFY_N_BATCH", "128") or "128"),
                "n_ubatch": int(os.environ.get("EDGE_GEMMA4_E4B_VERIFY_N_UBATCH", "64") or "64"),
                "use_mlx": False,
                "use_cgc_ir": False,
                "use_ggml": True,
                "baseline_tps": 26.2,
                "accept_nd1": 0.0,
                "accept_nd4": 0.0,
                "mtp_forward_ms_ggml": 0.0,
                "mtp_forward_ms_mlx": 0.0,
                "mtp_forward_ms_cpu": 0.0,
                "crossover_tps": 24,
            },
        },
        "dsv4": {
            "display_name": "DeepSeek V4 Flash",
            "draft_model_path": "",
            "draft_architecture": "DeepseekV4ForCausalLM",
            "draft_model_type": "deepseek_v4",
            "drafter_quant_sizes": [
                ("bf16", 0.90, "BF16 (需 GPU, MLA 架构)"),
            ],
            "draft_model_size_gb": 0.90,  # ~188.8M params bf16
            "draft_num_layers": 4,
            "draft_hidden_size": 4096,
            "draft_vocab_size": 129280,
            "draft_params_m": 188.8,
            "cloud_accept_rate": 0.85,
        },
        "qwen3vl": {
            "display_name": "Qwen3-VL-2B",
            "draft_model_path": "",
            "draft_architecture": "MTPHead",
            "draft_model_type": "mtphead",
            "drafter_quant_sizes": [
                ("bf16", 0.30, "BF16 (~59.8M params, 极小)"),
            ],
            "draft_model_size_gb": 0.30,
            "draft_num_layers": 4,
            "draft_hidden_size": 2048,
            "draft_vocab_size": 151936,
            "draft_params_m": 59.8,
            "cloud_accept_rate": 0.80,
        },
        "qwen25_05b": {
            "display_name": "Qwen2.5-0.5B",
            "draft_model_path": os.path.expanduser("~/models/gguf/qwen2.5-0.5b-instruct-fp16.gguf"),
            "draft_architecture": "MTPHead",
            "draft_model_type": "mtphead",
            "drafter_quant_sizes": [
                ("fp16", 1.27, "FP16 GGUF (可提取 lm_head/embedding)"),
                ("q4_km", 0.40, "Q4_K_M GGUF (量化, 不可提取权重)"),
            ],
            "draft_model_size_gb": 1.27,
            "draft_num_layers": 1,
            "draft_hidden_size": 896,
            "draft_vocab_size": 151936,
            "draft_params_m": 17.9,
            "cloud_accept_rate": 0.81,
            "verify_loop_config": {
                "mtp_checkpoint": "CGC_Phase2/mtp_output/qwen25_llamacpp_v2/mtp_head_qwen25-0.5b_decode.pt",
                "embed_head_path": "CGC_Phase2/mtp_train_data/qwen25_llamacpp/embed_head.pt",
                "num_heads": 14,
                "head_dim": 64,
                "intermediate_size": 4864,
                "n_ctx": 2048,
                "use_mlx": True,
                "use_cgc_ir": True,
                "use_ggml": True,
                "baseline_tps": 55.0,
                "accept_nd1": 0.91,
                "accept_nd4": 0.57,
                "mtp_forward_ms_ggml": 6.5,
                "mtp_forward_ms_mlx": 7.4,
                "mtp_forward_ms_cpu": 9.5,
                "crossover_tps": 42,
            },
        },
    }

    def __init__(
        self,
        cloud_mtp_url: str = "http://localhost:30001",
        cloud_plain_url: str = "http://localhost:30000",
        cloud_host: str = "",
    ):
        self.cloud_mtp_url = cloud_mtp_url
        self.cloud_plain_url = cloud_plain_url
        self.cloud_host = cloud_host
        self.state_abi = StateABI()
        self.system_profile: Optional[SystemProfile] = None
        self.bindings: dict[str, ProfileBinding] = {}

    def run(self, verbose: bool = False) -> BootstrapResult:
        """执行启动序列."""
        t0 = time.time()
        result = BootstrapResult(
            cloud_mtp_url=self.cloud_mtp_url,
            cloud_plain_url=self.cloud_plain_url,
        )

        try:
            # 1. 硬件检测
            if verbose:
                print("[bootstrap] Step 1: 硬体检测...")
            self.system_profile = SystemProfile.detect(cloud_host=self.cloud_host)
            result.system_profile = self.system_profile.to_dict()

            # 2. 平台基准测试 (MLX vs llama.cpp, 选快的)
            if verbose:
                print("[bootstrap] Step 2: 平台基准测试 (MLX vs llama.cpp)...")
            benchmark = PlatformBenchmark()
            bench_result = benchmark.run(
                system_profile=self.system_profile,
                verbose=verbose,
            )
            result.platform_benchmark = bench_result.to_dict()

            # 将 benchmark 结果写入 system_profile
            self.system_profile.preferred_edge_backend = bench_result.preferred_backend
            self.system_profile.mlx_available = bench_result.mlx_available
            self.system_profile.llamacpp_available = bench_result.llamacpp_available
            self.system_profile.mlx_benchmark_tps = bench_result.mlx_tps
            self.system_profile.llamacpp_benchmark_tps = bench_result.llamacpp_tps
            result.system_profile = self.system_profile.to_dict()

            if verbose:
                print(f"  首选后端: {bench_result.preferred_backend} "
                      f"(MLX={bench_result.mlx_tps:.1f} vs llama.cpp={bench_result.llamacpp_tps:.1f} tok/s, "
                      f"speedup={bench_result.speedup:.1f}x)")

            # 3. 模型注册表 → 评估 ProfileBinding (含后端选择)
            if verbose:
                print("[bootstrap] Step 3: ProfileBinding 评估...")
            for model_name, info in self.DRAFT_MODELS.items():
                binding = ProfileBinding.evaluate(
                    model_name=model_name,
                    model_display_name=info["display_name"],
                    draft_model_path=info["draft_model_path"],
                    draft_model_size_gb=info["draft_model_size_gb"],
                    draft_architecture=info["draft_architecture"],
                    draft_model_type=info["draft_model_type"],
                    draft_num_layers=info["draft_num_layers"],
                    draft_hidden_size=info["draft_hidden_size"],
                    draft_vocab_size=info["draft_vocab_size"],
                    draft_params_m=info["draft_params_m"],
                    system_profile=self.system_profile,
                    cloud_accept_rate=info["cloud_accept_rate"],
                    drafter_quant_sizes=info.get("drafter_quant_sizes", []),
                )
                self.bindings[model_name] = binding
                result.bindings[model_name] = binding.to_dict()
                if verbose:
                    status = "EDGE" if binding.can_run_on_edge else "CLOUD"
                    backend = f"[{binding.edge_backend}]" if binding.can_run_on_edge else ""
                    quant = f"[{binding.selected_quant}]" if binding.selected_quant else ""
                    savings = f"省{binding.cloud_compute_savings_pct:.0%}" if binding.can_run_on_edge else "N/A"
                    print(f"  {model_name}: {status} {backend} {quant} ({savings}) — {binding.reason}")

            # 4. 云端连接检查
            if verbose:
                print("[bootstrap] Step 4: 云端连接检查...")
            import urllib.request
            try:
                req = urllib.request.Request(f"{self.cloud_mtp_url}/health", method="GET")
                urllib.request.urlopen(req, timeout=5)
                result.cloud_reachable = True
                if verbose:
                    print(f"  云端 MTP: {self.cloud_mtp_url} ✓")
            except Exception as e:
                result.cloud_reachable = False
                if verbose:
                    print(f"  云端 MTP: 不可达 ({e})")

            # 5. 状态转换 → READY
            self.state_abi.transition("READY", "Bootstrap 完成")
            result.state = self.state_abi.current
            result.success = True

        except Exception as e:
            result.error = str(e)
            result.state = self.state_abi.current
            logger.error(f"[bootstrap] 失败: {e}")

        result.elapsed_ms = (time.time() - t0) * 1000
        return result


# ============================================================================
# 6.5 VerifyLoopBackend — 端侧 verify loop 执行后端
# ============================================================================
class VerifyLoopBackend:
    """端侧 MTP verify loop 后端 — 包装 MTPVerifyLoop.

    当 HermesRouter 决定 edge_draft 模式时, 用此后端执行:
      1. llama.cpp (Metal GPU) 加载 target model
      2. PyTorch MTP head (CPU) 生成 draft tokens
      3. llama.cpp verify draft tokens (accept/reject + KV cache rewind)

    性能 (M4 16GB, Qwen2.5-0.5B FP16):
      - baseline: 61.5 tok/s (llama.cpp Metal)
      - num_draft=1: 81% accept, 20.2 tok/s (MTP overhead > gain for fast models)
      - num_draft=4: 49% accept, 13.1 tok/s
      - crossover: base < 42 tok/s 时 MTP 有益

    用法:
        backend = VerifyLoopBackend(
            model_path="qwen2.5-0.5b-instruct-fp16.gguf",
            mtp_checkpoint="mtp_head.pt",
            embed_head_path="embed_head.pt",
        )
        result = backend.run("def hello():", max_tokens=50, num_draft=4)
        print(result["text"], result["tps"], result["accept_rate"])
    """

    # 已注册的 verify loop 实例 (单例, 避免重复加载模型)
    _instances: dict[str, "VerifyLoopBackend"] = {}

    def __init__(
        self,
        model_path: str,
        mtp_checkpoint: str,
        embed_head_path: str = "",
        assistant_model_path: str = "",
        hidden_size: int = 896,
        vocab_size: int = 151936,
        num_heads: int = 14,
        head_dim: int = 64,
        intermediate_size: int = 4864,
        n_gpu_layers: int = -1,
        n_ctx: int = 2048,
        n_batch: int = 512,
        n_ubatch: int = 512,
        n_threads: int = 4,
        n_threads_batch: int = 4,
        flash_attn: bool = True,
        offload_kqv: bool = True,
        use_mmap: bool = True,
        use_mlock: bool = False,
        use_mlx: bool = False,
        use_cgc_ir: bool = False,
        use_ggml: bool = False,
    ):
        self.model_path = model_path
        self.mtp_checkpoint = mtp_checkpoint
        self.embed_head_path = embed_head_path
        self.assistant_model_path = assistant_model_path
        self._use_mlx = use_mlx
        self._use_cgc_ir = use_cgc_ir
        self._use_ggml = use_ggml
        self.config = {
            "hidden_size": hidden_size,
            "vocab_size": vocab_size,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "intermediate_size": intermediate_size,
            "n_gpu_layers": n_gpu_layers,
            "n_ctx": n_ctx,
            "n_batch": n_batch,
            "n_ubatch": n_ubatch,
            "n_threads": n_threads,
            "n_threads_batch": n_threads_batch,
            "flash_attn": flash_attn,
            "offload_kqv": offload_kqv,
            "use_mmap": use_mmap,
            "use_mlock": use_mlock,
        }
        self._loop = None  # lazy init

    @classmethod
    def get_or_create(
        cls,
        model_key: str,
        model_path: str,
        mtp_checkpoint: str,
        embed_head_path: str = "",
        assistant_model_path: str = "",
        **kwargs,
    ) -> "VerifyLoopBackend":
        """获取或创建 verify loop 单例 (避免重复加载模型)."""
        if model_key not in cls._instances:
            cls._instances[model_key] = cls(
                model_path=model_path,
                mtp_checkpoint=mtp_checkpoint,
                embed_head_path=embed_head_path,
                assistant_model_path=assistant_model_path,
                **kwargs,
            )
        return cls._instances[model_key]

    def _ensure_loop(self):
        """延迟初始化 MTPVerifyLoop (首次调用时加载模型)."""
        if self._loop is not None:
            return

        import os
        import sys

        # 添加 CGC_Phase2 到 path
        repo_root = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", ".."
        ))
        cgc_phase2 = os.path.join(repo_root, "CGC_Phase2")
        for p in [repo_root, cgc_phase2]:
            if p not in sys.path:
                sys.path.insert(0, p)

        from mtp_verify_loop import MTPVerifyLoop

        ckpt = str(self.mtp_checkpoint or "").strip()
        if ckpt and not os.path.isabs(ckpt):
            ckpt = os.path.join(repo_root, ckpt)

        embed = self.embed_head_path
        if embed and not os.path.isabs(embed):
            embed = os.path.join(repo_root, embed)

        assistant_model = self.assistant_model_path
        if assistant_model and not os.path.isabs(assistant_model):
            assistant_model = os.path.join(repo_root, assistant_model)

        model = self.model_path
        if not os.path.isabs(model):
            model = os.path.join(repo_root, model)

        self._loop = MTPVerifyLoop(
            model_path=model,
            mtp_checkpoint=ckpt if ckpt and os.path.exists(ckpt) else None,
            hidden_size=self.config["hidden_size"],
            vocab_size=self.config["vocab_size"],
            num_heads=self.config["num_heads"],
            head_dim=self.config["head_dim"],
            intermediate_size=self.config["intermediate_size"],
            n_gpu_layers=self.config["n_gpu_layers"],
            n_ctx=self.config["n_ctx"],
            n_batch=self.config["n_batch"],
            n_ubatch=self.config["n_ubatch"],
            n_threads=self.config["n_threads"],
            n_threads_batch=self.config["n_threads_batch"],
            flash_attn=self.config["flash_attn"],
            offload_kqv=self.config["offload_kqv"],
            use_mmap=self.config["use_mmap"],
            use_mlock=self.config["use_mlock"],
            verbose=False,
            use_ngram_fallback=True,
            embed_head_path=embed if embed and os.path.exists(embed) else None,
            assistant_model_path=assistant_model if assistant_model and os.path.exists(assistant_model) else None,
            use_mlx=self._use_mlx,
            use_cgc_ir=self._use_cgc_ir,
            use_ggml=self._use_ggml,
        )

    def run(
        self,
        prompt: str,
        max_tokens: int = 100,
        num_draft: int = 4,
        runtime_unit_plan: Optional[dict[str, Any]] = None,
    ) -> dict:
        """执行 verify loop 生成.

        Args:
            prompt: 输入 prompt
            max_tokens: 最大生成 token 数
            num_draft: draft token 数量 (0 = 纯 baseline, 无 MTP)

        Returns:
            {
                "text": str,           # 生成文本
                "tps": float,          # tokens per second
                "accept_rate": float,  # MTP accept rate
                "avg_accept_len": float,
                "total_tokens": int,
                "prefill_ms": float,
                "draft_ms": float,
                "verify_ms": float,
                "mode": "edge_draft",
            }
        """
        self._ensure_loop()
        self._loop.stats = type(self._loop.stats)()  # reset stats per request
        runtime_request_payload = runtime_unit_plan
        if runtime_unit_plan and not is_unified_runtime_ir_v0(runtime_unit_plan):
            model_family = "gemma4" if "gemma4" in str(runtime_unit_plan.get("family") or self.model_path).lower() else ""
            backend_family = "mlx" if self._use_mlx or model_family == "gemma4" else ("gemma4_native" if self._use_ggml else "auto")
            runtime_backend = ""
            adapter_name = ""
            if backend_family == "mlx":
                runtime_backend = "turbofieldfare" if model_family == "gemma4" else "omlx_mlx_lm"
                adapter_name = "gemma4_a4b" if runtime_backend == "turbofieldfare" and model_family == "gemma4" else ""
            model_family = "gemma4" if "gemma4" in str(runtime_unit_plan.get("family") or self.model_path).lower() else ""
            model_format = "gguf" if str(self.model_path or "").lower().endswith(".gguf") else ""
            runtime_request_payload = build_unified_runtime_ir_v0(
                request_id=str(runtime_unit_plan.get("frontier_key") or ""),
                runtime_unit_plan=runtime_unit_plan,
                model_id=str(runtime_unit_plan.get("model") or self.model_path or ""),
                model_family=model_family or str(runtime_unit_plan.get("family") or ""),
                model_format=model_format,
                architecture="moe_decoder" if model_family == "gemma4" else "",
                runtime_mode="local_verify_loop",
                execution_intent="streaming_decode",
                backend_family=backend_family,
                runtime_backend=runtime_backend,
                adapter_name=adapter_name,
                device_class="apple_silicon",
                platform="macos",
                strategy_family="verify_loop",
                speculative_mode="mtp" if num_draft > 0 else "none",
                max_tokens=max_tokens,
                stream=False,
                residency_policy_family="tiered_streaming" if bool(runtime_unit_plan.get("enabled")) else "bypass",
                target_tier="ram",
                prefetch_semantics="best_effort" if bool(runtime_unit_plan.get("enabled")) else "noop",
                bootstrap_semantics="decode_preprime" if bool(runtime_unit_plan.get("enabled")) else "none",
                required_capabilities=["streaming_expert_units"] if bool(runtime_unit_plan.get("enabled")) else [],
                optional_capabilities=["decode_preprime", "runtime_unit_plan"],
            )
        runtime_request = self._loop.begin_request(runtime_request_payload)

        if num_draft == 0:
            # 纯 baseline, 无 MTP
            import time
            t0 = time.time()
            output = self._loop.llm.create_completion(
                prompt, max_tokens=max_tokens, temperature=0, top_p=1
            )
            dt = time.time() - t0
            return {
                "text": output["choices"][0]["text"],
                "tps": max_tokens / dt if dt > 0 else 0,
                "accept_rate": 0,
                "avg_accept_len": 0,
                "total_tokens": max_tokens,
                "prefill_ms": 0,
                "draft_ms": 0,
                "verify_ms": dt * 1000,
                "mode": "edge_baseline",
                "runtime_request": runtime_request,
            }

        # MTP verify loop
        text, stats = self._loop.generate_text(
            prompt, max_tokens=max_tokens, num_draft=num_draft
        )

        return {
            "text": text,
            "tps": stats.tps,
            "accept_rate": stats.accept_rate,
            "avg_accept_len": stats.avg_accept_len,
            "total_tokens": stats.total_tokens,
            "prefill_ms": stats.prefill_ms,
            "draft_ms": stats.draft_ms_total,
            "verify_ms": stats.verify_ms_total,
            "mode": "edge_draft",
            "runtime_request": runtime_request,
        }

    def bench(self, prompt: str, max_tokens: int = 50, num_draft: int = 4) -> dict:
        """Benchmark verify loop (打印详细统计)."""
        self._ensure_loop()
        return self._loop.bench(prompt, max_tokens=max_tokens, num_draft=num_draft)

    @staticmethod
    def is_available(model_path: str = "", mtp_checkpoint: str = "") -> bool:
        """检查 verify loop 是否可用 (依赖检查)."""
        try:
            import llama_cpp
            import torch
            if model_path:
                import os
                if not os.path.exists(model_path):
                    return False
            if mtp_checkpoint:
                import os
                if not os.path.exists(mtp_checkpoint):
                    return False
            return True
        except ImportError:
            return False


# ============================================================================
# 7. HermesRouter — 路由决策核心
# ============================================================================
class HermesRouter:
    """Hermes 认知路由 v3 — 完整版.

    路由决策流程:
      1. 缓存命中 → CACHE_HIT (TTFT 0-2ms)
      2. 离线 → LOCAL_ONLY
      3. ProfileBinding.can_run_on_edge == True → EDGE_DRAFT (省云端算力)
      4. ProfileBinding.can_run_on_edge == False → CLOUD_MTP (云端 NEXTN MTP)
      5. MTP 降级 (accept_rate < threshold) → CLOUD_ONLY
      6. MTP 不可用 → CLOUD_ONLY

    用法:
        bootstrap = Bootstrap(cloud_mtp_url="http://localhost:30001")
        result = bootstrap.run()
        router = HermesRouter(bootstrap)
        decision = router.decide(model_name="gemma4", prompt="def hello():", cache_hit=False)
    """

    def __init__(
        self,
        bootstrap: Optional[Bootstrap] = None,
        cloud_mtp_url: str = "http://localhost:30001",
        cloud_plain_url: str = "http://localhost:30000",
        mtp_degraded_threshold: float = 0.40,
    ):
        """初始化 Hermes 路由.

        Args:
            bootstrap: 已完成的 Bootstrap (含 SystemProfile + ProfileBindings)
            cloud_mtp_url: sglang NEXTN MTP endpoint
            cloud_plain_url: sglang plain endpoint (fallback)
            mtp_degraded_threshold: MTP accept rate 低于此值 → 降级
        """
        if bootstrap:
            self.bootstrap = bootstrap
            self.system_profile = bootstrap.system_profile
            self.bindings = bootstrap.bindings
            self.state_abi = bootstrap.state_abi
        else:
            self.bootstrap = None
            self.system_profile = SystemProfile.detect()
            self.bindings = {}
            self.state_abi = StateABI()
            self.state_abi.transition("READY", "默认初始化 (无 bootstrap)")

        self.cloud_mtp_url = cloud_mtp_url
        self.cloud_plain_url = cloud_plain_url
        self.mtp_degraded_threshold = mtp_degraded_threshold
        self._d5 = D5ContentAware()
        self._verify_backends: dict[str, VerifyLoopBackend] = {}

        self._stats = {
            "total": 0,
            "cache_hit": 0,
            "edge_draft": 0,
            "cloud_mtp": 0,
            "cloud_only": 0,
            "local_only": 0,
            "total_latency_ms": 0.0,
        }

    def _build_matrix_v2(
        self,
        *,
        model_name: str,
        prompt: str,
        binding: ProfileBinding,
        accept_rate: float,
        cache_hit: bool,
        route_context: Optional[dict[str, Any]] = None,
        expert_runtime: Optional[dict[str, Any]] = None,
    ) -> route_decision_v2_api.FourDMatrixV2:
        """构建 RouteDecisionV2 对应的 FeatureSchema 输入.

        expert_runtime: 来自 FullExpertDataPlaneManager.runtime_snapshot() 的真实
        expert 缓存/预取/路由指标. 若提供则覆盖 accept_rate 占位, 让 Hermes 真正
        感知 colibri streamed-MoE 运行态.
        """
        ex = expert_runtime or {}

        # 优先用 expert_data_plane 真实指标, 缺失才退回 accept_rate 占位
        expert_hit_rate = float(ex.get("cache_hit_rate", 0.0) or 0.0)
        prefetch_hit_rate = float(ex.get("prefetch_hit_rate", 0.0) or 0.0)
        hot_ratio = (
            float(ex.get("pinned_count", 0)) / max(int(ex.get("resident_count", 0)) or 1, 1)
            if ex.get("resident_count") else 0.0
        )
        warm_pin_gb = float(ex.get("pinned_gb", 0.0) or 0.0)
        resident_gb = float(ex.get("resident_gb", 0.0) or 0.0)
        ram_budget_gb = float(ex.get("ram_budget_gb", 0.0) or 0.0)
        pin_budget_gb = float(ex.get("pin_budget_gb", 0.0) or 0.0)
        # 预测冷读: 当前 plan 的 cold_bytes 占比, 缺失则按 1-hit_rate 估算
        last_plan = ex.get("last_plan") or {}
        cold_bytes_mb = float(last_plan.get("cold_bytes", 0) or 0) / 1024.0 / 1024.0
        if cold_bytes_mb <= 0.0:
            cold_bytes_mb = 0.0 if cache_hit else round(max(0.0, 1.0 - expert_hit_rate) * 128, 3)
        bytes_to_read_mb = cold_bytes_mb * 2.0  # 粗估: cold + prefetch window

        dynamic_heat = {
            "expert_hit_rate_ema": round(expert_hit_rate if expert_hit_rate > 0 else accept_rate, 3),
            "hot_expert_ratio": round(hot_ratio if hot_ratio > 0 else accept_rate, 3),
            "recent_expert_heat_entropy": 0.0,
            "layer_hotness_topk": [
                {"key": e.get("key"), "layer": e.get("layer_id"), "hot": e.get("hot_score")}
                for e in (ex.get("top_hot_experts") or [])[:5]
            ],
            "warm_pin_gb": round(warm_pin_gb, 3),
            "repin_recent_count": int(ex.get("promotions", 0) or 0),
            "prefetch_hit_rate_ema": round(prefetch_hit_rate if prefetch_hit_rate > 0 else accept_rate, 3),
            "predicted_cold_bytes_mb": round(cold_bytes_mb, 3),
            "predicted_bytes_to_read_mb": round(bytes_to_read_mb, 3),
        }

        # residency 判定: 用真实 resident/pin 占用 vs budget
        if resident_gb <= 0.0:
            # expert_data_plane 未启用或无 catalog
            predicted_mode = "hybrid_tier" if binding.can_run_on_edge else "streamed"
            can_partial = bool(binding.can_run_on_edge)
        elif warm_pin_gb >= pin_budget_gb * 0.8 and resident_gb >= ram_budget_gb * 0.8:
            # pin 和 resident 都接近预算上限 → 视为 warm_resident
            predicted_mode = "warm_resident"
            can_partial = True
        elif resident_gb > 0:
            # 有 resident 但未满 → hybrid_tier
            predicted_mode = "hybrid_tier"
            can_partial = True
        else:
            predicted_mode = "streamed"
            can_partial = bool(binding.can_run_on_edge)

        # I/O overlap gain 估算: prefetch 命中率 × route_agree (路由稳定才好 overlap)
        route_agree = float(ex.get("route_agree", 1.0) or 1.0)
        io_overlap_gain = round(prefetch_hit_rate * route_agree * 0.5, 3)

        storage_runtime = {
            "dense_resident_tier": "memory",
            "expert_resident_tier": "mixed" if resident_gb > 0 else ("mixed" if binding.can_run_on_edge else "unknown"),
            "predicted_residency_mode": predicted_mode,
            "can_partial_resident": can_partial,
            "nvme_bw_gbps": float(self.system_profile.nvme_bw_gbps or 0.0) if self.system_profile else 0.0,
            "io_queue_depth": int(ex.get("prefetch_inflight", 0) or 0),
            "secondary_nvme_available": bool(self.system_profile.secondary_nvme_available) if self.system_profile else False,
            "disk_mirror_mode": "mirror" if (self.system_profile and self.system_profile.secondary_nvme_available) else "none",
            "multi_store_read_gain_estimate": round(0.5 if (self.system_profile and self.system_profile.secondary_nvme_available) else 0.0, 3),
            "io_compute_overlap_gain": io_overlap_gain,
        }
        verify_cost_ms = float(self.system_profile.rtt_ms or 0.0) + 10.0
        draft_cost_ms = float(binding.estimated_edge_ttft_ms or 0.0)
        speculation_roi = {
            "accept_rate_ema": round(accept_rate, 3),
            "verify_cost_ms": verify_cost_ms,
            "draft_cost_ms": draft_cost_ms,
            "recent_speculation_roi": round(((verify_cost_ms - draft_cost_ms) / max(verify_cost_ms, 1.0)), 3)
            if draft_cost_ms > 0
            else 0.0,
            "recent_json_success_rate": 1.0,
            "grammar_accept_rate_ema": round(accept_rate, 3),
            "grammar_mode_roi": 0.0,
        }
        transport_features = {
            k: v for k, v in (route_context or {}).items()
            if k in route_decision_v2_api.TransportRouteFeatures.__dataclass_fields__
        }
        memory_pressure = {
            k: v for k, v in (route_context or {}).items()
            if k in route_decision_v2_api.MemoryPressureFeatures.__dataclass_fields__
        }
        model_info = SimpleNamespace(
            name=model_name,
            params_b=0.0,
            num_layers=0,
            is_moe=bool((route_context or {}).get("moe_candidate", False)),
            num_experts=0,
            experts_per_tok=0,
            hidden_size=0,
            vocab_size=0,
            quantization=(binding.selected_quant or "bf16").lower(),
            model_size_gb=float(binding.draft_model_size_gb or 0.0),
            per_layer_gb=0.0,
            has_native_mtp=bool(binding.draft_model_path),
            draft_model_path=str(binding.draft_model_path or ""),
        )
        return route_decision_v2_api.FourDMatrixV2.from_hardware_model(
            self.system_profile,
            model_info,
            prompt=prompt,
            history_accept_rate=accept_rate,
            cache_hit_rate=1.0 if cache_hit else 0.0,
            dynamic_heat=dynamic_heat,
            storage_runtime=storage_runtime,
            speculation_roi=speculation_roi,
            transport_route=transport_features,
            memory_pressure=memory_pressure,
        )

    @staticmethod
    def _to_route_decision_v2(
        decision: D4Decision,
        matrix_v2: route_decision_v2_api.FourDMatrixV2,
    ) -> route_decision_v2_api.RouteDecisionV2:
        """把 Hermes 原生 D4Decision 映射到 RouteDecisionV2."""
        mode_map = {
            "edge_draft": "edge_draft_cloud_verify",
            "cloud_mtp": "cloud_only",
            "cloud_only": "cloud_only",
            "local_only": "local_only",
            "cache_hit": "cache_hit",
        }
        response_contract = matrix_v2.request_contract.response_contract_hint
        grammar_mode = "json" if response_contract == "json" else ("tool_call" if response_contract == "tool" else "off")
        fallback_policy = "disable_speculation" if decision.use_mtp and decision.expected_accept_rate < 0.4 else ("plain_mtp" if decision.use_mtp else "cloud_only")
        return route_decision_v2_api.RouteDecisionV2(
            mode=mode_map.get(decision.mode, "cloud_only"),
            draft_n_tokens=getattr(decision, "num_draft", 0) if decision.use_mtp else 0,
            pivot_layer=0,
            use_flashmoe=bool(matrix_v2.D3.is_moe),
            draft_model_path=matrix_v2.D3.draft_model_path,
            confidence=decision.confidence,
            reason=decision.reason,
            expected_ttft_ms=decision.expected_ttft_ms,
            expected_decode_tps=decision.expected_decode_tps,
            expected_accept_rate=decision.expected_accept_rate,
            speculation_expected_roi=matrix_v2.speculation_roi.recent_speculation_roi,
            residency_policy=matrix_v2.storage_runtime.predicted_residency_mode,
            prefetch_policy="aggressive" if matrix_v2.heat.prefetch_hit_rate_ema >= 0.3 else "conservative",
            streaming_policy="overlap_io_compute" if matrix_v2.storage_runtime.can_partial_resident else "buffered",
            fallback_policy=fallback_policy,
            response_contract=response_contract,
            grammar_mode=grammar_mode,
        )

    @staticmethod
    def _merge_route_context_v2(
        decision_v2: route_decision_v2_api.RouteDecisionV2,
        route_context: Optional[dict[str, Any]],
    ) -> route_decision_v2_api.RouteDecisionV2:
        """把 edge_first 的四态 transport route 合并进 Hermes v2 决策."""
        if not route_context:
            return decision_v2
        route_mode = str(route_context.get("mode") or route_context.get("mode_hint") or "")
        route_reason = str(route_context.get("mode_switch_reason") or route_context.get("reason") or "")
        route_p = int(route_context.get("P") or 0)
        route_mac_time_est = float(route_context.get("mac_time_est") or 0.0)
        memory_pressure = str(route_context.get("memory_pressure") or "unknown")
        sticky_active = bool(route_context.get("sticky_active"))
        degrade_target_mode = str(route_context.get("degrade_target_mode") or "")
        degrade_suggested = bool(route_context.get("degrade_suggested"))

        reason_suffix = f"transport={route_mode}:{route_reason}".strip(":")
        if degrade_suggested:
            reason_suffix += f" | memory_pressure={memory_pressure}"
            if degrade_target_mode:
                reason_suffix += f" | degrade_target={degrade_target_mode}"
        if sticky_active:
            reason_suffix += " | sticky_window=active"

        if route_mode == "local_full":
            decision_v2.mode = "local_full"
            decision_v2.pivot_layer = 0
            decision_v2.expected_ttft_ms = route_mac_time_est or decision_v2.expected_ttft_ms
            decision_v2.fallback_policy = "cloud_only"
            decision_v2.reason = f"{decision_v2.reason} | {reason_suffix}".strip()
        elif route_mode == "layer_split_pd":
            decision_v2.mode = "layer_split_pd"
            decision_v2.pivot_layer = max(route_p, 0)
            decision_v2.reason = f"{decision_v2.reason} | {reason_suffix}".strip()
        elif route_mode == "cloud_pd":
            decision_v2.mode = "cloud_pd"
            decision_v2.pivot_layer = 0
            decision_v2.fallback_policy = "cloud_only"
            decision_v2.reason = f"{decision_v2.reason} | {reason_suffix}".strip()
        elif route_mode == "cloud_fallback":
            decision_v2.mode = "cloud_fallback"
            decision_v2.pivot_layer = 0
            decision_v2.fallback_policy = "cloud_only"
            decision_v2.reason = f"{decision_v2.reason} | {reason_suffix}".strip()
        return decision_v2

    def decide(
        self,
        model_name: str = "gemma4",
        prompt: str = "",
        cache_hit: bool = False,
        online: Optional[bool] = None,
        mtp_available: Optional[bool] = None,
        mtp_accept_rate: Optional[float] = None,
    ) -> D4Decision:
        """路由决策 — 十步流水线 + 4D 矩阵 + ProfileBinding.

        Args:
            model_name: 模型注册名 (gemma4 / dsv4 / qwen3vl)
            prompt: 请求 prompt
            cache_hit: 是否缓存命中
            online: 是否在线 (None → 从 SystemProfile 判断)
            mtp_available: 云端 MTP 是否可用 (None → 从 Bootstrap 判断)
            mtp_accept_rate: 最近 MTP accept rate (None → 从 ProfileBinding 判断)

        Returns:
            D4Decision: 路由决策
        """
        t0 = time.time()
        self._stats["total"] += 1

        # 获取 ProfileBinding
        binding = self.bindings.get(model_name)
        if binding is None:
            # 没有预评估的 binding, 创建默认
            binding = ProfileBinding(
                model_name=model_name,
                can_run_on_edge=False,
                reason="未评估的模型, 默认云端",
            )

        # 状态: ACTIVE
        self.state_abi.transition("ACTIVE", f"请求: {model_name}")

        # 网络/云端状态
        is_online = online if online is not None else (self.system_profile.online if self.system_profile else True)
        cloud_mtp_ok = mtp_available if mtp_available is not None else (self.bootstrap.result.cloud_reachable if self.bootstrap and hasattr(self.bootstrap, 'result') else True)
        accept_rate = mtp_accept_rate if mtp_accept_rate is not None else (
            binding.estimated_cloud_mtp_accept_rate if binding.estimated_cloud_mtp_accept_rate > 0 else 0.90
        )

        # 构建 4D 矩阵
        matrix = FourDMatrix.from_system_profile(
            profile=self.system_profile or SystemProfile(),
            model_name=model_name,
            draft_model_path=binding.draft_model_path,
            draft_architecture=binding.draft_architecture,
            draft_model_size_gb=binding.draft_model_size_gb,
            draft_params_m=binding.draft_params_m,
            prompt=prompt,
        )

        # 执行十步流水线 (内部使用)
        pipeline = TenStepPipeline()
        pipeline.execute(
            profile=self.system_profile or SystemProfile(),
            binding=binding,
            prompt=prompt,
            verbose=False,
        )

        # === D5 内容感知 ===
        d5 = self._d5.analyze(prompt)

        # D5 调整 accept_rate: 基于内容类型
        if mtp_accept_rate is None:
            # 无外部传入 → 用 D5 预估 + binding 基线
            base_accept = accept_rate
            if d5.prompt_type == "code_completion":
                accept_rate = min(base_accept + 0.15, 0.95)
            elif d5.prompt_type == "code_generation":
                accept_rate = min(base_accept + 0.05, 0.85)
            elif d5.prompt_type == "reasoning":
                accept_rate = max(base_accept - 0.30, 0.15)

        # D5 模式覆盖: 推理类 → 跳过 MTP
        d5_override = d5.suggested_mode_override

        # === 路由决策 ===

        # 1. 缓存命中
        if cache_hit:
            decision = D4Decision(
                mode="cache_hit",
                confidence=1.0,
                reason="L1-L5 缓存命中",
                expected_ttft_ms=0.3,
                expected_decode_tps=9999,
                cloud_url="",
                use_mtp=False,
            )
            self.state_abi.transition("CACHE_HIT", "缓存命中")
            self._stats["cache_hit"] += 1

        # 2. 离线
        elif not is_online:
            decision = D4Decision(
                mode="local_only",
                confidence=0.95,
                reason="离线模式: 本地推理",
                expected_ttft_ms=500,
                expected_decode_tps=26,
                cloud_url="",
                use_mtp=False,
            )
            self.state_abi.transition("LOCAL_ONLY", "离线")
            self._stats["local_only"] += 1

        # 3. 端侧能跑 draft → edge_draft (Hermes 调动端侧 draft model, 省云端算力!)
        elif binding.can_run_on_edge and not d5_override:
            savings = binding.cloud_compute_savings_pct
            backend = binding.edge_backend or "mlx"
            quant = binding.selected_quant or "default"
            # D5: 根据内容类型选择 num_draft
            num_draft = d5.suggested_num_draft if d5.suggested_num_draft > 0 else 2
            decision = D4Decision(
                mode="edge_draft",
                confidence=0.92,
                reason=(
                    f"Hermes 调动端侧 {backend} [{quant}] draft decode (省云端 {savings:.0%} 算力) "
                    f"+ 云端 verify — D5: {d5.prompt_type} (nd={num_draft}, accept~{accept_rate:.0%}) "
                    f"— {binding.reason}"
                ),
                expected_ttft_ms=binding.estimated_edge_ttft_ms,
                expected_decode_tps=binding.estimated_edge_tps * (1 + accept_rate * num_draft),
                expected_accept_rate=accept_rate,
                cloud_compute_savings_pct=savings,
                cloud_url=self.cloud_mtp_url,
                use_mtp=True,
                selected_quant=quant,
                edge_backend=backend,
            )
            # 附加 D5 信息到 decision (动态属性)
            decision.d5_content = d5
            decision.num_draft = num_draft
            self.state_abi.transition("EDGE_DECODE", f"端侧 {backend} [{quant}] draft (省 {savings:.0%}, D5: {d5.prompt_type})")
            self._stats["edge_draft"] += 1

        # 3.5 D5 override: 推理类 → 跳过 MTP, 直连云端
        elif d5_override == "cloud_only" and is_online:
            decision = D4Decision(
                mode="cloud_only",
                confidence=0.88,
                reason=(
                    f"D5 内容感知: {d5.prompt_type} → 跳过 MTP (推理类不可预测, accept~{accept_rate:.0%})"
                ),
                expected_ttft_ms=(self.system_profile.rtt_ms if self.system_profile else 50) + 10,
                expected_decode_tps=273,
                cloud_url=self.cloud_plain_url,
                use_mtp=False,
            )
            decision.d5_content = d5
            self.state_abi.transition("CLOUD_ONLY", f"D5: {d5.prompt_type} → cloud_only")
            self._stats["cloud_only"] += 1

        # 4. 端侧跑不动 + MTP 可用 + accept rate 良好 → cloud_mtp
        elif cloud_mtp_ok and accept_rate >= self.mtp_degraded_threshold:
            decision = D4Decision(
                mode="cloud_mtp",
                confidence=0.90,
                reason=(
                    f"云端 NEXTN MTP (accept={accept_rate:.0%}) "
                    f"— 端侧不支持: {binding.reason}"
                ),
                expected_ttft_ms=(self.system_profile.rtt_ms if self.system_profile else 50) + 10,
                expected_decode_tps=binding.estimated_cloud_mtp_tps or 273,
                expected_accept_rate=accept_rate,
                cloud_compute_savings_pct=0,
                cloud_url=self.cloud_mtp_url,
                use_mtp=True,
            )
            self.state_abi.transition("CLOUD_MTP", f"云端 MTP (accept={accept_rate:.0%})")
            self._stats["cloud_mtp"] += 1

        # 5. MTP 降级 → cloud_only
        elif cloud_mtp_ok and accept_rate < self.mtp_degraded_threshold:
            decision = D4Decision(
                mode="cloud_only",
                confidence=0.85,
                reason=f"MTP 降级 (accept={accept_rate:.0%} < {self.mtp_degraded_threshold:.0%})",
                expected_ttft_ms=(self.system_profile.rtt_ms if self.system_profile else 50) + 10,
                expected_decode_tps=273,
                cloud_url=self.cloud_plain_url,
                use_mtp=False,
            )
            self.state_abi.transition("CLOUD_ONLY", "MTP 降级")
            self._stats["cloud_only"] += 1

        # 6. MTP 不可用 → cloud_only
        else:
            decision = D4Decision(
                mode="cloud_only",
                confidence=0.90,
                reason="MTP 不可用: 直连云端",
                expected_ttft_ms=(self.system_profile.rtt_ms if self.system_profile else 50) + 10,
                expected_decode_tps=273,
                cloud_url=self.cloud_plain_url,
                use_mtp=False,
            )
            self.state_abi.transition("CLOUD_ONLY", "MTP 不可用")
            self._stats["cloud_only"] += 1

        # 回到 READY
        self.state_abi.transition("READY", "请求完成")

        elapsed_ms = (time.time() - t0) * 1000
        self._stats["total_latency_ms"] += elapsed_ms

        logger.info(
            f"[hermes] mode={decision.mode} conf={decision.confidence:.2f} "
            f"({elapsed_ms:.3f}ms) — {decision.reason}"
        )

        return decision

    def decide_v2(
        self,
        model_name: str = "gemma4",
        prompt: str = "",
        cache_hit: bool = False,
        online: Optional[bool] = None,
        mtp_available: Optional[bool] = None,
        mtp_accept_rate: Optional[float] = None,
        route_context: Optional[dict[str, Any]] = None,
        expert_runtime: Optional[dict[str, Any]] = None,
    ) -> tuple[route_decision_v2_api.FourDMatrixV2, route_decision_v2_api.RouteDecisionV2]:
        """Hermes v2 输出: FeatureSchema + RouteDecisionV2.

        expert_runtime: 来自 FullExpertDataPlaneManager.runtime_snapshot() 的真实
        expert 缓存/预取/路由指标. 若提供, colibri 相关的 heat/storage/residency
        字段将基于真实运行态而非 accept_rate 占位.
        """
        binding = self.bindings.get(model_name)
        if binding is None:
            binding = ProfileBinding(
                model_name=model_name,
                can_run_on_edge=False,
                reason="未评估的模型, 默认云端",
            )
        accept_rate = mtp_accept_rate if mtp_accept_rate is not None else (
            binding.estimated_cloud_mtp_accept_rate if binding.estimated_cloud_mtp_accept_rate > 0 else 0.90
        )
        matrix_v2 = self._build_matrix_v2(
            model_name=model_name,
            prompt=prompt,
            binding=binding,
            accept_rate=accept_rate,
            cache_hit=cache_hit,
            route_context=route_context,
            expert_runtime=expert_runtime,
        )
        decision = self.decide(
            model_name=model_name,
            prompt=prompt,
            cache_hit=cache_hit,
            online=online,
            mtp_available=mtp_available,
            mtp_accept_rate=mtp_accept_rate,
        )
        decision_v2 = self._to_route_decision_v2(decision, matrix_v2)
        decision_v2 = self._merge_route_context_v2(decision_v2, route_context)
        matrix_v2.D4 = decision_v2
        return matrix_v2, decision_v2

    def _get_verify_backend(self, model_name: str) -> Optional[VerifyLoopBackend]:
        """获取或创建模型的 verify loop 后端.

        从 DRAFT_MODELS 的 verify_loop_config 读取配置,
        创建 VerifyLoopBackend 单例.
        """
        if model_name in self._verify_backends:
            return self._verify_backends[model_name]

        model_info = Bootstrap.DRAFT_MODELS.get(model_name)
        if not model_info:
            return None

        vl_config = model_info.get("verify_loop_config")
        if not vl_config:
            return None

        import os
        repo_root = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", ".."
        ))

        model_path = str(vl_config.get("model_path") or model_info["draft_model_path"] or "")
        if not os.path.isabs(model_path):
            model_path = os.path.join(repo_root, model_path)

        ckpt = vl_config["mtp_checkpoint"]
        if not os.path.isabs(ckpt):
            ckpt = os.path.join(repo_root, ckpt)

        embed = vl_config.get("embed_head_path", "")
        if embed and not os.path.isabs(embed):
            embed = os.path.join(repo_root, embed)

        backend = VerifyLoopBackend(
            model_path=model_path,
            mtp_checkpoint=ckpt,
            embed_head_path=embed,
            hidden_size=model_info["draft_hidden_size"],
            vocab_size=model_info["draft_vocab_size"],
            num_heads=vl_config.get("num_heads", 14),
            head_dim=vl_config.get("head_dim", 64),
            intermediate_size=vl_config.get("intermediate_size", 4864),
            n_ctx=vl_config.get("n_ctx", 2048),
            use_mlx=vl_config.get("use_mlx", False),
            use_cgc_ir=vl_config.get("use_cgc_ir", False),
            use_ggml=vl_config.get("use_ggml", False),
        )
        self._verify_backends[model_name] = backend
        return backend

    def execute(
        self,
        decision: D4Decision,
        prompt: str,
        max_tokens: int = 100,
    ) -> dict:
        """执行路由决策 — 根据 decision.mode 分发到对应后端.

        Args:
            decision: decide() 返回的 D4Decision
            prompt: 输入 prompt
            max_tokens: 最大生成 token 数

        Returns:
            {
                "text": str,
                "mode": str,
                "tps": float,
                "accept_rate": float,
                "tokens": int,
                ...
            }
        """
        mode = decision.mode

        if mode == "cache_hit":
            return {
                "text": "",
                "mode": "cache_hit",
                "tps": 0,
                "accept_rate": 0,
                "tokens": 0,
                "reason": "Cache hit - no generation needed",
            }

        if mode == "edge_draft":
            # 端侧 verify loop
            model_name = getattr(decision, "model_name", "qwen25_05b")
            backend = self._get_verify_backend(model_name)
            if backend is None:
                # Fallback to cloud
                logger.warning(f"[hermes] No verify backend for {model_name}, falling back to cloud")
                return self._execute_cloud(decision, prompt, max_tokens)

            num_draft = getattr(decision, "num_draft", 2)
            try:
                result = backend.run(prompt, max_tokens=max_tokens, num_draft=num_draft)
                result["decision"] = decision.to_dict()
                return result
            except Exception as e:
                logger.error(f"[hermes] Verify loop failed: {e}, falling back to cloud")
                return self._execute_cloud(decision, prompt, max_tokens)

        elif mode in ("cloud_mtp", "cloud_only"):
            return self._execute_cloud(decision, prompt, max_tokens)

        elif mode == "local_only":
            # 离线模式: 用 verify loop 做 local inference (无 MTP)
            model_name = getattr(decision, "model_name", "qwen25_05b")
            backend = self._get_verify_backend(model_name)
            if backend is not None:
                return backend.run(prompt, max_tokens=max_tokens, num_draft=0)
            return {
                "text": "[offline mode: no local model available]",
                "mode": "local_only",
                "tps": 0,
                "accept_rate": 0,
                "tokens": 0,
            }

        else:
            return {
                "text": f"[unknown mode: {mode}]",
                "mode": mode,
                "tps": 0,
                "accept_rate": 0,
                "tokens": 0,
            }

    def _execute_cloud(
        self,
        decision: D4Decision,
        prompt: str,
        max_tokens: int,
    ) -> dict:
        """执行云端推理 (cloud_mtp / cloud_only)."""
        import urllib.request
        import json as _json

        url = decision.cloud_url or self.cloud_mtp_url
        payload = _json.dumps({
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }).encode("utf-8")

        try:
            req = urllib.request.Request(
                url + "/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = _json.loads(resp.read().decode("utf-8"))
                text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                usage = data.get("usage", {})
                return {
                    "text": text,
                    "mode": decision.mode,
                    "tps": 0,  # cloud TPS not measured here
                    "accept_rate": decision.expected_accept_rate,
                    "tokens": usage.get("completion_tokens", 0),
                    "cloud_url": url,
                }
        except Exception as e:
            return {
                "text": f"[cloud error: {e}]",
                "mode": "error",
                "tps": 0,
                "accept_rate": 0,
                "tokens": 0,
                "error": str(e),
            }

    def decide_and_execute(
        self,
        prompt: str,
        model_name: str = "qwen25_05b",
        max_tokens: int = 100,
        cache_hit: bool = False,
    ) -> tuple[D4Decision, dict]:
        """一步完成: 路由决策 + 执行.

        用法:
            router = get_hermes_router()
            decision, result = router.decide_and_execute("def hello():", max_tokens=50)
            print(result["text"])
        """
        decision = self.decide(
            model_name=model_name,
            prompt=prompt,
            cache_hit=cache_hit,
        )
        # 附加 model_name 到 decision 供 execute() 使用
        decision.model_name = model_name
        result = self.execute(decision, prompt, max_tokens=max_tokens)
        return decision, result

    def get_binding(self, model_name: str) -> Optional[ProfileBinding]:
        """获取模型的 ProfileBinding."""
        return self.bindings.get(model_name)

    def get_state(self) -> dict:
        """获取当前状态 (State ABI)."""
        return self.state_abi.to_dict()

    def get_stats(self) -> dict:
        """路由统计."""
        total = self._stats["total"]
        return {
            **self._stats,
            "avg_latency_ms": round(self._stats["total_latency_ms"] / max(total, 1), 3),
            "state": self.state_abi.to_dict(),
        }


# ============================================================================
# 全局单例
# ============================================================================
_hermes_router: Optional[HermesRouter] = None


def get_hermes_router(
    cloud_mtp_url: str = "http://localhost:30001",
    cloud_plain_url: str = "http://localhost:30000",
) -> HermesRouter:
    """获取全局 HermesRouter 单例."""
    global _hermes_router
    if _hermes_router is None:
        bootstrap = Bootstrap(
            cloud_mtp_url=cloud_mtp_url,
            cloud_plain_url=cloud_plain_url,
        )
        bootstrap.run()
        _hermes_router = HermesRouter(bootstrap=bootstrap)
    return _hermes_router


def reset_hermes_router():
    """重置全局单例 (测试用)."""
    global _hermes_router
    _hermes_router = None


# ============================================================================
# 自测
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Hermes Router v4 — PlatformBenchmark + 4D矩阵 + 十步流水线 + State ABI")
    print("=" * 70)

    # 1. Bootstrap
    print("\n[1] Bootstrap 初始化...")
    bootstrap = Bootstrap(
        cloud_mtp_url="http://localhost:30001",
        cloud_plain_url="http://localhost:30000",
    )
    result = bootstrap.run(verbose=True)

    print(f"\n  Bootstrap: {'✅' if result.success else '❌'} ({result.elapsed_ms:.1f}ms)")
    print(f"  State: {result.state}")
    print(f"  Cloud MTP: {'✅' if result.cloud_reachable else '❌'}")

    # 2. ProfileBinding 结果
    print("\n[2] PlatformBenchmark 结果:")
    bench = result.platform_benchmark
    if bench:
        print(f"  MLX:       {'✅' if bench['mlx_available'] else '❌'} {bench['mlx_tps']:.1f} tok/s")
        print(f"  llama.cpp: {'✅' if bench['llamacpp_available'] else '❌'} {bench['llamacpp_tps']:.1f} tok/s")
        print(f"  首选后端:  {bench['preferred_backend']} (speedup {bench['speedup']:.1f}x)")

    print("\n[3] ProfileBinding 评估结果:")
    print(f"  {'Model':<20s} {'Edge?':>6s} {'Backend':>10s} {'Quant':>8s} {'Savings':>8s} {'Reason':<50s}")
    print(f"  {'-'*20} {'-'*6} {'-'*10} {'-'*8} {'-'*8} {'-'*50}")
    for name, binding in bootstrap.bindings.items():
        edge = "YES" if binding.can_run_on_edge else "NO"
        backend = binding.edge_backend or "—"
        quant = binding.selected_quant or "—"
        savings = f"{binding.cloud_compute_savings_pct:.0%}" if binding.can_run_on_edge else "N/A"
        print(f"  {binding.model_display_name:<20s} {edge:>6s} {backend:>10s} {quant:>8s} {savings:>8s} {binding.reason[:50]}")

    # 4. Hermes 路由决策
    print("\n[4] Hermes 路由决策 (7 场景):")
    router = HermesRouter(bootstrap=bootstrap)
    test_cases = [
        ("Gemma4 缓存命中", "gemma4", "def hello():", True, None, None, None),
        ("Gemma4 离线", "gemma4", "def hello():", False, False, None, None),
        ("Gemma4 正常", "gemma4", "def fibonacci(n):\n    return", False, None, True, 0.95),
        ("Gemma4 MTP降级", "gemma4", "def hello():", False, None, True, 0.30),
        ("DSV4 正常", "dsv4", "def hello():", False, None, True, 0.85),
        ("Qwen3-VL 正常", "qwen3vl", "def hello():", False, None, True, 0.80),
        ("Gemma4 MTP不可用", "gemma4", "def hello():", False, None, False, None),
    ]
    for name, model, prompt, cache, online, mtp_avail, accept in test_cases:
        decision = router.decide(
            model_name=model, prompt=prompt,
            cache_hit=cache, online=online,
            mtp_available=mtp_avail, mtp_accept_rate=accept,
        )
        print(f"\n  {name}:")
        print(f"    mode:     {decision.mode}")
        print(f"    confidence: {decision.confidence:.0%}")
        if decision.selected_quant:
            print(f"    quant:    {decision.selected_quant} ({decision.edge_backend})")
        print(f"    reason:   {decision.reason}")
        print(f"    TTFT:     {decision.expected_ttft_ms:.1f}ms")
        print(f"    decode:   {decision.expected_decode_tps:.1f} tok/s")
        print(f"    savings:  {decision.cloud_compute_savings_pct:.0%}")
        print(f"    cloud:    {decision.cloud_url or '(local)'}")

    # 5. State ABI
    print(f"\n[5] State ABI:")
    print(f"  Current: {router.state_abi.current}")
    print(f"  History: {len(router.state_abi.get_history())} transitions")
    print(f"  Last: {router.state_abi.get_history()[-1] if router.state_abi.get_history() else 'N/A'}")

    # 6. 十步流水线 (verbose)
    print(f"\n[6] 十步流水线 (Gemma4):")
    binding = bootstrap.bindings["gemma4"]
    pipeline = TenStepPipeline()
    pipeline.execute(bootstrap.system_profile, binding, "def hello():", verbose=True)

    # 7. 统计
    print(f"\n[7] 统计:")
    print(json.dumps(router.get_stats(), indent=2, ensure_ascii=False))
