"""跨平台硬件感知模块 (Mac/Windows/Linux).

检测:
  - 内存 (总/可用)
  - 芯片型号 + 算力等级
  - 磁盘空间
  - GPU (Apple Metal / NVIDIA / AMD)
  - 网络 RTT (ping cloud)

输出:
  HardwareInfo dataclass → 用于 4D 感知矩阵 + 路由决策
"""
from __future__ import annotations

import os
import platform
import subprocess
import shutil
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class HardwareInfo:
    """跨平台硬件信息."""
    # OS
    os_name: str = ""          # Darwin / Windows / Linux
    os_version: str = ""
    arch: str = ""             # arm64 / x86_64

    # CPU
    cpu_brand: str = ""        # Apple M4 / Intel i7 / AMD Ryzen 9
    cpu_cores: int = 0

    # 内存 (GB)
    total_mem_gb: float = 0.0
    available_mem_gb: float = 0.0

    # 磁盘 (GB)
    disk_total_gb: float = 0.0
    disk_available_gb: float = 0.0

    # GPU
    gpu_type: str = ""         # apple_metal / nvidia / amd / none
    gpu_name: str = ""         # Apple M4 (Metal) / NVIDIA RTX 4090
    gpu_vram_gb: float = 0.0   # 独显 VRAM (0 for unified memory)

    # 算力
    compute_tier: str = ""     # weak / medium / strong / ultra
    tflops: float = 0.0        # 估算算力

    # 网络 (需传入 cloud host)
    rtt_ms: float = 0.0        # ping RTT to cloud

    # 推理引擎
    recommended_engine: str = ""  # mlx / cuda / cpu / cloud

    def to_dict(self) -> dict:
        return asdict(self)


def detect_os() -> tuple[str, str, str]:
    """检测 OS + 架构."""
    os_name = platform.system()  # Darwin / Windows / Linux
    os_version = platform.release()
    arch = platform.machine()    # arm64 / x86_64
    return os_name, os_version, arch


def detect_memory(os_name: str) -> tuple[float, float]:
    """检测总内存 + 可用内存 (GB)."""
    try:
        if os_name == "Darwin":
            # macOS: vm_stat
            result = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5)
            lines = result.stdout.strip().split("\n")
            # page size
            page_size = 4096
            for line in lines:
                if "page size" in line.lower():
                    # "Mach virtual memory statistics: (page size of 4096 bytes)"
                    import re
                    m = re.search(r"page size of (\d+)", line)
                    if m:
                        page_size = int(m.group(1))
                    break

            free_pages = 0
            inactive_pages = 0
            active_pages = 0
            wired_pages = 0
            for line in lines:
                if "free" in line.lower() and ":" in line:
                    free_pages = int(line.split(":")[-1].strip().rstrip("."))
                elif "inactive" in line.lower() and ":" in line:
                    inactive_pages = int(line.split(":")[-1].strip().rstrip("."))
                elif "active" in line.lower() and ":" in line:
                    active_pages = int(line.split(":")[-1].strip().rstrip("."))
                elif "wired" in line.lower() and ":" in line:
                    wired_pages = int(line.split(":")[-1].strip().rstrip("."))

            total_pages = free_pages + inactive_pages + active_pages + wired_pages
            total_gb = total_pages * page_size / 1e9
            # 可用 = free + inactive (inactive 可被回收)
            available_gb = (free_pages + inactive_pages) * page_size / 1e9
            return round(total_gb, 1), round(available_gb, 1)

        elif os_name == "Linux":
            result = subprocess.run(["free", "-g"], capture_output=True, text=True, timeout=5)
            lines = result.stdout.strip().split("\n")
            if len(lines) >= 2:
                parts = lines[1].split()
                total_gb = float(parts[1])
                available_gb = float(parts[6]) if len(parts) > 6 else float(parts[3])
                return total_gb, available_gb

        elif os_name == "Windows":
            result = subprocess.run(
                ["wmic", "OS", "get", "TotalVisibleMemorySize,FreePhysicalMemory", "/format:value"],
                capture_output=True, text=True, timeout=5
            )
            total_kb = 0
            free_kb = 0
            for line in result.stdout.split("\n"):
                if line.startswith("TotalVisibleMemorySize="):
                    total_kb = int(line.split("=")[1])
                elif line.startswith("FreePhysicalMemory="):
                    free_kb = int(line.split("=")[1])
            return round(total_kb / 1e6, 1), round(free_kb / 1e6, 1)

    except Exception as e:
        print(f"  [hw] Memory detection error: {e}")

    return 0.0, 0.0


def detect_disk(path: str = "/") -> tuple[float, float]:
    """检测磁盘空间 (GB)."""
    try:
        usage = shutil.disk_usage(path if os.path.exists(path) else os.path.expanduser("~"))
        total_gb = usage.total / 1e9
        free_gb = usage.free / 1e9
        return round(total_gb, 1), round(free_gb, 1)
    except:
        return 0.0, 0.0


def detect_cpu(os_name: str) -> tuple[str, int]:
    """检测 CPU 型号 + 核心数."""
    try:
        cpu_brand = platform.processor() or "Unknown"
        cpu_cores = os.cpu_count() or 1

        if os_name == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5
            )
            if result.stdout.strip():
                cpu_brand = result.stdout.strip()
            # 核心数 (P+E)
            result = subprocess.run(
                ["sysctl", "-n", "hw.ncpu"],
                capture_output=True, text=True, timeout=5
            )
            if result.stdout.strip():
                cpu_cores = int(result.stdout.strip())

        elif os_name == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        cpu_brand = line.split(":")[1].strip()
                        break

        elif os_name == "Windows":
            result = subprocess.run(
                ["wmic", "cpu", "get", "Name,NumberOfCores", "/format:value"],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.split("\n"):
                if line.startswith("Name="):
                    cpu_brand = line.split("=")[1].strip()
                elif line.startswith("NumberOfCores="):
                    try:
                        cpu_cores = int(line.split("=")[1])
                    except:
                        pass

        return cpu_brand, cpu_cores
    except:
        return "Unknown", 1


def detect_gpu(os_name: str, arch: str) -> tuple[str, str, float]:
    """检测 GPU 类型 + 名称 + VRAM."""
    gpu_type = "none"
    gpu_name = ""
    gpu_vram = 0.0

    try:
        if os_name == "Darwin" and arch == "arm64":
            # Apple Silicon: Metal GPU
            gpu_type = "apple_metal"
            result = subprocess.run(
                ["system_profiler", "SPHardwareDataType"],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.split("\n"):
                if "Chip:" in line:
                    gpu_name = line.split(":")[1].strip()
                elif "Memory:" in line:
                    # "Memory: 16 GB"
                    import re
                    m = re.search(r"(\d+)", line)
                    if m:
                        gpu_vram = float(m.group(1))  # unified memory
            # Apple Silicon GPU 是 unified memory, VRAM = total mem
            # gpu_vram 在 detect_memory 里设置

        elif os_name == "Linux":
            # NVIDIA
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                gpu_type = "nvidia"
                parts = result.stdout.strip().split(",")
                gpu_name = parts[0].strip()
                if len(parts) > 1:
                    import re
                    m = re.search(r"(\d+)", parts[1])
                    if m:
                        gpu_vram = float(m.group(1))
            else:
                # AMD ROCm
                result = subprocess.run(
                    ["rocm-smi", "--showproductname", "--showmeminfo", "vram"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0 and result.stdout.strip():
                    gpu_type = "amd"
                    gpu_name = "AMD ROCm"

        elif os_name == "Windows":
            result = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "Name,AdapterRAM", "/format:value"],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.split("\n"):
                if line.startswith("Name=") and ("NVIDIA" in line or "AMD" in line):
                    gpu_name = line.split("=")[1].strip()
                    gpu_type = "nvidia" if "NVIDIA" in line else "amd"
                elif line.startswith("AdapterRAM="):
                    try:
                        gpu_vram = int(line.split("=")[1]) / 1e9
                    except:
                        pass

    except Exception as e:
        print(f"  [hw] GPU detection error: {e}")

    return gpu_type, gpu_name, gpu_vram


def classify_compute_tier(cpu_brand: str, gpu_type: str, gpu_vram: float, total_mem: float) -> tuple[str, float]:
    """分类算力等级 + 估算 TFLOPS."""
    cpu_lower = cpu_brand.lower()

    # Apple Silicon
    if gpu_type == "apple_metal":
        if "ultra" in cpu_lower:
            return "ultra", 80.0
        elif "max" in cpu_lower:
            return "ultra", 60.0
        elif "pro" in cpu_lower:
            return "strong", 40.0
        elif "m4" in cpu_lower or "m3" in cpu_lower:
            return "medium", 30.0
        elif "m2" in cpu_lower:
            return "medium", 25.0
        elif "m1" in cpu_lower:
            return "weak", 15.0
        else:
            return "weak", 10.0

    # NVIDIA
    elif gpu_type == "nvidia":
        if gpu_vram >= 40:
            return "ultra", 100.0  # A100/H100
        elif gpu_vram >= 20:
            return "strong", 60.0  # RTX 4090
        elif gpu_vram >= 10:
            return "medium", 30.0  # RTX 3080
        else:
            return "weak", 15.0

    # AMD
    elif gpu_type == "amd":
        if gpu_vram >= 32:
            return "strong", 50.0
        elif gpu_vram >= 16:
            return "medium", 30.0
        else:
            return "weak", 15.0

    # CPU only
    else:
        if total_mem >= 32:
            return "weak", 5.0
        else:
            return "weak", 2.0


def recommend_engine(os_name: str, arch: str, gpu_type: str) -> str:
    """推荐推理引擎."""
    if gpu_type == "apple_metal":
        return "mlx"
    elif gpu_type == "nvidia":
        return "cuda"
    elif gpu_type == "amd":
        return "rocm"
    else:
        return "cpu"


def measure_rtt(host: str = "47.95.250.55", count: int = 3) -> float:
    """测量到 cloud 的 RTT (ms)."""
    try:
        os_name = platform.system()
        if os_name == "Darwin" or os_name == "Linux":
            result = subprocess.run(
                ["ping", "-c", str(count), "-W", "3", host],
                capture_output=True, text=True, timeout=10
            )
            # "round-trip min/avg/max/stddev = 55.123/55.456/55.789/0.123 ms"
            import re
            m = re.search(r"= ([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)", result.stdout)
            if m:
                return float(m.group(2))  # avg
        elif os_name == "Windows":
            result = subprocess.run(
                ["ping", "-n", str(count), host],
                capture_output=True, text=True, timeout=10
            )
            import re
            m = re.search(r"Average = (\d+)ms", result.stdout)
            if m:
                return float(m.group(1))
    except:
        pass
    return 55.0  # 默认


def detect_all(cloud_host: str = "47.95.250.55") -> HardwareInfo:
    """检测所有硬件信息 (十步流水线 Step 1-5.5)."""
    info = HardwareInfo()

    # Step 1: OS
    info.os_name, info.os_version, info.arch = detect_os()

    # Step 2: CPU
    info.cpu_brand, info.cpu_cores = detect_cpu(info.os_name)

    # Step 3: Memory
    info.total_mem_gb, info.available_mem_gb = detect_memory(info.os_name)

    # Step 4: Disk
    info.disk_total_gb, info.disk_available_gb = detect_disk()

    # Step 5: GPU
    info.gpu_type, info.gpu_name, info.gpu_vram = detect_gpu(info.os_name, info.arch)

    # Apple Silicon: VRAM = available unified memory
    if info.gpu_type == "apple_metal":
        info.gpu_vram_gb = info.available_mem_gb

    # Step 5.5: Compute tier
    info.compute_tier, info.tflops = classify_compute_tier(
        info.cpu_brand, info.gpu_type, info.gpu_vram, info.total_mem_gb
    )

    # Engine
    info.recommended_engine = recommend_engine(info.os_name, info.arch, info.gpu_type)

    # RTT
    info.rtt_ms = measure_rtt(cloud_host)

    return info


if __name__ == "__main__":
    print("=" * 60)
    print("跨平台硬件感知 (Mac/Windows/Linux)")
    print("=" * 60)

    info = detect_all()

    print(f"\n  OS: {info.os_name} {info.os_version} ({info.arch})")
    print(f"  CPU: {info.cpu_brand} ({info.cpu_cores} cores)")
    print(f"  Memory: {info.total_mem_gb}GB total, {info.available_mem_gb}GB available")
    print(f"  Disk: {info.disk_total_gb}GB total, {info.disk_available_gb}GB available")
    print(f"  GPU: {info.gpu_type} - {info.gpu_name} ({info.gpu_vram}GB)")
    print(f"  Compute: {info.compute_tier} ({info.tflops} TFLOPS)")
    print(f"  Engine: {info.recommended_engine}")
    print(f"  RTT: {info.rtt_ms}ms")

    print(f"\n  4D Matrix (hardware):")
    print(f"  {info.to_dict()}")
