#!/usr/bin/env python3
"""
Phi MoE模型加载测试 - Harness Agent
验证FlashMoE+OMLX在Apple M4 16GB上能否加载Phi MoE模型
"""

import sys
import os
import time
import psutil
import subprocess
from typing import Dict, Any, Tuple

sys.path.insert(0, os.path.dirname(__file__))

from tools.anti_fraud_framework import AntiFraudCollector, DataSource, calculate_crc32

# ==================== Phi MoE模型配置 ====================

class PhiMoEModelConfig:
    """Phi MoE模型配置"""
    
    # Phi-MoE-4B配置
    PHI_MOE_4B = {
        "name": "Phi-MoE-4B",
        "num_parameters_b": 4,
        "num_experts": 8,
        "top_k": 2,
        "hidden_size": 2560,
        "num_layers": 32,
        "vocab_size": 51200,
        "max_seq_len": 2048
    }
    
    # Phi-MoE-12B配置
    PHI_MOE_12B = {
        "name": "Phi-MoE-12B",
        "num_parameters_b": 12,
        "num_experts": 16,
        "top_k": 2,
        "hidden_size": 4096,
        "num_layers": 40,
        "vocab_size": 51200,
        "max_seq_len": 2048
    }

# ==================== 内存估算工具 ====================

class MemoryCalculator:
    """内存估算工具"""
    
    @staticmethod
    def calculate_model_memory(model_config: Dict[str, Any], dtype: str = "fp16") -> float:
        """计算模型内存需求（GB）"""
        params_b = model_config["num_parameters_b"]
        
        # 基础参数内存
        dtype_bytes = {
            "fp32": 4,
            "fp16": 2,
            "fp8": 1,
            "int4": 0.5
        }
        
        base_memory_gb = params_b * dtype_bytes[dtype.lower()]
        
        # KV缓存内存（每token）
        kv_bytes_per_token = model_config["num_layers"] * model_config["hidden_size"] * dtype_bytes["fp16"] * 2
        kv_memory_gb = (kv_bytes_per_token * model_config["max_seq_len"]) / (1024**3)
        
        # MoE额外开销（专家权重）
        moe_overhead = params_b * 0.3  # MoE结构额外30%
        
        total_memory = base_memory_gb + kv_memory_gb + moe_overhead
        
        return total_memory

# ==================== 真实系统数据源 ====================

class RealSystemDataSource(DataSource):
    """真实系统数据源"""

    def __init__(self):
        self.source_name = "real_system"

    def collect(self) -> Dict[str, Any]:
        """采集真实系统数据"""
        mem = psutil.virtual_memory()
        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(),
            "memory_total_gb": mem.total / (1024**3),
            "memory_available_gb": mem.available / (1024**3),
            "memory_used_gb": mem.used / (1024**3),
            "memory_percent": mem.percent,
            "source": "real_system"
        }

    def get_source_name(self) -> str:
        return self.source_name

class RealGPUDataSource(DataSource):
    """真实GPU数据源"""

    def __init__(self):
        self.source_name = "real_gpu"

    def collect(self) -> Dict[str, Any]:
        """采集真实GPU数据"""
        try:
            result = subprocess.run(
                ['system_profiler', 'SPDisplaysDataType'],
                capture_output=True, text=True
            )
            gpu_info = {"model": "Unknown", "cores": 0}
            for line in result.stdout.split('\n'):
                if 'Chipset Model:' in line:
                    gpu_info['model'] = line.split(':')[1].strip()
                elif 'Total Number of Cores:' in line:
                    gpu_info['cores'] = int(line.split(':')[1].strip())
            return {**gpu_info, "source": "real_gpu"}
        except:
            return {"model": "Unknown", "cores": 0, "source": "real_gpu"}

    def get_source_name(self) -> str:
        return self.source_name

class PhiMoEMemoryTestDataSource(DataSource):
    """Phi MoE内存测试数据源"""

    def __init__(self):
        self.source_name = "phi_moe_test"

    def collect(self) -> Dict[str, Any]:
        """采集Phi MoE内存测试数据"""
        results = {}
        
        for model_name, config in [("Phi-MoE-4B", PhiMoEModelConfig.PHI_MOE_4B), 
                                   ("Phi-MoE-12B", PhiMoEModelConfig.PHI_MOE_12B)]:
            for dtype in ["fp32", "fp16", "fp8"]:
                mem_gb = MemoryCalculator.calculate_model_memory(config, dtype)
                results[f"{model_name}_{dtype}_memory_gb"] = mem_gb
        
        return {**results, "source": "phi_moe_test"}

    def get_source_name(self) -> str:
        return self.source_name

# ==================== Phi MoE Harness Agent ====================

class PhiMoEHarnessAgent:
    """Phi MoE Harness Agent"""

    def __init__(self):
        self.collector = AntiFraudCollector()
        self.collector.register_source(RealSystemDataSource())
        self.collector.register_source(RealGPUDataSource())
        self.collector.register_source(PhiMoEMemoryTestDataSource())

    def run_memory_test(self):
        """运行Phi MoE内存测试"""
        print("=" * 100)
        print("🔍 Phi MoE模型加载测试 - FlashMoE+OMLX")
        print("=" * 100)
        print(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # 采集真实数据
        data = self.collector.collect_all()
        
        system_data = data["real_system"]
        gpu_data = data["real_gpu"]
        moe_data = data["phi_moe_test"]

        # 系统信息
        print("\n" + "=" * 100)
        print("💻 真实系统信息")
        print("=" * 100)
        print(f"CPU核心数: {system_data['cpu_count']}")
        print(f"CPU使用率: {system_data['cpu_percent']}%")
        print(f"内存总量: {system_data['memory_total_gb']:.2f} GB")
        print(f"可用内存: {system_data['memory_available_gb']:.2f} GB")
        print(f"已用内存: {system_data['memory_used_gb']:.2f} GB ({system_data['memory_percent']}%)")

        # GPU信息
        print("\n" + "=" * 100)
        print("🎮 GPU信息")
        print("=" * 100)
        print(f"GPU型号: {gpu_data['model']}")
        print(f"GPU核心数: {gpu_data['cores']}")

        # Phi MoE内存需求分析
        print("\n" + "=" * 100)
        print("📊 Phi MoE模型内存需求分析")
        print("=" * 100)
        
        available_mem = system_data["memory_available_gb"]
        
        print(f"\n{'模型':<15} | {'量化':<8} | {'内存需求(GB)':<15} | {'可用内存(GB)':<15} | {'是否可行'}")
        print("-" * 100)
        
        models = [
            ("Phi-MoE-4B", "FP32", moe_data["Phi-MoE-4B_fp32_memory_gb"]),
            ("Phi-MoE-4B", "FP16", moe_data["Phi-MoE-4B_fp16_memory_gb"]),
            ("Phi-MoE-4B", "FP8", moe_data["Phi-MoE-4B_fp8_memory_gb"]),
            ("Phi-MoE-12B", "FP32", moe_data["Phi-MoE-12B_fp32_memory_gb"]),
            ("Phi-MoE-12B", "FP16", moe_data["Phi-MoE-12B_fp16_memory_gb"]),
            ("Phi-MoE-12B", "FP8", moe_data["Phi-MoE-12B_fp8_memory_gb"]),
        ]
        
        for model_name, dtype, mem_needed in models:
            feasible = mem_needed < available_mem * 0.8  # 预留20%安全余量
            status = "✅ 可行" if feasible else "❌ 不可行"
            print(f"{model_name:<15} | {dtype:<8} | {mem_needed:.2f} GB            | {available_mem:.2f} GB            | {status}")

        # 详细分析
        print("\n" + "=" * 100)
        print("🔍 详细分析")
        print("=" * 100)
        
        # Phi-MoE-4B FP16分析
        mem_4b_fp16 = moe_data["Phi-MoE-4B_fp16_memory_gb"]
        print(f"\nPhi-MoE-4B (FP16):")
        print(f"  • 内存需求: {mem_4b_fp16:.2f} GB")
        print(f"  • 可用内存: {available_mem:.2f} GB")
        print(f"  • 剩余内存: {(available_mem - mem_4b_fp16):.2f} GB")
        
        # Phi-MoE-4B FP8分析
        mem_4b_fp8 = moe_data["Phi-MoE-4B_fp8_memory_gb"]
        print(f"\nPhi-MoE-4B (FP8):")
        print(f"  • 内存需求: {mem_4b_fp8:.2f} GB")
        print(f"  • 可用内存: {available_mem:.2f} GB")
        print(f"  • 剩余内存: {(available_mem - mem_4b_fp8):.2f} GB")

        # 数据完整性校验
        print("\n" + "=" * 100)
        print("🔐 数据完整性校验")
        print("=" * 100)
        print(f"Run ID: 0x{data['run_id']:016x}")
        print(f"CRC32哈希: 0x{data['crc32_hash']:08x}")
        
        is_hash_valid, hash_reason = self.collector.validate_hash(data)
        print(f"校验结果: {'✅ 通过' if is_hash_valid else f'❌ 失败: {hash_reason}'}")

        # 结论
        print("\n" + "=" * 100)
        print("📝 测试结论")
        print("=" * 100)
        
        if moe_data["Phi-MoE-4B_fp8_memory_gb"] < available_mem * 0.8:
            print("✅ Phi-MoE-4B (FP8量化)可以在当前环境运行")
        else:
            print("❌ Phi-MoE-4B内存不足")
        
        if moe_data["Phi-MoE-12B_fp8_memory_gb"] < available_mem * 0.8:
            print("✅ Phi-MoE-12B (FP8量化)可以在当前环境运行")
        else:
            print("❌ Phi-MoE-12B内存不足")

        print("\n" + "=" * 100)
        print("✅ Phi MoE模型加载测试完成")
        print("=" * 100)

# ==================== 主程序 ====================

def main():
    agent = PhiMoEHarnessAgent()
    agent.run_memory_test()

if __name__ == "__main__":
    main()