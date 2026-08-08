# Copyright (c) 2025 SandAI. All Rights Reserved.
"""
Smart Strategy Combiner - 智能策略组合引擎
基于场景(S0-S4) + 模型类型(LLM/VLM/MoE) 自动组合最优优化策略
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any

from .scenario_detector import ScenarioInfo

logger = logging.getLogger(__name__)


@dataclass
class StrategyCombination:
    """策略组合结果"""
    strategy_name: str
    scenario_id: str
    model_type: str
    optimizations: List[str]
    expected_speedup_x: float
    expected_memory_saving_gb: float
    description: str


class SmartStrategyCombiner:
    """智能策略组合引擎"""
    
    # 优化积木库
    OPTIMIZATION_BLOCKS = {
        "kda": "kda_attention",
        "pd_separate": "pd_separate_kv",
        "cudagraph": "cuda_graph",
        "mlx_jit": "mlx_jit_graph",
        "mtplx": "mtplx_apple_native_speculative",
        "dflash_mtp": "dflash_llm_special",
        "dual_gpu_parallel": "dual_gpu_tp2",
        "multi_gpu_3d": "multi_gpu_3d_parallel",
        "flash_moe": "flash_moe",
        "gds_spdk": "gds_spdk",
        "mtlheap": "mtlheap_kv_cache",
        "int4_quant": "int4_quantization",
        "tiling_128": "tiling_128x128",
        "tiling_64": "tiling_64x64",
    }
    
    # 策略组合矩阵表
    STRATEGY_MATRIX: Dict[str, Dict[str, Dict[str, Any]]] = {
        "s0": {  # 端侧纯本地 (Apple Silicon专属)
            "llm": {
                "optimizations": ["kda_attention", "pd_separate_kv", "mtlheap_kv_cache", "int4_quantization", "mlx_jit_graph", "mtplx_apple_native_speculative", "tiling_64x64"],
                "speedup": 5.0,
                "memory_save": 12.0,
                "description": "端侧LLM旗舰策略: KDA O(1)注意力 + PD分离KV + MTLHeap零拷贝 + 4bit量化 + MLX-JIT(替代CUDA Graph) + MTPLX Apple原生投机解码(2.0~2.3x加速)"
            },
            "vlm": {
                "optimizations": ["pd_separate_kv", "int4_quantization", "mlx_jit_graph", "tiling_64x64"],
                "speedup": 2.0,
                "memory_save": 8.0,
                "description": "端侧VLM专属策略: PD分离KV缓存 + 4bit量化 + MLX-JIT"
            },
            "moe": {
                "optimizations": ["pd_separate_kv", "int4_quantization", "mlx_jit_graph"],
                "speedup": 1.7,
                "memory_save": 15.0,
                "description": "端侧MoE轻量策略: PD分离KV + 4bit量化 + MLX-JIT"
            }
        },
        "s1": {  # 端云一体协同
            "llm": {
                "optimizations": ["kda_attention", "pd_separate_kv", "int4_quantization", "cuda_graph"],
                "speedup": 2.0,
                "memory_save": 15.0,
                "description": "端云一体LLM策略: 端侧KDA+PD分离，云侧CUDA Graph全图捕获"
            },
            "vlm": {
                "optimizations": ["pd_separate_kv", "int4_quantization", "cuda_graph"],
                "speedup": 1.8,
                "memory_save": 10.0,
                "description": "端云一体VLM策略: 图文分离流水线 + CUDA Graph"
            },
            "moe": {
                "optimizations": ["pd_separate_kv", "int4_quantization", "flash_moe", "gds_spdk"],
                "speedup": 2.2,
                "memory_save": 25.0,
                "description": "端云一体MoE策略: 云侧FlashMoE + GDS/SPDK SSD卸载"
            }
        },
        "s2": {  # 云双GPU并行
            "llm": {
                "optimizations": ["kda_attention", "cuda_graph", "dflash_mtp", "tiling_128x128"],
                "speedup": 3.0,
                "memory_save": 18.0,
                "description": "云双GPU LLM旗舰策略: KDA + CUDA Graph + DFlash&MTP语言模型专属 + TP=2双卡并行"
            },
            "vlm": {
                "optimizations": ["cuda_graph", "tiling_128x128"],
                "speedup": 2.2,
                "memory_save": 12.0,
                "description": "云双GPU VLM策略: 双卡专家并行 + CUDA Graph"
            },
            "moe": {
                "optimizations": ["flash_moe", "gds_spdk", "tiling_128x128"],
                "speedup": 2.8,
                "memory_save": 30.0,
                "description": "云双GPU MoE旗舰策略: FlashMoE 8专家Top2 + GDS/SPDK + TP=2双卡并行"
            }
        },
        "s3": {  # 多机分布式
            "llm": {
                "optimizations": ["kda_attention", "cuda_graph", "dflash_mtp", "multi_gpu_3d_parallel", "tiling_128x128"],
                "speedup": 4.5,
                "memory_save": 25.0,
                "description": "多机分布式LLM策略: DP+TP+PP三维并行全量组合"
            },
            "vlm": {
                "optimizations": ["cuda_graph", "multi_gpu_3d_parallel", "tiling_128x128"],
                "speedup": 3.5,
                "memory_save": 18.0,
                "description": "多机分布式VLM策略: 多模态图文三维并行流水线"
            },
            "moe": {
                "optimizations": ["flash_moe", "gds_spdk", "multi_gpu_3d_parallel", "tiling_128x128"],
                "speedup": 4.0,
                "memory_save": 40.0,
                "description": "多机分布式MoE策略: 分布式专家分片 + 三维并行 + GDS/SPDK"
            }
        },
        "s4": {  # 超大规模集群
            "llm": {
                "optimizations": ["kda_attention", "dflash_mtp", "multi_gpu_3d_parallel", "tiling_128x128"],
                "speedup": 6.0,
                "memory_save": 40.0,
                "description": "超大规模集群LLM终极策略: 万卡级DP+TP+PP调度 + DFlash&MTP全量加速"
            },
            "vlm": {
                "optimizations": ["multi_gpu_3d_parallel", "tiling_128x128"],
                "speedup": 5.0,
                "memory_save": 30.0,
                "description": "超大规模集群VLM终极策略: 万卡级多模态分布式训练"
            },
            "moe": {
                "optimizations": ["flash_moe", "gds_spdk", "multi_gpu_3d_parallel", "tiling_128x128"],
                "speedup": 5.5,
                "memory_save": 60.0,
                "description": "超大规模集群MoE终极策略: 万卡级MoE + 全量GDS/SPDK SSD卸载"
            }
        }
    }
    
    @classmethod
    def combine(cls, scenario_info: ScenarioInfo) -> StrategyCombination:
        """主入口: 根据场景信息自动组合最优策略"""
        logger.info(f"[SmartStrategyCombiner] 🎯 为场景 {scenario_info.scenario_id}-{scenario_info.model_type} 生成智能策略组合...")
        
        matrix_entry = cls.STRATEGY_MATRIX.get(scenario_info.scenario_id, {}).get(
            scenario_info.model_type,
            cls.STRATEGY_MATRIX["s0"]["llm"]
        )
        
        strategy = StrategyCombination(
            strategy_name=f"{scenario_info.scenario_id.upper()}_{scenario_info.model_type.upper()}_STRATEGY",
            scenario_id=scenario_info.scenario_id,
            model_type=scenario_info.model_type,
            optimizations=matrix_entry["optimizations"],
            expected_speedup_x=matrix_entry["speedup"],
            expected_memory_saving_gb=matrix_entry["memory_save"],
            description=matrix_entry["description"]
        )
        
        logger.info(f"[SmartStrategyCombiner] ✅ 策略组合生成: {strategy.strategy_name}")
        logger.info(f"  → 优化项数: {len(strategy.optimizations)}")
        logger.info(f"  → 预期提速: {strategy.expected_speedup_x:.1f}x")
        logger.info(f"  → 预期显存节省: {strategy.expected_memory_saving_gb:.1f}GB")
        
        return strategy
