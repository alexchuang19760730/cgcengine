# Copyright (c) 2025 SandAI. All Rights Reserved.
"""
Scenario Detector - Step0: 场景自动探测器
自动识别5大部署场景 (S0-S4) 和3大模型类型
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import platform

logger = logging.getLogger(__name__)


@dataclass
class ScenarioInfo:
    """场景信息数据类"""
    scenario_id: str  # s0 / s1 / s2 / s3 / s4
    scenario_name: str
    model_type: str  # llm / vlm / moe
    num_gpus: int
    has_nvlink: bool
    has_rdma: bool
    total_system_memory_gb: float
    gpu_memory_total_gb: List[float] = field(default_factory=list)
    confidence: float = 0.0


class ScenarioDetector:
    """场景自动探测器"""
    
    @classmethod
    def detect(cls, force_scenario: Optional[str] = None, force_model_type: Optional[str] = None) -> ScenarioInfo:
        """主入口: 自动探测场景和模型类型"""
        logger.info("[ScenarioDetector] 🔍 开始全场景智能探测...")
        
        hardware_info = cls._scan_hardware_topology()
        
        if force_scenario and force_scenario.startswith("s"):
            scenario_info = cls._get_manual_scenario(force_scenario, hardware_info)
        else:
            scenario_info = cls._auto_classify_scenario(hardware_info)
        
        scenario_info.model_type = force_model_type if force_model_type else cls._detect_model_type()
        
        logger.info(f"[ScenarioDetector] ✅ 探测完成: 场景={scenario_info.scenario_name}, 模型类型={scenario_info.model_type}")
        return scenario_info
    
    @classmethod
    def _scan_hardware_topology(cls) -> Dict[str, Any]:
        """扫描硬件拓扑信息"""
        hw = {
            "platform": platform.system(),
            "is_apple_silicon": False,
            "num_gpus": 0,
            "has_nvlink": False,
            "has_rdma": False,
            "gpu_memory_gbs": [],
            "system_memory_gb": 0.0,
        }
        
        if hw["platform"] == "Darwin":
            hw["is_apple_silicon"] = True
            hw["system_memory_gb"] = cls._get_mac_memory()
            hw["num_gpus"] = 1
            return hw
        
        try:
            import torch
            hw["num_gpus"] = torch.cuda.device_count()
            hw["gpu_memory_gbs"] = [torch.cuda.get_device_properties(i).total_memory / (1024**3) for i in range(hw["num_gpus"])]
            hw["has_nvlink"] = hw["num_gpus"] >= 2
        except Exception:
            pass
        
        return hw
    
    @classmethod
    def _get_mac_memory(cls) -> float:
        """获取Mac统一内存大小"""
        try:
            import subprocess
            result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
            mem_bytes = int(result.stdout.strip())
            return mem_bytes / (1024**3)
        except Exception:
            return 16.0
    
    @classmethod
    def _auto_classify_scenario(cls, hw: Dict[str, Any]) -> ScenarioInfo:
        """根据硬件信息自动分类场景S0-S4"""
        scenario_map = {
            "s0": ("端侧纯本地", 0.95),
            "s1": ("端云一体协同", 0.85),
            "s2": ("云双GPU并行", 0.90),
            "s3": ("多机分布式", 0.88),
            "s4": ("超大规模集群", 0.92),
        }
        
        scenario_id = "s0"
        if hw["num_gpus"] >= 8:
            scenario_id = "s4"
        elif hw["num_gpus"] >= 4:
            scenario_id = "s3"
        elif hw["num_gpus"] >= 2:
            scenario_id = "s2"
        elif hw["is_apple_silicon"] or hw["system_memory_gb"] < 16:
            scenario_id = "s0"
        
        name, conf = scenario_map[scenario_id]
        return ScenarioInfo(
            scenario_id=scenario_id,
            scenario_name=name,
            model_type="llm",
            num_gpus=hw["num_gpus"],
            has_nvlink=hw.get("has_nvlink", False),
            has_rdma=hw.get("has_rdma", False),
            total_system_memory_gb=hw["system_memory_gb"],
            gpu_memory_total_gb=hw.get("gpu_memory_gbs", []),
            confidence=conf
        )
    
    @classmethod
    def _get_manual_scenario(cls, scenario_id: str, hw: Dict[str, Any]) -> ScenarioInfo:
        """手动指定场景"""
        scenario_names = {
            "s0": "端侧纯本地",
            "s1": "端云一体协同",
            "s2": "云双GPU并行",
            "s3": "多机分布式",
            "s4": "超大规模集群",
        }
        return ScenarioInfo(
            scenario_id=scenario_id,
            scenario_name=scenario_names.get(scenario_id, scenario_id),
            model_type="llm",
            num_gpus=hw.get("num_gpus", 1),
            has_nvlink=hw.get("has_nvlink", False),
            has_rdma=hw.get("has_rdma", False),
            total_system_memory_gb=hw.get("system_memory_gb", 16.0),
            gpu_memory_total_gb=hw.get("gpu_memory_gbs", []),
            confidence=1.0
        )
    
    @classmethod
    def _detect_model_type(cls) -> str:
        """自动检测模型类型（默认LLM）"""
        return "llm"
