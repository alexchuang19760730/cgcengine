#!/usr/bin/env python3
"""
CGC编译器自动增强系统 - Agent核心模块
实现自动发现、智能决策、自动注入三大功能
优化版本：并行扫描 + 缓存机制 + 按需注入 + 智能预热
"""

import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional

# ==================== AutoDiscoveryAgent ====================

class AutoDiscoveryAgent:
    """自动发现Agent - 扫描系统中可用的优化技术
    优化特性：并行扫描 + 缓存机制 + 增量更新
    """
    
    def __init__(self):
        self.optimizations: Dict[str, List[Dict[str, Any]]] = {}
        self.backend_status: Dict[str, bool] = {}
        self.last_scan_time = 0
        self.cache_ttl = 3600  # 缓存有效期1小时
        self.scan_lock = threading.Lock()
    
    def scan_backends(self, force: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        """扫描所有可用的后端优化技术（支持缓存）"""
        now = time.time()
        
        # 检查缓存是否有效
        if not force and (now - self.last_scan_time) < self.cache_ttl:
            print("� 使用缓存的优化技术...")
            return self.optimizations
        
        return self._scan_backends_parallel()
    
    def _scan_backends_parallel(self) -> Dict[str, List[Dict[str, Any]]]:
        """并行扫描所有后端优化技术"""
        print("� Agent正在并行扫描可用优化技术...")
        
        # 清空之前的优化
        self.optimizations.clear()
        self.backend_status.clear()
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # 提交所有扫描任务
            futures = [
                executor.submit(self._discover_llama_compute),
                executor.submit(self._discover_llama_storage),
                executor.submit(self._discover_vllm_scheduler),
                executor.submit(self._discover_mlx_optimizations),
                executor.submit(self._discover_flashmoe_optimizations),
                executor.submit(self._discover_megatrain_optimizations),
                executor.submit(self._discover_mlx_tune_optimizations),
            ]
            
            # 等待所有任务完成
            for future in futures:
                future.result()
        
        self.last_scan_time = time.time()
        print(f"✅ 并行发现 {sum(len(v) for v in self.optimizations.values())} 项优化技术")
        return self.optimizations
    
    def _discover_llama_compute(self):
        """发现llama.cpp计算图优化"""
        optimizations = [
            {
                "name": "flash_attention",
                "layer": "graph_optim",
                "priority": 10,
                "description": "Flash Attention优化注意力计算",
                "backend": "llama.cpp",
                "requirements": ["cuda", "flash-attn"]
            },
            {
                "name": "fused_mlp",
                "layer": "graph_optim",
                "priority": 8,
                "description": "融合MLP层计算",
                "backend": "llama.cpp",
                "requirements": ["cuda"]
            },
            {
                "name": "rope_fusion",
                "layer": "graph_optim",
                "priority": 7,
                "description": "RoPE位置编码融合",
                "backend": "llama.cpp",
                "requirements": []
            },
            {
                "name": "swiglu_fusion",
                "layer": "graph_optim",
                "priority": 9,
                "description": "SwigLU激活函数融合",
                "backend": "llama.cpp",
                "requirements": ["cuda"]
            }
        ]
        self.optimizations["llama_compute"] = optimizations
        self.backend_status["llama.cpp"] = True
    
    def _discover_llama_storage(self):
        """发现llama.cpp内存优化"""
        optimizations = [
            {
                "name": "gguf_quantization",
                "layer": "memory",
                "priority": 10,
                "description": "GGUF量化格式支持",
                "backend": "llama.cpp",
                "requirements": ["gguf"]
            },
            {
                "name": "page_cache",
                "layer": "memory",
                "priority": 9,
                "description": "分页缓存管理",
                "backend": "llama.cpp",
                "requirements": []
            },
            {
                "name": "mmap_loading",
                "layer": "memory",
                "priority": 8,
                "description": "内存映射加载",
                "backend": "llama.cpp",
                "requirements": []
            },
            {
                "name": "gpu_offloading",
                "layer": "memory",
                "priority": 9,
                "description": "GPU分层卸载",
                "backend": "llama.cpp",
                "requirements": ["cuda", "metal"]
            }
        ]
        self.optimizations["llama_storage"] = optimizations
    
    def _discover_vllm_scheduler(self):
        """发现vLLM调度优化"""
        optimizations = [
            {
                "name": "paged_kv_cache",
                "layer": "scheduler",
                "priority": 10,
                "description": "分页KV缓存管理",
                "backend": "vLLM",
                "requirements": ["cuda"]
            },
            {
                "name": "continuous_batching",
                "layer": "scheduler",
                "priority": 9,
                "description": "连续批处理",
                "backend": "vLLM",
                "requirements": ["cuda"]
            },
            {
                "name": "speculative_decoding",
                "layer": "scheduler",
                "priority": 8,
                "description": "投机解码",
                "backend": "vLLM",
                "requirements": ["cuda"]
            },
            {
                "name": "dynamic_batching",
                "layer": "scheduler",
                "priority": 8,
                "description": "动态批处理",
                "backend": "vLLM",
                "requirements": ["cuda"]
            },
            {
                "name": "device_io_optimization",
                "layer": "scheduler",
                "priority": 7,
                "description": "设备IO优化",
                "backend": "vLLM",
                "requirements": ["cuda"]
            }
        ]
        self.optimizations["vllm_scheduler"] = optimizations
        self.backend_status["vLLM"] = True
    
    def _discover_mlx_optimizations(self):
        """发现MLX优化技术"""
        optimizations = [
            {
                "name": "mlx_graph_opt",
                "layer": "graph_optim",
                "priority": 8,
                "description": "MLX计算图优化",
                "backend": "mlx",
                "requirements": ["metal"]
            },
            {
                "name": "mlx_kv_cache",
                "layer": "memory",
                "priority": 9,
                "description": "MLX KV缓存优化",
                "backend": "mlx",
                "requirements": ["metal"]
            }
        ]
        self.optimizations["mlx_optimizations"] = optimizations
        self.backend_status["mlx"] = True
    
    def _discover_flashmoe_optimizations(self):
        """发现FlashMoE优化技术"""
        optimizations = [
            {
                "name": "flashmoe_expert",
                "layer": "graph_optim",
                "priority": 9,
                "description": "FlashMoE专家计算优化",
                "backend": "flashmoe",
                "requirements": ["cuda"]
            },
            {
                "name": "flashmoe_router",
                "layer": "scheduler",
                "priority": 8,
                "description": "FlashMoE路由优化",
                "backend": "flashmoe",
                "requirements": ["cuda"]
            }
        ]
        self.optimizations["flashmoe_optimizations"] = optimizations
        # FlashMoE是一种MoE技术，不是独立后端，不加入backend_status
    
    def _discover_megatrain_optimizations(self):
        """发现megatrain分布式训练优化技术"""
        optimizations = [
            {
                "name": "megatrain_data_parallel",
                "layer": "scheduler",
                "priority": 10,
                "description": "数据并行训练",
                "backend": "megatrain",
                "requirements": ["cuda", "nccl"]
            },
            {
                "name": "megatrain_model_parallel",
                "layer": "graph_optim",
                "priority": 10,
                "description": "模型并行训练",
                "backend": "megatrain",
                "requirements": ["cuda", "nccl"]
            },
            {
                "name": "megatrain_pipeline_parallel",
                "layer": "scheduler",
                "priority": 9,
                "description": "流水线并行训练",
                "backend": "megatrain",
                "requirements": ["cuda", "nccl"]
            },
            {
                "name": "megatrain_zero",
                "layer": "memory",
                "priority": 10,
                "description": "ZeRO内存优化",
                "backend": "megatrain",
                "requirements": ["cuda", "nccl"]
            },
            {
                "name": "megatrain_checkpoint",
                "layer": "memory",
                "priority": 8,
                "description": "分布式检查点",
                "backend": "megatrain",
                "requirements": ["cuda", "nccl"]
            }
        ]
        self.optimizations["megatrain_optimizations"] = optimizations
        self.backend_status["megatrain"] = True
    
    def _discover_mlx_tune_optimizations(self):
        """发现mlx-tune微调优化技术"""
        optimizations = [
            {
                "name": "lora_adapter",
                "layer": "graph_optim",
                "priority": 10,
                "description": "LoRA适配器微调",
                "backend": "mlx-tune",
                "requirements": ["metal"]
            },
            {
                "name": "qlora_adapter",
                "layer": "graph_optim",
                "priority": 9,
                "description": "QLoRA量化微调",
                "backend": "mlx-tune",
                "requirements": ["metal"]
            },
            {
                "name": "ia3_adapter",
                "layer": "graph_optim",
                "priority": 8,
                "description": "IA³适配器微调",
                "backend": "mlx-tune",
                "requirements": ["metal"]
            },
            {
                "name": "mlx_peft",
                "layer": "memory",
                "priority": 9,
                "description": "参数高效微调",
                "backend": "mlx-tune",
                "requirements": ["metal"]
            }
        ]
        self.optimizations["mlx_tune_optimizations"] = optimizations
        self.backend_status["mlx-tune"] = True
    
    def get_optimizations_by_layer(self, layer: str) -> List[Dict[str, Any]]:
        """按层级获取优化列表"""
        result = []
        for opt_list in self.optimizations.values():
            result.extend(opt for opt in opt_list if opt["layer"] == layer)
        return sorted(result, key=lambda x: x["priority"], reverse=True)

# ==================== OptimizationDecisionEngine ====================

class OptimizationDecisionEngine:
    """决策引擎 - 根据模型信息和硬件环境选择最优配置"""
    
    def __init__(self):
        self.rules = self._load_rules()
    
    def _load_rules(self) -> List[Dict[str, Any]]:
        """加载决策规则"""
        return [
            {
                "name": "model_size_check",
                "description": "模型大小检查",
                "conditions": [
                    {"opt_name": "continuous_batching", "model_size_min": "7B"}
                ]
            },
            {
                "name": "quantization_check",
                "description": "量化检查",
                "conditions": [
                    {"opt_name": "gguf_quantization", "quantization_required": ["Q4_K_M", "Q8_0"]}
                ]
            },
            {
                "name": "hardware_check",
                "description": "硬件检查",
                "conditions": [
                    {"opt_name": "flash_attention", "hardware_required": ["cuda", "metal"]}
                ]
            }
        ]
    
    def decide(self, model_info: Dict[str, Any], optimizations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """根据模型信息和可用优化做出决策"""
        print(f"🧠 决策引擎处理模型: {model_info.get('name', 'Unknown')}")
        
        decisions = []
        for opt in optimizations:
            if self._check_rules(opt, model_info):
                decisions.append(opt)
                print(f"   ✅ 选中: {opt['name']}")
        
        return self._prioritize(decisions)
    
    def _check_rules(self, opt: Dict[str, Any], model_info: Dict[str, Any]) -> bool:
        """检查优化是否适用于当前模型"""
        # 模型大小检查
        model_size = model_info.get("size", "0B")
        if opt["name"] == "continuous_batching" and self._parse_size(model_size) < self._parse_size("7B"):
            return False
        
        # 量化格式检查
        quantization = model_info.get("quantization", "")
        if opt["name"] == "gguf_quantization" and quantization not in ["Q4_K_M", "Q8_0", "FP16", "FP32"]:
            return False
        
        # 硬件要求检查
        requirements = opt.get("requirements", [])
        if requirements:
            available_hardware = model_info.get("hardware", [])
            if not any(req in available_hardware for req in requirements):
                return False
        
        return True
    
    def _parse_size(self, size_str: str) -> int:
        """解析模型大小字符串"""
        size_str = size_str.strip().upper()
        if size_str.endswith("B"):
            num = float(size_str[:-1])
            return int(num * 1e9)
        elif size_str.endswith("M"):
            num = float(size_str[:-1])
            return int(num * 1e6)
        return int(size_str)
    
    def _prioritize(self, decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """按优先级排序"""
        return sorted(decisions, key=lambda x: x["priority"], reverse=True)

# ==================== AutoInjector ====================

class AutoInjector:
    """自动注入器 - 将优化自动注入到CGC编译器对应层
    优化特性：按需注入 + 智能预热 + 优先级排序
    """
    
    def __init__(self, compiler):
        self.compiler = compiler
        self.injected_optimizations = []
    
    def inject(self, optimizations: List[Dict[str, Any]], layers: Optional[List[str]] = None) -> List[str]:
        """将优化自动注入到对应层（支持按需注入）"""
        if layers is None:
            layers = ["graph_optim", "memory", "scheduler"]
        
        print(f"\n🔧 Agent正在注入优化 (目标层: {layers})...")
        
        for opt in optimizations:
            layer = opt["layer"]
            if layer not in layers:
                continue
                
            try:
                if layer == "graph_optim":
                    self._inject_graph_optim(opt)
                elif layer == "memory":
                    self._inject_memory_optim(opt)
                elif layer == "scheduler":
                    self._inject_scheduler_optim(opt)
                else:
                    print(f"   ⚠️ 未知层级: {layer}")
            except Exception as e:
                print(f"   ❌ 注入失败 {opt['name']}: {e}")
        
        return self.injected_optimizations
    
    def inject_critical_only(self, optimizations: List[Dict[str, Any]], min_priority: int = 9) -> List[str]:
        """智能预热：只注入高优先级优化"""
        critical_opts = [opt for opt in optimizations if opt["priority"] >= min_priority]
        print(f"\n🔥 智能预热模式 - 只注入优先级 >= {min_priority} 的优化")
        return self.inject(critical_opts)
    
    def _inject_graph_optim(self, opt: Dict[str, Any]):
        """注入图优化层"""
        if hasattr(self.compiler, 'graph_optimizer'):
            self.compiler.graph_optimizer.add_gt(opt["name"])
            self.injected_optimizations.append(opt["name"])
            print(f"   ✅ 图优化层注入: {opt['name']}")
    
    def _inject_memory_optim(self, opt: Dict[str, Any]):
        """注入内存管理层"""
        if hasattr(self.compiler, 'memory_manager'):
            self.compiler.memory_manager.add_storage_gt(opt["name"])
            self.injected_optimizations.append(opt["name"])
            print(f"   ✅ 内存管理层注入: {opt['name']}")
    
    def _inject_scheduler_optim(self, opt: Dict[str, Any]):
        """注入执行调度层"""
        if hasattr(self.compiler, 'execution_scheduler'):
            self.compiler.execution_scheduler.add_scheduler_gt(opt["name"])
            self.injected_optimizations.append(opt["name"])
            print(f"   ✅ 执行调度层注入: {opt['name']}")

# ==================== CGCCompiler Mock ====================

class MockGraphOptimizer:
    """模拟图优化器"""
    def __init__(self):
        self.gts = []
    
    def add_gt(self, name: str):
        self.gts.append(name)

class MockMemoryManager:
    """模拟内存管理器"""
    def __init__(self):
        self.gts = []
    
    def add_storage_gt(self, name: str):
        self.gts.append(name)

class MockExecutionScheduler:
    """模拟执行调度器"""
    def __init__(self):
        self.gts = []
    
    def add_scheduler_gt(self, name: str):
        self.gts.append(name)

class CGCCompiler:
    """CGC编译器模拟"""
    def __init__(self):
        self.graph_optimizer = MockGraphOptimizer()
        self.memory_manager = MockMemoryManager()
        self.execution_scheduler = MockExecutionScheduler()

# ==================== AgentController ====================

class AgentController:
    """Agent控制器 - 整合自动发现、决策、注入
    优化特性：支持多种运行模式 + 智能预热 + 按需注入
    """
    
    def __init__(self):
        self.discovery_agent = AutoDiscoveryAgent()
        self.decision_engine = OptimizationDecisionEngine()
        self.injector = None
        self.last_model_info = None
        self.compiler = None
    
    def run(self, compiler, model_info: Dict[str, Any], 
            mode: str = "full", layers: Optional[List[str]] = None,
            force_rescan: bool = False) -> Dict[str, Any]:
        """执行自动增强流程（支持多种模式）
        
        Args:
            compiler: CGC编译器实例
            model_info: 模型信息字典
            mode: 运行模式 ('full', 'warmup', 'lazy')
            layers: 指定注入的层（仅在mode='lazy'时生效）
            force_rescan: 是否强制重新扫描（跳过缓存）
        """
        self.compiler = compiler
        self.last_model_info = model_info
        
        print("🚀 启动CGC编译器自动增强系统")
        print(f"   模式: {mode} | 强制扫描: {force_rescan}")
        
        # 1. 自动发现（支持缓存）
        optimizations = self.discovery_agent.scan_backends(force=force_rescan)
        
        # 2. 获取按层分类的优化
        all_optimizations = []
        for opt_list in optimizations.values():
            all_optimizations.extend(opt_list)
        
        # 3. 智能决策
        decisions = self.decision_engine.decide(model_info, all_optimizations)
        
        # 4. 根据模式执行注入
        self.injector = AutoInjector(compiler)
        
        if mode == "warmup":
            # 智能预热：只注入高优先级优化
            injected = self.injector.inject_critical_only(decisions)
        elif mode == "lazy":
            # 按需注入：只注入指定层
            injected = self.injector.inject(decisions, layers=layers)
        else:
            # 完整模式：注入所有优化
            injected = self.injector.inject(decisions)
        
        # 5. 返回结果
        return {
            "discovered": len(all_optimizations),
            "selected": len(decisions),
            "injected": len(injected),
            "injected_list": injected,
            "mode": mode
        }
    
    def warm_up(self, compiler, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """智能预热模式：快速注入关键优化"""
        return self.run(compiler, model_info, mode="warmup")
    
    def lazy_inject(self, compiler, model_info: Dict[str, Any], 
                    layers: List[str]) -> Dict[str, Any]:
        """按需注入模式：只注入指定层"""
        return self.run(compiler, model_info, mode="lazy", layers=layers)
    
    def refresh(self):
        """强制刷新优化缓存"""
        if self.compiler and self.last_model_info:
            print("🔄 强制刷新优化缓存...")
            return self.run(self.compiler, self.last_model_info, force_rescan=True)
        return {"error": "No active compiler session"}

# ==================== 测试函数 ====================

def test_auto_enhancement():
    """测试Agent自动增强功能"""
    print("=" * 60)
    print("🧪 测试CGC编译器自动增强系统")
    print("=" * 60)
    
    # 创建编译器和Agent
    compiler = CGCCompiler()
    agent = AgentController()
    
    # 模型信息
    model_info = {
        "name": "Qwen2.5-7B",
        "size": "7B",
        "quantization": "Q4_K_M",
        "hardware": ["cuda", "metal", "gguf"]
    }
    
    # 运行Agent
    result = agent.run(compiler, model_info)
    
    # 验证结果
    print("\n" + "=" * 60)
    print("📊 测试结果")
    print("=" * 60)
    print(f"发现优化技术: {result['discovered']} 项")
    print(f"选中优化技术: {result['selected']} 项")
    print(f"成功注入: {result['injected']} 项")
    print(f"注入列表: {result['injected_list']}")
    
    # 验证注入
    assert "flash_attention" in compiler.graph_optimizer.gts, "Flash Attention未注入"
    assert "gguf_quantization" in compiler.memory_manager.gts, "GGUF量化未注入"
    assert "paged_kv_cache" in compiler.execution_scheduler.gts, "Paged KV缓存未注入"
    
    print("\n✅ Agent自动增强测试通过!")

if __name__ == "__main__":
    test_auto_enhancement()