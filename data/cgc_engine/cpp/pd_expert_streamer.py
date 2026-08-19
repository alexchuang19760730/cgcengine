#!/usr/bin/env python3
"""
PD-Separated Expert Streamer for MoE Models

参考 moeexpert/qwen3.6/prime-agent-worktrees 的 TurboFieldfare 实现:
- repack_qwen36.py: 模型打包逻辑
- resident_writer.py: 常驻权重写入

实现:
1. PDLayerSplitter - 将层分配到 prefill/decode 阶段
2. ExpertPrefetcher - 基于路由历史的专家预取
3. PDExpertStreamer - PD 分离式专家流流器
4. MultiGPUCoordinator - 多 GPU 协调器

PD 分离策略:
- Prefill: GPU 0 (Intel UHD) - 处理整个 prompt，加载 prefill 层的所有专家
- Decode: GPU 1 (NVIDIA MX250) - 逐 token 生成，按需流式加载 decode 层专家
- Switch: Prefill 完成后，将 decode 层专家预加载到 GPU 1

Usage:
    from pd_expert_streamer import PDExpertStreamer
    
    streamer = PDExpertStreamer(gguf_path, gpu0_mem_gb=4, gpu1_mem_gb=2)
    streamer.prepare_prefill()  # 准备 prefill 阶段
    # ... prefill tokens ...
    streamer.switch_to_decode()  # 切换到 decode 阶段
    # ... decode tokens ...
"""

import os
import sys
import time
import json
import struct
from collections import defaultdict, OrderedDict
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unified_moe_streamer import (
    UnifiedExpertStreamer,
    ExpertLayout,
    GGML_TYPE_BYTES,
    parse_gguf_header,
    detect_layout,
)


class PDFase(Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    IDLE = "idle"


class PDLayerAssignment:
    """PD 层分配策略."""
    
    def __init__(self, total_layers: int, prefill_ratio: float = 0.5):
        self.total_layers = total_layers
        self.prefill_count = int(total_layers * prefill_ratio)
        self.decode_count = total_layers - self.prefill_count
        
        self.prefill_layers = list(range(self.prefill_count))
        self.decode_layers = list(range(self.prefill_count, total_layers))
    
    def get_device_for_layer(self, layer: int) -> str:
        if layer in self.prefill_layers:
            return "gpu0"
        elif layer in self.decode_layers:
            return "gpu1"
        else:
            return "unknown"


class ExpertPrefetcher:
    """
    基于路由历史的专家预取器.
    
    参考 moeexpert/qwen3.6 的 PrefillRoutedTileScheduler:
    - 记录每个 token 路由到的专家 ID
    - 基于频率统计预测下一个 token 的专家
    - 动态调整预取窗口
    """
    
    def __init__(self, max_history: int = 1000, prefetch_window: int = 3):
        self.max_history = max_history
        self.prefetch_window = prefetch_window
        
        # 每层的路由历史
        self.route_history: Dict[int, List[int]] = defaultdict(list)
        
        # 专家频率统计: {layer_id: {expert_id: count}}
        self.expert_freq: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        
        # 时间衰减因子
        self.decay_factor = 0.95
    
    def record_route(self, layer: int, expert_ids: List[int]):
        """记录 token 路由."""
        history = self.route_history[layer]
        history.extend(expert_ids)
        
        # 限制历史长度
        if len(history) > self.max_history:
            history[:] = history[-self.max_history:]
        
        # 更新频率（带时间衰减）
        freq = self.expert_freq[layer]
        for eid in freq:
            freq[eid] *= self.decay_factor
        for eid in expert_ids:
            freq[eid] += 1.0
    
    def predict_next_experts(self, layer: int, top_k: int = 8) -> List[int]:
        """预测下一个 token 可能用到的专家."""
        freq = self.expert_freq[layer]
        
        if not freq:
            # 无历史记录，返回均匀分布的专家
            return list(range(top_k))
        
        # 按频率排序
        sorted_experts = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        predicted = [eid for eid, _ in sorted_experts[:top_k * self.prefetch_window]]
        
        return predicted[:top_k]
    
    def get_prefetch_list(self, layer: int, current_experts: List[int], 
                          top_k: int = 8) -> List[int]:
        """获取预取列表（排除当前已加载的专家）."""
        predicted = self.predict_next_experts(layer, top_k * 2)
        prefetch = [e for e in predicted if e not in current_experts]
        return prefetch[:top_k]


class PDExpertStreamer:
    """
    PD 分离式专家流流器.
    
    核心功能:
    1. 层分配: 将 MoE 层分配到 prefill/decode 阶段
    2. 专家流式加载: 按需加载专家权重
    3. 缓存管理: LRU + 频率预取
    4. 阶段切换: prefill → decode 的缓存失效和预加载
    
    参考 moeexpert/qwen3.6/prime-agent-worktrees 的实现:
    - ExpertStreamer.swift: 基础流流器
    - PrefillMoEGrouping.swift: Prefill 阶段分组
    - ExpertAccessTrace.swift: 访问追踪
    """
    
    def __init__(self, gguf_path: str, 
                 gpu0_mem_gb: float = 4.0, 
                 gpu1_mem_gb: float = 2.0,
                 prefill_ratio: float = 0.5):
        """
        初始化 PD 专家流流器.
        
        Args:
            gguf_path: GGUF 模型文件路径
            gpu0_mem_gb: GPU 0 显存 (GB)
            gpu1_mem_gb: GPU 1 显存 (GB)
            prefill_ratio: Prefill 层占比
        """
        self.gguf_path = gguf_path
        self.gpu0_mem_gb = gpu0_mem_gb
        self.gpu1_mem_gb = gpu1_mem_gb
        
        # 基础流流器
        self.base_streamer = UnifiedExpertStreamer(gguf_path)
        
        # PD 配置
        layers = self.base_streamer.adapter.list_layers()
        self.total_layers = len(layers)
        self.layer_assignment = PDLayerAssignment(self.total_layers, prefill_ratio)
        
        # 预取器
        self.prefetcher = ExpertPrefetcher()
        
        # 当前阶段
        self.current_phase = PDFase.IDLE
        
        # GPU 0 缓存 (prefill 阶段)
        self.gpu0_cache: OrderedDict[str, dict] = OrderedDict()
        self.gpu0_max_experts = self._calc_max_experts(gpu0_mem_gb)
        
        # GPU 1 缓存 (decode 阶段)
        self.gpu1_cache: OrderedDict[str, dict] = OrderedDict()
        self.gpu1_max_experts = self._calc_max_experts(gpu1_mem_gb)
        
        # 统计
        self.stats = {
            'prefill_experts_loaded': 0,
            'decode_experts_loaded': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'prefetch_hits': 0,
            'tokens_processed': 0,
        }
        
        self._print_config()
    
    def _calc_max_experts(self, mem_gb: float) -> int:
        """计算 GPU 可缓存的最大专家数."""
        # 使用 Qwen3.6 典型值: ~774MB 每层 = ~3MB 每专家
        bytes_per_expert = 3 * 1024 * 1024  # 3MB 每专家
        max_experts = int(mem_gb * 1024**3 / bytes_per_expert * 0.85)  # 85% 利用率
        return max_experts
    
    def _print_config(self):
        """打印 PD 配置."""
        print("\n" + "=" * 60)
        print("PD EXPERT STREAMER CONFIGURATION")
        print("=" * 60)
        print(f"  Model: {os.path.basename(self.gguf_path)}")
        print(f"  Total Layers: {self.total_layers}")
        print(f"  Prefill Layers (GPU 0): {len(self.layer_assignment.prefill_layers)}")
        print(f"  Decode Layers (GPU 1): {len(self.layer_assignment.decode_layers)}")
        print(f"  GPU 0 Memory: {self.gpu0_mem_gb}GB → max {self.gpu0_max_experts} experts")
        print(f"  GPU 1 Memory: {self.gpu1_mem_gb}GB → max {self.gpu1_max_experts} experts")
        
        if hasattr(self.base_streamer.adapter, 'num_experts_val'):
            n_experts = self.base_streamer.adapter.num_experts_val
            print(f"  Experts/Layer: {n_experts}")
            print(f"  Total Expert Slots: {self.total_layers * n_experts}")
        print("=" * 60)
    
    def prepare_prefill(self, expert_ids_per_layer: list = None):
        """
        准备 Prefill 阶段.
        
        只加载需要的专家 (基于 top-K 或指定列表).
        默认加载 top_k 个专家.
        """
        print("\n[PD] Preparing Prefill Phase...")
        self.current_phase = PDFase.PREFILL
        
        prefill_layers = self.layer_assignment.prefill_layers
        n_experts = self.base_streamer.adapter.num_experts(prefill_layers[0])
        top_k = min(8, n_experts)  # 默认 top-8
        
        if expert_ids_per_layer is None:
            expert_ids_per_layer = list(range(top_k))
        
        loaded = 0
        failed = 0
        
        for layer in prefill_layers:
            for expert_id in expert_ids_per_layer:
                try:
                    expert = self._load_expert_to_gpu0(layer, expert_id)
                    if expert:
                        loaded += 1
                        
                        # 检查缓存限制
                        if len(self.gpu0_cache) >= self.gpu0_max_experts:
                            print(f"  GPU 0 cache full ({len(self.gpu0_cache)} experts), stopping prefill load")
                            print(f"  Loaded {loaded} experts (target: {len(prefill_layers)} layers × {len(expert_ids_per_layer)} experts)")
                            self.stats['prefill_experts_loaded'] = loaded
                            return loaded
                except MemoryError:
                    failed += 1
                    if failed >= 3:
                        print(f"  Memory pressure detected ({failed} failures), stopping prefill load")
                        print(f"  Loaded {loaded} experts before memory limit")
                        self.stats['prefill_experts_loaded'] = loaded
                        return loaded
                    continue
        
        print(f"  Loaded {loaded} experts for prefill phase")
        self.stats['prefill_experts_loaded'] = loaded
        return loaded
    
    def switch_to_decode(self):
        """
        切换到 Decode 阶段.
        
        1. 失效 GPU 0 缓存 (释放显存)
        2. 清除 GPU 1 缓存
        3. 预加载 decode 层的常用专家到 GPU 1
        """
        print("\n[PD] Switching to Decode Phase...")
        self.current_phase = PDFase.DECODE
        
        # 失效 GPU 0 缓存
        self.gpu0_cache.clear()
        print("  Cleared GPU 0 cache")
        
        # 预加载 decode 层的常用专家
        decode_layers = self.layer_assignment.decode_layers
        preloaded = 0
        
        for layer in decode_layers[:5]:  # 预加载前 5 层
            # 获取该层的专家频率
            freq = self.prefetcher.expert_freq.get(layer, {})
            if freq:
                sorted_experts = sorted(freq.items(), key=lambda x: x[1], reverse=True)
                top_experts = [eid for eid, _ in sorted_experts[:8]]
            else:
                top_experts = list(range(8))  # 默认前 8 个
            
            for eid in top_experts:
                expert = self._load_expert_to_gpu1(layer, eid)
                if expert:
                    preloaded += 1
        
        print(f"  Preloaded {preloaded} experts for decode phase")
        return preloaded
    
    def decode_load_expert(self, layer: int, expert_id: int) -> dict:
        """
        Decode 阶段加载专家.
        
        1. 检查 GPU 1 缓存
        2. 若未命中，加载并添加到缓存
        3. 执行预取策略
        """
        cache_key = f"L{layer}_E{expert_id}"
        
        # 检查缓存
        if cache_key in self.gpu1_cache:
            self.stats['cache_hits'] += 1
            # 移到末尾 (LRU 更新)
            self.gpu1_cache.move_to_end(cache_key)
            return self.gpu1_cache[cache_key]
        
        self.stats['cache_misses'] += 1
        
        # 加载专家
        expert = self.base_streamer.load_expert(layer, expert_id)
        if not expert:
            return {}
        
        # 添加到 GPU 1 缓存
        self.gpu1_cache[cache_key] = expert
        
        # 缓存溢出时淘汰最久未使用的
        if len(self.gpu1_cache) > self.gpu1_max_experts:
            self.gpu1_cache.popitem(last=False)
        
        self.stats['decode_experts_loaded'] += 1
        return expert
    
    def prefetch_for_token(self, layer: int, current_experts: List[int], 
                            top_k: int = 8) -> List[int]:
        """
        为下一个 token 预取专家.
        
        参考 moeexpert/qwen3.6 的 PrefetchStrategy:
        - 基于当前 token 的路由预测下一个 token 的专家
        - 预取到 GPU 1 缓存
        """
        prefetch_list = self.prefetcher.get_prefetch_list(layer, current_experts, top_k)
        prefetched = []
        
        for eid in prefetch_list:
            cache_key = f"L{layer}_E{eid}"
            if cache_key not in self.gpu1_cache:
                expert = self.base_streamer.load_expert(layer, eid)
                if expert:
                    self.gpu1_cache[cache_key] = expert
                    prefetched.append(eid)
                    self.stats['prefetch_hits'] += 1
                    
                    # 缓存溢出时淘汰
                    if len(self.gpu1_cache) > self.gpu1_max_experts:
                        self.gpu1_cache.popitem(last=False)
        
        return prefetched
    
    def record_token_routing(self, routes: Dict[int, List[int]]):
        """
        记录 token 路由用于未来预取.
        
        Args:
            routes: {layer_id: [expert_id_1, expert_id_2, ...]}
        """
        for layer, experts in routes.items():
            self.prefetcher.record_route(layer, experts)
        
        self.stats['tokens_processed'] += 1
    
    def get_prefill_experts(self, layer: int) -> List[dict]:
        """获取 Prefill 阶段指定层的所有专家."""
        if self.current_phase != PDFase.PREFILL:
            return []
        
        experts = []
        for eid in range(self.base_streamer.adapter.num_experts(layer)):
            cache_key = f"L{layer}_E{eid}"
            if cache_key in self.gpu0_cache:
                experts.append(self.gpu0_cache[cache_key])
        
        return experts
    
    def get_stats(self) -> dict:
        """获取统计信息."""
        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        
        return {
            'phase': self.current_phase.value,
            'gpu0_cache_size': len(self.gpu0_cache),
            'gpu0_max_size': self.gpu0_max_experts,
            'gpu1_cache_size': len(self.gpu1_cache),
            'gpu1_max_size': self.gpu1_max_experts,
            'cache_hit_rate': self.stats['cache_hits'] / max(total_requests, 1) * 100,
            'prefetch_hits': self.stats['prefetch_hits'],
            'tokens_processed': self.stats['tokens_processed'],
            'prefill_experts_loaded': self.stats['prefill_experts_loaded'],
            'decode_experts_loaded': self.stats['decode_experts_loaded'],
        }
    
    def _load_expert_to_gpu0(self, layer: int, expert_id: int) -> Optional[dict]:
        """加载专家到 GPU 0 缓存."""
        cache_key = f"L{layer}_E{expert_id}"
        
        if cache_key in self.gpu0_cache:
            self.gpu0_cache.move_to_end(cache_key)
            return self.gpu0_cache[cache_key]
        
        expert = self.base_streamer.load_expert(layer, expert_id)
        if not expert:
            return None
        
        self.gpu0_cache[cache_key] = expert
        
        # LRU 淘汰
        if len(self.gpu0_cache) > self.gpu0_max_experts:
            self.gpu0_cache.popitem(last=False)
        
        return expert
    
    def _load_expert_to_gpu1(self, layer: int, expert_id: int) -> Optional[dict]:
        """加载专家到 GPU 1 缓存."""
        cache_key = f"L{layer}_E{expert_id}"
        
        if cache_key in self.gpu1_cache:
            self.gpu1_cache.move_to_end(cache_key)
            return self.gpu1_cache[cache_key]
        
        expert = self.base_streamer.load_expert(layer, expert_id)
        if not expert:
            return None
        
        self.gpu1_cache[cache_key] = expert
        
        # LRU 淘汰
        if len(self.gpu1_cache) > self.gpu1_max_experts:
            self.gpu1_cache.popitem(last=False)
        
        return expert


def test_pd_streamer():
    """测试 PD 流流器."""
    print("=" * 80)
    print("PD EXPERT STREAMER - TEST")
    print("=" * 80)
    
    model_path = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found: {model_path}")
        return 1
    
    # 初始化 PD 流流器
    print("\n🚀 Initializing PDExpertStreamer...")
    t0 = time.time()
    pd_streamer = PDExpertStreamer(model_path, gpu0_mem_gb=4, gpu1_mem_gb=2)
    init_time = time.time() - t0
    print(f"  Initialized in {init_time:.2f}s")
    
    # 测试 Prefill 阶段 (只加载前 4 层的 top-4 专家)
    print("\n📦 Testing Prefill Phase (loading top-4 experts per layer)...")
    t0 = time.time()
    prefill_loaded = pd_streamer.prepare_prefill(expert_ids_per_layer=[0, 1, 2, 3])
    prefill_time = time.time() - t0
    print(f"  Loaded {prefill_loaded} experts in {prefill_time:.2f}s")
    
    # 模拟路由记录 (只记录前 5 层)
    print("\n🎯 Simulating Token Routing (5 layers, 10 tokens)...")
    test_routes = []
    layers = pd_streamer.base_streamer.adapter.list_layers()
    for i in range(10):
        routes = {}
        for layer in layers[:5]:  # 只记录前 5 层
            routes[layer] = [i % 256, (i + 1) % 256, (i + 2) % 256]
        pd_streamer.record_token_routing(routes)
        test_routes.append(routes)
    print(f"  Recorded 10 token routes for 5 layers")
    
    # 测试切换到 Decode
    print("\n🔄 Testing Switch to Decode Phase...")
    t0 = time.time()
    preloaded = pd_streamer.switch_to_decode()
    switch_time = time.time() - t0
    print(f"  Preloaded {preloaded} experts in {switch_time:.2f}s")
    
    # 测试 Decode 阶段加载 (只测前 3 层)
    print("\n📤 Testing Decode Phase Loading (3 layers)...")
    decode_layers = pd_streamer.layer_assignment.decode_layers[:3]
    for layer in decode_layers:
        for eid in [0, 50, 100]:
            t0 = time.time()
            expert = pd_streamer.decode_load_expert(layer, eid)
            load_time = (time.time() - t0) * 1000
            if expert:
                roles = expert.get('roles', {})
                total_kb = sum(r.get('size_bytes', 0) / 1024 for r in roles.values())
                print(f"    Layer {layer}, Expert {eid}: {load_time:.1f}ms ({total_kb:.0f}KB)")
    
    # 测试预取
    print("\n🔮 Testing Expert Prefetch...")
    for layer in decode_layers[:2]:
        prefetch_list = pd_streamer.prefetch_for_token(layer, [0, 1, 2], top_k=4)
        print(f"    Layer {layer}: prefetched {len(prefetch_list)} experts: {prefetch_list}")
    
    # 获取统计
    print("\n📊 Statistics:")
    stats = pd_streamer.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)
    print("✅ PD EXPERT STREAMER TEST COMPLETE")
    print("=" * 80)
    return 0


if __name__ == '__main__':
    sys.exit(test_pd_streamer())
