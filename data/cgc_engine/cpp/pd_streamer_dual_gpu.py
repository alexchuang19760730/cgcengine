#!/usr/bin/env python3
"""
Dual-GPU PD-Separated Expert Streamer

针对 Intel UHD (4GB) + NVIDIA MX250 (2GB) 的优化实现.

核心特性:
1. Per-Layer Expert Streaming: 每次只加载必要的专家权重
2. PD 分离: Prefill (GPU0) → Decode (GPU1) 无缝切换
3. 动态缓存: 基于 LRU + 频率预取的智能缓存管理
4. 路由感知: 基于 token 路由历史预测预取
5. 量化感知: 支持 GGUF 多种量化格式的切片加载

架构:
    GPU 0 (Intel UHD 4GB)          GPU 1 (NVIDIA MX250 2GB)
    ┌─────────────────────┐        ┌─────────────────────┐
    │ Prefill Phase       │        │ Decode Phase        │
    │ - 处理整个 prompt   │  ─────→ │ - 逐 token 生成     │
    │ - 加载 prefill 层   │  switch │ - 按需加载专家      │
    │ - top-K 专家缓存    │        │ - 动态预取          │
    └─────────────────────┘        └─────────────────────┘
"""

import os
import sys
import time
import struct
import json
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Tuple, Set

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unified_moe_streamer import (
    UnifiedExpertStreamer,
    ExpertLayout,
    GGML_TYPE_BYTES,
    parse_gguf_header,
    detect_layout,
)


class Phase(Enum):
    IDLE = "idle"
    PREFILL = "prefill"
    DECODE = "decode"


class QuantType(Enum):
    IQ3_XXS = 18
    IQ2_XXS = 16
    IQ4_XS = 23
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    BF16 = 30
    F16 = 1
    F32 = 0


@dataclass
class ExpertCacheEntry:
    """GPU 缓存中的专家条目."""
    layer: int
    expert_id: int
    gate_bytes: bytes = None
    up_bytes: bytes = None
    down_bytes: bytes = None
    gate_inp_bytes: bytes = None
    last_access: float = 0.0
    access_count: int = 0
    size_bytes: int = 0
    
    @property
    def key(self) -> str:
        return f"L{self.layer}_E{self.expert_id}"


@dataclass
class GPUCacheStats:
    """GPU 缓存统计."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    bytes_used: int = 0
    expert_count: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / max(total, 1) * 100


class DualGPUCache:
    """
    双 GPU 缓存管理器.
    
    特性:
    - LRU + 频率组合淘汰策略
    - 内存预算管理
    - 专家权重去重 (共享 gate_inp)
    """
    
    def __init__(self, gpu_id: str, max_memory_gb: float):
        self.gpu_id = gpu_id
        self.max_memory_bytes = int(max_memory_gb * 1024**3 * 0.85)  # 85% 利用率
        
        # 主缓存: {cache_key: ExpertCacheEntry}
        self._cache: OrderedDict[str, ExpertCacheEntry] = OrderedDict()
        
        # 共享权重缓存 (gate_inp 等所有专家共享的权重)
        self._shared_cache: Dict[str, bytes] = {}
        
        # 统计
        self.stats = GPUCacheStats()
        
        # 热专家追踪
        self._hot_experts: Set[str] = set()
        self._hot_threshold = 3  # 访问次数阈值
    
    @property
    def current_bytes(self) -> int:
        return self.stats.bytes_used
    
    @property
    def free_bytes(self) -> int:
        return max(0, self.max_memory_bytes - self.current_bytes)
    
    @property
    def expert_count(self) -> int:
        return len(self._cache)
    
    def get(self, layer: int, expert_id: int) -> Optional[ExpertCacheEntry]:
        """获取专家缓存."""
        key = f"L{layer}_E{expert_id}"
        
        if key in self._cache:
            entry = self._cache[key]
            self.stats.hits += 1
            entry.last_access = time.time()
            entry.access_count += 1
            
            # 升级为热专家
            if entry.access_count >= self._hot_threshold:
                self._hot_experts.add(key)
            
            # LRU: 移到末尾 (最近使用)
            self._cache.move_to_end(key)
            return entry
        
        self.stats.misses += 1
        return None
    
    def put(self, entry: ExpertCacheEntry) -> bool:
        """
        添加专家到缓存.
        
        Returns:
            True if successfully added, False if memory insufficient
        """
        key = entry.key
        
        # 如果已存在，先移除旧的
        if key in self._cache:
            old = self._cache.pop(key)
            self.stats.bytes_used -= old.size_bytes
        
        # 检查内存
        if self.current_bytes + entry.size_bytes > self.max_memory_bytes:
            # 需要淘汰
            while self._cache and self.current_bytes + entry.size_bytes > self.max_memory_bytes:
                if not self._evict_one():
                    return False
        
        # 添加新条目
        self._cache[key] = entry
        self.stats.bytes_used += entry.size_bytes
        self.stats.expert_count = len(self._cache)
        
        return True
    
    def _evict_one(self) -> bool:
        """淘汰一个专家 (冷优先 → 最近最少使用)."""
        if not self._cache:
            return False
        
        # 优先淘汰非热专家
        for key, entry in self._cache.items():
            if key not in self._hot_experts:
                self._cache.pop(key)
                self.stats.bytes_used -= entry.size_bytes
                self.stats.evictions += 1
                return True
        
        # 如果都是热专家，淘汰最久未使用的
        if self._cache:
            key, entry = next(iter(self._cache.items()))
            self._cache.pop(key)
            self.stats.bytes_used -= entry.size_bytes
            self.stats.evictions += 1
            return True
        
        return False
    
    def clear(self):
        """清空缓存."""
        self._cache.clear()
        self._shared_cache.clear()
        self.stats.bytes_used = 0
        self.stats.expert_count = 0
    
    def remove_layer(self, layer: int):
        """移除指定层的所有专家."""
        keys_to_remove = [k for k in self._cache if k.startswith(f"L{layer}_")]
        for key in keys_to_remove:
            entry = self._cache.pop(key)
            self.stats.bytes_used -= entry.size_bytes


class RouterHistory:
    """
    Token 路由历史追踪器.
    
    用于预测下一个 token 可能使用的专家.
    支持:
    - 时间衰减的频率统计
    - 专家共现分析
    - 短期/长期记忆
    """
    
    def __init__(self, max_history: int = 2000, decay_factor: float = 0.98):
        self.max_history = max_history
        self.decay_factor = decay_factor
        
        # 每层的路由历史: {layer: [expert_ids_list]}
        self._history: Dict[int, List[List[int]]] = defaultdict(list)
        
        # 专家频率: {layer: {expert_id: weight}}
        self._freq: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        
        # 专家共现: {layer: {(e1, e2): count}}
        self._cooccur: Dict[int, Dict[Tuple[int, int], float]] = defaultdict(lambda: defaultdict(float))
    
    def record(self, layer: int, expert_ids: List[int]):
        """记录一次 token 路由."""
        history = self._history[layer]
        history.append(expert_ids)
        
        # 限制历史长度
        if len(history) > self.max_history:
            history[:] = history[-self.max_history:]
        
        # 更新频率 (带时间衰减)
        freq = self._freq[layer]
        for eid in list(freq.keys()):
            freq[eid] *= self.decay_factor
        
        for eid in expert_ids:
            freq[eid] += 1.0
        
        # 更新共现
        cooccur = self._cooccur[layer]
        for i, e1 in enumerate(expert_ids):
            for e2 in expert_ids[i+1:]:
                pair = (min(e1, e2), max(e1, e2))
                cooccur[pair] += 1.0
    
    def predict_next(self, layer: int, current_experts: List[int], 
                    top_k: int = 8) -> List[int]:
        """
        预测下一个 token 可能使用的专家.
        
        策略:
        1. 基于历史频率的 Top-K
        2. 加上与当前专家共现频率高的专家
        """
        freq = self._freq[layer]
        if not freq:
            return list(range(top_k))
        
        # 基础预测: 频率最高的专家
        base_pred = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        predicted = set(eid for eid, _ in base_pred[:top_k])
        
        # 共现增强: 如果有当前专家，加上共现频率高的
        if current_experts:
            cooccur = self._cooccur[layer]
            cooc_experts: Dict[int, float] = defaultdict(float)
            
            for eid in current_experts:
                # 查找与 eid 共现频率高的专家
                for (e1, e2), count in cooccur.items():
                    if e1 == eid and e2 not in predicted:
                        cooc_experts[e2] += count
                    elif e2 == eid and e1 not in predicted:
                        cooc_experts[e1] += count
            
            # 添加共现频率最高的专家
            cooc_sorted = sorted(cooc_experts.items(), key=lambda x: x[1], reverse=True)
            for eid, _ in cooc_sorted[:top_k // 2]:
                predicted.add(eid)
        
        return list(predicted)[:top_k]
    
    def get_most_frequent(self, layer: int, top_k: int = 8) -> List[int]:
        """获取指定层最常用的 Top-K 专家."""
        freq = self._freq[layer]
        if not freq:
            return list(range(top_k))
        
        sorted_experts = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [eid for eid, _ in sorted_experts[:top_k]]


class PDExpertStreamer:
    """
    PD 分离式双 GPU 专家流流器.
    
    核心设计:
    1. 层分配: prefill 层 → GPU0, decode 层 → GPU1
    2. 按需加载: 每次只加载当前 token 需要的 top-K 专家
    3. 智能缓存: LRU + 频率 + 共现的多级缓存策略
    4. 无缝切换: Prefill → Decode 时自动迁移专家
    
    使用示例:
        streamer = PDExpertStreamer(
            gguf_path="model.gguf",
            gpu0_mem_gb=4.0,   # Intel UHD
            gpu1_mem_gb=2.0,   # NVIDIA MX250
            max_experts_per_layer=8,
            prefill_ratio=0.5,
        )
        
        # Prefill 阶段
        streamer.enter_prefill()
        for token in prompt_tokens:
            experts = streamer.load_prefill_experts(token_routes)
            # ... compute with experts ...
            streamer.record_routes(token_routes)
        
        # 切换到 Decode
        streamer.switch_to_decode()
        
        # Decode 阶段
        for token in range(max_new_tokens):
            experts = streamer.load_decode_experts(token_routes)
            # ... compute with experts ...
            streamer.record_routes(token_routes)
    """
    
    def __init__(self, 
                 gguf_path: str,
                 gpu0_mem_gb: float = 4.0,
                 gpu1_mem_gb: float = 2.0,
                 max_experts_per_layer: int = 8,
                 prefill_ratio: float = 0.5):
        """
        初始化 PD 流流器.
        
        Args:
            gguf_path: GGUF 模型路径
            gpu0_mem_gb: GPU 0 (Intel UHD) 显存
            gpu1_mem_gb: GPU 1 (NVIDIA MX250) 显存
            max_experts_per_layer: 每层最多缓存的专家数
            prefill_ratio: Prefill 层占比 (前 N 层用于 prefill)
        """
        self.gguf_path = gguf_path
        self.max_experts_per_layer = max_experts_per_layer
        
        # 基础流流器 (解析 GGUF 元数据)
        print("[PD] Initializing base streamer...")
        t0 = time.time()
        self.base_streamer = UnifiedExpertStreamer(gguf_path)
        print(f"[PD] Base streamer ready in {time.time() - t0:.2f}s")
        
        # 获取模型信息
        self.adapter = self.base_streamer.adapter
        self.layers = self.adapter.list_layers()
        self.total_layers = len(self.layers)
        
        # 每层的专家数
        if self.layers:
            self.experts_per_layer = self.adapter.num_experts(self.layers[0])
        else:
            self.experts_per_layer = 8
        
        # PD 层分配
        self.prefill_layers = self.layers[:int(self.total_layers * prefill_ratio)]
        self.decode_layers = self.layers[int(self.total_layers * prefill_ratio):]
        
        # 双 GPU 缓存
        self.gpu0_cache = DualGPUCache("Intel UHD", gpu0_mem_gb)
        self.gpu1_cache = DualGPUCache("NVIDIA MX250", gpu1_mem_gb)
        
        # 路由历史追踪
        self.router_history = RouterHistory()
        
        # 当前阶段
        self.current_phase = Phase.IDLE
        
        # 性能统计
        self.perf_stats = {
            'prefill_time_ms': 0,
            'decode_time_ms': 0,
            'expert_load_time_ms': 0,
            'prefetch_time_ms': 0,
            'tokens_processed': 0,
            'experts_loaded': 0,
            'bytes_transferred': 0,
        }
        
        self._print_config()
    
    def _print_config(self):
        """打印配置信息."""
        print("\n" + "=" * 70)
        print("PD DUAL-GPU EXPERT STREAMER CONFIGURATION")
        print("=" * 70)
        print(f"  Model: {os.path.basename(self.gguf_path)}")
        print(f"  Total Layers: {self.total_layers}")
        print(f"  Experts/Layer: {self.experts_per_layer}")
        print(f"  Max Experts/Layer (cache): {self.max_experts_per_layer}")
        print(f"  Total Expert Slots: {self.total_layers * self.experts_per_layer}")
        print()
        print(f"  GPU 0 (Intel UHD 4GB):")
        print(f"    - Prefill Layers: {len(self.prefill_layers)} layers")
        print(f"    - Cache: {self.gpu0_cache.max_memory_bytes / 1024**3:.2f} GB")
        print(f"    - Max Experts Cache: estimate based on quant size")
        print()
        print(f"  GPU 1 (NVIDIA MX250 2GB):")
        print(f"    - Decode Layers: {len(self.decode_layers)} layers")
        print(f"    - Cache: {self.gpu1_cache.max_memory_bytes / 1024**3:.2f} GB")
        print(f"    - Max Experts Cache: estimate based on quant size")
        print()
        print(f"  PD Separation:")
        print(f"    - Prefill → GPU 0 (Intel UHD)")
        print(f"    - Decode → GPU 1 (NVIDIA MX250)")
        print(f"    - Switch: After prefill completes")
        print("=" * 70)
    
    def enter_prefill(self):
        """进入 Prefill 阶段."""
        print("\n[PD] Entering PREFILL phase...")
        self.current_phase = Phase.PREFILL
        self.gpu0_cache.clear()
        self.gpu1_cache.clear()
        
        # 预加载 prefill 层的 top-K 专家
        print(f"[PD] Preloading top-{self.max_experts_per_layer} experts for {len(self.prefill_layers)} prefill layers...")
        loaded = 0
        failed = 0
        
        for layer in self.prefill_layers:
            for eid in range(min(self.max_experts_per_layer, self.experts_per_layer)):
                try:
                    if self._load_expert_to_gpu(layer, eid, self.gpu0_cache):
                        loaded += 1
                except Exception as e:
                    failed += 1
                    if failed >= 5:
                        print(f"[PD] Too many failures ({failed}), stopping prefill load")
                        break
        
        print(f"[PD] Loaded {loaded} experts for prefill")
        print(f"[PD] GPU 0: {self.gpu0_cache.expert_count} experts, {self.gpu0_cache.current_bytes / 1024**2:.1f} MB used")
    
    def load_prefill_experts(self, routes: Dict[int, List[int]]) -> Dict[int, List[dict]]:
        """
        加载 Prefill 阶段的专家.
        
        Args:
            routes: {layer_id: [expert_ids]} 当前 token 的路由
        
        Returns:
            {layer_id: [expert_data_dicts]} 加载的专家数据
        """
        if self.current_phase != Phase.PREFILL:
            return {}
        
        loaded_experts = {}
        
        for layer, expert_ids in routes.items():
            if layer not in self.prefill_layers:
                continue
            
            layer_experts = []
            for eid in expert_ids:
                entry = self.gpu0_cache.get(layer, eid)
                if entry is None:
                    # 缓存未命中，加载到 GPU 0
                    if self._load_expert_to_gpu(layer, eid, self.gpu0_cache):
                        entry = self.gpu0_cache.get(layer, eid)
                
                if entry:
                    layer_experts.append({
                        'layer': layer,
                        'expert_id': eid,
                        'gate': entry.gate_bytes,
                        'up': entry.up_bytes,
                        'down': entry.down_bytes,
                    })
            
            loaded_experts[layer] = layer_experts
        
        self.perf_stats['tokens_processed'] += 1
        return loaded_experts
    
    def switch_to_decode(self):
        """
        切换到 Decode 阶段.
        
        流程:
        1. 记录 prefill 阶段的路由历史
        2. 释放 GPU 0 缓存 (prefill 完成后不再需要)
        3. 预加载 decode 层的热门专家到 GPU 1
        4. 调整缓存策略 (decode 更激进的 LRU)
        """
        print("\n" + "=" * 70)
        print("[PD] SWITCHING TO DECODE PHASE")
        print("=" * 70)
        
        t0 = time.time()
        self.current_phase = Phase.DECODE
        
        # 1. 保存路由历史 (已在 record_routes 中完成)
        
        # 2. 释放 GPU 0 (prefill 完成)
        gpu0_experts = self.gpu0_cache.expert_count
        self.gpu0_cache.clear()
        print(f"[PD] Released GPU 0 cache ({gpu0_experts} experts)")
        print(f"[PD] GPU 0 now free: {self.gpu0_cache.free_bytes / 1024**2:.1f} MB")
        
        # 3. 预加载 decode 层的热门专家
        print(f"[PD] Preloading decode experts to GPU 1...")
        preloaded = 0
        
        for layer in self.decode_layers:
            # 获取该层的热门专家 (基于 prefill 路由历史)
            hot_experts = self.router_history.get_most_frequent(layer, self.max_experts_per_layer // 2)
            
            if not hot_experts:
                # 无历史，加载前 N 个
                hot_experts = list(range(min(4, self.experts_per_layer)))
            
            for eid in hot_experts:
                if self._load_expert_to_gpu(layer, eid, self.gpu1_cache):
                    preloaded += 1
        
        print(f"[PD] Preloaded {preloaded} decode experts to GPU 1")
        print(f"[PD] GPU 1: {self.gpu1_cache.expert_count} experts, {self.gpu1_cache.current_bytes / 1024**2:.1f} MB used")
        
        # 4. 预取预测的下一批专家
        for layer in self.decode_layers[:5]:
            predicted = self.router_history.predict_next(layer, [], self.max_experts_per_layer)
            for eid in predicted:
                if eid not in self._get_cached_expert_ids(layer, self.gpu1_cache):
                    self._load_expert_to_gpu(layer, eid, self.gpu1_cache)
        
        switch_time = (time.time() - t0) * 1000
        self.perf_stats['prefill_time_ms'] += switch_time
        print(f"[PD] Switch completed in {switch_time:.1f} ms")
    
    def load_decode_experts(self, routes: Dict[int, List[int]]) -> Dict[int, List[dict]]:
        """
        加载 Decode 阶段的专家.
        
        Args:
            routes: {layer_id: [expert_ids]} 当前 token 的路由
        
        Returns:
            {layer_id: [expert_data_dicts]} 加载的专家数据
        """
        if self.current_phase != Phase.DECODE:
            return {}
        
        loaded_experts = {}
        
        for layer, expert_ids in routes.items():
            if layer not in self.decode_layers:
                continue
            
            layer_experts = []
            for eid in expert_ids:
                entry = self.gpu1_cache.get(layer, eid)
                if entry is None:
                    # 缓存未命中，加载到 GPU 1
                    t_load = time.time()
                    if self._load_expert_to_gpu(layer, eid, self.gpu1_cache):
                        entry = self.gpu1_cache.get(layer, eid)
                        self.perf_stats['expert_load_time_ms'] += (time.time() - t_load) * 1000
                        self.perf_stats['experts_loaded'] += 1
                
                if entry:
                    layer_experts.append({
                        'layer': layer,
                        'expert_id': eid,
                        'gate': entry.gate_bytes,
                        'up': entry.up_bytes,
                        'down': entry.down_bytes,
                    })
            
            loaded_experts[layer] = layer_experts
        
        # 触发预取 (为下一个 token 预热)
        self._trigger_prefetch(routes)
        
        self.perf_stats['tokens_processed'] += 1
        return loaded_experts
    
    def record_routes(self, routes: Dict[int, List[int]]):
        """
        记录 token 路由用于预测.
        
        Args:
            routes: {layer_id: [expert_ids]}
        """
        for layer, expert_ids in routes.items():
            self.router_history.record(layer, expert_ids)
    
    def _load_expert_to_gpu(self, layer: int, expert_id: int, 
                            cache: DualGPUCache) -> bool:
        """
        加载单个专家到指定 GPU 缓存.
        
        Args:
            layer: 层 ID
            expert_id: 专家 ID
            cache: 目标 GPU 缓存
        
        Returns:
            True if successfully loaded
        """
        # 通过基础流流器获取专家数据
        expert_data = self.base_streamer.load_expert(layer, expert_id)
        
        if not expert_data:
            return False
        
        # 计算大小
        size_bytes = 0
        for key in ['gate', 'up', 'down', 'gate_inp']:
            data = expert_data.get(key)
            if isinstance(data, bytes):
                size_bytes += len(data)
        
        # 创建缓存条目
        entry = ExpertCacheEntry(
            layer=layer,
            expert_id=expert_id,
            gate_bytes=expert_data.get('gate'),
            up_bytes=expert_data.get('up'),
            down_bytes=expert_data.get('down'),
            gate_inp_bytes=expert_data.get('gate_inp'),
            last_access=time.time(),
            access_count=1,
            size_bytes=size_bytes,
        )
        
        return cache.put(entry)
    
    def _trigger_prefetch(self, current_routes: Dict[int, List[int]]):
        """
        触发专家预取.
        
        基于当前 token 的路由，预测并预取下一个 token 可能使用的专家.
        """
        if self.current_phase != Phase.DECODE:
            return
        
        t0 = time.time()
        prefetched = 0
        
        for layer in self.decode_layers:
            current_experts = current_routes.get(layer, [])
            predicted = self.router_history.predict_next(
                layer, current_experts, self.max_experts_per_layer
            )
            
            # 加载预测的专家 (如果不在缓存中)
            for eid in predicted:
                if eid not in self._get_cached_expert_ids(layer, self.gpu1_cache):
                    if self._load_expert_to_gpu(layer, eid, self.gpu1_cache):
                        prefetched += 1
        
        if prefetched > 0:
            self.perf_stats['prefetch_time_ms'] += (time.time() - t0) * 1000
    
    def _get_cached_expert_ids(self, layer: int, cache: DualGPUCache) -> Set[int]:
        """获取指定层在缓存中的专家 ID 集合."""
        ids = set()
        prefix = f"L{layer}_E"
        for key in cache._cache:
            if key.startswith(prefix):
                try:
                    eid = int(key.split("_E")[1])
                    ids.add(eid)
                except:
                    pass
        return ids
    
    def get_stats(self) -> dict:
        """获取完整统计信息."""
        return {
            'phase': self.current_phase.value,
            'model': {
                'total_layers': self.total_layers,
                'experts_per_layer': self.experts_per_layer,
                'max_experts_cache': self.max_experts_per_layer,
            },
            'gpu0': {
                'name': 'Intel UHD Graphics',
                'cache_count': self.gpu0_cache.expert_count,
                'cache_mb': self.gpu0_cache.current_bytes / 1024**2,
                'max_mb': self.gpu0_cache.max_memory_bytes / 1024**2,
                'hit_rate': self.gpu0_cache.stats.hit_rate,
                'hits': self.gpu0_cache.stats.hits,
                'misses': self.gpu0_cache.stats.misses,
                'evictions': self.gpu0_cache.stats.evictions,
            },
            'gpu1': {
                'name': 'NVIDIA GeForce MX250',
                'cache_count': self.gpu1_cache.expert_count,
                'cache_mb': self.gpu1_cache.current_bytes / 1024**2,
                'max_mb': self.gpu1_cache.max_memory_bytes / 1024**2,
                'hit_rate': self.gpu1_cache.stats.hit_rate,
                'hits': self.gpu1_cache.stats.hits,
                'misses': self.gpu1_cache.stats.misses,
                'evictions': self.gpu1_cache.stats.evictions,
            },
            'performance': {
                'tokens_processed': self.perf_stats['tokens_processed'],
                'experts_loaded': self.perf_stats['experts_loaded'],
                'prefill_time_ms': self.perf_stats['prefill_time_ms'],
                'decode_time_ms': self.perf_stats['decode_time_ms'],
                'expert_load_time_ms': self.perf_stats['expert_load_time_ms'],
                'prefetch_time_ms': self.perf_stats['prefetch_time_ms'],
            },
        }


def test_dual_gpu_streamer():
    """测试双 GPU PD 流流器."""
    print("=" * 80)
    print("DUAL-GPU PD EXPERT STREAMER - TEST")
    print("=" * 80)
    
    model_path = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found: {model_path}")
        return 1
    
    # 初始化
    print("\n🚀 Initializing DualGPU PD Streamer...")
    streamer = PDExpertStreamer(
        gguf_path=model_path,
        gpu0_mem_gb=4.0,   # Intel UHD
        gpu1_mem_gb=2.0,   # NVIDIA MX250
        max_experts_per_layer=8,
        prefill_ratio=0.5,
    )
    
    # ======== Prefill 阶段测试 ========
    print("\n📦 Prefill Phase Test")
    print("-" * 40)
    
    streamer.enter_prefill()
    
    # 模拟 10 个 token 的 prefill 路由
    print("  Simulating 10 prefill tokens...")
    for token_idx in range(10):
        routes = {}
        for layer in streamer.prefill_layers:
            # 随机选择 top-8 专家中的 4 个
            expert_ids = [(token_idx + i) % 256 for i in range(4)]
            routes[layer] = expert_ids
        
        experts = streamer.load_prefill_experts(routes)
        streamer.record_routes(routes)
    
    print(f"  Processed 10 prefill tokens")
    print(f"  GPU 0 cache: {streamer.gpu0_cache.expert_count} experts")
    print(f"  GPU 0 hit rate: {streamer.gpu0_cache.stats.hit_rate:.1f}%")
    
    # ======== 切换到 Decode ========
    print("\n🔄 Switch to Decode")
    print("-" * 40)
    
    streamer.switch_to_decode()
    
    # ======== Decode 阶段测试 ========
    print("\n⚡ Decode Phase Test")
    print("-" * 40)
    
    decode_results = []
    for token_idx in range(20):
        routes = {}
        for layer in streamer.decode_layers:
            expert_ids = [(token_idx * 3 + i) % 256 for i in range(4)]
            routes[layer] = expert_ids
        
        t0 = time.time()
        experts = streamer.load_decode_experts(routes)
        load_time = (time.time() - t0) * 1000
        streamer.record_routes(routes)
        
        decode_results.append({
            'token': token_idx,
            'experts_loaded': len(experts),
            'load_time_ms': load_time,
            'gpu1_cache': streamer.gpu1_cache.expert_count,
        })
    
    print(f"  Processed 20 decode tokens")
    print(f"  Average load time: {sum(r['load_time_ms'] for r in decode_results) / len(decode_results):.3f} ms/token")
    print(f"  GPU 1 cache: {streamer.gpu1_cache.expert_count} experts")
    print(f"  GPU 1 hit rate: {streamer.gpu1_cache.stats.hit_rate:.1f}%")
    
    # ======== 统计 ========
    print("\n📊 Final Statistics")
    print("-" * 40)
    
    stats = streamer.get_stats()
    print(json.dumps(stats, indent=2, default=str))
    
    return 0


def test_single_expert_loading():
    """测试单个专家的加载速度."""
    print("=" * 80)
    print("SINGLE EXPERT LOADING TEST")
    print("=" * 80)
    
    model_path = r"D:\alex\flashkv0516\models\gguf\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found: {model_path}")
        return 1
    
    streamer = PDExpertStreamer(
        gguf_path=model_path,
        gpu0_mem_gb=4.0,
        gpu1_mem_gb=2.0,
        max_experts_per_layer=8,
    )
    
    # 测试加载速度
    test_layers = [0, 5, 10, 15, 20]
    test_experts = [0, 3, 7]
    
    print("\n  Loading individual experts...")
    times = []
    
    for layer in test_layers:
        for expert_id in test_experts:
            t0 = time.time()
            expert = streamer.base_streamer.load_expert(layer, expert_id)
            load_time = (time.time() - t0) * 1000
            times.append(load_time)
            
            if expert:
                size_kb = sum(len(v) for v in expert.values() if isinstance(v, bytes)) / 1024
                print(f"  Layer {layer}, Expert {expert_id}: {load_time:.2f} ms, {size_kb:.1f} KB")
            else:
                print(f"  Layer {layer}, Expert {expert_id}: FAILED")
    
    avg_time = sum(times) / len(times) if times else 0
    print(f"\n  Average load time: {avg_time:.2f} ms/expert")
    print(f"  Estimated 8-expert load: {avg_time * 8:.2f} ms")
    print(f"  Estimated 32-layer × 8-expert load: {avg_time * 8 * 32 / 1000:.2f} s")
    
    return 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Dual-GPU PD Expert Streamer")
    parser.add_argument("--test", choices=["full", "single"], default="full",
                        help="Test type: full (PD flow) or single (load speed)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to GGUF model")
    
    args = parser.parse_args()
    
    if args.test == "single":
        test_single_expert_loading()
    else:
        test_dual_gpu_streamer()
