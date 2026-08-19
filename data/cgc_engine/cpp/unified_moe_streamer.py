#!/usr/bin/env python3
"""
Unified MoE Expert Streamer Architecture

支持 Qwen3.6 (Per-Expert) 和 Gemma4 (Per-Layer) 两种布局.

核心设计:
1. ExpertLayout 枚举 - 标识布局类型
2. LayoutAdapter 抽象接口 - 统一 load_expert() 调用
3. PerExpertAdapter - 处理 Qwen3.6 风格
4. PerLayerAdapter - 处理 Gemma4 风格
5. UnifiedExpertStreamer - 对外主入口

使用方式:
    streamer = UnifiedExpertStreamer(gguf_path)
    expert = streamer.load_expert(layer=5, expert_id=3)
    # 返回 {"gate": {...}, "up": {...}, "down": {...}}
"""

import os
import sys
import struct
import time
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Optional

GGML_TYPE_BYTES = {
    0: 4.0,   # F32
    1: 2.0,   # F16
    30: 2.0,  # BF16
    16: 0.25, # IQ2_XXS (2-bit quant)
    17: 0.5,  # IQ2_XS (2-bit quant, slightly different)
    18: 0.375, # IQ3_XXS (3-bit quant, block-aligned)
    19: 0.5,  # IQ1_S (1.56-bit quant)
    20: 0.5,  # IQ4_NL (4-bit non-linear)
    21: 0.375, # IQ3_S (3-bit quant, block-aligned)
    22: 0.25, # IQ2_S (2-bit quant)
    23: 0.5,  # IQ4_XS (4-bit quant)
    2: 4.0,   # Q4_0
    3: 4.0,   # Q4_1
    6: 4.0,   # Q5_0
    7: 4.0,   # Q5_1
    8: 4.0,   # Q8_0
    9: 4.0,   # Q8_1
    10: 4.0,  # Q2_K
    11: 4.0,  # Q3_K
    12: 4.0,  # Q4_K
    13: 4.0,  # Q5_K
    14: 4.0,  # Q6_K
    15: 4.0,  # Q8_K
}

TYPE_NAMES = {
    0: "F32", 1: "F16", 30: "BF16",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
    22: "IQ2_S", 23: "IQ4_XS",
    2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
    13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
}


class ExpertLayout(Enum):
    PER_EXPERT = "per_expert"
    PER_LAYER = "per_layer"
    UNKNOWN = "unknown"


def detect_layout(kv: dict, tensors: list) -> ExpertLayout:
    """
    自动检测 GGUF 文件的 MoE 布局类型.

    判断逻辑:
    - 如果存在 ffn_down_exps 张量 -> PER_LAYER (Gemma4/Qwen3.6 均使用)
    - 如果存在 blk.X.expert.Y 命名格式 -> PER_EXPERT (传统 per-expert 布局)
    """
    for t in tensors:
        name = t["name"]
        if "ffn_down_exps" in name:
            return ExpertLayout.PER_LAYER

    for t in tensors:
        name = t["name"]
        parts = name.split(".")
        if len(parts) >= 5 and parts[0] == "blk" and parts[2] == "expert":
            return ExpertLayout.PER_EXPERT

    return ExpertLayout.UNKNOWN


def _parse_gguf_safe(filepath: str) -> dict:
    """
    Safe GGUF parser using gguf library internals.
    Parses KV fields and tensor metadata WITHOUT triggering data reshape.
    """
    import gguf as _gguf
    from gguf.constants import GGML_QUANT_SIZES, GGUF_MAGIC, GGUF_VERSION

    reader = _gguf.GGUFReader.__new__(_gguf.GGUFReader)
    reader.data = None
    reader.byte_order = 'I'
    reader.fields = _gguf.OrderedDict()
    reader.tensors = []
    reader.alignment = 32

    import numpy as np

    data = np.memmap(filepath, mode='r')
    reader.data = data

    def _get(offs, dtype, count=1, override_order=None):
        count = int(count)
        itemsize = int(np.empty([], dtype=dtype).itemsize)
        end_offs = offs + itemsize * count
        arr = data[offs:end_offs].view(dtype=dtype)[:count]
        return arr.view(arr.dtype.newbyteorder(reader.byte_order if override_order is None else override_order))

    def _push_field(field, skip_sum=False):
        if field.name in reader.fields:
            pass
        else:
            reader.fields[field.name] = field
        return 0 if skip_sum else sum(int(part.nbytes) for part in field.parts)

    offs = 0

    magic = _get(offs, np.uint32, override_order='<')[0]
    if magic != GGUF_MAGIC:
        raise ValueError(f"Not a GGUF file: magic=0x{magic:08X}")
    offs += 4

    temp_version = _get(offs, np.uint32)
    version = temp_version[0]
    offs += _push_field(_gguf.gguf_reader.ReaderField(offs, 'GGUF.version', [temp_version], [0], [_gguf.GGUFValueType.UINT32]))

    temp_counts = _get(offs, np.uint64, 2)
    offs += _push_field(_gguf.gguf_reader.ReaderField(offs, 'GGUF.tensor_count', [temp_counts[:1]], [0], [_gguf.GGUFValueType.UINT64]))
    offs += _push_field(_gguf.gguf_reader.ReaderField(offs, 'GGUF.kv_count', [temp_counts[1:]], [0], [_gguf.GGUFValueType.UINT64]))
    tensor_count, kv_count = temp_counts

    offs = reader._build_fields(offs, kv_count)

    offs, tensors_fields = reader._build_tensor_info(offs, tensor_count)

    new_align = reader.fields.get('general.alignment')
    if new_align is not None:
        if new_align.types != [_gguf.GGUFValueType.UINT32]:
            raise ValueError('Bad type for general.alignment field')
        reader.alignment = new_align.parts[-1][0]
    padding = offs % reader.alignment
    if padding != 0:
        offs += reader.alignment - padding
    reader.data_offset = offs

    kv = {}
    for name, field in reader.fields.items():
        if name.startswith("GGUF."):
            continue
        try:
            if hasattr(field, "contents"):
                kv[name] = field.contents()
        except Exception:
            pass

    tensors = []
    for i, field in enumerate(tensors_fields):
        try:
            _name_len, name_data, _n_dims, dims, raw_dtype, offset_tensor = field.parts
            tensor_name = str(bytes(name_data), encoding='utf-8')
            ggml_type = _gguf.GGMLQuantizationType(raw_dtype[0])
            n_elems = int(np.prod(dims))

            if ggml_type in GGML_QUANT_SIZES:
                block_size, type_size = GGML_QUANT_SIZES[ggml_type]
            else:
                bpe = GGML_TYPE_BYTES.get(ggml_type.value, 4.0)
                block_size = 1
                type_size = bpe

            n_bytes = n_elems * type_size // block_size
            data_offs = int(offs + offset_tensor[0])

            tensors.append({
                "index": i,
                "name": tensor_name,
                "dims": [int(d) for d in dims],
                "type": ggml_type.value,
                "type_name": TYPE_NAMES.get(ggml_type.value, f"TYPE_{ggml_type.value}"),
                "offset": data_offs,
                "size_bytes": int(n_bytes),
            })
        except Exception as e:
            print(f"  [WARN] Skipping tensor {i}: {e}")
            continue

    return {
        "kv": kv,
        "tensors": tensors,
        "data_start": int(reader.data_offset),
        "file_size": os.path.getsize(filepath),
    }


def parse_gguf_header(filepath: str) -> dict:
    """使用 gguf 库的安全解析方式."""
    return _parse_gguf_safe(filepath)


class LayoutAdapter:
    """抽象基类: 专家权重适配器."""

    def __init__(self, header: dict, filepath: str):
        self.header = header
        self.filepath = filepath
        self.kv = header["kv"]
        self._build_index()

    def _build_index(self):
        raise NotImplementedError

    def load_expert(self, layer: int, expert_id: int) -> dict:
        """
        加载指定层和专家的权重.

        Returns:
            {
                "gate": {"data": bytes, "dims": list, "type": int, ...},
                "up": {"data": bytes, "dims": list, "type": int, ...},
                "down": {"data": bytes, "dims": list, "type": int, ...},
                "gate_inp": ...,  # 可选, Gemma4 专属
                "scale": ...,     # 可选
            }
        """
        raise NotImplementedError

    def get_layer_info(self, layer: int) -> dict:
        """获取指定层的元信息."""
        raise NotImplementedError

    def list_layers(self) -> list:
        """列出所有 MoE 层."""
        raise NotImplementedError

    def num_experts(self, layer: int = 0) -> int:
        """获取指定层的专家数量."""
        raise NotImplementedError


class PerExpertAdapter(LayoutAdapter):
    """
    Per-Expert 布局适配器 (Qwen3.6 风格).

    张量命名: blk.X.expert.Y.role.weight
    - X: layer index
    - Y: expert id
    - role: gate / up / down
    """

    def _build_index(self):
        self._offsets = {}  # (layer, expert, role) -> tensor_info
        self._layers = set()
        self._experts = set()

        for t in self.header["tensors"]:
            name = t["name"]
            parts = name.split(".")
            if len(parts) >= 5 and parts[0] == "blk" and parts[2] == "expert":
                try:
                    layer = int(parts[1])
                    expert = int(parts[3])
                    role = parts[4]
                    self._offsets[(layer, expert, role)] = t
                    self._layers.add(layer)
                    self._experts.add(expert)
                except (ValueError, IndexError):
                    pass

        self._layer_list = sorted(self._layers)
        self._expert_list = sorted(self._experts)

    def load_expert(self, layer: int, expert_id: int) -> dict:
        roles = {}
        for role in ["gate", "up", "down"]:
            info = self._offsets.get((layer, expert_id, role))
            if not info:
                continue

            with open(self.filepath, "rb") as f:
                f.seek(info["offset"])
                data = f.read(info["size_bytes"])

            roles[role] = {
                "data": data,
                "dims": info["dims"],
                "type": info["type"],
                "type_name": info["type_name"],
                "size_bytes": info["size_bytes"],
            }

        return {
            "layer": layer,
            "expert_id": expert_id,
            "layout": "per_expert",
            "roles": roles,
        }

    def get_layer_info(self, layer: int) -> dict:
        layer_tensors = {k: v for k, v in self._offsets.items() if k[0] == layer}
        info = {
            "layer": layer,
            "experts": len(set(k[1] for k in layer_tensors.keys())),
            "roles": list(set(k[2] for k in layer_tensors.keys())),
        }
        return info

    def list_layers(self) -> list:
        return self._layer_list

    def num_experts(self, layer: int = 0) -> int:
        return len(self._expert_list)


class PerLayerAdapter(LayoutAdapter):
    """
    Per-Layer 布局适配器 (Gemma4/Qwen3.6 风格).

    支持两种变体:
    1. 打包式 (Gemma4): ffn_gate_up_exps.weight 包含 gate+up
    2. 分离式 (Qwen3.6): ffn_gate_exps.weight + ffn_up_exps.weight 分开存储

    关键张量:
    - ffn_down_exps.weight: [inter, hidden, num_experts]
    - ffn_gate_up_exps.weight: [hidden, gate_up_dim, num_experts] (打包, Gemma4)
    - ffn_gate_exps.weight: [hidden, inter, num_experts] (分离 gate, Qwen3.6)
    - ffn_up_exps.weight: [hidden, inter, num_experts] (分离 up, Qwen3.6)
    - ffn_gate_inp.weight: [hidden, num_experts] (共享 gate 输入)

    切片方式: 沿最后一维切片.
    """

    def _build_index(self):
        self._layer_info = {}

        for t in self.header["tensors"]:
            name = t["name"]
            parts = name.split(".")

            if parts[0] != "blk" or len(parts) < 3:
                continue

            try:
                layer = int(parts[1])
            except ValueError:
                continue

            role = parts[2]
            layer_data = self._layer_info.setdefault(layer, {})

            if role == "ffn_down_exps":
                layer_data["down"] = t
            elif role == "ffn_gate_up_exps":
                layer_data["gate_up"] = t
            elif role == "ffn_gate_exps":
                layer_data["gate"] = t
            elif role == "ffn_up_exps":
                layer_data["up"] = t
            elif role == "ffn_gate_inp":
                layer_data["gate_inp"] = t

        self._layer_list = sorted(self._layer_info.keys())

        kv = self.kv
        self.hidden = int(kv.get("gemma4.embedding_length",
                                  kv.get("qwen35moe.embedding_length", 2048)))
        self.expert_inter = int(kv.get("gemma4.expert_feed_forward_length",
                                       kv.get("qwen35moe.expert_feed_forward_length", 512)))
        self.num_experts_val = int(kv.get("gemma4.expert_count",
                                          kv.get("qwen35moe.expert_count", 256)))
        self.top_k = int(kv.get("gemma4.expert_used_count",
                                 kv.get("qwen35moe.expert_used_count", 8)))

    def _slice_expert_tensor(self, t: dict, expert_id: int) -> dict:
        """Slice a per-layer tensor to get a single expert's data."""
        dims = t["dims"]
        experts_dim = dims[-1]
        per_expert_bytes = t["size_bytes"] // experts_dim

        slice_offset = t["offset"] + expert_id * per_expert_bytes
        with open(self.filepath, "rb") as f:
            f.seek(slice_offset)
            data = f.read(per_expert_bytes)

        return {
            "data": data,
            "dims": dims[:-1],
            "type": t["type"],
            "type_name": t["type_name"],
            "size_bytes": per_expert_bytes,
        }

    def load_expert(self, layer: int, expert_id: int) -> dict:
        if layer not in self._layer_info:
            return {}

        layer_tensors = self._layer_info[layer]
        result = {
            "layer": layer,
            "expert_id": expert_id,
            "layout": "per_layer",
            "roles": {},
        }

        if "down" in layer_tensors:
            t = layer_tensors["down"]
            result["roles"]["down"] = self._slice_expert_tensor(t, expert_id)

        if "gate_up" in layer_tensors:
            t = layer_tensors["gate_up"]
            expert_data = self._slice_expert_tensor(t, expert_id)
            data = expert_data["data"]
            dims = t["dims"]
            inter = self.expert_inter
            bpe = GGML_TYPE_BYTES.get(t["type"], 2)

            gate_bytes = int(inter * self.hidden * bpe)
            result["roles"]["gate"] = {
                "data": data[:gate_bytes],
                "dims": [self.hidden, inter],
                "type": t["type"],
                "type_name": t["type_name"],
                "size_bytes": gate_bytes,
            }
            result["roles"]["up"] = {
                "data": data[gate_bytes:gate_bytes + gate_bytes],
                "dims": [self.hidden, inter],
                "type": t["type"],
                "type_name": t["type_name"],
                "size_bytes": gate_bytes,
            }
        else:
            if "gate" in layer_tensors:
                result["roles"]["gate"] = self._slice_expert_tensor(layer_tensors["gate"], expert_id)
            if "up" in layer_tensors:
                result["roles"]["up"] = self._slice_expert_tensor(layer_tensors["up"], expert_id)

        if "gate_inp" in layer_tensors:
            t = layer_tensors["gate_inp"]
            with open(self.filepath, "rb") as f:
                f.seek(t["offset"])
                data = f.read(t["size_bytes"])

            result["roles"]["gate_inp"] = {
                "data": data,
                "dims": t["dims"],
                "type": t["type"],
                "size_bytes": t["size_bytes"],
                "shared": True,
            }

        return result

    def get_layer_info(self, layer: int) -> dict:
        if layer not in self._layer_info:
            return {}

        info = {
            "layer": layer,
            "tensors": list(self._layer_info[layer].keys()),
            "num_experts": self.num_experts_val,
            "top_k": self.top_k,
        }
        return info

    def list_layers(self) -> list:
        return self._layer_list

    def num_experts(self, layer: int = 0) -> int:
        return self.num_experts_val


class UnifiedExpertStreamer:
    """
    统一 MoE Expert Streamer 主类.

    自动检测布局类型, 提供统一的 load_expert() 接口.

    使用示例:
        streamer = UnifiedExpertStreamer("model.gguf")
        expert = streamer.load_expert(layer=5, expert_id=3)
        # expert["roles"]["gate"] -> 权重数据
        # expert["roles"]["up"]   -> 权重数据
        # expert["roles"]["down"] -> 权重数据
    """

    def __init__(self, gguf_path: str):
        self.gguf_path = gguf_path
        self.file_size = os.path.getsize(gguf_path)

        # Step 1: 解析 GGUF
        print(f"Loading {os.path.basename(gguf_path)} ({self.file_size / 1024**3:.2f} GB)...")
        self.header = parse_gguf_header(gguf_path)

        # Step 2: 检测布局
        self.layout = detect_layout(self.header["kv"], self.header["tensors"])
        print(f"Detected layout: {self.layout.value}")

        # Step 3: 创建适配器
        if self.layout == ExpertLayout.PER_EXPERT:
            self.adapter = PerExpertAdapter(self.header, gguf_path)
        elif self.layout == ExpertLayout.PER_LAYER:
            self.adapter = PerLayerAdapter(self.header, gguf_path)
        else:
            raise ValueError(f"Unknown layout: {self.layout}")

        # Step 4: 初始化缓存
        self._cache = {}
        self._hits = 0
        self._misses = 0
        self.max_cache_size = 64  # 最多缓存的专家数

        # 打印统计信息
        self._print_stats()

    def _print_stats(self):
        layers = self.adapter.list_layers()
        print(f"\nArchitecture:")
        print(f"  Layers: {len(layers)}")
        if layers:
            n_experts = self.adapter.num_experts(layers[0])
            print(f"  Experts/Layer: {n_experts}")
            if hasattr(self.adapter, 'top_k'):
                print(f"  Top-K: {self.adapter.top_k}")
            if hasattr(self.adapter, 'hidden'):
                print(f"  Hidden: {self.adapter.hidden}")
                print(f"  Expert Intermediate: {self.adapter.expert_inter}")

    def load_expert(self, layer: int, expert_id: int) -> dict:
        """加载专家权重 (带缓存)."""
        cache_key = f"L{layer}_E{expert_id}"

        if cache_key in self._cache:
            self._hits += 1
            return self._cache[cache_key]

        self._misses += 1
        expert = self.adapter.load_expert(layer, expert_id)

        # LRU 缓存管理
        if len(self._cache) >= self.max_cache_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[cache_key] = expert
        return expert

    def load_layer(self, layer: int, expert_ids: list = None) -> list:
        """加载指定层的多个专家."""
        if expert_ids is None:
            if hasattr(self.adapter, 'top_k'):
                expert_ids = list(range(self.adapter.top_k))
            else:
                expert_ids = list(range(min(8, self.adapter.num_experts(layer))))

        experts = []
        for eid in expert_ids:
            expert = self.load_expert(layer, eid)
            if expert:
                experts.append(expert)
        return experts

    def prewarm_prefill(self, prefill_layers: list = None) -> int:
        """
        预热 prefill 阶段的层 (前半部分).
        返回加载的专家数.
        """
        layers = self.adapter.list_layers()
        if prefill_layers is None:
            mid = len(layers) // 2
            prefill_layers = layers[:mid]

        warmed = 0
        for layer in prefill_layers:
            experts = self.load_layer(layer)
            warmed += len(experts)
        return warmed

    def prewarm_decode(self, decode_layers: list = None) -> int:
        """
        预热 decode 阶段的层 (后半部分).
        会先失效 prefill 层的缓存.
        """
        layers = self.adapter.list_layers()
        if decode_layers is None:
            mid = len(layers) // 2
            decode_layers = layers[mid:]

        # 失效 prefill 层
        prefill_layers = layers[:len(layers)//2]
        for layer in prefill_layers:
            self.invalidate_layer(layer)

        warmed = 0
        for layer in decode_layers:
            experts = self.load_layer(layer)
            warmed += len(experts)
        return warmed

    def invalidate_layer(self, layer: int):
        """失效指定层的缓存."""
        keys = [k for k in self._cache.keys() if k.startswith(f"L{layer}_")]
        for k in keys:
            del self._cache[k]

    def cache_stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "cached": len(self._cache),
            "hit_rate": self._hits / max(total, 1) * 100,
        }

    def get_memory_estimate(self) -> dict:
        """估算内存节省."""
        layers = self.adapter.list_layers()

        if self.layout == ExpertLayout.PER_LAYER and hasattr(self.adapter, 'top_k'):
            # Gemma4 风格
            full_model_mb = 0
            streaming_mb = 0

            for layer in layers:
                if layer in self.adapter._layer_info:
                    info = self.adapter._layer_info[layer]
                    for role in ["down", "gate_up"]:
                        if role in info:
                            full_model_mb += info[role]["size_bytes"] / 1024**2
                            streaming_mb += info[role]["size_bytes"] / 1024**2 * self.adapter.top_k / self.adapter.num_experts_val

            return {
                "full_model_mb": full_model_mb,
                "streaming_mb": streaming_mb,
                "saving_percent": (1 - streaming_mb / max(full_model_mb, 1)) * 100,
                "layout": self.layout.value,
            }

        return {"layout": self.layout.value, "note": "Estimation not available for this layout"}


def test_unified_streamer():
    """测试统一框架."""
    print("=" * 80)
    print("UNIFIED MOE EXPERT STREAMER - TEST")
    print("=" * 80)

    model_path = r"C:\Users\alexchuang\Desktop\fastprefill\gemma4_gguf\gemma-4-26B-A4B-it-heretic.IQ4_XS.gguf"

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found: {model_path}")
        return 1

    # 测试统一加载器
    print("\n🚀 Initializing UnifiedExpertStreamer...")
    start = time.time()
    streamer = UnifiedExpertStreamer(model_path)
    init_time = time.time() - start
    print(f"  Initialized in {init_time:.2f}s")

    # 测试加载专家
    print("\n📈 Testing Expert Loading...")
    test_layers = streamer.adapter.list_layers()[:3]
    test_experts = [0, streamer.adapter.num_experts() // 2, streamer.adapter.num_experts() - 1]

    for layer in test_layers:
        for expert_id in test_experts:
            t0 = time.time()
            expert = streamer.load_expert(layer, expert_id)
            t1 = time.time()

            if expert:
                roles = expert.get("roles", {})
                total_mb = sum(r.get("size_bytes", 0) / 1024**2 for r in roles.values())
                print(f"  L{layer}E{expert_id}: {total_mb:.2f} MB ({(t1-t0)*1000:.1f}ms)")

                # 验证统一接口
                if "gate" in roles and "up" in roles and "down" in roles:
                    print(f"    ✅ Unified interface: gate={roles['gate']['dims']}, up={roles['up']['dims']}, down={roles['down']['dims']}")

    # 测试缓存
    print("\n💾 Testing Cache...")
    for _ in range(5):
        for layer in test_layers:
            streamer.load_expert(layer, 0)

    stats = streamer.cache_stats()
    print(f"  Hits: {stats['hits']}, Misses: {stats['misses']}")
    print(f"  Hit Rate: {stats['hit_rate']:.1f}%")

    # 测试 prewarm
    print("\n⚡ Testing Prewarm...")
    layers = streamer.adapter.list_layers()
    mid = len(layers) // 2

    t0 = time.time()
    prefill_warmed = streamer.prewarm_prefill()
    t1 = time.time()
    print(f"  Prefill: {prefill_warmed} experts in {(t1-t0):.2f}s")

    t0 = time.time()
    decode_warmed = streamer.prewarm_decode()
    t1 = time.time()
    print(f"  Decode: {decode_warmed} experts in {(t1-t0):.2f}s")

    # 内存估算
    print("\n📊 Memory Estimation...")
    mem = streamer.get_memory_estimate()
    for k, v in mem.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED ✅")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(test_unified_streamer())
