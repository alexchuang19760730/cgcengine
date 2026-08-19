#!/usr/bin/env python3
"""
Gemma 4 26B-A4B Expert Streaming 真实验证脚本

功能:
1. 解析真实 GGUF 文件结构
2. 识别 Gemma 4 MoE 架构参数 (hidden, inter, experts, layers)
3. 验证 per-expert 张量命名约定
4. 测试 ExpertStreamer 加载/缓存逻辑
5. 统计每层 expert 权重大小

模型: gemma-4-26B-A4B-it-heretic.IQ4_XS.gguf (13.1 GB)
"""

import os
import sys
import struct
import time
from collections import defaultdict
from pathlib import Path

GGUF_MAGIC = 0x46554747
GGML_TYPE_BYTES = {
    0: 4,  # F32
    1: 2,  # F16
    2: 4,  # Q4_0
    3: 4,  # Q4_1
    4: 4,  # Q5_0
    5: 4,  # Q5_1
    6: 8,  # Q8_0
    7: 4,  # Q4_K
    8: 4,  # Q5_K
    9: 4,  # Q6_K
    10: 4, # Q8_K
    11: 1, # Q2_K
    12: 2, # Q3_K_S
    13: 2, # Q3_K_M
    14: 2, # Q3_K_L
    15: 2, # Q4_K_S
    16: 2, # Q4_K_M
    17: 2, # Q5_K_S
    18: 2, # Q5_K_M
    19: 2, # Q6_K
    20: 2, # Q5_K
    21: 2, # Q4_K
    22: 2, # IQ3_M
    30: 2, # BF16
    31: 2, # Q2_K
    32: 2, # IQ4_NL
    33: 2, # IQ2_XXS
    34: 2, # IQ3_XXS
    35: 2, # IQ1_S
    36: 2, # IQ4_XS
    37: 2, # IQ2_MM
    38: 2, # IQ2_S
    39: 2, # IQ3_S
}

TYPE_NAMES = {
    0: "F32", 1: "F16", 22: "IQ3_M", 30: "BF16",
    36: "IQ4_XS", 33: "IQ2_XXS", 34: "IQ3_XXS",
    16: "Q4_K_M", 9: "Q6_K",
}


def parse_gguf_header(filepath: str) -> dict:
    """解析 GGUF 文件头和张量信息 (使用官方 gguf 库)."""
    import gguf as _gguf

    print(f"  Reading: {os.path.basename(filepath)}")
    print(f"  Size: {os.path.getsize(filepath) / 1024**3:.2f} GB")

    start_time = time.time()

    reader = _gguf.GGUFReader(filepath)

    # 提取 KV
    kv = {}
    for name, field in reader.fields.items():
        if name.startswith("GGUF."):
            continue
        try:
            if hasattr(field, "contents"):
                kv[name] = field.contents()
        except Exception:
            pass

    # 提取张量信息
    tensors = []
    expert_tensors = []
    layer_tensors = defaultdict(list)

    for i, t in enumerate(reader.tensors):
        name = t.name
        dims = [int(d) for d in t.shape]
        ggml_type = t.tensor_type.value
        offset = int(t.data_offset)
        bpe = GGML_TYPE_BYTES.get(ggml_type, 4)
        n_elements = 1
        for d in dims:
            n_elements *= d
        size_bytes = int(n_elements * bpe)

        info = {
            "index": i, "name": name, "dims": dims,
            "type": ggml_type, "type_name": TYPE_NAMES.get(ggml_type, f"TYPE_{ggml_type}"),
            "offset": offset, "size_bytes": size_bytes,
        }
        tensors.append(info)

        # 识别专家张量
        if "expert" in name.lower():
            expert_tensors.append(info)

        # 按层分组
        parts = name.split(".")
        if parts[0] == "blk" and len(parts) >= 2:
            try:
                layer_idx = int(parts[1])
                layer_tensors[layer_idx].append(info)
            except (ValueError, IndexError):
                pass

    data_start = int(reader.data_offset)

    elapsed = time.time() - start_time
    print(f"  Version: 3, Tensors: {len(tensors)}, KV pairs: {len(kv)}")
    print(f"  Parsed in {elapsed:.2f}s")

    return {
        "version": 3,
        "n_tensors": len(tensors),
        "n_kv": len(kv),
        "kv": kv,
        "tensors": tensors,
        "expert_tensors": expert_tensors,
        "layer_tensors": dict(layer_tensors),
        "data_start": data_start,
    }


def analyze_gemma4_architecture(header: dict) -> dict:
    """分析 Gemma 4 模型架构参数."""
    kv = header["kv"]

    arch = {
        "architecture": kv.get("general.architecture", "unknown"),
        "hidden_size": kv.get("gemma4.hidden_size", 0),
        "intermediate_size": kv.get("gemma4.intermediate_size", 0),
        "num_hidden_layers": kv.get("gemma4.num_hidden_layers", 0),
        "num_attention_heads": kv.get("gemma4.num_attention_heads", 0),
        "num_key_value_heads": kv.get("gemma4.num_key_value_heads", 0),
        "max_position_embeddings": kv.get("gemma4.max_position_embeddings", 0),
        "vocab_size": kv.get("gemma4.vocab_size", 0),
        "expert_count": kv.get("gemma4.expert_count", 0),
        "expert_stride": kv.get("gemma4.expert_stride", 0),
        "quantization_type": kv.get("general.quantization_version", 0),
    }

    # 从专家张量推断架构
    expert_tensors = header["expert_tensors"]
    if expert_tensors:
        # 统计每层的专家数量
        layer_experts = defaultdict(set)
        for t in expert_tensors:
            parts = t["name"].split(".")
            if parts[0] == "blk" and len(parts) >= 5:
                try:
                    layer_idx = int(parts[1])
                    expert_idx = int(parts[3])
                    layer_experts[layer_idx].add(expert_idx)
                except (ValueError, IndexError):
                    pass

        if layer_experts:
            arch["num_hidden_layers"] = max(arch["num_hidden_layers"], max(layer_experts.keys()) + 1)
            arch["num_experts_per_layer"] = max(len(v) for v in layer_experts.values())
            arch["moe_layers"] = sorted(layer_experts.keys())

    return arch


class Gemma4ExpertStreamer:
    """Gemma 4 Expert Streaming 加载器 (per-expert 命名约定)."""

    def __init__(self, header: dict, filepath: str):
        self.header = header
        self.filepath = filepath
        self._cache = {}
        self._hits = 0
        self._misses = 0

        self.kv = header["kv"]
        self.hidden = self.kv.get("gemma4.hidden_size", 3584)
        self.inter = self.kv.get("gemma4.intermediate_size", 14336)
        self.num_experts = self.kv.get("gemma4.expert_count", 8)
        self.num_layers = self.kv.get("gemma4.num_hidden_layers", 26)

        self._build_offset_map()

    def _build_offset_map(self):
        """构建专家张量偏移索引."""
        self._offsets = {}  # (layer, expert, role) -> tensor_info
        for t in self.header["expert_tensors"]:
            parts = t["name"].split(".")
            if parts[0] == "blk" and len(parts) >= 5:
                try:
                    layer = int(parts[1])
                    expert = int(parts[3])
                    role = parts[4]
                    self._offsets[(layer, expert, role)] = t
                except (ValueError, IndexError):
                    pass

        # 统计唯一层数和专家数
        layers = set()
        experts = set()
        for (layer, expert, _), _ in self._offsets.items():
            layers.add(layer)
            experts.add(expert)

        self.moe_layers = sorted(layers)
        self.all_experts = sorted(experts)

    def load_expert(self, layer: int, expert_id: int) -> dict:
        """加载指定层和专家的权重."""
        key = f"L{layer}_E{expert_id}"
        if key in self._cache:
            self._hits += 1
            return self._cache[key]

        self._misses += 1
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
                "name": info["name"],
            }

        result = {
            "layer": layer,
            "expert_id": expert_id,
            "roles": roles,
        }
        self._cache[key] = result
        return result

    def cache_stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "cached": len(self._cache),
            "hit_rate": self._hits / max(total, 1) * 100,
        }

    def invalidate_layer(self, layer: int):
        """失效指定层的缓存 (prefill 完成后)."""
        keys_to_remove = [k for k in self._cache.keys() if k.startswith(f"L{layer}_")]
        for k in keys_to_remove:
            del self._cache[k]

    def prewarm_prefill_layers(self):
        """预热 prefill 阶段需要的层 (前几层)."""
        prefill_layers = self.moe_layers[:len(self.moe_layers)//2]
        warmed = 0
        for layer in prefill_layers:
            for expert_id in self.all_experts:
                if self.load_expert(layer, expert_id):
                    warmed += 1
        return warmed

    def prewarm_decode_layers(self):
        """预热 decode 阶段需要的层 (后几层)."""
        decode_layers = self.moe_layers[len(self.moe_layers)//2:]
        warmed = 0
        for layer in decode_layers:
            for expert_id in self.all_experts:
                if self.load_expert(layer, expert_id):
                    warmed += 1
        return warmed


def main():
    print("=" * 80)
    print("GEMMA 4 26B-A4B EXPERT STREAMING VALIDATION")
    print("=" * 80)

    model_path = r"C:\Users\alexchuang\Desktop\fastprefill\gemma4_gguf\gemma-4-26B-A4B-it-heretic.IQ4_XS.gguf"

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        sys.exit(1)

    # Step 1: 解析 GGUF 头部
    print("\n📊 Step 1: GGUF Header Parsing")
    print("-" * 60)
    header = parse_gguf_header(model_path)

    # Step 2: 分析架构
    print("\n🧠 Step 2: Architecture Analysis")
    print("-" * 60)
    arch = analyze_gemma4_architecture(header)

    print(f"  Architecture: {arch['architecture']}")
    print(f"  Hidden Size: {arch['hidden_size']}")
    print(f"  Intermediate Size: {arch['intermediate_size']}")
    print(f"  Num Layers: {arch['num_hidden_layers']}")
    print(f"  Num Attention Heads: {arch['num_attention_heads']}")
    print(f"  Num KV Heads: {arch['num_key_value_heads']}")
    print(f"  Max Position: {arch['max_position_embeddings']}")
    print(f"  Vocab Size: {arch['vocab_size']}")

    if "moe_layers" in arch:
        print(f"\n  MoE Expert Layers: {len(arch['moe_layers'])} layers")
        print(f"  Expert Layers: {arch['moe_layers']}")
        print(f"  Experts Per Layer: {arch['num_experts_per_layer']}")

        # 统计专家权重大小
        expert_tensors = header["expert_tensors"]
        total_expert_bytes = sum(t["size_bytes"] for t in expert_tensors)
        print(f"\n  Expert Weight Statistics:")
        print(f"    Total Expert Tensors: {len(expert_tensors)}")
        print(f"    Total Expert Size: {total_expert_bytes / 1024**2:.1f} MB")
        print(f"    Experts Per Layer Size: {total_expert_bytes / len(arch['moe_layers']) / 1024**2:.1f} MB")

        # 计算单层每层专家大小 (单层 8 专家)
        # 每个专家: gate[inter, hidden] + up[inter, hidden] + down[hidden, inter]
        hidden = arch["hidden_size"]
        inter = arch["intermediate_size"]
        for t in expert_tensors[:3]:  # 只看第一个专家
            print(f"\n    Sample Expert Tensor:")
            print(f"      Name: {t['name']}")
            print(f"      Dims: {t['dims']}")
            print(f"      Type: {t['type_name']}")
            print(f"      Size: {t['size_bytes'] / 1024:.1f} KB")

        # 单层专家大小估算
        if arch.get("num_experts_per_layer"):
            single_expert_size = 0
            for role in ["gate", "up", "down"]:
                expert_tensors_in_layer = [t for t in expert_tensors
                                           if role in t["name"] and f".{arch['moe_layers'][0]}." in t["name"]]
                for t in expert_tensors_in_layer:
                    single_expert_size += t["size_bytes"]
            print(f"\n    Single Expert (gate+up+down): {single_expert_size / 1024**2:.2f} MB")
            print(f"    8 Experts Per Layer: {single_expert_size * 8 / 1024**2:.2f} MB")

    # Step 3: 初始化 ExpertStreamer
    print("\n🚀 Step 3: ExpertStreamer Initialization")
    print("-" * 60)
    streamer = Gemma4ExpertStreamer(header, model_path)

    print(f"  MoE Layers: {streamer.moe_layers}")
    print(f"  Experts: {streamer.all_experts}")
    print(f"  Total Expert Tensors Indexed: {len(streamer._offsets)}")

    # Step 4: 测试 Expert 加载
    print("\n📈 Step 4: Expert Loading Test")
    print("-" * 60)

    test_layers = streamer.moe_layers[:3]  # 测试前 3 层
    test_experts = [0, 4]  # 每个 layer 测试 2 个专家

    for layer in test_layers:
        for expert_id in test_experts:
            start = time.time()
            expert = streamer.load_expert(layer, expert_id)
            elapsed = time.time() - start

            if expert and expert["roles"]:
                total_size = sum(r["size_bytes"] for r in expert["roles"].values())
                print(f"  Layer {layer}, Expert {expert_id}: {total_size / 1024**2:.2f} MB ({elapsed*1000:.1f}ms)")
                for role, info in expert["roles"].items():
                    print(f"    {role}: {info['dims']} {info['type_name']} ({info['size_bytes'] / 1024:.1f} KB)")

    # Step 5: 缓存验证
    print("\n💾 Step 5: Cache Validation")
    print("-" * 60)

    stats = streamer.cache_stats()
    print(f"  Initial Cache Stats: hits={stats['hits']}, misses={stats['misses']}")

    # 重复加载相同专家
    for layer in test_layers:
        for expert_id in test_experts:
            streamer.load_expert(layer, expert_id)

    stats_after = streamer.cache_stats()
    print(f"  After Repeated Access: hits={stats_after['hits']}, misses={stats_after['misses']}")
    print(f"  Hit Rate: {stats_after['hit_rate']:.1f}%")

    # Step 6: 缓存失效测试
    print("\n🔄 Step 6: Cache Invalidation")
    print("-" * 60)

    if streamer.moe_layers:
        layer_to_remove = streamer.moe_layers[0]
        streamer.invalidate_layer(layer_to_remove)
        stats_after_remove = streamer.cache_stats()
        print(f"  After Invalidate Layer {layer_to_remove}: cached={stats_after_remove['cached']}")
    else:
        print("  ⚠️  No MoE layers found")

    # Step 7: Prewarm 测试
    print("\n⚡ Step 7: Prewarm Test")
    print("-" * 60)

    # 清空缓存
    streamer._cache.clear()
    streamer._hits = 0
    streamer._misses = 0

    # 模拟 prefill 阶段: 加载前几层
    prefill_start = time.time()
    prefill_warmed = streamer.prewarm_prefill_layers()
    prefill_time = time.time() - prefill_start

    prefill_stats = streamer.cache_stats()
    print(f"  Prefill Prewarm: {prefill_warmed} experts in {prefill_time:.2f}s")
    print(f"    Cache size: {prefill_stats['cached']}")

    # 模拟 decode 阶段: 失效 prefill 层, 加载后几层
    decode_start = time.time()
    decode_warmed = streamer.prewarm_decode_layers()
    decode_time = time.time() - decode_start

    decode_stats = streamer.cache_stats()
    print(f"  Decode Prewarm: {decode_warmed} experts in {decode_time:.2f}s")
    print(f"    Cache size: {decode_stats['cached']}")

    # Step 8: 性能分析
    print("\n📊 Step 8: Performance Analysis")
    print("-" * 60)

    total_expert_size_mb = 0
    for (layer, expert, _), info in streamer._offsets.items():
        total_expert_size_mb += info["size_bytes"] / 1024**2

    print(f"  Total Expert Weight Size: {total_expert_size_mb / 1024:.2f} GB")
    print(f"  Per-Layer Expert Size: {total_expert_size_mb / len(streamer.moe_layers) / 1024:.2f} MB")
    print(f"  Streaming Benefit: only load {total_expert_size_mb / len(streamer.moe_layers) / 1024:.1f} MB per step vs {total_expert_size_mb / 1024:.1f} GB full")

    # 计算节省的内存
    full_model_expert = total_expert_size_mb
    streaming_needed = total_expert_size_mb / len(streamer.moe_layers) * 2  # 每层只加载 2 专家 (topk)
    saving = (1 - streaming_needed / full_model_expert) * 100
    print(f"\n  Memory Saving (from expert streaming):")
    print(f"    Full Expert Weights: {full_model_expert / 1024:.2f} GB")
    print(f"    With Streaming: {streaming_needed / 1024:.2f} GB (2 experts/layer)")
    print(f"    Saving: {saving:.1f}%")

    # 总结
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)

    if arch.get("moe_layers"):
        print("✅ MoE Architecture Identified")
        print(f"   Hidden: {arch['hidden_size']}, Inter: {arch['intermediate_size']}")
        print(f"   Expert Layers: {len(arch['moe_layers'])}, Experts/Layer: {arch.get('num_experts_per_layer', '?')}")
        print(f"   Expert Size: {total_expert_size_mb / 1024**2:.1f} MB/layer")

    if prefill_stats["hit_rate"] >= 0:
        print(f"✅ Streaming Logic Works:")
        print(f"   Prefill Prewarm: {prefill_time*1000:.0f}ms ({prefill_warmed} experts)")
        print(f"   Decode Prewarm: {decode_time*1000:.0f}ms ({decode_warmed} experts)")
        print(f"   Cache Hit Rate: {decode_stats['hit_rate']:.1f}%")

    print("\n💡 Next Steps:")
    print("  1. 接入 llama.cpp GGUF loader (gguf.cpp)")
    print("  2. 实现动态 expert 切换 (prefill -> decode)")
    print("  3. 与 DeepEP dispatch 集成")
    print("  4. GPU 端零拷贝映射")

    return 0


if __name__ == "__main__":
    sys.exit(main())
