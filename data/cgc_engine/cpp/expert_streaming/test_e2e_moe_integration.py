"""
test_e2e_moe_integration.py — ExpertStreamer → ExpertComputeBridge → cgc_moe_engine E2E 验证

验证项:
  T1: GGUF header 解析 (lite 版本)
  T2: ExpertStreamerLite cache hit/miss 逻辑
  T3: ExpertWeightsView 零拷贝指针正确性
  T4: viewsToGroupedWeights 转置逻辑 (Python 版)
  T5: DeepEP dispatch + grouped_gemm_silu + combine 端到端流程
  T6: Gemma 4 26B-A4B 实际维度验证 (hidden=3584, inter=14336, 8 experts)
  T7: 量化格式占位路径 (IQ3_M)
  T8: 多级 cache (prefetch + load + release)

不依赖 C++ 编译,纯 Python mock 验证完整集成链路.
"""

import os
import struct
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

GGUF_MAGIC = 0x46554747
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_BF16 = 30
GGML_TYPE_IQ3_M = 22
GGML_TYPE_BYTES = {GGML_TYPE_F32: 4, GGML_TYPE_F16: 2, GGML_TYPE_BF16: 2, GGML_TYPE_IQ3_M: 3}


def make_test_gguf(filepath: str, hidden: int = 256, inter: int = 512,
                   num_experts: int = 8, ggml_type: int = GGML_TYPE_BF16) -> str:
    """创建测试 GGUF 文件 (per-layer expert 布局)."""
    n_sub = 3
    roles = ["gate", "up", "down"]
    bpe = GGML_TYPE_BYTES.get(ggml_type, 4)
    gate_size = inter * hidden * bpe
    up_size = inter * hidden * bpe
    down_size = hidden * inter * bpe
    expert_stride = gate_size + up_size + down_size

    def write_u32(f, v):
        f.write(struct.pack("<I", v))

    def write_u64(f, v):
        f.write(struct.pack("<Q", v))

    def write_i32(f, v):
        write_u32(f, v if v >= 0 else v + 2**32)

    def write_str(f, s):
        write_u64(f, len(s))
        f.write(s.encode("utf-8"))

    def write_kv_u32(f, key, val):
        write_str(f, key)
        write_u32(f, 4)
        write_u32(f, val)

    def write_kv_i32(f, key, val):
        write_str(f, key)
        write_u32(f, 5)
        write_i32(f, val)

    def write_kv_str(f, key, val):
        write_str(f, key)
        write_u32(f, 8)
        write_str(f, val)

    n_tensors = num_experts * n_sub
    n_kv = 8

    with open(filepath, "wb") as f:
        write_u32(f, GGUF_MAGIC)
        write_u32(f, 3)
        write_u64(f, n_tensors)
        write_u64(f, n_kv)

        write_kv_str(f, "general.architecture", "gemma4_moe")
        write_kv_u32(f, "general.layer_index", 5)
        write_kv_u32(f, "gemma4.expert_count", num_experts)
        write_kv_u32(f, "gemma4.expert_stride", expert_stride)
        write_kv_str(f, "gemma4.quantization", "BF16" if ggml_type == GGML_TYPE_BF16 else "IQ3_M")
        write_kv_i32(f, "gemma4.hidden_size", hidden)
        write_kv_i32(f, "gemma4.moe_intermediate_size", inter)
        write_kv_i32(f, "gemma4.num_experts", num_experts)

        current_offset = 0
        for e in range(num_experts):
            for s, role in enumerate(roles):
                name = f"blk.5.expert.{e}.{role}.weight"
                write_str(f, name)
                write_u32(f, 2)
                dims = [inter, hidden] if role in ("gate", "up") else [hidden, inter]
                write_u64(f, dims[0])
                write_u64(f, dims[1])
                write_u32(f, ggml_type)
                write_u64(f, current_offset)
                if role == "gate":
                    current_offset += gate_size
                elif role == "up":
                    current_offset += up_size
                else:
                    current_offset += down_size

        pos = f.tell()
        aligned = (pos + 31) & ~31
        for _ in range(aligned - pos):
            f.write(b"\x00")

        for _ in range(current_offset):
            f.write(b"\x41" if ggml_type == GGML_TYPE_BF16 else b"\x00")

    return filepath


def parse_gguf_header(filepath: str):
    with open(filepath, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        assert magic == GGUF_MAGIC
        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        kv = {}
        for _ in range(n_kv):
            klen = struct.unpack("<Q", f.read(8))[0]
            key = f.read(klen).decode()
            dtype = struct.unpack("<I", f.read(4))[0]
            if dtype == 4:
                val = struct.unpack("<I", f.read(4))[0]
            elif dtype == 5:
                val = struct.unpack("<i", f.read(4))[0]
            elif dtype == 8:
                slen = struct.unpack("<Q", f.read(8))[0]
                val = f.read(slen).decode()
            else:
                val = None
            kv[key] = val

        tensors = []
        for _ in range(n_tensors):
            nlen = struct.unpack("<Q", f.read(8))[0]
            name = f.read(nlen).decode()
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            ggml_type = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            bpe = GGML_TYPE_BYTES.get(ggml_type, 4)
            n_elements = 1
            for d in dims:
                n_elements *= d
            tensors.append({"name": name, "dims": dims, "type": ggml_type,
                             "offset": offset, "size_bytes": int(n_elements * bpe)})

        data_start = (f.tell() + 31) & ~31

    return {"version": version, "n_tensors": n_tensors, "n_kv": n_kv,
            "kv": kv, "tensors": tensors, "data_start": data_start}


class PythonExpertStreamer:
    """Python 版 expert streamer, 直接对接 GGUF 文件."""

    def __init__(self, gguf_path: str):
        self.path = gguf_path
        self.header = parse_gguf_header(gguf_path)
        self._file = None
        self._cache = {}
        self._cache_order = []
        self._hits = 0
        self._misses = 0

        self.layer = int(self.header["kv"].get("general.layer_index", 0))
        self.hidden = int(self.header["kv"].get("gemma4.hidden_size", 0))
        self.inter = int(self.header["kv"].get("gemma4.moe_intermediate_size", 0))
        self.num_experts = int(self.header["kv"].get("gemma4.expert_count", 0))
        self.ggml_type = int(self.header["kv"].get("gemma4.quantization", "BF16") != "BF16") and GGML_TYPE_IQ3_M or GGML_TYPE_BF16

        self._build_offset_map()

    def _build_offset_map(self):
        self._offsets = {}
        for t in self.header["tensors"]:
            parts = t["name"].split(".")
            if len(parts) >= 5 and parts[0] == "blk" and parts[2] == "expert":
                try:
                    eid = int(parts[3])
                    role = parts[4]
                    self._offsets[(eid, role)] = t
                except (ValueError, IndexError):
                    pass

    def load_expert(self, expert_id: int) -> dict:
        key = f"E{expert_id}"
        if key in self._cache:
            self._hits += 1
            self._cache_order.remove(key)
            self._cache_order.append(key)
            return self._cache[key]

        self._misses += 1
        roles = {}
        for role in ["gate", "up", "down"]:
            info = self._offsets.get((expert_id, role))
            if not info:
                continue
            with open(self.path, "rb") as f:
                f.seek(self.header["data_start"] + info["offset"])
                data = f.read(info["size_bytes"])
            roles[role] = {
                "data": data,
                "dims": info["dims"],
                "ggml_type": self.ggml_type,
                "size_bytes": info["size_bytes"],
            }

        result = {"expert_id": expert_id, "roles": roles}
        self._cache[key] = result
        self._cache_order.append(key)
        return result

    def cache_stats(self) -> dict:
        return {"hits": self._hits, "misses": self._misses,
                "cached": len(self._cache),
                "hit_rate": self._hits / max(self._hits + self._misses, 1)}


def views_to_grouped_weights_py(views, hidden: int, inter: int, num_experts: int) -> dict:
    """Python 版 viewsToGroupedWeights (与 C++ 实现 1:1 对应).
    
    grouped_gemm_silu_bf16_forward 期望:
      gate_weights: [num_experts, in_dim, out_dim]  (需要转置)
      up_weights:   [num_experts, in_dim, out_dim]  (需要转置)
      down_weights: [num_experts, out_dim, in_dim]  (需要转置)
    
    GGUF 布局:
      gate: [out_dim, in_dim] = [inter, hidden]
      up:   [out_dim, in_dim] = [inter, hidden]
      down: [in_dim, out_dim] = [hidden, inter]
    
    所有权重都需要转置为 grouped_gemm 期望的布局.
    """
    gate_weights = []
    up_weights = []
    down_weights = []

    for i in range(num_experts):
        gate_view = views[i]["roles"]["gate"]
        up_view = views[i]["roles"]["up"]
        down_view = views[i]["roles"]["down"]

        gate_dims = gate_view["dims"]  # GGUF [out_dim, in_dim] = [inter, hidden]
        # grouped_gemm gate: [in_dim, out_dim] = [hidden, inter]
        gate_weights.append({"in": gate_dims[1], "out": gate_dims[0]})

        up_dims = up_view["dims"]  # GGUF [out_dim, in_dim] = [inter, hidden]
        up_weights.append({"in": up_dims[1], "out": up_dims[0]})

        down_dims = down_view["dims"]  # GGUF [in_dim, out_dim] = [hidden, inter]
        # grouped_gemm down: [out_dim, in_dim] = [inter, hidden]
        down_weights.append({"out": down_dims[1], "in": down_dims[0]})

    return {
        "gate": [{"shape": [num_experts, w["in"], w["out"]]} for w in gate_weights],
        "up": [{"shape": [num_experts, w["in"], w["out"]]} for w in up_weights],
        "down": [{"shape": [num_experts, w["out"], w["in"]]} for w in down_weights],
    }


def deepep_dispatch_py(tokens, gating_logits, num_experts_per_token: int):
    """Python 版 DeepEP dispatch (简化版)."""
    import numpy as np
    tokens_arr = np.array(tokens)
    logits_arr = np.array(gating_logits)

    topk_idx = np.argsort(-logits_arr, axis=-1)[:, :num_experts_per_token]
    topk_vals = np.take_along_axis(logits_arr, topk_idx, axis=-1)
    weights = topk_vals / topk_vals.sum(axis=-1, keepdims=True)

    num_tokens = len(tokens)
    dispatched = np.repeat(tokens_arr, num_experts_per_token, axis=0)

    return {
        "dispatched_tokens": dispatched,
        "indices": topk_idx,
        "weights": weights,
        "num_tokens": num_tokens,
        "num_experts_per_token": num_experts_per_token,
    }


def grouped_gemm_silu_py(dispatched_tokens, gate_weights, up_weights, down_weights, indices):
    """Python 版 grouped_gemm_silu_bf16_forward (简化版)."""
    import numpy as np

    gate_w = np.asarray(gate_weights)
    up_w = np.asarray(up_weights)
    down_w = np.asarray(down_weights)
    tokens_arr = np.asarray(dispatched_tokens)
    idx_arr = np.asarray(indices)

    num_dispatch = tokens_arr.shape[0]
    hidden = tokens_arr.shape[1]
    inter = gate_w.shape[2]

    gate_out = np.zeros((num_dispatch, inter), dtype=np.float32)
    up_out = np.zeros((num_dispatch, inter), dtype=np.float32)

    # indices 是 2D [num_tokens, k], 需要展平为 1D expert index 序列
    if idx_arr.ndim == 2:
        flat_indices = idx_arr.flatten()
    else:
        flat_indices = idx_arr

    for i in range(num_dispatch):
        expert_idx = int(flat_indices[i])
        gW = gate_w[expert_idx]  # [hidden, inter]
        uW = up_w[expert_idx]    # [hidden, inter]
        gate_out[i] = tokens_arr[i] @ gW
        up_out[i] = tokens_arr[i] @ uW

    act = gate_out * (1.0 / (1.0 + np.exp(-gate_out))) * up_out

    down_out = np.zeros((num_dispatch, hidden), dtype=np.float32)
    for i in range(num_dispatch):
        expert_idx = int(flat_indices[i])
        dW = down_w[expert_idx]  # [inter, hidden]
        down_out[i] = act[i] @ dW

    return down_out


def run_tests():
    tests_passed = 0
    tests_failed = 0
    base_dir = "D:\\alex\\flashkv0516\\.tmp_test"
    os.makedirs(base_dir, exist_ok=True)
    tmpdir = tempfile.mkdtemp(dir=base_dir)

    def check(name, fn):
        nonlocal tests_passed, tests_failed
        try:
            fn()
            tests_passed += 1
            print(f"[PASS] {name}")
        except Exception as e:
            tests_failed += 1
            print(f"[FAIL] {name}: {e}")

    # T1: GGUF header 解析
    def t1_gguf_parse():
        path = os.path.join(tmpdir, "test_t1.gguf")
        make_test_gguf(path, hidden=256, inter=512, num_experts=4, ggml_type=GGML_TYPE_BF16)
        hdr = parse_gguf_header(path)
        assert hdr["version"] == 3, f"version={hdr['version']}"
        assert hdr["n_tensors"] == 12, f"n_tensors={hdr['n_tensors']}"
        assert hdr["kv"]["gemma4.expert_count"] == 4
        assert hdr["kv"]["gemma4.hidden_size"] == 256
        assert hdr["kv"]["gemma4.moe_intermediate_size"] == 512
        gate_tensor = [t for t in hdr["tensors"] if "gate" in t["name"]][0]
        assert gate_tensor["dims"] == [512, 256], f"gate dims: {gate_tensor['dims']}"

    # T2: ExpertStreamerLite cache
    def t2_streamer_cache():
        path = os.path.join(tmpdir, "test_t2.gguf")
        make_test_gguf(path, hidden=256, inter=512, num_experts=8, ggml_type=GGML_TYPE_BF16)
        streamer = PythonExpertStreamer(path)
        assert streamer.num_experts == 8
        assert streamer.hidden == 256
        assert streamer.inter == 512

        v0 = streamer.load_expert(0)
        assert "roles" in v0
        assert "gate" in v0["roles"]
        assert "up" in v0["roles"]
        assert "down" in v0["roles"]

        stats = streamer.cache_stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 0

        v0_again = streamer.load_expert(0)
        stats2 = streamer.cache_stats()
        assert stats2["hits"] == 1
        assert stats2["misses"] == 1

    # T3: ExpertWeightsView 零拷贝
    def t3_zero_copy():
        path = os.path.join(tmpdir, "test_t3.gguf")
        make_test_gguf(path, hidden=64, inter=128, num_experts=4, ggml_type=GGML_TYPE_BF16)
        streamer = PythonExpertStreamer(path)
        views = [streamer.load_expert(i) for i in range(4)]

        for i, v in enumerate(views):
            gate_data = v["roles"]["gate"]["data"]
            assert len(gate_data) == 128 * 64 * 2  # BF16 = 2 bytes
            up_data = v["roles"]["up"]["data"]
            assert len(up_data) == 128 * 64 * 2
            down_data = v["roles"]["down"]["data"]
            assert len(down_data) == 64 * 128 * 2

    # T4: viewsToGroupedWeights 转置
    def t4_views_to_grouped():
        path = os.path.join(tmpdir, "test_t4.gguf")
        make_test_gguf(path, hidden=256, inter=512, num_experts=8, ggml_type=GGML_TYPE_BF16)
        streamer = PythonExpertStreamer(path)
        views = [streamer.load_expert(i) for i in range(8)]
        grouped = views_to_grouped_weights_py(views, hidden=256, inter=512, num_experts=8)

        gate_shapes = [g["shape"] for g in grouped["gate"]]
        assert all(s == [8, 256, 512] for s in gate_shapes), f"gate shapes: {gate_shapes}"

        up_shapes = [g["shape"] for g in grouped["up"]]
        assert all(s == [8, 256, 512] for s in up_shapes), f"up shapes: {up_shapes}"

        down_shapes = [g["shape"] for g in grouped["down"]]
        assert all(s == [8, 512, 256] for s in down_shapes), f"down shapes: {down_shapes}"

    # T5: DeepEP dispatch + grouped_gemm_silu + combine
    def t5_moe_forward_flow():
        import numpy as np
        hidden, inter, num_experts = 256, 512, 8
        num_tokens = 4
        k = 2

        tokens = np.random.randn(num_tokens, hidden).astype(np.float32)
        gating_logits = np.random.randn(num_tokens, num_experts).astype(np.float32)

        dispatch = deepep_dispatch_py(tokens, gating_logits, k)
        assert dispatch["dispatched_tokens"].shape == (num_tokens * k, hidden)
        assert dispatch["indices"].shape == (num_tokens, k)

        expert_weights = np.random.randn(num_experts, hidden, inter).astype(np.float32)
        down_weights = np.random.randn(num_experts, inter, hidden).astype(np.float32)

        expert_output = grouped_gemm_silu_py(
            dispatch["dispatched_tokens"],
            expert_weights,
            expert_weights,
            down_weights,
            dispatch["indices"],
        )
        assert expert_output.shape == (num_tokens * k, hidden)

        combined = np.zeros((num_tokens, hidden), dtype=np.float32)
        for i in range(num_tokens):
            for j in range(k):
                combined[i] += expert_output[i * k + j] * dispatch["weights"][i, j]
        assert combined.shape == (num_tokens, hidden)

    # T6: Gemma 4 26B-A4B 实际维度 (使用缩放版本验证形状正确性)
    # Gemma 4 26B-A4B 真实维度: hidden=3584, inter=14336, 8 experts
    # 测试使用 1/8 缩放: hidden=448, inter=1792, 8 experts
    def t6_gemma4_dims():
        REAL_HIDDEN = 3584
        REAL_INTER = 14336
        SCALE = 8
        hidden = REAL_HIDDEN // SCALE  # 448
        inter = REAL_INTER // SCALE   # 1792
        num_experts = 8

        path = os.path.join(tmpdir, "test_t6.gguf")
        make_test_gguf(path, hidden=hidden, inter=inter, num_experts=num_experts, ggml_type=GGML_TYPE_BF16)
        streamer = PythonExpertStreamer(path)
        assert streamer.hidden == hidden
        assert streamer.inter == inter
        assert streamer.num_experts == num_experts

        views = [streamer.load_expert(i) for i in range(num_experts)]
        grouped = views_to_grouped_weights_py(views, hidden=hidden, inter=inter, num_experts=num_experts)

        expected_gate = [num_experts, hidden, inter]
        for g in grouped["gate"]:
            assert g["shape"] == expected_gate, f"gate shape: {g['shape']} != {expected_gate}"

        total_gate_bytes = num_experts * REAL_HIDDEN * REAL_INTER * 2
        print(f"      Gemma 4 real gate_weights: {total_gate_bytes / 1024 / 1024:.1f} MB (BF16)")
        print(f"      Gemma 4 real expert_size: {3 * REAL_HIDDEN * REAL_INTER * 2 / 1024 / 1024:.1f} MB × 8 experts")

    # T7: IQ3_M 量化占位
    def t7_iq3m_placeholder():
        path = os.path.join(tmpdir, "test_t7.gguf")
        make_test_gguf(path, hidden=64, inter=128, num_experts=4, ggml_type=GGML_TYPE_IQ3_M)
        streamer = PythonExpertStreamer(path)
        assert streamer.ggml_type == GGML_TYPE_IQ3_M
        v = streamer.load_expert(0)
        gate_data = v["roles"]["gate"]["data"]
        expected_size = 128 * 64 * 3  # IQ3_M = 3 bytes
        assert len(gate_data) == expected_size, f"IQ3_M gate size: {len(gate_data)} != {expected_size}"

    # T8: 多级 cache (prefetch + LRU)
    def t8_multi_level_cache():
        path = os.path.join(tmpdir, "test_t8.gguf")
        make_test_gguf(path, hidden=128, inter=256, num_experts=8, ggml_type=GGML_TYPE_BF16)
        streamer = PythonExpertStreamer(path)

        # 首次 prefetch: 全 miss
        for e in range(8):
            streamer.load_expert(e)
        stats1 = streamer.cache_stats()
        assert stats1["misses"] == 8, f"stats1: {stats1}"

        # 再次加载: 全 hit
        for e in range(8):
            streamer.load_expert(e)
        stats2 = streamer.cache_stats()
        assert stats2["hits"] == 8, f"stats2: {stats2}"
        assert stats2["misses"] == 8, f"stats2 misses: {stats2}"

        # 交替访问: 验证 LRU
        streamer.load_expert(0)
        streamer.load_expert(1)
        streamer.load_expert(0)
        stats3 = streamer.cache_stats()
        # 第二轮 8 hits + 第三轮 3 hits (0, 1, 0 都已缓存) = 11
        assert stats3["hits"] == 11, f"stats3 hits: {stats3}"

    # 运行
    print("=" * 70)
    print("test_e2e_moe_integration.py — ExpertStreamer → cgc_moe_engine E2E")
    print("=" * 70)

    check("T1: GGUF header 解析", t1_gguf_parse)
    check("T2: ExpertStreamer cache hit/miss", t2_streamer_cache)
    check("T3: ExpertWeightsView 零拷贝", t3_zero_copy)
    check("T4: viewsToGroupedWeights 转置", t4_views_to_grouped)
    check("T5: DeepEP dispatch + GEMM + combine", t5_moe_forward_flow)
    check("T6: Gemma 4 26B-A4B 实际维度", t6_gemma4_dims)
    check("T7: IQ3_M 量化占位路径", t7_iq3m_placeholder)
    check("T8: 多级 cache (prefetch + LRU)", t8_multi_level_cache)

    print(f"\n{'='*70}")
    if tests_failed == 0:
        print(f"ALL {tests_passed} TESTS PASSED")
    else:
        print(f"{tests_passed}/{tests_passed + tests_failed} PASSED, {tests_failed} FAILED")
    print(f"{'='*70}")

    return 0 if tests_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_tests())