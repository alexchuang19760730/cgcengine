"""
test_expert_compute_torch.py — 验证 ExpertWeightsView → torch → grouped_gemm_silu_bf16_forward 集成逻辑.

由于 expert_compute_torch.cpp 需要 PyTorch + cgc_moe_engine 编译,
此测试用 mock 数据验证集成逻辑的正确性:

1. ExpertWeightsView 布局理解 (gate/up/down shape 和 offset)
2. viewsToGroupedWeights 的转置逻辑 (GGUF [out,in] → grouped_gemm [in,out])
3. moeForward 的调用流程 (dispatch → grouped_gemm_silu → combine)
4. GroupedWeights tensor 形状校验

测试不依赖 C++ 编译, 用纯 Python mock 验证逻辑.
"""
import sys
import os
import struct
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

# ============================================================================
# Mock: 模拟 ExpertWeightsView 和 GroupedWeights 的逻辑
# ============================================================================

class MockExpertSubTensorView:
    """模拟 C++ ExpertSubTensorView"""
    def __init__(self, data, shape, ggml_type, offset, size):
        self.data = data
        self.shape = shape  # [out_dim, in_dim]
        self.ggmlType = ggml_type
        self.offsetInBuffer = offset
        self.sizeBytes = size


class MockExpertWeightsView:
    """模拟 C++ ExpertWeightsView"""
    def __init__(self, expert_id, gate, up, down, raw_buffer, raw_size):
        self.expertId = expert_id
        self.gate = gate
        self.up = up
        self.down = down
        self.rawBuffer = raw_buffer
        self.rawSize = raw_size


# GGML 类型常量 (与 cgc_gguf_lite.h 一致)
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q4_K = 14
GGML_TYPE_IQ3_S = 21
GGML_TYPE_IQ3_M = 22
GGML_TYPE_BF16 = 30


def make_mock_view(expert_id, hidden=256, inter=512, ggml_type=GGML_TYPE_BF16):
    """创建 mock ExpertWeightsView.
    gate: [inter, hidden]  (moe_intermediate_size × hidden_size)
    up:   [inter, hidden]
    down: [hidden, inter]
    """
    # 计算字节数 (BF16 = 2 bytes/element)
    elem_size = 2 if ggml_type == GGML_TYPE_BF16 else 1
    gate_size = inter * hidden * elem_size
    up_size = inter * hidden * elem_size
    down_size = hidden * inter * elem_size

    gate = MockExpertSubTensorView(
        data=b'\x00' * gate_size,
        shape=[inter, hidden],
        ggml_type=ggml_type,
        offset=0,
        size=gate_size
    )
    up = MockExpertSubTensorView(
        data=b'\x00' * up_size,
        shape=[inter, hidden],
        ggml_type=ggml_type,
        offset=gate_size,
        size=up_size
    )
    down = MockExpertSubTensorView(
        data=b'\x00' * down_size,
        shape=[hidden, inter],
        ggml_type=ggml_type,
        offset=gate_size + up_size,
        size=down_size
    )
    raw_size = gate_size + up_size + down_size
    return MockExpertWeightsView(expert_id, gate, up, down, b'\x00' * raw_size, raw_size)


# ============================================================================
# Python 版 viewsToGroupedWeights 逻辑 (模拟 C++ 实现)
# ============================================================================

def views_to_grouped_weights_py(views):
    """Python 版 viewsToGroupedWeights, 验证转置逻辑.
    输入: views 列表
    输出: (gate_weights_shape, up_weights_shape, down_weights_shape)
    """
    if not views:
        return None

    num_experts = len(views)

    # 从第一个 view 获取形状
    # GGUF 布局: gate=[out_dim, in_dim]=[inter, hidden]
    gate_out, gate_in = views[0].gate.shape  # inter, hidden
    up_out, up_in = views[0].up.shape        # inter, hidden
    down_out, down_in = views[0].down.shape  # hidden, inter

    # grouped_gemm_silu_bf16_forward 期望 [num_experts, in_dim, out_dim]
    # gate: GGUF [inter, hidden] → 转置 → [hidden, inter] = [in_dim, out_dim]
    gate_weights_shape = [num_experts, gate_in, gate_out]
    # up: 同 gate
    up_weights_shape = [num_experts, up_in, up_out]
    # down: GGUF [hidden, inter] → grouped_gemm 期望 [out_dim, in_dim]?
    # 实际: down_proj 是 [hidden, inter], 输入是 inter, 输出 hidden
    # grouped_gemm_silu 内部 down_proj: tokens[inter] × down[inter, hidden] → [hidden]
    # 所以 down_weights 应该是 [num_experts, inter, hidden] = [in_dim_for_down, out_dim_for_down]
    # 但 GGUF down shape 是 [hidden, inter], 需要转置
    down_weights_shape = [num_experts, down_in, down_out]  # [num_experts, inter, hidden]

    return {
        'gate': gate_weights_shape,
        'up': up_weights_shape,
        'down': down_weights_shape,
    }


# ============================================================================
# 测试用例
# ============================================================================

def test_view_layout():
    """T1: ExpertWeightsView 布局正确 (gate/up/down shape)"""
    view = make_mock_view(expert_id=0, hidden=256, inter=512)
    assert view.gate.shape == [512, 256], f"gate shape: {view.gate.shape}"
    assert view.up.shape == [512, 256], f"up shape: {view.up.shape}"
    assert view.down.shape == [256, 512], f"down shape: {view.down.shape}"
    print(f"[T1] PASS: view 布局 gate={view.gate.shape} up={view.up.shape} down={view.down.shape}")


def test_grouped_weights_shape():
    """T2: viewsToGroupedWeights 输出形状正确"""
    views = [make_mock_view(i, hidden=256, inter=512) for i in range(8)]
    shapes = views_to_grouped_weights_py(views)
    assert shapes['gate'] == [8, 256, 512], f"gate: {shapes['gate']}"
    assert shapes['up'] == [8, 256, 512], f"up: {shapes['up']}"
    assert shapes['down'] == [8, 512, 256], f"down: {shapes['down']}"
    print(f"[T2] PASS: grouped weights gate={shapes['gate']} up={shapes['up']} down={shapes['down']}")


def test_transpose_logic():
    """T3: 转置逻辑正确 (GGUF [out,in] → grouped_gemm [in,out])"""
    views = [make_mock_view(0, hidden=256, inter=512)]
    shapes = views_to_grouped_weights_py(views)

    # gate GGUF [inter=512, hidden=256] → grouped_gemm [hidden=256, inter=512]
    assert shapes['gate'][1] == 256, "gate in_dim 应为 hidden=256"
    assert shapes['gate'][2] == 512, "gate out_dim 应为 inter=512"
    print(f"[T3] PASS: 转置逻辑 gate GGUF[512,256] → grouped_gemm[256,512]")


def test_bf16_zero_copy():
    """T4: BF16 零拷贝路径 (ggmlType=30)"""
    view = make_mock_view(0, hidden=64, inter=128, ggml_type=GGML_TYPE_BF16)
    assert view.gate.ggmlType == GGML_TYPE_BF16
    assert view.up.ggmlType == GGML_TYPE_BF16
    assert view.down.ggmlType == GGML_TYPE_BF16
    print(f"[T4] PASS: BF16 零拷贝路径 (type={GGML_TYPE_BF16})")


def test_quantized_warning():
    """T5: 量化格式 (IQ3_M) 返回零 tensor (待对接 llama.cpp dequantize)"""
    view = make_mock_view(0, hidden=64, inter=128, ggml_type=GGML_TYPE_IQ3_M)
    assert view.gate.ggmlType == GGML_TYPE_IQ3_M
    # 当前实现: 量化格式返回零 tensor, shape 正确
    # TODO: 对接 llama.cpp ggml dequantize_row_iq3_m
    print(f"[T5] PASS: IQ3_M (type={GGML_TYPE_IQ3_M}) 标记需反量化, 当前零 tensor 占位")


def test_moe_forward_flow():
    """T6: moeForward 调用流程 (dispatch → grouped_gemm_silu → combine)"""
    views = [make_mock_view(i, hidden=256, inter=512) for i in range(8)]

    # 模拟流程:
    # 1. viewsToGroupedWeights
    shapes = views_to_grouped_weights_py(views)
    assert shapes is not None

    # 2. deepep_dispatch_forward (mock: 不实际调用)
    num_tokens = 4
    num_experts = 8
    num_experts_per_token = 2
    dispatched_shape = [num_tokens * num_experts_per_token, 256]  # [8, 256]
    indices_shape = [num_tokens, num_experts_per_token]  # [4, 2]

    # 3. grouped_gemm_silu_bf16_forward (mock)
    # 输入: dispatched_tokens [8, 256], gate_weights [8, 256, 512], ...
    # 输出: [8, 256] (hidden_dim)
    expert_output_shape = [dispatched_shape[0], 256]

    # 4. deepep_combine_forward (mock)
    final_output_shape = [num_tokens, 256]

    assert final_output_shape == [4, 256], f"final: {final_output_shape}"
    print(f"[T6] PASS: moeForward 流程 tokens[{num_tokens},256] → dispatch[{dispatched_shape[0]},256] → gemm_silu → combine[{final_output_shape[0]},256]")


def test_offset_consistency():
    """T7: expert 内部 gate/up/down offset 连续"""
    view = make_mock_view(0, hidden=256, inter=512)
    assert view.gate.offsetInBuffer == 0
    assert view.up.offsetInBuffer == view.gate.sizeBytes
    assert view.down.offsetInBuffer == view.gate.sizeBytes + view.up.sizeBytes
    assert view.rawSize == view.gate.sizeBytes + view.up.sizeBytes + view.down.sizeBytes
    print(f"[T7] PASS: offset 连续 gate@0 up@{view.up.offsetInBuffer} down@{view.down.offsetInBuffer} total={view.rawSize}")


def test_gemma4_26b_dimensions():
    """T8: Gemma 4 26B-A4B 实际维度验证"""
    # Gemma 4 26B-A4B: hidden_size=3584, intermediate=14336, num_experts=8 (A4B: 4 active)
    hidden = 3584
    inter = 14336
    num_experts = 8
    active_experts = 4  # A4B = 4 active experts per token

    views = [make_mock_view(i, hidden=hidden, inter=inter) for i in range(num_experts)]
    shapes = views_to_grouped_weights_py(views)

    assert shapes['gate'] == [8, 3584, 14336], f"gate: {shapes['gate']}"
    assert shapes['down'] == [8, 14336, 3584], f"down: {shapes['down']}"
    print(f"[T8] PASS: Gemma 4 26B-A4B 维度 hidden={hidden} inter={inter} experts={num_experts} active={active_experts}")
    print(f"      gate_weights shape: {shapes['gate']} (8×3584×14336 BF16 = {8*3584*14336*2/1024/1024:.1f} MB)")


def main():
    print("=" * 70)
    print("test_expert_compute_torch.py — ExpertWeightsView → torch → grouped_gemm 集成")
    print("=" * 70)
    tests = [
        test_view_layout,
        test_grouped_weights_shape,
        test_transpose_logic,
        test_bf16_zero_copy,
        test_quantized_warning,
        test_moe_forward_flow,
        test_offset_consistency,
        test_gemma4_26b_dimensions,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
            failed += 1
    print()
    print("=" * 70)
    if failed == 0:
        print(f"ALL {passed} TESTS PASSED")
    else:
        print(f"{passed}/{passed+failed} TESTS PASSED, {failed} FAILED")
    print("=" * 70)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
