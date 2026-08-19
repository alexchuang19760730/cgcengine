#!/usr/bin/env python3
"""
验证 Gemma4 量化块对齐 (gate_up 拆分)

Gemma4 使用的量化类型:
- Q4_K (TYPE_7): 块大小 256 元素, 每个块 18 bytes (含 scale)
- IQ4_XS (TYPE_23): 块大小 256 元素

需要验证:
1. gate_up 切片后，gate 部分的元素数是否为块大小的整数倍
2. up 部分的元素数是否为块大小的整数倍
"""

import math

# Gemma4 架构参数
hidden = 2816
expert_inter = 704  # expert_feed_forward_length
num_experts = 128
gate_up_dim = expert_inter * 2  # 1408

# 量化类型参数
# Q4_K: block_size = 256, block_bytes = 18
# IQ4_XS: block_size = 256 (需要确认)

def check_alignment(name, elements, block_size=256, block_bytes=18):
    """检查元素数是否对齐到量化块."""
    blocks = elements / block_size
    is_aligned = elements % block_size == 0

    # 如果不对齐, 计算实际存储大小
    if not is_aligned:
        actual_blocks = math.ceil(elements / block_size)
        actual_bytes = actual_blocks * block_bytes
        wasted_bytes = actual_bytes - (elements * block_bytes // block_size)
    else:
        actual_blocks = elements // block_size
        actual_bytes = actual_blocks * block_bytes
        wasted_bytes = 0

    print(f"  {name}:")
    print(f"    Elements: {elements}")
    print(f"    Block size: {block_size}")
    print(f"    Blocks: {blocks:.2f} ({actual_blocks} actual)")
    print(f"    Aligned: {is_aligned}")
    print(f"    Size: {actual_bytes} bytes ({actual_bytes/1024:.2f} KB)")
    if not is_aligned:
        print(f"    ⚠️  NOT ALIGNED! Wasted: {wasted_bytes} bytes")
    print()

    return is_aligned, actual_bytes

print("=" * 80)
print("GEMMA4 QUANTIZATION BLOCK ALIGNMENT CHECK")
print("=" * 80)

# 检查 ffn_gate_up_exps 切片
print("\n📊 ffn_gate_up_exps 切片分析")
print("-" * 60)
print(f"  原始 shape: [{hidden}, {gate_up_dim}, {num_experts}]")
print(f"  每个专家: [{hidden}, {gate_up_dim}] = {hidden * gate_up_dim} elements")
print()

# 每个专家的元素数
expert_elements = hidden * gate_up_dim  # 2816 * 1408 = 3,964,928

# Gate 部分: [hidden, expert_inter] = [2816, 704]
gate_elements = hidden * expert_inter  # 2816 * 704 = 1,982,464

# Up 部分: [hidden, expert_inter] = [2816, 704]
up_elements = hidden * expert_inter  # 同上

print(f"  Expert total elements: {expert_elements}")
print(f"  Gate elements: {gate_elements}")
print(f"  Up elements: {up_elements}")
print()

# Q4_K 对齐检查
print("🔍 Q4_K (block_size=256, block_bytes=18):")
print()

# 整个专家 (gate+up)
total_aligned, total_bytes = check_alignment("Full Expert (gate+up)", expert_elements, 256, 18)

# Gate 部分
gate_aligned, gate_bytes = check_alignment("Gate Part", gate_elements, 256, 18)

# Up 部分
up_aligned, up_bytes = check_alignment("Up Part", up_elements, 256, 18)

# IQ4_XS 对齐检查
print("🔍 IQ4_XS (block_size=256, block_bytes=?):")
print()

# IQ4_XS 实际每个元素 4 bits = 0.5 bytes, 加 scale
# 块大小 256 元素 = 128 bytes + 2 bytes scale = 130 bytes
# 但实际 IQ4_XS 的存储格式可能不同
iq4xs_block_size = 256
iq4xs_bytes_per_block = 256 * 1  # 假设 1 byte/element (实际约 0.5 + scale)

print(f"  Note: IQ4_XS format needs verification")
print(f"  Typical: 4 bits/element + per-block scale")
print()

# 更精确的分析
print("📈 Detailed Analysis:")
print("-" * 60)

# 检查 2816 * 704 的因数
print(f"  hidden=2816 factorization: {2816} = {2**6} * {44} = {64} * {44}")
print(f"  expert_inter=704 factorization: {704} = {2**6} * {11} = {64} * {11}")
print(f"  gate_elements = 2816 * 704 = {gate_elements}")
print(f"  gate_elements / 256 = {gate_elements / 256:.2f}")
print(f"  gate_elements % 256 = {gate_elements % 256}")

if gate_elements % 256 == 0:
    print(f"  ✅ Gate elements divisible by 256!")
else:
    print(f"  ❌ Gate elements NOT divisible by 256!")
    # 分析原因
    gcd = math.gcd(gate_elements, 256)
    print(f"  GCD({gate_elements}, 256) = {gcd}")
    
    # 如何调整?
    # 方案: 按行切片 (不破坏量化块)
    print(f"\n  💡 Solution: Slice by rows, not elements")
    print(f"  Each row has {gate_up_dim} elements")
    print(f"  Row elements / 256 = {gate_up_dim / 256:.2f}")
    print(f"  gate_up_dim % 256 = {gate_up_dim % 256}")
    
    if gate_up_dim % 256 == 0:
        print(f"  ✅ gate_up_dim divisible by 256!")
        print(f"  Can slice by rows (each row is complete quant blocks)")

# 验证实际大小
print("\n📊 Verify with actual file sizes:")
print("-" * 60)

# 从之前的输出:
# ffn_gate_up_exps.weight: [2816, 1408, 128], size=1936 MB
# 每个专家: 1936 / 128 = 15.125 MB
# 每个专家 gate_up: 15.125 MB = 15,854,488 bytes (约)
# gate 部分: 7.5625 MB = 7,930,244 bytes (约)
# up 部分: 同上

total_expert_bytes = 1936 * 1024 * 1024 / 128  # 每个专家的字节数
gate_part_bytes = total_expert_bytes / 2  # gate 占一半
up_part_bytes = total_expert_bytes / 2  # up 占一半

print(f"  Full expert gate_up: {total_expert_bytes:.0f} bytes")
print(f"  Gate part: {gate_part_bytes:.0f} bytes")
print(f"  Up part: {up_part_bytes:.0f} bytes")

# 计算实际 bpe
actual_bpe = total_expert_bytes / (hidden * gate_up_dim)
print(f"  Actual bytes per element: {actual_bpe:.4f}")
print(f"  This corresponds to type with bpe ≈ {actual_bpe:.2f}")

# IQ4_XS 通常是约 1.5 bytes/element (4 bits 数据 + scale)
# Q4_K 是 2 bytes/element (但存储更大, 因为 scale 开销)

print("\n🔍 Verify block alignment with actual data:")
# 如果 gate_part_bytes 能被块大小整除, 则对齐
# 块大小 = 256 元素 × bpe
gate_block_bytes = 256 * actual_bpe
print(f"  Gate block size (bytes): {gate_block_bytes:.1f}")
print(f"  Gate part / block = {gate_part_bytes / gate_block_bytes:.2f}")

if gate_part_bytes % gate_block_bytes < 1:
    print(f"  ✅ Aligned to actual blocks!")
else:
    print(f"  ⚠️  May not be perfectly aligned (need to verify format)")

# 总结
print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

if gate_elements % 256 == 0 and up_elements % 256 == 0:
    print("\n✅ PERFECT ALIGNMENT!")
    print("   Gate and up split points align to quantization block boundaries.")
    print("   The naive split (midpoint of gate_up) is safe.")
else:
    print("\n⚠️  NEEDS VERIFICATION")
    print(f"   Gate elements mod 256 = {gate_elements % 256}")
    print("   Recommend: Verify with actual byte-level inspection.")
    print("   If not aligned, need to adjust split point to nearest block boundary.")

print(f"\n  gate_elements = {gate_elements}")
print(f"  gate_elements / 256 = {gate_elements / 256}")
print(f"  All good: {gate_elements % 256 == 0}")
