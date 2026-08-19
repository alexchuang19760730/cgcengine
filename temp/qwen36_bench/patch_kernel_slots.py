#!/usr/bin/env python3
"""Change q36_moe_expert kernel: replace single blob with 8 per-rank slot buffers.
The body uses `slotBuf + gateQOff` (no e*expertStride base) since each slot buffer
holds exactly one expert's data at offset 0."""
path = "/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/turbo-fieldfare/Sources/TurboFieldfare/Metal/MoE/moe_qwen36.metal"
src = open(path).read()

# 1. Replace the kernel signature: blob -> 8 slot buffers, shift param indices
old_sig = """kernel void q36_moe_expert(
    device const half*   xIn       [[buffer(0)]],   // [n, 2048]
    device const uint*   idsIn     [[buffer(1)]],   // [n, 8]
    device const uchar*  blob      [[buffer(2)]],   // layer_XX.bin
    device float*        expertOut [[buffer(3)]],   // [n, 8, 2048]
    constant uint&       expertStride [[buffer(4)]],
    constant uint&       gateQOff  [[buffer(5)]],
    constant uint&       gateSOff  [[buffer(6)]],
    constant uint&       upQOff    [[buffer(7)]],
    constant uint&       upSOff    [[buffer(8)]],
    constant uint&       downQOff  [[buffer(9)]],
    constant uint&       downSOff  [[buffer(10)]],"""

new_sig = """kernel void q36_moe_expert(
    device const half*   xIn       [[buffer(0)]],   // [n, 2048]
    device const uint*   idsIn     [[buffer(1)]],   // [n, 8]
    device const uchar*  slotBuf0  [[buffer(2)]],   // per-rank expert slot
    device const uchar*  slotBuf1  [[buffer(3)]],
    device const uchar*  slotBuf2  [[buffer(4)]],
    device const uchar*  slotBuf3  [[buffer(5)]],
    device const uchar*  slotBuf4  [[buffer(6)]],
    device const uchar*  slotBuf5  [[buffer(7)]],
    device const uchar*  slotBuf6  [[buffer(8)]],
    device const uchar*  slotBuf7  [[buffer(9)]],
    device float*        expertOut [[buffer(10)]],   // [n, 8, 2048]
    constant uint&       expertStride [[buffer(11)]],
    constant uint&       gateQOff  [[buffer(12)]],
    constant uint&       gateSOff  [[buffer(13)]],
    constant uint&       upQOff    [[buffer(14)]],
    constant uint&       upSOff    [[buffer(15)]],
    constant uint&       downQOff  [[buffer(16)]],
    constant uint&       downSOff  [[buffer(17)]],"""

assert old_sig in src, "kernel sig not found"
src = src.replace(old_sig, new_sig, 1)

# 2. After `const uint rank = tg % Q36_TOP_K;`, add slot buffer select
old_body = """    const uint rank = tg % Q36_TOP_K;
    const uint e = idsIn[t * Q36_TOP_K + rank];

    for (uint i = lid; i < Q36_HIDDEN; i += Q36_TG_E) {"""

new_body = """    const uint rank = tg % Q36_TOP_K;
    const uint e = idsIn[t * Q36_TOP_K + rank];
    // Select the per-rank slot buffer (streamer loads each expert into its own).
    threadgroup const uchar* slotBuf;
    if (rank == 0) { slotBuf = slotBuf0; }
    else if (rank == 1) { slotBuf = slotBuf1; }
    else if (rank == 2) { slotBuf = slotBuf2; }
    else if (rank == 3) { slotBuf = slotBuf3; }
    else if (rank == 4) { slotBuf = slotBuf4; }
    else if (rank == 5) { slotBuf = slotBuf5; }
    else if (rank == 6) { slotBuf = slotBuf6; }
    else { slotBuf = slotBuf7; }

    for (uint i = lid; i < Q36_HIDDEN; i += Q36_TG_E) {"""

assert old_body in src, "kernel body anchor not found"
src = src.replace(old_body, new_body, 1)

# 3. Replace `const uint base = e * expertStride;` with `const uint base = 0;`
old_base = "    const uint base = e * expertStride;"
new_base = "    const uint base = 0u;  // slot buffer has one expert at offset 0"
assert old_base in src, "base line not found"
src = src.replace(old_base, new_base, 1)

# 4. Replace blob references with slotBuf in the body
old_blobref = "    device const uchar*  gq = blob + base + gateQOff;   // [512, 1024] u8"
new_blobref = "    device const uchar*  gq = slotBuf + base + gateQOff;   // [512, 1024] u8"
assert old_blobref in src, "gq line not found"
src = src.replace(old_blobref, new_blobref, 1)

old_gs = "    device const float*  gs = (device const float*)(blob + base + gateSOff); // [512, 32]"
new_gs = "    device const float*  gs = (device const float*)(slotBuf + base + gateSOff); // [512, 32]"
src = src.replace(old_gs, new_gs, 1)

old_uq = "    device const uchar*  uq = blob + base + upQOff;"
new_uq = "    device const uchar*  uq = slotBuf + base + upQOff;"
src = src.replace(old_uq, new_uq, 1)

old_us = "    device const float*  us = (device const float*)(blob + base + upSOff);"
new_us = "    device const float*  us = (device const float*)(slotBuf + base + upSOff);"
src = src.replace(old_us, new_us, 1)

old_dq = "    device const uchar* dq = blob + base + downQOff;  // [2048, 256] u8"
new_dq = "    device const uchar* dq = slotBuf + base + downQOff;  // [2048, 256] u8"
src = src.replace(old_dq, new_dq, 1)

old_ds = "    device const float* ds = (device const float*)(blob + base + downSOff); // [2048, 8]"
new_ds = "    device const float* ds = (device const float*)(slotBuf + base + downSOff); // [2048, 8]"
src = src.replace(old_ds, new_ds, 1)

open(path, "w").write(src)
print("moe_qwen36.metal: q36_moe_expert -> 8 per-rank slot buffers")