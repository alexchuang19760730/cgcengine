#!/usr/bin/env python3
"""Add GPU-buffer accessors + reset to the three Qwen36 runners so the
Qwen36ForwardRunner can chain layers on-GPU (no CPU readback per layer):

- DeltaNetRunner:  reset() clears h/conv; expose nothing extra (writes caller y)
- GatedAttnRunner: yOutBuffer accessor + reset() clears KV caches + rope state
- Qwen36MoERunner: moeOutBuffer accessor (fp32 [n,2048])
"""
base = "/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/turbo-fieldfare/Sources/TurboFieldfare/Kernels/Attention"

# ---- GatedAttnRunner: yOutBuffer + reset -----------------------------------
p = base + "/GatedAttnRunner.swift"
src = open(p).read()

old1 = """    public func readScores(head: Int, seqLen: Int) -> [Float] {"""
new1 = """    /// GPU output buffer [2048] fp16 (o_proj result) — for on-GPU chaining.
    public var yOutBuffer: MTLBuffer { yOut }

    /// Clear KV caches and per-call scratch (start of a new generation).
    public func reset() {
        if let kc = kCache.contents().assumingMemoryBound(to: Float16.self) {
            memset(kc, 0, kCache.length)
        }
        if let vc = vCache.contents().assumingMemoryBound(to: Float16.self) {
            memset(vc, 0, vCache.length)
        }
        if let q = qOut.contents().assumingMemoryBound(to: Float16.self) {
            memset(q, 0, qOut.length)
        }
        if let k = kOut.contents().assumingMemoryBound(to: Float16.self) {
            memset(k, 0, kOut.length)
        }
        if let v = vOut.contents().assumingMemoryBound(to: Float16.self) {
            memset(v, 0, vOut.length)
        }
        if let g = gateOut.contents().assumingMemoryBound(to: Float16.self) {
            memset(g, 0, gateOut.length)
        }
        if let s = scoresBuf.contents().assumingMemoryBound(to: Float.self) {
            memset(s, 0, scoresBuf.length)
        }
    }

    public func readScores(head: Int, seqLen: Int) -> [Float] {"""
assert old1 in src, "GatedAttn readScores anchor not found"
src = src.replace(old1, new1)
open(p, "w").write(src)
print("GatedAttnRunner: yOutBuffer + reset added")

# ---- Qwen36MoERunner: moeOutBuffer accessor ---------------------------------
p = base + "/../MoE/Qwen36MoERunner.swift"
src = open(p).read()

old2 = """    public func readIds(nTokens: Int) -> [UInt32] {"""
new2 = """    /// GPU MoE output [n, 2048] fp32 — for on-GPU chaining (forward path).
    public var moeOutBuffer: MTLBuffer { moeOutBuf }

    public func readIds(nTokens: Int) -> [UInt32] {"""
assert old2 in src, "MoE readIds anchor not found"
src = src.replace(old2, new2)
open(p, "w").write(src)
print("Qwen36MoERunner: moeOutBuffer accessor added")

# ---- DeltaNetRunner: reset() clears h/conv state ----------------------------
p = base + "/DeltaNetRunner.swift"
src = open(p).read()

old3 = """    /// 读回状态 h [32,128,128] FP32 (跨 token 持久)。"""
new3 = """    /// Clear persistent state (h + conv) — start of a new generation.
    public func reset() {
        if let hp = hState.contents().assumingMemoryBound(to: Float.self) {
            memset(hp, 0, hState.length)
        }
        if let cp = convState.contents().assumingMemoryBound(to: Float16.self) {
            memset(cp, 0, convState.length)
        }
        if let tp = convStateTmp.contents().assumingMemoryBound(to: Float16.self) {
            memset(tp, 0, convStateTmp.length)
        }
    }

    /// 读回状态 h [32,128,128] FP32 (跨 token 持久)。"""
assert old3 in src, "DeltaNet readH anchor not found"
src = src.replace(old3, new3)
open(p, "w").write(src)
print("DeltaNetRunner: reset() added")
