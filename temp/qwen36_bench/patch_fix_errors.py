#!/usr/bin/env python3
"""Fix the two compile-error classes from the Qwen36 wiring:
1. context.queue is `any MTLCommandQueue` (non-optional) -> drop `guard let`
2. MTLBuffer.contents().assumingMemoryBound() returns a non-optional pointer
   -> drop the `if let` around memset in GatedAttnRunner.reset()
"""
base = "/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/turbo-fieldfare/Sources/TurboFieldfare"

# ---- Qwen36ForwardRunner: queue is non-optional -----------------------------
p = base + "/Runtime/Inference/Qwen36ForwardRunner.swift"
src = open(p).read()
old = "        guard let queue = context.queue else { return }\n        guard let cb = queue.makeCommandBuffer() else { return }"
new = "        let queue = context.queue\n        guard let cb = queue.makeCommandBuffer() else { return }"
assert old in src, "queue guard not found"
src = src.replace(old, new)
open(p, "w").write(src)
print("Qwen36ForwardRunner: queue guard fixed")

# ---- GatedAttnRunner.reset(): non-optional contents pointer -----------------
p = base + "/Kernels/Attention/GatedAttnRunner.swift"
src = open(p).read()
old = """    public func reset() {
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
    }"""
new = """    public func reset() {
        let kc = kCache.contents().assumingMemoryBound(to: Float16.self)
        memset(kc, 0, kCache.length)
        let vc = vCache.contents().assumingMemoryBound(to: Float16.self)
        memset(vc, 0, vCache.length)
        let q = qOut.contents().assumingMemoryBound(to: Float16.self)
        memset(q, 0, qOut.length)
        let k = kOut.contents().assumingMemoryBound(to: Float16.self)
        memset(k, 0, kOut.length)
        let v = vOut.contents().assumingMemoryBound(to: Float16.self)
        memset(v, 0, vOut.length)
        let g = gateOut.contents().assumingMemoryBound(to: Float16.self)
        memset(g, 0, gateOut.length)
        let s = scoresBuf.contents().assumingMemoryBound(to: Float.self)
        memset(s, 0, scoresBuf.length)
    }"""
assert old in src, "GatedAttn reset not found"
src = src.replace(old, new)
open(p, "w").write(src)
print("GatedAttnRunner.reset: non-optional pointer fixed")
