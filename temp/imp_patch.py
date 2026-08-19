#!/usr/bin/env python3
"""Add Q36IMP per-layer importance accumulation to Qwen36ForwardRunner.swift.
Gated by existing Q36_DUMP_LAYERS; normal path untouched."""
import sys

P = "/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/turbo-fieldfare/Sources/TurboFieldfare/Runtime/Inference/Qwen36ForwardRunner.swift"
src = open(P).read()
orig = src

# ---- Edit 1: class properties after dumpLayers ----
a1 = "    private let dumpLayers: Bool\n"
b1 = a1 + """    /// Layer-importance accumulation (Q36_DUMP_LAYERS only): per-layer RMS
    /// of attnOut / moeOut / hidden accumulated across decode tokens, printed
    /// as Q36IMP every 32 tokens. relT = (rmsA + rmsM) / rmsH.
    private var impAttnSq = [Float](repeating: 0, count: 64)
    private var impMoeSq = [Float](repeating: 0, count: 64)
    private var impHiddenSq = [Float](repeating: 0, count: 64)
    private var impCount = 0
"""
assert src.count(a1) == 1, "E1 anchor"
src = src.replace(a1, b1, 1)

# ---- Edit 2: helper methods before dumpStats32 ----
a2 = "    private func dumpStats32(_ label: String, _ buf: MTLBuffer, _ n: Int) {\n"
b2 = """    private func impAccHalf(_ buf: MTLBuffer, _ n: Int, _ L: Int, _ arr: inout [Float]) {
        guard dumpLayers else { return }
        let tmp = context.device.makeBuffer(length: n * MemoryLayout<Float16>.size,
                                            options: .storageModeShared)!
        let cb = context.queue.makeCommandBuffer()!
        if let blt = cb.makeBlitCommandEncoder() {
            blt.copy(from: buf, sourceOffset: 0, to: tmp, destinationOffset: 0,
                     size: n * MemoryLayout<Float16>.size)
            blt.endEncoding()
        }
        cb.commit()
        cb.waitUntilCompleted()
        let p = tmp.contents().assumingMemoryBound(to: Float16.self)
        var s: Float = 0
        for i in 0..<n { let f = Float(p[i]); s += f * f }
        arr[L] += s
    }

    private func impAccF32(_ buf: MTLBuffer, _ n: Int, _ L: Int, _ arr: inout [Float]) {
        guard dumpLayers else { return }
        let tmp = context.device.makeBuffer(length: n * MemoryLayout<Float>.size,
                                            options: .storageModeShared)!
        let cb = context.queue.makeCommandBuffer()!
        if let blt = cb.makeBlitCommandEncoder() {
            blt.copy(from: buf, sourceOffset: 0, to: tmp, destinationOffset: 0,
                     size: n * MemoryLayout<Float>.size)
            blt.endEncoding()
        }
        cb.commit()
        cb.waitUntilCompleted()
        let p = tmp.contents().assumingMemoryBound(to: Float.self)
        var s: Float = 0
        for i in 0..<n { s += p[i] * p[i] }
        arr[L] += s
    }

    private func printImportance() {
        guard dumpLayers, impCount > 0 else { return }
        print("Q36IMP tokens=\\(impCount) D=\\(D)")
        for L in 0..<cfg.numLayers {
            let n = Float(Int(D) * impCount)
            let rmsA = sqrt(impAttnSq[L] / n)
            let rmsM = sqrt(impMoeSq[L] / n)
            let rmsH = sqrt(impHiddenSq[L] / n)
            let relA = rmsH > 0 ? rmsA / rmsH : 0
            let relM = rmsH > 0 ? rmsM / rmsH : 0
            let kind = deltanetLayers.contains(L) ? "DN" : "GA"
            print("Q36IMP L\\(L) \\(kind) rmsA=\\(String(format: "%.4f", rmsA)) rmsM=\\(String(format: "%.4f", rmsM)) rmsH=\\(String(format: "%.4f", rmsH)) relA=\\(String(format: "%.4f", relA)) relM=\\(String(format: "%.4f", relM)) relT=\\(String(format: "%.4f", relA + relM))")
        }
    }

""" + a2
assert src.count(a2) == 1, "E2 anchor"
src = src.replace(a2, b2, 1)

# ---- Edit 3: dumpFront accumulation ----
a3 = '            dumpStats("\\(L).normed", ffnNormedLayers[L], Int(D))\n'
b3 = """            if L == 0 { impCount += 1 }
            impAccHalf(attnOut, Int(D), L, &impAttnSq)
            impAccHalf(hidden, Int(D), L, &impHiddenSq)
""" + a3
assert src.count(a3) == 1, "E3 anchor"
src = src.replace(a3, b3, 1)

# ---- Edit 4: dumpBack accumulation ----
a4 = '                dumpStats32("\\(L).moeOut", moes[L].moeOutBuffer, Int(D))\n            }\n        }\n'
b4 = '                dumpStats32("\\(L).moeOut", moes[L].moeOutBuffer, Int(D))\n            }\n            impAccF32(moes[L].moeOutBuffer, Int(D), L, &impMoeSq)\n        }\n'
assert src.count(a4) == 1, "E4 anchor"
src = src.replace(a4, b4, 1)

# ---- Edit 5: periodic summary print in produce() ----
a5 = '        if dumpLayers {\n            dumpStats("final.hidden", hidden, Int(D))\n'
b5 = '        if dumpLayers {\n            if position % 32 == 31 { printImportance() }\n            dumpStats("final.hidden", hidden, Int(D))\n'
assert src.count(a5) == 1, "E5 anchor"
src = src.replace(a5, b5, 1)

open(P, "w").write(src)
print("patched OK, delta bytes:", len(src) - len(orig))
