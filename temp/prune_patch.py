#!/usr/bin/env python3
"""Add Q36_PRUNE_LAYERS layer-pruning support to Qwen36ForwardRunner.swift.
Pruned layers: front/back skipped (residual passthrough); deltaNets/gatedAttns
counter indices consumed so subsequent layers stay aligned."""
P = "/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/turbo-fieldfare/Sources/TurboFieldfare/Runtime/Inference/Qwen36ForwardRunner.swift"
src = open(P).read()
orig = src

# ---- Edit 1: property ----
a1 = "    private let deltanetLayers: Set<Int>\n"
b1 = a1 + """    /// Q36_PRUNE_LAYERS=... zero-training layer pruning: comma-separated layer
    /// indices whose front/back are skipped (residual passthrough). Runner
    /// counter indices are consumed so later layers stay aligned.
    private let prunedLayers: Set<Int>
"""
assert src.count(a1) == 1, "E1"
src = src.replace(a1, b1, 1)

# ---- Edit 2: init env read ----
a2 = "        let manifestLayers = model.manifest.arch.deltanetLayers\n        self.deltanetLayers = Set(manifestLayers ?? Self.defaultDeltanetLayers(numLayers: cfg.numLayers))\n"
b2 = a2 + """        if let pruneEnv = ProcessInfo.processInfo.environment["Q36_PRUNE_LAYERS"],
           !pruneEnv.isEmpty {
            let ids = pruneEnv.split(separator: ",").compactMap {
                Int($0.trimmingCharacters(in: .whitespaces))
            }
            self.prunedLayers = Set(ids)
        } else {
            self.prunedLayers = []
        }
"""
assert src.count(a2) == 1, "E2"
src = src.replace(a2, b2, 1)

# ---- Edit 3: CB(0) guards ----
a3 = """        // ---- CB(0): front(0) ---- (no back(0) yet — nothing to merge with)
        if kernelTimingEnabled {
            try encodeTimedFrontNormAttn(0)
        } else {
            guard let cb0 = queue.makeCommandBuffer() else {
                throw PrefillError.chunkedUnsupported("qwen36: front command buffer")
            }
            let tEnc0 = Date()
            try encodeFront(0, cb0)
            cpuEncodeAcc += Date().timeIntervalSince(tEnc0)
            cb0.commit()
            commitAndWait(cb0)
            record(cb0)
        }
"""
b3 = """        // ---- CB(0): front(0) ---- (no back(0) yet — nothing to merge with)
        if kernelTimingEnabled {
            if !prunedLayers.contains(0) { try encodeTimedFrontNormAttn(0) }
        } else {
            guard let cb0 = queue.makeCommandBuffer() else {
                throw PrefillError.chunkedUnsupported("qwen36: front command buffer")
            }
            let tEnc0 = Date()
            if !prunedLayers.contains(0) { try encodeFront(0, cb0) }
            cpuEncodeAcc += Date().timeIntervalSince(tEnc0)
            cb0.commit()
            commitAndWait(cb0)
            record(cb0)
        }
"""
assert src.count(a3) == 1, "E3"
src = src.replace(a3, b3, 1)

# ---- Edit 4: main loop ----
a4 = """        // ---- CB(1..39): back(L-1) + front(L) in one command buffer ----
        for L in 1..<cfg.numLayers {
            if kernelTimingEnabled {
                try runSeg(1, L - 1) { encodeBackExperts(L - 1, $0) }
                try runSeg(2, L - 1) { encodeBackShared(L - 1, $0) }
                try runSeg(3, L - 1) { encodeBackMerge(L - 1, $0) }
                try encodeTimedFrontNormAttn(L)
            } else {
                guard let cb = queue.makeCommandBuffer() else {
                    throw PrefillError.chunkedUnsupported("qwen36: merged command buffer")
                }
                let tEnc = Date()
                encodeBack(L - 1, cb)
                try encodeFront(L, cb)
                cpuEncodeAcc += Date().timeIntervalSince(tEnc)
                cb.commit()
                commitAndWait(cb)
                record(cb)
            }
            dumpBack(L - 1)
            dumpFront(L)
            let moe = moes[L]
            let tRead = Date()
            let ids = moe.readIdsFast(nTokens: 1)
            cpuReadAcc += Date().timeIntervalSince(tRead)
            if dumpLayers && (L == 36 || L == 37) {
                print("Q36DUMP \\(L).routerIds = \\(ids)")
            }
            if moeSpAcEnabled { moeSpAcLayerIds[L] = ids }
            let tFill = Date()
            try moe.fillSlotTable(ids, decodeStepIndex: position)
            fillAcc += Date().timeIntervalSince(tFill)
            if realRoutePrefetch {
                realRouteByLayer[L] = ids.map { Int($0) }
"""
b4 = """        // ---- CB(1..39): back(L-1) + front(L) in one command buffer ----
        // §13.118 層裁剪（Q36_PRUNE_LAYERS）：pruned 層 front/back 整個跳過、
        // hidden 殘差原樣通過（層跳躍語義）。deltaNets/gatedAttns 用計數器索引
        // ——pruned 層必須消耗自己的 deltaIdx/gatedIdx 保持後續層對齊。
        for L in 1..<cfg.numLayers {
            let pruneL = prunedLayers.contains(L)
            let prunePrev = prunedLayers.contains(L - 1)
            if pruneL {
                if deltanetLayers.contains(L) { deltaIdx += 1 } else { gatedIdx += 1 }
            }
            if kernelTimingEnabled {
                if !prunePrev {
                    try runSeg(1, L - 1) { encodeBackExperts(L - 1, $0) }
                    try runSeg(2, L - 1) { encodeBackShared(L - 1, $0) }
                    try runSeg(3, L - 1) { encodeBackMerge(L - 1, $0) }
                }
                if !pruneL { try encodeTimedFrontNormAttn(L) }
            } else {
                if prunePrev && pruneL { continue }
                guard let cb = queue.makeCommandBuffer() else {
                    throw PrefillError.chunkedUnsupported("qwen36: merged command buffer")
                }
                let tEnc = Date()
                if !prunePrev { encodeBack(L - 1, cb) }
                if !pruneL { try encodeFront(L, cb) }
                cpuEncodeAcc += Date().timeIntervalSince(tEnc)
                cb.commit()
                commitAndWait(cb)
                record(cb)
            }
            if !prunePrev { dumpBack(L - 1) }
            if pruneL { continue }
            dumpFront(L)
            let moe = moes[L]
            let tRead = Date()
            let ids = moe.readIdsFast(nTokens: 1)
            cpuReadAcc += Date().timeIntervalSince(tRead)
            if dumpLayers && (L == 36 || L == 37) {
                print("Q36DUMP \\(L).routerIds = \\(ids)")
            }
            if moeSpAcEnabled { moeSpAcLayerIds[L] = ids }
            let tFill = Date()
            try moe.fillSlotTable(ids, decodeStepIndex: position)
            fillAcc += Date().timeIntervalSince(tFill)
            if realRoutePrefetch {
                realRouteByLayer[L] = ids.map { Int($0) }
"""
assert src.count(a4) == 1, "E4"
src = src.replace(a4, b4, 1)

# ---- Edit 5: final back(39) guard ----
a5 = """        // ---- CB(40): back(39) ----
        if kernelTimingEnabled {
            try runSeg(1, cfg.numLayers - 1) { encodeBackExperts(cfg.numLayers - 1, $0) }
            try runSeg(2, cfg.numLayers - 1) { encodeBackShared(cfg.numLayers - 1, $0) }
            try runSeg(3, cfg.numLayers - 1) { encodeBackMerge(cfg.numLayers - 1, $0) }
        } else {
            guard let cbLast = queue.makeCommandBuffer() else {
                throw PrefillError.chunkedUnsupported("qwen36: final back command buffer")
            }
            let tEncL = Date()
            encodeBack(cfg.numLayers - 1, cbLast)
            cpuEncodeAcc += Date().timeIntervalSince(tEncL)
            cbLast.commit()
            commitAndWait(cbLast)
            record(cbLast)
        }
        dumpBack(cfg.numLayers - 1)
"""
b5 = """        // ---- CB(40): back(39) ----
        if !prunedLayers.contains(cfg.numLayers - 1) {
            if kernelTimingEnabled {
                try runSeg(1, cfg.numLayers - 1) { encodeBackExperts(cfg.numLayers - 1, $0) }
                try runSeg(2, cfg.numLayers - 1) { encodeBackShared(cfg.numLayers - 1, $0) }
                try runSeg(3, cfg.numLayers - 1) { encodeBackMerge(cfg.numLayers - 1, $0) }
            } else {
                guard let cbLast = queue.makeCommandBuffer() else {
                    throw PrefillError.chunkedUnsupported("qwen36: final back command buffer")
                }
                let tEncL = Date()
                encodeBack(cfg.numLayers - 1, cbLast)
                cpuEncodeAcc += Date().timeIntervalSince(tEncL)
                cbLast.commit()
                commitAndWait(cbLast)
                record(cbLast)
            }
        }
        if !prunedLayers.contains(cfg.numLayers - 1) { dumpBack(cfg.numLayers - 1) }
"""
assert src.count(a5) == 1, "E5"
src = src.replace(a5, b5, 1)

open(P, "w").write(src)
print("patched OK, delta bytes:", len(src) - len(orig))
