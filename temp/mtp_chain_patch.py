#!/usr/bin/env python3
"""Apply Swift patches for EAGLE-2 chain-dump support (knob A) + baseline bridge gating (knob B)."""

def patch(path, old, new, count=1, must=True):
    src = open(path).read()
    if old not in src:
        if must:
            raise SystemExit(f"OLD NOT FOUND in {path}:\n{old[:200]!r}")
        print(f"skip (absent): {path}")
        return
    assert src.count(old) >= count, f"count mismatch in {path}: {src.count(old)} < {count}"
    src = src.replace(old, new, count)
    open(path, "w").write(src)
    print(f"patched {path} ({count})")

D = "turbo-fieldfare-github-official/Sources/TurboFieldfare/Runtime/Generation/TrainingDataDump.swift"
R = "turbo-fieldfare-github-official/Sources/TurboFieldfare/Runtime/Inference/RealForwardRunner.swift"
M = "turbo-fieldfare-github-official/Sources/TurboFieldfare/Runtime/Generation/MTPCompletion.swift"
C = "turbo-fieldfare-github-official/Sources/TurboFieldfare/Runtime/Generation/RawCompletion.swift"

# ---------- 1. TrainingDataDump: chain handle ----------
patch(D, '''    static let handle: FileHandle? = {
        guard let p = path else { return nil }
        if !FileManager.default.fileExists(atPath: p) {
            FileManager.default.createFile(atPath: p, contents: nil)
        }
        return FileHandle(forWritingAtPath: p)
    }()''',
'''    static let handle: FileHandle? = {
        guard let p = path else { return nil }
        if !FileManager.default.fileExists(atPath: p) {
            FileManager.default.createFile(atPath: p, contents: nil)
        }
        return FileHandle(forWritingAtPath: p)
    }()

    private static let chainPath =
        ProcessInfo.processInfo.environment["TURBO_FIELDFARE_MTP_CHAIN_DUMP"]

    static let chainHandle: FileHandle? = {
        guard let p = chainPath else { return nil }
        if !FileManager.default.fileExists(atPath: p) {
            FileManager.default.createFile(atPath: p, contents: nil)
        }
        return FileHandle(forWritingAtPath: p)
    }()''')

# ---------- 2. extract KV writer into a shared private static ----------
old_kv = '''        func appendKV(_ snap: AssistantBridgeKVSnapshot?) {
            if let s = snap {
                let valid = s.key.validTokenCount
                var pos = Int32(valid)
                var nk = Int32(s.numKVHeads)
                var hd = Int32(s.headDim)
                var stride = Int32(s.key.stride)
                var ringCap = Int32(s.key.ringCapacity)
                var startSlot = Int32(s.key.startSlot)
                rec.append(Data(bytes: &pos, count: 4))
                rec.append(Data(bytes: &nk, count: 4))
                rec.append(Data(bytes: &hd, count: 4))
                rec.append(Data(bytes: &stride, count: 4))
                rec.append(Data(bytes: &ringCap, count: 4))
                rec.append(Data(bytes: &startSlot, count: 4))
                func copyTokens(_ b: MTLBuffer, offset: Int) {
                    guard valid > 0 else { return }
                    let halfsPerToken = Int(nk) * Int(hd)
                    let byteStride = s.key.stride
                    let p = b.contents().advanced(by: offset * 2)
                        .assumingMemoryBound(to: Float16.self)
                    var tok = Data()
                    tok.reserveCapacity(valid * halfsPerToken * 2)
                    for t in 0..<valid {
                        var slot = t
                        if Int(startSlot) > 0 {
                            slot = (Int(startSlot) + t) % max(Int(ringCap), 1)
                        }
                        let base = p.advanced(by: slot * byteStride / 2)
                        tok.append(Data(bytes: base, count: halfsPerToken * 2))
                    }
                    rec.append(tok)
                }
                copyTokens(s.key.buffer, offset: s.key.offset)
                copyTokens(s.value.buffer, offset: s.value.offset)
            } else {
                var pos = Int32(0), nk = Int32(0), hd = Int32(0), stride = Int32(0), ringCap = Int32(0), startSlot = Int32(-1)
                rec.append(Data(bytes: &pos, count: 4))
                rec.append(Data(bytes: &nk, count: 4))
                rec.append(Data(bytes: &hd, count: 4))
                rec.append(Data(bytes: &stride, count: 4))
                rec.append(Data(bytes: &ringCap, count: 4))
                rec.append(Data(bytes: &startSlot, count: 4))
            }
        }
        appendKV(bridge.slidingAttentionKV)
        appendKV(bridge.fullAttentionKV)
        handle.write(rec)
    }'''
new_kv = '''        Self.writeKV(bridge.slidingAttentionKV, into: &rec)
        Self.writeKV(bridge.fullAttentionKV, into: &rec)
        handle.write(rec)
    }

    /// Append one KV snapshot (6-int header + token-major FP16 k/v) to `rec`.
    private static func writeKV(_ snap: AssistantBridgeKVSnapshot?, into rec: inout Data) {
        if let s = snap {
            let valid = s.key.validTokenCount
            var pos = Int32(valid)
            var nk = Int32(s.numKVHeads)
            var hd = Int32(s.headDim)
            var stride = Int32(s.key.stride)
            var ringCap = Int32(s.key.ringCapacity)
            var startSlot = Int32(s.key.startSlot)
            rec.append(Data(bytes: &pos, count: 4))
            rec.append(Data(bytes: &nk, count: 4))
            rec.append(Data(bytes: &hd, count: 4))
            rec.append(Data(bytes: &stride, count: 4))
            rec.append(Data(bytes: &ringCap, count: 4))
            rec.append(Data(bytes: &startSlot, count: 4))
            func copyTokens(_ b: MTLBuffer, offset: Int) {
                guard valid > 0 else { return }
                let halfsPerToken = Int(nk) * Int(hd)
                let byteStride = s.key.stride
                let p = b.contents().advanced(by: offset * 2)
                    .assumingMemoryBound(to: Float16.self)
                var tok = Data()
                tok.reserveCapacity(valid * halfsPerToken * 2)
                for t in 0..<valid {
                    var slot = t
                    if Int(startSlot) > 0 {
                        slot = (Int(startSlot) + t) % max(Int(ringCap), 1)
                    }
                    let base = p.advanced(by: slot * byteStride / 2)
                    tok.append(Data(bytes: base, count: halfsPerToken * 2))
                }
                rec.append(tok)
            }
            copyTokens(s.key.buffer, offset: s.key.offset)
            copyTokens(s.value.buffer, offset: s.value.offset)
        } else {
            var pos = Int32(0), nk = Int32(0), hd = Int32(0), stride = Int32(0), ringCap = Int32(0), startSlot = Int32(-1)
            rec.append(Data(bytes: &pos, count: 4))
            rec.append(Data(bytes: &nk, count: 4))
            rec.append(Data(bytes: &hd, count: 4))
            rec.append(Data(bytes: &stride, count: 4))
            rec.append(Data(bytes: &ringCap, count: 4))
            rec.append(Data(bytes: &startSlot, count: 4))
        }
    }'''
patch(D, old_kv, new_kv)

# ---------- 3. appendChain at end of enum ----------
old_end = '''        handle.write(rec)
    }
}'''
new_end = '''        handle.write(rec)
    }

    /// EAGLE-2-style chain record for rollout training
    /// (`TURBO_FIELDFARE_MTP_CHAIN_DUMP=<path>`). One record per MTP verify of
    /// `span = [ctx] + drafts`:
    ///
    ///   B i32 | hidden[2816] f32 | embed_ctx[2816] f32 | ctx i32
    ///   drafts[B] i32 | predictions[B+1] i32
    ///   rowHidden[(B+1)*2816] f16 | embed_drafts[B][2816] f32
    ///   [sliding KV header + k/v] [full KV header + k/v]
    ///
    /// `predictions[i]` = target greedy after span[0...i] (the label draft i
    /// should match). `rowHidden[i+1]` = backbone hidden after span[0...i]
    /// (the feature target for the head's predicted hidden ĥ_i). Row hiddens
    /// are conditioned on the *drafted* prefix — exactly the self-consistent
    /// rollout distribution EAGLE-2/3 trains on.
    public static func appendChain(bridge: AssistantBridgeSnapshot,
                                   ctx: Int32,
                                   drafts: [Int32],
                                   predictions: [Int32],
                                   rowHiddens: [[Float]],
                                   draftEmbeddings: [[Float]?]) {
        guard let chainHandle, bridge.lastHiddenState.count > 0 else { return }
        var rec = Data()
        var B = Int32(drafts.count)
        rec.append(Data(bytes: &B, count: 4))
        bridge.lastHiddenState.withUnsafeBufferPointer { buf in
            rec.append(UnsafeBufferPointer(start: buf.baseAddress, count: buf.count))
        }
        let embLen = bridge.lastHiddenState.count
        if let emb = bridge.lastTokenEmbedding, emb.count == embLen {
            emb.withUnsafeBufferPointer { buf in
                rec.append(UnsafeBufferPointer(start: buf.baseAddress, count: buf.count))
            }
        } else {
            rec.append(Data(count: embLen * 4))
        }
        var c = ctx
        rec.append(Data(bytes: &c, count: 4))
        for d in drafts { var dd = d; rec.append(Data(bytes: &dd, count: 4)) }
        for p in predictions { var pp = p; rec.append(Data(bytes: &pp, count: 4)) }
        for row in rowHiddens {
            var half = row.map { Float16($0) }
            half.withUnsafeBufferPointer { buf in
                rec.append(UnsafeBufferPointer(start: buf.baseAddress, count: buf.count))
            }
        }
        for emb in draftEmbeddings {
            if let e = emb, e.count == embLen {
                e.withUnsafeBufferPointer { buf in
                    rec.append(UnsafeBufferPointer(start: buf.baseAddress, count: buf.count))
                }
            } else {
                rec.append(Data(count: embLen * 4))
            }
        }
        Self.writeKV(bridge.slidingAttentionKV, into: &rec)
        Self.writeKV(bridge.fullAttentionKV, into: &rec)
        chainHandle.write(rec)
    }
}'''
patch(D, old_end, new_end)

# ---------- 4. RealForwardRunner: copyVerifyRowHidden ----------
old_rh = '''    /// Copy the last decode hidden state as Float32 for MTP draft generation.
    public func copyLastHiddenState() -> [Float] {
        let count = cfg.hiddenSize
        var result = [Float](repeating: 0, count: count)
        let ptr = hidden.contents().bindMemory(to: Float16.self, capacity: count)
        for i in 0..<count { result[i] = Float(ptr[i]) }
        return result
    }'''
new_rh = '''    /// Copy the last decode hidden state as Float32 for MTP draft generation.
    public func copyLastHiddenState() -> [Float] {
        let count = cfg.hiddenSize
        var result = [Float](repeating: 0, count: count)
        let ptr = hidden.contents().bindMemory(to: Float16.self, capacity: count)
        for i in 0..<count { result[i] = Float(ptr[i]) }
        return result
    }

    /// Copy one row of the most recent verify chunk's hidden scratch as
    /// Float32. Row `i` is the backbone hidden at chunk position `i` — used
    /// by the chain dump as the feature target for draft `i-1`'s ĥ.
    public func copyVerifyRowHidden(_ row: Int) -> [Float] {
        guard let scratch = prefillScratch else { return [] }
        let rowBytes = cfg.hiddenSize * MemoryLayout<Float16>.stride
        guard row >= 0, (row + 1) * rowBytes <= scratch.hidden.length else { return [] }
        var out = [Float](repeating: 0, count: cfg.hiddenSize)
        let src = scratch.hidden.contents()
            .advanced(by: row * rowBytes)
            .assumingMemoryBound(to: Float16.self)
        for i in 0..<cfg.hiddenSize { out[i] = Float(src[i]) }
        return out
    }'''
patch(R, old_rh, new_rh)

# ---------- 5. MTPCompletion: chain-dump call after verify ----------
old_mtp = '''        guard predictions.count == span.count else {
            throw PrefillError.unsupportedPrefillSeed(
                "verifyBatch returned \\(predictions.count) rows for \\(span.count) tokens")
        }
        let verifyEnd = Date()'''
new_mtp = '''        guard predictions.count == span.count else {
            throw PrefillError.unsupportedPrefillSeed(
                "verifyBatch returned \\(predictions.count) rows for \\(span.count) tokens")
        }
        let verifyEnd = Date()

        // EAGLE-2 rollout supervision: the verify chunk's row hiddens are the
        // backbone's hidden after each drafted prefix, and `predictions[i]` is
        // the target's greedy after `span[0...i]` — together they give the
        // (CE token, feature MSE) pair for every draft position.
        if ProcessInfo.processInfo.environment["TURBO_FIELDFARE_MTP_CHAIN_DUMP"] != nil {
            var rowHiddens: [[Float]] = []
            rowHiddens.reserveCapacity(span.count + 1)
            for r in 0...span.count { rowHiddens.append(producer.copyVerifyRowHidden(r)) }
            let draftEmbs: [[Float]?] = drafts.map { producer.copyTokenEmbedding($0) }
            TrainingDataDump.appendChain(
                bridge: bridge, ctx: Int32(currentToken),
                drafts: drafts,
                predictions: predictions.map { Int32(bitPattern: $0) },
                rowHiddens: rowHiddens,
                draftEmbeddings: draftEmbs)
        }'''
patch(M, old_mtp, new_mtp)

# ---------- 6. RawCompletion: gate the unconditional bridge build ----------
old_rc = '''        if let fr = fusedRunner {
            TrainingDataDump.append(
                bridge: fr.makeBridgeSnapshot(currentToken: tokenID),
                ctx: tokenID,
                next: Int32(bitPattern: fr.lastGreedyToken),
                drafts: [])
        }'''
new_rc = '''        if let fr = fusedRunner,
           ProcessInfo.processInfo.environment["TURBO_FIELDFARE_MTP_HIDDEN_DUMP"] != nil {
            TrainingDataDump.append(
                bridge: fr.makeBridgeSnapshot(currentToken: tokenID),
                ctx: tokenID,
                next: Int32(bitPattern: fr.lastGreedyToken),
                drafts: [])
        }'''
patch(C, old_rc, new_rc)

print("ALL PATCHES APPLIED")
