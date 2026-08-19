#!/usr/bin/env python3
"""Apply remaining Swift patches: copyVerifyRowHidden, MTPCompletion chain call, RawCompletion gate."""

def patch(path, old, new, count=1, must=True):
    src = open(path).read()
    if old not in src:
        if must:
            raise SystemExit(f"OLD NOT FOUND in {path}:\n{old[:200]!r}")
        print(f"skip (absent): {path}")
        return
    src = src.replace(old, new, count)
    open(path, "w").write(src)
    print(f"patched {path}")

R = "turbo-fieldfare-github-official/Sources/TurboFieldfare/Runtime/Inference/RealForwardRunner.swift"
M = "turbo-fieldfare-github-official/Sources/TurboFieldfare/Runtime/Generation/MTPCompletion.swift"
C = "turbo-fieldfare-github-official/Sources/TurboFieldfare/Runtime/Generation/RawCompletion.swift"

# 4. copyVerifyRowHidden
patch(R, '''    /// Copy the last decode hidden state as Float32 for MTP draft generation.
    public func copyLastHiddenState() -> [Float] {
        let count = cfg.hiddenSize
        var result = [Float](repeating: 0, count: count)
        let ptr = hidden.contents().bindMemory(to: Float16.self, capacity: count)
        for i in 0..<count { result[i] = Float(ptr[i]) }
        return result
    }''',
'''    /// Copy the last decode hidden state as Float32 for MTP draft generation.
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
    }''')

# 5. MTPCompletion chain call
patch(M, '''        guard predictions.count == span.count else {
            throw PrefillError.unsupportedPrefillSeed(
                "verifyBatch returned \\(predictions.count) rows for \\(span.count) tokens")
        }
        let verifyEnd = Date()''',
'''        guard predictions.count == span.count else {
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
        }''')

# 6. RawCompletion gate
patch(C, '''        if let fr = fusedRunner {
            TrainingDataDump.append(
                bridge: fr.makeBridgeSnapshot(currentToken: tokenID),
                ctx: tokenID,
                next: Int32(bitPattern: fr.lastGreedyToken),
                drafts: [])
        }''',
'''        if let fr = fusedRunner,
           ProcessInfo.processInfo.environment["TURBO_FIELDFARE_MTP_HIDDEN_DUMP"] != nil {
            TrainingDataDump.append(
                bridge: fr.makeBridgeSnapshot(currentToken: tokenID),
                ctx: tokenID,
                next: Int32(bitPattern: fr.lastGreedyToken),
                drafts: [])
        }''')

print("PATCHES 4-6 APPLIED")
