#!/usr/bin/env python3
"""Fix copyVerifyRowHidden: scratch.hidden is .storageModePrivate — blit to shared staging first."""

def patch(path, old, new, count=1):
    src = open(path).read()
    if old not in src:
        raise SystemExit(f"OLD NOT FOUND in {path}:\n{old[:200]!r}")
    src = src.replace(old, new, count)
    open(path, "w").write(src)
    print(f"patched {path}")

R = "turbo-fieldfare-github-official/Sources/TurboFieldfare/Runtime/Inference/RealForwardRunner.swift"
M = "turbo-fieldfare-github-official/Sources/TurboFieldfare/Runtime/Generation/MTPCompletion.swift"

# 1. replace the broken single-row contents() method with a blit-based bulk method
patch(R, '''    /// Copy one row of the most recent verify chunk's hidden scratch as
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
    }''',
'''    /// Staging buffer for `copyVerifyRowHiddens` (shared so CPU can read it;
    /// the prefill hidden scratch itself is .storageModePrivate).
    private var rowStaging: MTLBuffer?

    /// Copy rows `start..<start+count` of the last verify chunk's hidden
    /// scratch as Float32, using one blit command buffer into a shared staging
    /// buffer (GPU-private memory is not CPU-readable). Row `i` of the chunk
    /// is the backbone hidden at chunk position `i` — used by the chain dump
    /// as the feature target for draft `i-1`'s predicted hidden.
    public func copyVerifyRowHiddens(start: Int, count: Int) -> [[Float]] {
        guard let scratch = prefillScratch, count > 0 else { return [] }
        let rowBytes = cfg.hiddenSize * MemoryLayout<Float16>.stride
        guard start >= 0, (start + count) * rowBytes <= scratch.hidden.length else { return [] }
        if rowStaging == nil {
            rowStaging = ctx.device.makeBuffer(length: count * rowBytes,
                                               options: .storageModeShared)
            rowStaging?.label = "assistant.rowStaging"
        }
        guard let staging = rowStaging else { return [] }
        runSync { cb in
            if let blit = cb.makeBlitCommandEncoder() {
                for i in 0..<count {
                    blit.copy(from: scratch.hidden,
                              sourceOffset: (start + i) * rowBytes,
                              to: staging,
                              destinationOffset: i * rowBytes,
                              size: rowBytes)
                }
                blit.endEncoding()
            }
        }
        var rows: [[Float]] = []
        rows.reserveCapacity(count)
        let base = staging.contents().assumingMemoryBound(to: Float16.self)
        for i in 0..<count {
            var out = [Float](repeating: 0, count: cfg.hiddenSize)
            let src = base.advanced(by: i * cfg.hiddenSize)
            for j in 0..<cfg.hiddenSize { out[j] = Float(src[j]) }
            rows.append(out)
        }
        return rows
    }''')

# 2. update the MTPCompletion call site
patch(M, '''            var rowHiddens: [[Float]] = []
            rowHiddens.reserveCapacity(span.count + 1)
            for r in 0...span.count { rowHiddens.append(producer.copyVerifyRowHidden(r)) }
            let draftEmbs: [[Float]?] = drafts.map { producer.copyTokenEmbedding($0) }''',
'''            let rowHiddens = producer.copyVerifyRowHiddens(start: 0, count: span.count + 1)
            let draftEmbs: [[Float]?] = drafts.map { producer.copyTokenEmbedding($0) }''')

print("FIX APPLIED")
