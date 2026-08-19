#!/usr/bin/env python3
"""Append appendChain method to TrainingDataDump enum (final patch)."""

path = "turbo-fieldfare-github-official/Sources/TurboFieldfare/Runtime/Generation/TrainingDataDump.swift"
src = open(path).read()

anchor = '''            rec.append(Data(bytes: &startSlot, count: 4))
        }
    }
}'''
assert anchor in src, "anchor not found"
assert src.count(anchor) == 1, f"anchor ambiguous: {src.count(anchor)}"

new_method = '''            rec.append(Data(bytes: &startSlot, count: 4))
        }
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
    /// (the feature target for the head's predicted hidden h_i). Row hiddens
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

src = src.replace(anchor, new_method, 1)
open(path, "w").write(src)
print("appendChain appended OK")
