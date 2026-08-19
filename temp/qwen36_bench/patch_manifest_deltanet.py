#!/usr/bin/env python3
"""Add optional deltanetLayers field to ManifestArch (Qwen3.6 provides it,
Gemma-4 manifest has no such key -> nil)."""
path = "/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/turbo-fieldfare/Sources/TurboFieldfare/Infrastructure/ModelIO/ManifestReader.swift"
src = open(path).read()

# 1. field declaration after fullAttentionLayerMask
old1 = """    public let fullAttentionLayerMask: [Int]?

    private enum CodingKeys: String, CodingKey {"""
new1 = """    public let fullAttentionLayerMask: [Int]?
    /// Qwen3.6 hybrid: layer indices that are DeltaNet (the rest are GatedAttn).
    /// Gemma-4 manifest has no such key -> nil.
    public let deltanetLayers: [Int]?

    private enum CodingKeys: String, CodingKey {"""
assert old1 in src, "field anchor not found"
src = src.replace(old1, new1)

# 2. CodingKeys
old2 = """        case tieWordEmbeddings, attentionKEqV, hiddenActivation
        case fullAttentionLayerMask
    }"""
new2 = """        case tieWordEmbeddings, attentionKEqV, hiddenActivation
        case fullAttentionLayerMask, deltanetLayers
    }"""
assert old2 in src, "codingkeys anchor not found"
src = src.replace(old2, new2)

# 3. decode
old3 = """        fullAttentionLayerMask = try c.decodeIfPresent([Int].self, forKey: .fullAttentionLayerMask)
    }
}"""
new3 = """        fullAttentionLayerMask = try c.decodeIfPresent([Int].self, forKey: .fullAttentionLayerMask)
        deltanetLayers = try c.decodeIfPresent([Int].self, forKey: .deltanetLayers)
    }
}"""
assert old3 in src, "decode anchor not found"
src = src.replace(old3, new3)

open(path, "w").write(src)
print("ManifestArch: deltanetLayers added")
