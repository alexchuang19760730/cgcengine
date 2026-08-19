#!/usr/bin/env python3
"""Patch DeltaNetRunner: add layerIndex so it can serve layers other than 0.
The weight() helper re-scopes 'layers.0.' -> 'layers.<layerIndex>.' at call
time; call sites keep the template string so no other edits are needed."""
path = "/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/turbo-fieldfare/Sources/TurboFieldfare/Kernels/Attention/DeltaNetRunner.swift"
src = open(path).read()

# 1. init signature + assignment
old1 = """    public init(context: MetalContext, model: Model) throws {
        self.device = context.device
        self.queue = context.queue
        self.config = Config()"""
new1 = """    public init(context: MetalContext, model: Model, layerIndex: Int = 0) throws {
        self.device = context.device
        self.queue = context.queue
        self.config = Config()
        self.layerIndex = layerIndex"""
assert old1 in src, "init block not found"
src = src.replace(old1, new1)

# 2. stored property
old2 = """    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let config: Config"""
new2 = """    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let config: Config
    private let layerIndex: Int"""
assert old2 in src, "stored prop anchor not found"
src = src.replace(old2, new2)

# 3. weight() helper: scope layer names
old3 = """    private func weight(_ model: Model, _ name: String) throws -> TensorView {
        try model.resident(name: name)
    }"""
new3 = """    private func weight(_ model: Model, _ name: String) throws -> TensorView {
        let scoped = name.replacingOccurrences(
            of: "model.language_model.layers.0.",
            with: "model.language_model.layers.\\(layerIndex).")
        return try model.resident(name: scoped)
    }"""
assert old3 in src, "weight helper not found"
src = src.replace(old3, new3)

open(path, "w").write(src)
print("patched OK. literal 'layers.0' occurrences (template+comment):", src.count("layers.0"))
