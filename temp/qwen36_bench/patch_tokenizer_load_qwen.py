#!/usr/bin/env python3
"""1. GFTokenizer: add public static loadQwen36(from:) that loads the underlying
   tokenizer from a folder with .qwen36 compatibility (reuses the internal
   coordinator-free AutoTokenizer path).
2. Run.swift: use loadQwen36 for the qwen36 arch branch (no AutoTokenizer
   import needed in the CLI)."""
# ---- 1. GFTokenizer.loadQwen36 ----
p = "/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/turbo-fieldfare/Sources/TurboFieldfare/Tokenization/Tokenizer.swift"
src = open(p).read()

old = """    public static func load(from folder: URL) async throws -> GFTokenizer {
        try await GFTokenizerLoadCoordinator.shared.load(.local(folder.standardizedFileURL.path))
    }"""
new = """    public static func load(from folder: URL) async throws -> GFTokenizer {
        try await GFTokenizerLoadCoordinator.shared.load(.local(folder.standardizedFileURL.path))
    }

    /// Load a Qwen3.6 tokenizer (no BOS, <|im_start|>/<|im_end|> framing) from
    /// a local folder. `AutoTokenizer.from(modelFolder:)` is not exposed to the
    /// CLI target, so this lives here next to the Gemma loader.
    public static func loadQwen36(from folder: URL) async throws -> GFTokenizer {
        let underlying = try await AutoTokenizer.from(modelFolder: folder)
        return try GFTokenizer(tokenizer: underlying, compatibility: .qwen36)
    }"""
assert old in src, "load(from:) anchor not found"
src = src.replace(old, new)
open(p, "w").write(src)
print("Tokenizer.swift: loadQwen36 added")

# ---- 2. Run.swift: use loadQwen36 ----
p = "/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/turbo-fieldfare/Sources/TurboFieldfareCLI/Run.swift"
src = open(p).read()

old2 = """        let tokenizer: GFTokenizer
        if detectedArch == .qwen36_35B_A3B {
            if let folder = GFTokenizer.tokenizerFolder(forModelDirectory: modelURL) {
                let underlying = try await AutoTokenizer.from(modelFolder: folder)
                tokenizer = try GFTokenizer(tokenizer: underlying, compatibility: .qwen36)
            } else {
                tokenizer = try await GFTokenizer.load(forModelDirectory: modelURL)
            }
        } else {
            tokenizer = try await GFTokenizer.load(forModelDirectory: modelURL)
        }"""
new2 = """        let tokenizer: GFTokenizer
        if detectedArch == .qwen36_35B_A3B,
           let folder = GFTokenizer.tokenizerFolder(forModelDirectory: modelURL) {
            tokenizer = try await GFTokenizer.loadQwen36(from: folder)
        } else {
            tokenizer = try await GFTokenizer.load(forModelDirectory: modelURL)
        }"""
assert old2 in src, "Run.swift tokenizer block not found"
src = src.replace(old2, new2)
open(p, "w").write(src)
print("Run.swift: uses loadQwen36")
