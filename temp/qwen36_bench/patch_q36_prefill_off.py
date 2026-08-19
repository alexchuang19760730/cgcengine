#!/usr/bin/env python3
"""Qwen36 branch: use .off prefill (per-token produce) instead of the
chunked default. Qwen36ForwardRunner does not conform ChunkedPrefillRunner,
so the shared runRawCompletion loop throws otherwise. gemma4 path untouched."""
path = "/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/turbo-fieldfare/Sources/TurboFieldfareCLI/Run.swift"
src = open(path).read()

old = """            let qStats = try await runRawCompletion(
                producer: qRunner,
                tokenizer: tokenizer,
                promptIds: promptIds,
                config: config,
                context: context,
                scratch: scratch,
                prefillConfig: runtime.prefillConfig) { progress in"""

new = """            let qStats = try await runRawCompletion(
                producer: qRunner,
                tokenizer: tokenizer,
                promptIds: promptIds,
                config: config,
                context: context,
                scratch: scratch,
                // Qwen36ForwardRunner drives every token through produce()
                // (no chunked prefill path yet) — per-token .off mode.
                prefillConfig: .off) { progress in"""

assert old in src, "qwen36 runRawCompletion block not found"
src = src.replace(old, new, 1)
open(path, "w").write(src)
print("Run.swift: qwen36 branch prefill -> .off")
