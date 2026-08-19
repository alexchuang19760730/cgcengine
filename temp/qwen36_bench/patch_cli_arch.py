#!/usr/bin/env python3
"""Wire arch auto-detection into the CLI:
1. Read manifest.json from the model dir to decide gemma4 vs qwen36.
2. Model.load expecting: pick .gemma4_26B_A4B or .qwen36_35B_A3B.
3. Qwen36 branch: run the full decode loop through Qwen36ForwardRunner via
   runRawCompletion (any LogitProducer) and return — gemma4 path untouched.
"""
path = "/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/turbo-fieldfare/Sources/TurboFieldfareCLI/Run.swift"
src = open(path).read()

# 1. helper to detect arch from manifest (insert before `public func run`)
old0 = "public func run(args: Args,"
new0 = """/// Detect the model architecture from manifest.json.
/// Returns nil if the manifest cannot be read (caller falls back to gemma4).
private func detectArch(_ modelURL: URL) -> ArchConfig? {
    let manifestURL = modelURL.appendingPathComponent("manifest.json")
    guard let data = try? Data(contentsOf: manifestURL) else { return nil }
    guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let arch = obj["arch"] as? [String: Any] else { return nil }
    let numLayers = arch["numLayers"] as? Int
    let hidden = arch["hiddenSize"] as? Int
    if numLayers == 40, hidden == 2048 {
        return .qwen36_35B_A3B
    }
    if numLayers == 30, hidden == 2816 {
        return .gemma4_26B_A4B
    }
    return nil
}

public func run(args: Args,"""
assert old0 in src, "run() anchor not found"
src = src.replace(old0, new0)

# 2. Model.load: pass detected arch + Qwen36 inline branch
old1 = """        let model = try Model.load(
            directoryURL: modelURL,
            device: context.device,
            streamingMode: .pread(slotCount: runtime.expertCacheSlots),
            expertCachePolicy: runtime.modelExpertCachePolicy,
            prefillExpertBoundedParallelMissReadWorkers: readWorkers,
            integrityPolicy: args.trustReceipt ? .sizeCheckTrustedReceipt
                                               : .fullSha256)
        let runner = try RealForwardRunner(
            model: model,
            context: context,
            maxContext: args.maxContext,
            runtimeConfiguration: runtime)"""
new1 = """        let detectedArch = detectArch(modelURL)
        let model = try Model.load(
            directoryURL: modelURL,
            device: context.device,
            expecting: detectedArch ?? .gemma4_26B_A4B,
            streamingMode: .pread(slotCount: runtime.expertCacheSlots),
            expertCachePolicy: runtime.modelExpertCachePolicy,
            prefillExpertBoundedParallelMissReadWorkers: readWorkers,
            integrityPolicy: args.trustReceipt ? .sizeCheckTrustedReceipt
                                               : .fullSha256)

        // Qwen3.6 hybrid path: driven by the shared runRawCompletion loop via
        // the LogitProducer protocol. Expert streaming knobs (hot pool /
        // EXPERT_SLOTS / READ_WORKERS) are NOT consumed by this runner yet —
        // Qwen36MoERunner loads each layer file wholesale (all-resident).
        if detectedArch == .qwen36_35B_A3B {
            let qRunner = try Qwen36ForwardRunner(
                context: context,
                model: model,
                maxSeq: args.maxContext)
            let scratch = try RawCompletionScratch(context: context,
                                                   vocab: model.config.vocabSize)
            let qStats = try await runRawCompletion(
                producer: qRunner,
                tokenizer: tokenizer,
                promptIds: promptIds,
                config: config,
                context: context,
                scratch: scratch,
                prefillConfig: runtime.prefillConfig) { progress in
                switch progress {
                case .prefill:
                    break
                case .token(_, _, let delta):
                    if !delta.isEmpty { stdout.write(Data(delta.utf8)) }
                case .tail(let tail):
                    stdout.write(Data(tail.utf8))
                }
            }
            if !args.quiet {
                let tps = qStats.decodeSeconds > 0
                    ? Double(qStats.newTokens) / qStats.decodeSeconds : 0
                let footer = "\\n[stop=\\(String(describing: qStats.reason)) prefill=\\(qStats.prefillTokens)tok new=\\(qStats.newTokens)tok ttft=\\(String(format: \\"%.2f\\", qStats.prefillSeconds))s decode=\\(String(format: \\"%.2f\\", qStats.decodeSeconds))s tok/s=\\(String(format: \\"%.3f\\", tps))]\\n"
                stderr.write(Data(footer.utf8))
            }
            return RunResult(exitCode: 0)
        }

        let runner = try RealForwardRunner(
            model: model,
            context: context,
            maxContext: args.maxContext,
            runtimeConfiguration: runtime)"""
assert old1 in src, "Model.load block not found"
src = src.replace(old1, new1)

open(path, "w").write(src)
print("Run.swift: arch detection + Qwen36 inline branch wired")
