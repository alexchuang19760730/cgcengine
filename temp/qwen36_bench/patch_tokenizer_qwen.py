#!/usr/bin/env python3
"""Add a .qwen36 compatibility mode to GFTokenizer so the Qwen3.6 model can be
driven by the shared runRawCompletion loop:

- special-token guards fail open (bos==nil -> eos; pad==nil -> eos; missing
  Gemma tool/channel markers -> those stop IDs are dropped)
- applyChatTemplate switches to the Qwen <|im_start|>/<|im_end|> framing
- encode() never prepends a fake BOS (Qwen has none)

Gemma 4 behaviour is unchanged (the default compatibility is .gemma4).
"""
path = "/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/turbo-fieldfare/Sources/TurboFieldfare/Tokenization/Tokenizer.swift"
src = open(path).read()

# 1. Compatibility enum + property
old1 = """    @usableFromInline
    let tokenizer: any Tokenizer

    public static func load() async throws -> GFTokenizer {"""
new1 = """    /// Tokenizer dialect: Gemma 4 (strict special tokens + Gemma chat
    /// framing) or Qwen3.6 (no BOS, <|im_start|>/<|im_end|> framing).
    public enum Compatibility: Sendable {
        case gemma4
        case qwen36
    }

    public let compatibility: Compatibility

    @usableFromInline
    let tokenizer: any Tokenizer

    public static func load() async throws -> GFTokenizer {"""
assert old1 in src, "tokenizer property anchor not found"
src = src.replace(old1, new1)

# 2. init: add compatibility param + fail-open guards
old2 = """    public init(tokenizer: any Tokenizer) throws {
        self.tokenizer = tokenizer

        guard let bos = tokenizer.bosTokenId else {
            throw GFTokenizerError.missingSpecialToken("<bos>")
        }
        guard let eos = tokenizer.eosTokenId else {
            throw GFTokenizerError.missingSpecialToken("<eos>")
        }
        guard let pad = tokenizer.convertTokenToId("<pad>") else {
            throw GFTokenizerError.missingSpecialToken("<pad>")
        }
        guard let eot = tokenizer.convertTokenToId("<turn|>") else {
            throw GFTokenizerError.missingSpecialToken("<turn|>")
        }
        guard let toolResponse = tokenizer.convertTokenToId("<|tool_response>") else {
            throw GFTokenizerError.missingSpecialToken("<|tool_response>")
        }
        guard let toolCallStart = tokenizer.convertTokenToId("<|tool_call>"),
              let toolCallEnd = tokenizer.convertTokenToId("<tool_call|>"),
              let toolResponseEnd = tokenizer.convertTokenToId("<tool_response|>"),
              let channelStart = tokenizer.convertTokenToId("<|channel>"),
              let channelEnd = tokenizer.convertTokenToId("<channel|>") else {
            throw GFTokenizerError.missingSpecialToken("Gemma tool/channel markers")
        }

        self.bosID = Int32(bos)
        self.eosID = Int32(eos)
        self.padID = Int32(pad)
        self.endOfTurnID = Int32(eot)
        self.toolCallStartID = Int32(toolCallStart)
        self.toolCallEndID = Int32(toolCallEnd)
        self.toolResponseID = Int32(toolResponse)
        self.toolResponseEndID = Int32(toolResponseEnd)
        self.channelStartID = Int32(channelStart)
        self.channelEndID = Int32(channelEnd)
        self.stopTokenIDs = [self.eosID, self.endOfTurnID, self.toolResponseID]
        self.vocabSize = 262_144
    }"""
new2 = """    public init(tokenizer: any Tokenizer,
                compatibility: Compatibility = .gemma4) throws {
        self.tokenizer = tokenizer
        self.compatibility = compatibility

        let bos = tokenizer.bosTokenId
        let eos = tokenizer.eosTokenId
        guard let eos else {
            throw GFTokenizerError.missingSpecialToken("<eos>")
        }
        let bosFallback = compatibility == .qwen36 ? Int32(eos) : nil
        let pad: Int32?
        if let p = tokenizer.convertTokenToId("<pad>") {
            pad = Int32(p)
        } else if compatibility == .qwen36 {
            pad = Int32(eos)
        } else {
            throw GFTokenizerError.missingSpecialToken("<pad>")
        }
        let eot: Int32?
        if let v = tokenizer.convertTokenToId("<turn|>") {
            eot = Int32(v)
        } else if compatibility == .qwen36 {
            eot = Int32(eos)
        } else {
            throw GFTokenizerError.missingSpecialToken("<turn|>")
        }
        let toolResponse = tokenizer.convertTokenToId("<|tool_response>").map(Int32.init)
        let toolCallStart = tokenizer.convertTokenToId("<|tool_call>").map(Int32.init)
        let toolCallEnd = tokenizer.convertTokenToId("<tool_call|>").map(Int32.init)
        let toolResponseEnd = tokenizer.convertTokenToId("<tool_response|>").map(Int32.init)
        let channelStart = tokenizer.convertTokenToId("<|channel>").map(Int32.init)
        let channelEnd = tokenizer.convertTokenToId("<channel|>").map(Int32.init)
        if compatibility == .gemma4 {
            guard let toolResponse, let toolCallStart, let toolCallEnd,
                  let toolResponseEnd, let channelStart, let channelEnd else {
                throw GFTokenizerError.missingSpecialToken("Gemma tool/channel markers")
            }
        }

        if let bos {
            self.bosID = Int32(bos)
        } else if let bosFallback {
            self.bosID = bosFallback
        } else {
            throw GFTokenizerError.missingSpecialToken("<bos>")
        }
        self.eosID = Int32(eos)
        self.padID = pad ?? Int32(eos)
        self.endOfTurnID = eot ?? Int32(eos)
        self.toolCallStartID = toolCallStart ?? -1
        self.toolCallEndID = toolCallEnd ?? -1
        self.toolResponseID = toolResponse ?? -1
        self.toolResponseEndID = toolResponseEnd ?? -1
        self.channelStartID = channelStart ?? -1
        self.channelEndID = channelEnd ?? -1
        var stops: Set<Int32> = [self.eosID]
        if self.endOfTurnID >= 0 { stops.insert(self.endOfTurnID) }
        if self.toolResponseID >= 0 { stops.insert(self.toolResponseID) }
        self.stopTokenIDs = stops
        self.vocabSize = 262_144
    }"""
assert old2 in src, "init block not found"
src = src.replace(old2, new2)

# 3. applyChatTemplate: dispatch on compatibility
old3 = """    public func applyChatTemplate(_ messages: [Message]) throws -> String {
        var s = Self.bosMark
        for (index, message) in messages.enumerated() {
            guard let rawContent = message.content else {
                throw GFTokenizerError.invalidChatTemplate("text-only messages require content")
            }
            let content = rawContent.trimmingCharacters(in: .whitespacesAndNewlines)
            if message.role == .system && index != 0 {
                throw GFTokenizerError.invalidChatTemplate("system message must be first")
            }
            let role = message.role == .assistant ? "model" : message.role.rawValue
            s += Self.turnOpen + role + "\\n" + content + Self.turnClose + "\\n"
        }
        s += Self.turnOpen + "model\\n<|channel>thought\\n<channel|>"
        return s
    }"""
new3 = """    public func applyChatTemplate(_ messages: [Message]) throws -> String {
        switch compatibility {
        case .qwen36:
            var s = ""
            for (index, message) in messages.enumerated() {
                guard let rawContent = message.content else {
                    throw GFTokenizerError.invalidChatTemplate("text-only messages require content")
                }
                let content = rawContent.trimmingCharacters(in: .whitespacesAndNewlines)
                if message.role == .system && index != 0 {
                    throw GFTokenizerError.invalidChatTemplate("system message must be first")
                }
                let role = message.role == .assistant ? "assistant" : message.role.rawValue
                s += "<|im_start|>" + role + "\\n" + content + "<|im_end|>\\n"
            }
            s += "<|im_start|>assistant\\n"
            return s
        case .gemma4:
            var s = Self.bosMark
            for (index, message) in messages.enumerated() {
                guard let rawContent = message.content else {
                    throw GFTokenizerError.invalidChatTemplate("text-only messages require content")
                }
                let content = rawContent.trimmingCharacters(in: .whitespacesAndNewlines)
                if message.role == .system && index != 0 {
                    throw GFTokenizerError.invalidChatTemplate("system message must be first")
                }
                let role = message.role == .assistant ? "model" : message.role.rawValue
                s += Self.turnOpen + role + "\\n" + content + Self.turnClose + "\\n"
            }
            s += Self.turnOpen + "model\\n<|channel>thought\\n<channel|>"
            return s
        }
    }"""
assert old3 in src, "applyChatTemplate not found"
src = src.replace(old3, new3)

# 4. encode(): qwen36 never prepends BOS
old4 = """    public func encode(_ text: String, addBOS: Bool = true) -> [Int32] {
        let base = tokenizer.encode(text: text, addSpecialTokens: false).map(Int32.init)
        return addBOS ? [bosID] + base : base
    }"""
new4 = """    public func encode(_ text: String, addBOS: Bool = true) -> [Int32] {
        let base = tokenizer.encode(text: text, addSpecialTokens: false).map(Int32.init)
        if compatibility == .qwen36 { return base }
        return addBOS ? [bosID] + base : base
    }"""
assert old4 in src, "encode not found"
src = src.replace(old4, new4)

open(path, "w").write(src)
print("Tokenizer.swift: qwen36 compatibility mode added")
