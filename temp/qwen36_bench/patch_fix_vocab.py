#!/usr/bin/env python3
"""Fix GFTokenizer.vocabSize: read from the underlying tokenizer instead of
hardcoding 262_144 (gemma4's vocab). Qwen3.6 is 248320."""
path = "/Users/alexchuang/Documents/flashkv0516/prime-agent-worktrees/turbo-fieldfare/Sources/TurboFieldfare/Tokenization/Tokenizer.swift"
src = open(path).read()

# The qwen36 branch and gemma4 branch both end with the same vocabSize line
old = """        var stops: Set<Int32> = [self.eosID]
        if self.endOfTurnID >= 0 { stops.insert(self.endOfTurnID) }
        if self.toolResponseID >= 0 { stops.insert(self.toolResponseID) }
        self.stopTokenIDs = stops
        self.vocabSize = 262_144
    }"""
new = """        var stops: Set<Int32> = [self.eosID]
        if self.endOfTurnID >= 0 { stops.insert(self.endOfTurnID) }
        if self.toolResponseID >= 0 { stops.insert(self.toolResponseID) }
        self.stopTokenIDs = stops
        self.vocabSize = tokenizer.vocabularySize
    }"""
assert old in src, "qwen36 vocabSize line not found"
src = src.replace(old, new)

# gemma4 branch (the original hardcode)
old2 = """        self.stopTokenIDs = [self.eosID, self.endOfTurnID, self.toolResponseID]
        self.vocabSize = 262_144
    }"""
new2 = """        self.stopTokenIDs = [self.eosID, self.endOfTurnID, self.toolResponseID]
        self.vocabSize = tokenizer.vocabularySize
    }"""
assert old2 in src, "gemma4 vocabSize line not found"
src = src.replace(old2, new2)

open(path, "w").write(src)
print("Tokenizer.swift: vocabSize from tokenizer.vocabularySize")
