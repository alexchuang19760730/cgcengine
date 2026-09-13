# CGC Devserver Development Whitepaper (2026-09-04)

## 1. Executive Summary

This document records the current development snapshot of the `devserver` branch for the Nail/Qwen3.6 MTP server path. The work in this snapshot focuses on two goals:

1. recover `llama-server` MTP runtime behavior to near CLI parity;
2. isolate and debug the semantic drift observed on the server chat path, especially the mismatch between native Qwen ChatML thinking behavior and server-side prompt / parser assembly.

At the runtime layer, the branch has already recovered the major MTP acceptance regression by introducing an opt-in CLI-parity initialization path. At the semantic layer, the investigation has converged from generic "answer channel" speculation to a narrower native thinking-contract alignment problem.

This snapshot is valuable as a development checkpoint, but it is **not yet a release candidate**. The short-answer (`qa-zh`) path is still semantically unstable, and the current QA-only override is intentionally scoped as a debugging aid rather than a production-ready default.

## 2. Problem Statement

The baseline problem was a large gap between CLI and server behavior for the same Nail/Qwen3.6 MTP carrier:

- CLI speculative path: around `25+ tok/s`, with healthy draft acceptance.
- Server MTP path: initially near `2-3 tok/s`, with draft acceptance collapsing toward `0%`.

After the runtime path was repaired, a second class of issues remained:

- longform responses could recover text but still show semantic oddities;
- short-answer QA frequently produced empty content, structural tokens, or recursive thinking loops;
- native GGUF template behavior did not fully match the assumptions made by the server chat path.

The central question became:

> How do we align the native Qwen ChatML thinking contract with the server chat path without introducing side effects to longform traffic?

## 3. Design Principles

This branch follows a few hard constraints:

1. **Semantic correctness before speed**  
   Performance gains only count once the server returns stable, clean outputs.

2. **Controlled experiments**  
   Each cut changes one variable at a time: runtime init, parser boundary, generation prompt, or profile-specific scaffold.

3. **Profile-aware behavior**  
   QA and longform have materially different failure modes. A fix that helps short answers must not silently poison longform.

4. **No hidden global defaults**  
   If a workaround is only valid for QA, it must be explicitly gated by profile-level context rather than applied globally.

## 4. Major Changes in This Snapshot

### 4.1 CLI-parity MTP initialization

Files:

- `src/llama.cpp/tools/server/server-context.cpp`
- `scripts/run_server.sh`

This snapshot adds an opt-in server initialization path that mirrors the effective order used by `llama-speculative-simple`:

- initialize speculative decoding context earlier under `CGC_MTP_CLI_PARITY=1`;
- optionally skip the server-side `seq_rm` probe;
- optionally move warmup after speculative topology is established;
- expose the knobs in `scripts/run_server.sh`.

This change is the main reason the server runtime recovered from the earlier `0%` draft-acceptance failure mode and returned to a useful MTP operating point.

### 4.2 Minimal Nail template cleanup

File:

- `src/llama.cpp/models/templates/Nail-Qwen3.6-Minimal-Chat.jinja`

The template was tightened so that:

- assistant turns can preserve explicit reasoning blocks when they are real;
- empty think scaffolds are no longer blindly forced into every continuation path;
- profile-level kwargs such as `disable_think_scaffold` can be passed through cleanly.

This is part of separating "real reasoning output" from "server-injected scaffold".

### 4.3 Specialized native chat handling for Nail minimal template

File:

- `src/llama.cpp/common/chat.cpp`

This snapshot adds a specialized native parser path for the Nail minimal template and expands debug visibility around:

- generated prompt construction;
- continuation assembly;
- final prompt tail sent into parsing.

The intent is not to add broad special cases for their own sake, but to create a controlled place where the native contract can be observed and adjusted with minimal blast radius.

### 4.4 Semantic debug instrumentation

Files:

- `src/llama.cpp/common/chat.cpp`
- `src/llama.cpp/tools/server/server-task.cpp`

The branch now emits structured debug events for:

- generation prompt construction;
- partial chat-state updates during streaming;
- final OpenAI-compatible response assembly.

This instrumentation was critical in ruling out several false leads. In particular, it let us separate:

- what the model actually generated,
- what the parser interpreted as reasoning,
- what the final response assembly exposed as `content` vs `reasoning_content`.

### 4.5 QA-only override for native assistant prefill

Files:

- `src/llama.cpp/common/chat.cpp`
- `scripts/check/replay_server_profile.py`
- `scripts/run_server.sh`

The branch now supports a **profile-aware** short-answer override gated by:

```json
{"disable_think_scaffold": true}
```

The reason for this override is specific to QA:

- native thinking-prefill can trap short-answer requests in a structural or recursive-think path;
- longform requests should remain on the native path unless explicitly overridden.

This logic is intentionally gated through `chat_template_kwargs` so that QA and longform can be tested side by side on the same server without global side effects.

## 5. Investigation Findings

### 5.1 What was ruled out

The following hypotheses were tested and rejected as primary fixes:

- generic short-answer prefixes;
- dynamic question-derived continuations;
- "answer slot" strings such as `答案：`;
- raw `<|output|>` channel speculation;
- parser-only explanations for the recursive think loop.

These experiments helped establish that the core issue was not just "finding the right magic answer prefix".

### 5.2 Native contract mismatch was real, but incomplete

Direct GGUF inspection confirmed that the model's native template is ChatML-style, not the older `<|assistant|>/<|user|>/<|output|>` contract. That mattered because:

- some server-side assumptions were based on non-native control tokens;
- native Qwen-style generation behavior enters the assistant turn differently.

However, simply switching back to the native template did **not** automatically fix QA behavior. The server could still fall into recursive or malformed thinking output, which meant the root cause was deeper than template selection alone.

### 5.3 `generation_prompt` and actual model prompt are different levers

One key finding from this snapshot is that the server uses:

- `generation_prompt` for parser alignment,
- `prompt` as the actual text fed to the model.

Changing only `generation_prompt` can therefore change the parser/assembly outcome without changing what the model actually sees. That is why an earlier assistant-only experiment shifted the failure mode from `reasoning_content` into `content`, while still producing the same bad model output.

### 5.4 QA-only scoping is necessary

Once the native assistant-only override was made **profile-aware**, the observed behavior improved in an important way:

- QA remained isolated for short-answer debugging;
- longform stopped inheriting the broken QA-specific override path.

This does **not** solve QA semantics yet, but it proves the control boundary is correct: the override belongs to QA-only experimentation, not to the global server path.

## 6. Current Validation Status

### 6.1 What looks good

- server MTP runtime no longer behaves like the original `0%` acceptance failure path;
- debugability is much better than before;
- QA-only controls can now be injected without contaminating longform by default;
- longform is no longer forced into the catastrophic 2-token failure mode caused by a global assistant-only override.

### 6.2 What is still failing

The branch still does **not** meet the intended devserver gate:

1. **QA semantics are not solved**  
   `qa-zh` can still fall into question echo / wrong answer-track behavior.

2. **Longform speed is below target**  
   The desired devserver target remains `25+ tok/s` with passing semantics. Current longform numbers in this debugging line are still materially below that target.

3. **Bit-identical parity is not established**  
   The runtime path is much healthier, but semantic parity with the intended contract is still under active investigation.

## 7. Why This Snapshot Still Merits a Commit

Even though this is not the final releaseable state, this snapshot deserves to be committed because it captures several durable advances:

- the server runtime no longer has to be debugged blind;
- the native thinking-contract investigation is now grounded in evidence rather than prompt folklore;
- the branch has a clean separation between:
  - runtime parity work,
  - parser / assembly visibility,
  - QA-only semantic experiments.

In other words, this commit turns an ambiguous failure surface into a structured debugging surface.

## 8. Recommended Next Steps

The next development cut should stay narrow:

1. continue only on the QA-only branch of the logic;
2. search for the minimal short-answer scaffold that prevents question echo without touching longform;
3. keep longform on the native path while measuring whether MTP acceptance and decode speed remain healthy;
4. only promote any QA workaround to a broader default after three conditions are met:
   - QA semantics pass,
   - longform semantics remain clean,
   - speed stays within the devserver gate.

## 9. File Map

This snapshot is mainly represented by the following files:

- `scripts/check/replay_server_profile.py`
- `scripts/run_server.sh`
- `src/llama.cpp/common/chat.cpp`
- `src/llama.cpp/models/templates/Nail-Qwen3.6-Minimal-Chat.jinja`
- `src/llama.cpp/tools/server/server-context.cpp`
- `src/llama.cpp/tools/server/server-task.cpp`

Formal documentation added in this commit:

- `docs/CGC_DEVSERVER_DEVELOPMENT_WHITEPAPER_20260904.md`

## 10. Closing Note

This branch has moved from "server path is broken" to "server path is measurable, partially aligned, and locally controllable." That is meaningful progress. The remaining work is no longer broad exploration; it is a focused semantic repair on the QA-only branch of the native thinking contract.
