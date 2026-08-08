# Debug Session: shape-split-trace
- **Status**: [CLOSED]
- **Issue**: inst1/inst3 and inst2/inst4 produce different prompt-token counts and response shapes for the same chat/completions SWE-style prompt.
- **Debug Server**: pending
- **Log File**: .dbg/trae-debug-log-shape-split-trace.ndjson

## Reproduction Steps
1. Send the same `/v1/chat/completions` request to `inst2_direct`, `inst4_direct`, `inst1_direct`, and `inst3_direct`.
2. Compare `usage.prompt_tokens`, final prompt text before tokenization, and response content before gateway serialization.
3. Confirm where the 6-token delta and fenced JSON shape are introduced.

## Hypotheses & Verification
| ID | Hypothesis | Likelihood | Effort | Evidence |
|----|------------|------------|--------|----------|
| A | inst1/inst3 receive different final prompt text before tokenization, causing the 6-token delta. | High | Med | Pending |
| B | prompt text is the same, but host2 response serialization adds fenced JSON after decode. | Med | Med | Pending |
| C | host2 serving path injects an extra preamble/system wrapper before prompt build. | High | Med | Pending |
| D | tokenizer-side preprocessing differs by instance group even when final prompt text is identical. | Low | High | Pending |

## Log Evidence
- Host2 backend prompt-before-tokenizer evidence:
  - `serving_chat.py:_apply_conversation_template`
  - prompt SHA `80a5a34298298a2ef6c62195df6a5381439687c6d2fce9449d7a651509be7e1b`
  - prompt token count local `41`
- Host2 backend response-shape evidence:
  - `inst1` backend returned plain JSON with `meta_prompt_tokens=41`
  - `inst3` backend returned fenced JSON with `meta_prompt_tokens=41`
- Host2 gateway response-shape evidence after fixing instrumentation and scoped relaunch:
  - `inst1` gateway assistant SHA matched backend plain JSON SHA
  - `inst3` gateway assistant SHA matched backend fenced JSON SHA
  - gateway `usage.prompt_tokens` stayed `41` for both probes
- Host1 scoped relaunch evidence:
  - `inst2` long triage replay now reports `usage.prompt_tokens=122`
  - historical `116` prompt-token baseline did not reproduce after moving to the new runtime package generation
- Host1 `inst4` long triage A/B/gateway evidence:
  - `serving_chat.py:_apply_conversation_template` captured prompt SHA `0d55e3a641b7d7fef220e7837d58ed99ee725de9954bdb9d0f6a2b77253b19f5`
  - prompt token count local `122`
  - backend response stayed plain JSON with `meta_prompt_tokens=122`
  - gateway response stayed plain JSON with `usage.prompt_tokens=122`
- Host2 long triage A/B/gateway evidence:
  - `inst1` prompt SHA `0d55e3a641b7d7fef220e7837d58ed99ee725de9954bdb9d0f6a2b77253b19f5`, prompt token count local `122`, backend fenced JSON, gateway fenced JSON
  - `inst3` prompt SHA `0d55e3a641b7d7fef220e7837d58ed99ee725de9954bdb9d0f6a2b77253b19f5`, prompt token count local `122`, backend plain JSON, gateway plain JSON
  - host2 long-prompt replay flips the earlier short-prompt fenced/plain pairing, proving the shape split is not stably bound to a specific instance

## Verification Conclusion
- Hypothesis A: `PARTIAL`
  - For the short isolation prompt on host2, prompt text before tokenizer is identical across `inst1` and `inst3`, so fenced-vs-plain is not caused by prompt drift there.
  - For the historical long triage prompt family, host1 `inst4` plus host2 `inst1/inst3` now all show `122` at prompt-build, backend usage, and gateway usage, and the old `116 vs 122` split no longer reproduces after scoped runtime realignment.
- Hypothesis B: `REJECTED`
  - Gateway serialization does not add fences. It forwards the backend payload unchanged.
- Hypothesis C: `SUPPORTED`
  - The stale Serve controller/runtime package generation was materially involved in the historical shape split.
- Additional conclusion:
  - The fenced/plain divergence is a backend generation-style variability issue and is not a stable per-instance runtime trait, because the long-prompt host2 replay flips the earlier short-prompt instance pairing.
- Hypothesis D: `UNNEEDED`
  - No tokenizer-only divergence is needed to explain the currently observed evidence.

## Cleanup
- Removed the temporary `shape-split-trace` instrumentation from:
  - `ComputeGraphCompiler-main/Backend/CGC/cloud_sglang/python/sglang/srt/entrypoints/openai/serving_chat.py`
  - `ComputeGraphCompiler-main/Backend/CGC/ray_serve_sglang_gateway.py`
- Synced cleaned source to host1 and host2 repository paths.
- Removed `shape-split-trace.env` from host1 and host2.
- Removed the temporary host2 debug server used only for this trace round.
