# Debug Session: host2-decode-quality
- **Status**: [CLOSED]
- **Issue**: host2 `inst1/inst3` single-question completions initially succeeded with poor latency, abnormal output, and high `/data2/ray` pressure; this session closes after chat-path output and runtime stability were verified.
- **Debug Server**: N/A
- **Log File**: N/A

## Reproduction Steps
1. Launch host2 `inst1` or `inst3` with the current synchronized runtime.
2. Send a minimal `/v1/completions` request with a short arithmetic prompt.
3. Observe completion latency, output text, and request-path traces.
4. Inspect host2 `/data2/ray` usage and worker log growth.

## Hypotheses & Verification
| ID | Hypothesis | Likelihood | Effort | Evidence |
|----|------------|------------|--------|----------|
| A | output corruption is caused by prompt construction or tokenizer/backend mismatch rather than SWA/decode budget shortage | High | Med | Pending |
| B | the 64s latency is in decode/runtime execution after successful admission, not in request admission itself | High | Med | Pending |
| C | host2 `/data2/ray` pressure and oversized logs are adding runtime instability and startup overhead | High | Low | Pending |
| D | `TokenizersBackend after retries` is a meaningful signal for output quality regression on this model path | Med | Med | Pending |
| E | host2 no longer needs extra SWA headroom for the minimal single-question path; remaining issues are downstream | High | Low | Pending |

## Log Evidence
- Baseline replay before cleanup:
  - `inst1` completion returned in about `64.16s`.
  - `inst3` completion returned in about `63.99s`.
  - Both returned the same abnormal text: `';\nconst string = '4';\nconst`.
- Admission/path evidence:
  - Both requests still showed `pending_swa_budget=384` and `rem_swa_tokens=128` after prefill budget update.
  - This continues to support that the minimal path is not blocked by extra SWA headroom.
- Disk/log pressure evidence:
  - `df -h /data2` showed `/data2` at `100%`.
  - `/data2/ray/inst1` and `/data2/ray/inst3` occupied about `25G` and `66G`.
  - Largest `worker*.err` files reached `2.1G`, `2.4G`, and historical `14G`.
- Trace-noise evidence:
  - The dominant high-frequency spam came from `prefill_return_none_early` in idle loops.
  - After suppressing idle-loop logging for that trace and redoing instance-scoped cleanup/relaunch, replay latency dropped sharply.
- Replay after cleanup and quieter tracing:
  - `inst1` completion returned in about `16.8s`.
  - `inst3` completion returned in about `16.64s`.
  - Output text remained unchanged and still abnormal: `';\nconst string = '4';\nconst`.
- Remaining output-quality signals:
  - Host2 still logs `Tokenizer for /data/models/DeepSeek-V4-Flash-UD-IQ2 is still TokenizersBackend after retries with --trust-remote-code`.
  - Gateway/backend still logs `No HuggingFace chat template found` and defaults to `string` content format.
- Template-path experiment:
  - Raw `/v1/completions` with the naked prompt still produced JS-like garbage.
  - Manually wrapping the prompt as `<｜begin▁of▁sentence｜><|User|>...<|Assistant|>` returned the correct answer `4` and stopped cleanly.
  - This showed the main output-quality issue was prompt/template shape rather than tokenizer round-trip corruption.
- Productized fix:
  - Added a built-in `deepseek-v4` conversation template and mapped `model_type=deepseek_v4` to it in `sglang/srt/parser/conversation.py`.
  - Synced `conversation.py` into host2 repo-tree and live `site-packages`.
  - Relaunch logs now show `Inferred chat template from model path: deepseek-v4`.
- Post-fix API verification:
  - `POST /v1/chat/completions` on host2 `inst1` now returns assistant content `"4"` in about `10.9s`.
- Remaining disk-pressure signal:
  - Even after cleanup, new sessions still reported `/data2/ray/... is over 95% full, available space: 0 GB`.
- Root-level cleanup evidence:
  - Root-cause inventory on host2 `/data2` showed:
    - `egodex`: about `687G`
    - `models`: about `167G`
    - `ray`: about `93G`
  - Safe cleanup targeted only historical Ray runtime artifacts, not models or datasets.
  - Removed `24` old `session_*` directories under `/data2/ray/inst1` and `/data2/ray/inst3`, keeping only the current `session_latest` targets.
  - Reclaimed about `76.6 GB`.
  - Post-cleanup disk state:
    - `/data2`: `93%` used with about `72G` available
    - `/data2/ray`: reduced to about `137M`
- Post-cleanup service verification:
  - `POST /v1/chat/completions` on host2 `inst1` still returns assistant content `"4"` in about `4.36s`.
- Final target-close verification:
  - First re-check found `inst3` down on port `50073`; it was relaunched with the synchronized runtime.
  - Final host2 verification passed on both instances:
    - `inst1` (`50053`): `/health=ok`, `/v1/chat/completions -> "4"` in about `4.51s`
    - `inst3` (`50073`): `/health=ok`, `/v1/chat/completions -> "4"` in about `10.39s`
  - Verification artifact written to `ComputeGraphCompiler-main/Output/cli_gate_upkg39/host2_chat_single_question_report.json`.

## Verification Conclusion
- Hypothesis A: Refined and confirmed. The dominant issue was missing DeepSeek V4 prompt/template shaping, not tokenization round-trip of plain text.
- Hypothesis B: Confirmed. Admission succeeds; a large part of prior latency came from downstream runtime overhead rather than admission.
- Hypothesis C: Confirmed. Host2 disk pressure and runaway worker logs materially worsened runtime conditions.
- Hypothesis D: Partially downgraded. `TokenizersBackend after retries` is still a warning and may matter for advanced model-specific attributes, but it is not the immediate blocker for correct chat answers once the DeepSeek V4 chat template is supplied.
- Hypothesis E: Confirmed for the current proof path. No extra SWA headroom is needed to make single-question requests complete.
- Current status:
  - Latency improved significantly after suppressing idle trace noise and clearing instance-scoped runtime state.
  - Chat-path output quality is fixed for the verified single-question case through the DeepSeek V4 template mapping.
  - Raw `/v1/completions` remains prompt-shape sensitive because OpenAI completions do not auto-apply the chat template.
  - The urgent `/data2` runtime pressure is mitigated for serving; remaining root usage is now dominated by retained datasets (`egodex`) and model artifacts, not Ray runtime residue.
  - The immediate host2 target is now closed: `inst1` and `inst3` both serve real single-question `chat/completions` requests successfully.
