# Debug Session: warm-decode-nan
- **Status**: [OPEN]
- **Issue**: Warm local server decode path produces Metal NaN and CPU fallback, especially around router/qmatmul, polluting warm-path KPI.
- **Debug Server**: pending
- **Log File**: .dbg/trae-debug-log-warm-decode-nan.ndjson

## Reproduction Steps
1. Start local warm server on `127.0.0.1:18080` via `colibri/c/openai_server.py`.
2. Send short decode requests (`max_tokens=1` or `8`) to `/v1/chat/completions`.
3. Observe `server.log` for `Metal NaN`, `qmatmul retry on CPU`, `router scores NaN`, and `forcing expert 0`.

## Hypotheses & Verification
| ID | Hypothesis | Likelihood | Effort | Evidence |
|----|------------|------------|--------|----------|
| A | Metal `qmatmul` decode kernel first emits non-finite values for `S=1` on specific shapes, and router NaN is a downstream symptom. | High | Low | Pending |
| B | Decode attention input/state becomes non-finite before router/qmatmul, so later NaN is only propagated. | Medium | Low | Pending |
| C | Router normalization or router scales drift out of range on upper layers, causing Metal/CPU numeric divergence. | Medium | Medium | Pending |
| D | Warm server only warms layer residency, not expert/data path, causing repeated fallback and unstable decode behavior. | Medium | Low | Pending |

## Log Evidence
- Pending.

## Verification Conclusion
- Pending.
