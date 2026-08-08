# Debug Session: decode-critical-path
- **Status**: [OPEN]
- **Issue**: Shared-pool preferred admission reduced churn metrics, but decode throughput remains around 4.x tok/s instead of the expected higher baseline. Need to determine whether the critical path is dominated by disk, wait, read, or prompt-cache mismatch.
- **Debug Server**: Pending startup
- **Log File**: .dbg/trae-debug-log-decode-critical-path.ndjson

## Reproduction Steps
1. Start the debug server for session `decode-critical-path`.
2. Add runtime instrumentation only, without modifying business logic.
3. Reproduce the same multi-turn continuation workload on baseline and shared-pool servers.
4. Compare pre-fix evidence for prompt-cache, decode wait/read, and shared-pool handoff timing.

## Hypotheses & Verification
| ID | Hypothesis | Likelihood | Effort | Evidence |
|----|------------|------------|--------|----------|
| A | Prompt-cache mismatch causes turn2/turn3 to miss cached prefix, inflating apparent decode cost. | High | Low | Pending |
| B | Shared-pool admission reduces churn but shifts time into decode wait/read, so throughput barely improves. | High | Medium | Pending |
| C | DecodeProtected policy is over-constraining hot experts, lowering short-horizon payoff despite fewer evictions. | Medium | Medium | Pending |
| D | External PD handoff contract is threaded through runtime, but the actual decode critical path still bypasses the useful shared-pool fast path. | Medium | Medium | Pending |

## Log Evidence
- Pending

## Verification Conclusion
- Pending
