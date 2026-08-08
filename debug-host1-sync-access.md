# Debug Session: host1-sync-access

- Status: OPEN
- Scope: host1 readonly access and Gate 3.1 sync evidence
- Symptom: local Gate 3.1 assets still mark host1 sync as pending, and current readonly probe to host1 fails with `AuthenticationException`.
- Current frontier: determine whether the blocker is auth method, credential mismatch, path drift, or missing remote directory.

## Hypotheses

1. host1 no longer accepts password auth for `root`, and requires key-based or alternate authentication.
2. host1 still accepts password auth, but the credential differs from host2.
3. host1 is reachable, but the repo root/path differs from `/root/flashkv0516/ComputeGraphCompiler-main/docs/technical_whitepapers`.
4. host1 is reachable and the path is correct, but `CGC_Gate_3.1_self_harness` is genuinely absent.

## Evidence Log

- User supplied a new host1 credential for readonly probing: `root@39.106.118.206` with a password distinct from host2.
- Re-running the readonly probe confirms host1 access is now unblocked.
- Both `host1` and `host2` report `DIR_EXISTS=1` for `/root/flashkv0516/ComputeGraphCompiler-main/docs/technical_whitepapers/CGC_Gate_3.1_self_harness`.
- The initial probe showed both remote directories were empty shells (`total 8`, only `.` and `..`) and all four expected files were absent:
  - `CGC_Gate_3.1_self_harness_Technical_Whitepaper_v1.0_zh_CN.md`
  - `CGC_Gate_3.1_self_harness_gate_map.json`
  - `CGC_Gate_3.1_self_harness_checkin.example.json`
  - `CGC_Gate_3.1_self_harness_summary.example.json`
- `sync_docs_host1_host2_20260619.py` does not include any `CGC_Gate_3.1_self_harness` files in its `UPLOADS`, which matches the observed "directory shell without content" state.
- A minimal backfill helper uploaded the four formal assets to both hosts and verified all uploads by SHA-256 match.
- The re-run readonly probe confirms both hosts now expose the four files with `EXISTS=1`.

## Next Action

- Local Gate 3.1 sync status assets can now be updated from `pending` to `synced`.
- If desired, extend the same backfill pattern to include `README.md` for complete directory parity, although the four formal assets required for sync evidence are now present on both hosts.
