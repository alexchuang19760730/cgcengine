# PD To DOPD Upgrade Plan

## Scope
- Keep existing PD capabilities intact: KV blocks, prefix cache, KDA ops, CQ4 transport, KV compression, zero-copy VRAM, and State ABI.
- Upgrade the current `Prefetch Distribution Service` into a DOPD-ready runtime contract.
- This phase only covers `P0-P2`: mapping, contract formalization, and ABI-aligned resume payload schema.

## Current PD Mapping

| Current asset | Current role | DOPD interpretation | Status |
| --- | --- | --- | --- |
| `pd_service.proto` | KV / prefix / KDA service contract | Base transport and cache RPC layer | Keep |
| `AllocateBlocks` | KV block allocation | Retain as data-plane primitive | Keep |
| `StorePrefixKV` / `GetPrefixKV` | Prefix cache storage | Reuse as payload / resume state carrier | Keep |
| `ExecuteCGCCommand` | PD to worker command bridge | Reuse for worker-side control bridging | Keep |
| `ExecuteKDAForward` / `UpdateKDABasis` | KDA preprocessing and basis update | Reuse as compression / KDA support path | Keep |
| `encode_pd_kv_blocks_v1()` | Existing PD payload encoding | Legacy resume / KV payload compatibility | Keep |
| `State ABI` + `CacheSchema` | Semantic compatibility contract | Source of truth for DOPD resume payload | Bind explicitly |

## Identified Gap
- Current PD is data-plane capable but does not expose session / phase semantics as first-class protocol objects.
- DOPD needs explicit runtime meaning for:
  - session identity
  - handoff identity
  - phase role
  - resume offset / decode takeover
  - rollback / recovery acknowledgement

## P1 Contract Upgrade
- Extend `pd_service.proto` with source-of-truth DOPD RPCs:
  - `PrepareHandoff`
  - `CommitHandoff`
  - `ResumeDecode`
  - `AbortHandoff`
  - `QuerySessionState`
- Keep old RPCs unchanged for backward compatibility.
- Do not enable the new RPCs until `pb2` regeneration and runtime implementation are ready.

## P2 Resume Payload Upgrade
- Add a new ABI-aligned payload format: `DOPD resume payload v2`.
- Preserve `pd_kv_blocks_v1` as the legacy format.
- `v2` payload binds the following State ABI concepts explicitly:
  - `cache_schema`
  - `kv_variant`
  - `abi_descriptor`
  - `layout_meta`
- Runtime transport / handoff concepts carried by `v2`:
  - `session_id`
  - `handoff_id`
  - `phase_role`
  - `resume_position`
  - `prefill_done`
  - `decode_resume`
  - `transport_codec`
  - `compression_codec`
  - `zero_copy_vram`
  - `integrity_checksum`

## Deliverables In This Change
- `dopd_schema.py`: canonical DOPD resume payload encoding / decoding helpers.
- `pd_client.py`: backward-compatible helpers for legacy and DOPD resume payload retrieval.
- `pd_service.proto`: DOPD contract source additions for later runtime enablement.

## Deferred To Later Phases
- `pb2` / `pb2_grpc` regeneration
- server-side implementation of DOPD RPCs
- cloud decode worker binding
- scheduler-driven dynamic handoff
- rollback / recovery execution path
