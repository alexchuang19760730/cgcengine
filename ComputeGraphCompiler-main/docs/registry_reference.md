# Registry Reference

## Release Alias Registry

| Alias | Schema Key | Schema Path | Purpose |
| --- | --- | --- | --- |
| `agent_execution` | `cgc.agent_execution.v1` | `docs/gate_whitepapers/CGC_AGENT_EXECUTION_SCHEMA_v1.0.json` | Normalize SWE agent workload execution into a machine-readable release alias. |
| `deepep_release_guard` | `cgc.deepep_release_guard.v1` | `docs/gate_whitepapers/CGC_DEEPEP_RELEASE_GUARD_SCHEMA_v1.0.json` | Normalize DeepEP runtime/release claim into a machine-readable release alias. |
| `fresh_host1_probe` | `cgc.fresh_host1_probe.v1` | `docs/gate_whitepapers/CGC_FRESH_HOST1_PROBE_SCHEMA_v1.0.json` | Formalize the fresh host1 health probe used by the DeepEP release guard. |

## Ref Object Shape

All release aliases use a named ref object instead of a plain path string.

```json
{
  "ref_kind": "json_payload_ref",
  "source_path": "/abs/path/to/file.json",
  "schema_key": "cgc.agent_execution.v1",
  "section": "formal_evidence.swe_verified_formal_summary.payload",
  "source": "system_execution_manifest",
  "profile_id": "optional_profile_id",
  "profile_version": "optional_profile_version",
  "binding_key": "optional_binding_key",
  "artifact_name": "optional_artifact_name"
}
```

## Governance Boundary

- Formal release-facing aliases must map back to `Gate 6.0` scoped capabilities and an executable CLI entrypoint.
- Work that lacks a `Gate 6.0` capability boundary or CLI entrypoint remains exploration evidence and must not be promoted into formal release claims.
