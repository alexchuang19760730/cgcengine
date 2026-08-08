from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = REPO_ROOT / "ComputeGraphCompiler-main"
for candidate in (REPO_ROOT, ENGINE_ROOT):
    raw = str(candidate.resolve())
    if raw not in sys.path:
        sys.path.insert(0, raw)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _extract_gate(report: Dict[str, Any]) -> Dict[str, Any]:
    gate_result = report.get("gate_result") if isinstance(report.get("gate_result"), dict) else {}
    for key in ("m72", "m71", "m7"):
        gate = gate_result.get(key)
        if isinstance(gate, dict) and gate:
            if key == "m72" and isinstance(gate_result.get("m7"), dict):
                return gate_result["m7"]
            return gate
    raise RuntimeError("missing_m7_audit_payload")


def _extract_audit_paths(report: Dict[str, Any]) -> Tuple[Path, Path]:
    gate = _extract_gate(report)
    audit = gate.get("audit") if isinstance(gate.get("audit"), dict) else {}
    events_path = Path(str(audit.get("events_path") or "")).expanduser().resolve()
    head_path = Path(str(audit.get("chain_head_path") or "")).expanduser().resolve()
    if not events_path.exists() or not head_path.exists():
        raise RuntimeError("missing_audit_paths_in_report")
    return events_path, head_path


def run_audit(*, output_dir: str, strict: bool) -> Dict[str, Any]:
    from cgc_engine.product import run_m7_gate, run_m72_gate

    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if strict:
        m7_output_dir = output_root / "m7_artifacts"
        m7_report = run_m7_gate(output_dir=str(m7_output_dir))
        m7_gate = ((m7_report.get("gate_result") or {}).get("m7") or {}) if isinstance(m7_report, dict) else {}
        m72_output_dir = output_root / "m72_industrial"
        m72_report = run_m72_gate(
            output_dir=str(m72_output_dir),
            cgc_report={"gate_result": {"m7": m7_gate}},
        )
        m72_gate = ((m72_report.get("gate_result") or {}).get("m72") or {}) if isinstance(m72_report, dict) else {}
        report = {
            "name": "CGC Audit CLI",
            "status": "PASS"
            if str(m7_gate.get("status") or "") == "PASS" and str(m72_gate.get("status") or "") == "PASS"
            else "FAIL",
            "mode": "strict",
            "report_path": str((output_root / "report.json").resolve()),
            "gate_result": {
                "m7": m7_gate,
                "m72": m72_gate,
            },
            "subreports": {
                "m7": str((m7_output_dir / "m7_industrial" / "m7_report.json").resolve()),
                "m72": str((m72_output_dir / "report.json").resolve()),
            },
        }
    else:
        m7_report = run_m7_gate(output_dir=str(output_root))
        m7_gate = ((m7_report.get("gate_result") or {}).get("m7") or {}) if isinstance(m7_report, dict) else {}
        report = {
            "name": "CGC Audit CLI",
            "status": str(m7_gate.get("status") or "FAIL"),
            "mode": "baseline",
            "report_path": str((output_root / "report.json").resolve()),
            "gate_result": {
                "m7": m7_gate,
            },
            "subreports": {
                "m7": str((output_root / "m7_industrial" / "m7_report.json").resolve()),
            },
        }

    report_path = output_root / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def verify_audit(*, report_path: str = "", log_path: str = "", head_path: str = "") -> Dict[str, Any]:
    if report_path:
        events_path, chain_head_path = _extract_audit_paths(_read_json(Path(report_path).expanduser().resolve()))
    else:
        events_path = Path(log_path).expanduser().resolve()
        chain_head_path = Path(head_path).expanduser().resolve()
    rows = _read_jsonl(events_path)
    head = _read_json(chain_head_path)

    from cgc_engine.audit.chain import canonical_json_bytes, sha256_bytes

    prev = "0" * 64
    ok = True
    event_count = 0
    for row in rows:
        event = row.get("event")
        event_hash = str(row.get("event_hash") or "")
        chain_hash = str(row.get("chain_hash") or "")
        computed_event_hash = sha256_bytes(canonical_json_bytes(event))
        prev = sha256_bytes((prev + computed_event_hash).encode("utf-8"))
        event_count += 1
        if computed_event_hash != event_hash or prev != chain_hash:
            ok = False
            break
    head_hash = str(head.get("chain_head_hash") or "")
    ok = bool(ok and prev == head_hash and int(head.get("event_count", -1)) == event_count)
    return {
        "status": "PASS" if ok else "FAIL",
        "verify_ok": bool(ok),
        "events_path": str(events_path),
        "chain_head_path": str(chain_head_path),
        "event_count": int(event_count),
        "chain_head_hash": head_hash,
    }


def trace_audit(*, report_path: str = "", log_path: str = "", stage: str = "", limit: int = 20) -> Dict[str, Any]:
    if report_path:
        events_path, _ = _extract_audit_paths(_read_json(Path(report_path).expanduser().resolve()))
    else:
        events_path = Path(log_path).expanduser().resolve()
    rows = _read_jsonl(events_path)
    normalized_stage = str(stage or "").strip().lower()
    filtered: List[Dict[str, Any]] = []
    for row in rows:
        event = row.get("event") if isinstance(row.get("event"), dict) else {}
        kind = str(event.get("kind") or "").strip().lower()
        if normalized_stage and kind != normalized_stage:
            continue
        filtered.append(
            {
                "kind": str(event.get("kind") or ""),
                "event_hash": str(row.get("event_hash") or ""),
                "chain_hash": str(row.get("chain_hash") or ""),
                "event": event,
            }
        )
    return {
        "status": "PASS",
        "events_path": str(events_path),
        "stage": str(stage or ""),
        "count": len(filtered),
        "events": filtered[: max(1, int(limit))],
    }


def export_audit(*, report_path: str, output_path: str, export_format: str) -> Dict[str, Any]:
    report = _read_json(Path(report_path).expanduser().resolve())
    gate = _extract_gate(report)
    audit = gate.get("audit") if isinstance(gate.get("audit"), dict) else {}
    dynamic = gate.get("dynamic_trace_l1") if isinstance(gate.get("dynamic_trace_l1"), dict) else {}
    replay = gate.get("soft_rt_replay") if isinstance(gate.get("soft_rt_replay"), dict) else {}
    industrial = gate.get("industrial_audit") if isinstance(gate.get("industrial_audit"), dict) else {}

    payload = {
        "status": str(gate.get("status") or "UNKNOWN"),
        "audit": {
            "events_path": str(audit.get("events_path") or ""),
            "chain_head_path": str(audit.get("chain_head_path") or ""),
            "verify_ok": bool(audit.get("verify_ok")),
            "event_count": int(audit.get("event_count", 0) or 0),
        },
        "dynamic_trace_l1": dynamic,
        "soft_rt_replay": replay,
        "industrial_audit": industrial,
    }

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fmt = str(export_format).strip().lower()
    if fmt == "json":
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    elif fmt == "html":
        html = (
            "<html><body><h1>CGC Audit Report</h1>"
            f"<pre>{json.dumps(payload, ensure_ascii=False, indent=2)}</pre>"
            "</body></html>"
        )
        output.write_text(html, encoding="utf-8")
    else:
        md = [
            "# CGC Audit Report",
            "",
            f"- status: {payload['status']}",
            f"- events_path: {payload['audit']['events_path']}",
            f"- chain_head_path: {payload['audit']['chain_head_path']}",
            f"- verify_ok: {payload['audit']['verify_ok']}",
            f"- event_count: {payload['audit']['event_count']}",
            f"- compile_success_rate: {dynamic.get('compile_success_rate', '')}",
            f"- cache_hit_rate: {dynamic.get('cache_hit_rate', '')}",
            f"- p99_latency_ms: {replay.get('p99_latency_ms', '')}",
            f"- event_integrity: {industrial.get('event_integrity', '')}",
        ]
        output.write_text("\n".join(md) + "\n", encoding="utf-8")

    return {
        "status": "PASS",
        "format": fmt,
        "output_path": str(output),
    }

