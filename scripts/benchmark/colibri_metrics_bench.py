#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS_ROOT = REPO_ROOT / "var" / "colibri_metrics"

OPENAI_CASES = [
    {
        "name": "openai_short_nonstream",
        "stream": False,
        "max_tokens": 24,
        "messages": [
            {"role": "user", "content": "Reply in one sentence about Colibri metrics readiness."},
        ],
    },
    {
        "name": "openai_short_stream",
        "stream": True,
        "max_tokens": 24,
        "messages": [
            {"role": "user", "content": "Reply in one concise sentence about Gemma4 request telemetry."},
        ],
    },
    {
        "name": "openai_medium_nonstream",
        "stream": False,
        "max_tokens": 48,
        "messages": [
            {"role": "user", "content": "Summarize request latency, queue wait, and throughput in three short bullet points."},
        ],
    },
]

OPENAI_CONTINUATION_CASES = [
    {
        "name": "ttft_cache_probe",
        "seed_user": "Reply with exactly: alpha",
        "next_user": "Now reply with exactly: beta",
        "max_tokens": 2,
    },
    {
        "name": "decode_cache_probe",
        "seed_user": "Reply with exactly: alpha",
        "next_user": "Count from one to eight using digits separated by spaces.",
        "max_tokens": 8,
    },
]

SERVICE_CASES = [
    {
        "name": "service_submit_ready",
        "receipt": {
            "status": "ready",
            "message": "benchmark receipt",
            "cache_tier": "ram",
            "bytes_loaded": 1048576,
            "load_ms": 12.5,
            "resident_handles": ["colibri-runtime://benchmark/layer/0"],
            "artifacts": ["benchmark://artifact/0"],
            "metrics": {"resident_unit_count": 1},
            "unit_results": [
                {
                    "key": "layer0-full",
                    "unit_kind": "layer",
                    "cache_tier": "ram",
                    "bytes_loaded": 1048576,
                    "load_ms": 12.5,
                    "runtime_confirmed": True,
                }
            ],
        },
    }
]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _http_json(
    method: str,
    url: str,
    *,
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 180.0,
) -> tuple[int, str, dict[str, Any] | None]:
    raw_body = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url,
        data=raw_body,
        headers={"Content-Type": "application/json", **(headers or {})},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
            payload = json.loads(text) if text and text[:1] in "{[" else None
            return int(getattr(resp, "status", 200) or 200), text, payload if isinstance(payload, dict) else None
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        payload = json.loads(text) if text and text[:1] in "{[" else None
        return int(exc.code or 500), text, payload if isinstance(payload, dict) else None


def _http_stream(
    url: str,
    *,
    body: dict[str, Any],
    headers: dict[str, str],
    timeout: float = 180.0,
) -> tuple[int, str]:
    req = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return int(getattr(resp, "status", 200) or 200), resp.read().decode("utf-8", errors="replace")


def _health_url_from_openai_base(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[:-3]
    return normalized + "/health"


def _extract_assistant_text(payload: dict[str, Any] | None) -> str:
    choice = ((payload or {}).get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content")
    return str(content or "").strip()


def _fetch_json(url: str, timeout: float = 30.0) -> dict[str, Any] | None:
    status, _text, payload = _http_json("GET", url, timeout=timeout)
    if status != 200 or not isinstance(payload, dict):
        return None
    return payload


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * (pct / 100.0)
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return ordered[lower]
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _metric_summary(records: list[dict[str, Any]], fields: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {"count": len(records)}
    for field in fields:
        values = [
            float(item[field])
            for item in records
            if item.get(field) is not None and str(item.get(field)) != ""
        ]
        if not values:
            continue
        summary[field] = {
            "p50": _percentile(values, 50.0),
            "p95": _percentile(values, 95.0),
            "p99": _percentile(values, 99.0),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
        }
    return summary


def _openai_headers(run_id: str, case_name: str) -> dict[str, str]:
    return {
        "X-Coli-Benchmark-Run-Id": run_id,
        "X-Coli-Benchmark-Case": case_name,
    }


def _run_openai_matrix(base_url: str, model: str, iterations: int, run_id: str) -> None:
    chat_url = base_url.rstrip("/") + "/chat/completions"
    for case in OPENAI_CASES:
        for _ in range(iterations):
            body = {
                "model": model,
                "messages": list(case["messages"]),
                "max_tokens": int(case["max_tokens"]),
                "stream": bool(case["stream"]),
                "temperature": 0.0,
            }
            headers = _openai_headers(run_id, str(case["name"]))
            if bool(case["stream"]):
                status, _text = _http_stream(chat_url, body=body, headers=headers)
            else:
                status, _text, _payload = _http_json("POST", chat_url, body=body, headers=headers)
            if status != 200:
                raise RuntimeError(f"OpenAI benchmark case {case['name']} failed with HTTP {status}")
    for case in OPENAI_CONTINUATION_CASES:
        for _ in range(iterations):
            seed_headers = _openai_headers(run_id, f"{case['name']}_seed")
            seed_body = {
                "model": model,
                "messages": [{"role": "user", "content": str(case["seed_user"])}],
                "max_tokens": int(case["max_tokens"]),
                "stream": False,
                "temperature": 0.0,
            }
            status, _text, payload = _http_json("POST", chat_url, body=seed_body, headers=seed_headers)
            if status != 200:
                raise RuntimeError(f"OpenAI benchmark case {case['name']}_seed failed with HTTP {status}")
            assistant = _extract_assistant_text(payload)
            headers = _openai_headers(run_id, str(case["name"]))
            body = {
                "model": model,
                "messages": [
                    {"role": "user", "content": str(case["seed_user"])},
                    {"role": "assistant", "content": assistant},
                    {"role": "user", "content": str(case["next_user"])},
                ],
                "max_tokens": int(case["max_tokens"]),
                "stream": False,
                "temperature": 0.0,
            }
            status, _text, _payload = _http_json("POST", chat_url, body=body, headers=headers)
            if status != 200:
                raise RuntimeError(f"OpenAI benchmark case {case['name']} failed with HTTP {status}")


def _service_submit_payload(session_id: str) -> dict[str, Any]:
    return {
        "protocol": "colibri_engine_bridge/v1",
        "action": "begin_or_update_session",
        "session": {
            "session_id": session_id,
            "engine": "colibri",
            "state": "submitted",
            "request_seq": 1,
            "frontier_key": "benchmark-frontier",
            "model": "gemma4-colibri",
            "mode": "benchmark",
            "created_at_ms": int(time.time() * 1000),
            "submitted_at_ms": int(time.time() * 1000),
        },
        "transport": {
            "kind": "http",
            "target": "benchmark",
            "delivery": "benchmark",
            "metadata": {},
        },
        "request": {
            "enabled": True,
            "summary": {
                "engine_bridge_ready": True,
                "unresolved_source_count": 0,
            },
            "lanes": {
                "current": [],
                "next": [],
                "next_next": [],
                "far": [],
            },
        },
        "client": {
            "adapter": "scripts.benchmark.colibri_metrics_bench",
            "pid": os.getpid(),
            "cwd": os.getcwd(),
        },
        "response_contract": {
            "required_fields": [
                "accepted",
                "session_state",
                "worker_id",
                "queue_depth",
            ],
            "normalization_version": 1,
        },
    }


def _run_service_matrix(base_url: str, iterations: int, run_id: str) -> None:
    submit_url = base_url.rstrip("/") + "/session"
    for case in SERVICE_CASES:
        for idx in range(iterations):
            session_id = f"{run_id}-{case['name']}-{idx:02d}"
            headers = _openai_headers(run_id, str(case["name"]))
            status, _text, payload = _http_json(
                "POST",
                submit_url,
                body=_service_submit_payload(session_id),
                headers=headers,
            )
            if status != 202 or not payload or str(payload.get("session_id") or "") != session_id:
                raise RuntimeError(f"Service benchmark submit failed for {session_id} with HTTP {status}")
            receipt = {
                "session_id": session_id,
                "worker_id": "benchmark-worker",
                "completed_at_ms": int(time.time() * 1000),
                **dict(case["receipt"]),
            }
            receipt_url = base_url.rstrip("/") + f"/session/{session_id}/receipt"
            status, _text, _payload = _http_json("POST", receipt_url, body=receipt, headers=headers)
            if status != 200:
                raise RuntimeError(f"Service benchmark receipt failed for {session_id} with HTTP {status}")
            status_url = base_url.rstrip("/") + f"/session/{session_id}"
            deadline = time.time() + 5.0
            while time.time() < deadline:
                status_code, _text, status_payload = _http_json("GET", status_url, headers=headers)
                if status_code == 200 and isinstance(status_payload, dict):
                    session_state = str(status_payload.get("session_state") or status_payload.get("state") or "")
                    if session_state == "ready":
                        break
                    if session_state in {"blocked", "failed"}:
                        raise RuntimeError(f"Service benchmark session {session_id} ended in {session_state}")
                time.sleep(0.1)
            else:
                raise RuntimeError(f"Service benchmark session {session_id} did not reach ready")


def _collect_openai_run(metrics_root: Path, run_id: str) -> list[dict[str, Any]]:
    requests_dir = metrics_root / "openai_server" / "requests"
    if not requests_dir.exists():
        return []
    records = []
    for path in sorted(requests_dir.glob("*.json")):
        payload = _read_json(path)
        if str(payload.get("benchmark_run_id") or "") == run_id:
            records.append(payload)
    return records


def _collect_service_run(metrics_root: Path, run_id: str) -> list[dict[str, Any]]:
    sessions_dir = metrics_root / "colibri_service" / "sessions"
    if not sessions_dir.exists():
        return []
    records = []
    for path in sorted(sessions_dir.glob("*.json")):
        payload = _read_json(path)
        if str(payload.get("benchmark_run_id") or "") == run_id:
            records.append(payload)
    return records


def _collect_gemma4_run(metrics_root: Path, openai_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    requests_dir = metrics_root / "gemma4" / "requests"
    if not requests_dir.exists():
        return []
    case_by_engine_id = {
        str(item.get("engine_request_id") or ""): str(item.get("benchmark_case") or "")
        for item in openai_records
        if str(item.get("engine_request_id") or "")
    }
    records = []
    for engine_request_id, benchmark_case in sorted(case_by_engine_id.items()):
        path = requests_dir / f"{engine_request_id}.json"
        if not path.exists():
            continue
        payload = _read_json(path)
        payload["benchmark_case"] = benchmark_case
        records.append(payload)
    return records


def _group_by_case(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in records:
        case = str(item.get("benchmark_case") or "unlabeled")
        grouped.setdefault(case, []).append(item)
    return grouped


def _build_report(run_id: str, metrics_root: Path) -> dict[str, Any]:
    openai_records = _collect_openai_run(metrics_root, run_id)
    service_records = _collect_service_run(metrics_root, run_id)
    gemma4_records = _collect_gemma4_run(metrics_root, openai_records)
    return {
        "schema_version": 1,
        "run_id": run_id,
        "generated_at_ms": int(time.time() * 1000),
        "metrics_root": str(metrics_root),
        "layers": {
            "openai_server": {
                case: _metric_summary(
                    items,
                    [
                        "total_ms",
                        "queue_wait_ms",
                        "accept_latency_ms",
                        "tokens_per_second",
                        "prompt_tokens",
                        "completion_tokens",
                        "cached_tokens",
                    ],
                )
                for case, items in sorted(_group_by_case(openai_records).items())
            },
            "gemma4": {
                case: _metric_summary(
                    items,
                    [
                        "total_ms",
                        "accept_latency_ms",
                        "prompt_tokens",
                        "completion_tokens",
                        "prompt_bytes",
                        "prefill_ms",
                        "decode_ms",
                        "attention_s",
                        "expert_disk_s",
                        "expert_matmul_s",
                        "lm_head_s",
                    ],
                )
                for case, items in sorted(_group_by_case(gemma4_records).items())
            },
            "colibri_service": {
                case: _metric_summary(items, ["ready_latency_ms", "queue_wait_ms", "staging_ms", "receipt_load_ms"])
                for case, items in sorted(_group_by_case(service_records).items())
            },
        },
        "record_counts": {
            "openai_server": len(openai_records),
            "gemma4": len(gemma4_records),
            "colibri_service": len(service_records),
        },
    }


def _format_metric_block(metrics: dict[str, Any]) -> list[str]:
    lines = []
    for name, summary in metrics.items():
        if name == "count":
            continue
        lines.append(
            f"- `{name}`: p50={summary['p50']:.3f} p95={summary['p95']:.3f} "
            f"p99={summary['p99']:.3f} avg={summary['avg']:.3f}"
        )
    return lines or ["- no numeric metrics"]


def _write_report(report: dict[str, Any], metrics_root: Path) -> tuple[Path, Path]:
    reports_dir = metrics_root / "reports"
    _ensure_dir(reports_dir)
    json_path = reports_dir / f"{report['run_id']}.json"
    md_path = reports_dir / f"{report['run_id']}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [f"# Colibri Metrics Report: {report['run_id']}", ""]
    lines.append("## Record Counts")
    for layer, count in sorted(dict(report.get("record_counts") or {}).items()):
        lines.append(f"- `{layer}`: {count}")
    environment = dict(report.get("environment") or {})
    if environment:
        lines.append("")
        lines.append("## Environment")
        openai_health = dict(environment.get("openai_health") or {})
        if openai_health:
            for key in ("cpu", "gpu", "gpus", "vram_total_gb"):
                if key in openai_health:
                    lines.append(f"- `openai_health.{key}`: {openai_health[key]}")
    for layer, cases in sorted(dict(report.get("layers") or {}).items()):
        lines.append("")
        lines.append(f"## {layer}")
        if not cases:
            lines.append("- no records")
            continue
        for case, metrics in sorted(cases.items()):
            lines.append(f"### {case}")
            lines.append(f"- samples: {metrics.get('count', 0)}")
            lines.extend(_format_metric_block(metrics))
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed Colibri metrics benchmark matrix")
    parser.add_argument("--openai-base-url", default="http://127.0.0.1:8000/v1", help="OpenAI server base URL")
    parser.add_argument("--service-base-url", default="http://127.0.0.1:30110/colibri", help="Colibri service base URL")
    parser.add_argument("--model", default="gemma4-colibri", help="Model id for OpenAI benchmark calls")
    parser.add_argument("--metrics-root", default=str(DEFAULT_METRICS_ROOT), help="Metrics root path")
    parser.add_argument("--iterations", type=int, default=3, help="Iterations per benchmark case")
    parser.add_argument("--skip-openai", action="store_true", help="Skip OpenAI server benchmark cases")
    parser.add_argument("--skip-service", action="store_true", help="Skip Colibri service benchmark cases")
    parser.add_argument("--require-openai-gpu", action="store_true", help="Fail if OpenAI health does not report a GPU backend")
    parser.add_argument("--run-id", default="", help="Explicit benchmark run id")
    args = parser.parse_args()

    metrics_root = Path(args.metrics_root).resolve()
    run_id = str(args.run_id or f"bench_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}")

    if not args.skip_openai:
        _run_openai_matrix(str(args.openai_base_url), str(args.model), int(args.iterations), run_id)
    if not args.skip_service:
        _run_service_matrix(str(args.service_base_url), int(args.iterations), run_id)

    # Let the service worker loop flush the receipt-driven transition before reading snapshots.
    time.sleep(0.5)
    report = _build_report(run_id, metrics_root)
    openai_health = None if args.skip_openai else _fetch_json(_health_url_from_openai_base(str(args.openai_base_url)))
    if args.require_openai_gpu:
        gpu_count = int((openai_health or {}).get("gpus") or 0)
        gpu_name = str((openai_health or {}).get("gpu") or "")
        if gpu_count < 1 or gpu_name in {"", "none"}:
            raise RuntimeError(f"OpenAI server is not reporting a GPU backend: {openai_health}")
    report["environment"] = {"openai_health": openai_health or {}}
    json_path, md_path = _write_report(report, metrics_root)
    print(json.dumps({"run_id": run_id, "json_report": str(json_path), "markdown_report": str(md_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
