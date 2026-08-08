#!/usr/bin/env python3
import argparse
import glob
import json
import os
import statistics
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS_ROOT = REPO_ROOT / "var" / "colibri_metrics"
DEFAULT_WARM_SERVER_ENV = {
    "COLI_METAL_GEMMA4_MOE_BLOCK": "1",
    "COLI_METAL_GEMMA4_MOE_BLOCK_MAX_S": "32",
    "GEMMA4_EXPERT_PREFETCH": "0",
    "GEMMA4_EXPERT_PREFETCH_DEPTH": "1",
    "GEMMA4_EXPERT_W_ADOPT_BUDGET": "2",
    "GEMMA4_EXPERT_W_ADOPT_RECENT_WINDOW": "0",
    "GEMMA4_EXPERT_W_SLOT1_PROTECT_ADOPT_WINDOW": "1",
    "GEMMA4_EXPERT_W_SLOT1_PROTECT_HIT_WINDOW": "2",
    "GEMMA4_EXPERT_W_SLOT1_PROTECT_DECODE_ONLY": "1",
    "GEMMA4_EXPERT_W_SLOT1_PROTECT_SCORE_FLOOR": "10",
    "GEMMA4_EXPERT_W_SLOT1_PROTECT_GAP": "1",
    "GEMMA4_EXPERT_W_SLOT1_ADOPT_GAP": "1",
    "GEMMA4_EXPERT_DECODE_BOOTSTRAP_ANCHOR_FLOOR": "12",
    "GEMMA4_EXPERT_DECODE_BOOTSTRAP_SLOT1_FLOOR": "12",
    "GEMMA4_EXPERT_DECODE_BOOTSTRAP_MAX_LAYERS": "4",
    "GEMMA4_EXPERT_DECODE_BOOTSTRAP_SYNC_LOAD": "1",
}


def _post_json(url: str, payload: dict, headers: dict[str, str], timeout: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in headers.items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _response_record(response: dict, *, case: str, sample: int) -> dict:
    diagnostics = response.get("diagnostics") or {}
    routed = diagnostics.get("routed_expert_cache") or {}
    aggregate = routed.get("aggregate") or {}
    usage = response.get("usage") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    timing = diagnostics.get("timing") or {}
    completion_tokens = usage.get("completion_tokens")
    decode_ms = timing.get("decode_ms")
    tokens_per_second = None
    if completion_tokens is not None and decode_ms not in (None, 0):
        tokens_per_second = float(completion_tokens) / (float(decode_ms) / 1000.0)
    return {
        "sample": sample,
        "case": case,
        "request_id": response.get("id"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion_tokens,
        "cached_tokens": prompt_details.get("cached_tokens"),
        "prefill_ms": timing.get("prefill_ms"),
        "decode_ms": decode_ms,
        "total_ms": (
            float(timing.get("prefill_ms") or 0.0) + float(timing.get("decode_ms") or 0.0)
            if timing.get("prefill_ms") is not None and timing.get("decode_ms") is not None
            else None
        ),
        "tokens_per_second": tokens_per_second,
        "diagnostics": diagnostics,
        "current_request_raw_counter_focus_layers":
            ((aggregate.get("current_request_raw_counters") or {}).get("focus_layers")),
        "current_request_raw_counter_tracked_steps":
            ((aggregate.get("current_request_raw_counters") or {}).get("tracked_steps")),
        "aggregate_routed_expert_cache": aggregate,
    }


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _median(values: list[float]) -> Optional[float]:
    values = [float(v) for v in values if v is not None]
    return statistics.median(values) if values else None


def _read_openai_run(run_id: str, metrics_root: Path) -> list[dict]:
    items = []
    for p in glob.glob(str(metrics_root / "openai_server" / "requests" / "*.json")):
        d = _load_json(Path(p))
        if str(d.get("benchmark_run_id") or "") == run_id:
            items.append(d)
    items.sort(key=lambda d: int(d.get("started_at_ms") or 0))
    return items


def _join_gemma(openai_items: list[dict], metrics_root: Path) -> list[dict]:
    rows = []
    for item in openai_items:
        engine_request_id = str(item.get("engine_request_id") or "")
        if not engine_request_id:
            continue
        gp = metrics_root / "gemma4" / "requests" / f"{engine_request_id}.json"
        if not gp.exists():
            continue
        gemma = _load_json(gp)
        rows.append(
            {
                "case": str(item.get("benchmark_case") or ""),
                "request_id": str(item.get("request_id") or ""),
                "engine_request_id": engine_request_id,
                "openai_total_ms": item.get("total_ms"),
                "accept_latency_ms": item.get("accept_latency_ms"),
                "cached_tokens": item.get("cached_tokens"),
                "prompt_tokens": gemma.get("prompt_tokens"),
                "completion_tokens": gemma.get("completion_tokens"),
                "prefill_ms": gemma.get("prefill_ms"),
                "decode_ms": gemma.get("decode_ms"),
                "total_ms": gemma.get("total_ms"),
                "tokens_per_second": gemma.get("tokens_per_second"),
                "expert_disk_s": gemma.get("expert_disk_s"),
                "expert_matmul_s": gemma.get("expert_matmul_s"),
                "prefill_expert_load_wait_s": gemma.get("prefill_expert_load_wait_s"),
                "prefill_expert_load_evict_s": gemma.get("prefill_expert_load_evict_s"),
                "prefill_expert_load_read_s": gemma.get("prefill_expert_load_read_s"),
                "prefill_expert_load_decode_s": gemma.get("prefill_expert_load_decode_s"),
                "decode_expert_load_wait_s": gemma.get("decode_expert_load_wait_s"),
                "decode_expert_load_evict_s": gemma.get("decode_expert_load_evict_s"),
                "decode_expert_load_read_s": gemma.get("decode_expert_load_read_s"),
                "decode_expert_load_decode_s": gemma.get("decode_expert_load_decode_s"),
                "wcache_hit": gemma.get("wcache_hit"),
                "wcache_adopt": gemma.get("wcache_adopt"),
                "wcache_miss": gemma.get("wcache_miss"),
                "wcache_refuse": gemma.get("wcache_refuse"),
                "wcache_victim_refs_busy_refuse": gemma.get("wcache_victim_refs_busy_refuse"),
                "wcache_adopt_attempt": gemma.get("wcache_adopt_attempt"),
                "wcache_adopt_skip_cold": gemma.get("wcache_adopt_skip_cold"),
                "wcache_adopt_skip_not_recent": gemma.get("wcache_adopt_skip_not_recent"),
                "wcache_adopt_skip_budget": gemma.get("wcache_adopt_skip_budget"),
                "wcache_anchor_hit": gemma.get("wcache_anchor_hit"),
                "wcache_opportunistic_hit": gemma.get("wcache_opportunistic_hit"),
                "wcache_anchor_adopt": gemma.get("wcache_anchor_adopt"),
                "wcache_opportunistic_adopt": gemma.get("wcache_opportunistic_adopt"),
                "wcache_adopt_skip_floor": gemma.get("wcache_adopt_skip_floor"),
                "wcache_adopt_skip_gap": gemma.get("wcache_adopt_skip_gap"),
                "wcache_slot1_evict": gemma.get("wcache_slot1_evict"),
                "wcache_slot1_protect_armed": gemma.get("wcache_slot1_protect_armed"),
                "prefill_moe_gpu_s": gemma.get("prefill_moe_gpu_s"),
                "prefill_moe_cpu_s": gemma.get("prefill_moe_cpu_s"),
                "decode_moe_gpu_s": gemma.get("decode_moe_gpu_s"),
                "decode_moe_cpu_s": gemma.get("decode_moe_cpu_s"),
                "compute_backend": gemma.get("compute_backend"),
                "gpu_name": gemma.get("gpu_name"),
                "expert_mode": gemma.get("expert_mode"),
            }
        )
    return rows


def _post_continuation_case(
    *,
    url: str,
    model: str,
    timeout: float,
    run_id: str,
    case: str,
    seed_user: str,
    next_user: str,
    max_tokens: int,
) -> dict:
    seed_payload = {
        "model": model,
        "messages": [{"role": "user", "content": seed_user}],
        "max_tokens": max_tokens,
        "stream": False,
        "temperature": 0.0,
    }
    seed_response = _post_json(
        url,
        seed_payload,
        headers={
            "X-Coli-Benchmark-Run-Id": run_id,
            "X-Coli-Benchmark-Case": f"{case}_seed",
        },
        timeout=timeout,
    )
    assistant = str(
        (((seed_response.get("choices") or [{}])[0]).get("message") or {}).get("content") or ""
    ).strip()
    continuation_payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": seed_user},
            {"role": "assistant", "content": assistant},
            {"role": "user", "content": next_user},
        ],
        "max_tokens": max_tokens,
        "stream": False,
        "temperature": 0.0,
    }
    return _post_json(
        url,
        continuation_payload,
        headers={
            "X-Coli-Benchmark-Run-Id": run_id,
            "X-Coli-Benchmark-Case": case,
        },
        timeout=timeout,
    )


def _build_summary(rows: list[dict]) -> dict:
    ttft_rows = [r for r in rows if r["case"] == "ttft_probe"]
    decode_rows = [r for r in rows if r["case"] == "decode_probe"]
    ttft_cache_rows = [r for r in rows if r["case"] == "ttft_cache_probe"]
    decode_cache_rows = [r for r in rows if r["case"] == "decode_cache_probe"]
    return {
        "ttft_probe": {
            "samples": len(ttft_rows),
            "p50_total_ms": _median([r.get("total_ms") for r in ttft_rows]),
            "p50_prefill_ms": _median([r.get("prefill_ms") for r in ttft_rows]),
            "p50_decode_ms": _median([r.get("decode_ms") for r in ttft_rows]),
            "p50_cached_tokens": _median([r.get("cached_tokens") for r in ttft_rows]),
            "p50_expert_disk_s": _median([r.get("expert_disk_s") for r in ttft_rows]),
            "p50_prefill_expert_load_wait_s": _median([r.get("prefill_expert_load_wait_s") for r in ttft_rows]),
            "p50_prefill_expert_load_evict_s": _median([r.get("prefill_expert_load_evict_s") for r in ttft_rows]),
            "p50_prefill_expert_load_read_s": _median([r.get("prefill_expert_load_read_s") for r in ttft_rows]),
            "p50_prefill_expert_load_decode_s": _median([r.get("prefill_expert_load_decode_s") for r in ttft_rows]),
            "p50_decode_expert_load_wait_s": _median([r.get("decode_expert_load_wait_s") for r in ttft_rows]),
            "p50_decode_expert_load_evict_s": _median([r.get("decode_expert_load_evict_s") for r in ttft_rows]),
            "p50_decode_expert_load_read_s": _median([r.get("decode_expert_load_read_s") for r in ttft_rows]),
            "p50_decode_expert_load_decode_s": _median([r.get("decode_expert_load_decode_s") for r in ttft_rows]),
            "p50_wcache_hit": _median([r.get("wcache_hit") for r in ttft_rows]),
            "p50_wcache_adopt": _median([r.get("wcache_adopt") for r in ttft_rows]),
            "p50_wcache_miss": _median([r.get("wcache_miss") for r in ttft_rows]),
            "p50_wcache_refuse": _median([r.get("wcache_refuse") for r in ttft_rows]),
            "p50_wcache_victim_refs_busy_refuse": _median([r.get("wcache_victim_refs_busy_refuse") for r in ttft_rows]),
            "p50_wcache_adopt_attempt": _median([r.get("wcache_adopt_attempt") for r in ttft_rows]),
            "p50_wcache_adopt_skip_cold": _median([r.get("wcache_adopt_skip_cold") for r in ttft_rows]),
            "p50_wcache_adopt_skip_not_recent": _median([r.get("wcache_adopt_skip_not_recent") for r in ttft_rows]),
            "p50_wcache_adopt_skip_budget": _median([r.get("wcache_adopt_skip_budget") for r in ttft_rows]),
            "p50_wcache_anchor_hit": _median([r.get("wcache_anchor_hit") for r in ttft_rows]),
            "p50_wcache_opportunistic_hit": _median([r.get("wcache_opportunistic_hit") for r in ttft_rows]),
            "p50_wcache_anchor_adopt": _median([r.get("wcache_anchor_adopt") for r in ttft_rows]),
            "p50_wcache_opportunistic_adopt": _median([r.get("wcache_opportunistic_adopt") for r in ttft_rows]),
            "p50_wcache_adopt_skip_floor": _median([r.get("wcache_adopt_skip_floor") for r in ttft_rows]),
            "p50_wcache_adopt_skip_gap": _median([r.get("wcache_adopt_skip_gap") for r in ttft_rows]),
            "p50_wcache_slot1_evict": _median([r.get("wcache_slot1_evict") for r in ttft_rows]),
            "p50_wcache_slot1_protect_armed": _median([r.get("wcache_slot1_protect_armed") for r in ttft_rows]),
            "rows": ttft_rows,
        },
        "decode_probe": {
            "samples": len(decode_rows),
            "p50_total_ms": _median([r.get("total_ms") for r in decode_rows]),
            "p50_prefill_ms": _median([r.get("prefill_ms") for r in decode_rows]),
            "p50_decode_ms": _median([r.get("decode_ms") for r in decode_rows]),
            "p50_cached_tokens": _median([r.get("cached_tokens") for r in decode_rows]),
            "p50_tokens_per_second": _median([r.get("tokens_per_second") for r in decode_rows]),
            "p50_expert_disk_s": _median([r.get("expert_disk_s") for r in decode_rows]),
            "p50_prefill_expert_load_wait_s": _median([r.get("prefill_expert_load_wait_s") for r in decode_rows]),
            "p50_prefill_expert_load_evict_s": _median([r.get("prefill_expert_load_evict_s") for r in decode_rows]),
            "p50_prefill_expert_load_read_s": _median([r.get("prefill_expert_load_read_s") for r in decode_rows]),
            "p50_prefill_expert_load_decode_s": _median([r.get("prefill_expert_load_decode_s") for r in decode_rows]),
            "p50_decode_expert_load_wait_s": _median([r.get("decode_expert_load_wait_s") for r in decode_rows]),
            "p50_decode_expert_load_evict_s": _median([r.get("decode_expert_load_evict_s") for r in decode_rows]),
            "p50_decode_expert_load_read_s": _median([r.get("decode_expert_load_read_s") for r in decode_rows]),
            "p50_decode_expert_load_decode_s": _median([r.get("decode_expert_load_decode_s") for r in decode_rows]),
            "p50_wcache_hit": _median([r.get("wcache_hit") for r in decode_rows]),
            "p50_wcache_adopt": _median([r.get("wcache_adopt") for r in decode_rows]),
            "p50_wcache_miss": _median([r.get("wcache_miss") for r in decode_rows]),
            "p50_wcache_refuse": _median([r.get("wcache_refuse") for r in decode_rows]),
            "p50_wcache_victim_refs_busy_refuse": _median([r.get("wcache_victim_refs_busy_refuse") for r in decode_rows]),
            "p50_wcache_adopt_attempt": _median([r.get("wcache_adopt_attempt") for r in decode_rows]),
            "p50_wcache_adopt_skip_cold": _median([r.get("wcache_adopt_skip_cold") for r in decode_rows]),
            "p50_wcache_adopt_skip_not_recent": _median([r.get("wcache_adopt_skip_not_recent") for r in decode_rows]),
            "p50_wcache_adopt_skip_budget": _median([r.get("wcache_adopt_skip_budget") for r in decode_rows]),
            "p50_wcache_anchor_hit": _median([r.get("wcache_anchor_hit") for r in decode_rows]),
            "p50_wcache_opportunistic_hit": _median([r.get("wcache_opportunistic_hit") for r in decode_rows]),
            "p50_wcache_anchor_adopt": _median([r.get("wcache_anchor_adopt") for r in decode_rows]),
            "p50_wcache_opportunistic_adopt": _median([r.get("wcache_opportunistic_adopt") for r in decode_rows]),
            "p50_wcache_adopt_skip_floor": _median([r.get("wcache_adopt_skip_floor") for r in decode_rows]),
            "p50_wcache_adopt_skip_gap": _median([r.get("wcache_adopt_skip_gap") for r in decode_rows]),
            "p50_wcache_slot1_evict": _median([r.get("wcache_slot1_evict") for r in decode_rows]),
            "p50_wcache_slot1_protect_armed": _median([r.get("wcache_slot1_protect_armed") for r in decode_rows]),
            "rows": decode_rows,
        },
        "ttft_cache_probe": {
            "samples": len(ttft_cache_rows),
            "p50_total_ms": _median([r.get("total_ms") for r in ttft_cache_rows]),
            "p50_prefill_ms": _median([r.get("prefill_ms") for r in ttft_cache_rows]),
            "p50_decode_ms": _median([r.get("decode_ms") for r in ttft_cache_rows]),
            "p50_cached_tokens": _median([r.get("cached_tokens") for r in ttft_cache_rows]),
            "rows": ttft_cache_rows,
        },
        "decode_cache_probe": {
            "samples": len(decode_cache_rows),
            "p50_total_ms": _median([r.get("total_ms") for r in decode_cache_rows]),
            "p50_prefill_ms": _median([r.get("prefill_ms") for r in decode_cache_rows]),
            "p50_decode_ms": _median([r.get("decode_ms") for r in decode_cache_rows]),
            "p50_cached_tokens": _median([r.get("cached_tokens") for r in decode_cache_rows]),
            "p50_tokens_per_second": _median([r.get("tokens_per_second") for r in decode_cache_rows]),
            "rows": decode_cache_rows,
        },
    }


def _attach_response_diagnostics(rows: list[dict], response_records: list[dict]) -> None:
    by_request_id = {
        str(record.get("request_id") or ""): record
        for record in response_records
        if record.get("request_id")
    }
    for row in rows:
        record = by_request_id.get(str(row.get("request_id") or ""))
        if not record:
            continue
        row["current_request_raw_counter_focus_layers"] = record.get(
            "current_request_raw_counter_focus_layers")
        row["current_request_raw_counter_tracked_steps"] = record.get(
            "current_request_raw_counter_tracked_steps")


def _fallback_rows_from_responses(response_records: list[dict]) -> list[dict]:
    rows = []
    for record in response_records:
        aggregate = record.get("aggregate_routed_expert_cache") or {}
        current_request_delta = aggregate.get("current_request_delta") or {}
        current_request_raw_counters = aggregate.get("current_request_raw_counters") or {}
        rows.append(
            {
                "case": record.get("case"),
                "request_id": record.get("request_id"),
                "engine_request_id": None,
                "openai_total_ms": record.get("total_ms"),
                "accept_latency_ms": None,
                "cached_tokens": record.get("cached_tokens"),
                "prompt_tokens": record.get("prompt_tokens"),
                "completion_tokens": record.get("completion_tokens"),
                "prefill_ms": record.get("prefill_ms"),
                "decode_ms": record.get("decode_ms"),
                "total_ms": record.get("total_ms"),
                "tokens_per_second": record.get("tokens_per_second"),
                "expert_disk_s": None,
                "expert_matmul_s": None,
                "prefill_expert_load_wait_s": None,
                "prefill_expert_load_evict_s": None,
                "prefill_expert_load_read_s": None,
                "prefill_expert_load_decode_s": None,
                "decode_expert_load_wait_s": None,
                "decode_expert_load_evict_s": None,
                "decode_expert_load_read_s": None,
                "decode_expert_load_decode_s": None,
                "wcache_hit": None,
                "wcache_adopt": None,
                "wcache_miss": None,
                "wcache_refuse": None,
                "wcache_victim_refs_busy_refuse": None,
                "wcache_adopt_attempt": None,
                "wcache_adopt_skip_cold": None,
                "wcache_adopt_skip_not_recent": None,
                "wcache_adopt_skip_budget": None,
                "wcache_anchor_hit": None,
                "wcache_opportunistic_hit": None,
                "wcache_anchor_adopt": None,
                "wcache_opportunistic_adopt": None,
                "wcache_adopt_skip_floor": None,
                "wcache_adopt_skip_gap": None,
                "wcache_slot1_evict": None,
                "wcache_slot1_protect_armed": None,
                "prefill_moe_gpu_s": None,
                "prefill_moe_cpu_s": None,
                "decode_moe_gpu_s": None,
                "decode_moe_cpu_s": None,
                "compute_backend": "turbofieldfare_server_response",
                "gpu_name": None,
                "expert_mode": "response_diagnostics_fallback",
                "current_request_raw_counter_focus_layers":
                    current_request_raw_counters.get("focus_layers"),
                "current_request_raw_counter_tracked_steps":
                    current_request_raw_counters.get("tracked_steps"),
                "decode_protected_cap": aggregate.get("decode_protected_cap"),
                "decode_protected_budget": aggregate.get("decode_protected_budget"),
                "decode_protected_slots": aggregate.get("decode_protected_slots"),
                "prefill_transient_slots": aggregate.get("prefill_transient_slots"),
                "shared_resident_slots": aggregate.get("shared_resident_slots"),
                "total_evictions": aggregate.get("total_evictions"),
                "current_request_delta_evictions": current_request_delta.get("evictions"),
                "current_request_delta_decode_protected_evictions":
                    current_request_delta.get("decode_protected_evictions"),
                "current_request_delta_shared_resident_evictions":
                    current_request_delta.get("shared_resident_evictions"),
                "current_request_delta_decode_protected_promotions":
                    current_request_delta.get("decode_protected_promotions"),
                "current_request_delta_decode_protected_demotions":
                    current_request_delta.get("decode_protected_demotions"),
                "current_request_delta_decode_protected_admission_rejected":
                    current_request_delta.get("decode_protected_admission_rejected"),
                "current_request_delta_decode_shared_pool_handoff_hits":
                    current_request_delta.get("decode_shared_pool_handoff_hits"),
            }
        )
    return rows


def _build_production_gate(summary: dict, ttft_threshold_ms: float, decode_threshold_tps: float) -> dict:
    ttft_p50 = summary.get("ttft_probe", {}).get("p50_total_ms")
    decode_p50 = summary.get("decode_probe", {}).get("p50_tokens_per_second")
    ttft_pass = bool(ttft_p50 is not None and float(ttft_p50) <= float(ttft_threshold_ms))
    decode_pass = bool(decode_p50 is not None and float(decode_p50) >= float(decode_threshold_tps))
    return {
        "ttft_threshold_ms": float(ttft_threshold_ms),
        "decode_threshold_tps": float(decode_threshold_tps),
        "ttft_p50_total_ms": ttft_p50,
        "decode_p50_tokens_per_second": decode_p50,
        "ttft_pass": ttft_pass,
        "decode_pass": decode_pass,
        "pass": ttft_pass and decode_pass,
    }


def _write_report(report: dict, metrics_root: Path) -> tuple[Path, Path, Path]:
    reports = metrics_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    run_id = str(report["run_id"])
    json_path = reports / f"{run_id}.json"
    md_path = reports / f"{run_id}.md"
    diagnostics_path = reports / f"{run_id}_diagnostics.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    diagnostics_path.write_text(
        json.dumps({"run_id": run_id, "responses": report.get("diagnostic_responses", [])},
                   ensure_ascii=False,
                   indent=2) + "\n",
        encoding="utf-8")
    lines = [
        f"# Gemma4 Decode/TTFT Probe `{run_id}`",
        "",
        f"- `base_url`: {report['base_url']}",
        f"- `model`: {report['model']}",
        f"- `expert_mode`: {report.get('expert_mode')}",
        f"- `warm_server_env_defaults`: {json.dumps(report.get('warm_server_env_defaults', {}), ensure_ascii=False)}",
        f"- `ttft_samples`: {report['summary']['ttft_probe']['samples']}",
        f"- `decode_samples`: {report['summary']['decode_probe']['samples']}",
        f"- `ttft_cache_samples`: {report['summary']['ttft_cache_probe']['samples']}",
        f"- `decode_cache_samples`: {report['summary']['decode_cache_probe']['samples']}",
        f"- `production_pass`: {report['production_gate']['pass']}",
        f"- `diagnostics_json`: {diagnostics_path}",
        "",
        "## TTFT",
        f"- `p50_total_ms`: {report['summary']['ttft_probe']['p50_total_ms']}",
        f"- `p50_prefill_ms`: {report['summary']['ttft_probe']['p50_prefill_ms']}",
        f"- `p50_decode_ms`: {report['summary']['ttft_probe']['p50_decode_ms']}",
        f"- `p50_cached_tokens`: {report['summary']['ttft_probe']['p50_cached_tokens']}",
        f"- `p50_expert_disk_s`: {report['summary']['ttft_probe']['p50_expert_disk_s']}",
        f"- `p50_prefill_expert_load_wait_s`: {report['summary']['ttft_probe']['p50_prefill_expert_load_wait_s']}",
        f"- `p50_prefill_expert_load_evict_s`: {report['summary']['ttft_probe']['p50_prefill_expert_load_evict_s']}",
        f"- `p50_prefill_expert_load_read_s`: {report['summary']['ttft_probe']['p50_prefill_expert_load_read_s']}",
        f"- `p50_prefill_expert_load_decode_s`: {report['summary']['ttft_probe']['p50_prefill_expert_load_decode_s']}",
        f"- `p50_decode_expert_load_wait_s`: {report['summary']['ttft_probe']['p50_decode_expert_load_wait_s']}",
        f"- `p50_decode_expert_load_evict_s`: {report['summary']['ttft_probe']['p50_decode_expert_load_evict_s']}",
        f"- `p50_decode_expert_load_read_s`: {report['summary']['ttft_probe']['p50_decode_expert_load_read_s']}",
        f"- `p50_decode_expert_load_decode_s`: {report['summary']['ttft_probe']['p50_decode_expert_load_decode_s']}",
        f"- `p50_wcache_hit`: {report['summary']['ttft_probe']['p50_wcache_hit']}",
        f"- `p50_wcache_adopt`: {report['summary']['ttft_probe']['p50_wcache_adopt']}",
        f"- `p50_wcache_miss`: {report['summary']['ttft_probe']['p50_wcache_miss']}",
        f"- `p50_wcache_refuse`: {report['summary']['ttft_probe']['p50_wcache_refuse']}",
        f"- `p50_wcache_victim_refs_busy_refuse`: {report['summary']['ttft_probe']['p50_wcache_victim_refs_busy_refuse']}",
        f"- `p50_wcache_adopt_attempt`: {report['summary']['ttft_probe']['p50_wcache_adopt_attempt']}",
        f"- `p50_wcache_adopt_skip_cold`: {report['summary']['ttft_probe']['p50_wcache_adopt_skip_cold']}",
        f"- `p50_wcache_adopt_skip_not_recent`: {report['summary']['ttft_probe']['p50_wcache_adopt_skip_not_recent']}",
        f"- `p50_wcache_adopt_skip_budget`: {report['summary']['ttft_probe']['p50_wcache_adopt_skip_budget']}",
        f"- `p50_wcache_anchor_hit`: {report['summary']['ttft_probe']['p50_wcache_anchor_hit']}",
        f"- `p50_wcache_opportunistic_hit`: {report['summary']['ttft_probe']['p50_wcache_opportunistic_hit']}",
        f"- `p50_wcache_anchor_adopt`: {report['summary']['ttft_probe']['p50_wcache_anchor_adopt']}",
        f"- `p50_wcache_opportunistic_adopt`: {report['summary']['ttft_probe']['p50_wcache_opportunistic_adopt']}",
        f"- `p50_wcache_adopt_skip_floor`: {report['summary']['ttft_probe']['p50_wcache_adopt_skip_floor']}",
        f"- `p50_wcache_adopt_skip_gap`: {report['summary']['ttft_probe']['p50_wcache_adopt_skip_gap']}",
        f"- `p50_wcache_slot1_evict`: {report['summary']['ttft_probe']['p50_wcache_slot1_evict']}",
        f"- `p50_wcache_slot1_protect_armed`: {report['summary']['ttft_probe']['p50_wcache_slot1_protect_armed']}",
        f"- `threshold_ms`: {report['production_gate']['ttft_threshold_ms']}",
        f"- `pass`: {report['production_gate']['ttft_pass']}",
        "",
        "## Decode",
        f"- `p50_total_ms`: {report['summary']['decode_probe']['p50_total_ms']}",
        f"- `p50_prefill_ms`: {report['summary']['decode_probe']['p50_prefill_ms']}",
        f"- `p50_decode_ms`: {report['summary']['decode_probe']['p50_decode_ms']}",
        f"- `p50_cached_tokens`: {report['summary']['decode_probe']['p50_cached_tokens']}",
        f"- `p50_tokens_per_second`: {report['summary']['decode_probe']['p50_tokens_per_second']}",
        f"- `p50_expert_disk_s`: {report['summary']['decode_probe']['p50_expert_disk_s']}",
        f"- `p50_prefill_expert_load_wait_s`: {report['summary']['decode_probe']['p50_prefill_expert_load_wait_s']}",
        f"- `p50_prefill_expert_load_evict_s`: {report['summary']['decode_probe']['p50_prefill_expert_load_evict_s']}",
        f"- `p50_prefill_expert_load_read_s`: {report['summary']['decode_probe']['p50_prefill_expert_load_read_s']}",
        f"- `p50_prefill_expert_load_decode_s`: {report['summary']['decode_probe']['p50_prefill_expert_load_decode_s']}",
        f"- `p50_decode_expert_load_wait_s`: {report['summary']['decode_probe']['p50_decode_expert_load_wait_s']}",
        f"- `p50_decode_expert_load_evict_s`: {report['summary']['decode_probe']['p50_decode_expert_load_evict_s']}",
        f"- `p50_decode_expert_load_read_s`: {report['summary']['decode_probe']['p50_decode_expert_load_read_s']}",
        f"- `p50_decode_expert_load_decode_s`: {report['summary']['decode_probe']['p50_decode_expert_load_decode_s']}",
        f"- `p50_wcache_hit`: {report['summary']['decode_probe']['p50_wcache_hit']}",
        f"- `p50_wcache_adopt`: {report['summary']['decode_probe']['p50_wcache_adopt']}",
        f"- `p50_wcache_miss`: {report['summary']['decode_probe']['p50_wcache_miss']}",
        f"- `p50_wcache_refuse`: {report['summary']['decode_probe']['p50_wcache_refuse']}",
        f"- `p50_wcache_victim_refs_busy_refuse`: {report['summary']['decode_probe']['p50_wcache_victim_refs_busy_refuse']}",
        f"- `p50_wcache_adopt_attempt`: {report['summary']['decode_probe']['p50_wcache_adopt_attempt']}",
        f"- `p50_wcache_adopt_skip_cold`: {report['summary']['decode_probe']['p50_wcache_adopt_skip_cold']}",
        f"- `p50_wcache_adopt_skip_not_recent`: {report['summary']['decode_probe']['p50_wcache_adopt_skip_not_recent']}",
        f"- `p50_wcache_adopt_skip_budget`: {report['summary']['decode_probe']['p50_wcache_adopt_skip_budget']}",
        f"- `p50_wcache_anchor_hit`: {report['summary']['decode_probe']['p50_wcache_anchor_hit']}",
        f"- `p50_wcache_opportunistic_hit`: {report['summary']['decode_probe']['p50_wcache_opportunistic_hit']}",
        f"- `p50_wcache_anchor_adopt`: {report['summary']['decode_probe']['p50_wcache_anchor_adopt']}",
        f"- `p50_wcache_opportunistic_adopt`: {report['summary']['decode_probe']['p50_wcache_opportunistic_adopt']}",
        f"- `p50_wcache_adopt_skip_floor`: {report['summary']['decode_probe']['p50_wcache_adopt_skip_floor']}",
        f"- `p50_wcache_adopt_skip_gap`: {report['summary']['decode_probe']['p50_wcache_adopt_skip_gap']}",
        f"- `p50_wcache_slot1_evict`: {report['summary']['decode_probe']['p50_wcache_slot1_evict']}",
        f"- `p50_wcache_slot1_protect_armed`: {report['summary']['decode_probe']['p50_wcache_slot1_protect_armed']}",
        f"- `threshold_tps`: {report['production_gate']['decode_threshold_tps']}",
        f"- `pass`: {report['production_gate']['decode_pass']}",
        "",
        "## TTFT Cache",
        f"- `p50_total_ms`: {report['summary']['ttft_cache_probe']['p50_total_ms']}",
        f"- `p50_prefill_ms`: {report['summary']['ttft_cache_probe']['p50_prefill_ms']}",
        f"- `p50_decode_ms`: {report['summary']['ttft_cache_probe']['p50_decode_ms']}",
        f"- `p50_cached_tokens`: {report['summary']['ttft_cache_probe']['p50_cached_tokens']}",
        "",
        "## Decode Cache",
        f"- `p50_total_ms`: {report['summary']['decode_cache_probe']['p50_total_ms']}",
        f"- `p50_prefill_ms`: {report['summary']['decode_cache_probe']['p50_prefill_ms']}",
        f"- `p50_decode_ms`: {report['summary']['decode_cache_probe']['p50_decode_ms']}",
        f"- `p50_cached_tokens`: {report['summary']['decode_cache_probe']['p50_cached_tokens']}",
        f"- `p50_tokens_per_second`: {report['summary']['decode_cache_probe']['p50_tokens_per_second']}",
        "",
        "## Rows",
        "```json",
        json.dumps(report["rows"], ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path, diagnostics_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Run fixed Gemma4 TTFT/decode probes and summarize KPI files.")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1/chat/completions")
    ap.add_argument("--model", default="gemma4-colibri")
    ap.add_argument("--metrics-root", default=str(DEFAULT_METRICS_ROOT))
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--run-id", default="")
    ap.add_argument("--ttft-threshold-ms", type=float, default=300.0)
    ap.add_argument("--decode-threshold-tps", type=float, default=30.0)
    args = ap.parse_args()

    metrics_root = Path(args.metrics_root)
    run_id = args.run_id or f"gemma4_probe_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    response_records = []
    ttft_payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": "Reply with exactly one token: OK"}],
        "max_tokens": 1,
        "stream": False,
        "temperature": 0.0,
    }
    decode_payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": "Count from one to eight using digits separated by spaces."}],
        "max_tokens": 8,
        "stream": False,
        "temperature": 0.0,
    }

    for case, payload in (("ttft_probe", ttft_payload), ("decode_probe", decode_payload)):
        for sample in range(1, args.samples + 1):
            response = _post_json(
                args.base_url,
                payload,
                headers={
                    "X-Coli-Benchmark-Run-Id": run_id,
                    "X-Coli-Benchmark-Case": case,
                },
                timeout=args.timeout,
            )
            response_records.append(_response_record(response, case=case, sample=sample))

    for sample in range(1, args.samples + 1):
        response = _post_continuation_case(
            url=args.base_url,
            model=args.model,
            timeout=args.timeout,
            run_id=run_id,
            case="ttft_cache_probe",
            seed_user="Reply with exactly: alpha",
            next_user="Now reply with exactly: beta",
            max_tokens=2,
        )
        response_records.append(_response_record(response, case="ttft_cache_probe", sample=sample))
        response = _post_continuation_case(
            url=args.base_url,
            model=args.model,
            timeout=args.timeout,
            run_id=run_id,
            case="decode_cache_probe",
            seed_user="Reply with exactly: alpha",
            next_user="Count from one to eight using digits separated by spaces.",
            max_tokens=8,
        )
        response_records.append(_response_record(response, case="decode_cache_probe", sample=sample))

    time.sleep(0.5)
    openai_items = _read_openai_run(run_id, metrics_root)
    rows = _join_gemma(openai_items, metrics_root)
    if not rows:
        rows = _fallback_rows_from_responses(response_records)
    _attach_response_diagnostics(rows, response_records)
    report = {
        "run_id": run_id,
        "base_url": args.base_url,
        "model": args.model,
        "rows": rows,
        "diagnostic_responses": response_records,
        "expert_mode": rows[0].get("expert_mode") if rows else "unknown",
        "warm_server_env_defaults": DEFAULT_WARM_SERVER_ENV,
        "summary": _build_summary(rows),
    }
    report["production_gate"] = _build_production_gate(
        report["summary"],
        ttft_threshold_ms=args.ttft_threshold_ms,
        decode_threshold_tps=args.decode_threshold_tps,
    )
    json_path, md_path, diagnostics_path = _write_report(report, metrics_root)
    print(json.dumps({
        "run_id": run_id,
        "json": str(json_path),
        "markdown": str(md_path),
        "diagnostics_json": str(diagnostics_path),
        "summary": report["summary"],
        "production_gate": report["production_gate"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
