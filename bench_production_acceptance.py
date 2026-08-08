#!/usr/bin/env python3
"""Gemma4-26B-A4B 纯端侧单机量产验收压测脚本。

目标：
  - 对 edge_first_proxy 发送连续多轮真实请求
  - 触发 live report 中的生产 gate 计算
  - 强制要求请求走纯端侧 local_full 路径
  - 最终直接读取 single_node_candidate_matrix.live.json 输出 GO / NO_GO
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import aiohttp


DEFAULT_PROXY_URL = os.environ.get("PROXY_URL", "http://127.0.0.1:30011")
DEFAULT_MODEL = os.environ.get("PRODUCTION_MODEL", os.environ.get("ACTIVE_MODEL", "gemma4"))
DEFAULT_REPORT_PATH = os.environ.get(
    "EDGE_FIRST_REPORT_PATH",
    "/Users/alexchuang/Documents/flashkv0516/ComputeGraphCompiler-main/Output/edge_first_proxy_reports/single_node_candidate_matrix.live.json",
)
DEFAULT_REPORT_DIR = str(Path(DEFAULT_REPORT_PATH).resolve().parent)
DEFAULT_BASELINE_OUTPUT = os.environ.get(
    "EDGE_BASELINE_OUTPUT",
    str(Path(DEFAULT_REPORT_DIR) / "dense_local_baseline.json"),
)

PROMPTS = [
    "Write a Python function to merge overlapping intervals and explain the edge cases.",
    "Refactor this API handler for better error handling and smaller functions.",
    "Generate strict JSON with keys summary, risks, and next_steps for a deployment checklist.",
    "Review this caching strategy and point out correctness risks before performance concerns.",
    "Implement a deterministic parser for a simple config format with comments and blank lines.",
    "Fix this bug: requests occasionally return duplicated records after retry logic.",
    "Provide a concise architecture summary for a pure edge speculative decoding pipeline.",
    "Return valid JSON only with fields ttft_target, decode_target, and readiness.",
]

SCENARIO_NORMAL = "normal"
SCENARIO_LOW_MEMORY = "low-memory"

_SCENARIO_ALLOWED_ROUTES = {
    SCENARIO_NORMAL: ("local_full",),
    SCENARIO_LOW_MEMORY: ("layer_split_pd", "cloud_pd"),
}

_SCENARIO_PRIMARY_ROUTE = {
    SCENARIO_NORMAL: "local_full",
    SCENARIO_LOW_MEMORY: "layer_split_pd",
}

FRONTIER_MODE_REUSE = "reuse"
FRONTIER_MODE_DIVERSE = "diverse"
_BENCH_TOKENIZER_CACHE: dict[str, object | None] = {}


def _candidate_tokenizer_paths(model: str) -> list[str]:
    repo_root = Path(__file__).resolve().parent
    paths: list[str] = []
    for candidate in (
        os.environ.get("EDGE_BENCH_TOKENIZER_PATH", ""),
        os.environ.get("DSV4_TOKENIZER_PATH", ""),
    ):
        text = str(candidate or "").strip()
        if text:
            paths.append(text)
    try:
        from app.shared.model_registry import get_model_config
        cfg = get_model_config(model)
        tokenizer_path = str(getattr(cfg, "tokenizer_path", "") or "").strip()
        if tokenizer_path:
            if not os.path.isabs(tokenizer_path):
                tokenizer_path = str((repo_root / tokenizer_path).resolve())
            paths.append(tokenizer_path)
    except Exception:
        pass
    default_gemma = repo_root / "models" / "gemma-4-mtp-head"
    paths.append(str(default_gemma))
    deduped: list[str] = []
    for path in paths:
        if path and path not in deduped:
            deduped.append(path)
    return deduped


def _load_bench_tokenizer(model: str):
    cache_key = str(model or "").strip().lower() or "default"
    if cache_key in _BENCH_TOKENIZER_CACHE:
        return _BENCH_TOKENIZER_CACHE[cache_key]
    tokenizer = None
    try:
        from tokenizers import Tokenizer as _RustTokenizer
        for root in _candidate_tokenizer_paths(model):
            tok_path = Path(root) / "tokenizer.json"
            if tok_path.exists():
                tokenizer = _RustTokenizer.from_file(str(tok_path))
                break
    except Exception:
        tokenizer = None
    _BENCH_TOKENIZER_CACHE[cache_key] = tokenizer
    return tokenizer


def _estimate_output_tokens(text: str, model: str) -> int:
    cleaned = str(text or "")
    if not cleaned.strip():
        return 0
    tokenizer = _load_bench_tokenizer(model)
    if tokenizer is not None:
        try:
            return max(len(tokenizer.encode(cleaned, add_special_tokens=False).ids), 0)
        except Exception:
            pass
    return max(len(cleaned) // 4, 1)


def _route_expectation_for_scenario(
    scenario: str,
    explicit_expect_route: str,
) -> tuple[str, tuple[str, ...]]:
    normalized = str(scenario or SCENARIO_NORMAL).strip().lower()
    primary = str(explicit_expect_route or _SCENARIO_PRIMARY_ROUTE.get(normalized, "local_full")).strip()
    allowed = tuple(_SCENARIO_ALLOWED_ROUTES.get(normalized, (primary,)))
    if explicit_expect_route:
        allowed = (primary,)
    return primary, allowed


def _build_route_validation(
    *,
    scenario: str,
    actual_route: str,
    primary_route: str,
    allowed_routes: tuple[str, ...],
    route_payload: dict | None = None,
) -> dict:
    transport_route = (route_payload or {}).get("transport_route") or {}
    actual = str(actual_route or "")
    allowed = tuple(str(route).strip() for route in allowed_routes if str(route).strip())
    route_ok = actual in allowed if allowed else (actual == primary_route)
    normalized_scenario = str(scenario or SCENARIO_NORMAL).strip().lower()
    admissible_degrade = (
        normalized_scenario == SCENARIO_LOW_MEMORY
        and route_ok
        and actual in allowed
    )
    return {
        "scenario": normalized_scenario,
        "actual_route": actual,
        "primary_route": str(primary_route or ""),
        "allowed_routes": list(allowed),
        "route_ok": route_ok,
        "admissible_degrade": admissible_degrade,
        "degrade_suggested": bool(transport_route.get("degrade_suggested")),
        "degrade_target_mode": str(transport_route.get("degrade_target_mode") or ""),
        "mode_switch_reason": str(transport_route.get("mode_switch_reason") or transport_route.get("reason") or ""),
        "memory_pressure": str(transport_route.get("memory_pressure") or ""),
    }


def _routing_status(
    *,
    scenario: str,
    actual_route: str,
    primary_route: str,
    allowed_routes: tuple[str, ...],
) -> str:
    validation = _build_route_validation(
        scenario=scenario,
        actual_route=actual_route,
        primary_route=primary_route,
        allowed_routes=allowed_routes,
    )
    if not validation["route_ok"]:
        return "FAIL"
    if validation["admissible_degrade"] and actual_route != primary_route:
        return "ADMISSIBLE_DEGRADE"
    return "PASS"


async def _post_json(session: aiohttp.ClientSession, url: str, payload: dict) -> dict:
    async with session.post(url, json=payload) as resp:
        data = await resp.json()
        if resp.status >= 400:
            raise RuntimeError(f"{url} -> HTTP {resp.status}: {data}")
        return data


async def _reset_runtime(session: aiohttp.ClientSession, proxy_url: str) -> None:
    await _post_json(session, f"{proxy_url}/stats/reset", {})
    await _post_json(session, f"{proxy_url}/acceptance/reset", {})


async def _route_test(
    session: aiohttp.ClientSession,
    proxy_url: str,
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    conversation_id: str,
    request_id: str,
) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "conversation_id": conversation_id,
        "request_id": request_id,
    }
    async with session.post(f"{proxy_url}/route-test", json=payload) as resp:
        data = await resp.json()
        if resp.status >= 400:
            raise RuntimeError(f"/route-test -> HTTP {resp.status}: {data}")
        return data


async def _stream_request(
    session: aiohttp.ClientSession,
    proxy_url: str,
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    scenario: str,
    primary_route: str,
    allowed_routes: tuple[str, ...],
    conversation_id: str,
    request_id: str,
) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "conversation_id": conversation_id,
        "request_id": request_id,
    }
    started = time.perf_counter()
    first_token_at = None
    response_parts: list[str] = []
    validation = _build_route_validation(
        scenario=scenario,
        actual_route="",
        primary_route=primary_route,
        allowed_routes=allowed_routes,
    )

    async with session.post(
        f"{proxy_url}/v1/chat/completions",
        json=payload,
        timeout=aiohttp.ClientTimeout(total=180),
    ) as resp:
        if resp.status != 200:
            text = await resp.text()
            raise RuntimeError(f"/v1/chat/completions -> HTTP {resp.status}: {text[:300]}")
        actual_route = str(
            resp.headers.get("x-edge-router")
            or resp.headers.get("x-cgc-hermes-route-mode")
            or ""
        )
        validation = _build_route_validation(
            scenario=scenario,
            actual_route=actual_route,
            primary_route=primary_route,
            allowed_routes=allowed_routes,
        )
        if not validation["route_ok"]:
            raise RuntimeError(
                f"unexpected route header: expected one of {validation['allowed_routes']} "
                f"actual={actual_route or '(empty)'}"
            )
        async for raw in resp.content:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            try:
                obj = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            choices = obj.get("choices", [])
            if not choices:
                continue
            content = str((choices[0].get("delta") or {}).get("content") or "")
            if content and first_token_at is None:
                first_token_at = time.perf_counter()
            if content:
                response_parts.append(content)

    ended = time.perf_counter()
    ttft_ms = (first_token_at - started) * 1000 if first_token_at else 0.0
    total_ms = (ended - started) * 1000
    decode_elapsed_ms = max(total_ms - ttft_ms, 0.0)
    response_text = "".join(response_parts)
    approx_tokens = _estimate_output_tokens(response_text, model)
    return {
        "route_mode": actual_route,
        "route_validation": validation,
        "ttft_ms": round(ttft_ms, 1),
        "total_ms": round(total_ms, 1),
        "decode_tps_approx": round(approx_tokens / decode_elapsed_ms * 1000, 1) if decode_elapsed_ms > 0 and approx_tokens > 0 else 0.0,
        "response_chars": len(response_text),
    }


async def _fetch_stats(session: aiohttp.ClientSession, proxy_url: str) -> dict:
    async with session.get(f"{proxy_url}/stats") as resp:
        return await resp.json()


def _read_live_report(report_path: str) -> dict:
    path = Path(report_path)
    if not path.exists():
        raise FileNotFoundError(f"live report not found: {report_path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_report(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"missing evidence file: {path}")
    if path.stat().st_size <= 2:
        raise RuntimeError(f"evidence file too small: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_evidence_chain(report_dir: str) -> dict[str, dict]:
    base = Path(report_dir)
    filenames = [
        "route_policy_v2.live.json",
        "route_heat_snapshot.live.json",
        "draft_mode_acceptance.live.json",
        "single_node_candidate_matrix.live.json",
    ]
    return {name: _read_json_report(base / name) for name in filenames}


def _write_json(path: str, payload: dict) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_dense_baseline_summary(
    *,
    args,
    stats: dict,
    report: dict,
    readiness: dict,
    execution_route_summary: dict,
) -> dict:
    acceptance_summary = report.get("acceptance_summary") or {}
    transport_runtime = report.get("transport_runtime") or {}
    transport_route = report.get("transport_route") or {}
    path_metrics = stats.get("path_metrics") or {}
    cold_stats = path_metrics.get("cold_path") or {}
    warm_hot_stats = path_metrics.get("warm_hot_path") or {}
    warm_hot_bottleneck = ((acceptance_summary.get("path_bottlenecks") or {}).get("warm_hot_path") or {})
    expert_plan = report.get("expert_data_plane") or {}
    return {
        "schema_version": "v1",
        "kind": "dense_local_baseline",
        "generated_at_epoch_sec": int(time.time()),
        "proxy_url": args.proxy_url,
        "model": args.model,
        "scenario": args.scenario,
        "rounds": int(args.rounds),
        "max_tokens": int(args.max_tokens),
        "routing": execution_route_summary,
        "runtime": {
            "local_model_path": transport_runtime.get("local_model_path") or transport_runtime.get("local_full_model_path"),
            "layer_split_model_path": transport_runtime.get("layer_split_model_path"),
            "local_num_layers": transport_route.get("local_num_layers") or transport_runtime.get("local_num_layers"),
            "memory_pressure": transport_route.get("memory_pressure"),
            "moe_candidate": transport_route.get("moe_candidate"),
            "moe_streaming_admissible": transport_route.get("moe_streaming_admissible"),
        },
        "baseline_status": {
            "cold_path_status": str(((acceptance_summary.get("cold_path") or {}).get("status")) or "UNKNOWN"),
            "production_readiness_status": str(readiness.get("status") or "UNKNOWN"),
            "baseline_ready": bool(
                ((acceptance_summary.get("cold_path") or {}).get("status") == "PASS")
                and float(cold_stats.get("decode_tps_avg", 0.0) or 0.0) > 0
            ),
        },
        "metrics": {
            "overall": {
                "ttft_ms_avg": float(acceptance_summary.get("ttft_ms_avg", 0.0) or 0.0),
                "decode_tps_avg": float(acceptance_summary.get("decode_tps_avg", 0.0) or 0.0),
                "completed_requests": int(acceptance_summary.get("completed_requests", 0) or 0),
                "failed_requests": int(acceptance_summary.get("failed_requests", 0) or 0),
            },
            "cold_path": {
                "ttft_ms_avg": float(cold_stats.get("ttft_ms_avg", 0.0) or 0.0),
                "decode_tps_avg": float(cold_stats.get("decode_tps_avg", 0.0) or 0.0),
                "completed_requests": int(cold_stats.get("completed_requests", 0) or 0),
                "failed_requests": int(cold_stats.get("failed_requests", 0) or 0),
                "execution_success_rate": float(cold_stats.get("execution_success_rate", 0.0) or 0.0),
                "content_success_rate": float(cold_stats.get("content_success_rate", 0.0) or 0.0),
            },
            "warm_hot_path": {
                "ttft_ms_avg": float(warm_hot_stats.get("ttft_ms_avg", 0.0) or 0.0),
                "decode_tps_avg": float(warm_hot_stats.get("decode_tps_avg", 0.0) or 0.0),
                "completed_requests": int(warm_hot_stats.get("completed_requests", 0) or 0),
                "failed_requests": int(warm_hot_stats.get("failed_requests", 0) or 0),
                "status": str(((acceptance_summary.get("warm_hot_path") or {}).get("status")) or "UNKNOWN"),
                "not_evaluated_reason": str(warm_hot_bottleneck.get("summary") or ""),
            },
        },
        "expert_data_plane": {
            "catalog_source": expert_plan.get("catalog_source"),
            "last_plan_enabled": bool((expert_plan.get("last_plan") or {}).get("enabled")),
            "last_plan_reason": str(((expert_plan.get("last_plan") or {}).get("reason")) or ""),
        },
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description="Gemma4-26B-A4B 量产验收压测")
    parser.add_argument("--proxy-url", default=DEFAULT_PROXY_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
    parser.add_argument("--scenario", choices=[SCENARIO_NORMAL, SCENARIO_LOW_MEMORY], default=SCENARIO_NORMAL)
    parser.add_argument("--expect-route", default="")
    parser.add_argument("--frontier-mode", choices=[FRONTIER_MODE_REUSE, FRONTIER_MODE_DIVERSE], default=FRONTIER_MODE_REUSE)
    parser.add_argument("--baseline-output", default="")
    args = parser.parse_args()
    primary_route, allowed_routes = _route_expectation_for_scenario(args.scenario, args.expect_route)
    run_conversation_id = f"bench-{args.scenario}-{int(time.time())}"

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{args.proxy_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"proxy health returned HTTP {resp.status}")
                health = await resp.json()
        except Exception as exc:
            print(f"[ERROR] proxy unreachable: {exc}")
            return 2
        if not bool((health or {}).get("edge_model_loaded")):
            print("[ERROR] edge model not loaded; pure edge-only acceptance cannot start")
            return 3

        print(f"[INFO] reset runtime stats on {args.proxy_url}")
        await _reset_runtime(session, args.proxy_url)

        print(
            f"[INFO] running {args.rounds} rounds on model={args.model} scenario={args.scenario} "
            f"primary_route={primary_route} allowed_routes={list(allowed_routes)} "
            f"frontier_mode={args.frontier_mode} conversation_id={run_conversation_id}"
        )
        for idx in range(args.rounds):
            prompt = PROMPTS[0] if args.frontier_mode == FRONTIER_MODE_REUSE else PROMPTS[idx % len(PROMPTS)]
            route_request_id = f"{run_conversation_id}-route-{idx + 1}"
            stream_request_id = f"{run_conversation_id}-stream-{idx + 1}"
            route_preview = await _route_test(
                session,
                args.proxy_url,
                model=args.model,
                prompt=prompt,
                max_tokens=args.max_tokens,
                conversation_id=run_conversation_id,
                request_id=route_request_id,
            )
            preview_final_route = str(route_preview.get("final_route_mode") or "")
            preview_transport_route = str(((route_preview.get("transport_route") or {}).get("mode")) or "")
            preview_validation = _build_route_validation(
                scenario=args.scenario,
                actual_route=preview_final_route,
                primary_route=primary_route,
                allowed_routes=allowed_routes,
                route_payload=route_preview,
            )
            if not preview_validation["route_ok"]:
                print(
                    f"[ERROR] round {idx + 1}: route-test final_route_mode={preview_final_route} "
                    f"transport_route={preview_transport_route}, allowed={preview_validation['allowed_routes']}"
                )
                return 4
            result = await _stream_request(
                session,
                args.proxy_url,
                model=args.model,
                prompt=prompt,
                max_tokens=args.max_tokens,
                scenario=args.scenario,
                primary_route=primary_route,
                allowed_routes=allowed_routes,
                conversation_id=run_conversation_id,
                request_id=stream_request_id,
            )
            route_validation = result["route_validation"]
            print(
                f"  round {idx + 1}/{args.rounds}: "
                f"route={result['route_mode']} "
                f"admissible_degrade={route_validation['admissible_degrade']} "
                f"ttft={result['ttft_ms']}ms total={result['total_ms']}ms "
                f"decode~={result['decode_tps_approx']} tok/s chars={result['response_chars']}"
            )

        stats = await _fetch_stats(session, args.proxy_url)
        print("[INFO] runtime stats:")
        print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))

    report = _read_live_report(args.report_path)
    evidence_chain = _read_evidence_chain(args.report_dir)
    readiness = report.get("production_readiness") or {}
    gates = report.get("production_gates") or {}
    transport_route = report.get("transport_route") or {}
    hardware_profile = report.get("hardware_profile") or {}
    acceptance_summary = report.get("acceptance_summary") or {}
    print("[INFO] production gates:")
    print(json.dumps(gates, ensure_ascii=False, indent=2, sort_keys=True))
    print("[INFO] execution route:")
    routing_status = _routing_status(
        scenario=args.scenario,
        actual_route=str(transport_route.get("mode") or ""),
        primary_route=primary_route,
        allowed_routes=allowed_routes,
    )
    admissible_degrade = (
        args.scenario == SCENARIO_LOW_MEMORY
        and str(transport_route.get("mode") or "") in allowed_routes
        and str(transport_route.get("mode") or "") != primary_route
    )
    execution_route_summary = {
        "scenario": args.scenario,
        "primary_route": primary_route,
        "allowed_routes": list(allowed_routes),
        "transport_route_mode": transport_route.get("mode"),
        "hardware_route_mode": hardware_profile.get("route_mode"),
        "degrade_suggested": transport_route.get("degrade_suggested"),
        "degrade_target_mode": transport_route.get("degrade_target_mode"),
        "mode_switch_reason": transport_route.get("mode_switch_reason") or transport_route.get("reason"),
        "memory_pressure": transport_route.get("memory_pressure"),
        "admissible_degrade": admissible_degrade,
        "routing_status": routing_status,
    }
    print(json.dumps(execution_route_summary, ensure_ascii=False, indent=2, sort_keys=True))
    print("[INFO] acceptance summary:")
    print(json.dumps(acceptance_summary, ensure_ascii=False, indent=2, sort_keys=True))
    print("[INFO] path bottlenecks:")
    print(json.dumps((acceptance_summary.get("path_bottlenecks") or {}), ensure_ascii=False, indent=2, sort_keys=True))
    print("[INFO] production readiness:")
    print(json.dumps(readiness, ensure_ascii=False, indent=2, sort_keys=True))
    print("[INFO] bench summary:")
    print(
        json.dumps(
            {
                "routing_status": routing_status,
                "production_readiness_status": str(readiness.get("status") or "UNKNOWN"),
                "admissible_degrade": admissible_degrade,
                "transport_route_mode": transport_route.get("mode"),
                "degrade_target_mode": transport_route.get("degrade_target_mode"),
                "mode_switch_reason": transport_route.get("mode_switch_reason") or transport_route.get("reason"),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    print("[INFO] evidence chain:")
    print(
        json.dumps(
            {
                "report_dir": str(Path(args.report_dir).resolve()),
                "files": sorted(evidence_chain.keys()),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    baseline_output = str(args.baseline_output or "").strip()
    if baseline_output:
        baseline_summary = _build_dense_baseline_summary(
            args=args,
            stats=stats,
            report=report,
            readiness=readiness,
            execution_route_summary=execution_route_summary,
        )
        _write_json(baseline_output, baseline_summary)
        print("[INFO] dense baseline written:")
        print(json.dumps({"path": str(Path(baseline_output).resolve())}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if readiness.get("status") == "GO" else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
