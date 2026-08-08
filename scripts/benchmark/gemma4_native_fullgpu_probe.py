#!/usr/bin/env python3
import argparse
import importlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from app.edge_engine.mlx_tokenizer_shim import load as load_tokenizer  # type: ignore
from app.shared import transport_route_context as transport_route  # type: ignore


DEFAULT_MODEL = REPO_ROOT / "models" / "gemma-4-26B-A4B-it-qat-4bit"
DEFAULT_BINARY = REPO_ROOT / "colibri" / "c" / "gemma4"
DEFAULT_OUT_DIR = REPO_ROOT / "var" / "benchmark" / "gemma4_native_fullgpu_probe"

SHORT_PROMPT = (
    "Explain in one concise paragraph why MoE expert streaming can reduce memory usage "
    "but still hurt decode latency on consumer GPUs."
)

LONG_PROMPT = (
    "We are investigating native Gemma4 full-GPU decode on Apple Silicon. "
    "Please analyze the interaction between attention compute, routed expert matmul, "
    "expert cache residency, warm and cold expert sets, and KV cache updates. "
    "Focus on why a system can show acceptable memory footprint yet still have poor TTFT "
    "and decode throughput. Include the roles of prefill cost, per-token expert routing, "
    "disk-backed expert loads, per-group quantized matmuls, and the difference between "
    "moving control decisions into the runtime versus leaving them in the outer control plane. "
    "Then summarize what measurements are needed to separate attention overhead from expert "
    "compute overhead, and why a short smoke test can miss the long-prompt stability and "
    "latency behaviors that appear only after many decode steps. "
    * 24
)


def build_prompt(kind: str) -> str:
    if kind == "short":
        return SHORT_PROMPT
    if kind == "long":
        return LONG_PROMPT
    raise ValueError(f"unknown prompt kind: {kind}")


def tokenize_prompt(model_path: Path, prompt: str) -> list[int]:
    tokenizer = load_tokenizer(model_path)
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return list(tokenizer.encode(text, add_special_tokens=False))
    return list(tokenizer.encode(prompt, add_special_tokens=False))


def write_token_file(path: Path, token_ids: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(" ".join(str(x) for x in token_ids) + "\n", encoding="utf-8")


def maybe_trim_tokens(token_ids: list[int], limit: Optional[int]) -> list[int]:
    if limit is None or limit <= 0 or len(token_ids) <= limit:
        return token_ids
    return token_ids[:limit]


def parse_output(stdout: str, stderr: str) -> dict:
    merged = stdout + "\n" + stderr
    json_line = None
    output_ids: list[int] = []
    lines = merged.splitlines()
    for line in merged.splitlines():
        if line.startswith("JSON_STANDALONE "):
            json_line = line[len("JSON_STANDALONE ") :]
    for idx, line in enumerate(lines):
        if line.strip() == "[gemma4] Output token ids:":
            for probe_line in lines[idx + 1:]:
                stripped = probe_line.strip()
                if not stripped:
                    continue
                if stripped.startswith("[gemma4]"):
                    break
                try:
                    output_ids = [int(part) for part in stripped.split() if part]
                except ValueError:
                    output_ids = []
                break
    if not json_line:
        raise RuntimeError("missing JSON_STANDALONE in gemma4 output")
    metrics = json.loads(json_line)
    metrics["output_token_ids"] = output_ids

    prefill_match = re.search(r"\[gemma4\] Prefill: ([0-9.]+)s \(([0-9.]+) tok/s for (\d+) tokens\)", merged)
    decode_match = re.search(r"\[gemma4\] Decode: ([0-9.]+)s \(([0-9.]+) tok/s for (\d+) tokens\)", merged)
    hit_match = re.search(r"\[gemma4\] Expert cache hit rate: ([0-9.]+)%  \(hit=(\d+) miss=(\d+)\)", merged)
    if prefill_match:
        metrics["prefill_line_s"] = float(prefill_match.group(1))
        metrics["prefill_line_tps"] = float(prefill_match.group(2))
    if decode_match:
        metrics["decode_line_s"] = float(decode_match.group(1))
        metrics["decode_line_tps"] = float(decode_match.group(2))
    if hit_match:
        metrics["cache_hit_percent_line"] = float(hit_match.group(1))
        metrics["cache_hits"] = int(hit_match.group(2))
        metrics["cache_misses"] = int(hit_match.group(3))
    return metrics


def write_summary(path: Path, run_id: str, model: Path, binary: Path,
                  cap: int, max_tokens: int, results: list[dict]) -> None:
    summary = {
        "run_id": run_id,
        "model": str(model),
        "binary": str(binary),
        "max_tokens": max_tokens,
        "cap": cap,
        "results": results,
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def route_preflight(prompt_text: str, model: Path, max_tokens: int) -> dict:
    os.environ.setdefault("EDGE_LOCAL_MAIN_MODEL_PATH", str(model))
    os.environ.setdefault("EDGE_LOCAL_MODEL_PATH", str(model))
    os.environ.setdefault("CGC_LOCAL_FLASHMOE_MODEL", str(model))
    route_mod = importlib.reload(transport_route)
    body = {
        "model": str(model),
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": int(max_tokens),
        "stream": True,
    }
    route = route_mod.build_transport_route_context(body)
    target_mode = str(route.get("mode") or route.get("desired_mode") or "unknown")
    low_memory_guard = bool(route.get("external_low_memory_detected")) or str(route.get("memory_pressure") or "") == "critical"
    admissibility_guard = low_memory_guard or bool(route.get("degrade_suggested"))
    degrade = admissibility_guard and target_mode in {
        route_mod.ROUTE_LAYER_SPLIT_PD,
        route_mod.ROUTE_CLOUD_PD,
        route_mod.ROUTE_CLOUD_FALLBACK,
    }
    return {
        "enabled": os.environ.get("GEMMA4_NATIVE_ROUTE_DEGRADE", "1") != "0",
        "target_mode": target_mode,
        "degrade": degrade,
        "low_memory_guard": low_memory_guard,
        "admissibility_guard": admissibility_guard,
        "reason": str(route.get("mode_switch_reason") or route.get("reason") or ""),
        "route_context": route,
    }


def decode_tokens(model_path: Path, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    tokenizer = load_tokenizer(model_path)
    text = tokenizer.decode(token_ids)
    return str(text or "")


def extract_response_text(payload: dict) -> str:
    choices = list(payload.get("choices") or [])
    if not choices:
        return ""
    message = dict(choices[0].get("message") or {})
    if message.get("content") is not None:
        return str(message.get("content") or "")
    delta = dict(choices[0].get("delta") or {})
    return str(delta.get("content") or "")


def call_edge_handoff(prompt_text: str, assistant_prefix: str, model: Path,
                      remaining_tokens: int, route_mode: str, route_reason: str,
                      route_context: dict, out_dir: Path, case_name: str) -> dict:
    handoff_url = str(
        os.environ.get("EDGE_HANDOFF_URL")
        or os.environ.get("EDGE_FIRST_PROXY_URL")
        or ""
    ).strip()
    if not handoff_url:
        return {"handoff_ok": False, "handoff_error": "missing EDGE_HANDOFF_URL"}
    if remaining_tokens <= 0:
        return {"handoff_ok": False, "handoff_error": "no remaining tokens for handoff"}
    if not handoff_url.endswith("/v1/chat/completions"):
        handoff_url = handoff_url.rstrip("/") + "/v1/chat/completions"
    messages = [{"role": "user", "content": prompt_text}]
    if assistant_prefix:
        messages.append({"role": "assistant", "content": assistant_prefix})
    extra_body = {
        "cgc_route_override": route_mode,
        "cgc_route_override_reason": route_reason or "gemma4_midrun_handoff",
        "cgc_handoff_source": "gemma4_native_fullgpu_probe",
        "cgc_acceptance_case": case_name,
    }
    if route_mode == transport_route.ROUTE_LAYER_SPLIT_PD and route_context.get("P"):
        extra_body["cgc_route_override_pivot_layer"] = int(route_context.get("P") or 0)
    body = {
        "model": str(model),
        "messages": messages,
        "max_tokens": int(remaining_tokens),
        "temperature": 0.0,
        "stream": True,
        "extra_body": extra_body,
    }
    req = urllib.request.Request(
        handoff_url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.time()
    raw_chunks: list[str] = []
    text_parts: list[str] = []
    response_status = 0
    content_type = ""
    response_headers: dict[str, str] = {}
    finish_reason = ""
    response_id = ""
    response_model = ""
    usage: dict | None = None
    handoff_error = ""
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            response_status = int(getattr(resp, "status", 200) or 200)
            content_type = str(resp.headers.get("Content-Type") or "")
            response_headers = {
                key.lower(): value for key, value in resp.headers.items()
                if key.lower().startswith("x-edge-router")
            }
            raw = resp.read().decode("utf-8", errors="replace")
            raw_chunks.append(raw)
            if "text/event-stream" in content_type:
                for line in raw.splitlines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[len("data: "):].strip()
                    if payload == "[DONE]":
                        continue
                    try:
                        event = json.loads(payload)
                    except Exception:
                        continue
                    response_id = response_id or str(event.get("id") or "")
                    response_model = response_model or str(event.get("model") or "")
                    choices = list(event.get("choices") or [])
                    if not choices:
                        continue
                    choice = dict(choices[0] or {})
                    if choice.get("finish_reason"):
                        finish_reason = str(choice.get("finish_reason") or "")
                    delta = dict(choice.get("delta") or {})
                    content = delta.get("content")
                    if content:
                        text_parts.append(str(content))
            else:
                try:
                    payload_json = json.loads(raw)
                except Exception:
                    payload_json = {}
                response_id = str(payload_json.get("id") or "")
                response_model = str(payload_json.get("model") or "")
                usage = dict(payload_json.get("usage") or {}) or None
                choices = list(payload_json.get("choices") or [])
                if choices:
                    finish_reason = str((choices[0] or {}).get("finish_reason") or "")
                text_parts.append(extract_response_text(payload_json))
    except urllib.error.HTTPError as exc:
        response_status = int(getattr(exc, "code", 0) or 0)
        content_type = str(exc.headers.get("Content-Type") or "") if exc.headers else ""
        response_headers = {
            key.lower(): value for key, value in (exc.headers.items() if exc.headers else [])
            if key.lower().startswith("x-edge-router")
        }
        raw_chunks.append(exc.read().decode("utf-8", errors="replace"))
        handoff_error = f"http {response_status}"
    handoff_text = "".join(text_parts)
    appended_text = handoff_text
    prefix_match = False
    if assistant_prefix and handoff_text.startswith(assistant_prefix):
        prefix_match = True
        appended_text = handoff_text[len(assistant_prefix):]
    tokenizer = load_tokenizer(model)
    handoff_text_tokens = len(tokenizer.encode(handoff_text, add_special_tokens=False)) if handoff_text else 0
    appended_text_tokens = len(tokenizer.encode(appended_text, add_special_tokens=False)) if appended_text else 0
    raw_path = out_dir / f"{case_name}.handoff.raw.txt"
    json_path = out_dir / f"{case_name}.handoff.json"
    raw_path.write_text("".join(raw_chunks), encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "url": handoff_url,
                "ok": not handoff_error,
                "http_status": response_status,
                "content_type": content_type,
                "headers": response_headers,
                "route_mode": route_mode,
                "route_reason": route_reason,
                "assistant_prefix": assistant_prefix,
                "assistant_prefix_chars": len(assistant_prefix),
                "assistant_prefix_match": prefix_match,
                "remaining_tokens": remaining_tokens,
                "elapsed_s": time.time() - started,
                "response_id": response_id,
                "response_model": response_model,
                "finish_reason": finish_reason,
                "usage": usage,
                "text": handoff_text,
                "text_token_estimate": handoff_text_tokens,
                "appended_text": appended_text,
                "appended_text_token_estimate": appended_text_tokens,
                "handoff_error": handoff_error,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return {
        "handoff_ok": not handoff_error,
        "handoff_http_status": response_status,
        "handoff_content_type": content_type,
        "handoff_headers": response_headers,
        "handoff_finish_reason": finish_reason,
        "handoff_response_id": response_id,
        "handoff_response_model": response_model,
        "handoff_usage": usage,
        "handoff_text": handoff_text,
        "handoff_appended_text": appended_text,
        "handoff_prefix_match": prefix_match,
        "handoff_text_token_estimate": handoff_text_tokens,
        "handoff_appended_text_token_estimate": appended_text_tokens,
        "handoff_raw_path": str(raw_path),
        "handoff_json_path": str(json_path),
        "handoff_elapsed_s": time.time() - started,
        "handoff_error": handoff_error,
    }


def run_case(binary: Path, model: Path, out_dir: Path, case_name: str, prompt_kind: str,
             prompt_text: str, token_file: Path, cap: int, max_tokens: int, extra_env: dict[str, str]) -> dict:
    preflight = route_preflight(prompt_text=prompt_text, model=model, max_tokens=max_tokens)
    route_context = dict(preflight.get("route_context") or {})
    route_context_path = out_dir / f"{case_name}.route.json"
    route_context_path.write_text(json.dumps(route_context, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if preflight.get("enabled") and preflight.get("degrade"):
        target_mode = str(preflight.get("target_mode") or "unknown")
        degrade_action = "abort-to-cloud_pd"
        if target_mode == transport_route.ROUTE_LAYER_SPLIT_PD:
            degrade_action = "abort-to-layer_split_pd"
        elif target_mode == transport_route.ROUTE_CLOUD_FALLBACK:
            degrade_action = "abort-to-cloud_fallback"
        metrics = {
            "stdout_path": None,
            "stderr_path": None,
            "route_context_path": str(route_context_path),
            "case": case_name,
            "prompt_kind": prompt_kind,
            "wall_s": 0.0,
            "env": {k: os.environ.get(k, "") for k in (
                "CGC_LOW_MEMORY_DETECTED",
                "CGC_MEMORY_PRESSURE",
                "CGC_LOW_MEMORY_REASON",
                "EDGE_LOW_MEMORY_DETECTED",
                "EDGE_MEMORY_PRESSURE",
                "EDGE_LOW_MEMORY_REASON",
                "GEMMA4_NATIVE_ROUTE_DEGRADE",
            ) if os.environ.get(k) is not None},
            "prompt_chars": len(prompt_text),
            "returncode": 0,
            "ok": True,
            "degraded": True,
            "local_executed": False,
            "degrade_action": degrade_action,
            "route_mode": target_mode,
            "route_reason": str(preflight.get("reason") or ""),
            "memory_pressure": str(route_context.get("memory_pressure") or ""),
            "external_low_memory_detected": bool(route_context.get("external_low_memory_detected")),
            "admissibility_guard": bool(preflight.get("admissibility_guard")),
            "degrade_suggested": bool(route_context.get("degrade_suggested")),
            "moe_streaming_admissible": bool(route_context.get("moe_streaming_admissible")),
            "moe_streaming_headroom_bytes": int(route_context.get("moe_streaming_headroom_bytes") or 0),
        }
        if os.environ.get("GEMMA4_ENABLE_HANDOFF_CONTINUE", "1") != "0":
            try:
                handoff_result = call_edge_handoff(
                    prompt_text=prompt_text,
                    assistant_prefix="",
                    model=model,
                    remaining_tokens=max_tokens,
                    route_mode=target_mode,
                    route_reason=str(preflight.get("reason") or degrade_action),
                    route_context=route_context,
                    out_dir=out_dir,
                    case_name=case_name,
                )
                metrics["handoff_continued"] = bool(handoff_result.get("handoff_ok"))
                metrics.update(handoff_result)
            except Exception as exc:
                metrics["handoff_continued"] = False
                metrics["handoff_error"] = str(exc)
        return metrics

    env = os.environ.copy()
    env.update(
        {
            "SNAP": str(model),
            "GEMMA4_EXPERT_MODE": "streaming",
            "COLI_METAL_GEMMA4_PERF_MODE": "1",
            "COLI_METAL_GEMMA4_MOE_VERIFY": extra_env.get("COLI_METAL_GEMMA4_MOE_VERIFY", "0"),
            "COLI_METAL_GEMMA4_MOE_VERIFY_BLOCKS": extra_env.get("COLI_METAL_GEMMA4_MOE_VERIFY_BLOCKS", "0"),
            "CGC_ROUTE_DEGRADE_SUGGESTED": "1" if route_context.get("degrade_suggested") else "0",
            "CGC_MOE_STREAMING_ADMISSIBLE": "1" if route_context.get("moe_streaming_admissible") else "0",
            "CGC_MOE_STREAMING_HEADROOM_BYTES": str(int(route_context.get("moe_streaming_headroom_bytes") or 0)),
            "CGC_ADMISSIBILITY_REASON": str(preflight.get("reason") or ""),
            "CGC_MEMORY_PRESSURE": str(route_context.get("memory_pressure") or ""),
            "CGC_LOW_MEMORY_DETECTED": "1" if route_context.get("external_low_memory_detected") else "0",
            "GEMMA4_MIDRUN_ROUTE_TARGET": str(preflight.get("target_mode") or ""),
        }
    )
    env.update(extra_env)
    cmd = [str(binary), str(cap), str(max_tokens), "_", str(token_file)]
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(binary.parent),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.time() - started
    stdout_path = out_dir / f"{case_name}.stdout.log"
    stderr_path = out_dir / f"{case_name}.stderr.log"
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    metrics = {
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "route_context_path": str(route_context_path),
        "case": case_name,
        "prompt_kind": prompt_kind,
        "wall_s": elapsed,
        "env": {k: env[k] for k in sorted(extra_env)},
        "prompt_chars": len(prompt_text),
        "returncode": proc.returncode,
        "degraded": False,
        "local_executed": True,
        "route_mode": transport_route.ROUTE_LOCAL_FULL,
        "route_reason": "local_probe_direct",
        "memory_pressure": str(route_context.get("memory_pressure") or ""),
        "external_low_memory_detected": bool(route_context.get("external_low_memory_detected")),
    }
    if proc.returncode != 0:
        metrics["ok"] = False
        metrics["error"] = f"process exited with code {proc.returncode}"
        return metrics
    try:
        metrics.update(parse_output(proc.stdout, proc.stderr))
    except Exception as exc:
        metrics["ok"] = False
        metrics["error"] = str(exc)
        return metrics
    if metrics.get("degraded") and os.environ.get("GEMMA4_ENABLE_HANDOFF_CONTINUE", "1") != "0":
        try:
            output_ids = list(metrics.get("output_token_ids") or [])
            assistant_prefix = decode_tokens(model, output_ids)
            handoff_result = call_edge_handoff(
                prompt_text=prompt_text,
                assistant_prefix=assistant_prefix,
                model=model,
                remaining_tokens=max(int(max_tokens) - int(metrics.get("completion_tokens") or 0), 0),
                route_mode=str(metrics.get("route_mode") or preflight.get("target_mode") or transport_route.ROUTE_CLOUD_PD),
                route_reason=str(metrics.get("route_reason") or metrics.get("degrade_action") or ""),
                route_context=route_context,
                out_dir=out_dir,
                case_name=case_name,
            )
            metrics["handoff_continued"] = bool(handoff_result.get("handoff_ok"))
            metrics.update(handoff_result)
        except Exception as exc:
            metrics["handoff_continued"] = False
            metrics["handoff_error"] = str(exc)
    metrics["ok"] = True
    return metrics


def summarize_hotspots(metrics: dict) -> dict:
    decode_s = metrics.get("decode_ms", 0.0) / 1000.0
    if decode_s <= 0:
        return {}
    hotspot_pairs = [
        ("decode_expert_matmul_s", metrics.get("decode_expert_matmul_s", metrics.get("expert_matmul_s", 0.0))),
        ("decode_attention_s", metrics.get("decode_attention_s", metrics.get("attention_s", 0.0))),
        ("decode_expert_disk_s", metrics.get("decode_expert_disk_s", metrics.get("expert_disk_s", 0.0))),
        ("decode_lm_head_s", metrics.get("decode_lm_head_s", metrics.get("lm_head_s", 0.0))),
    ]
    return {
        name: {
            "seconds": value,
            "decode_share": (value / decode_s) if decode_s > 0 else None,
        }
        for name, value in hotspot_pairs
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Fixed native Gemma4 full-GPU probe for short/long prompts.")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--binary", default=str(DEFAULT_BINARY))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--cap", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--prompts", nargs="+", default=["short", "long"], choices=["short", "long"])
    ap.add_argument("--cases", nargs="+", default=["baseline", "attn_only", "fullgpu"],
                    choices=["baseline", "attn_only", "fullgpu"])
    ap.add_argument("--prompt-token-limit", type=int, default=0,
                    help="Trim each prompt to at most this many input tokens (0 = no trim).")
    args = ap.parse_args()

    model = Path(args.model)
    binary = Path(args.binary)
    out_root = Path(args.out_dir)
    run_id = f"native_fullgpu_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    all_cases = [
        ("baseline", {"COLI_METAL_GEMMA4_ATTN_DECODE": "0", "COLI_METAL_GEMMA4_MOE_BLOCK": "0"}),
        ("attn_only", {"COLI_METAL_GEMMA4_ATTN_DECODE": "1", "COLI_METAL_GEMMA4_MOE_BLOCK": "0"}),
        ("fullgpu", {
            "COLI_METAL_GEMMA4_ATTN_DECODE": "1",
            "COLI_METAL_GEMMA4_MOE_BLOCK": "1",
        }),
    ]
    cases = [item for item in all_cases if item[0] in set(args.cases)]

    results = []
    for prompt_kind in args.prompts:
        prompt_text = build_prompt(prompt_kind)
        raw_token_ids = tokenize_prompt(model, prompt_text)
        token_ids = maybe_trim_tokens(raw_token_ids, args.prompt_token_limit or None)
        prompt_label = prompt_kind if len(token_ids) == len(raw_token_ids) else f"{prompt_kind}_tok{len(token_ids)}"
        token_file = out_dir / f"{prompt_label}.tokens.txt"
        write_token_file(token_file, token_ids)
        for case_name, extra_env in cases:
            full_case_name = f"{prompt_label}_{case_name}"
            metrics = run_case(
                binary=binary,
                model=model,
                out_dir=out_dir,
                case_name=full_case_name,
                prompt_kind=prompt_label,
                prompt_text=prompt_text,
                token_file=token_file,
                cap=args.cap,
                max_tokens=args.max_tokens,
                extra_env=extra_env,
            )
            metrics["raw_prompt_tokens"] = len(raw_token_ids)
            metrics["effective_prompt_tokens"] = len(token_ids)
            metrics["prompt_token_limit"] = args.prompt_token_limit
            if metrics.get("ok"):
                metrics["hotspots"] = summarize_hotspots(metrics)
            results.append(metrics)
            write_summary(out_dir / "summary.json", run_id, model, binary, args.cap, args.max_tokens, results)

    summary_path = out_dir / "summary.json"
    write_summary(summary_path, run_id, model, binary, args.cap, args.max_tokens, results)
    print(summary_path.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
