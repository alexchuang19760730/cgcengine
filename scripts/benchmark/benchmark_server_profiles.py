#!/usr/bin/env python3
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


BAD_MARKERS = ("<|user|>", "<|output|>", "<think>", "</think>")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run repeated fixed-payload server benchmarks via replay profiles"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["qa-zh", "longform-zh"],
        choices=["qa-zh", "longform-zh"],
    )
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--model", default="test")
    parser.add_argument("--json", action="store_true", help="Print aggregate JSON summary")
    return parser.parse_args()


def mean_or_none(values):
    vals = [float(v) for v in values if isinstance(v, (int, float))]
    if not vals:
        return None
    return sum(vals) / len(vals)


def median_or_none(values):
    vals = [float(v) for v in values if isinstance(v, (int, float))]
    if not vals:
        return None
    return statistics.median(vals)


def run_profile(script_path, profile, args):
    cmd = [
        sys.executable,
        str(script_path),
        "--base-url",
        args.base_url,
        "--profile",
        profile,
        "--timeout",
        str(args.timeout),
        "--model",
        args.model,
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(proc.stdout)


def validate_result(result):
    errors = []
    if result.get("finish_reason") != "stop":
        errors.append(f"finish_reason={result.get('finish_reason')!r}")

    content = str(result.get("content") or "")
    leaked = [marker for marker in BAD_MARKERS if marker in content]
    if leaked:
        errors.append("marker_leak=" + ",".join(leaked))

    if result.get("decode_tps") is None:
        errors.append("missing decode_tps")

    return errors


def main():
    args = parse_args()
    replay_script = Path(__file__).resolve().parents[1] / "check" / "replay_server_profile.py"
    all_runs = []
    grouped = {profile: [] for profile in args.profiles}

    for profile in args.profiles:
        for idx in range(1, args.iterations + 1):
            result = run_profile(replay_script, profile, args)
            errors = validate_result(result)
            if errors:
                raise RuntimeError(f"{profile} iteration {idx} failed validation: {'; '.join(errors)}")
            grouped[profile].append(result)
            all_runs.append({"profile": profile, "iteration": idx, "result": result})
            print(
                f"[run {idx}/{args.iterations}] {profile} "
                f"finish={result.get('finish_reason')} "
                f"decode_tps={result.get('decode_tps')} "
                f"prompt_tps={result.get('prompt_tps')} "
                f"completion_tokens={result.get('completion_tokens')} "
                f"draft_accept_pct={result.get('draft_accept_pct')}"
            )

    summary = []
    for profile in args.profiles:
        runs = grouped[profile]
        decode_tps = [item.get("decode_tps") for item in runs]
        prompt_tps = [item.get("prompt_tps") for item in runs]
        completion_tokens = [item.get("completion_tokens") for item in runs]
        accept_pct = [item.get("draft_accept_pct") for item in runs]
        finish_reasons = [item.get("finish_reason") for item in runs]
        summary.append(
            {
                "profile": profile,
                "iterations": len(runs),
                "finish_reasons": finish_reasons,
                "decode_tps_mean": mean_or_none(decode_tps),
                "decode_tps_median": median_or_none(decode_tps),
                "prompt_tps_mean": mean_or_none(prompt_tps),
                "completion_tokens_mean": mean_or_none(completion_tokens),
                "draft_accept_pct_mean": mean_or_none(accept_pct),
            }
        )

    if args.json:
        print(json.dumps({"base_url": args.base_url, "summary": summary, "runs": all_runs}, ensure_ascii=False, indent=2))
    else:
        print("")
        print("== summary ==")
        for item in summary:
            print(
                f"{item['profile']}: "
                f"decode_tps_mean={item['decode_tps_mean']} "
                f"decode_tps_median={item['decode_tps_median']} "
                f"prompt_tps_mean={item['prompt_tps_mean']} "
                f"completion_tokens_mean={item['completion_tokens_mean']} "
                f"draft_accept_pct_mean={item['draft_accept_pct_mean']} "
                f"finish_reasons={item['finish_reasons']}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
