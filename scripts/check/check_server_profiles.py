#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path


BAD_MARKERS = ("<|user|>", "<|output|>", "<think>", "</think>")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run fixed replay-profile regression checks against llama-server"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["qa-zh", "longform-zh"],
        choices=["qa-zh", "longform-zh"],
    )
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--model", default="test")
    parser.add_argument("--json", action="store_true", help="Print aggregate JSON summary")
    return parser.parse_args()


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
    replay_script = Path(__file__).with_name("replay_server_profile.py")
    summaries = []
    failed = False

    for profile in args.profiles:
        result = run_profile(replay_script, profile, args)
        errors = validate_result(result)
        summaries.append(
            {
                "profile": profile,
                "finish_reason": result.get("finish_reason"),
                "decode_tps": result.get("decode_tps"),
                "prompt_tps": result.get("prompt_tps"),
                "completion_tokens": result.get("completion_tokens"),
                "draft_accept_pct": result.get("draft_accept_pct"),
                "errors": errors,
            }
        )
        status = "PASS" if not errors else "FAIL"
        print(
            f"[{status}] {profile} "
            f"finish={result.get('finish_reason')} "
            f"decode_tps={result.get('decode_tps')} "
            f"prompt_tps={result.get('prompt_tps')} "
            f"completion_tokens={result.get('completion_tokens')} "
            f"draft_accept_pct={result.get('draft_accept_pct')}"
        )
        if errors:
            failed = True
            for err in errors:
                print(f"  - {err}")

    if args.json:
        print(json.dumps({"base_url": args.base_url, "results": summaries}, ensure_ascii=False, indent=2))

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
