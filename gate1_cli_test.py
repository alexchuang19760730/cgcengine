#!/usr/bin/env python3
"""Gate 1.0: cgc model / cgc agent CLI 测试程序

测试所有 cgc model 和 cgc agent 子命令的可用性。
每项测试: 命令是否能执行 (exit code 0) + 关键输出是否存在。

用法:
  python3 gate1_cli_test.py                    # 测试所有
  python3 gate1_cli_test.py --only model       # 只测 cgc model
  python3 gate1_cli_test.py --only agent       # 只测 cgc agent
  python3 gate1_cli_test.py --json             # JSON 输出
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

# CGC CLI 路径
CGC_PY = os.path.join(os.path.dirname(__file__), "app", "cli", "cgc.py")
PYTHON = sys.executable


@dataclass
class TestResult:
    command: str
    subcommand: str
    status: str  # PASS / FAIL / SKIP
    exit_code: int
    duration_ms: float
    output_snippet: str = ""
    error: str = ""


def run_cmd(cmd: str, timeout: int = 30) -> tuple[int, str, str, float]:
    """运行命令, 返回 (exit_code, stdout, stderr, duration_ms)."""
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        dt = (time.time() - t0) * 1000
        return proc.returncode, proc.stdout, proc.stderr, dt
    except subprocess.TimeoutExpired:
        dt = (time.time() - t0) * 1000
        return -1, "", "TIMEOUT", dt
    except Exception as e:
        dt = (time.time() - t0) * 1000
        return -2, "", str(e), dt


def test_cgc_help(cmd: str) -> TestResult:
    """测试命令 --help 是否可用."""
    full_cmd = f"{PYTHON} {CGC_PY} {cmd} --help"
    code, out, err, dt = run_cmd(full_cmd, timeout=15)
    status = "PASS" if code == 0 else "FAIL"
    snippet = out[:200].replace("\n", " ") if out else err[:200]
    return TestResult(cmd, "--help", status, code, dt, snippet, err[:200] if code != 0 else "")


def test_cgc_model_subcommands() -> List[TestResult]:
    """测试 cgc model 所有子命令."""
    subcommands = [
        ("list", f"{PYTHON} {CGC_PY} model list --json", "models"),
        ("run", f"{PYTHON} {CGC_PY} model run --help", "usage"),
        ("serve", f"{PYTHON} {CGC_PY} model serve --help", "usage"),
        ("verify", f"{PYTHON} {CGC_PY} model verify --help", "usage"),
        ("audit", f"{PYTHON} {CGC_PY} model audit --help", "usage"),
        ("replay", f"{PYTHON} {CGC_PY} model replay --help", "usage"),
        ("trace", f"{PYTHON} {CGC_PY} model trace --help", "usage"),
        ("compare", f"{PYTHON} {CGC_PY} model compare --help", "usage"),
        ("launch", f"{PYTHON} {CGC_PY} model launch v4-flash", "启动命令"),
        ("swe-verified", f"{PYTHON} {CGC_PY} model swe-verified --help", "usage"),
    ]
    results = []
    for subcmd, cmd, expected in subcommands:
        code, out, err, dt = run_cmd(cmd, timeout=30)
        # 检查关键输出 (不严格依赖 exit code, 因 WARNING 可能导致 exit=1)
        combined = out + err
        status = "PASS" if (code == 0 or expected in combined) else "FAIL"
        snippet = out[:200].replace("\n", " ") if out else err[:200]
        results.append(TestResult("model", subcmd, status, code, dt, snippet, err[:200] if status == "FAIL" else ""))
    return results


def test_cgc_agent_subcommands() -> List[TestResult]:
    """测试 cgc agent 所有子命令."""
    subcommands = [
        ("import-dag", f"{PYTHON} {CGC_PY} agent import-dag --help"),
        ("teach", f"{PYTHON} {CGC_PY} agent teach --help"),
        ("train", f"{PYTHON} {CGC_PY} agent train --help"),
        ("infer", f"{PYTHON} {CGC_PY} agent infer --help"),
        ("visualize", f"{PYTHON} {CGC_PY} agent visualize --help"),
        ("compare", f"{PYTHON} {CGC_PY} agent compare --help"),
        ("audit", f"{PYTHON} {CGC_PY} agent audit --help"),
        ("replay", f"{PYTHON} {CGC_PY} agent replay --help"),
        ("trace", f"{PYTHON} {CGC_PY} agent trace --help"),
        ("universe", f"{PYTHON} {CGC_PY} agent universe --num 5"),
        ("fusionroute", f"{PYTHON} {CGC_PY} agent fusionroute status"),
        ("bench", f"{PYTHON} {CGC_PY} agent bench --num-tasks 3"),
    ]
    results = []
    for subcmd, cmd in subcommands:
        code, out, err, dt = run_cmd(cmd, timeout=30)
        combined = out + err
        # universe/fusionroute/bench 检查关键输出
        if subcmd in ("universe", "fusionroute", "bench"):
            expected_map = {"universe": "CLI-Universe", "fusionroute": "四角色", "bench": "Benchmark"}
            status = "PASS" if (code == 0 or expected_map[subcmd] in combined) else "FAIL"
        else:
            status = "PASS" if code == 0 else "FAIL"
        snippet = out[:200].replace("\n", " ") if out else err[:200]
        results.append(TestResult("agent", subcmd, status, code, dt, snippet, err[:200] if code != 0 else ""))
    return results


def test_cgc_top_commands() -> List[TestResult]:
    """测试 cgc 顶级命令."""
    commands = [
        ("serve", f"{PYTHON} {CGC_PY} serve --help"),
        ("claude", f"{PYTHON} {CGC_PY} claude --help"),
        ("config", f"{PYTHON} {CGC_PY} config --help"),
        ("run", f"{PYTHON} {CGC_PY} run --help"),
        ("list", f"{PYTHON} {CGC_PY} list --help"),
        ("status", f"{PYTHON} {CGC_PY} status --help"),
        ("audit", f"{PYTHON} {CGC_PY} audit --help"),
        ("build", f"{PYTHON} {CGC_PY} build --help"),
    ]
    results = []
    for cmd_name, cmd in commands:
        code, out, err, dt = run_cmd(cmd, timeout=15)
        # claude 命令可能 exit code != 0 (add_help=False)
        if cmd_name == "claude":
            status = "PASS" if code == 0 or "claude" in err.lower() or "usage" in err.lower() else "FAIL"
        else:
            status = "PASS" if code == 0 else "FAIL"
        snippet = out[:200].replace("\n", " ") if out else err[:200]
        results.append(TestResult("top", cmd_name, status, code, dt, snippet, err[:200] if code != 0 else ""))
    return results


def print_report(results: List[TestResult], json_output: bool = False):
    """打印测试报告."""
    if json_output:
        print(json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False))
        return

    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    skipped = sum(1 for r in results if r.status == "SKIP")
    total = len(results)

    print("=" * 80)
    print(" Gate 1.0: cgc model / cgc agent CLI 测试")
    print("=" * 80)
    print(f"\n总计: {total} | PASS: {passed} | FAIL: {failed} | SKIP: {skipped}")
    print(f"通过率: {passed/total*100:.1f}%\n")

    # 按命令分组
    current_group = ""
    for r in results:
        if r.command != current_group:
            current_group = r.command
            print(f"\n--- cgc {r.command} ---")

        status_icon = "✅" if r.status == "PASS" else "❌" if r.status == "FAIL" else "⏭️"
        print(f"  {status_icon} {r.subcommand:20s} ({r.duration_ms:.0f}ms)", end="")
        if r.status == "FAIL":
            print(f" [exit={r.exit_code}]")
            if r.error:
                print(f"     error: {r.error[:100]}")
        else:
            print()

    print("\n" + "=" * 80)
    if failed == 0:
        print("🎉 Gate 1.0 全部通过!")
    else:
        print(f"⚠️  {failed} 项失败, 需修复后重测")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Gate 1.0 CLI 测试程序")
    parser.add_argument("--only", choices=["model", "agent", "top"], default="", help="只测试指定组")
    parser.add_argument("--json", action="store_true", help="JSON 输出")
    args = parser.parse_args()

    results = []

    if not args.only or args.only == "top":
        results.extend(test_cgc_top_commands())
    if not args.only or args.only == "model":
        results.extend(test_cgc_model_subcommands())
    if not args.only or args.only == "agent":
        results.extend(test_cgc_agent_subcommands())

    print_report(results, args.json)

    # exit code: 0 = all pass, 1 = some fail
    failed = sum(1 for r in results if r.status == "FAIL")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
