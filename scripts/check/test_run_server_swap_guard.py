#!/usr/bin/env python3
"""Test the prod-new swap-arm gate in `scripts/run_server.sh`.

The gate is the enforcement half of "P0/P1/P2 are the default, not a manual switch": without it,
a launch that quietly drops one of the three arms runs a *different* shape and gets quoted as
prod-new. Two things must hold, and neither is checkable by reading the file once:

  1. the three states behave differently -- armed passes, disarmed refuses, disarmed + declared
     bypass continues with a warning;
  2. the gate sits AFTER the `CGC_DUMP_ENV` block. Reversed, `llama_bench_matrix.resolve()` (the
     single source of truth for every bench arm's env, which calls `run_server.sh CGC_DUMP_ENV=1`)
     would be refused by the very gate meant to protect it -- i.e. the fix would silently disable
     the measurement entry point instead of guarding the launch.

The assertions run the REAL function bodies, extracted from the script, rather than a copy: a
fixture that re-implements the guard would pass while the shipped guard was broken (the same
class of defect this repo keeps catching -- see `test_decode_window_harness.py`).

Run: python3 scripts/check/test_run_server_swap_guard.py
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SH = os.path.join(HERE, "..", "run_server.sh")
ARMS = ("CGC_EXPERT_SKIP_READRAW", "CGC_POOL_MADVISE", "CGC_B_SCHEME")


def _source() -> str:
    with open(SH, encoding="utf-8") as fh:
        return fh.read()


def _function_bodies(src: str) -> str:
    want = ("cgc_env_value", "cgc_swap_guard")
    bodies = []
    for name in want:
        m = re.search(r"^%s\(\) \{.*?^\}\n" % name, src, re.S | re.M)
        if not m:
            raise SystemExit(f"could not extract {name}() from run_server.sh")
        bodies.append(m.group(0))
    return "\n".join(bodies)


def _run(env_entries: list, extra: str = "") -> subprocess.CompletedProcess:
    """Drive the extracted guard exactly as the launcher would: SERVER_ENV array + one call."""
    script = (
        "set -euo pipefail\n"
        f"SERVER_ENV=({' '.join(env_entries)})\n"
        f"{_function_bodies(_source())}\n"
        "set +e\n"
        f"{extra} cgc_swap_guard\n"
        'echo "RC=$?"\n'
    )
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


def main() -> int:
    bad = 0
    n = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal bad, n
        n += 1
        print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
        if not ok:
            bad += 1

    armed = [f"{k}=1" for k in ARMS]

    r = _run(armed)
    check("all three armed -> launch proceeds", "RC=0" in r.stdout, r.stderr.strip()[:80])

    for drop in ARMS:
        r = _run([e for e in armed if not e.startswith(drop + "=")])
        check(f"missing {drop} -> refused (exit 2)",
              r.returncode == 2 and "RC=0" not in r.stdout, f"rc={r.returncode}")
        check(f"  ...and the message names {drop}", drop in r.stderr, "")

    r = _run([e if not e.startswith("CGC_B_SCHEME=") else "CGC_B_SCHEME=0" for e in armed])
    check("explicit 0 counts as OFF (run_server.sh transmits only non-zero)",
          r.returncode == 2, f"rc={r.returncode}")

    r = _run(["CGC_EXPERT_SKIP_READRAW=0"], extra="CGC_SWAP_GUARD=off")
    check("CGC_SWAP_GUARD=off continues, loudly, instead of refusing",
          "RC=0" in r.stdout and "WARNING" in r.stderr, r.stderr.strip().splitlines()[0][:60])

    # (2) ordering: the gate must not sit in front of the dump/resolve path.
    src = _source()
    dump_at = src.index('if [ "${CGC_DUMP_ENV:-}" = "1" ]; then')
    def_at = src.index("cgc_swap_guard() {")
    call_at = src.index("\n    cgc_swap_guard\n", def_at)
    launch_at = src.index('env "${SERVER_ENV[@]}" "$BIN"')
    check("gate defined after the dump block, called before the exec",
          dump_at < def_at < call_at < launch_at,
          f"dump={dump_at} def={def_at} call={call_at} exec={launch_at}")
    check("gate is scoped to prod-new (other profiles keep their own shape)",
          'if [ "$SERVER_PROFILE" = "prod-new" ]; then\n    cgc_swap_guard' in src)

    print()
    if bad:
        print(f"FAIL ({bad}/{n})")
        return 1
    print(f"PASS ({n}/{n})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
