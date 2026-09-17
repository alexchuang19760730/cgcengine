#!/usr/bin/env python3
"""去隱私：把日誌／紀錄裡的絕對路徑、區網 IP、主機名、金鑰遮罩掉。

WHY THIS IS A SEPARATE MODULE AND NOT INLINE IN pack_evidence.py
----------------------------------------------------------------
PLAN §8 names it as its own file, and the reason is the same one that made `sft_common.py` a
separate module: masking is a POLICY that more than one thing will need (log packing today, an
evidence export tomorrow), and two inline copies drift on the first edit -- silently, because both
still produce output and neither output looks wrong.

THE CASE THAT MOTIVATED IT (PLAN §8, verbatim)
-----------------------------------------------
> `run_server.sh` 會印「連線卡」（含區網 IP 與 port）→ `sanitize.py` 必擋。

So a LAN address is not hypothetical: the server banner prints one on every start, and the packed
evidence is meant to be committed. `--self-test` therefore includes that exact shape.

WHAT IS AND IS NOT MASKED
-------------------------
  /Users/…/flashkv-devserver/…   ->  $REPO/…          (the repo root, longest match first)
  /Users/<name>/…                ->  $HOME/…          (a home path leaks the account name)
  <local hostname>               ->  $HOST            (from socket.gethostname(), not a guess)
  192.168.x.x / 10.x.x.x …       ->  $IP              (any dotted quad, so public ones too)
  sk-…, ghp_…, xox?-…            ->  $KEY             (API tokens)
  Authorization: Bearer …        ->  $TOKEN
  user:pass@host                 ->  $CRED@host       (credentials inside a URL)

MASKING MUST BE IDEMPOTENT. Running it on already-masked text has to be a no-op, or a second pass
double-masks (`$REPO/$REPO/…`) and the output depends on how many times it ran. `--self-test`
asserts `f(f(x)) == f(x)`.

Usage:
    python3 agent_harness/shared/sanitize.py --self-test
    cat some.log | python3 agent_harness/shared/sanitize.py
"""
from __future__ import annotations

import argparse
import getpass
import re
import socket
import sys
from pathlib import Path

REPO_ENV = "$REPO"
HOME_ENV = "$HOME"


def _repo_root() -> str:
    """agent_harness/shared/sanitize.py -> the repository root, without assuming a name."""
    return str(Path(__file__).resolve().parents[2])


def _home() -> str:
    return str(Path.home())


def _host() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return ""


def sanitize_text(text: str, *, repo_root: str | None = None, home: str | None = None,
                  host: str | None = None, user: str | None = None) -> str:
    """The policy, as one pure function so it can be tested without touching a file."""
    repo = (repo_root if repo_root is not None else _repo_root()).rstrip("/")
    home = (home if home is not None else _home()).rstrip("/")
    host = host if host is not None else _host()
    user = user if user is not None else getpass.getuser()

    # Longest first: the repo lives under $HOME, so masking $HOME first would leave "$HOME/repo".
    for real, tag in ((repo, REPO_ENV), (home, HOME_ENV)):
        if real:
            text = text.replace(real, tag)
    if home and user:
        # A bare account name still identifies a person even with no path around it.
        text = text.replace(f"/Users/{user}", HOME_ENV).replace(f"/home/{user}", HOME_ENV)
    if host:
        text = re.sub(rf"(?<![\w.-]){re.escape(host)}(?![\w.-])", "$HOST", text)

    # Credentials inside a URL, before the generic dotted-quad rule can see the host part.
    text = re.sub(r"\b([A-Za-z][\w+.-]*://)([^/\s:@]+):([^/\s@]+)@", r"\1$CRED@", text)
    # Any dotted quad. \b does not help here (dots are non-word), so spell out the boundaries.
    text = re.sub(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])", "$IP", text)
    text = re.sub(r"\bsk-[A-Za-z0-9_\-]{6,}", "$KEY", text)
    text = re.sub(r"\bgh[pousr]_[A-Za-z0-9]{6,}", "$KEY", text)
    text = re.sub(r"\b(xox[baprs])-[A-Za-z0-9\-]{6,}", "$KEY", text)
    text = re.sub(r"(?i)\b(bearer)\s+[A-Za-z0-9._\-]{8,}", r"\1 $TOKEN", text)
    # A machine-local uid still points at an account that may exist elsewhere.
    text = re.sub(rf"(?<![\w-]){re.escape(user)}(?![\w-])", "$USER", text) if user else text
    return text


CASES: list[tuple[str, str]] = [
    # (input, expected substring that must appear / must NOT appear)
    ("open $REPO/src/llama.cpp/build/bin/llama-server", "$REPO/src/llama.cpp/build/bin/llama-server"),
    ("curl http://192.168.101.90:8080/v1/models", "$IP:8080"),
    ("[連線卡] host=192.168.3.24 port=8080", "$IP"),
    ("api_key=sk-cgc-edge-key-abcdef", "$KEY"),
    ("Authorization: Bearer abcdef123456", "$TOKEN"),
    ("https://user:hunter2@example.internal/x", "$CRED@example.internal"),
    ("CGC-S1: SLOT-OWNER graph=0 il=0 sum=10153", "sum=10153"),
    ("CGC-HOOK: ctx=0x12e80c000 il=0 ntok=2", "0x12e80c000"),
    ("wait=441.02 gpu_busy_sum=749.74", "441.02"),
]

FORBIDDEN = [r"/Users/[A-Za-z]", r"\b\d{1,3}(?:\.\d{1,3}){3}\b", r"\bsk-[A-Za-z0-9]", r"Bearer [A-Za-z0-9]"]


def self_test() -> int:
    fails: list[str] = []
    total = [0]

    def check(ok: bool, label: str, detail: str = "") -> None:
        total[0] += 1
        print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + (f"   {detail}" if detail and not ok else ""))
        if not ok:
            fails.append(label)

    print("=== 1. 每一類都要被遮到，而數字／指標不能被誤遮 ===")
    for src, must in CASES:
        got = sanitize_text(src, repo_root="/Users/nobody/repo", home="/Users/nobody",
                            host="nobody-mbp.local", user="nobody")
        check(must in got, f"保留：{must[:46]}", f"got={got[:80]}")
        for pat in FORBIDDEN:
            if re.search(pat, got):
                check(False, f"洩漏：{pat}", f"in {got[:90]}")

    print()
    print("=== 2. 冪等（跑兩次 == 跑一次）—— 否則輸出取決於執行了幾次 ===")
    for src, _ in CASES:
        once = sanitize_text(src, repo_root="/Users/nobody/repo", home="/Users/nobody",
                             host="nobody-mbp.local", user="nobody")
        twice = sanitize_text(once, repo_root="/Users/nobody/repo", home="/Users/nobody",
                              host="nobody-mbp.local", user="nobody")
        check(once == twice, f"冪等：{src[:40]}", f"{once[:60]} != {twice[:60]}")

    print()
    print("=== 3. 真實的連線卡形狀（PLAN §8 指名的那一個）===")
    banner = ("[連線卡] server ready on http://192.168.101.90:8080  api_key=sk-cgc-edge-key-zzzz\n"
              "         logged to /Users/alexchuang/Documents/flashkv-devserver/Backup/cgc_logs/x.log")
    out = sanitize_text(banner, repo_root="/Users/alexchuang/Documents/flashkv-devserver",
                        home="/Users/alexchuang", host="alexchuang-mbp.local", user="alexchuang")
    print("     ", out.replace("\n", "\n      "))
    check("$IP:8080" in out and "$KEY" in out and "$REPO/Backup" in out,
          "連線卡三處（IP／金鑰／絕對路徑）全遮")

    print()
    if fails:
        print(f"  {len(fails)}/{total[0]} FAILED: {fails}")
        return 1
    print(f"  all {total[0]} checks passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--in-place", type=Path, default=None, help="sanitize this file to stdout")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    text = args.in_place.read_text(encoding="utf-8", errors="replace") if args.in_place else sys.stdin.read()
    sys.stdout.write(sanitize_text(text))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
