#!/usr/bin/env python3
"""唯一 gate 規格的漂移探測器（MEASUREMENT_CONTRACT §3）。

WHY THIS EXISTS
---------------
`docs/MEASUREMENT_CONTRACT_2026-09-25.md` §3 定義了「唯一的 gate 規格」。
但規格寫得再清楚，只要**沒有一支程式去比對實作**，它就會漂移 ——
2026-09-25 的實際狀態就是活證：同一個 commit 內三處 swap 門檻各不相同
（`arm_two_pass` 起跑 1024 / `lane_watchdog` 止血 3072 / `budget_gate` 超訂即拒），
而 `--reps` 預設是 1（讓散度判據永不觸發）。

⇒ 這支程式把「規格」變成「可機檢的期望」：
   **現在是紅的**（§3 尚未接線）正是它該有的樣子 —— 接線後自動轉綠。

它只讀檔、不改檔，所以可以安全地在任何時候跑。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

FILES = {
    "two_pass": "scripts/check/arm_two_pass.py",
    "watchdog": "scripts/check/lane_watchdog.py",
    "harness": "scripts/check/harness.py",
    "matrix": "scripts/check/llama_bench_matrix.py",
    "commit_bench": "scripts/check/commit_bench.py",
}

# §3 的 canonical 值（改這裡＝改規格，必須同時改 docs/MEASUREMENT_CONTRACT §3）
CANON_REPS = 3
CANON_LAUNCH_SWAP_MB = 2048.0
CANON_POOL_BYTES = 8589934592
# 只比對「旗標名」，不比對值 —— 各檔把它寫成 `"--prompt", prompt` 或 `--prompt 2048` 都算合格。
CANON_FLAGS = ("--prompt", "--gen", "--depths", "--warm-skip", "--ctx-size")


def _search(text: str, pattern: str, cast=float):
    m = re.search(pattern, text)
    return cast(m.group(1)) if m else None


def check_all(files: dict[str, str]) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []

    def add(name: str, ok: bool, detail: str):
        out.append((name, bool(ok), detail))

    tp = files.get("two_pass", "")
    wd = files.get("watchdog", "")
    hs = files.get("harness", "")
    mx = files.get("matrix", "")

    # 1) 起跑 swap 門檻：arm_two_pass 的 --max-swap-mb 預設必須 = lane_watchdog 的 LAUNCH_SWAP_KILL
    tp_swap = _search(tp, r'"--max-swap-mb"[^)]*?default=([0-9.]+)')
    wd_launch = _search(wd, r"LAUNCH_SWAP_KILL\s*=\s*([0-9.]+)")
    add("起跑 swap 門檻對齊（arm_two_pass == lane_watchdog.LAUNCH_SWAP_KILL）",
        tp_swap is not None and wd_launch is not None and tp_swap == wd_launch == CANON_LAUNCH_SWAP_MB,
        f"arm_two_pass={tp_swap} / lane_watchdog={wd_launch} / 期望 {CANON_LAUNCH_SWAP_MB}")

    # 2) reps 預設 = 3（否則 judge_artifact 的散度判據永不觸發）
    tp_reps = _search(tp, r'"--reps"[^)]*?default=([0-9]+)', int)
    hs_reps = _search(hs, r"_BENCH_DEFAULTS\s*=\s*dict\([^)]*?reps=([0-9]+)", int)
    add(f"reps 預設 = {CANON_REPS}（arm_two_pass 與 harness）",
        tp_reps == hs_reps == CANON_REPS,
        f"arm_two_pass={tp_reps} / harness={hs_reps}")

    # 3) G1 的看門狗偵測必須包含現行名字
    pat = _search(tp, r'_procs_matching\(\[([^\]]*)\]\)', str) or ""
    has_daemon = "watchdog_daemon" in tp or "lane_watchdog" in tp
    add("G1 看門狗偵測含現行名字（watchdog_daemon / lane_watchdog）",
        has_daemon,
        f"目前 G1 只查 {'auto_bench_watchdog' if 'auto_bench_watchdog' in tp else '?'}")

    # 4) budget_gate 必須接進 llama_bench_matrix
    add("llama_bench_matrix.py 已接 budget_gate.sh",
        "budget_gate.sh" in mx,
        "找到 budget_gate.sh 引用" if "budget_gate.sh" in mx else "沒有 source budget_gate.sh")

    # 5) pool = 8 GiB（cell 的一部分，不得為通過閘門而縮）
    tp_pool = _search(tp, r'CGC_EXPERT_CACHE_BYTES"\s*:\s*"([0-9]+)"', int)
    add(f"pool = {CANON_POOL_BYTES}（8 GiB）", tp_pool == CANON_POOL_BYTES, f"arm_two_pass PIN={tp_pool}")

    # 6) canonical cell 形狀必須在兩支 runner 裡都出現
    for key, label in (("commit_bench", "commit_bench.py"), ("two_pass", "arm_two_pass.py")):
        txt = files.get(key, "")
        missing = [f for f in CANON_FLAGS if f not in txt]
        add(f"canonical cell 旗標齊全於 {label}", not missing, f"缺 {missing}" if missing else "齊全")

    # 7) MTP 必須 ABSENT（= off）
    add("MTP 必須 ABSENT（PIN_ABSENT 含 CGC_SERVER_MTP）",
        "PIN_ABSENT" in tp and "CGC_SERVER_MTP" in tp,
        "PIN_ABSENT 缺 CGC_SERVER_MTP" if "PIN_ABSENT" not in tp else "ok")

    return out


def run(root: Path) -> int:
    files = {}
    for key, rel in FILES.items():
        p = root / rel
        files[key] = p.read_text(encoding="utf-8") if p.exists() else ""
    rows = check_all(files)
    width = max(len(n) for n, _, _ in rows)
    for name, ok, detail in rows:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<{width}}  {detail}")
    bad = [n for n, ok, _ in rows if not ok]
    print()
    if bad:
        print(f"§3 尚未接線：{len(bad)}/{len(rows)} 項未達標（這正是 2026-09-25 的現狀）")
        print("⇒ 接線後本檢查自動轉綠；規格見 docs/MEASUREMENT_CONTRACT_2026-09-25.md §3")
        return 1
    print(f"PASS — {len(rows)} 項全部與 §3 規格一致")
    return 0


def selftest() -> int:
    ok: list[tuple[str, bool]] = []

    def chk(n, c):
        ok.append((n, bool(c)))

    good_two_pass = (
        'ap.add_argument("--max-swap-mb", type=float, default=2048, help="x")\n'
        'ap.add_argument("--reps", type=int, default=3)\n'
        '"CGC_EXPERT_CACHE_BYTES": "8589934592",\n'
        'PIN_ABSENT = ["CGC_SERVER_MTP"]\n'
        '_procs_matching(["llama-bench", "llama-server", "watchdog_daemon"])\n'
        '"--prompt", "--gen", "--depths", "--warm-skip", "--ctx-size"\n')
    good = {
        "two_pass": good_two_pass,
        "watchdog": "LAUNCH_SWAP_KILL = 2048.0\nSWAP_KILL = 3072.0\n",
        "harness": "_BENCH_DEFAULTS = dict(prompt=2048, gen=128, depths='512', reps=3,\n                       warm_skip=64, ctx_size=0, batch=5632, ubatch=5632)\n",
        "matrix": '. "${REPO}/scripts/check/budget_gate.sh"\n',
        "commit_bench": "--prompt 2048 --gen 128 --depths 512 --warm-skip 64 --ctx-size 0\n",
    }
    rows = check_all(good)
    chk("全綠：符合規格的樣本全 PASS", all(o for _, o, _ in rows))
    chk("樣本檢查項數 >= 8", len(rows) >= 8)

    bad = dict(good)
    bad["two_pass"] = good_two_pass.replace("default=2048", "default=1024")
    rows = check_all(bad)
    chk("起跑 swap 不對齊 ⇒ 紅",
        any("起跑 swap" in n and not o for n, o, _ in rows))

    bad = dict(good)
    bad["two_pass"] = good_two_pass.replace("--reps\", type=int, default=3", "--reps\", type=int, default=1")
    rows = check_all(bad)
    chk("reps=1 ⇒ 紅", any(n.startswith("reps 預設") and not o for n, o, _ in rows))

    bad = dict(good)
    bad["two_pass"] = good_two_pass.replace(", \"watchdog_daemon\"", "")
    rows = check_all(bad)
    chk("缺 watchdog_daemon ⇒ 紅", any("看門狗偵測" in n and not o for n, o, _ in rows))

    bad = dict(good)
    bad["matrix"] = "# 沒有 budget gate\n"
    rows = check_all(bad)
    chk("matrix 沒接 budget_gate ⇒ 紅", any("budget_gate" in n and not o for n, o, _ in rows))

    bad = dict(good)
    bad["two_pass"] = good_two_pass.replace("8589934592", "3221225472")
    rows = check_all(bad)
    chk("pool 被縮 ⇒ 紅", any("pool =" in n and not o for n, o, _ in rows))

    for n, c in ok:
        print(f"  [{'PASS' if c else 'FAIL'}] {n}")
    print(f"\n  selftest: {sum(1 for _, c in ok if c)}/{len(ok)}")
    return 0 if all(c for _, c in ok) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    return run(Path(args.root))


if __name__ == "__main__":
    sys.exit(main())
