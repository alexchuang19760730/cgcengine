#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
p012_ab.py — P0/P1/P2 的 A/B runner（重開機後跑這一支就對了）

為什麼要有這支：三個開關都編在 engine 裡、預設全關，所以「對照臂」不是另一個 build，
而是**同一個 binary 不設 env**。A/B 的整個可信度建立在「同一個 build」這件事上 ——
換 build 就換了所有東西，比較就沒有意義（本 repo 的所有配對比較都靠這個前提）。

三臂：

| arm  | env                                                | 測什麼                     |
|------|----------------------------------------------------|----------------------------|
| ctrl | （無）                                               | 基線；同時證明「預設關」等價 |
| p0   | `CGC_EXPERT_SKIP_READRAW=1`                          | skip-load expert 不 read_raw |
| p012 | `CGC_EXPERT_SKIP_READRAW=1` + `CGC_POOL_MADVISE=2`   | 再疊 fill 前／evict 時丟頁   |

每臂都記 swap 前後（`sysctl vm.swapusage`）＋ 把子行程的完整輸出 tee 到 `<out>/logs/<arm>_r<n>/`，
這樣事後才能用 `miss_attr_gate.py` 驗「三條判據」。

用法（重開機、機器冷、沒別人在跑之後）：

    python3 scripts/check/p012_ab.py                      # 1 輪 3 臂，decode cell（prod25），reps=3
    python3 scripts/check/p012_ab.py --profile prod-new --extra-env CGC_SERVER_MTP=1

⚠ **用 prod-new 必須疊 `CGC_SERVER_MTP=1`**：prod-new 的 MTP=0 分支會把 MODEL 換成
  `Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`（**沒有 nextn head 的另一個 checkpoint**，
  `run_server.sh:153-161`）⇒ 那不是交付模型，三臂都換了模型就完全不能比。
  疊了 `CGC_SERVER_MTP=1` 才會回到 Nail checkpoint（與 prod25 同一個檔）。
    python3 scripts/check/p012_ab.py --rounds 2           # ABBA（第 2 輪反序）
    python3 scripts/check/p012_ab.py --dry-run            # 只看會跑什麼（零 GPU）
    python3 scripts/check/p012_ab.py --self-test          # 零 GPU 自檢

跑完接著驗三條判據（三條同時成立才算修好，單條不算）：

    python3 scripts/check/miss_attr_gate.py \\
        --row ctrl=<out>/logs/ctrl_r1/decode-delivery.out \\
        --row p0=<out>/logs/p0_r1/decode-delivery.out \\
        --row p012=<out>/logs/p012_r1/decode-delivery.out

⚠ 開 P0 的第一件事不是看數字，是**看輸出還是不是正常文字**：layer-0 的 guard 若失效，
整網會輸出垃圾且完全不報錯（`CGC_SERVER_SKIP0` 預設 0 ⇒ 交付配置目前安全）。
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

# 只有這兩個 env 會被注入 —— 白名單是刻意的：多帶一個 knobs 進去，A/B 就少一個自由度可控。
ARM_ENV = {
    "ctrl": {},
    "p0":   {"CGC_EXPERT_SKIP_READRAW": "1"},
    "p012": {"CGC_EXPERT_SKIP_READRAW": "1", "CGC_POOL_MADVISE": "2"},
}
ALLOWED_KEYS = {"CGC_EXPERT_SKIP_READRAW", "CGC_POOL_MADVISE"}
ORDER = ["ctrl", "p0", "p012"]

SWAP_RE = re.compile(r"used\s*=\s*([0-9]+(?:\.[0-9]+)?)M")


# ---------------------------------------------------------------- 零 GPU 的純函式

def parse_swap_mib(text):
    """`sysctl vm.swapusage` → used MiB（float）。抓不到回 None，不要憑空給 0。"""
    if not text:
        return None
    m = SWAP_RE.search(text)
    if not m:
        return None
    return float(m.group(1))


def swap_now():
    """讀一次 swap；讀不到（非 macOS / sysctl 被擋）回 None，欄位顯示 UNREAD 而不是 0。"""
    try:
        p = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True)
        return parse_swap_mib(p.stdout)
    except Exception:
        return None


def arm_env(arm):
    if arm not in ARM_ENV:
        raise KeyError("unknown arm %r (have %s)" % (arm, list(ARM_ENV)))
    return dict(ARM_ENV[arm])


def parse_extra_env(items):
    """`["K=V", "K=V"]` → dict。缺 `=` 直接 SystemExit：拼錯的 env 會靜默變成沒設。"""
    out = {}
    for it in items or []:
        for kv in it.split(","):
            kv = kv.strip()
            if not kv:
                continue
            if "=" not in kv:
                raise SystemExit("--extra-env wants K=V, got %r" % kv)
            k, v = kv.split("=", 1)
            out[k] = v
    return out


def arm_order(rounds):
    """第 1 輪正序、第 2 輪反序 —— 3 臂的 ABBA。單輪沒有 drift 保護，多輪才有。"""
    out = []
    for r in range(1, max(1, rounds) + 1):
        seq = list(ORDER) if r % 2 == 1 else list(reversed(ORDER))
        for a in seq:
            out.append((r, a))
    return out


def extract(rec):
    """prod_profile.py 的 json → 這臂的數字。缺任何東西都回 None，不補 0。"""
    if not isinstance(rec, dict):
        return None
    if rec.get("refused"):
        return {"refused": rec["refused"], "ts": None, "sd": None,
                "launch": None, "worst": None}
    axes = rec.get("axes") or []
    if not axes:
        return None
    a0 = axes[0]
    # prod_profile.py 把 "refused" 放在 **axis 那一層**（不是最外層），而且拒跑那一臂
    # 根本不會有 rows。漏掉這一層會把「機器太熱所以沒跑」顯示成「跑了但沒數字」。
    if a0.get("refused"):
        return {"refused": a0["refused"], "ts": None, "sd": None,
                "launch": None, "worst": None}
    rows = a0.get("rows") or []
    if not rows:
        return {"refused": "no row", "ts": None, "sd": None, "launch": None, "worst": None}
    r0 = rows[0]
    v = a0.get("verdict") or {}
    return {"ts": r0.get("t/s"), "sd": r0.get("±"),
            "refused": None,
            "launch": (a0.get("thermal") or {}).get("launch", {}).get("label")
                      or v.get("launch"),
            "worst": (a0.get("thermal") or {}).get("worst", {}).get("label")
                     or v.get("worst"),
            "build": r0.get("build_commit")}


# ---------------------------------------------------------------- 執行

def run_arm(arm, rnd, out, profile, reps, dry_run, extra_env=None):
    tag = "%s_r%d" % (arm, rnd)
    logd = out / "logs" / tag
    js = out / ("%s.json" % tag)
    env = dict(os.environ)
    env.update(extra_env or {})
    env.update(arm_env(arm))   # 手臂的開關永遠贏：extra_env 不能蓋掉 A/B 的自變數
    cmd = [sys.executable, str(HERE / "prod_profile.py"),
           "--profile", profile, "--axes", "decode", "--no-ref",
           "--reps", str(reps), "--json", str(js), "--log-dir", str(logd)]
    rec = {"arm": arm, "round": rnd, "tag": tag, "cmd": " ".join(cmd[1:]),
           "env": arm_env(arm), "extra_env": extra_env or {}, "profile": profile,
           "json": str(js), "log_dir": str(logd)}
    if dry_run:
        rec["dry"] = True
        return rec
    logd.mkdir(parents=True, exist_ok=True)
    rec["swap_before"] = swap_now()
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, env=env)
    rec["wall_s"] = round(time.time() - t0, 1)
    rec["rc"] = p.returncode
    (logd / "runner.out").write_text(p.stdout or "")
    (logd / "runner.err").write_text(p.stderr or "")
    rec["swap_after"] = swap_now()
    if js.exists():
        try:
            got = extract(json.loads(js.read_text()))
        except Exception as e:
            got = {"refused": "json unreadable: %s" % e}
        rec.update(got or {"refused": "no axes in json"})
    else:
        rec["refused"] = "no json written"
    return rec


def budget_note(out):
    """提醒：8 GiB pool 在 16 GB 上是靜態超訂，P0 也救不回來（10262 + 8192 = 18454 > 16384）。"""
    return ("    ⚠ 8 GiB pool 在 16 GB 上靜態超訂：model resident 10262 + pool 8192 = 18454 > 16384\n"
            "      （P0 之後仍超訂 2070 MiB）⇒ 這組 A/B 的數字活在壓縮/swap 之上，\n"
            "       只能三臂互比，不可當交付錨點。要合法得把 pool 降到 ≤ 6122 MiB。\n")


def report(recs, out):
    lines = ["", "=== P0/P1/P2 A/B ===",
             "    %-6s %-3s %-9s %-8s %-9s %-9s %s" %
             ("arm", "r", "t/s", "±", "launch", "worst", "Δswap MiB")]
    for r in recs:
        ts = r.get("ts")
        sd = r.get("sd")
        sb, sa = r.get("swap_before"), r.get("swap_after")
        dsw = "UNREAD" if (sb is None or sa is None) else "%+.1f" % (sa - sb)
        lines.append("    %-6s %-3s %-9s %-8s %-9s %-9s %s" % (
            r["arm"], r.get("round", "-"),
            "REFUSED" if r.get("refused") else ("%.2f" % ts if ts is not None else "NA"),
            "%.2f" % sd if sd is not None else "-",
            r.get("launch") or "-", r.get("worst") or "-", dsw))
        if r.get("refused"):
            lines.append("           ^ %s" % r["refused"])
    lines.append("")
    lines.append("驗三條判據（三條同時成立才算修好）：")
    rows = " ".join("--row %s=%s/decode-delivery.out" % (r["tag"], r.get("log_dir", "<out>/logs/" + r["tag"]))
                    for r in recs if not r.get("dry"))
    if rows:
        lines.append("    python3 scripts/check/miss_attr_gate.py " + rows)
    lines.append("")
    lines.append("先看這個，再看 t/s：P0 臂的輸出是不是正常文字（layer-0 guard 失效 = 輸出垃圾且完全不報錯）。")
    lines.append("    grep -h 'CGC-SHAPE\\|final stats' %s/logs/*/*.out | head" % out)
    lines.append(budget_note(out))
    return "\n".join(lines)


# ---------------------------------------------------------------- self-test

def self_test():
    p = f = 0

    def check(name, got, want):
        nonlocal p, f
        if got == want:
            p += 1
            print("  ok   %s" % name)
        else:
            f += 1
            print("  FAIL %s (got %r want %r)" % (name, got, want))

    # 1-3 swap 解析
    check("swap 解析（真實格式）",
          parse_swap_mib("vm.swapusage: total = 7168.00M  used = 6296.12M  free = 871.88M  (encrypted)"),
          6296.12)
    check("swap 解析（used=0）",
          parse_swap_mib("total = 1024.00M  used = 0.00M  free = 1024.00M"), 0.0)
    check("swap 解析失敗回 None（不補 0）", parse_swap_mib("garbage"), None)
    check("swap 空字串回 None", parse_swap_mib(""), None)
    check("swap None 輸入回 None", parse_swap_mib(None), None)

    # 5-7 arm env
    check("ctrl 不帶任何 env", arm_env("ctrl"), {})
    check("p0 只有 SKIP_READRAW", arm_env("p0"), {"CGC_EXPERT_SKIP_READRAW": "1"})
    check("p012 帶兩個", arm_env("p012"),
          {"CGC_EXPERT_SKIP_READRAW": "1", "CGC_POOL_MADVISE": "2"})
    try:
        arm_env("nope")
        raised = False
    except KeyError:
        raised = True
    check("arm_env('nope') 真的 raise", raised, True)

    # 9 白名單
    bad = [a for a, e in ARM_ENV.items() if not set(e) <= ALLOWED_KEYS]
    check("所有注入的 env 都在白名單內", bad, [])

    # 10-11 順序（ABBA）
    check("單輪順序", arm_order(1), [(1, "ctrl"), (1, "p0"), (1, "p012")])
    check("兩輪反序（ABBA）",
          arm_order(2),
          [(1, "ctrl"), (1, "p0"), (1, "p012"),
           (2, "p012"), (2, "p0"), (2, "ctrl")])
    check("rounds=0 仍跑一輪", len(arm_order(0)), 3)

    # extra-env 解析
    check("extra-env 解析", parse_extra_env(["CGC_SERVER_MTP=1"]), {"CGC_SERVER_MTP": "1"})
    check("extra-env 逗號分隔", parse_extra_env(["A=1,B=2"]), {"A": "1", "B": "2"})
    check("extra-env 空", parse_extra_env([]), {})
    try:
        parse_extra_env(["NOEQUALS"])
        raised2 = False
    except SystemExit:
        raised2 = True
    check("extra-env 缺 = 直接 SystemExit（不靜默吞掉）", raised2, True)

    # 13-16 extract
    fake = {"axes": [{"rows": [{"t/s": 12.57, "±": 2.26, "build_commit": "abc"}],
                      "thermal": {"launch": {"label": "NOMINAL"}, "worst": {"label": "NOMINAL"}},
                      "verdict": {}}]}
    e = extract(fake)
    check("extract t/s", e["ts"], 12.57)
    check("extract ±", e["sd"], 2.26)
    check("extract launch/worst/build", (e["launch"], e["worst"], e["build"]),
          ("NOMINAL", "NOMINAL", "abc"))
    check("extract axis 層 refused 帶出原因",
          extract({"axes": [{"refused": "usable 12%"}]})["refused"], "usable 12%")
    check("extract 最外層 refused 也帶出原因",
          extract({"refused": "machine still MODERATE"})["refused"], "machine still MODERATE")
    check("extract 空 axes 回 None", extract({"axes": []}), None)
    check("extract refused 帶出原因",
          extract({"axes": [{"refused": "usable 12%"}]})["refused"], "usable 12%")
    check("extract 非 dict 回 None", extract("nope"), None)
    check("extract 沒有 rows → refused 而非 0",
          extract({"axes": [{"rows": []}]})["refused"], "no row")
    check("extract 沒有 rows 時 ts 是 None 不是 0",
          extract({"axes": [{"rows": []}]})["ts"], None)

    # 21 dry run 不碰 GPU
    r = run_arm("p012", 1, Path("/tmp/p012_selftest_out"), "prod25", 3, dry_run=True)
    check("dry-run 不產生 json", r.get("dry"), True)
    check("dry-run 命令含 --no-ref", "--no-ref" in r["cmd"], True)
    check("dry-run 命令含 profile（server profile 名，不是 arm 名）",
          "--profile prod25" in r["cmd"], True)

    print("\n  self-test: %d passed, %d failed" % (p, f))
    return 0 if f == 0 else 1


def main():
    ap = argparse.ArgumentParser(description="P0/P1/P2 A/B runner（同一個 build，只有 env 不同）")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    # ⚠ 這裡要的是 **server profile 名**（prod25），不是 arm 名。餵 `prod25-stream` 會在
    #   run_server.sh 的 CGC_SERVER_PROFILE 枚舉上直接報錯（它只收 prod25/prefill250/prod-new…）；
    #   prod25-stream 是 llama_bench_matrix 的 arm，只能出現在 `--ref-arm`。
    ap.add_argument("--profile", default="prod25")
    ap.add_argument("--extra-env", action="append", default=[], metavar="K=V",
                    help="額外的環境變數（可重複或用逗號分隔）。用 prod-new 時要給 "
                         "CGC_SERVER_MTP=1，否則 MODEL 會換成沒有 nextn head 的另一個 checkpoint。")
    ap.add_argument("--out", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        return self_test()

    out = Path(a.out) if a.out else Path("/tmp/p012_ab_%s" % time.strftime("%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    extra_env = parse_extra_env(a.extra_env)
    print("=== P0/P1/P2 A/B  out=%s  profile=%s  extra=%s  reps=%d  rounds=%d ==="
          % (out, a.profile, extra_env or "-", a.reps, a.rounds))
    print(budget_note(out))
    recs = []
    for rnd, arm in arm_order(a.rounds):
        print("\n--- arm %s (round %d) env=%s ---" % (arm, rnd, arm_env(arm) or "(none)"), flush=True)
        r = run_arm(arm, rnd, out, a.profile, a.reps, a.dry_run, extra_env)
        recs.append(r)
        print("    %s" % (("rc=%s t/s=%s" % (r.get("rc"), r.get("ts")))
                          if not r.get("dry") else "dry-run: " + r["cmd"][:100]), flush=True)
    print(report(recs, out))
    (out / "summary.json").write_text(json.dumps(recs, ensure_ascii=False, indent=2))
    print("json -> %s/summary.json" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
