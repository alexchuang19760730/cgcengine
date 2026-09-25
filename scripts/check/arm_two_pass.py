#!/usr/bin/env python3
"""arm_two_pass.py — 每個 arm 強制跑兩輪 + 生產級硬 gate + 全參數/log 歸檔。

為什麼存在
-----------
這個 repo 的速度數字反覆被三件事污染：
  1. arm 的 env 手拼錯（漏參數、開關沒接、profile 沒 resolve）→ 跑出 7-8 t/s 還當結果；
  2. 儀器開著量速度（profiling 本身拖慢），或反過來要診斷時儀器沒開；
  3. 在髒環境（高 swap / 熱節流 / 殘留進程）裡測，跨時間數字不可比。

本腳本對「每一個」提交的 arm，固定執行兩輪，且兩輪只差儀器（單變數）：
  * clean        — 所有已知儀器強制關閉 → 這才是對外可引用的真實速度；
  * instrumented — 在完全相同的 env 上、只疊加「當前 binary 真存在」的儀器 → 採全資料。

跑之前先過生產級 gate（預設 fail-closed，不符就拒跑）：
  G1 環境潔淨：swap 水位、thermal NOMINAL、無殘留 llama-bench/server、無 watchdog；
  G2 生產釘對：prod-new 關鍵 env/scalar 必須等於 PIN（arm 顯式覆寫 → 標 deviation 不阻斷）；
  G3 兩輪等價：resolve 兩輪 env，差異集合必須 ⊆ 儀器鍵，否則拒跑。

所有 log（llama-bench stderr / json、系統快照、build 指紋、複現命令）都保留到 run-dir，
再由 arm_report_html.py 生成一份 HTML。

用法
----
  python3 scripts/check/arm_two_pass.py --arm "prod-new" \
      --run-dir /tmp/armrun --json /tmp/armrun/result.json
  # 髒環境也硬跑（只會把結果標紅、不擋）：
  python3 scripts/check/arm_two_pass.py --arm "prod-new" --allow-dirty
  # 自测：
  python3 scripts/check/arm_two_pass.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PY = sys.executable or "python3"


def _load(alias: str, filename: str):
    spec = importlib.util.spec_from_file_location(alias, str(HERE / filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────────────────────────────────────
# 儀器定義
# ─────────────────────────────────────────────────────────────────────────────
# 全部已知儀器鍵（clean 輪逐個顯式關 0；不管它在不在當前 binary）。
ALL_INSTRUMENT_KEYS = [
    "CGC_DECODE_PROFILE", "CGC_DECODE_PROFILE_ALL",
    "CGC_GPU_TIMING", "CGC_GPUOPK",
    "CGC_GPU_NODES", "CGC_GPU_NODES_MATRIX", "CGC_GPU_NODES_TRACE",
    "CGC_HOOK_PROFILE", "CGC_PHASE_DBG", "CGC_PHASE_TIMING",
    "CGC_VERIFY_OP_TIMING",
]

# instrumented 輪實際打開的儀器 = 已在「當前 binary」驗證存在的子集（位於 libggml-base）。
# 若 binary 換了、這裡可能有鍵失效 → gate G3 / 報告會如實標注，不把 null 當成功。
INSTRUMENTS_ON = {
    "CGC_DECODE_PROFILE": "1",
    "CGC_DECODE_PROFILE_ALL": "1",
    "CGC_GPU_TIMING": "1",
    "CGC_GPU_NODES_MATRIX": "1",
    "CGC_VERIFY_OP_TIMING": "1",
}


def instruments_off() -> dict:
    return {k: "0" for k in ALL_INSTRUMENT_KEYS}


# ─────────────────────────────────────────────────────────────────────────────
# 生產級 PIN（prod-new resolve 後必須匹配；以測試卡 §3 + 實際 dump 為準）
# arm 顯式覆寫其中某鍵 → 記為 declared deviation（警告、不阻斷，因為是有意的實驗）。
# ─────────────────────────────────────────────────────────────────────────────
PIN_ENV = {
    "CGC_OA_ASYNC": "1",
    "CGC_SPAC": "1",
    "CGC_SPAC_ALPHA": "0.75",
    "CGC_MM_BITIDENT": "1",
    "CGC_PREFILL_STREAM": "1",
    "CGC_GATHER_SLAB_CAP": "256",
    "CGC_EXPERT_CACHE_BYTES": "8589934592",
    "CGC_N_CB": "8",
    "CGC_WAKE_POLL_US": "15",
    "CGC_EVICTED_RING": "0",
    "CGC_SERVER_AUTO_ANCHOR": "0",
    "CGC_SERVER_DEFAULT_MARKER_STOPS": "1",
    "LLAMA_EXPERT_CACHE_ALLOW_NGL": "1",
    "LLAMA_EXPERT_CACHE_WORKERS": "8",
    "LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0": "0",
}
PIN_SCALARS = {
    "BATCH": "5632",
    "UBATCH": "5632",
    "CTX": "8192",
    "BUDGET": "8589934592",
    "LOAD_MODE": "none",
    "NGL": "99",
}
# 這些鍵在「預設關」時不應出現在 resolve env 裡；出現 = 偏離生產預設。
PIN_ABSENT = ["CGC_SERVER_MTP"]


# ─────────────────────────────────────────────────────────────────────────────
# arm 解析（複用 harness 的語法 PROFILE:!K=V;K=V）
# ─────────────────────────────────────────────────────────────────────────────
def parse_arm(spec: str) -> tuple[str, dict, set]:
    return _load("harness_mod", "harness.py")._parse_arm(spec)


def _arm_spec(profile: str, env: dict) -> str:
    """把 env dict 組回 matrix 吃的 arm spec（matrix 不要求 ! 宣告）。"""
    if not env:
        return profile
    return profile + ":" + ";".join(f"{k}={v}" for k, v in env.items())


def build_pass_env(user_env: dict, which: str) -> dict:
    """clean = user_env 後疊儀器全關（儀器設定壓過使用者，保證乾淨）；
    instrumented = user_env 後疊儀器開。使用者若在 arm 裡手動設了儀器，
    clean 仍強制關、instrumented 以 INSTRUMENTS_ON 為準。"""
    env = dict(user_env)
    if which == "clean":
        env.update(instruments_off())
    else:
        env.update(INSTRUMENTS_ON)
    return env


# ─────────────────────────────────────────────────────────────────────────────
# G1 環境潔淨
# ─────────────────────────────────────────────────────────────────────────────
def _procs_matching(patterns: list[str]) -> list[str]:
    out: list[str] = []
    try:
        run = subprocess.run(["pgrep", "-fl", "llama|auto_bench_watchdog"],
                             capture_output=True, text=True, timeout=10)
    except Exception:
        return out
    self_pid = str(os.getpid())
    for line in run.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        pid = line.split()[0]
        if pid == self_pid:
            continue
        # 不把「正在跑本檢查」的父 harness / pgrep 自己算進去
        if "arm_two_pass" in line or "arm_report_html" in line:
            continue
        if any(p in line for p in patterns):
            out.append(line)
    return out


def gate_environment(max_swap_mb: float) -> dict:
    harness = _load("harness_mod2", "harness.py")
    tp = _load("tp_g1", "thermal_pressure.py")
    mp = _load("mp_g1", "memory_pressure.py")
    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    # thermal
    try:
        t = tp.stamp()
        label = t.get("label")
        add("thermal_nominal", label == "NOMINAL", f"current={label}")
    except Exception as e:
        add("thermal_nominal", False, f"thermal probe error: {e}")

    # swap
    try:
        swap = mp.swap_used_mb()
        add("swap_level", swap is not None and swap <= max_swap_mb,
            f"swap={swap:.0f} MiB (max {max_swap_mb:.0f})")
    except Exception as e:
        add("swap_level", False, f"swap probe error: {e}")

    # 殘留進程（llama-bench / llama-server）
    residual = _procs_matching(["llama-bench", "llama-server"])
    add("no_residual", len(residual) == 0,
        "none" if not residual else "; ".join(residual[:4]))

    # watchdog
    watchdog = _procs_matching(["auto_bench_watchdog"])
    add("no_watchdog", len(watchdog) == 0,
        "none" if not watchdog else "; ".join(watchdog[:2]))

    return {"pass": all(c["pass"] for c in checks), "checks": checks}


# ─────────────────────────────────────────────────────────────────────────────
# G2 生產 PIN
# ─────────────────────────────────────────────────────────────────────────────
def gate_pin(profile: str, user_env: dict) -> dict:
    matrix = _load("matrix_g2", "llama_bench_matrix.py")
    resolved = matrix.resolve(profile, user_env)
    env, scalars = resolved["env"], resolved["scalars"]

    failures: list[str] = []
    deviations: list[str] = []

    def check(kind: str, key: str, want: str, got):
        if str(got) == str(want):
            return
        msg = f"{kind} {key}: want={want!r} got={got!r}"
        # arm 顯式要求改這個鍵 → 有意的實驗，標 deviation 不阻斷
        if key in user_env:
            deviations.append(msg + " (declared in arm)")
        else:
            failures.append(msg)

    for k, want in PIN_ENV.items():
        check("env", k, want, env.get(k))
    for k, want in PIN_SCALARS.items():
        check("scalar", k, want, scalars.get(k))
    for k in PIN_ABSENT:
        if k in env:
            if k in user_env:
                deviations.append(f"env {k}: expected ABSENT, got={env[k]!r} (declared in arm)")
            else:
                failures.append(f"env {k}: expected ABSENT in default-off, got={env[k]!r}")

    return {
        "pass": len(failures) == 0,
        "failures": failures,
        "deviations": deviations,
        "resolved_env": env,
        "resolved_scalars": scalars,
    }


# ─────────────────────────────────────────────────────────────────────────────
# G3 兩輪等價（差異 ⊆ 儀器鍵）
# ─────────────────────────────────────────────────────────────────────────────
def gate_two_pass(profile: str, user_env: dict) -> dict:
    matrix = _load("matrix_g3", "llama_bench_matrix.py")
    clean_env = build_pass_env(user_env, "clean")
    inst_env = build_pass_env(user_env, "instrumented")
    c = matrix.resolve(profile, clean_env)["env"]
    i = matrix.resolve(profile, inst_env)["env"]

    diff = {k for k in set(c) | set(i) if c.get(k) != i.get(k)}
    instr_set = set(ALL_INSTRUMENT_KEYS)
    leaked = sorted(diff - instr_set)

    # 儀器真的生效了嗎：instrumented resolve 應能看到打開的儀器鍵（=1）。
    # 注意部分儀器可能不走白名單、resolve 看不到 → 如實列出，不當成功。
    effective = sorted(k for k, v in INSTRUMENTS_ON.items() if i.get(k) == v)
    not_visible = sorted(k for k in INSTRUMENTS_ON if i.get(k) != INSTRUMENTS_ON[k])

    return {
        "pass": len(leaked) == 0,
        "diff_keys": sorted(diff),
        "non_instrument_diff": leaked,
        "instruments_effective_in_resolve": effective,
        "instruments_not_visible_in_resolve": not_visible,
    }


# ─────────────────────────────────────────────────────────────────────────────
# build 指紋（每次測試記實際 digest；含 server / impl / libllama / ggml-base / metal）
# ─────────────────────────────────────────────────────────────────────────────
def _md5_12(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def build_fingerprint() -> dict:
    bindir = ROOT / "src/llama.cpp/build/bin"
    targets = ["llama-server", "libllama-server-impl.dylib", "libllama.0.dylib",
               "libggml-base.0.dylib", "libggml-metal.0.dylib"]
    out: dict[str, str] = {}
    for t in targets:
        p = bindir / t
        try:
            real = p.resolve()
            out[t] = f"{_md5_12(real)} ({real.name})"
        except Exception as e:
            out[t] = f"(absent: {e})"
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 系統快照（複用 harness）
# ─────────────────────────────────────────────────────────────────────────────
def sys_snapshot():
    return _load("harness_snap", "harness.py")._sys_snapshot()


# ─────────────────────────────────────────────────────────────────────────────
# 跑一輪（delegate 給 llama_bench_matrix；量測邏輯不複製）
# ─────────────────────────────────────────────────────────────────────────────
def run_pass(which: str, profile: str, user_env: dict, pass_dir: Path, a: argparse.Namespace) -> dict:
    pass_env = build_pass_env(user_env, which)
    arm_spec = _arm_spec(profile, pass_env)
    logs_dir = pass_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    matrix_json = pass_dir / "matrix.json"

    cmd = [PY, str(HERE / "llama_bench_matrix.py"),
           "--arms", arm_spec,
           "--prompt", str(a.prompt), "--gen", str(a.gen),
           "--depths", str(a.depths), "--reps", str(a.reps),
           "--warm-skip", str(a.warm_skip), "--ctx-size", str(a.ctx_size),
           "--workdir", str(logs_dir), "--json", str(matrix_json)]

    print(f"\n{'─'*88}\n  PASS [{which}]  arm={arm_spec}\n{'─'*88}", flush=True)
    before = sys_snapshot()
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=str(ROOT))
    wall = time.time() - t0
    after = sys_snapshot()

    matrix_data = None
    try:
        matrix_data = json.loads(matrix_json.read_text())
    except Exception as e:
        print(f"  !! could not read matrix product {matrix_json}: {e}", flush=True)

    # 收集這輪保留的 log 檔（matrix 按 tag+shape 存 .stderr.log/.json）
    kept = sorted(str(p.relative_to(ROOT)) if str(p).startswith(str(ROOT)) else str(p)
                  for p in logs_dir.glob("*"))
    result = {
        "pass_name": which, "arm_spec": arm_spec, "rc": rc, "wall_s": round(wall, 1),
        "sys_before": before, "sys_after": after,
        "matrix_json": str(matrix_json), "kept_logs": kept,
        "matrix": matrix_data,
    }
    _print_pass_summary(result)
    return result


def _row_metric(matrix_data, kind):
    """從 matrix 產物取 pp/tg 的 (avg,std)；matrix 產物是 list[arm]。"""
    if not matrix_data:
        return None
    vals = []
    for arm in matrix_data:
        for r in arm.get("rows", []):
            is_pp = r.get("n_prompt", 0) > 0
            if (kind == "pp") == is_pp:
                vals.append((r.get("avg_ts"), r.get("stddev_ts")))
    if not vals:
        return None
    # 同形狀多 rep 已由 llama-bench 聚合，這裡取第一個（matrix 每 arm 一個 shape）
    return vals[0]


def _print_pass_summary(res: dict):
    md = res.get("matrix")
    for kind in ("pp", "tg"):
        m = _row_metric(md, kind)
        if m and m[0] is not None:
            print(f"  {kind}: {m[0]:.2f} ± {m[1]:.2f} t/s", flush=True)
    b, a = res.get("sys_before"), res.get("sys_after")
    if b and a:
        print(f"  swap: {b.get('swap_used_mb'):.0f} -> {a.get('swap_used_mb'):.0f} MiB  "
              f"thermal after={ (a.get('thermal') or {}).get('label') }", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# 污染判定（測試卡 §5.1：HEAVY 或 swap growth>500 → 標污染）
# ─────────────────────────────────────────────────────────────────────────────
def pass_pollution(pass_res: dict) -> dict:
    md = pass_res.get("matrix")
    heavy = False
    if md:
        for arm in md:
            hist = (arm.get("thermal") or {}).get("hist")
            if isinstance(hist, dict) and hist.get("HEAVY", 0) > 0:
                heavy = True
            if (arm.get("thermal") or {}).get("worst", {}).get("label") == "HEAVY":
                heavy = True
    b = (pass_res.get("sys_before") or {}).get("swap_used_mb")
    a = (pass_res.get("sys_after") or {}).get("swap_used_mb")
    growth = (a - b) if (a is not None and b is not None) else 0.0
    polluted = heavy or growth > 500
    return {"polluted": polluted, "thermal_heavy": heavy,
            "swap_growth_mb": round(growth, 0)}


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def cmd_run(args) -> int:
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    fingerprint = build_fingerprint()
    t_wall0 = time.strftime("%Y-%m-%d %H:%M:%S")

    arms_out = []
    hard_block = False

    for spec in args.arm:
        profile, user_env, overrides = parse_arm(spec)
        print(f"\n{'#'*88}\n# ARM {spec}  (profile={profile}, user_env={user_env})\n{'#'*88}",
              flush=True)

        g1 = gate_environment(args.max_swap_mb)
        g2 = gate_pin(profile, user_env)
        g3 = gate_two_pass(profile, user_env)

        for name, g in (("G1_environment", g1), ("G2_production_pin", g2),
                        ("G3_two_pass_equivalence", g3)):
            print(f"  gate {name}: {'PASS' if g.get('pass') else 'FAIL'}", flush=True)
            for line in g.get("failures", []) + g.get("non_instrument_diff", []):
                print(f"      ✗ {line}", flush=True)
            for line in g.get("deviations", []):
                print(f"      ⚠ {line}", flush=True)

        gate_pass = g1["pass"] and g2["pass"] and g3["pass"]
        if not gate_pass and not args.allow_dirty:
            print("\n  生產級 gate 未過 — 拒跑。修正環境/參數，或加 --allow-dirty（結果會標紅）。",
                  flush=True)
            hard_block = True
            arms_out.append({"arm": spec, "profile": profile, "user_env": user_env,
                             "ran": False,
                             "gates": {"G1": g1, "G2": g2, "G3": g3}})
            continue

        safe = re.sub(r"[^A-Za-z0-9._-]", "_", spec)[:60]
        arm_dir = run_dir / safe
        clean_res = run_pass("clean", profile, user_env, arm_dir / "clean", args)
        if args.cool_s > 0:
            print(f"  cool {args.cool_s}s between passes ...", flush=True)
            time.sleep(args.cool_s)
        inst_res = run_pass("instrumented", profile, user_env, arm_dir / "instrumented", args)

        arms_out.append({
            "arm": spec, "profile": profile, "user_env": user_env, "ran": True,
            "gates": {"G1": g1, "G2": g2, "G3": g3},
            "clean": clean_res, "instrumented": inst_res,
            "clean_pollution": pass_pollution(clean_res),
            "instrumented_pollution": pass_pollution(inst_res),
        })

    meta = {
        "generated_at": t_wall0,
        "machine": f"{platform.machine()} / {platform.node()}",
        "macos": platform.mac_ver()[0],
        "shape": {"prompt": args.prompt, "gen": args.gen, "depths": args.depths,
                  "reps": args.reps, "warm_skip": args.warm_skip, "ctx_size": args.ctx_size},
        "build_fingerprint": fingerprint,
        "allow_dirty": bool(args.allow_dirty),
        "reproduce": " ".join(["python3 scripts/check/arm_two_pass.py"] +
                              sum([["--arm", a] for a in args.arm], []) +
                              ["--run-dir", args.run_dir]),
    }
    result = {"meta": meta, "arms": arms_out}
    Path(args.json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_path).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n產物已寫入 {args.json_path}（含 gates + 兩輪 + 指紋 + 全部 log 路徑）", flush=True)

    if hard_block:
        return 2
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# selftest（不跑 GPU）
# ─────────────────────────────────────────────────────────────────────────────
def selftest() -> int:
    fails = []

    def expect(name, cond):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    # 1) arm 解析
    p, e, o = parse_arm("prod-new:!CGC_SPAC=0;CGC_SERVER_MTP=1")
    expect("parse profile", p == "prod-new")
    expect("parse env", e.get("CGC_SERVER_MTP") == "1" and e.get("CGC_SPAC") == "0")
    expect("parse override set", "CGC_SPAC" in o)

    # 2) clean 輪一定關儀器，即使使用者手動開
    ce = build_pass_env({"CGC_GPU_TIMING": "1"}, "clean")
    expect("clean forces instruments off", ce["CGC_GPU_TIMING"] == "0")
    expect("clean covers all known instruments",
           all(ce.get(k) == "0" for k in ALL_INSTRUMENT_KEYS))

    # 3) instrumented 輪打開
    ie = build_pass_env({}, "instrumented")
    expect("instrumented turns instruments on",
           all(ie.get(k) == v for k, v in INSTRUMENTS_ON.items()))

    # 4) 兩輪 env 差異只在儀器（純 dict 邏輯；不依賴 resolve）
    base = {"CGC_SPAC": "1", "CGC_OA_ASYNC": "1"}
    c = dict(base); c.update(instruments_off())
    i = dict(base); i.update(INSTRUMENTS_ON)
    diff = {k for k in set(c) | set(i) if c.get(k) != i.get(k)}
    expect("two-pass diff subset of instruments",
           diff <= set(ALL_INSTRUMENT_KEYS))

    # 5) 污染判定
    fake = {"sys_before": {"swap_used_mb": 100},
            "sys_after": {"swap_used_mb": 700, "thermal": {"label": "NOMINAL"}},
            "matrix": [{"thermal": {"hist": {"HEAVY": 0}, "worst": {"label": "NOMINAL"}}}]}
    pol = pass_pollution(fake)
    expect("pollution on swap growth>500", pol["polluted"] is True)
    fake2 = {"sys_before": {"swap_used_mb": 100},
             "sys_after": {"swap_used_mb": 200, "thermal": {"label": "NOMINAL"}},
             "matrix": [{"thermal": {"hist": {"HEAVY": 1}, "worst": {"label": "HEAVY"}}}]}
    expect("pollution on thermal HEAVY", pass_pollution(fake2)["polluted"] is True)

    # 6) 指紋函數可跑（不強制每個檔案都存在）
    fp = build_fingerprint()
    expect("fingerprint returns entries", len(fp) >= 3)

    print(f"\n  selftest: {6 - len(fails)}/6 groups pass")
    return 1 if fails else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", action="append",
                    help='PROFILE:!K=V;K=V（可多臂）。每臂強制 clean+instrumented 兩輪')
    ap.add_argument("--prompt", type=int, default=2048)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--depths", default="512")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--warm-skip", type=int, default=64)
    ap.add_argument("--ctx-size", type=int, default=0)
    ap.add_argument("--max-swap-mb", type=float, default=1024,
                    help="G1 允許的最大起測 swap（MiB）；生產嚴格可調 512")
    ap.add_argument("--cool-s", type=float, default=0,
                    help="兩輪之間冷卻秒數（預設 0，避免拖慢開發）")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="gate 未過也硬跑，結果在報告裡標紅（不建議用於 commit）")
    ap.add_argument("--run-dir", default="/tmp/arm_two_pass")
    ap.add_argument("--json", dest="json_path", default="/tmp/arm_two_pass/result.json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if not args.arm:
        ap.error("--arm is required (unless --selftest)")
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
