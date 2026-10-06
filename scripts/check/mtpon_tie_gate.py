#!/usr/bin/env python3
"""[CGC 2026-10-06] MTP-on 打平 gate：把「on 有沒有打平 off」變成一個乾淨窗自動判的閘。

為什麼需要它 —— 打平式一直是**手算**的，每一次都要有人回頭從兩份 log 導出
`emit/round` 與 R_on/R_off，而這個專案的實測史顯示手算會踩三個坑：
  ① **跑序效應**：同 session ABBA 的兩趟可以差 28%（t1w2_roundcost.py 的 W2/W1：0.871 vs 1.264）
     ⇒ 任何單趟比值不可引用；
  ② **時代差**：off 與 on 的絕對值都動過（off 9.55 vs 11.52、鏈 12.9 vs 10.90）
     ⇒ 跨 session 抓數字就是在比較不同機器；
  ③ **單位**：2.05／1.5–1.6 是 delivery cell 的 token/decode（分母含 replay 輪），
     **樹臂 replay=0 ⇒ 不可互換**（T1C_BAR_205_DERIVATION_2026-10-06.md §4）。
本檔把判準寫成唯一的式子，並把①②用「同 session ABBA」與「用同一 session 的跑序差當雜訊帶」
直接消掉。

判準（唯一一條）
----------------
      on 打平 off  ⟺  emit_on · T_off  ≥  R_on
  T_off = off 的 ms/token（純解碼，1 token/步）
  R_on  = on 的 ms/verify 輪（含 replay 輪）
  emit_on = on 的 emit_per_round（counter，`CGC-BENCH-ACCEPT`；樹臂 replay=0）
等價式：`gain = emit_on · T_off / R_on ≥ 1`。兩側都同 session 取，故比值穩健。
**雜訊帶不寫死**：margin = max(`--margin-min`（預設 2%）, 本 session 兩跑序的 |1−Δ|)。
gain ≥ 1+margin ⇒ TIE；否則 SHORTFALL（附「還缺幾 %」與「emit 要到多少」）。

機制閘（照 charter 的驗收，任一不過 ⇒ 該 session VOID，不給 TIE/SHORT 判詞）
--------------------------------------------------------------------------
  replay/rounds < 10%、trim_fail == 0、每臂 rc==0、無 STALL/TIMEOUT、
  盒閘（l255_close.box_reading/verdict）在起跑前 PASS（或在明確 `--waive-reason` 下放行）。

用法
----
  python3 scripts/check/mtpon_tie_gate.py --check                 # 只問盒子能不能跑
  python3 scripts/check/mtpon_tie_gate.py --go                    # 檢查過才跑一趟 ABBA 判詞
  python3 scripts/check/mtpon_tie_gate.py --watch                 # 輪詢盒閘，綠了自動跑（＝「盒閘綠即跑」）
  python3 scripts/check/mtpon_tie_gate.py --replay A.log B.log …  # 離線重算既有 log（審計/回歸）
  python3 scripts/check/mtpon_tie_gate.py --selftest              # 純函式自測（不吃 GPU）

產物：`Backup/tie_gate_<ts>/` 下的 `tie_gate.json` ＋ 每臂 stderr/stdout ＋ 本檔 stdout。
exit code：0=TIE、1=SHORTFALL、2=拒跑/VOID。
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import os
import re
import select
import shutil
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.expanduser("~/Documents/flashkv-devserver")
DEFAULT_PROMPT = os.path.join(ROOT, "Backup/bench_parity2_20260918/real_prompt_2025.txt")

# ── 正確性閘（輸出健康度）：退化的重複迴圈 ⇒ 不給判詞 ────────────────────────────
# 資料來源是 llama-bench 的 `LLAMA_BENCH_SPEC_DUMP=1`（只有 on 臂／spec 路徑會 dump）。
# 門檻的來歷跑 `scripts/check/output_health_calibrate.py`。
sys.path.insert(0, HERE)
import output_health as oh   # noqa: E402  （同目錄；純函式，不吃 GPU）

# 沒有 dump ⇒ 沒有正確性見證。預設 fail-closed；`--no-output-health` 只放行這一種，
# **永不**放行 DEGENERATE（退化的臂一律 VOID）。
OH_REQUIRE_WITNESS = True

STALL_S, TIMEOUT_S = 420.0, 1800.0
KEY = ("CGC-BENCH-ACCEPT", "CGC-SPEC-TREE", "CGC-ORACLE", "prompt-file",
       "assert", "Assertion", "GGML_ASSERT", "failed", "error", "abort", "Metal")

# ── arm 定義（off = 無 spec；on = --spec-type draft-mtp，樹/鏈由 env 決定）───────────
def arm_env(mode: str, tree: bool, width: int, k: int, seed: int, caps: str) -> dict:
    if mode == "off":
        return {"CGC_SERVER_MTP": "1", "CGC_SERVER_LAYER_CAPS": caps,
                "CGC_SERVER_SEED": str(seed)}
    env = {"CGC_SERVER_MTP": "1", "CGC_SERVER_LAYER_CAPS": caps,
           "CGC_DRAFT_CTX_ALIGN": "1", "CGC_DRAFT_SMALL_BATCH": "1",
           "LLAMA_BENCH_SPEC": "1", "LLAMA_BENCH_SPEC_DRAFT_N_MAX": str(k),
           "CGC_SERVER_SEED": str(seed)}
    if tree:
        env["CGC_SPEC_TREE"] = "1"
        env["CGC_SPEC_TREE_B"] = str(width)
        env["CGC_SPEC_TREE_RS"] = "2"
    return env

def arm_cli(mode: str, tree: bool, k: int) -> list[str]:
    if mode == "off":
        return []
    cli = ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(k)]
    return cli

def shape_args(a) -> list[str]:
    s = ["-b", str(a.batch), "-ub", str(a.ubatch), "-p", str(a.prompt),
         "-n", str(a.gen), "-d", str(a.depth), "-r", str(a.reps), "-o", "json"]
    if a.ctx_size:
        s += ["--ctx-size", str(a.ctx_size)]
    if a.warm_skip:
        s += ["--warm-skip", str(a.warm_skip)]
    if a.prompt_file:
        s += ["--prompt-file", a.prompt_file]
    return s

# ── 解析（純函式；--replay 與 --selftest 都走這裡）────────────────────────────────
def parse_accept(text: str) -> dict:
    lines = [l for l in text.splitlines() if l.startswith("CGC-BENCH-ACCEPT")]
    if not lines:
        return {}
    f = dict(re.findall(r"([A-Za-z0-9_]+)=([^\s]+)", lines[-1]))
    out = {k: float(v) for k, v in f.items() if re.fullmatch(r"-?\d+(\.\d+)?", v)}
    out["_line"] = lines[-1]
    return out

def parse_token_line(text: str, tag: str) -> dict:
    lines = [l for l in text.splitlines() if l.startswith(tag)]
    if not lines:
        return {}
    return dict(re.findall(r"([A-Za-z0-9_]+)=([^\s]+)", lines[-1]))

def parse_json_row(stdout_text: str) -> dict:
    data = json.loads(stdout_text)
    rows = data if isinstance(data, list) else [data]
    rows = [r for r in rows if int(r.get("n_gen", 0)) > 0] or rows
    return rows[0] if rows else {}

# ── 被打斷時的收尾：gate 死掉不可以留下還在跑的 llama-bench ────────────────────────
_LIVE = {"pid": None}   # 目前在跑的 llama-bench 行程群組（Popen 後設、wait 後清）


def _kill_live(sig=None):
    """收掉在跑的那支 llama-bench（連它的行程群組）。gate 只用於乾淨窗，
    所以「gate 沒了、測量還在跑」是不能接受的：它會污染下一個窗，而且沒人記得它存在。"""
    pid = _LIVE.get("pid")
    if pid:
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass                                    # 已經沒了（正常）
        except PermissionError:
            pass                                    # 不歸我們管 ⇒ 已經不是我們的子行程
        else:
            for _ in range(30):                     # 最多等 3 s
                time.sleep(0.1)
                try:
                    os.killpg(pid, 0)
                except ProcessLookupError:
                    break
                except PermissionError:
                    # 實測（macOS）：子行程剛死、群組還在半死狀態時回 EPERM ⇒ 再探下去只會
                    # 把收尾本身變成 traceback（rc 1 而不是 143）。把「不能再 signal」當「結束」。
                    break
            else:
                try:
                    os.killpg(pid, signal.SIGKILL)  # 不放手就硬收
                except OSError:
                    pass
    if sig is not None:
        nm = signal.Signals(sig).name if hasattr(signal, "Signals") else str(sig)
        print(f"\n[gate] 收到 {nm} ⇒ 已收掉在跑的 llama-bench，離開（沒有判詞）",
              file=sys.stderr, flush=True)
        sys.exit(128 + sig)


def _install_handlers():
    for nm in ("SIGINT", "SIGTERM", "SIGHUP", "SIGPIPE"):
        sg = getattr(signal, nm, None)
        if sg is None:
            continue
        # SIGPIPE：輸出被管道截斷（`| head`）時預設是靜靜死掉並且**留下子行程** ⇒ 要接住。
        signal.signal(sg, lambda sn, _frm: _kill_live(sn))


def arm_metrics(stderr_text: str, stdout_text: str, mode: str,
                allow_prefill: bool = False) -> dict:
    """一支臂的三個量：emit_per_round（on 才有）、avg_ns、rounds。缺一即 VOID。

    ⛔ n_prompt > 0 一律拒收：`avg_ns` 是計時區的總時間，而 llama-bench 把 prompt run 放在
    時鐘**內**（`if (t.n_prompt > 0)` 在 t_start 之後）⇒ prefill 的分母會混進 T_off 與 R_on。
    decode 支柱是 `-p 0`（只有 depth run，時鐘外）。`allow_prefill=True` 只給事后審計用
    （replay 既有 log），且報告會標成審計、不當判詞。
    """
    row = parse_json_row(stdout_text)
    acc = parse_accept(stderr_text)
    tre = parse_token_line(stderr_text, "CGC-SPEC-TREE")
    avg_ns = row.get("avg_ns")
    if avg_ns is None:
        raise ValueError("JSON 沒有 avg_ns —— 這一支臂的計時不可用")
    n_prompt = int(row.get("n_prompt", 0))
    if n_prompt > 0 and not allow_prefill:
        raise ValueError(f"n_prompt={n_prompt}>0：計時含 prefill（llama-bench 把 prompt run 放在時鐘內），"
                         "R_on/T_off 不可用 —— 請用 -p 0 的 decode 支柱")
    avg_ns = float(avg_ns)
    m = {"mode": mode, "avg_ns": avg_ns, "n_prompt": n_prompt,
         "avg_ts": float(row.get("avg_ts", 0.0)),
         "n_gen": int(row.get("n_gen", 0)), "n_kept": int(row.get("n_kept", 0)),
         "rounds": int(acc.get("rounds", 0)) if acc else int(row.get("n_gen", 0)),
         "emit": float(acc.get("emit_per_round", 1.0)) if mode == "off" else float(acc.get("emit_per_round", 0.0)),
         "replay": int(acc.get("replay", 0)) if acc else 0,
         "drafted": int(acc.get("drafted", 0)) if acc else 0,
         "trim_fail": int(tre.get("trim_fail", 0)) if tre else 0,
         "accept_line": acc.get("_line", "")}
    if m["rounds"] <= 0:
        raise ValueError("rounds<=0 —— 無法把 avg_ns 攤成每輪成本")
    m["round_ms"] = avg_ns / 1e6 / m["rounds"]      # on 的 R_on（off 時 = 1 token 的 R）
    m["token_ms"] = avg_ns / 1e6 / max(m["n_gen"], 1)
    # 正確性見證：只有 on 臂（spec 路徑）會 dump 輸出 ⇒ 只有它判健康度。
    # off 臂（純解碼）沒有這條 ⇒ 已知缺口，寫在 output_health.py 檔尾，不假裝有。
    m["output_health"] = oh.health_from_stderr(stderr_text) if mode == "on" else None
    return m

def gain(emit_on: float, token_ms_off: float, round_ms_on: float) -> float:
    if round_ms_on <= 0:
        raise ValueError("round_ms_on<=0")
    return emit_on * token_ms_off / round_ms_on

def decide(sessions: list[dict], margin_min: float) -> dict:
    """sessions = [{'order': 'AB'|'BA', 'off': metrics, 'on': metrics}, …]；
    gain_bal = 兩個跑序 gain 的幾何平均（比值量用幾何平均）；margin 取本 session 的跑序散度。
    """
    gs = []
    for s in sessions:
        gs.append(gain(s["on"]["emit"], s["off"]["token_ms"], s["on"]["round_ms"]))
    g_bal = (gs[0] * gs[1]) ** 0.5 if len(gs) > 1 else gs[0]
    spread = abs(gs[0] - gs[1]) / ((gs[0] * gs[1]) ** 0.5) if len(gs) > 1 else 0.0
    margin = max(margin_min, spread)
    out = {"gains": [round(g, 4) for g in gs], "gain_bal": round(g_bal, 4),
           "order_spread": round(spread, 4), "margin": round(margin, 4)}
    # 機制閘（charter 的驗收）
    reasons = []
    for s in sessions:
        on = s["on"]
        if on["rounds"] and on["replay"] / max(on["rounds"], 1) >= 0.10:
            reasons.append(f"replay {on['replay']}/{on['rounds']} ≥10%")
        if on["trim_fail"]:
            reasons.append(f"trim_fail={on['trim_fail']}")
        if on["emit"] <= 0:
            reasons.append("on 臂沒有 emit_per_round（spec 沒跑？）")
    out["void_reasons"] = reasons
    if reasons:
        out["verdict"] = "VOID"
    elif g_bal >= 1 + margin:
        out["verdict"] = "TIE"
    else:
        out["verdict"] = "SHORTFALL"
        out["need_emit_for_tie"] = round(1 + margin, 4)
        out["emit_now"] = round(sessions[0]["on"]["emit"], 4)
        out["needed_emit"] = round((1 + margin) * sessions[0]["on"]["round_ms"] / sessions[0]["off"]["token_ms"], 4)
    return out

# ── 盒閘（沿用 l255_close 的讀數與判詞；不另立第二套定義）────────────────────────
def box_gate():
    """盒閘讀數。`TIE_FAKE_BOX=short` 是**只會強制不綠**的測試鉤子（給自測釘住
    「不綠時 --watch 不會發車」這條路徑；它無法強制綠，所以不可能誘發真量測）。"""
    fake = os.environ.get("TIE_FAKE_BOX")
    if fake in ("short", "foreign"):
        r = {"reclaimable_mb": 100.0 if fake == "short" else 9000.0, "need_mb": 7703.0,
             "memory_ok": fake != "short", "port_held": False,
             "foreign_llama": fake == "foreign", "compressor_ok": True, "compressor_reason": "",
             "thermal": 0, "launcher_class": "full-mtp", "launcher_free_pct": 30,
             "binding": "harness", "agree": False, "swap_used_mb": 0.0}
        return (fake != "short"), [f"（TIE_FAKE_BOX={fake} 測試鉤子）"], \
               (["記憶體"] if fake == "short" else []), r
    sys.path.insert(0, HERE)
    import l255_close as l
    r = l.box_reading()
    ok, lines, blocked = l.verdict(r)
    return ok, lines, blocked, r

# ── 跑一支臂（串流、stall/timeout 守衛；同 t1w2_realprompt.py 的實作）──────────────
def run_arm(a, mode: str, tag: str, outdir: str) -> dict:
    sys.path.insert(0, HERE)
    m = importlib.import_module("llama_bench_matrix")
    envs = arm_env(mode, a.tree, a.tree_w, a.k, a.seed, a.caps)
    res = m.resolve("prod-new", envs)
    fwd = m.forward_argv(res["server_argv"])
    cmd = [str(m.LLAMA_BENCH)] + fwd + arm_cli(mode, a.tree, a.k) + shape_args(a) + \
          ["--fixed-fill-seed", str(a.seed)]
    run_env = dict(os.environ)
    run_env.update(res["env"])
    run_env.update(m.sampling_env(res["server_argv"]))
    run_env.update(envs)
    run_env["LLAMA_BENCH_SPEC_DBG"] = "1"
    if mode == "on":
        # 正確性閘的資料來源（spec 路徑才有）。off 臂不會印 ⇒ 見 output_health.py 檔尾。
        run_env["LLAMA_BENCH_SPEC_DUMP"] = "1"
    if a.oracle:
        run_env["LLAMA_BENCH_SPEC_ORACLE"] = "1"
    logp, outp = os.path.join(outdir, f"{tag}.stderr.log"), os.path.join(outdir, f"{tag}.stdout.json")
    print(f"\n=== {tag} ({mode}{', tree W%d' % a.tree_w if mode == 'on' and a.tree else ''}, k={a.k}) ===",
          flush=True)
    print("cmd:", " ".join(cmd), flush=True)
    t0 = time.time()
    killed = None
    with open(logp, "w") as lf, open(outp, "w") as of:
        p = subprocess.Popen(cmd, cwd=ROOT, env=run_env, stdout=of, stderr=subprocess.PIPE,
                             start_new_session=True, text=True, bufsize=1)
        _LIVE["pid"] = p.pid
        buf, last, last_prog = "", time.time(), "(no output yet)"
        while True:
            r, _, _ = select.select([p.stderr], [], [], 5.0)
            if r:
                chunk = os.read(p.stderr.fileno(), 65536).decode("utf-8", "replace")
                if chunk == "":
                    break
                buf += chunk
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    lf.write(line + "\n"); lf.flush()
                    if any(k2 in line for k2 in KEY):
                        print("   " + line[:200], flush=True)
                    if not line.startswith("CGC-RSS:"):
                        last, last_prog = time.time(), line
            if p.poll() is not None:
                break
            now = time.time()
            if now - last > STALL_S:
                killed = "STALL"; print(f"   [STALL] 420s 無進展，kill pgid（last: {last_prog[:120]}）", flush=True)
                os.killpg(p.pid, 15); break
            if now - t0 > TIMEOUT_S:
                killed = "TIMEOUT"; print("   [TIMEOUT] 1800s，kill pgid", flush=True)
                os.killpg(p.pid, 15); break
        if buf:
            lf.write(buf); lf.flush()
    try:
        rc = p.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, 9); rc = -9
    _LIVE["pid"] = None            # 已 reap ⇒ 別讓後來的訊號打到可能被重用的 pgid
    wall = time.time() - t0
    err, out = open(logp, errors="replace").read(), open(outp, errors="replace").read()
    rec = {"tag": tag, "mode": mode, "rc": rc, "wall_s": round(wall, 1), "killed": killed,
           "cmd": " ".join(cmd), "log": logp, "stdout_json": outp}
    try:
        rec.update(arm_metrics(err, out, mode))
    except Exception as e:  # noqa: BLE001 -- 缺量就是 VOID，不猜
        rec["parse_error"] = str(e)
    print(f"   rc={rc} wall={wall:.0f}s " +
          (f"emit={rec.get('emit')} rounds={rec.get('rounds')} round_ms={rec.get('round_ms', 0):.1f} "
           f"token_ms={rec.get('token_ms', 0):.2f}" if "parse_error" not in rec
           else f"PARSE FAIL: {rec['parse_error']}"), flush=True)
    return rec

# ── 一整個 session：AB 與 BA 兩跑序 ────────────────────────────────────────────────
def run_session(a, outdir: str, order: str) -> dict:
    steps = [("off", "off"), ("on", "on")] if order == "AB" else [("on", "on"), ("off", "off")]
    recs = {}
    for i, (mode, _t) in enumerate(steps):
        recs[mode] = run_arm(a, mode, f"{order}{i+1}_{mode}", outdir)
    return {"order": order, "off": recs["off"], "on": recs["on"]}

def sessions_valid(sessions: list[dict]) -> list[str]:
    """有效性檢查。`tag` 只在真跑時存在（replay 的 metrics 來自 log）⇒ 標籤要能降級，
    不能 KeyError —— 這是實測抓到的（replay 一按就炸）。"""
    bad = []
    for s in sessions:
        for role in ("off", "on"):
            r = s[role]
            who = r.get("tag") or r.get("log") or f"{s.get('order')}:{role}"
            if r.get("rc") not in (None, 0):
                bad.append(f"{who} rc={r.get('rc')}")
            if r.get("killed"):
                bad.append(f"{who} {r['killed']}")
            if "parse_error" in r:
                bad.append(f"{who}: {r['parse_error']}")
            h = r.get("output_health")
            if role == "on" and h:
                if h["state"] == "DEGENERATE":
                    # 「快但壞掉」：退化會抬高接受率 ⇒ 抬高 emit ⇒ 判詞必須拒收。
                    # 這一條**沒有**放行旗標。
                    bad.append(f"{who} 輸出退化：{'; '.join(h['reasons'])}")
                elif h["state"] == "NO_DATA" and OH_REQUIRE_WITNESS:
                    bad.append(f"{who} 輸出無見證（沒有 CGC-SPEC-DUMP）")
    return bad

# ── selftest：純函式，不吃 GPU ────────────────────────────────────────────────────
def cmd_selftest() -> int:
    ok = True
    def chk(name, cond):
        nonlocal ok
        print(f"  {'✅' if cond else '⛔'} {name}")
        ok = ok and bool(cond)

    # 打平式：off 100 ms/token，on 每輪 200 ms、emit 2.0 ⇒ gain 1.00（臨界）
    chk("gain 臨界 = 1.00", abs(gain(2.0, 100.0, 200.0) - 1.0) < 1e-9)
    chk("emit 較高 ⇒ gain 上升", gain(2.5, 100.0, 200.0) > 1.0)
    def mk(emit, round_ms, token_ms, replay=0, trim=0):
        return {"emit": emit, "round_ms": round_ms, "token_ms": token_ms, "rounds": 50,
                "replay": replay, "trim_fail": trim}
    d = decide([{"order": "AB", "on": mk(2.0, 200.0, 100.0), "off": mk(1.0, 100.0, 100.0)},
                {"order": "BA", "on": mk(2.0, 200.0, 100.0), "off": mk(1.0, 100.0, 100.0)}], 0.02)
    chk("臨界 1.00 < 1.02 ⇒ SHORTFALL", d["verdict"] == "SHORTFALL")
    chk("SHORTFALL 附 needed_emit", d.get("needed_emit", 0) > d["emit_now"])
    d = decide([{"order": "AB", "on": mk(2.2, 200.0, 100.0), "off": mk(1.0, 100.0, 100.0)},
                {"order": "BA", "on": mk(2.2, 200.0, 100.0), "off": mk(1.0, 100.0, 100.0)}], 0.02)
    chk("gain 1.10 ⇒ TIE", d["verdict"] == "TIE")
    # 跑序散度吃掉表面領先：兩跑序 gain 差 30% ⇒ margin 升到 0.30，1.10 不再過
    d = decide([{"order": "AB", "on": mk(2.2, 200.0, 100.0), "off": mk(1.0, 100.0, 100.0)},
                {"order": "BA", "on": mk(1.7, 200.0, 100.0), "off": mk(1.0, 100.0, 100.0)}], 0.02)
    chk("跑序散度 ⇒ margin 由 0.02 升到 ≥0.25", d["margin"] >= 0.25)
    d = decide([{"order": "AB", "on": mk(2.9, 200.0, 100.0, replay=8), "off": mk(1.0, 100.0, 100.0)},
                {"order": "BA", "on": mk(2.9, 200.0, 100.0, replay=8), "off": mk(1.0, 100.0, 100.0)}], 0.02)
    chk("replay 16% ⇒ VOID", d["verdict"] == "VOID")
    d = decide([{"order": "AB", "on": mk(2.9, 200.0, 100.0, trim=1), "off": mk(1.0, 100.0, 100.0)},
                {"order": "BA", "on": mk(2.9, 200.0, 100.0, trim=1), "off": mk(1.0, 100.0, 100.0)}], 0.02)
    chk("trim_fail ⇒ VOID", d["verdict"] == "VOID")

    # 解析：off 臂沒有 ACCEPT 行 ⇒ emit=1、rounds=n_gen、token_ms 由 avg_ns 攤
    off_err = "SPECDBG: nothing\n"
    off_json = json.dumps([{"n_gen": 128, "avg_ns": 12_800_000_000, "avg_ts": 10.0, "n_kept": 3}])
    m_off = arm_metrics(off_err, off_json, "off")
    chk("off: rounds=n_gen=128", m_off["rounds"] == 128)
    chk("off: token_ms = 100 ms", abs(m_off["token_ms"] - 100.0) < 1e-6)
    chk("off: emit=1（缺 ACCEPT 行時的唯一誠實值）", m_off["emit"] == 1.0)
    on_err = ("CGC-BENCH-ACCEPT phase=timed rounds=50 drafted=100 acc_drafts=60 mean_len=2.2 "
              "draft_ratio=0.6 gen_tokens=110 n_gen=128 replay=0 emit_per_round=2.2\n"
              "CGC-SPEC-TREE phase=timed rounds_tree=50 branchB=0 path_partial=10 trim_fail=0 replay=0\n")
    on_json = json.dumps([{"n_gen": 128, "avg_ns": 10_000_000_000, "avg_ts": 11.0, "n_kept": 3}])
    m_on = arm_metrics(on_err, on_json, "on")
    chk("on: round_ms = 200 ms", abs(m_on["round_ms"] - 200.0) < 1e-6)
    chk("on: emit=2.2", abs(m_on["emit"] - 2.2) < 1e-9)
    chk("on: trim_fail 讀到 0", m_on["trim_fail"] == 0)
    try:
        arm_metrics(on_err, json.dumps([{"n_gen": 128}]), "on")
        chk("缺 avg_ns ⇒ 拋錯（不猜）", False)
    except ValueError:
        chk("缺 avg_ns ⇒ 拋錯（不猜）", True)
    # 時鐘含 prefill（-p 32）⇒ 拒收；審計模式才放行
    pf_json = json.dumps([{"n_prompt": 32, "n_gen": 64, "avg_ns": 5_000_000_000,
                           "avg_ts": 19.2, "n_kept": 1}])
    try:
        arm_metrics(on_err, pf_json, "on")
        chk("n_prompt>0 ⇒ 拒收（時鐘含 prefill）", False)
    except ValueError:
        chk("n_prompt>0 ⇒ 拒收（時鐘含 prefill）", True)
    m_pf = arm_metrics(on_err, pf_json, "on", allow_prefill=True)
    chk("審計模式放行且記下 n_prompt", m_pf["n_prompt"] == 32)

    # 回歸釘（實測抓到的第三個 bug）：污染 log 走 --replay 必須是**乾淨拒收 rc=2**，
    # 不能是 uncaught traceback。純函式自測看不到 CLI 這一層，所以用子程序釘。
    import subprocess, tempfile
    with tempfile.TemporaryDirectory() as td:
        lp = os.path.join(td, "p.stderr.log")
        with open(lp, "w") as f:
            f.write("CGC-BENCH-ACCEPT rounds=64 replay=0 emit_per_round=2.0\n")
        with open(lp + ".stdout", "w") as f:
            f.write(pf_json)
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--replay", lp, lp],
                           capture_output=True, text=True, timeout=120)
        blob = r.stdout + r.stderr
        chk("污染 log --replay ⇒ rc=2（乾淨拒收，非 traceback）",
            r.returncode == 2 and "Traceback" not in blob and "拒收" in blob)
        r2 = subprocess.run([sys.executable, os.path.abspath(__file__), "--replay", lp, lp,
                             "--replay-allow-prefill", "--no-output-health"],
                            capture_output=True, text=True, timeout=120)
        chk("審計旗標放在 --replay 之後仍生效（位置無關；legacy log 需 --no-output-health）",
            r2.returncode in (0, 1) and "審計模式" in r2.stdout and "expected at least one argument" not in r2.stderr)

        # 端到端：退化 on 臂 ⇒ rc=2；`--no-output-health` **擋不住**它（本閘的存在理由）。
        # 再加一組對照（同一對 log、dump 換成健康序列 ⇒ 不再 VOID），證明擋的是退化、不是別的。
        onj = json.dumps([{"n_gen": 40, "avg_ns": 8_000_000_000, "avg_ts": 5.0, "n_kept": 2}])
        offj = json.dumps([{"n_gen": 40, "avg_ns": 4_000_000_000, "avg_ts": 10.0, "n_kept": 2}])
        onp, offp = os.path.join(td, "on.stderr.log"), os.path.join(td, "off.stderr.log")
        head = ("CGC-BENCH-ACCEPT phase=timed rounds=20 drafted=40 acc_drafts=30 mean_len=2.5 "
                "draft_ratio=0.75 gen_tokens=40 n_gen=40 replay=0 emit_per_round=2.0\n"
                "CGC-SPEC-TREE phase=timed rounds_tree=20 branchB=0 path_partial=5 trim_fail=0 replay=0\n")
        with open(onp, "w") as f:
            f.write(head + "CGC-SPEC-DUMP phase=timed n=40 ids=" + ",".join(["7"] * 40) + "\n")
        with open(onp + ".stdout", "w") as f:
            f.write(onj)
        with open(offp, "w") as f:
            f.write("SPECDBG: nothing\n")
        with open(offp + ".stdout", "w") as f:
            f.write(offj)
        r6 = subprocess.run([sys.executable, os.path.abspath(__file__), "--replay", offp, onp],
                            capture_output=True, text=True, timeout=120)
        b6 = r6.stdout + r6.stderr
        chk("退化 on 臂 ⇒ --replay rc=2、判詞含 DEGENERATE、無 traceback",
            r6.returncode == 2 and "DEGENERATE" in b6 and "Traceback" not in b6)
        r7 = subprocess.run([sys.executable, os.path.abspath(__file__), "--replay", offp, onp,
                             "--no-output-health"], capture_output=True, text=True, timeout=120)
        b7 = r7.stdout + r7.stderr
        chk("--no-output-health **擋不住**退化（仍 rc=2 且 DEGENERATE）",
            r7.returncode == 2 and "DEGENERATE" in b7)
        with open(onp, "w") as f:
            f.write(head + "CGC-SPEC-DUMP phase=timed n=40 ids=" +
                    ",".join(str(9000 + (i * 37) % 500) for i in range(40)) + "\n")
        r8 = subprocess.run([sys.executable, os.path.abspath(__file__), "--replay", offp, onp],
                            capture_output=True, text=True, timeout=120)
        chk("同一對 log、dump 換健康序列 ⇒ 不再 VOID（rc=1 SHORTFALL）", r8.returncode == 1)

        # 回歸釘（2026-10-06 實測抓到的第五個 bug）：一趟 arm 每個 rep 印一條 dump，
        # 舊版只讀**最後一條** ⇒ rep 0 的退化會被後面健康的 rep 蓋掉。逐 dump 取最壞後，
        # 「先退化、後健康」必須仍然 VOID。（用同一對 log，只改 dump 的順序與內容。）
        with open(onp, "w") as f:
            f.write(head + "CGC-SPEC-DUMP phase=timed n=40 ids=" + ",".join(["7"] * 40) + "\n" +
                    "CGC-SPEC-DUMP phase=timed n=40 ids=" +
                    ",".join(str(9000 + (i * 37) % 500) for i in range(40)) + "\n")
        r9 = subprocess.run([sys.executable, os.path.abspath(__file__), "--replay", offp, onp],
                            capture_output=True, text=True, timeout=120)
        b9 = r9.stdout + r9.stderr
        chk("退化在**前一條** dump、最後一條健康 ⇒ 仍 rc=2（逐 dump 取最壞）",
            r9.returncode == 2 and "DEGENERATE" in b9 and "rep 0" in b9)
        # 對照：兩條都健康 ⇒ 不再 VOID（證明擋的是退化、不是「多條 dump」本身）
        with open(onp, "w") as f:
            f.write(head + "CGC-SPEC-DUMP phase=timed n=40 ids=" +
                    ",".join(str(9000 + (i * 37) % 500) for i in range(40)) + "\n" +
                    "CGC-SPEC-DUMP phase=timed n=40 ids=" +
                    ",".join(str(7000 + (i * 53) % 400) for i in range(40)) + "\n")
        r10 = subprocess.run([sys.executable, os.path.abspath(__file__), "--replay", offp, onp],
                             capture_output=True, text=True, timeout=120)
        chk("兩條都健康 ⇒ 仍 SHORTFALL（rc=1，不因多條 dump 誤判退化）", r10.returncode == 1)

    # 回歸釘（實測抓到的第四個 bug）：--watch 的主用法（沒 waive、盒子不綠）必須走得到輪詢迴圈，
    # 且**永不在不綠時發車**。用只會強制不綠的鉤子 + max-min 0 讓它有界結束。
    env2 = dict(os.environ, TIE_FAKE_BOX="short")
    r3 = subprocess.run([sys.executable, os.path.abspath(__file__), "--watch",
                         "--poll-s", "1", "--watch-max-min", "0"],
                        capture_output=True, text=True, timeout=180, env=env2)
    b3 = r3.stdout + r3.stderr
    chk("不綠時 --watch ⇒ 進輪詢迴圈、有界放棄、rc=2（不發車）",
        r3.returncode == 2 and "輪詢" in b3 or ("超過最長等待" in b3 and r3.returncode == 2))
    chk("不綠時 --watch ⇒ 沒有產生量測目錄",
        "[watch]" in r3.stdout and "產物目錄" not in b3)
    r4 = subprocess.run([sys.executable, os.path.abspath(__file__), "--go"],
                        capture_output=True, text=True, timeout=180, env=env2)
    chk("不綠時 --go（無 waive）⇒ rc=2 拒跑", r4.returncode == 2 and "拒跑" in (r4.stdout + r4.stderr))

    # Bug 6 的兩面：①別的 llama 一定要擋（foreign 由盒閘同一套偵測來）
    #                 ②gate 不可以再自帶「掃命令列字串」的第二套偵測
    r5 = subprocess.run([sys.executable, os.path.abspath(__file__), "--go",
                         "--waive-reason", "selftest"],
                        capture_output=True, text=True, timeout=180,
                        env=dict(os.environ, TIE_FAKE_BOX="foreign"))
    b5 = r5.stdout + r5.stderr
    chk("foreign llama ⇒ rc=2 拒跑（waive 也不能放行）",
        r5.returncode == 2 and "別的 llama" in b5 and "產物目錄" not in b5)
    chk("gate 不再自帶第二套 foreign 偵測（用盒閘同一套）",
        not hasattr(sys.modules.get("__main__"), "foreign_llama"))

    # 收尾函式本身不可以拋錯（否則訊號處理會變成 traceback，rc 變 1 而不是 128+sig）
    m_self = sys.modules.get("__main__")
    try:
        m_self._LIVE["pid"] = 999999               # 不存在的 pid
        m_self._kill_live()                        # 不得拋錯、也不得 exit
        chk("_kill_live 對不存在的 pid 安靜收場", True)
    except BaseException as e:                     # noqa: BLE001
        chk(f"_kill_live 對不存在的 pid 安靜收場（got {type(e).__name__}）", False)
    finally:
        m_self._LIVE["pid"] = None

    # 產物自洽：doc 承諾「本檔 stdout」是產物之一 ⇒ report.txt 必須真的出現（純函式，不吃 GPU）
    import contextlib
    import io
    import tempfile as _tf

    def full(emit, round_ms, tok_ms):
        m = dict(mk(emit, round_ms, tok_ms))
        m.update({"avg_ts": 5.0, "n_kept": 1, "tag": "t", "rc": 0,
                  "mode": "off" if emit == 1 else "on"})
        return m

    sess = [{"order": "AB", "on": full(2.2, 200.0, 100.0), "off": full(1.0, 100.0, 100.0)},
            {"order": "BA", "on": full(2.2, 200.0, 100.0), "off": full(1.0, 100.0, 100.0)}]
    with _tf.TemporaryDirectory() as td:
        with contextlib.redirect_stdout(io.StringIO()):
            rc0 = report(sess, 0.02, td)
        rp = os.path.join(td, "report.txt")
        txt = open(rp).read() if os.path.exists(rp) else ""
        chk("report() 寫出 report.txt（doc 承諾的產物）", os.path.exists(rp))
        chk("report.txt 帶判詞，且 rc 與判詞一致（TIE ⇒ 0）", "TIE" in txt and rc0 == 0)
        chk("report() 也寫出 tie_gate.json", os.path.exists(os.path.join(td, "tie_gate.json")))

    # ── 正確性閘（輸出健康度）：退化的重複迴圈必須讓判詞 VOID ────────────────────────
    chk("退化偵測：週期 1 迴圈 ⇒ DEGENERATE", oh.health([7] * 40)["state"] == "DEGENERATE")
    chk("退化偵測：週期 2 迴圈 ⇒ DEGENERATE", oh.health([7, 9] * 20)["state"] == "DEGENERATE")
    chk("退化偵測：健康序列 ⇒ OK",
        oh.health([1, 5, 9, 2, 7, 3, 8, 4, 6, 1, 2, 9, 4, 7, 5, 3, 8])["state"] == "OK")
    chk("沒有 dump ⇒ NO_DATA（fail-closed）", oh.health(None)["state"] == "NO_DATA")
    chk("parse_dump 只吃 phase=timed",
        oh.parse_dump("CGC-SPEC-DUMP phase=warmup n=3 ids=1,2,3\n") is None
        and oh.parse_dump("CGC-SPEC-DUMP phase=timed n=3 ids=1,2,3\n") == [1, 2, 3])
    _off = full(1.0, 100.0, 100.0)
    _deg = dict(full(2.5, 200.0, 100.0)); _deg["output_health"] = oh.health([7] * 40)
    _nod = dict(full(2.5, 200.0, 100.0)); _nod["output_health"] = oh.health(None)
    _ok_ = dict(full(2.5, 200.0, 100.0)); _ok_["output_health"] = oh.health([1, 5, 9, 2, 7, 3, 8, 4, 6])
    s_deg = [{"order": "AB", "on": _deg, "off": _off}]
    s_nod = [{"order": "AB", "on": _nod, "off": _off}]
    s_ok = [{"order": "AB", "on": _ok_, "off": _off}]
    chk("退化 on 臂 ⇒ 有效性閘擋下（VOID）", len(sessions_valid(s_deg)) > 0)
    chk("健康 on 臂 ⇒ 不擋", len(sessions_valid(s_ok)) == 0)
    chk("無見證 on 臂 ⇒ 預設 fail-closed", len(sessions_valid(s_nod)) > 0)
    globals()["OH_REQUIRE_WITNESS"] = False
    chk("--no-output-health 放行無見證、**仍擋退化**",
        len(sessions_valid(s_nod)) == 0 and len(sessions_valid(s_deg)) > 0)
    globals()["OH_REQUIRE_WITNESS"] = True
    with _tf.TemporaryDirectory() as td2:
        with contextlib.redirect_stdout(io.StringIO()):
            rc_deg = report(s_deg, 0.02, td2)
        d_deg = json.load(open(os.path.join(td2, "tie_gate.json")))
        chk("退化 ⇒ report 判 VOID、rc=2", rc_deg == 2 and d_deg["verdict"] == "VOID")
        chk("退化 ⇒ tie_gate.json 留下 on 臂的健康度（可事後對帳）",
            d_deg["sessions"][0]["on"].get("output_health", {}).get("state") == "DEGENERATE")

    # 盒閘拒跑：短少時不給數字（只驗判詞函式，不碰真盒子）
    sys.path.insert(0, HERE)
    import l255_close as l
    fake = {"reclaimable_mb": 100.0, "need_mb": 7703.0, "memory_ok": False, "port_held": False,
            "foreign_llama": False, "compressor_ok": True, "compressor_reason": "", "thermal": 0,
            "launcher_class": "full-mtp", "launcher_free_pct": 50, "binding": "harness",
            "agree": False, "swap_used_mb": 0.0}
    ok2, lines, blocked = l.verdict(fake)
    chk("盒閘短少 ⇒ 拒（blocked 含記憶體）", (not ok2) and ("記憶體" in blocked))
    print("\nSELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1

# ── replay：用既有 log 重算判詞（審計/回歸；不發車）──────────────────────────────
def cmd_replay(paths: list[str]) -> int:
    """paths = 2 或 4 份 stderr log。2 份 ⇒ 單跑序（margin 用 --margin-min）。
    4 份 ⇒ 依序視為 AB 的兩臂 + BA 的兩臂。每份 log 旁的 `.stdout`（同名加後綴 .stdout）給不計時欄位。
    """
    def load(p):
        err = open(p, errors="replace").read()
        mode = "on" if "CGC-BENCH-ACCEPT" in err else "off"
        allow = bool(os.environ.get("TIE_REPLAY_ALLOW_PREFILL"))
        cands = [p + ".stdout", os.path.splitext(p)[0] + ".stdout", p.replace(".stderr.log", ".stdout")]
        out = ""
        for c in cands:
            if os.path.exists(c):
                out = open(c, errors="replace").read()
                break
        if not out:
            raise SystemExit(f"replay: 找不到 {p} 的 JSON（試過 {cands}）—— 沒有 avg_ns 就沒有判詞")
        return mode, arm_metrics(err, out, mode, allow_prefill=allow)
    if len(paths) not in (2, 4):
        raise SystemExit("replay 需要 2 或 4 份 log")
    audit = bool(os.environ.get("TIE_REPLAY_ALLOW_PREFILL"))
    if audit:
        print("⚠ 審計模式：放行了 n_prompt>0 的 log ⇒ 判詞含 prefill 污染，**不可引用為結果**。")
    loaded = []
    for p in paths:
        try:
            loaded.append(load(p))
        except SystemExit:
            raise                      # 「找不到 JSON」已經是乾淨訊息
        except Exception as e:         # noqa: BLE001 -- 形狀污染/缺量 ⇒ 沒有數字可以判
            # 這一支以前會直接 traceback 出場（rc=1）：與 doc 承諾的「2＝拒跑/VOID」不符，
            # 而且 traceback 會把「其實是形狀問題」讀成「程式壞了」。實測抓到（pg512 舊 log）。
            print(f"\n⛔ REPLAY 拒收：{p}\n   {e}", flush=True)
            if not audit:
                print("   形狀污染/缺量 ⇒ 這對 log 給不出判詞。", flush=True)
                print("   若只是事後審計：加 --replay-allow-prefill（或 TIE_REPLAY_ALLOW_PREFILL=1）。",
                      flush=True)
            return 2

    if len(loaded) == 2:
        (m0, a0), (m1, a1) = loaded
        off = a0 if m0 == "off" else a1
        on = a1 if m0 == "off" else a0
        sessions = [{"order": "run1", "off": off, "on": on}]
    else:
        sessions = []
        for i, order in ((0, "run1"), (2, "run2")):
            (m0, a0), (m1, a1) = loaded[i], loaded[i + 1]
            off = a0 if m0 == "off" else a1
            on = a1 if m0 == "off" else a0
            sessions.append({"order": order, "off": off, "on": on})
    return report(sessions, margin_min=float(os.environ.get("TIE_MARGIN_MIN", "0.02")),
                  outdir=None, replay=paths)

# ── 判詞輸出 ──────────────────────────────────────────────────────────────────────
class _Tee:
    """把判詞同時寫進 `report.txt` 與 stdout（doc 承諾的產物之一，先前只存在 json 裡）。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, txt):
        for st in self.streams:
            st.write(txt)
        return len(txt)

    def flush(self):
        for st in self.streams:
            st.flush()


def report(sessions: list[dict], margin_min: float, outdir, replay=None) -> int:
    # 判詞 stdout 也存成產物（doc 承諾的產物之一；先前只存在 json 裡）。
    # `--replay` 沒有 outdir ⇒ 只印不存。
    _f = open(os.path.join(outdir, "report.txt"), "w") if outdir else None
    _old = sys.stdout
    if _f:
        sys.stdout = _Tee(sys.stdout, _f)
    try:
        d = decide(sessions, margin_min)
        bad = sessions_valid(sessions)
        print("\n=== MTP-on tie gate ===")
        if replay:
            print("（REPLAY：由既有 log 重算，不算新量測）")
        for r, s in zip(d["gains"], sessions):
            print(f"  order {s['order']}: emit_on={s['on']['emit']:.4f} T_off={s['off']['token_ms']:.2f} ms/tok "
                  f"R_on={s['on']['round_ms']:.1f} ms/round  gain={r:.4f}  "
                  f"(off avg_ts={s['off']['avg_ts']:.2f} n_kept={s['off']['n_kept']}, "
                  f"on replay={s['on']['replay']}/{s['on']['rounds']})")
        for s_ in sessions:
            h = s_["on"].get("output_health")
            if h is not None:
                print(f"  order {s_['order']} on 輸出健康度：{oh.summary_line(h)}")
                if h["state"] == "NO_DATA" and not OH_REQUIRE_WITNESS:
                    print("      ⛔ --no-output-health：這一趟**沒有正確性見證** ⇒ 判詞不可引用。")
        print(f"  gain_bal={d['gain_bal']:.4f}  跑序散度={d['order_spread']:.3f}  margin={d['margin']:.3f} "
              f"(max(--margin-min, 跑序散度))")
        if bad:
            print("  ⛔ 有效性：" + "; ".join(bad))
        if d["void_reasons"]:
            print("  ⛔ 機制閘：" + "; ".join(d["void_reasons"]))
        if bad:
            d["verdict"] = "VOID"
        if d["verdict"] == "TIE":
            print(f"  ✅ TIE —— on 打平：per-round 產出 {d['gain_bal']:.3f}× off 的每 token 速率")
        elif d["verdict"] == "SHORTFALL":
            print(f"  ⛔ SHORTFALL —— 還缺 {100*(1+d['margin']-d['gain_bal']):.1f}%"
                  f"（gain {d['gain_bal']:.3f} vs 需 {1+d['margin']:.3f}）")
            print(f"     同形狀下 emit/round 需 ≥ {d['needed_emit']:.4f}（現 {d['emit_now']:.4f}）")
        else:
            print("  ⛔ VOID —— 這一趟不作判（機制閘或有效性不過）")
        d["sessions"] = [{k: v for k, v in s.items() if k in ("order", "off", "on")} for s in sessions]
        if outdir:
            os.makedirs(outdir, exist_ok=True)
            json.dump(d, open(os.path.join(outdir, "tie_gate.json"), "w"), indent=1)
            print(f"  產物：{os.path.join(outdir, 'tie_gate.json')}")
        return 0 if d["verdict"] == "TIE" else (1 if d["verdict"] == "SHORTFALL" else 2)
    finally:
        if _f:
            sys.stdout = _old
            _f.close()


# ── main ─────────────────────────────────────────────────────────────────────────
def main() -> int:
    _install_handlers()
    # `--replay` 是 nargs="+"，任何放在它後面的旗標都會被當成路徑（實測 argparse 會直接報
    # 「expected at least one argument」）⇒ 這個旗標先掃出來、位置無關。
    if "--replay-allow-prefill" in sys.argv:
        os.environ["TIE_REPLAY_ALLOW_PREFILL"] = "1"
        sys.argv = [a for a in sys.argv if a != "--replay-allow-prefill"]
    # 同一個 `--replay` 是 nargs="+" 的陷阱：放在它後面的旗標會被吃成 log 路徑 ⇒ 先掃掉。
    if "--no-output-health" in sys.argv:
        os.environ["TIE_NO_OUTPUT_HEALTH"] = "1"
        sys.argv = [a for a in sys.argv if a != "--no-output-health"]
    ap = argparse.ArgumentParser(description="MTP-on 打平 gate（同 session ABBA；盒閘綠才發車）")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true", help="只問盒子能不能跑（不吃 GPU）")
    g.add_argument("--go", action="store_true", help="盒閘過才跑 ABBA 判詞")
    g.add_argument("--watch", action="store_true", help="輪詢盒閘，綠了自動跑（盒閘綠即跑）")
    g.add_argument("--replay", nargs="+", metavar="LOG", help="由既有 log 重算判詞（2 或 4 份）")
    ap.add_argument("--replay-allow-prefill", action="store_true",
                    help="審計用：放行 n_prompt>0 的舊 log（計時含 prefill，判詞不可引用）。"
                         "位置無關：main() 先掃掉它再讓 argparse 解析（--replay 是 nargs=+，"
                         "放在它後面會被當成 log 路徑而報錯）")
    ap.add_argument("--no-output-health", action="store_true",
                    help="審計用：放行『沒有 CGC-SPEC-DUMP、因而沒有正確性見證』的舊 log。"
                         "⛔ **不放行退化**（DEGENERATE 一律 VOID）。位置無關。")
    g.add_argument("--selftest", action="store_true", help="純函式自測")
    ap.add_argument("--waive-reason", default="", help="盒閘未綠時的唯一放行方式（必須寫理由）")
    ap.add_argument("--poll-s", type=int, default=60, help="--watch 的輪詢間隔（預設 60）")
    ap.add_argument("--watch-max-min", type=int, default=180, help="--watch 的最長等待（分，預設 180）")
    ap.add_argument("--margin-min", type=float, default=0.02, help="最小雜訊帶（預設 2%%）")
    # 形狀：預設＝測試卡的 delivery cell（decode 支柱），兩臂同形狀
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--ubatch", type=int, default=512)
    ap.add_argument("--prompt", type=int, default=0)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--depth", type=int, default=512)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--ctx-size", type=int, default=4096)
    ap.add_argument("--warm-skip", type=int, default=64)
    ap.add_argument("--prompt-file", default=DEFAULT_PROMPT,
                    help="真文字（預設 server 自己那段巴黎文本）；給 '' 則退回隨機填充")
    ap.add_argument("--seed", type=int, default=1, help="CGC_SERVER_SEED ＋ --fixed-fill-seed")
    ap.add_argument("--k", type=int, default=2, help="--spec-draft-n-max（預設 2）")
    ap.add_argument("--tree", type=int, default=1, help="1=樹臂（CGC_SPEC_TREE=1）、0=鏈臂")
    ap.add_argument("--tree-w", type=int, default=1, help="樹寬（1 或 2）；預設 1")
    ap.add_argument("--caps", default="40-40:140")
    ap.add_argument("--oracle", type=int, default=1, help="一併開 CGC-ORACLE（附 head 天花板）")
    a = ap.parse_args()
    global OH_REQUIRE_WITNESS
    OH_REQUIRE_WITNESS = not bool(a.no_output_health or os.environ.get("TIE_NO_OUTPUT_HEALTH"))

    if a.selftest:
        return cmd_selftest()
    if a.replay:
        if a.replay_allow_prefill:
            os.environ["TIE_REPLAY_ALLOW_PREFILL"] = "1"
        return cmd_replay(a.replay)

    ok, lines, blocked, reading = box_gate()
    print("== 盒閘（l255_close 的讀數與判詞，同一套定義）==")
    for ln in lines:
        print(ln)
    print(f"  VERDICT: {'PASS' if ok else 'SHORT（' + ','.join(blocked) + '）'}")
    if a.check:
        return 0 if ok else 1

    # `--watch` ＝「盒閘綠即跑」：**等待本身就是使用者給的授權**，所以不要求 --waive-reason。
    # （這一支以前把不綠的盒子直接拒跑在前面，於是 --watch 的主用法永遠走不到輪詢迴圈 ——
    #   文件的承諾與程式不符，實測抓到。）waive 的語意相反：我知道不綠、照跑，故不等。
    if a.watch and not ok and a.waive_reason:
        print(f"[watch] 已 waive ⇒ 不輪詢，直接起跑（{a.waive_reason}）", flush=True)
    waited = 0
    while a.watch and not ok and not a.waive_reason:
        print(f"[watch] 盒閘未綠（{','.join(blocked) or '-'}），{a.poll_s}s 後再看（已等 {waited//60} 分）",
              flush=True)
        time.sleep(a.poll_s)
        waited += a.poll_s
        if waited > a.watch_max_min * 60:
            print("[watch] 超過最長等待，放棄（沒有量測產生）。", file=sys.stderr)
            return 2
        ok, lines, blocked, reading = box_gate()
        for ln in lines:
            print("  " + ln, flush=True)
        if ok:
            print("[watch] 盒閘綠 ⇒ 起跑", flush=True)
            break

    if not ok and not a.waive_reason:
        print("\n拒跑：盒閘未綠。要等它綠用 --watch；要現在跑必須 --waive-reason '<理由>'"
              "（照專案慣例留痕）。", file=sys.stderr)
        return 2
    if not ok:
        print(f"\n⚠ 盒閘未綠但已 waive：{a.waive_reason}", file=sys.stderr)
    # 「別的 llama」＝盒閘那一套（`server_window.decision` 優先走
    # `decode_window_harness.foreign_llama`，**以執行檔分類**，並有 pgrep 退路）。
    # 這裡先前自寫了一版掃「命令列字串」的：launch 它的 shell、`pgrep -f llama-bench`、編輯器
    # 全都會被判成 foreign ⇒ 沒人在跑也拒跑（實測抓到）。絕不再立第二套定義。
    if reading.get("foreign_llama"):
        print("拒跑：有其他 llama 行程在跑（同盒閘那行『別的 llama』）。", file=sys.stderr)
        return 2

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(ROOT, "Backup", f"tie_gate_{ts}")
    os.makedirs(outdir, exist_ok=True)
    print(f"\n產物目錄：{outdir}")
    print(f"形狀：-b {a.batch} -ub {a.ubatch} -p {a.prompt} -n {a.gen} -d {a.depth} -r {a.reps} "
          f"--ctx-size {a.ctx_size} --warm-skip {a.warm_skip} "
          f"--prompt-file {a.prompt_file or '(none: 隨機填充)'}")
    print(f"on 臂：--spec-type draft-mtp -k {a.k} tree={bool(a.tree)} W{a.tree_w}；"
          f"兩臂同 session、AB 與 BA 各一跑序（跑序效應由 margin 吸收）")
    json.dump({"reading_before": reading, "args": vars(a), "ts": ts},
              open(os.path.join(outdir, "meta.json"), "w"), indent=1)

    sessions = []
    for order in ("AB", "BA"):
        re_read = box_gate()[3]
        print(f"\n--- session {order}（起跑讀數 free%={re_read.get('launcher_free_pct')} "
              f"swap={re_read.get('swap_used_mb')}）")
        sessions.append(run_session(a, outdir, order))
        json.dump(sessions, open(os.path.join(outdir, "sessions.json"), "w"), indent=1)
    json.dump({"reading_after": box_gate()[3]}, open(os.path.join(outdir, "meta_after.json"), "w"), indent=1)

    rc = report(sessions, a.margin_min, outdir)
    print("\n（t/s 不引用；判詞只用 counters ＋ 同 session 的 avg_ns 配對）")
    return rc


if __name__ == "__main__":
    sys.exit(main())
