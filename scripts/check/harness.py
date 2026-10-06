#!/usr/bin/env python3
"""One front door for every windowed measurement on this line. `show windows` first.

WHY THIS EXISTS
---------------
Four things on this line each reimplemented "wait until the box is free, then run":

  * `decode_window_harness.py` (defines the probes; lane runner + HTML report)
  * `m123_gate_window.py`      (wraps the M1/M2/M3 oracle gate)
  * `plain_match_window.py`    (wraps plain_match_ab)
  * `server_window.py cmd_wait` (the shared probe itself, with its own wait loop)

They do not agree with each other, and the disagreement is not cosmetic. A launch into a busy box
measures the neighbour: the measured spread on identical configs here is 17%, and 2.5x on a bad day.
That failure is invisible in the number it produces, so it has to be caught before the launch -- and
today whether it is caught depends on which wrapper you happened to pick.

This file does NOT replace those wrappers. It is the FRONT DOOR: one place that knows

  1. which tools need a quiet window at all (the registry),
  2. what the box says right now (one shared probe, asked once),
  3. whether that verdict can be trusted physically, not just by bookkeeping
     (`--certify` runs the throughput sentinel; thermal NOMINAL alone has certified boxes that were
     ~5x slow -- 2026-09-20),
  4. and where the evidence went (`harness_runs.jsonl`, appended per run).

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does not run a server itself and it does not redefine any probe. Everything it knows about the
box comes from `server_window` and `window_sentinel`, so this file cannot drift from them.

HONEST BOUNDARY ON PROVENANCE
-----------------------------
The provenance recorded here is what THIS process saw: the box state before the run started, and
after it returned. The child process runs in its own address space. If the child is one of the
gated tools it asks the same probe for itself and carries its own evidence; if it is not, this
record is a statement about the box around the run, NOT about the child's internals. The record
says which case it is (`child_gated`) rather than implying more than it measured.

USAGE
-----
    python3 scripts/check/harness.py show                 # the window board (default)
    python3 scripts/check/harness.py show --json
    python3 scripts/check/harness.py run cb_headroom_probe -- --tag x --band any
    python3 scripts/check/harness.py run profile_duo --certify -- --profile prefill250
    python3 scripts/check/harness.py list                 # the registry
    python3 scripts/check/harness.py audit                # ungated launchers
    python3 scripts/check/harness.py selftest

BENCH (統一量測入口)
--------------------
    python3 scripts/check/harness.py bench \
        --arm "prod-new:!CGC_WAKE_POLL_US=999;CGC_SERVER_MTP=1" \
        --json /tmp/bench_out.json

    - 基底 = prod-new（CGC_DUMP_ENV=1 唯一權威）；arm 語法 PROFILE:!OVERRIDE;KEY=VAL
    - base gate：撞 base 鍵需 ! 宣告（實驗開關白名單 _EXPERIMENT_KNOBS 自動放行 + 記錄）
    - 產物契約：每臂含 base_check + sys_before/after（thermal/swap/pageins/memory_pressure/iostat）
    - 報告必須基於 prod-new + 自己的 env 增量；llama-bench 完整側參數見 _BENCH_DEFAULTS
"""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS_LOG = ROOT / "Backup" / "phase_decomp" / "harness_runs.jsonl"

PY = sys.executable or "python3"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- GPU 窗口原子鎖
# 窗口協調裡唯一原子的一道。wait_for_window 的 ps 快照是「開跑前看一眼」：兩個 harness 可在
# 對方 llama-bench 尚未 fork 的間隙都拿到連續 quiet、同時放行（TOCTOU），兩個 GPU 實驗於是並發、
# 互搶記憶體（2026-09-27 off_dense 被並發的 null_series 擠到 pp47/tg3.6）。flock 由內核持有、
# 綁定 open file description：第二個進程在第一個全程結束前阻塞；進程退出（含 kill/崩潰）內核自動
# 關 fd 釋鎖——沒有協作哨兵失效、殘留註冊的問題。
GPU_LOCK_PATH = "/tmp/flashkv_gpu_window.lock"


@contextmanager
def gpu_window_lock(held_by: str = "", timeout_s: float = None, poll_s: float = 3.0):
    """排他持有 GPU 窗口鎖，包住「實際跑 GPU 子進程」的那段；崩潰也由內核自動釋放。"""
    lf = open(GPU_LOCK_PATH, "a+")
    lf.seek(0)
    holder = lf.read().strip()
    acquired = False
    t0 = time.time()
    try:
        while True:
            try:
                fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                if timeout_s is not None and time.time() - t0 > timeout_s:
                    raise TimeoutError(
                        "GPU 窗口鎖等待逾時（%.0fs）；當前持有者：%s" % (timeout_s, holder))
                time.sleep(poll_s)
        lf.seek(0); lf.truncate()
        lf.write("%d\t%s\t%s\n" % (os.getpid(), time.strftime("%H:%M:%S"), held_by))
        lf.flush()
        yield lf
    finally:
        if acquired:
            lf.seek(0); lf.truncate(); lf.flush()
            fcntl.flock(lf, fcntl.LOCK_UN)
        lf.close()


# --------------------------------------------------------------------------- registry
# `needs_window` is the whole point of the registry: it is the question no single tool could answer
# for another. True = a busy box corrupts this tool's output. False = it parses files / does
# arithmetic, and a neighbour's server does not change its answer (only its speed). None = not yet
# labelled, which `show` reports as a gap instead of quietly treating as False.
#
# `launches` is not hand-maintained: it is merged from `server_window.audit()`, so a new launcher
# shows up here whether or not anyone remembered to edit this table.
REGISTRY: dict[str, dict] = {
    # --- line A (ace): engine layer, S1 / segment boundaries / per-layer KINDxOP
    "cb_headroom_probe.py": dict(needs_window=True, owner="lineA",
                                 purpose="re-read cb under a declared memory band"),
    "decode_step_profile.py": dict(needs_window=True, owner="lineA",
                                   purpose="per-step DECPROF capture (wait/cb/submit/gpu/union)"),
    "decode_sweep.py": dict(needs_window=True, owner="lineA", purpose="phase decomposition sweep"),
    "http_duo.py": dict(needs_window=True, owner="lineA", purpose="service-path two-axis probe"),
    "joint_reconcile.py": dict(needs_window=False, owner="lineA",
                               purpose="reconcile one step's account from a saved log"),
    "gate_registry.py": dict(needs_window=True, owner="lineA",
                             purpose="G0-G7 gate registry (long probe needs a server)"),
    "prod_profile.py": dict(needs_window=True, owner="lineA", purpose="delivery bars: prefill+decode"),
    "profile_duo.py": dict(needs_window=True, owner="lineA", purpose="llama-bench two-axis (delivery)"),
    "prod_matrix.py": dict(needs_window=True, owner="lineA", purpose="production matrix"),
    "llama_bench_matrix.py": dict(needs_window=True, owner="lineA", purpose="bench matrix resolver"),
    "paired_ab.py": dict(needs_window=True, owner="lineA", purpose="paired A/B (AB/BA interleaved)"),
    "ab_interleave.py": dict(needs_window=True, owner="lineA", purpose="interleaved A/B"),
    "window_sentinel.py": dict(needs_window=True, owner="lineA",
                               purpose="certify a window by measured throughput"),
    "decode_window_harness.py": dict(needs_window=True, owner="lineA",
                                     purpose="two decode lanes, judged report"),
    "server_window.py": dict(needs_window=False, owner="shared",
                             purpose="the shared probe + launcher audit"),
    # --- 窗口流程（2026-09-30）：流程本身與「怎麼跑」分開 ——
    #   window_runner.py 是**流程**（步驟以資料宣告：skip／fail-fast／可接續），
    #   rho_window.py 是**第一個宣告**（ρ 切分：S1 套用→S6 判詞）。
    # 為什麼兩者都標 needs_window=True：runner 不產生自己的讀數，它的 output 就是各步的 receipts
    # —— 那要由步驟的窗口決定（宣告裡發車的步不少），所以從嚴標。owner 留空（沿用預設的共用
    # 歸屬文件）：猜錯線名會把「請你先 gate 這支」送錯人。
    "window_runner.py": dict(needs_window=True,
                             purpose="generic window runner: steps as data (skip / fail-fast / resumable)"),
    "rho_window.py": dict(needs_window=True,
                          purpose="rho flag-split window: apply -> build -> oracle -> 4 authority rows -> verdict"
                                  "; --shadow = zero-GPU rehearsal of S1/S2 on a copy of src/ (repeatable)"),
    "window_gate.py": dict(needs_window=False, owner="shared", purpose="commit gate: launchers only go down"),
    "l2010_verdict.py": dict(needs_window=False, owner="shared",
                             purpose="L20-10 verdict: satisfiability -> quote gate -> pre-registered discriminators"),
    "skip_probe.py": dict(needs_window=False, owner="shared",
                          purpose="L25-2 skip-probe receptor: gate scan / paired plan / pre-registered delta-step verdict"),
    "l2010_window.py": dict(needs_window=True, owner="shared",
                            purpose="L20-10 clean-window pair: window check -> A -> B -> verdict (one command)"),
    "landing_ledger.py": dict(needs_window=False, owner="shared",
                              purpose="L20/L25 landing ledger: every card is closed-with-endpoint or "
                                      "blocked-with-entry (patches re-checked with git apply --check)"),
    # --- line I: instrumentation / cache geometry
    "cb_miss_regression.py": dict(needs_window=True, owner="lineI", purpose="cb miss regression"),
    "m123_oracle_gate.py": dict(needs_window=True, owner="lineI", purpose="M1/M2/M3 oracle gate"),
    "m123_gate_window.py": dict(needs_window=True, owner="lineI", purpose="oracle gate in a window"),
    # --- launchers found by the audit whose owning line is not recorded here. `owner` is left empty
    # on purpose so `show` falls back to the shared ownership document: a wrong line name sends the
    # "please gate this" request to someone who does not own the file.
    "mtp_accept_ab.py": dict(needs_window=True, purpose="MTP draft acceptance per carrier"),
    "mtp_long_sequence_verify.py": dict(needs_window=True, purpose="MTP long-sequence M1/M2/M3 verify"),
    "phase_split_ab.py": dict(needs_window=True, purpose="phase-split prefill A/B: armed slab vs clamp"),
    "plain_match_ab.py": dict(needs_window=True,
                              purpose="does batch-verify compute the same thing as per-token decode"),
    "plain_match_window.py": dict(needs_window=True, purpose="wrapper: plain_match_ab in a window"),
    "pool_curve.py": dict(needs_window=True, purpose="pool-size sweep, one budget per restart"),
    "slab_handoff_ab.py": dict(needs_window=True, purpose="slab to pool handoff A/B"),
    "test_decode_window_harness.py": dict(needs_window=True, purpose="self-test for harness verdicts"),
    # A driver whose `--measure` path launches, and whose `--pair` path does not. It is listed as
    # needing a window because the stance has to hold for the mode that launches: `harness.py
    # selftest` fails if a launcher has no stance at all, which is how this entry came to exist.
    "target25_report.py": dict(needs_window=True,
                               purpose="one-shot driver: measure -> decompose -> HTML report"),
    # Re-runs the two families that a reboot invalidates (P0/P1/P2 four arms, S1 E-series). It reads
    # its own window before EVERY step, which is why it is a launcher and not a wrapper: the gate has
    # to hold per step, because a run inside a family is what dirties the slot for the next row.
    "restart_rerun.py": dict(needs_window=True, owner="lineA",
                             purpose="reboot-gated re-run: P0/P1/P2 arms + S1 E-series, session-stamped"),
    # Runs one production arm and points `footprint --swapped` at the worker mid-arm. It is its own
    # gate (compressor quiet + no rival llama) and it records the other box numbers as labels; it is
    # listed here because a launcher with no stance is what `harness.py selftest` refuses.
    "arm_pressure_attrib.py": dict(needs_window=True, owner="lineA",
                                   purpose="per-process attribution of one arm's compressor traffic"),
    # Two launchers that arrived inside the 2026-09-25 multi-line commit and were never registered.
    # They are the reason `selftest` was red before this entry existed; the stance is declared, the
    # gating is not (that is the owning line's job, and `window_gate.py` tracks it separately).
    "cap_oomsweep.py": dict(needs_window=True, purpose="OOM sweep (owner's line not recorded here)"),
    "s1_ksweep.py": dict(needs_window=True, purpose="S1 k-sweep (owner's line not recorded here)"),
    # Promotion prerequisite for MTP-on: the candidate must carry its own same-shape off control
    # measured in one session against the frozen baseline (scripts/check/mtp_off_baseline.json).
    # It launches nothing -- it evaluates artifact JSONs, so no window stance is needed.
    "mtp_promotion_gate.py": dict(needs_window=False, owner="shared",
                                  purpose="MTP-on promotion prerequisite: paired off control vs frozen tg baseline"),
    # The corridor builder: matrix-direct bench products get the gate's product contract injected
    # (base_check/cell recomputed by harness's own predicates), and everything the harness gate
    # would have READ but this run did not (box_gate/charter/sys_*) is marked synthesized or null.
    # Reads files, shapes JSON -- proves no window of its own.
    "bench_ingest.py": dict(needs_window=False, owner="shared",
                            purpose="reshape bench products into the gate contract + corridor stats"),
    # 2026-09-28：這支被 audit 的**文字比對**誤判成 launcher —— 它列了執行檔路徑當掃描目標，
    # 又把啟動檔的名稱存成一個要**讀**的常數。兩者都不是「要跑一個 server」。
    # 它一個 server 都不會啟動：它跑 `--selftest`、`git log` 與讀 JSON。
    # 未宣告 ⇒ `selftest` 的「every launcher has a window stance」一直是紅的。
    # ⚠ 寫在這裡的註解也會被那個比對看到：別在註解裡照抄啟動形狀。
    "claim_instrument_check.py": dict(needs_window=False, owner="lineA",
                                      purpose="verify a cited number's instrument is armable/alive (reads files, launches nothing)"),
    # 同理：讀 mindmap.json 與 Backup/*.json，不啟動任何東西。
    "mindmap_void_check.py": dict(needs_window=False, owner="lineA",
                                  purpose="no VOID throughput may sit on a mindmap node's result surface (reads JSON, launches nothing)"),
    # 量具綁定：讀 stderr log 與產物 JSON，不啟動任何東西。
    # 2026-09-29：交付 cell 的四場空白嘗試，事後才發現日誌裡每個計數器都是 0 —— 旗標、白名單、
    # 符號、產物四項檢查全過。這支把「量具有沒有真的綁上圖」變成布林。
    "instrument_binding.py": dict(needs_window=False, owner="lineA",
                                  purpose="does the instrument actually run (reads logs, launches nothing)"),
}


def registry_with_audit(sw) -> list[dict]:
    """Hand labels merged with the live audit. Audit wins on `launches` because it reads the tree."""
    rep = sw.audit()
    launched = {r["file"] for r in rep["launchers"]}
    gated = {r["file"] for r in rep["gated"]}
    # The audit puts the file that DEFINES the probes under `source`: it is the truth these gates
    # are checked against, not a client of them. Counting it as an ungated launcher made the board
    # demand that the harness gate itself.
    source = {r["file"] for r in rep["source"]}
    owned = sw.ownership()
    rows = []
    for name in sorted(set(REGISTRY) | launched):
        e = dict(REGISTRY.get(name, {}))
        rows.append({
            "tool": name,
            "needs_window": e.get("needs_window", None),
            "launches_server": name in launched,
            "gated": name in gated,
            "is_source": name in source,
            "owner": e.get("owner") or owned.get(name) or "",
            "purpose": e.get("purpose", ""),
            "labelled": name in REGISTRY,
        })
    return rows


def needs_window_for(tool: str, rows: dict) -> bool:
    """Unlabelled means NEEDS a window.

    The unsafety is asymmetric: treating a measurement that needs quiet as one that does not
    produces a number that looks fine and describes the neighbour, while the reverse only makes
    someone wait. So the default errs toward waiting.
    """
    v = rows.get(tool, {}).get("needs_window", None)
    return True if v is None else bool(v)


# --------------------------------------------------------------------------- box view
def box_view(sw, port: int, need_mb: float) -> dict:
    d = sw.decision(port, need_mb)
    try:
        sentinel_others = _load("ws", "window_sentinel.py").others()
    except Exception:
        sentinel_others = None
    return {"port": port, "need_mb": need_mb, "decision": d,
            "other_session_pids": sentinel_others,
            "when": time.strftime("%Y-%m-%d %H:%M:%S")}


def _verdict(box: dict) -> tuple[str, str]:
    d = box["decision"]
    if box.get("other_session_pids"):
        # str() each item: the pids arrive as ints from one producer and as strings from another.
        return ("BUSY", "another session is running (pids %s)"
                % ", ".join(str(p) for p in box["other_session_pids"]))
    if not d["admits"]:
        why = "; ".join(f for f in [("port held" if d["port_held"] else ""),
                                    ("foreign llama: %s" % llama_desc(d["foreign_llama"])
                                     if d["foreign_llama"] else ""),
                                    ("reclaimable %.0fMB < %.0f" % (d["reclaimable_mb"],
                                                                    d["need_mb"])
                                     if d["reclaimable_mb"] < d["need_mb"] else ""),
                                    ("launcher refuses (%s)" % d["launcher_class"]
                                     if d["launcher_admits"] is False else "")] if f)
        # Say when the launcher would have started anyway: "refused" alone reads as "the box is
        # busy" when the truth is "our bar is stricter than the launcher's", and the two call for
        # opposite responses (wait vs lower the bar deliberately).
        if d["agree"] is False:
            why += " [DISAGREE: the launcher's own probe would admit; binding=%s]" % d["binding"]
        return ("REFUSED", why or ("binding=%s" % d["binding"]))
    if d["agree"] is False:
        return ("ADMITS*", "the two definitions disagree; binding=%s" % d["binding"])
    return ("ADMITS", "both definitions agree the box is quiet")


def llama_desc(items) -> str:
    """Render the shared probe's foreign-llama list as text.

    The producers hand back `(pid, exe)` PAIRS (`server_window._fallback_llama`, `decode_window_
    harness.foreign_llama`), while the fixtures in this file's own selftest used bare strings. So
    `",".join(...)` passed the selftest and raised `TypeError: sequence item 0: expected str
    instance, tuple found` in `show` -- the one command whose job is to say WHO is holding the box --
    precisely when a neighbour was holding it. The formatter takes every shape the probe can return,
    and is used by both readers of this field (the verdict's reason and the board's header).
    """
    out = []
    for it in items or []:
        if isinstance(it, (tuple, list)):
            out.append(" ".join(str(x) for x in it) if it else "")
        else:
            out.append(str(it))
    return ", ".join(x for x in out if x)


def box_lines(box, verdict, why) -> list[str]:
    """`show`'s box header, as data, so a selftest can assert on it without a live probe."""
    d = box["decision"]
    # "-" and not 0 for an absent launcher verdict: the launcher's bar and "no bar published" are
    # different statements, and a 0 would read as "the launcher would refuse anything".
    dash = lambda v: "-" if v is None else str(v)
    return [
        "WINDOW BOARD  %s" % box["when"],
        "  port %-5d %s" % (box["port"], "HELD by %s" % d["port_held"] if d["port_held"] else "free"),
        "  foreign llama: %s" % (llama_desc(d["foreign_llama"]) or "none"),
        "  reclaimable %.0f MB (need %.0f)   launcher %s%% (req %s%%, class %s)" % (
            d["reclaimable_mb"], box["need_mb"], dash(d["launcher_free_pct"]),
            dash(d["launcher_req_pct"]), dash(d["launcher_class"])),
        # `other_session_pids` is None whenever the sentinel is absent -- which is the common case,
        # and it used to reach `join()` as None (same defect class as the pairs above: the fixture
        # passed a list, the real box passed None).
        "  other sessions: %s" % (", ".join(str(p) for p in (box["other_session_pids"] or []))
                                  or "none"),
        "  VERDICT: %s -- %s" % (verdict, why),
    ]


def cmd_show(args) -> int:
    sw = _load("sw", "server_window.py")
    box = box_view(sw, args.port, args.need_mb)
    verdict, why = _verdict(box)
    d = box["decision"]
    rows = registry_with_audit(sw)

    if args.json:
        print(json.dumps({"box": box, "verdict": verdict, "why": why, "tools": rows}, indent=1))
        return 0

    print("\n".join(box_lines(box, verdict, why)))
    print()

    can = verdict == "ADMITS"
    print("%-34s %-7s %-8s %-6s %-6s %s" % ("TOOL", "WINDOW", "LAUNCHES", "GATED", "OWNER", "NOW"))
    print("-" * 100)
    for r in rows:
        nw = {True: "yes", False: "no", None: "?"}[r["needs_window"]]
        if r["needs_window"] is False:
            now = "runnable"
        elif can:
            now = "runnable"
        else:
            now = "blocked"
        if r["is_source"]:
            gate = "src"
        elif r["gated"]:
            gate = "yes"
        else:
            gate = "NO" if r["launches_server"] else "-"
        print("%-34s %-7s %-8s %-6s %-6s %s" % (
            r["tool"], nw, "yes" if r["launches_server"] else "-", gate,
            r["owner"] or "-", now))
    unlabelled = [r["tool"] for r in rows if not r["labelled"]]
    ungated = [r["tool"] for r in rows
               if r["launches_server"] and not r["gated"] and not r["is_source"]]
    print()
    print("ungated launchers (%d): %s" % (len(ungated), ", ".join(ungated) or "none"))
    print("unlabelled in registry (%d): %s" % (len(unlabelled), ", ".join(unlabelled) or "none"))
    if ungated:
        print("  -> a launch from any of these cannot be attributed on a busy box; run them through "
              "`harness.py run` or gate them (docs/SERVER_WINDOW_LEDGER_2026-09-19.md)")
    print("windows: %s" % ("open -- `harness.py run <tool> -- ...`" if can else "closed -- the box "
                           "is not admitting a launch right now"))
    if not can and d["agree"] is False and d["launcher_admits"] is not False:
        print("  note: the launcher's own probe would admit this box. The bar that refused is ours "
              "(--need-mb %.0f). Passing a lower `--need-mb` runs at the launcher's bar instead; "
              "do it deliberately, because the record will say which bar was used." % d["need_mb"])
    return 0


def cmd_list(args) -> int:
    sw = _load("sw", "server_window.py")
    rows = registry_with_audit(sw)
    if args.json:
        print(json.dumps(rows, indent=1))
        return 0
    for r in rows:
        print("%-34s window=%-5s launches=%-5s gated=%-5s owner=%-6s %s" % (
            r["tool"], {True: "yes", False: "no", None: "?"}[r["needs_window"]],
            "yes" if r["launches_server"] else "-", "yes" if r["gated"] else "no",
            r["owner"] or "-", r["purpose"]))
    return 0


# --------------------------------------------------------------------------- run
def wait_for_window(sw, port: int, need_mb: float, timeout_s: float, poll_s: float,
                    quiet_samples: int = 2) -> tuple[bool, str, float]:
    """Poll the shared probe until it admits, needing `quiet_samples` consecutive quiet readings.

    Two, not one: a single quiet reading here has repeatedly been a trough between two neighbours'
    launches, and the run started on it was measured inside someone else's load.
    """
    t0 = time.time()
    streak = 0
    last = ""
    while True:
        ok, why = sw.record(port, need_mb, where="harness.wait", gated=False)
        if ok:
            streak += 1
            if streak >= quiet_samples:
                return True, why, time.time() - t0
        else:
            streak = 0
            last = why
        if time.time() - t0 > timeout_s:
            return False, last or "timed out without a quiet window", time.time() - t0
        time.sleep(poll_s)


def cmd_run(args) -> int:
    sw = _load("sw", "server_window.py")
    tool = args.tool if args.tool.endswith(".py") else args.tool + ".py"
    path = HERE / tool
    if not path.exists():
        print("no such tool: %s (looked in %s)" % (tool, HERE), file=sys.stderr)
        print("`harness.py list` shows what is registered.", file=sys.stderr)
        return 2

    rows = {r["tool"]: r for r in registry_with_audit(sw)}
    meta = rows.get(tool, {})
    needs = needs_window_for(tool, rows)
    if meta.get("needs_window", None) is None:
        print("[harness] %s is not in the registry: assuming it needs a window. Label it in "
              "REGISTRY if that is wrong." % tool, file=sys.stderr)

    # 跑前立項閘：需要窗口的速度實驗必須帶 charter（現狀/目標/驗收）；純分析工具（needs=False）
    # 不強制。run 不直接對應 arm profile，故不檢查 profile 維度（傳 []）。
    charter_rec = None
    cv = getattr(args, "charter", None)
    if needs:
        if cv is None:
            print("[harness] 速度實驗需 --charter（現狀/目標/驗收）；純分析工具才可不帶。",
                  file=sys.stderr)
            return 2
        if cv == "waive":
            reason = getattr(args, "waive_reason", None)
            if not reason:
                print("[harness] --charter waive 必須帶 --waive-reason。", file=sys.stderr)
                return 2
            charter_rec = {"waived": True, "reason": reason}
            print(f"[harness][charter] 急件豁免（waive）：{reason}")
        elif cv == "none":
            print("[harness] --charter none 後門已關閉：給路徑立項或 --charter waive "
                  "--waive-reason。", file=sys.stderr)
            return 2
        else:
            ch = _load_charter(cv)
            cok, cerr = _charter_gate(ch, [])
            if not cok:
                print("[harness] charter gate FAIL — 拒跑：", file=sys.stderr)
                for e in cerr:
                    print("    " + e, file=sys.stderr)
                return 2
            charter_rec = {"id": ch.get("id"), "owner": ch.get("owner")}
            print("[harness][charter] 立項通過：%s" % ch.get("id"))

    if needs:
        ok, why, waited = wait_for_window(sw, args.port, args.need_mb, args.window_timeout_s,
                                          args.poll_s)
        if not ok:
            print("[harness] no window in %.0fs: %s" % (waited, why), file=sys.stderr)
            print("[harness] nothing was run, so nothing was recorded as data.", file=sys.stderr)
            return 3
        sw.record(args.port, args.need_mb, where="harness.launch", gated=True)
        print("[harness] window admitted after %.0fs" % waited)
    else:
        print("[harness] %s does not need a window (registry); running directly" % tool)

    cert = None
    if args.certify:
        ws = _load("ws", "window_sentinel.py")
        ts, rc = ws.measure()
        ref = None
        try:
            ref = json.loads((HERE / "window_sentinel_ref.json").read_text())
        except Exception:
            pass
        cert = {"measured_ts": ts, "rc": rc, "ref": ref}
        if ts is None or rc != 0:
            print("[harness] sentinel could not measure throughput -- refusing to certify",
                  file=sys.stderr)
            return 4
        floor = (ref or {}).get("median_ts")
        if floor and ts < float(floor) * args.certify_frac:
            print("[harness] REFUSED: %.2f t/s < %.2f x reference %.2f -- this window is degraded "
                  "even though the box looks quiet" % (ts, args.certify_frac, float(floor)),
                  file=sys.stderr)
            return 5
        print("[harness] certified: %.2f t/s (ref %.2f)" % (ts, float(floor or 0)))

    cmd = [PY, str(path)] + list(getattr(args, "rest", []) or [])
    print("[harness] $ %s" % " ".join(cmd))
    t0 = time.time()
    if needs:
        # 實際 GPU 運行全程持原子鎖，避免與其他 harness 並發
        with gpu_window_lock(held_by="run " + tool, timeout_s=args.window_timeout_s):
            p = subprocess.run(cmd, cwd=str(ROOT))
    else:
        p = subprocess.run(cmd, cwd=str(ROOT))
    dur = time.time() - t0

    prov = sw.provenance(args.port, args.need_mb)
    rec = {"when": time.strftime("%Y-%m-%dT%H:%M:%S"), "tool": tool, "argv": cmd,
           "exit": p.returncode, "duration_s": round(dur, 1),
           "needs_window": needs, "child_gated": bool(meta.get("gated")),
           "certified": cert, "window_class": prov["class"], "window_why": prov["why"],
           "decision_now": prov["now"], "owner": meta.get("owner", ""),
           "charter": charter_rec}
    try:
        RUNS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with RUNS_LOG.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
    except Exception as e:
        print("[harness] could not append the run record: %s" % e, file=sys.stderr)
    print("[harness] exit=%s in %.0fs | window=%s | record: %s" % (
        p.returncode, dur, prov["class"], RUNS_LOG))
    return p.returncode


def cmd_audit(args) -> int:
    wg = _load("wg", "window_gate.py")
    if args.json:
        print(json.dumps(wg.audit_now(), indent=1))
        return 0
    ok, msgs = wg.check()
    for m in msgs:
        print(m)
    return 0 if ok else 1


# --------------------------------------------------------------------------- selftest
def selftest() -> int:
    fails = []

    def expect(name, got, want):
        ok = got == want
        print("  %s %s: %r" % ("ok  " if ok else "FAIL", name, got))
        if not ok:
            fails.append(name)

    print("registry is the point of the tool: every launcher must appear, labelled or not")
    sw = _load("sw", "server_window.py")
    rows = registry_with_audit(sw)
    launched = {r["tool"] for r in rows if r["launches_server"]}
    expect("audit found launchers", len(launched) > 0, True)
    expect("every launcher has a window stance", all(
        r["needs_window"] is not None for r in rows if r["launches_server"]), True)

    print("\nthe three stances are distinct; unknown is not silently False")
    expect("no-window tool exists", any(r["needs_window"] is False for r in rows), True)
    expect("yes-window tool exists", any(r["needs_window"] is True for r in rows), True)

    print("\nverdict names the blocker instead of just refusing")
    base = {"decision": {"admits": False, "port_held": "1234", "foreign_llama": [],
                         "reclaimable_mb": 9999.0, "need_mb": 1.0, "launcher_admits": True,
                         "binding": "harness", "agree": True, "launcher_class": "k"},
            "other_session_pids": None, "need_mb": 1.0}
    v, why = _verdict(base)
    expect("refused", v, "REFUSED")
    expect("and says why", "port held" in why, True)

    print("\nthe blocker is RENDERED on the probe's real shape (pid, exe) -- regression")
    # The tuple shape is what `server_window._fallback_llama` / `decode_window_harness.foreign_llama`
    # actually return. `",".join(...)` on it raised TypeError and killed `show` -- so the board
    # crashed exactly when a neighbour held the box, and the fixtures above (bare strings) could
    # never catch it. Both readers of the field are asserted below, on the real shape.
    busy_decision = dict(base["decision"], foreign_llama=[(44076, "llama-server")],
                         launcher_free_pct=91.0, launcher_req_pct=30.0)
    busy_box = {"decision": busy_decision, "other_session_pids": None, "need_mb": 1.0,
                "port": 8080, "when": "fixture"}
    v, why = _verdict(busy_box)
    expect("still refuses", v, "REFUSED")
    expect("the reason names the pid and the exe", "44076" in why and "llama-server" in why, True)
    lines = box_lines(busy_box, v, why)
    expect("show's own header renders the pair", any("44076 llama-server" in ln for ln in lines), True)
    expect("and never prints an empty list as a blocker",
           any("foreign llama: none" in ln for ln in box_lines(
               {"decision": dict(busy_decision, foreign_llama=[]), "other_session_pids": None,
                "need_mb": 1.0, "port": 8080, "when": "fixture"}, "ADMITS", "q")), True)
    expect("the probe's pair shape", llama_desc([(1, "a"), (2, "b")]), "1 a, 2 b")
    expect("the bare-int shape other probes return", llama_desc([999]), "999")
    expect("empty stays empty", llama_desc([]), "")

    print("\nanother session overrides a box that otherwise looks quiet")
    v, _ = _verdict({"decision": dict(base["decision"], admits=True, port_held=False),
                     "other_session_pids": ["999"], "need_mb": 1.0})
    expect("busy wins", v, "BUSY")

    print("\ndisagreement is visible, not averaged away")
    v, _ = _verdict({"decision": dict(base["decision"], admits=True, port_held=False, agree=False),
                     "other_session_pids": [], "need_mb": 1.0})
    expect("admits-with-asterisk", v, "ADMITS*")

    print("\nwait needs a streak, not one lucky reading")
    calls = {"n": 0}

    def fake_record(port, need_mb, where="", gated=False):
        calls["n"] += 1
        return (calls["n"] >= 3), "why%d" % calls["n"]

    real = sw.record
    sw.record = fake_record
    try:
        ok, why, waited = wait_for_window(sw, 8080, 1.0, timeout_s=5, poll_s=0.01,
                                          quiet_samples=2)
        expect("waits for 2 consecutive", (ok, calls["n"]), (True, 4))
    finally:
        sw.record = real

    print("\nthe box view comes from the shared probe's real producer, not from a shape invented here")
    # This check exists because the old selftest built the decision dict as a FIXTURE and asserted on
    # it, so `show` -- this file's own default command -- called `sw.decision()` for months while that
    # function did not exist, and the selftest stayed green. Assert the producer, not a mock of it.
    got = sw.decision(need_mb=1.0)
    for k in ("admits", "harness_admits", "launcher_admits", "agree", "binding", "reclaimable_mb",
              "port_held", "foreign_llama", "launcher_class"):
        expect("decision() carries %s" % k, k in got, True)
    expect("agree is never True without a published launcher verdict",
           got["agree"] is True and got["launcher_admits"] is not None or got["agree"] is not True,
           True)

    print("\nan unregistered tool is treated as needing a window, never as not needing one")
    by = {r["tool"]: r for r in rows}
    expect("known no-window tool -> False", needs_window_for("joint_reconcile.py", by), False)
    expect("known yes-window tool -> True", needs_window_for("profile_duo.py", by), True)
    expect("unregistered -> True (errs toward waiting)",
           needs_window_for("no_such_tool.py", by), True)

    print("\nbench: the prod-new swap arms are enforced, and a disarmed A/B arm is declared")
    armed_env = {k: v for k, v in zip(_SWAP_ARM_KEYS, ("1", "2", "1"))}
    ok, missing = _swap_arm_gate(_BASE_PROFILE_DEFAULT, armed_env, set())
    expect("all three armed -> pass, nothing missing", (ok, missing), (True, []))
    ok, missing = _swap_arm_gate(_BASE_PROFILE_DEFAULT, {**armed_env, "CGC_B_SCHEME": "0"}, set())
    expect("a key set to 0 counts as OFF (run_server.sh drops it, so it is absent upstream)",
           (ok, [m["key"] for m in missing]), (False, ["CGC_B_SCHEME"]))
    ok, _ = _swap_arm_gate(_BASE_PROFILE_DEFAULT,
                           {k: armed_env[k] for k in _SWAP_ARM_KEYS if k != "CGC_B_SCHEME"}, set())
    expect("an absent key counts as OFF", ok, False)
    ok, missing = _swap_arm_gate(_BASE_PROFILE_DEFAULT,
                                 {k: armed_env[k] for k in _SWAP_ARM_KEYS if k != "CGC_B_SCHEME"},
                                 {"CGC_B_SCHEME"})
    expect("declared with ! -> allowed (the A/B arm is legitimate) and recorded as declared",
           (ok, [m["declared"] for m in missing]), (True, [True]))
    expect("a non-prod-new profile gets no opinion from this gate",
           _swap_arm_gate("prefill250", {}, set())[0], True)

    print("\nbench: the cell knobs actually reach the llama-bench command line")
    ba = argparse.Namespace(**_BENCH_DEFAULTS, workdir="/tmp/x", json_path="/tmp/x.json",
                            spec_type="", spec_draft_n_max=None)
    c0 = _bench_cmd(ba, ["prod-new"])
    expect("fixed-fill-seed is on by default and reaches the command (the run3-shaped hole)",
           ("--fixed-fill-seed" in c0, c0[c0.index("--fixed-fill-seed") + 1]), (True, "1"))
    expect("steady-state cell is echoed as one dict",
           _cell(ba)["fixed_fill_seed"], 1)
    c1 = _bench_cmd(argparse.Namespace(**{**vars(ba), "fixed_fill_seed": 0}), ["prod-new"])
    expect("an explicit seed=0 passes through rather than being overridden by the default",
           c1[c1.index("--fixed-fill-seed") + 1], "0")
    # 【CGC 2026-10-04】`--ubatch` 曾是 `_BENCH_DEFAULTS` 的一個死欄位：只有 `-b` 到得了 matrix，
    # 而 matrix 的舊語意「有 -b 就 -ub 同值」會把新預設格宣告的 512 蓋成 2048 ⇒ 契約拒跑。
    # 這條把「成對送出」釘住——缺一邊就是無聲地量錯形狀（同 fixed-fill-seed 的課）。
    expect("--batch and --ubatch both reach the matrix (default cell is b2048/ub512)",
           (c0[c0.index("--batch") + 1], c0[c0.index("--ubatch") + 1]), ("2048", "512"))
    c6 = _bench_cmd(argparse.Namespace(**{**vars(ba), "ubatch": 256}), ["prod-new"])
    expect("an explicit non-default ubatch passes through, not overridden by -b",
           c6[c6.index("--ubatch") + 1], "256")
    expect("no --no-warmup unless asked (warm-up is the pool-warm equivalent)",
           "--no-warmup" in c0, False)
    expect("and it appears when asked",
           "--no-warmup" in _bench_cmd(argparse.Namespace(**{**vars(ba), "warmup": False}),
                                       ["prod-new"]), True)
    expect("--prompt-file only when set",
           ("--prompt-file" in c0,
            "--prompt-file" in _bench_cmd(argparse.Namespace(**{**vars(ba), "prompt_file": "/p.txt"}),
                                          ["prod-new"])), (False, True))

    print("\nbench: 第五道閘門（量具綁定）—— 跑前擋前提、跑後擋拒寫")
    ib = _load("ibind_selftest", "instrument_binding.py")
    # 跑前那一半：用假的 resolve（真的 resolve 要跑 run_server.sh，而這裡要驗的是閘門本身）
    def _fake_resolve(envs):
        return lambda s: {"env": {**dict(_parse_arm(s)[1]), **(envs or {})}}

    # ⚠ 變數名不要叫 `fails`：那是本 selftest 的失敗收集器（:555），蓋掉它會讓
    # 「有失敗」與「沒失敗」反過來報（2026-09-29 自己踩到）。
    okp, pfails, pwarns = _instrument_precheck(
        ib, _fake_resolve({}), ["prod-new:CGC_MISS_MASK_DBG=1"], [])
    expect("跑前：DBG 開了但前提 MISS_MASK 不在 ⇒ 拒跑", (okp, len(pfails)), (False, 1))
    okp, _, _ = _instrument_precheck(
        ib, _fake_resolve({"CGC_MISS_MASK": "1"}), ["prod-new:CGC_MISS_MASK_DBG=1"], [])
    expect("跑前：把前提補上 ⇒ 放行（prod-new 基底本來就有它）", okp, True)
    okp, pfails, pwarns = _instrument_precheck(
        ib, _fake_resolve({"CGC_MISS_MASK": "1", "CGC_MISS_MASK_DBG": "1"}),
        ["prod-new"], ["rb_feed"])
    expect("跑前：要求 rb_feed 但 SEG_BATCH 沒開 ⇒ 拒跑",
           (okp, "SEG_BATCH" in pfails[0][1][0]), (False, True))
    okp, _, _ = _instrument_precheck(
        ib, _fake_resolve({"CGC_MISS_MASK": "1", "CGC_MISS_MASK_DBG": "1"}),
        ["prod-new:CGC_SEG_BATCH=1"], ["rb_feed"])
    expect("... 把 SEG_BATCH 補上就放行（同一條探針，只差一個開關）", okp, True)
    okp, _, pwarns = _instrument_precheck(
        ib, _fake_resolve({}), ["prod-new"], ["prefetch"])
    expect("跑前：跑前判不了的探針 ⇒ 放行但要警告（不是静默放行）",
           (okp, len(pwarns) >= 1), (True, True))
    expect("跑前：沒武裝任何受檢量具的臂 ⇒ 放行（沒有承諾）",
           _instrument_precheck(ib, _fake_resolve({}), ["prod-new"], [])[0], True)
    # 跑後那一半：拒寫的判定，以及它必須說出是哪一臂
    g = _instrument_gate(ib, [{"tag": "a", "instrument": {"verdict": "BOUND", "why": "x"}},
                              {"tag": "b", "instrument": {"verdict": "UNBOUND", "why": "全 0"}}])
    expect("跑後：任一臂 UNBOUND ⇒ 整份拒寫", (g["ok"], [d["tag"] for d in g["arms"]]), (False, ["a", "b"]))
    g2 = _instrument_gate(ib, [{"tag": "a", "instrument": {"verdict": "BOUND"}},
                               {"tag": "b", "instrument": {"verdict": "N/A"}}])
    expect("跑後：BOUND 與 N/A 都放行", g2["ok"], True)
    g3 = _instrument_gate(ib, [{"tag": "a"}])
    expect("跑後：產物沒有 instrument 欄位 ⇒ 拒寫，且理由不是「量具沒綁上」",
           (g3["ok"], g3["arms"][0]["verdict"]), (False, "NO-FIELD"))
    g4 = _instrument_gate(ib, [{"tag": "a", "instrument": {"verdict": "UNBOUND"}}],
                          waive_reason="急件")
    expect("跑後：豁免則放行，但理由要留在產物裡",
           (g4["ok"], g4["waived"]), (False, "急件"))
    c3 = _bench_cmd(argparse.Namespace(**{**vars(ba), "require_instrument": "rb_feed,prefetch"}),
                    ["prod-new"])
    expect("--require-instrument 只在宣告時出現，且真的傳到 matrix",
           ("--require-instrument" in c0, c3[c3.index("--require-instrument") + 1]),
           (False, "rb_feed,prefetch"))
    # 【CGC 2026-09-29】剎車與看門狗：同一個課——預設不出現，宣告了就要真的送到 matrix
    # （否則「我以為我設了逾時」會變成又一場沒有產物的等待）。
    c4 = _bench_cmd(argparse.Namespace(**{**vars(ba), "arm_timeout": 420.0,
                                          "stall_watch": 45.0, "stall_kill": True}),
                    ["prod-new"])
    expect("剎車／看門狗預設不出現在命令列",
           [f in c0 for f in ("--arm-timeout", "--stall-watch", "--stall-kill")],
           [False, False, False])
    expect("宣告了就真的傳到 matrix（含 stall-kill 這種旗標）",
           (c4[c4.index("--arm-timeout") + 1], c4[c4.index("--stall-watch") + 1],
            "--stall-kill" in c4), ("420.0", "45.0", True))

    # verify 這扇門（另一條路：arm_two_pass 直接叫 matrix，不經過上面的 cmd_bench）
    def _vres(emb):
        return {"arms": [{"arm": "prod-new", "clean": {"matrix": emb}, "instrumented": {"matrix": emb}}]}

    gv = _instrument_gate_verify(ib, _vres([{"tag": "i", "instrument": {"verdict": "UNBOUND"}}]))
    expect("verify：嵌在 result.json 裡的 UNBOUND 臂 ⇒ 拒收報告", (gv["ok"], gv["checked"]), (False, 2))
    gv2 = _instrument_gate_verify(ib, _vres([{"tag": "i", "instrument": {"verdict": "N/A"}},
                                             {"tag": "c", "instrument": {"verdict": "BOUND"}}]))
    expect("verify：clean 輪 N/A ＋ instrumented 輪 BOUND ⇒ 放行", gv2["ok"], True)
    gv3 = _instrument_gate_verify(ib, _vres([{"tag": "old", "rows": []}]))
    expect("verify：閘門存在之前的舊 result.json ⇒ 標 legacy 而不回溯拒收",
           (gv3["ok"], gv3["legacy"]), (True, True))
    gv4 = _instrument_gate_verify(ib, _vres([{"tag": "i", "instrument": {"verdict": "UNBOUND"}}]),
                                  waive_reason="急件")
    expect("verify：豁免要留在 result.json 裡（不是只印在 stdout）", gv4["waived"], "急件")
    gv5 = _instrument_gate_verify(ib, {"arms": []})
    expect("verify：沒有 matrix 產物（例如 gate 在跑任何 GPU 前就擋下）⇒ 不拒收",
           (gv5["ok"], gv5["checked"]), (True, 0))

    # 第六道閘門：spec 臂的 draft 死亡（逐 rep drafted=0）——判準與理由都要能離線重讀。
    gd1 = _spec_draft_gate([{"tag": "dead", "spec_draft": {
        "state": "INVALID", "n_reps": 5,
        "dead": [{"phase": "timed", "rounds": 128, "drafted": 0, "emit_per_round": "1.0000"}],
        "why": "4/5 輪 rep 的 draft 死亡"}}])
    expect("spec 健康閘：任一 rep drafted=0 ⇒ 拒收", gd1["ok"], False)
    gd2 = _spec_draft_gate([{"tag": "ok", "spec_draft": {
        "state": "OK", "n_reps": 5, "dead": [], "why": "5/5 輪 rep 的 drafted>0（最小 drafted=98）"}}])
    expect("spec 健康閘：每一輪 drafted>0 ⇒ 放行", gd2["ok"], True)
    gd3 = _spec_draft_gate([{"tag": "off", "spec_draft": {
        "state": "NOT_SPEC", "n_reps": 0, "dead": [], "why": "不是 spec 路徑"}}])
    expect("spec 健康閘：非 spec 臂（off）⇒ 與它無關，放行", gd3["ok"], True)
    gd4 = _spec_draft_gate([{"tag": "old", "rows": [], "avg_ts": 99.9}])
    expect("spec 健康閘：舊產物沒有 spec_draft 欄 ⇒ 不回溯（但記錄 N/A）",
           (gd4["ok"], gd4["checked"], gd4["arms"][0]["verdict"]), (True, 0, "N/A"))

    # 成對閘（兩扇門共用同一個判準：只有「產物旁邊有沒有 log」）
    # ⚠ 這個函式後面還有 `import tempfile`（:802/:863），所以 `tempfile` 對整個函式而言是區域名；
    # 這裡要先自己匯入一次，否則會 UnboundLocalError（2026-09-29 自己踩到）。
    import tempfile
    _rp = tempfile.mkdtemp(prefix="hv_pair_")
    try:
        _mj = os.path.join(_rp, "matrix.json")
        with open(_mj, "w") as f:
            json.dump([{"tag": "prod-new", "rows": [{"n_prompt": 0, "n_gen": 64}]}], f)
        bare = ib.pair_status(_mj)
        expect("成對閘：產物旁邊沒 log ⇒ paired=False，且說得出它缺什麼",
               (bare["paired"], "不可引用" in bare["why"] or "缺成對" in bare["why"]), (False, True))
        ib.write_pair_bundle(_mj, [("prod-new", "p0_n64", "CGC-RB-FEED: feeds=1\n")])
        expect("成對閘：補上後就成對（同一份判準，不必重寫）",
               ib.pair_status(_mj)["paired"], True)
        expect("auto-sync 的界線：repo 內的產物才回寫決策面（/tmp 的診斷產物不回寫）",
               (_inside_root("Backup/x.json"), _inside_root("/tmp/x.json")), (True, False))
        vres = {"arms": [{"arm": "prod-new", "clean": {"matrix_json": _mj},
                          "instrumented": {"matrix_json": os.path.join(_rp, "missing.json")}}]}
        pg = _pair_gate_verify(ib, vres)
        expect("成對閘（verify）：一輪有、一輪沒有 ⇒ 整份拒收，且逐輪點名",
               (pg["paired"], [p["pass"] for p in pg["passes"]], pg["checked"]),
               (False, ["prod-new/clean", "prod-new/instrumented"], 2))
        expect("成對閘（verify）：兩輪都沒有產物（gate 在跑前就擋）⇒ 不拒收",
               _pair_gate_verify(ib, {"arms": [{"arm": "a"}]})["paired"], True)
    finally:
        shutil.rmtree(_rp, ignore_errors=True)

    print("\nthe arm's swap line reads where the matrix actually puts the numbers")
    real = {"launch": {"swap_used_mb": 7862.44}, "end": {"swap_used_mb": 10000.19},
            "worst": {"max_swap_mb": 9060.81}}
    line = _swap_line(real)
    expect("growth is computed from launch/end", "growth=+2138 MiB" in line, True)
    expect("and worst is the real one, not None", "worst=9060.81" in line, True)
    empty = _swap_line({})
    expect("absent memory prints n/a", "growth=n/a" in empty, True)
    expect("and names the field that does have it",
           "attribution.swap_growth_mb" in empty, True)
    expect("absence is NEVER rendered as a zero (the 14-arm bug)", "+0" in empty, False)
    expect("and neither is a half-read memory — one end missing is not a zero either",
           "growth=n/a" in _swap_line({"launch": {"swap_used_mb": 1.0}}), True)
    stock = _swap_stock_line({"verdict": "advise", "why": "swap 存量 8037 MiB > 舊起跑線"})
    expect("the stock line says it is a label", "標籤" in stock and "起跑閘" in stock, True)
    expect("and it is not spelled as a verdict (the old `[swap] verdict=advise`)",
           "verdict=" in stock, False)

    print("\ncharter gate: 跑前立項（現狀/目標/假設/驗收），缺欄或出處不存在就拒跑")
    import tempfile
    tmpd = Path(tempfile.mkdtemp())
    base_src = tmpd / "baseline.json"
    base_src.write_text(json.dumps([{"warm_skip_applied": True}]))

    def good_charter(**over):
        c = {
            "id": "e-x", "owner": "lineA",
            "question": "ρ 批次化能否把 decode 從 13.99 提到 ≥15 t/s？",
            "baseline": {"metric": "decode 13.99 t/s",
                         "cell": "prod-new 穩態 warm-skip 64",
                         "source": str(base_src)},
            "hypothesis": {"mechanism": "cb 藏進 GPU busy",
                           "expected": "+14% ~ +19%",
                           "basis": "docs/CB_DELIVERY_SETTLED_2026-09-23.md"},
            "acceptance": {"success": "配對增益 ≥3%",
                           "falsify": "增益 <3% → 判死",
                           "on_fail": "revert、登記"},
        }
        c.update(over)
        return c

    expect("完整合法 charter -> pass",
           _charter_gate(good_charter(), ["prod-new"], root=tmpd)[0], True)
    expect("缺頂層欄位 -> fail",
           _charter_gate(good_charter(id=""), ["prod-new"], root=tmpd)[0], False)
    expect("question「測一下」-> fail",
           _charter_gate(good_charter(question="測一下速度"), ["prod-new"], root=tmpd)[0], False)
    expect("baseline 缺 metric -> fail",
           _charter_gate(good_charter(baseline={"cell": "x", "source": str(base_src)}),
                         ["prod-new"], root=tmpd)[0], False)
    expect("baseline.source 不存在 -> fail",
           _charter_gate(good_charter(baseline={"metric": "x", "cell": "x",
                                               "source": str(tmpd / "nope.json")}),
                         ["prod-new"], root=tmpd)[0], False)
    expect("hypothesis 缺 basis -> fail",
           _charter_gate(good_charter(hypothesis={"mechanism": "x", "expected": "+5%"}),
                         ["prod-new"], root=tmpd)[0], False)
    expect("acceptance 缺 falsify -> fail",
           _charter_gate(good_charter(acceptance={"success": "x", "on_fail": "y"}),
                         ["prod-new"], root=tmpd)[0], False)
    expect("非 prod-new 無理由 -> fail",
           _charter_gate(good_charter(), ["prefill250"], root=tmpd)[0], False)
    expect("非 prod-new 有 non_prod_reason -> pass",
           _charter_gate(good_charter(non_prod_reason="只能在 prefill250 回答"),
                         ["prefill250"], root=tmpd)[0], True)
    cold = tmpd / "cold.json"
    cold.write_text(json.dumps([{"warm_skip_applied": False}]))
    expect("宣稱穩態但 warm_skip_applied 全 False -> fail",
           _charter_gate(good_charter(baseline={"metric": "x", "cell": "穩態",
                                               "source": str(cold)}),
                         ["prod-new"], root=tmpd)[0], False)
    warm = tmpd / "warm.json"
    warm.write_text(json.dumps([{"cell": {"warm_skip_applied": True}}]))
    expect("穩態且 warm_skip_applied True -> pass",
           _charter_gate(good_charter(baseline={"metric": "x", "cell": "穩態",
                                               "source": str(warm)}),
                         ["prod-new"], root=tmpd)[0], True)

    print("GPU 窗口原子鎖：flock 互斥、釋放可獲取、逾時報錯（崩潰由內核自動釋放）")
    global GPU_LOCK_PATH
    import tempfile
    _orig_lock = GPU_LOCK_PATH
    with tempfile.TemporaryDirectory() as _lkd:
        GPU_LOCK_PATH = str(Path(_lkd) / "w.lock")
        try:
            with gpu_window_lock(held_by="selftestA"):
                _f2 = open(GPU_LOCK_PATH); _blocked = False
                try:
                    fcntl.flock(_f2, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    _blocked = True
                _f2.close()
                expect("持有期間第二個獲取被擋", _blocked, True)
            _f3 = open(GPU_LOCK_PATH)
            fcntl.flock(_f3, fcntl.LOCK_EX | fcntl.LOCK_NB); _f3.close()
            expect("釋放後可重新獲取", True, True)
            _f4 = open(GPU_LOCK_PATH)
            fcntl.flock(_f4, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _timed_out = False
            try:
                with gpu_window_lock(held_by="selftestB", timeout_s=0.5, poll_s=0.2):
                    pass
            except TimeoutError:
                _timed_out = True
            fcntl.flock(_f4, fcntl.LOCK_UN); _f4.close()
            expect("拿不到鎖時逾時報錯", _timed_out, True)
        finally:
            GPU_LOCK_PATH = _orig_lock

    print()
    if fails:
        print("SELFTEST FAIL (%d): %s" % (len(fails), fails))
        return 1
    print("SELFTEST PASS -- one front door, four wrappers behind it")
    return 0


# --------------------------------------------------------------------------- bench
# 統一量測入口：所有報告數字必須基於 prod-new（或顯式 profile）＋自己的 env 增量。
# base gate：arm env 撞到 base 已有鍵但未用 `!` 宣告 override → fail-closed 拒跑。
# 產物契約：每臂 json 注入 base_check + cell + box_gate（thermal/swap 讀數與冷卻結果）+ 量測紀律字段。
# prod-new 的三個 swap 支柱（P0/P1/P2＋B_SCHEME）未武裝且未用 `!` 宣告 → 拒跑（見 _swap_arm_gate）。

_BASE_PROFILE_DEFAULT = "prod-new"

# 實驗開關白名單（測試卡 §4 臂差異開關 + 支柱開關）：這些鍵允許覆蓋 base（不需 `!`，自動記錄）。
# 其餘 base 鍵 = 環境/儀器/模型常數，鎖死——要改必須 `!` 顯式宣告。
_EXPERIMENT_KNOBS = {
    # ── dump 內實驗開關（撞 base 自動放行 + 記錄）──
    "CGC_EXPERT_CACHE_BYTES", "LLAMA_EXPERT_CACHE_ALLOW_NGL",
    "CGC_OA_ASYNC", "CGC_GATHER_SLAB_CAP", "CGC_PREFILL_STREAM",
    "CGC_SPAC", "CGC_SPAC_ALPHA", "CGC_MM_BITIDENT",
    # ── 不在 dump 的已知實驗開關（arm 設了 = 新增鍵，gate 不擋、產物 extra_env 記錄）──
    "CGC_EXPERT_SKIP_READRAW", "CGC_POOL_MADVISE", "CGC_SERVER_MTP",
    "CGC_SERVER_MTP_N_MAX", "CGC_SEG_BATCH", "CGC_B_SCHEME",
    "CGC_SLOT_TABLE_GPU", "CGC_SPAC_HOT", "CGC_SERVER_PREFIX_REUSE_CKPT",
    "CGC_SERVER_DENSE_IQ4X", "CGC_SERVER_OA_ASYNC", "CGC_SERVER_NO_SEQ_RM_PROBE",
    "CGC_DOWN_COMBINE", "CGC_FORCE_TEMP0", "CGC_HOOK_PROFILE",  # opt-in 儀器
}

# prod-new 的 swap 結構組合：這三個是「預設就該開」的支柱，不是實驗開關。
# 缺一不可的理由各自獨立（P0 砍匿名副本＝超訂根因、P1/P2 壓 churn、B_SCHEME 讓 miss 收斂），
# 而三者關掉之後量到的 decode 會被讀成「prod-new 的效能」——所以缺了要拒跑，不是只警告。
# 要跑關掉其中一個的 A/B，用 `!KEY=VAL` 顯式宣告（會進產物 base_check.overrides）。
_SWAP_ARM_KEYS = ("CGC_EXPERT_SKIP_READRAW", "CGC_POOL_MADVISE", "CGC_B_SCHEME")

# llama-bench 完整側默認（測試卡 §2.5 權威 CELL，可覆寫但記錄在產物；reps=3 中位數口徑）。
# fixed_fill_seed=1 是 docs/DECODE_STEADY_BASELINE_2026-09-19.md:45 的結論（σ 3.46→1.09）：
# 沒有它，每個 rep 換一條隨機填充流，量到的是 cold/steady 的混合（run3 = 4.13/11.85/11.67），
# 而那個混合會直接被讀成「噪音」。它進產物的 cell 區塊，不是隱含 default。
# 2026-10-03 operator 裁定：prod-new + harness bench **完全貼齊 llama-bench 出廠形狀**。
# 出廠值就在原始碼裡（`llama-bench.cpp:392-439` 的 `cmd_params_defaults`）：
#   -p 512 -n 128 -b 2048 -ub 512 -d 0 -r 5、warmup 開。
# 這一組**同時是測試卡 §2.5 的預設 cell**——兩邊不一致時 `cell_contract.check_cell` 會
# fail-closed 拒跑，所以改這裡就必須改那裡（反之亦然）。
# 刻意不貼的兩項：KV 維持 prod-new 的 q8_0、池子／ngl 不動——那是記憶體側的專案硬約束，
# 不是 llama-bench 的形狀維度（operator 2026-10-03 選的是「形狀貼齊」）。
_BENCH_DEFAULTS = dict(prompt=512, gen=128, depths="0", reps=5,
                       warm_skip=0, ctx_size=0, batch=2048, ubatch=512,
                       fixed_fill_seed=1, prompt_file="", warmup=True)


def _parse_arm(spec: str) -> tuple[str, dict, set]:
    """'prod-new:!A=1;B=2' -> ('prod-new', {'A':'1','B':'2'}, {'A'})"""
    if ":" in spec:
        profile, envs = spec.split(":", 1)
    else:
        profile, envs = spec, ""
    env, overrides = {}, set()
    for piece in envs.split(";") if envs else []:
        piece = piece.strip()
        if not piece:
            continue
        declared = piece.startswith("!")
        if declared:
            piece = piece[1:]
        if "=" not in piece:
            raise SystemExit(f"bad arm env piece: {piece!r} (want KEY=VAL or !KEY=VAL)")
        k, v = piece.split("=", 1)
        env[k.strip()] = v.strip()
        if declared:
            overrides.add(k.strip())
    return profile, env, overrides


def _swap_arm_gate(profile: str, arm_env: dict, overrides: set) -> tuple[bool, list[dict]]:
    """prod-new 的三個 swap 支柱必須武裝；未宣吿就缺 → FAIL（fail-closed）。

    `arm_env` 是 matrix.resolve() 出來的**完全解析後**環境（不是我們寫下的清單）：
    run_server.sh 只傳非 0 值，所以「鍵不在」跟「=0」是同一件事，兩者都算沒開。
    `!K=0` 這種顯式關閉是合法做法（A/B 需要它）——它在 base gate 已被記錄，這裡不重複審。
    """
    if profile != _BASE_PROFILE_DEFAULT:
        return True, []
    missing = []
    for k in _SWAP_ARM_KEYS:
        v = arm_env.get(k)
        armed = v not in (None, "", "0")
        if not armed:
            missing.append({"key": k, "value": v,
                            "declared": k in overrides})
    undeclared = [m for m in missing if not m["declared"]]
    return (len(undeclared) == 0), missing


def _base_gate(profile: str, arm: dict, overrides: set) -> tuple[bool, list[str], list[str]]:
    """base vs 已解析的 arm；未宣告的 base 鍵變更 → FAIL（fail-closed）。"""
    matrix = _load("matrix", "llama_bench_matrix.py")
    base = matrix.resolve(profile, {})
    diffs, ovr = [], []
    # 雙向比對：arm 改 base 鍵值、或設 0 讓 base 鍵從 dump 消失（= 值變更），都要歸因
    for k in sorted(set(base["env"]) | set(arm["env"])):
        bv, av = base["env"].get(k), arm["env"].get(k)
        if bv is None:          # arm 新增鍵（base 沒有）
            continue
        if av == bv:
            continue
        if k in overrides:
            ovr.append(f"{k}: {bv!r} -> {av!r} (declared override)")
        elif k in _EXPERIMENT_KNOBS:
            ovr.append(f"{k}: {bv!r} -> {av!r} (experiment knob)")
        else:
            diffs.append(f"{k}: base={bv!r} arm={av!r} (NOT declared, use !)")
    return (len(diffs) == 0), diffs, ovr


# --------------------------------------------------------------------------- charter gate
# 跑前「立項」閘。契約 §4 的目標/判準要求「跑之前定好」，舊流程只在事後登記表檢查，agent
# 可以不聲明現狀/目標/驗收就開跑——甚至直跑冷 server 的第一請求、把冷啟動數字當穩態 decode
# 引用（2026-09-27 的 13.69：decode 僅 50 token、池全冷）。這個閘在 launch 前強制四件事：
# 現狀有出處、目標可判定、預期有依據、驗收有判據。缺欄或出處不存在 → fail-closed 拒跑。
_CHARTER_TOP_FIELDS = ("id", "owner", "question", "baseline", "hypothesis", "acceptance")
_CHARTER_VAGUE_PREFIX = ("測一下", "测一下", "測個", "测个", "看看", "試試", "试试", "跑一下")


def _load_charter(path: str) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise SystemExit(f"charter 不存在: {path}（相對於 repo root 解析）")
    try:
        import yaml
    except ImportError:
        raise SystemExit("需要 PyYAML 讀 charter（pip install pyyaml）")
    data = yaml.safe_load(p.read_text())
    if not isinstance(data, dict):
        raise SystemExit(f"charter 格式錯誤: {path}（頂層必須是 mapping）")
    data["_path"] = str(p)
    return data


def _charter_source_path(source, root):
    s = str(source).split("#", 1)[0].strip()
    p = Path(s)
    return p if p.is_absolute() else root / p


def _json_safe(o):
    """Charter 由 YAML 載入，`created: 2026-09-27` 會被解析成 datetime.date，json.dumps 無法
    序列化（2026-09-27 實測 TypeError: Object of type date is not JSON serializable）。遞迴把
    date/datetime 轉成 ISO 字串；其餘型別原樣返回。"""
    import datetime as _dt
    if isinstance(o, (_dt.datetime, _dt.date)):
        return o.isoformat()
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    return o


def _charter_gate(charter: dict, profiles: list[str], root=ROOT) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for f in _CHARTER_TOP_FIELDS:
        if not charter.get(f):
            errors.append(f"缺頂層欄位 {f}")

    q = " ".join(str(charter.get("question", "")).split())
    if q:
        if len(q) < 10:
            errors.append("question 太短（要一個可判定的問題，不是「測一下 X」）")
        if q.startswith(_CHARTER_VAGUE_PREFIX):
            errors.append("question 不可用「測一下／看看」開頭（必須可判定）")

    bl = charter.get("baseline")
    if not isinstance(bl, dict):
        errors.append("baseline 必須是 mapping（metric/cell/source）")
        bl = {}
    for f in ("metric", "cell", "source"):
        if not str(bl.get(f, "")).strip():
            errors.append(f"baseline.{f} 為空（現狀必須有數字、口徑、出處）")
    src = bl.get("source")
    srcp = _charter_source_path(src, root) if src else None
    if src and not srcp.exists():
        errors.append(f"baseline.source 不存在: {src}（基線數字必須可追溯）")

    hp = charter.get("hypothesis")
    if not isinstance(hp, dict):
        errors.append("hypothesis 必須是 mapping（mechanism/expected/basis）")
        hp = {}
    for f in ("mechanism", "expected", "basis"):
        if not str(hp.get(f, "")).strip():
            errors.append(f"hypothesis.{f} 為空（預期收益須有機制、區間、依據）")
    exp = str(hp.get("expected", ""))
    if exp and not any(c.isdigit() for c in exp):
        errors.append("hypothesis.expected 要給含數字的收益（優先給區間）")

    ac = charter.get("acceptance")
    if not isinstance(ac, dict):
        errors.append("acceptance 必須是 mapping（success/falsify/on_fail）")
        ac = {}
    for f in ("success", "falsify", "on_fail"):
        if not str(ac.get(f, "")).strip():
            errors.append(f"acceptance.{f} 為空（跑前要定成功／否證／失敗處置）")

    # 臂一律基於 prod-new；非 prod-new 須在 charter 補 non_prod_reason
    non_prod = [p for p in profiles if p != _BASE_PROFILE_DEFAULT]
    if non_prod and not str(charter.get("non_prod_reason", "")).strip():
        errors.append(f"非 prod-new 臂 {non_prod} 需在 charter 補 non_prod_reason")

    # 宣稱「穩態」的 baseline，json 產物必須能證明走了熱口徑；全為 False ⇒ 冷口徑不可當穩態。
    cell = str(bl.get("cell", ""))
    claims_steady = ("穩" in cell or "暖" in cell or "steady" in cell.lower()
                     or "warm" in cell.lower())
    if claims_steady and srcp is not None and srcp.suffix == ".json" and srcp.exists():
        try:
            arr = json.loads(srcp.read_text())
            rows = arr if isinstance(arr, list) else [arr]

            def _ws_flag(r):
                # 顯式檢查，不可用 `or`：warm_skip_applied=False 是 falsy，會被 or 跳過讀成 None。
                if not isinstance(r, dict):
                    return None
                if "warm_skip_applied" in r:
                    return r.get("warm_skip_applied")
                c = r.get("cell")
                return c.get("warm_skip_applied") if isinstance(c, dict) else None

            flags = [_ws_flag(r) for r in rows]
            if flags and all(f is False for f in flags):
                errors.append("baseline 宣稱穩態，但產物 warm_skip_applied 全為 False"
                              "（冷口徑不能當穩態 decode）")
        except Exception:
            pass

    return (len(errors) == 0), errors


def _sys_snapshot() -> dict:
    """測試前/後各採一次：thermal + swap 水位 + pageins/pageouts + memory_pressure + iostat。"""
    snap = {}
    tp = _load("tp_snap", "thermal_pressure.py")
    mp = _load("mpp_snap", "memory_pressure.py")
    try:
        t = tp.stamp()
        # `stamp()` returns {level, label, t} and has never had an `lv` key, so this field read
        # back as None in EVERY harness artifact until 2026-09-28: the label survived and the
        # number did not (measured live: {'label': 'NOMINAL', 'lv': None}). Worse, None is also
        # `thermal_pressure`'s UNREADABLE value, so the defect looked like a missing instrument
        # rather than a wrong key. Same shape as the `--fixed-fill-seed` wiring bug this file
        # already documents: a knob/field going missing is not a typo, it is a missing consumer.
        snap["thermal"] = {"label": t.get("label"), "lv": t.get("level"), "t": t.get("t")}
    except Exception as e:
        snap["thermal"] = {"error": str(e)}
    try:
        snap["swap_used_mb"] = mp.swap_used_mb()
    except Exception:
        snap["swap_used_mb"] = None
    try:  # vm_stat 累計 pageins/pageouts（欄位大小寫不敏感）
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10)
        pi = po = None
        for line in out.stdout.splitlines():
            low = line.lower()
            if low.startswith("pageins:"):
                pi = int(line.split()[1].rstrip("."))
            elif low.startswith("pageouts:"):
                po = int(line.split()[1].rstrip("."))
        snap["pageins"], snap["pageouts"] = pi, po
    except Exception:
        pass
    try:  # memory_pressure 狀態（System-wide memory free percentage 行）
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=10)
        lines = (out.stdout or out.stderr or "").splitlines()
        snap["memory_pressure"] = next((l.strip() for l in lines if "percentage" in l.lower()), "")
    except Exception:
        snap["memory_pressure"] = ""
    try:  # iostat 磁盤活動（最後一行 = 一次樣本）
        out = subprocess.run(["iostat", "-d", "1", "1"], capture_output=True, text=True, timeout=10)
        lines = [l.strip() for l in out.stdout.splitlines() if l.strip()]
        snap["iostat"] = lines[-1].split() if lines else []
    except Exception:
        snap["iostat"] = []
    return snap


def _bench_cmd(args, specs) -> list[str]:
    """llama-bench command for this cell. Extracted so the selftest can prove a knob reached it.

    This exists because the way a flag goes missing is not a typo: `harness bench` simply did
    not have `--fixed-fill-seed`, so every run through the "unified" entry point measured the
    mixed cold/steady regime while the recipe was fine. A builder that a test can inspect is
    the cheapest fix for that class.
    """
    cmd = [PY, str(HERE / "llama_bench_matrix.py"),
           "--arms", ",".join(specs),
           "--prompt", str(args.prompt), "--gen", str(args.gen),
           "--depths", args.depths, "--reps", str(args.reps),
           "--ctx-size", str(args.ctx_size), "--warm-skip", str(args.warm_skip),
           "--fixed-fill-seed", str(args.fixed_fill_seed),
           "--workdir", str(args.workdir), "--json", str(args.json_path)]
    if args.prompt_file:
        cmd += ["--prompt-file", args.prompt_file]
    if args.batch:
        cmd += ["--batch", str(args.batch)]
    # 與 `--batch` 成對送出（見 argparse 處的 2026-10-04 註解）：缺它 ⇒ matrix 的舊語意會把
    # ubatch 設成 batch，而新預設格是 b2048/ub512 ⇒ 契約拒跑。
    if getattr(args, "ubatch", None):
        cmd += ["--ubatch", str(args.ubatch)]
    # Named cell from the test card §2.5. Omitted = the default (prod-new) cell, so no existing
    # caller changes. Threaded here rather than inline so the selftest can prove it reached the
    # matrix -- the `--fixed-fill-seed` lesson is that a knob going missing is a wiring bug, not
    # a typo, and this builder is the one place a test can see it.
    if getattr(args, "cell", None):
        cmd += ["--cell", str(args.cell)]    # 【CGC 2026-09-29】需明文要求的探針（第五道閘門）：跑前擋前提、跑後擋拒寫。它以
    # `getattr` 取（不在），否則 selftest 的假 Namespace 一呼叫就會 AttributeError。
    if getattr(args, "require_instrument", ""):
        cmd += ["--require-instrument", str(args.require_instrument)]
    # 【CGC 2026-09-29】驅動側的剎車與看門狗（見 llama_bench_matrix `_stream_child`）。只在宣告時
    # 出現 ⇒ 既有 cell 的命令列一個字都不變。它屬於「跑不完那一類診斷」：
    # 子行程 stderr 逐行落地（被殺也留得下停滯點）＋ 逾時只殺自己的 pgid（驅動活下來寫產物）。
    if getattr(args, "arm_timeout", 0):
        cmd += ["--arm-timeout", str(args.arm_timeout)]
    if getattr(args, "stall_watch", 0):
        cmd += ["--stall-watch", str(args.stall_watch)]
    if getattr(args, "stall_kill", False):
        cmd += ["--stall-kill"]
    if getattr(args, "stall_progress_re", ""):
        cmd += ["--stall-progress-re", str(args.stall_progress_re)]
    if getattr(args, "live_echo", ""):
        cmd += ["--live-echo", str(args.live_echo)]
    if not args.warmup:
        cmd += ["--no-warmup"]
    if args.spec_type:
        cmd += ["--spec-type", args.spec_type]
        if args.spec_draft_n_max is not None:
            cmd += ["--spec-draft-n-max", str(args.spec_draft_n_max)]
    return cmd


def _instrument_precheck(ibind, resolve_fn, specs, require) -> tuple:
    """(ok, fails, warnings) —— 第五道閘門的**跑前那一半**：武裝了量具卻沒開它的前提。

    這一族在「不跑」的情況下就已經矛盾（引擎自己會印 `CGC-MISS-MASK: ... without ...`
    那一行），所以它应该在花 GPU 時間之前就拒掉 —— 而那種「空白場次」事後只看得到 0，
    看不到原因。`resolve_fn(spec) -> arm` 抽出來是為了讓 selftest 不用真的跑 `run_server.sh`。
    """
    fails, warnings = [], []
    for spec in specs:
        arm = resolve_fn(spec)
        ok, reasons, warns = ibind.static_precheck((arm or {}).get("env", {}) or {}, require=require)
        warnings += ["%s：%s" % (spec, w) for w in warns]
        if not ok:
            fails.append((spec, reasons))
    return (not fails), fails, warnings


_INSTRUMENT_RULE = ("每一臂的 instrument verdict 必須是 BOUND 或 N/A；"
                    "UNBOUND／缺欄位 ⇒ 拒寫進決策面")

# 【CGC 2026-10-06】第六道閘門：spec 臂的 draft 有沒有死。
# 判準資料在產物裡（matrix 的 `spec_draft`，逐臂讀自己那份 stderr，所以歸屬不會錯）。
_SPEC_DRAFT_RULE = ("spec 臂的每一輪 rep 的 CGC-BENCH-ACCEPT `drafted` 必須 > 0；"
                    "任一轮 drafted==0（draft 死亡）或 spec 開了卻沒有 ACCEPT 行 "
                    "⇒ 拒收（無豁免）")


_SPEC_DRAFT_MOD = None


def _spec_draft_dead(arm) -> bool:
    """[CGC 2026-10-06] 判準的唯一出處在產物端（`llama_bench_matrix.spec_draft_dead`）。

    這裡刻意**不做**本地 fallback 複本：三個消費端各留一份就是先前不一致的來源。
    載入失敗就讓它炸（可見），而不是靜默換一套判準。
    """
    global _SPEC_DRAFT_MOD
    if _SPEC_DRAFT_MOD is None:
        _SPEC_DRAFT_MOD = _load("llama_bench_matrix", "llama_bench_matrix.py")
    return _SPEC_DRAFT_MOD.spec_draft_dead(arm)


def _spec_draft_gate(data) -> dict:
    """第六道閘門：spec 臂的 **draft 有沒有死**（逐 rep `drafted=0`）。

    WHY（2026-10-06 `p_chain`／`p_chain2`）：`CGC_SPEC_TREE=0` 的純鏈臂第 1–4 個 rep 全是
    `drafted=0 rounds=128 emit_per_round=1.0000` —— draft 一顆都沒吐，那幾輪量的是**純解碼**，
    而它的 t/s（9.47／9.50）與健康的 tg 讀數長得一模一樣，差一點被當成「MTP-on 純鏈」引用。

    `drafted=0` 不是雜訊而是**簽名**：那一輪的 emit ≡ 1.0（= 純解碼）、rounds ≡ n_gen、
    draft_ratio ≡ 0。判定放在跑後、寫進產物（`spec_draft_gate`），並**拒收**（rc=6）。
    與成對 log 閘同一類：**沒有豁免** —— 要嘛修掉 draft 為什麼死，要嘛這一場不算數
    （它沒有量到你以為的那件事）。舊產物（沒有 `spec_draft` 欄）不回溯，只記錄。
    """
    arms, bad, checked = [], [], 0
    for a in data or []:
        tag = (a or {}).get("tag")
        d = (a or {}).get("spec_draft")
        if not isinstance(d, dict) or not d:
            arms.append({"tag": tag, "ok": True, "verdict": "N/A", "dead": [], "n_reps": 0,
                         "why": "產物沒有 spec_draft 欄（閘門存在之前的產物，或這一臂不經 spec 路徑）"
                                "⇒ 不回溯"})
            continue
        state = str(d.get("state") or "NO_STATE")
        ok = not _spec_draft_dead(a)          # 判準共用（fail-closed），見 _spec_draft_dead
        checked += 1
        arms.append({"tag": tag, "ok": ok, "verdict": state, "why": d.get("why") or "",
                     "dead": d.get("dead") or [], "n_reps": int(d.get("n_reps") or 0)})
        if not ok:
            bad.append(tag)
    return {"ok": not bad, "arms": arms, "bad": bad, "rule": _SPEC_DRAFT_RULE, "checked": checked}


def _instrument_gate(ibind, data, waive_reason=None) -> dict:
    """第五道閘門的**跑後那一半**：這一輪的量具到底有沒有動（決定這份產物能不能進決策面）。

    只有 BOUND（動過）與 N/A（沒武裝，所以沒有承諾）放行；UNBOUND（武裝了卻沒動）與
    缺欄（拿不出證明）都不放。回傳值直接寫進產物的 `instrument_gate`，不另外靠人記得。
    """
    ok, details = ibind.runs_unbound(data)
    return {"ok": ok, "waived": waive_reason, "arms": details, "legacy": False,
            "rule": _INSTRUMENT_RULE, "checked": len(details)}


def _instrument_gate_verify(ibind, result, waive_reason=None) -> dict:
    """第五道閘門在 `harness verify` 這扇門上的判定。

    `verify` 的每一輪是**另一次** `llama_bench_matrix` 執行，產物原封不動嵌在
    `result.json` 的 `arms[*].{clean,instrumented}.matrix` 裡 ⇒ 判準不必重寫，只要把那些臂攤平。
    這一扇門是必要的：`arm_two_pass` 直接把 matrix 叫起來，不經過 `cmd_bench` 的閘門，
    所以「產線硬條件」若只做在 bench 上，verify 這條路就是空的（2026-09-29 補）。

    **政策不回溯**：一份在閘門存在之前跑的 `result.json`（matrix 產物完全沒有 `instrument`
    欄位）標成 `legacy` 而不拒 —— 否則一次政策改動會靜默改寫整份舊報告，那正是 §7 拒絕做的事。
    """
    embedded, total = [], 0
    for a in (result or {}).get("arms") or []:
        for which in ("clean", "instrumented"):
            for arm in ((a.get(which) or {}).get("matrix") or []):
                total += 1
                if isinstance(arm, dict) and "instrument" in arm:
                    embedded.append(arm)
    if total and not embedded:
        return {"ok": True, "legacy": True, "waived": waive_reason, "arms": [], "checked": 0,
                "why": "這份 result.json 是閘門存在之前跑的（matrix 產物沒有 instrument 欄位）⇒ 不回溯",
                "rule": _INSTRUMENT_RULE}
    ok, details = ibind.runs_unbound(embedded)
    return {"ok": ok, "legacy": False, "waived": waive_reason, "arms": details,
            "checked": len(details), "rule": _INSTRUMENT_RULE}


def _inside_root(p) -> bool:
    """這個路徑在 repo 裡嗎？── 只有 repo 內的產物才可以回寫決策面。

    相對路徑算在 ROOT 底下（呼叫端與 selftest 從不同 cwd 跑時，答案不該不同）。
    """
    s = str(p or "")
    ap = os.path.abspath(s) if os.path.isabs(s) else os.path.join(ROOT, s)
    try:
        Path(ap).relative_to(Path(ROOT).resolve())
        return True
    except Exception:  # noqa: BLE001  不在 repo 內、或路徑不可解析
        return False


def _pair_gate_verify(ibind, result) -> dict:
    """`verify` 的成對判定：每一輪的 `matrix.json` 旁邊必須有它的 stderr log。

    `bench` 那扇門檢查的是 `--json` 的 sibling；這裡的產物是 `arm_two_pass` 寫的 `matrix.json`，
    每一輪（clean／instrumented）一份，所以逐輪檢查。缺 ⇒ 那一輪不可引用（無豁免）。
    """
    passes = []
    for a in (result or {}).get("arms") or []:
        for which in ("clean", "instrumented"):
            p = a.get(which) or {}
            mj = p.get("matrix_json")
            if not mj:
                continue
            tag = "%s/%s" % (a.get("arm"), which)
            if not os.path.exists(mj):
                passes.append({"pass": tag, "paired": False, "canonical": None,
                               "why": "找不到這一輪的產物 %s" % mj})
                continue
            st = ibind.pair_status(mj)
            passes.append({"pass": tag, "paired": st["paired"],
                           "canonical": st["canonical"], "why": st["why"]})
    return {"paired": all(p["paired"] for p in passes) if passes else True,
            "passes": passes, "checked": len(passes)}


def _cell(args) -> dict:
    """The cell, as a dict -- a number is only comparable to another number from the same cell."""
    return {"prompt": args.prompt, "gen": args.gen, "depths": args.depths, "reps": args.reps,
            "warm_skip": args.warm_skip, "ctx_size": args.ctx_size,
            "fixed_fill_seed": args.fixed_fill_seed, "prompt_file": args.prompt_file,
            "warmup": args.warmup, "spec_type": args.spec_type,
            "spec_draft_n_max": args.spec_draft_n_max,
            # Which §2.5 cell this artifact belongs to. Recorded because two cells' numbers are
            # not comparable to each other, and the record has to say which one it holds.
            "named_cell": getattr(args, "cell", None) or "(default)"}


def _swap_line(memory: dict) -> str:
    """The arm's swap numbers, read from where the matrix actually puts them.

    The keys are `memory.launch.swap_used_mb` / `memory.end.swap_used_mb` / `memory.worst.max_swap_mb`.
    Until 2026-09-26 this reader asked for `launch_swap`/`end_swap`/`worst_swap`, got None three
    times, and printed `growth=+0` on all 14 arms -- a judgement-shaped number ("growth=+0 ⇒ no
    paging problem") for a field it never read, while `attribution.swap_growth_mb` in the same arm
    said 2137.75 MiB. Absence now prints as itself, and names the field that does have the value.
    """
    launch = (memory.get("launch") or {}).get("swap_used_mb")
    end = (memory.get("end") or {}).get("swap_used_mb")
    worst = (memory.get("worst") or {}).get("max_swap_mb")
    growth = (end - launch) if (launch is not None and end is not None) else None
    g = "n/a" if growth is None else f"{growth:+.0f} MiB"
    tail = "" if growth is not None else "  (unpopulated — use attribution.swap_growth_mb)"
    return f"  swap: launch={launch} end={end} worst={worst} growth={g}{tail}"


def _swap_stock_line(advice: dict) -> str:
    """The stock is RENDERED as a label. The old form read `[swap] verdict=advise ... > 2048 MiB`,
    which is the same shape as the 14-arm `growth=+0`: a stock number wearing a verdict's clothes.
    The deciding gate for this launch is the compressor line printed right after it.
    """
    return f"[swap-stock] 標籤（不是起跑閘）：{advice['verdict']} {advice['why']}"


def _box_gate(args) -> dict:
    """Launch 前的盒況閘：thermal 冷卻 → **壓縮機安靜度**（決定者）→ swap 存量（標籤）。

    2026-09-26 的替換：swap 存量分不出「8 GB 陳年 swap + 壓縮機安靜」（實測 0.00 MiB/s，而舊
    閘會拒跑）與「小 swap + 壓縮機忙」（同一臂 245–540 MiB/s），而後者才是喫 t/s 的那個。
    存量的處置仍是**提醒**（那上面的頁不是我們的，P0/P1/P2 管不到別人的匿名頁），但它的措辭
    必須是標籤、不能長得像 verdict —— 「讀起來像判決」正是這一輪在修的儀器缺陷。
    """
    tp = _load("tp_gate", "thermal_pressure.py")
    mp = _load("mp_gate", "memory_pressure.py")
    gate = {"cool_max_s": args.cool_max_s, "thermal_gate": not args.no_thermal_gate}
    if args.no_thermal_gate:
        gate["thermal"] = {"skipped": True}
        print("[thermal] gate 關閉（--no-thermal-gate）-- 本輪的 thermal 狀態不會擋住啟動")
    else:
        gate["thermal"] = tp.wait_nominal(timeout_s=args.cool_max_s)
    advice = mp.swap_advice()
    gate["swap_stock_label"] = advice
    print(_swap_stock_line(advice))
    for row in advice["top"]:
        print(f"       {row['rss_mb']:8.1f} MiB  {row['proc']}")
    try:
        cpt = _load("cp_gate", "compressor_pressure.py")
        cq_ok, cq_why = cpt.require(where="harness bench")
    except Exception as e:  # noqa: BLE001  fail-closed: 讀不到不是通關
        cq_ok, cq_why = False, f"unknown: compressor probe unavailable: {e}"
    gate["compressor"] = {"quiet": cq_ok, "reason": cq_why}
    print(f"[compressor] {cq_why}   <- 起跑閘")
    if not cq_ok:
        override = os.environ.get("CGC_WINDOW_OVERRIDE") == "1"
        gate["refused"] = not override
        print("[compressor] BUSY：這輪會被歸因給壓縮機流量"
              + ("（已 override，狀態照記進產物）" if override else
                 " —— 拒跑（CGC_WINDOW_OVERRIDE=1 可改為照跑但記錄）"))
    return gate


def cmd_bench(args) -> int:
    matrix = _load("matrix", "llama_bench_matrix.py")
    specs = list(args.arm)

    # 跑前立項閘：速度實驗必須帶一張 charter（現狀/目標/假設/驗收）；--charter none 可豁免但標記。
    profiles = [_parse_arm(s)[0] for s in specs]
    charter = None
    cv = getattr(args, "charter", None)
    if cv is None:
        print("!! 缺 --charter：速度實驗跑前必須立項（現狀/目標/驗收）。")
        print("   用 scripts/check/charters/_TEMPLATE.yaml 建一張；純分析/急件可 --charter none。")
        return 2
    if cv == "waive":
        reason = getattr(args, "waive_reason", None)
        if not reason:
            print("!! --charter waive 必須帶 --waive-reason \"為何急件不立項\"；否則正常立項。")
            return 2
        charter = {"waived": True, "reason": reason, "when": time.strftime("%F %T")}
        print(f"[charter] 急件豁免（waive）：{reason} ——已記進產物")
    elif cv == "none":
        print("!! --charter none 後門已關閉：要嘛給 charter 路徑立項，要嘛 --charter waive "
              "--waive-reason \"...\"。")
        return 2
    else:
        charter = _load_charter(cv)
        cok, cerr = _charter_gate(charter, profiles)
        if not cok:
            print("!! charter gate FAIL — 拒跑：")
            for e in cerr:
                print(f"    {e}")
            return 2
        print(f"[charter] 立項通過：{charter.get('id')} — owner={charter.get('owner')}")

    # 第五道閘門的跑前那一半：量具武裝了但前提沒開 ⇒ 拒跑（比拒寫便宜：不花 GPU 時間）。
    ibind = _load("ibind", "instrument_binding.py")
    require_probes = [x.strip() for x in str(getattr(args, "require_instrument", "") or "").split(",")
                      if x.strip()]
    ip_ok, ip_fails, ip_warns = _instrument_precheck(
        ibind, lambda s: matrix.resolve(*_parse_arm(s)[:2]), specs, require_probes)
    for w in ip_warns:
        print(f"[instrument] {w}")
    if not ip_ok:
        print("!! 量具前提閘 FAIL — 拒跑（這是不花 GPU 時間就能判的矛盾）：")
        for spec, reasons in ip_fails:
            for r in reasons:
                print(f"    {spec}：{r}")
        print("   理由：武裝了而沒開前提的量具，它的計數器一定是 0，而那種「空白場次」事後只看得到 0。")
        print("   要跑就先把前提開上（例：CGC_MISS_MASK=1），或不要把那個量具的開關打開。")
        return 2

    ok_all = True
    gate_report = []
    for spec in specs:
        profile, env, overrides = _parse_arm(spec)
        arm = matrix.resolve(profile, env)
        ok, diffs, ovr = _base_gate(profile, arm, overrides)
        armed, missing = _swap_arm_gate(profile, arm.get("env", {}), overrides)
        report = {"arm": spec, "profile": profile, "overrides": ovr,
                  "diffs": diffs, "pass": ok and armed,
                  "swap_arms": {k: arm.get("env", {}).get(k) for k in _SWAP_ARM_KEYS}}
        gate_report.append(report)
        if not ok:
            ok_all = False
            print(f"!! base gate FAIL for {spec}:")
            for d in diffs:
                print(f"    {d}")
        if not armed:
            ok_all = False
            undeclared = [m for m in missing if not m["declared"]]
            print(f"!! swap-arm gate FAIL for {spec}: "
                  f"{', '.join(str(m['key']) for m in undeclared)} 沒開")
            print("   prod-new 的 decode 是在這三個開著的前提下量的（P0 砍匿名副本、"
                  "P1/P2 壓 churn、B_SCHEME 讓 miss 收斂）；關著跑出來的數字会被讀成 "
                  "prod-new 的效能。要跑關掉的對照臂："
                  + " ".join(f"--arm '{profile}:!{m['key']}=0,...'" for m in undeclared))
    if not ok_all:
        print("gate 未過 — 拒跑。用 !KEY=VAL 顯式宣告覆蓋（A/B 對照臂正當），或補上缺的開關。")
        return 2

    # launch 前的盒況閘（thermal 冷卻 → 壓縮機閘 → swap 標籤），產物會帶上它自己的讀數
    box_gate = _box_gate(args)
    if box_gate.get("refused"):
        print("拒跑：壓縮機忙（流量問題，不是 swap 存量問題）。見 [compressor]。")
        return 2

    # 測試前系統快照（thermal / swap / pageins / memory_pressure / iostat）
    t_bench0 = time.time()
    before = _sys_snapshot()
    print(f"系統快照 [before] thermal={before.get('thermal')} swap={before.get('swap_used_mb'):.0f} MiB "
          f"pageins={before.get('pageins')} pageouts={before.get('pageouts')} "
          f"pressure={before.get('memory_pressure')} iostat={before.get('iostat')}", flush=True)

    # delegate 給 llama_bench_matrix（量測邏輯不複製；matrix 不自動建 workdir）
    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    cmd = _bench_cmd(args, specs)
    print("$ " + " ".join(cmd), flush=True)
    # 實際 GPU 運行全程持內核原子鎖：第二個 harness 在這段結束前阻塞、無法並發
    try:
        with gpu_window_lock(held_by="bench " + " ".join(specs), timeout_s=args.window_timeout_s):
            # matrix 的 internal 守衛認這個變數：harness 是唯一正門、內部調用自帶、繞過守衛。
            rc = subprocess.call(cmd, cwd=str(ROOT), env={**os.environ, "CGC_INTERNAL_CALL": "1"})
    except TimeoutError as e:
        print(f"拒跑：{e}")
        return 3
    if rc != 0:
        return rc

    # 測試後系統快照 + 速率（wall 用牆鐘，thermal t 是字串不能用來減）
    after = _sys_snapshot()
    wall = time.time() - t_bench0
    rate = {}
    if before.get("pageins") is not None and after.get("pageins") is not None:
        rate["pageins_per_s"] = (after["pageins"] - before["pageins"]) / wall
        rate["pageouts_per_s"] = ((after.get("pageouts") or 0) - (before.get("pageouts") or 0)) / wall
    print(f"系統快照 [after]  thermal={after.get('thermal')} swap={after.get('swap_used_mb'):.0f} MiB "
          f"pageins={after.get('pageins')} pageouts={after.get('pageouts')} "
          f"pressure={after.get('memory_pressure')} iostat={after.get('iostat')}", flush=True)
    if rate:
        print(f"page 速率: in={rate['pageins_per_s']:.1f}/s out={rate['pageouts_per_s']:.1f}/s "
              f"(wall={wall:.0f}s)", flush=True)

    # 產物注入 base_check + cell + box_gate + sys_before/after（每個 arm 一臂）
    data = json.loads(Path(args.json_path).read_text())
    for arm, report in zip(data, gate_report):
        arm["base_check"] = {"profile": report["profile"], "pass": report["pass"],
                             "overrides": report["overrides"], "diffs": report["diffs"],
                             "swap_arms": report["swap_arms"]}
        arm["cell"], arm["box_gate"] = _cell(args), box_gate
        arm["charter"] = _json_safe(charter)
        arm["sys_before"], arm["sys_after"] = before, after
        if rate:
            arm["sys_rate"] = rate

    # 【CGC 2026-09-29】第五道閘門的跑後那一半：量具沒綁上 ⇒ **拒寫進決策面**。
    # 「記錄 + 事後檢查」已經在樹上（`instrument_gate` 欄位、`mindmap_void_check`、`claim_instrument_check`），
    # 但它們都是事後的；這裡是把它變成硬條件的第一道：跑得再漂亮，量具沒動就不會進 mindmap。
    # 注意「拒寫」≠「抹掉證據」：產物照寫本地檔（含最後判定），只是不覆寫決策面。
    waive_ib = getattr(args, "waive_instrument", None) \
        or os.environ.get("CGC_IGNORE_INSTRUMENT_GATE") or None
    # 【CGC 2026-09-29】產物 ↔ stderr log 成對（寫入端已落地；這裡是跑後自檢）：
    # 缺 log ⇒ 這一場的量具根本無法查證 ⇒ 不可引用。這一條**沒有豁免**：補救的方法是把 log
    # 寫對（或 `instrument_binding --pair-backfill` 把真的那一份登記回來），不是宣告
    # 「我知道它不可驗但還是要算」——量具閘可以豁免，因為那是「我知道這輪儀器沒開」；
    # 這條不行，因為沒有 log 時你不知道自己錯過了什麼。
    pair_gate = ibind.pair_status(args.json_path)
    ib_gate = _instrument_gate(ibind, data, waive_reason=waive_ib)
    # 【CGC 2026-10-06】第六道閘門：spec 臂的 draft 死沒死（判定來自產物裡的 `spec_draft`）。
    spec_draft_gate = _spec_draft_gate(data)
    for arm in data:
        arm["instrument_gate"] = ib_gate
        arm["pair_gate"] = pair_gate
        arm["spec_draft_gate"] = spec_draft_gate
    Path(args.json_path).write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"\n產物已寫入 {args.json_path}（每臂含 base_check + sys_before/after + 系統指標）\n", flush=True)
    for d in ib_gate["arms"]:
        mark = "✅" if d["ok"] else "⛔"
        print(f"  {mark} 量具 {d['verdict']:10s} {d['tag']}  {d['why']}", flush=True)
    print(f"  {'✅' if pair_gate['paired'] else '⛔'} 成對 log  {pair_gate['why']}", flush=True)
    for d in spec_draft_gate["arms"]:
        print(f"  {'✅' if d['ok'] else '⛔'} spec draft {d['verdict']:10s} {d['tag']}  {d['why']}",
              flush=True)
    _ib_bad = (not ib_gate["ok"]) and not waive_ib
    if _ib_bad:
        print("!! 量具閘 FAIL — **拒寫**：這份產物不會進 mindmap（決策面）。", flush=True)
        print("   理由：旗標設了但量具沒動 ⇒ 那些計數器與由此而來的結論不可引用；"
              "這一輪的 t/s 也不是你以為的那件事。", flush=True)
        print(f"   產物仍在 {args.json_path}（證據不抹掉，只是不上成績面）。", flush=True)
        print("   要明知不可引用還是寫入：--waive-instrument \"理由\"（理由會進產物）。", flush=True)
    if not ib_gate["ok"] and waive_ib:
        print(f"!! 量具閘 FAIL，但已豁免（--waive-instrument）：{waive_ib}", flush=True)
    if not pair_gate["paired"]:
        print("!! 成對閘 FAIL — **拒收**：這一場的產物旁邊沒有 stderr log。", flush=True)
        print(f"   修法（不是豁免）：{pair_gate['why']}", flush=True)
        print("   理由：log 是唯一能證明「量具真的動了」的東西；沒有它，產物裡每個計數器都無法查證，"
              "而「沒有 log 所以驗不了」不該是一種狀態。", flush=True)
        print("   補救：確認 llama_bench_matrix 的成對 log 寫得進去，"
              "或用 scripts/check/instrument_binding.py --pair-backfill 把真的那一份登記回來。", flush=True)
        return 5
    if not spec_draft_gate["ok"]:
        print("!! spec 健康閘 FAIL — **拒收**：至少一臂有 rep 的 draft 死亡"
              "（drafted=0 ⇒ 那一輪量的是純解碼，不是 spec）。", flush=True)
        for d in spec_draft_gate["arms"]:
            if not d["ok"]:
                print(f"    ⛔ {d['tag']}（{d['verdict']}）：{d['why']}", flush=True)
        print("   理由：draft 死掉那一輪的 emit ≡ 1.0（就是純解碼），它的 t/s 與健康的 tg 讀數"
              "長得一樣，但量的不是你以為的那件事。", flush=True)
        print("   修法（不是豁免）：先修 draft 為什麼死（例：CGC_SPEC_TREE=0 的純鏈缺 draft-ctx "
              "mirror trim），再重跑；這一場的 t/s 不上決策面。", flush=True)
        return 6
    if _ib_bad:
        return 4

    # D 自動回寫：產物完整後直接 sync 到 mindmap（免手動 experiment_sync sync），best／目標進度即時更新。
    # 只在有真立項（非 waive）時做；回寫失敗不影響本次產物與成績。
    auto_cid = charter.get("id") if isinstance(charter, dict) and not charter.get("waived") else None
    # 【CGC 2026-09-29】只有**repo 內**的產物才回寫決策面。`--json /tmp/...` 是診斷／煙霧測試用的，
    # 它不該變成實驗節點上的一筆 run —— 我自己的 stub 測試（不跑 GPU、假產物）就因此把一筆
    # 假的 9.9 t/s 寫進了 `exp-singlesubmit-fillahead`，還把它複製到 `Backup/exp_runs/`。
    # 避免那一族不需要什麼智慧：產物不在 repo 內，就不回寫。
    if auto_cid and not _inside_root(args.json_path):
        print(f"[auto-sync] 略過：產物 {args.json_path} 不在 repo 內（診斷產物不回寫決策面）", flush=True)
        auto_cid = None
    if auto_cid:
        try:
            esync = _load("esync", "experiment_sync.py")
            esync.sync_artifact_file(args.json_path, cid_default=auto_cid, do_rebuild=True)
        except Exception as e:  # noqa: BLE001
            # 第三扇門（`experiment_sync.sync_gate`）擋下時帶著編號 —— 說出來比「回寫失敗」有用。
            _c = getattr(e, "code", None)
            _tag = f"（量具閘 rc={_c}）" if _c else ""
            print(f"[auto-sync] mindmap 回寫失敗{_tag}（不影響本次產物）：{e}", flush=True)

    # 量測紀律輸出：pp + tg + thermal + swap（取自產物欄位）
    for arm in data:
        print(f"=== {arm.get('tag')} (base_check: {'PASS' if arm['base_check']['pass'] else 'FAIL'}) ===")
        for r in arm["rows"]:
            kind = "pp" if r.get("n_prompt", 0) > 0 else "tg"
            ts = r.get("avg_ts")
            print(f"  {kind:2s}: {ts:.2f} t/s" if ts is not None else f"  {kind:2s}: (no avg_ts)")
        th = arm.get("thermal", {})
        print(f"  thermal: launch={th.get('launch')} worst={th.get('worst')}")
        print(_swap_line(arm.get("memory", {})))
        print(f"  attribution: {arm.get('attribution')}")
    # 0 = 正常；2 = 跑前閘拒跑；3 = 等不到 window；4 = 量具 UNBOUND；5 = 缺成對 log；
    # 6 = spec 健康閘 FAIL（draft 死亡／拿不出逐輪帳）——4/5/6 皆拒收（不上決策面）。
    return 0


def cmd_verify(args) -> int:
    """統一的「雙輪驗證」入口：每 arm 跑 clean + instrumented 兩輪、過生產級 gate、出 HTML。

    順序：arm_two_pass（雙輪 + gate + 全 log）→ 只要 result.json 在就 arm_report_html。
    arm_two_pass 回 2（gate 硬擋、有臂沒跑）仍生成 HTML（報告標 blocked），最終回非 0。
    """
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    result_json = run_dir / "result.json"
    report_html = run_dir / "report.html"

    cmd = [PY, str(HERE / "arm_two_pass.py")]
    for a in args.arm:
        cmd += ["--arm", a]
    cmd += ["--prompt", str(args.prompt), "--gen", str(args.gen),
            "--depths", str(args.depths), "--reps", str(args.reps),
            "--warm-skip", str(args.warm_skip), "--ctx-size", str(args.ctx_size),
            "--max-swap-mb", str(args.max_swap_mb), "--cool-s", str(args.cool_s),
            "--run-dir", str(run_dir), "--json", str(result_json)]
    if args.allow_dirty:
        cmd.append("--allow-dirty")

    print("「verify」雙輪驗證（每 arm：clean 無儀器 + instrumented 有儀器）", flush=True)
    # 雙輪 GPU 運行全程持原子鎖（arm_two_pass 內部跑兩輪 bench），避免並發
    try:
        with gpu_window_lock(held_by="verify " + " ".join(args.arm),
                             timeout_s=args.window_timeout_s):
            rc = subprocess.call(cmd, cwd=str(ROOT))
    except TimeoutError as e:
        print(f"拒跑：{e}")
        rc = 3

    # 【CGC 2026-09-29】第五道閘門在 verify 這扇門上：`arm_two_pass` 把每一輪直接交給 matrix，
    # 不經過 `cmd_bench` 的閘門（量具 UNBOUND 時那邊拒寫且 rc=4）。這裡不補，這條路就只剩「記錄」。
    waive_ib = getattr(args, "waive_instrument", None) \
        or os.environ.get("CGC_IGNORE_INSTRUMENT_GATE") or None
    ib_gate = None
    if result_json.exists():
        res = None
        try:
            res = json.loads(result_json.read_text())
        except Exception as e:  # noqa: BLE001
            print(f"[instrument] verify 產物讀不出來，量具閘這一輪無判定：{e}", flush=True)
        if isinstance(res, dict):
            ib_gate = _instrument_gate_verify(_load("ibind_v", "instrument_binding.py"), res,
                                              waive_reason=waive_ib)
            for d in ib_gate["arms"]:
                print(f"  {'✅' if d['ok'] else '⛔'} 量具 {d['verdict']:10s} {d['tag']}  {d['why']}",
                      flush=True)
            if ib_gate.get("legacy"):
                print(f"[instrument] {ib_gate['why']}", flush=True)
            # 判定寫進 result.json（與 bench 的產物同名同義）——報告與記錄看的是同一份。
            # 同一扇門也要同樣的成對要求：`verify` 的每一輪是獨立一次 matrix 執行，
            # 它的 log 必須躺在那一輪的 `matrix.json` 旁邊（見 llm_bench_matrix 的寫入端）。
            pair_gate = _pair_gate_verify(_load("ibind_vp", "instrument_binding.py"), res)
            res["instrument_gate"] = ib_gate
            res["pair_gate"] = pair_gate
            result_json.write_text(json.dumps(res, indent=2, ensure_ascii=False))
            for p in pair_gate["passes"]:
                print(f"  {'✅' if p['paired'] else '⛔'} 成對 log  {p['pass']}: {p['why']}", flush=True)
            if not pair_gate["paired"]:
                print("!! 成對閘 FAIL — **拒收**：至少一輪的產物旁邊沒有 stderr log（不可引用，"
                      "且這一條沒有豁免）。", flush=True)
                if rc == 0:
                    rc = 5
            if not ib_gate["ok"] and not waive_ib:
                print("!! 量具閘 FAIL — 這份 verify 報告不算數：量具沒動，"
                      "那些計數器與由此而來的結論不可引用。", flush=True)
                print("   （HTML 仍生成，看得到是哪一臂；要明知不可引用還是接受："
                      "--waive-instrument \"理由\"）", flush=True)
                if rc == 0:
                    rc = 4
            elif not ib_gate["ok"] and waive_ib:
                print(f"!! 量具閘 FAIL，但已豁免（--waive-instrument）：{waive_ib}", flush=True)

    if result_json.exists():
        hcmd = [PY, str(HERE / "arm_report_html.py"),
                "--json", str(result_json), "--out", str(report_html)]
        hrc = subprocess.call(hcmd, cwd=str(ROOT))
        if hrc == 0:
            print(f"\nHTML 報告：{report_html}", flush=True)
    else:
        print("\n（無 result.json — 通常是 gate 在跑任何 GPU 前就擋下；未生成 HTML）", flush=True)

    # 0 = 全部臂兩輪通過；2/3 = gate／窗口擋下；4 = 量具 UNBOUND；5 = 缺成對 log（皆拒收）
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("show", aliases=("windows",), help="the window board (default)")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--need-mb", type=float, default=0)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("run", help="wait for a window, then run one tool through it")
    run_p = p
    p.add_argument("tool")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--need-mb", type=float, default=0)
    p.add_argument("--window-timeout-s", type=float, default=900)
    p.add_argument("--poll-s", type=float, default=15)
    p.add_argument("--certify", action="store_true",
                   help="measure throughput against the sentinel reference first (refuses a "
                        "degraded window that looks quiet)")
    p.add_argument("--certify-frac", type=float, default=0.85)
    p.add_argument("--charter", default=None,
                   help="跑前立項卡（charters/*.yaml）；速度實驗必填，純分析工具可不帶")
    p.add_argument("--waive-reason", default=None,
                   help='--charter waive 時必填：急件不立項的理由（記進產物）')
    # NOT argparse.REMAINDER: a REMAINDER positional placed after `tool` swallows the harness's own
    # options too (it is greedy from the first token it cannot name), so `--window-timeout-s 4`
    # was handed to the child and the run waited at the default 900 s instead. The split is done
    # by hand at the first bare `--` in main().
    p.set_defaults(func=cmd_run, rest=[])

    p = sub.add_parser("list", help="the registry")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("audit", help="ungated server launchers (window gate)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("selftest")
    p.set_defaults(func=lambda a: selftest())

    p = sub.add_parser("bench", help="統一量測入口：prod-new 基底 + env 增量 + base gate + llama-bench 完整側")
    p.add_argument("--arm", action="append", required=True,
                   help='PROFILE:!OVERRIDE;KEY=VAL 語法（可多次）。! = 顯式覆蓋 base 鍵；未宣告的 base 變更 → 拒跑')
    p.add_argument("--prompt", type=int, default=_BENCH_DEFAULTS["prompt"])
    p.add_argument("--gen", type=int, default=_BENCH_DEFAULTS["gen"])
    p.add_argument("--depths", default=_BENCH_DEFAULTS["depths"])
    p.add_argument("--reps", type=int, default=_BENCH_DEFAULTS["reps"])
    p.add_argument("--warm-skip", type=int, default=_BENCH_DEFAULTS["warm_skip"])
    p.add_argument("--ctx-size", type=int, default=_BENCH_DEFAULTS["ctx_size"])
    p.add_argument("--fixed-fill-seed", type=int, default=_BENCH_DEFAULTS["fixed_fill_seed"],
                   help="llama-bench --fixed-fill-seed（預設 1＝每個 rep 重播同一條填充流，池才到得住穩態；"
                        "0 = llama-bench 歷史行為＝每 rep 換一條 ⇒ cold/steady 混樣）")
    p.add_argument("--prompt-file", default=_BENCH_DEFAULTS["prompt_file"],
                   help="llama-bench --prompt-file（真 prompt；隨機填充的路由不是真實語言分佈）")
    p.add_argument("--no-warmup", dest="warmup", action="store_false",
                   default=_BENCH_DEFAULTS["warmup"],
                   help="跳過 llama-bench 暖機（預設暖機 ON：它才是「池已暖」的等價物）")
    p.add_argument("--batch", type=int, default=_BENCH_DEFAULTS["batch"], help="llama-bench -b")
    # [CGC 2026-10-04] `--ubatch` 一直是 `_BENCH_DEFAULTS` 裡的一個**死欄位**（沒有任何地方傳它）⇒
    # 只有 `--batch` 到得了 matrix，而 matrix 的舊語意「-b X ⇒ -ub 也 X」再把格的 ubatch 蓋掉
    # ⇒ 契約以「ubatch: 實際 2048 ≠ 權威 512」拒跑（新預設格 b2048/ub512 踩到）。
    # 形狀既然由卡片宣告，這條路就必須把該維度**明文送出去**，不是靠下游猜。
    p.add_argument("--ubatch", type=int, default=_BENCH_DEFAULTS["ubatch"], help="llama-bench -ub")
    p.add_argument("--cool-max-s", type=float, default=420.0,
                   help="thermal 閘的最長冷卻秒數（420 = repo 量到的最小充分冷卻）")
    p.add_argument("--no-thermal-gate", action="store_true",
                   help="不等待 NOMINAL（只警告）-- 非量測用途才用")
    p.add_argument("--spec-type", default="",
                   help="llama-bench --spec-type（僅 draft-mtp）；空 = 一般 cell（預設行為）")
    p.add_argument("--spec-draft-n-max", type=int, default=None,
                   help="llama-bench --spec-draft-n-max（1..16）；需配合 --spec-type")
    p.add_argument("--cell", default=None,
                   help="測試卡 §2.5 的 named cell（預設＝權威 prod-new cell）。未宣告的名字是 "
                        "fail-closed。交付 cell（batch 512 / prompt 0）宣告為 `delivery`。")
    p.add_argument("--require-instrument", default="",
                   help="第五道閘門：宣告「這一輪必須證明這些量具真的動過」（逗號分隔，例："
                        "rb_feed,prefetch）。跑前擋「武裝了沒開前提」，跑後若 UNBOUND 就拒寫進 mindmap。"
                        "判準見 scripts/check/instrument_binding.py")
    p.add_argument("--waive-instrument", default=None,
                   help="理由字串：明知量具不會動而仍要寫入決策面（會連理由一起記進產物的 "
                        "instrument_gate.waived）。rc=4 的閘門用這個放行")
    # 【CGC 2026-09-29】「跑不完」那一類診斷用的三根桿子（見 llama_bench_matrix `_stream_child`）：
    # 子行程 stderr 逐行落地（被殺也留得下停滯點）＋ 停滯時取堆疊 ＋ 逾時只殺自己的 pgid。
    p.add_argument("--arm-timeout", type=float, default=0.0,
                   help="單臂牆鐘上限（秒）：逾時只殺自己的 process group，驅動留下來寫 live log／產物。"
                        " 0 = 不設（歷史行為）")
    p.add_argument("--stall-watch", type=float, default=0.0,
                   help="停滯看門狗（秒）：連續這麼久沒有新的一行 stderr ⇒ 記下停滯點並 sample 堆疊。"
                        " 0 = 關")
    p.add_argument("--stall-kill", action="store_true",
                   help="第一次抓到停滯就先取堆疊再殺（預設只記錄，讓 --arm-timeout 決定收尾）")
    p.add_argument("--stall-progress-re", default="",
                   help="「進展行」的判準（regex）。空 = 矩陣的預設（心跳 CGC-RSS 不算進展——引擎有心跳，"
                        "所以「有沒有新的一行」量不出卡住）")
    p.add_argument("--live-echo", default="",
                   help="額外把符合這個 regex 的 stderr 行即時印到 stdout（預設用矩陣內建的那一組）")
    p.add_argument("--window-timeout-s", type=float, default=900)
    p.add_argument("--workdir", default="/tmp/harness_bench")
    p.add_argument("--json", dest="json_path", required=True, help="產物 json 路徑（含 base_check）")
    p.add_argument("--charter", default=None,
                   help="跑前立項卡路徑（scripts/check/charters/*.yaml）；速度實驗必填，"
                        "急件用 --charter waive --waive-reason")
    p.add_argument("--waive-reason", default=None,
                   help='--charter waive 時必填：急件不立項的理由（記進產物）')
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("verify",
        help="雙輪驗證：每 arm clean（無儀器）+ instrumented（有儀器）+ 生產級 gate + HTML 報告")
    p.add_argument("--arm", action="append", required=True,
                   help='PROFILE:!K=V;K=V（可多臂）。每臂強制兩輪、兩輪只差儀器')
    p.add_argument("--prompt", type=int, default=2048)
    p.add_argument("--gen", type=int, default=128)
    p.add_argument("--depths", default="512")
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--warm-skip", type=int, default=64)
    p.add_argument("--ctx-size", type=int, default=0)
    p.add_argument("--max-swap-mb", type=float, default=1024,
                   help="起測允許的最大 swap（MiB）；生產嚴格可調 512")
    p.add_argument("--cool-s", type=float, default=0,
                   help="兩輪之間冷卻秒數（預設 0，避免拖慢開發）")
    p.add_argument("--allow-dirty", action="store_true",
                   help="gate 未過也硬跑、報告標紅（不建議用於 commit）")
    p.add_argument("--run-dir", default="/tmp/harness_verify")
    p.add_argument("--waive-instrument", default=None,
                   help="理由字串：明知量具不會動而仍要收下這份 verify 報告（會記進 "
                        "result.json 的 instrument_gate.waived）。rc=4 的閘門用這個放行")
    p.set_defaults(func=cmd_verify)

    raw = list(sys.argv[1:] if argv is None else argv)
    cmd = raw[0] if raw and not raw[0].startswith("-") else None
    if cmd == "run":
        # Everything after the first bare `--` belongs to the child, verbatim; everything before it
        # is ours. No guessing which flags belong to whom.
        tail = raw[1:]
        if "--" in tail:
            i = tail.index("--")
            head, rest = tail[:i], tail[i + 1:]
        else:
            head, rest = tail, []
        args = run_p.parse_args(head)
        args.rest = rest
    elif cmd is None:
        args = ap.parse_args(raw + ["show"])
    else:
        args = ap.parse_args(raw)
    if not getattr(args, "func", None):
        args = ap.parse_args(raw + ["show"])
    # 0 means "ask the shared probe what it wants", so a caller that passes nothing inherits the
    # same bar the launcher uses instead of silently demanding zero.
    if getattr(args, "need_mb", 0) == 0:
        try:
            args.need_mb = float(_load("sw", "server_window.py").NEED_MB)
        except Exception:
            args.need_mb = 8000.0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
