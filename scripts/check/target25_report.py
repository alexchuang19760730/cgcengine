#!/usr/bin/env python3
"""One command from measurement to report: run the chain, then render ONE artifact with its provenance.

WHY THIS EXISTS
---------------
The six tools each answer one question (`docs/TOOLCHAIN_25TPS_2026-09-22.html`), and each one already
refuses when its input cannot support an answer. What was still manual was the ORDER: run the sweep,
then the round budget, then the ladder, then the accept axis, then paste the outputs into a document.
That is how the numbers got mixed before -- a reading lifted from one arm into another arm's table, a
report quoting a step total from a different regime. So the driver runs the stages in order, captures
each tool's OWN stdout verbatim, and renders it.

IT DOES NOT RE-DERIVE A SINGLE NUMBER.
Every figure in the HTML is quoted from a tool's own output (or from the sweep's record). There is no
second parser and no recomputation, which is deliberate: this repo already paid once for having two
definitions of "free memory", and a report that recomputes the number its tool printed can disagree
with it. If a tool refuses, its section says INVALID with the tool's own reason -- it is never dropped,
because a missing section reads as "nothing to report".

WHAT "PROVENANCE" MEANS HERE, precisely
---------------------------------------
Three different things get called provenance, and conflating them is what produces fabricated numbers:

  engine_digest      `mtp_k_sweep.engine_digest()` -- sha256[:16] of every dylib/so plus llama-bench.
                     Taken NOW, and per arm from the sweep's own record.
  measure-time window  per arm, from `arms.jsonl` (`window_before` / `window_after`: usable_pct,
                     reclaimable, swap_used_mb) plus the arm's thermal labels. This is the state the
                     measurement actually ran under.
  report-time window   `server_window.provenance()`, taken in THIS process, and it carries `class`
                     (clean / busy-then-cleared / busy-overridden / unknown). It is NOT the window the
                     measurement ran under. Both are printed, labelled, side by side.

`server_window`'s own docstring states the rule this follows: provenance is taken in process, never
stamped on afterwards, "because the window state at write time is not the window state the run
measured under, so a post-hoc stamp would be a fabricated provenance".

The per-arm `window` block IS a derivation -- the sweep records `window_before`, not a class -- so it
names its rule in `derived` and falls back to `unknown` with a reason rather than guessing. It exists
because the repo's own `server_window.py audit-products` asks of every measured object: can this
artifact answer "was the box busy?". A reading that cannot answer it should say so in a field, not by
being silent.

HONEST BOUNDARY ON THE GATE
---------------------------
`--measure` spawns `mtp_k_sweep.py`, which spawns the launcher. So this file is an INDIRECT launcher:
it calls `require_first()` before spawning anything, which refuses on a busy box, and it checks that
`scripts/run_server.sh` exists before starting. What it cannot do is re-check the box per arm -- the
sweep owns that, and its own gate (via prod_matrix) is the one that decides each launch.

Usage
-----
    python3 scripts/check/target25_report.py --pair /tmp/verify_layers
    python3 scripts/check/target25_report.py --measure --logdir /tmp/mtp_k_sweep --reps 3
    python3 scripts/check/target25_report.py --pair /tmp/verify_layers \\
        --roofline-capture Backup/roofline/phase2_154436.txt --gguf models/gguf/<carrier>.gguf
    python3 scripts/check/target25_report.py --selftest
"""
from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECK = Path(__file__).resolve().parent
sys.path.insert(0, str(CHECK))

# The launcher this driver ultimately starts (through the sweep). Checked for existence before a
# measurement, so a missing launcher is a named error instead of a sweeper that "measured" nothing.
LAUNCHER = ROOT / "scripts" / "run_server.sh"

import mtp_k_sweep as ks            # noqa: E402  (engine_digest, window: one source, not a second)
import server_window as sw          # noqa: E402  (the shared probe + its provenance taxonomy)

PY = sys.executable
# rc != 0 is a refusal; rc == 0 with one of these is still a refusal the tool printed on purpose.
REFUSAL_MARKERS = ("INVALID:", "INVALID --", "no valid arm yet", "refused --", "FATAL:")


# ------------------------------------------------------------------ process plumbing

def run_stage(argv: list[str], timeout: int = 900) -> dict:
    """Run one tool and keep its output VERBATIM. Stderr is kept separately: several tools print
    their instruments there, and merging the two would misattribute a warning as a result."""
    try:
        p = subprocess.run(argv, capture_output=True, text=True, cwd=str(ROOT), timeout=timeout)
    except subprocess.TimeoutExpired as e:
        return {"rc": -9, "out": e.stdout or "", "err": (e.stderr or "") + f"\nTIMEOUT after {timeout}s"}
    return {"rc": p.returncode, "out": p.stdout, "err": p.stderr}


def classify(rc: int, text: str) -> tuple[str, str]:
    """(status, why). A tool that refused on purpose and a tool that crashed are not the same thing:
    the first is a supported outcome, the second means the report lost a section."""
    if rc != 0:
        why = next((l.strip() for l in text.splitlines()
                    if any(m in l for m in REFUSAL_MARKERS)), "")
        if why:
            return "refused", why
        tail = [l for l in text.splitlines() if l.strip()][-1:] or ["(no output)"]
        return "error", f"exit {rc}: {tail[0].strip()[:200]}"
    for m in REFUSAL_MARKERS:
        if m in text:
            why = next((l.strip() for l in text.splitlines() if m in l), m)
            return "refused", why
    return "ok", ""


def selftest_count(text: str) -> str:
    """The tools print three formats (`N/N passed`, `selftest: PASS`, `SELFTEST PASS`). All three are
    read here rather than normalising six tools that each print what their own check style wants."""
    if (m := re.search(r"(\d+)\s*/\s*(\d+)\s+passed", text)):
        return f"{m.group(1)}/{m.group(2)}"
    if re.search(r"(?i)\d+\s+FAILED", text):
        return "FAILED"
    ok = len([l for l in text.splitlines() if re.match(r"^\s*(?:\[ok\]|ok)\s+\S", l)])
    if re.search(r"(?i)selftest:?\s*PASS", text):
        return f"{ok} ok" if ok else "PASS"
    return "?"


# ------------------------------------------------------------------ the stages

def stage_defs(pair: Path, *, k: int, per_miss: float, layer_log: str,
               capture: str, gguf: str, gguf_n_used: int | None) -> list[dict]:
    """The chain, in the order the docs run it. Each entry names the script and the exact argv, so
    the report can show what was run instead of describing it."""
    def s(name, title, script, argv, note="", selftest_argv=("--selftest",), **kw):
        return {"name": name, "title": title, "script": script, "argv": argv, "note": note,
                "selftest_argv": list(selftest_argv), **kw}

    out = [
        s("pair", "量測來源：這一對 k 臂的身分",
          "mtp_k_sweep.py", ["--report-only", "--logdir", str(pair)],
          "同 harness、同窗口的 k 成本曲線。這一節是唯一引用「量測時」狀態的地方。"),
        s("marginal", "邊際 verify token 的三路拆分 + round 預算收斂",
          "verify_marginal.py", ["--logdir", str(pair), "--per-miss-ms", repr(per_miss)],
          "miss / verify-matmul / host 三軸，以及 remainder 在邊際裡佔多少。"),
        s("ladder", "25 t/s 必要條件階梯（每往下一階＝移除一個實測項）",
          "target25_grid.py", ["--from-pair", str(pair), "--k", str(k)],
          "「impossible」＝連 α=1 都到不了；那是這張表存在的理由。"),
        s("accept", "accept 軸的天花板與最佳 k",
          "draft_budget.py", [], "沒有正規化：預設值就是 S0=93.3 / step_T4=247.98 的實測點。"),
    ]
    if layer_log:
        out.append(s("layers", "逐層：空窗是 miss 買的、還是固定成本",
                     "gap_vs_miss.py", ["--log", layer_log],
                     "相位邊界＝最後一條 CGC-POST:；缺它就不給判定。"))
    if capture and gguf:
        argv = ["--capture", capture, "--gguf", gguf]
        if gguf_n_used is not None:
            argv += ["--n-used", str(gguf_n_used)]
        out.append(s("roofline", "kernel 的頭頂：天花板（算術）vs 實測 kernel",
                     "shape_roofline.py", argv,
                     "未知型別或缺測資一律拒算 —— 靜默代換的 roofline 表比沒有表更糟。"))
    # server_window is a SUBCOMMAND cli (`selftest`, `audit`) -- `--selftest` is an argparse error
    # (rc 2), which the first version of this driver rendered as "FAILED" for a passing tool. A
    # failing selftest line in a report is worse than none: it trains the reader to ignore it.
    out.append(s("gate", "這條鏈自己的閘門狀態（誰會在啟動前問共有探針）",
                 "server_window.py", ["audit"],
                 "把註冊的啟動器數量留在產物裡，讓「未接閘門的啟動器只能下降」可以從這份報告問。",
                 selftest_argv=["selftest"]))
    return out


def run_stages(defs: list[dict]) -> list[dict]:
    rows = []
    for d in defs:
        argv = [PY, str(CHECK / d["script"]), *d["argv"]]
        r = run_stage(argv)
        status, why = classify(r["rc"], r["out"] + r["err"])
        rows.append({**{kk: v for kk, v in d.items() if kk != "argv"},
                     "argv": argv, "rc": r["rc"], "status": status, "why": why,
                     "out": r["out"], "stderr": r["err"]})
    return rows


def selftests(argvs: dict[str, list[str]]) -> list[dict]:
    rows = []
    for name in sorted(argvs):
        r = run_stage([PY, str(CHECK / name), *argvs[name]], timeout=300)
        text = r["out"] + r["err"]
        rows.append({"tool": name, "rc": r["rc"],
                     "status": "ok" if r["rc"] == 0 else "FAILED",
                     "counts": selftest_count(text),
                     "why": "" if r["rc"] == 0 else classify(r["rc"], text)[1]})
    return rows


# ------------------------------------------------------------------ the pair (measure-time truth)

def load_arms(jsonl: Path) -> tuple[list[dict], str]:
    if not jsonl.exists():
        return [], f"no {jsonl}"
    recs = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
    if not recs:
        return [], f"{jsonl} is empty"
    return recs, ""


# The rule is about the MEMORY claim, which is what the class is for. It deliberately does NOT read
# the record's `rc`/`error`: prod_matrix sets `error` only when its subprocess exited non-zero, and
# that subprocess is llama-bench, which exits non-zero on the first bad shape while the matrix
# recovers the instances that succeeded (its own comment says so, and `n_kept=2` with `rc=1` is the
# normal shape on this carrier). So `rc != 0` here is a caveat to print, not a verdict -- folding it
# into the class would have labelled all six measured arms `unknown` and hidden the real answer.
ARM_WINDOW_RULE = ("clean iff usable_pct >= the sweep's --min-usable-pct; unknown if the arm has "
                   "the sweep's own `invalid` key, or usable_pct is missing, or it is below the "
                   "gate (this artifact cannot tell whether the sweep proceeded or skipped)")


def arm_window(rec: dict, min_usable: float) -> dict:
    w = rec.get("window_before") or {}
    pct = w.get("usable_pct")
    note = "" if rec.get("rc") in (0, None) else f"; matrix rc={rec.get('rc')} (recovered instances)"
    if rec.get("invalid"):
        return {"class": "unknown", "why": "the arm recorded no usable number",
                "derived": ARM_WINDOW_RULE, "source": "measure-time"}
    if pct is None:
        return {"class": "unknown", "why": "no window_before.usable_pct in the record",
                "derived": ARM_WINDOW_RULE, "source": "measure-time"}
    if pct >= min_usable:
        return {"class": "clean", "why": f"usable_pct={pct:.1f} >= gate {min_usable:.0f}{note}",
                "derived": ARM_WINDOW_RULE, "source": "measure-time"}
    return {"class": "unknown",
            "why": f"usable_pct={pct:.1f} < gate {min_usable:.0f} — the sweep's own gate would "
                   f"have skipped this arm; the record cannot say whether it did{note}",
            "derived": ARM_WINDOW_RULE, "source": "measure-time"}


ARM_KEYS = ("rep", "arm", "k", "tps", "round_ms", "E", "mean_len", "residual_ms", "engine_digest",
            "rc", "wall_s", "n_kept", "extra_env", "window_before", "window_after", "pool",
            "launch_thermal", "worst_thermal", "dir")


def summarize_arms(recs: list[dict], min_usable: float) -> dict:
    """Every object here that carries a measured key also carries a `window`, including the two
    nested ones the repo's `audit-products` counts (`measure_time` carries `arms`, each arm's `pool`
    carries `requests`). Without that, an artifact that DOES record the box state still answers
    "was the box busy?" for only half its objects -- and the audit's second number is the one that
    matters."""
    arms = []
    for r in recs:
        row = {k: r.get(k) for k in ARM_KEYS}
        row["window"] = arm_window(r, min_usable)
        if isinstance(row.get("pool"), dict):
            # the pool counters are this arm's, so they carry this arm's window
            row["pool"] = {**row["pool"], "window": row["window"]}
        row["invalid"] = r.get("invalid")
        arms.append(row)
    classes: dict[str, int] = {}
    for a in arms:
        classes[a["window"]["class"]] = classes.get(a["window"]["class"], 0) + 1
    digests = sorted({json.dumps(a["engine_digest"], sort_keys=True) for a in arms if a["engine_digest"]})
    if len(classes) == 1:
        agg, why = next(iter(classes)), f"all {len(arms)} arm record(s): {json.dumps(classes)}"
    else:
        agg = "unknown"
        why = f"mixed per-arm classes {json.dumps(classes, sort_keys=True)} -- see arms[].window"
    return {"n_records": len(recs), "arms": arms, "window_classes": classes,
            "window": {"class": agg, "why": why, "derived": ARM_WINDOW_RULE,
                       "source": "measure-time", "per_arm": "arms[].window"},
            "engine_digests": [json.loads(d) for d in digests],
            "nonzero_rc_records": sum(1 for r in recs if r.get("rc") not in (0, None)),
            "usable_arms": sorted({a["arm"] for a in arms if a["arm"] and not a["invalid"]})}


def git_state() -> dict:
    def g(*argv):
        try:
            return subprocess.run(["git", "-C", str(ROOT), *argv], capture_output=True,
                                  text=True).stdout.strip()
        except Exception:
            return ""
    dirty = [l for l in g("status", "--porcelain").splitlines() if l.strip()]
    return {"head": g("rev-parse", "--short", "HEAD"),
            "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_files": len(dirty),
            "dirty_sample": dirty[:6]}


def provenance() -> dict:
    """The window block this artifact is audited on (`server_window.py audit-products`)."""
    prov = sw.provenance()
    prov["swap_used_mb"] = ks.window().get("swap_used_mb")
    prov["taken"] = "report time, in this process"
    prov["not_the_measurement_window"] = (
        "this is the box NOW. The window each arm ran under is in measure_time, read from "
        "arms.jsonl (window_before/window_after) -- the two are different states and are printed "
        "separately on purpose")
    return prov


# ------------------------------------------------------------------ rendering

CSS = """
:root{--bg:#fff;--fg:#111827;--muted:#6b7280;--line:#e5e7eb;--blue:#2563eb;--blue-bg:#eff6ff;
--red:#dc2626;--red-bg:#fef2f2;--green:#059669;--green-bg:#ecfdf5;--amber:#d97706;--amber-bg:#fffbeb;
--code-bg:#f6f8fa}
*{box-sizing:border-box}
body{margin:0;padding:32px 20px 80px;background:var(--bg);color:var(--fg);
font:15px/1.75 -apple-system,BlinkMacSystemFont,"PingFang TC","Helvetica Neue",Arial,sans-serif}
.wrap{max-width:1080px;margin:0 auto}
h1{font-size:26px;margin:0 0 6px;letter-spacing:-.2px}
h2{font-size:20px;margin:40px 0 12px;padding-bottom:8px;border-bottom:2px solid var(--blue)}
h3{font-size:15.5px;margin:24px 0 8px}
.sub{color:var(--muted);font-size:13px;margin-bottom:24px}
code,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12.5px}
code{background:var(--code-bg);padding:1px 5px;border-radius:4px;color:#b91c1c}
pre{background:var(--code-bg);border:1px solid var(--line);border-radius:8px;padding:14px 16px;
overflow-x:auto;font-size:12.5px;line-height:1.6;white-space:pre-wrap}
pre code{background:none;padding:0;color:#111827}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:13.5px}
th,td{border:1px solid var(--line);padding:8px 10px;text-align:left;vertical-align:top}
th{background:#f9fafb;font-weight:600}
td.num,th.num{font-family:ui-monospace,Menlo,monospace;white-space:nowrap;text-align:right}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:18px 0}
.card{border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.card .k{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px}
.card .v{font-size:21px;font-weight:700;margin-top:2px}
.ok{color:var(--green)} .bad{color:var(--red)} .warn{color:var(--amber)}
.box{border-radius:8px;padding:12px 16px;margin:16px 0;border:1px solid}
.box p{margin:6px 0}
.b-red{background:var(--red-bg);border-color:#fecaca}
.b-green{background:var(--green-bg);border-color:#a7f3d0}
.b-amber{background:var(--amber-bg);border-color:#fde68a}
.b-blue{background:var(--blue-bg);border-color:#bfdbfe}
.stage{border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:20px 0}
.stage.bad-stage{border-color:#fecaca;background:#fffbfb}
.pill{display:inline-block;font-size:11.5px;padding:1px 8px;border-radius:999px;
border:1px solid var(--line);background:var(--code-bg);font-family:ui-monospace,Menlo,monospace}
ul,ol{margin:8px 0 8px 22px;padding:0} li{margin:5px 0}
.small{font-size:12.5px;color:var(--muted)}
.dg{word-break:break-all;font-family:ui-monospace,Menlo,monospace;font-size:12px}
summary{cursor:pointer;color:var(--blue);font-size:13.5px;margin:6px 0}
"""

STATUS_STYLE = {"ok": ("ok", "OK"), "refused": ("warn", "REFUSED (fail-closed)"),
                "error": ("bad", "ERROR")}


def es(x) -> str:
    return html.escape("" if x is None else str(x))


def cmdline(argv: list[str]) -> str:
    """Repo-relative, so the line printed in the report can be pasted. The absolute interpreter path
    and the worktree prefix are this machine's, not part of the measurement. The sidecar keeps the
    real argv verbatim."""
    def rel(a: str) -> str:
        try:
            return Path(a).relative_to(ROOT).as_posix()
        except ValueError:
            return a
    out = ["python3" if i == 0 and Path(a).name.startswith("python") else rel(a)
           for i, a in enumerate(argv)]
    return " ".join(out)


def fmt(x, nd: int) -> str:
    """Display rounding only. The raw float is what `arms.jsonl` holds and what the sidecar keeps --
    `7.3117300000000001` and `62.713050842285156` are what json round-tripping leaves behind, and a
    table nobody can read at a glance is not more honest, just harder to check against the tools."""
    if x is None:
        return "—"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return es(x)


def digest_brief(d: dict) -> str:
    """A one-line digest summary. The full map is 35 artifacts and printing it inline collapsed the
    provenance table into a single character-wrapped column on the first render.

    The named picks go by PREFIX, not by versioned filename: `mtp_k_sweep.engine_digest` is a glob
    precisely so a version bump cannot make it go quietly stale, and hard-coding `libllama.0.0.279`
    here would put that staleness back.
    """
    def pick(pred):
        return next((f"{k}={v}" for k, v in sorted(d.items()) if pred(k)), "")
    named = [pick(lambda k: k == "llama-bench"), pick(lambda k: k.startswith("libggml-metal")),
             pick(lambda k: k.startswith("libllama."))]
    return f"{len(d)} 個產物" + (" · " + " · ".join(x for x in named if x) if any(named) else "")


def digest_block(title: str, digests: list[dict], now: dict) -> str:
    """The full maps, folded away. Folded rather than omitted: a digest nobody can read is how a
    report claims provenance without providing it."""
    if not digests:
        return ""
    same = all(d == now for d in digests)
    mark = ("<strong class='ok'>相同</strong>" if same else
            "<strong class='bad'>不同</strong>（量測來自別的 build）")
    rows = [f"<tr><td>與現在的 build</td><td>{mark}</td></tr>"]
    for d in digests:
        rows.append(f"<tr><td class='dg'>{es(digest_brief(d))}</td><td class='dg'>"
                    + "<br>".join(f"{es(k)} <span class='small'>{es(v)}</span>"
                                   for k, v in sorted(d.items())) + "</td></tr>")
    return (f"<details><summary>{es(title)}（{len(digests)} 組）</summary><table><tbody>"
            + "".join(rows) + "</tbody></table></details>")


def render_html(doc: dict) -> str:
    st = doc["stages"]
    good = [s for s in st if s["status"] == "ok"]
    bad = [s for s in st if s["status"] != "ok"]
    ftests = doc["selftests"]
    tests_ok = sum(1 for t in ftests if t["status"] == "ok")
    prov = doc["window"]
    mt = doc["measure_time"]
    arms = mt["arms"]
    usable = [a for a in arms if a.get("tps") is not None]

    P = []
    A = P.append
    A(f"<!DOCTYPE html><html lang='zh-Hant'><head><meta charset='utf-8'>"
      f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
      f"<title>25 t/s 報告 {es(doc['generated_at'])}</title><style>{CSS}</style></head><body>"
      f"<div class='wrap'>")
    A(f"<h1>25 t/s 報告 · <span class='mono'>{es(doc['generated_at'])}</span></h1>")
    A(f"<div class='sub'>由 <code>scripts/check/target25_report.py</code> 產生 —— 這個檔案裡的每一個數字"
      f"都是<strong>引用</strong>自某支工具的原始輸出，不是重算的。<br>"
      f"量測來源 <code>{es(doc['pair']['dir'])}</code> · 分支 <code>{es(doc['git']['branch'])}</code> "
      f"@ <code>{es(doc['git']['head'])}</code>"
      f"{'（工作區有 ' + str(doc['git']['dirty_files']) + ' 個未提交檔案）' if doc['git']['dirty_files'] else ''}</div>")

    A("<div class='cards'>")
    A(f"<div class='card'><div class='k'>量測臂</div><div class='v'>{len(usable)}/{len(arms)}</div>"
      f"<div class='small'>有可用 t/s 的臂；--min-usable-pct={es(doc['min_usable_pct'])}</div></div>")
    A(f"<div class='card'><div class='k'>分析階段</div>"
      f"<div class='v {'ok' if not bad else 'warn'}'>{len(good)}/{len(st)}</div>"
      f"<div class='small'>{'全部完成' if not bad else '有階段拒算，見下'}</div></div>")
    A(f"<div class='card'><div class='k'>工具自測</div>"
      f"<div class='v {'ok' if tests_ok == len(ftests) else 'bad'}'>{tests_ok}/{len(ftests)}</div>"
      f"<div class='small'>產生這份報告的每一支工具</div></div>")
    cls = prov.get("class", "unknown")
    # `unknown` has two very different causes and they must not be shown the same way: the probe was
    # asked and the box was busy (alarming, affects the numbers) versus an analysis-only run that
    # never asked (neutral -- the numbers came from an earlier run whose own window is in section 2).
    busy_now = cls in ("busy-then-cleared", "busy-overridden")
    A(f"<div class='card'><div class='k'>報告時的盒況（class）</div>"
      f"<div class='v {'ok' if cls == 'clean' else 'warn'}'>{es(cls)}</div>"
      f"<div class='small'>這是<strong>現在</strong>，不是量測當時 —— 見下</div></div>")
    A("</div>")

    if busy_now or mt["window_classes"].get("unknown"):
        A("<div class='box b-amber'><p><strong>盒況不是乾淨的。</strong>")
        A(f"報告時：{es(prov.get('why', ''))}</p>")
        if mt["window_classes"].get("unknown"):
            A(f"<p>量測時有 {mt['window_classes']['unknown']} 個臂的窗口<em>無法判定</em>"
              f"（規則：{es(ARM_WINDOW_RULE)}）。這些讀數要當成「可能描述鄰居」處理。</p>")
        A("</div>")
    elif cls == "unknown":
        A("<div class='box b-blue'><p><strong>報告時的盒況＝unknown，因為這一輪沒有問探針</strong>"
          "（<code>--pair</code> 分析模式只讀既有紀錄，不啟動任何服務）。這是誠實的「不知道」，"
          "不是「盒況有問題」：這份報告的數字來自先前那一次量測，而<strong>那一次</strong>的窗口"
          "逐臂列在第二節。<code>--measure</code> 模式會在啟動前問探針，class 就會變成 "
          "<code>clean</code> 或指出誰占著盒子。</p></div>")

    if bad:
        A("<div class='box b-red'><p><strong>以下階段拒絕給答案</strong>（這是設計，不是失敗）：</p><ul>")
        for s in bad:
            A(f"<li><code>{es(s['script'])}</code> — {es(s['why'])}</li>")
        A("</ul><p class='small'>拒算的階段仍然完整列在下面，不會被省略：一個消失的段落會被讀成"
          "「沒東西可報」。</p></div>")

    # ---- provenance
    A("<h2>一、出處（provenance）</h2>")
    A("<p>這份報告刻意把三件常被混為一談的東西分開列。混在一起就會生出偽造的出處。</p>")
    A("<table><thead><tr><th>項</th><th>值</th></tr></thead><tbody>")
    A(f"<tr><td>引擎 digest（現在）</td><td>{es(digest_brief(doc['engine_digest']))} · "
      f"<code>llama-bench={es(doc['engine_digest'].get('llama-bench'))}</code></td></tr>")
    meas = mt["engine_digests"]
    if not meas:
        verdict = "—"
    elif all(d == doc["engine_digest"] for d in meas):
        verdict = "<strong class='ok'>與現在相同</strong>"
    else:
        verdict = "<strong class='bad'>與現在不同</strong>"
    A(f"<tr><td>引擎 digest（量測時）</td>"
      f"<td>{es(digest_brief(meas[0])) if meas else '—'} · {verdict}</td></tr>")
    A(f"<tr><td>報告時的盒況（<code>server_window.provenance()</code>，in process）</td>"
      f"<td><code>{es(prov.get('why', ''))}</code> · class=<strong>{es(cls)}</strong> · "
      f"n_samples={es(prov.get('n_samples'))} · busy={es(prov.get('busy_samples'))}</td></tr>")
    A(f"<tr><td>量測時的盒況</td><td>{es(mt['n_records'])} 筆臂紀錄，逐臂列在下一節"
      f"（window_before / window_after 原文照登）</td></tr>")
    A(f"<tr><td>arms.jsonl</td><td><code>{es(doc['pair']['arms_jsonl'])}</code> · "
      f"mtime {es(doc['pair']['mtime'])}</td></tr>")
    A(f"<tr><td>工作區</td><td><code>{es(doc['git']['branch'])}</code> @ "
      f"<code>{es(doc['git']['head'])}</code> · 未提交 {es(doc['git']['dirty_files'])} 檔</td></tr>")
    A(f"<tr><td>工具自測</td><td>" + " · ".join(
        f"<code>{es(t['tool'])}</code> {es(t['counts'])}" for t in ftests) + "</td></tr>")
    A("</tbody></table>")
    A(digest_block("完整引擎 digest：現在（量這份報告的 build）", [doc["engine_digest"]],
                   doc["engine_digest"]))
    A(digest_block("完整引擎 digest：量測當時", mt["engine_digests"], doc["engine_digest"]))
    A(f"<div class='box b-blue'><p><strong>為什麼「報告時的盒況」不是「量測時的盒況」：</strong>"
      f"<code>server_window</code> 自己的規則就是出處必須在行程內取得、不能事後補蓋 —— "
      f"因為寫檔時的盒況不是量測當時的盒況，事後補蓋就是偽造。所以這裡兩個都印，並且分開標。</p></div>")

    # ---- measure-time window per arm
    A("<h2>二、量測時的窗口條件（逐臂，原文照登）</h2>")
    A(f"<p class='small'>每個臂的 <code>window</code> 是我從 <code>window_before</code> 推出來的，"
      f"規則寫在欄位裡（{es(ARM_WINDOW_RULE)}）。推不出來的一律 <code>unknown</code> 並附理由，"
      f"而不是猜一個 <code>clean</code>。</p>")
    A("<table><thead><tr><th>rep</th><th>arm</th><th class='num'>t/s</th><th class='num'>round ms</th>"
      "<th class='num'>E</th><th class='num'>usable%</th><th class='num'>swap MB</th>"
      "<th class='num'>rc</th><th>thermal</th><th>window class</th></tr></thead><tbody>")
    for a in arms:
        wb = a.get("window_before") or {}
        A("<tr>"
          f"<td class='num'>{es(a.get('rep'))}</td><td>{es(a.get('arm'))}</td>"
          f"<td class='num'>{fmt(a.get('tps'), 2)}</td>"
          f"<td class='num'>{fmt(a.get('round_ms'), 1)}</td>"
          f"<td class='num'>{fmt(a.get('E'), 3)}</td>"
          f"<td class='num'>{fmt(wb.get('usable_pct'), 1)}</td>"
          f"<td class='num'>{fmt(wb.get('swap_used_mb'), 0)}</td>"
          f"<td class='num'>{es(a.get('rc'))}</td>"
          f"<td>{es(a.get('launch_thermal'))} / {es(a.get('worst_thermal'))}</td>"
          f"<td>{es(a['window']['class'])}<div class='small'>{es(a['window']['why'])}</div></td>"
          "</tr>")
    A("</tbody></table>")
    if mt.get("nonzero_rc_records"):
        A(f"<div class='box b-amber'><p><strong>{es(mt['nonzero_rc_records'])} 筆紀錄的 "
          f"<code>rc!=0</code>。</strong>那個 rc 是 <code>llama-bench</code> 的：它在第一個壞形狀就非零退出，"
          f"而 matrix 會把已完成的實例撈回來（它的註解自己這樣寫，<code>n_kept</code> 就是撈回的數量）。"
          f"所以它在這裡是<strong>要印出來的警告，不是判定</strong> —— 把它折進 window class 會讓所有臂都變成 "
          f"<code>unknown</code>，反而把真正的答案藏起來。</p></div>")
    A(f"<p class='small'>class 分佈：<code>{es(json.dumps(mt['window_classes']))}</code> · "
      f"有可用數字的臂：<code>{es(', '.join(mt['usable_arms']))}</code> · "
      f"表格數字是<strong>顯示用四捨五入</strong>，原文在 sidecar 的 <code>measure_time.arms</code>。</p>")

    # ---- stages
    A("<h2>三、分析階段（原文輸出）</h2>")
    for i, s in enumerate(st, 1):
        cssclass, label = STATUS_STYLE[s["status"]]
        A(f"<div class='stage{' bad-stage' if s['status'] != 'ok' else ''}'>")
        A(f"<h3>{i}. {es(s['title'])} <span class='pill'>{es(s['script'])}</span> "
          f"<span class='{cssclass}'>{label}</span></h3>")
        if s["note"]:
            A(f"<p class='small'>{es(s['note'])}</p>")
        A(f"<p class='small'>指令：<code>{es(cmdline(s['argv']))}</code> · exit={es(s['rc'])}</p>")
        if s["status"] != "ok":
            A(f"<div class='box b-red'><p><strong>拒算理由（工具自己的話）：</strong>"
              f"<code>{es(s['why'])}</code></p></div>")
        A(f"<pre><code>{es(s['out'].strip() or '(no stdout)')}</code></pre>")
        if s["stderr"].strip():
            A(f"<p class='small'>stderr（儀器輸出常在這裡）：</p>"
              f"<pre><code>{es(s['stderr'].strip())}</code></pre>")
        A("</div>")

    A("<h2>四、邊界</h2><ul>")
    A("<li><strong>這份報告不重算任何數字。</strong>沒有第二個解析器：若某個數字與工具自己的輸出不符，"
      "那是工具的輸出，不是我加的。</li>")
    A(f"<li><strong>絕對步時綁窗口。</strong>量測時的 swap 與 usable% 逐臂列在第二節；"
      f"比值可跨 rep 引用，絕對 t/s 不可與乾淨窗口的家族並排。</li>")
    A("<li><strong>逐臂的 window class 是衍生欄位</strong>，不是量測值；它存在的理由是讓產物能回答"
      "「這個數字是不是在繁忙盒況下量的」，推不出來時它說是 unknown。</li>")
    A("<li><strong>M1/M2/M3 逐位元閘門不在這條鏈裡。</strong>這些階段只讀既有日誌、只改環境變數，"
      "不重建引擎；engine digest 逐臂照登，但「逐位元一致」要另外跑閘門才算成立。</li>")
    A("<li><strong>間接啟動器的閘門邊界：</strong><code>--measure</code> 會 spawn <code>mtp_k_sweep</code>"
      "再 spawn launcher；本 driver 只在啟動前問一次共有探針（<code>require_first</code>），"
      "逐臂的閘門由 sweep 自己負責。</li>")
    A("</ul>")
    A(f"<p class='small'>產物：本 HTML 與其 sidecar <code>{es(doc['json_path'])}</code>。"
      f"sidecar 帶著 <code>window.class</code>，所以它可以回答 repo 自己的 "
      f"<code>server_window.py audit-products</code> 那個問題。</p>")
    A("</div></body></html>")
    return "".join(P)


# ------------------------------------------------------------------ driver

def build(args) -> tuple[dict, int]:
    pair = Path(args.pair or args.logdir)
    min_usable = args.min_usable_pct

    # fail closed on the one input that cannot be worked around: the analysis needs the pair.
    if args.measure:
        if not LAUNCHER.exists():
            print(f"INVALID: launcher missing: {LAUNCHER}")
            return {}, 1
        sw.require_first(port=args.port, need_mb=args.need_mb, where="target25_report --measure")
        argv = [PY, str(CHECK / "mtp_k_sweep.py"), "--reps", str(args.reps),
                "--logdir", str(pair), "--min-usable-pct", str(min_usable)]
        if args.extra_env:
            argv += ["--extra-env", args.extra_env]
        r = run_stage(argv, timeout=args.measure_timeout)
        sys.stdout.write(r["out"])
        sys.stderr.write(r["err"])
        if r["rc"] != 0:
            print(f"INVALID: the measurement refused (exit {r['rc']}) -- nothing to report")
            return {}, 1

    recs, why = load_arms(pair / "arms.jsonl")
    if why:
        print(f"INVALID: {why} -- the ladder and the round budget both need a k=1/k=3 pair")
        return {}, 1
    kinds = {r.get("arm") for r in recs}
    need = {f"k{k}" for k in (1, args.k)}
    if not need.issubset(kinds):
        print(f"INVALID: the pair needs both a k1 and a k{args.k} arm (found: {sorted(kinds)})")
        return {}, 1

    defs = stage_defs(pair, k=args.k, per_miss=args.per_miss_ms, layer_log=args.layer_log,
                      capture=args.roofline_capture, gguf=args.gguf, gguf_n_used=args.n_used)
    stages = run_stages(defs)
    tests = [] if args.no_selftests else selftests({d["script"]: d["selftest_argv"] for d in defs})

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) if args.out else ROOT / "docs" / f"TARGET25_REPORT_{stamp}.html"
    doc = {
        "kind": "target25_report",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "engine_digest": ks.engine_digest(),
        "window": provenance(),
        "min_usable_pct": min_usable,
        "pair": {"dir": str(pair), "arms_jsonl": str(pair / "arms.jsonl"),
                 "mtime": time.strftime("%Y-%m-%d %H:%M:%S",
                                        time.localtime((pair / "arms.jsonl").stat().st_mtime))},
        "measure_time": summarize_arms(recs, min_usable),
        "stages": stages,
        "selftests": tests,
        "git": git_state(),
        "json_path": str(out.with_suffix(".json")),
        "argv": sys.argv[1:],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(doc))
    Path(doc["json_path"]).write_text(json.dumps(doc, indent=2, default=str))
    bad = [s for s in stages if s["status"] != "ok"]
    print(f"report : {out}")
    print(f"sidecar: {doc['json_path']}")
    print(f"stages : {len(stages) - len(bad)}/{len(stages)} ok"
          + (f" -- refused: {', '.join(s['script'] for s in bad)}" if bad else ""))
    print(f"gate   : class={doc['window'].get('class')} ({doc['window'].get('why', '')})")
    return doc, 0


def selftest() -> int:
    """The three ways this driver could lie: a refusal rendered as an empty section, a fabricated
    window class, and a report that never got written."""
    bad = 0

    def check(name, cond):
        nonlocal bad
        print(f"  {'ok  ' if cond else 'FAIL'} {name}")
        bad += 0 if cond else 1

    check("rc!=0 is a refusal, not a crash", classify(1, "INVALID: no rows")[0] == "refused")
    check("rc!=0 with no marker is an error", classify(2, "Segmentation fault")[0] == "error")
    check("rc==0 with an INVALID line is still a refusal",
          classify(0, "  rep 1: refused -- no BATCHDBG series")[0] == "refused")
    check("rc==0 clean output is ok", classify(0, "42 t/s\n")[0] == "ok")
    check("the refusal reason is the tool's own line",
          "BATCHDBG" in classify(0, "x\n  rep 1: refused -- no BATCHDBG series\ny")[1])

    check("selftest count: NN/NN passed", selftest_count("target25 selftest: 14/14 passed") == "14/14")
    check("selftest count: FAILED is not PASS", selftest_count("selftest: 2 FAILED") == "FAILED")
    check("selftest count: PASS form counts the [ok] lines",
          selftest_count("  [ok] a\n  [ok] b\nselftest: PASS") == "2 ok")
    check("selftest count: the no-colon SELFTEST PASS form too",
          selftest_count("  ok   a\n  [ok] b\nSELFTEST PASS") == "2 ok")

    # the window derivation
    clean = {"rep": 0, "arm": "k1", "tps": 11.0, "window_before": {"usable_pct": 55.0}}
    low = {"rep": 0, "arm": "k1", "tps": 11.0, "window_before": {"usable_pct": 12.0}}
    none = {"rep": 0, "arm": "k1", "window_before": {}}
    check("above the gate derives clean", arm_window(clean, 30.0)["class"] == "clean")
    check("below the gate is unknown, not clean", arm_window(low, 30.0)["class"] == "unknown")
    check("a missing usable_pct is unknown, not 0", arm_window(none, 30.0)["class"] == "unknown")
    check("an invalid arm is unknown regardless of memory",
          arm_window({**clean, "invalid": "only 1 sample"}, 30.0)["class"] == "unknown")
    check("every derived class names its rule", arm_window(low, 30.0)["derived"] == ARM_WINDOW_RULE)
    # measured, not assumed: all six arms of the 2026-09-22 pair carry rc=1 (llama-bench exits
    # non-zero on the first bad shape, the matrix recovers the rest). Folding rc into the class
    # labelled every measured arm unknown and hid the answer.
    check("a non-zero matrix rc is a caveat, not an unknown class",
          arm_window({**clean, "rc": 1}, 30.0)["class"] == "clean")
    check("...and the caveat is printed with the class",
          "rc=1" in arm_window({**clean, "rc": 1}, 30.0)["why"])
    check("an rc=0 arm carries no caveat", "rc=" not in arm_window(clean, 30.0)["why"])

    # a missing pair must fail closed
    rc_stages = stage_defs(Path("/tmp/nope"), k=3, per_miss=0.695, layer_log="", capture="",
                           gguf="", gguf_n_used=None)
    check("the chain always has the four core stages",
          [s["name"] for s in rc_stages][:4] == ["pair", "marginal", "ladder", "accept"])
    check("the subcommand-CLI tool gets its own selftest argv",
          [s["selftest_argv"] for s in rc_stages if s["script"] == "server_window.py"] == [["selftest"]])
    check("...and every other stage keeps --selftest",
          all(s["selftest_argv"] == ["--selftest"] for s in rc_stages
              if s["script"] != "server_window.py"))
    check("the roofline stage only exists when given a capture+gguf",
          "roofline" not in [s["name"] for s in rc_stages])
    withgy = stage_defs(Path("/tmp/nope"), k=3, per_miss=0.695, layer_log="", capture="c.txt",
                        gguf="m.gguf", gguf_n_used=8)
    check("...and exists when it is", "roofline" in [s["name"] for s in withgy])
    check("it does not re-implement n_used on its own",
          "--n-used" in withgy[-2]["argv"] and "8" in withgy[-2]["argv"])

    # every measured object in the artifact must answer "was the box busy?" -- the nested ones too
    withpool = {"rep": 0, "arm": "k1", "tps": 9.0, "rc": 0, "window_before": {"usable_pct": 50.0},
                "pool": {"requests": 10, "hits": 9}}
    mt = summarize_arms([withpool], 30.0)
    check("the aggregate carries a window", mt["window"]["class"] == "clean")
    check("an arm's pool carries its arm's window", mt["arms"][0]["pool"]["window"]["class"] == "clean")
    check("a mixed pair does not get summarised as clean",
          summarize_arms([withpool, {**withpool, "window_before": {"usable_pct": 5.0}}],
                         30.0)["window"]["class"] == "unknown")

    # the renderer must show a refusal INSTEAD of dropping the section, and must escape
    ftests = [{"tool": "x.py", "rc": 0, "status": "ok", "counts": "1/1", "why": ""}]
    doc = {
        "generated_at": "2026-01-01 00:00:00", "min_usable_pct": 30.0,
        "engine_digest": {"libllama.dylib": "deadbeef"},
        "window": {"class": "clean", "why": "quiet (reclaimable 9000MB)", "n_samples": 1,
                   "busy_samples": 0},
        "pair": {"dir": "/tmp/pair", "arms_jsonl": "/tmp/pair/arms.jsonl", "mtime": "t"},
        "measure_time": {"n_records": 1, "arms": [{**clean, "window": arm_window(clean, 30.0)}],
                         "window_classes": {"clean": 1},
                         "window": {"class": "clean", "why": "all 1 arm record(s)"},
                         "nonzero_rc_records": 0,
                         "engine_digests": [{"d": "deadbeef"}], "usable_arms": ["k1"]},
        "stages": [{"name": "ladder", "title": "L", "script": "target25_grid.py", "note": "",
                    "argv": ["py", "grid"], "rc": 1, "status": "refused",
                    "why": "INVALID: needs both a k1 and a k3 arm", "out": "<script>alert(1)</script>",
                    "stderr": ""}],
        "selftests": ftests, "git": {"head": "abc", "branch": "b", "dirty_files": 0},
        "json_path": "/tmp/x.json",
    }
    page = render_html(doc)
    check("a refused stage's reason reaches the html", "needs both a k1 and a k3 arm" in page)
    check("...and its section is NOT dropped", "REFUSED" in page)
    check("tool output is escaped", "<script>alert(1)</script>" not in page
          and "&lt;script&gt;" in page)
    check("the report-time window is labelled as not the measurement window",
          "不是量測當時" in page)
    # `unknown` from an analysis-only run and `unknown` from a busy probe look identical in the
    # class string but must NOT read the same: one is "not asked", the other affects the numbers.
    unk = render_html({**doc, "window": {**doc["window"], "class": "unknown",
                                         "why": "this process never asked the shared probe"}})
    check("an analysis-only 'unknown' is explained, not alarmed",
          "沒有問探針" in unk and "盒況不是乾淨的" not in unk)
    check("the digest brief names llama-bench and counts the artifacts",
          digest_brief({"llama-bench": "aaa", "libggml-metal.0.19.0.dylib": "bbb"})
          == "2 個產物 · llama-bench=aaa · libggml-metal.0.19.0.dylib=bbb")
    check("the digest brief survives a version bump",
          "libllama.0.0.999.dylib" in digest_brief({"libllama.0.0.999.dylib": "c"}))
    check("a measurement from another build says so",
          "不同" in digest_block("t", [{"llama-bench": "other"}], {"llama-bench": "now"}))
    check("the full map is folded away, not omitted",
          "<details>" in digest_block("t", [{"llama-bench": "x"}], {"llama-bench": "x"}))
    check("the printed command drops the machine's interpreter path and is repo-relative",
          cmdline(["/Library/Developer/CommandLineTools/usr/bin/python3",
                   str(ROOT / "scripts" / "check" / "x.py"), "--log", "/tmp/y"])
          == "python3 scripts/check/x.py --log /tmp/y")
    check("...and an absolute argument outside the repo is left alone",
          cmdline(["python3", "x.py", "/tmp/y"]).endswith("/tmp/y"))
    check("display rounding keeps a float readable without turning it into a string",
          fmt(7.3117300000000001, 2) == "7.31" and fmt(None, 2) == "—"
          and fmt("n/a", 2) == "n/a")
    check("a demonstrably busy box IS raised as a warning",
          "盒況不是乾淨的" in render_html({**doc, "window": {**doc["window"],
                                                        "class": "busy-overridden",
                                                        "why": "2 busy sample(s)"}}))
    check("the html is complete", page.startswith("<!DOCTYPE html>") and page.rstrip().endswith("</html>"))

    print(f"selftest: {38 - bad}/38 passed")
    return 0 if bad == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pair", default="", help="existing pair dir with arms.jsonl")
    ap.add_argument("--measure", action="store_true",
                    help="run mtp_k_sweep first (gated); pair defaults to --logdir")
    ap.add_argument("--logdir", default="/tmp/mtp_k_sweep", help="where --measure writes")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--extra-env", default="")
    ap.add_argument("--k", type=int, default=3, help="draft width of the wide arm")
    ap.add_argument("--per-miss-ms", type=float, default=0.695, help="F1's borrowed constant")
    ap.add_argument("--min-usable-pct", type=float, default=30.0)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--need-mb", type=float, default=8000.0)
    ap.add_argument("--measure-timeout", type=int, default=3600)
    ap.add_argument("--layer-log", default="", help="optional: a server log for gap_vs_miss")
    ap.add_argument("--roofline-capture", default="", help="optional: test-backend-ops perf output")
    ap.add_argument("--gguf", default="", help="optional: the carrier, for shape_roofline")
    ap.add_argument("--n-used", type=int, default=None)
    ap.add_argument("--out", default="", help="html path (default docs/TARGET25_REPORT_<ts>.html)")
    ap.add_argument("--no-selftests", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.pair and not args.measure:
        ap.error("give --pair DIR (analyse an existing pair) or --measure (make one)")
    _, rc = build(args)
    return rc


if __name__ == "__main__":
    sys.exit(main())
