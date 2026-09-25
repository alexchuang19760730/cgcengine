#!/usr/bin/env python3
"""arm_report_html.py — 把 arm_two_pass.py 的 result.json 渲染成一份自包含 HTML 報告。

報告內容（固定結構）：
  * 總覽：時間 / 機器 / shape / build 指紋 / 總 gate 狀態；
  * 每個 arm：G1/G2/G3 gate 細節、clean vs instrumented 成績對比（含儀器開銷）、
    污染標記、thermal/swap 前後快照、cache 統計、完整 resolved env、保留的 log 路徑；
  * 複現命令。

設計：純內聯 CSS、無外部依賴、無 JS 也能完整閱讀（靜態可讀優先）。
用法：
  python3 scripts/check/arm_report_html.py --json /tmp/armrun/result.json \
      --out /tmp/armrun/report.html
  python3 scripts/check/arm_report_html.py --selftest
"""
from __future__ import annotations

import argparse
import html as _html
import json
import sys
from pathlib import Path


def esc(x) -> str:
    return _html.escape("" if x is None else str(x))


# ─────────────────────────────────────────────────────────────────────────────
# 從 matrix 產物取成績 / cache
# ─────────────────────────────────────────────────────────────────────────────
def _rows(matrix):
    if not matrix:
        return []
    out = []
    for arm in matrix:
        out.extend(arm.get("rows", []))
    return out


def metric(pass_res: dict, kind: str):
    """回傳 (avg, std, n_batch)；kind: pp/tg。"""
    for r in _rows((pass_res or {}).get("matrix")):
        is_pp = r.get("n_prompt", 0) > 0
        if (kind == "pp") == is_pp:
            return r.get("avg_ts"), r.get("stddev_ts"), r.get("n_batch")
    return None, None, None


def cache_stats(pass_res: dict):
    if not pass_res or not pass_res.get("matrix"):
        return {}
    for arm in pass_res["matrix"]:
        c = arm.get("cache")
        if c:
            return c
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# 小元件
# ─────────────────────────────────────────────────────────────────────────────
def badge(pass_: bool, ok="PASS", no="FAIL") -> str:
    cls = "b-ok" if pass_ else "b-no"
    return f'<span class="badge {cls}">{esc(ok if pass_ else no)}</span>'


def kv_table(rows, kcol="項", vcol="值") -> str:
    body = "".join(
        f"<tr><td class=k>{esc(k)}</td><td>{esc(v)}</td></tr>" for k, v in rows)
    return (f'<table class=t><thead><tr><th>{esc(kcol)}</th><th>{esc(vcol)}</th></tr>'
            f"</thead><tbody>{body}</tbody></table>")


def env_table(env: dict) -> str:
    if not env:
        return '<p class=muted>（resolve 無 env）</p>'
    rows = sorted(env.items())
    body = "".join(f"<tr><td class=k>{esc(k)}</td><td>{esc(v)}</td></tr>" for k, v in rows)
    return (f'<table class=t env><thead><tr><th>env key</th><th>value（resolve 實際值）</th></tr>'
            f"</thead><tbody>{body}</tbody></table>")


def delta_pct(clean, inst) -> str | None:
    try:
        d = (float(inst) - float(clean)) / float(clean) * 100.0
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    cls = "b-no" if d > 1.0 else ("b-ok" if d < -1.0 else "b-warn")
    return f'<span class="badge {cls}">{d:+.1f}%</span>'


# ─────────────────────────────────────────────────────────────────────────────
# gate 區塊
# ─────────────────────────────────────────────────────────────────────────────
def render_gates(gates: dict) -> str:
    blocks = []

    # G1
    g1 = gates.get("G1", {})
    rows = "".join(
        f"<tr><td class=k>{esc(c.get('check'))}</td>"
        f"<td>{badge(c.get('pass'))}</td><td>{esc(c.get('detail'))}</td></tr>"
        for c in g1.get("checks", []))
    blocks.append(
        f'<div class="gate"><h4>G1 環境潔淨 {badge(g1.get("pass"))}</h4>'
        f'<table class=t><thead><tr><th>檢查</th><th>結果</th><th>細節</th></tr></thead>'
        f'<tbody>{rows}</tbody></table></div>')

    # G2
    g2 = gates.get("G2", {})
    fails = "".join(f"<li class=no>{esc(x)}</li>" for x in g2.get("failures", []))
    devs = "".join(f"<li class=warn>{esc(x)}</li>" for x in g2.get("deviations", []))
    scalars = g2.get("resolved_scalars", {})
    sc_rows = "".join(
        f"<tr><td class=k>{esc(k)}</td><td>{esc(v)}</td></tr>"
        for k, v in sorted(scalars.items()))
    blocks.append(
        f'<div class="gate"><h4>G2 生產參數釘對 {badge(g2.get("pass"))}</h4>'
        f'{("<ul>"+fails+devs+"</ul>") if (fails or devs) else ""}'
        f'<details><summary>resolved scalars</summary>'
        f'<table class=t><tbody>{sc_rows}</tbody></table></details></div>')

    # G3
    g3 = gates.get("G3", {})
    leak = "".join(f"<li class=no>{esc(x)}</li>" for x in g3.get("non_instrument_diff", []))
    eff = esc(", ".join(g3.get("instruments_effective_in_resolve", [])) or "（無）")
    notvis = esc(", ".join(g3.get("instruments_not_visible_in_resolve", [])) or "（無）")
    blocks.append(
        f'<div class="gate"><h4>G3 兩輪等價（差異僅限儀器）{badge(g3.get("pass"))}</h4>'
        f'{("<ul>"+leak+"</ul>") if leak else ""}'
        f'<p class=muted>儀器在 resolve 可見：{eff}<br>resolve 不可見（不代表沒生效，需查 log）：{notvis}</p>'
        f'</div>')

    return '<div class="gates">' + "".join(blocks) + "</div>"


# ─────────────────────────────────────────────────────────────────────────────
# 一輪的成績 + 快照
# ─────────────────────────────────────────────────────────────────────────────
def pass_metric_table(clean: dict, inst: dict) -> str:
    head = ("<thead><tr><th>指標</th><th>clean（無儀器‧可引用）</th>"
            "<th>instrumented（有儀器）</th><th>儀器開銷</th></tr></thead>")
    body_rows = []
    for kind, label in (("pp", "prefill"), ("tg", "decode")):
        ca, cs, cb = metric(clean, kind)
        ia, is_, ib = metric(inst, kind)
        ctxt = f"{ca:.2f} ± {cs:.2f}" if ca is not None else "—"
        itxt = f"{ia:.2f} ± {is_:.2f}" if ia is not None else "—"
        d = delta_pct(ca, ia) or "—"
        body_rows.append(
            f"<tr><td class=k>{label} t/s</td><td>{ctxt}</td><td>{itxt}</td><td>{d}</td></tr>")
    return f'<table class=t>{head}<tbody>{"".join(body_rows)}</tbody></table>'


def snapshot_lines(pass_res: dict) -> str:
    if not pass_res:
        return ""
    b, a = pass_res.get("sys_before"), pass_res.get("sys_after")

    def line(tag, s):
        if not s:
            return f"<tr><td class=k>{tag}</td><td>—</td></tr>"
        th = (s.get("thermal") or {}).get("label")
        return (f"<tr><td class=k>{tag}</td><td>thermal={esc(th)} · "
                f"swap={esc(s.get('swap_used_mb'))} MiB · "
                f"pageins={esc(s.get('pageins'))} pageouts={esc(s.get('pageouts'))}<br>"
                f"<span class=muted>{esc(s.get('memory_pressure'))}</span></td></tr>")

    c = cache_stats(pass_res)
    crow = ""
    if c:
        crow = (f"<tr><td class=k>cache</td><td>hit={esc(c.get('hit_rate_pct'))}% · "
                f"misses={esc(c.get('misses'))} · evictions={esc(c.get('evictions'))} · "
                f"cap%={esc(c.get('capacity_pct'))}</td></tr>")
    logs = pass_res.get("kept_logs", [])
    lrow = ""
    if logs:
        items = "".join(f"<li>{esc(p)}</li>" for p in logs)
        lrow = f'<tr><td class=k>保留 log</td><td><ul class=logs>{items}</ul></td></tr>'
    rcrow = (f"<tr><td class=k>rc / wall</td><td>rc={esc(pass_res.get('rc'))} · "
             f"wall={esc(pass_res.get('wall_s'))}s</td></tr>")
    return (f'<table class=t><tbody>{line("before", b)}{line("after", a)}{crow}{rcrow}{lrow}'
            f"</tbody></table>")


def pollution_banner(pol: dict) -> str:
    if not pol or not pol.get("polluted"):
        return ""
    why = []
    if pol.get("thermal_heavy"):
        why.append("thermal 出現 HEAVY")
    why.append(f"swap growth = {pol.get('swap_growth_mb'):+.0f} MiB（>500 即污染）")
    return ('<div class="pollute">⚠ 污染讀數（測試卡 §5.1）：' + esc("；".join(why)) +
            " — 僅供診斷，不可進錨點 / commit 標題</div>")


# ─────────────────────────────────────────────────────────────────────────────
# 單個 arm
# ─────────────────────────────────────────────────────────────────────────────
def render_arm(arm: dict) -> str:
    head = (f'<h3>{esc(arm.get("arm"))} '
            f'<span class=muted>（profile={esc(arm.get("profile"))}）</span></h3>')
    if arm.get("user_env"):
        head += f'<p class=muted>arm 增量 env：{esc(arm.get("user_env"))}</p>'

    if not arm.get("ran"):
        return (f'<section class=arm>{head}'
                f'<div class="blocked">✗ 生產級 gate 未過、本臂未執行（未消耗 GPU）。</div>'
                f"{render_gates(arm.get('gates', {}))}</section>")

    clean, inst = arm.get("clean"), arm.get("instrumented")
    pol_c, pol_i = arm.get("clean_pollution"), arm.get("instrumented_pollution")

    # 完整 resolved env（用 clean 輪 G2 的、= 生產 env；含偏差）
    resolved_env = arm.get("gates", {}).get("G2", {}).get("resolved_env", {})

    parts = [head, render_gates(arm.get("gates", {})),
             '<h4>成績對比（同形狀、兩輪只差儀器）</h4>',
             pass_metric_table(clean, inst),
             pollution_banner(pol_c),
             '<h4>clean 輪（真實速度）</h4>', snapshot_lines(clean),
             pollution_banner(pol_i),
             '<h4>instrumented 輪（全儀器）</h4>', snapshot_lines(inst),
             '<details class=envdetails><summary>完整 resolved env（'
             f'{len(resolved_env)} keys）</summary>{env_table(resolved_env)}</details>']
    return '<section class=arm>' + "".join(parts) + "</section>"


# ─────────────────────────────────────────────────────────────────────────────
# 整份報告
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
:root{--ink:#1f2933;--mut:#64748b;--line:#d9e0e6;--bg:#f5f7f9;--ok:#1a7f37;
--no:#c0392b;--warn:#b7791f;--card:#fff;}
*{box-sizing:border-box;}
body{margin:0;padding:0;background:var(--bg);color:var(--ink);
 font:14px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;}
.wrap{max-width:1080px;margin:0 auto;padding:28px 22px 60px;}
h1{font-size:24px;margin:0 0 4px;} h2{font-size:18px;margin:30px 0 10px;}
h3{font-size:16px;margin:26px 0 6px;padding-top:14px;border-top:2px solid var(--line);}
h4{font-size:14px;margin:18px 0 7px;}
.muted{color:var(--mut);font-weight:400;}
.lead{color:var(--mut);margin:0 0 18px;}
table.t{border-collapse:collapse;width:100%;background:var(--card);margin:6px 0 12px;
 font-size:13px;}
table.t th,table.t td{border:1px solid var(--line);padding:6px 9px;text-align:left;
 vertical-align:top;}
table.t th{background:#eef2f5;}
td.k{white-space:nowrap;font-weight:600;background:#fafbfc;}
table.env td:nth-child(2){font-family:ui-monospace,Menlo,monospace;}
.badge{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;
 font-weight:700;color:#fff;}
.b-ok{background:var(--ok);} .b-no{background:var(--no);} .b-warn{background:var(--warn);}
.gates{display:grid;grid-template-columns:1fr;gap:10px;margin:8px 0;}
.gate{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 12px;}
.gate h4{margin:4px 0 8px;}
.arm{background:var(--card);border:1px solid var(--line);border-radius:10px;
 padding:6px 18px 18px;margin:18px 0;}
.pollute{background:#fdecea;border:1px solid #e6a39c;color:#8a2a20;
 border-radius:7px;padding:8px 12px;margin:8px 0;font-weight:600;}
.blocked{background:#fdecea;border:1px solid #e6a39c;color:#8a2a20;border-radius:7px;
 padding:10px 12px;margin:10px 0;font-weight:600;}
li.no{color:var(--no);} li.warn{color:var(--warn);}
ul.logs{margin:4px 0;padding-left:18px;font-family:ui-monospace,Menlo,monospace;
 font-size:12px;}
summary{cursor:pointer;font-weight:600;}
details.envdetails{margin-top:10px;}
code{background:#eef2f5;padding:1px 5px;border-radius:4px;
 font-family:ui-monospace,Menlo,monospace;}
"""


def render(result: dict) -> str:
    meta = result.get("meta", {})
    arms = result.get("arms", [])

    overall_ok = (all(
        (a.get("gates", {}).get("G1", {}).get("pass") and
         a.get("gates", {}).get("G2", {}).get("pass") and
         a.get("gates", {}).get("G3", {}).get("pass"))
        for a in arms) and all(a.get("ran", True) for a in arms))
    any_polluted = any(
        (a.get("clean_pollution", {}) or {}).get("polluted") or
        (a.get("instrumented_pollution", {}) or {}).get("polluted")
        for a in arms if a.get("ran"))

    overall = badge(overall_ok, "ALL GATES PASS", "GATE FAILURE")
    polflag = (' <span class="badge b-no">含污染讀數</span>' if any_polluted else "")

    fp = meta.get("build_fingerprint", {})
    fp_rows = [(k, v) for k, v in fp.items()]
    shape = meta.get("shape", {})
    shape_rows = [(k, shape.get(k)) for k in
                  ("prompt", "gen", "depths", "reps", "warm_skip", "ctx_size")]

    head = (
        f'<h1>Arm 雙輪驗證報告 {overall}{polflag}</h1>'
        f'<p class=lead>每臂強制 clean（無儀器）+ instrumented（有儀器），兩輪只差儀器；'
        f'生產級 gate fail-closed。</p>'
        f'{kv_table([("generated_at", meta.get("generated_at")),
                     ("machine", meta.get("machine")),
                     ("macos", meta.get("macos")),
                     ("allow_dirty", meta.get("allow_dirty"))], "項", "值")}'
        f'<h2>量測 shape</h2>{kv_table(shape_rows)}'
        f'<h2>build 指紋（實際 digest）</h2>{kv_table(fp_rows)}')

    body = '<h2>各 arm 結果</h2>' + "".join(render_arm(a) for a in arms)

    repro = meta.get("reproduce", "")
    foot = (f'<h2>複現命令</h2><p><code>{esc(repro)}</code></p>'
            '<p class=muted>解讀規則：對外引用只用 clean 輪成績；instrumented 僅供診斷；'
            '污染/未跑臂不可進 commit 標題或錨點。</p>')

    return (
        '<html style="margin:0;padding:0;"><head><meta charset="utf-8">'
        '<title>Arm 雙輪驗證報告</title>'
        f'<style>{CSS}</style></head><body><div class="wrap">'
        f'{head}{body}{foot}</div></body></html>')


# ─────────────────────────────────────────────────────────────────────────────
# selftest
# ─────────────────────────────────────────────────────────────────────────────
def selftest() -> int:
    fails = []

    def expect(name, cond):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    fake = {
        "meta": {"generated_at": "2026-09-25 09:00", "machine": "arm64 / x",
                 "macos": "15.0", "shape": {"prompt": 2048, "gen": 128, "depths": "512",
                                            "reps": 1, "warm_skip": 64, "ctx_size": 0},
                 "build_fingerprint": {"llama-server": "abc123"},
                 "reproduce": "python3 arm_two_pass.py --arm prod-new"},
        "arms": [{
            "arm": "prod-new", "profile": "prod-new", "user_env": {}, "ran": True,
            "gates": {
                "G1": {"pass": True, "checks": [{"check": "swap_level", "pass": True,
                                                 "detail": "swap=0"}]},
                "G2": {"pass": True, "failures": [], "deviations": [],
                       "resolved_env": {"CGC_SPAC": "1"},
                       "resolved_scalars": {"BATCH": "5632"}},
                "G3": {"pass": True, "non_instrument_diff": [],
                        "instruments_effective_in_resolve": ["CGC_GPU_TIMING"],
                        "instruments_not_visible_in_resolve": []}},
            "clean": {"rc": 0, "wall_s": 100,
                      "sys_before": {"thermal": {"label": "NOMINAL"}, "swap_used_mb": 0},
                      "sys_after": {"thermal": {"label": "NOMINAL"}, "swap_used_mb": 100},
                      "matrix": [{"rows": [
                          {"n_prompt": 2048, "avg_ts": 300.0, "stddev_ts": 5.0},
                          {"n_prompt": 0, "avg_ts": 12.0, "stddev_ts": 0.2}],
                          "cache": {"hit_rate_pct": 96, "misses": 10, "evictions": 9}}],
                      "kept_logs": ["x.stderr.log"]},
            "instrumented": {"rc": 0, "wall_s": 110,
                             "sys_before": {"thermal": {"label": "NOMINAL"},
                                            "swap_used_mb": 100},
                             "sys_after": {"thermal": {"label": "NOMINAL"},
                                           "swap_used_mb": 150},
                             "matrix": [{"rows": [
                                 {"n_prompt": 2048, "avg_ts": 295.0, "stddev_ts": 5.0},
                                 {"n_prompt": 0, "avg_ts": 11.5, "stddev_ts": 0.2}],
                                 "cache": {"hit_rate_pct": 96}}],
                             "kept_logs": []},
            "clean_pollution": {"polluted": False},
            "instrumented_pollution": {"polluted": False}}]}

    out = render(fake)
    expect("has html title", "<title>Arm 雙輪驗證報告</title>" in out)
    expect("renders prefill clean 300", "300.00" in out)
    expect("renders decode clean 12", "12.00" in out)
    expect("shows ALL GATES PASS", "ALL GATES PASS" in out)
    expect("escapes unsafe", "<script>" not in out)

    # blocked arm
    fake["arms"][0]["ran"] = False
    out2 = render(fake)
    expect("blocked arm message", "本臂未執行" in out2)
    expect("blocked => GATE FAILURE", "GATE FAILURE" in out2)

    print(f"\n  selftest: {6 - len(fails)}/6 groups pass")
    return 1 if fails else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", dest="json_path")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if not args.json_path or not args.out:
        ap.error("--json and --out are required (unless --selftest)")
    result = json.loads(Path(args.json_path).read_text())
    Path(args.out).write_text(render(result))
    print(f"HTML 報告已寫入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
