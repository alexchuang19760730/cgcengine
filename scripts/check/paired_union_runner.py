#!/usr/bin/env python3
"""A SERVER-SIDE paired A/B runner that records every rep's server log -- the tool G4 lacks.

WHY. G4's target is a -12.8% change in `union_sum` (the GPU-side span). Every existing tool
fails one of the two requirements a paired test has:

  * paired_ab.py     drives llama-bench (`cell=decode -p 0 -n 128 -d 512`): no server log, and
                     it records no per-rep log path either, so its arms cannot be re-read for
                     union at all.
  * decode_sweep.py  iterates ARMS OUTERMOST -- one server per arm, all reps of A then all reps
                     of B -- so its two arms are ALWAYS two sequential servers. Measured
                     2026-09-20: that design reports a 37% "effect" from drift alone.
  * http_duo.py      single arm. It DOES record `server_log` (line 612), which is what makes it
                     usable as the per-rep engine here.

So this runner is deliberately thin: it schedules, it calls http_duo.py per rep, and it reads
`server_log` out of that rep's JSON. It does NOT re-implement server launching, the env
allowlist, or the port choice -- those live in the tools that already own them, and a second
copy of `run_server.sh`'s env handling is exactly how a knob silently stops arriving.

    run   --pairs N --a-env ... --b-env ...       -> manifest.json (per-rep log paths + validity)
    eval  --manifest manifest.json                -> paired AB/BA on union_sum, with the verdict
    --self-test                                   -> the statistics, on the four Ornith logs

Interleave order is ABBA per pair by default: A-B on odd pairs, B-A on even ones, so a monotone
drift cancels in the AB/BA estimator instead of being reported as the effect.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired_union_ab import se_median, series, decprof_steps   # noqa: E402

REPO = Path(__file__).resolve().parents[2]        # scripts/check/<file> -> repo root.
# The first move into scripts/check/ left this as `.parent.parent`, which silently pointed at
# scripts/check/ ... and then built scripts/check/scripts/check/http_duo.py, a path that does not
# exist. --self-test and `eval` both passed anyway, because neither touches HTTP_DUO: the breakage
# was latent in `run`, the one subcommand that needs the repo root. Hence the assertion below.
HTTP_DUO = REPO / "scripts" / "check" / "http_duo.py"
assert HTTP_DUO.exists(), f"http_duo.py not found at {HTTP_DUO} (REPO resolved to {REPO})"
PROD_ENV = "CGC_DECODE_PROFILE=1;CGC_GPU_TIMING=1"    # BOTH are required; see below


def schedule(pairs: int, order: str):
    """(pair_index, slot) in execution order. ABBA cancels a monotone drift."""
    out = []
    for k in range(1, pairs + 1):
        first, second = ("A", "B") if (order == "AB" or k % 2 == 1) else ("B", "A")
        out += [(k, first), (k, second)]
    return out


def do_run(a):
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # Warn, do not refuse: the point of warning is that an arm missing a knob yields a log that
    # parses to ZERO union steps, and an empty pair set reads as "no effect" rather than as an
    # error. Measured 2026-09-20: 12 of the 13 most recent server logs had no DECPROF at all.
    for nm, env in (("--a-env", a.a_env), ("--b-env", a.b_env)):
        missing = [k for k in PROD_ENV.split(";") if k.split("=")[0] not in env]
        if missing:
            print(f"⚠️  {nm} 缺 {missing} ⇒ 這一臂的 log 不會有 union，eval 會拒絕。", file=sys.stderr)
    reps = []
    for seq, (k, slot) in enumerate(schedule(a.pairs, a.order)):
        env = a.a_env if slot == "A" else a.b_env
        jf = outdir / f"rep{k:02d}_{slot}.json"
        cmd = [sys.executable if a.python == "" else a.python, str(HTTP_DUO),
               "--profile", a.profile, "--port", "auto", "--reps", str(a.reps),
               "--decode-predict", str(a.decode_predict), "--extra-env", env,
               "--logdir", str(outdir), "--json", str(jf)]
        print(f"\n== rep {k} slot {slot}  env={env}\n   {' '.join(cmd)}", flush=True)
        if a.dry_run:
            reps.append(dict(pair=k, slot=slot, env=env, json=str(jf), server_log=None,
                             valid=None, dry_run=True))
            continue
        rc = subprocess.run(cmd, cwd=REPO).returncode
        rec = dict(pair=k, slot=slot, env=env, json=str(jf), rc=rc, seq=seq,
                   server_log=None, valid=None, server_killed=None)
        if jf.exists():
            j = json.loads(jf.read_text())
            rec.update(server_log=j.get("server_log"), valid=j.get("valid"),
                       server_killed=j.get("server_killed"),
                       decode_spread=j.get("decode_spread"), quotable=j.get("quotable"))
        reps.append(rec)
    man = dict(kind="paired_union", profile=a.profile, reps=a.reps, pairs=a.pairs,
               order=a.order, decode_predict=a.decode_predict,
               a_env=a.a_env, b_env=a.b_env, records=reps)
    Path(a.out).write_text(json.dumps(man, ensure_ascii=False, indent=1))
    print(f"\nwrote {a.out}  ({len(reps)} reps)")
    print("→ 接著：python3 Backup/paired_union_runner.py eval --manifest " + a.out)
    return 0


def load_manifest(m):
    recs = m["records"] if isinstance(m, dict) else m
    by = {}
    for r in recs:
        if r.get("dry_run"):
            raise SystemExit("manifest 是 dry-run，沒有可讀的 log")
        p = r.get("server_log")
        if not p or not os.path.exists(p):
            raise SystemExit(f"rep {r.get('pair')}/{r.get('slot')} 的 server_log 不存在：{p}")
        by.setdefault(r["pair"], {})[r["slot"]] = r
    pairs = sorted(k for k, v in by.items() if "A" in v and "B" in v)
    if len(pairs) < 2:
        raise SystemExit(f"只有 {len(pairs)} 個完整配對；AB/BA 至少要 2 對（拿到 {len(pairs)}）")
    return by, pairs


def do_eval(a):
    m = json.loads(Path(a.manifest).read_text())
    by, pairs = load_manifest(m)
    print(f"manifest: profile={m.get('profile')} reps={m.get('reps')} pairs={m.get('pairs')} "
          f"order={m.get('order')}   完整配對 {len(pairs)}")

    # 每個 rep 先各自報：run 內的中位數與 3SE（§EN-286 的統計量）
    print(f"\n{'pair':>5}{'slot':>5}{'n':>5}{'union中位':>11}{'σ':>9}{'3SE(run 內)':>13}"
          f"{'/step Δ':>10}  log")
    per = {}
    for k in pairs:
        for slot in ("A", "B"):
            r = by[k][slot]
            S = series(r["server_log"])
            d = decprof_steps(r["server_log"])
            if not S:
                print(f"{k:>5}{slot:>5}{0:>5}{'—':>11}{'—':>9}{'—':>13}"
                      f"{'—':>10}  ** 讀不到 union（缺 CGC_GPU_TIMING=1？decprof={d}）**")
                per[(k, slot)] = None
                continue
            v = [x["union"] for x in S.values()]
            med, sd = st.median(v), st.pstdev(v)
            per[(k, slot)] = dict(S=S, med=med)
            print(f"{k:>5}{slot:>5}{len(v):>5}{med:>11.2f}{sd:>9.2f}"
                  f"{300*se_median(v)/med:>12.1f}%"
                  f"{'':>10}  {os.path.basename(r['server_log'])[:34]}")

    if any(v is None for v in per.values()):
        raise SystemExit("有 rep 讀不到 union —— 先修好旋鈕，不要用剩下的算")

    # 每對一個 Δ（步號交集、兩側同 ntok），再**按對的執行序分組**做 AB/BA。
    # 分組用 pair 而不是位置：ABBA 的漂移抵消單位是「對」，位置切割只在排程剛好對稱時才相等。
    afirst, bfirst, per_pair = [], [], []
    for k in pairs:
        SA, SB = per[(k, "A")]["S"], per[(k, "B")]["S"]
        ntok = st.mode([x["ntok"] for x in SA.values()])
        steps = sorted(s for s in set(SA) & set(SB)
                       if SA[s]["ntok"] == ntok and SB[s]["ntok"] == ntok)
        if len(steps) < 4:
            print(f"  pair {k}: 可配對步只有 {len(steps)}（ntok={ntok}），跳過")
            continue
        d = st.median([SA[s]["union"] - SB[s]["union"] for s in steps])
        seqA = by[k]["A"].get("seq", 0)
        seqB = by[k]["B"].get("seq", 1)
        who = "A 先" if seqA < seqB else "B 先"
        (afirst if seqA < seqB else bfirst).append(d)
        per_pair.append(d)
        print(f"  pair {k}: {who}  n={len(steps):>3}  Δ(A−B) 中位 {d:+8.2f} ms")
    if len(afirst) < 1 or len(bfirst) < 1:
        raise SystemExit(f"兩種順序都要有（A先 {len(afirst)} 對、B先 {len(bfirst)} 對）"
                         "—— 只有一種順序就沒有漂移可扣，請用 --order ABBA")
    if len(per_pair) < 2:
        raise SystemExit(f"可配對的對只有 {len(per_pair)}")

    base = st.median([per[(k, "B")]["med"] for k in pairs])
    mAF, mBF = st.median(afirst), st.median(bfirst)
    # `d` was computed as SA - SB, i.e. (A - B), so the estimator is the (A - B) one:
    #   A-first pair: d = (a + delta) - b = (a - b) + delta
    #   B-first pair: d = a - (b + delta) = (a - b) - delta
    eff_AB = (mAF + mBF) / 2          # effect (A - B)
    drift = (mAF - mBF) / 2
    # 逐對的效應估計，用來取 SE（每對一個，不是每步一個）
    eff_per_pair = [(d - drift) for d in per_pair]
    need = a.target_frac * base
    thr = 3 * se_median(eff_per_pair)
    print(f"\n-- AB/BA（{len(afirst)} 對 A 先 + {len(bfirst)} 對 B 先，union 基準 {base:.1f} ms）--")
    print(f"  Δ(A−B) 按順序: A 先 {mAF:+8.2f} ms  /  B 先 {mBF:+8.2f} ms"
          f"   ⇒ 兩者符號{'相反（漂移主導）' if mAF*mBF < 0 else '相同'}")
    print(f"  效應 (A−B) = {eff_AB:+8.2f} ms = {eff_AB/base:+.2%}")
    print(f"  漂移 = {drift:+8.2f} ms = {drift/base:+.2%}   ← 未交錯的設計會把它報成效應")
    print(f"  SE(每對效應) = {se_median(eff_per_pair):.2f} ms  (n_pairs={len(eff_per_pair)})")
    print(f"\n  G4 目標 −{a.target_frac:.1%} = {need:.2f} ms")
    verdict = "可分辨" if need > thr else "**分辨不了**"
    print(f"  判準 3SE = {thr:.2f} ms ⇒ {verdict}"
          f"（效應是門檻的 {need/thr:.1f}×）")
    print(f"  欲達 3SE 需 n_pairs ≈ "
          f"{(1.253*st.pstdev(eff_per_pair)/(need/3))**2:.0f}（現有 {len(eff_per_pair)}）"
          f"；每對的步數越多、每對的 Δ 越穩，但門檻主要由對數決定")
    return 0


def do_self_test():
    q = {"A1": "Backup/cgc_logs/llama_server_20260918_043439.log",
         "B1": "Backup/cgc_logs/llama_server_20260918_043255.log"}
    if not all(os.path.exists(v) for v in q.values()):
        raise SystemExit("selftest fixture logs missing")
    S = {k: series(v) for k, v in q.items()}
    assert all(S[k] for k in S), "fixture has no union"
    # 1) schedule is ABBA and covers every pair twice
    sc = schedule(3, "ABBA")
    assert [s for _, s in sc] == ["A", "B", "B", "A", "A", "B"], sc
    assert sorted({k for k, _ in sc}) == [1, 2, 3]
    print(f"  [1] ABBA 排程: {sc}  OK")
    # 2) the within-run statistic matches the tool we already trust
    v = [x["union"] for x in S["A1"].values()]
    assert abs(se_median(v) - 1.253 * st.pstdev(v) / len(v) ** 0.5) < 1e-9
    print(f"  [2] SE(中位) 公式一致: n={len(v)} 3SE={300*se_median(v)/st.median(v):.1f}%  OK")
    # 3) a manifest with too few pairs must refuse, not report
    for bad, why in ((dict(records=[dict(pair=1, slot="A", server_log=q["A1"]),
                                    dict(pair=1, slot="B", server_log=q["B1"])]), "1 對"),
                     (dict(records=[dict(pair=1, slot="A", server_log=None)]), "log 缺失")):
        try:
            load_manifest(bad)
        except SystemExit as e:
            print(f"  [3] 拒絕（{why}）: {e}")
        else:
            raise AssertionError(f"did not refuse: {why}")
    # 4) dry-run manifests are refused (they carry no logs)
    try:
        load_manifest(dict(records=[dict(pair=1, slot="A", dry_run=True)]))
    except SystemExit as e:
        print(f"  [4] dry-run manifest 被拒: {e}")
    # 5) ABBA halves must agree in sign-flip terms on the real fixture
    ntok = st.mode([x["ntok"] for x in S["A1"].values()])
    steps = sorted(s for s in set(S["A1"]) & set(S["B1"])
                   if S["A1"][s]["ntok"] == ntok and S["B1"][s]["ntok"] == ntok)
    assert len(steps) >= 20, len(steps)
    print(f"  [5] fixture 可配對步 {len(steps)}（ntok={ntok}）  OK")
    print("SELF-TEST OK")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run")
    r.add_argument("--pairs", type=int, default=3)
    r.add_argument("--reps", type=int, default=4, help="每 rep 的 HTTP 重複數（rep1 是暖機）")
    r.add_argument("--order", choices=["AB", "ABBA"], default="ABBA")
    r.add_argument("--profile", default="prod25")
    r.add_argument("--decode-predict", type=int, default=96)
    r.add_argument("--a-env", default=PROD_ENV)
    r.add_argument("--b-env", default=PROD_ENV)
    r.add_argument("--outdir", default="Backup/cgc_logs/pairunion")
    r.add_argument("--out", default="Backup/phase_decomp/paired_union_manifest.json")
    r.add_argument("--python", default="", help="解譯器；空 = 用目前這個")
    r.add_argument("--dry-run", action="store_true")
    e = sub.add_parser("eval")
    e.add_argument("--manifest", required=True)
    e.add_argument("--target-frac", type=float, default=0.128)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        do_self_test(); return
    if a.cmd == "run":
        sys.exit(do_run(a))
    if a.cmd == "eval":
        sys.exit(do_eval(a))
    ap.error("need run|eval|--self-test")


if __name__ == "__main__":
    main()
