#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
miss_rate_series.py — 把 MISSMASK / BATCHDBG 行还原成「每一步的稳态 miss 率」时间序列。

为什么单独一支工具：
  gate 判据是「单段提交 + 背景 fill 臂的 *稳态* miss 率」，而「稳态」只能从时间序列看：
  · 第 1 步几乎必然是全 miss（冷池），混进去会把均值拖高；
  · 背景 fill 若真的有在填，曲线会往下走并收敛；若没有（= 零 fill），曲线是平的。
  所以本工具不做「总体均值」这一个数，而是给分段统计 + 收敛判据。

行格式（两种都可以，可同时喂进来做交叉比對）：
  MISSMASK il=<l> step=<s> ntok=<n> nsel=<k> misses=<m> exps: <e> <e> ...
  BATCHDBG layer=<l> misses=<m> slots: e<e>->s<d> ...

⚠ 切 step 不能用单调性：miss=0 的 step 一行都不印（见 skill 第 64 条）。
  本工具用「显式 step 字段」（MISSMASK 有）或「按 layer 回绕」来切；
  BATCHDBG 没有 step 字段，只能用 layer 序列回绕，因此对 BATCHDBG 只输出
  「每层的 miss 分布」，不输出 per-step 序列（除非 --layer-wrap 明确要求）。

用法：
  --self-test                     跑内置自测
  --log F [--log F2 ...]          解析真实 log
  --warmup N                      跳过前 N 个 step（默认 1，因为第 1 步是冷池）
  --bins N                        把剩下的 step 切成 N 段报均值（默认 3）
  --json PATH                     输出 json
"""
import argparse
import json
import re
import sys

# ntok 是可选的：单段提交臂印的是 `MISSMASK il=1 step=4 nsel=8 misses=4 exps: ...`
# （没有 ntok），带 hook 的臂会带上。写成非捕获可选组，其余组号不动。
RE_MISSMASK = re.compile(
    r"^MISSMASK il=(\d+) step=(\d+) (?:ntok=(\d+) )?nsel=(\d+) misses=(\d+)"
    r"(?: exps:(.*))?$")
RE_BATCHDBG = re.compile(
    r"^BATCHDBG layer=(\d+) misses=(\d+) slots:(.*)$")


def parse_text(text):
    """-> {"steps": [...], "layers": {il: [misses...]}, "src": {mask:n, batch:n}}

    steps 只由 MISSMASK 构成（它有显式 step 字段）。BATCHDBG 进 layers。
    """
    steps = []          # [{"step": int, "il": int, "k": int, "m": int, "exps": [..]}]
    layers = {}         # il -> [misses per observation]
    src = {"mask": 0, "batch": 0}
    for ln in text.splitlines():
        ln = ln.strip()
        m = RE_MISSMASK.match(ln)
        if m:
            src["mask"] += 1
            exps = [int(x) for x in (m.group(6) or "").split()]
            steps.append({"step": int(m.group(2)), "il": int(m.group(1)),
                          "ntok": int(m.group(3)) if m.group(3) else None,
                          "k": int(m.group(4)),
                          "m": int(m.group(5)), "exps": exps})
            continue
        m = RE_BATCHDBG.match(ln)
        if m:
            src["batch"] += 1
            il = int(m.group(1))
            layers.setdefault(il, []).append(int(m.group(2)))
    return {"steps": steps, "layers": layers, "src": src}


def per_step(steps):
    """把 per-(step, il) 观测聚成 per-step：k 总和 / miss 总和 / 出现的层数。"""
    by = {}
    for s in steps:
        d = by.setdefault(s["step"], {"k": 0, "m": 0, "nl": 0})
        d["k"] += s["k"]
        d["m"] += s["m"]
        d["nl"] += 1
    out = []
    for st in sorted(by):
        d = by[st]
        out.append({"step": st, "req": d["k"], "miss": d["m"], "nl": d["nl"],
                    "rate": (d["m"] / d["k"]) if d["k"] else 0.0})
    return out


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def analyse(parsed, warmup=1, bins=3):
    """-> dict：分段统计 + 收敛判据。"""
    rows = per_step(parsed["steps"])
    if not rows:
        return {"ok": False, "reason": "no MISSMASK rows", "src": parsed["src"]}
    body = rows[warmup:]
    if not body:
        return {"ok": False, "reason": "all steps swallowed by warmup",
                "n_steps": len(rows), "src": parsed["src"]}
    n = len(body)
    edge = [0]
    for i in range(1, bins):
        edge.append(n * i // bins)
    edge.append(n)
    segs = []
    for i in range(bins):
        sl = body[edge[i]:edge[i + 1]]
        if not sl:
            continue
        segs.append({"bin": i, "n": len(sl),
                     "miss_per_step": round(_mean([r["miss"] for r in sl]), 2),
                     "req_per_step": round(_mean([r["req"] for r in sl]), 1),
                     "rate": round(_mean([r["rate"] for r in sl]), 4),
                     "layers_per_step": round(_mean([r["nl"] for r in sl]), 2)})
    first = segs[0]["rate"]
    last = segs[-1]["rate"]
    # 收敛判据：后段相对前段下降比例。>0.3 视为「有在填」；<0.1 视为「平的 = 零 fill」。
    decay = (first - last) / first if first > 0 else 0.0
    if decay >= 0.30:
        conv = "DECAYING"
    elif decay <= 0.10:
        conv = "FLAT"
    else:
        conv = "MIXED"
    return {"ok": True, "n_steps_total": len(rows), "n_steps_used": n,
            "warmup": warmup, "src": parsed["src"],
            "step0": {"req": rows[0]["req"], "miss": rows[0]["miss"],
                      "rate": round(rows[0]["rate"], 4)},
            "bins": segs, "decay_rel": round(decay, 3), "converge": conv,
            "steady_rate": last, "steady_miss_per_step": segs[-1]["miss_per_step"],
            "steady_req_per_step": segs[-1]["req_per_step"]}


# ------------------------------- self-test --------------------------------
def _t(steps):
    return steps


def self_test():
    ok = 0
    bad = []

    def chk(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            bad.append(name)

    # 1. 解析一条 MISSMASK
    p = parse_text("MISSMASK il=2 step=6 ntok=1 nsel=8 misses=3 exps: 18 202 234\n")
    chk("parse one", p["src"]["mask"] == 1 and p["steps"][0]["m"] == 3
        and p["steps"][0]["exps"] == [18, 202, 234])
    # 2. 无 exps 也解析
    p = parse_text("MISSMASK il=3 step=6 ntok=1 nsel=8 misses=0\n")
    chk("no exps", p["steps"][0]["m"] == 0 and p["steps"][0]["exps"] == [])
    # 3. BATCHDBG 进 layers 不进 steps
    p = parse_text("BATCHDBG layer=2 misses=3 slots: e18->s43 e202->s9\n")
    chk("batchdbg", p["src"]["batch"] == 1 and p["steps"] == []
        and p["layers"][2] == [3])

    # 4. 合成 30 步：每步 39 层 × 8，前 10 步 60% miss，后 20 步 10% miss（DECAYING）
    txt = []
    for st in range(30):
        for il in range(39):
            m = 5 if st < 10 else 1
            if m:
                txt.append(f"MISSMASK il={il} step={st} ntok=1 nsel=8 misses={m} exps: 1 2 3 4 5")
    a = analyse(parse_text("\n".join(txt)), warmup=1, bins=3)
    chk("decaying bins", a["ok"] and len(a["bins"]) == 3)
    chk("decaying flag", a["converge"] == "DECAYING")
    chk("decaying steady", abs(a["steady_rate"] - 0.125) < 0.01)

    # 5. 全程平的一串（FLAT）
    txt = []
    for st in range(30):
        for il in range(39):
            txt.append(f"MISSMASK il={il} step={st} ntok=1 nsel=8 misses=5 exps: 1 2 3 4 5")
    a = analyse(parse_text("\n".join(txt)), warmup=1, bins=3)
    chk("flat flag", a["converge"] == "FLAT")
    chk("flat rate", abs(a["steady_rate"] - 0.625) < 1e-6)
    chk("flat miss/step", a["steady_miss_per_step"] == 195.0)

    # 6. miss=0 的 step 不产生行 ⇒ 不能靠单调性切；这里 step 号显式，不应合并
    txt = ["MISSMASK il=0 step=0 ntok=1 nsel=8 misses=8 exps: 1",
           "MISSMASK il=1 step=5 ntok=1 nsel=8 misses=2 exps: 1"]
    a = analyse(parse_text("\n".join(txt)), warmup=0, bins=2)
    chk("gap kept", a["n_steps_total"] == 2)

    # 7. warmup 把第 1 步（冷池全 miss）排掉
    txt = ["MISSMASK il=%d step=0 ntok=1 nsel=8 misses=8 exps: 1" % il for il in range(39)]
    txt += ["MISSMASK il=%d step=1 ntok=1 nsel=8 misses=1 exps: 1" % il for il in range(39)]
    a0 = analyse(parse_text("\n".join(txt)), warmup=0, bins=2)
    a1 = analyse(parse_text("\n".join(txt)), warmup=1, bins=2)
    chk("warmup drops step0", a0["n_steps_total"] == 2 and a1["n_steps_used"] == 1)
    chk("warmup rate", abs(a1["steady_rate"] - 0.125) < 1e-6)

    # 8. 空输入不炸
    a = analyse(parse_text(""), warmup=1, bins=3)
    chk("empty ok=False", a["ok"] is False)

    # 9. bins 数 > step 数不炸
    a = analyse(parse_text("MISSMASK il=0 step=0 ntok=1 nsel=8 misses=4 exps: 1\n"
                           "MISSMASK il=0 step=1 ntok=1 nsel=8 misses=4 exps: 1\n"),
                warmup=0, bins=5)
    chk("bins>steps", a["ok"] and 0 < len(a["bins"]) <= 2)

    # 10. req 统计：39 层 × 8 = 312/step
    txt = ["MISSMASK il=%d step=3 ntok=1 nsel=8 misses=2 exps: 1" % il for il in range(39)]
    a = analyse(parse_text("\n".join(txt)), warmup=0, bins=1)
    chk("req per step", a["steady_req_per_step"] == 312.0)

    print(f"[self-test] {ok}/{ok+len(bad)} passed")
    for b in bad:
        print(f"  FAIL: {b}")
    return len(bad) == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--log", action="append", default=[])
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--bins", type=int, default=3)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    if a.self_test:
        sys.exit(0 if self_test() else 1)
    if not a.log:
        ap.error("需要 --log 或 --self-test")

    merged = {"steps": [], "layers": {}, "src": {"mask": 0, "batch": 0}}
    for f in a.log:
        p = parse_text(open(f, errors="replace").read())
        merged["steps"].extend(p["steps"])
        for k, v in p["layers"].items():
            merged["layers"].setdefault(k, []).extend(v)
        for k in merged["src"]:
            merged["src"][k] += p["src"][k]
    res = analyse(merged, warmup=a.warmup, bins=a.bins)
    print(json.dumps(res, indent=1, ensure_ascii=False))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    return 0 if res.get("ok") else 2


if __name__ == "__main__":
    sys.exit(main())
