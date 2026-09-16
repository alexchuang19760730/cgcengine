#!/usr/bin/env python3
"""Prove the T1 distillation path works, without a model and without touching traces/.

WHY THIS EXISTS (and what it does NOT claim)
--------------------------------------------
tb_loop already learned this lesson: "已本地驗證（假 prime-agent 捕獲 argv + 假 subprocess）".
The failure it prevents is specific -- a refine script whose prompt lost its scope marker, or
whose id hand-off never made it into the prompt, looks EXACTLY like a working one: it runs, it
exits 0, and it just distils worse. Nothing downstream can tell.

So this test checks the hand-offs, not the model quality:
  * the prompt's first line is `/refine --global` (the global scope is load-bearing)
  * the prompt carries the `[engine]` scope marker
  * the prompt carries the evidence block AND the next-free-id list, and no leftover `{{...}}`
  * the fake agent is invoked with `--offline` and the prompt as its final argument
  * the candidate the fake emits -- built FROM the id the prompt handed over -- is accepted by
    traces/validate.py **together with the real records**, so cross-record integrity is exercised
  * `--accept` appends to a redirected traces dir and stays valid
  * `--dry-run` prints the prompt and writes nothing

It does NOT claim the distillation is any good. Nothing here has ever seen a real model.

Usage: python3 agent_harness/engine_loop/distill/selftest.py
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parent
REPO = ENGINE.parent.parent
SCRIPT = HERE / "refine_engine.sh"
VALIDATE = ENGINE / "traces" / "validate.py"

FAKE = r"""#!/usr/bin/env python3
import json, os, re, sys
argv = sys.argv[1:]
open(os.environ["FAKE_ARGV_OUT"], "w", encoding="utf-8").write(json.dumps(argv, ensure_ascii=False))
prompt = argv[-1]
# Take the id from the MARKED block, which the prompt declares to be the only usable source.
# If the hand-off breaks, this test fails on a collision instead of passing on a lucky free number.
blk = re.search(r"NEXT-FREE-IDS-BEGIN -->(.*?)<!-- NEXT-FREE-IDS-END", prompt, re.S)
if not blk:
    sys.stderr.write("fake prime-agent: prompt carried no marked next-free-id block\n"); sys.exit(3)
m = re.search(r"eng-([a-z]+)-(\d{4})", blk.group(1))
if not m:
    sys.stderr.write("fake prime-agent: marked block carried no id\n"); sys.exit(3)
lid = f"eng-{m.group(1)}-{m.group(2)}"
print("（模型可能先說幾句廢話，這一行必須被解析器忽略）")
print(json.dumps({
    "type": "lesson", "lesson_id": lid, "class": "smoke",
    "rule": "SELFTEST ONLY -- this rule exists to be rejected by review; it must never reach traces/.",
    "because": "It is emitted by a fake prime-agent in distill/selftest.py to prove the pipeline's hand-offs.",
    "counterexample_observed": None,
    "applies_to": ["agent_harness/engine_loop/distill/refine_engine.sh"],
    "superseded_by": None,
}, ensure_ascii=False))
print("```")   # a stray fence must NOT be parsed as a record
print(json.dumps({"type": "something-else", "x": 1}, ensure_ascii=False))
"""


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def main() -> int:
    fails: list[str] = []

    def check(ok: bool, label: str, detail: str = "") -> None:
        print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + (f"   {detail}" if detail and not ok else ""))
        if not ok:
            fails.append(label)

    tmp = Path(tempfile.mkdtemp(prefix="refine_engine_selftest_"))
    try:
        bindir = tmp / "bin"
        bindir.mkdir()
        fake = bindir / "prime-agent"
        fake.write_text(FAKE, encoding="utf-8")
        fake.chmod(0o755)
        argv_out = tmp / "fake_argv.json"
        traces = tmp / "traces"
        shutil.copytree(ENGINE / "traces", traces)          # redirect --accept away from the real one

        env = dict(os.environ)
        env["PATH"] = f"{bindir}:{env.get('PATH', '')}"
        env["FAKE_ARGV_OUT"] = str(argv_out)
        env["REFINE_ENGINE_MODEL"] = "fake/model-for-selftest"
        env["TRACES_DIR"] = str(traces)

        print("=== 1. --dry-run 不呼叫模型、不寫任何東西 ===")
        before = set((HERE / "out").glob("*")) if (HERE / "out").is_dir() else set()
        r = run(["bash", str(SCRIPT), "--dry-run"], env=env)
        after = set((HERE / "out").glob("*")) if (HERE / "out").is_dir() else set()
        check(r.returncode == 0, "--dry-run rc=0", r.stderr[-200:])
        check(before == after, "--dry-run 沒有產生 out/ 目錄")
        check(not argv_out.exists(), "--dry-run 沒有呼叫 prime-agent")
        prompt = r.stdout
        check(prompt.splitlines()[0].strip() == "/refine --global", "prompt 首行是 /refine --global",
              repr(prompt.splitlines()[0][:60]))
        check("[engine]" in prompt, "prompt 帶 [engine] scope 標記")
        check("還沒有任何 decision 引用過的 episode" in prompt, "prompt 帶證據區塊")
        check(not re.search(r"\{\{[A-Z_]+\}\}", prompt), "prompt 沒有殘留 {{...}} 佔位符")
        check(bool(re.search(r"eng-[a-z]+-\d{4}", prompt)), "prompt 帶下一個可用 id")
        check("只輸出 JSONL" in prompt, "prompt 要求 JSONL 輸出")
        check("NEXT-FREE-IDS-BEGIN" in prompt and "NEXT-FREE-IDS-END" in prompt,
              "可用 id 被哨兵包起來（唯一可取的來源）")
        check(prompt.count("NEXT-FREE-IDS-BEGIN") == 1 and prompt.count("NEXT-FREE-IDS-END") == 1,
              "哨兵在整份 prompt 裡只出現一次（指示文字引用它但不得複製字面，否則第一個配對會是錯的）",
              f"begin={prompt.count('NEXT-FREE-IDS-BEGIN')} end={prompt.count('NEXT-FREE-IDS-END')}")
        check("一律不可重用" in prompt, "prompt 明文禁止重用證據區裡的既有 id")

        # 負向對照：證明「抓文件中第一個看到的 id」這個策略**真的會撞號**。
        # 這一條是這個測試最重要的產物——它把一個 prompt 設計缺陷變成可回歸的斷言。
        first_any = re.search(r"eng-[a-z]+-\d{4}", prompt)
        blk = re.search(r"NEXT-FREE-IDS-BEGIN -->(.*?)<!-- NEXT-FREE-IDS-END", prompt, re.S)
        first_free = re.search(r"eng-[a-z]+-\d{4}", blk.group(1)) if blk else None
        check(first_any is not None and first_free is not None and first_any.group(0) != first_free.group(0),
              "文件中第一個出現的 id ≠ 可用 id（所以『抓第一個』會撞號，哨兵是必要的）",
              f"first_any={first_any.group(0) if first_any else None} first_free={first_free.group(0) if first_free else None}")

        print("=== 2. 接上假 prime-agent：prompt 的手遞必須真的到達 argv ===")
        r = run(["bash", str(SCRIPT)], env=env)
        check(r.returncode == 0, "run rc=0", (r.stderr or r.stdout)[-200:])
        check(argv_out.exists(), "prime-agent 被呼叫了")
        if argv_out.exists():
            argv = json.loads(argv_out.read_text(encoding="utf-8"))
            check("--offline" in argv, "argv 帶 --offline")
            check(argv[-1].strip().startswith("/refine --global"), "prompt 是 argv 最後一項且帶 --global")
            check(any(a == "--model" for a in argv), "argv 帶 --model")
            check("fake/model-for-selftest" in argv, "argv 帶入我們指定的模型（腳本沒有自己猜一個）")

        outs = sorted((HERE / "out").glob("*"))
        check(bool(outs), "產生了 out/<ts>/")
        ts_dir = outs[-1] if outs else None
        cand = (ts_dir / "lessons.candidate.jsonl") if ts_dir else None
        check(bool(cand and cand.exists() and cand.stat().st_size > 0), "候選 lesson 檔非空")
        if cand and cand.exists():
            recs = [json.loads(l) for l in cand.read_text(encoding="utf-8").splitlines() if l.strip()]
            check(len(recs) == 1, "只解析出 1 筆（散文與 code fence 與別的 type 被擋掉）", f"got {len(recs)}")
            check(recs and recs[0]["type"] == "lesson", "型別是 lesson")
        rej = (ts_dir / "rejected.txt") if ts_dir else None
        check(bool(rej and rej.stat().st_size > 0), "被拒的行有留下紀錄（不是靜默丟掉）")

        print("=== 3. 候選要與正式 record 一起通過驗證器 ===")
        if cand and cand.exists():
            r = run([sys.executable, str(VALIDATE), *[str(p) for p in sorted(traces.glob("*.jsonl"))],
                     str(cand)])
            check(r.returncode == 0, "validate.py 接受「正式 + 候選」", (r.stdout + r.stderr)[-300:])

        print("=== 4. --accept 追加到被重導的 traces（證明寫入路徑可測）===")
        if ts_dir:
            n_before = len(traces.joinpath("lessons.jsonl").read_text(encoding="utf-8").splitlines())
            r = run(["bash", str(SCRIPT), "--accept", ts_dir.name], env=env)
            check(r.returncode == 0, "--accept rc=0", (r.stderr or r.stdout)[-200:])
            n_after = len(traces.joinpath("lessons.jsonl").read_text(encoding="utf-8").splitlines())
            check(n_after == n_before + 1, f"lessons.jsonl 多了一筆（{n_before} -> {n_after}）")
            r = run([sys.executable, str(VALIDATE), *[str(p) for p in sorted(traces.glob("*.jsonl"))]])
            check(r.returncode == 0, "追加後整份 traces 仍通過驗證",
                  (r.stdout + r.stderr)[-300:])
            real = ENGINE / "traces" / "lessons.jsonl"
            check("SELFTEST ONLY" not in real.read_text(encoding="utf-8"),
                  "真的 traces/lessons.jsonl 沒有被污染")

        print("=== 5. 沒有指定模型時必須拒跑，而不是猜一個 ===")
        env2 = dict(env); env2.pop("REFINE_ENGINE_MODEL", None)
        r = run(["bash", str(SCRIPT)], env=env2)
        check(r.returncode == 2, "未設 REFINE_ENGINE_MODEL -> exit 2", f"rc={r.returncode}")
        check("will not guess" in (r.stderr or ""), "錯誤訊息說明為什麼不猜")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        # 清掉本次 selftest 產生的 out/ 目錄（它們是假的候選）
        out_root = HERE / "out"
        if out_root.is_dir():
            for d in out_root.glob("*"):
                if d.is_dir() and any(d.glob("*.candidate.jsonl")):
                    txt = "".join(p.read_text(encoding="utf-8", errors="replace") for p in d.glob("*"))
                    if "SELFTEST ONLY" in txt:
                        shutil.rmtree(d, ignore_errors=True)
            if not any(out_root.iterdir()):
                out_root.rmdir()

    print()
    if fails:
        print(f"  {len(fails)} FAILED: {fails}")
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
