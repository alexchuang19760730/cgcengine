#!/usr/bin/env python3
"""3a (CGC_ZERO_MISS) 的正確性收益：四臂 answer-digest 比對。

為什麼不能沿用 seg_batch_abba.py：那是 llama-bench，不產文字，無法回答「輸出對不對」。
B 臂（CGC_SEG_BATCH 單段提交）沒有 hook ⇒ 沒有 idseq dump，所以只能用**答案本身**
（/completion 的 return_tokens，token id 陣列）當觀測。

臂（全部同 build、同 checkpoint、同 prod-new argv，只有 CGC_* 不同）：
  A0  基線：分段提交、有 fill、無 vmask                ⇒ 視為參考輸出
  A1  A0 + CGC_SLOT_TABLE_GPU + CGC_MISS_MASK + CGC_ZERO_MISS
        ⇒ 有 fill 的情況下，把 miss 的 expert 權重置零。若 decode 穩態 miss=0，
          A1 必須**逐 token 等於 A0**（強判據：3a 無回歸）。
  B0  CGC_SEG_BATCH + CGC_B_SCHEME + CGC_SLOT_TABLE_GPU（無 fill，garbage）
  B1  B0 + CGC_MISS_MASK + CGC_ZERO_MISS（無 fill，但缺失項被置零）

⚠ vmask 建在 `if (cgc_slot_table_gpu)` 內（llama-graph.cpp:2426），所以
  **沒有 CGC_SLOT_TABLE_GPU 的臂上 CGC_ZERO_MISS 是 no-op** —— A1 必須帶它，
  否則「A1 == A0」會是一個從沒生效的開關給出的假綠燈。

判據（由 --verdict 印出）：
  1. A0 reps 之間完全一致 ⇒ 視窗乾淨、解碼確定性成立（否則後面全部不可引用）。
  2. A1 == A0（或重合率 1.0）⇒ 3a 在有 fill 時不破壞輸出。
  3. B1 != B0 ⇒ 3a 在無 fill 臂上確實改變了輸出（不是 no-op）。
  4. overlap(B1,A0) vs overlap(B0,A0) ⇒ 正確性收益的方向。
     ⚠ 若 B 臂 miss=100%（無 fill），vmask 全 0 ⇒ B1 是「全零 MoE 輸出」，
       不是「k−1 項正確和」。腳本會用 BATCHDBG 的 misses/slots 把這件事說清楚，
       不要讓「B1 變了」被讀成「B1 變對了」。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "src" / "llama.cpp" / "build" / "bin" / "llama-server"

# prod-new 的 argv（CGC_DUMP_ENV=1 CGC_SERVER_PROFILE=prod-new scripts/run_server.sh 導出，
# 2026-09-25 00:05）。--temp 0.4 被改成 --temp 0，seed 由 CGC_FORCE_TEMP0 釘死。
ARGV = [
    "-m", "models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf",
    "-ngl", "99", "--load-mode", "none", "-t", "8", "-c", "8192", "-np", "1",
    "--no-kv-unified", "-sps", "0", "--host", "127.0.0.1", "--port", "{port}",
    "--jinja", "-expert-cache", "8589934592", "--cache-type-k", "q8_0",
    "--cache-type-v", "q8_0", "-b", "5632", "-ub", "5632",
    "--reasoning", "off", "--reasoning-format", "none",
    "--temp", "0", "--top-k", "0", "--top-p", "0.8",
    "--repeat-penalty", "1.0", "--dry-multiplier", "1.0",
    "--dry-allowed-length", "10", "--dry-penalty-last-n", "512",
]

BASE_ENV = {
    "CGC_EXPERT_CACHE_BYTES": "8589934592",
    "LLAMA_EXPERT_CACHE_ALLOW_NGL": "1",
    "LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0": "0",
    "LLAMA_EXPERT_CACHE_WORKERS": "8",
    "CGC_WAKE_POLL_US": "15",
    "CGC_EVICTED_RING": "0",
    "CGC_N_CB": "8",
    "CGC_OA_ASYNC": "1",
    "CGC_SERVER_AUTO_ANCHOR": "0",
    "CGC_SERVER_DEFAULT_MARKER_STOPS": "1",
    "CGC_GATHER_SLAB_CAP": "256",
    "CGC_PREFILL_STREAM": "1",
    "CGC_DBUF": "1",
    "CGC_SPAC": "1",
    "CGC_SPAC_ALPHA": "0.75",
    "CGC_SOFT_POOL_L0": "0",
    "CGC_SOFT_POOL_L1": "0",
    "CGC_LOOP_GUARD": "1",
    "CGC_GLU_FUSED_DOWN": "1",
    "CGC_WATCHDOG": "1",
    "CGC_MM_BITIDENT": "1",
    "CGC_FORCE_TEMP0": "1",
    "CGC_EXPERT_SKIP_READRAW": "1",
    "LLAMA_EXPERT_CACHE_BATCH_DBG": "1",
}

S1 = {"CGC_SLOT_TABLE_GPU": "1"}
SEG = {"CGC_SEG_BATCH": "1", "CGC_B_SCHEME": "1", "CGC_SLOT_TABLE_GPU": "1"}
MASK = {"CGC_MISS_MASK": "1", "CGC_MISS_MASK_DBG": "1"}
ZERO = {"CGC_ZERO_MISS": "1"}

ARMS = {
    "A0": {},
    "A1": {**S1, **MASK, **ZERO},
    "B0": dict(SEG),
    "B1": {**SEG, **MASK, **ZERO},
}

PROMPT = ("用條列方式說明快取置換策略的取捨，並比較 LRU 與成本感知淘汰在長序列推論下的差異。"
          "請具體舉例，不要只列名詞。")


def engine_digest() -> dict:
    out = {}
    for p in sorted((ROOT / "src/llama.cpp/build/bin").glob("*.dylib")):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            h.update(f.read(65536))
        out[p.name] = {"size": p.stat().st_size, "sha256_64k": h.hexdigest()[:16]}
    return out


def launch(arm: str, port: int, logdir: Path) -> tuple[subprocess.Popen, Path, float]:
    logdir.mkdir(parents=True, exist_ok=True)
    log = logdir / f"{arm}.log"
    env = dict(os.environ)
    env.update(BASE_ENV)
    env.update(ARMS[arm])
    f = open(log, "wb")
    argv = [BIN.as_posix()] + [a.replace("{port}", str(port)) for a in ARGV]
    p = subprocess.Popen(argv, env=env, cwd=str(ROOT), stdout=f, stderr=f)
    return p, log, time.time()


def wait_health(port: int, proc, timeout: int = 600) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def decode(port: int, n_predict: int, seed: int = 0) -> dict:
    body = json.dumps({
        "prompt": PROMPT, "n_predict": n_predict, "temperature": 0,
        "top_k": 1, "seed": seed, "cache_prompt": False, "stream": False,
        "return_tokens": True,
        # [2026-09-25] garbage 輸出會提前撞 EOS：B 臂實測只吐 8 個 token 就停，8 個 token 不夠
        # 做任何重合率判斷。這是 /completion 的 API 參數，**不改動引擎配置**（prod-new 的
        # 形狀／池／採樣全部照舊），只是叫 sampler 不要因為 EOS 收尾。
        "ignore_eos": True,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())


def stop(proc) -> None:
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
    # run_server.sh:755 —— kill 後 Metal 端 GPU buffer 不是立刻回收，等一拍再起下一支。
    time.sleep(10)


def overlap(a: list, b: list) -> dict:
    """逐位置 token 重合率 + 首次分歧位置。兩個空陣列必須被判成 NO DATA，不是 1.0。"""
    if not a or not b:
        return {"rate": None, "first_diff": None, "n": 0}
    n = min(len(a), len(b))
    if n == 0:
        return {"rate": None, "first_diff": None, "n": 0}
    same = sum(1 for i in range(n) if a[i] == b[i])
    first = next((i for i in range(n) if a[i] != b[i]), None)
    return {"rate": same / n, "first_diff": first, "n": n}


def prefix_profile(a: list, b: list, cuts=(4, 8, 16, 32, 64)) -> dict:
    """分段重合率。

    hot prewarm（llama-context.cpp:1972）是在 hook 之外、context 層填的，所以 **B 臂也會跑**：
    池在第一步有一批 expert 是真的在裡面的，之後無 fill ⇒ 越往後 miss 越多。因此「3a 有沒有
    正確性收益」的正確形態不是一個全域數字，而是一條**遞減曲線**：前綴高、尾部掉。
    只報全域重合率會把「前 8 個 token 對、後面全錯」和「全程均勻地錯一半」混成同一個數。
    """
    if not a or not b:
        return {}
    out = {}
    for c in cuts:
        n = min(c, len(a), len(b))
        if n == 0:
            continue
        out[str(n)] = round(sum(1 for i in range(n) if a[i] == b[i]) / n, 4)
    return out


def miss_stats(text: str) -> dict:
    """從 BATCHDBG 行抽 misses/slots。misses/slots == 1.0 表示全 miss（無 fill）。"""
    ms = [int(m) for m in re.findall(r"BATCHDBG layer=\d+ misses=(\d+)", text)]
    # prewarm 在 hook 之外，是 B 臂唯一真會填池的來源；它的 hit/miss 決定 vmask 有多少個 1。
    pw = re.findall(r"prewarm req=(\d+) hit=(\d+) miss=(\d+)", text)
    return {"batch_dbg_lines": len(ms), "misses_sum": sum(ms) if ms else None,
            "misses_max": max(ms) if ms else None,
            "prewarm": {"req": int(pw[-1][0]), "hit": int(pw[-1][1]), "miss": int(pw[-1][2]),
                        "hit_rate": round(int(pw[-1][1]) / int(pw[-1][0]), 4)} if pw else None}


def run_arm(arm: str, args) -> dict:
    proc, log, t0 = launch(arm, args.port, Path(args.workdir))
    try:
        if not wait_health(args.port, proc, args.health_timeout):
            return {"arm": arm, "error": "server never became healthy", "log": str(log)}
        outs = [decode(args.port, args.n_predict) for _ in range(args.reps)]
        ids = [o.get("tokens") or [] for o in outs]
        texts = [o.get("content") or "" for o in outs]
        txt = log.read_text(errors="replace")
        # [2026-09-25 修正] 實體 log 行是 `CGC-ZEROMISS: applied=%d skipped=%d`（**連字符**，
        # llama-graph.cpp:2777），不是 ZERO_MISS。第一版寫成底線 ⇒ 匹配不到 ⇒ applied 假報 None，
        # 會被讀成「3a 沒生效」，其實是正則沒對上。
        # [2026-09-25 再改] 修好一次性 dump 後格式變成 `CGC-ZEROMISS[<n>]: applied=%d skipped=%d`
        # （每滿 40 層印一次）。取**最後一筆**：第一筆是第一次建圖（常是 reserve，vmask 未建）
        # ⇒ 假報 applied=0；後面的筆數才是真實的 apply 狀態。
        applied = re.findall(r"CGC-ZEROMISS\[\d+\]: applied=(\d+) skipped=(\d+)", txt)
        return {
            "arm": arm, "env": ARMS[arm], "reps": args.reps,
            "ids": ids, "texts": texts,
            "n_tokens": [len(x) for x in ids],
            "within_arm_identical": all(ids[0] == x for x in ids) and all(texts[0] == t for t in texts),
            "compared_by": "ids" if all(len(x) >= 8 for x in ids) else "NO_DATA",
            "zero_miss_applied": applied[-1] if applied else None,
            "miss_stats": miss_stats(txt),
            "log": str(log),
        }
    finally:
        stop(proc)


def verdict(res: dict) -> str:
    a0, a1, b0, b1 = (res.get(k) for k in ("A0", "A1", "B0", "B1"))
    L = []
    if not a0 or a0.get("error"):
        return "A0 沒起來 ⇒ 整輪不可引用（視窗/載入問題，先查 log）"
    if a0.get("compared_by") != "ids":
        L.append("⚠ A0 沒拿到 token id 陣列 ⇒ 只能比文字，證據等級下降")
    L.append(f"A0 臂內一致：{a0.get('within_arm_identical')}（ false ⇒ 解碼不確定，後面全不可引用）")
    for name, r in (("A1", a1), ("B0", b0), ("B1", b1)):
        if not r or r.get("error"):
            L.append(f"{name}：{r.get('error') if r else '未跑'}")
            continue
        ov = overlap(a0["ids"][0], r["ids"][0]) if a0.get("ids") else {"rate": None}
        pf = prefix_profile(a0["ids"][0], r["ids"][0]) if a0.get("ids") else {}
        L.append(f"{name} vs A0 重合率={ov['rate']} 首分歧={ov['first_diff']} 前綴剖面={pf}")
        L.append(f"    applied/skipped={r.get('zero_miss_applied')} miss={r.get('miss_stats')}")
    if b0 and b1 and not b0.get("error") and not b1.get("error"):
        same = b0["ids"][0] == b1["ids"][0]
        L.append(f"B1 == B0：{same}（True ⇒ 3a 是 no-op，白做）")
    return "\n".join(L)


def self_test() -> int:
    ok = 0

    def chk(name, cond):
        nonlocal ok
        ok += 1 if cond else 0
        print(f"  [{'ok' if cond else 'FAIL'}] {name}")

    chk("四臂都定義了", set(ARMS) == {"A0", "A1", "B0", "B1"})
    chk("A1 帶 SLOT_TABLE_GPU（否則 ZERO_MISS 是 no-op）",
        ARMS["A1"].get("CGC_SLOT_TABLE_GPU") == "1")
    chk("B1 = B0 + mask + zero",
        all(ARMS["B1"].get(k) == v for k, v in ARMS["B0"].items())
        and ARMS["B1"].get("CGC_ZERO_MISS") == "1"
        and ARMS["B1"].get("CGC_MISS_MASK") == "1")
    chk("A0 是乾淨基線（沒有任何 S1/seg/mask/zero）", ARMS["A0"] == {})
    chk("BASE_ENV 帶 ALLOW_NGL（不帶則 expert cache 整個不啟用）",
        BASE_ENV.get("LLAMA_EXPERT_CACHE_ALLOW_NGL") == "1")
    chk("BASE_ENV 帶 P0 SKIP_READRAW（不帶則 prefill OOM）",
        BASE_ENV.get("CGC_EXPERT_SKIP_READRAW") == "1")
    chk("temp 0（確定性）", ARGV[ARGV.index("--temp") + 1] == "0")
    chk("port 佔位符存在", any("{port}" in a for a in ARGV))
    chk("overlap: 空陣列回 None 不是 1.0", overlap([], [])["rate"] is None)
    chk("overlap: 全同 = 1.0", overlap([1, 2, 3], [1, 2, 3])["rate"] == 1.0)
    chk("overlap: 首分歧位置正確", overlap([1, 2, 3], [1, 9, 3])["first_diff"] == 1)
    chk("overlap: 長度不同取 min", overlap([1, 2, 3], [1, 2])["n"] == 2)
    chk("miss_stats 抽得到 misses", miss_stats("BATCHDBG layer=3 misses=7")["misses_sum"] == 7)
    chk("miss_stats 無資料回 None", miss_stats("")["misses_sum"] is None)
    chk("binary 存在（不跑就不該存在）", BIN.exists())
    print(f"selftest {ok}/15" if ok == 15 else f"selftest {ok}/15")
    return 0 if ok == 15 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A0,A1,B0,B1")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--health-timeout", type=int, default=600)
    ap.add_argument("--workdir", default="Backup/zeromiss_quality")
    ap.add_argument("--json", default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()

    res: dict = {"engine_digest": engine_digest(), "n_predict": args.n_predict,
                 "reps": args.reps, "prompt": PROMPT[:20] + "..."}
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm!r}")
        print(f"[{arm}] launching ...", flush=True)
        res[arm] = run_arm(arm, args)
        print(f"[{arm}] done n_tokens={res[arm].get('n_tokens')}", flush=True)
    out = Path(args.json) if args.json else Path(args.workdir) / "zeromiss_quality.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # [2026-09-25] 分批跑（先 --arms A0，再 --arms A1,B0,B1）時直接寫會**蓋掉上一批的 ids**
    # —— token id 只存在 json 裡（server log 沒有），掉了就必須重跑一次 13 GB 載入才能拿回來。
    # 所以先把舊檔裡本次沒跑的臂併回來。
    if out.exists():
        try:
            prev = json.loads(out.read_text())
            for k, v in prev.items():
                if k not in res:
                    res[k] = v
        except Exception as e:
            print(f"[warn] 無法合併既有 {out}: {e}", file=sys.stderr)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print("\n=== VERDICT ===")
    print(verdict(res))
    print(f"\njson: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
