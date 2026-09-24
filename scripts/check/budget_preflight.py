#!/usr/bin/env python3
"""超訂預檢：launch 前檢查「model resident + expert pool vs 實體記憶體」。

口徑與 scripts/run_server.sh:1049-1062 完全相同：
    _b_res  = model 檔大小（load_mode=mmap 時為 0，OS 可回收；否則匿名頁、不可回收、全算）
    _b_pool = expert pool 上限（BUDGET bytes）
    _b_dem  = _b_res + _b_pool
    _b_phys = 實體記憶體（hw.memsize）
    OVERSUBSCRIBED = _b_dem > _b_phys

預設行為：超訂 → exit 2 拒跑（並提示 STRICT_BUDGET / 減 pool）。
--warn-only：超訂只警告（exit 0）——給「明知超訂仍要跑」的相容模式。

用法：
    python3 scripts/check/budget_preflight.py [--pool-bytes 8589934592] [--model PATH] [--load-mode none|mmap] [--warn-only]
exit code：0 = OK／未超訂；2 = OVERSUBSCRIBED（拒跑）

設計理由（2026-09-24）：
run_server.sh 預設「只警告不拒跑」（避免擋掉 prefill250 既有量測），但量測 runner 不知道
這件事——MTP on + ρ 影子節點在超訂 4838 MiB 下跑出 B 臂全 0.00 t/s，50 分鐘才發現不可引用。
量測腳本必須在 launch 前自己先做這道檢查，否則「跑完才發現環境不可引用」會反覆發生。
"""
from __future__ import annotations
import argparse, os, subprocess, sys

MIB = 1048576

def phys_mem_mib() -> int:
    try:
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                             text=True, timeout=5).stdout.strip()
        return int(out) // MIB
    except Exception:
        return 0

def model_bytes(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-bytes", type=int, default=8589934592,
                    help="expert pool 上限（BUDGET），預設 prod-new 8 GiB")
    ap.add_argument("--model", default=None,
                    help="模型檔路徑；預設 prod-new Nail checkpoint")
    ap.add_argument("--load-mode", choices=["none", "mmap", "mmap+mlock"], default="none",
                    help="load_mode；mmap 系 => model resident=0（OS 可回收）")
    ap.add_argument("--warn-only", action="store_true",
                    help="超訂時只警告不拒跑（exit 0）")
    args = ap.parse_args()

    model = args.model or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../models/gguf/"
        "Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf")

    mb = model_bytes(model)
    if args.load_mode in ("mmap", "mmap+mlock"):
        res, kind = 0, "file-backed、OS 可回收"
    else:
        res, kind = mb, "匿名頁、不可回收"

    b_model = mb // MIB
    b_res = res // MIB
    b_pool = args.pool_bytes // MIB
    b_phys = phys_mem_mib()
    b_dem = b_res + b_pool

    print(f"[budget] model       {b_model} MiB（load_mode={args.load_mode} => {kind}）")
    print(f"[budget] expert pool {b_pool} MiB")
    print(f"[budget] 靜態需求    {b_dem} MiB（model resident {b_res} + pool {b_pool}） vs 實體 {b_phys} MiB")

    if b_phys > 0 and b_dem > b_phys:
        over = b_dem - b_phys
        print(f"[budget] OVERSUBSCRIBED by {over} MiB：此啟動只能在 macOS 記憶體壓縮 + swap 之上執行，"
              f"量測數字不可引用。")
        print(f"[budget] 槓桿：CGC_SERVER_EXPERT_CACHE_BYTES（減 pool）／CGC_SERVER_UBATCH（compute buffer）"
              f"／重開機 + purge 清 swap（swap 只累積不回收）。")
        if args.warn_only:
            print("[budget] --warn-only：降級為警告，繼續。")
            return 0
        print("[budget] 拒跑。要硬跑請用 --warn-only（明知超訂仍要跑），或設 "
              "CGC_SERVER_STRICT_BUDGET=1 讓 run_server.sh 自己擋。")
        return 2
    print("[budget] OK：靜態需求在實體記憶體內。")
    return 0

if __name__ == "__main__":
    sys.exit(main())
