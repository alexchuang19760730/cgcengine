#!/usr/bin/env python3
"""超訂預檢：launch 前檢查「固定非池常駐 + expert pool vs Metal working set 上限」。

口徑歷程：
  v1（作廢）：模型檔案全量 13030 + pool 8192 = 21222 vs hw.memsize 16384 ⇒ 雙重計 expert。
  v2（task#37，2026-10-17，已被本版修正）：dense 555 + pool 8192 = 8747 vs hw.memsize 16384 ⇒
      仍誤判 OK，並假設 pool 是檔案 mmap 可回收。
  v3（本版，2026-10-18，依 docs/L4_POOL_ANON_NONRECLAIMABLE_VERDICT_2026-10-18.md）：
      ① 上限改用引擎自報的 recommendedMaxWorkingSetSize（本機實測 11453.25 MiB，
         Backup/metal_working_set.json），不是 hw.memsize 16384 —— 系統/GPU wired 已扣在 ceiling 外。
      ② expert pool 是 **Metal shared 匿名 dirty 頁**（ggml_metal_host_malloc→vm_allocate +
         newBufferWithBytesNoCopy:storageModeShared），pread 寫入後不可被內核回收；evict 後
         madvise 歸還路徑只覆蓋 L3 malloc pool、不覆蓋 L4 metal pool（cgc_in_malloc_pool 守衛）。
         故 pool 全額按「不可回收匿名」計，不是檔案頁。
      ③ 補固定非池開銷：dense + KV(q8_0) + gather slab + compute/graph/prefill stream。
      ④ 分窄/寬工作集：
           narrow（harness 單 prompt，如 C17）：靜態容量門即可，窄集 fill 不滿池。
           wide（常駐 server 連跑多 profile）：池被 churn 全滿、冷專家頁只在 swap，需額外
           預留寬集系統/wired 餘量（wide_headroom，預設 2048 MiB，G6 一手數據校準）。
         8G 寬集實測進程獨佔髒足跡 ≈12.9G（resident 8.1+swapped 4.8，見判詞），撞 ceiling
         ⇒ free 0、coding 退化、tier D；這是結構性、非腳本可修。

用法：
    python3 scripts/check/budget_preflight.py --workload narrow --pool-bytes 8589934592
    python3 scripts/check/budget_preflight.py --workload wide   --pool-bytes 6442450944
exit code：0 = 過；2 = 超訂（拒跑，--warn-only 可降級警告）。

模型「檔案全量」仍印出但僅作上限參考（legacy），不參與判定。
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys

MIB = 1048576

# ── 本機實測默認值（M4 Air 16GB；均可被參數/env 覆寫）──────────────────────────
# Metal recommendedMaxWorkingSetSize（Backup/metal_working_set.json）
DEFAULT_CEILING_MIB = 11453.25
# 非 expert 固定常駐（denseIQ4X 實測，load_mode=none）
DEFAULT_DENSE_MIB = 555
# KV cache：q8_0 @ ctx 8192，僅 10 層帶 K/V（full_attention_interval=4，i=3,7…39），每 token 2KiB
DEFAULT_KV_MIB = 85
# gather slab：kind0=110 + kind1=110 + kind2=136（CGC-GATHER-SLAB cap=256，啟動日誌實測）
DEFAULT_GATHER_MIB = 356
# compute graph / prefill stream / 其餘 Metal 常駐（保守估計，G6 校準）
DEFAULT_COMPUTE_MIB = 400
# 寬工作集（常駐多主題 churn）額外預留：系統/wired 峰值 + 冷頁 swap 緩衝
# 依 8G 寬集實測冷頁 4.55G 中至少需留給系統約 2G 才不壓垮；工程默認、非物理常數，G6 校準。
DEFAULT_WIDE_HEADROOM_MIB = 2048
# 靜態容量安全係數（不把池頂到 ceiling 滿格，留碎片/波動）
SAFETY_FRAC = 0.90


def phys_mem_mib() -> int:
    try:
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                             text=True, timeout=5).stdout.strip()
        return int(out) // MIB
    except Exception:
        return 0


def ceiling_from_artifact() -> float:
    """從 repo Backup/metal_working_set.json 讀引擎自報 ceiling；找不到回默認。"""
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(here, "../../Backup/metal_working_set.json")
    try:
        with open(cand) as f:
            return float(json.load(f)["mb"])
    except Exception:
        return DEFAULT_CEILING_MIB


def model_bytes(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-bytes", type=int,
                    default=int(os.environ.get("CGC_SERVER_EXPERT_CACHE_BYTES",
                                               os.environ.get("CGC_EXPERT_CACHE_BYTES", 8589934592))),
                    help="expert pool 上限 bytes（匿名不可回收），預設 prod-new 8 GiB")
    ap.add_argument("--model", default=None, help="模型檔路徑（僅印檔案全量作上限參考）")
    ap.add_argument("--load-mode", choices=["none", "mmap", "mmap+mlock"], default="none",
                    help="load_mode；mmap 系 => dense 常駐=0（OS 可回收）")
    ap.add_argument("--workload", choices=["narrow", "wide"], default=os.environ.get("CGC_BUDGET_WORKLOAD", "narrow"),
                    help="narrow=harness 單 prompt（預設）；wide=常駐 server 連跑多 profile（churn 冷頁）")
    ap.add_argument("--ceiling-mib", type=float, default=None,
                    help="Metal working set ceiling；預設讀 Backup/metal_working_set.json（11453.25）")
    ap.add_argument("--dense-mib", type=float,
                    default=float(os.environ.get("CGC_DENSE_MIB", DEFAULT_DENSE_MIB)))
    ap.add_argument("--kv-mib", type=float, default=DEFAULT_KV_MIB)
    ap.add_argument("--gather-mib", type=float, default=DEFAULT_GATHER_MIB)
    ap.add_argument("--compute-mib", type=float, default=DEFAULT_COMPUTE_MIB,
                    help="compute graph/prefill stream/其他 Metal 常駐 MiB")
    ap.add_argument("--wide-headroom-mib", type=float, default=DEFAULT_WIDE_HEADROOM_MIB,
                    help="寬工作集額外預留 MiB（系統/wired/冷頁 swap 緩衝），G6 校準")
    ap.add_argument("--warn-only", action="store_true", help="超訂只警告不拒跑（exit 0）")
    args = ap.parse_args()

    model = args.model or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../models/gguf/"
        "Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf")

    b_model = model_bytes(model) // MIB
    b_phys = phys_mem_mib()
    ceiling = args.ceiling_mib if args.ceiling_mib is not None else ceiling_from_artifact()
    usable = ceiling * SAFETY_FRAC

    if args.load_mode in ("mmap", "mmap+mlock"):
        b_dense, dense_kind = 0, "file-backed、OS 可回收（dense 不計匿名）"
    else:
        b_dense, dense_kind = args.dense_mib, "匿名 dirty（dense 常駐）"
    b_kv = args.kv_mib
    b_gather = args.gather_mib
    b_compute = args.compute_mib
    b_fixed = b_dense + b_kv + b_gather + b_compute

    b_pool = args.pool_bytes // MIB
    b_dem_static = b_fixed + b_pool
    b_dem_wide = b_dem_static + args.wide_headroom_mib

    print("=" * 72)
    print(f"[budget] 實體記憶體 hw.memsize      = {b_phys} MiB（僅參考；系統/GPU wired 不可用）")
    print(f"[budget] Metal working set ceiling = {ceiling:.0f} MiB（recommendedMaxWorkingSetSize）")
    print(f"[budget] 可用門檻 ceiling×{SAFETY_FRAC:g}      = {usable:.0f} MiB")
    print(f"[budget] 模型檔案全量              = {b_model} MiB（load_mode={args.load_mode}；非常駐，僅上限參考）")
    print("-" * 72)
    print(f"[budget] 固定非池常駐 = {b_fixed:.0f} MiB")
    print(f"[budget]   dense={b_dense:.0f}（{dense_kind}） + kv={b_kv:.0f} + "
          f"gather_slab={b_gather:.0f} + compute/graph={b_compute:.0f}")
    print(f"[budget] expert pool  = {b_pool} MiB（**匿名 dirty、不可回收**：vm_allocate metal shared）")
    print("-" * 72)

    over = []
    # 門 1：靜態容量門（任何工作集都必須過）
    if b_dem_static > usable:
        over.append(("靜態容量門", b_dem_static,
                     "固定+池 即便窄集也超可用門檻 ⇒ Metal OOM/必然 swap"))
    else:
        print(f"[budget] 靜態容量門（narrow）：需求 {b_dem_static:.0f} ≤ 可用 {usable:.0f}  ✓ 餘 {usable-b_dem_static:.0f} MiB")

    # 門 2：寬工作集門（僅 wide 要求）
    if args.workload == "wide":
        print(f"[budget] 寬工作集門（wide）：需求 {b_dem_static:.0f} + 寬集餘量 {args.wide_headroom_mib:.0f} "
              f"= {b_dem_wide:.0f} MiB vs 可用 {usable:.0f}")
        if b_dem_wide > usable:
            over.append(("寬工作集門", b_dem_wide,
                         "常駐多主題 churn 冷頁（8G 實測獨佔髒足跡≈12.9G、冷頁 swap 4.55G）⇒ "
                         "free 0、coding 退化、tier D；建議縮池到 6G/4G 或開啟 L4 冷頁歸還（見 L4 判詞 C 項）"))

    if over:
        for name, demand, why in over:
            print(f"[budget] ✗ {name} 超訂：需求 {demand:.0f} MiB > 可用 {usable:.0f} MiB（+{demand-usable:.0f}）")
            print(f"         {why}")
        print(f"[budget] 槓桿：CGC_SERVER_EXPERT_CACHE_BYTES（8G→6G={8192-6144}MB／→4G={8192-4096}MB）、"
              f"退 WorkBuddy/Chrome 省 2–3GB、L4 冷頁歸還（根治，C 項）、重開機 purge。")
        if args.warn_only:
            print("[budget] --warn-only：降級為警告，繼續（數字可能活在 swap 上、不可引用）。")
            return 0
        print("[budget] 拒跑。要硬跑用 --warn-only 或 BUDGET_GATE=warn；穩態權威請用 harness 單 prompt（narrow）。")
        return 2

    print(f"[budget] OK：workload={args.workload} 下固定+池在可用門檻內。")
    if args.workload == "narrow" and b_dem_static > usable - 1024:
        print(f"[budget] 注意：窄集靜態僅餘 {usable-b_dem_static:.0f} MiB；若改跑常駐多 profile 請用 --workload wide 復檢。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
