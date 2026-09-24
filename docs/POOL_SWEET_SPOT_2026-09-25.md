# Pool 甜點曲線定案：8GiB → 4GiB（prod-new 預設）

日期：2026-09-25
狀態：**定案並落地**（run_server.sh prod-new BUDGET 8589934592 → 4294967296）
依據：同場量測（重開機後乾淨窗口，build 1ca685490，prod-new 口徑，P0=CGC_EXPERT_SKIP_READRAW=1）

## 1. 甜點曲線（單臂，thermal 全程 NOMINAL）

| pool | decode | prefill | hit | run 內 swap growth |
|---|---:|---:|---:|---:|
| 8G | 12.48 | 288.79 | 93.1% | +596 MiB |
| 6G | 13.12 | 285.50 | 93.3% | +105 MiB |
| 5G | 12.64 | 283.77 | 92.7% | +346 MiB |
| **4G** | **13.11** | 274.10 | 93.0% | **−3 MiB（≈0）** |

## 2. 同場 ABBA 確認（4G vs 8G，A B A B 交錯）

| 對 | 臂 A（4G） | 臂 B（8G） | 4G/8G |
|---|---|---|---|
| 1 | 12.47 | 12.56 | 0.993 |
| 2 | 12.79 | ~~4.04~~（contention，外來進程） | 無效 |
| 3 | 12.32 | 12.06 | 1.022 |

**中位比率 ≈ +0.7%**（兩對有效，±2% 噪音內）→ **decode 持平**。
「4G = 8G 的 0.79×」舊結論是**跨環境污染**：8G 名義容量在 16GB 機上實際駐留僅 ~6.2G
（resident 6197 MiB），多出來的容量被記憶體壓力逼去 swap、從未被用上。

## 3. 為什麼 hit 幾乎不降

路由重尾：143/256 專家吃掉 98.8% 訪問（CGC_MASSCOV 實測，K=143 → 98.9%）。
4G pool 就裝得下 93% 的每步命中 → pool 4-8G 的 hit 差異僅 92.7-93.3%（2.7pp 內）。

## 4. 結構收益

- run 內 swap growth：8G +596 MiB → **4G ≈ 0** → swap 從結構消失 → 量測可比
- 生產 profile 記憶體預算：21.6 GiB（模型 13 + pool 8）→ 17.6 GiB（13 + 4）
  —— 結構性超訂（4838 MiB）顯著緩解
- wired 峰值壓力下降（4G 臂大多 <100 MiB growth，8G 臂有 611）

## 5. 決策

- **prod-new 預設 BUDGET = 4294967296（4GiB）**；顯式 CGC_SERVER_EXPERT_CACHE_BYTES 仍可覆寫
- 6G 亦為甜點（decode 同級 13.12、prefill 略高 285 vs 274）——若後續 prefill 權重上升可再評估 6G
- 5G 是局部低點（12.64），不採

## 6. 產物

- /tmp/harness_first.json（8G 基線 12.36/280.73）
- /tmp/seg_mtp_abba_result.json（A1 12.48 / B1 13.10，單段×MTP 判定：單流 25 無路）
- /tmp/pool_6g_result.json、/tmp/pool_45g_result.json（曲線）
- /tmp/pool_4g8g_abba.json（同場 ABBA）

---

## 7. 覆盤（2026-09-25 晚間）— 4G 方案撤銷

**override 語法 bug 推翻先前全部「4G≈8G」結論**：
- harness 正確的 pool override 是 `CGC_SERVER_EXPERT_CACHE_BYTES`（run_server.sh 守門變數，內部導出 CGC_EXPERT_CACHE_BYTES）。
- 先前 ABBA / 甜點曲線誤用無前綴的 `CGC_EXPERT_CACHE_BYTES` → 被守門忽略 → **兩臂其實都是 8G** →「+0.7% 持平」「4G swap≈0」全部是 8G vs 8G 的假結論。

**同窗口真曲線（2026-09-25 00:18-00:28，override 正確）**：

| pool | hit | cap% | decode | swap growth |
|---|---:|---:|---:|---:|
| 4GiB | 85.1% | 32.1% | **7.67** | ≈0 |
| 6GiB | ~89% | ~15% | **10.48** | ≈0 |
| 8GiB | 92.8% | 3.8% | **12.06** | +1529 MiB |

**結論**：
1. 熱集（143/256 專家 = 98.8% 路由）需要 8GiB 才裝得下 → hit 93% → 12 t/s。4G/6G 都裝不下 → capacity miss 32%/15% → decode 掉 34%/13%。
2. **4G 的慢是穩定態容量不足，不是冷啟動**——warm-skip 預熱救不了（8.39 ≈ 8.46、7.91 ≈ 7.95 ≈ 7.67）。
3. **C 方案（4G + 預熱）撤銷**；prod-new 回 8GiB。
4. 「測試 arm 不到 10 vs 原本 12.57」的根源 = 真 4G（<10）vs 8G（12+），不是環境漂移。
5. swap 結構解回 L1-L4（docs/SWAP_STRUCTURAL_FIX_2026-09-24.md），L2 單一駐留優先——不靠縮 pool。

**教訓固化**：profile override 一律用 `CGC_SERVER_*` 前綴；任何 ABBA 前先驗證 `env` resolve 後實際值（harness base_check 的 env dump）。
