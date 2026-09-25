# 重開機後的重測包：P0/P1/P2 歸因 ＋ S1（2026-09-25 存檔）

這份是重開機前的存檔點與重跑指令。**它不是新結論**——本輪所有數字都落在污染盒況裡，只能當
診斷價；這份檔的作用是讓重開機後的乾淨窗口能一次把兩個問題問清楚，而不是再重踩一次同樣的
協定缺陷。

## 0. 存檔點（重開機不會改變的東西）

| 項 | 值 | 出處 |
|---|---|---|
| branch / HEAD | `demo/sweet-spot-windows-fix` / `356d6c298`（本輪；前一個 `d242af39e`） | `git log -1` |
| engine digest | `2ac1cbb9cfcb130a`（`libllama-common.0.dylib` → `…0.0.578.dylib`） | `md5 -q src/llama.cpp/build/bin/libllama-common.0.dylib` |
| 原始碼↔binary | 同步（本輪 **無 llama 原始碼變更**） | pre-commit 檢查 8 PASS |
| 重開機前的盒況 | swap **7704 / 9216 MiB**、thermal NOMINAL、0 個 llama 行程 | `sysctl -n vm.swapusage`、`thermal_pressure.py` |
| 產物位置 | `Backup/` 在 `.gitignore:396` ⇒ **產物與 driver 只在磁碟**，見 §5 的 md5 | — |

**重開機後第一件事**：`sysctl -n vm.swapusage` 必須回到 ≈0。本輪八臂起跑前是 3849–6004 MiB，
S1 那些格是 7.9–8.0 GiB（E4 起跑 `8129.12` MiB、E0 `8180` MiB）——這正是它們只能當診斷價的原因。

## 1. 為什麼要重跑（不是為了看一次新數字）

`MEASUREMENT_CONTRACT` §5.1 的污染門：swap 增量 > 500 MiB 的 run 不得當可引用值。本輪：

- **P0/P1/P2**：八臂 worst thermal 全 `HEAVY`、起跑 swap 3849–6004 MiB。
- **S1**：E0/E4 兩輪起跑 swap 7.7–8.2 GiB、worst thermal HEAVY／MODERATE。

而更關鍵的是**方向在同一 cell 上 40 分鐘內翻轉**：pass1（19:46，無 `--warm-skip`）`p012` 比
`ctrl2` 慢 −13.7%；pass2（19:54，`--warm-skip 64`＋冷卻）同一對變成 **+6.9%**。同 build、同
cell、同 thermal 標籤 ⇒ **自變數是 session，不是開關**。這也是為什麼重跑必須在同一個 session
內帶自己的對照臂。

## 2. 跑之前的閘門（不成立就別跑，產物會再一次不可引用）

1. `sysctl -n vm.swapusage` 的 `used` **≤ 2048 MiB**。
2. `python3 scripts/check/thermal_pressure.py`：`NOMINAL`。
3. 零個 `llama-server`／`llama-bench`、零個 peer driver。
4. 每臂之間 **420 s 冷卻**（兩支 driver 都已內建，見 §3/§4）。
5. 產物自證欄位必須在：`warm_skip_applied`（`gen − n_gen`，用來抓「名義有 skip、引擎沒生效」
   那 10 筆同型缺陷）、`fixed_fill_seed`、`engine_build`、`box_gate`。

小盒子注意：這台是 16 KB page，用 4096 解 `vm_stat` 會把可用記憶體看成 4 倍小。

## 3. 線 1：P0/P1/P2 歸因（驅動已重寫，自足）

```bash
python3 Backup/mtpoff_base/p0_vs_madv_iso.py --passes 2 --cool 420 --workdir /tmp/p0iso
# 產物：/tmp/p0iso/summary.json（每臂即時寫入，見 res.arms[]）
```

**四臂**——因為 `run_server.sh` 現在**預設就武裝**這三個支柱，所以「prod-new 原樣」＝處理組，
控制臂必須顯式關閉（`!KEY=0` 是 harness 認可的宣告式關閉，會進 `base_check.overrides`）：

| tag | 臂規格 | 意義 |
|---|---|---|
| `armed` | `prod-new` | 交付預設（P0 ＋ P1/P2 ＋ B_SCHEME） |
| `p0only` | `prod-new:!CGC_POOL_MADVISE=0` | armed 減 P1/P2 |
| `m2only` | `prod-new:!CGC_EXPERT_SKIP_READRAW=0` | armed 減 P0 |
| `alloff` | `prod-new:!CGC_EXPERT_SKIP_READRAW=0;!CGC_POOL_MADVISE=0;!CGC_B_SCHEME=0` | 11.2–11.6 那個舊形狀 |

所有臂都走 `harness bench`（唯一允許的起跑口；直跑 `llama_bench_matrix`／`prod_matrix` 的數字
口徑不明）。`--passes 2` 會把臂序輪替一次，讓「啟動序」從中位數裡消掉。

**判定（跑前凍結，已用 7 個合成案例驗過，含本輪真的踩到的兩種失敗）**：每個 pct 是
`100×(armed − arm)/armed`，**負 = 該臂比 armed 慢**；兩個「移除臂」對照這一對的兩個端點（0 與
`alloff` 的 pct）：

| 觀測 | 判定 |
|---|---|
| \|pct(alloff)\| ≤ 5% | 這個結構組合**在本 session 無可測效應** |
| `p0only` ≈ alloff 的 pct、`m2only` ≈ 0 | 效應是 **P1/P2** 的 |
| `m2only` ≈ alloff 的 pct、`p0only` ≈ 0 | 效應是 **P0** 的 |
| 兩個都在中間 | **兩者都貢獻**（兩個 pct 就是份額） |
| 兩者都等於整個效應 | 兩者在這裡**不可分離**（不指認單一開關） |
| 缺 `armed`（或 `alloff`） | **PRECONDITION FAILED**：跨 session 的參考臂答不了這題 |
| 任一臂 rep 散度 >12% | 標 `mixed_regime` ⇒ 該臂不可比（契約 §3.3b） |

**要一起讀的可否證指紋**（兩開關的預測不同，這是分開它們的第二條路）：

- `reads` 幾乎不動（本輪八臂 81966–83277，散佈 1.2%）⇒ 兩者都**不改變「讀多少」**，只改變
  「頁／副本留在哪」。所以若速度差重現，它不會是「讀更多」造成的。
- `pread_usec` 上升而 reads 不變 ⇒ P1/P2 的指紋（頁被 `DONTNEED` 後回磁碟）。
- `reads` 略升（白皮書 ctrl 43857 → p0 44589，+1.7%）⇒ P0 的指紋。

**上一次（本輪）的讀數，僅供對照，不可引用**：

| 臂 | warm_skip | tg | 樣本 | swap growth |
|---|---:|---:|---|---:|
| pass2 `ctrl`（熱啟動，無冷卻） | 64 | 7.75 | 8.46/8.23/**6.57** | +476 |
| pass2 `p012` | 64 | 10.17 | 10.80/10.39/9.31 | +1106 |
| pass2 `ctrl2` | 64 | 9.27 | 7.97/9.22/10.62 | +1723 |
| iso `m2only` | 64 | 9.21 | 6.26/10.43/10.95 | +1957 |
| iso `p0only` | 64 | **10.35** | 9.95/10.20/10.90 | +1850 |

⇒ 本輪**分不出來**：指定的判準前提（焊接臂要慢）在權威 cell 上失敗，白皮書「P0 單獨 −25%」
降級成單一 session 的觀察。同帶的參照：prod-new 重現地板 **11.22**（run5）／**11.64**（run4），
卡面 `11.90 ± 0.15`。

## 4. 線 2：S1 的 E0–E4（驅動改了三處）

```bash
ESERIES_COOL_S=420 python3 Backup/eseries/driver.py E0 E4 E2   # 每次只跑一個 step 也可以
# 產物：Backup/eseries/results.json（逐步即時寫入）
```

這輪改的三處（都是本輪量測的缺陷，不是新功能）：

1. **每臂冷卻**：`bench_step` 之前會 sleep `ESERIES_COOL_S`（預設 420 s）。
2. **E4 拆成單臂啟動**（`bench_arms_cooled`）：`harness bench --arm A --arm B` 會讓 B 緊接 A
   啟動，兩臂的盒況不同——這正是本輪 E4 的比值只能是診斷價的原因。拆開後仍寫同一份
   `summary.json`（同樣是 arm list），另存 `runs.json` 記啟動序與每臂 swap。
3. **probe 落盤**（`probe_of`）：E0 的決定性證據是**答案位元**（分段臂 `42` vs S1 臂
   `文摘文摘…`），它原本只存在 console——S1 臂在 gate 寫 summary 之前就退出。

**E0 的前提與陷阱**：`--write-ref` 那條路會拿**預設 pin（prefill250）**當比較基準，所以 E0-seg
自己會被判 `INVALID COMPARISON`（36 個 config diff）。**這不是 S1 的缺陷**，但意思很明確：
S1 兩臂必須以那份新寫的 `Backup/eseries/E0/seg_ref.jsonl` 為基準，且 `probe_prompt_md5` 三臂
必須一致——那個欄位本輪沒有落盤，是這次新加 `probe_of` 要修的事。

**本輪 S1 的讀數（診斷價，swap 7.7–8.2 GiB／worst HEAVY）**：

| 格 | 讀數 | 狀態 |
|---|---|---|
| E0 | 分段臂 `probe_answer=42`；兩個 S1 臂（8 GiB 暖池／1 GiB 小池）都回 `文摘文摘…`，且引擎自蓋 INVALID `non-finite values 248320 of 248320` | 前提成立：S1 在交付形狀下不是「值錯」而是 **NaN** |
| E1 | fill 可引用值 **20.8 ms/step**；3.955 降級為「兩臂步時差的殘差歸屬」 | 0 GPU，可離線複核 |
| E2 | **40/40 層都有 miss**、rho **−0.32**（round-trip-bound）；同臂同框 decode 步 75.09 ms ＝ wait **88.7%** ＋ cb **6.7%** ＋ submit 4.5%，EBTIMER 4.32 ms/步 ⇒ **FILL_IN_CB** ⇒ 每 miss **0.69–0.86 ms** | 形狀 1（同步等 fill）判死：可移除的只有 6.7% |
| E3 | combine 熔在 `kernel_mul_mv_id_down_combine_*` ⇒ 補算只能重跑融合 combine；`pool_ext_buf` 是非擁有視圖 ⇒ 異步 fill 必須 staging＋copy-in | 0 GPU 讀碼 |
| E4 | 分段臂 pp 233.29／**tg 11.396**（11.3544/11.4325/11.4025）；S1 臂 pp 187.03／**tg 16.136**（17.2755/14.2276/16.904，misses 0、reads 0） ⇒ decode ×1.42／prefill ×0.80。兩臂 `swap_arms` 都是 `1/2/1` ⇒ 這一對只差 S1 的三個開關 | 只能當**診斷價**：兩臂 worst HEAVY、起跑 swap `8129.12` MiB、非交錯（B 緊接 A） |

## 5. 產物與 md5（Backup 不入版控，所以指紋寫在這裡）

| 檔 | md5 |
|---|---|
| `Backup/mtpoff_base/p0_vs_madv_iso.py` | `46dff2a42ea4f9d9d27ef5926e8ab4bf` |
| `Backup/eseries/driver.py` | `47ffcfdce6a118d8b20497bcfc3ce7dd` |
| `scripts/check/s1_wait_budget.py` | `c9ef008d56cc735fa09e99d3f8c54bd5` |
| `scripts/check/column_census.py` | `fff40db1cc0c32a2704da60cfccc3cce` |
| `Backup/p0attrib/iso_20260925_2011/summary.json` | `3a59f52c3e8edfd42d89e07be0f9fe3b` |
| `Backup/eseries/results.json` | `05f99edfbbd35efda7c150c47c90d3c6` |
| `Backup/mtpoff_base/run5_20260925_193721_prodnew_ws64.json` | `0fc0d45193a50f1d948ee6dadd63b736` |

散落的產物目錄：`Backup/p0attrib/{pass1_20260925_1946,pass2_20260925_1956,iso_20260925_2011}/`、
`Backup/eseries/{E0,E2b,E2c,E4}/`、`Backup/mtpoff_base/`。

## 6. 不要重複的四個協定缺陷（本輪我自己犯的）

1. **pass1 三臂漏 `--warm-skip 64`** ⇒ 落在「冷啟動含在內」的口徑，不能與 11.22/11.64 併排。
2. **iso driver 的 idle 判準只看 `llama-bench`**，看不到「正在冷卻、沒有子行程」的 peer driver
   ⇒ 20:09 兩支 driver 撞在一起，兩邊都作廢。已修：連 peer driver 一起判。
3. **pass2 的首臂沒有冷卻**，與後兩臂不同源 ⇒ 那個 `ctrl 7.75` 不是對照。
4. **E0 的決定性證據只存在 console**（答案位元）⇒ 現在由 `probe_of()` 落盤。

## 7. 這份檔沒有主張的事

- 沒有宣稱任何吞吐：本輪 P0/P1/P2 八臂與 S1 五格全部是診斷價。
- 沒有裁決 P0/P1/P2 的肇因：判準的前提在權威 cell 上失敗，所以它回的是
  `PRECONDITION FAILED`，不是「P0」或「P1/P2」。
- 沒有把 S1 的 NaN 歸因到任何一個 env：E0 只有「S1 臂 NaN、分段臂 `42`」這個對比，三個
  S1 開關彼此未分離。
