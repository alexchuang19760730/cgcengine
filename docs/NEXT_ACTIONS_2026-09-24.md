# 下一步行動清單 — 2026-09-24

> 優先級排序後的待辦事項。做完手邊的事，照此順序。

> **⚠️ 強制前置（所有速度 A/B）**：讀 `docs/ABBA_MEASUREMENT_PROTOCOL_2026-09-23.md`
> 並照做（交錯、深冷卻 ≥300s、配對中位、log-space 校正、窗口守門、build 指紋）。
> 缺協議的數字不進報告。

***

## 🔴 兩個已定案的前提（2026-09-24 夜，勿再重複推翻）

> 全文與來源：`docs/PREFILL250_MET_AND_S1_REJUDGE_2026-09-24.md`（靜態核對，0 GPU）。

### P1. **prefill 250 已達標** —— 不要再把它列為缺口
同一個 cell（`-p 2048`，prod-new 口徑）在可指名來源裡有 **9 次 launch ≥ 250 t/s**：
283.01（`PROD_NEW_MTP_OFF_2026-09-23.md`）、296.24 / 275.98 / 269.23 / 262.01 / 275.39
（`Backup/seg_batch_s1_pairs/`）、289.28 / 280.5 / 289.81（commit-gate 標題）。

⚠ **但白皮書 §3.1/§5.1 的 `ctrl 162.5 → p0+B 209.5（+29%）` 不成立**：那不是程式碼的性質。
同一支臂在同一個視窗十分鐘內的四次 launch 是 **275.39 / 196.72 / 207.33 / 204.98**
（同臂散佈 **78.7 t/s = 中位的 38%**）；`177.36` 那次自己的標題就寫「swap 8.6GB 脏，环境不可引用」。
**要報一個可引用的 prefill，必須連視窗區塊與 build digest 一起報**（照 `PROD_NEW_MTP_OFF_2026-09-23.md` 的格式）。
未達的是**可重現的地板**，不是峰值。

### P2. **單段提交（S1）的「2.2×」重判為 1.72×，且不可歸因為 shape**
`Backup/seg_batch_s1_pairs/abba_212809.json`（8 次 launch、交錯、同 build、同 cell、兩臂均無 `--spec-type`）：
A 分段 **11.84** / B 單段 **20.32** ⇒ 中位比值 **1.72×**（被引用的 2.2× 其分母 A 取自 8.24 那段已失敗的子視窗）。

而兩臂的 **I/O 結構不同**：A `file_reads` **82,293** / `misses` **4,835** / `read_mib` 2,440；
B **全部為 0**（無 hook ⇒ 無 demand fill ⇒ 填充路徑根本沒跑）。
⇒ **這個比值是「S1 ＋ 填充路徑不執行」的合併效果，不是 S1 的 shape 收益**；
而且兩臂 `k_eff = 1`，交付 regime 是 `mean_len 3.117` ⇒ **S1 從未在背著 4-token verify 下量過**。

**⇒ 新硬規則（已實作）**：`scripts/check/io_symmetry.py`（`--selftest` 11/11 PASS）；
兩臂 I/O 結構不一致時**拒絕把比值簽成效果**（`asymmetric` / `unknown` 皆 blocking）。
已接進 `scripts/check/decode_carrier_ab.py`：verdict 變 `shape-confounded`，**exit 2**。
```sh
python3 scripts/check/io_symmetry.py --dir Backup/seg_batch_s1_pairs   # exit 2
```

> **🔴 commit 口徑統一**：commit 統一用 `prod-new`（單一介面）。任何寫進
> 報告/白皮書/commit 標題的數字一律 prod-new 口徑產出。

> **🔴 量測紀律（2026-09-24 新增，測試卡 §5，兩個 Agent 與 MainAgent 都必須遵守）**：
> ① 每臂必須**同時測並同時報 prefill + decode**（pp + tg 兩行）——只報 tg = 不完整產物，不可引用；
> ② 每個報出的數字必須帶 **thermal state（launch/worst/hist）+ swap state（launch/end/worst/growth）**
> 標注，取自產物欄位（matrix json 已內嵌），不許手寫或跨產物借用；
> ③ **HEAVY 或 swap growth > 500 MiB = 污染讀數**，只能當診斷價，不可進錨點 / commit 標題。
> 全文：`docs/PROD_NEW_TEST_CARD_2026-09-24.md` §5。

> **🔴 commit gate**：每次 commit 前跑 `python3 scripts/check/commit_bench.py`
> （prod-new，prefill≥120 / decode≥10，ABBA 協議）。標題帶 `[commit-gate]` 成績。

> **🔴 budget 預檢（2026-09-24 新增）**：所有速度 runner / launch 前必須跑
> `python3 scripts/check/budget_preflight.py`（同 run_server.sh 口徑：model resident + pool vs 實體
> 記憶體），超訂（OVERSUBSCRIBED）即 exit 2 拒跑——不要跑完才發現不可引用。
> 實證：MTP on + ρ 影子節點在 16GB 超訂 **4838 MiB**（model 13030 + pool 8192 = 21222 > 16384）
> ⇒ B 臂全 0.00 t/s OOM，50 分鐘白跑。**MTP on + 影子節點的 insert 在 16GB 結構性不可量**；
> 要量需 pool ≤ 3GiB 或換更小模型。
> 相容模式：`--warn-only`（明知超訂仍要跑）。
>
> **接線面（2026-09-24 10:0x）**：`scripts/check/budget_gate.sh` —— 量測腳本 launch 前
> `. "${REPO}/scripts/check/budget_gate.sh"` 一行即可（已接 `rho_fill_ab.sh`／
> `route_overlap_3prompt.sh`／`masscov_decode_shape.sh`；`--self-test` 7/7）。
> `BUDGET_GATE=strict`（預設，超訂 exit 2）／`warn`（放行但樣本帶 `CGC_BUDGET_OVERSUBSCRIBED=1`
> 標記）／`off`。工具不會自己被叫到 —— **沒接線的 runner 等於沒有閘門**。

***

## 方向決策（2026-09-24 03:15，量 h 閉合）

### ❌ A（speculative slot binding / 跨 token 預指派）——從候選清單刪除
實測（docs/H_MEASURED_2026-09-24.md §四，prod-new / MTP off / 3721 層樣本）：
- **全中率 h_all = 0.0030**（每步 union 100% 被上一步 union 蓋住只有 0.3%）
- 贏 ρ 上界需 h ≥ 0.75；原門檻 0.65——實測差 200 倍以上
- 跨 token 猜語義上不可行：每次固定錯 ~2.7 個 expert
- **任何 agent 不要再碰 A 路線**（包括預指派 slot、跨 token 候選集加寬）

### ✅ ρ 路線 = 唯一正路（覆蓋率已飽和，剩 insert）

> **⚠️ 已被取代（2026-09-24 18:2x 起）**：本節寫於 03:15。當天稍晚的
> `docs/SHAPE_GAP_TO_TARGETS_2026-09-24.md` §2.4 與 `MEMORY.md` 已把第一順位換成
> **單段提交（S1）＋ miss 處理**，ρ 與 prebind 降為第二順位；同一天的
> `docs/RHO_BATCH_REGRESSION_2026-09-24.md` 另記 ρ-batch 實測 −38~51%。
> 「唯一正路」這四個字在 15 小時內換過四次（見
> `docs/PREFILL250_MET_AND_S1_REJUDGE_2026-09-24.md` §2），**不要照字面排優先級**。
實測：cov_uni=0.8585 / rho_tok=0.8581（>0.70 飽和點，覆蓋率不是瓶頸）
剩餘槓桿（唯一正路）：
1. **ρ insert 實測**（白皮書 §10 下一步）——影子節點插入成本 4.76 ms/步 是估值，
   需實測歸因（CGC-RHO 影子節點 + gate+topk 的實際 GPU 時間）
2. cb 42–51 ms 遮蔽（CB_DELIVERY_SETTLED 已定讞，+14.5–19% 真實收益）

***

## 待辦（照序）

- [x] **build（2026-09-24 10:47）** ✅ `cmake --build src/llama.cpp/build --target llama-server
      llama-bench -j 8` rc=0；`strings libllama.dylib` 驗到 `CGC_EXPERT_SKIP_READRAW`、
      `CGC_POOL_MADVISE` 兩個字串都在 ⇒ 三個開關確實編進去了。
      ⚠ 這個 binary **同時含線 I 未提交的 294+ 行 `llama-expert-cache.cpp` 改動**
      ⇒ **ctrl 臂 ≠ 12.57 錨點**，只能三臂互比。
- [ ] **⚠ 重建（P0.g 還沒進 binary）**：`CGC_EXPERT_SKIP_READRAW` 的判定已從「存在即開」
      改成「取值判定」（`=0`/空 = 關），原始碼已改但**尚未 build**（11:2x 有別條線的
      llama-bench 在跑，不能蓋 binary）。重開機後、A/B 之前先：
      ```sh
      cmake --build src/llama.cpp/build --target llama-server llama-bench -j 8
      ```
      （三個閘門：8080 無 listener／無量測行程／無 cmake 在跑。）
- [ ] **開關已接進 prod-new profile**（`run_server.sh`）：白名單 ＋ prod-new 底下的
      顯式預設。`CGC_DUMP_ENV=1` 已驗證：預設不傳、`=1`/`=2` 會傳、**`=0` 不傳**。
      ⚠ 白名單是必須的：launch line 走 `env "${SERVER_ENV[@]}"`，沒列到的 `CGC_*` 被靜默丟掉
      ⇒ 不列進來 A/B 三臂會跑出一模一樣的數字而不報錯。
- [ ] **重開機後跑 A/B（一步到位）**：
      ```sh
      python3 scripts/check/p012_ab.py --rounds 2                                  # prod25
      python3 scripts/check/p012_ab.py --rounds 2 --profile prod-new \
              --extra-env CGC_SERVER_MTP=1                                         # prod-new
      python3 scripts/check/p012_ab.py --dry-run          # 零 GPU 先看會跑什麼
      ```
      ⚠ **prod-new 一定要疊 `CGC_SERVER_MTP=1`**：MTP=0 分支會把 MODEL 換成
      `Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`（**沒有 nextn head 的另一個 checkpoint**）。
      手臂順序：ctrl（不設 env）→ p0（`CGC_EXPERT_SKIP_READRAW=1`）→
      p012（`+CGC_POOL_MADVISE=2`）；每臂記 swap 前後、tee 完整輸出到
      `<out>/logs/<arm>_r<n>/`。跑完接著
      ```sh
      python3 scripts/check/miss_attr_gate.py \
          --row ctrl=<out>/logs/ctrl_r1/decode-delivery.out \
          --row p0=<out>/logs/p0_r1/decode-delivery.out \
          --row p012=<out>/logs/p012_r1/decode-delivery.out
      ```
      ⚠ 開 P0 時**必須先看輸出是否還是正常文字**（layer-0 guard 若失效就是垃圾且不報錯）。
      ⚠ 8 GiB pool 在 16 GB 上**靜態超訂**：P0 後 10262 + 8192 = 18454 > 16384，仍超 2070 MiB
      ⇒ 這組數字只能三臂互比，**不可當交付錨點**；要合法得把 pool 降到 ≤ 6122 MiB。
- [ ] **P0（正路）**：ρ insert 實測——量影子節點 + gate+topk 的實際成本，
      把 4.76 ms/步 從估值變成讀數；若 insert 不可壓，評估 ρ-batch 按層批次化（freebuff 的活）
- [ ] 環境治理：swap 8648MB 髒——重開機/purge 拿乾淨基線後再跑任何速度 A/B
- [ ] H_MEASURED §四 與 NEXT_ACTIONS 方向已 commit（含 h_all 儀器）
- [ ] freebuff：CGC_IDSEQ_DUMP 儀器不可達（CGC-CANON=0）——若要 per-call ids
      序列，移到真正執行的 topk hook 路徑；不急（h_all 已覆蓋判決需求）


## 🔴 統一量測入口（harness bench）— 2026-09-24
- 所有報告數字必須基於 prod-new + 自己的 env 增量，唯一入口：
  `python3 scripts/check/harness.py bench --arm "prod-new:!OVERRIDE;KEY=VAL" --json <out>`
- base gate：撞 base 鍵需 `!` 宣告；實驗開關白名單 `_EXPERIMENT_KNOBS` 自動放行 + 產物記錄
- 產物契約：每臂 base_check + sys_before/sys_after（thermal / swap 水位 / pageins/pageouts 速率 / memory_pressure / iostat）——測試前後系統狀態必附
- 委託 llama_bench_matrix 跑量測（邏輯不複製）；`--dry-run` 走 matrix 相同語義
- 現有 harness 窗口守門（show/run/list/audit）未動


## ⚠️ 重開機通知（2026-09-24 23:2x）— 兩分鐘內重開機
- **即刻停手**：不要啟動新的 GPU 量測；正在跑的會被中斷：
  - `seg_batch_abba.py --mode identity`（pid 63457/64173，-n 8 -r 3）可能跑不完——結果以寫入 Backup/zm_identity2 的為準，跑一半不算數
  - `s1_ksweep.py --wait-minutes 240`（pid 64421）會被殺——重開機後需重啟該 driver 才能續
- **重開機後第一動作**（swap 應歸零）：
  1. `bash /tmp/seg_mtp_abba/run_clean.sh` —— 單段×MTP 單對判定（A1/B1，兩臂帶 P0=1）
  2. `python3 scripts/check/harness.py bench --arm "prod-new" --json /tmp/harness_first.json` —— 乾淨基線（prefill/decode + 測試前後系統快照）
- **重開機前 swap 現況**：used 7834.5 MiB（total 9216）——重開機後對比用
- 統一量測入口 harness bench 已就緒（見上方條目）；所有後續報告基於 prod-new + 自己的 env 增量

## [watchdog 2026-09-25 08:53:31] 巡檢問題（看門人自動通知）

- 污染產物 **不可引用/不可進 commit 標題**：prod-new pp=260.41 tg=12.20 （啟動就髒 launch_swap=2057; swap growth=+3446; attribution=swap）@ `/Users/alexchuang/Documents/flashkv-devserver/Backup/mw_ab/mw_ctrl.json`
- 請相關 agent：修正參數/環境後重跑，勿把壓力下數字當生產成績。

## [watchdog 2026-09-25 08:58:42] 巡檢問題（看門人自動通知）

- 污染產物 **不可引用/不可進 commit 標題**：prod-new pp=260.41 tg=12.20 （啟動就髒 launch_swap=2057; swap growth=+3446; attribution=swap）@ `/Users/alexchuang/Documents/flashkv-devserver/Backup/mw_ab/mw_ctrl.json`
- 污染產物 **不可引用/不可進 commit 標題**：prod-new pp=277.42 tg=12.12 （啟動就髒 launch_swap=3302; swap growth=+2044; attribution=swap）@ `/Users/alexchuang/Documents/flashkv-devserver/Backup/mw_ab/mw_w2.json`
- 請相關 agent：修正參數/環境後重跑，勿把壓力下數字當生產成績。

## [watchdog 2026-09-25 09:08:43] 巡檢問題（看門人自動通知）

- 污染產物 **不可引用/不可進 commit 標題**：prod-new pp=260.41 tg=12.20 （啟動就髒 launch_swap=2057; swap growth=+3446; attribution=swap）@ `/Users/alexchuang/Documents/flashkv-devserver/Backup/mw_ab/mw_ctrl.json`
- 污染產物 **不可引用/不可進 commit 標題**：prod-new pp=277.42 tg=12.12 （啟動就髒 launch_swap=3302; swap growth=+2044; attribution=swap）@ `/Users/alexchuang/Documents/flashkv-devserver/Backup/mw_ab/mw_w2.json`
- 請相關 agent：修正參數/環境後重跑，勿把壓力下數字當生產成績。

## [watchdog 2026-09-25 09:20:44] 巡檢問題（看門人自動通知）

- 已隔離失效數字：['prod-new'] @ `/Users/alexchuang/Documents/flashkv-devserver/Backup/pf_ab/pf_new.json` → `/Users/alexchuang/Documents/flashkv-devserver/Backup/quarantine/pf_new.20260925_092044.json`
- 量具失效、數字作廢：prod-new pp=268.78 tg=8.21 （量具散度 150%>12%（不可重複））@ `/Users/alexchuang/Documents/flashkv-devserver/Backup/pf_ab/pf_new.json`
- 宣稱vs實際待驗：workers 同宣稱=8、但 us/job 離散 27454~30891（13%）；可疑 ['/Users/alexchuang/Documents/flashkv-devserver/Backup/mw_ab/mw_ctrl.json']
    高 us/job 臂的並行度可能未實際生效（workers 實際或較少）；這是『請驗證』、非定論。需對實際下發值/進程命令行核對。
- 環境承壓但**數字有效、可引用**：prod-new pp=260.41 tg=12.20 （啟動 swap=2057; swap growth=+3446（環境承壓、需結構修復；數字本身仍有效））；swap 結構問題另需修復
- 環境承壓但**數字有效、可引用**：prod-new pp=277.42 tg=12.12 （啟動 swap=3302; swap growth=+2044（環境承壓、需結構修復；數字本身仍有效））；swap 結構問題另需修復

## [watchdog 2026-09-25 09:32:16] 巡檢問題（看門人自動通知）

- 已隔離失效數字：['prod-new'] @ `/Users/alexchuang/Documents/flashkv-devserver/Backup/quarantine/pf_new.20260925_092044.json` → `/Users/alexchuang/Documents/flashkv-devserver/Backup/quarantine/pf_new.20260925_092044.20260925_093216.json`
- 量具失效、數字作廢：prod-new pp=268.78 tg=8.21 （量具散度 150%>12%（不可重複））@ `/Users/alexchuang/Documents/flashkv-devserver/Backup/quarantine/pf_new.20260925_092044.json`
- 宣稱vs實際待驗：workers 同宣稱=8、但 us/job 離散 27454~30891（13%）；可疑 ['/Users/alexchuang/Documents/flashkv-devserver/Backup/mw_ab/mw_ctrl.json']
    高 us/job 臂的並行度可能未實際生效（workers 實際或較少）；這是『請驗證』、非定論。需對實際下發值/進程命令行核對。
- 環境承壓但**數字有效、可引用**：prod-new pp=260.41 tg=12.20 （啟動 swap=2057; swap growth=+3446（環境承壓、需結構修復；數字本身仍有效））；swap 結構問題另需修復
- 環境承壓但**數字有效、可引用**：prod-new pp=277.42 tg=12.12 （啟動 swap=3302; swap growth=+2044（環境承壓、需結構修復；數字本身仍有效））；swap 結構問題另需修復

## [watchdog 2026-09-25 09:42:16] 巡檢問題（看門人自動通知）

- 已隔離失效數字：['prod-new'] @ `/Users/alexchuang/Documents/flashkv-devserver/Backup/quarantine/pf_new.20260925_092044.20260925_093216.json` → `/Users/alexchuang/Documents/flashkv-devserver/Backup/quarantine/pf_new.20260925_092044.20260925_093216.20260925_094216.json`
- 量具失效、數字作廢：prod-new pp=268.78 tg=8.21 （量具散度 150%>12%（不可重複））@ `/Users/alexchuang/Documents/flashkv-devserver/Backup/quarantine/pf_new.20260925_092044.20260925_093216.json`
- 宣稱vs實際待驗：workers 同宣稱=8、但 us/job 離散 27454~30891（13%）；可疑 ['/Users/alexchuang/Documents/flashkv-devserver/Backup/mw_ab/mw_ctrl.json']
    高 us/job 臂的並行度可能未實際生效（workers 實際或較少）；這是『請驗證』、非定論。需對實際下發值/進程命令行核對。
- 環境承壓但**數字有效、可引用**：prod-new pp=260.41 tg=12.20 （啟動 swap=2057; swap growth=+3446（環境承壓、需結構修復；數字本身仍有效））；swap 結構問題另需修復
- 環境承壓但**數字有效、可引用**：prod-new pp=277.42 tg=12.12 （啟動 swap=3302; swap growth=+2044（環境承壓、需結構修復；數字本身仍有效））；swap 結構問題另需修復

## [watchdog 2026-09-25 09:52:16] 巡檢問題（看門人自動通知）

- 已隔離失效數字：['prod-new'] @ `/Users/alexchuang/Documents/flashkv-devserver/Backup/quarantine/pf_new.20260925_092044.20260925_093216.20260925_094216.json` → `/Users/alexchuang/Documents/flashkv-devserver/Backup/quarantine/pf_new.20260925_092044.20260925_093216.20260925_094216.20260925_095216.json`
- 量具失效、數字作廢：prod-new pp=268.78 tg=8.21 （量具散度 150%>12%（不可重複））@ `/Users/alexchuang/Documents/flashkv-devserver/Backup/quarantine/pf_new.20260925_092044.20260925_093216.20260925_094216.json`
- 宣稱vs實際待驗：workers 同宣稱=8、但 us/job 離散 27454~30891（13%）；可疑 ['/Users/alexchuang/Documents/flashkv-devserver/Backup/mw_ab/mw_ctrl.json']
    高 us/job 臂的並行度可能未實際生效（workers 實際或較少）；這是『請驗證』、非定論。需對實際下發值/進程命令行核對。
- 環境承壓但**數字有效、可引用**：prod-new pp=260.41 tg=12.20 （啟動 swap=2057; swap growth=+3446（環境承壓、需結構修復；數字本身仍有效））；swap 結構問題另需修復
- 環境承壓但**數字有效、可引用**：prod-new pp=277.42 tg=12.12 （啟動 swap=3302; swap growth=+2044（環境承壓、需結構修復；數字本身仍有效））；swap 結構問題另需修復

## [watchdog 2026-09-25 10:02:16] 巡檢問題（看門人自動通知）

- 已隔離失效數字：['prod-new'] @ `/Users/alexchuang/Documents/flashkv-devserver/Backup/quarantine/pf_new.20260925_092044.20260925_093216.20260925_094216.20260925_095216.json` → `/Users/alexchuang/Documents/flashkv-devserver/Backup/quarantine/pf_new.20260925_092044.20260925_093216.20260925_094216.20260925_095216.20260925_100216.json`
- 量具失效、數字作廢：prod-new pp=268.78 tg=8.21 （量具散度 150%>12%（不可重複））@ `/Users/alexchuang/Documents/flashkv-devserver/Backup/quarantine/pf_new.20260925_092044.20260925_093216.20260925_094216.20260925_095216.json`
- 宣稱vs實際待驗：workers 同宣稱=8、但 us/job 離散 27454~30891（13%）；可疑 ['/Users/alexchuang/Documents/flashkv-devserver/Backup/mw_ab/mw_ctrl.json']
    高 us/job 臂的並行度可能未實際生效（workers 實際或較少）；這是『請驗證』、非定論。需對實際下發值/進程命令行核對。
- 環境承壓但**數字有效、可引用**：prod-new pp=260.41 tg=12.20 （啟動 swap=2057; swap growth=+3446（環境承壓、需結構修復；數字本身仍有效））；swap 結構問題另需修復
- 環境承壓但**數字有效、可引用**：prod-new pp=277.42 tg=12.12 （啟動 swap=3302; swap growth=+2044（環境承壓、需結構修復；數字本身仍有效））；swap 結構問題另需修復

## [watchdog 2026-09-25 10:12:16] 巡檢問題（看門人自動通知）

- 已隔離失效數字：['prod-new'] @ `/Users/alexchuang/Documents/flashkv-devserver/Backup/quarantine/pf_new.20260925_092044.20260925_093216.20260925_094216.20260925_095216.20260925_100216.json` → `/Users/alexchuang/Documents/flashkv-devserver/Backup/quarantine/pf_new.20260925_092044.20260925_093216.20260925_094216.20260925_095216.20260925_100216.20260925_101217.json`
- 量具失效、數字作廢：prod-new pp=268.78 tg=8.21 （量具散度 150%>12%（不可重複））@ `/Users/alexchuang/Documents/flashkv-devserver/Backup/quarantine/pf_new.20260925_092044.20260925_093216.20260925_094216.20260925_095216.20260925_100216.json`
- 環境承壓但**數字有效、可引用**：prod-new pp=277.42 tg=12.12 （啟動 swap=3302; swap growth=+2044（環境承壓、需結構修復；數字本身仍有效））；swap 結構問題另需修復
- 環境承壓但**數字有效、可引用**：prod-new pp=300.51 tg=11.79 （啟動 swap=2496; swap growth=+2128（環境承壓、需結構修復；數字本身仍有效））；swap 結構問題另需修復

## [watchdog 2026-09-25 10:22:17] 巡檢問題（看門人自動通知）

- 已終止 pid=44235（terminated (SIGTERM)）：swap 5011 > 3072 MiB; 大 batch 5632 在高危記憶體下跑
- 已隔離失效數字：['prod-new'] @ `/Users/alexchuang/Documents/flashkv-devserver/Backup/quarantine/pf_new.20260925_092044.20260925_093216.20260925_094216.20260925_095216.20260925_100216.20260925_101217.json` → `/Users/alexchuang/Documents/flashkv-devserver/Backup/quarantine/pf_new.20260925_092044.20260925_093216.20260925_094216.20260925_095216.20260925_100216.20260925_101217.20260925_102217.json`
- 量具失效、數字作廢：prod-new pp=268.78 tg=8.21 （量具散度 150%>12%（不可重複））@ `/Users/alexchuang/Documents/flashkv-devserver/Backup/quarantine/pf_new.20260925_092044.20260925_093216.20260925_094216.20260925_095216.20260925_100216.20260925_101217.json`
- 環境承壓但**數字有效、可引用**：prod-new pp=277.42 tg=12.12 （啟動 swap=3302; swap growth=+2044（環境承壓、需結構修復；數字本身仍有效））；swap 結構問題另需修復
- 環境承壓但**數字有效、可引用**：prod-new pp=300.51 tg=11.79 （啟動 swap=2496; swap growth=+2128（環境承壓、需結構修復；數字本身仍有效））；swap 結構問題另需修復

## [watchdog 2026-09-25 10:32:18] 巡檢問題（看門人自動通知）

- 已隔離失效數字：['prod-new'] @ `/Users/alexchuang/Documents/flashkv-devserver/Backup/quarantine/pf_new.20260925_092044.20260925_093216.20260925_094216.20260925_095216.20260925_100216.20260925_101217.20260925_102217.json` → `/Users/alexchuang/Documents/flashkv-devserver/Backup/quarantine/pf_new.20260925_092044.20260925_093216.20260925_094216.20260925_095216.20260925_100216.20260925_101217.20260925_102217.20260925_103218.json`
- 量具失效、數字作廢：prod-new pp=268.78 tg=8.21 （量具散度 150%>12%（不可重複））@ `/Users/alexchuang/Documents/flashkv-devserver/Backup/quarantine/pf_new.20260925_092044.20260925_093216.20260925_094216.20260925_095216.20260925_100216.20260925_101217.20260925_102217.json`
- 環境承壓但**數字有效、可引用**：prod-new pp=300.51 tg=11.79 （啟動 swap=2496; swap growth=+2128（環境承壓、需結構修復；數字本身仍有效））；swap 結構問題另需修復
- 環境承壓但**數字有效、可引用**：prod-new pp=302.56 （啟動 swap=2180; swap growth=+2735（環境承壓、需結構修復；數字本身仍有效））；swap 結構問題另需修復

## [watchdog 2026-09-25 10:42:18] 巡檢問題（看門人自動通知）

- 已隔離失效數字：['prod-new'] @ `/Users/alexchuang/Documents/flashkv-devserver/Backup/quarantine/pf_new.20260925_092044.20260925_093216.20260925_094216.20260925_095216.20260925_100216.20260925_101217.20260925_102217.20260925_103218.json` → `/Users/alexchuang/Documents/flashkv-devserver/Backup/quarantine/pf_new.20260925_092044.20260925_093216.20260925_094216.20260925_095216.20260925_100216.20260925_101217.20260925_102217.20260925_103218.20260925_104218.json`
- 量具失效、數字作廢：prod-new pp=268.78 tg=8.21 （量具散度 150%>12%（不可重複））@ `/Users/alexchuang/Documents/flashkv-devserver/Backup/quarantine/pf_new.20260925_092044.20260925_093216.20260925_094216.20260925_095216.20260925_100216.20260925_101217.20260925_102217.20260925_103218.json`
- 宣稱vs實際待驗：workers 同宣稱=8、但 us/job 離散 27422~30747（12%）；可疑 ['/Users/alexchuang/Documents/flashkv-devserver/Backup/quarantine/pf_new.20260925_092044.20260925_093216.20260925_094216.20260925_095216.20260925_100216.20260925_101217.20260925_102217.20260925_103218.json']
    高 us/job 臂的並行度可能未實際生效（workers 實際或較少）；這是『請驗證』、非定論。需對實際下發值/進程命令行核對。
- 環境承壓但**數字有效、可引用**：prod-new pp=267.34 tg=14.24 （啟動 swap=5986）；swap 結構問題另需修復
- 環境承壓但**數字有效、可引用**：prod-new pp=240.52 tg=11.03 （啟動 swap=6043）；swap 結構問題另需修復
- 環境承壓但**數字有效、可引用**：prod-new pp=300.51 tg=11.79 （啟動 swap=2496; swap growth=+2128（環境承壓、需結構修復；數字本身仍有效））；swap 結構問題另需修復
- 環境承壓但**數字有效、可引用**：prod-new pp=302.56 （啟動 swap=2180; swap growth=+2735（環境承壓、需結構修復；數字本身仍有效））；swap 結構問題另需修復

## [watchdog 2026-09-25 10:52:18] 巡檢問題（看門人自動通知）

- 環境承壓但**數字有效、可引用**：prod-new pp=267.34 tg=14.24 （啟動 swap=5986）；swap 結構問題另需修復
- 環境承壓但**數字有效、可引用**：prod-new pp=240.52 tg=11.03 （啟動 swap=6043）；swap 結構問題另需修復
- 環境承壓但**數字有效、可引用**：prod-new pp=300.51 tg=11.79 （啟動 swap=2496; swap growth=+2128（環境承壓、需結構修復；數字本身仍有效））；swap 結構問題另需修復
- 環境承壓但**數字有效、可引用**：prod-new pp=302.56 （啟動 swap=2180; swap growth=+2735（環境承壓、需結構修復；數字本身仍有效））；swap 結構問題另需修復

## [watchdog 2026-09-25 11:02:18] 巡檢問題（看門人自動通知）

- 環境承壓但**數字有效、可引用**：prod-new pp=267.34 tg=14.24 （啟動 swap=5986）；swap 結構問題另需修復
- 環境承壓但**數字有效、可引用**：prod-new pp=240.52 tg=11.03 （啟動 swap=6043）；swap 結構問題另需修復
- 環境承壓但**數字有效、可引用**：prod-new pp=300.51 tg=11.79 （啟動 swap=2496; swap growth=+2128（環境承壓、需結構修復；數字本身仍有效））；swap 結構問題另需修復
- 環境承壓但**數字有效、可引用**：prod-new pp=302.56 （啟動 swap=2180; swap growth=+2735（環境承壓、需結構修復；數字本身仍有效））；swap 結構問題另需修復

## [watchdog 2026-09-25 11:12:19] 巡檢問題（看門人自動通知）

- 環境承壓但**數字有效、可引用**：prod-new pp=267.34 tg=14.24 （啟動 swap=5986）；swap 結構問題另需修復
- 環境承壓但**數字有效、可引用**：prod-new pp=240.52 tg=11.03 （啟動 swap=6043）；swap 結構問題另需修復
- 環境承壓但**數字有效、可引用**：prod-new pp=300.51 tg=11.79 （啟動 swap=2496; swap growth=+2128（環境承壓、需結構修復；數字本身仍有效））；swap 結構問題另需修復
- 環境承壓但**數字有效、可引用**：prod-new pp=302.56 （啟動 swap=2180; swap growth=+2735（環境承壓、需結構修復；數字本身仍有效））；swap 結構問題另需修復

## [watchdog 2026-09-25 11:22:19] 巡檢問題（看門人自動通知）

- 環境承壓但**數字有效、可引用**：prod-new pp=267.34 tg=14.24 （啟動 swap=5986）；swap 結構問題另需修復
- 環境承壓但**數字有效、可引用**：prod-new pp=240.52 tg=11.03 （啟動 swap=6043）；swap 結構問題另需修復
- 環境承壓但**數字有效、可引用**：prod-new pp=300.51 tg=11.79 （啟動 swap=2496; swap growth=+2128（環境承壓、需結構修復；數字本身仍有效））；swap 結構問題另需修復
- 環境承壓但**數字有效、可引用**：prod-new pp=302.56 （啟動 swap=2180; swap growth=+2735（環境承壓、需結構修復；數字本身仍有效））；swap 結構問題另需修復

## [watchdog 2026-09-25 11:33:15] 巡檢問題（看門人自動通知）

- 環境承壓但**數字有效、可引用**：prod-new pp=267.34 tg=14.24 （啟動 swap=5986）；swap 結構問題另需修復
- 環境承壓但**數字有效、可引用**：prod-new pp=240.52 tg=11.03 （啟動 swap=6043）；swap 結構問題另需修復
- 環境承壓但**數字有效、可引用**：prod-new pp=300.51 tg=11.79 （啟動 swap=2496; swap growth=+2128（環境承壓、需結構修復；數字本身仍有效））；swap 結構問題另需修復
- 環境承壓但**數字有效、可引用**：prod-new pp=302.56 （啟動 swap=2180; swap growth=+2735（環境承壓、需結構修復；數字本身仍有效））；swap 結構問題另需修復

## [watchdog 2026-09-25 11:46:30] 巡檢問題（看門人自動通知）

- 環境承壓但**數字有效、可引用**：prod-new pp=267.34 tg=14.24 （啟動 swap=5986）；swap 結構問題另需修復
- 環境承壓但**數字有效、可引用**：prod-new pp=240.52 tg=11.03 （啟動 swap=6043）；swap 結構問題另需修復
- 環境承壓但**數字有效、可引用**：prod-new pp=302.56 （啟動 swap=2180; swap growth=+2735（環境承壓、需結構修復；數字本身仍有效））；swap 結構問題另需修復

## [watchdog 2026-09-25 12:05:49] 巡檢問題（看門人自動通知）

- 環境承壓但**數字有效、可引用**：prod-new pp=240.52 tg=11.03 （啟動 swap=6043）；swap 結構問題另需修復
