# 「關掉 WorkBuddy 跑 10 GiB 一定跑得起來」—— 逐項查證（2026-10-07 17:1x）

## 0. 判詞：**不成立，而且不是因為 WorkBuddy 佔太多**

三項獨立檢查，全部指向同一個結論：**10 GiB 的 REFUSE 與 WorkBuddy 無關。**

## 1. 閘門的算式與它的輸入（`budget_gate.py:225-250`）

```
total = pool_mb + wired_mb + reserve_mb      預設 reserve = 1024
deficit = total − ceiling_mb                 ceiling = 引擎 recommendedMaxWorkingSetSize = 11453.25
```

`mtry10g.log` 的實際輸入與算式：
```
pool = 10240（宣告值，固定）
wired = 1350（起跑時的 pages_wired_mb）
reserve = 1024
total = 12614  vs ceiling 11453  ⇒ deficit 1160 MiB ⇒ REFUSE
```

★ **`pool` 是宣告值、`ceiling` 是引擎自報值，兩者都不受 WorkBuddy 影響。**
只有 `wired` 受影響，而且它是**整機的** wired —— 包含系統核心檔案與驅動，不只是某個 app。

## 2. ★ 關掉 WorkBuddy 能省多少 wired：要降 **1621 MiB** 才可能讓 10 GiB 過

```
要 PASS 需要 wired ≤ 11453 − 10240 − 1024 = 189 MiB
起跑時實際 wired            = 1350 MiB
現在（WorkBuddy 開著）wired = 1810 MiB（vm_stat 115830 頁 × 16 KiB）
```
★ **關掉 WorkBuddy 要讓 wired 從 1350 掉到 189 MiB —— 降幅 86%。**
★ **而 `free` 這一側不可能是它判拒的理由**：10 GiB 臂起跑時 `free = 8852 MiB`
（8 GiB 臂的 4538 的**近兩倍**）、`swap = 0`、`thermal = NOMINAL`、`memory_gate.ok = true`
⇒ ★★ **那趟的機器狀態比 8 GiB 那趟更乾淨，卻仍被 REFUSE。**

## 3. ★★ 反證：那個 wired 不是 WorkBuddy 造成的，是 baseline

8 GiB 臂（跑成了）的 wired 時間序列：
```
launch wired = 1479 MiB   llama_procs = 0
end    wired = 1358 MiB   llama_procs = 0
差異 = −121 MiB
```
★ 引擎在整趟裡對 wired 的影響是 **−121 MiB**（沒增加）。
⇒ 那 1479 MiB 是**起跑時就在的東西**（系統 + 驅動 + 其他 app），不是池也不是 WorkBuddy 單獨造成。
⇒ ★ 而且 **10 GiB 臂的 wired（1350）比 8 GiB 臂（1479）還低** —— WorkBuddy 在那兩趟的存在感
   遠小於 run 間的環境差異。**要把 wired 壓到 189 MiB，關掉一個 app 做不到。**

## 4. 那麼 REFUSE 的真正來源是什麼：★ 是 `pool` 這個**宣告值**本身

```
8 GiB :  8192 + 1350 + 1024 = 10566 ≤ 11453  ⇒ deficit −887（負） ⇒ 不觸發 REFUSE
10 GiB: 10240 + 1350 + 1024 = 12614 >  11453  ⇒ deficit +1160          ⇒ REFUSE
```
⇒ ★★ **差 1160 MiB 的唯一原因是池從 8192 變成 10240（+2048）。**
WorkBuddy 在算式裡的權重是 `wired` 的那幾百 MiB，**方向對但量級不夠**：
即使省掉 500 MiB，仍差 660 MiB。

⇒ ★ **這條閘在本機的語意**（`budget_gate.py:32-42` 註解明寫）：`deficit > 0` 不是「跑不成」，
而是「**池子必然有一部分在 swap 上**」。2026-09-30 登記 **IN-BUDGET 0 / 2629 場**，
`deficit` 在每一場皆正，**包含已認證的 11.703 錨點**。
⇒ ★★ **所以正確的讀法是：「10 GiB 的 deficit 使數字必然不可引用」，
不是「WorkBuddy 讓它跑不起來」。** 而 10 GiB 那一趟確實**連量都沒量**
（`refused_preflight=true`），所以我們沒有它的速度數字 —— **這才是真正的損失。**

## 5. ★ 若仍要硬跑（`--warn-only` 路線），會得到什麼
- 可以跑完（`BUDGET_GATE=warn` 只降級不擋），**但產物必帶 `oversubscribed` 標記 ⇒ t/s 不可引用**。
- 機制列（`CGC-SYNCFILL cold_filled`、`CGC-COLD-GUARD`）仍可讀 ⇒ ★ **能回答那個唯一的問題**：
  **2 GiB 更大的池會不會把 MTP-on 的 `cold_filled` 平台（今日 8 GiB 實測 ≈3.67/事件，384 事件）壓下去？**
- ⇒ ★ **這是關掉 WorkBuddy 唯一有價值的作用：讓 `free` 與熱態乾淨，
  提高機制列的可讀性 —— 但它不能讓 10 GiB 通過 REFUSE。**

## 6. ★★ 但算式給出一個**沒被試過**的位置：8.87 GiB

```
pool_max = 11453.25 − wired(1350) − reserve(1024) = 9079 MiB = 8.87 GiB
```

| 池 | total | 結果 |
|---|---:|---|
| 8 GiB（8192） | 10566 | ✅ PASS（餘裕 887 MiB） |
| **8.5 GiB（8704）** | **11078** | ★ **PASS（餘裕 375 MiB）** |
| 9 GiB（9216） | 11590 | ❌ REFUSE（超 137） |
| 10 GiB（10240） | 12614 | ❌ REFUSE（超 1161） |

★★ **`CGC_METAL_RESERVE_MB` 歸零也救不了 10 GiB**（已驗算：仍超 326 MiB），
而唯一的變數 `wired` 要從 1350 降到 189 MiB（**降幅 86%**）⇒ **沒有任何 env 旋鈕能救 10 GiB。**

⇒ ★ **已加入可跑的選項**：`--pool8g5-mtp`（8 GiB vs **8.5 GiB**，MTP on，2 臂 / ~12 分鐘）
＋ 測試卡 §2.5 新格 `pool8g5`（`expert_cache_bytes: 9126805504`，`cell_contract --selftest` **47/47**）。
**它回答 8 GiB 基線留下的唯一問題**：`cold_filled` 平台今天只降了 4%（183→176），
多 512 MiB 會不會讓它降？降 ⇒ 池是槓桿；不降 ⇒ 池在這個範圍內不是槓桿。**兩種答案都有用。**

## 7. 誠實邊界
- 本輪**未跑量測**，全部從 `pool10g_20261007_164641` 的產物與 `budget_gate.py` 讀出。
- ★ **「WorkBuddy 關掉後 wired 實際會降到多少」未實測** —— 我算的是門檻（189 MiB）與現值
  （1350 MiB），推不出實測降幅。但門檻本身已使結論成立：要降 86%。
- ⚠ 「WorkBuddy 佔 wired 的幾百 MiB」是量級常識，**本輪未量**；不過 §3 的證據顯示
  它在兩趟之間的差異是小量（1350 vs 1479），而 86% 的降幅遠超任何單一 app 的量級。
- ⚠ 8.5 GiB 那一格**沒有實測過**（新加，僅通過契約 selftest 與閘門算式）。