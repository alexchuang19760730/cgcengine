# S²-MoE 能不能解決本專案的 MTP verify 成本？（2026-09-24）

**性質**：靜態評估 + 定量門檻（0 GPU，0 重建）　**對象**：`arXiv 2608.15018` / `github.com/angerybob/S2-MoE`
**結論一句話**：**它的 draft 機制不能用（會把我們的 accept 從 3.117 打到 ~2.50），
但「reuse-aware gating」那一條是對題的 —— 只是它在我們這裡**拿不到輸入**。
真正該抄的不是 S²-MoE，而是同源的 **AcceptMoE**（按快取駐留限制 expert 資格）。

---

## 1. S²-MoE 的三個技術，逐條對表

來源：README（`orin` 分支）的代表命令行與機制說明。

| 技術 | 它做什麼 | 在本專案的適配 |
|---|---|---|
| **① Routing-aware adaptive expansion** | 只在「預測的接受收益 ≥ 專家傳輸成本」時才擴大 draft 樹；成本模型 `--prune-expert-bytes 0.571875`（每專家 0.57 MB）/`--prune-bandwidth 3.66`（GB/s） | **對題但要重寫**：我們用 MTP，沒有「draft 樹」可擴。等價物是「動態決定 draft 深度 k」—— 而 `M-W` 已判 ~+1.7%、低於 3% 門檻 |
| **② Reuse-aware gating**<br>`--moe-reuse-expert-cap 18 --moe-reuse-runtime` | **引導 target 的 verification 走向 draft 階段已激活的專家** | ★ **對題，但見 §2：我們沒有這個輸入** |
| **③ Context-aligned self-speculation**<br>`--draft-share-kv`、`--model-draft = TARGET`、`--draft-expert-topk 2` | draft 就是 target 自己，但**每層只跑 top-2 專家**（target 是 top-8），共享 KV | **不適用**：見 §1.1 |

### 1.1 為什麼 ③（self-draft）在本專案是退步

README 的實測表給了 `effective acceptance length`（**含 target 自己生成的那個 token**，
與我們「每 step 3.117 token」同一口徑）：

| 方法 | OLMoE/32GB | DeepSeek/64GB |
|---|---:|---:|
| DFlash | 3.15 | 3.08 |
| Domino | 3.71 | 3.29 |
| **S²-MoE** | **2.56** | **2.50** |

**我們現有 MTP 的 accept = 3.117 > S²-MoE 的 2.50~2.56（高 22~25%）。**

⇒ S²-MoE 的 self-draft（top-2 專家）**接受率比我們現有的 MTP draft head 低**。
它的 1.87×/1.60× 是**相對 target-only 自回歸**算出來的，不是相對「已經有 MTP 的我們」。
⇒ **拿它替換 MTP 是拿 2.50 換 3.117，方向反了。**

---

## 2. ★ 真正的阻斷：② 的輸入我們沒有

reuse gating 的輸入是「**draft 階段已激活的專家集合**」。這個集合只在
**draft 真的跑過 target 的 MoE 層**時才存在 —— 也就是 ③ 的 self-draft。

而 llama.cpp 的 `--spec-type draft-mtp` 用的是**獨立的 MTP draft module**，
draft 階段**不呼叫 target 的 expert FFN** ⇒ **沒有任何「已激活專家」資訊可 reuse**。

⇒ 三條技術互相綁定：想拿 ② 就得先上 ③，而 ③ 會把 accept 從 3.117 打到 ~2.50。

| 組合 | accept | 專家約束資訊 | 淨效 |
|---|---|---|---|
| 現況（MTP） | **3.117** | 無 | 基準 |
| S²-MoE 整套 | ~2.50 | 有（self-draft） | 待算（§3） |
| **現況 + AcceptMoE 式資格約束** | **3.117（不改 draft）** | **不需要**（用駐留狀態本身） | §4 |

---

## 3. 定量：這筆交換的盈虧平衡在哪

用本專案實測時間帳（`STEP_SERIALIZATION`，per-step）與 `steps/round = 1.55` 橋：

```
現況          total 161.0 ms → step 249.6 ms（對上錨點 247.98，差 0.7%）
              token/step 3.117                ⇒ 12.57 t/s
gap 44.4 的成分：cb 阻塞填池 25.0（56%）/ submit 11.3（26%）/ 截距 10.1（23%）
```

若專家資格被約束到「已駐留」⇒ miss → 0 ⇒ **cb 阻塞填池 25.0 ms 消失**：

```
per-step 161.0 − 25.0 = 136.0  ⇒ step = 136.0 × 1.55 = 210.8 ms
token 若仍 3.117               ⇒ 3.117 / 0.2108 = 14.79 t/s（+17.7%）
```

⇒ **省下的上限是 17.7%**。而 accept 是**乘性**作用在分子上：

| accept 損失 | 新 accept | ≈ t/s | vs 12.57 |
|---|---:|---:|---:|
| 0% | 3.117 | 14.79 | **+17.7%** |
| −10% | 2.805 | 13.66 | +8.7% |
| **−20%** | **2.494** | **12.50** | **≈ 打平** |
| −25%（S²-MoE 水準） | 2.338 | 11.9 | −5% |

> ⚠ 上表是**上界**：它假設 cb 25.0 ms 可以整段消失，且 GPU 時間不受影響。
> 實際上 verify 的 GPU 時間會隨 batch 縮小略降，但 `wait 123.2`（GPU 等 CPU）不會等比縮，
> 所以真實值會低於這一行。**不要拿它當預測，只能當門檻。**

**判據**：只要 accept 掉超過 ~20%，這筆交換就是負的。
而 S²-MoE 自己報的 accept 損失（3.117 → 2.50）正好是 **−20%** ⇒ **落在打平線上，沒有安全邊際。**

---

## 4. 真正該抄的是 AcceptMoE，不是 S²-MoE

搜尋時冒出的同源工作（`arXiv 2608.02989`，Shuang Liang 等）：

> "Under offloading, AcceptMoE conditions expert eligibility on **cache residency**
> instead of predicting natural routes and prefetching the corresponding expert weights."
> —— offloading 下達 **2.06×**（全在 GPU 記憶體時 1.29×），**host-to-device 流量減 73.6~77.1%**，
> 12 組 model-task 的平均精度只掉 **0.27 pp**。

這與我們本輪實測完全對上：

| | 做法 | 需要搬資料？ | 本專案實測 |
|---|---|---|---|
| ρ / prebind | 預測路由 → **預取**未駐留的專家 | **要** | **−38~51%**（搬不動，IO 10 MiB/s） |
| PIN_PROFILE | 靜態釘住固定專家集 | 不用（但失去彈性） | **−23%**（且 profile 由 `--prompt 0` 產生 = 洩題，不能泛化） |
| **AcceptMoE 式** | **按駐留狀態限制資格** | **不用** | **未測** ← 唯一沒被判死的一格 |

⇒ **ρ 的失敗根因是「要搬」**（`fill_wait` 佔牆鐘 46~56%，IO 被 swap 壓到 10 MiB/s）。
AcceptMoE 那條路**根本不需要搬** —— 它只是「不選那些不在池裡的專家」。
⇒ **它繞開的正是殺死 ρ 的那個約束**，而付出的代價是「分布改變」（0.27pp 精度）。

### 4.1 它與兩個已判死近親的差別（必須先說清楚，否則會被當成重做）

| | PIN_PROFILE（−23%） | A 預指派（h=0.373 判死） | **AcceptMoE 式** |
|---|---|---|---|
| 集合來源 | 固定 profile（**靜態、會洩題**） | 上一步 union（**跨 token**） | **當下駐留狀態**（同 step） |
| 預測口徑 | — | h = 0.373（**弱**） | 不需預測 |
| 失敗歸因 | 不能泛化 + 池失去 LRU 彈性 | 跨 token 路由局部性弱 | — |

⇒ 它不是「PIN_PROFILE 再做一次」。PIN_PROFILE 死在**靜態**；這一格是**逐 step 由駐留狀態決定**，
而我們已實測「同 step」的覆蓋率是強的那一側（`cov_uni = 0.824`，vs 跨 token 的 `h = 0.373`）。

---

## 5. 建議順序

```
0  環境：swap = 0（本輪已被實驗自己壓到 8308 MiB，q4/q8 一輪全 REFUSED ⇒ 必須重開機）
1  先做「只讀」的可行性檢查（0 重建）：從既有 log 算「top-8 中有多少比例不在池內」
     ⇒ 這個數 = 資格約束會改變多少專家的選擇 = accept 損失的代理量。
     若 > 30% ⇒ 分布改太大，直接停。
2  靜態模擬：把「只選已駐留專家」寫成離線評估（不進關鍵路徑），量 accept 與輸出差異
3  才決定要不要進代碼。進代碼時**必須帶開關**，且與 ρ 互斥（ρ 要搬、這個不要搬）。
```

**不要做的事**：不要因為「S²-MoE 報 5.3×/2.0×」就整套移植 ——
那兩個數字是相對**自回歸**，我們已經有 MTP（+28.5%），基準不同。

---

## 6. 來源

| 數字 | 來源 |
|---|---|
| S²-MoE 三技術、`--draft-expert-topk 2`、`--moe-reuse-expert-cap 18`、成本模型 `0.571875 / 3.66` | `github.com/angerybob/S2-MoE` README（`orin` 分支） |
| accept length：S²-MoE 2.50/2.56、DFlash 3.08/3.15、Domino 3.29/3.71（含 target token） | 同上，README 的 DFlash/Domino 比較表 |
| AcceptMoE：2.06× offloading、1.290× 全駐留、流量 −73.6~77.1%、精度 −0.27pp | `arXiv 2608.02989` 摘要（**2026-09-24 已讀原文覆核，數字無誤**） |
| 我們 accept 3.117、step 247.98 ms、12.57 t/s | `MEMORY.md`（交付 decode 權威讀數）／`prod_profile.py` |
| 時間帳 161.0 / gap 44.4 / cb 阻塞填池 25.0、`steps/round` 1.55 | `docs/STEP_SERIALIZATION_2026-09-23.md`、`ROUND_REMAINDER_IDENTIFIED` |
| `cov_uni` 0.824（同 step）、`h` 0.373（跨 token） | `docs/H_MEASURED_2026-09-24.md` |
| ρ −38~51%、PIN_PROFILE −23% | `docs/RHO_BATCH_REGRESSION_2026-09-24.md` §1、`2026-09-24.md` §EN-476/479 |
| MTP on/off +28.5%、M-W ~+1.7% | `MEMORY_PERF.md`（MTP／speculative 節） |

---

## 7. 待閉合

1. ~~AcceptMoE 原文未讀~~ **已閉合**（2026-09-24 讀原文摘要覆核，數字無誤）。
2. ~~「top-8 有多少比例不在池內」沒有實測值~~ **已閉合，而且它否決了這條路**：
   `masscov_base_2026-09-23.txt` 實測 **冷訪問 34.1%**（`cur`=0.6585），
   而靜態 top-96 就能覆蓋 **95.65%**、top-143 覆蓋 **98.88%** ⇒ **駐留落差 33pp**。
   AcceptMoE 「以駐留為資格」在我們這裡要丟掉 34% 的訪問，換 ≤+17.5%
   ⇒ **詳見 `docs/ACCEPTMOE_ADAPTATION_2026-09-24.md`，結論是先修駐留、暫不抄。**
3. swap 8308 MiB ⇒ 本輪任何新的 GPU 實驗都跑不動，**下一輪必須先重開機**。
