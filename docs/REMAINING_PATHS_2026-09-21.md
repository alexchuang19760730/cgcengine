# 25 t/s — 剩餘可行之路的清點（2026-09-21）

承 `docs/IO_PATH_AB_2026-09-21.md`（§EN-359）。使用者新增兩項約束後重算：

- **accept 不能用 `CGC_RN_ROUTING=1 + CGC_WCOLD_EN=1` 那一支**（25.9 t/s／98.7% accept，但
  quality 0.3，renorm mask bug）⇒ 那條速度路徑是量測假象。
- 09-05 的 accept 0.9974／27.71 t/s 同理（09-16 已標 `auditable=NO`）。

本文只列**在「不動模型／不降 bpw／16 GB／無損」下仍開著的路**，每條標上量級與狀態。

---

## 1. 已關閉（有實測，不要再投）

| 路 | 判死證據 |
|---|---|
| 關池／換 IO／駐留路徑 | `recommendedMaxWorkingSetSize = 11453 MB` < 模型 13026 MB ⇒ `CGC-METAL-FAIL status 5`。池是**讓這支模型跑起來的前提**，不是優化 |
| 縮池到 4 GiB | +3.1%，同場控制自己 +23% ⇒ 噪音內 |
| 減 miss | miss 1.91×（capacity 3.15×）而 t/s 10.60 vs 10.28 ⇒ **miss 不是驅動項** |
| 加大池 143→256 slots | 引擎自記 counterfactual 上界 ≤+2.5% |
| 降專家 bpw | 使用者排除（XXS 已是精度下限） |
| 換機器 | 使用者排除 |
| 有損路由提 accept | 使用者排除（quality 0.3） |
| kernel 融合／省 dispatch（G4） | 單次 dispatch 實測 0.0736%；集群 1 上界 5.3% < 門檻 12.8% |
| 加深 k | `a=0.537` 與 `m=0.474` 同階；**k→∞ 且 100% 接受 = 24.59 t/s**，兩個不現實條件同時成立仍 <25 |
| 把 `gap` 當獨立靶 | `gap` 只有兩成分：`cb` ＋ 每邊界 0.29–0.35 ms（40 段 ⇒ 地板 ≈12.5 ms）。本輪 `cb=41.8`、`gap=52.46`，差 10.66 ≈ 地板 ⇒ **gap 就是 cb，不是新靶** |

---

## 2. 還開著的路（按量級排序）

### ① M4 — draft 的採樣分佈從來不是 request 的（**無損、已實作、未測**）

這是本次清點最重要的發現，來自 `docs/M4_DRAFT_DISTRIBUTION_2026-09-20.md`：

- 交付跑 **temp 0.4**，但 draft chain 在 **init 時**從 `params_base.sampling` 建，而啟動行把它釘在
  **`--temp 0`（greedy）**。request 用自己的溫度採樣，**沒有任何人把 request 的 chain 交給 draft**。
- ⇒ **draft 是 greedy，target 是 temp 0.4 採樣。今天 accept = 0.537 有結構性原因。**
- `CGC_MTP_SAMPLER_PARITY` 就是為此存在，但 **no profile sets it ⇒ 此處跑過的每一個 MTP 臂都是 off**。
- `--spec-draft-p-min` 已 plumbed（`CGC_SERVER_MTP_P_MIN`），此前**每個臂都是 `p_min=0`**。

**上界**：交付 `k=3` ⇒ `mean_len ≤ 4.0`。對照可引用基線 12.57 t/s（step 268 ms 反推）⇒
**step 不變下 14.9 t/s**。

⚠️ **必須聯合判讀**：G6 已實測「+11% mean_len 但 +14% step ⇒ 淨 **−0.45 t/s**」。
accept 會同時推高 step。單看 mean_len 會被誤導（本線已因此錯兩次）。

### ② 並行 decode（吞吐口徑）—— 從未測過，量級最大

`F = 79.9 ms/step` 是**每 step 固定**、不隨序列數變的。若 25 指的是**服務吞吐**而非單流延遲，
batch 2–4 可攤薄它：dense 權重讀一次服務多個序列，只有專家部分隨序列數增加。

現有唯一相關實測是 `decode-up`（`-b 2048`）的 **5.32 t/s（HEAVY）**——那是把 decode 形狀
prefill 化，**不是**多序列並行。多序列並行在此 repo **零實測**。

### ③ 降低 `m`（每 draft token 邊際成本 0.474）

`m → 0` 且維持今天 accept ⇒ `step = 85.8 ms`、`mean_len = 2.611` ⇒ **30.4 t/s**。

但：09-05 的 `m ≈ 0` 極可能是那個 bug 的副產品（accept 0.9974 意味 draft 近乎平凡），
且 IO 實驗已排除「池／miss」這個機制。**m 的物理來源未歸因 ⇒ 目前沒有可執行手段**，
只能列為待研究，不能列為計畫。

### ④ 縮小 `S0 = 85.8 ms` 的非頻寬部分

頻寬下限 **9.76 ms/token（102 t/s）**，實測 85.8 ⇒ **8.8× 是開銷不是頻寬**。
但已證：不是 miss、不是池、不是 dispatch 數量。**未歸因 ⇒ 沒有可執行手段。**

---

## 3. 判決

**單流 decode 25 t/s：在「不動模型／不降精度／16 GB／無損」下沒有已知可執行路徑。**

- 最樂觀的現實路徑是 ① ⇒ **14.9 t/s**（且 G6 警告它可能淨負）。
- ② 是唯一量級足夠的路，但它把目標從「單流延遲」改成「服務吞吐」。
- ③④ 是空間最大的兩塊，但目前只有「有空間」沒有「有手段」。

**建議把目標改寫為 14–16**，或先由使用者裁定 25 是單流還是吞吐口徑。prefill 250 不受影響（281.51 已達成）。

---

## 4. 下一步（最便宜，命令已寫好）

跑 M4 的 A/B —— 它**無損、已實作、零建置成本**，是目前唯一能動的槓桿：

```bash
# 治療
CGC_MTP_SAMPLER_PARITY=1 CGC_SERVER_MTP_P_MIN=0.1 ./scripts/run_server.sh
# 控制 = 同一命令去掉這兩個變數
```

要求（`M4_DRAFT_DISTRIBUTION_2026-09-20.md` §3）：

- `mean_len` 與 `step` 必須取自**同一 round**（`scripts/check/mtp_ruler.py` 就是為此存在）；
- 交錯 ≥3 reps、同一 binary、哨兵認可的窗口；
- M1/M2/M3 oracle gate 只跑控制臂（draft 分佈改的是生成文本，不是 kernel；gate 的 probe 是 greedy，
  兩條 chain 在 greedy 下重合）。

**若 M4 也是淨負 ⇒ accept 軸封版，25 正式下架，目標定 14 封版。**
