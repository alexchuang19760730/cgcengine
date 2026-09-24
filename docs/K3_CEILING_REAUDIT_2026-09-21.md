# K3 上界複審：「72 個」到底是不是 dispatch？（2026-09-21 18:0x，純讀碼 + 重算，0 重建）

接 `docs/K2_CPY_AUDIT_2026-09-21.md` §8.5 留下的問題：
> 「K3 那個『最大集群 72 個』—— 若 72 也是圖節點計數而非實測 dispatch 數，5.3% 還要再打折。」

## 1. 原問題的答案：**72 是真 dispatch，不打折**

`72 = 24 × 3`（`ffn_moe_weights_sum` 的 SUM_ROWS、`_clamped` 的 CLAMP、`_norm` 的 DIV，各 24 次/步），
來自 `CGC_ELEMW_CENSUS`（`docs/G4_ELEMENTWISE_TARGETS_2026-09-20.md` §3.1）。

論據（純讀碼，三條都在同一條路徑上）：

| # | 事實 | 位置 |
|---|---|---|
| ① | 普查程式碼位於 `ggml-metal-ops.cpp:573-609`，**在 dispatch `switch` 之後**（`:565` 收尾） | 它數的是「已經發出去的那一次」 |
| ② | 它在 `ggml_is_empty` 過濾（`:68-74`）、`is_consumed` 跳過（`:232`）、`ggml_is_empty` 第二道閘門（`:238`）**全部之後** | 被濾掉的 empty 節點**永遠到不了這裡**（這正是 §8 判 60 個零位元組 CPY 為 0 的同一條路徑） |
| ③ | 旁邊的 `CGC_DISPATCH_CENSUS`（`:611-659`）**同時印 `dispatches=` 與 `nodes=` 兩欄** | 該儀器生來就區分兩者；G4 那張 AB 表裡「ADD 27/76（ON）vs 67/67（OFF）」就是 `dispatch/nodes` |

⇒ **72 不是圖節點數，是 dispatch 數。5.3% 不必因為「node vs dispatch」打折。**

---

## 2. 但順手挖到一個**方向相反**的問題：單價 0.0736% 的分母不可靠

這個才是真正該坐實的東西 —— 而且它讓 K3 上界**往上**，不是往下。

### 2.1 分母是怎麼來的

`Backup/metal_fusion_dispatch_cost_ab.py:140`：

```python
m = re.match(r"CGC-DISPATCH:\s+(\S+)\s+dispatches=(\d+)\s+nodes=(\d+)", line.strip())
...
"disp_total": sum(v[0] for v in disp.values()),
```

`Δ = 1119 − 1004 = 115`，`8.5% / 115 = 0.0736%`（`docs/G4_FUSION_ANSWER_2026-09-20.md` §2）。

### 2.2 問題一（可證）：這個 regex **同時吃匯總行**，所以 `disp_total` 大概翻倍

儀器每張圖會印兩種行（`ggml-metal-ops.cpp:632` 與 `:642`）：

```
CGC-DISPATCH: graph=1 dispatches=29 nodes=32 (fusion saved 3)     ← 匯總行
CGC-DISPATCH:   ADD                  dispatches=27     nodes=76   ← 逐 op 行
```

實測 regex 行為（兩種都 `MATCH`，`group(1)` 分別是 `graph=1` 與 `ADD`）：

```
MATCH ('graph=1', '1098', '1200')  <- CGC-DISPATCH: graph=1 dispatches=1098 ...
MATCH ('ADD',     '27',   '76')    <- CGC-DISPATCH:   ADD   dispatches=27 ...
```

而 `disp` 是以 `group(1)` 為 key 的 dict ⇒ **`graph=N` 自成一批 key**，
`disp_total` = `Σ(逐 op 行)` **＋** `Σ(匯總行)` ≈ **2 × 真實總量**。

⇒ 真實 `Δ ≈ 115 / 2 ≈ 58`，**單價 ≈ 0.148%/dispatch（不是 0.0736%）**。

### 2.3 問題二（可證）：普查裡的「graph」是 **segment**，不是一步

`docs/G4_DISPATCH_CENSUS_2026-09-20.md` §2 的原文自己就標了：

```
CGC-DISPATCH: graph=2 dispatches=1 nodes=1 ← ARGSORT（top-k 段）
```

「段」＝ 一個 command buffer（`ggml-metal-context.m:1471-1482`：`idx` 是**每個 cb 各自的局部索引**，
`idx == 0` 每顆 cb 都會觸發一次 flush）。而一個 decode step ≈ 31 段（「一個 decode segment 就是一層」）
⇒ **48 段 ≈ 1.5 步**，不是 1 步。

⇒ 那份文件寫的「一步 1098 次 dispatch」**口徑錯了**：1098 是 48 段的合計。
若按段數歸一，真實「省下的 dispatch/步」≈ `58 / 1.5 ≈ 38` ⇒ **單價 ≈ 0.22%/dispatch**。

### 2.4 問題三（原本就自承）：8.5% 是上界

`G4_FUSION_ANSWER` §2 已寫：OFF 臂被推到 thermal HEAVY ⇒ 8.5% 偏大 ⇒ 單價偏大。
方向與 2.2／2.3 **相反**。

---

## 3. 重算：集群 1（72 dispatch/步）的上界區間

| 分母口徑 | Δ dispatch | 單價 | **72 個值多少** |
|---|---:|---:|---:|
| 文件原用（未修正） | 115 | 0.0736% | **5.3%** |
| 只修雙算 | ≈58 | ≈0.148% | **≈10.6%** |
| 雙算＋段數歸一 | ≈38 | ≈0.222% | **≈16.0%** |
| 再扣 thermal 上界 | — | 偏小 | 往下 |

⇒ **集群 1 的上界不是 5.3%，而是「5%～16%，中央 ≈10%」區間的某處。**

### 這翻轉了什麼

`G4_FUSION_ANSWER` 的結論「**別為了 G4 去寫 300 行融合 kernel**」是建立在
「5.3% 連 12.8% 門檻的一半都不到」。若真值是 10～16%，**它可能貼近甚至越過那個門檻**。

⚠️ 但這不是結論，是**待測** —— 三個修正項各自都還沒被獨立坐實，
而且 2.2／2.3 往上、2.4 往下，**不能直接相乘也不能相加**。

---

## 4. 下一步（**0 重建**，儀器已在現有 binary 裡）

1. **修 regex**：`(\S+)` 要排除 `graph=`（或改成只接受已知 op name），並把匯總行與逐 op 行分開存。
   順便存一份 raw stderr（上次沒留，這是現在複查不了的根本原因）。
2. **重跑 2 臂**（`on,off`；`--sentinel-retries 2`），直接讀匯總行得到「dispatch/段」與「段/步」。
3. 用重算的單價重乘 72。

成本：約兩臂 × sentinel；風險：thermal 配額（09-20 四臂只跑成兩臂，**這是既有限制不是缺陷**）。

---

## 5. 本文弱點

1. 2.2 的「≈2×」是**程式碼＋regex 行為**推出來的，沒有 raw log 可對帳（上次執行沒存 stderr）。
2. 2.3 的「48 段 ≈ 1.5 步」用了「一層一段」的既有結論，**未在本輪實測段/步**。
3. 72 的「24/步」本身也是推斷：普查有 3000 行上限，`G4_ELEMENTWISE_TARGETS` §5 自承
   「每步」是 `raw ÷ 8` 的估計，而那個 8 是從 GLU 反推的。
4. 全部是既有資料的重算，**沒有新的量測**。
