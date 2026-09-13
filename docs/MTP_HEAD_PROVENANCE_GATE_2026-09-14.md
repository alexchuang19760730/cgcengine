# MTP head provenance gate — 用「官方的 head」保證初始 accept rate

日期：2026-09-14 · 分支：`demo/sweet-spot-windows-fix` · 機器：MacBook Air M4 16GB
工具：`scripts/check/mtp_head_identity.py` · sidecar：`models/gguf/*.mtphead.json`

---

## 0. 策略（2026-09-14 定案）

**每個 base 只用它自己的 head，永不跨接；我們的工程是 decode framework，不是 head。**

理由是量出來的，不是偏好：MTP 的 accept rate 是 **(base, head) 配對**的性質，不是 head 單獨的性質。
`flashkv-devserver` 這側過去把「F16 儲存型別」當成 0% accept 的嫌疑犯，那是錯的方向——
本文件 §2 用同一套 gate 證明兩個 Edge0 MTP artifact 的 **型別表完全一樣**，而差別在權重本身。

四條不變量：

| # | 不變量 | 今天由什麼保證 |
|---|---|---|
| I1 | 每個 artifact 的 head 身分有記錄，且 gate 會拒比 | `*​.mtphead.json` sidecar + `check` |
| I2 | head 數值上是活的（router 沒有塌掉） | `fingerprint` 的 `degenerate` 檢查（§3） |
| I3 | 跨模型：head 由**語意**定位（`blk.<n>.nextn.*`），不是硬編碼層號 | `head_layer()`（§1 的註解說明為何不能靠 `block_count`） |
| I4 | 跨記憶體：head 的大小與 pool 無關 | head 是檔案內容，pool 只影響它的載入路徑 |

---

## 1. gate 是什麼

```
fingerprint --gguf G [--out J]     記錄 head 身分（sidecar）
compare A B [--sigma]              兩個 artifact 差在哪
check --gguf G --expect J          PASS / FAIL
```

head 的定義是**最高編號區塊的張量＋共享的 `output.weight`（draft 與 target 共用的 lm_head）**。
身分是 `name / ggml type / nbytes / sha256(raw bytes)` 的 sha256，所以**任何一個 head 張量變了就翻**，
包含「大小與型別都不變但位元組變了」的那種改動——那正是 0.81% 那次移植的形態。

> **一個必須記下來的陷阱**：這顆模型的 `qwen35moe.block_count` 是 **41**（blk.0..40，MTP 層被算進去），
> 所以 `head = block_count` 會走出界、只match 到 `output.weight`，產生一個**永遠 PASS 的閘門**。
> 第一次跑就踩到。因此 head 是**語意定位**：找帶 `nextn.*` 的那一層。而且若只 match 到 lm_head，
> 腳本會直接拒絕產出指紋，而不是給一個不會動的身分。

---

## 2. 事實：型別表分不出兩個 head

| artifact | head 身分 | 型別 | head 大小 |
|---|---|---|---|
| `Edge0-35B-Q4_0-MTP.gguf`（Nail head 移植） | `bc0cfecf…` | BF16=2 F32=7 IQ4_XS=7 Q2_K=2 Q3_K=1 **Q4_0**=1 Q8_0=1 | 575.8 MiB |
| `Edge0-35B-Q4_0-MTP-edge0head.gguf`（Edge0 自己的 head） | `aa16b5e9…` | **完全相同** | **完全相同** |
| `Nail-…-denseIQ4X.gguf`（Nail 的 head＋base） | `219b27c4…` | 同上但 lm_head 是 **Q6_K** | 700.8 MiB |

前兩者：**型別集合相同、張量數相同、總位元組相同**，身分不同。所以「打開型別表看它是不是官方 head」
在原理上就不可能——這正是需要 sidecar 的原因。

`compare --sigma` 另外把 head 的**無損張量**（BF16/F16/F32）解出來比：

| 張量 | 型別 | rel_l2 | d/σ |
|---|---|---|---|
| `blk.40.ffn_gate_inp.weight`（router） | BF16 | **1.00000** | **1.00003** |
| `blk.40.nextn.shared_head_norm.weight` | F32 | 0.36223 | **3.59094** |
| `blk.40.nextn.enorm.weight` | F32 | 0.01951 | 0.09324 |
| `blk.40.attn_norm.weight` | F32 | 0.01218 | 0.07163 |

判讀規則（腳本自己印）：`d/σ ≤ 0.005` → 同權重、只是精度不同；`d/σ ≥ 0.05` → **不同權重，不要移植**。

---

## 3. 這輪抓到的東西（兩個都在 `-edge0head` 那個 artifact 裡）

### 3.1 router 塌成零 —— 這是 0% accept 的直接機制

`Edge0-35B-Q4_0-MTP-edge0head.gguf` 的 `blk.40.ffn_gate_inp.weight`（MTP head 的**專家 router**，BF16 無損）：

| artifact | nonzero | min | max | std |
|---|---|---|---|---|
| `Edge0-35B-Q4_0-MTP`（Nail head） | 524288/524288 | −0.141602 | +0.125 | **0.00963** |
| **`-edge0head`** | 524287/524288 | **−3.26e−09** | **+9.31e−10** | **7.74e−12** |
| `Nail-…-denseIQ4X` | 524288/524288 | −0.141602 | +0.125 | 0.00963 |

**router 全部是 denormal 級的塵埃。** router 全零 ⇒ 每個專家的 logit 相同 ⇒ top-k 選擇退化（變成
依索引決定的固定集合）⇒ draft 的分佈與 target 無關 ⇒ accept 塌成 0。
**這比「不同訓練的 head」更根本：那個 head 在數值上是死的，任何 base 都拿不到 accept。**

這也解釋了先前觀測到的 `blk.40.ffn_gate_inp_shexp` 異常讀值（`−0.00002` vs Nail `123.27` 那個線索）。

### 3.2 `shared_head_norm` 少了 +1.0 的 norm shift

同一個 artifact 的 `blk.40.nextn.shared_head_norm.weight`：min = **−0.176758**，而另外兩個 artifact 都是
min = +0.829102。兩者相差恰為 1.0（0.829 − 1 = −0.171 ≈ −0.177），即 **B 存的是 npz 的 delta 慣例**，
而 loader（`create_qwen35_mtp_head`）在載入時會 **+1.0**——所以 GGUF 裡必須存**已 shift 的值**。
`merge_mtp_head_edge0.py` 的註解第 1 條講的就是這件事（`NORM_SHIFT_SUFFIXES`），但這個張量漏了。

### 3.3 反面：Nail head 的移植是位元完美的

`Edge0-35B-Q4_0-MTP` vs `Nail-…-denseIQ4X` 的**每一個**無損 head 張量：rel_l2 = 0.00000、d/σ = 0.00000
（含 router、`attn_norm`、`enorm`、`shared_head_norm`、`ffn_gate_inp_shexp`）。所以那個 artifact 確實
**逐位元組裝著 Nail 的 head**。它的 0.81% accept **不是轉換缺陷，就是 (base, head) 不匹配**——
與 `merge_mtp_head_edge0.py` 的結論一致，現在有了在 devserver 這側可重跑、不依賴 npz 的證據。

---

## 4. gate 現在會自己擋下來

```
$ python3 scripts/check/mtp_head_identity.py fingerprint --gguf models/gguf/Edge0-35B-Q4_0-MTP-edge0head.gguf
identity aa16b5e9…
head     blk.40.* + output.weight  21 tensors, 575.8 MiB
types    {'BF16': 2, 'F32': 7, 'IQ4_XS': 7, 'Q2_K': 2, 'Q3_K': 1, 'Q4_0': 1, 'Q8_0': 1}
DEAD     {'blk.40.ffn_gate_inp.weight': 'collapsed (peak |v| = 3.26e-09)'}
         a collapsed head tensor drafts tokens the base cannot accept
$ echo $?
1
```

`check --expect` 同樣會 FAIL（而且**先**報數值死，再比身分——身分相符但數值死是最危險的狀態）。

---

## 5. 還沒覆蓋的部分（要講清楚）

1. **accept rate 本身沒有閘門。** 引擎有遙測（`server-context.cpp` 的 `n_draft_total` / `n_draft_accepted`，
   以及 `draft_ratio` / `mean_acc_len` / per-position 表），但 harness 沒有任何一格去判它。
   §0 的「保證最初的 accept rate」目前只有**前置條件**（head 正確且活著）被閘住，**結果**沒有。
2. **MTP-on 沒有自己的 oracle cell。** 五格掃描全部是 MTP ON 的*引擎*，但 oracle 探針是單輪 chat；
   已知 `plain_match=False`（batch verify 與逐 token 解碼不一致）⇒ **oracle 綠不代表 verify 路徑綠**。
3. **只有一顆 base 被這樣驗過。** I3（語意定位）在架構上支援其他模型，但沒有第二顆模型實測過。
4. **`output.weight` 在兩顆 Edge0 artifact 都是 Q4_0**，而 Nail 載體是 Q6_K。這個 lm_head 是 draft 與
   target **共用**的，所以它的精度同時影響 target 品質——這條線沒有被量過。
5. **修好之後的重測順序**：先修 3.1（router）與 3.2（norm shift）→ 重建 artifact → 重新 fingerprint →
   量 accept → 才談 M4 的 60% 離開條件。**在 3.1 修好之前，任何 accept 數字都沒有意義。**

---

## 6. 用法（放進流程）

```bash
# 產生或更新 sidecar（換模型 / 換 head 時）
python3 scripts/check/mtp_head_identity.py fingerprint \
    --gguf models/gguf/<artifact>.gguf --out models/gguf/<artifact>.mtphead.json

# 要跑 MTP 之前先過閘
python3 scripts/check/mtp_head_identity.py check \
    --gguf models/gguf/<artifact>.gguf --expect models/gguf/<artifact>.mtphead.json

# 想知道兩個 artifact 到底差在權重還是精度
python3 scripts/check/mtp_head_identity.py compare A.gguf B.gguf --sigma
```

`fingerprint` 只讀 head 的 21 個張量（575.8 MiB），不是 18 GB 全檔，所以它可以放在每次啟動前的路徑上。
