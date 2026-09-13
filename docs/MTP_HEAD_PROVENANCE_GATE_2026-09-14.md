# MTP head provenance gate — 用「官方的 head」保證初始 accept rate

日期：2026-09-14 · 分支：`demo/sweet-spot-windows-fix` · 機器：MacBook Air M4 16GB
工具：`scripts/check/mtp_head_identity.py` · sidecar：`models/gguf/*.mtphead.json`

> **2026-09-14 修訂**：本文件初版把 `-edge0head` 的 `shared_head_norm` 判成「少了 +1.0 norm shift」，
> **那是錯的**。詳見 §3.2。真因（§3.1）已定位並修好，gate 現在 PASS。

---

## 0. 策略（2026-09-14 定案）

**每個 base 只用它自己的 head，永不跨接；我們的工程是 decode framework，不是 head。**

理由是量出來的，不是偏好：MTP 的 accept rate 是 **(base, head) 配對**的性質，不是 head 單獨的性質。
`flashkv-devserver` 這側過去把「F16 儲存型別」當成 0% accept 的嫌疑犯，方向錯了——
§2 用同一套 gate 證明兩個 Edge0 MTP artifact 的 **型別表完全一樣**，差別在權重與**位元組編碼**本身。

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
fix-encoding --gguf G [--apply]    修「宣告型別與實際編碼不符」（§3.1）
```

head 的定義是**最高編號區塊的張量＋共享的 `output.weight`（draft 與 target 共用的 lm_head）**。
身分是 `name / ggml type / nbytes / sha256(raw bytes)` 的 sha256，所以**任何一個 head 張量變了就翻**，
包含「大小與型別都不變但位元組變了」的那種改動——那正是 0.81% 那次移植的形態。

> **一個必須記下來的陷阱**：這顆模型的 `qwen35moe.block_count` 是 **41**（blk.0..40，MTP 層被算進去），
> 所以 `head = block_count` 會走出界、只 match 到 `output.weight`，產生一個**永遠 PASS 的閘門**。
> 第一次跑就踩到。因此 head 是**語意定位**：找帶 `nextn.*` 的那一層。而且若只 match 到 lm_head，
> 腳本會直接拒絕產出指紋，而不是給一個不會動的身分。

---

## 2. 事實：型別表分不出兩個 head

| artifact | head 身分 | 型別 | head 大小 |
|---|---|---|---|
| `Edge0-35B-Q4_0-MTP.gguf`（Nail head 移植） | `bc0cfecf…` | BF16=2 F32=7 IQ4_XS=7 Q2_K=2 Q3_K=1 **Q4_0**=1 Q8_0=1 | 575.8 MiB |
| `Edge0-35B-Q4_0-MTP-edge0head.gguf`（修好後） | `d29f7e9d…` | **完全相同** | **完全相同** |
| `Nail-…-denseIQ4X.gguf`（Nail 的 head＋base） | `219b27c4…` | 同上但 lm_head 是 **Q6_K** | 700.8 MiB |

前兩者：**型別集合相同、張量數相同、總位元組相同**，身分不同。所以「打開型別表看它是不是官方 head」
在原理上就不可能——這正是需要 sidecar 的原因。

`compare --sigma` 另外把 head 的**無損張量**（BF16/F16/F32）解出來比：

| 張量 | 型別 | rel_l2 | d/σ |
|---|---|---|---|
| `blk.40.ffn_gate_inp.weight`（router） | BF16 | **1.00000** | **1.00003** |
| `blk.40.nextn.shared_head_norm.weight` | F32 | 0.36223 | 3.59094 |
| `blk.40.nextn.enorm.weight` | F32 | 0.01951 | 0.09324 |
| `blk.40.attn_norm.weight` | F32 | 0.01218 | 0.07163 |

判讀規則（腳本自己印）：`d/σ ≤ 0.005` → 同權重、只是精度不同；`d/σ ≥ 0.05` → **不同權重，不要移植**。

---

## 3. 這輪抓到的東西

### 3.1 真因：`ffn_gate_inp` 的**位元組是 F16，標頭宣告 BF16**（已修）

`Edge0-35B-Q4_0-MTP-edge0head.gguf` 的 `blk.40.ffn_gate_inp.weight`（MTP head 的**專家 router**）：

| 讀法 | peak | std | nonzero |
|---|---|---|---|
| 依宣告的 **BF16** 解讀 | **3.26e−09** | 7.75e−12 | 524287/524288 |
| 把同一批位元組當 **F16** 解讀 | **0.167969** | 0.009695 | 524287/524288 |
| npz 權威值（`mtp.layers.0.mlp.gate.weight`） | 0.167969 | 0.009695 | — |

把位元組當 F16 解讀後與 npz 比：**rel_l2 = 0.000000、corr = 1.000000**。
所以**檔案裡的數字是對的，標籤是錯的**。因為 BF16 與 F16 都是 16 bit，`nbytes` 完全相同，
所以大小、張量數、偏移、sha256 沒有任何事情看得出來——唯一的症狀是**每個權重被乘上約 2^−26**，
router 因此變成常數（所有 expert 的 logit 相同）⇒ top-k 退化 ⇒ draft 分佈與 target 無關 ⇒ **accept 塌成 0**。

修法（`fix-encoding`）：把檔案裡既有的 F16 值**重新編碼成 BF16**（round-to-nearest-even），
**就地、同長度**寫回。因此沒有任何標頭欄位、偏移或檔案長度改變——artifact 的其他部分不可能被動到。
寫入前先驗「檔案在 `data_offset` 的位元組等於 reader 看到的位元組」，不符就拒寫；並用 **APFS clone** 備份
（19,853,282,144 bytes，磁碟用量 0，因為共用 block）。

修好後：peak 0.167969、std 0.009695，與 npz **rel_l2 = 0.000000**；新身分 `d29f7e9d…`；
`degenerate` = `{}`；`check` PASS。

### 3.2 我上一輪的誤判：那不是 norm shift（撤回）

初版把同一個 artifact 的 `blk.40.nextn.shared_head_norm.weight`（min = −0.176758）與 Nail 的
（min = +0.829102）相比，因為差距恰為 1.0，就判成「少了 +1.0 norm shift」。**錯。**

- 該張量在 artifact 裡的值與 npz 的 `mtp.norm.weight` **完全相同**（min −0.176758、max 2.9375、std 0.287555）。
- 而 loader 明確寫著**不 shift**：
  `src/edge0/mtp/mtp_head_qwen35.py:190` → `self.norm.weight = tensors["mtp.norm.weight"]  # mtp.norm 不需要 shift`
- 也就是說 **artifact 是對的**，那個 ~1.0 的差距純粹來自「Edge0 的 head 與 Nail 的 head 是不同訓練結果」。
- 更難看的是：`merge_mtp_head_edge0.py` 的註解**早就明文警告過這件事**——
  「a ~1.0 offset against a different head (e.g. Nail's) means the heads differ, not that this one is wrong.
  Doing exactly that produced a bogus +1.0 patch that survived a full diagnostic round on 2026-09-13.」
  我重犯了同一個錯。

**教訓寫進工具的行為**：§2 的 `d/σ ≥ 0.05 → 不同權重` 規則是對的，但**「不同權重」就是結論，
不該再從跨 head 的偏移去推論某個慣例 bug**。gate 只回報「這兩個 head 不同」，不診斷慣例。
（反過來，`degenerate` 不涉及另一個 head：它只問「這個張量自己讀起來是不是空的」，所以 §3.1 那種
缺陷它可以單獨斷定，這也是它抓到真因、而 `--sigma` 沒抓到的原因。）

### 3.3 反面：Nail head 的移植是位元完美的

`Edge0-35B-Q4_0-MTP` vs `Nail-…-denseIQ4X` 的**每一個**無損 head 張量：rel_l2 = 0.00000、d/σ = 0.00000
（含 router、`attn_norm`、`enorm`、`shared_head_norm`、`ffn_gate_inp_shexp`）。所以那個 artifact 確實
**逐位元組裝著 Nail 的 head**。它的 0.81% accept **不是轉換缺陷，就是 (base, head) 不匹配**。

---

## 4. gate 現在會自己擋下來（修好前後的實錄）

修好前：

```
$ python3 scripts/check/mtp_head_identity.py fingerprint --gguf models/gguf/Edge0-35B-Q4_0-MTP-edge0head.gguf
identity aa16b5e9…
head     blk.40.* + output.weight  21 tensors, 575.8 MiB
types    {'BF16': 2, 'F32': 7, 'IQ4_XS': 7, 'Q2_K': 2, 'Q3_K': 1, 'Q4_0': 1, 'Q8_0': 1}
DEAD     {'blk.40.ffn_gate_inp.weight': 'collapsed (peak |v| = 3.26e-09); bytes read as F16 give
          peak 0.1680 -- the declared type does not match the stored encoding (run fix-encoding)'}
         a collapsed head tensor drafts tokens the base cannot accept
$ echo $?
1
```

修好後：

```
$ python3 scripts/check/mtp_head_identity.py check --gguf models/gguf/Edge0-35B-Q4_0-MTP-edge0head.gguf \
      --expect models/gguf/Edge0-35B-Q4_0-MTP-edge0head.mtphead.json
expect d29f7e9d…
got    d29f7e9d…
PASS  head blk.40.* + output.weight, 21 tensors, 575.8 MiB, {'BF16': 2, 'F32': 7, ...}
```

`degenerate` 的訊息會**直接指出機制**（不是只說「塌了」），因為「塌了」會讓讀者去找「少的權重」，
而實際上是「錯的標籤」。

---

## 5. 還沒覆蓋的部分（要講清楚）

1. ~~accept rate 本身沒有閘門~~ → **2026-09-14 已補上**：`scripts/check/mtp_accept_ab.py` 先過 head 閘門，
   再從 HTTP `timings.draft_n_accepted / draft_n` 量每個 carrier 的 accept。實測（8 GiB pool，3 prompt × 96 tokens）：

   | carrier | head | accept | acc / gen | mean len | decode t/s |
   |---|---|---|---|---|---|
   | `nail`（Nail head ＋ Nail base） | `219b27c4…` | **70.39%** | 107 / 152 | 36.67 | 6.74 |
   | `edge0head`（修好後的 Edge0 head ＋ Edge0 base） | `d29f7e9d…` | **26.81%** | 37 / 138 | 13.33 | 3.14 |
   | `graft`（Nail head ＋ Edge0 base） | `bc0cfecf…` | **0.61%** | 5 / 822 | 2.67 | 1.18 |
   | `nail_nomtp`（同一 carrier，MTP OFF） | — | n/a | 0 / 0 | 1.00 | **8.87** |

   §0 的第 2 條（accept 是 (base, head) 配對的性質）在這張表上有了另一組獨立證據：graft 用的是
   **與 nail 逐位元組相同的 head**，只換 base，accept 就從 70.39% 掉到 0.61%。
   而 Nail 配對已達 70.39%，超過 M4 的 60% 離開條件；**該配對仍未過的是
   `MTP-on ≥ MTP-off`（6.74 < 8.87）**，與 accept 無關。
2. **MTP-on 沒有自己的 oracle cell。** 五格掃描全部是 MTP ON 的*引擎*，但 oracle 探針是單輪 chat；
   已知 `plain_match=False`（batch verify 與逐 token 解碼不一致）⇒ **oracle 綠不代表 verify 路徑綠**。
3. **只有一顆 base 被這樣驗過。** I3（語意定位）在架構上支援其他模型，但沒有第二顆模型實測過。
4. **`output.weight` 在兩顆 Edge0 artifact 都是 Q4_0**，而 Nail 載體是 Q6_K。這個 lm_head 是 draft 與
   target **共用**的，所以它的精度同時影響 target 品質——這條線沒有被量過。
5. **同一個編碼缺陷類別可能還在別處。** `fix-encoding` 今天只看 head 的 21 個張量。
   任何 F16/F32/BF16 的**成對同尺寸**誤標（F16↔BF16 都是 16 bit）都不會被大小檢查抓到；
   主幹的 753 個張量沒有做過這個稽核。
6. ~~量測順序~~：§3.1 已修、gate 已 PASS、accept 已量（§5.1）。剩下的就是 M4 自己那個
   `MTP-on ≥ MTP-off` 的缺口（verify 批次成本）與 `plain_match=False`（batch verify 的逐值一致性）。

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

# 宣告型別與實際編碼不符時（先看 dry run）
python3 scripts/check/mtp_head_identity.py fix-encoding --gguf models/gguf/<artifact>.gguf
python3 scripts/check/mtp_head_identity.py fix-encoding --gguf models/gguf/<artifact>.gguf --apply \
    --out models/gguf/<artifact>.mtphead.json
```

`fingerprint` 只讀 head 的 21 個張量（575.8 MiB），不是 18 GB 全檔，所以它可以放在每次啟動前的路徑上。
`fix-encoding` 的寫入是同長度就地寫，所以它不需要重排、不需要重建，也不需要動 disk 空間。
