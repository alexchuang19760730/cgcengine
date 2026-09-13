# Roadmap — prefill 250 t/s · decode 25 t/s · 每個里程碑維持 M1/M2 bit-identical

Date: 2026-09-13 · Branch: `demo/sweet-spot-windows-fix` · Box: MacBook Air M4 16 GB (internal
NVMe: 1404 MiB/s cold device, 7721 MiB/s page-cache hot — `MISS_PATH_COST_AUDIT_2026-09-13.md`)

## 0. 這份計畫要解的三面牆

| 牆 | 內容 | 現在的位置 |
|---|---|---|
| **W1 頻寬天花板** | `t/s = 頻寬 ÷ bytes/token`，bytes/token 由「層數 × top_k × per_expert」決定 | decode：Edge0-int4 **580 MB/tok**、Nail-IQ3 **368 MB/tok** |
| **W2 每 token 計算** | T=1 是 ~100 ms/token，與 I/O 幾乎無關 | MTP 額外成本的 84% 在這裡 |
| **W3 `cap × top_k ≤ usable slots`** | pool 用「縮小 expert tensor」實作 → ubatch 被鎖在 8 | union 只有 28.7/256 = 11% → 沒有 batching 紅利 |

**W3 是總開關**：它同時造成 prefill 無法批次、以及 MTP verify 拿不到紅利。所以 M1 是所有
事情的依賴來源，其他里程碑都不能繞過它。

## 0.1 每個里程碑都必須通過的「不變閘門」

每個里程碑結尾都要重跑，缺一不可：

1. **oracle M1**（bit-identical logits）**= 100%**，且與 M2 分開報告（`scripts/check/cgc_logits_oracle_compare.py`）
2. **oracle M2**（argmax 一致）**= 100%**
3. **oracle M3（top-k 集合一致）= 100%**。
   > **2026-09-14 修正**：這條原本只是「列出來」而沒有標準，而**每個里程碑自己的離開條件
   > 表只寫了「M1 / M2 = 100%」—— 所以 M3 實際上沒有歸屬，它會靜默不回歸。**
   > **本文件中所有離開條件表的「M1 / M2 = 100%」一律讀作「M1 / M2 / M3 = 100%」。**
4. template（closed scaffold）· 5. buffer-nil · 6. preflight（`knifeedge_matrix.py --gates-only`）
7. **pool-size sweep 4 / 6 / 8 / 10 GiB 全部 M1 = 100%**，而**參考 oracle 必須用當下的
   binary 重生**。sidecar 會因為 `llama-context.cpp` / `llama-expert-cache.cpp` 的 source
   digest 改變而主動拒絕比對 —— **這是正常運作，不是壞掉**；要嘛重生，要嘛
   `--allow-stale-oracle`，但**不可以**讓舊參考靜默通過。
8. **RSS 必須在 4 / 6 / 8 / 10 GiB 四個 pool 都量，且 slab/scratch 分開報**（見 §0.2）。
   原本只有「RSS @ 10 GiB = 9.08 ± 0.5 GiB」一條 —— 而 scratch 的成本**正是落在小 pool 上**
   （4 GiB 那格的 pool 只有 4096 MiB），只看最大的配置等於沒有閘到。

> **每個里程碑的 M1 失敗都優先於該里程碑的速度數字。** 這個專案的歷史就是「先有數字、後
> 發現數字是假的」，所以閘門先於產出。

---

## 0.2 新增閘門：scratch/slab 大小必須有模型上界（2026-09-14）

**為什麼要這條**：`4/6/8/10 GiB` 是 **pool 大小**的掃描，不是**模型**的掃描。原本整份計畫
沒有一條離開條件把 scratch/slab 的大小與**模型幾何**綁在一起 —— 而那正是「換一顆模型就
壞掉」的地方。以下是 2026-09-14 的實測，它顯示這個缺口有多大。

> **範圍**：本文件的主線是 **Mac（16 GB MacBook Air M4，內接 NVMe）**。手機（UFS + 12 GB
> RAM）是一條**獨立的裝置線，見 §0.3** —— 它不與 Mac 共用任何 gate，因為它的儲存頻寬、
> 「一個 process 真正可用的 RAM」、散熱都不同構。

**實測幾何**（`scripts/gguf_retensor.py list`，兩顆本 branch 實際跑的 GGUF，`blk.1`）：

| kind | 張量 | Nail IQ3_XXS 的型別 | Nail per-expert bytes | Edge0 Q4_0 per-expert |
|---|---|---|---|---|
| 0 | `ffn_gate_exps` | iq2_s / iq3_s / q2_K | 335,872 / 450,560 / 344,064 | 589,824 |
| 1 | `ffn_up_exps` | 同上 | 同上 | 589,824 |
| 2 | `ffn_down_exps` | iq3_s / iq4_xs（q3_K 同值） | 450,560 / 557,056 | 589,824 |
| 3 | `ffn_gate_up_exps` | **兩顆都沒有** → 只有 3 個 kind 活著 | — | — |

**關鍵：Nail 是混合量化 —— 光是這一顆就有 8 個不同的 `(kind, stride)` 組合。** 所以「slab
多大」不是一個常數，它是**模型的函數**。任何「per kind 一個固定 size」的估算都會錯。

**slab 大小（C = 64，即 M1 的 `cap × top_k`）：**

| 設計 | Nail IQ3_XXS | Edge0 Q4_0 |
|---|---|---|
| 每 `(kind, stride)` 一個 slab | **199.5 MiB** | 108.0 MiB |
| **每 kind 一個 slab、以該 kind 的 max stride 定尺寸** | **89.0 MiB** | 108.0 MiB |

**採用後者。** 因為 `ggml_nbytes = union × 本層 stride ≤ cap × max_stride = slab 大小`，所以
小 stride 的層可以共用「為大 stride 配的」slab —— 不需要為每個 stride 各配一份。（這也
避免了一個真實的 bug：按 `(kind, stride)` 逐層重配時，舊 slab 被 free 而其他層的
`wt->buffer` 還指在裡面。）

**M2 的整層 slab（C = 256）：** Nail **366 MiB**；double-buffer 後 **732 MiB**。在 4 GiB 的
pool 上那是 **18%** —— 而它完全不在原本「±0.5 GiB @ 10 GiB」那條的視野內。

### 離開條件（每個里程碑都要報）

- 報出 `slab(C) = Σ_kind (C × max_stride(kind))` 的**實測值**，並與「裝置可用 RAM − dense −
  KV − OS − pool」比較，**在 4 / 6 / 8 / 10 GiB 四個 pool 都要過**
- **換任何一顆模型都要重跑這一條**（這就是讓「跨模型」變成可回歸的方式）
- slab 必須是 **lazy 配置**：預設 10 GiB pool 若不走寬 union 路徑，就應該配置 **0 bytes**
- slab 的**容量 C 必須是常數**，不得由 pool 或觀測到的 union 推導（否則 pass 邊界會隨 pool
  移動，M1 不變性直接垮——這是 M1 唯一的數值陷阱）

---

## 0.3 手機裝置線（獨立，2026-09-14 加入）

**這是一條獨立的裝置線，不與 Mac 共用任何 gate。** 手機與 Mac 在三件事上不同構，所以把
Mac 的 4/6/8/10 GiB 掃描套過來會得到錯的結論：(a) 儲存頻寬**與存取模式**、(b) 對一個
process 真正可用的 RAM 不是標稱值、(c) 持續負載下的散熱。

> **最重要的誠實標註：我們沒有手機可以量。** 本章所有數字都是**外推 + 廠商規格**，不是實測。
> 所以它們的形式是「**預測門檻**」—— 有了硬體之後要用**同一套 oracle 與計時器**去驗證或推翻。
> 任何一個都不得在沒有實測前寫進報告當結論。

### 0.3.1 裝置參數（區分「規格」與「實測」）

| 項 | 值 | 來源 |
|---|---|---|
| UFS 4.0 循序讀，**spec 峰值** | 4.2 GB/s | 廠商規格 |
| UFS 4.0 循序讀，**實機持續** | 1.5–2.8 GB/s | 第三方量測，**不是本機** |
| UFS 4.0 **4K 隨機**讀 | 一、兩百 MB/s 級 | 第三方量測 |
| 標稱 RAM | 12 GB | — |
| **一個 process 真正可用** | 顯著低於 12 GB（Android LMK / iOS jetsam） | **未知，必須在目標平台上量** |

**而我們的存取模式不是循序的**：per-expert 切片 1.136 MB（Nail）/ 1.769 MB（Edge0），一層
的 expert 散在檔案各處 → 落在「隨機」與「循序」之間。所以上表第二列（1.5–2.8 GB/s）
**是高估**；保守估計要往下修，而修多少只能實測。

### 0.3.2 幾何（由 GGUF 直讀，與 §0.2 同一套工具）

| 模型 | per-expert（全模型平均） | expert set（41 層 × 3 proj） | bytes/token @ top_k 8 |
|---|---|---|---|
| Nail IQ3_XXS | 1.136 MB | **11.92 GB** | **372.6 MB** |
| Edge0-35B-Q4_0 | 1.769 MB | **18.41 GB** | **580.4 MB** |
| edge0-8b（參考） | ≈1.33 MB | ≈4.09 GB | ≈250 MB |

> 校正：先前引用的 Nail「1.122 MB/expert、368.1 MB/token」是 **blk.1 單層**的值；全模型平均
> 是 1.136 MB、**372.6 MB/token**。Edge0 的 18.41 GB / 580.4 MB 經復測**正確**。

### 0.3.3 12 GB 手機能給多少常駐率 h

扣掉 OS + dense + KV 後，池子樂觀估 **6 GiB**（見 0.3.1 的警告，這個數字很可能偏大）：

| 模型 | 6 GiB pool → slots/layer | 佔 256 | 對應 hit（用 Mac 上實測的三點內插） |
|---|---|---|---|
| Nail IQ3_XXS | **138** | 54% | **~86%** |
| Edge0-35B-int4 | **89** | 35% | **~61%** |

（內插基準：71 slots/27.7% → 54.8%、143/55.9% → 88.6%、179/69.9% → 95.7–97.0%。這是 **Mac
上**的曲線；手機的池子相對工作集更小、還有降頻 → **實際會更差**。）

注：hit **高於** slot 佔比，因為 routing 本身就偏斜。這也意味著**用 slot 佔比估算會偏保守**。

### 0.3.4 預測門檻

| 目標 | Nail IQ3_XXS（h 86%） | Edge0-35B-int4（h 61%） | 對 1.5–2.8 GB/s 的裝置 |
|---|---|---|---|
| **decode 25 t/s** | 372.6 × 0.14 = 52.2 MB/tok → **1.30 GB/s** | 580.4 × 0.39 = 226.4 MB/tok → **5.66 GB/s** | Nail **⚠️ 壓在下緣**；Edge0 **✗ 不可能** |
| **prefill 250 t/s**（chunk 2048、union 飽和） | 11.92 GB × 0.14 / 2048 = 0.81 MB/tok → **0.20 GB/s** | 18.41 × 0.39 / 2048 = 3.51 MB/tok → **0.88 GB/s** | 兩颗 **✅ 都有餘裕** |

**反推需要的 h（Nail，decode 25）**：`372.6 × (1−h) × 25 = BW`

| 裝置有效頻寬 | 需要的 h |
|---|---|
| 1.5 GB/s（保守） | **84%** |
| 2.0 GB/s | 79% |
| 2.8 GB/s（樂觀） | 70% |

→ **6 GiB pool 的 ~86% 剛好壓過 84% 這條線**，但它是外推值 + 樂觀的 pool 假設 + 未計降頻
→ **這是擦邊，不是過關**。而 4 GiB pool（87 slots、h≈60%）需要 **3.73 GB/s** → ✗。

> 校正 2：M6 的表用 h=84% 算出 Nail decode 25 需要 **1.47 GB/s**；0.3.3 的 6 GiB 估算內插
> 到 h=86% → **1.30 GB/s**。兩者都在外推誤差內，也都是擦邊 —— 不要把它們當成兩個獨立證據。

### 0.3.5 結論：手機的形狀跟 Mac 不一樣

1. **手機的 prefill I/O 是過的，decode I/O 是擦邊或不可能** —— 跟 Mac 相反（Mac 是 decode 的
   容量牆）。手機的 prefill 反而比 Mac 寬鬆，因為 union 飽和下只需讀「非常駐的 14–39%」。
2. **35B 這一級在 12 GB 手機上做不到 decode 25 t/s**（Nail 是擦邊、Edge0 差 2–3.7×）。
   要 25 t/s 只有往 **8B tier** 走：expert set 4.09 GB 可以**整個**放進 12 GB → h ≈ 100%
   → 需求降到 ~0.1 GB/s。
3. **最大的槓桿是量化幾何，不是儲存速度**：per-expert 從 1.769 → 1.136 MB 就把 Nail 的
   decode 需求從 5.66 壓到 1.30 GB/s。**所以對手機線而言，M6 的優先度最高，不是 M1。**
4. **prefill 250 的真正牆在 compute，跟 Mac 一樣**（見 M3）。所以手機線**也要** M3，
   不會因為「手機比較慢」而繞過。

### 0.3.6 手機線的離開條件（與 Mac 的 gate 分開命名）

| 代號 | 條件 |
|---|---|
| **P1** | 在**真實目標裝置**上量到「一個 process 可用 RAM」，用它取代 0.3.3 那個樂觀的「6 GiB 池」假設 |
| **P2** | 量到 per-expert 讀取的**有效**持續速率（1.136/1.769 MB 切片，非循序），與 1.5–2.8 GB/s 的循序值**分開報告** |
| **P3** | `slab(C)` 在手機的可用 RAM 內（依 §0.2 公式；C=256 時 Nail 要 366 MiB、double-buffer 732 MiB） |
| **P4** | oracle **M1 / M2 / M3 = 100%**，用**同一套 gate 與同一個 binary**（不可因為換裝置就放寬） |
| **P5** | **持續負載（≥ 10 分鐘）**的速率與首次請求分開報告（散熱） |

**這五條沒有一條可以在 Mac 上替代量測。** 在拿到真實裝置之前，手機線的狀態是「**未量測**」，
不是「待驗證」—— 這兩個詞在報告裡的意義完全不同。

### 0.3.7 手機線的依賴排序

```
M6（換量化幾何）──► 手機線的最高槓桿
M1（pool/graph 解耦）──► 正確性前提（P4）+ slab
M2（prefill 整層串流）──► 讓最後那 14–39% 不再是瓶頸
M3/M4（decode compute）──► 手機同樣需要（P 線不能繞過）
```

---

## M0 — 先把量測能力建起來（前置，無功能改動）

**目標**：讓「decode 的 100 ms/token 花在哪」變成一個可回答的問題，並讓 M1/M2 harness
能覆蓋 `union > slots` 這個**目前完全沒有被測到的**區域。

### 工作項

1. **decode 逐層剖面**：把 T=1 的 100 ms/token 拆成 `dispatch` / `gather` / `eval-sync` /
   `attention` / `MoE-math`。沿用 `CGC-MTP-PERF` 的做法（stderr + 寫進 `run_server.sh` 的
   env allowlist，否則會被靜默丟掉），並在最終輸出一張「41 層 × 各項 ms」的表。
2. **`union-fit` 閘門語義變更**：現在它是硬性 PASS/FAIL，認證「pool path 不可能被觸發」。
   解耦之後 `union > slots` 必須是**合法**的（走串流），所以改成 `union-routable`：
   `union ≤ slots` **或** 串流路徑被走過。**同時新增一個 `union > slots` 的 M1/M2 測試案例**
   —— 今天這條路的數值等價性沒有任何證據。
3. **重生 8 GiB 參考 oracle** 並更新 sidecar（commit、cap、模型 sha、source digest）。

### 離開條件（數字）

- 一張表，各項相加 = 實測的 ms/token（誤差 ±5%）
- 新增的 `union > slots` 案例：**M1/M2 目前預期是 FAIL**，並把失敗的 step/層記錄下來 ——
  這是交付物，不是障礙
- 新參考 oracle 生效，4/6/8 GiB 對它 M1 = 100%

**投入**：1–1.5 天 · **風險**：低 · **依賴**：無

---

## M1 — 解耦 pool 與圖形（W3）★ 最關鍵

**目標**：expert tensor 不再被縮小；pool 變成**獨立的 cache buffer**，只回答「哪些位元組
已經在 RAM」；ubatch 寬度不再由 pool 大小決定。

### 工作項

1. **loader**：pool 改成獨立配置的 Metal buffer；expert tensor（或它的可用視圖）保持全尺寸
   256。
2. **graph 按 phase 分流，在建圖前決定**（依 request 的 token 數，**不看常駐**）：
   - `T ≥ T_prefill`（先用 512 當起點）→ **PREFILL GRAPH**：整層 slab，256 experts
   - `T ≤ T_decode`（16 = `floor(slots/top_k)` 的上界）→ **DECODE GRAPH**：compacted gather
3. **`cap` 不再是從 slots 推導的常數**：它退化成「decode graph 的寬度上限」。
4. **canonical gather order**：compacted 之後的 expert **按 expert id 排序**再進 batched
   matmul。**這是這個新架構唯一的 M1 陷阱**——slot index 是 pool 相關的，若累加順序跟著
   slot 走，換 pool 大小就會改變舍入，M1 直接垮。
5. 計數器：`resident bytes` vs `pool capacity`、per-layer slots、`phase`（prefill/decode）
   每請求各跑幾次。

### 離開條件

| 閘門 | 標準 |
|---|---|
| M1 / M2 @ 4/6/8/10 GiB | **100%**，**且包含 `union > slots` 案例** |
| `union-routable` | PASS（串流路徑被走過） |
| prefill chunk 2048 | 被接受，無 OOB、無 NaN、無 `buffer is nil` |
| RSS @ 10 GiB | 與現在（9.08 GiB）差 ±0.5 GiB |
| decode 速度 | **不得退步**（現在 9.3–10.5 t/s） |

**投入**：3–5 天 · **風險**：**高**（動到 loader 與 graph 建構，是這個 repo 最貴的兩處）
· **依賴**：M0

---

## M2 — prefill 走整層串流（W1 + W3）

**目標**：prefill 一次性 materialize 整層 slab（常駐的從 pool、其餘逐層順序串流），
且算 L 時讀 L+1。

### 為什麼這條路成立

2048 tokens × top_k 8 = 16384 次抽取分佈在 256 顆上 → **union 飽和到 256**。所以
prefill 的預取是**無條件且 100% 準確的，不需要預測**（prerouter 在 prefill 沒有角色）。
實測 cap=8 時 union 只有 28.7/256 = 11%，正是同一公式的另一端。

### 工作項

1. **整層 slab 讀取**：按 `(proj, part)` 對整層發批次 `preadv`，而不是 9 tensor × 145 expert
   各自一次（這是「245 MB/s → 823 MB/s」那條未完成的路）。
2. **double-buffer**：算 layer L 的 FFN 時讀 L+1（prefill 沒有依賴問題，所以可以無條件做）。
3. 量測：`bytes/token`、裝置 GB/s、`load_wall`、compute ms/token。

### 離開條件（Nail IQ3_XXS 幾何：11.92 GB expert set、1.122 MB/expert）

| 量 | 標準 |
|---|---|
| bytes/token @ chunk 2048 | **≤ 3.0 MB**（理論 2.79：只讀非常駐的 48%） |
| 裝置持續速率 | **≥ 1.0 GB/s** |
| I/O 與 compute 重疊 | 請求 wall < I/O + compute（證明有 overlap） |
| prefill t/s | 記錄實測值；**250 t/s 可達/不可達要誠實寫明** |
| M1 / M2 | **100%**（prefill 因為永遠 materialize 256，不變性是建構保證） |

**投入**：3–4 天 · **風險**：中（I/O 路徑已知可行：0.70 GB/s 需求 = 裝置的 48%）
· **主要未知**：compute 需要在 4 ms/token 內完成（見 M3）

---

## M3 — decode 的 compute 削減（W2）★ 最可能失敗

**目標**：找出 `100 ms/token → 40 ms/token` 的那 **2.4×**。

### 為什麼這一格是關鍵

- pool 從 143 → 179 slots 讓 I/O 少 3.7×，**decode 只 +9%** → I/O 不是瓶頸。
- MTP-on 的 profile：draft head 只佔多出來成本的 16%，**84% 是 verify 批量拿不到批量紅利**
  （T=1 是 92 ms/token、T=2.6 是 104 ms/token）。
- 那筆 run 的 pool hit 是 87.3%，而 `pread` 只用到 **8 條 worker 裡的 1 條** → 加預取沒用。

### 工作項（按 M0 剖面的結果排序，最貴的先動）

1. 消除 per-layer 的 `eval()` 同步
2. MoE gather 的融合
3. **batched-union gather**：一次收 union、一次 batched matmul（現在是 T 次各自的 gather）

### 離開條件

- **成功**：decode（MTP off）≥ **15 t/s**，且 M1/M2 = 100%
- **失敗**：寫一份「找不到路」的結論，**附上 M0 的剖面表當證據**。這個里程碑允許以否定
  結果結束，但**不允許用沒拆開的數字下去做決策**

**投入**：2–3 天 · **風險**：**高**（可能真的沒有 2.4×）· **依賴**：M0、M1

---

## M4 — MTP 拒絕取樣 + verify 真批次（W2）

**目標**：accept 從 19.9% 拉到 60–80%，並讓 verify 真的吃到批量紅利。

### 工作項

1. **拒絕取樣**：以 `min(1, p_t/p_d)` 接受、被拒時從殘差 `max(0, p_t − p_d)` 重抽。
   現在的規則是**精確相等**（`sampling.cpp`），所以 `accept = Σ p_t·p_d`，而這在
   `p_d` 是 one-hot(argmax) 時**最大** —— 這解釋了為什麼我們改 draft 取樣器量到 18.3% → 18.1%
   （**沒有改善，而且不可能改善**）。拒絕取樣是前置條件，不是替代品。
2. **batched-union gather 套進 verify**（M1 的副產品）。
3. 量 accept、tokens/forward、ms/forward 的三段拆解。

### 離開條件

- accept **≥ 60%**；**MTP-on ≥ MTP-off**（今天是 **1.8× 淨損失**）
- M1 / M2 = 100% **with MTP ON**（現在 `plain_match=False` 在四條臂上全部成立）

**投入**：2–3 天 · **風險**：中 · **依賴**：M1、M3

---

## M5 — prerouter 只當預取提示（可選，期望值低）

**前置結論（已量測）**：placement 天花板 **0.2pp**（mass coverage 90.3% vs 最佳 90.5%）；
`start_layer = 7` → owner layers 6..38，而 churn 最重的是 **layer 1/2（distinct 250–253/256）**
—— **最需要預測的層它沒有 head**。edge0 自己的成績：ON 有 1.5× 但品質 0.3、
PREFETCH_ONLY 12/12 bit-identical 但增益未證實、LRU 64 格 / 40 層 = 1.6 格/層 → prefill `hits=0`。

### 工作項

只做 **decode、只做 L+1 的 8 顆、只影響時間**（PREFETCH_ONLY 語義：predictions 永不決定
routing）。量的是 **overlap（ms）**，不是 hit rate。

### 離開條件

- overlap 增益 ≥ **10%** 才保留；< 10% **記錄為已排除的死路**並把旗標預設關閉
- M1 / M2 = 100%

**投入**：2 天 · **風險**：低（預期以否定結果結束）

---

## M6 — 換量化幾何（W1，不動引擎就有量級）

**目標**：per-expert 從 1.769 MB 降到 ~1.122 MB。**這一步同時降低「要讀的」和「要放的」。**

| | Edge0-int4 | 目標（IQ3 級） |
|---|---|---|
| 6 GB pool 的 slots/layer | 85（33% → hit ~61%） | **133（52% → hit ~84%）** |
| decode 25 t/s 需要的殘餘頻寬 | **5.66 GB/s ✗** | **1.47 GB/s ⚠️** |
| prefill 250 t/s 需要的殘餘頻寬 | 1.51 GB/s（103% of device） | **0.70 GB/s ✅** |

工具已在：`scripts/gguf_retensor.py`（`scripts/gguf_retensor_qctl.c` 的驅動）、
`verify_edge0_gguf.py` 的 LoRA byte-exact 檢查、APFS clone 的可逆備份。

### 離開條件

- `verify_edge0_gguf.py` 全通（含多層 LoRA 目標）
- M1 / M2 = 100%（換量化是**換參考**，不是換引擎 → 必須重建 oracle）
- 48 題用**新舊兩把尺**並排報告

**投入**：1–2 天 · **風險**：低（工具已存在）· **依賴**：無（可與 M1 平行）

---

## 依賴與排序

```
M0 ──► M1 ──┬──► M2 ─┐
            └──► M3 ──┴──► M4
M6（可平行）
M5（可選，最後，期望值低）
```

## 該先做什麼（如果只能做一件事）

**M1。** 其他每一件事都被 W3 擋住：prefill 無法批次、MTP verify 無法批次、`cap` 無法當
旋鈕。而且 M1 的離開條件裡有一項是「decode 不得退步」—— 它不會給你速度，但它是唯一能
同時解開兩個症狀的改動。

## 明確不要做的事

- **不要把 `CGC_POOL_MAX_TOKENS` 調大**當加速旋鈕：沒有紅利的是 pool path 本身，不是那個
  邊界；調大只是讓 guard 更容易踩到（4 GiB + cap 8 的 union 理論上界 64 > 52 usable slots）。
- **不要用 mmap 當低 RSS 手段**：現在 `--no-mmap` + 10 GiB pool 已經是 9.08 GiB RSS；
  mmap 會把我們放回 fault 路徑（同 bundle：fault 3.50 ms vs preadv 1.98 ms）。
- **不要在 prerouter 上先投資**：它蓋不到 churn 最重的層，placement 天花板 0.2pp。
- **不要引用未經重生的 oracle 數字**：sidecar 的拒絕是保護，不是故障。
