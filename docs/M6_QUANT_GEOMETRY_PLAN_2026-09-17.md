# M6 換量化幾何：怎麼做（2026-09-17 二版 —— **只改設定、不動模型**）

> 對象：`docs/ROADMAP_PREFILL250_DECODE25_2026-09-13.md` 的 **M6**。
> **決定（2026-09-17 13:0x）：M6 只做無損的那半 —— 不動模型位元。**
> 因此 §3 的模型路線**留作記錄但不執行**，採用的是 §4 的設定路線（同一份預算、零精度代價）。
> 全部數字為 2026-09-17 實測，未動任何模型檔；§4.3 有三組與引擎自報數字逐格相符的驗證。

---

## 0. 一句話

**M6 的頭條目標早已達成（而且超過）；真正剩下的價值是一個「綁定層」，而它可以在不改模型、
不重建、不動引擎的前提下拿回來 —— 用一條既有的 env 字串。**

出貨模型在現行預設 **10 GiB** 池下是 **179 slots/layer**（roadmap 的目標是 133）。問題在
`capacity = budget / (41 × per_slot)` **取所有層的最大值**，於是 **blk.39 一層厚、41 層付錢**，
而池實際只佔 **7.85 GiB／10 GiB（78.5%）**——**2.15 GiB 的預算買不到任何 slot**。

⇒ 讓每一層用自己的 cap（§4），同一份預算下 **37 個典型層 179 → 232（+29.6%）**，
池佔滿 99.7%，**不改任何模型位元、不需要重建、不需要 D5**（無 `src/` 變更）。

---

## 1. 先更正前提：roadmap 的 1.769 MB 是 **Edge0**，不是出貨模型

| | roadmap 寫的（M6 表） | 實測（prod25 實際載入的模型） |
|---|---|---|
| 模型 | Edge0-int4 | `Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf`（12.72 GiB） |
| per-expert | **1.769 MB** | **1.0703 MiB（典型層）／1.3906 MiB（綁定層 blk.39）** |
| 6 GB pool 的 slots | 85 | **107** |
| hit rate | 61% → 目標 84% | **143 slots 量到 86.7–86.8%**；**179 slots 量到 87.9%**（見 §4.5） |

`scripts/run_server.sh:411-419` 自己記著這件事：
```
capacity = budget / (41 layers * per_slot 1.465MB)   ->   8GiB = 143 slots/layer
… at 143 slots on Nail denseIQ4X, hit rate 90.8%
counterfactual: K=96 79.7%  K=128 87.7%  K=192 97.1%  K=256 100.0%
```
⇒ **「把模型壓成 IQ3 級」在載體換成 denseIQ4X 時已經做掉了。** 照 roadmap 字面再做一次，
是拿一個已經 86–88% 的東西去換 84%。

---

## 2. 實測幾何（41 層 × 256 experts）

```
blk.0–33, blk.35–37（37 層）  gate=IQ2_S up=IQ2_S down=IQ3_S   274.00 MiB/層  1.0703 MiB/expert
blk.34, blk.38                down=IQ4_XS                      300.00 MiB     1.1719 MiB
★ blk.39                      gate/up=IQ3_S, down=IQ4_XS        356.00 MiB     1.3906 MiB  ← 綁定
blk.40（MTP／NextN）           gate=Q2_K up=Q2_K down=Q3_K       278.00 MiB     1.0859 MiB
```

**公式（`llama-model-loader.cpp:1100-1191`）**：
```
per_slot = MAX over TRUNK layers only      ← il >= n_decoder（NextN/MTP）被跳過
capacity = clamp(budget / (n_layers_ALL × per_slot), 8, 256)
```
兩處容易寫錯、都已在工具裡更正：
- **NextN/MTP 層不參與 `per_slot` 的 MAX**（loader 的註解寫明了理由：否則主幹的池會取決於
  *head* 剛好怎麼存），但 **`denom` 仍然數它**。這條在出貨模型上「剛好」看不出來（blk.40 比 blk.39 薄），
  換一個**頭比主幹厚**的模型就立刻現形 —— 見 §4.3 的 Ornith。
- `per_slot` 是**每顆專家**的位元組，不是每層（每層要乘 256）。用錯了答案會大 256 倍，
  而它仍然長得像一個數字。

---

## 3. 模型路線：把綁定層拉平（**已否決 —— 保留為記錄**）

改動 = 3 個張量（blk.39 gate/up `IQ3_S→IQ2_S`、down `IQ4_XS→IQ3_S`），工具是
`scripts/gguf_retensor.py set-type`（dry run 預設、`--apply` 才寫、寫前備份、寫後 re-verify，
可逐位元組 `restore`）。§4.1 的三種情境：

| 情境 | per_slot | 6 GiB | 8 GiB | 10 GiB |
|---|---|---|---|---|
| 現況（綁定層 blk.39） | 1,458,176 B | 107 | 143 | **179** |
| 只修 blk.39 ⇒ 綁定層換成 blk.34／38 | 1,228,800 B | 127 | 170 | **213**（+19.0%） |
| 修掉全部 4 層「比典型厚」的 | 1,122,304 B | 140 | 186 | **233**（+30.2%） |

### 3.1 為什麼否決：**它會掉精度，而且掉的方向是「被校準判定值得花位元」的那幾層**

| 張量 | 現在 | 改後 | 每個權重的位元 | 變化 |
|---|---|---|---|---|
| blk.39 `gate`／`up` | `IQ3_S` 3.4375 bpw | `IQ2_S` 2.5625 bpw | **−25.5%** | 實質降 |
| blk.39 `down` | `IQ4_XS` 4.25 bpw | `IQ3_S` 3.4375 bpw | **−19.1%** | 實質降 |
| blk.34／38 `down` | `IQ4_XS` 4.25 bpw | `IQ3_S` 3.4375 bpw | **−19.1%** | 實質降 |
| blk.40 `gate`／`up` | `Q2_K` 2.625 bpw | `IQ2_S` 2.5625 bpw | −2.4% | 小，但**換家族** |
| blk.40 `down` | `Q3_K` 3.4375 bpw | `IQ3_S` 3.4375 bpw | 0 | **換家族**，大小相同 |

**而那幾層的厚度不是這條 repo 的選擇。** `scripts/gen_denseiq4x_tt.py` 的政策是
「250 個 dense Q6_K 張量 → IQ4_XS；`every OTHER tensor is pinned to its CURRENT type … →
byte-copy, bit-identical by construction`」⇒ **所有 expert 張量的型別是從上游 `UD-IQ3_XXS`
（Unsloth Dynamic）逐位元組繼承的**：blk.34／38 的 down、blk.39 的三個張量之所以厚，
是**一個逐張量重要性校準的決定**。拉平＝用全班平均覆蓋它 ⇒ 不是中性重排。

**⇒ 決定：不做。** 要用也是先用 48 題（`scripts/check/bare_48.py`）＋ PPL 證明沒有損失，
而且先確立「slots → t/s 有因果」（§4.5）。**§4 的設定路線沒有這個代價。**

---

## 4. 設定路線（**採用**）：per-layer caps —— 同一份預算、零精度代價

### 4.1 機制：`LLAMA_EXPERT_CACHE_LAYER_CAPS` 是**雙邊讀取**的既有開關

```
LLAMA_EXPERT_CACHE_LAYER_CAPS="start-end:cap;..."      例："0-33:232;34-34:212;…;40-40:256"
    ├─ loader：每個 expert 張量的 ne[2] = cgc_layer_cap(il, capacity)   ← 池區的寬度
    │           （llama-model-loader.cpp:1492／1502）
    └─ cache ：n_slots_l[l] 逐層填，所有 per-layer 向量按自己的 cap 配置
                （llama-expert-cache.cpp:3136-3142；llama-context.cpp:6666 的 wt->ne[2]）
```
- 解析語意（`cgc_layer_cap`，`src/llama.cpp/src/llama-expert-cache.h:511`）：分段用 `;` 或 `,`；**後面的段落覆蓋前面的**；
  **沒被覆蓋的層保留 `def`（＝當天的 uniform capacity）**。
- **`scripts/run_server.sh` 每次啟動都已經設它**（`:1804-1807`）：沒指定時是 `"40-40:256"`
  （只把 MTP 層釘在 256）。所以要覆蓋，只要 `CGC_SERVER_LAYER_CAPS="…"`。
- ⇒ **零程式改動、零重建、零 D5**（沒有 `src/` 變更）。這條路徑的存在本身就是 M1 工作項 4 的前置。

**規則（`gguf_pool_geometry.py --layer-caps` 實作）**：最大化 L，使
`Σ_trunk max(base, min(256, L // ps_il)) × ps_il ≤ budget − mtp_cap × ps_mtp`。
也就是**每層等位元組**，但**夾在 `base`（今天的 uniform 值）之上**。
夾這個地板不是保守，是必要：池路徑要求 top-k 的聯集放得進該層的 slots，而**slots 太少會靜默讀到空**
（Edge0 實測：33 slots ⇒ 585 次 `buffer is nil`、48 題 0/48）。所以這個改動是
**嚴格加法**：薄層變多、厚層不動。

### 4.2 結果（`--layer-caps` 直接印出來的字串）

| 預算 | 今天 uniform | 今天池實際 | **典型 37 層** | blk.34／38 | blk.39（綁定） | 改後池 |
|---|---|---|---|---|---|---|
| 6 GiB | 107 | 4.80 GiB（80.0%） | **136（+27.1%）** | 125 | 107（不動） | 5.96 GiB（99.4%） |
| 8 GiB | 143 | 6.32 GiB（79.0%） | **184（+28.7%）** | 168 | 143（不動） | 7.97 GiB（99.6%） |
| **10 GiB（預設）** | **179** | **7.85 GiB（78.5%）** | **232（+29.6%）** | 212 | 179（不動） | **9.97 GiB（99.7%）** |

```
6 GiB : CGC_SERVER_LAYER_CAPS="0-33:136;34-34:125;35-37:136;38-38:125;39-39:107;40-40:256"
8 GiB : CGC_SERVER_LAYER_CAPS="0-33:184;34-34:168;35-37:184;38-38:168;39-39:143;40-40:256"
10 GiB: CGC_SERVER_LAYER_CAPS="0-33:232;34-34:212;35-37:232;38-38:212;39-39:179;40-40:256"
```

**「今天池實際」是算出來的，不是猜的**：40 層 × base × ps_il ＋ 256 × ps_40。
`resident` 的引擎自報值與它相符（§4.3）。

### 4.3 端到端驗證：三組與引擎自報數字逐格相符

引擎在初始化時就印自己的普查，**所以驗證是一行 grep**：
`LAYER_CAPS per-layer caps: total N slots (avg A/layer, min M/layer)`（`total = Σ caps`）。

| 模型 | 預算 | 引擎 `n_slots` | 工具 | 引擎普查 | 工具的預測 |
|---|---|---|---|---|---|
| Nail | 8 GiB | **143** | 143 | `5976 / 145.8 / 143` | — |
| Nail | 10 GiB | **179** | 179 | `7416 / 180.9 / 179` | — |
| **Ornith** | 2 GiB | **29** | **29** | `1416 / 34.5 / 29` | **1416 ✓** |
| **Ornith** | 4 GiB | **59** | **59** | `2616 / 63.8 / 59` | **2616 ✓** |

（`total = 40×base + 256`：143→5976、179→7416、29→1416、59→2616 ✓。日誌：
`llama_server_20260917_105409.log`（179，`resident=7990.71 MiB`）、`…_114123.log`（143，
`resident=6430.6 MiB`）、`…_20260914_032706/033111.log`（Ornith 29／59）。）

**★ Ornith 這一列是工具的一個真 bug 被抓出來的證據。** 該模型的 MTP 層是 `Q8_0`（比主幹厚），
所以「MAX over **所有**層」會得到 `per_slot = 3,342,336 B` ⇒ 2 GiB 只有 **15** slots，
**而引擎實際是 29**。修正成「MAX over **TRUNK**」之後兩者相符。出貨模型上這個 bug 完全看不出來
（它的 blk.40 比 blk.39 薄）—— **一個只在單一 artifact 上驗證過的公式，會把那顆 artifact 的偶然編碼進去。**

第二道讀數：`final stats` 的 `resident=` 與工具的「池實際」相符（179 → 7990.71 MiB vs 7.85 GiB；
143 → 6430.6 MiB vs 6.32 GiB）。

### 4.4 Runbook（**不需要建置**）

> **名詞（2026-09-17 補）**：本節的「第一道…第五道」＝ §4.7／§4.8 的「閘門 1…5」，
> 指的是同一組東西 —— **交付閘門**，每一道是一個「能不能出貨」的判準。
> 它與**起跑檢查**（`lsof -nP -iTCP:8080 -sTCP:LISTEN` ＋ `ps` 看別條線的量測行程，
> 任一非空就 `exit 3`）**不是同一件事**：後者是「不要踩到別人」的資源紀律，不是交付標準。
> 兩者都叫「閘門」會讓人把「閘門擋下了」誤讀成「交付閘門紅了」。

```sh
M=models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf

# 0) 算字串（純讀標頭、不吃 GPU；可以在別人的量測跑著時做）
python3 scripts/check/gguf_pool_geometry.py --layer-caps          # 6/8/10 GiB 三條

# 1) 起 server（預設 profile 就是 10 GiB 池；把字串蓋上去）
CGC_SERVER_LAYER_CAPS='0-33:232;34-34:212;35-37:232;38-38:212;39-39:179;40-40:256' \
  bash scripts/run_server.sh

# 2) 第一道：引擎自己的普查行必須等於工具印的「expect in the log」
grep "LAYER_CAPS per-layer caps" "$LOG"     # 10 GiB 應為 total 9443 (avg 230.3, min 179)
grep "resident=" "$LOG"                     # 應約 9.97 GiB（今天 7.85）

# 3) 第二道：正確性 —— 這個路徑的失效模式是「靜默讀到空」，不是崩潰
python3 scripts/check/bare_48.py …          # 48 題；今天的基準是 9/48（cap 8）

# 4) 第三道：數值閘門（換池幾何**不該**動到 logits）
python3 scripts/check/m123_oracle_gate.py --tag m6-layercaps   # 先讀 comparable，M1/M2/M3 應 9/9

# 5) 第四道：hit%（同一輪負載才可比）——讀 final stats 的 hit rate
# 6) 第五道：t/s —— 交錯 A/B ×3（有／無字串），同 build、md5 對齊
RUN_REPLAY_BENCH=0 python3 scripts/check/decode_sweep.py --profile prod25 \
  --arms p25-gputime,p25-slotgpu … --rounds 3 --warmup 0 --n-predict 24
```

### 4.5 風險與界線（都不是理論風險）

1. **逐層變動的 trunk caps 從未被量測過。** 用過的只有 `1-39:32`（全班 32）與 `40-40:256`
   （只有 MTP 不同）。解析器支援、向量是按層配置、`wt->ne[2]` 也是逐層給的 —— 但「支援」不等於
   「量過」。**先跑第 2–4 道再談速度。**
2. **地板不能拆。** slots 低於今天的值就有「聯集放不下 ⇒ 靜默讀空」的風險；工具的規則已經把它夾住
   （薄層只升不降），所以 §4.2 的三條字串**每一層都 ≥ 今天**。
3. **MTP 的 cap 與預算無關。** `40-40:256` 是固定 256 個 slot，不管預算多大、頭有多肥：
   Ornith 的 `Q8_0` 頭 ⇒ `256 × 3,342,336 B = 0.80 GiB`，**佔 2 GiB 預算的 40%**
   （工具在 2 GiB 那格會印出「連地板都裝不下、今天的形狀已經超支 118.5%」）。
   要調就 `--mtp-cap` 算，但那是另一個交換（MTP accept 率）。
4. **hit% 不是 slots 的函數而已。** 真實 log 的分佈（同一台機器、同一支模型、不同輪次）：
   179 slots → 83.7–97.0%（29 輪）／143 → 0.0–99.4%（430 輪，負載混雜）／71 → 44.9–82.8%（45 輪）。
   最可比的一對：**179 → 87.9%**（10:54，18.9 萬次請求）vs **143 → 86.7%**。
   ⇒ **在這個區間，負載比 slots 更能決定 hit%**。+29% slots 的收益**要量，不能推**。
5. **M6 不是 25 t/s 的路。** 池容量買的是 hit%，而「加大 pool／提高 hit rate 是槓桿」已在 09-15
   被推翻（09-05 的 **27.71 t/s** 是 **71 slots／4 GiB** 跑出來的，今天 143 slots 只有 6.5–10.4）。
   ⇒ **先確立「slots → t/s」有因果，再談值不值得**；M6 的價值是「同預算下少讀 I/O」。
6. **§4 是無損的**：不改模型、不改 `src/`、不重建 ⇒ **不需要 D5 才能提交**（但第 4 道仍然要跑，
   因為它驗的是**引擎行為**，不是「有沒有改到 src」）。

---

## 4.6 會快多少？（**預估，不是量測** —— 附上界與否證設計）

**結論：decode ＋1～3%（上界 ~3%，且無法排除 0）；prefill ＝ 0，而且是「可證的 0」。**

### decode：機制只有一條，量級由三個**已量到**的數字決定

| 輸入 | 值 | 出處 |
|---|---|---|
| 每層的**請求工作集** | **≈149 experts**（原文並註明「這不是淘汰政策的產物」） | `docs/MISS_PATH_COST_AUDIT_2026-09-13.md` |
| 現行池（8 GiB／143 slots） | **143 < 149**，而池**已經飽和**：`resident` 6430.6 MiB ÷ 1.07 MiB ≈ **150/層** | `llama_server_20260917_114123.log` |
| 錯過的組成（連貫文本） | **51% compulsory ／ 49% capacity** | skill `cgc-decode-attribution` 的實測分解 |
| 錯過佔抓取的比例 | ~12–13%（179 slots：hits 166201／misses 22883） | `final stats` @10:54 |

⇒ **capacity miss ≈ 0.49 × 12.5% ≈ 6% 的抓取。** 而 §4.2 的 caps 在 **8 GiB→184**、
**10 GiB→232**，兩者都 **≥ 149** ⇒ **「工作集塞不下」這一類應該歸零**，位元組約 **−6%**。
若 IO 與 compute 是序列化的（稽核 Verdict 3：IO ≈ 請求牆的一半）⇒ **≤ −3%**。
換算到現行暖平台 **10.78–10.91 t/s**（`prefill250+SPAC=1`、d512、MTP=0）：**→ 約 10.9–11.2 t/s**。

**★ 兩個界線、兩個可否證的預測**

1. **上界是 3%，不是 1.8×。** 1.8× 是「IO 與 compute 的序列化」（稽核 Verdict 3 的天花板），
   那是 **D3／M3** 的事，不是 M6。**M6 買的是「少讀」，不是「重疊」。**
2. **下界 0，而且這不是免責條款**：decode 快路徑 `fill_wait = 0`（填槽由背景執行緒做，關鍵路徑不等待）
   ⇒ 一部分重讀本來就被重疊掉了。**所以不能拿 6% 位元組直接乘。**
3. **預測 A（可否證）**：**收益不隨 slots 成長。** 工作集一旦塞得下（149），**8 GiB(184) 與 10 GiB(232)
   的 decode 應該一樣快**。若 10 GiB 明顯快於 8 GiB，機制就不是這裡寫的這一條。
4. **預測 B（可否證）**：`final stats` 的 **`misses` 應顯著下降、`file_reads`／bytes 降幅 ≈6%，
   而 compulsory 那半不動**。若 misses 不降 ⇒ 機制被否證，那就只剩 hit% 的裝飾效果。

### prefill：0 —— 而且是「沒有機制 ＋ 量不出來」的 0

- **沒有機制。** 池路徑的閘是 `n_tokens <= cgc_pool_max_tokens()`（預設 **8**，`llama-graph.h:18-37`），
  而 prefill 的 chunk 是 **2048／5632** ⇒ 它走**全張量／整層串流**那條路。**池的 slot 數碰不到 prefill。**
  而 `run_server.sh:1804-1807` 給 prefill250 的 `40-40:256` 只釘 MTP 層 ⇒ **改 trunk 的 caps 對它無效**。
- **有反向證據**：SPAC 那輪讓池**少讀 3.9% 位元組**，prefill **「不買也不付」**（skill §3）。
- **而且現行協定解析不到這麼小的效果。** prefill 的**自身變異**遠大於任何可能的效果：
  COLD ≈ **254–290** vs HOT ≈ **167–200**、`ABAB` 的**位置 1 溢價 +72.50 t/s**、
  11 次獨立啟動只有 **1 次**越過 250。⇒ 要談 prefill 得先有 **COLD-STATE（≥1800 s 靜置）＋丟棄臂**；
  在那個協定下，這個改動的效果（若有）仍在解析度之下。
  **正確的述句是：沒有機制預測它會變，且現行協定解析不到這麼小的效果。**

### 這張預估要成立，需要什麼

| 前提 | 現在的狀態 |
|---|---|
| 工作集 ≈149 experts 在**這個 profile／這個負載**下仍成立 | **舊量測**（09-13，67+32 token）⇒ 第 5 道時一併重讀 |
| caps 真的被套用（`min 179/layer` → `min 179、avg 230`） | **未驗**（§4.4 第 2 道） |
| 換池幾何不動數值（否則 M1 先紅） | **未驗**（§4.4 第 4 道） |
| 不得拿 prefill 的變異當 decode 的證據 | skill `cgc-decode-attribution`：兩個 decode 儀器不可並排、prefill 的 profile 不可當 decode 標準 |

---

## 4.7 實測結果（2026-09-17 13:2x）：**兩個真缺陷 ＋ 一個新的取捨**

### ① 第一輪：閘門 1 顯示 caps **完全沒生效** —— 找到並修掉一個真 bug

第一輪用 `--profile prod25 --arms p25-gputime`（＝解碼基準臂）帶 caps 跑，引擎卻印
`n_slots=179` 且**完全沒有 `LAYER_CAPS per-layer caps` 普查行**。零成本 dump 定位：

```
CGC_SERVER_MTP=1 -> LLAMA_EXPERT_CACHE_LAYER_CAPS=0-33:232;34-34:212
CGC_SERVER_MTP=0 -> (空)
```

**根因：`run_server.sh` 把 LAYER_CAPS 的 push 放在 `if [ "$SERVER_MTP" = "1" ]` 區塊內**（原 1804-1808）。
而 **解碼基準臂 `p25-gputime` 是 `CGC_SERVER_MTP=0`**（`decode_sweep.py:273`）⇒
**這個旋鈕剛好在我們要量它的那一臂上被靜默丟棄**，而「沒設到」與「沒有效果」**同形**。
（唯一的救援是引擎自己那行普查 —— 它的一行 grep 就能分辨兩者。沒有它，這一輪會被寫成「M6 無效」。）

**修法（`scripts/run_server.sh`，3 行，**不需要重建**）：把 push 移出 MTP 區塊，並保留既有預設：

```sh
if [ -n "$SERVER_LAYER_CAPS" ]; then
    SERVER_ENV+=(LLAMA_EXPERT_CACHE_LAYER_CAPS="$SERVER_LAYER_CAPS")
elif [ "$SERVER_MTP" = "1" ]; then
    SERVER_ENV+=(LLAMA_EXPERT_CACHE_LAYER_CAPS="40-40:256")
fi
```
**四種組合實測**：MTP=1 無 caps → `40-40:256`（與修前相同）；MTP=1 有 caps → 生效；
MTP=0 無 caps → 空（與修前相同）；**MTP=0 有 caps → 生效（＝新增的能力）**。
⇒ **對所有既有配置逐位元等價，只有「MTP=0 帶 caps」這一格從無效變成有效。**

### ② 第二輪：閘門 1 **通過**，算術逐格相符

```
LAYER_CAPS per-layer caps: total 9187 slots (avg 229.7/layer, min 179/layer)
```
工具預測的是 9443，差 **256 ＝ MTP 層那一格**：**MTP=0 時只池化 40 層**（trunk），
所以正確的預測是 `37×232 + 2×212 + 1×179 = 9187`、avg `229.7`、min `179` —— **三者全中**。
（`min 179` 尤其要緊：它證明**沒有任何一層低於今天的值**，也就是 §4.5 的地板成立。）

### ③ **D5 閘門 3 現在 OOM：`Killed: 9`（載入期）** —— 而這是個有資訊的失敗

第二輪的 D5（prefill250 形狀：ub 6144、ctx 8192、8 GiB）帶著 caps 起 server 時被 **SIGKILL**。
`run_server.sh` 自己的 `[budget]` 註解寫著機制：「**OOM 的邊界不是 ub 一個旋鈕，而是 (pool + compute buffer) 的和**」。
caps 把實際池由 **6.32 → 7.97 GiB（+1.65 GiB）**，而那 1.65 GiB 正是 ub 6144 需要的餘裕。
（同一組 caps 在 **MTP=0 的 10 GiB 解碼臂**上**存活並跑完** —— 那個形狀有餘裕。）

### ④ 因此「等位元組重分配」是**空操作** —— 這是本輪最重要的幾何結論

`--keep-pool-bytes`（同樣的池位元組、重新分配）跑出來是 `0-39:143`，**一格都沒變**。
原因不是實作錯誤，是**算術**：今天 uniform 143 已經只比「等位元組」低 **0.7%**
（綁定層只比典型厚 1.30×，而 143 是 `budget/(41×per_slot)`）⇒ **沒有東西可以重分配**。
**⇒ M6 的全部收益都只能來自「花掉那筆沒被用到的 slack」，而 slack 正是記憶體吃緊的 profile 需要的那筆。**

### ⑤ 正確的介面：`--slack`（用多少記憶體換多少 slot）

| slack | 典型層 cap | blk.39 | 池實際 | 是否蓋過工作集 149 |
|---|---|---|---|---|
| **0** | 143 | 143 | 6.32 GiB | ✗（今天） |
| **256 MiB** | **149** | 143 | 6.56 GiB | **✓ 剛好** |
| 512 MiB | 155 | 143 | 6.79 GiB | ✓（+6） |
| 1 GiB | 168 | 143 | 7.32 GiB | ✓（+19） |
| 2 GiB | 192 | 148 | 8.30 GiB ⚠ | ✓ |

```
slack 256MiB : CGC_SERVER_LAYER_CAPS="0-33:149;34-34:143;35-37:149;38-39:143;40-40:256"
slack 1GiB   : CGC_SERVER_LAYER_CAPS="0-33:168;34-34:154;35-37:168;38-38:154;39-39:143;40-40:256"
```
**★ 最省的可行配置是 `--slack 256MiB`：+256 MiB 就把典型層從 143 推到 149 ＝ 覆蓋量到的工作集。**
以 §4.6 的模型，這一代價買到的正是「capacity miss ≈ 6% 的抓取」那一類 ——
也就是說 **§4.6 預估的收益，只需要它原本假設的記憶體的一小部分**（10 GiB 版要 +2.1 GiB）。
（`--slack` 可以超過預算：2 GiB 那格就是 103.8%，工具會把百分比印出來當警訊。）

### ⑥ 還沒做的（下一步，順序不能顛倒）

1. **帶 `--slack 256MiB` 重跑閘門 1**（10 GiB 解碼臂已證 caps 生效；這裡要驗 8 GiB 也能存活）。
2. **閘門 2**（48 題，基準 9/48）—— 這條路的失效模式是靜默讀到空。
3. **閘門 3**（D5）—— 注意 **預期會報 INVALID COMPARISON**：resolved env 的
   `LLAMA_EXPERT_CACHE_LAYER_CAPS` 由 `40-40:256` 變成新字串 ⇒ 需要 skill `cgc-commit-gate` §2.5
   的**控制臂 ＋ 重新基線**流程，不能直接讀 M1/M2/M3。
4. 之後才是 `final stats` 的 misses／bytes 與 t/s A/B（§4.6 的兩個可否證預測）。

---

## 4.8 實測（第二輪，13:5x）：**閘門 1 通過（兩個形狀逐格吻合）；閘門 2 撞 GPU OOM**

### ① 先修工具：`expect in the log` 原本只按 41 層算

那行是用 `len(caps)`（＝41，含 MTP）算的，但**引擎只池化它實際啟用的層**，而 MTP=0 的臂
只池化 40 層（trunk）。上一輪的「預測 9443、實見 9187」差 256 ＝ `mtp_cap`，成因就是這裡。
現在兩個臂都印，操作者在跑之前就知道該期待哪個數字：

```
expect in the log: "LAYER_CAPS per-layer caps: total 6198 slots (avg 151.2/layer, min 143/layer)"   <- MTP=1, all 41 layers
expect in the log: "LAYER_CAPS per-layer caps: total 5942 slots (avg 148.6/layer, min 143/layer)"   <- MTP=0, trunk only (40 layers)
expect resident ~ 6.56 GiB (today ~ 6.32 GiB); with MTP=0 ~ 6.28 GiB
```

### ② 閘門 1 ＝ **PASS，兩個形狀都通過，且普查行逐格相同**

| 形狀 | profile（MTP） | 引擎普查行 | 工具預測 | 載入 |
|---|---|---|---|---|
| 解碼臂（`prod25`，ctx 4096） | MTP=0 | `total 5942 (avg 148.6/layer, min 143/layer)` | `5942 / 148.6 / 143` | 存活 |
| prefill250（Nail，ub 5632，ctx 8192） | MTP=1 | `total 6198 (avg 151.2/layer, min 143/layer)` | `6198 / 151.2 / 143` | 存活 |

`min 143` ＝ **沒有任何一層低於今天** ⇒ §4.5 的地板成立。

**額外驗證（原本會漏掉的一條）**：`prod25` 起的是**非 Nail 的 `Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`**
（40 層、無 NextN），而同一份字串對它算出的 5942 與引擎完全相符
⇒ **這份幾何是「專家佈局」的性質，不是那一支檔案（更不是 MTP 層）的性質。**

**`resident` 也對上了**：非 Nail 的新形狀算術 ＝ `37×149×1.0703 + 2×143×1.1719 + 143×1.3906`
＝ **6435.1 MiB**，引擎 `final stats` 報 **6434.65 MiB**（差 **0.007%**）。

### ③ 閘門 2 ＝ **不是分數難看，是 server 在第一個請求上就 GPU OOM**

`bare_48.py` 的 qa-zh #1 花了 7.5 s，接著連續 5 題「瞬間空回應」⇒ 工具自己的死伺服器防護
判定「伺服器已不在」並宣告報告無效。死因**不是 host SIGKILL，是 Metal 側**：

```
E ggml_metal_synchronize: error: command buffer 8 failed with status 5
E error: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)
E CGC-METAL-FAIL: command buffer 8 failed (status 5, Insufficient Memory) - refusing to return stale output
  #3 ggml_metal_synchronize  ← #4 ggml_backend_sched_alloc_graph  ← #5 llama_context::process_ubatch
  #7 common_speculative_impl_draft_mtp::draft      ← 失敗的是 MTP **draft 圖**的配置
```

⇒ **「載入期存活」不是這個形狀的充分條件。** 兩輪的失敗面不同：
7.97 GiB 死在**載入**（`Killed: 9`）；`--slack 256MiB`（6.56 GiB）活過載入、
死在**第一個真實請求**。**這一條要進 §4.5 的風險清單。**

**歸因：未歸因**（兩個候選都無法排除）：
- **(a)** 池多出來的 **+238 MiB**（37 個薄層 143→149；`blk.40` 仍是 256，沒動）；
- **(b)** 前一個 server（`kill -INT` 後 6 s 就起新的）留下的 **GPU／swap 殘留狀態**
  —— 當時 `vm.swapusage total=3072M used=2199M`。

能裁決它的實驗只有一個：**控制臂** ＝ 同形狀、同 build、**不帶 caps**，跑同一份 48 題。

**⚠ 控制臂跑了，但它沒有裁決任何事 —— 它在第 14 題被外部的 `SIGTERM` 殺掉**（機制見 ⑦）。
所以閘門 2 對**兩臂都是無結論**：死法不同、觸發點不同（治療臂第 1 題、控制臂第 14 題），
但**兩臂都没有跑完一次**。⇒ 不能說「控制臂活著，所以是 (a)」；
也不能說「兩臂都死，所以與 caps 無關」（控制臂是被外力殺的，不是自己死的）。
**正確的述句是：這個形狀上的 48 題協定在本輪一次都沒跑完 ⇒ 閘門 2 既未通過、也未否證。**

### ④ 兩個**出處陷阱**（會讓閘門 2／3 的數字用錯地方）

1. **「cap 8」不是 8 GiB。** `docs/CAP_AB_2026-09-13.md` 唯一的變數是 `CGC_POOL_MAX_TOKENS`
   （`--pool-cap`）⇒ 9/48 那格是 **`POOL_MAX_TOKENS=8` ＝ 今天的生產預設**，不是預算。
   而那份基準的模型是 **Nail denseIQ4X**、配方 `--expert-cache 8GiB --ngl 99 --no-mmap`。
   ⇒ 閘門 2 必須在 **Nail 檔**上跑（`prod25` 起的是非 Nail 檔，跑出來不可比），
   且必須 `CGC_SERVER_CHAT_AB=off`（否則 `assistant_prefill ... mode=always` 是拐杖，裸問不再裸）。
2. **★ `DEFAULT_REF` 還是 v5，而 v5 對這個 build 已經過期 —— 現在是活的陷阱。**
   別條線 13:45 生了 `ref_...v6_nbaware.jsonl`（`md5 72d82a33ad79e0e69bc935acd24228f2`），
   與 v5（`a0a0ca742ca94e843c54b39981742738`）**不同** ⇒ 這是**真的重基線**，不是蓋章。
   而 `scripts/check/m123_oracle_gate.py:146` 的 `DEFAULT_REF` **仍指著 v5**。證據：

   | tag | 比的參考 | M1 / M2 / M3 | comparable |
   |---|---|---|---|
   | `r34_oldref` | v5 | **5/9 · 6/9 · 5/9** | True |
   | `r34_newref` | v5 | **5/9 · 6/9 · 5/9** | True |
   | `r35_confirm` | **v6** | **9/9 · 9/9 · 9/9** | True |

   ⇒ **任何不帶 `--ref` 的 gate 執行現在都會拿到假的 5/9。** skill §2.5 重基線清單的第 4 步
   （「改 `DEFAULT_REF` 並在旁邊寫 dated 註解」）**還沒做完** —— 這是別條線的未竟項，
   本線不代改（跨線檔案），但必須揭露。

### ⑤ 還沒做（被機器佔用擋住）

- **控制臂**（③ 的裁決實驗）。
- **閘門 3**：
  1. 控制臂：不帶 `--env`（＝`40-40:256`）對 v6，應 `comparable=True`；`r35_confirm`（13:50）
     已經是同 build 的一次。
  2. 治療臂：`--env CGC_SERVER_LAYER_CAPS='0-33:149;34-34:143;35-37:149;38-39:143;40-40:256'`
     對 v6 ＋ `--allow-incomparable`，應報 **INVALID COMPARISON**，diff 形狀是
     `ENV.LLAMA_EXPERT_CACHE_LAYER_CAPS: ref='40-40:256' now='0-33:149;...'`
     （而 M1/M2/M3 仍會印 9/9 —— 印著 *printed for information, NOT a verdict*）。
  3. 重基線：`--write-ref ref_..._v7_layercaps.jsonl --ref-note "..."` ＋ `md5` 對一眼。
     **若新參考與 v6 逐位元相同 ⇒ 換池幾何不動任何一個 logit**，這是最強的無損證據；
     **不要用「M1 = 9/9」當證據**（那是弱述句，見 skill §2.5）。
     ⚠ 治療臂在 `prefill250` 形狀上可能連第一個請求都過不去（③）。
- 之後才是 `final stats` 的 misses／bytes 與 t/s A/B。

### ⑥ 建置出處（本輪兩個形狀都用同一個 build）

| 產物 | md5 | mtime |
|---|---|---|
| `libggml-metal.0.19.0.dylib` | `037c931ec6169f60189e7fde19b1232e` | 11:50:22 |
| `libllama.0.0.279.dylib` | `127680f71e213bf6240c0b10324e84b8` | 13:24:06 |
| `llama-server` | `054fb22f04a01c5cb93b362883dd3f82` | 13:24:06 |

⚠ **這個 build 含別條線未提交的 r33 nb-aware 修正**（13:24 建的）⇒ 與 09-13 的 48 題基準
**跨 build**，任何「B 有／沒有變差」的比較都要多一個同 build 的控制臂才成立。

### ⑦ 控制臂的死因是**別的 session**，不是記憶體 —— 而機制是設計出來的

控制臂的 log 最後是這一行，**不是** `Insufficient Memory`：

```
[CGC] Received SIGTERM — initiating graceful shutdown (expert cache + Metal buffers will be freed).
Do NOT use kill -9; it leaks GPU memory.
```

時間線（同一個 repo、兩條 session）：

| 時刻 | 事件 |
|---|---|
| 13:55:58 | 別條線起 `m123_oracle_gate --tag r36_pool6` |
| 13:56:20 | `summary_r36_pool6.json`（他們那輪結束，港埠放開） |
| **13:56:54** | **本線過閘門（當下 8080 確實空、無量測行程）→ 起控制臂（denseIQ4X、5976 槽）** |
| **14:01:53** | **控制臂的 server 收到 `SIGTERM`（進行到第 14 題）** |
| 14:01:56 | 別條線的 `cap_r37_pool4.json` |
| 14:02:16 | `summary_r37_pool4.json` |

機制不是偶然：**`run_server.sh` 有一道強制 preflight，會先把任何殘留的 `llama-server` SIGTERM 掉，
才起自己的**（`scripts/run_server.sh:638-740`，`[防護 1] 清殘留`；註解寫明目的是防止「行程疊加」
造成的 `GPU OOM ret=-3` 與 kernel panic；可用 `CGC_PREFLIGHT_KILL=0` 關掉）。
⇒ **在這個 repo 上，「起跑前檢查港埠」只能保護起跑那一瞬間，保護不了 20–30 分鐘的長跑。**
一個 48 題的協定需要的是**互斥**：雙方都認的 lockfile、或請對方設 `CGC_PREFLIGHT_KILL=0`、
或把它切成長度受控的片段（`--profiles` / `--max-per-profile`）。
**這就是閘門 2 本輪沒有結論的機制原因**，也是今天第四次資源碰撞 —— 這次本線是擋路的一方。

### ⑧ 順帶修掉一個工具缺陷：`bare_48.py --compare` 在「有臂早停」時崩潰

死伺服器防護會產出**只有部分 profile** 的結果檔，而 `--compare` 逐 `a["profiles"]` 去取
`b["profiles"][profile]` ⇒

```
KeyError: 'longform-zh'      # A 有 {qa-zh, longform-zh}、B 只有 {qa-zh}
```

**它偏偏在「該說這臂無效」的時候崩掉。** 修法：走 union、缺的一側印 `n/a`、
**有 `invalid` 就在表頭大聲標出來並且不讓它的通過數被當成分數**、
總計列在題數不同時拒絕相減。修好之後：

```
!! m6-control  IS INVALID -- …中止於 longform-zh #6；covered 14 question(s)
!! m6-slack256 IS INVALID -- …中止於 qa-zh #6；covered 6 question(s)

Profile          m6-control   m6-slack256  差距
qa-zh            5/8          0/8          -5
longform-zh      0/6          n/a          n/a
總計              5/14         0/6          n/a (不同題數)
```

⚠ **不要引用「control 5/14」或「control 3/48（legacy）」。** 它只覆蓋 14 題、被外力中斷、
工具自己標成 `invalid` ⇒ 它是一個**未完成的樣本**，不是分數。

---

## 4.9 閘門 4（14:49–14:53）：**「149 個 slot 就能蓋住工作集」被否證**

第一次真的量到這條路要的兩個量。儀器：`run_server.sh` ＋ 一個**確定性負載**
（同一 prompt ×6、`max_tokens=96`、`temperature=0`），計數器**全部取自引擎自己**的兩行：
`final stats` 與 `miss attribution: compulsory=… capacity=… evictions=… layers_distinct_over_slots=…
worst=layer … distinct=… slots=…`（後者在 `llama-expert-cache.cpp:2360`）。

兩臂同 build、同 profile（`prefill250` ＋ `CGC_SERVER_MTP=0` ⇒ 非 Nail 檔、8 GiB、ub 5632、ctx 8192）、
同 prompt；唯一差別是 `CGC_SERVER_LAYER_CAPS`。

| | 控制（無 caps ＝ 均勻 143） | 治療（`--slack 256MiB` ⇒ 薄層 149） |
|---|---|---|
| 普查行 | 空（均勻 143 不印） | `total 5942 (avg 148.6, min 143)` ✓ |
| requests / hits / misses | 164719 / 159510 / **5209** | 215120 / 205965 / **9155** |
| hit rate | **96.8%** | **95.7%** |
| capacity 佔比 | **2374 / 5209 ＝ 45.6%** | **3780 / 9155 ＝ 41.3%** |
| evictions | 5209 | 8933 |
| **distinct > slots 的層數** | **3** | **9** |
| **最差層** | layer 2：**distinct 173 > 143** | layer 2：**distinct 239 > 149** |
| resident | 6197.04 MiB | 6434.65 MiB（**+238 ✓ 幾何生效**） |
| decode t/s | 13.99 | 6.00 |
| prompt eval t/s | 18.07 | 6.04 |

### ① ★ 判決性的（**單臂即可讀、不依賴 A/B 可比**）：§4.6「capacity 應該歸零」是錯的

§4.6 的前提是「工作集 ≈149 ⇒ 149 個 slot 就蓋住它」。引擎的**逐層**普查否證了它：
**最差的層需要 239 個 distinct 專家，遠高於 149**；且治療臂**有 9 層 distinct > slots**。
⇒ `--slack 256MiB` 買到的 +6 slot/層**不可能**消掉 capacity 那一類（實測 41.3%；控制臂 45.6%）。

**這一條不依賴兩臂可比** —— 它是治療臂**自己**的讀數：即使給了 149，那一層還是要 239。

**⇒ 修正 §4.6**：「工作集 ≈149」應改為「**逐層 distinct 需求，最差層 173–239**」。
要真的消掉 capacity class，需要的不是 +6 slot/層而是**接近 256**（或改淘汰／填充策略）。
連帶地，§4.2 的 8 GiB 字串（184）與 10 GiB 字串（232）**兩者都落在不夠的那一側**。

### ② ⚠ 這次的 A/B **不可比** —— 而「為什麼不可比」本身是個訊號

負載**發散了**：同一個 greedy prompt ×6，控制臂六次都回 81 tokens（**486**），治療臂回
81/71/71/71/82/75（**451**）。**同一個 server 上六個相同的請求給了不同的長度** ⇒ 治療臂
**自身不確定**，控制臂確定。而治療臂同時慢 **2.33×**（decode 13.99 → 6.00）。

兩個候選都無法排除 ⇒ **歸因：未歸因**：
- **(a) 記憶體代價**：resident +238 MiB；`vm.swapusage` 的 **total 由 3072M 變 6144M、used 4716M**；
- **(b) 熱**：`thermalpressurelevel` 由 **0 → 1**（14:48 是 0、14:53 是 1）。

**不是別條線干的**：14:50–14:55 的 `Backup/` 沒有任何他們的產物，也沒有他們的行程。

**裁決實驗**：同一熱狀態下**交錯**跑兩臂 ×3（A,B,A,B,A,B），且**每臂把同一負載跑兩遍**看 token 數
是否自洽。控制臂仍自洽＋治療臂仍發散 ⇒ 不是熱，是幾何。

### ③ 對 §4.6 上界的最終影響

- 「capacity ≈ 5.3% 的抓取會歸零」**不成立**（①）。
- 而且 §4.6 缺一個關鍵數字：**capacity miss 的位元組佔比**。這次有 `file_reads`（42876 → 167832）
  與 `pread_usec`（0.99 → 4.91 s），但負載發散 ⇒ 不可比。
- ⇒ **§4.6 的 +1–3% 上界降級為「未確立」**（不是「樂觀到 0」）：要給數字仍需要一場**乾淨的**計數器 A/B。

### ④ 這輪的副產品（可直接引用）

- **引擎已有逐層 miss 歸因普查**（`src/llama.cpp/src/llama-expert-cache.cpp:2360`），含 `evictions`
  與 `layers_distinct_over_slots`／`worst=layer … distinct=… slots=…` ⇒ **閘門 4 不需要新儀器**。
- 這個負載的 hit 是 **96.8%**，而同機器上 oracle／不同 prompt 的負載是 72–89% ⇒ 再次印證
  **hit% 由負載主導**；「+N slot 會讓 hit 上升多少」必須指定負載才有意義。
- 檔案：`Backup/cgc_logs/llama_server_20260917_{144857,145047}.log`；負載驅動 `/tmp/pool_load.py`（暫存）。

---

## 4.10 判決（15:0x）：**設定路線「不可交付」，不是「機制被判死」**

分三層講，因為混在一起就會得到錯的結論。

### ① M6 的本體（roadmap 原意：換量化幾何）—— **不是死了，是早就達成了**

roadmap 的目標是 per-expert **1.769 → 1.122 MB**，用來把 6 GB pool 的 slots 由 85 推到 133、
把 decode-25 的殘餘頻寬由 5.66 壓到 1.30 GB/s。**出貨的 `Nail-…-denseIQ4X.gguf` 實測 per-expert
＝ 1.0703 MiB（典型層）**，也就是 roadmap 的目標值**已經在檔案裡**。進一步動模型（拉平 `blk.39`）
已被否決：那是上游逐張量重要性校準的結果，拉平＝**有方向的降精度**（gate/up −25.5%、down −19.1%）。
⇒ **M6 的頭條價值已兌現**，這件事今天沒有被推翻。
（`verify_edge0_gguf.py` 在本 repo **不存在**，所以那條離開條件無對象；等價物是 §4.4 的第 2–4 道。）

### ② 小額 slack（`--slack 256MiB`）—— **判死**

§4.9 的量測：薄層 143 → 149 **消不掉** capacity class（最差層要 **239** 個 distinct）。
而且同一條字串在 MTP-on 的 `prefill250` 形狀上**第一個請求就 GPU OOM**（§4.8 ③）。
⇒ **+238 MiB 這個價位買不到東西。**

### ③ 「把池用滿」這個想法 —— **沒死，但價錢被量出來了，而且落在買不起的一側**

引擎自己報的瓶頸是**逐層 distinct 需求**（最差 239），所以覆蓋它需要 **cap ≈ 240–256**：

| 要覆蓋 | 需要的 cap | 8 GiB 預算下的池 | 可行性 |
|---|---|---|---|
| 今天的均勻 143 | 143 | 6.32 GiB | 現狀 |
| §4.2 的 8 GiB 字串 | 184 | 7.97 GiB | 已在 MTP-on 上 **載入即 OOM** |
| `--slack 256MiB` | 149 | 6.56 GiB | 載入活、**請求即 OOM**；且不夠 |
| **覆蓋瓶頸層** | **≈240–256** | **≈9.9 GiB** | **超過 8 GiB 預算** |

**收益上限也比我原本寫的小**（用今天同一個負載自己的數字反推）：requests 164719、misses 5209
⇒ miss 只佔 **3.16%** 的抓取；capacity 2374 ⇒ **1.4% 的抓取**；IO ≈ 牆的一半 ⇒ **≤ +0.7%**。
若改用生產負載的 hit（89.2%，今天 13:39–13:50 那三輪）反推：capacity ≈ 0.45 × 10.8% ≈ 4.9%
的抓取 ⇒ **≤ +2.5%**。

**⇒ 正確述句**：**機制存在、上限存在（~+2.5%，取決於負載的 hit）、代價（+2 GiB 等級）落在
記憶體吃緊的形狀買不起的一側。** 所以 M6 設定路線是**不可交付**，不是機制被否證。

**另一個轉折**：別條線的 r33 修正把 hit 由 57.1% 抬到 **72.0%**（**與池無關**）⇒
**有一部分本來要 M6 去買的東西，已經被「修路由」買走了。**

**能復活它的三條路**：① 加預算（> +2 GiB，但那正是吃緊的形狀沒有的）；
② **改淘汰／填充策略**（不靠更多 slot 提高複用 —— 這是唯一不需要預算的路）；
③ 改目標（「少讀」本來就是 M2 的地盤）。

---

## 5. 離開條件（逐條，並更正一條）

| roadmap 的離開條件 | 現在該怎麼寫 |
|---|---|
| `verify_edge0_gguf.py` 全通 | **該檔在本 repo 不存在**（find 0 筆）。§4 路線的等價物是第 2–4 道（普查行／48 題／oracle gate） |
| M1／M2 = 100% | §4 不動模型 ⇒ 參考不變，直接跑 `m123_oracle_gate`（**先讀 `comparable`**）。§3 若哪天要做，才需要 `--write-ref --no-pin-oracle-env` |
| 48 題用新舊兩把尺並排 | 工具已在：`scripts/check/bare_48.py` ＋ `bare48_answer_keys_v1.json`；設計見 `docs/BARE48_SCORER_TWO_RULERS_2026-09-13.md` |
| （新增）**hit rate 與 slots** | 讀 `final stats` 的 hit rate，**同一支模型、同一輪負載**；速率（64 題 vs 200 token）不同不可比 |

---

## 6. 這份計畫還缺什麼（誠實）

1. **沒有量測。** §4 的所有數字都是**同一份標頭的算術**，加上與引擎自報值的對帳（§4.3）。
   「232 slots 會讓 hit% 上升多少、t/s 上升多少」**未經任何實驗**。
2. **逐層 caps 的正確性未驗**（第 3–4 道）：這一族失效是**靜默讀到空**，不是崩潰。
3. **`--mtp-cap` 的建議值沒有依據**：256 是 `run_server.sh` 的既有預設，不是量出來的結論。
4. **工具的測試**：目前只有實跑＋錯誤路徑（非 GGUF→rc=3、缺檔→rc=2、未知型別 id→硬錯、
   `-q --layer-caps` 拒收、地板溢出、`--mtp-cap`）＋自我檢查（emitted 字串用 `cgc_layer_cap`
   的語意重新解析、逐層比對）。**沒有 mock 的 GGUF fixture 測試。**
5. **替代的分配規則沒比較**：這裡只實作「每層等位元組、夾地板」。若要最大化 **平均** hit%，
   正確的目標是 `max Σ_il hit(s_il) s.t. Σ s_il ps_il ≤ B` ⇒ `hit'(s_il) ∝ 1/ps_il`，
   差距只有 `τ·ln(ps 比)`（ps 比 1.30 ⇒ 約 13 個 slot @ τ≈50），比「等位元組」更平緩。
   **要不要換規則，應該在拿到第一個真實 hit% 之後再決定。**

---

## 7. 出處

- `docs/ROADMAP_PREFILL250_DECODE25_2026-09-13.md` §M6、§依賴與排序
- `scripts/run_server.sh:411-419`（143 slots／90.8%／counterfactual；`:419 BUDGET_DEFAULT=10 GiB`；
  `:1804-1807` 每次啟動都寫 `LLAMA_EXPERT_CACHE_LAYER_CAPS`）
- `src/llama.cpp/src/llama-model-loader.cpp:1100-1191`（公式、NextN 排除）、`:1468-1513`（ne[2] 收縮、
  `cgc_layer_cap` 的呼叫點與「必須與 cache 的 n_slots_l 相符，否則 OOB」）
- `src/llama.cpp/src/llama-expert-cache.h:506-531`（`cgc_layer_cap` 的解析語意）、
  `:152-158`（`slots_l` 逐層）、`src/llama-expert-cache.cpp:3133-3142`（`n_slots_l` 的填充）
- `models/gguf/MANIFEST.md`（載體的重生方式）、`scripts/gen_denseiq4x_tt.py`（逐張量釘選＝byte-copy）
- `scripts/check/gguf_pool_geometry.py`（本檔所有幾何數字的產生器；`--layer-caps` 是 §4）
- `scripts/gguf_retensor.py`（§3 的模型工具；SAFETY 一節）、`scripts/check/bare_48.py`、
  `docs/BARE48_SCORER_TWO_RULERS_2026-09-13.md`、`docs/POOLSIZE_INVARIANCE_2026-09-13.md`（4/6/8 GiB 的數值不變性）
