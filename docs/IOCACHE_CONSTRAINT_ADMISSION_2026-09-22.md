# 把「25 的約束」壓在 expert cache 的 Metal I/O 介面上？——能不能到

日期：2026-09-22 · 作者：本線（線 A，引擎層） · 0 重建（純 code 閱讀 ＋ 既有實測引用）
前提（你指定的）：**8 GiB expert cache**、`Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X`、
`Mac16,12`/M4/8GPU/**16 GB**、Metal backend、單流交付 cell。
上游文件：`docs/SHAPE_FOR_25_2026-09-22.md`（本文會更正它三處）。

---

## §0 結論（先答）

**不能。不是「可能不夠」，是「這七條裡只有三條歸這個介面管，而那三條今天已經是現況」。**

| 分數 | 結果 |
|---|---|
| 七條約束中，**歸 expert-cache Metal I/O 介面擁有**的 | **3 / 7**（#2 #3 #4） |
| 其中**今天尚未實作**的 | **0 / 3**（三條都已實作：union 去重、<=143 bucket gate、只讀一次） |
| 其中**施加形式約束能帶來速度**的 | **0 / 3**（詳 §2：可以「拒絕」或「裁剪」，但裁剪＝丟專家＝改答案） |
| 真正還沒 solution 的四條（#1 #5 #6 #7） | 住在 `llama-graph.cpp` / `common/speculative.cpp` / `ggml-metal` / mmapped weights，**不在這個函式呼叫裡** |

而我最樂觀的物理帳（§5）在最好的那個 ml 假設下，**做完七條仍超標 12%**；在中間假設下**超標 2.2×**。
唯一的例外是「允許專家再量化一檔」那一條（§7），它會同時鬆開三件事，但它撞的是你 09-20 的精度裁定。

### 順手更正我自己昨天的三處（都要記，不要用舊值）

| # | 昨天 `SHAPE_FOR_25` 寫的 | 更正後 | 依據 |
|---|---|---|---|
| ① | union 去重次線性：`distinct(M) = 256(1−(248/256)^M)`（8 → 30.5 → 57.4） | **本 repo 實測是線性 8·M**（M=5 → **40**、M=8 → **64**、MTP 穩態 4 列 → **min=max=32**） | `docs/CAP_INVARIANCE_LOCALIZATION_2026-09-14.md:97-101`（`gather=0`）、`Backup/cgc_logs/union_evidence_2gib_20260914.txt` |
| ② | per-slot `1.045 MiB`（⇒ 專家 342.5 MiB/token-row） | **`1.391 MiB`**（⇒ 專家 **456.2 MiB/token-row**，+33%） | `docs/EXPERT_CACHE_THRASH_2026-09-19.md:115` binding blk.39；自洽檢查：`8 GiB / (41 層 × 1.391 MiB) = 143.6` ✓ 對上實測 `n_slots=143` |
| ③ | 「dense 常駐 page-cache」＝ §EN-358 第一槓桿 | **該槓桿已於 09-21 撤下**（減 miss 不驅動 t/s；且 miss 翻倍時 t/s 不動） | `docs/IO_PATH_AB_2026-09-21.md` §三結論② ③ |

---

## §1 介面解剖：真正的「輸入輸出面」只有四個函式

expert cache 承接 Metal 的那條縫，就這五個位置，**SSD 位元 ⇢ GPU 可見位元的唯一路徑**：

| 角色 | 位置 | 做什麼 |
|---|---|---|
| **請求形狀**（入口） | `llama-context.cpp:5499-5525`（`expert_cache_on_topk`） | 把 top-k ids 張量（`[n_expert_used, n_tokens]`）**跨所有列去重**成 `uni`，排序，然後一次呼叫 |
| **許可閘門** | `llama-context.cpp:6707` → `llama-expert-cache.cpp:906` `ensure_batch` | 一次鎖，整層的 miss 同時分配 slot 並發 pread 工作 |
| **dst 產生** | `llama-expert-cache.cpp:649` `fill_pool_direct_collect` | `dst = pool_region(layer,kind) + slot_idx * stride`；每專家 4 個 kind ⇒ 4 段 |
| **真正的 I/O** | `llama-expert-cache.cpp:3163` `fill_segments_pool` | 按 `(file_idx, file_offset)` 排序 → **檔案相鄰的段落合成一個 `preadv`**（iovec scatter 到多個 dst）→ 推到 worker pool → `:3280` **同步等待** `pool_outstanding == 0` |
| GPU 可見記憶體來源 | `:632` `pool_region` ／ `:1715` `pool_split_alloc` | L4：**adopt expert tensor 的 Metal storage**（non-owning）；否則 malloc pool。`GGML_BACKEND_BUFFER_USAGE_WEIGHTS` |

實測 `n_slots=143`（`Backup/cgc_logs/llama_server_20260922_141654.log`）：
`L4 metal pool: 143 slots/layer, regions adopted from expert tensors`、`LAYER_CAPS total 5976 (avg 145.8, min 143)` ⇒ pooled 層 ≈ **41**。

⇒ 這一點很重要：**L4 下 pool 就是 Metal 權重 buffer 本身**（`regions adopted from expert tensors`），
所以「填池」沒有第二跳拷貝 —— 這條 I/O 面已經是它自己的 zero-copy 版本。

---

## §2 七條約束逐條裁決（能不能在這個介面 enforce）

| # | 約束 | 擁有者（file:line） | 歸此介面？ | 今天狀態 | 要做到要動什麼 |
|---|---|---|---|---|---|
| 1 | 每 forward **3–4 列**，樹狀 | `cgc_pool_max_tokens()` default **8**（`llama-graph.h:23`，`CGC_POOL_MAX_TOKENS` clamped [2,64]）、`cgc_decode_width(min_usable=143, top_k=8, cap)`（`llama-context.cpp:292-303`）、spec 排程 `common/speculative.cpp` | ❌ **它是被呼叫者** | 今天 **M 的上限就已經是 8**（cap=8）；routable bound `⌊143/8⌋=17` ⇒ `CGC_POOL_MAX_TOKENS=17` 環境變數 **0 重建即試** | 提高 cap ＋ 樹狀 draft/verify 排程，都不在這個函式裡 |
| 2 | dense 搬運不要動 | — | ✅ 天然符合（此介面只搬專家） | **已結案**（93.2 GB/s ＝ 86% 峰值，NSG 1..32 無趨勢） | 無 |
| 3 | 同層列聯集、每 tensor 每步只讀一次 | `llama-context.cpp:5499-5513` union ＋ `:6707` ensure_batch | ✅ **歸它，且早已實作** | 已實作（實測 union = 8·M） | 無。**但收益 ≈ 0：能被去重掉的東西從源頭就不存在，列間本來就不重疊**（見 §3） |
| 4 | 工作集 ≤30/層、硬頂 143 | gate `uni.size() <= n_usable`（`:6166`）、超限走 gather 或 abort（`:1126`） | ✅ **歸它** | 143/層；ρ(4)=32/143=0.224 | ⚠ 若真把「≤30」寫成**形式約束**（超了就裁掉）：那不是加速，是**丟掉選中的專家 ⇒ 輸出改變 ⇒ M1 bit-identity 歸零**，不是加速 |
| 5 | dense 常駐 page-cache／pool 只放專家 | model mmap ＋ unified memory 預算 | ❌ | 已 **oversubscribed 6.9 GB**；且 §EN-358 已撤下 | 此介面沒有任何槓桿（且 has been 撤下） |
| 6 | graph ≲100 buffers／≤8 segments | `llama-graph.cpp` ＋ `ggml-metal` graph build/split | ❌ | 633 buffers／40 segments | 見 §6（它是 #3/#4 能否 async 的**前提**） |
| 7 | GDN chunked / parallel scan | op/graph | ❌ | 逐步；`LLM_FUSED_OP_GDN_CH`（`llama-graph.h:71`）有 enum 但 `llama-graph.cpp` **無任何引用** ⇒ 未實作 | 需新 kernel |

**讀表的重點**：介面能施加的約束，若不維持 bit-identity，它就是**品質旋鈕不是速度旋鈕**。
介面真正能新加的只有「形式拒絕」那一類（超限就拒絕／報告／abort）——而那一類今天已經在 enforce 了。

---

## §3 更正①：union 實測是線性 —— 「每步只讀一次」本來就成立，但它的收益是 0

本 repo 自己的量測：

- `docs/CAP_INVARIANCE_LOCALIZATION_2026-09-14.md:100-101`：`Max union at cap5 is **40 (= 5·top_k8)**, at cap8 **64**, against 143 usable slots — gather=0 everywhere`
- `Backup/cgc_logs/union_evidence_2gib_20260914.txt`（MTP 穩態，2 GiB）：`layer=1..8 union avg=32.0 **min=32 max=32** of usable=34`
- 由此反推硬上限：M=16 ⇒ union 128、M=18 ⇒ union **144 > 143 池滿** —— 不是昨天寫的 M=26

理由很樸素：**不同 token 的路由近似獨立**，期望列間重疊只有 `C(M,2)·8²/256` 個（M=4 才 ~1.5 個），量不出來。

⇒ **所以 #3 那條「同層列取聯集、每步只讀一次」今天確實成立 —— 但它去重掉的東西從源頭就不存在。
加寬一列，就是多 8 個不同專家的 bytes，沒有去重可以撿。**

（對照 §EN/THRASH：「足跡成長 ≈ 0.56 個新 expert/層/token」那個是**長跑趨緩後的增量**，不是單 STEP 內的去重，兩者別混。）

---

## §4 更正後重掃 M（不是 79–90，是 ~50 t/s）

輸入全實測：Peak 108.8 GB/s ／ dense 家族實跑 **93.2 GB/s**；
per-forward dense bytes = trunk 827.0 ＋ lm_head 397.9 = **1224.9 MiB**（13.78 ms）＋ MTP 分支 303.0 MiB（3.41 ms）＝ **17.19 ms/forward，與 M 無關**；
專家：41 層 × 8M distinct × **1.391 MiB**；每列額外 FLOP **2 ms**〔未實測，採 ~3 TFLOPS FP16 量級〕；
接受：鏈式 `E(M) = (1−p^M)/(1−p)`、`p=0.5825`〔實測〕。

| M | 專家 MiB/forward | bytes+compute ms/forward | E(M) | **ms/token** | **屋頂 t/s** |
|---:|---:|---:|---:|---:|---:|
| 1 | 456.2 | 24.3 | 1.000 | 24.3 | 41.1 |
| **2** | 912.5 | 31.5 | 1.583 | **19.9** | **50.3** |
| **3** | 1368.7 | 38.6 | 1.922 | **20.1** | **49.8** |
| 4 | 1825.0 | 45.7 | 2.119 | 21.6 | 46.4 |
| 8 | 3650.0 | 74.3 | 2.363 | 31.4 | 31.8 |
| 12 | 5475.0 | 102.8 | 2.392 | 43.0 | 23.3 |
| 17 | 7756.2 | 138.5 | 2.395 | 57.8 | 17.3 |

（若用舊的 1.045 MiB，同樣的表是 43.4 / 54.8 / **55.3** / 52.2 —— **不影響任何定性結論**。）

三件事：

1. **屋頂線仍在 50 t/s ≫ 25** ⇒ 「25 不是被頻寬/FLOPS 擋住」這個結論**不受更正影響**。
2. 但甜蜜點從「M=3–4 ⇒ 90 t/s」往下修成 **「M=2–3 ⇒ 50 t/s」**。
3. **所需的進步重新定義**：markup 要從今天的 `79.55 / 20.08 = 3.96×` 降到 `40 / 20.08 = **1.99×**`
   （昨天寫 6.3×→3.2×，那是因為昨天沒算 MTP 303 MiB、且 per-slot 低估 33%）。
   ⇒ 目標是「**overhead 砍一半**」，不是「砍到三分之一」。

---

## §5 這樣到底能不能到 25：不能（三個 ml 假設全超標）

把每 forward 的非 byte 成本也放進來（都取自既有結案報告）：

- **miss I/O**：`cb_layer = 0.46 ms + 0.695 ms × m`（F1 線性回歸 R²=0.86）⇒ 328 請求/token-row × 22.5% miss = 73.8 miss × 0.695 = **51.3 ms**
- **graph 固定**：dispatch 18.6 µs × 633 buffer ≈ 11.8 ms ／ buffer 邊界 0.29–0.35 ms × 40 seg ≈ 12.5 ms ⇒ **24.3 ms**

以 M=3 為例（屋頂 20.08 ms/token ⇒ **剩 19.92 ms/token 給所有非 byte 成本**）：

| ml（token/步） | miss ms/token | graph ms/token | 合計 | vs 可用 19.92 | 超標 |
|---|---:|---:|---:|---:|---:|
| 1.28 | 40.08 | 18.98 | **59.06** | 19.92 | **2.96×** |
| 1.70 | 30.18 | 14.29 | **44.47** | 19.92 | **2.23×** |
| **3.375** | 15.20 | 7.20 | **22.40** | 19.92 | **1.12×** |

（M=2 / M=4 結果同量：2.94–3.21×／2.21–2.41×／1.11–1.22×。）

⇒ **「把七條全做到」也只在 ml=3.375 的樂觀情形接近，還差 12%，而且那份樂觀還沒扣 GDN、CPU hook、序列化。**

⚠ 交叉檢查（很重要）：把上面所有分量相加，M=1 會預言 99.9 ms/token，實測是 **79.55** ⇒
**分量相加過預測 26%**，與 09-20「`gap ⊆ cb+submit` 交叉檢查以 10.09 ms 失敗 ⇒ 該 regime 分量相加不閉合」
完全一致。**所以上表是樂觀上界；真相比它更差。**

⚠ 反向證據也不能不說（`docs/IO_PATH_AB_2026-09-21.md` §③）：4 GiB 池的 miss 數是 8 GiB 的 **1.91×**、
capacity miss **3.15×**、`cb` 41.8 → **90.0** ms —— **而 t/s 是 10.60 vs 10.28，不動**。
⇒ **存在一段「miss 成本被吃在 idle 裡」的鬆弛（該場約 48 ms/步）**。
這兩個量測打架（F1 回歸 vs 池 A/B），本文採「保守上界」那一邊；**要消歧的成本就是下面實驗 ①**。

---

## §6 依賴是 DAG，不是加法 —— #6 是 #3/#4 能不能 async 的前提

- 要把 miss 藏起來，必需「填池時 GPU 有別的事可做」。實測（09-21）：**5811/5811 的 MoE 落在 segment 的第一個 command buffer**
  ⇒ **可遮窗口 = 0**（`docs/L3_WINDOW_ZERO_2026-09-21.md`）。
- 而且 `:3280` 是**同步等待** `pool_outstanding == 0`。想要重疊，要嘛
  (a) 先把圖切成 ≤8 段、讓 MoE 不再永遠站在段首（**#6**），
  (b) 要嘛靠 fill 發得更早（`prefetch_slot` / layer-ahead prefetch / `WIN_PIN` / prerouter，全都有 flag）——
  但已有負結果：`docs/L3_PREFETCH_RETRACTED_2026-09-21.md`。
- ⇒ **#6 沒做之前，#3 的 async 版本不可達**；兩者不能並列相加（也違反 09-20 的「別相加」判決）。

---

## §7 8 GiB 前提下，唯一在物理上有巨量空間的那一格（以及為什麼它今天不值錢）

最有意思的數字其實在這裡（既有實測，不是本文新算）：

> 每個 miss 搬 ~1.11 MiB，被 **1.6 GiB/s** 服務（0.695 ms）；
> 同一台機器的 DRAM 峰值是 **108.8 GB/s** ⇒ **只用了 1.5%**。
> 而且它**比磁碟自己的 823 MB/s 峰值還快一半** ⇒ **不是 disk-bound，是被 overhead 統治的 RAM 拷貝**。
> —— `docs/F1_CB_MISS_REGRESSION_RESULT_2026-09-20.md:13,87-95`、`docs/CB_IS_THE_DEVICE_RESULT_2026-09-20.md:75`

也就是說：**這條 I/O 面確實有 ~60× 的物理空間**，而它正好在你問的那個函式裡。
但同一條縫早在 2026-08-29 就已經做過：`(file,offset)` 排序 ＋ 檔案相鄰段落合併成一個 `preadv`（`:3170-3245`，2026-08-29，
`LLAMA_EXPERT_CACHE_NO_MERGE` 可 A/B）＋ `rdadvise` ＋ persistent worker pool（預設 8）。
剩下的選項（加大 span / 更激進 batching）有專門的判準在 `cgc-io-request-shape` skill 裡。

而 IO A/B 又說：**這段成本今天多半落在 idle slack 裡**（miss 翻倍 t/s 不動）。
⇒ 空間存在，但**只有在別的固定成本先消失之後才會顯現** —— 這正是 §5 上表「total 超標」的來源。

### 唯一能讓帳變正的兩條路（都要你先裁定）

**(A) per-slot 降到 ≤ 0.929 MiB（−33%）** —— 同一個 8 GiB 買到 **215 slot/層**（`8192 / (41 × 0.929) = 215`），
capacity miss 34% → 0.7%。它同時鬆三件事：專家 bytes −33%、單次 miss 成本 −33%、slot ×1.5。
**但它要把專家再量化一檔，撞的是 09-20「精度下限不再壓」的裁定**（那份裁定講的是 dense xx s；
專家是不是同一個裁定，需要你明文確認）。
不做重新量化（維持 `@143 slot`）的話，capacity miss 只佔 miss 的 **34%**，把它清零也只是 51.3 → 33.9 ms —— **不夠**（§5）。

**(B) 提高 ml 到 ≥ 3.4（樹狀 speculation）** —— §5 表裡唯一接近過線的那一行。
它不是新的引擎工作：粗糙地說鏈式已在 `p=0.58` 飽和於 **2.395 token/步**，
必需樹狀（加深 k 已算過只到 24.59 t/s，`MEMORY_PERF.md:718`）。
而它把戰場推回 **#1/#7 的老問題**：接受數取決於 accept 率與品質 —— `CGC_RN_ROUTING=1 + CGC_WCOLD_EN=1`
曾實測 25.9 t/s / 98.7% accept，quality **0.3**。**速度路存在，擋路的是品質**（`IO_PATH_AB` §剩兩條 2）。

---

## §8 要試就照這個順序（前兩個 0 重建）

1. **消歧 miss 到底是貴還是被吸收**（最高優先，下面三項的解讀都取決於它）
   既有儀器：`CGC_UNION_LOG=1`（per-layer union min/avg/max，`llama-context.cpp:6162`）
   ＋ `µs/miss`。做一場 pool budget 4 GiB vs 8 GiB 的**同場配對**（single-run ±27% ⇒ 不能跨場比），
   直接看 `cb` 成長時 GPU idle 有沒有等幅吸收。
2. **`CGC_POOL_MAX_TOKENS=17` 掃 M**（0 重建，單純放寬 cap； routable bound `⌊143/8⌋=17`）
   ⇒ 這是 #1 的全部成本：union 會立刻長到 8·M，同時抄 ρ(union/143)、miss/步 與 `ml=E(M)`。
   ⚠ 期望很低：§4 表說 M=2–3 才是甜蜜點，而 pool full 在 **M=18**。
3. **把 union=8·M 與 per-slot=1.391 MiB 回寫** `SHAPE_FOR_25` §1 §3（本文 §0 已更正，舊值停用）。
4. **如果 (A) 被解鎖**：專家再量化 ⇒ `215 slot/層` ⇒ 重跑 §5 表；那是唯一讓 §5 的帳由負轉正的路。

---

## §9 誠實欄

- §4 的 `2 ms/列` FLOP 是〔未實測〕：本專案從沒量過 FP16 峰值，它可能大到某個 M 之後由純頻寬受限轉成算力受限（昨天清單的第 4 項仍未量）。
- MTP 303.0 MiB/forward 是當作**與 M 無關**處理（樂觀）；若它隨列成長，每行再加 3.4 ms。
- `0.695 ms/miss` 是 **within-run 回歸**的邊際值；cross-run 會長得不一樣（`IO_PATH_AB` §③）。
- ml 仍然沒有乾淨值；本文列出三種 ml 是要攤開不確定性，不是假裝有三個精度。
- 全程 0 重建、只讀 source；但 tree 上仍有上次未提交的 `ggml-metal-device.cpp` patch（另一個實驗的），本文未使用。
