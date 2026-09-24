# 「swap 大 ⇒ miss 也高」的機制判別與修復決策

日期：2026-09-24 · 方法：讀碼定案 + 既有實測重算（0 GPU、0 重建）
口徑：16GB MacBook Air M4；模型 `Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf` 13.65GB；pool 8GiB

---

## 0. 結論先講

**這個前提要拆成兩條通道，而它們的答案相反：**

| 通道 | swap 大時會怎樣 | 判決 |
|---|---|---|
| **cost 通道**（一次 miss／hit 要付多少時間） | 顯著變貴 | ✅ **成立，且是主效應** |
| **count 通道**（miss 的次數） | 基本不動 | ❌ **大概率不成立** |

⇒ **你觀測到的「swap 大時 miss 也高」，最可能是混淆不是因果**：swap 在本機是**單調累積、永不回收**的
（`MEMORY_HYGIENE.md:44` 實測 `purge` 不減 swap_used），所以任何隨時間惡化的東西
（熱／DVFS 降階、背景 GPU 客戶端、累積負載）都會跟它同向。`MEMORY_PERF.md:537` 早已為 prefill
判過同一句話：**「`swap` 不是因是果」**；`§EN-351` 用 n=4 相關係數給了同方向的否證
（`swap +0.761`，臨界 `0.950`，沒過），且有明確反例（r2 水位四軸全比 r1 好，`cb` 卻 +42%）。

**但這不代表沒事**：真正的病是**結構性超訂 4.7 GB**，swap 只是它的症狀。修法必須對著超訂打，
不能對著 swap 打 —— 下面 §4 會證明「縮 pool 降 swap」這帖藥會**真的把 miss 抬高**。

---

## 1. count 通道的判決證據

`SWAP_STRUCTURAL_FIX_2026-09-24.md` §2 的兩趟 run（完全一致）：

```
累計填池 24.7 GiB ／ 從盤讀 19.6 GiB ／ evictions 3872·4037
miss 歸因： compulsory 91.4% / capacity 8.6%   （run2：90.8% / 9.2%）
decode hit 95.6% / 95.4%
```

**compulsory = 這個 (layer, expert) 從沒被載過**（`llama-expert-cache.cpp:893-900` 的定義：
「first demand touch … = compulsory；a second touch means it was resident once and got evicted = capacity」）。

⇒ **91% 的 miss 是「第一次摸到」，跟剔除、跟水位、跟 swap 都無關。**
swap 頂多能碰得到剩下那 **8.6%** 的 capacity miss。就算 swap 把 capacity miss 翻倍，
總 miss 也只動 8.6pp —— 而下面是反向的實測。

### 1.1 一條反向的實測（同文 §4，熱污染但相對可信）

| pool | decode t/s | hit |
|---|---:|---:|
| 8G | 9.98 | **96.0%** |
| 6G | 6.56 | 93.3% |
| 5G | 7.99 | 91.4% |
| 4G | 8.34 | **87.6%** |

**pool 越大 ⇒ 記憶體壓力越大 ⇒ swap 越大，但 hit 越高（miss 越低）。**
方向上與「swap 大 ⇒ miss 高」相反。真正控制 miss 的是**槽數（capacity）**，swap 是它的下游結果。

---

## 2. cost 通道為什麼成立（機制）

`--load-mode none`（`run_server.sh:507` 的**預設**）下，模型的 13.65GB 被 `read_raw` 讀成
**匿名頁**（`run_server.sh:1014-1017`：`_MODEL_KIND="匿名頁、不可回收"`）。
匿名頁被回收要先**寫**到 swap 再**讀**回來 —— 讀寫兩次 IO；
而 file-backed clean 頁是直接丟掉、重讀一次。這是 2× 的 IO 不對稱。

結果就是：**hit 在快取中繼資料上仍然是 hit（不計入 miss），但它觸發的頁已經在 swap 上了**
⇒ 你看到 t/s 崩、hit% 卻紋風不動。這跟 §1 的數字完全吻合。

---

## 3. 真因：結構性超訂 4838 MiB（有代碼行號）

```
模型檔      13030 MiB   （13.65GB；其中 expert = 256×40×1.0703 = 10960 MiB，dense/其他 2070 MiB）
expert pool  8192 MiB
─────────────────────
靜態需求    21222 MiB   vs 實體 16384 MiB  ⇒ 超訂 4838 MiB
```

`run_server.sh:1056-1057` 本身就會印這一行（並在超訂時印 `OVERSUBSCRIBED`，
可用 `CGC_SERVER_STRICT_BUDGET=1` 讓它直接拒跑）。

**expert 權重其實有兩份潛在駐留**（`SWAP_STRUCTURAL_FIX` §7.1 的事實鏈）：

| # | 事實 | 證據 |
|---|---|---|
| 1 | prod-new 設了 `LLAMA_EXPERT_CACHE_ALLOW_NGL=1` | `pool_sweet_spot.sh:14` |
| 2 | `ALLOW_NGL` ⇒ `expert_cache_skip_load = true` ⇒ expert 走 skip-load | `llama.cpp:384`；`llama-model-loader.cpp:1113` |
| 3 | skip-load 的 tensor 載入時**仍被 `read_raw` 全量讀進 CPU RAM**（匿名堆） | `llama-model-loader.cpp:1844-1850` |
| 4 | 但 compute 時 GPU FFN 讀的是 **pool**（`pread` 直接寫 pool buffer） | `llama-expert-cache.cpp:35/149`；`adopt_pool_region :1881` |

⇒ **expert 的 CPU 副本是純佔位**，佔掉 ~10.9 GB 匿名記憶體，卻從沒被算過。

---

## 4. 修法排序（按「能不能真的解決超訂」排）

| 方案 | 作法 | 駐留需求 | 對 miss 的影響 | 狀態 |
|---|---|---:|---|---|
| **P0 = L2** | skip-load 的 expert **不 `read_raw`**（資料由 pool fill 的 pread 供給，CPU buffer 保持佔位） | **10262 MiB（餘裕 6122）** | **不動**（槽數不變） | ✅ **已套用**（`CGC_EXPERT_SKIP_READRAW=1`，預設關） |
| ~~L1 縮 pool~~ | 8G→4G | 17126 MiB（**仍超訂 742**） | **hit 96→87.6%，−8.4pp** ❌ | **不要用** |
| ~~切 `--load-mode mmap~~ | 模型變 file-backed | 8192 MiB（餘裕 8192） | 理想 | **mmap 存活 0/3，不可用** ❌ |
| **P1** | **fill 前對目標 slot 範圍 `madvise(MADV_DONTNEED)`** | 不減需求，但砍掉「覆寫前的 swap-in 讀」 | 不動 | ✅ **已套用**（`CGC_POOL_MADVISE=1`，預設關） |
| P2 | evict 時對該 slot `madvise(MADV_DONTNEED)`（內容本來就作廢） | 砍掉 garbage 的 swap 寫 | 不動 | ✅ **已套用**（`CGC_POOL_MADVISE=2`，預設關） |
| P3 = L3 | swap 進 autotuner 目標；`LAYER_CAPS` 逐層容量；swap 觸發就收 pool | 閉環 | 受控 | 未動手 |

### 4.1 為什麼「縮 pool」這帖藥是錯的

`run_server.sh` 的算術顯示：**光靠縮 pool，要壓到 3072 MiB 才不超訂**——那時每層只剩約 53 槽
（今天 143 槽），hit 會崩。而 8G→4G 只把超訂從 4838 砍到 **742，病根還在**，
卻先付掉 8.4pp 的 hit（§1.1 實測）。**拿 miss 換 swap，方向反了。**

### 4.2 為什麼「切成 mmap」這帖藥也不可用

`run_server.sh:577` 的 interleaved 存活紀錄：

```
none ub=4096/5120/5632/6144 pool 8GiB   survived 5/5, 3/3, 4/4, 0/5
mmap ub=4096/5120/6144      (any width) survived 0/3      ← 全滅
```

所以雖然 `CGC_MMAP_POOL_FIX` 的 `_Pool` buft **已在當前 binary 裡**（本輪實測
`strings libggml-metal*.dylib | grep -c _Pool` = 1），mmap 這個組合本身仍然站不住。
**結論：修法必須落在 skip-load 的 `read_raw` 分支，不是切 load-mode。**

### 4.3 P1（fill 前 madvise）為什麼值得做

pool 的目標頁接下來會被 `pread` **整塊覆寫**。但內核在覆寫前必須先把那頁**物化**——
若它在 swap 上，就要先做一次 swap-in 讀，讀進來的內容下一刻就被丟掉。
這是**純浪費的 IO**。`madvise(MADV_DONTNEED)` 直接把這步省掉。

安全性：只對 **malloc'd pool** 做（用 `pool_region()` 區分 `pool_ext`（Metal）與 `pool`（malloc），
`llama-expert-cache.cpp:715-730`），且緊接著就是整塊覆寫，語義安全。

### 4.4 P0／P1／P2 落地規格（2026-09-24 10:4x 已套用、**10:47 已 build**，env 預設關閉）

**build**：`cmake --build src/llama.cpp/build --target llama-server llama-bench -j 8` → rc=0；
`strings src/llama.cpp/build/bin/libllama.dylib | grep -E '^(CGC_EXPERT_SKIP_READRAW|CGC_POOL_MADVISE)$'`
兩個都命中 ⇒ 開關確實編進去。

⚠ **這個 binary 同時含線 I 未提交的 294+ 行 `llama-expert-cache.cpp` 改動**，所以
  「不設 env」的對照臂 **不等於 12.57 錨點**（那個數字是 ρ layer-batch 之前的 build）。
  A/B 只能三臂互比，不能拿單臂數字去對 12.57。

**狀態：三個都寫進引擎了（`scripts/check/swap_p0p1p2_apply.py --apply`，`--self-test` 11/11、
`--check` 11/11 ALREADY），兩個檔案都通過 `c++ -fsyntax-only`（未 build、未跑 GPU）。**
但**三個開關預設都是關的** —— 不設 env，行為與舊路徑逐位元組相同，所以它可以安全地躺在
線 I 的髒檔裡，不會動到任何正在跑的量測。

| 開關 | 值 | 做什麼 |
|---|---|---|
| `CGC_EXPERT_SKIP_READRAW=1` | 預設關 | **P0**：skip-load expert 不 `read_raw`（省 ~10.9 GiB 匿名駐留） |
| `CGC_POOL_MADVISE=1` | 預設關 | **P1**：fill 前對目標 slot 丟頁 |
| `CGC_POOL_MADVISE=2` | 預設關 | **P1 ＋ P2**：再加上 evict 時丟頁 |

還原：`python3 scripts/check/swap_p0p1p2_apply.py --revert`（備份在 `.workbuddy/patch_bak/`，
**故意不放原始檔旁邊** —— 放旁邊會變 untracked 新檔、污染別人的 `git status`）。

⚠ **落地時踩到並修掉的東西**：`cgc_pool_madvise_mode()` 在 `pick_slot`（約 :700）被呼叫，
但定義在 `pool_region` 之後（約 :800，因為要用 `slots_l`）⇒ 少一個前向宣告就是
**編譯錯誤**，而這個檔的編譯錯誤會擋掉所有人的 build。已補兩個前向宣告。

⚠ **P1/P2 最危險的一點（helper 存在的理由）**：expert stride = 1.0703 MiB = 68.5 個 16 KiB 頁，
**不是頁的整數倍** ⇒ 範圍若往外取整，`madvise` 會清掉**前後 slot 的頭尾**，而且不報錯。
所以只丟**完整落在範圍內**的頁（start 往上取整、end 往下取整，無整頁就不丟）。
實測（自檢第 2 項）：6 個連續 slot 全部零越界，每個 slot 丟到 68 頁。
另外 Metal 的 `pool_ext` 不是匿名 malloc，`cgc_in_malloc_pool` 會先把非 malloc 的 dst 擋掉。

改動點是單一支、單一行 —— `llama_model_loader::load_data_for()`
（`llama-model-loader.cpp:1640`）：

```cpp
    } else {
        GGML_ASSERT(cur->data != nullptr);
        GGML_ASSERT(w.idx < files.size());
        const auto & file = files.at(w.idx);
        file->seek(w.offs, SEEK_SET);
        file->read_raw(cur->data, ggml_nbytes(cur));   // ← :1655，skip-load expert 走這裡
    }
```

要的是「**這一行對 skip-load expert 不做**」——buffer 照樣 malloc（佔位），位元組不讀。
省下來的正是那 10960 MiB（匿名、只寫一次、從此不再讀）。

⚠️ **三個必須一起守的邊界**（讀碼翻出，漏一個就靜默出錯或白做）：

1. **`expert_cache_l4_skip_layer0 && l4_il == 0`（`:1505-1511`）例外。**
   `L4_SKIP_LAYER0` 下的 blk.0 被**故意**降回「普通 CPU skip-load tensor」（`l4_kind = -1`），
   它的 FFN **真的從 CPU buffer 讀**（`llama-context.cpp:6437`）。P0 一把跳過 ⇒ layer-0 讀到
   zeros ⇒ 整網輸出變垃圾、且不報錯。
   觸發條件 `CGC_SERVER_SKIP0`（`run_server.sh:1445`，預設 0 ⇒ 交付配置安全），
   但 **P0 必須寫成 `if (!(skip_load_expert && !(l4_skip_layer0 && il == 0)))`**，不能只判 skip-load。
2. **`check_tensors` 要一起擋**（`:1657` 緊接著 `ggml_validate_row_data`）。
   跳過 read_raw 之後 buffer 是未初始化，`--check-tensors` 會直接 throw `invalid data`
   （`:1698` 的 `use_mmap || check_tensors` 分支同理）。P0 跳過時要連 validation 一起跳。
3. **`load_data_for` 拿不到「我是不是 skip-load expert」這個事實** —— 它只看
   `require_weight(name)`。skip-load 的判定在 `:1364-1371`
   （`expert_cache_skip_load && strstr(name,"_exps") && strstr(name,"blk.")`，**且前提是 `!buft`**）。
   ⇒ 正確做法是在 `:1364` 那個分支裡把 `is_skip_load_expert` 記進 weight/Tensor 的旁表
   （`load_data_for` 再查），**不要在 `load_data_for` 裡重寫一次 `strstr`**（`!buft` 前提會漏判）。

**P1 的安全邊界**（同場讀碼確認）：`pool` 是 malloc、`pool_ext` 是 Metal（`pool_region()`，
`llama-expert-cache.cpp:715-730`）⇒ 只對 `pool` 做 `madvise`，對 Metal 做會踩別人的 buffer。

---

## 5. 動 P0 之前必須先量的三個前提（§7.3 既有清單，本輪未推進）

1. **expert 實際駐留形態**：跑一次 load + 看 RSS／swap 增長，確認 skip-load expert 真的被 `read_raw`。
   （`footprint` 的 `MALLOC_LARGE` 只有 161MB，與「全量讀進 RAM」矛盾 ⇒ **需實測**，
   可能 `ALLOW_NGL=1` 讓 expert 走 adopt 而非 skip-load。）
2. **MoE kernel 100% 從 pool 讀**：確認 `MUL_MAT_ID`／`GLU_FUSED_DOWN`／`CGC_MM_BITIDENT` 路徑
   沒有任何 fallback 去讀 `tensor->data`（CPU 副本）。**有 fallback 就不能跳 `read_raw`。**
3. `LLAMA_EXPERT_CACHE_ALLOW_NGL` 的實際語意（決定 adopt vs skip-load）。

---

## 6. 驗證判據（怎麼知道修好了）

### 6.0 讀碼覆核（2026-09-24 10:1x，逐行對過）

miss 分解計數器**一直都在跑**（`n_miss_compulsory` / `n_miss_capacity`）——兩個累計點
`llama-expert-cache.cpp:891-902`（`ensure_slot`）與 `:1167-1172`（batch 路徑）**都沒有 env 門控**
（只有 per-layer dump 被 `LLAMA_EXPERT_CACHE_MISS_ATTR_LAYERS` 門住）。✅ 原敘述成立。

印出處（**原敘述有一處行號與一處欄位名要更正**）：

- `llama-shape-knob.cpp:230-250`（fprintf 起始 :230，欄位在 :234）：
  `req= hits= misses= hit_pct= compulsory= capacity= evict= ... read_mib= pread_us= fill_wait_us=`
  ✅ 與原敘述一致（這裡確實是縮寫 `evict=`）。由 destructor 無條件呼叫
  （`llama-expert-cache.cpp:2573`）⇒ **每趟一定會印**（前提是乾淨關機）。
- `llama-expert-cache.cpp:2764`（**不是 2765**；fprintf 起始行）stats block：
  `miss attribution: compulsory=N capacity=N (x% / y% of T)  evictions=N  layers_distinct_over_slots=N ...`
  ⚠ 欄位是 **`evictions=`（全名）不是 `evict=`**，且這一行**沒有 hit_pct**。
  hit rate 與 `pread_usec=` 在另一行 `final stats:`（`:2553`）。
- `pread_usec` 有三個出口，門控與可靠性**完全不同**（這是 09-23 log 全部零命中的原因）：
  1. `CGC-RIG-SNAPSHOT ... pread_usec=`（`:520`）—— **有 env 門控**：
     必須設 `CGC_PREFILL_PROTECT_FILE`（`:535-537` 沒設就直接 return），且檔案內容變動才印。
  2. `final stats: ... pread_usec=`（`:2553`）—— 無門控，但**依賴乾淨關機**
     （同檔 `:515-518` 自己的註解：SIGINT 時 destructor 可能跑不到）。
  3. `CGC-SHAPE ... pread_us=`（`:236`）—— 無門控，同上。
  ⚠ `pread_usec` 是**跨 worker thread 的加總**（`llama-expert-cache.h:600`），不是 wall clock
  ⇒ 比大小要在 WORKERS 相同下（prod-new `WORKERS=8` ✓）。

**判據（三條同時成立才算修好，單條不算）：**

1. `dswap = swap_after − swap_before` ≈ 0（連跑多趟不累積）——既有 `pool_sweet_spot.sh:51`／
   `pin_abba.sh:57` 已在記這個欄位 ✅ 覆核無誤。
2. `hit_pct` 不降（槽數沒動，hit 沒有理由降）。
3. `capacity` 不升（**用 per-1k-req，不要用佔比**，理由見下）＋ `pread_usec`/`pread_us` 顯著下降。

⚠️ **第 3 條的口徑陷阱（本次覆核新發現，原敘述漏了）**：`ever_loaded[layer][expert]`
只在 init 清一次（`:3738`），跑途中**永不重置** ⇒ `compulsory` 的上界是
`n_layer × n_expert`（40×256 = **10240**），之後分子飽和、分母（總 miss）繼續長
⇒ **capacity 佔比會隨 run 長度機械性上升**，兩趟 req 差很多時直接比佔比會誤判。
所以判據改用 `capacity × 1000 / req`；req 差異 >10% 時該比較本身不可信。

⚠️ 只看到 1 不成立而 2 成立 = 只是把 swap 推遲；只看到 2 不成立 = 撞到 §5 前提 2 的 fallback。

**機檢工具**：`scripts/check/miss_attr_gate.py`（`--self-test` 15/15）——
`--row base=LOG:SWAP_B:SWAP_A --row fix=LOG2:...`，逐條印 PASS/FAIL；
**缺欄位一律判「未驗」並當 FAIL**（不能靠省略欄位混過）。

---

## 7. 歸屬

- P0／P1／P2 都動 `llama-expert-cache.cpp`（池填充/剔除）與 `llama-model-loader.cpp`（載入）
  —— 2026-09-24 10:4x 已由本線套用（env 預設關）、10:47 已 build，詳 §4.4；
  **尚未實測**（等重開機拿乾淨視窗）。A/B 一步到位：`scripts/check/p012_ab.py`。
  ⇒ **線 I**（`cb`＝expert cache 填池 IO、快取命中儀器）。
- 本輪只做讀碼定案與既有實測重算，**未改引擎代碼、未 build、未跑 GPU**（swap 7766/9216 MiB 已污染）。

### 7.1 立即可做項已完成（2026-09-24 10:0x）：`scripts/check/budget_gate.sh`

- 最新 commit `3f0e97140` 只加了**工具**（`budget_preflight.py`）與**文件規則**（NEXT_ACTIONS），
  **沒有**把閘門接進任何量測腳本 ⇒ 工具不會自己被叫到。補上接線面：
  `scripts/check/budget_gate.sh`（可 source，一行接入；`--self-test` 7/7）。
- 已接：`rho_fill_ab.sh`／`route_overlap_3prompt.sh`／`masscov_decode_shape.sh`（本線）。
  其他 runner（含線 I 的 `pool_curve.py`、`decode_sweep.py`）要接只需一行：
  `. "${REPO}/scripts/check/budget_gate.sh"`（launch 之前）。
- 模式：`BUDGET_GATE=strict`（預設，超訂 exit 2 拒跑）／`warn`（放行但 export
  `CGC_BUDGET_OVERSUBSCRIBED=1`，樣本帶 OVERSUBSCRIBED 標記）／`off`（不檢查）。
  未超訂時額外 export `CGC_SERVER_STRICT_BUDGET=1`，讓 `run_server.sh` 自己也擋一次（同口徑）。
- ⚠ **16 GB 這台的現實**：prod-new pool 8 GiB 靜態超訂 **4838 MiB**（13030 + 8192 = 21222 > 16384，
  與實測 resident 21222 MiB 對上）⇒ **strict 會把目前的交付 cell 一併拒跑**。
  所以「拒跑」不是常態開關，而是**強迫選擇**：要麼 `BUDGET_GATE=warn` 明確承認樣本帶污染標記，
  要麼把 pool 降到 ≤ 3 GiB（16384 − 13030 = 3354 MiB）——兩者都不做就拿不到數字，
  這正是本閘門的目的（不再有「跑完才發現不可引用」）。
- ⚠ 兩個實作的坑（第一版都踩了）：① 被 source 的檔案用 `return` 只會結束自己、呼叫方會繼續跑
  ⇒ 拒跑必須用 `exit`；② `$( . gate; printf "$VAR" )` 抓不到 export ⇒ 標記測試要在
  **同一個 subshell 內** source 完再判斷。

---

## 8. 一句話回答

**swap 不會讓 miss 變多，它讓 hit 和 fill 都變貴——miss 次數由槽數與置換策略決定，swap 是下游。
真正的病是 expert 權重在 CPU 副本與 pool 裡各住一份，導致 21222 MiB 塞進 16384 MiB；
修法是讓 skip-load 的 expert 不再 `read_raw`（駐留降到 10262 MiB，槽數與 hit 都不動），
而不是縮 pool（那會先付 8.4pp 的 hit 卻只砍掉 84% 的超訂）。**
