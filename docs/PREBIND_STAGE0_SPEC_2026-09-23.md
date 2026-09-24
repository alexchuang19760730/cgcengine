# 階段 0 儀器規格：`CGC-PREBIND-PROBE`

**日期** 2026-09-23 · **性質** 只加儀器、不改行為的實作規格 · **歸本線（線 A）**
**上游**：`docs/PREBIND_TRADEOFF_2026-09-23.md`（定價與門檻）、
`docs/PREBIND_SLOT_DESIGN_2026-09-23.md`（原設計，§6／§7 已被上游更正）
**配套工具**：`scripts/check/prebind_probe_parse.py`（解析本規格的 log，含 selftest）

---

## 0. 一句話

**一次跑就把四個數字量回來：`uni_size`（真 U）、`q8/q16/q24`（三種預測寬度的覆蓋率）、
`p_res0`（真 union 的常駐率）、`all_clean`（整層零 miss 的層數）。**
不改任何行為、不改 `slot_table`／`slot_owner`／remap、預設關閉。

---

## 1. 為什麼 gate 要改寫成「量 q 本身」

原設計 §7 的門檻是 `all_cov_layers / n_routable >= 0.5`。作廢，兩個理由：

1. **`f_clean`（= `all_cov_layers/n_routable`）是 q 的函數**（`f_clean = r^U`、
   `r = p_res0 + (1-p_res0)q`）。拿函數當 gate，等於把「門檻」和「模型」綁在一起 ——
   模型一改，門檻的意義就跟著變，而門檻必須是**可以事前寫死、事後不改**的東西。
2. **q 才是能直接跟定價對齊的量**。`prebind_ev.py --breakeven` 給的是
   「step −10% 需要 q >= 0.731」，量 q 就能直接查表；量 f_clean 還要再轉一次。

⇒ **gate：`q24 >= 0.73`。** `all_clean` 降級為**只記錄**（它剛好是模型的驗算點）。

---

## 2. ★ 一個重要發現：這四個數字有 3 個已經在樹上，只是沒被接起來

| 要量的東西 | 樹上既有的東西 | 位置 |
|---|---|---|
| 真 union 大小 U | `std::vector<uint32_t> uni`（已去重、已排序） | `llama-context.cpp:5530-5547` |
| 常駐率 `p_res0` | `slot_table.data() + il*n_expert`，`st[e] < 0` ⇒ cold | `llama-context.cpp:6353-6360` |
| **覆蓋率 q（寬度 8）** | 既有 hitrate 區塊 `hit/(hit+miss)`，遍歷 `uni` 對 `pred_set` | `llama-context.cpp:5746-5770` |

⇒ 既有的 hitrate 區塊**已經在量 q**，只是：

- 被關在 `CGC_DRAFT_PREFETCH` 後面（那個 env 同時會**改變行為**：在 `il==1` 真的發 prefetch）；
- 只取 draft 的 **j=0 一行（8 ids）** ⇒ 量到的是寬度 8，而寬度 8 的結構上限只有 51.6%；
- 只在 `il <= 1` 印，40 層裡 38 層的數字只有累計值、沒有分佈。

**本規格做的事：把這個量測獨立成自己的 env、把寬度開到 24、把每層都印出來。**

---

## 3. 插入點（五處）

| # | 檔案 | 錨點 | 動作 |
|---|---|---|---|
| 1 | `src/llama.cpp/src/llama-expert-cache.h` | `:512-513`（`draft_prefetch_ids` / `draft_prefetch_valid` 旁） | 新增兩個成員 |
| 2 | `src/llama.cpp/src/llama-expert-cache.cpp` | `:3492-3493`（兩個 `assign` 旁） | 初始化 |
| 3 | `src/llama.cpp/src/llama-context.cpp` | `:5386-5396`（draft COLLECT 區塊內） | 收集**全部** token 行 |
| 4 | `src/llama.cpp/src/llama-context.cpp` | `:5746-5770`（既有 hitrate 區塊旁） | 新增 probe 區塊 |
| 5 | 同上 | 同區塊結尾 | 逐層印 ＋ 每步彙總 |

⚠ 行號會漂移。**以錨點字串為準**：
#3 找 `// [CGC MTP Draft Prefetch 2026-09-07] COLLECT phase`；
#4 找 `// [CGC MTP Draft Prefetch 2026-09-07] HIT-RATE measurement`。

---

## 4. 程式碼

### 4.1 宣告（`llama-expert-cache.h`，貼在 `draft_prefetch_valid` 後面）

```cpp
    // [CGC prebind probe 2026-09-23] MEASUREMENT ONLY. draft ctx 一次算好 n_tokens 行
    // top-8；既有 draft_prefetch_ids 只留下來的第一行（prefetch 消費端依賴那個寬度，
    // 加寬它會改變行為）。這裡另外存全部的行，只給 probe 讀。
    std::vector<std::vector<uint32_t>> draft_all_ids;   // [layer] 全部 token 行的 ids
    std::vector<bool> draft_all_valid;                  // [layer] 本 round 收集過
```

### 4.2 初始化（`llama-expert-cache.cpp`，兩個 `assign` 旁）

```cpp
    cache->draft_all_ids.assign(max_layer, std::vector<uint32_t>());
    cache->draft_all_valid.assign(max_layer, false);
```

### 4.3 收集（`llama-context.cpp` COLLECT 區塊內，緊接 `cache->draft_prefetch_valid[il] = true;`）

```cpp
        // [CGC prebind probe 2026-09-23] 同一個 ids tensor，但把全部 token 行都存下來。
        // ids layout 是 [n_expert_used, n_tokens]，元素 (i,j) 在 i + j*n_expert_used。
        // 既有程式碼只取 j=0（理由：「verify 從第一個預測 token 驗證」）—— 那個理由對
        // prefetch 的優先順序成立，對 prebind 的覆蓋率不成立。這裡不改既有行為。
        if (cgc_prebind_probe) {
            std::vector<uint32_t> & dstAll = cache->draft_all_ids[il];
            dstAll.resize((size_t)(n_tokens * n_expert_used));
            for (int64_t j = 0; j < n_tokens * n_expert_used; ++j) {
                dstAll[(size_t) j] = (uint32_t) ids[j];
            }
            cache->draft_all_valid[il] = true;
        }
```

其中 `cgc_prebind_probe` 在該函式內靜態初始化一次（沿用 `:5574` 的寫法）：

```cpp
    static const bool cgc_prebind_probe = getenv("CGC_PREBIND_PROBE") != nullptr &&
                                          getenv("CGC_PREBIND_PROBE")[0] != '0';
```

⚠ 這一處必須放在 COLLECT 區塊的 `if (cgc_draft_prefetch_on() && ...)` **外面**，
否則沒開 `CGC_DRAFT_PREFETCH` 就收集不到 —— 而我們要的正是**不開那個 env**。

### 4.4 probe 區塊（`llama-context.cpp`，貼在既有 hitrate 區塊之後）

```cpp
    // [CGC prebind probe 2026-09-23] MEASUREMENT ONLY —— 不寫 slot_table／slot_owner／
    // remap，不呼叫 ensure/prefetch/touch。只讀 uni、draft_all_ids、slot_table 做比較。
    //
    // 放在這裡（既有 hitrate 區塊旁）是刻意的：本區塊在 ensure_batch（:6735 附近）
    // 之前，所以 slot_table 讀到的是「hook 當時」的常駐狀態 = p_res0，不是填完之後。
    if (cgc_prebind_probe &&
        cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT && n_tokens >= 1 &&
        il >= 0 && (size_t) il < cache->draft_all_ids.size() &&
        cache->draft_all_valid[(size_t) il] && !uni.empty()) {

        const std::vector<uint32_t> & allp = cache->draft_all_ids[(size_t) il];
        const size_t per_row = (size_t) n_expert_used;
        const size_t n_rows  = per_row ? allp.size() / per_row : 0;

        // 逐寬度覆蓋率：pred_m = draft 的前 m/8 行。q = |uni ∩ pred| / |uni|。
        auto cov_hits = [&](size_t rows_used) -> size_t {
            std::unordered_set<uint32_t> ps;
            const size_t rr = rows_used < n_rows ? rows_used : n_rows;
            ps.reserve(rr * per_row);
            for (size_t r = 0; r < rr; ++r) {
                for (size_t k = 0; k < per_row; ++k) ps.insert(allp[r * per_row + k]);
            }
            size_t hit = 0;
            for (uint32_t e : uni) if (ps.count(e)) ++hit;
            return hit;
        };
        const size_t h8  = cov_hits(1);
        const size_t h16 = cov_hits(2);
        const size_t h24 = cov_hits(3);

        // 常駐率（p_res0）：與 :6353 完全同一個口徑。
        const int32_t * st = cache->slot_table.data() + (size_t) il * cache->n_expert;
        size_t resident = 0;
        for (uint32_t e : uni) {
            if (e < cache->n_expert && st[e] >= 0) ++resident;
        }
        const size_t U = uni.size();

        // 累計（probe 用；單執行緒 hook，多 slot 併發時數字會混，見 §8）
        static uint64_t s_layers = 0, s_uni = 0, s_h8 = 0, s_h16 = 0, s_h24 = 0;
        static uint64_t s_res = 0, s_clean = 0, s_steps = 0;
        static int s_last_il = -1;
        if (il <= s_last_il) ++s_steps;          // 新的一步（il 回到 0 或更小）
        s_last_il = il;
        s_layers += 1; s_uni += U; s_h8 += h8; s_h16 += h16; s_h24 += h24;
        s_res += resident;
        if (resident == U) ++s_clean;

        if (getenv("CGC_PREBIND_PROBE_VERBOSE") != nullptr) {
            fprintf(stderr, "CGC-PREBIND-PROBE: il=%d ntok=%lld uni=%zu rows=%zu "
                            "cov8=%.3f cov16=%.3f cov24=%.3f res=%.3f clean=%d\n",
                    il, (long long) n_tokens, U, n_rows,
                    (double) h8  / (double) U, (double) h16 / (double) U,
                    (double) h24 / (double) U, (double) resident / (double) U,
                    resident == U ? 1 : 0);
        }
        if (il == 0) {
            fprintf(stderr, "CGC-PREBIND-SUM: steps=%llu layers=%llu uni_avg=%.2f "
                            "q8=%.4f q16=%.4f q24=%.4f p_res0=%.4f clean_pct=%.1f%%\n",
                    (unsigned long long) s_steps, (unsigned long long) s_layers,
                    s_layers ? (double) s_uni / (double) s_layers : 0.0,
                    s_uni ? (double) s_h8  / (double) s_uni : 0.0,
                    s_uni ? (double) s_h16 / (double) s_uni : 0.0,
                    s_uni ? (double) s_h24 / (double) s_uni : 0.0,
                    s_uni ? (double) s_res / (double) s_uni : 0.0,
                    s_layers ? 100.0 * (double) s_clean / (double) s_layers : 0.0);
        }

        cache->draft_all_valid[(size_t) il] = false;   // one-shot，避免讀到上一 round 的
    }
```

---

## 5. ★ 零行為改變的證明（這五點缺一不可）

1. **新 buffer 沒有任何既有 consumer。** `draft_all_ids` 只有 probe 讀；
   `draft_prefetch_ids`（既有 prefetch 消費端在 `:5629`）**寬度維持 8**。
2. **env 預設關。** `CGC_PREBIND_PROBE` 未設 ⇒ `cgc_prebind_probe == false`
   ⇒ 收集與 probe 都不跑。
3. **不寫任何狀態。** probe 只讀 `uni`／`draft_all_ids`／`slot_table`；
   不呼叫 `ensure_batch`／`ensure_slot`／`prefetch_slot`／`touch`／`wait_loading`。
4. **`draft_all_valid` 不影響既有邏輯**（它是新的 flag，既有 `draft_prefetch_valid` 不動）。
5. **不進 graph。** 不在 remap leaf 的路徑上，不改變 `ne[2]`。

⇒ 這是「可以安心編進去留著」的儀器，不是一次性的補丁。

---

## 6. gate（**事前寫死，事後不改**）

| # | 判準 | 門檻 | 不過的話 |
|---|---|---|---|
| 0 | 有效性 | `layers > 0` | 宣告本輪無效，不解讀其他數字（F2 的坑） |
| 0b | ★ q 有沒有量到 | `pred_pct > 0` | `pred_pct == 0` ⇒ 宣告「**q 不可得**」，**不判 FAIL** |
| 1 | **主判準** | 見 §6.3 三區規則 | 預指派不值得 ⇒ **停在階段 0**，改走 `pread_usec` 路線 |
| 2 | 模型檢定 | 見 §6.2「q=0 檢定」 | logistic-normal 隨機效應不適用 ⇒ 先修模型，不進階段 1 |
| 3 | 輸入檢查 | 實測 `uni_avg`/`p_res0` 與導出門檻時用的輸入一致 | 門檻失效 ⇒ 退出碼 3，重算後再判 |
| 4 | 衝突拆解 | 回報 `p_res0`（見 §7） | ——（這是記錄項，不是 gate） |

### 6.1 ★ 門檻不是常數：它是 (U, p_res0, sigma) 的函數

`q24 >= 0.73` 是**舊的模型輸入**（U=15.5、p_res0=0.85、sigma=0）導出來的。
2026-09-23 第一支實測 log 把 U 與 p_res0 直接量出來 ⇒ 門檻必須跟著變：

| 輸入 | q 門檻（step −10% 兩平） |
|---|---|
| U=15.5, p=0.85, sigma=0（舊假設） | **0.7312** |
| U=20.23, p=0.9016, sigma=0（實測 U/p，仍假設獨立） | **0.7042** |
| U=20.23, p=0.9016, sigma=1.019（實測 + 相關性） | **0.5948** |

重算：`prebind_ev.py --breakeven 0.10 --u 20.23 --p-res0 0.9016 --sigma <s>`。
⚠ 這是「模型輸入被量測更新」，不是事後放寬 gate：決策規則（step −10% 兩平）
從頭到尾沒變，變的是代入的 U／p_res0／sigma。

### 6.2 ★ sigma 只能由「逐層 res 的變異數」估，不能由 `clean_pct` 反解

sigma 若用 `clean_pct` 反解就是**循環論證**：`clean_pct` 正是模型要預測的量 `E[r_z^U]`，
拿它配適出的 sigma 再去降門檻 ⇒ 用來放寬的數字 == 放寬後要預測的數字。

改用**另一個統計量**：逐層常駐比例 `res=` 的變異數（二階矩）。

    res_layer = (1/U) · Σ Bern(r_z)   =>   Var[res] = Var[r_z] + E[r_z(1-r_z)]/U

`prebind_ev.sigma_from_res_spread()` 用不動點迭代反解（二項修正項本身也依賴 sigma）。
這樣 `clean_pct`（U 階矩）就從「配適點」變成**驗證點** ——
「二階矩能不能預測 U 階矩」是一個可以為偽的檢定。

實測（第一支 log，1880 層）：mean(res)=0.9062、sd=0.1078 ⇒ **sigma_hat = 1.019**，
模型預測 clean(q=0) = 28.9% vs 實測 32.9%，**差 +4.0 pp（±10 pp 內 ✓）**。
對照：由 `clean_pct` 反解會得到 1.191 —— 兩個數字接近是函數形式的佐證，
但**只有前者可以用來定門檻**。

★ 附帶修掉一個會吃掉 gate 的 bug：舊版把實測 `clean_pct` 拿去比 `f_clean(q24)`。
但 probe 不改行為 ⇒ 實測值永遠是 **q=0 的基線**，拿基線比介入後是兩個不同的點，
差值符號固定為負 ⇒ 每一輪都會被判「模型不符」而 FAIL。現在比的是 `f_clean(q=0)`。

### 6.3 ★ 三區判決規則（事先寫死）

決策規則固定為「step −10% 損益兩平」，數字隨實測輸入變。令
`g0 = 門檻(sigma=0)`、`gs = 門檻(sigma_hat)`（本次 = 0.7042 / 0.5948）：

| 區間 | 結論 |
|---|---|
| `q24 >= g0` | **PASS**：連獨立下界都過，結論不依賴相關性假設 |
| `gs <= q24 < g0` | **PASS（附條件）**：只在實測相關性下過；獨立假設會判 FAIL |
| `q24 < gs` | **FAIL**：連實測相關性都救不了 ⇒ 停在階段 0 |

解析器印出落在哪一區；PASS/FAIL 仍以 `--gate` 為準（**門檻是輸入，不是輸出**），
但 `--gate` 必須是實測輸入導出的值，否則判「門檻失效」（退出碼 3）。

**寬度選擇規則（事先寫死）**：若主判準過，階段 1 用**能過門檻的最小寬度**
（先試 16，不行才用 24）—— 寬度越大，影子 ensure 越貴、LRU 擾動越大。

---

## 7. 順手拆掉 `p_res0` 的 3 倍矛盾

`docs/PREBIND_TRADEOFF_2026-09-23.md` §8 記的矛盾：註解寫 15% 冷缺口（⇒ 0.85），
而由 F1 係數反解實測 `cb=39.3` 得到 ≈ 0.95。本規格的 `p_res0` 是**直接量**的
（`slot_table` 查表，與 `:6353` 同口徑），一次解決。

⚠ 既有 `CGC-COLD-GUARD`（`:6375`）雖然也印 `cold/uni`，但**只在不合格時印**
（`metric > CGC_FAST_COLD_MAX`，預設 0.30）⇒ 樣本被截斷偏高，**不能拿來當 p_res0**。
本規格是無條件量。

---

## 8. 跑法（一次就夠；不需要配對）

```
./scripts/check/prebind_probe_run.sh            # 等窗口 -> 跑 -> 印解析指令
python3 scripts/check/prebind_probe_parse.py $(ls -t /tmp/prebind_probe/*.stderr.log | head -1)
```

`prebind_probe_run.sh` 自己帶三道閘門（8080 無 listener／無其他 llama 行程／usable ≥ 30%），
並走 `llama_bench_matrix.py --arms prod25-stream`（= 12.57 anchor 臂，同一條路徑）。

- **只需要一支 log。** 主指標是 step 級計數器，一支 log 有 40 層 × 上千步的樣本，
  **不需要跨 launch 配對**（`cgc-decode-attribution` 第 33 條）。
- 解析：`python3 scripts/check/prebind_probe_parse.py <log>`（或 `-` 讀 stdin；
  `--self-test` **63/63**；`pred=`／`cov32`／`q32`／`hist=`／`w*` 都是**選配欄位**，
  沒有它們的舊 log 仍可解析）。
  解析器**按 `ntok` 分組**、以 `uni` 加權
  （`q = Σhit/Σuni`，不是逐層平均的平均），並直接印出 gate 判決。
  交付形狀是 `ntok=4`（`--ntok` 可改）；門檻 `--gate` 是**輸入**，
  用實測輸入導出（見 §6.3），並配 `--expect-u`／`--expect-p` 讓「門檻失效」能被抓到。
- ⚠ **多 slot 併發**時 static 累計會混：`s_steps` 的判斷靠 `il` 回繞，併發下不可靠。
  ⇒ **跑的時候用單一請求**。這是本階段唯一的執行限制，沒有繞過的方法
  （要修得把累計移到 cache 成員並按 ctx 分開，超出階段 0 的範圍）。

---

## 9. 未解

1. ~~**`q32`（draft 24 ∪ prev token 8）本階段不量**~~ —— **已解（2026-09-23 同一輪）**：
   `prev_token_expert_ids` 的收集確實被 gate 在 `CGC_PREV_TOKEN_PREFETCH`／
   `CGC_LAYER_AHEAD_PREFETCH` 後面，`prev_token_expert_ids`（`:5448`），
   開它們會改變行為 —— 但那只是「既有 buffer 的收集」被 gate，**ids 張量本身**在
   hook 裡就在手上。所以 probe 自己維護一份雙緩衝 `prebind_prev_ids`／`prebind_curr_ids`
   （`llama-expert-cache.h`），用**與既有邏輯同一條規則**（`il==0` 時 swap、然後以 j=0 那列
   覆寫 curr），只寫 probe 自己的 buffer，不改既有 `prev_token_*` ⇒ 仍然零行為改變。
   `cov32` = 全部 draft 列 ∪ 上一 token 的 ids。
   ★ **q32 不是寬度選擇的候選**：gate 打在 q24，q24 不過時主判準已經 FAIL，所以 q32 永遠
   不可能成為「能過門檻的最小寬度」——把它放進寬度迴圈等於偷偷放寬 gate。解析器因此把
   q32 當**獨立的備援結論**陳述（「q24 不夠時，多買一個 prev 源能不能救」），不併入 PASS/FAIL。
2. ~~**draft 的 `n_tokens` 是多少沒有實測**~~ —— **已量，答案是 0，而且寬度軸已經換掉**。
   實測 `rows` 分佈 = `{0: 1880}`：draft 一次都沒收集到（MTP draft graph 只跑 nextn
   block，collect 只在 `il == n_layer` 觸發，verify 用的是 0..39，永遠對不上）。
   ⇒ **寬度軸改用歷史 token**（最近 1/2/3 個 decode 步的 per-layer ids 聯集 = 8/16/24），
   `rows` 降級為「draft 有沒有跑」的診斷欄位。
   ★ 換源之後「寬度」的證據也換了：q24 的 24 只是**名義**寬度，相鄰 token 的 top-8
   會重疊，實測聯集可能只有 19 ⇒ 它是「至多 24 寬」。所以 probe 多印
   `w8/w16/w24/w32`（候選集合的實際大小）與 `hist=`（非空歷史個數 0..3，
   冷啟動那幾步不進 `q24w` 那一桶）。解析器由此報告**真實**寬度。
3. ~~**相關性（sigma）本階段量不到**~~ —— **已解（見 §6.2）**：由逐層 `res` 的變異數
   （二階矩）反解，實測 sigma_hat = 1.019（n=1880）。`clean_pct` 不再被拿去配適，
   改成驗證點：模型預測 28.9% vs 實測 32.9%（+4.0 pp，在 ±10 pp 內）。
4. **static 累計的多 slot 問題**（§8）未解，靠「單一請求」避開。
