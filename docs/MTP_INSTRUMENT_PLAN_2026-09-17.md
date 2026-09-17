# MTP 的生產級儀器：路 i 的實作方案與一個需要拍板的取捨（2026-09-17）

> **本文件的狀態：計劃已完成，實作已跑起來（但還沒完成）。**
> §1–§5 是動工前的計畫；**最後一節「實作結果」是 2026-09-17 21:1x–22:0x 的實測**，
> 它列出四個只有跑起來才會知道的坑，以及**還缺的一步（部分接受的 checkpoint 移植）**。
> 讀這份文件請直接跳到最後一節 —— 前面的計畫在「插入點」與「取捨」上仍然有效，
> 但「未動工」那句話已經不作數了。

---

## 1. 為什麼需要這份計劃

**使用者 21:03 的判準**：「請用生產級 prefill/decode 腳本跑出來的數字才算真實數字」。

生產級的權威定義在 `docs/PROD_MATRIX_STANDARD_20260917_1630.html`：單一入口
`scripts/check/prod_matrix.py`，而 **`llama-bench` 是 prefill 與 decode 共用的唯一 instrument of record**
（`decode_bench` 當日退休）。

**但 `llama-bench` 對投機是瞎的 —— 本輪自己驗過，不是轉述**：

- `llama-bench --help` 共 78 行，含 `spec/draft/mtp/sampl` 的只有 **2 行假命中**
  （`-hff, --hf-file` 的 "specified" 與 "specifying"）；
- 整個 `src/llama.cpp/tools/llama-bench/` 目錄對 `speculat`／`spec_type`／`draft`／`mtp` **零命中**；
- 又因為它**不經過 server**（自己解析參數，**不吃** profile 的 `mtp=1`）。

⇒ `prod_matrix.py --cells decode` 量到的是 **MTP off** 的 decode（那正好是 M3 離開條件要的），
而 **M4 的產出（MTP 收益）在「只認生產級」之下沒有 instrument of record**。
標準文件自己列的兩條補救路**未定**：

| 路 | 內容 | 對使用者的判準 |
|---|---|---|
| **ii** | 保留 `llama-speculative-simple` 當 MTP 臂、**永不並排** | ❌ **不滿足** —— 它不是生產級腳本 |
| **i** | 把 `--spec-*` 加進 `llama-bench` | ✅ 唯一根本解 |

---

## 2. 接線口是現成的

| 需要的東西 | 現況 |
|---|---|
| `common_params` 物件 | llama-bench **已有**（`llama-bench.cpp:1093` `common_params p;`） |
| 從 params 建 spec | `common_speculative_init_from_params(common_params &, llama_model * model_tgt, llama_context * ctx_tgt)` |
| 一圈的動詞 | `common_speculative_begin/process/draft/get_draft_params/accept` |
| 收尾 | `common_speculative_print_stats`、`_free` |
| 可照抄的範例 | `examples/speculative-simple/speculative-simple.cpp` |

**真正的成本在別處** —— 在「llama-bench 量的是什麼」這件事上（見 §3）。

---

## 3. ★ 需要拍板的取捨：兩個變體量到的**不是同一件事**

**決定性的事實**（`llama-bench.cpp:2167-2186`）：`test_gen` 的 token 是
`std::rand() % n_vocab` —— **它根本不看 logits**，而且它硬編 `llama_batch_get_one(&token, 1)`
（**一次一個 token**）。

而 `common_speculative` 的 draft 需要**取樣**（`speculative-simple.cpp` 用
`common_sampler_sample(smpl, ctx_dft, ...)`）⇒ 它需要 logits。兩者不相容，於是：

### 變體 i-a — 接完整的 spec（真正的 MTP）

在 llama-bench 裡引入 sampler 與 `common_speculative`，把 `test_gen` 換成
draft → verify → accept 的迴圈。

- **量到**：真正的 MTP 收益（含 sampling 與接受率）。
- **代價**：改動大（要接 sampler、要處理 draft context 與 KV、要處理 `--spec-type draft-mtp`
  對 nextn layer 的依賴），而且**改變了這個 cell 的性質** —— 它不再是「純 GEMV 的 tg」。
- ⚠️ 這正是 §4「不要污染 `decode` cell」要處理的事。

### 變體 i-b — 只改批次形狀（verify 批量紅利的上界）

不動 sampler、不動 token 產生（仍用 `std::rand()`），**只把每輪 decode 的寬度從 1 改成 k+1**，
模擬 verify 的批次寬度。

- **量到**：**「verify 拿不到批量紅利」這一項的上界** —— 而 `M3_VERDICT` 說那一項正是
  MTP-on 多出來成本的 **84%**（draft head 只佔 16%；T=1 是 92 ms/token、T=2.6 是 104 ms/token）。
- **代價**：小（改一個迴圈的批次寬度）。
- ⚠️ **但它不是 M4** —— 沒有 draft、沒有 accept，所以它給的是「如果 verify 批次化，上限是多少」，
  而不是「我的拒絕取樣做得怎麼樣」。

### 兩者的關係（我的建議）

**先 i-b 後 i-a**，理由是可判別性：

- i-b 便宜，且它**直接檢驗「工作項 2 值不值得做」**（若批次化的上界 < 10%，那 M4 的工作項 2
  就不用做了，M4 也就到頂了）。
- i-a 貴，且它的結果只有在 i-b 顯示「有空間」時才有意義。
- 兩者都不污染 `decode` cell（見 §4）。

---

## 4. 不要污染現有的 `decode` cell

`decode` cell 有明確用途：**與上游 `tg128 @ d512` 可比**（`-p 0 -n 128 -d 512 -b 512`）。
加了 spec 之後它就不再是純 tg ⇒ 正確做法是**新增 cell**，並在兩處登記：

| 要改的地方 | 內容 |
|---|---|
| `scripts/check/prod_matrix.py` 的 cell 表 | 新增 `decode-mtp`（形狀待定 —— 取決於 §3 選哪個變體） |
| `docs/PROD_MATRIX_STANDARD_*.html` | 在新版標準裡登記它與它的用途、以及它**不能**與誰並排 |

⇒ 這是「**一次性的儀器投入 ＋ 一次標準修訂**」，不是一行 patch。

---

## 5. 動工需要的兩個條件（都要）

1. **`prod_matrix.py` 的 preflight 不 abort** —— 即沒有別條線的量測行程。
   ⚠️ 判準是**讓它自己回答**，不要自己寫檢查：本輪我的檢查（比對 `comm == llama-server/llama-bench`）
   在 21:07 說「環境空了」，而閘門抓到一個 `bash -c 'sleep 240; …'` 的 driver。
2. **`free` 回到 GB 級** —— 21:08 實測 `Pages free` = 4905 × 16384 = **80 MB**、
   `swap used` = **8.36 GB**。
   ⚠️ 反例就在同一小時：另一條線在 `free_gib 0.08` 下量到 `decode=5.96 t/s`，
   對照生產級 `10.78` —— **在那個環境下跑出來的數字看著像真實數字，但不是**。

**建置也要這兩個條件**（`cmake --build` 是對機器上所有正在跑的實驗的一次寫入），
所以建置與跑 cell 應該排在**同一個窗口**裡。

---

## 6. 現在可以／不可以做的（21:08 實況）

| 動作 | 現在可以做嗎 | 理由 |
|---|---|---|
| 寫程式（本文件就是它的前置） | ✅ | 不碰模型、不佔記憶體 |
| 建置 | ❌ | 條件 2 不滿足（free 80 MB），且會蓋掉別人的產物 |
| 跑任何 cell | ❌ | 條件 1 可能滿足、**條件 2 不滿足** ⇒ 拿到的不是真實數字 |

---

## 7. 未做／未主張

- 本文件**沒有**動 llama-bench、**沒有**建置、**沒有**跑任何 cell。
- §3 的兩個變體**都未實作**，所以「M4 的真實數字」仍然不存在 ——
  任何在此之前的 M4 t/s（含 `mtp_accept_ab.py` 的 12.57 vs 9.59）都**不是生產級數字**。
- `test_gen` 不看 logits 這件事是本輪讀出來的（`:2167-2186`），
  它的**推論**（「所以 i-a 必須引入 sampler」）依據是 `speculative-simple.cpp` 的用法；
  **未驗證** llama-bench 引入 sampler 之後是否還有其他阻礙。

---

## 實作結果（2026-09-17 21:1x–22:0x，i-a 第一次真的跑起來）

> 這一節是**實測**，不是計畫。**尚未 commit**；`src/llama.cpp/tools/llama-bench/llama-bench.cpp`
> +222/−1、`scripts/run_server.sh` +16/−0（allowlist 兩個非 CGC 變數）。

### 四個「照抄來源沒告訴你」的東西 —— 全部是跑出來的，不是想出來的

下列四項每一項都讓那條路直接死掉，而且**死法不同**（編譯錯／assert／靜默 rc≠0）：

| # | 症狀 | 根因 | 修法 |
|---|---|---|---|
| 1 | `llama_context::synchronize` 裡 `CGC-METAL-FAIL: command buffer 8 failed (status 5, Insufficient Memory)`，**連續 5 次**（8 GiB 與 2 GiB pool、`-b 256` 與 `-b 512`、swap 8.7–12.4 GB 全都一樣） | **我自己手組 argv**，只帶 `-m`；少了 profile 的 `--load-mode none`（以及 `-ngl 99 -t 8 -ctk/-ctv q8_0 -expert-cache`） | 用 `llama_bench_matrix.forward_argv(resolve(...)['server_argv'])`。改完**同一個形狀 23 秒 rc=0** |
| 2 | `GGML_ASSERT(hparams.n_layer_nextn…)` 之後在 `qwen35moe.cpp:566` `layer.nextn.eh_proj` assert | `llama_model_params.load_mtp` 預設 **false**；qwen35moe loader 用 `TENSOR_SKIP` 建 MTP 區塊（`qwen35moe.cpp:45`）。server 是 `--spec-type draft-mtp` 經 `common.cpp:1635` 把它打開的，**llama-bench 沒有那條路** | `to_llama_mparams()` 裡 `mparams.load_mtp = getenv("LLAMA_BENCH_SPEC") != nullptr` |
| 3 | `speculative.cpp:1318` `MTP requires ctx_tgt and ctx_dft to be set` | `common_speculative_init()` 只在 **`params.draft.ctx_dft != nullptr`** 時啟用 MTP（`speculative.cpp:2633`）；impl ctor 又要 `ctx_tgt`。`common_speculative_init_from_params()` **只回傳 holder，不會幫你填這兩個欄位** | 照 `speculative-simple.cpp:142/180/181`：`release_context()` → 自持 `ctx_dft` → 填 `draft.ctx_tgt` ＋ `draft.ctx_dft` |
| 4 | 第三輪 `verify decode failed: ret=-1`（**靜默**，`llama_decode` 不印任何 ERROR） | **部分接受時我無條件 `llama_memory_seq_rm` 修剪**，而 `common_context_can_seq_rm(ctx)` 回報 **FULL(2)** —— 卻仍讓下一次 decode 失敗。參考實作在 `can_seq_rm == FULL` 時走的是 **`common_prompt_checkpoint` 還原**那條路（`speculative-simple.cpp:570-593`），不是修剪 | **未修**。`LLAMA_BENCH_SPEC_NOTRIM=1` 拿掉修剪後 **rc=0、跑完 16 tokens** ⇒ 缺陷被侷限在**這一個分支** |

### 現在的狀態（誠實版）

- **off 臂（不設 `LLAMA_BENCH_SPEC`）：rc=0，是可信的控制臂。** 形狀 `-p 0 -n 16 -d 0 -b 512`，23 s，輸出表列有
  `| qwen35moe 35B.A3B IQ3_XXS | 12.71 GiB | MTL | ngl 99 | threads 8 | n_batch 512 | q8_0 | none | tg16 |`。
- **on 臂：`NOTRIM=1` 才跑完**（`MTP fast path: draft calls=N union=8N`、`verify` 計數器有值），
  但**修剪一開就在第 2–3 輪死** ⇒ 這條路**還不能當儀器用**。
- 所以 **M4 還沒有「用生產級儀器量到的」數字**；差的是 #4 的 checkpoint 移植。
- 診斷用的開關（都在 llama-bench 內，預設不影響任何東西）：`LLAMA_BENCH_SPEC_DBG=1`（每輪 n_done/n_past/draft
  ＋ `seq_rm_type`）、`LLAMA_BENCH_SPEC_NOTRIM=1`（跳過修剪）。

### 動工前必須知道的兩個 repo 陷阱

1. **`bin/llama-bench` 的 md5 是假指紋**：33 KB thin stub，程式碼在 `libllama-bench-impl.dylib`
   ⇒ 判新鮮度要看後者（`LLAMA_BENCH_SPEC` 字串也只在後者裡）。
2. **`llama-bench` 的 `n_ctx = n_prompt + n_gen + n_depth`**：`-n 16 -d 0` ⇒ ctx 只有 16（本次實測印出
   `n_ctx=256`、`n_batch=16`）。生產 cell `-n 128 -d 512` ⇒ 640，有餘裕。**驗證批次需要
   `1 + draft.size()` 個空位**，所以短形狀要自己設界（已加）。

---

## 更新 22:0x–22:35：checkpoint 移植完成，短形狀通了；cell 形狀卡在「第二個 context」

第 #4 項**已修**（照 `speculative-simple.cpp:570-593` 與 server 版 `server-context.cpp:4186-4221`）：

- `common_prompt_checkpoint` 的兩半分開取（`update_pos` 在 draft 前、`update_tgt` 在 draft 後，因為 MTP 的
  draft ctx 與 target **共用 KV**）；部分接受 ⇒ `draft = std::move(ids)` → `load_tgt` →
  `seq_rm(pos_max+1, -1)` → **`common_sampler_copy(smpl_save, smpl)`** → `n_past = ckpt.n_tokens` → `continue`。
- **sampler 還原是必要的**，不是裝飾：少了它，replay 那輪抽到不同的 token，同一個部分接受會反覆發生
  （實測 `SPECDBG` 行數 ~2000 → 7）。

| 形狀 | 結果 |
|---|---|
| `-n 16 -d 0`（spec on） | **rc=0**、`tg16 1.63 ± 0.00` |
| `-n 128 -d 512`（spec off） | **rc=0**、`tg128 @ d512 5.09`（熱狀態 1／swap 11 GB ⇒ 不可引用） |
| `-n 128 -d 512`（spec on） | **rc=1**：第一個 verify batch `ret=-1 n_tokens=4(pos 0..3) n_ctx=768` |
| 同上 ＋ `--no-warmup` | **rc=1**，同一個錯 ⇒ 與熱身殘留狀態無關 |

同一次 teardown 印出 **`CGC-M2-UNREPOINT: teardown restored 120 expert tensor(s) ... before freeing 6
slab(s) (a second context built from this model would otherwise read a freed buffer)`** ⇒ 池熱過之後，
從同一個 model 建第二個 context（MTP draft）會把 expert tensor 重指到 slab，target 的第一個 decode 就 `-1`。
**這一關屬於 CGC 池／slab 的 owner，不是 llama-bench。** 短形狀（無 depth prefill、池不熱）不觸發。
