# S1 的 abort 不是 S1 的：一條診斷探針讀到了別人的張量

日期：2026-09-20 07:0x–07:2x（+08）
儀器：`m123_oracle_gate.py`（D5 數值閘門，自行起 `run_server.sh`）＋ 三次 A/B 對照
對象：`CGC_S1_DBG` 這條探針，以及 07:00 那份「S1 現在會 abort」的讀數
相關：`docs/S2_IMPLEMENTATION_PLAN_2026-09-20.md` §7（本檔推翻它的結論）、
`.workbuddy/memory/2026-09-20.md` §EN-306／§EN-307

---

## 0. 一句話

**S1 在交付配置上不會 abort；abort 的是 `CGC_S1_DBG` 這條診斷指令自己** ——
它在**不是它自己建的圖**上遍歷一張從不清空的捕獲表，讀到的張量被按 **4 bytes/元素** 讀，
其中一層（`il=14`）的實際位元組數只有讀取長度的一半 ⇒ `ggml-backend.cpp:349` 的越界斷言。

判準是同一個 build、同一棵樹上的一次對照：**只差 `s1_dbg` 這一格**。

| 臂 | 旋鈕（banner 逐字） | 結果 |
|---|---|---|
| `s1-nodbg` | `slot_table_gpu=1 … s1_dbg=off` | **不 abort**；gate `PASS`（`comparable=true`、`config_diffs=[]`）、M1 9/9、M2 9/9、M3 9/9、`zero_mapped_selected=0` |
| `s1-baseline` | `slot_table_gpu=1 … s1_dbg=1` | abort（`llama_server_20260920_070035.log:368`） |

```
launch_s1-nodbg.log:29     [perf]  diag: submit_ahead=off slot_table_gpu=1 canon_order=off pool_split_dbg=off s1_dbg=off
launch_s1-baseline.log:29  [perf]  diag: submit_ahead=off slot_table_gpu=1 canon_order=off pool_split_dbg=off s1_dbg=1
```

⇒ **「S1 的前置不是『還沒過閘』，是『目前會崩』」這句推論作廢。**
`llama-graph.cpp:2144-2145` 要求的前提（S1 先過 bit-identical 閘門）**已經滿足**。

---

## 1. 根因：三件事疊起來

### (1) 捕獲表只在**建 decode 圖**時被填，而它**從不清空**

`cache_slots_out_tensors` / `cache_slot_table_tensors` / `cache_remap_tensors` 在
`llama-context.cpp:7142-7173` 的 `graph_get_cb` 裡寫入，而那個 capture 位在
`llama-graph.cpp:2191` 的條件內：

```cpp
if (expert_cache_active && !(dw_env && dw_env[0]) && n_tokens >= 1 &&
        cgc_is_decode_graph(n_tokens, expert_cache_decode_max_tokens) && il >= 0 && ...)
```

⇒ 只有 decode 圖會填它。**而三張表都沒有任何 `.clear()`**（全檔 grep 的結果：只有賦值、
`count()`、`find()`、以及 `operator[]`，沒有清空）。所以在 prefill 圖上，它們裝的是
「**最後一次建 decode 圖時**」的指標。

### (2) ggml 每個 build 都 reset 並**重用 arena** ⇒ 舊指標會落在**當前圖的別的張量**上

這一條是關鍵，而它是**實測**出來的，不是推論。同一次啟動裡探針跑了兩次
（`if (cgc_s1_post_n < 6)`），兩次互相矛盾：

| 呼叫 | 行號 | 內容 |
|---|---|---|
| 第 1 次 | `254-292` | `il=1..39`，**每一條都是 `ntok=2 n_expert=256`** —— 真正的 decode 形狀，表是新鮮的 |
| 第 2 次 | `355-367` | **同 39 個 key，每一條的形狀都是別的張量的** |

而第一次修法加的「**這個位址是不是當前圖的節點**」檢查，**一條都沒攔下**（`SKIPPED` 計數 = 0）。
原因：arena 被重用之後，那個舊位址**真的**是當前圖某個節點的位址 ——
**「是不是本圖的節點」不是「是不是我當初那顆張量」**。

### (3) 讀取長度假設 4 bytes/元素，而那些張量不保證

探針每一處讀都是 `ggml_backend_tensor_get(t, buf, 0, n * sizeof(int32_t))`，
而 `ggml_nbytes()` 是**由 `nb[]` 算出來的**。落在舊位址上的張量若是 F16／量化型別，
`4n` 就會大於它自己的 `ggml_nbytes` ⇒ `ggml-backend.cpp:349`：

```cpp
GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds") failed
```

---

## 2. 定量證據（修後那次運行，`llama_server_20260920_071337.log`）

加護之後同一條探針的 82 條輸出，逐條分類：

| 類別 | 條數 | 說明 |
|---|---|---|
| `reported` | **65** | 兩個判準都過，正常印出 |
| `over`（`ggml_nbytes > 4n`） | 13 | 別人的張量，讀起來**安全但是錯的報告** |
| `zero`（`ntot = 0`） | 1 | 退化形狀 |
| **`under`（`ggml_nbytes < 4n`）** | **1** | **這就是 abort 的那一處** |
| `membership`（不是本圖節點） | 2 | 舊指標已經完全不在圖裡 |

那唯一一條下溢，逐字：

```
CGC-S1: POST il=14 SKIPPED (ne=[256,256] but ggml_nbytes=139264 is not 65536 int32s: ...)
```

`ne=[256,256]` ⇒ 要讀 `65536 × 4 = 262144` bytes，而 `ggml_nbytes = 139264`
⇒ **越界 122880 bytes**。

而 07:00 那份 log 崩潰前的**最後一行是 `il=13`** —— 它死在 `il=14`，與這裡指出的那一層**同一個**。
（兩次運行是不同的行程，形狀不保證逐位元相同；但「13 印完、14 死」與「14 是唯一一條下溢」互相印證。）

---

## 3. 修法（`src/llama.cpp/src/llama-context.cpp`，+~60 行，只在 `CGC_S1_DBG` 下生效）

兩個判準，**都是必要的**，而且不是同一件事：

```cpp
// ① 這個位址屬於剛剛跑的那張圖（必要，但不充分）
static bool cgc_node_in_graph(ggml_cgraph * gf, const ggml_tensor * t);

// ② 這顆張量真的裝著 n 個連續的 4-byte 元素 —— 讀取本身的前置條件
static bool cgc_is_i32_n(const ggml_tensor * t, int64_t n) {
    return t != nullptr && t->data != nullptr && n > 0 &&
           ggml_nbytes(t) == (size_t) n * sizeof(int32_t);
}
```

- ① 用**公開**訪問器 `ggml_graph_n_nodes()` / `ggml_graph_node()` —— `ggml_cgraph` 這個 TU 只有
  前向宣告（完整定義在 `ggml/src/ggml-impl.h`，`src/` 底下沒有任何檔案 include 它）
  ⇒ 直接寫 `gf->n_nodes` 會在建置時得到兩條 `member access into incomplete type`。
- ② 用 `==` 而不是 `>=`：它同時要求**連續**（非連續的 4-byte 張量 `nbytes > 4n`），
  而探針正是把張量當扁平緩衝讀的。
- `tb` / `rm` / `ids_src` 的讀取套用同一組判準；任何一項不過就把該指標降成 `nullptr`，
  而下游每一處都已經處理 null（表頭印 `-1` / `0x0`，差分自動跳過）。

### 留下的殘餘限制（明寫，不藏）

`cgc_is_i32_n` 讓讀取**安全**，沒有讓它**可歸屬**：一個舊指標若剛好落在
「長度對得上的 I32 張量」上，兩個判準都會過，那一行印的就是**那顆張量**而不是 S1 的 gather。
要真正關掉它，需要**建置世代戳記** —— 讓 capture 端（`llama-graph.cpp`）記下「是哪一次 build 寫的」。
那是後續工作，不在這次改動裡；在它做完之前，每一條 POST 行都必須連同 `ids_src_valid` 一起讀
（這條紀律本來就寫在探針自己的表頭註解裡：*IT ONLY MEANS SOMETHING WHEN ids_src_valid=1*）。

---

## 4. 三次運行的判決（`Backup/m123_oracle_gate/summary_*.json`）

| tag | binary | `comparable` | `config_diffs` | M1/M2/M3 | 判決 |
|---|---|---|---|---|---|
| `s1-nodbg` | 補丁**前** | `true` | `[]` | 9/9 9/9 9/9 | **PASS** |
| `s1-dbg-fixed2` | 補丁**後** | `false` | `["ENV.CGC_S1_DBG: ref='<absent>'  now='1'"]` | 9/9 9/9 9/9 | INVALID COMPARISON（**差距就是那顆診斷旋鈕自己**） |
| `s1-postfix` | 補丁**後** | `true` | `[]` | 9/9 9/9 9/9 | **PASS**（補丁後的可比基線） |

`s1-postfix` 的歸屬：`cap_s1-postfix.json` 的 `created` = `2026-09-20T07:14:45`，晚於
`libllama.0.0.279.dylib` 的 mtime `07:13:33` ⇒ 兩者不重疊，判決可歸給那一顆 binary
（`md5 f9bb8693e593e5ae`）。`ref_md5 = 72d82a33ad79e0e69bc935acd24228f2` 與釘住的 `REF_PINS` 相符
（`ref_pinned=true`）。

⚠️ **`s1-dbg-fixed2` 的 INVALID 不是故障**：參考檔是在沒有 `CGC_S1_DBG` 的配置下 dump 的，
而 `CGC_S1_DBG` **不在** `DIAGNOSTIC_KEYS` 裡（`CGC_SLOT_TABLE_GPU` 在）
⇒ 開它必然產生一筆 `config_stamp` 差異。這也是為什麼 `s1-nodbg` 是 `comparable=true` 而它不是。
**兩個 9/9 都只是 `printed for information, NOT a verdict`。**

---

## 5. 對 S2／G1 的後果

- ❌ 作廢：「S2 卡在一個要先修的 defect 上」、「S1 的前置是會崩」。
- ✅ 成立：S1 的 bit-identical 前提**已滿足**（`s1-nodbg`：`comparable=true`、9/9/9、`zero_mapped_selected=0`）。
- ⇒ S2（`docs/S2_IMPLEMENTATION_PLAN_2026-09-20.md` §4 的裝置化）**沒有數值前置阻塞**；
  剩下的阻塞是**流程性的**：§5 的 owner 問題，以及那個檔上本線自己尚未提交的 G4 儀器。

---

## 6. 誠實邊界

- 本檔**沒有**主張「S1 在交付配置上功能正確」——它主張的是「**閘門判它 bit-identical**」，
  以及在 `s1-nodbg` 那一個配置上 `zero_mapped_selected=0`。兩者都不是「S2 可以照做」的證明。
- **`id_oob` 仍然出現**（`s1-nodbg` 的 log 有 8 行），而它的結果是 9/9/9。
  這與既有裁決一致（`CGC-MMID-ASSERT` 在 encode 期讀的是配置器的殘留，不是消費者讀到的值），
  但本檔沒有為它新增證據。
- 「`il=14` 就是舊 log 死去的那一層」是**跨行程的形狀對應**，不是同一份 dump 內的同一顆張量；
  它的成立依賴「arena 重用後佈局穩定」，而本檔只量到**兩次運行都有同一條 il=14 下溢**的其中一次。
- 修法**只動診斷路徑**：`CGC_S1_DBG` 未設時一行都不執行，不動 dispatch、不動圖、不動任何數值。
