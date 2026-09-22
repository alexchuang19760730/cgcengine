# 兩個形狀函數落地（不是分析）——2026-09-22

本文是「落地」的交付記錄：`src/llama.cpp/src/llama-shape-knob.{h,cpp}` 兩個函數已寫進引擎、
通過 build、在 `llama-bench` 裡實測印出；`scripts/check/shape_knob_search.py` 是讀它們的搜索器，
也已實跑。**它沒有把我們帶到 25 t/s**，下面 §4 講為什麼——而且它是自己拒絕給結論的。

## 1. 兩個函數是什麼

| 函數 | 角色 | 做什麼 |
|---|---|---|
| `cgc_shape_knobs()` | **INPUT**（受 env 控制） | 所有形狀 env 只 parse 一次；跨軸驗證（`union = top_k × M` 是否塞得進 routable slots）；每軸記狀態 `APPLIED / CLAMPED / CONFLICT / ABSENT / REFUSED` |
| `cgc_shape_report_{init,final}` | **OUTPUT**（給 harness） | 每相位一行 `CGC-SHAPE v=1 ...`，帶**實收幾何**＋ cache 自己的計數器 |

為什麼要把這兩件事分開做：今天的環境裡，一個沒落地的 knob 和一個「試過、沒用」的 knob
長得一模一樣。有了 output 行的 `M_stat`，這兩者第一次可以被分開。

**今日真實輸出**（`Backup/shape_search/prod25-stream_width/*/*.stderr.log`）：

```
CGC-SHAPE v=1 phase=final M=17 M_stat=APPLIED width=17 union=136 fits=1 pool_cap_slots=143
slots_layer=143 n_layer=41 req=13430 hits=9096 misses=4334 hit_pct=67.73 compulsory=4071
capacity=263 evict=4186 zero_mapped=0 verify_refused=0 inv_viol=0 read_mib=4538.6
fill_wait_us=52040 final_counters=1
```

其中三個計數器是這次搜索能被自動裁決的原因：`zero_mapped`＝被靜默丟掉貢獻的被選中專家、
`verify_refused`＝因 cold 被拒的 fast-path 步、`inv_viol`＝batch invariance 破壞。
**任一非零 ⇒ 那一行做的不是同一份工，它的 t/s 不算量測。**（三個計數器實測都是 0。）

## 2. 四條軸現在各自的真實狀態

| 軸 | env | 狀態 |
|---|---|---|
| #1 寬度 M | `CGC_SHAPE_M`（別名，等同既有 `CGC_POOL_MAX_TOKENS`） | **可掃**。已被 `cgc_pool_max_tokens()` 收斂成單一來源（`llama-graph.h`） |
| #6 graph cb 數 | `CGC_N_CB` | **可掃**。已改由 `cgc_shape_n_cb()` 供值（`llama-context.cpp:495`） |
| #5 memory split | `LLAMA_ARG_EXPERT_CACHE`（`-expert-cache` 的 env） | **可掃**。實收量是 `pool_cap_slots`（注意：`llama_model::expert_cache_pool_capacity` 是**每層 slot 數**，不是 bytes——第一版報告列就叫 pool_bytes 導致列出版了 143 這個假 bytes，已改名） |
| #7 GDN | `CGC_SHAPE_GDN_CH` | **ABSENT**。這棵樹只有 `LLM_FUSED_OP_GDN_CH` 這個 enum 成員，**沒有任何人引用它**。設了會印 `ABSENT` 並明說請求被忽略——不會假裝有效果 |

`run_server.sh` 是 explicit allowlist（未列的 `CGC_*` 被靜默丟掉），所以也把 `CGC_SHAPE_M` /
`CGC_SHAPE_TAG` / `CGC_SHAPE_GDN_CH` 三個加進去了；已用 `CGC_DUMP_ENV=1` 驗證真的到得了子行程。

## 3. harness 的裁決規則（`shape_knob_search.py --selftest` 14/14）

- **VOID**：`M_stat` 不是 `APPLIED/DEFAULTED`（knob 沒落地）；品質計數器任一非零；
  `fits=0`（union > routable ⇒ 已知會 silent gather、`buffer is nil` 的那個形狀）；沒取到 `CGC-SHAPE` 行。
- **UNANCHORED**：cell 夾的兩個 reference 相差 > 10%，或 sweep 期間離開 NOMINAL。
- **NO_EVIDENCE / FASTER / SLOWER**：以 ±3% 為界（本 repo 門檻；llama-bench decode 單臂噪音底 ≈ ±27%）。
- cell 之間會 **settle**（等 NOMINAL ＋ `--min-free-pct`），arm 自己的 registry env 與 knobs 衝突時**拒跑**而不是靜默覆蓋。

## 4. 今晚實測的結果：三個結論，沒有一個是「變快」

跑了三輪（reps=1，交付 decode cell，`prod25-stream`）：

1. **knob 全部落地**：M=4 → `width=4 union=32`；M=17 → `width=17 union=136 fits=1`。
   ⇒ **M 的上限 17 是被實測確認的**（`floor(142/8)`；routable 是 142 不是 143，因為 ZERO slot）。M=24 會走 CLAMPED，尚未量。
2. **這台機器在 5 次 launch 內單調崩掉**：reference 序列 11.72 → 10.83 → **5.22** t/s
   （另兩輪 10.50→8.47、10.75→6.04）。這個漂移比要找的效應還大 ⇒ harness 對兩個 cell 都判
   `UNANCHORED`，**沒有給任何速度結論**。這是對的：今晚任何「M=4 比較慢」的說法都測不出來。
   同一現象也說明為什麼以前那些單次 A/B 不可引用。
3. **一條真正新的量測**（跟 #4/#5 直接相關）：**capacity miss 只佔 miss 的 6~7%**
   （M=4/8/17 → `capacity=67 / 227–260 / 263`，`compulsory=3697 / 3491–3588 / 4071`）。
   ⇒ 143 slot/層在這個 workload 上**不是容量飢餓**；之前提的「route A：把 per-slot 砍到
   0.929 MiB 換 215 slot」最多只能處理 7% 的 miss。反過來讀，**該試的是把池縮小**
   （4 GiB ⇒ ~71 slot），把 RAM 還給 dense page-cache —— `budget` 軸已經接好了，就等一個乾淨窗口。

## 5. 距離 25 的現狀

今晚這顆 Binary 的交付 cell 實測在 **10.8–11.7 t/s**（與記錄的 12.57 同階），加寬沒有幫上忙，
品質計數器全 0。**沒有任何東西往 25 移動。** 兩個函數做的是把「四條軸的值多少」變成一個
可以被機讀、可以被自動拒答的界面；它今天立刻產出的東西是一個否定式收穫：**以前那些針對
pool 容量的處方，在這個 workload 上最多只能處理 7% 的 miss**。

還缺的一步是硬的：**#7 沒有實作**。想在 #7 上有 Data，得先有 chunked GatedDeltaNet operator，
knob 才有地方施力 —— 這是唯一無法靠 env 生出來的東西。

## 6. 檔案

| 檔案 | 狀態 |
|---|---|
| `src/llama.cpp/src/llama-shape-knob.h` / `.cpp` | 新增（兩個函數） |
| `src/llama.cpp/src/CMakeLists.txt` | 加入 source |
| `src/llama.cpp/src/llama-graph.h` | `cgc_pool_max_tokens()` 改由表供值（行為不變：default 8、[2,64] clamp） |
| `src/llama.cpp/src/llama-context.cpp` | n_cb 改由表供值；寬度決算後 `note_width`＋`report_init` |
| `src/llama.cpp/src/llama-expert-cache.cpp` | destructor 呼叫 `report_final` |
| `scripts/run_server.sh` | allowlist 加入三個 shape env |
| `scripts/check/shape_knob_search.py` | 新增（搜索器，`--plan/--run/--selftest`） |
| `Backup/shape_search/prod25-stream_width/{search.json,*/*.stderr.log}` | 三輪 raw |
