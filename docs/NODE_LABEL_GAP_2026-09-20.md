# `node` 桶的內容盤點與命名缺口（2026-09-20，零 GPU）

## 0. 為什麼有這份文件

`docs/G4_WORK_ORDER_2026-09-20.md` 的 `label_gap` 記著一個結論：**每張圖最大的那個 kind 不是一種工作，
是一個「沒有名字」的自動兜底名**——`ggml.c:7191-7193` 對任何進圖時 `name` 仍為空的節點執行
`ggml_format_name(node, "node_%d", …)`，而 kind 詞彙表（`ggml-backend.cpp:2228`）有一條 `"node"` 前綴
把它整桶收走（`(other)` 桶因此是 0.00 ms）。

那份記錄說「便宜的動作是在 builder 補名」。**這份文件做的是那個動作的取證部分**：哪些節點是無名的、
它們是什麼、以及缺口確切在哪一行。**它不加速任何東西**，它只讓下一張（時間加權的）圖可讀。

## 1. 盤點（離線，從既有的 `CGC-GRPH` 轉儲）

來源：`Backup/cgc_logs/llama_server_20260918_050331.log`，第一張圖 **2604 個節點**，
其中 **350 個（13%）是 `node_*`**。

| op | 數量 | 常見形狀 `ne` | 前一個**具名**節點 | 讀出來的身分 |
|---|---:|---|---|---|
| `MUL_MAT` | 130 | (32,2)×60、(8192,2)×30、(64,16)×10 | `Kcur-<l>` | GDN 路徑的投影 |
| `GET_ROWS` | 90 | (24576,0)×30、(524288,1)×30、(524288,0)×30 | `beta_sigmoid-<l>` | 專家 gather ＋ GDN 狀態 gather（每 GDN 層 3 個） |
| `MUL` | 60 | (128,32) | `norm-<l>` | norm 之後的 elementwise（每層 2 個） |
| `ADD` | 30 | (32,2) | `beta-<l>` / `conv_input-<l>` | GDN 的 beta/conv 相加 |
| `UNARY` | 30 | (128,32) | `gdn_out_raw-<l>` | GDN 輸出上的激活 |
| **`FLASH_ATTN_EXT`** | **10** | (256,16) | `Kcur-<l>` | **10 個 full-attn 層的注意力本體** |

兩件值得注意的：**注意力本體在 10 個 full-attn 層上是無名的**（那是真的 kernel，不是 elementwise）；
而 `node` 桶的 op 組成裡 GET_ROWS 佔最大一項 —— 所以「`node` 61% 是未 fuse 的 elementwise」這個
曾被使用的說法，**方向是錯的**。

## 2. 缺口確切在哪些行

### 2.1 已釘死的一個（可以一行修）

`src/llama.cpp/src/llama-graph.cpp:3160-3165`：

```c
cur = ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale, hparams.f_max_alibi_bias, …);
res->add_fused_node({LLM_FUSED_OP_FLASH_ATTN, cur, il});   // ← 有 il，但沒有名字
ggml_flash_attn_ext_add_sinks(cur, sinks);
ggml_flash_attn_ext_set_prec (cur, GGML_PREC_F32);
```

同一個檔案別處的慣例是 `cb(Kcur, "Kcur", il);`（:1686）⇒ 缺的就是一行
`cb(cur, "FLASH_ATTN", il);`。**這解釋了那 10 個無名的 `FLASH_ATTN_EXT`。**

### 2.2 其餘（**沒有**寫成補丁，理由見 §3）

- GDN 家族（MUL_MAT 130、MUL 60、ADD 30、UNARY 30，以及部分 GET_ROWS）：`delta-net-base.cpp` 與
  `qwen35moe.cpp` 的替用路徑，`cb()` 密度已經很高但仍然有洞 ⇒ **逐個 call site 對照**，不是一個模式。
- 其餘 GET_ROWS 90：部分在專家 gather（`llama-graph.cpp:2353` 附近的 `slots`/`selected_experts` 鏈）。

**⇒ 這是一個跨 ≥3 個檔、每處要個別判斷的編輯，不是一個 patch。**

## 3. 為什麼**只**釘死 §2.1 那一行，不動 `src/`

1. **需要一次 build 才生效**，而 `cmake --build` 會蓋掉別條線正在用的 `libggml-*`／`libllama`
   （2026-09-17 11:29 的代價是別人一整輪 A/B 橫跨兩個 build）。**⇒ 排隊。**
2. **提交 `src/` 會觸發 D5**（數值閘門）⇒ 需要一次 GPU oracle run ⇒ 需要一個窗口。**⇒ 排隊。**
3. 未經 build 的 C++ 改動**不可驗證**。把一行寫進 `src/` 而不建置，只會在樹上留下一個
   「已修改但未驗證」的引擎檔，而那是別條線 `git add -A` 會掃到的東西。

⇒ 補丁以**腳本**形式放在 `Backup/patch_node_name_flash_attn_20260920.py`（預設 dry-run，
錨點計數不符就整批不寫）。**它沒有被套用，也沒有被 build。**

## 4. 這件事的價值與代價（說清楚，免得被當成進度）

| | |
|---|---|
| 買到什麼 | 下一張**時間加權**的 per-kind 圖（`CGC-GPULAYK`）不再有一個 13% 的無名桶；「哪個 kind 貴」變成可讀 |
| **不買到什麼** | **任何速度。** 這是標籤，不是最佳化 |
| 前置 | 一次 build（排隊）＋ 一次 run 看名字有沒有出現 |
| 風險 | 低 —— `cb()` 只呼叫 `ggml_format_name`，**不碰數值路徑** |
