# `shape_roofline.py` 能不能拿來「依硬體／模型建立 kernel」

**日期** 2026-09-22 · 審查對象 `scripts/check/shape_roofline.py`（20 KB，別條線 15:59 落地）
+ `docs/SHAPE_ROOFLINE_2026-09-22.md` · 審查者本線（demo/sweet-spot-windows-fix）

---

## 0. 一句話

**它是「自適應測量層」，不是 kernel 生成器 —— 而真正缺的那一段（runtime 掃參）
其實是現成的，只是被一個拼錯的旋鈕名擋住了。**

自適應的**三半都是真的**（讀真模型、讀本機真峰值、讀真形狀耗時），11/11 自測、
未知型別一律拒算。它能驅動的決策只有「選型別」一件；而「依硬體自動選 kernel 參數」
這件事，樹上早就有 `CGC_MMV_NSG` 在做 —— 它是 Metal function constant，
**不用改 .metal、不用重建**，而且**不破壞 bit-identity**。

---

## 1. 它做了什麼（已驗證）

```bash
/Users/alexchuang/.workbuddy/binaries/python/versions/3.13.12/bin/python3 \
    scripts/check/shape_roofline.py --selftest
# selftest: 11/11 passed
```

| 自適應的那一半 | 怎麼來的 | 真／假 |
|---|---|---|
| **模型** | `gguf_tensor_table()` 直讀 GGUF header → `carrier_projections()` 取每個 projection 的 dominant 型別 + K/N/層數，並**列出少數派 variants**（37 IQ3_S + 3 IQ4_XS + 1 Q3_K 會被印出來，不藏） | ✅ 真自適應，非硬編碼 |
| **硬體** | `--peak-gb-s` 吃**本機實測** 100.5 GB/s；預設 120 的 `--peak-note` 自帶「cited M4 spec, NOT measured on this box」 | ✅ 真自適應，且強制標注 |
| **形狀** | `--capture` 吃 `test-backend-ops perf` 輸出，`us/run` 抓不到就丟掉那一格而不是猜 | ✅ 真自適應 |

**誠實設計（三條硬拒，都在自測裡）**：需要的形狀缺 → exit 1；型別沒有 bpw → exit 1；
proxy 必須顯式 `--proxy TYPE=SRC` 且輸出標 `*`。

---

## 2. 它不做什麼

`grep -nE "codegen|emit.*kernel|generate.*kernel|autotune|write.*\.metal"` → **0 命中**。

⇒ 它不產生任何 kernel 原始碼、不寫 `.metal`、不派發。它是**報告器**：
告訴你「這族 kernel 離裝置峰值多遠、寫到完美值多少」。

---

## 3. ★ 缺環其實是現成的：`CGC_MMV_NSG`（文檔結論錯，只差一個字母）

`SHAPE_ROOFLINE_2026-09-22.md` §1 寫：

> `knob_space()` 掃 `CGC_MM_NSG` / `CGC_MM_NXPSG`；兩者在 `ggml-metal/` **0 命中**
> ⇒ 要 patch 才能掃

**字面正確，結論錯誤。** `ggml-metal/` 裡 `CGC_` 開頭的環境變數有 **57 個**，
其中 MoE GEMV 專用的一整族都在：

```
CGC_MMV_NSG    CGC_MMV_NR0    CGC_MMV_FUSE    CGC_MM_BITIDENT    CGC_DC_NSG
```

`m1_harness.py` 自己的 docstring 第 24 行就寫著
`-> TODO: readenv("CGC_MM_NSG") / ("CGC_MM_NXPSG") to sweep.` —— **TODO 裡的兩個名字
是不存在的近似拼寫**，真名是 `CGC_MMV_NSG`（多一個 V）。

### 3.1 `CGC_MMV_NSG` 的性質（`ggml-metal-device.cpp:818-835`）

```c
// CGC P1-3c: runtime nsg (simdgroups-per-threadgroup) selection for the per-expert GEMV
// kernels (env CGC_MMV_NSG=N; default = the per-type N_SG_* constant). nsg is a Metal
// function constant (FC_MUL_MV+0) so the pipeline compiles/caches per value at runtime —
// no .metal change needed. Each row's dot product is computed inside ONE simdgroup, so
// nsg does not change the per-row reduction order (bit-identity holds).
// Clamped to the M4 threadgroup limit (1024 threads / 32 per simdgroup = 32).
```

三件事，每一件都直接命中本線這兩天在吵的題目：

1. **Metal function constant ⇒ 依值在 runtime 編譯並快取，不用改 .metal、不用重建**
   ⇒ 「依硬體自適應」這件事**樹上已經在做**，不是缺功能。
2. **clamp 1..32 正是 M4 的 threadgroup 上限（1024 threads ÷ 32）**
   ⇒ 換硬體就是換這個上限，是**可移植的那個參數**。
3. **「每行點積在單一 simdgroup 內 ⇒ 不改每行歸約順序 ⇒ bit-identity 保持」**
   ⇒ 這就是 §FP_ORDER 那場辯論要的形狀：拿平行度（nsg 決定同時跑幾行）
   **而不動歸約順序**。**不用關 fast-math、不用重新 baseline。**

### 3.2 但覆蓋表只有兩個型別

```c
switch (t) {
    case GGML_TYPE_IQ2_S:   return N_SG_IQ2_S;
    case GGML_TYPE_IQ3_XXS: return N_SG_IQ3_XXS;
    default:                return -1;      // ← 其他型別不可覆寫
}
// CGC_MMV_NR0 同理，且只接受字串 "8"
```

對照載體（`dense_bytes_census.py` 與 roofline §3 兩邊獨立吻合）：

| projection | 型別 | NSG/NR0 可掃？ |
|---|---|---|
| `ffn_gate_exps` | **IQ2_S** | ✅ **可掃** |
| `ffn_up_exps` | **IQ2_S** | ✅ **可掃** |
| `ffn_down_exps` | **IQ3_S** | ❌ **不可掃**（表裡只有 IQ3_XXS） ⇒ 要 patch |

⇒ **不用動 C++ 就能掃 gate/up（佔 MoE GEMV 時間的 55%）；
down 那格（45%，且是離峰值最遠的 36%）要 patch 才能掃。**

---

## 4. 收益邊界（別高估）

roofline §4/§6 的實測表（µs = 一次 dispatch、n_used=8、n=1）：

```
ffn_gate_exps  iq2_s  104.8 MB/tok   60.22 us   44.6 GB/s  44% peak
ffn_up_exps    iq2_s  104.8 MB/tok   60.22 us   44.6 GB/s  44% peak
ffn_down_exps  iq3_s  133.4 MB/tok   98.86 us   36.5 GB/s  36% peak
TOTAL          342.9 MB/tok  8.355 ms/token (39 layers)
X = 8.355 / 97.90 ms = 8.5% of GPU busy
```

⇒ **這族 kernel 寫到完美，端到端也只有 2–3%**，而只掃 gate/up（可掃的那半）
量級只剩 **~1–2%**。**都低於 3% 門檻。**

**而且順序是反的**：in situ 實測 12.0 GB/s（= 峰值 12%），全熱 8.355 ms 的
expert bytes 在 in situ 要 28.6 ms ⇒ **居住性懲罰 3.4×**。先造 kernel 是倒著做。

---

## 5. ★ 本線要更正的一條（我自己的舊分析）

我在 `docs/FP_ORDER_TARGETS_2026-09-22.md` 把
`kernel_mul_mv_id_glu_iq3_xxs_impl`（p2 那 24 行改的 kernel）列為候選目標 ③。

**載體全檔 `IQ3_XXS = 0`**（`dense_bytes_census.py` 獨立確認：gate/up = IQ2_S、
down = IQ3_S）⇒ **那個 kernel 在這顆載體上根本不會被執行**。
⇒ 目標 ③ 是空的，我那條「依賴鏈被 threadgroup 間接查表掩蓋」的推論
**對這顆載體無對象可套**。該節結論（方法論在 decode 上無目標）不受影響，
但**理由要換成「目標不存在」，不是「目標鏈不在關鍵路徑」**。

---

## 6. 交叉驗證（兩條獨立路徑對上）

| 量 | roofline（別條線） | `dense_bytes_census.py`（本線） | 差 |
|---|---|---|---|
| experts / step（top_k=8） | 342.9 MiB | **342.5 MiB** | 0.1% |
| trunk dense / step | 827.0 MiB | **827.0 MiB** | 0% |
| expert_count | 256 | 256（自 dims 讀） | — |
| gate/up 型別 | IQ2_S | IQ2_S | — |
| down 型別 | IQ3_S | IQ3_S | — |

⇒ 兩份數字互相獨立而吻合，**可以引用**。

本線那份另外多給出一格 roofline 沒算的：**`lm_head` 397.9 MiB = 全部 dense bytes 的
32.5%**（`DENSE subtotal 1527.8 MiB`）。「非 MoE 佔 70%」那格沒把它算進去。

---

## 7. 現場狀態（審查時）

| 項 | 值 |
|---|---|
| dylib md5 | `544b6c94bc7a372d` = **乾淨 HEAD 重建**（roofline §9 記錄的同一顆） |
| 源碼 `sum_parts` 計數 | **0** ⇒ **產物與源碼一致**（本線先前標的「不一致」已解除） |
| 分支 | `demo/sweet-spot-windows-fix` |
| 門禁 | **ok**（usable 8.59 GiB，swap 4031/5120 = 78.7%） |
| `test-backend-ops` | 已編，mtime 15:55 |

⚠ 別條線 15:55 動過共享 dylib（§9 有記錄、有備份 `.bak`）；本線任何重建前
仍要走「8080 + 量測行程」雙閘。

---

## 8. 結論：要建「依硬體／模型」的 autotuner，配方是三塊拼起來

| 塊 | 現況 | 要做 |
|---|---|---|
| ① 清單（這顆模型有哪些形狀／型別） | ✅ `shape_inventory.py` + `CGC_MM_DBG`（`ggml-metal-ops.cpp:2533`）可用 | 無 |
| ② 微基準（形狀隔離的 µs） | ❌ `m1_harness.py` 壞（用 llama-bench 全模型 t/s，無形狀隔離）<br>✅ `test-backend-ops perf` 可替代 | **把 harness 的量測半換成 `test-backend-ops perf`** |
| ③ 掃參 | ✅ `CGC_MMV_NSG`（1..32）/ `CGC_MMV_NR0=8` / `CGC_MMV_FUSE=1` 都是真 env<br>❌ 但 `m1_harness` 的 TODO 名字寫錯 | **knob 名改成真的那三個**；down（IQ3_S）要 patch 加進覆蓋表 |

**⇒ 「依硬體／模型建立 kernel」這條路，缺的不是功能，是把三塊接起來的那一層。**

但接起來之後值多少，§4 已經給了上界：**2–3%（全族）/ 1–2%（只掃可掃的 gate/up），
低於 3% 門檻**。所以它值得做，是因為它讓「寫 kernel 值不值」這件事**從此有數字**，
不是因為它本身能給出 >3%。

---

## 9. 與本線既有結案的關係

* 不推翻 `K5 判不做`（18 µs/dispatch 固定開銷）—— 那是 5-node 小 op 群，
  本節是大 op，兩者不衝突。
* 不推翻 `FP_ORDER_TARGETS` 的總結論（方法論在 decode 上無目標），
  但**理由更正**（§5）。
* 與 `BIGGER_FISH` 的第 3 條（dense 827 MB 重複讀）**指向同一件事**：
  兩份獨立量測都說 expert 只佔 ~343 MB / step，而 dense 是它的 2.4 倍。
