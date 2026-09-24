# 形狀天花板：先掃 roofline 再來造 kernel

**日期** 2026-09-22 · 載體 `Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf`
· 引擎 dylib **`md5 544b6c94bc7a372d`**（15:55 重鏈，乾淨 HEAD 的建置）
· 工具 `scripts/check/shape_roofline.py` · 產物 `Backup/roofline/phase2_*.txt`

---

## 0. 一句話

方向對，儀器要換：`m1_harness.py` 做不了（沒有形狀隔離），`test-backend-ops perf`
可以。用它掃完的結論是三條**否定**，每條都有數字：

1. **這族 kernel 的天花板是步時的 8.5%**（8.355 ms / 97.90 ms）⇒ 寫到完美也只有 2–3%。
2. **MTP verify 批次在 kernel 層完全不攤薄**：每多一個 draft token 就是全額再讀一次
   （n=1 60.22 → n=4 229.97 µs，per-token 成本 +4.6% 以內不變）。
3. **裝置峰值不是 120 GB/s，是實測 100.5 GB/s**；而這族 kernel 只跑 36–45 GB/s。

配套的肯定結論只有一條：**型別替換是唯一免寫 kernel 的槓桿**（gate/up `iq2_s`→`q2_K`
1.35×，down `iq3_s`→`iq4_nl` 1.46×）。

---

## 1. 為什麼 `m1_harness.py` 不能掃

| 問題 | 證據 |
|---|---|
| **沒有形狀隔離** | `run_timed()` = `subprocess.run(llama-bench)` + `re.search(r"decode.*([0-9.]+)\s*t/s")`。量到的是**整個模型**的 t/s；docstring 寫 "measures µs/dispatch"，實作沒有 |
| **要 patch 才能掃** | `knob_space()` 掃 `CGC_MM_NSG` / `CGC_MM_NXPSG`；兩者在 `ggml-metal/` **0 命中** |
| 但清單那半可用 | `CGC_MM_DBG` / `MMDBG` 存在（`ggml-metal-ops.cpp:2533`）⇒ `shape_inventory.py` 可用 |

架構（先清單、再微基準）是對的；壞掉的只有微基準那一格。

---

## 2. 實際用的儀器：`test-backend-ops perf`

* 已編在磁碟上；編出真 pipeline（`kernel_mul_mv_id_iq2_s_f32_nsg=2`）
* op 無 flops 時它**直接印 GB/s**；有 flops 時印 µs/run + GFLOPS
* **perf 模式有自己獨立的 case 清單** `make_test_cases_perf()`（`:9960`），
  與 `make_test_cases_eval()`（`:8212`）不同 —— 加測資要加在**生效的那一份**。
  本次修補前，qwen36 的探針在 eval 清單裡有一份被 `#if 0` 關掉的複製品，容易誤改。

```bash
src/llama.cpp/build/bin/test-backend-ops perf -o MUL_MAT_ID -p n_mats=256 > cap.txt 2>&1
python3 scripts/check/shape_roofline.py --capture cap.txt --gguf <carrier.gguf> \
    --peak-gb-s 100.5 --peak-note "MEASURED: Q6_K lm_head 417 MB contiguous"
```

---

## 3. 載體事實（從 GGUF 表讀的）

```
layers=41  experts=256  top-k=8  d_model=2048  ff=512
ffn_gate_exps  iq2_s  K=2048 N=512  39 layers   (+1 iq3_s, +1 q2_K)
ffn_up_exps    iq2_s  K=2048 N=512  39 layers   (+1 iq3_s, +1 q2_K)
ffn_down_exps  iq3_s  K=512  N=2048 37 layers   (+3 iq4_xs, +1 q3_K)
```

1. **`top-k = 8`，不是 4。** 兩個獨立來源：GGUF `expert_used_count=8`，以及引擎橫幅
   `CGC-PHASE-SPLIT: ... top_k=8`（`Backup/cgc_logs/llama_server_20260922_141654.log:20`）。
2. **down 是 `IQ3_S`，不是 `IQ3_XXS`**：全檔 **IQ3_XXS = 0** ⇒ p2 那 24 行
   （`kernel_mul_mv_id_glu_iq3_xxs_impl`）在這顆載體上**仍然不可達**。
   原本 `test-backend-ops.cpp:9969` 的註解自稱 "qwen36 ffn_down" 卻寫 IQ3_XXS —— 已改正。

---

## 4. Roofline 表（全部實測）

```
projection     type     MB/tok  us/layer    GB/s  %peak  ideal us
ffn_gate_exps  iq2_s     104.8     60.22    44.6    44%     26.74
ffn_up_exps    iq2_s     104.8     60.22    44.6    44%     26.74
ffn_down_exps  iq3_s     133.4     98.86    36.5    36%     35.87
TOTAL                    342.9    8.355 ms/token => 120 t/s ceiling (MoE GEMV only)
```

* `us/layer` = 一次 dispatch、`n_used=8`、`n=1`。`ideal us` = 同 bytes 在實測峰值下的時間。
* **沒有一格是模型化的**（上一版的 down 是用 iq3_xxs 推的；實測 98.86 vs 推的 99.45，
  巧合地只差 0.6%，但現在它是讀數）。
* `342.9 MB/token` 與 `WHY_MODEL_MISSED` 的「專家 hit 361 MB」一致（層數口徑 39/39/37 vs 40）。

---

## 5. 裝置峰值（實測，取代引用的 120 GB/s）

perf 清單裡既有的 qwen36 lm_head 探針就是一個大流量探針（一次讀 417 MB 連續權重）：

```
MUL_MAT q6_K    m=248320 n=1 k=2048   bytes=417.2 MB  4152.71 us -> 100.5 GB/s
MUL_MAT iq3_xxs m=248320 n=1 k=2048   bytes=194.7 MB  3048.99 us ->  63.9 GB/s
```

* **裝置實測峰值 = 100.5 GB/s**（引用的 120 高了 19%）。
* 同一個形狀，**Q6_K 比 IQ3_XXS 快 1.57×**（100.5 vs 63.9 GB/s）—— IQ 系列的
  table lookup 在頻寬上是真的貴，這正是 §7 那條型別槓桿的機制。
* **否定結果**：`DUP` 不是有效探針。新加的 128 MiB `test_dup` 在 MTL0 上量到
  **11.89 GB/s**（`Backup/roofline/phase2_devicepeak_*.txt`）——256 MB 的複製不可能
  只有 12 GB/s，所以那是 kernel 的限制，不是裝置的。**不要用 DUP 當頻寬基準**；
  用大流量 GEMV。

---

## 6. X × Y：kernel 值多少

* **X = 8.355 / 97.90 = 8.5%** 的 GPU busy
* **Y**（同形狀下最快型別）：

```
shape m=512,k=2048: q2_K 44.75 | iq3_xxs 58.58 | iq2_s 60.22 | iq4_nl 60.25 | iq3_s 62.00
    iq2_s -> q2_K  = 1.35x  => 端到端 2.2%
shape m=2048,k=512: iq4_nl 67.58 | iq3_xxs 94.61 | iq3_s 98.86 | q3_K 138.33
    iq3_s -> iq4_nl = 1.46x => 端到端 2.7%
```

⇒ 把這族 kernel 寫到完美，端到端 **2–3%**（12.57 → ~12.9）。

**in situ 對照**：`dense 827 MB + experts 343 MB = 1170 MB/token ÷ 97.90 ms = 12.0 GB/s`
= 實測峰值的 **12%**。同樣的 expert bytes，全熱 8.355 ms、in situ 343 MB/12.0 GB/s
= 28.6 ms ⇒ **居住性懲罰 3.4×**。所以「先造 kernel」的順序是反的。

---

## 7. MTP verify 批次：**不攤薄**（這是補上的最大一格）

補上 `n=2,3,4` 的 256-expert 測資之後，第一次可以在 kernel 層回答
「verify 每多一個 draft token 要多少」：

```
iq2_s m=512,k=2048   (gate/up)
  n=1   60.22 us    60.22 us/token    44.6 GB/s
  n=2  114.61 us    57.30 us/token    46.9 GB/s   +54.39
  n=3  172.72 us    57.57 us/token    46.7 GB/s   +58.11
  n=4  229.97 us    57.49 us/token    46.7 GB/s   +57.25
  -> batched mean 57.48 us/token vs n=1 的 60.22 = +4.6%  => NO amortisation

iq3_s m=2048,k=512   (down)
  n=1   98.86 us    98.86 us/token    36.5 GB/s
  n=4  400.23 us   100.06 us/token    36.0 GB/s   (平均)
  -> batched mean 102.56 us/token vs n=1 的 98.86 = -3.7%  => NO amortisation
```

**每一個 token 都帶著自己的 top-k ids，所以每個 token 都要再讀一次自己的 8 顆 expert。**
`n=4` 的 GB/s 與 `n=1` 相同 ⇒ 批次拿到的不是頻寬攤薄，只是「同一個 kernel 被餵更多行」。

### 這條讀數怎麼接到引擎

每多一個 draft token、每層的成本 = gate + up + down
= 57.25 + 57.25 + 100.73 = **215.2 µs/層 × 40 層 = 8.61 ms**

| 口徑 | 每 draft token | 相對實測峰值 |
|---|---|---|
| 峰值地板（359 MB @ 100.5 GB/s） | 3.57 ms | — |
| **kernel 實測（全熱）** | **8.61 ms** | 41% |
| in situ（359 MB @ 12.0 GB/s） | ~29.9 ms | 12% |
| 引擎側實測邊際成本（先前）：46–59 ms | 46–59 ms | 6–8% |

⇒ **kernel 只解釋引擎邊際成本的 15–19%，其餘 81–85% 是居住性與派送往返。**
這同時否證了「批次會攤薄 verify」這個假設 —— 它不會，一個 token 一份 bytes。

---

## 8. 唯一免寫 kernel 的槓桿：型別替換

| 位置 | 現在 | 候選 | kernel | bytes |
|---|---|---|---|---|
| gate/up | IQ2_S | **Q2_K** | 60.22 → 44.75（1.35×） | 2.56 → 2.625 bpw（+2.4%） |
| down | IQ3_S | **IQ4_NL** | 98.86 → 67.58（1.46×） | 3.44 → 4.5 bpw（+31%） |

合計 **bytes +14%、kernel 時間 −26%**，而 down 那格是**同時**升精度又變快。
§5 的機制解釋：K-quant 直解碼能跑到 100 GB/s，IQ 的 table lookup 只有 64。

代價（必須誠實標）：換型別會改值 ⇒ **M1/M2 oracle 參考必須重做**，品質要用 48 題重測。
工具已在：`scripts/gguf_retensor.py set-type`。

---

## 9. 建置 provenance（本次動了共享 dylib，必須記錄）

| 時點 | dylib md5 | 說明 |
|---|---|---|
| 動工前 | `430f9d315cbcf6c4` | p2 rebased 樹編的（含 `sum_parts`）；**已備份**至 `Backup/roofline/libggml-metal.P2-430f9d31.20260922_1551.bak` |
| 動工後 | `544b6c94bc7a372d` | 乾淨 HEAD（demo tip）的重建，**與 p2 記錄的 baseline 逐位元相同** |

* 動工前確認：**0 個 build 行程、build/ 15 分鐘內 0 次寫入**，`ggml-metal/` 的
  `git diff` 為空（那 14:30 的 mtime 不是內容改動）。
* 只改了 `tests/test-backend-ops.cpp`（+23 −2，全部在 perf 清單與註解）。
* 所有數字來自**後建置的同一顆** dylib；舊版 capture 已不並排引用。

---

## 10. 產物

* `scripts/check/shape_roofline.py` —— 11/11 自測。未知型別、缺測資、缺 top-k
  一律**拒算**（exit 1）；proxy 必須明示且輸出標 `*`；新增 verify-batch 的 n 伸縮段。
* `Backup/roofline/phase2_mulmatid_n256_*.txt`（15 格）、`phase2_bwprobe_*.txt`（裝置峰值）、
  `phase2_devicepeak_*.txt`（DUP 否定結果）、`libggml-metal.P2-*.bak`。
* 本文。沒有起 server；沒有改引擎源碼以外的檔案。
