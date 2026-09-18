# MTP(k=3) 到 25 tok/s 的算術：需要多少 token/步，瓶頸在哪 —— 2026-09-18 18:4x

**一句話**：25 t/s ＝ 40 ms/token。而「每步要幾個 token」＝ **步時 ÷ 40 ms**，所以答案取決於步時：

| 步時 | 25 t/s 需要的 mean_len | 判斷 |
|---|---|---|
| **85.65 ms**（MTP-off 的一步，＝假設 verify 不變貴） | **2.14** | 實測 mean_len 就是 **2.07–2.14** ⇒ **剛好擦線** |
| **168.4 ms**（實測 MTP-on 的 verify 步，最乾淨那幾筆） | **4.21** | **k=3 的上限是 4.0** ⇒ **差 5%，物理上不可達** |
| 281.7 ms（今天受干擾的樣本） | 7.04 | 不可能 |

⇒ **瓶頸不是 accept 率，是 verify 一步的成本（實測 1.97× 單步）**；而那個成本的本質是
**一步要讀 19.05 個 distinct expert 去換 2.14 個 token**（每 token **8.90** 個 expert，
比單 token 一步的 8.0 **還差 11%**）⇒ **MTP 在「池／IO 綁定」這條軸上結構性為負**。

---

## 1. 「每步 token 數」的定義（先讀原始碼，不猜）

`src/llama.cpp/tools/server/server-context.cpp:673-674`：

```cpp
const float  draft_ratio  = (float) n_draft_accepted / n_draft_total;
const double mean_acc_len = n_draft_verif_steps > 0
                          ? 1.0 + (double) n_draft_accepted / (double) n_draft_verif_steps : 1.0;
```

⇒ **`mean len`（日誌裡那格）就是「每步 token 數，含基底 token」**。同一行還會印
`draft acceptance = X (a accepted / g generated), mean len = Y`（`:684`），
而 **驗證步數另有累計值**：`spec_decode_num_drafts_total`（`/metrics`，
"Total speculative decoding verification steps"，`:4778`）。

⇒ 所以「每步幾個 token」「每步幾 ms」**都可以直接量**，不必建模：
`steps = predicted_n / mean_len`、`ms/step = predicted_ms / steps`。

## 2. 實測（服務路徑、`prod25`、8 GiB pool、MTP on、k=3）

資料來源是**今天各次 MTP-on 服務的 log**（`Backup/cgc_logs/llama_server_20260918_*.log`），
把同一 task 的 `eval time` 與 `draft acceptance` 配對，抽到 **160 組**。**乾淨子集（t/s ≥ 10）**：

| log | task | mean_len | ms/step | t/s |
|---|---|---|---|---|
| 18_013 | 34 | 2.14 | **168.4** | **12.71** |
| 18_013 | 17 | 2.14 | **172.4** | **12.41** |
| 18_120 | 219 | 2.07 | 195.9 | 10.57 |
| 18_131 | 0 | 2.07 | 203.0 | 10.19 |
| 18_041 | 26 | 2.20 | 211.8 | 10.39 |
| 18_163 | 183 | 2.40 | 233.6 | 10.27 |

- **mean_len 的分布**：1.00（全被拒）／2.07／2.14／2.20／2.30／2.44／2.50／2.75／4.00
  ⇒ 常態落在 **2.07–2.50**。
- **ms/step 的分布**：**157.7 → 7769.5 ms**（同一 mean_len 下差 20 倍）⇒ 這是**環境**，
  不是 treatment。所以只引用「乾淨子集」。

**★ 兩個獨立印證（這輪沒有我自己跑的樣本，伺服器兩次都被別條線的 preflight 殺掉，見 §5）**：

1. 乾淨子集最佳 **168.4–172.4 ms/step ⇒ 12.41–12.71 t/s**，與記錄的 HTTP **12.62 t/s** 相符。
2. MTP-off 的一步是 **85.65 ms**（`docs/MTP_POOL_CONFOUND_2026-09-17.md:28,56`），
   而本輪稍早我自己量的 HTTP 口徑 **11.75 t/s ＝ 85.1 ms/step**（`docs/HTTP_VS_BENCH_CALIBER_2026-09-18.md`）
   ⇒ 兩個來源互證。

⇒ **verify 步 / 單步 = 168.4 / 85.65 = 1.97×** ⇒ **MTP 的淨增益 = 2.14 / 1.97 = 1.09×**
⇒ 預測 `11.75 × 1.09 = 12.8 t/s`，**實測 12.4–12.7** ✓（模型與觀測自洽）。

> ⚠ **18:45 補註（影響本節的推導，不影響結論的方向）**：上面那個 `11.75`（HTTP 中位）與
> `168.4 ms/step` **都來自「服務路徑」**，而 `caliber_env.py --equiv` 已判該路徑與 bench cell
> **NOT CONFIG-EQUIVALENT**（`-b/-ub` 未對齊，`prod_matrix.py:319`）⇒
> **1.09× 這個淨增益是「同為服務路徑」的前提下量到的，故仍然自洽**（分子分母同側），
> 但**不要**把它與任何 **llama-bench 側**的數字相乘。`85.65 ms` 那條來自 2026-09-17 的獨立量測，
> 不受此影響。

## 3. 需求表：要 25 t/s，每步至少幾個 token

`每步 token 數 × 步速 = t/s`，而 `t/s = mean_len / (step_ms/1000)` ⇒ **`mean_len ≥ step_ms / 40`**。

| 情境 | 步時 | 需要的 mean_len | k=3（上限 4.0）夠嗎 |
|---|---|---|---|
| verify 一步「免費」（＝單步水準） | 85.65 ms | **2.14** | ✔ 而且實測已達 2.07–2.14 |
| 現行 verify 步（乾淨窗） | 168.4 ms | **4.21** | ✘ **超出上限 5%** |
| 現行 verify 步（受干擾窗） | 281.7 ms | 7.04 | ✘ 遠超 |

**反過來算上限**：即使 **accept 100%**（mean_len = 4.0，因為 draft_n_max=3 ⇒ 一步最多 1+3 個 token），
以現行 verify 步時 168.4 ms ⇒ **4.0 / 0.1684 = 23.75 t/s**。
⇒ **mtp 3 ＋ 現行 verify 步時，理論天花板是 23.75 t/s < 25** —— 差 **5%**，但**不是靠 accept 能補的**。

## 4. 「攤三次」的預算

把步時攤到每步的 token 上（步預算 = 每步 token 數 × 40 ms）：

| 每步 token | 步預算上限 | 現行 168.4 ms | 差 |
|---|---|---|---|
| **3**（1 + 2 個被接受） | **120 ms** | 168.4 ms | **超 40%** |
| **4**（1 + 3，k=3 最大值） | **160 ms** | 168.4 ms | **超 5%** |
| 2.14（實測） | 85.6 ms | 168.4 ms | **超 97%** |

⇒ 要「攤三次攤到 40 ms」，**verify 一步必須壓到 120 ms 以下**（若每步真能出 3 個 token）。
現在的 168.4 ms 意味著**每個 token 分攤到 56.1 ms**（168.4/3）—— 而 40 ms 是目標。

## 5. 瓶頸：是「一步讀幾個 expert」，不是 accept

**機制（都是既有儀器印出來的）**：

| 量 | 值 | 出處 |
|---|---|---|
| verify 一步的 union | **19.05** distinct expert（`9028 / 474 / 32 = 0.595`） | `llama-expert-cache.cpp` 的 `MTP fast path: … verify: calls=474 union=9028` |
| 單 token 一步 | **8** expert（top-k） | 同上，`draft: calls=36 union=288` ⇒ `288/36 = 8.00` ✓ |
| ⇒ 每 token 的 expert 讀取量 | **19.05 / 2.14 = 8.90** vs 單步 **8.0** ⇒ **+11%（更差）** | 由上面兩格推得 |
| 單次 verify 會不會打穿池 | **不會**（`cold(ZERO) = 0`） | 同上 |
| 但長期工作集 | `layers_distinct_over_slots` 3(off) → 18(server)；`capacity` miss 218 → 2229（×10.2）；hit 96.0 → 87.1%；`worst` 層 167/143 → 240/143 | `M3_M4_STATUS` §2、`MEMORY_PERF` |

⇒ **兩層瓶頸，先後不同**：
1. **第一步（結構性）**：verify 必須把 4 個 draft 全算過（不管最後接受幾個）⇒ 一步要 **19.05** 個
   distinct expert。實測 mean_len 只有 2.14 ⇒ **那 19 個 expert 攤到 2.14 個 token**，
   每 token 成本反而**高於**不做投機（8.90 vs 8.0）。
   這個結構解释了為什麼 **llama-bench 路徑量到 MTP ≈ 1.0**（它 verify 走 `ensure_batch`、
   `verify: calls=0`）而 **server 路徑量到 ×0.695**：同一個現象、兩條池分支、兩個量級。
2. **第二步（長期）**：即使單次不打穿池，**長期工作集**（18 個 distinct 層超額、capacity miss ×10.2）
   讓 evictions／IO 上升 ⇒ 這才是 server 路徑比 llama-bench 更低的那一段。

**⇒ 要 25 的唯一算術出路**：把 **19.05 壓下來**（讓 4 個 verify token 共享更多 expert ⇒ draft 品質
／**池的淘汰填充策略（M6，唯一不需預算的路）**），或**讓同樣 19 個 expert 攤到更多 token**（提高 accept）。
⚠ **在 greedy 下 accept 不可由 accept rule 移動**——它是 `(base, head)` 配對的性質
（`MEMORY_PERF:351`，差 60% 只有 1.75pp）⇒ 只剩前者。

## 6. 操作危害（這輪量不到的真正原因，具名）

`scripts/run_server.sh` 的 preflight 用**固定二進位名清單 ＋ `pgrep -f`**（`CGC_PREFLIGHT_PATTERNS`，
`:659-673`），**不看 port、不看 session**，而且**清理排在 memory guard 之前**（`:813` 之後才判 guard）
⇒ **任何一條線啟動 `run_server.sh`，都會 SIGTERM 掉其他所有 session 正在跑的 llama 進程**；
他們自己就算被 guard 擋下來，被殺的那一邊也已經死了。

本輪兩次撞到（`Terminated: 15`）：`llama_server_20260918_181929.log`（18:19，第 C3 臂）、
`llama_server_20260918_183958.log`（18:40）。第二次是**在前景、我的命令還活著**的時候死的
⇒ **推翻我上一輪對 C3 的「nohup／進程組連坐」解釋**：同樣的症狀，真因不同。
**建議**：preflight 至少按 port 或按 `CGC_SERVER_PORT` 縮範圍，或在殺之前檢查對方是否為本 session 的。

## 7. 沒做的（具名）

- **我自己沒有本輪的一手樣本**：兩次起的 server 都被別條線的 preflight 殺掉（§6）。§2 的數字來自
  今天其他 session 的服務日誌（同樣是生產路徑、同一支腳本解析的 env），但我**沒有親手重跑**。
- **k 的掃描**：沒有。以「每 token 的 union 專家數」看，**k 越小可能越好**（k 的 union 成長
  比 mean_len 快）——這是可驗證的猜測，未量。
- **`CGC_MTP_REJECTION`**：沒有跑（temp 0 下它 by construction 是 null）。
- **D5**：本輪 0 個 `src/`（只有 `docs/`）⇒ oracle 那一半不適用。
