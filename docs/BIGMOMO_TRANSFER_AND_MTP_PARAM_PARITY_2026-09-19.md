# BigMoMo 能不能救我們的 MTP，＋bench/http 的 MTP 參數落差
`docs/BIGMOMO_TRANSFER_AND_MTP_PARAM_PARITY_2026-09-19.md` · 2026-09-19 · 只有既有數據與程式碼，沒重跑量測

## 0. 三句話

1. **BigMoMo（arXiv:2609.14643，北大 Luo/Chen 組，2026-09-13）是我們這個問題的正確文獻**，
   但它對我們有用的數字是 **1.76×（平均）／1.82×（最大）over best SD-aware baseline**，不是首頁那個 4.83×
   —— 4.83× 是打「沒有 speculative decoding、按需專家載入」的自迴歸基線。
2. **它最賺的那一刀（消掉「未被計算掩蓋的專家搬移停頓」），正好是我們自己量出來只有 ~15% 空間的那一項**
   ⇒ 紙上的主力機制搬到這台機器上，**變不出 1.7×，也救不了 m**。
3. 它給我們真正新的東西有兩個：**(i)「接受率加權的專家剪枝」打到的是 per-token 計算而不是搬移**
   （唯一對上我們主項的一招）；**(ii) 把每 token 成本拆成 draft / expert compute / exposed stall 的記帳法**
   —— 我們自己的 prefill vs decode 兩軸其實已經有這個量了，見 §3。

---

## 1. 論文的關鍵讀數（不是二手轉述）

| 讀數 | 值 | 出處 |
|---|---|---|
| 4.83× 的對象 | **OnDemand**（自迴歸＋按需載入，**沒有** SD） | 5.1/5.2 |
| 我們該看的 | **1.76× 平均、1.82× 最大**，over SpecMoEOff / SP-MoE / MoE-SpAc 中 TPOT 最低者 | Table 9 + Conclusion |
| 那個 best baseline 有什麼 | 已經有**視窗級專家需求預取、批量 I/O、視窗內權重複用、搬移-計算重疊** | 5.1 (3)(4)(5) |
| 時間都去哪（Qwen3-30B, 8 Elite Gen5） | exposed expert-movement stall：SD-OnDemand **237.86** → best baseline **142.41** → BigMoMo **13.62** ms/token | Fig.14 |
| NPU 利用率 | 預設 18.96–45.03% → 83.78–98.30% | Table 7 |
| 硬體 | 3 台 OnePlus，可用 DRAM 6.00–18.23 GB，UFS ~4 GB/s，Hexagon NPU/VTCM 8 MiB | Table 1 |

**它的整篇論述是：「一個專家被搬進來只服務很少幾個 token」，所以每 token 的搬移成本無法攤。**

## 2. 搬到這裡：三招各自的天花板

| BigMoMo 招數 | 我們樹上有什麼 | 自己的數據給的天花板 |
|---|---|---|
| ① 視窗級 union 去重／批量搬移 | **已做**：MTP fast path union remap（`union/call` 12.2–22.9 vs 單 token 8.00，`zero_mapped_selected=0`） | ≤ 已經吃到 |
| ② On-flash 依共載模式重排 | 沒有（GGUF 佈局固定） | 它在 UFS 上也只拿到 −15.06% 讀次數／1.23× 讀帶寬；我們的 byte 彈性 0.10–0.17 ⇒ 幾個百分點 |
| ③ 就緒組批 ＋ 雙緩衝 ＋ 計算排序去掩蓋搬移 | 近似物＝我們自己的 prefetch，但 **`CGC_NO_PREFETCH=1` 是預設**（`run_server.sh:2029`） | ⇒ 這是真的沒 A/B 過，但可掩蓋的量受 §下一個 bound 限制 |
| ④（額外）依 acceptance rate / routing impact 剪 draft 分支與專家激活 | 沒有 | **唯一打到主項的一招**，見下 |

**為什麼 ①②③ 的天花板這麼低 —— 我們自己的數字：

- 跨池、同 k 比較（benchmark 第二軸）：bytes 乘 4.68× 時時間只乘 1.25×（彈性 **0.145**）；
  k=1/3 甚至負彈性（`docs/POOL_BUDGET_COST_DECOMP_2026-09-18.md` §3）。
- 而且 **bytes 隨 k 是飽和的**（7.69→34→55→56→65 MiB），**時間隨 k_eff 是線性的**（59 ms/token，R²=0.842）
  ⇒ 就算 bytes 彈性是 1，也解釋不了線性項。
- 8 GiB、k=7：超額 269 ms 裡，byte 係數只能解釋 ≈40 ms ＝ **15%**。

**結論：這台機器的 exposed stall 不是主詞**（我們整個 baseline 一步才 106–133 ms，
而它在同一個量上是 142–238 ms/token）。把 BigMoMo 整套搬來，最多是把那 15% 拿乾淨。

### 2.1 那 ④ 呢：它跟主項對得上嗎

我們的主項是「每個 draft token 46–59 ms 的固定代價」。POOL_BUDGET_COST_DECOMP §6.2 的那句話要補一個但書：
**budget 第二軸動的是「搬移成本」，不是「每 token 的 MoE 計算」** —— 它證明的是「搬移不是主詞」，
**沒有**證明「每多做一個 expert 的 tensor 計算不是主詞」。
BigMoMo 的 expert pruning 動的正是後者：`I_e = Σ V_d · min|s_e − s_j|`（router score 相近 ⇒ 可替代），
剪掉後把 routing weight 重分配；他們量到 −29% 專家請求 ⇒ −23% TPOT，**精確度掉了 ≤0.26%**（GSM8K −0.13 / HumanEval −0.05 / MMLU-Pro −0.18）。

⇒ 這是三招裡唯一對著我們主項開的，**但它付的是精確度，要在品質閘門下做**。
它的 node pruning 那一半我們不用看：那是樹狀 draft 的浪費（82.1% 的 verify 計算花在被丟掉的分支上），
我們的 MTP 是鏈狀，而我們自己的 oracle dynamic-k 早就把它封頂在 **1.035×／1.068×**。

## 3. 用我們自己的兩軸，把「46–59 ms」再切一刀（零成本）

他們的記帳法是 `TPOT = draft gen + expert compute + exposed stall`。我們其實有同一個量的兩端：

```
同一支 run（-p 512 -n 128，prefill250，b=ub=5632）
   prefill  ~85 t/s   ⇒ 每 token ≈ 11.8 ms（權重已被 512 個 token 攤薄）
   decode   k=0        ⇒ 106 ms／步
   verify   邊際       ⇒ +46–59 ms／draft token     ← 同一批權重、同一層、同一 GPU
```

把 prefill 拆成「共用讀權重 S＋每 token c」：`512/85 s = 6024 ms`，若 S ≈ 90 ms（≈ decode 一步的水位），

```
c_prefill ≈ (6024 − 90) / 512 ≈ 11.6 ms / token        vs      c_verify ≈ 46–59 ms / token
```

**同一個 model、同一顆 GPU，批次裡多一個 token 的代價差了 ~4–5×。**
這不是「prefill 天生比較便宜」（那是 batch 512 vs 8 的分母差，邊際量不受影響），
而是 **verify 那個 T=k+1 的批次沒有吃到 prefill 吃到的攤薄**。

⚠ 但書：prefill 數字是在 HEAVY 熱態下量的（`docs/SPEC_COST_CURVE_2026-09-18.md` §2），
而且 prefill 可能走不同的 kernel/graph 路徑。所以這個 4–5× 是「值得去追的量級」，不是定論。

## 4. 所以要 MTP 轉正，第一個做的是什麼

`S = E / (1 + m·k_eff)`，今天的 `E`、`k_eff`（實測，`SPEC_COST_CURVE_2026-09-18.md` §3）：

| k | k_eff | E | **轉正所需 m ≤ (E−1)/k_eff** |
|---|---|---|---|
| 1 | 1.000 | 1.306 | 0.306 |
| 2 | 1.716 | 1.362 | **0.211** |
| 3 | 2.460 | 1.662 | 0.269 |
| 5 | 3.986 | 1.882 | 0.221 |
| 7 | 4.676 | 1.662 | **0.142** |

實測 `m = 0.474`（NOMINAL 那一點 0.689）⇒ **至少要把每個 draft token 的成本砍到今天的一半以下**（k=7 要砍到 30%）。
把 bytes/residency 修乾淨只到 `m ≈ 0.40` ⇒ k_eff=4.7 時仍然是 `S = 1.88/2.88 = 0.65×`。

**第一個動作不是引進 BigMoMo，也不是練 draft head，而是把 §3 那個 4–5× 的差距問出名字來：**

1. **n-gram 消融**（已經會用：`--spec-type ngram-map-k`）：同一個 T=k+1 的 verify 批次，去掉 draft 前向。
   - m 崩到 ≈0.1 ⇒ 46–59 ms 是 draft 前向 ⇒ 換便宜 draft；
   - m 不動 ⇒ 住在 verify 的每 token 路徑上 ⇒ 主戰場是「批次沒攤薄」。
2. **同一支 verify step 的 node 級歸因**（k=0 vs k=7）：時間長在 MoE 的 mul_mat，還是長在
   gather / cpy / ENSURE_SLOT 這類「每 token 準備」上 —— BigMoMo 整個 hardware side story
   其實就是這種 per-stage preparation 的另一個版本，只是我們這台機器上那些 stage 不從 SSD 付帳。
3. 順手（最便宜，已探在樹上）：`CGC_P_ROUTE=1` 一次 forward 都不用多，直接量
   `P(top8_{t+j} ⊆ top8_t)`，把 RSL 那條線判生判死。

## 5. 附：bench 與 http 的 MTP 參數，到底一不一樣

來源：`llama_bench_matrix.resolve()`（唯一的真相來源，`CGC_DUMP_ENV=1`，無副作用）與
`scripts/check/caliber_env.py --equiv --show-mtp`。

### 5.1 一樣的（by construction，兩條路都走同一個 resolve()）

`-m`（同一個 gguf）、`-ngl 99`、`--load-mode none`、`-t 8`、`-expert-cache 8589934592`、
`--cache-type-k/v q8_0`，以及 **8 個 engine env 全部同源**：

```
CGC_DRAFT_DECODE=1  CGC_MM_BITIDENT=1  CGC_MTP_NO_WARMUP=1  CGC_NO_PREFETCH=1
CGC_NO_SEQ_RM_PROBE=1  CGC_VERIFY_DECODE=1  CGC_WARM_NPAST=0
LLAMA_EXPERT_CACHE_LAYER_CAPS=40-40:256
```

⇒ **兩條路今天都是在 prefetch 關閉的狀態下量的**，這件事過去被當成「只有 server 是這樣」，其實同源。

### 5.2 不一樣的

| knob | run_server.sh（prefill250） | llama-bench（我們的工具） | 影響 |
|---|---|---|---|
| `-b`/`-ub` | 5632 / 5632 | **5632 / 5632** | ⚠ 見 5.3，**相等** |
| `-c` | 8192（prod25: 4096） | 沒有這個概念 | KV 配置不同 |
| `-np 1` / `--no-kv-unified` / `-sps 0` | 有 | 未轉發 | server-only |
| `--spec-type draft-mtp --spec-draft-n-max` | **固定 3** | 我們掃 0..7 | 只有 k=3 那一格可直接比 |
| sampling | `--temp 0.4 --top-k 0 --top-p 0.8 --repeat-penalty 1.0 --dry-*`、`assistant_prefill` | greedy、無懲罰 | ⚠ **直接改變 E**（接受率），因此改變 m |
| prompt 來源 | HTTP＋jinja chat template | bench 自產 | routing 軌跡不同 ⇒ miss/thrash 不同 |

⇒ 「`m` 是 0.21（服務）還是 0.474（bench）」目前**至少**被三件事混著：profile（prod25 vs prefill250）、
sampling（E 的分母）、prompt 軌跡。batch **不是**這一個的解釋。

### 5.3 更正一條舊結論

`MEMORY.md` 寫的「我們的工具 `-b 512` vs server `-b 5632`（11 倍）」，**對象是 `prod_matrix`／`profile_duo`
的 cell**（`prod_matrix.py` ~line 319 `b = ub = spec["batch"]` 讓 cell 蓋掉 profile），
**不是 `spec_cost_curve.py`**：它走 `lbm.default_batch()`（優先用 profile 自身的 BATCH/UBATCH），
09-18 那 18 支 run 的 JSON 裡記錄的就是 `batch=5632, ubatch=5632`。
⇒ 那份 cost curve 的 batch 與 server 相同；把它拿去和 -b 512 的兩軸 cell 數字混用會出錯。

## 6. 附：hist 這個 option 有──但現在沒有接入口

- **C++ 有實作**：`llama-context.cpp:2002-2004` 支援 `CGC_PREFETCH_SRC = step（預設）| prev | hist`，
  另有 `CGC_PREFETCH_WINDOW`（預設 4，clamp 1..16，`:2012`）。
  hist 的定義：每層最近 N 步 union 的滾動視窗，把其中「已被 LRU 踢掉的熱專家」重新預取回來。
- **run_server.sh 的 SERVER_ENV 已經把它刪了**（2026-09-13，只剩 `:1322-1326` 的註解，
  註解還寫著「Opt back in with `CGC_PREFETCH_SRC=hist` for A/B」）。
  ⚠ SERVER_ENV 是 **allowlist，未列出的 `CGC_*` 靜默丟棄** ⇒ **今天不管從哪一條路設它都不會生效**，
  而且症狀是「看起來有設、其實没設」。`CGC_PREFETCH_WINDOW` 有被放行（`:1942`），但沒有 hist 就惰性。
- **唯一現在就能用的 prefetch 開關**：`CGC_SERVER_NO_PREFETCH=0`（`run_server.sh:2029-2032`）
  ⇒ 讓 `CGC_NO_PREFETCH=1` 那個變數**整個不出現**；兩條路都吃得到（bench 工具用 `--extra CGC_SERVER_NO_PREFETCH=0`）。
  這就是 BigMoMo 招數③在我們樹上的可用替身。
- ⚠ **已知危險**：`scripts/check_build_tracked.sh:21` 記著
  「`CGC_PREFETCH_SRC=hist` 非 MTP prefill 100% 掛死於 `ensure_batch(il=1)`」；
  而 C++ `:2000` 說明 MTP verify 期間為什麼要關 prefetch（背景執行緒填/踢 slot 時 GPU 正在 verify，
  會蓋掉還在讀的 slot）⇒ 真要測，要在 MTP 關閉與開啟下分別掛 watchdog。
- 還有兩個 launch 路徑仍在使用它：`scripts/run_n30cache.sh:254`、`agent_harness/tb_loop/scripts/start_local_model.sh:86`
  （可當成「曾經跑得起來」的參考實作）。

## 7. 補驗：把 BigMoMo 的主軸（請求數，不只是位元組）用同一批 run 测一次

BigMoMo 的論述有兩個版本：一是「位元組太多」，二是**「請求太碎」**（fragmented reads、
每次預備只服務很少的 token）。09-18 那 36 支 run 的 stderr 同時有 `read jobs=` 與 `bytes=`，
所以不必跑新實驗就能把第二個版本也测掉（`Backup/phase_decomp/pool_budget_runs_20260918_2251.json`）：

| 模型（response = ms/step，n = 36 支 run） | r² | 係數 |
|---|---|---|
| 只有 `k_eff` | **0.842** | +59.14 ms / draft token |
| 只有 `jobs/step` | 0.590 | +0.295 ms / job |
| 只有 `MiB/step` | 0.651 | +2.043 ms / MiB |
| `k_eff` + `jobs/step` | **0.877** | +47.61 ms/token ＋ **0.0999 ms/job** |
| `k_eff` + `jobs` + `MiB` | 0.878 | +46.32 ＋ 0.0685 ＋ 0.2583 |

同一個 k、只換池子（跨池比較，是這批資料裡唯一把「搬移量」與「draft 深度」解耦的設計）：

| k | jobs 倍率 4G/8G | 時間倍率 4G/8G | 對 jobs 的彈性 |
|---|---|---|---|
| 0 | 1.68× | 1.25× | **0.431** |
| 1 | 1.62× | 0.76× | **−0.574** |
| 3 | 1.46× | 0.87× | **−0.379** |
| 5 | 1.99× | 1.09× | **0.126** |
| 7 | 1.96× | 1.19× | **0.255** |

**讀法：請求數軸跟位元組軸一樣塌。** 拿 §EN-185 那邊「輪時 ≈ 每輪請求數 × 1.59 ms/請求」來對一下：

- 同一個 counter、k=7、8 GiB 是 **633 jobs/輪**，`× 1.59 ms` 會得到 **1006 ms/輪**，實測卻是 **375 ms/輪**
  ⇒ 那個斜率在這個 counter 上不成立（它大概是另一個請求定義，或是由兩點過原點擬出來的 —— 過原點的
  兩點擬合會把截距整套塞進斜率裡）。
- 這批 run 自身的 `ms/job` 只有 **0.27–0.59 ms**，而且**跨池不守恆**。

⇒ **「位元組」與「請求數」兩個版本在這裡都解釋不了主項**，§4 的結論因此更穩，
BigMoMo 招數 ①②③ 的天花板也再往下壓一層：即使把 jobs 全砍光，上界也只有
`0.0999 ms/job × 633 ≈ 63 ms`，而跨池實驗顯示實際拿不到。

需要被分開答的兩件事：§EN-185 的「capacity miss 主導、217 distinct vs 143 slots」回答的是
**為什麼會 miss**；本節回答的是 **miss 完之後對時間有多少影響**。推薦還是分開唸，不要合成一句。

## 8. 可證偽

- 若 `CGC_SERVER_NO_PREFETCH=0` 在配對設計下讓 k=7 的時間掉 >20% ⇒ §2 的「exposed stall 不是主詞」要重寫；
- 若 n-gram 讓 `m` 崩掉 ⇒ §3 的「每 draft token 固定代價」要改名成「draft 前向成本」；
- 若有人在拉丁方設計下量到 pool 對**絕對 t/s** 有 >20% 效果 ⇒ `POOL_BUDGET_COST_DECOMP` §3 的彈性估計作廢。
