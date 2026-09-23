# 那 154 ms 是什麼：裝置忙碌時間，不是 host（2026-09-23）

**工具** `scripts/check/verify_marginal.py --device-split`（selftest 全綠，新增 6 項）
**載體** Nail IQ3_XXS-denseIQ4X · MTP on k=3 · pool 8 GiB · `prod25` 血統 · llama-bench decode-spec
**產物** `Backup/phase_decomp/node_attr/nsm_{k1,k2,k3,shard_k1,shard_k3,capture_decode_spec}_20260923.log`

---

## 0. 一句話

交付 step 裡那塊「沒人認領的 ~154 ms」**是裝置忙碌時間**，不是主機開銷；它的 **57% 住在 MoE 區（層內節點 0–39）、42% 住在 attention/GDN 區（40–89）**；
而**邊際 verify token** 的 +26.3 ms 裝置時間裡，MoE 區只佔 +14.2（其中 kernel 自己的地板是 +7.9），另外 **+13.4 是 attention/GDN 投影與掃描**。

---

## 1. 先修「可用的 device-busy 數字」——答案是不用重建引擎，但要用細切的擷取

### 1.1 step 級尾線 (`gpu_sum`/`union_sum`) 不可以引用，證據在同一行上

```
CGC-DECPROF: step=69 segs=41 layers=40 total=251.72 ms | wait=160.49 (64%) cb=84.45 (34%) submit=6.78 (3%) ntok=4 | layer gpu_sum=685570162.37 union_sum=171392728.16 gap_sum=94.06 ms
CGC-DECPROF: step=70 segs= 2 layers= 1 total=  2.16 ms | wait=  1.99 (92%) cb= 0.04 ( 2%) submit=0.13 ( 6%) ntok=4 | layer gpu_sum=171393038.76 union_sum= 85696515.61 gap_sum=0.00 ms
```

一個 **2.16 ms** 的 draft step 帶著 **85.7 ms** 的 `union_sum`，而兩條 step 的 `union_sum` 只差最後幾位數 ⇒ 那是**每步 2 個假 buffer**（`GPUEndTime − GPUStartTime` 回傳絕對開機時戳 ≈1.7e14 ns，通過了 emitter 的 `s > 0 && e > s` 守衛）在灌同一個累加器。
**我上一輪報的「裝置 span 邊際 ~31 ms/token」就是從這條線算出來的，作廢。**

### 1.2 逐 buffer 的 NSM 是健康的 —— 只要套上「buffer 不得長於它自己的 step」

把每步 2 個假 buffer 丟掉之後，逐 buffer 的 GPU 時長**與 host 的 `wait` 視窗閉合**：

| 擷取 | T | n | step total | wait | cb | submit | **device Σdur** | **device/wait** | bufs/step |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 粗切（預設 `n_nodes_0=MAX(64,…)`） | 4 | 34 | 159.2 | 110.5 | 43.5 | 4.7 | **114.66** | 1.037 | 318 |
| 細切（`CGC_CB_N_MAIN=1` + 大 `n_cb`） | 2 | 16 | 112.6 | 79.1 | 17.6 | 8.4 | **103.00** | 1.303 | 622 |
| 細切 | 4 | 48 | 234.6 | 142.6 | 81.1 | 9.5 | **155.61** | 1.091 | 631 |
| 細切 | 8 | 64 | 411.2 | 316.7 | 90.5 | 6.5 | **315.02** | 0.995 | 589 |

⇒ 比例 1.0 上下（>1 是因為同一 step 內的 buffer 互相重疊，Σdur 會重複計），**所以 `wait` 視窗內 GPU 是滿的**：交付 regime 沒有「CPU 造成的大段 GPU 空窗」在 trunk 的 wait 裡。

### 1.3 為什麼一定要細切

粗切下 `n_nodes_0 = MAX(64, 0.1N)` ⇒ **每層 ~100 個節點裡前 64 個擠在同一個 buffer**，40 個這種 buffer 吃掉 **61.5%** 的裝置時間，而是 10 個節點區塊裡只認得出 4 塊 —— 這張表**讀起來像歸屬、其實什麼都沒識別**。`CGC_CB_N_MAIN=1` 覆寫那個樓，buffer 變成 1–7 個節點，圖的節點索引才變成時間軸上的位置。
⚠ 這個模式會改變絕對步時 ⇒ **該輪的 t/s 不可引用**。

---

## 2. 歸屬：層內節點位置（細切，T=4，n=48）

| 節點區 | ms/step | 佔比 | 區內主要 kind |
|---|---:|---:|---|
| 0–9 | 54.49 | 32.7% | `ffn_moe_`, `shared_expert_gate`, `ffn_moe_gate`, `ffn_moe_up` |
| 10–19 | 26.26 | 15.7% | `ffn_moe_`, `ffn_`, `ffn_moe_down`, `conv` |
| 20–29 | 5.99 | 3.6% | `ffn_moe_add`, `node`, `conv` |
| 30–39 | 8.67 | 5.2% | `ffn_moe_add`, `norm`, `ffn_`, `l_out` |
| 40–49 | 22.51 | 13.5% | `Kcur`, `Qcur`, `conv`, `attn_norm` |
| 50–59 | 6.60 | 4.0% | `conv`, `beta`, `alpha` |
| 60–69 | 5.02 | 3.0% | `cache`, `gate`, `q_conv` |
| 70–79 | 24.15 | 14.5% | `z-`, `k_conv`, `v_conv`, `attn_output` |
| 80–89 | 11.05 | 6.6% | `linear_attn`, `dn_normg_mul`, `final_output` |
| 90–99 | 2.09 | 1.3% | `ffn_moe_probs`, `ffn_moe_logits`, `ffn_moe_argsort` |

**MoE 區（0–39）= 95.4 ms/step＝57%**；**attention/GDN 區（40–89）= 69.3 ms/step＝42%**。
⚠ 位置與家族**有重疊**（`conv` 節點也出現在 0–19，`ffn_moe_probs` 出現在 90–99），所以這是「圖上位置」的分割，不是純家族分割。

---

## 3. 邊際 verify token：+26.3 ms 裝置，**MoE 只佔一半**

同一次啟動內 T=2 → T=4（n=16 / n=48）：

| 項目 | 每 token |
|---|---:|
| step total | **+61.0 ms** |
| ├ `wait`（host 時鐘，健康） | +31.77 |
| │  └ **device Σdur** | **+26.30** |
| └ `cb`（host hook） | +31.71 |
| └ `submit` | +0.53 |

**裝置邊際逐區**：

| 區域 | 每 token | 家族 |
|---|---:|---|
| 0–9 | +6.59 | MoE gate/up + shared |
| 10–19 | +4.74 | MoE down |
| 20–29 | +0.86 | MoE add/residual |
| 30–39 | +2.03 | MoE post + l_out |
| 40–49 | +5.57 | attn q/k/v 投影（GDN 與全注意力） |
| 50–59 | +0.33 | GDN conv/beta/alpha |
| 60–69 | +1.01 | GDN gate/q_conv |
| 70–79 | +3.90 | GDN z-/k_conv/v_conv + attn_out |
| 80–89 | +2.60 | linear_attn 掃描 + out norm |
| 90–99 | +0.13 | topk/argsort/logits |
| **MoE 合計** | **+14.22** | |
| **attn/GDN 合計** | **+13.41** | |

**對照 kernel 自己的地板**（`Backup/phase_decomp/L3/shape_probe/mmid_gather.json`，真幾何 256 experts × top-8 × 40 層；同值也印在 `docs/MOE_GATHER_BOUND_2026-09-22.md` 的 §2 表）：**expert bank 家族 = 8.09 / 15.98 / 23.79 / 31.81 ms/步**（T=1..4），**邊際 +7.91 ms/token，`extra_over_first = 0.977`**，頻寬利用率 38.5%（`down_exps` 只到 33.5 GiB/s＝30.8%）。

> ⚠ **單位／身分警語**：`15.98` 是 **T=2 時該家族的毫秒/步（成本）**，不是吞吐。
> 這個 repo 裡唯一數值相近的速率是 **15.89 t/s**，那是 `Backup/prod_matrix/20260923_155637_prod25_decode-spec/`  2026-09-23 15:56:48 的**單次帶儀器**讀數（見 §5）。
> 兩者**不可放在同一個句子裡比較**，也不可互相引用（一個是每步毫秒、一個是每秒 token）。

⇒ 兩句話：
1. **expert bank 只解釋裝置邊際的 30%**（7.91 / 26.30）。它自己已經證實**線性**（第二個 token 付第一個的 97.7%：top-8 of 256 幾乎不重疊）⇒ **「batch verify 讓第二個 token 只花 20–30%」在 host 側（派送 +0.53/token）與 kernel 側（97.7% 線性）都已被量掉**。
2. **另外 70% 的裝置邊際在 attention/GDN 與 MoE 的其它項**，而它們**沒有 kernel 地板可對照**（`mmid_shapes.py` 只價了 expert bank）。

---

## 4. 這對 25 t/s 的意思

```
交付 anchor                 step 247.98 ms  →  12.57 t/s
把 cb 全藏（CB_DELIVERY）     step 197.6   →  15.8 t/s   （＋14~19% 那條線的上界）
25 t/s 需要                  step ≤ 124.7 ms
本細切 cell 的 device 單獨    155.6 ms      →  已經超過 124.7
```

⇒ **就算 cb 歸零，25 t/s 還要求「裝置自己」再降 ~20%。** 而裝置那 155.6 ms 由兩個數字撐著：

| 候選 | 今天 | 地板/對照 | 可拿的量級 |
|---|---:|---|---|
| MoE 區（0–39） | 95.4 ms/step | kernel 對 expert bank **家族**（不含 shared expert / norm / conv）T=4：31.8 ms/步 | **上界 63.6 ms/step** ⚠ 該節點區含的比那個家族多（`shared_expert_gate`、`norm`、`conv` 都在裡面）⇒ 3.0× 是**差距的上界，不是量到的差距** |
| attention/GDN 區（40–89） | 69.3 ms/step | 無 | 未知（KV 讀取的地板約 0.4 ms/token，離 13.4 很遠） |

這兩格就是「25 還有沒有機會」的全部空間，而且**第一格已經有地板可對照**（這是它比第二格值錢的原因）。

---

## 5. 仍然不能歸屬的，以及工具自己標出來的假值

- **模型自由那條路（`nk=1`，無需任何模型）只覆蓋 15% 的裝置時間**（25.3 / 166.8 ms），其餘 85% 落在 573 個多節點 buffer/step 裡。
- 而且它最大的一項是**假值**：`ffn_moe_topk` **T=2 21.68 → T=4 15.45 ms/step**、`ffn_moe_` 9.94 → 9.48 —— **token 越多反而越小**，是 emitter 自己記過的 VIEW/RESHAPE 類（該 buffer 沒有 GPU 命令，時戳跨到下一個東西）。工具現在**按名字 flag** 這兩項，不讓它們被加進任何歸屬。
- 要拆掉剩下的 85%，只有兩條：**形狀鍵化的 kind**（emitter 改動）或已經存在的反卷積矩陣 `--nsm`（rank 20/44，且已證實對不上 marginal）。
- **`build_commit` 不是引擎身分**（2026-09-23 查證）：`Backup/prod_matrix/*/summary.json` 只記 `build_commit` + `build_number`，沒有 binary digest；而 `1ca685490` 本身是**純文檔**（`docs/MOE_GATHER_BOUND_2026-09-22.md`，+17/−9，零代碼）⇒ 任何把引擎速度接到這個 commit 的說法都是假的。那輪的 binary 現在也無法回推（`Backup/` 下 15:40–16:10 之間沒有任何 witness/md5 檔）。
- **最高的那個 decode 讀數（15.89 t/s）因此沒有引擎身分**，而且它同時是：`n_kept=1`（單次，`stddev_ts=0.0` 無意義）、**7 個儀器旋鈕**（`DECODE_PROFILE[_ALL]`＋`GPU_TIMING`＋`GPU_NODES[_MATRIX]`＋`MTP_PERF`，其中 NSM 矩陣那顆按本 repo 自己的規則會改變絕對步時）、**`b=8/ub=8`**、**無 `--warm-skip`** ⇒ 與 12.57 那條 anchor（b=512、warm-skip 64、無儀器）**不可比**。它的 accept 0.951／`gen_tok_per_round=3.000` 是唯一可引用的部分，而其成因尚未查。

---

## 6. 重現

```sh
# 細切擷取（⚠ 該輪 t/s 不可引用）
CGC_GPU_NODES=1 CGC_GPU_NODES_MATRIX=1 CGC_DECODE_PROFILE=1 CGC_CB_N_MAIN=1 CGC_N_CB=127 \
python3 scripts/check/llama_bench_matrix.py --arms prod25-stream --reps 3 --prompt 0 --gen 128 \
  --depths 512 --batch 512 --ctx-size 4096 --warm-skip 64 --spec-type draft-mtp --k 1,3 \
  --workdir /tmp/dev_split

python3 scripts/check/verify_marginal.py --device-split \
  Backup/phase_decomp/node_attr/nsm_shard_k3_20260923.log
python3 scripts/check/verify_marginal.py --selftest
```

三條規則（都在工具裡，違反就拒答而不是照報）：buffer 不得長於自己的 step（丟棄並計數）、逐 buffer 時長必須對 `wait` 閉合（0.5–1.5，超出即拒算裝置欄）、**隨 token 變小而變小的 solo kind 一律按名字標成 VIEW/RESHAPE 假值**。
