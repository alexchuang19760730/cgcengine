# 聯立 Shape × Kernel：哪一邊還活著（2026-09-22 夜）

承接 §EN-441（knob 清單核對）。回答的問題是：

> 上層調度（餵什麼形狀）+ 下層 kernel（怎麼處理那個形狀）要一起優化。

**同意。但這句話在本 workload 上有一個已經被實測切開的地方，事先講清楚可以省掉一整夜的 GPU。**

---

## 0. 一句話

| | 結論 |
|---|---|
| 「把**同一個 kernel** 在某個形狀下跑快」 | **dense 那一半已判死**（上界 2.42% < 3% 門檻），而且 **`CGC_MMV_NSG` 在 IQ4_XS 上早已 1..32 掃過並否證** —— 我上一輪講反了，見 §1 |
| 「改變一步裡的 **dispatch / graph 結構**」（融合＝少一次 dispatch） | **還沒排除**，也是 joint 論點真正的落點 |
| 「MoE expert 路徑（`MUL_MAT_ID`）的 shape 與 tile」 | **唯一數值上還有空間的地方**，而且 joint 儀器**已經存在**（§3），只是**跑出來是 INVALID**（§4） |

---

## 1. 先撤回：我上一輪講錯的一句話

§EN-441 / `docs/KNOB_INVENTORY_VERIFIED_2026-09-22.md` 寫：

> 「舊的 NSG 全射程否證……**不覆蓋佔 dense 81.1% 的 IQ4_XS**」

**這句是錯的。** 證據在我們自己寫的 code 裡：

- `scripts/check/shape_probe/sweep_nsg.py:38-42` 的 `SWEEP` 字典，三條撰掃的形狀就是
  `("iq4_xs", 2048, 8192)` = `attn_qkv`、`("iq4_xs", 4096, 2048)` = `ssm_out`、
  `("iq4_xs", 2048, 512)` = `ffn_gate_shexp` —— **全是 IQ4_XS**。
- `docs/DENSE_GEMV_INSTRUMENT_2026-09-22.md` §4 有 pipeline 名作證：
  `CGC_MMV_NSG=8 → kernel_mul_mv_iq4_xs_f32_nsg=8_ne12=1_r2=1_r3=1`（用 `/tmp/cgc-nsg-build`，
  P1-3d 分支，`libggml-metal` 09-22 19:19）。
- §5 那張表（unset/1/4/8/16/32）因此**就是在 IQ4_XS 上的全射程掃描**，判決是不採用。

⇒ **`CGC_MMV_NSG` 對 dense 這條通道已經測過了，不是「還沒測」。**
（`docs/BEST_SHAPE…` 那列「IQ4_XS = 0 knob」仍然過期 —— 通道確實已經 opens
—— 但「通道開了」跟「通道有錢」是兩件事。）

配套的硬上界（同文 §6）：把 **dense + lm_head 全部打到 100% DRAM 峰值**，
13.39 ms → 11.47 ms，可回收 **1.92 ms = 2.42% < 3% 門檻**。
⇒ **tile size / unroll / vector width 也救不了**：那些只改內部常數，分子都是同一批位元組，上界還是 2.42%。

---

## 2. 三層成本階梯（這決定「一起調」值不值得）

| 層 | 旋鈕例子 | 單次實驗成本 | 會不會換 binary／重新 anchor |
|---|---|---|---|
| 調度 | `CGC_POOL_MAX_TOKENS`、`CGC_SERVER_EXPERT_CACHE_BYTES`、`CGC_SERVER_N_CB` | **94 s**（一次 llama-bench launch，實測） | 不會 |
| kernel·泛函常數 | `CGC_MMV_NSG`（IQ2_S/IQ3_XXS/IQ4_XS/Q6_K/Q8_0/IQ3_S，clamp 1..32）、`CGC_MMV_NR0`（僅 IQ2_S/IQ3_XXS） | 94 s（+ 每個值一次 pipeline 編譯） | **不會**（Metal function constant `FC_MUL_MV+0`，runtime 編並快取） |
| kernel·原始碼 | tile size / unroll / vector width | **build + 重跑 m123 oracle gate 三 PASS** | **會**（dylib md5 換掉 ⇒ 探針拒測） |

⇒ **不要 flat 地說「kernel 層要掃」**：第三層的成本不是 94 s，是「一次 rebuild 會蓋掉別條線正在 mmap 的 dylib」
（`~/.workbuddy/MEMORY.md`：這是本 repo 已知會殺掉別人實驗的動作）。

---

## 3. joint 儀器**已經存在**（這點可能出乎意料）

`scripts/check/shape_probe/mmid_shapes.py`：

```
--tokens 1,2,3,4      # ← shape 軸：help 原文 "the MTP axis: tokens per op"
--nsg-sweep "unset,1,2,4,8,16,32"   # ← kernel 軸
--reps / --target-ms / --json
```

也就是說 **speculative width 決定 `mul_mat_id` 一次處理幾個 token（ne21），NSG 決定那
個形狀怎麼被切成 simdgroup** —— 這正是「餵什麼形狀 ⊗ 怎麼處理形狀」的聯立版本，
而且用的是已確立的權威計費口徑 `marginal = (g(64) − g(32)) / 32`。

- `--selftest` **27/27**（含：arm 順序旋轉每輪都不同、marginal 精確消掉固定項、
  **null cell 寬 31% 的通道必須判 REFUSED 而不是給中位數**、knob 沒進 kernel 就判 INVALID）。
- 每個 arm 都對自己**實際編出來的 pipeline 名**驗 PV（`kernel_mul_mv_id_iq3_s_f32_nsg=8`），
  不是對著 env 變數驗 —— 因為 IQ3_S 是 P1-3e 才接上的，舊 dylib 會靜默忽略。

---

## 4. 但它**已經跑過，而且兩條形狀都是 INVALID**

`Backup/phase_decomp/L3/shape_probe/mmid_nsg_sweep_rotated.json`（09-22 20:00，tokens=4，reps=3）：

| 形狀 | 型別 | %DRAM 峰值（中位數） | per-rep 離散 | verdict |
|---|---|---|---|---|
| `gate_exps` 2048×512×256 | IQ2_S | unset **30.1**、nsg=1 26.1、nsg=4 **36.2**、8 32.9、16 30.3、32 30.0 | **21–44%** | **INVALID** |
| `down_exps` 512×2048×256 | IQ3_S | unset **19.6**、1 18.1、4 20.9、8 17.5、16 18.5、32 **11.5** | 20–39% | **INVALID** |

同一份 json 自己交代為什麼：

```
window.class = "busy-overridden"
window.why   = "usable 5.18 GiB vs the shared admission bar 7.8 GiB ... swap 3711 MiB"
```

**null cell（unset 那一格自身）寬 44%**，所以工具拒絕給中位數，只給 status。這是对的：
在一條 44% 寬的通道上取中位數，讀出來的是漂遷不是旋鈕。

### 但這個數量級值得注意 —— 它跟 dense 完全不同

| 家族 | 有效頻寬 / 峰值 | notes |
|---|---|---|
| dense `attn_qkv`/`ssm_out`/`attn_gate`（IQ4_XS） | **76.2 / 76.5 / 77.4%** |幾乎貼著屋頂 ⇒ 才會有「打滿也只有 2.42%」那個自我否證的上界 |
| `lm_head`（Q6_K） | 91.2% | |
| **MoE `gate_exps`（IQ2_S）** | **~30%** | |
| **MoE `down_exps`（IQ3_S）** | **~20%** | **離屋頂有 3–5 倍** |

數量級估算（experts 342.5 MiB/步，取自 `docs/BEST_SHAPE_IQ3XXS_M4_2026-09-22.md`；
**只是把 placement 換算一下，不是量測**）：
以 25% 峰值 ≈ 27 GB/s ⇒ ~12.7 ms/步；若到 dense 的水準 77% ⇒ ~4.3 ms/步，
差 ~8 ms/步 ≈ **10% of a 79.5 ms step** —— 是門檻的 3 倍多。

⚠ **這串數字目前不能引用**：中位數是站在 20–44% 寬的 null cell 上，place 有可能整體是 0。
它的用途只有一個：**說明這一族「還沒被否證」，值得先把它量乾淨**，不像 dense 那樣已經有自我否證的上界。

---

## 5. 剩下的一條：`CGC_MMV_FUSE`，以及它跟既有實測**打架**（未解）

- - `ggml-metal-ops.cpp:2800` 說 `CGC_MMV_FUSE=1` 才開
  `mul_mat_id_glu_fused`，而 `can_batch_mmv_glu_down`（`:3242`）只在它裡面被叫
  ⇒ prod 現有的 `CGC_GLU_FUSED_DOWN=1` 今天是惰性的。它在 `run_server.sh:107` 引用的
  「§8.113 +6.5%」**從未生效過**。
- **但 `scripts/check/autotune_mmv.py:108-115` 記載相反的事实**：
  FUSE 宣稱 bit-identical，**實測 777/800 rows divergent**，端到端 **15.94 vs 22.07 t/s（更慢）**，
  draft accept 崩掉 ⇒ 該檔案帶 `prior=measured_worse` + `adoption=needs_m1_gate`。

⇒ **我上一輪把 `CGC_MMV_FUSE=1` 列為「第一步先跑」是不完整的**：那次記錄存在，
只是它的 profile（ symptom "draft accept collapsed" 指向一個與 `--spec-type draft-mtp`
不同的配置）與今天的 prod25 cell 未必同一個 regime。下一步要麼先把那筆搬到 prod25 cell 重現，
要麼明確宣告它不適用，**不能當不存在**。

---

## 6. 所以「一起調」該怎麼做

1. **不要在 dense 上再找 tile。** 通道開了、東西掃了、上界自己也否證自己（2.42% < 3%）。
2. **joint 的戰場是 MoE `MUL_MAT_ID`**：它的形狀真的被 M（tokens/op）決定，而且 %峰值只有 20–36%。
   儀器現成（`mmid_shapes.py --tokens 1,2,3,4 --nsg-sweep`），缺的是**乾淨窗口**。
3. **順序（省的不是想法，是 σ）**：
   ① 等窗口回到 **usable ≥ 7.8 GiB**（今晚 json 是 5.18 GiB／swap 3.7 GiB）；
   ② 同一個 libdir 下先把 **null cell 壓到 ≤5%**（reps↑、`--target-ms`、輪轉）；
   ③ 才跑 `--tokens 1,2,3,4` 全軸；
   ④ op 級若出現一條 ≥10% 的 arm，**才**把它送上 94 s/launch 的端到端。
   （σ 降不下來就別上加axis —— 盲加 axis 只會把 3% 的效應埋在 11% 的離散裡。）
4. `CGC_MMV_FUSE` 要先解 §5 的衝突。

---

## 7. 本文會變動的地方

- 若乾淨窗口下的 mmid joint sweep 把 null cell 壓到 ~2%，**表裡那兩個 %峰值會整片移動**（可能到 0）。
- `--tokens` 目前只跑過 **4**；1/2/3 完全沒有資料。
- `CGC_MMV_NR0` 只對 IQ2_S/IQ3_XXS 有效，`down_exps`（IQ3_S）吃不到 ⇒ joint 軸目前只有 NSG 一個 kernel 維。

---

## 8. 2026-09-23 01:0x–01:3x 追加：試了兩次，**儀器兩次都拒答**，而且原因找到了

§6 排程的 ①② 今晚真的下去試了。結果不是答案，是**一個環境事實** —— 但它比答案重要，
因為它決定了「能不能問」，不是「問一次多少錢」。

### 8.1 兩次 joint sweep（同一 build，`/tmp/cgc-nsgid-build`，dylib `650dd9d1d9fb89a4`）

| 次 | 起始窗口 | Null cell 跨度（gate / down） | 門檻 | 結果 |
|---|---|---|---|---|
| 9-22 20:00（舊） | usable 5.18 GiB | 44% / — | ≤5% | `INVALID` |
| **9-23 01:05** | usable **1.47 GiB**、swap 5300→5424 MiB | **57.7% / 21.0%** | ≤5% | `NO READING` |
| **9-23 01:21** | usable **0.83 GiB**、swap 7437 MiB | **241.5% / 13.6%** | ≤5% | `NO READING` |

第二輪甚至跑出**負的 %峰值**（`unset -25.1%`、`4 -9.1%`）：marginal 模具 `(g(64)-g(32))/32`
在這種情況下會直接給出負數（後一次 batch 反而更快）⇒ 那是堆在打架，不是 kernel 在跑。
三次都是同一句：`the CHANNEL moved: the unset arm itself spans X% ... rerun in a quieter window`。

### 8.2 根因：這台機器今晚**沒有** 8 GB 可回收的空檔（實測）

用 `box_probe_compare.py --n 4 --interval 5` 對照兩個探針（結論很一致，binding=harness）：

```
reclaimable 7153–7544 MB / need 8000 MB  → harness refuses
launcher: FREE% 82–83 / req 40           → admits
samples=4  quiet=0  disagreements=4  refused_by=['harness:memory']
```

再連採 5 次（`server_window.decision()`，間隔 4 s）：`6376 / 6242 / 6229 / 6385 / 6433 MB`，
全程 `foreign_llama=none`、`port_held=none`、`admits=False`。

也就是說：**在沒有任何 llama 行程的狀況下**，reclaimable 也只有 ~6.3 GB。

而這一夜它的擺幅是 **0.83 → 9.43 GiB**：9.43 GB 那一次是另一條線的 `llama-server`
（pid 64026，RSS 8.1 GiB）退出的那一瞬間量到的；它 01:07 起來、01:2x 結束，
剛好把兩次 sweep 的前後都包住了。⇒ 所以我的讀法是：**「乾淨窗口」在這種多人共用 + 桌面 app
常駐的 16 GB 盒子上不是排隊等得到的東西，它取決於別人的行程週期。**

### 8.3 兩個副作用（必須寫下來）

1. **我把 swapfile 從 5.1 GiB 推到 6.1 GiB**（`vm.swapusage` total 5.1→6.1，used 3.4→7.4 GiB 峰值）。
   這是 microbench 自己造成的（`--target-ms 1500` × 72 次 probe invocation），而且**不會自動縮回去**
   ⇒ 它可能影響到別條線接下來的 run。記在這裡以便之後劃得清楚是誰的。
2. 也因此我**停止了第三次的嘗試**：再跑一次會繼續把 swap 往上推，換到的還是一次 `NO READING`。

### 8.4 順手補起來的洞：`shape_knob_search.py` 根本沒接共用窗口閘門

端到端 harness 有自己的 `--min-free-pct 15` + 逐 cell 的 settle（看 thermal），
但**從來沒問過「這台機器現在是誰的」** —— 而 `server_window` 是其他所有 launcher 問的那個模組，
它看得見**外來的 llama-server**，那是一個 sweep 在自己印出來的數字裡完全看不見的東西。

今晚我自己就踩進去了：00:59 在 usable 6.99 GiB（低於共用門檻 7.81 GiB）時就發了 sweep，
而我自己的排程白紙黑字寫著「要在 ≥7.8 GiB 的乾淨窗口跑」。

已加（全部零 GPU，`--selftest` **27 → 36/36**）：

- `judge_window(before, after)` —— 起點決定能不能發，終點決定機器有沒有在過程中換人；
  換人 = `hijacked`（不是 drift）。陷阱是有向的：**只有「失去」機器算 hijack**，
  反向（忙→淨）仍是 `busy-overridden`（有 selftest 突變保證）。
- `maybe_proceed(before, allow_busy)` —— 拒絕時把原因印進 log，單看 log 就足以讓那一輪 VOID。
- `--need-mb`（0 = 共用 `server_window.NEED_MB`）／`--allow-busy`（逃生門，不是通行證），
  對齊既有 `--allow-inert` 的哲學。

所以此刻的狀態是： `--run` 會**直接 rc=2 拒絕**（reclaim 6.3 < 8.0 GB）。
這意味著今晚 §6 排程的 ①② **都 run 不了** —— 而這是正確的結果，不是 bug。

### 8.5 這給出一個待決的問題（不是我單方面能定的）

`NEED_MB=8000` 的校準根據是「guarded run 低於此線會載不進模型」（server_window:67），
那是**載入失敗**的門檻，不是**噪音**的門檻。真正決定能不能引用讀數的是儀器自己的
`nsg_decide` 判準（null cell ≤5%）。這中間的落差現在沒有人量過：

- (A) **維持**：端到端 sweeps 也要 reclaim ≥8 GB ⇒ 今晚、很可能明晚都發不出去，除非關掉約 1.7 GB 的 app。
- (B) **改用「讀得穩」當門檻**：把 joint probe 的 `--experts` 開一個 footprint 把手（現在寫死 256），
  讓 footprint 小到不去 thrash，然後用 null cell ≤5% 當真正的通行判準。
  但這樣就是在動**共用常數的適用範圍**，應該要在 `server_window` 層連 `--need-mb` 的證據一起改，不是在這裡偷偷放寬。

我沒有單方面選 (B)，因為放寬自己要用的門檻正是這份筆記第 2 節在罵的那件事。

## 9. 2026-09-23 02:30 追加：(B) 做完了，**答案是「不是 footprint 的鍋」**

使用者同意先試半步 (B)。做完的完整記錄在 **`docs/FOOTPRINT_HALFSTEP_2026-09-23.md`**，這裡只留結論：

- 新工具 `scripts/check/shape_probe/footprint_ab.py`（selftest 26/26）把三個可分離的成分定價：
  pool 320→146/174 MiB 讓**盒子可用下陷降 26.4%**（可分辨），但**行程 RSS 一動都沒動**（三臂皆 232 MB）
  —— pool 裡超過 bank 的頁從來沒被 fault-in，所以要看的是共用視窗的 `vm_free_mb`，不是 child RSS。
- 帶著瘦池重跑 joint sweep：null cell 從 241.5% 降到 **26.2% / 15.7%**，仍然是門檻的 3–5 倍 ⇒ 依然拒答。
- `--sequence` 模式（每臂連打 12 次、三臂交錯）顯示累積掉的是 **~1.2 GB，三種 config 一模一樣**
  ⇒ 那 ~1.2 GB 不是我們能調的那兩個旋鈕造成的。
- **真兇**：新加的 win begin/end 抓到跑 sweep 的兩分鐘裡 `foreign=[(80017, 'llama-server')]`，
  而盒子自己的可用記憶體在數十秒內擺盪 **1463 ↔ 9869 MB（8.5 GB）**。
  我們能調的整套 footprint 是 0.13 GB —— **比真正的原因小 20 倍**。

⇒ **`NEED_MB=8000` 不動**，但不是「預設如此」，而是「量過了，降 footprint 換不到可讀性」。
⇒ 真正缺的是閘門：`nsg_sweep` 原本只會**印**窗口不會**擋**，已經補上（`return 2` + `--allow-busy` 留記錄）。
⇒ 五次 joint sweep，五次都沒有一次跑在乾淨盒子上 —— 那五個 null cell 數字測的是別人的記憶體環境。

順便更正本文件 §8.3 我上一輪自己寫錯的兩句話（詳細在 §9 指到的那份文件第 9 節）：
swapfile **會**縮回去（實測回到 4096 MiB）；這台機器**有** 8 GB 空檔（實測 9364–9428 MB 連續六次），只是不穩定。
兩句都是同一個毛病：用「我看到的」代替「它的性質」。
