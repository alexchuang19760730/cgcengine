# Knobs 清單核對：50+ 個 env、18 個「值得掃」——哪些是真的

日期 2026-09-22 23:5x · 方法：**全部零 GPU**（掃原始碼＋問 `run_server.sh CGC_DUMP_ENV=1`＋讀今晚那一輪 sweep 自己的 stderr）

---

## 0. 一句話

那份表和 `docs/SHAPE_WORLD_MODEL_2026-09-23.md` §四 的 A–E 表同源，而那張表的「✅ 可調」是**從 env 名字推出來的，沒有驗證過**。
實測：**35 個被點名的 knobs 裡 8 個名字/預設值錯、7 個根本到不了子行程、3 個不是性能旋鈕、4 個已有判決。**
然後是今晚最重要的一件事：**我們今晚跑的那輪 budget sweep 是無效的**（見 §2），它的軸從頭到尾沒動過。

---

## 1. 怎麼驗證的（兩支零 GPU 儀器）

**① 名字存不存在**：用 getenv 呼叫點當唯一權威，掃 `src/llama.cpp/{src,ggml/src,tools,common,examples,tests}`（排除 `build/`）。
`getenv/cgc_env_*` 的字面量共 **416 個不同 env 名**，其中 `CGC_*` 約 150 個（一半是 `*_DBG` 診斷開關）。

**② 到不到得了**：`scripts/check/shape_knob_search.py` 現在有 `resolved_env()` / `check_reachability()`。冷知識：

```
llama_bench_matrix.run_arm:  run_env = dict(os.environ); run_env.update(<resolved env>)
```

我們的 knob 是從**命令列**進來的（`--arms prod25:KEY=VAL`），不在 `os.environ` 裡，而 `<resolved env>` 就是
`run_server.sh` 印出來的 `SERVER_ENV` 那一份。⇒ **`run_server.sh` 沒有 push 進 `SERVER_ENV` 的變數，llama-bench 一輩子看不到。**
（這個陷阱 `run_server.sh:1102` 自己寫過一次，`run_server.sh:1552` 又寫了一次，這是第三次抓到它。）

判準不問「我的 key 有沒有活下來」，而是**把整份 resolved env 對 reference cell 做 diff**——真的旋鈕一定會動到某個東西
（可能換名字，diff 會印出來給你看）。加上 `--allow-inert` 逃生門，但印的是警告而不是通行證。

---

## 2. 今晚的 budget sweep 是無效的（撤回）

`GRIDS["budget"]` 用的 key 是 `LLAMA_ARG_EXPERT_CACHE`。**它不是任何程式讀的名字。**

| 事實 | 值 |
|---|---|
| 真正決定 pool 的鏈 | `CGC_SERVER_EXPERT_CACHE_BYTES` → `BUDGET`(`run_server.sh:468`) → `CGC_EXPERT_CACHE_BYTES`(`:1315`) |
| `LLAMA_ARG_EXPERT_CACHE=3/4/6/8 GiB` 的 resolved env | 全部仍是 `CGC_EXPERT_CACHE_BYTES=8589934592` |
| 10 個 arm cell ＋ 6 個 ref 的 stderr | **全部印 `n_slots=143`**（8 GiB 的幾何；10 GiB 會是 179 slots/layer） |

⇒ 那一輪的 4 個「arm」是**同一個配置的 10 次重複**。因此必須撤回 `docs/CLOSED_LOOP_AUTOTUNE_2026-09-22.md` §六 的：

> ~~`best=6 GiB +3.40% P(best)=0.44`~~ —— 它是同一配置內的次序/飄移，不是 budget 效應。

**但這一輪沒有白跑，它現在是我們手上最乾淨的噪音標定**（因為四臂恆等）：

```
10 個 arm cell: 7.73 / 9.38 / 9.51 / 9.80 / 9.83 / 9.94 / 10.22 / 10.26 / 11.05 / 12.05
mean 9.976  sd 1.119  CV 11.2%      ← 全程 NOMINAL、同一個 build、同一份 pool 幾何
6 個 ref:    9.43 / 10.09 / 10.20 / 10.47 / 10.47 / 11.44
```

也就是說：**單一 cell 的 ±11%、括號式 braveket 後 log 空間 σ≈0.28，兩者都只是 replica noise。**（§5 的算術用它。）

修好了：`GRIDS["budget"]` 改成 `CGC_SERVER_EXPERT_CACHE_BYTES`，ladder 改 `(3,4,6) GiB`——**8 GiB 不進 ladder**，
因為它就是 reference cell 本身（每個括號的兩個 ref 都在跑它），把它當 arm 只是花一次 cell 的量測費去量 baseline。
`--selftest` 現在 **27/27**，其中兩條就是這件事的突變測試：新 key 必須 VOID-free 地可達、`LLAMA_ARG_EXPERT_CACHE` 必須被判 INERT。

---

## 3. 逐條核對（標 ⚠ 的是與原表不同的地方）

### A. 形狀 / 調度

| 原表 | 實測 | 結論 |
|---|---|---|
| ⚠ `CGC_M` | **不存在**。真名 `CGC_POOL_MAX_TOKENS`（`llama-shape-knob.cpp:64`），別名 `CGC_SHAPE_M`(`:65`) | 改名；值仍可調 ✔ |
| `CGC_N_CB` | prod 已釘 8。**直接設 `CGC_N_CB=4` 會被覆寫回 8**，必須用 `CGC_SERVER_N_CB`（實測 8→4 ✔） | ⭐ 可掃，但要用把手 |
| ⚠ `CGC_PREFILL_STREAM` | prod=1，且被 **arm registry** 釘住 ⇒ `shape_knob_search` 的 conflict guard 會拒這個 cell | prefill 軸；對 `--prompt 0` 的交付格無關 |
| ⚠ `CGC_GATHER_SLAB_CAP` | 同上（arm registry），prefill slab | 同上 |
| `CGC_DBUF` | prod=1；設 `CGC_DBUF=0` ✔（從 SERVER_ENV 消失） | 可掃 |
| ⚠ `CGC_SPAC` / `_ALPHA` | prod=1 / 0.75，可改成 0 ✔，**但不要用**：EMA refresh 是 pool membership 的驅動器，關掉會回到 loader 的 identity prepopulate（code comment 實測 ~44% count-cold → zero-slot logits collapse） | 承載柱，不是旋鈕 |
| `CGC_POOL_SPLIT` | ✔ 可達；load 期分割（`docs/M1_POOL_SPLIT_COST_2026-09-14.md` 有成本） | ⭐ 可掃 |
| ⚠ `CGC_POOL_CAPTURE` | ✔ 可達，但它是**註解器**：給 POOL rows 加上 owner/expect 標註（`ggml-metal-ops.cpp:3744`），不是性能旋鈕 | 掃它只會變慢 |

### B. Kernel

| 原表 | 實測 | 結論 |
|---|---|---|
| ★ `CGC_MMV_NSG` | ✔ 可達 (1..32)。**射程今天變大了**：P1-3d 把 plain `mul_mv` 的 `IQ4_XS/Q6_K/Q8_0` 接上 env（`device.cpp:1006/921/954`），P1-3e 把 `mul_mv_id` 的 `IQ3_S` 接上（已 commit `0e5b56b0c`）；**而且已構進現行 binary**（metal dylib 20:48 > src 20:10，md5 與 ANCHOR 相符） | **見下 ★★★** |
| ⚠ `CGC_GLU_FUSED_DOWN` | prod=1，**但沒有 `CGC_MMV_FUSE=1` 就進不去**：`ggml_metal_op_can_fuse_mmv_glu` 先擋（`ops.cpp:2810`），而 `can_batch_mmv_glu_down` 只在 `mul_mat_id_glu_fused` 內被呼叫（`ops.cpp:3242`）。⇒ **這個「已開著」的旋鈕今天是惰性的**，而 `run_server.sh:107` 引用的「§8.113 +6.5%」並沒有在生效 | **見下 ★★★** |
| ⚠ `CGC_IDS_LINEAR_READ` | 這是**故意重現一個已知缺陷**的開關，註解明寫 *never quote quality or throughput from that arm* | 不是候選 |
| ⚠ `CGC_IDS_MAX_LINES` | debug dump 的行數預算，預設 0；開了會在熱路徑上做數千次 unbuffered stderr write | 反效果，不是候選 |
| `CGC_OA_ASYNC` | prod=1；要用 `CGC_SERVER_OA_ASYNC=0`（實測 1→0 ✔），歷史 +12.6% | 已知方向，別期待新錢 |

**★★★ 今晚唯一的兩個真東西：**

1. **`CGC_MMV_FUSE=1`（＋保持 `CGC_GLU_FUSED_DOWN=1`）** —— 這是清單**漏掉**的那一個。`ops.cpp:2798` 自稱
   「把 gate/up 兩次 GEMV 對同一次 y 載入算完並直接吐 `silu(gate)*up`，**bit-identical**」，而且融合後才輪到
   `can_batch_mmv_glu_down`（`GLU_FUSED_DOWN` 的那條路）。今天兩pa都在 prod env 裡，但**前門關著**。
2. **`CGC_MMV_NSG` 重掃** —— 之前「NSG 全射程否證」（`docs/AUTOTUNER_WIRING_2026-09-22.md`）是在**舊射程**下測的：
   那時 NSG 只碰得到 `IQ2_S/IQ3_XXS`，而佔 dense 81.1% 位元組的 **`IQ4_XS` 完全是編譯期宏**（`docs/BEST_SHAPE_IQ3XXS_M4_2026-09-22.md` §3）。
   P1-3d 落地後那句「否證」**不再覆蓋** IQ4_XS/Q6_K/Q8_0。bit-identity 論據仍在（每行點積留在單一 simdgroup，nsg 不變歸約順序）。

### C. Prefetch / 快取

| 原表 | 實測 | 結論 |
|---|---|---|
| ⚠ `CGC_NO_PREFETCH` | prod=1，但 `llama-context.cpp:2086` 的閘門是 `(NO_PREFETCH==nullptr \|\| spac_on)`，而 prod 的 `CGC_SPAC=1` ⇒ **這個 prefetch 區塊本來就在跑**。常見誤讀「prod 沒開預取」是錯的 | 不是 ⭐⭐⭐ 的那條開關 |
| ⚠ `CGC_DRAFT_PREFETCH` / `_SYNC` | **INERT**（launcher 沒 port）：設了之後 resolved env 與 reference 完全相同 | 要掃得先加 allowlist |
| `CGC_LAYER_AHEAD_PREFETCH` | ✔ 可達；09-19 那一臂 bit-identity 9/9。**但 `docs/F2_F5_OVERLAP_AND_AUDIT_RESULT_2026-09-20.md:33` 的教訓：那一輪的 B 臂 env 裡有這個 flag，`prefetch=0/0`、零 PFDBG ⇒ 治療從未執行**；`docs/L3_M0_DOSE_RESULT_2026-09-20.md:88` 說這個區塊只長在批次池路徑上，而 verify ntok=4 多數層走 fast path | 可測，但**必須先有「真的執行了」的計數閘門**，不能只看 t/s |
| `CGC_PREV_TOKEN_PREFETCH` | ✔ 可達；無端到端判決 | ⭐ 唯一乾淨的 prefetch 候選 |
| ⚠ `CGC_PREFETCH_WINDOW` | 只作用於 `CGC_PREFETCH_SRC=hist`（預設是 step），而 hist/prev **與 `CGC_SPAC=1` 互斥**（`ctx:2060` 註解）；`CGC_PREFETCH_SRC` 本身 INERT | 在交付配置下無法單獨掃 |
| ⚠ `CGC_EVICTED_RING` | **INERT**：`run_server.sh:1361` 硬寫 `CGC_EVICTED_RING=0`，怎麼設都是 0 | 要改 launcher |
| `CGC_SLOT_TABLE_GPU` | ✔ 可達；但這是 **S1 線**（分工屬另一條 session），且 S2 那端已判不可達 | 別在本線動 |
| ⚠ `CGC_SYNCFILL_COLD` | 預設是 **ON**（原表寫 0 是錯的，`ctx:6426` 是 `nullptr → true`），且它是**正確性不變式**（refuse fast path when any selected expert still cold） | 不是性能旋鈕 |
| ⚠ `CGC_SYNCFILL_SERIAL` | 舊串行填法的 A/B 還原 | 預期更慢，不是候選 |

### D. MTP / Speculative

| 原表 | 實測 | 結論 |
|---|---|---|
| ⚠ `CGC_SERVER_MTP` | **引擎不讀這個名字**（getenv 掃描無此 key）；它是 `run_server.sh:88` 的 shell 變數。bench 側的 MTP 開關是 `--spec-type draft-mtp` | 不是 autotuner 的軸；且 MTP off 9.82 vs on 12.62 ⇒ 別動 |
| `CGC_DRAFT_DECODE`/`CGC_VERIFY_DECODE` | prod=1，MTP fast path 的一部分 | 已知在天秤對面 |
| ⚠ `CGC_DRAFT_STRICT` | **INERT** | — |
| ⚠ `CGC_PREROUTER`/`_TOP_K` | ✔ 可達，**但 09-17 已經量過**：`calls=80 queued=0 scored=40 pred_total=320 hit=41 precision=12.8%`，`prefetch_slot` 拒了全部 320 次（`prewarm_hot` 用同一份 `freq` 排序先把它們裝進去了）；roadmap 自己給的上限 0.2pp | 已記錄的死路（還附一個 `prefetch_slot` 其實會 evict 的 bug） |
| ⚠ `CGC_P_ROUTE` | **這是量測儀表**（記錄 RSL-MTP 要的 p_route，`ctx:6149`），不是性能旋鈕；且 INERT | 不是候選 |
| `CGC_RN_ROUTING`/`CGC_RN_NOOP` | **INERT** | — |

### E. 原表沒列、但可達且有意義的

`CGC_MMV_FUSE`（★）、`CGC_MMV_NR0`（**INERT**，要先用 allowlist）、`CGC_SUBMIT_AHEAD`（zero-code 天花板探針）、
`CGC_DOWN_COMBINE`、`CGC_MASSCOV`、`CGC_SERVER_WORKERS`（`LLAMA_EXPERT_CACHE_WORKERS` 直接設是 INERT；已知 8→2 是 µs/miss −9.8%、
端到端 ≈+1.7% 低於門檻）、`CGC_MM_BITIDENT=0`（拆 bit-identity 支柱）。

---

## 4. 統計訂正

| 類別 | 原表 | 實測（可真正 moves something 的） |
|---|---|---|
| 形狀 / 調度 | 5 | **3**（`CGC_POOL_MAX_TOKENS`、`CGC_SERVER_N_CB`、`CGC_POOL_SPLIT`） |
| Kernel 效率 | 3 | **2**（`CGC_MMV_NSG` 重掃、`CGC_MMV_FUSE` ＋ GLU 配對）＋原表漏的這兩個才是主力 |
| Prefetch / 快取 | 6 | **2**（`CGC_PREV_TOKEN_PREFETCH`、`CGC_LAYER_AHEAD_PREFETCH`＋執行計數） |
| MTP / Speculative | 4 | **0**（MTP 別動、PREROUTER 已死、P_ROUTE 是儀表、SERVER_MTP 不存在） |
| **合計** | **18** | **7**，其中只有 2 個有 ≥5% 的機制理由 |

---

## 5. 就算 18 個都成立，量得起嗎（用今晚的數字算）

一次 launch（含 model load）實測：**1507 s / 16 次 ≈ 94 s**。
單一 cell 的噪音（恆等配置實測）：log 空間 **σ ≈ 0.28**（raw CV 11.2%）。
兩臂在 95% 下要認證一個 g 的差距，每臂需要 `n ≥ 2·(1.96σ/g)²`：

| 要認證的差距 | 每臂 n | 總 launch | 時間 |
|---|---|---|---|
| 3%（本 repo 門檻） | 674 | ~1350 | **≈ 35 h** |
| 5% | 242 | ~485 | **≈ 12.7 h** |
| 10% | 61 | ~121 | **≈ 3.2 h** |
| 20% | 15 | ~30 | ≈ 47 min |

⇒ **一夜的量測能量約等於「一條軸、一個效應 ≥10%」**。18 條軸是三週不間斷的量測。
所以真正的槓桿不是「加更多 knob」，而是把 σ 從 0.28 降到 ~0.10（更長 generation、更多 reps、每側多一次 ref 壓插值噪音）——
因為時間是 σ²。

---

## 6. 建議的下一步（按價值排序，全部只需加 mesh 的一行）

1. **`CGC_MMV_FUSE` 兩臂**（`=off` vs `=1`，`CGC_GLU_FUSED_DOWN` 維持 1）—— 兩臂 · 4 launch · ~7 min。
   它同時回答「為什麼 §8.113 的 +6.5% 一直沒出現」和「這條融合路到底值多少」。自稱 bit-identical ⇒ 不污染 bit-identity。
2. **`CGC_MMV_NSG` 重掃**（`{2,4,8,16}`）針對新射程 —— 4 臂 · 6 launch · ~10 min 起。若 nsg 對 IQ4_XS 有 5%+，
   這是唯一可以直接落在 81% dense 位元組上的 tile 維度。
3. **`CGC_PREV_TOKEN_PREFETCH=1` 兩臂** —— 但先加一個「prefetch 真的有 queue」的計數（F2/F5 的坑）。

（以上三條我都只改 `GRIDS`/`NAME_OF_AXIS` 各一行即可，`--plan` 會用 `check_reachability` 先證明軸真的會動。）

---

## 附錄：核對用的命令（全部零 GPU）

```sh
python3 scripts/check/shape_knob_search.py --selftest                 # 27/27
python3 scripts/check/shape_knob_search.py --plan --grid budget       # 印每格的 resolved env diff
python3 scripts/check/prod_profile.py --emit-spec                     # 交付格實際 pin 的 30 個 env
```
