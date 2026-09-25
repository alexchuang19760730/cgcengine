# miss_axis 實跑結果（2026-09-24 20:2x）

> 工具 `scripts/check/miss_axis.py`（selftest 26/26）。raw trace：
> `Backup/miss_axis/llama_bench_prod-new_p2048_n128_d512_r3.stderr.log`（`Backup/` 被 gitignore）
> 重現：`python3 scripts/check/miss_axis.py --run --p0 1 --profile prod-new --reps 3 \
>   --gen 128 --depth 512 --prompt 2048 --ubatch 5632 --workdir Backup/miss_axis`
> 離線重算：`python3 scripts/check/miss_axis.py --log <上面的 .stderr.log>`

## 0. 這趟是怎麼跑起來的（三個坑，各花了十趟）

| 坑 | 症狀 | 處置 |
|---|---|---|
| **`--ctx-size 8192`** | prefill/decode 收尾 `kIOGPUCommandBufferCallbackErrorOutOfMemory`，`rc=-6` | **真兇**。標準測試卡 §2 的指令沒有這個 flag、§7 寫 `--ctx-size 0`。只差這一個 flag：無它存活 6 steps、有它當場 rc=-6 |
| **缺 `LLAMA_EXPERT_CACHE_ALLOW_NGL`** | `pool_cap_slots=0` / `routable=0` ⇒ 13.0 GB 模型整包上 Metal（上限 11453 MB）⇒ OOM | 只對「直接跑 llama-bench」成立；`run_server.sh` 自己有 export，harness 路徑不受影響。已補進 `REQUIRED_ENV` |
| **parse 錯來源** | harness 用 `capture_output=True` 吃掉子行程 stderr，只寫進 `{workdir}/llama_bench_*.stderr.log`，自己 stderr 只有一行摘要 ⇒ 永遠 `steps=0` | 改成 glob 出子行程 log 再 parse |

⚠ 兩個坑同時存在時會互相掩蓋（看起來都像「跑不起來」）。
⚠ 「去掉 `CGC_GPU_TIMING` 仍 OOM」「`--ctx-size` 8192 與 0 都 OOM」「`-ub` 2048 也 OOM」這三次隔離
**全部無效**，因為那時 `pool_cap_slots=0`——**OOM 的第一個檢查點是印 `pool_cap_slots` 去跟 `Backup/`
的歷史分佈對**（主流 `143`，`0` 在整個 Backup 只出現 2 次）。

## 1. 量到的東西（48 decode steps、96 hook rows，ntok=1）

```
baseline  72.52 ms/step -> 13.79 t/s   (cb 4.30 ms, gap 15.08 ms)
```

### Q1 「正常操作的固定成本 ≈ 0，cb 跟窗口幾乎 100% 是 cache miss 造成的」→ **FAIL（推翻 09-20）**

| 項 | floor（m=0 的截距） | per-miss 斜率 | r² | miss 可解釋比例 |
|---|---:|---:|---:|---:|
| `cb` | **0.077 ms** | 0.110 ms | 0.17 | 0.66 |
| `gap` | **0.350 ms** | 0.058 ms | **0.05** | **0.18** |

- `cb` 的 floor ≈ 0 —— 這一半**支持** 09-20（`m=0 的層只付 0.01 ms`）。
- **`gap` 的 floor = 0.350 ms/段**，×40 段 ≈ 14 ms/step ≈ 就是 `gap` 15.08 ms 的主體，
  而 **miss 只能解釋它的 18%（r² = 0.05）** ⇒ **段間空隙主要是結構性的，不是 miss 造成的。**
  這跟 09-20「`idle_host` ∝ miss、miss-free 每層 ≤0.2–0.3 ms」的結論**相反**。

### Q2 / Q3 兩個上界 —— **不相等（使用者預期不成立）**

| | 做法 | 結果 | 增益 |
|---|---|---:|---:|
| **Q2** | 消掉 `cb` + 串列空隙（＝單段提交 S1） | 53.14 ms → **18.82 t/s** | **+26.7%** |
| **Q3** | 只把 cache miss 消到 0，保留結構性成本 | 65.11 ms → **15.36 t/s** | +10.2% |

```
Q2 vs Q3   DIVERGENT  (rel delta 0.251, tol 0.05)
structural (non-miss) part of the segmented dispatcher: 11.97 ms/step
```

**兩者差 25.1%，差額就是結構性的 11.97 ms/step。**
⇒ 單段提交（S1）**不只是「避開 miss」**：它另外有 ~12 ms/step（≈ +16%）的獨立收益，
是「41 段分別提交」這個形狀本身的成本，跟池子命中率無關。

## 2. 對 S1 的意義

- B 臂診斷值 17.7–18 t/s（`docs/B_PLAN_PREFLIGHT_2026-09-24.md` §7.2）與這裡的 **18.82** 同帶 ⇒ 互相印證。
- 只做「消 miss」（例如加大池、更好的替換策略）天花板是 **15.36**；
  做單段提交天花板是 **18.82**。**要碰 18 就必須做形狀，不能只做命中率。**
- ⚠ 但 S1 要落地仍卡在 miss 處理（錯層重算），那一步的淨值見 `§7.4`（10–14 t/s 髒環境 / 14.3–16.1 乾淨且可後台 fill）。

## 3. ⚠ 未閉合：兩個儀器對 miss 成本的讀數差 9.5×

```
hook split us/call: pre 4.4  ensure 128.3  drain 0.2  tail 0.3   (n=15360, 320 calls/step)
ensure 41.06 ms/step   vs   cb 4.30 ms/step   -> ratio 9.54
```

`CGC_HOOK_SPLIT` 的 `ensure`（demand fill，直接量、不靠回歸）是 **41 ms/step**，
而 `CGC-DECPROF` 的 `cb` 欄只有 **4.30 ms/step**。兩者差 9.5 倍 ⇒
**`cb` 欄不是 hook 的全部時間**（很可能 fill 的時間被記到 `wait`/`gap` 裡）。
⇒ Q2/Q3 用的是 `cb`+`gap` 的口徑，若改用 `ensure` 的口徑，結論的數字會動，**方向不變**（結構性項仍存在），
但**「Q2 vs Q3 差多少」這個數字在口徑閉合前不可當交付數字**。

其他注意：
- 這趟全程帶儀器（DECPROF_ALL / HOOK_SPLIT）⇒ **13.79 t/s 不可當交付數字**；
  卡錨點是 11.90 / 11.37（`docs/PROD_NEW_TEST_CARD_2026-09-24.md` §6）。
- `-p 2048` 的 prefill 圖也會吐 `CGC-DECPROF step` 行（`ntok=2048`），parser 已按 `ntok` 濾掉。
- swap 在跑之前是 3.9 GB（非乾淨窗口）⇒ 建議在乾淨窗口複跑一次確認。
