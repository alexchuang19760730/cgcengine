# 量測方式審計 — `2eb84420d`（看門狗 / 雙輪 arm 量具 / 斷言協議）

**審計對象**：`docs/ASSERTION_PROTOCOL_2026-09-25.md`、`scripts/check/{lane_watchdog.py,
watchdog_daemon.sh, arm_two_pass.py, arm_report_html.py, harness.py}`
（6 檔 1946 行，2026-09-25 09:33）。
**基準**：`scripts/check/commit_bench.py` + `docs/PROD_NEW_TEST_CARD_2026-09-24.md`。
**口徑**：pool 大小不放進判準（可商議）；**量測腳本本身按標準卡逐項對**。

---

## 1. 對的地方（先講，因為這些是改進）

| 項 | 說明 |
|---|---|
| **不複製量測邏輯** | `arm_two_pass.py:320` 直接 delegate 給 `llama_bench_matrix.py` ⇒ **與本線同一支 runner**，不會兩套口徑漂移 ✅ |
| **生產 PIN** | `PIN_ENV` 把 `CGC_EXPERT_CACHE_BYTES=8589934592`、`CGC_SPAC`、`CGC_PREFILL_STREAM`、`WORKERS=8` 等 15 鍵釘死；`PIN_SCALARS` 釘 `BATCH/UBATCH=5632`、`CTX=8192`、`LOAD_MODE=none`、`NGL=99` ✅ |
| **MTP 必須缺席** | `PIN_ABSENT = ["CGC_SERVER_MTP"]` ⇒ 把「MTP off」變成可機檢的斷言 ✅ |
| **build 指紋** | `build_fingerprint()` 記 5 檔 md5（含 `libggml-base.0.dylib`、`libggml-metal.0.dylib`、server impl）⇒ **比本線只記 libllama 強** ✅ |
| **同時報 pp + tg** | `_row_metric(..., "pp"/"tg")`，符合 09-24 的量測紀律 ✅ |
| **狀態標注** | `sys_before`/`sys_after` 快照 + matrix 的 thermal ⇒ 數字帶得出 thermal/swap ✅ |
| **clean / instrumented 兩輪** | 乾淨輪強制關全部已知儀器（`build_pass_env`）⇒ 儀器不污染測速 ✅ |

---

## 2. 逐項與標準卡對比

| 項 | 標準卡 / `commit_bench.py` | 新 harness | 判定 |
|---|---|---|---|
| runner | `llama_bench_matrix.py` | 同 | ✅ |
| profile | prod-new | prod-new | ✅ |
| `-p` / `-n` / `-d` | 2048 / 128 / 512 | 2048 / 128 / 512 | ✅ |
| `--warm-skip` / `--ctx-size` | 64 / 0 | 64 / 0 | ✅ |
| batch | profile 推導 5632 | PIN 5632 | ✅ |
| MTP | off | `PIN_ABSENT` | ✅ |
| pool | 8 GiB | PIN 8 GiB | ✅ |
| **`-r`（reps）** | **3** | **1**（`_BENCH_DEFAULTS` 與 `arm_two_pass --reps` 皆 1） | ❌ **見 §3B** |
| 兩輪之間冷卻 | — | `--cool-s` **0** | ❌ **見 §3D** |
| 起跑 swap 閘門 | `budget_gate`（超訂 4838 MiB 就拒） | **swap ≤ 1024 MiB** | ⚠ **見 §3C** |

---

## 3. 四個缺陷

### A. G1 的 `no_watchdog` 只認**舊**看門狗的名字 —— 它會放行一個已上膛的殺手
`arm_two_pass.py:201`：`watchdog = _procs_matching(["auto_bench_watchdog"])`。
**實際在跑的是 `scripts/check/watchdog_daemon.sh`（PID 39559，`bash .../watchdog_daemon.sh`）
→ `lane_watchdog.py --kill --quarantine`，每 600 s 一 tick。**
⇒ `pgrep` 用的是 `llama|auto_bench_watchdog` 這個模式，**抓不到 `watchdog_daemon`**
⇒ **G1 的 `no_watchdog` 會 PASS，然後那一槍由看門狗在 tick 上開**。
這不是假想：**本線 10:21:30 發射的 arm 死於 10:22:19，而 tick 正是 10:22:18**。
⇒ **同一份 commit 裡的兩個工具互不知道對方存在。**

### B. 預設 `--reps 1` 讓 `judge_artifact` 的散度判據**結構性失效**
`lane_watchdog.py:212-224`：
```python
if avg and sams and len(sams) >= 2:
    rep_spread[kind] = (max(sams) - min(sams)) / avg * 100
...
spread_high = spread_tg is not None and spread_tg > SPREAD_UNRELIABLE   # 12.0
unreliable  = thermal_bad or spread_high
```
`-r 1` ⇒ `samples_ts` 只有 1 個 ⇒ `rep_spread` **無定義** ⇒ `spread_high=False`
⇒ **「量具被壓垮 ⇒ 數字作廢」這條最強的可靠性判據，在預設形狀下永遠不會觸發。**
而它自己的註解寫著：`SPREAD_UNRELIABLE = 12.0  # 生產 cell 4.3%、壞 cell 38%`。
⇒ **按預設跑，harness 報出的數字不可能被判「量具被壓垮」**；偏偏它同時又主打
「swap 高不廢數字」⇒ 兩個失效模式疊在一起，輸出的是「什麼都不會紅」的綠燈。
（本線的配對一律用 `-r 3`；`-r 3` 才有 3 個 `samples_ts`。）

### C. 同一 commit 內**兩把尺對 swap 不一致**
- `lane_watchdog.judge_artifact` 的 docstring（`:205-206`）：
  > 「數字作廢與否由「thermal + 量具 rep 散度」裁決，**不由 swap**；
  > swap 高只標 stressed（環境承壓、需結構修復），量具穩時數字仍可引用。」
- `arm_two_pass` 的 G1（`--max-swap-mb`，`:545` **預設 1024**）把 **swap > 1024 當拒跑**。

⇒ 一個說 swap 無關，一個說 swap>1 GiB 就不准開始。
**而本線交付口徑的實測起跑 swap 是 2180–2496 MiB**（`Backup/mw_ab/mw_ctrl.json`、
`nf_fill` 的 `啟動 swap=2496`）⇒ 按 G1 全該拒跑，按 `judge_artifact` 全屬可引用
（thermal NOMINAL、rep 散度 3.2%／4.3%）。**兩把尺必須統一。**

### D. 兩輪之間 `--cool-s 0`：第二輪繼承第一輪的熱
本項目自己量過：**連發會繼承熱量**（§EN-1323 四輪連發全掛；`r5`/`r6` 隔 15 分 20 秒才乾淨）。
`clean → instrumented` 預設 0 秒 ⇒ **第二輪系統性偏熱/偏冷**，而兩輪之差正好是它判決依據之一。
⇒ 標準做法應是 `--cool-s 420`（實測過的最小冷卻），或至少在報告裡把兩輪間隔當作已知偏差列出來。

### E.（附）kill 不區分「有意的診斷臂」
`judge_proc`（`:197`）的 kill 條件是 `dangerous_now`（swap>3072 或 free<150），**與臂的性質無關**。
診斷臂（如 `CGC_EB_NOFILL`）的設計本來就讓 swap 長高（權重不進池）⇒
**看門狗結構上會殺診斷臂、並把那個資料點毀掉**（10:21 那支正是 nofill 臂）。

---

## 4. 對 commit 自報錨點的意義

commit 訊息寫 `prefill=271.44t/s decode=11.55t/s mtp=off commit_bench prod-new p2048 n128 d512 r3`。
- 這組數字**與本線的交付讀數同量級**（11.79／11.03／12.195），**互相印證** ✅
- 但它標的是 `r3`，而 **harness 的預設是 r1** ⇒ **錨點不是 harness 產出的**，是 `commit_bench` 的
  （`commit_bench.py --dry-run` 確認：`--reps 3`）。⇒ **harness 沒跑出過這組數字。**
- ⚠ `/tmp/arm_two_pass`／`/tmp/harness_bench`／`/tmp/harness_verify` **目前都不存在**
  ⇒ 沒有 harness 跑過的痕跡（`/tmp` 會被清，所以**不能據此斷言「從未跑過」**）。

---

## 5. 建議修法（全部低成本）

1. **G1 的看門狗偵測改成問看門狗自己**：把 `"watchdog_daemon"`、`"lane_watchdog"` 加進
   `_procs_matching` 的模式，或直接跑 `lane_watchdog.py --report` 並在 `level == "kill"` 時拒跑。
   （現行做法只會偵測到那支**沒在跑**的 `auto_bench_watchdog`。）
2. **`_BENCH_DEFAULTS.reps` 改 3**，並讓 `judge_artifact` 把 `len(sams) < 2` 當成
   **「散度不可測 ⇒ 不得判可引用」**，而不是靜默通過。
3. **統一 swap 口徑**：G1 用 `LAUNCH_SWAP_KILL=2048`（與 `judge_artifact` 同一條線），
   或把 `judge_artifact` 的 stressed 也升成不可引用 —— 但不能一個 1024、一個無關。
4. **`--cool-s` 預設 420**（實測最小冷卻），或把它寫進報告的已知偏差。
5. **讓看門狗認得自己人**：harness 起跑時把自己的 PID 樹寫進一個 marker 檔，
   `lane_watchdog` 對「已宣告的診斷臂」只 warn 不 kill。
