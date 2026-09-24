# ABBA 測量協議（decode A/B 的強制口徑）— 2026-09-23

> **為什麼存在**（§EN-473 實測）：同一顆錨點 binary 在「GPU 閒 10 分鐘」後讀 **12.37 t/s**、
> 在「只冷卻 3 分鐘」後讀 **9.9–10.5 t/s**（samples 11.07/8.96/17.09 的 12.37 是被 17.09
> outlier 拉高的均值）。**本機單一配置的 launch-to-launch 波動 ±20%，大於多數 A/B 想測的
> 差異。** 任何不遵守本協議的 A/B 結論（「快了/慢了 X%」）都在噪音裡打轉，不可引用。

**適用範圍**：所有 decode / prefill 的速度 A/B、所有「某個 knob / commit 快了 X%」的宣稱。
**不適用**：bit-exact 閘門（另有 oracle gate）、非速度的儀器產物。

---

## 0. 一句話

```
交錯（ABBA 或 bracketed）＋ 每 launch 前深度冷卻 ≥300s ＋ 交付用中位數（配對比率的中位）＋
log-space 校正 ＋ 窗口守門 ＋ build 指紋。少了任何一項，數字不進報告。
```

---

## 1. 交錯順序（把 drift 變成配對內可消的項）

- **兩個 arm**：`A B A B ...`（ABBA 旋轉；`ab_interleave.py` 就是這個）。
- **參考 + 測試**（drift 形狀未知時）：bracketed `R T R T R ...`，對比用
  `T − (R_before + R_after)/2`（`bracketed_ab.py --drift geometric`；線性漂移在括號內精確抵消，
  不需要估計）。
- **禁止**：先跑完 A 全部 reps 再跑 B（順序偏誤），或兩個 arm 各一次取「兩個 median 的比」
  （把 launch 間 drift 混進差異）。

## 2. 深度冷卻閘門（新增，§EN-473）

- 每個 launch 前，**GPU 空閒 ≥ 300 秒**（`--min-idle-s 300`，可配）。
- 空閒起點 = 上次測量完成（`ab_interleave.py` 用 json mtime；其他 harness 用同口徑的時間戳）。
- **NOMINAL label 只是必要條件，不是充分條件**：這台 M4 Air 無風扇，thermal 在 NOMINAL 之下
  是連續的。只認 label 會把「深冷卻 12.37 vs 淺冷卻 9.9」都標成 NOMINAL。
- 記錄實際 gap_s 進每一行（bracketed_ab 已做），供事後檢查。
- 首跑（無上次測量記錄）：假設冷機，但註記 `assuming cold start`。

## 3. 中位數口徑（交付數字用中位，不用均值）

- **交付數字**：rep 中位數；兩 arm 用**配對 per-rep 比率的中位**（`ab_interleave.report` 的
  `paired median`），不是「ratio of medians」。
- **均值只能當參考**，且**必須列出 samples**——12.37 的 samples `[11.07, 8.96, 17.09]`
  （stddev 4.22）一望即知被 outlier 拉高；12.57 錨點（±2.26，stddev 佔 18%）同類。
- 任何「12.5+ t/s」的宣稱，先問：是 samples 的中位還是均值？samples 長什麼樣？

## 4. log-space 校正

- throughput 被機器狀態**乘**在 t/s 上：`log y = log θ_c + log ρ(t)`。均值/插值必須在 log 空間
  做（`ORDER_DRIFT_CORRECTION_2026-09-22.md` 已實測：同一行用 arith vs geom 校正，
  頭條數字差 7.9pp）。
- `bracketed_ab.py` 的 `adjusted()`（OLS on log tps × [arm, order]）與 `bracketed()` 是官方
  估計器；`MAX_DRIFT_PCT = 10%`：參考兩側差 >10% 的行 = UNANCHORED，**不算測量**。

## 5. 窗口守門

- launch 前檢查**沒有別人的** `llama-bench` / `llama-server` 進程（`ab_interleave.others_measuring`；
  污染是硬閘門 rc=2，不是 covariate）。
- 多 session 並行時：`.bench_lock/` 鎖定目錄 + 進程檢查雙保險。
- swap 水位寫進每行（`memory_watch.sh` / bracketed_ab 的 swap_mb）；**swap 是追蹤變數，
  不是事後推鍋對象**。

## 6. build 指紋

- 每行帶 md5，**至少含**：`llama-server`（或 launcher）、`libllama-server-impl.dylib`、
  `libggml-metal`、`libllama`（§6f 教訓：server 邏輯在 impl dylib，只 hash launcher 是盲的）。
- `answer_md5_set` 必帶：一個宣稱「改了行為」的 knob 如果 md5 沒變 = flag 沒到達進程 =
  speedup 是假的。

## 7. 判定標準

- **差異 < 3%：不引用**（可寫「低於 3% 閘門」）。
- 差異 ≥ 3% 但 reps < 3：只算「待確認」，不算結論。
- 樣本波動 > 差異時（本機常態）：報告中位 + samples + 條件（gap_s / thermal / swap），
  不給單點結論。
- 「復現 12.57」類的單均值宣稱：先做 §3 檢查——它可能根本不是穩定基線。

---

## 工具對照

| 要做什麼 | 用哪個 |
|---|---|
| 兩 arm 交錯（A,B,A,B）+ 配對比率中位 | `scripts/check/ab_interleave.py`（已加 `--min-idle-s` / 窗口守門） |
| 參考夾測試（R T R T R）+ log 校正 | `scripts/check/bracketed_ab.py` |
| 模擬/估計器選擇 | `bracketed_ab.py --simulate / --analyze` |
| 順序/漂移校正原理 | `docs/ORDER_DRIFT_CORRECTION_2026-09-22.md` |
| swap/記憶體追蹤 | `scripts/check/memory_watch.sh` → `.workbuddy/memory/swap_log.tsv` |

---

## 引用

兩個 agent 的所有速度 A/B 產物，文件頭必須寫「按 ABBA_MEASUREMENT_PROTOCOL 測量」，
並附：交錯模式、min-idle、reps、samples、build 指紋。缺引用 = 數字不進任何報告。
