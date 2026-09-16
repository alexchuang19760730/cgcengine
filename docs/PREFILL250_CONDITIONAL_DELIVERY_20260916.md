# prefill 250 的**可量測**交付條件

專案：`/Users/alexchuang/Documents/flashkv-devserver`
模型：`Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf`（`CGC_SERVER_PROFILE=prefill250`，出廠預設）
機器：MacBook Air M4 16GB（`Mac16,12`，**無風扇**）
日期：2026-09-16
權威版本：`docs/PREFILL250_THERMAL_TRANSIENT_20260916.html` §11.15（本文是它的可操作摘要）

---

## 0. 一頁結論

**條件**：在發射驗收臂之前，讀一次

```bash
notifyutil -g com.apple.system.thermalpressurelevel
```

**必須是 `0`（Nominal）。** 不需要 root、不需要 `powermetrics`、不需要擷取檔，單次讀取約 11 ms。

**判準的實測分離度（request 級，零重疊）**

| 發射時等級 | 樣本（t/s） | 是否 ≥250 |
|---|---|---|
| `0` Nominal | 266.81、269.35、271.64、257.41、253.42、262.64 | **6/6 全部 ≥250**（最低 253.42） |
| `1` Moderate / `2` Heavy | 211.65、196.42、189.49、169.09、167.91、155.23、152.50、150.27、150.25、144.36、143.89、142.14、141.75、129.02、127.43、125.46、124.33、120.74、120.60、117.53、104.88 | **0/21 ≥250**（最高 211.65） |

間隙：`253.42 − 211.65 = 41.77 t/s`，**沒有重疊**。

所以這不是相關，是一條**決策規則**：讀到 0 就可以交付，讀到非 0 就不行。

---

## 1. 儀器：為什麼是這個鍵

`com.apple.system.thermalpressurelevel` 是 notify(3) 的 key，**就是 `powermetrics` 的 thermal sampler 用來印 `Current pressure level` 的那一條**。刻度直接用 IOKit 常數：

| 值 | 意義 |
|---|---|
| 0 | Nominal |
| 1 | Moderate |
| 2 | Heavy |
| 3 | Trapping |
| 4 | Sleeping |

成本：20 次讀取 0.225 s ⇒ **約 11 ms/次**，可以 2 Hz 取樣而不擾動被測量本身。

### 不是這個儀器（都實測過）

| 候選 | 結論 |
|---|---|
| `powermetrics --samplers gpu_power,thermal` | 需要 root。本機 `sudo` 對 agent 是硬封鎖（`operation not permitted`，關閉沙箱亦同）⇒ 只能由人啟動，不能當每臂的閘門。 |
| `pmset -g therm` | 只回 `No thermal warning level has been recorded`。無資訊。 |
| `sysctl -a \| grep -i thermal` | 沒有相關鍵。 |
| `ioreg -c IOAccelerator` | 有 `Device Utilization %`（即時值），但**沒有時脈**；`GPU Performance States` 只暴露 IOReport 的 channel id 與 unit，**不是讀數**。 |
| `NSProcessInfo.thermalState`（JXA，非 root） | **不具區辨力，已否證。** 367 個樣本、跨越 ABBA 滿載與 4 分鐘閒置，**全部讀到 `1/fair`**；同一時間 `notifyutil` 從 2 → 0。⇒ 不要用它當閘門。 |

**教訓**：「沒有非 root 儀器可讀」這句話是**關於某個工具的推論**，被寫成了**關於系統的事實**。儀器一直在那裡，只是搜尋停在「`powermetrics` 要 root」就結束了。

---

## 2. 條件是「發射時」，不是「全程」

兩次正向臂的行為不同，這件事必須說清楚：

**臂 A（15:31:04，乾淨）** — 等級全程 0：

```
before launch = 0/NOMINAL
before req1   = 0/NOMINAL   req1 266.81 t/s
after  req1   = 0/NOMINAL
before req2   = 0/NOMINAL   req2 269.35 t/s
after  req2   = 0/NOMINAL
before req3   = 0/NOMINAL   req3 271.64 t/s
after  req3   = 1/MODERATE   <- 三個 prefill 都跑完之後才翻
```

**臂 B（15:18:46）** — 等級在臂中途上升：

```
15:18:46 launch      0/NOMINAL
15:19:13 (req1 之後) 1/Moderate
15:19:24 (req2 期間) 2/Heavy
req1 257.41 / req2 253.42 / req3 262.64   <- 仍然全部 ≥250
```

⇒ **臂中途的等級上升不會撤回該臂的數字。** 所以閘門讀的是**發射前那一刻**，而不是要求全程為 0。機制上合理的解釋是 governor 的反應對熱壓訊號有落後（該臂的熱累積尚未追上讀數），但**這是假設**：HTTP 路徑上沒有 per-request 的 DVFS 證據，所以不寫成機制結論。

`run_req2_retest.sh` 仍然把 `before launch` 與每個 request 邊界的讀數都印出來，因為它們是**證據**（顯示該臂處在哪個過程），而閘門只有一個：發射時。

---

## 3. powermetrics 的獨立佐證（block 級）

兩次擷取（`prefill_certifiability.py` / llama-bench `-p 2048`），配對的 pp t/s：

| 擷取 | block | 熱壓 | 有效時脈 | pp t/s | ≥250 |
|---|---|---|---|---|---|
| 09:08（b=6144） | cold | Nominal ×55 | 1400 MHz | 276.25 | ✓ |
| 09:08 | hot | Heavy48+Mod20 | 1082 MHz | 227.27 | ✗ |
| 09:08 | hot2 | Heavy ×104 | 700 MHz | 151.28 | ✗ |
| 14:55（b=5632） | cold | Nominal ×52 | 1387 MHz | 300.43 | ✓ |
| 14:55 | hot | Heavy ×86 | 843 MHz | 182.39 | ✗ |
| 14:55 | hot2 | Heavy ×109 | 649 MHz | 145.79 | ✗ |

**Nominal 2/2 → ≥250；非 Nominal 4/4 → <250。** 與 HTTP 路徑的結論一致，且是兩個獨立的入口（llama-bench vs HTTP）。

連續形式（時脈 → 吞吐）也吻合：以 `t(ms/token) = a + b/f_eff` 擬合，250 t/s 對應的有效時脈是 **1154–1227 MHz**（三組擬合：09:08 → 1227、14:55 → 1154、合併 → 1195）。殘差 ±3.8 t/s（組內），合併時 ±15.6 t/s（因為兩次擷取的 `n_batch` 不同：6144 vs 5632）。

---

## 4. 怎麼跑

### 4.1 閘門（交付樣本的唯一入口）

```bash
cd /Users/alexchuang/Documents/flashkv-devserver

# 等閘門打開，然後跑一個正向臂 + 一個負向對照
ARMS=2 OUTDIR=Backup/cgc_logs/thermal_gate bash Backup/run_thermal_gate.sh
```

- 條件不成立就**不跑臂，exit 3**（fail closed）。要刻意產熱態樣本才用 `GATE_FORCE=1`。
- `MAX_WAIT`（預設 1800 s）是等待上限。實測：滿載停止後等級約 **47 秒**就回到 0 ⇒ 等待通常很短。
- 腳本同時起 2 Hz 的外掛探針（`Backup/thermal_pressure_probe.sh`），所以臂中途的等級變化事後看得見。

### 4.2 驗收臂本身帶證據

```bash
IDLE_BEFORE=0 OUTDIR=Backup/cgc_logs/x bash Backup/run_req2_retest.sh
```

報告會出現：

```
# [thermal pressure] before launch = 0/NOMINAL   (notifyutil -g com.apple.system.thermalpressurelevel)
  [thermal pressure] before req1 = 0/NOMINAL
...
  --- thermal condition for this arm (in-band) ---
    high-water : 0/NOMINAL -- condition SATISFIED across this arm
    -> a 250-class t/s reading from this arm is licensed by whitepaper 11.15.
```

讀不到時印 `?/UNREADABLE`，**不會**靜默當成 0（「讀不到」與「條件成立」同形是這個變數最糟的混淆）。

### 4.3 DVFS 的權威版本（有 root 時）

```bash
sudo scripts/check/powermetrics_gpu_freq.sh
python3 scripts/check/powermetrics_gpu_freq_parse.py Backup/cgc_logs/powermetrics_prefill_*.log
```

這是升級版的佐證（會給時脈與 DVFS 駐留分佈），但**不是閘門**：閘門要能在每一臂前跑，且不需要人。

---

## 5. 不能說的話

1. **不能說條件是「靜置 ≥150 s」。** 已推翻：安靜 18 s 的臂 req1 = 289.86，安靜 34 s 的臂只有 166.95。安靜長度本身不預測吞吐。
2. **不能說「安靜夠久 ⟹ 等級一定回到 0」。** 等級回到 0 是**讀出來的**，不是推出來的。反例就在 §2 臂 B：發射時是 0，但該臂中途就翻了。
3. **不能說 `swap` 高低是條件。** 反例順序相反（289.86 t/s @ `used 4975.06M`，188.35 t/s @ `4549.69M`，13:51 慢臂 @ `5066.06M`），而且它在安靜期間自己從 5066.06M 回到 2743.00M ⇒ 它是果。
4. **不能說 `NSProcessInfo.thermalState` 可以用。** 367/367 讀到 `fair`。
5. **不能說「250 隨時可重現」。** 可交付的是「**讀到 0 的那一臂**，其 req1–req3 全部 ≥250」——本輪 2/2 臂成立（6/6 request）。母數小，如實計數。
6. **不能說閘門驗證了機制。** 它驗證的是**判準**。時脈 → 吞吐的定量關係來自 `powermetrics`（§3），但 HTTP 路徑上沒有 per-request 的 DVFS 讀數。

---

## 6. 副產物：(c) 那 2.25× 已歸因 —— 用**否證**的方式

§11.13 留下一個未歸因的 2.25×（12:44 產物 `268.09/273.55/256.61` vs 最終產物 `118.30–154.77`，同 A11 指紋），唯一內容差異是 `libggml-metal`。

**做了 ABBA 交錯**（`Backup/run_lib_ab.sh`，檔案互換 + 一臂暖機丟棄，共 5 臂）：

| 臂 | 庫 | md5 | req1 | req2 | req3 |
|---|---|---|---|---|---|
| 0（暖機，丟棄） | A | `dd4d1bc4` | 248.18 | 201.94 | 210.46 |
| 1 | B | `f2d1c961` | 124.33 | 129.02 | 152.50 |
| 2 | A | `dd4d1bc4` | 117.53 | 150.27 | 167.91 |
| 3 | A | `dd4d1bc4` | 125.46 | 169.09 | 141.75 |
| 4 | B | `f2d1c961` | 120.60 | 144.36 | 155.23 |

- `req1`：A=121.50 / B=122.47（**+0.8%**）
- `req2`：A=159.68 / B=136.69（−14.4%）
- `req3`：A=154.83 / B=153.87（**−0.6%**）

**組間差無一致方向，且組內散佈大於組間差**（A 的 span = 51.56 t/s，B 的 span = 34.63，組間均差 = 7.66 ⇒ 散佈是組間差的 **6.7 倍**）。

⇒ **庫的假設被否證。** 剩下的解釋是機器狀態，而 §0 的判準提供了第二條獨立線索：`268.09/273.55/256.61` 落在 Nominal 帶（≥253.42），`118.30–154.77` 落在 Heavy 帶（≤211.65），**兩組分屬分離的兩個帶**。這是**一致的歸因**，不是直接觀測——那些臂當時沒有等級讀數（儀器是今天 15:2x 才找到的），如實標記。

**附帶驗證**：`load_mode=none` 下兩版庫統計上無差別，正是設計的預測（`_Pool` buft 只在 `use_mmap` 時被選中），所以那個改動對出廠路徑是中性的。

---

## 7. 復現

```bash
cd /Users/alexchuang/Documents/flashkv-devserver

# 檔案互換前的 ABI 安全檢查（install name / deps / imports / 簽章）
otool -D  Backup/pre_flagalign_20260916/libggml-metal.0.19.0.dylib
nm -u     Backup/pre_flagalign_20260916/libggml-metal.0.19.0.dylib | wc -l   # 135，與最終版相同

# 判準的證據：兩組分離
grep -E 'before launch|pp_t/s' Backup/cgc_logs/thermal_gate2/req2retest_20260916_15310{4,7}.txt
grep -E 'before launch|pp_t/s' Backup/cgc_logs/thermal_gate/req2retest_20260916_15184{6}.txt \
                                                   Backup/cgc_logs/thermal_gate/req2retest_20260916_1519{49}.txt

# ABBA
cat Backup/cgc_logs/libab/summary_20260916_150641.tsv

# 閘門
ARMS=2 OUTDIR=Backup/cgc_logs/thermal_gate bash Backup/run_thermal_gate.sh
```

證據檔：
`Backup/cgc_logs/thermal_gate2/{req2retest_20260916_153104.txt,req2retest_20260916_153207.txt,pressure_20260916_153102.tsv,gate_20260916_153102.txt,summary_20260916_153102.tsv}`、
`Backup/cgc_logs/thermal_gate/{req2retest_20260916_151846.txt,…_151949.txt,…_152126.txt,pressure.tsv}`、
`Backup/cgc_logs/libab/summary_20260916_150641.tsv`、
`Backup/cgc_logs/powermetrics_parse_145506.json`、`Backup/cgc_logs/powermetrics_prefill_20260916_090808.parse.txt`。
