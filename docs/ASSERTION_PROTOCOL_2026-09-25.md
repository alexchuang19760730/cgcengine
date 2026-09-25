# 知識斷言與勘誤協議（Assertion & Erratum Protocol）

> 適用：`.workbuddy/memory/`、各 `SKILL.md`、技術白皮書、`docs/` 研究記錄。
> 生效：2026-09-25。所有 agent（含看門人）讀寫上述文件時遵守。

## 0. 為什麼需要這個協議

memory / SKILL.md / 白皮書是後續 agent 的**權威依據**。一條錯誤斷言（假設當結論、數字標反、
被證偽的方向沒標記）會被後續反覆引用、**錯誤沿著依賴鏈傳播放大**。這兩天的具體教訓：

- 把「讀代碼看到 buffer」當成「buffer 有數據」（prebind 的 draft_prefetch_ids 從未被填）；
- 把樂觀預估當成量測結果（25 t/s、+27% 等先後被證偽）；
- 同一數字多處複製、改一處後其他過期（4G/8G、cb 各種口徑互相矛盾）；
- 「unavoidable / 必然 / 100%」類結論在無數據支撐下被寫死、誤導後續決策。

結論：**不能靠「發現了再悄悄改掉」**。需要寫入分級、證偽留痕、勘誤閉環、單一事實源。

## 1. 斷言分級（寫入時必標）

每條可被引用的結論性陳述，開頭帶標籤；不帶標籤的結論視為【假設】、不得當依據：

| 標籤 | 含義 | 必須附帶 |
|---|---|---|
| 【實測】 | 來自具體一次 run / 產物 | 產物路徑 + 量測時間 + 口徑（shape、warm-skip、thermal、swap） |
| 【推斷】 | 第一性原理 / 邏輯 / 代碼必然 | 前提假設 + 代碼或算術依據（檔:行 / 公式） |
| 【假設】 | 待驗證、猜測 | 驗證方法（怎麼證實/證偽） |
| 【已證偽】 | 曾經主張、已被推翻 | 推翻證據 + 日期 + 指向正確結論 |

範例：
- 【實測】乾淨基線 decode = 11.49 t/s（harness bench prod-new r3、swap launch 0、thermal NOMINAL、
  產物 `/tmp/harness_bench/...json`，2026-09-24）。
- 【推斷】讀數據與解壓縮串行（12.3+18.4≈總時間、前提是計時口徑一致）。
- 【假設】殘差流局部性使 ρ 重合率 >0.4（待一次 GPU run 定價）。

## 2. 證偽與勘誤：不刪除、留痕

一條斷言被新數據/代碼推翻時：

1. **不刪除原文**，在原處加 `【已證偽 YYYY-MM-DD】` 標記；
2. 寫明推翻證據（產物路徑 / 數據 / 代碼行）與正確結論；
3. 在當日 `docs/NEXT_ACTIONS_*.md` 寫一條**勘誤通知**，讓引用過它的 agent 看到；
4. 若錯誤已傳播到其他文件，逐處加同樣標記（保留追溯鏈）。

**禁止靜默覆蓋**——「曾經為什麼這麼想」本身有復盤價值，也是防止重犯的依據。

勘誤通知格式：

```markdown
## [勘誤 date] <一句話標題>
- 位置：<file:line>，原主張：<…>
- 推翻證據：<產物/數據/代碼行>
- 正確結論：<…>
- 已連帶修正：<其他位置清單>
```

## 3. 單一事實源（SSOT）：引用、不複製

- 權威數字只存在於**產物 / 錨點表**一處；memory、白皮書、commit 標題**引用**它（帶路徑），不複製數值。
- 任何數字帶：單位 + 口徑 + 量測條件（thermal、swap、shape、warm-skip）。
- 條件不同的數字不得直接相比（跨 thermal、跨 swap、跨 shape 都是不同母體）。

## 4. 自動核查（看門人）

`scripts/check/lane_watchdog.py` 提供入口；常駐 daemon `watchdog_daemon.sh` 每 600 s 跑 `--kill --quarantine`：

| 階段 | 命令 | 作用 |
|---|---|---|
| run 前 | `lane_watchdog.py gate` | thermal/swap/free/殘留進程/watchdog 不合格 → 非 0、runner 拒跑 |
| run 中 | daemon 定時巡檢 | 對 kill 級危險進程止損 |
| run 後 / 寫報告 | `lane_watchdog.py [--kill] [--quarantine]` | 產物三態判決 + 失效隔離 + NEXT_ACTIONS 通知 |
| 文件核查 | `lane_watchdog.py audit [files]` | 掃「絕對化措辭但同行無數字/來源」、列複核清單 |

### 4.1 產物三態判決（關鍵口徑）

數字**作廢與否由「thermal + rep 散度」裁決，不由 swap**：

| 判決 | 條件 | 數字可否引用 |
|---|---|---|
| clean | thermal NOMINAL、rep 散度 ≤12%、swap 不高 | 可引用 |
| STRESSED* | swap 高（啟動 >2048 或 growth >1500），但 thermal NOMINAL、散度 ≤12% | **可引用**（附承壓備註；swap 另需結構修復） |
| UNRELIABLE | thermal worst 非 NOMINAL（HEAVY 真實降 GPU 時脈），或 rep 散度 >12% | **作廢** |

- rep 散度口徑 = `(max(samples)−min(samples))/avg*100`（生產 cell 約 4.3% 可靠；壞 cell 可達 150%）。
- 教訓修正：**有 swap 仍可跑出正常值**（生產 cell 在 swap growth 3446 MiB 下仍給 12.20、散度 4.3%），
  不得再以 swap 高為由廢數字；但 swap 太大仍要結構修復、且必須在 thermal 正確狀態量測。
- `--quarantine` 對 UNRELIABLE 產物**主動隔離**：多 arm 檔在檔內給失效 arm 注入 `_quarantine`、
  整檔失效移到 `Backup/quarantine/` 並在原路徑留指針、登記 `QUARANTINE_REGISTRY.json`；STRESSED* 不動。

### 4.2 宣稱 vs 實際

開關是否生效要看**行為指標**、不看 json 記的 env（那是 profile 解析值、非實際下發值）。
看門人對「同 workers 宣稱、但 us/job 離散 >12%」標「並行度可能未生效、請驗證（非定論）」——
否則「沒生效」和「沒效果」看上去一模一樣。實例：兩臂都記 WORKERS=8、一臂 us/job 偏高。

`audit` 是**初篩、非定讞**：邏輯必然（如 union>sum 不可能）屬於【推斷】、補標籤即可；
真正無依據的絕對化結論必須改寫或降級為【假設】。

## 5. 責任

- **誰寫誰標**：下筆時自帶分級標籤與來源，不留無標籤結論。
- **誰發現誰勘誤**：任何 agent 發現文件錯誤，走第 2 節流程，不得忽略或私改。
- 看門人發現「無來源絕對化 / 已證偽未標記 / 數字與產物不一致」→ 主動開勘誤並通知。
