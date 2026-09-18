# `fleet_auto.py` —— 讓「dump → 帶回 → 映射 → 匯出」這條鏈定期跑

> 這一頁只講一件事：**這條鏈被自動化了，以及它自動不了什麼。**
> 前者是功能，後者是紀律 —— 少了後者，這條鏈會變成一條「看起來在跑」的鏈。

## 這條鏈的四步，與它們各自的家

| 步 | 在哪台機器跑 | 誰負責 |
|---|---|---|
| ① dump：寫出一份 edge status | **端點自己**（鴻蒙／Windows 上） | `edge_server.py --dump-status` |
| ② 帶回：把 dump 搬進 `endpoints/dumps/` | 端點 ↔ 這裡 | **既有通道**（scp／git push），不是這支 |
| ③ 映射：dump → 合契約的報告 | 這裡 | `report_endpoint_status.py --from-edge-status` |
| ④ 匯出：報告 → 網站吃的 JSON | 這裡 | `build_fleet_portal.py --export` |

`fleet_auto.py` 自動化的是 **③＋④，以及對 ② 的「有沒有東西可讀」的檢查**。
它**不**自動化 ①（那台機器不在我們的網路上）與 ②（通道是既有的東西）。

## 它做不到什麼（不會假裝做到）

- **端點上的 dump**：離線端不在我們的網路上，那一步只能在**那台機器**上跑。
- **通道**：scp／git push 是既有的實作，這支不負責把它們打開。
  ⇒ 所以結果裡最常見的一格是 `absent`，而**它不是失敗**。
- **`commit`**：它只寫檔，不 commit。要不要把 `endpoints/`／`docs/fleet_export.json`
  的變動進版控，是人的決定。（多 session 共寫同一個 repo，自動 commit 會踩到別人的線。）
- **讓本機端點一定有資料**：本機的觀測來自「edge_server 正在跑」。
  服務沒跑 ⇒ 本機端點也是 `absent`，這是**對的**（服務真的沒跑）。

## 四種結果，只有一種是錯

| 結果 | 意思 | rc |
|---|---|---|
| `updated` | 有新的觀測，報告已更新 | 0 |
| `unchanged` | 同一份觀測已經處理過（**不重寫**） | 0 |
| `absent` | 沒拿到 —— 服務沒跑／dump 不存在／通道沒開。**不知道，不是失敗** | 0 |
| `reject` | 拿到了但拒收（見下） | **1** |

`reject` 的四種：`object` 不是 `powerauto.edge.status`（拿錯 JSON）／dump 的 `endpoint_id`
與註冊不符（接受它就會**寫到別人的檔名**）／dump 沒有 `endpoint_id`／**拿到的觀測比現有的舊**
（一份舊 dump 不該把新的報告蓋回去）。

★ 把 `absent` 與 `reject` 混成一格，看門狗就分不出「該不該叫人」——
前者不需要處理，後者需要。

## 三條硬規矩

**① 不偽造新鮮度。** 報告的 `reported_at` 取**端點自己的觀測時刻**（`dump` 裡那一格），
不是這一輪的執行時刻；處理時刻另記在 `mapped_at`。
端點離線三天、dump 今天才被帶回來時，這兩個時刻差三天。手動跑人會發現；
**每小時跑一次會把這個錯誤每小時抹平一次，永遠看起來是新的** —— 那是自動化最容易製造的謊。

**② 同一份觀測只處理一次。** 冪等判準就是 `reported_at`。沒有新觀測 ⇒ 不重寫檔案
（自測會驗 mtime 不變），也不重生 `fleet_export.json`（否則 git 每天多出一堆只有時間戳
不同的 diff，而「上次真的有變化是什麼時候」就看不見了）。
匯出仍會在**跨日**時跑一次 —— 七日趨勢與動能有機會變。

**③ 覆蓋時要保住人的工作。** 自動跑覆蓋一份報告之前，會把**只有人知道**的東西接過來：
`what_ran`／`capabilities`／`not_measured`／`gpu`／`build_profile`／`metrics`。
切分線是報告裡的 `edge_derived`（映射層寫下的邊界：前 N 條是 edge 導出的）。
**沒有那一格 ⇒ 整份都算人工的。**

> 這條不是潔癖，是**實測踩到的**：一台剛 `--no-worker` 起來的 edge_server 覆蓋之後，
> `mac-local` 從「3 條能力 ＋ 3 個實測 metrics ＋ 3 條未量測 ＋ GPU」變成
> 「uptime 2.6s、服務 0 次」——**用零換掉實質**。
> 一個會默默丟掉別人工作的自動化，比不自動化更糟。

## 用法

```bash
python3 agent_harness/portal/fleet_auto.py                    # 跑一輪（= --run）
python3 agent_harness/portal/fleet_auto.py --plan             # 只說會做什麼（不連網、不寫檔）
python3 agent_harness/portal/fleet_auto.py --check            # 看門狗
python3 agent_harness/portal/fleet_auto.py --status           # 最近幾輪的留痕
python3 agent_harness/portal/fleet_auto.py --json             # 機器可讀
python3 agent_harness/portal/fleet_auto.py --self-test        # 21 格突變式黑箱自測
```

### 看門狗（`--check`）的五種判讀

| 排程 | 留痕 | 判讀 | rc |
|---|---|---|---|
| 無 | 無 | 未啟用 | 0 |
| 無 | 有 | 只人手跑過 | 0 |
| 有 | 無 | ★ **裝了卻從沒跑過** | **1** |
| 有 | 有，且超過 2.5× 間隔 | ★ **停擺了** | **1** |
| 有 | 有，且新鮮 | 健康 | 0 |

另外它會檢查兩件**症狀會誤導**的事：
- 原始 dump 被放進 `endpoints/`（上一層）⇒ 指名「放錯位置」
- `endpoints/dumps/` 有對不到任何註冊端點的孤兒 ⇒ 出聲（它永遠不會被採集）

★ 看門狗存在的理由：「排程裝了卻沒跑」是自動化最貴的失敗模式 ——
**入口上的數字看起來一樣，而它其實停在三天前。**

## 排程（macOS launchd）

```bash
python3 agent_harness/portal/fleet_auto.py --install-schedule --interval-min 60
python3 agent_harness/portal/fleet_auto.py --uninstall-schedule
```

- 排程檔：`~/Library/LaunchAgents/ai.powerauto.fleet-auto.plist`
- `RunAtLoad=true` ⇒ **裝完會立刻跑一次**，留痕裡會出現一筆 `trigger=schedule`
  —— 這是「排程真的被觸發過」唯一的證據（否則你只有一個你自己相信的設定檔）。
- 日誌：`~/Library/Logs/powerauto/fleet_auto.{out,err}.log`（**repo 外** —— 執行期產物
  不是版控資產；放在 repo 裡會讓 `git status` 一直多一個 untracked 目錄，而那種雜訊
  正好會掩蓋真正的變更）
  （launchd 是 append、沒有輪轉；檔案大了自己清。）
- 非 macOS：`--install-schedule` 會**印出 cron 行而不裝**，理由在輸出裡 ——
  在別人的機器上偷偷多一條 cron，比不裝更糟。

## 留痕（`auto_runs.jsonl`）

每一輪一行，只追加、永不覆蓋。**連「這一輪什麼都沒拿到」也留** ——
沒記錄與沒上報是兩件事，前者是這一支的責任。

```json
{"ts": "2026-09-18 16:04:26", "trigger": "schedule", "repo_head": "0a3d48cf7",
 "endpoints": [{"id": "mac-local", "reaches": "self", "source": null,
                "result": "absent", "reported_at": null, "why": "dump 不存在（…）"}],
 "updated": [], "absent": ["mac-local", "cloud-host2", "harmonyos-matebook14", "windows-rtx4090"],
 "rejected": [], "unchanged": [], "exported": false,
 "export_why": "今天已經匯出過，且這一輪沒有新觀測"}
```

檔案只會長大（一輪約 0.5 KB）。超過 2 MB 時 `--run` 會印一行提示 ——
**出聲，但不自動刪**：自動刪歷史是危險動作，而「沒人記得它已經 4 MB」才是真正的問題。

## 來源從哪來（不建第二份真相）

`fleet.json` 的 `channels.edge_status` 是**唯一**的來源宣告：

```json
"edge_status": {
  "url":  "http://127.0.0.1:8080/v1/edge/status",   // 服務活著時走這條；null ＝ 這條路不存在
  "dump": "agent_harness/portal/endpoints/dumps/mac-local.json",
  "how":  "…"
}
```

`reaches == "offline"` 且 `url` 為 `null` ⇒ **不會去試那條路**。
這不是最佳化：離線端本來就不在網路上，去試只換來一次 timeout，
而**那個 timeout 長得像「服務掛了」**。`reaches` 是唯一該問的問題。
