# 加入 CGC 機隊：Windows 端（RTX 4090 / VS2022）（`windows-rtx4090`）

> ★ 本檔由 `agent_harness/portal/build_fleet_portal.py --gen-onboarding` 從
> `agent_harness/portal/join.json` 產生。**不要手改** —— `--check` 會比對磁碟與重算結果。
> 改了註冊表就重生成；手改文件會讓閘門變紅（刻意的：文件與註冊表漂移是靜默的）。

## 0. 你是誰

| 欄位 | 值 |
|---|---|
| endpoint_id | `windows-rtx4090` |
| 我們的標籤 | windows / x86_64 |
| 角色 | CUDA 或 Vulkan 建置；repo 裡預期 Qwen3.6-35B-A3B IQ3_XXS 40–60 t/s |
| 現在的狀態 | **未加入**（我們沒有收過你的任何上報） |
| 你的上報檔 | `agent_harness/portal/endpoints/windows-rtx4090.json` |

> 這是我們最貴的通道缺口：它的 40–60 t/s 預期是唯一能把 decode 目標從「本機 12.62」往上推的外部讀數。

## 1. 連結資訊（唯一一條要打通的通道）

### 主要通道：`git`

- **repo**：`git@github.com:alexchuang19760730/cgcengine0907.git`
- **remote**：`你那邊的 remote 名（現有腳本裡是 cgc0907）`
- **branch**：`demo/sweet-spot-windows-fix`
- **你要寫的檔案**：`agent_harness/portal/endpoints/windows-rtx4090.json`
- ★ **注意**：現有腳本 `auto_git_push.ps1` 推的是 `fusionroutemot` ⇒ 與上面那條**不同**（見 §2 第 3 項）
- 自動化：`agent_harness/scripts/auto_git_push.ps1（監看 agent_harness/，定期 commit ＋ push）`
- 為什麼：本入口讀的是 demo/sweet-spot-windows-fix。你推到 fusionroutemot 會成功，但入口看不到 —— 這是已知的靜默缺口。

先量一次通道有沒有通：

```bash
git ls-remote --heads <你的 remote> demo/sweet-spot-windows-fix
```

### 備援通道：`scp/rsync`

- 把 agent_harness/portal/endpoints/windows-rtx4090.json 直接 scp 回我們這台（或貼進 PR）

```bash
scp agent_harness/portal/endpoints/windows-rtx4090.json <our-host>:~/incoming/
```

### 心跳（可選，不是必要條件）

- 現在是 `null` —— ★ **這不是「失敗」**。我們沒有你的可達位址 ⇒ 這一格永遠是
  `unknown`、不會變綠。連得上時把實際 URL 回報給我們，我們才填進註冊表。
- 要起它的話：`若真的連得上區網：python3 agent_harness/pd/compute_sharing.py --role node --port 9100`
- ★ null 不是「失敗」。現在沒有可達位址 ⇒ 這一格永遠是 unknown、不會變綠。連得上時把實際 URL 回報給我們，我們才把它填進註冊表。

## 2. 你要做的事（7 項；做完一項，入口上就少一格 todo）

### 1. （可選）先讓入口看到這台機器：dump 一份狀態

**為什麼**：★ 這條**不等於**下面的項目：它上報的是「這台機器現在是什麼」（平台／CPU／RAM／後端提示／服務狀態），不含建置結果、加速器型號、實測 t/s —— 那些欄位要人寫。自動跑不會把它們洗掉：覆蓋報告時會把人工寫的部分接過來。

**怎麼做**：

```bash
python3 installer/edge_server.py --dump-status status.json --endpoint-id windows-rtx4090
放到 agent_harness/portal/endpoints/dumps/windows-rtx4090.json（★ 是 dumps/ 子目錄，不是 endpoints/ 本身）
推上去；映射與匯出由我們這邊的 fleet_auto.py 定期接手
```

### 2. 產生第一份上報檔 `endpoints/windows-rtx4090.json`

**為什麼**：契約要求 not_measured 必填且不得留白；兩個都沒給，產生器會拒跑（rc=2）。

**怎麼做**：

```bash
python3 agent_harness/portal/report_endpoint_status.py --id windows-rtx4090 \
    --what-ran '…' --not-measured '…'（或 --claim-nothing-unmeasured）
```

### 3. 上報的 `platform` 寫對（Windows 是 `win32`，不是 `windows`）

**為什麼**：註冊表用 windows 當**我們的標籤**，而 sys.platform 是 win32 ⇒ 不對齊會讓自動比對靜默失效。

**怎麼做**：

```bash
不要覆寫 --platform，讓它取 sys.platform（= win32）
```

### 4. 把上報推到入口讀的那條分支（或告訴我們改讀你推的那條）

**為什麼**：★ 這一格現在就是 todo —— 刻意的。它把「你推成功了但我們看不到」這個沉默缺口變成一格看得見的紅。

**怎麼做**：

```bash
改 agent_harness/scripts/auto_git_push.ps1 的 $BranchName = "demo/sweet-spot-windows-fix"
或：叫我們改讀 fusionroutemot（二選一，但兩邊要一致）
```

### 5. 真的建過一次（CUDA，或退回 Vulkan），把結果寫進 `what_ran`

**為什麼**：「能不能建」現在只是腳本裡的一行意圖；只有跑過才是能力。

**怎麼做**：

```bash
deploy-harmonyos/windows/build-windows.bat
--what-ran 'CUDA 建置完成（GGML_CUDA=ON）'
```

### 6. 回報 GPU 型號與建置設定（`gpu` / `build_profile`）

**為什麼**：沒有它，「40–60 t/s 的預期」沒有對應的機器可查。

**怎麼做**：

```bash
--gpu 'RTX 4090 24GB' --build-profile 'VS2022 + CUDA 12.x'
```

### 7. 跑一次真實 decode，把 t/s 寫進 `metrics.decode_tps`

**為什麼**：README 的 40–60 t/s 是**預期**，本入口不把它當實績。唯一能改變這一點的就是量。

**怎麼做**：

```bash
跑一次生成（llama-server 或 Playground），再：
--metric decode_tps=<實測值>
```

## 3. 送出

```bash
python3 agent_harness/portal/report_endpoint_status.py --id windows-rtx4090 \
    --what-ran '實際跑了什麼' \
    --capability '我實際驗證過的能力' \
    --not-measured '我不知道什麼'   # 或 --claim-nothing-unmeasured

# 先在端點上自檢（不寫檔）：
python3 agent_harness/portal/report_endpoint_status.py --id windows-rtx4090 --print
```

然後把 `agent_harness/portal/endpoints/windows-rtx4090.json` 送到我們這邊（通道見 §1）。

## 4. 邊界（不要做這些）

- 不要 push 到 `main`、不要 `--force`、不要動 `agent_harness/` 以外的檔。
- 不要 `git add -A`（這個 repo 有多條線同時在寫；`-A` 會把別人的變更一起提交）。
- **不要填你沒量到的數字。** `--not-measured` 就是為此存在的：
  缺席要出聲，留白會被讀成「沒有未量的東西」，而那是最貴的一種沉默。
- 不要手改本檔（`--check` 會抓到）。

## 5. 我們這邊怎麼驗

```bash
python3 agent_harness/portal/build_fleet_portal.py --check      # 會逐條指名哪一格沒做
python3 agent_harness/portal/build_fleet_portal.py --no-net     # 產生入口頁
```

做完之後，入口的機隊表上會出現你的卡片，並且帶著：
**加入進度**（上面那幾格）、**能力**（你上報的 ＋ 我們從 repo 驗到的）、
**未量測**（你宣告的 ＋ 我們的）、以及**復盤**（歷次建置的能力數與未量測數變化）。

