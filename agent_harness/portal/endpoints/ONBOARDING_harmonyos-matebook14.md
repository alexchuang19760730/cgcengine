# 加入 CGC 機隊：鴻蒙端（MateBook 14 / Kirin 9030）（`harmonyos-matebook14`）

> ★ 本檔由 `agent_harness/portal/build_fleet_portal.py --gen-onboarding` 從
> `agent_harness/portal/join.json` 產生。**不要手改** —— `--check` 會比對磁碟與重算結果。
> 改了註冊表就重生成；手改文件會讓閘門變紅（刻意的：文件與註冊表漂移是靜默的）。

## 0. 你是誰

| 欄位 | 值 |
|---|---|
| endpoint_id | `harmonyos-matebook14` |
| 我們的標籤 | harmonyos / aarch64 |
| 角色 | CPU-only 建置與執行（Kirin 9030 沒有 Metal ⇒ GGML_METAL=OFF） |
| 現在的狀態 | **未加入**（我們沒有收過你的任何上報） |
| 你的上報檔 | `agent_harness/portal/endpoints/harmonyos-matebook14.json` |

## 1. 連結資訊（唯一一條要打通的通道）

### 主要通道：`git`

- **repo**：`git@github.com:alexchuang19760730/cgcengine0907.git`
- **remote**：`你那邊的 remote 名`
- **branch**：`demo/sweet-spot-windows-fix`
- **你要寫的檔案**：`agent_harness/portal/endpoints/harmonyos-matebook14.json`
- 為什麼：與 Windows 同一條路：推上 demo/sweet-spot-windows-fix 我們才讀得到。

先量一次通道有沒有通：

```bash
git ls-remote --heads <你的 remote> demo/sweet-spot-windows-fix
```

### 備援通道：`scp`

- 本機已有 deploy-harmonyos/deploy-to-harmonyos.sh（scp -r 整包過去 ＋ ssh 進去跑 build.sh）；反向把 endpoints/harmonyos-matebook14.json scp 回來也可以。

```bash
bash deploy-harmonyos/deploy-to-harmonyos.sh <user>@<host>
```

### 心跳（可選，不是必要條件）

- 現在是 `null` —— ★ **這不是「失敗」**。我們沒有你的可達位址 ⇒ 這一格永遠是
  `unknown`、不會變綠。連得上時把實際 URL 回報給我們，我們才填進註冊表。
- 要起它的話：`若真的連得上：python3 agent_harness/pd/compute_sharing.py --role node --port 9100`
- ★ null 不是「失敗」：沒有可達位址 ⇒ unknown，不會變綠。

## 2. 你要做的事（6 項；做完一項，入口上就少一格 todo）

### 1. （可選）先讓入口看到這台機器：dump 一份狀態

**為什麼**：★ 這條**不等於**下面的項目：它上報的是「這台機器現在是什麼」（平台／CPU／RAM／後端提示／服務狀態），不含建置結果、加速器型號、實測 t/s —— 那些欄位要人寫。自動跑不會把它們洗掉：覆蓋報告時會把人工寫的部分接過來。

**怎麼做**：

```bash
python3 installer/edge_server.py --dump-status status.json --endpoint-id harmonyos-matebook14
放到 agent_harness/portal/endpoints/dumps/harmonyos-matebook14.json（★ 是 dumps/ 子目錄，不是 endpoints/ 本身）
推上去；映射與匯出由我們這邊的 fleet_auto.py 定期接手
```

### 2. 產生第一份上報檔 `endpoints/harmonyos-matebook14.json`

**為什麼**：沒有它，入口上這一格永遠是「未上報」。

**怎麼做**：

```bash
python3 agent_harness/portal/report_endpoint_status.py --id harmonyos-matebook14 \
    --what-ran '…' --not-measured '…'
```

### 3. 回報你的環境：`notes` 要寫 REMOTE 的 `user@host`、鴻蒙版本、可用 RAM　`[弱檢查]`

**為什麼**：我們現在的 not_measured 有三條「不知道」：鴻蒙版本、RAM、build.sh 最後成功時間。★ 這一格只證明 notes 不是空的，不代表內容對 —— 標成弱檢查，不假裝它證明了更多。

**怎麼做**：

```bash
--notes 'REMOTE=<user>@<host>；鴻蒙版本=<…>；RAM=<…>GB'
```

### 4. 上報的 `platform` 是實話（鴻蒙 PC 多半回 `linux`）

**為什麼**：註冊表用 harmonyos 當我們的標籤；若 Python 報 linux 也要能被接受，但不能容忍它空著或填錯。

**怎麼做**：

```bash
不要覆寫 --platform，讓它取 sys.platform
```

### 5. 跑過 `build.sh`（CPU-only、GGML_METAL=OFF），把結尾輸出寫進 `what_ran`

**為什麼**：Kirin 9030 沒有 Metal ⇒ 這是唯一能回答「這台到底跑不跑得動」的路徑。

**怎麼做**：

```bash
ssh <user>@<host> 'cd ~/llama-cpp/harmonyos && ./build.sh'
--what-ran 'aarch64 CPU-only 建置完成（GGML_METAL=OFF）'
```

### 6. 跑一次真實 decode，把 t/s 寫進 `metrics.decode_tps`

**為什麼**：CPU-only 的 decode t/s 是「decode ≥25 t/s 能不能跨機複製」的關鍵讀數；沒有它，我們只有「應該可以」。

**怎麼做**：

```bash
ssh <user>@<host> 'cd ~/llama-cpp/harmonyos && ./run.sh -m ~/models/<m>.gguf'
--metric decode_tps=<實測值>
```

## 3. 送出

```bash
python3 agent_harness/portal/report_endpoint_status.py --id harmonyos-matebook14 \
    --what-ran '實際跑了什麼' \
    --capability '我實際驗證過的能力' \
    --not-measured '我不知道什麼'   # 或 --claim-nothing-unmeasured

# 先在端點上自檢（不寫檔）：
python3 agent_harness/portal/report_endpoint_status.py --id harmonyos-matebook14 --print
```

然後把 `agent_harness/portal/endpoints/harmonyos-matebook14.json` 送到我們這邊（通道見 §1）。

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

