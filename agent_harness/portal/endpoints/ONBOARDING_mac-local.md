# 加入 CGC 機隊：本機 Mac（M4 / 16 GB unified）（`mac-local`）

> ★ 本檔由 `agent_harness/portal/build_fleet_portal.py --gen-onboarding` 從
> `agent_harness/portal/join.json` 產生。**不要手改** —— `--check` 會比對磁碟與重算結果。
> 改了註冊表就重生成；手改文件會讓閘門變紅（刻意的：文件與註冊表漂移是靜默的）。

## 0. 你是誰

| 欄位 | 值 |
|---|---|
| endpoint_id | `mac-local` |
| 我們的標籤 | macos / arm64 |
| 角色 | decode 生產線：Metal kernel ＋ SSD expert streaming；也是這個入口自己跑的地方 |
| 現在的狀態 | 已加入 |
| 你的上報檔 | `agent_harness/portal/endpoints/mac-local.json` |

## 1. 連結資訊（唯一一條要打通的通道）

### 主要通道：`local`

- **你要寫的檔案**：`agent_harness/portal/endpoints/mac-local.json`
- 為什麼：本機就是入口自己跑的地方 ⇒ 直接寫檔，不需要任何通道。

先量一次通道有沒有通：

```bash
python3 agent_harness/portal/report_endpoint_status.py --id mac-local --print
```

### 心跳（可選，不是必要條件）

- `http://127.0.0.1:9100/v1/compute/status`
- 要起它的話：`agent_harness/pd/compute_sharing.py --role node --port 9100`
- 只在服務真的起來時才有回應；沒起來是 unknown，不是紅。

## 2. 你要做的事（3 項；做完一項，入口上就少一格 todo）

### 1. （可選）先讓入口看到這台機器：dump 一份狀態

**為什麼**：本機的觀測來自「edge_server 正在跑」。服務沒跑 ⇒ 這一格是 absent，那是**對的**（服務真的沒跑）。自動鏈每一輪的留痕都會記下這一格是 updated 還是 absent。

**怎麼做**：

```bash
服務跑著時不用 dump —— fleet_auto.py 直接抓 http://127.0.0.1:8080/v1/edge/status
服務沒跑：python3 installer/edge_server.py --dump-status status.json --endpoint-id mac-local
放到 agent_harness/portal/endpoints/dumps/mac-local.json
```

### 2. 有上報檔（`endpoints/mac-local.json`）

**為什麼**：沒有上報檔，這一格永遠是「未上報」。

**怎麼做**：

```bash
python3 agent_harness/portal/report_endpoint_status.py --id mac-local --what-ran '…'
```

### 3. 在 `fleet.json` 的端點註冊表裡

**為什麼**：★ 沒註冊的上報檔不會被讀（`--check` 也會指名它）。

**怎麼做**：

```bash
在 fleet.json 的 endpoints 加一筆 id=mac-local
```

## 3. 送出

```bash
python3 agent_harness/portal/report_endpoint_status.py --id mac-local \
    --what-ran '實際跑了什麼' \
    --capability '我實際驗證過的能力' \
    --not-measured '我不知道什麼'   # 或 --claim-nothing-unmeasured

# 先在端點上自檢（不寫檔）：
python3 agent_harness/portal/report_endpoint_status.py --id mac-local --print
```

然後把 `agent_harness/portal/endpoints/mac-local.json` 送到我們這邊（通道見 §1）。

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

