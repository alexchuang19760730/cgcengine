# `endpoints/` —— 端點狀態上報的落點

這個目錄是**離線端進入入口的唯一入口**。任何端點只要能產生一份符合下面契約的
`<endpoint_id>.json` 並讓它落到這裡，就會出現在 `docs/FLEET_PORTAL.html` 的機隊表上。

## 從 edge_server 產生上報（`--from-edge-status`）

★ **契約形狀只存在這一側。** `edge_server.py` 吐的是**事實**（它觀察到什麼），
這一支把事實映射成契約。若讓 edge 端也吐契約形狀，就會有第二份契約 —— 而兩份必然漂移。

```bash
# 服務在跑（--no-worker 也能起，不需模型、不佔記憶體）
python3 installer/edge_server.py --host 0.0.0.0 --port 8080 \
    --api-key <key> --endpoint-id mac-local --no-worker

# 把它的 status 映射成上報（URL 或檔案皆可）
python3 agent_harness/portal/report_endpoint_status.py \
    --from-edge-status http://192.168.101.90:8080/v1/edge/status \
    --id mac-local \
    --not-measured '<你自己知道的未知>' \
    --out-dir agent_harness/portal/endpoints
```

**`what_ran` 從哪來**：契約要求它不得為空，而 edge 端唯一誠實的來源就是它自己的觀測 ——
啟動時間、worker 模式與可達性、已服務計數、上一次實際走過的 chat 路由、模型是否存在、
探測到的旗標數。**每一條都能回溯到 status 裡的一個欄位**；`notes` 會寫明
「前 N 條由觀測導出、其餘 N 條由 `--what-ran` 人工提供」。**不推測、不補話。**

**會拒跑的情形**（寧可拒跑也不要寫出錯的東西）：
`object` 不是 `powerauto.edge.status`（拿錯 JSON）／`--id` 與 status 的 `endpoint_id` 不一致
（會寫到**錯的檔名**）／兩邊都沒有 id／edge 回空 `not_measured` 而沒明確宣告
（空陣列是一句**主張**，見上面的規則）。

★ **URL 一律繞代理**：環境裡的 `HTTP_PROXY`（本機實測 `http://127.0.0.1:7897`）會把
`http://192.168.101.90:…` 也丟進代理 ⇒ 靜默失敗，而失敗長得像「端點沒在跑」。
實測：`urlopen` 失敗、`build_opener(ProxyHandler({}))` 通（測 LAN 位址，不是 127.0.0.1）。

★ **`host` 缺 platform／arch 時**：那兩個值取自**產生器所在的機器**，`notes` 會明說。
若產生器不在端點上跑，它們就是錯的 —— 這一句必須留著，否則兩格看起來像端點的事實。


## 為什麼是檔案，不是協議

鴻蒙端與 Windows 端**目前不在我們的網路上**。在它們真的離線時，任何「即時協議」都不成立
—— 包含 `agent_harness/pd/compute_sharing.py` 的心跳（`:9100/v1/compute/status`）。
所以這裡採 store-and-forward：端點寫檔，靠**既有**的通道帶回來。

| 通道 | 既有實作 | 適用 |
|---|---|---|
| `scp` + `ssh` | `deploy-harmonyos/deploy-to-harmonyos.sh`（`scp -r` 整包過去、ssh 進去跑 `build.sh`） | 鴻蒙端 |
| `git` | `agent_harness/scripts/auto_git_push.ps1`（監看 `agent_harness/`、推 remote `cgc0907`、branch `fusionroutemot`） | Windows 端 |
| `rsync` | `CGC-main/cgc_engine/tools/_archive_v1/server/sync_all_gates_to_hosts.sh` | 雲端主機 |

★ 兩個要注意的對齊問題：
1. `auto_git_push.ps1` 推的是 **`fusionroutemot`**，而入口讀的是 **`demo/sweet-spot-windows-fix`**
   ⇒ 兩邊對齊之前，Windows 的上報**不會**出現在入口上。
2. `auto_git_push.ps1` 在本機**從未被執行過**（2026-09-17 實查）。「它會自動推」目前是
   設計意圖，不是已驗證的行為。

## 契約（`contract_version = 1`）

```json
{
  "contract_version": 1,
  "endpoint_id": "windows-rtx4090",
  "reported_at": "2026-09-18 14:02:11",
  "hostname": "DESKTOP-XXXX",
  "platform": "win32",
  "arch": "AMD64",
  "produced_by": "report_endpoint_status.py v1",
  "what_ran": ["建 CUDA 版（GGML_CUDA=ON）", "跑 llama-bench，Qwen3.6-35B-A3B IQ3_XXS"],
  "capabilities": ["CUDA backend 可建置"],
  "gpu": "RTX 4090 24GB",
  "build_profile": "VS2022 + CUDA 12.x",
  "metrics": {"decode_tps": 52.3, "prefill_tps": 487.1},
  "not_measured": ["沒有量過 M1/M2/M3 oracle 身分"],
  "notes": ""
}
```

| 欄位 | 必填 | 說明 |
|---|---|---|
| `contract_version` | ✔ | 必須是 `1` |
| `endpoint_id` | ✔ | 要對應 `fleet.json` 的端點 id；只能小寫英數與 `-_.`（它會變成檔名） |
| `reported_at` | ✔ | `YYYY-MM-DD HH:MM:SS`（本機時區）。**不能是空字串**——新鮮度全靠它 |
| `platform` / `arch` | ✔ | 沒偵測到就寫實話，不要猜 |
| `what_ran` | ✔ | **不能是空陣列**。一輪什麼都沒跑也要寫一句，例如「本輪沒有跑任何東西」 |
| `capabilities` | ✔ | 可以空（沒驗證過就別寫）。**只寫驗證過的** |
| `not_measured` | ✔ | ★ 可以空，但空是一個**主張**；產生器強迫你明說（見下） |
| `gpu` / `build_profile` / `metrics` / `notes` | | 選填 |

### ★ `not_measured` 為什麼要逼你表態

空陣列 = 「這個端點沒有未量的東西」，這是一個**主張**。它不能由「使用者忘了填」推得 ——
留白會被讀成前者，而那是最貴的一種沉默。所以產生器要求你二擇一：

```bash
# 要嘛逐條寫出你不知道什麼
--not-measured "沒有量過 M1/M2/M3 身分" --not-measured "不知道鴻蒙版本"

# 要嘛明確宣告沒有未量項
--claim-nothing-unmeasured
```

兩個都不給 ⇒ **拒跑**（rc=2）並說明理由。

## 怎麼產生

```bash
# 在端點上（只需要 Python 標準函式庫；沒有這個 repo 也能跑）
python3 report_endpoint_status.py --id windows-rtx4090 \
    --what-ran "建 CUDA 版" --metric decode_tps=52.3 \
    --not-measured "沒量過 M1/M2/M3 身分"

python3 report_endpoint_status.py --self-test     # 7 格黑箱自測

# 沒有 Python 的端點：照上面的契約手寫同一份 JSON 就行 —— 契約是欄位，不是這支腳本
```

產生器預設寫到**它自己旁邊的** `endpoints/`；用 `--out-dir` 指到你的 checkout 也可以。

## 這個目錄的兩條規矩

1. **這裡放的是「現況」，不是「趨勢」。** 同一端點重複上報會**覆蓋**。趨勢落在
   `agent_harness/portal/fleet_status.jsonl`（只追加），由入口建置時寫入。
2. **`fleet.json` 與這裡必須雙向對齊。** 註冊表裡的端點沒上報 ⇒ 表上顯示「未上報」；
   這裡有檔而註冊表沒有那個 id ⇒ **入口會紅**。少一邊都是靜默的漏洞。
