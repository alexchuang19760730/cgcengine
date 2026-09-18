# `dumps/` —— 還沒被映射的原始 edge status

這裡放的是**運輸物**，不是報告。

```
endpoints/<id>.json          ← 報告：映射過的、合契約的、入口讀的
endpoints/dumps/<id>.json    ← 原始 dump：edge_server --dump-status 的輸出，還沒映射
```

## 為什麼要分開放

`build_fleet_portal.py --check` 會 glob 這個目錄的**上一層**（`endpoints/*.json`），並要求
每個檔案都對得上 `fleet.json` 裡的一個端點。如果 dump 直接放在上一層：

```
endpoints/windows-rtx4090.status.json
```

檢查器會說：

> endpoints/windows-rtx4090.status.json 存在，但 fleet.json 沒有註冊 windows-rtx4090.status

—— 那句話聽起來像**註冊表少了東西**，其實只是**檔案放錯目錄**。
檢查器不該把使用者指去修錯的地方，所以 dump 有自己的子目錄（`glob("*.json")` 不遞迴）。
`fleet_auto.py --check` 對這個錯放會**直接指名**。

## 怎麼產生

在端點上（鴻蒙／Windows 都不在我們的網路上，所以只能在**那台機器**上跑）：

```bash
python3 installer/edge_server.py --dump-status status.json --endpoint-id windows-rtx4090
```

不起服務、不開埠、不需要模型、不需要 llama-server。然後把 `status.json` 放成
`dumps/<endpoint_id>.json`，再由**既有通道**（scp／git push）帶回來。

放進來就不用管了：`agent_harness/portal/fleet_auto.py` 會定期讀它、映射成
`endpoints/<id>.json`，並把結果寫進 `auto_runs.jsonl`。

## 兩件容易搞錯的事

1. **檔名必須是 `<endpoint_id>.json`，而且那個 id 要在 `fleet.json` 裡。**
   對不上就是孤兒 —— `fleet_auto.py --check` 會出聲（它永遠不會被採集）。
2. **dump 裡的 `reported_at` 是那台機器的鐘，不會被改寫。** 這是刻意的：一份三天前產生、
   今天才被帶回來的 dump，映射之後仍然顯示三天前 —— 入口上「它多久沒說話」才看得出來。
   若拿映射時刻充當 `reported_at`，自動跑會每小時把它抹平一次。
