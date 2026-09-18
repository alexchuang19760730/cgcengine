# `agent_harness/a2a/` —— 把 harness 與 pd 服務暴露成 A2A agent

> 這一頁回答一個問題：**「對面那個 agent 是哪一類、能決定什麼、能執行什麼？」**
> ——而且答案必須是**機檢的**（別的 agent 在呼叫前就讀得到），不是寫在文件裡的宣稱。

## 為什麼不是「再加一個服務」

`agent_harness/` 底下已經有兩套東西在跑：兩條迴路（engine loop／tb loop）與 **pd 推論子系統**
（gRPC，KV cache ＋ DOPD handoff）。它們共同缺的不是「更多功能」，而是**一個對外的身分**：
別的 agent 想知道「你能做什麼」時，只能讀文件或試錯；而試錯失敗與「你沒這個能力」
在日誌裡長得一樣。

A2A 把這件事變成協議：**能力與類別寫在卡片上，呼叫前就讀得到**。

## 三類 agent，與兩個刻意分開的維度

| 類別 | 是什麼 | 宏觀決策域（macro） |
|---|---|---|
| `developer` | 把引擎移植到新平台（windows／鴻蒙）並證明它跑起來 | `porting` —— 移植目標與構建策略 |
| `operator` | 讓端／雲／端的服務持續可觀測、可部署（pd 與 harness 入口都在這） | `fleet-ops` —— 機隊部署與可用性 |
| `explorer` | 解決宏觀決策：目標還成不成立、哪條路已封閉、哪些結論已被推翻 | `decision` —— 目標、路線與假設的取捨 |

**macro 與 micro 不可互相推導。** 一個能跑 `cmake` 的 agent（micro）不代表它有權決定
「要不要移植到鴻蒙」（macro）。把兩者併成一張清單，下一個讀者就會以為「能執行」等於「有權決定」。
所以在協議上它們是分開的：

- **微觀能力** → A2A **標準**的 `skills[]`（每一項是一個可呼叫的動作，標準客戶端也看得懂）
- **類別與宏觀決策域** → 擴展欄位 `x-agent-class` / `x-macro-scope`

★ 為什麼不把類別也塞進 `skills[]`：A2A 的 `skills[]` 語意是「可呼叫的能力」。把類別塞進去，
標準客戶端會以為 `class:developer` 是一個可以呼叫的動作 —— 它會去呼叫，然後失敗。

## 分類的**實際作用**：路由會因為它拒絕

只把類別寫在卡片上，它是一個標籤。真正讓它是分類的是這條規則：

```json
{"error": {"code": -32005,
  "message": "skill 'register-decision' 不屬於 developer 這一類（它是 探索 agent）",
  "data": {"agentClass": "developer", "skillOwnerClass": "explorer",
           "availableSkills": ["build-target", "port-and-verify", ...]}}}
```

把 explorer 的 skill 送給 developer agent ⇒ **拒收並指名誰才是 owner**。
沒有這一條，「能區分 agent 屬於哪一類」就只是一句話。

同理，**沒指名 skill 也拒收**（`-32602`）：猜一個預設 skill 會讓打錯字的請求看起來像成功。

## 四組能力（使用者要的是完整規格，四組都真的能用）

| 方法 | 說明 |
|---|---|
| `message/send` | 同步任務；回 `completed` 的 task 或 error |
| `message/stream` | SSE，逐步回報（自測會真的收事件流，不是只驗標頭） |
| `tasks/get` / `tasks/cancel` | 查詢與取消；取消一個已完成的任務 ⇒ **拒絕**（-32002），不是靜默接受 |
| `tasks/pushNotificationConfig/set\|get\|list\|delete` | webhook 設定，**且真的投遞**（自測會起一個接收器驗這一格） |

★ `capabilities.pushNotifications: true` 宣告了就要送得出去。宣告 true 而不實作，
是這一類協議最常見的謊。投遞結果會留在 `tasks/get` 的 `metadata.pushDeliveries` 裡
—— 否則「push 已設定」與「push 送達過」長得一樣。

## 端點

```bash
python3 agent_harness/a2a/a2a_server.py --serve --port 9210

GET  /.well-known/agent-card.json                      # 閘道目錄（本閘道擴展：這裡有誰）
GET  /a2a/<agent_id>/.well-known/agent-card.json       # 那個 agent 的 card
POST /a2a/<agent_id>                                   # JSON-RPC
GET  /healthz
```

四個 agent：`cgc-dev-porting`（developer）、`fleet-operator`（operator）、
`pd-inference`（operator）、`cgc-explorer`（explorer）。

```bash
python3 agent_harness/a2a/agent_card.py --check        # 分類與卡片的一致性
python3 agent_harness/a2a/agent_card.py --agent cgc-explorer
python3 agent_harness/a2a/a2a_server.py --call cgc-explorer register-decision --text "..."
python3 agent_harness/a2a/a2a_server.py --self-test    # 17 格
```

## 與既有東西的整合面（這一輪做到哪）

| 對象 | 關係 |
|---|---|
| `pd/`（gRPC 推論子系統） | **不改它**。它在本套件裡是一個 backend：`pd-inference` 這個 agent 是它的對外身分。★ 目前只映射了「健康狀態／會話查詢」這一類；`AllocateBlocks`／`StorePrefixKV` 等 KV 操作**還沒有** A2A 對應的 skill（見下面「不主張什麼」）。 |
| `portal/fleet.json`（機器） | agent 的 `runs_on` 指向 fleet 的端點 id。**兩個正交的軸**：fleet 回答「這台機器上有什麼」，registry 回答「誰在跑」。併成一張表就再也回答不了「這台機器上有誰」。 |
| `tb_loop/agents/`（執行器） | 那三個是 **CLI 適配器**（terminal-bench 的 `AbstractInstalledAgent`），不是類別。它們可以被 executor 包起來，但目前**沒有**接。 |
| `fleet_auto.py`／看門狗 | 無耦合。運營 agent 的 `watchdog-check` skill 是**呼叫它**，不是重寫它。 |

## 不主張什麼

- **不是四個獨立部署**。四個 agent 共用同一個閘道進程與同一份 registry；它們在協議上是四個
  身分，在部署上是一個服務。要拆成四個進程，registry 的路徑前綴已經為此準備好了。
- **pd 的 KV/DOPD 操作沒有全部映射過來**。`pd-inference` 目前的能力是它的**身分**，
  不是它的全部 rpc。說「整合完成」會是假的。
- **push 沒有重試**。投遞失敗會留痕，但不會重送 —— 重試策略需要一個「什麼算送達」的判準，
  而那個判準目前不存在。
- **沒有認證**。閘道預設綁 `127.0.0.1`。要對外就得先有 `--api-key` 這一層，
  目前**沒有**（與 `edge_server` 的 `--api-key` 是兩套，未對齊）。
- **executor 只有 12 個 ＋ 1 個擴展**。它們是為了讓每一格微觀能力「真的能跑」而寫的，
  不是生產實作。`shell-allowlist` 只放行白名單命令，且不做管線與重導向 ——
  `shell=True` 在這個情境等於把閘道變成遠端 shell。

## 自測 17 格（黑箱：真的起服務、真的發 JSON-RPC、真的收 SSE、真的收 webhook）

| 格 | 在驗什麼 |
|---|---|
| 1–3 | 目錄卡與 registry 雙向對齊；每張 card 的 `x-agent-class` 與 taxonomy 一致；★ `skills` 與 `x-taxonomy.skill_owner` 一致（**對端可以自己核對分類**） |
| 4 | `message/send` ⇒ completed 且帶 artifact |
| 5 | ★★ 把 explorer 的 skill 送給 developer ⇒ 拒收並指名 owner |
| 6 | ★ 沒指名 skill ⇒ 拒收（不猜預設） |
| 7–8 | 未知 agent ／ 未知 task 的錯誤碼 |
| 9 | 取消已完成的任務 ⇒ 拒絕（-32002） |
| 10 | ★★ SSE 逐步收到事件，最後一個是 final |
| 11 | ★★ push 四種操作 ＋ **webhook 真的收到 POST** |
| 12 | ★ 卡片上有但沒有 executor ⇒ `failed` 並附原因 |
| 13–14 | 壞 JSON ⇒ -32700；未實作方法 ⇒ -32601 |
| 15 | ★★ taxonomy 的每一個 skill 都有 executor |
| 16 | ★ 每個 agent 都寫了 `not_capable` |
| 17 | `/healthz` 與 404 |

★ 第 15 格是最重要的一格：taxonomy 裡多一個 skill 而漏了 executor，症狀**不是報錯**，
是「那個能力永遠不會被執行」——而它在卡片上看起來完全正常。
