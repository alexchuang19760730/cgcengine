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

JSON-RPC 方法：

| 方法 | 做什麼 |
|---|---|
| `message/send` | 同步執行一個 skill ⇒ 完成後回 artifact |
| `message/stream` | 同上，但用 SSE 逐步回報（最後一個事件是 final） |
| `tasks/get` / `tasks/cancel` | 查／取消任務（已完成 ⇒ 取消被拒 `-32002`） |
| `tasks/pushNotificationConfig/{set,get,list,delete}` | webhook 設定（投遞**真的** POST；失敗留痕但不改任務狀態） |
| `agent/brief` | ★★ 這一類的**資產／能力／進度／復盤**，並附**出處**；帶 `email`／`jwt` ⇒ 換一類 |
| `identity/check` | 身分錨 vs 雲側帳號。`asClass`＝用哪一類帳號讀；`probeLogin`＝**明示才做**登入探測 |
| `identity/here` | ★★ 出處（repo／HEAD／未提交改動／錨文件）——**不是**身分 |

四個 agent：`cgc-dev-porting`（developer）、`fleet-operator`（operator）、
`pd-inference`（operator）、`cgc-explorer`（explorer）。

```bash
python3 agent_harness/a2a/agent_card.py --check        # 分類與卡片的一致性
python3 agent_harness/a2a/agent_card.py --agent cgc-explorer
python3 agent_harness/a2a/a2a_server.py --call cgc-explorer register-decision --text "..."
python3 agent_harness/a2a/a2a_server.py --self-test    # 23 格
```

## 與既有東西的整合面（這一輪做到哪）

| 對象 | 關係 |
|---|---|
| `pd/`（gRPC 推論子系統） | **不改它**。它在本套件裡是一個 backend：`pd-inference` 這個 agent 是它的對外身分。★ 目前只映射了「健康狀態／會話查詢」這一類；`AllocateBlocks`／`StorePrefixKV` 等 KV 操作**還沒有** A2A 對應的 skill（見下面「不主張什麼」）。 |
| `portal/fleet.json`（機器） | agent 的 `runs_on` 指向 fleet 的端點 id。**兩個正交的軸**：fleet 回答「這台機器上有什麼」，registry 回答「誰在跑」。併成一張表就再也回答不了「這台機器上有誰」。 |
| `tb_loop/agents/`（執行器） | 那三個是 **CLI 適配器**（terminal-bench 的 `AbstractInstalledAgent`），不是類別。它們可以被 executor 包起來，但目前**沒有**接。 |
| `fleet_auto.py`／看門狗 | 無耦合。運營 agent 的 `watchdog-check` skill 是**呼叫它**，不是重寫它。 |

## 身分錨：session 屬於哪一類（以及那一類的四維）

「這個 session 是哪一類」如果靠它**自己宣稱**或靠 session 標題，它就是不可機檢的。
最不會漂移的判準是**它用哪一個帳號登入雲側**（Supabase `profiles`）：

| 類別 | 帳號 | 雲側角色 |
|---|---|---|
| `operator` 運營 | `alexchuang@powerauto.ai` | `super_admin` |
| `developer` 開發 | `developer@powerauto.ai` | `super_admin` |
| `explorer` 探索 | `frontier@powerauto.ai` | `super_admin` |

★ **只做精確比對。** `nobody@powerauto.ai`（同網域但不在錨上）⇒ **拒答**，不會被吸進某一類。
★ **判不出來就拒答**：猜一個預設類別，會讓一個打錯帳號的 session 拿到「運營者」的四維，
而在它自己看起來完全正常。

> ★★ 2026-09-18 16:38 實況：三個帳號都是 `super_admin`。
> 所以 **role 不承擔「區分類別」的職責，區分只在 email 上** ——
> 如果有人以為「role 低的那個是開發者」，他會做錯判斷。
> `identity.py --check` 拿實況比對：角色不符報「不一致」、帳號不存在報「absent」，**兩件事分開講**。

### 四維的形狀一樣，**來源不同**

`agent/brief`（JSON-RPC）回一個類別的**資產／能力／進度／復盤**：

| 類別 | 資產 | 能力 | 進度 | 復盤 |
|---|---|---|---|---|
| operator | 機隊**全部**端點 | 所有端點已驗證的能力 | 各端加入單完成度 | 逐次上報的能力／未量測變化 |
| developer | 只看看**移植目標** | 那些端的建置／移植能力 | 那些端的加入單 | 它們的建置變化 |
| explorer | 決策與假設（不是端點） | 4 個決策類 skill | decisions 的 pending 數 | `momentum_block()` 實算的動能 |

★ developer 的判準是 **`reaches == "offline"`**（我們連不上、得主動送過去），
不是 `platform` 白名單。第一版我用「platform 不是 darwin」篩，把 cloud-host2（linux 雲端主機）
也算了進來 —— 而它根本不需要移植。

```bash
python3 agent_harness/a2a/identity.py --email alexchuang@powerauto.ai --brief   # 運營者的四維
python3 agent_harness/a2a/identity.py --jwt "$TOKEN" --brief                    # 從 JWT 判別
python3 agent_harness/a2a/identity.py --check                                   # 身分錨 vs 雲側帳號（含角色）
python3 agent_harness/a2a/identity.py --check --as-class explorer               # 用探索者帳號**自己**去看
python3 agent_harness/a2a/identity.py --probe-login                             # 三個帳號各登入一次（★ 只讀）
python3 agent_harness/a2a/identity.py --here                                    # 這一輪的**出處**（見下）
```

```bash
# 透過 A2A 問（閘道要跑著）
curl -s -X POST http://127.0.0.1:9210/a2a/fleet-operator -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"agent/brief","params":{}}'
# 帶 email ⇒ 換一類（讓運營者用**別的**身分問同一件事）
#   "params":{"email":"frontier@powerauto.ai"}
```

### 出處（不是身分）：為什麼**不做** session 註冊

★★ 2026-09-18 決定：**身分只到「帳號」這一層，不做 session 級註冊。**
同一帳號同時開兩個 session（一個做移植、一個做運營）會拿到**同一類**的四維 ——
這是**有意接受**的，不是漏掉的功能。

理由是：能區分兩者的東西（**本機檔案、git 操作**）本來就擺在那裡，而且兩個 session
看到的是**同一份**。再加一層註冊表只會多一份會漂移的真相（這個 repo 已經為「兩份真相」付過學費）。

所以四維帶的是**出處**：

```bash
python3 agent_harness/a2a/identity.py --here
#   出處（★ 這不是身分，是歸因）：
#       repo    /Users/alexchuang/Documents/flashkv-devserver
#       HEAD    demo/sweet-spot-windows-fix 22e7cb9919d3
#       未提交  8 筆（分不出是誰改的：多 session 共用一棵樹）
#       錨文件  7 個讀得到、0 個讀不到
#       ★ 同一帳號的多個 session 會判到**同一類**（level=none, byDesign=True）
```

於是同帳號的兩個 session 雖然同類，卻可以互相知道「**你讀的是哪一版**」——
歸因靠出處，不靠註冊。`agent/brief` 與 `identity/here` 都回這一包，
而 `sessionDiscrimination.level` 固定是 `"none"`：自測有一格釘住它，
免得之後有人偷偷加回一個註冊表、卻忘了改這個宣告。

「錨文件」就是四維的來源清單（`taxonomy.py`／`fleet.json`／`decisions.jsonl`／
`fleet_export.json`／端側的 `edge_server.py`…）。**缺檔也留在清單裡**（標 `exists:false`）——
從清單消失會讓「還沒匯出」與「我忘了檢查」長得一樣。

### 憑證怎麼放（**不要**貼進 repo）

`identity.py` 的原始碼裡**沒有任何 key**，憑證只從環境變數或
`~/.config/powerauto/supabase.env`（repo 外、`chmod 600`）讀。而且它**刻意只用 anon key
＋ 各類帳號自己的 JWT**，不用 Service Role Key —— 這樣讀到的就是那條路徑**真的讀得到**的東西，
RLS 在雲側那一頭生效。

一個類別**一個帳號**，密碼共用同一把（2026-09-18 機檢：三個都能登入、各有自己的 UUID）：

```bash
POWERAUTO_PASSWORD=...
POWERAUTO_EMAIL_OPERATOR=alexchuang@powerauto.ai
POWERAUTO_EMAIL_DEVELOPER=developer@powerauto.ai
POWERAUTO_EMAIL_EXPLORER=frontier@powerauto.ai
```

★ 使用者說「同樣的密碼」，但**說不是證據** —— `--probe-login` 真的各登一次
（**只讀**：不建帳號、不改角色）。三個帳號可能沒建、密碼不同、或某一個被停用，
而這三種情況的處置完全不同。

★ 已知缺口：`docs/fleet_export.json` **沒有導出 `reaches`**（portal 的 `export_payload()` 的缺口），
所以這支得自己回頭讀 `fleet.json`。一份「給別的網站吃」的資料少了判斷所需的欄位，
每個消費者都要多讀一個檔 —— 值得之後在 export 補上。

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

## 自測 23 格（黑箱：真的起服務、真的發 JSON-RPC、真的收 SSE、真的收 webhook）

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
| 18–19 | ★★ 身分錨：`agent/brief` 的四維按**它那一類**整理並帶身分錨；換 `email` ⇒ 換一類，不認得的帳號 ⇒ 拒答 |
| 20 | ★ `identity/check`（`noCloud`）⇒ 明說沒查，不是靜默通過 |
| 21 | ★★ `asClass`：不在類別裡 ⇒ 拒答（`-32602`）；合法的要記下來 |
| 22 | ★★ `probeLogin` **明示才做**（不給就沒有 `loginProbe`）；給了 ⇒ 三類各一列 |
| 23 | ★★ `identity/here` ⇒ 出處帶 repo／HEAD，且 `sessionDiscrimination.level=none` |

另有兩支獨立自測：`identity.py`（**17 格**）與 `agent_card.py --check`。

★ 第 15 格是最重要的一格：taxonomy 裡多一個 skill 而漏了 executor，症狀**不是報錯**，
是「那個能力永遠不會被執行」——而它在卡片上看起來完全正常。

★ 第 22 格防的是另一種：一個**每次都被順手做**的網路動作（登入探測），
會讓「沒查」這件事從輸出裡消失。
