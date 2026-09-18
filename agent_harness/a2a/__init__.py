# -*- coding: utf-8 -*-
"""`agent_harness/a2a/` —— 把 harness 與 pd 服務暴露成 A2A agent 的閘道。

這個套件回答一個問題：**「對面那個 agent 是哪一類、能決定什麼、能執行什麼？」**
——而且答案必須是**機檢的**（別的 agent 在呼叫前就能讀到），不是寫在文件裡的宣稱。

四個檔：
  - `taxonomy.py`    三類 agent（開發／運營／探索）的分類體系，macro（宏觀決策域）
                     與 micro（微觀能力）的**單一真相**
  - `registry.json`  有哪些 agent **實例**，各自屬哪一類、接哪些 skill、在哪裡跑
  - `agent_card.py`  把上面兩者編譯成 A2A 的 Agent Card（標準欄位 ＋ `x-agent-class`
                     ／`x-macro-scope` 擴展），並做嚴格校驗
  - `a2a_server.py`  JSON-RPC 閘道：`message/send`、`message/stream`（SSE）、
                     `tasks/get|cancel`、`tasks/pushNotificationConfig/*`
                     —— 含真實的 webhook 投遞

`agent_harness/pd/` 是引擎側的 PD 推論子系統（gRPC），**不是** harness 的一條迴路；
它在本套件裡只是一個 backend，透過 executor 抽象接進來，本身不需要改。
"""
