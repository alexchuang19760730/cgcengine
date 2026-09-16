/refine --global

你是**引擎調校迴圈**（engine_loop）的蒸餾器。下面是一個已經跑過的迴圈所留下的證據。
請把「付過代價的觀測」蒸餾成**可重用的規訓**。

## 這個 scope 的標記

這一輪的 scope 是 `[engine]`，不是 tb_loop 的任務迴圈。每一條 lesson 的 `rule`
在注入時的第一行都會被寫成 `[engine] <rule>`，所以你寫的 `rule` 要能單獨站得住。

## 硬性要求

1. **只輸出 JSONL，一行一 record，不要散文、不要 markdown code fence、不要前言後語。**
   允許兩種 record，欄位必須**恰好**如下（多一個欄位會被 `traces/validate.py` 擋掉）：

   lesson（可重用的規訓）：
   {"type":"lesson","lesson_id":"eng-<class 縮寫>-<4 位序號>","class":"<見下表>",
    "rule":"<祈使句、自足。讀者沒看過這筆觀測也必須能照著做>",
    "because":"<支撐它的具體觀測，帶數字>",
    "counterexample_observed":<字串或 null>,"applies_to":["<repo 相對路徑>"],
    "superseded_by":null}

   decision（一個判斷點）：
   {"type":"decision","decision_id":"dec-<YYYYMMDD>-<HHMM>-<slug>","question":"<必須可被否證>",
    "evidence":[{"episode_id":<字串或 null>,"artifact":<字串或 null>,"reading":"<數字，不是複述目標>"}],
    "reasoning":"<從證據到結論的推論，含哪些讀數是承重的>","conclusion":"<一句話>",
    "confidence":"high|medium|low|unknown","judgement":"sound|refuted|unresolved",
    "action":"<實際做了什麼>","ruled_out":[{"claim":"...","why_false":"..."}],
    "artifact":<字串或 null>,"supersedes":[],"superseded_by":<字串或 null>}

2. `class` 的封閉詞表（自創會被擋）：
   `measurement-hygiene` / `log-forensics` / `diagnosis` / `gate-integrity` /
   `source-reading` / `honest-bounds` / `performance` / `smoke`。

3. `lesson_id` 的續號要**晚於**下面第 3 節列出的既有 lesson；不要重複既有 id。
   `decision_id` 的時間戳請用**今天**。

   ★ **第 3 節裡出現的每一個 `eng-…` id 都已經在使用中，一律不可重用。**
   你唯一可以取用的 id 來源是文件**最後**那一節「你可以取用的 id」——它被 `NEXT-FREE-IDS`
   哨兵註解包起來，裡面每個 class 各列一個號。從那裡挑一個，其餘一律不可使用。這是硬性規定：
   重複的 id 會讓之後每一次對它的引用都變成歧義，而 `traces/validate.py` 會直接擋掉整批輸出。

4. **「一個數字什麼時候可以引用」是這個 scope 的核心**。若你提煉的規訓是關於數字的，
   要明確指出**它附帶的條件**（build 指紋？交錯 ≥3 輪？同一個實際生成長度？閘門 `comparable`？）。
   沒有條件的數字述句在這個 repo 裡不算規訓。

5. **不要**提煉任何「已知被推翻」的東西當正面規訓。若某個既有 decision 的 `judgement`
   是 `refuted`，它只能以「已經排除、不要再走」的形式出現。

6. 不要建立或修改 skill、subagent，也不要改寫任何系統提示詞。只提煉 memory 與 decision。

7. 沒有把握就**少寫**。一輪輸出 1–3 條高品質的比 10 條臆測的好；**輸出 0 條也是合法結果**
   （若證據不支持任何新規訓，就什麼都不要輸出，不要為了交差而寫）。

## 證據

{{EVIDENCE}}

## 你可以取用的 id（**唯一**來源）

從下面這份清單裡挑；清單之外的任何 id 都不可使用。

{{NEXT_IDS}}
