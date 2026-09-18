# `agent_harness/portal/` — 目標／決策／驗證／資產，同一頁

這一頁回答三個問題，而且**每一題的答案都是一個可重跑的指令**：

| 問題 | 來源 | 為什麼不是文件 |
|---|---|---|
| 我們在追求什麼、已經決定了什麼？ | `goals.json` ＋ `engine_loop/traces/decisions.jsonl` | 決策有 `judgement`（sound／refuted）與取代鏈 ⇒ 是**可復盤的狀態**，不是流水帳 |
| 驗證是不是真的跑過、而且能重跑？ | `gates.json`，**入口實跑** | 綠燈必須是「剛剛跑出來的 rc=0」，不是「上次看到是綠的」 |
| 資產有多少、有沒有在動？ | 實掃記憶／skill／代碼／軌跡／衍生物 | 數字由掃描得到，不由人填 |

## 檔案

```
agent_harness/portal/
├── build_portal.py    產生器（唯一入口）
├── gates.json         ★ 驗證閘門的單一真相來源
├── goals.json         ★ 目標樹（手維護，但指針受機檢）
├── data.json          本輪的機器可讀快照（~18 KB；下一輪的「上一次」）
├── history.jsonl      **只追加**的量化歷史（趨勢圖來源）
└── README.md          本檔
docs/AGENT_HARNESS_PORTAL.html   ★ 產物：自帶樣式與腳本，離線可開
```

## 用法

```bash
python3 agent_harness/portal/build_portal.py               # 跑快速閘門 ＋ 產 portal（預設）
python3 agent_harness/portal/build_portal.py --gates-only  # 只跑閘門並印表；任何紅燈 ⇒ rc=1
python3 agent_harness/portal/build_portal.py --no-gates    # 不跑閘門（離線／趕時間）
python3 agent_harness/portal/build_portal.py --check       # 只驗註冊表與目標樹指針，不寫任何檔
python3 agent_harness/portal/build_portal.py --self-test   # 15 格黑箱自測（含陰性對照）
```

`--gates-only` 是目前這個 repo 最快的一條「整體健康檢查」：**14 條快速閘門、約 2 秒**，
涵蓋憲章引用、共用檢查器、軌跡 schema、衍生物、索引、快照、portal 自身。

## 三條硬規矩（每一條都對應一個已付過的學費）

**① 單一真相來源。** 閘門清單只在 `gates.json`。portal 顯示的「復現指令」與
`--gates-only` 實際跑的是同一份 ⇒ **不存在「文件寫的指令已經不是實際跑的那條」**。
新增閘門時要填 `proves`（證什麼）、`not_proves`（**不**證什麼）、`expect`（什麼算通過）。

**② 不假裝。** `cost=heavy` 的閘門（需要 GPU／build 產物／docker／模型端點）**不自動跑**，
在表上是 `SKIP` 並附前題，**不會被畫成綠燈**。把跑不動的閘門畫綠，比沒有閘門更糟 ——
本 repo 的提交閘門自己就有這個著名陷阱（不帶 `BIN_DIR` ⇒ 11 段全 SKIP 卻仍印 `OK`）。

**③ 缺席要出聲。** 沿用 `CONVENTIONS.md:7-14` 的驗收形式：目標樹每一節要嘛有
**可解析**的指針（可帶 `contains` 子串，因此不怕行號漂移），要嘛有一行 `evidence_note`
明說它的證據形態。閘門同理：快速閘門沒有自測時要交代理由，或它的 `cmd` 本身就是 `--self-test`。
**缺口從來不是「沒有檔案」，是沉默。**

## `--self-test` 的 15 格

| 格 | 在驗什麼 |
|---|---|
| A–E | 目標樹：懸空指針／**沉默（無指針也無標記）**／`contains` 漂移／`blocked` 缺 `blocked_by` ⇒ 都必須 rc=1 |
| F–H | 閘門註冊表：缺 `not_proves`／fast 無自測又無理由 ⇒ rc=1；壞 JSON ⇒ rc=1 且不 traceback |
| I | `--check` **不寫任何檔**（比對跑前後所有檔的 mtime_ns） |
| J | `history.jsonl` **只追加**（兩次執行 ⇒ 兩行，第二次不覆蓋第一次） |
| K | HTML 產出且四個分頁錨點齊全 |
| L | `--no-gates` ⇒ **沒有任何 PASS**（不跑就不准宣稱綠） |
| M | 有紅燈 ⇒ `--gates-only` rc=1 且列出 id |
| N | heavy 閘門 ⇒ `SKIP` 而不是 `PASS` |
| O | ★ **自我測試不寫任何檔到真正的 repo** —— 子行程必須帶 `PA_PORTAL_REPO` |

**O 格存在的原因是真實的**：本檔第一版忘了把 `PA_PORTAL_REPO` 傳給子行程，於是「跑自我測試」
等於「在真正的 repo 原地產生一份 portal」（`data.json` 與 `history.jsonl` 真的被寫出來了，
而 13 格 fixture 全部「通過」）。**自測有副作用**是這個 repo 最不能接受的失效模態之一，
所以它現在有自己的陰性對照。

## 怎麼加東西

**加一條閘門** → 在 `gates.json` 的 `gates` 陣列加一筆，然後跑 `--check`。
欄位缺漏會被擋（`proves`／`not_proves`／`cost` 都是必填）。

**加一個目標** → 在 `goals.json` 的樹裡加一節。`acceptance`（什麼算完成）必填，
`status=blocked` 必填 `blocked_by`。指針建議用 `{"path": ..., "contains": "..."}` 形式，
因為它對行號漂移免疫。

**加一個 lesson 分類／決策** → 不用動這裡。決策樹與 lesson 分類是從
`engine_loop/traces/*.jsonl` **衍生**的。

## 已知邊界（不主張什麼）

- **`cost=heavy` 的閘門在本頁是未知狀態**，不是綠。要它們的狀態就得手動跑並看輸出。
- **「重跑就會得到同一個結果」只在同一 `fingerprint` 下成立**。頁首的輸入指紋是
  `lessons.jsonl`／`decisions.jsonl`／`episodes.jsonl`／`CONVENTIONS.md`／`gates.json`／
  `goals.json` 六個檔的 sha256 前 12 位串接。指紋不同 ⇒ 兩次執行**不可比較**。
- **`index-memory` 與 `snapshot-freshness` 的輸入含多 session 共寫的每日日誌** ⇒
  另一個 session 正在 append 的那一刻它們會紅，**而那個紅是對的**（快照真的落後了）。
  這兩條在 portal 上會多一個「輸入易變」標籤。
- **目標樹的措辭對不對，機器驗不了** —— `--check` 只驗它有沒有指針、有沒有 `acceptance`。
- **`versions/` 是歷史存檔**（3835 檔 / 1.55M 行），**不計入** headline 的「現行代碼 LOC」。
