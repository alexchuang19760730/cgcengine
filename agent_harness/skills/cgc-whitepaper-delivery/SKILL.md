---
name: cgc-whitepaper-delivery
description: 在 flashkv-devserver（TurboFieldfare / llama.cpp CGC fork）撰寫 docs/*.html 技術白皮書的版式、取材清單與交付流程。當要求為該 repo 產生白皮書 / 技術架構文件 / 里程碑文件、或 D5 要求「每次 commit 附一份白皮書」時使用。也適用於不確定 docs/ 底下的文件該長什麼樣、數字該從哪裡來、以及新增一份 docs/*.html 會不會弄紅索引閘門。
agent_created: true
---

> **這是快照，不是權威副本。**
> 權威位置：`~/.workbuddy/skills/cgc-whitepaper-delivery/SKILL.md`（由 host 持續寫入）。
> 本檔於 2026-09-17 由 `agent_harness/scripts/import_harness_snapshot.py` 複製進 repo，唯一目的是讓 `agent_harness/`
> 底下的內容能被 `agent_harness/scripts/auto_git_push.ps1` 定時推送；原檔改了這裡**不會**自動跟上。
> 要改 skill 請改原檔，再重跑 `python3 agent_harness/scripts/import_harness_snapshot.py`。

# flashkv-devserver 技術白皮書交付

專案：`/Users/alexchuang/Documents/flashkv-devserver`
（**是 git worktree**，`.git` 是一個檔案 → `.../flashkv0516/.git/worktrees/flashkv-devserver`）

## 30 秒版

```sh
cd /Users/alexchuang/Documents/flashkv-devserver

# 0) 先確認沒有別的 session 正在同一個 repo 寫檔（見陷阱六，這條最容易被忽略）
git status --porcelain --untracked-files=all

# 1) 跑現場閘門，把輸出「原文」貼進白皮書（不要轉述）
python3 agent_harness/engine_loop/traces/validate.py
python3 agent_harness/engine_loop/traces/selftest.py
python3 agent_harness/engine_loop/memory/build_memory_index.py --check
python3 agent_harness/engine_loop/index_assets.py --check      # 要濾 OK:/error

# 2) 寫 docs/<NAME>.html（版式照抄 docs/PREFILL250_CONDITIONAL_DELIVERY_20260916.html）

# 3) 驗結構（見「交付前自檢」）
```

**D5（`CONVENTIONS.md:889`）要求每次 commit 附一份技術白皮書**，所以這條路每輪都會走一次。
格式沿用既有 HTML 版式。

---

## 1. 版式：照抄對象與硬性規範

**照抄對象**：`docs/PREFILL250_CONDITIONAL_DELIVERY_20260916.html`（＋ 它的 `.md` 雙生檔）。
這是最近的「房內風格」基準：**淺色、`lang="zh-Hant"`、繁中技術中文**。

硬性規範（逐項都有既有檔案為證）：

| 項目 | 規定 | 為什麼 |
|---|---|---|
| 語言 | `<html lang="zh-Hant">`，內文繁體中文 | 與 `docs/*.html` 全體一致 |
| 主題 | **淺色**（`--bg:#ffffff`），`body` 用 `-apple-system,"PingFang TC"` 字型堆疊 | 列印與截圖友善；深色只出現在更早的別支白皮書 |
| 寬度 | `.wrap{max-width:1040px}` 上下留白 `32px 20px 80px` | 與同族檔案並排時不會跳動 |
| 圖 | **內嵌 `<svg>`，不要載 CDN 的 mermaid** | 使用者在中國大陸，CDN 不穩；離線可讀 |
| 程式碼 | `<pre><code>` ＋ 等寬字型堆疊；行內用 `<code>` | 同族 |
| 表格 | `<table>` ＋ `th{background:#f9fafb}`；數字欄加 `.num`（`white-space:nowrap`） | 數字不可斷行 |
| 標籤 | `.tag` 膠囊（`t-blue/t-green/t-red/t-amber/t-purple/t-gray`） | 狀態一眼可掃 |
| 提示框 | `.box`（`.warn`/`.ok`/`.bad`/`.pur` 變體），左側 4px 色條 | 同族 |
| 檔名 | `<TOPIC>_<YYYYMMDD>.html`，同日多版本加 `_HHMM` | 見陷阱一 |

**不要**引入 JS（除必要）；不要引入外部字型；不要用 emoji。

---

## 2. 取材清單（這是白皮書的「事實來源」，不是參考資料）

白皮書的每一句話都要指得到下列其中之一。**動筆前先讀完，不要憑印象寫。**

### 2.1 判準與規劃（必讀）

| 檔案 | 取什麼 |
|---|---|
| `agent_harness/CONVENTIONS.md` | **判準憲章**：A（數字什麼時候可以引用，A1–A20）、B（診斷）、C（讀原始碼）、D（流程，D1–D6）、E（自我約束）。87 KB。用小節標題定位，不要整檔讀 |
| `agent_harness/engine_loop/README.md` | 兩個 loop 的同構表、紅線（不得複製 `scripts/check/*`）、三種 record 與三條硬規則、E0–E4 現況、**已知的誠實邊界**（第 7 節） |
| `agent_harness/PLAN_ENGINE_LOOP_2026-09-15.md` | §9 的 E0–E4 驗收表（可機檢條件）、§10 的 R1–R6 風險 |
| `agent_harness/README.md` | `tb_loop` 的既有設計（Terminal-Bench × prime-agent），以及它的誠實聲明 |

### 2.2 索引與痕跡（計數要現場算，不要複製舊文件的數字）

| 檔案 | 取什麼 |
|---|---|
| `agent_harness/engine_loop/MANIFEST.jsonl` | 資產清單與 `role` 十值分佈；**`role` 是人工判斷，其餘自動推導** |
| `agent_harness/engine_loop/memory/INDEX.jsonl` | 段落級定位的形狀（`path/line_start/line_end/subs[]`） |
| `agent_harness/memory/SNAPSHOT.jsonl`、`agent_harness/skills/SNAPSHOT.jsonl` | 快照的 `source_sha256`／`source_mtime` 語意 |
| `agent_harness/memory/README.md`、`agent_harness/skills/README.md` | 「權威在哪、快照是非權威」的三條防線 |
| `agent_harness/engine_loop/traces/{episodes,decisions,lessons}.jsonl` | 三種 record 的真實樣本與**分類分佈** |

現場採數（每次都要重跑，數字會變）：

```sh
python3 - <<'EOF'
import json, collections
for f,key in (('episodes.jsonl','usable_as_evidence'),('decisions.jsonl','judgement'),('lessons.jsonl','class')):
    p='agent_harness/engine_loop/traces/'+f
    rows=[json.loads(l) for l in open(p) if l.strip()]
    print(f, len(rows), collections.Counter(r.get(key) for r in rows))
EOF
```

### 2.3 指紋與戳記（白皮書最核心的素材）

七個機制的實作與 docstring 就在這些行上——**docstring 本身就是最好的白皮書文字**，因為它記錄了
「當初為什麼要這樣做」：

| 機制 | 位置 |
|---|---|
| `build_fingerprint()` | `scripts/check/decode_sweep.py:34`（`ab_interleave.py:42` 委派同一份） |
| `binary_stamp()` | `scripts/check/knifeedge_matrix.py:469` |
| `source_stamp()` | `scripts/check/knifeedge_matrix.py:974`（`NUMERIC_SOURCES` 在 `:957`） |
| `pool_geometry_stamp()` | `scripts/check/knifeedge_matrix.py:447`（`POOL_GEOMETRY_SOURCES` 在 `:440`） |
| `model_stamp()` | `scripts/check/knifeedge_matrix.py:1051` |
| `mtp_head_identity.fingerprint()` | `scripts/check/mtp_head_identity.py:198` |
| A11 啟動環境指紋 | `CONVENTIONS.md:178` ＋ `scripts/run_server.sh:1497` |
| M1/M2/M3 判決 | `scripts/check/m123_oracle_gate.py:1`（定義）與 `DEFAULT_REF`／`ORACLE_PINNED_ENV` |

### 2.4 技能（方法論的成熟形式）

`~/.workbuddy/skills/cgc-commit-gate/SKILL.md`、`cgc-decode-attribution/SKILL.md`、
`cgc-prefill-thermal-delivery/SKILL.md` — 引它們可以省掉整節的重述。

---

## 3. 現場閘門輸出：**必須真的跑，且原文貼上**

這是「可驗收」的唯一具體含義。白皮書必須有一節把**產出當時**的閘門輸出原文列出，
而不是「四條閘門都通過」這種轉述。

```sh
cd /Users/alexchuang/Documents/flashkv-devserver

python3 agent_harness/engine_loop/traces/validate.py                     # 122 / 38 / 98 → OK
python3 agent_harness/engine_loop/traces/selftest.py                     # 10/10 rejected
python3 agent_harness/engine_loop/memory/build_memory_index.py --check    # 3 file(s), N section(s)
python3 agent_harness/engine_loop/index_assets.py --check                 # manifest OK: N assets
```

**要白皮書談引擎效能時**，另外跑（成本高，自己起 server）：

```sh
lsof -nP -iTCP:8080 -sTCP:LISTEN && pgrep -fl llama-server   # 必須都空
python3 scripts/check/m123_oracle_gate.py --tag <標籤>        # ~40–60 s；M1/M2/M3 9/9
```

規則：**引用前先讀 gate 的 `comparable` 欄位**。`comparable=False` 時那三個 9/9 是
*printed for information, NOT a verdict*，且 gate 會 `exit 2`（INVALID COMPARISON）而不是 `exit 1`。

---

## 4. 陷阱

### 陷阱一（使用者特別點名）：同日同主題的多版本，檔名要指名到 `_HHMM`

本 repo 已經踩過這個。證據：

```
PREFILL250_DECODE25_WHITEPAPER_2026-09-15.html          ← 舊式，無時刻
PREFILL250_DECODE25_WHITEPAPER_20260915_1745.html       ← 有時刻
PREFILL250_DECODE25_WHITEPAPER_20260915_1810.html
PREFILL250_DECODE25_WHITEPAPER_20260915_1900.html
PREFILL250_DECODE25_WHITEPAPER_20260915_2240.html
PREFILL250_DECODE25_WHITEPAPER_20260915_2355.html
```

同一天同一個主題被重寫了五次。**只要「同一天、同一個 TOPIC 可能再寫一次」，就要帶 `_HHMM`**，
否則後一份會與前一份同名（或要靠 `_2026-09-15` vs `_20260915` 這種不一致的寫法區分，
讀者無法判斷哪份是最終版）。

對照組：`SKIP0_VALUE_SEMANTICS_20260916_0021.html` 也是帶時刻的；
而 `ZERO_REGION_TRIAGE_20260916.html`、`PREFILL250_THERMAL_TRANSIENT_20260916.html`
是單發型，不帶時刻——**單發可以省，會再修的主題必須帶。**

推論：若白皮書在**同一輪內**就被修訂，就直接覆蓋並在文末記修訂；
跨輪修訂才新開一個 `_HHMM` 檔（因為 D5 要求它跟 commit 綁定，舊檔是舊 commit 的附件）。

### 陷阱二：新增 `docs/*.html` **不會**讓 `index_assets.py --check` 漂移

`index_assets.py` 的自動 glob **只掃 `scripts/check/*`**（`:314`），
`docs/` 只由 `CURATED` 逐條列出（`:227` 起）。所以：

- 新增一份 `docs/*.html` → **不會**紅，也不必重生 manifest。已實測：新增後 `--check` 仍印
  `manifest OK: 73 assets`。
- 但**編輯 `CURATED` 裡已列出的 docs 條目**（例如 `docs/PREFILL250_THERMAL_TRANSIENT_20260916.html`
  的 note）→ 會紅，要重生 manifest。
- 改 `agent_harness/CONVENTIONS.md`、`agent_harness/PLAN_ENGINE_LOOP_*.md`、
  `traces/*.jsonl`、`.workbuddy/memory/*.md` → 都會紅。

不確定就跑 `--check`，它會直接印出漂移的 path 與 bytes 變化。

### 陷阱三：**不要**把白皮書寫進 repo 的 `.workbuddy/memory/`，也**不要**手動加快照

- 寫進 `.workbuddy/memory/` 會觸發 INDEX/MANIFEST 的重生義務（D6）。
- `agent_harness/memory/` 與 `agent_harness/skills/` 的 dated 快照**只能**由
  `python3 agent_harness/scripts/import_harness_snapshot.py` 產生——手動新增會造出一份沒有
  `SNAPSHOT.jsonl` 對應列的檔案，那正是 D6 要防的「第二個權威來源」。
- 要讓白皮書／筆記跟 repo 走，就放 `docs/`（會被 commit）；要跨機器帶記憶，跑匯入腳本。
- 那支腳本 2026-09-16 從 `Backup/import_harness_snapshot.py` 搬到版控裡（`Backup/` 被
  `.gitignore:396` 排除 ⇒ 舊位置上的東西**進不了 commit**，clone 出來的機器上沒有它）。
  同一輪把它的兩個硬編碼清單改成 glob：**新增 skill 或新增一天的日誌都不需要改任何清單**。
  **寫白皮書時要引用正確的路徑**，否則你就成了那個「把死路徑抄進文件」的人。

### 陷阱四：`index_assets.py --check` 的結論行會被說明文字包住

直接 `tail` 只會看到尾端那段 `--out` 的說明。要濾：

```sh
python3 agent_harness/engine_loop/index_assets.py --check 2>&1 | grep -E "OK:|error|drift"
```

### 陷阱五：**不要**用 `index_assets.py --out`，也不要為了「看一下」而跑不帶 flag 的重建

不帶 `--check` 就是**重建**（會覆寫 `MANIFEST.jsonl`）。要看 by-role 分佈而跑重建，
就會在別的 writer 也在動這個 repo 時插入一次寫入（見陷阱六）。
`--out` 是 cwd 相對路徑，會寫出**第二份 manifest** 而讓原檔 stale。

### 陷阱六：動筆前先確認**沒有另一個 session 正在同一個 repo 工作**

這個 repo 會有多個 WorkBuddy session 併行。實例（2026-09-16 17:10）：
在 `flashkv-devserver` 目錄下有兩個 session——一個在做
`docs/PREFILL250_THERMAL_TRANSIENT_*` 與 `binary_stamp()` 的陽性對照，
另一個（當前）在寫 agent harness 白皮書。前者在 17:10:28／17:10:39 **連續寫入**
`traces/lessons.jsonl`（新增 `eng-mh-0037`）與 `PLAN_ENGINE_LOOP_2026-09-15.md`，
而後者剛好在 17:10:59 跑了一次**不帶 `--check` 的 `index_assets.py`** ⇒ 兩邊的寫入交錯。

判準與做法：

```sh
git status --porcelain --untracked-files=all     # 動筆前跑一次；開工後再跑一次比對
# 有沒有出現「不是我改的」檔案？有的話先停下，別碰索引／別 commit
```

- 看到**不是自己改的** modified 檔 ⇒ 有別的 writer。此時**只新增不修改**（新增 `docs/*.html`
  是安全的），不要重生索引、不要 commit。
- 這種競態的失效方式是**安靜的**：`MANIFEST.jsonl` 會記錄對方尚未完成的中間狀態 bytes/mtime，
  而 `--check` 在下一次重生之前都會說「OK」。

**補：收尾時要再比對一次，因為「你已經寫好的敘述」也會隨對方的前進而變成假話。**
實例（2026-09-16 20:0x，E3 白皮書）：動筆時另一條線往 `traces/lessons.jsonl` 追加的 5 筆 lesson
**還沒提交**，所以白皮書裡寫了一節「必須揭露的副作用：重生投影會把另一個 session 的
**工作區位元組**寫進我的 commit」。收尾前再查一次 —— 他們已經把**同一組**提交了（`cd91aab01`），
而且那個 commit 的訊息自己寫著「那 5 條 lesson 就是 `build_memories.py` 的輸入」
（刻意只 append 權威、把衍生物的重生留給這條線）。於是
`git diff HEAD -- agent_harness/engine_loop/traces/lessons.jsonl` 變成空的 ⇒
**事實沒變，但性質變了**（未提交的副作用 → 已消解的競態），那一節改寫成後者。

```sh
git status --porcelain --untracked-files=all      # 收尾時重跑陷阱六那一條
git diff HEAD -- <你引用過的權威檔>                # 空 ⇒ 你的重生對應的是已提交狀態
```

判準：**收尾時，對每一個「關於別人未提交狀態」的句子重新驗一次。**
程式裡的過期註解不會被引用；**白皮書裡過期的敘述會** —— 它印在文件上、有人會照著它行動，
而它讀起來和事實一模一樣。

### 陷阱七：不要為了讓數字好看而平滑、估整或轉述

專案憲章 §E 的標準是「每一條都必須能指到具體的證據（檔名／log 行／欄位值）」。
白皮書的數字一律**現場量、原樣貼**；算出來是 22.1% 就寫 22.1%，不要寫「約兩成」。
**未跑的東西要明寫未跑與為什麼**（這是本 repo 的 commit 風格，也適用於白皮書）。

### 陷阱八：同一個標記名在本 repo 常有多義，動筆時要先拆開

實例：`m1/m2/m3` 同時指「引擎調校里程碑」與「D5 的 M1/M2/M3 三個判決指標」；
`M1/M2` 另有 `M1_POOL_SPLIT_COST`／`M2_PREFILL` 文件與 `feat(m2)` commit 的用法。
**先寫一張「指涉 → 在哪裡 → 被誰守」的表把它分開**，再往下寫；否則整節的「守護」會守錯對象。
這個動作同時是最省事的提問替代品。

---

## 5. 收尾：兩個 commit ＋ 索引重生順序

白皮書是 D5 的附件，所以它跟**它所屬的那個 commit** 一起提交。收尾永遠是**兩個** commit：

```sh
# 第一個：本次的引擎／文件改動 ＋ 白皮書
git add -A
BIN_DIR='src/llama.cpp/build/bin' RUN_REPLAY_BENCH=0 \
  bash scripts/check_build_tracked.sh --repo "$PWD"
RUN_REPLAY_BENCH=0 git commit -F <message 檔>

# 收尾記憶（commit hash / gate 輸出 / push 結果）只有 commit 之後才知道
# → append 進 .workbuddy/memory/YYYY-MM-DD.md，然後重生索引：
python3 agent_harness/engine_loop/memory/build_memory_index.py   # 先（寫 INDEX.jsonl）
cd agent_harness/engine_loop && python3 index_assets.py && cd -  # 後（記錄 INDEX 的 bytes/mtime）

# 第二個：resync（只動 INDEX.jsonl + MANIFEST.jsonl，無 src/ ⇒ D5 不必重跑，message 要寫明）
git add -A && RUN_REPLAY_BENCH=0 git commit -m 'docs(index): resync ...'
```

- **順序不可顛倒**：`build_memory_index.py` 每次都會重寫 `INDEX.jsonl`（即使內容不變），
  所以 `MANIFEST` 必須在它之後取得 bytes/mtime。顛倒的簽名是 **`INDEX.jsonl` 的 mtime 漂移**
  而不是 bytes 漂移——看到就重跑一次兩條，不要去找內容差異。
- 收尾驗證：`traces/validate.py`、`traces/selftest.py`（10/10）、兩條 `--check`、
  `git status --porcelain --untracked-files=all` 必須空。
- 純 doc 的 commit 技術上可以不跑 D5，但**跑它只花 40–60 s**，而本 repo 已明確把
  「省下來然後寫一段話解釋為何不跑」判為錯的做法（lesson `eng-gate-0016`）。

---

## 6. 交付前自檢

```sh
# a) HTML 結構：沒有未閉合或錯配的標籤（SVG 標籤要當 void 處理）
python3 - <<'EOF'
import html.parser
p='docs/<你的檔名>.html'; s=open(p,encoding='utf-8').read()
class C(html.parser.HTMLParser):
    def __init__(s):
        super().__init__(); s.st=[]; s.err=[]
        s.void={'meta','br','hr','img','link','input','path','rect','line','circle','text','marker','use','stop'}
    def handle_starttag(s,t,a):
        if t not in s.void: s.st.append((t,s.getpos()))
    def handle_endtag(s,t):
        if t in s.void: return
        if s.st and s.st[-1][0]==t: s.st.pop()
        else: s.err.append((t,s.getpos()))
c=C(); c.feed(s)
print('unclosed:',c.st[:5],'errors:',c.err[:5])
EOF

# b) 所有的檔名引用都真的存在（白皮書大量引用路徑，最常見的錯是打錯檔名/行號）
grep -oE '(docs|scripts|agent_harness|Backup)/[A-Za-z0-9_./-]+' docs/<你的檔名>.html | sort -u | while read f; do [ -e "$f" ] || echo "MISSING: $f"; done

# c) 現場閘門四條（見 §3），把輸出原文貼進白皮書

# d) markdown 殘留掃描（2026-09-16 實測：三處 **粗體** 與四處 `反引號` 直接寫進了 HTML，
#    會原樣顯示。這是「用 markdown 的習慣寫 HTML」的必然產物，不是粗心。）
python3 - <<'EOF'
import re
p='docs/<你的檔名>.html'; s=open(p,encoding='utf-8').read()
chk=re.sub(r'<(pre|code)\b[^>]*>.*?</\1>','',s,flags=re.S)   # pre/code 內原樣，不看
print('markdown 殘留:', re.findall(r'\*\*[^*\n]{1,80}\*\*|`[^`\n]{1,80}`', chk) or '無')
EOF
# 修法（保留 pre/code 區塊，其餘把 **X** -> <b>X</b>、`X` -> <code>X</code>）：
#   逐段掃 <(pre|code)\b[^>]*>.*?</\1>，只對「區塊外」的文字做替換。
#   注意：**「含 <code> 的長句」** 這種巢狀情況正則抓不到（我在 E1 漏了一個），
#   所以掃完要人工看一眼殘留清單，不要只看它印「無」。

# ★★ 兩次實測（E1 漏 4 處、E2 漏 11 處）之後的結論：**用單行正則一定不夠。**
#    最常見的殘留是**跨行**的粗體（`` **…`` 換行 ``…** ``），單行 pattern 看不到。
#    而用 `re.S` 補第二輪時，我又弄出了巢狀 `<b>`——第二輪把第一輪已經轉好的
#    `<b>X</b>` 又包了一層，症狀是 HTML 解析器回報「unclosed」而看不出為什麼。
#    可用的做法（兩輪，且第二輪要驗巢狀）：
#      r1: 單行 pattern（見上），把明確的處理掉
#      r2: 對「區塊外」的段落用 `re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', t, flags=re.S)`
#      r3: 驗巢狀 —— `'<b>' in re.sub(r'<b>[^<]*</b>','',html)` 必須為 False
#    最後 `s.count('<b>') == s.count('</b>')` 必須相等（E2 實測 35/35）。
#    ⇒ 更省事的做法是**一開始就用 <b>**，把 markdown 習慣省下來；但實測做不到，
#      所以把「一輪轉換 + 巢狀驗證」當成固定步驟，不要當成意外。

# e) `<pre>` 區塊內的 `<` 與 `>` 必須轉義
#    實測：`TB_REPO_ROOT = <repo>` 讓解析器把 <repo> 當成一個永不閉合的標籤，
#    (a) 會回報 unclosed [html, body, div, pre, code] —— 症狀與「標籤真的少了一個」一模一樣。
#    寫成 &lt;repo root&gt;。
```

自檢清單：

- [ ] 每個數字都有出處；沒有「約」「多數」「大幅」這類無法證偽的詞
- [ ] 有一節列出**現場跑過**的閘門輸出原文
- [ ] 有一節「誠實邊界」，明寫**不主張什麼**（含未完成的事與未跑的測試）
- [ ] 檔名帶了 `_HHMM`（若同主題同日可能再寫）
- [ ] 沒有改到不是自己改的檔案（陷阱六）
- [ ] 沒有寫進 `.workbuddy/memory/`、沒有手動加快照（陷阱三）
- [ ] 淺色主題、`lang="zh-Hant"`、無 CDN 依賴
- [ ] 引用索引時用 `path:line_start`，**不複述內容**
- [ ] markdown 殘留掃過（d）；`<pre>` 內的 `<`/`>` 已轉義（e）
- [ ] **路徑引用檢查的「MISSING」清單逐條看過**——白皮書會刻意引用「不存在的那個路徑」
      （搬遷後指錯的位置、懸空指標），那些是內容不是錯誤。要看的是**沒打算缺失卻缺失**的那些

## 6.1 已提交的白皮書：修訂它，還是取代它

白皮書是 D5 的附件，所以它與**它所屬的那個 commit** 綁定。當後續的一個 commit 讓它的某幾句
變成假話時（E1 就讓上一份的「tb_loop 三個入口跑不起來」與 README 路徑兩處過期）：

- **不要重寫已提交的文件**——那會讓它不再是那個 commit 的附件，而讀者無法分辨哪一版屬於哪一輪。
- **也不要留著不處理**——那會讓後人照著一份已知是假話的文件工作。
- **做法**：在原文**最前面加一個標註框**，逐條列出「哪幾處被取代、被誰取代」，
  並**明說哪些部分仍然成立**；原文一個字不刪。新內容寫進新的一份白皮書，
  用相對路徑互相連結（`AGENT_HARNESS_E1_RESTRUCTURE_20260916.html`）。
- 依據是 `CONVENTIONS.md` §E：「條文本身可以被推翻：若某條的證據被後續實驗推翻，該條標
  `superseded_by` 而不是刪掉——刪掉會讓後人重新踩一次。」文件同理。
- 同一個 commit 內若另一份白皮書的 note 也過期（`index_assets.py` 的 `CURATED` note
  就有這個毛病），**一起改**，因為編輯 `CURATED` 會讓 `--check` 紅。

---

## 7. 這條 skill 自己的邊界

- 它規範的是**版式與流程**，不是內容。一份白皮書的價值來自 §2 的取材與 §3 的現場輸出。
- 它**不**代替 `cgc-commit-gate`：提交閘門、`check 8`、stash 換臂、dylib ABI 那些都在那份。
- 它**不**代替 `CONVENTIONS.md` 的 A 節：什麼數字可以引用是憲章說的，這裡只說「要現場跑並原樣貼」。
