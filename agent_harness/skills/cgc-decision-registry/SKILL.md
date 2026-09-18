---
name: cgc-decision-registry
description: 把 flashkv-devserver 的日誌決策節（`.workbuddy/memory/YYYY-MM-DD.md` 的 §EN-／§TB- 節）登錄成 `agent_harness/engine_loop/traces/decisions.jsonl` 的決策記錄。當使用者要求「把某段時間的技術決策納入宏觀技術決策」「決策 registry 覆蓋落後」「補決策/補可復盤」時使用。也適用於不確定決策記錄的 schema、id 格式、粒度、或改完之後要重生哪些衍生物。
agent_created: true
---

> **這是快照，不是權威副本。**
> 權威位置：`~/.workbuddy/skills/cgc-decision-registry/SKILL.md`（由 host 持續寫入）。
> 本檔於 2026-09-18 由 `agent_harness/scripts/import_harness_snapshot.py` 複製進 repo，唯一目的是讓 `agent_harness/`
> 底下的內容能被 `agent_harness/scripts/auto_git_push.ps1` 定時推送；原檔改了這裡**不會**自動跟上。
> 要改 skill 請改原檔，再重跑 `python3 agent_harness/scripts/import_harness_snapshot.py`。

# flashkv-devserver 決策 registry 的登錄

專案：`/Users/alexchuang/Documents/flashkv-devserver`（是 git worktree）

**用途**：`traces/decisions.jsonl` 是入口「宏觀技術決策」那一頁的資料源。日誌（`.workbuddy/memory/*.md`）
本身是 append-only 的敘事，而 registry 是它的**可機檢骨架**。這條路每次都會走一次（覆蓋會落後）。

---

## 0. 30 秒版

```sh
cd /Users/alexchuang/Documents/flashkv-devserver

# ① 量缺口（只讀）
python3 - <<'PY'
import json
rows=[json.loads(l) for l in open('agent_harness/engine_loop/traces/decisions.jsonl') if l.strip()]
print("registry:", len(rows), "筆")
for r in rows[-3:]: print("  ", r['decision_id'])
PY

# ② 量來源有幾節（★ `^## ` 與 `^### ` 都要算）
python3 - <<'PY'
import re, pathlib
for f in ('2026-09-17','2026-09-18'):
    s=pathlib.Path(f'.workbuddy/memory/{f}.md').read_text(encoding='utf-8')
    print(f, len(re.findall(r'^###? §', s, re.M)), "節")
PY

# ③ 寫片段檔 → 一支 append 腳本（見 §3）→ 驗 → 重生兩個投影 → 入口 → commit
```

---

## 1. 首先確認的三件事（不確認就會做錯方向）

1. **切點**。「有 m1/m2/m3 oracle identical 以後」＝
   `Backup/m123_oracle_gate/summary_*.json` 中 `M1/M2/M3 = 9/9/9` **第一次出現**的時刻
   ＝ **2026-09-15 17:22**（`summary_final_v2.json`；之前是 0/4、3/13、1/4…）。
   量法：
   ```sh
   python3 - <<'PY'
   import json, glob, os, datetime
   rows=[(os.path.getmtime(p), p, json.load(open(p))) for p in glob.glob('Backup/m123_oracle_gate/summary_*.json')]
   for ts,p,d in sorted(rows)[:14]:
       print(datetime.datetime.fromtimestamp(ts).strftime('%m-%d %H:%M'), d.get('m1_numeric_identity'),
             d.get('m2_decision_agreement'), d.get('m3_topk_set_agreement'), os.path.basename(p))
   PY
   ```
   注意：`cacheoff`／`s1_outcap` 那幾筆是**刻意的陰性對照**（0/1、5/9），不是退化。
2. **粒度**（一節一筆 vs 宏觀收斂）與**交付方式**（分批 vs 一次全部）。
   缺口常是 100+ 節 ⇒ 一定要先問，否則差 10 倍工作量。實測規模：09-17 有 70 節、09-18 有 49 節。
3. **切點之前的節有沒有漏**。registry 的覆蓋是用 `decision_id` 的日期前綴算的 ⇒
   補舊日期是**允許的**（append-only 不要求時間遞增），但記得 `days` 那一格會跟著變。

---

## 2. `decisions.jsonl` 的 schema（`traces/schema/decision.schema.json`）

必填：`type` / `decision_id` / `question` / `evidence` / `reasoning` / `conclusion` / `judgement` /
`action` / `superseded_by`（可為 `null`，但**必須存在**）。
選填：`confidence`（`high|medium|low|unknown`）、`ruled_out`（`{claim, why_false}`）、`artifact`、`supersedes`。

- `decision_id` 的 pattern 是 **`^dec-[0-9]{8}-[0-9]{4}-[a-z0-9-]+$`**
  ⇒ 小寫英數與連號，**不能有底線、大寫或中文**。時間取來源節自己寫的時刻；沒寫就用該節在日誌裡的相對位置估，
  並在 `because` 之外的地方老實承認那是估的（例如 `question` 裡不宣稱精確時間）。
- `judgement` 是**反毒閘門**：`refuted` 的記錄會被排除出兩個 SFT 投影的正向集 ⇒
  **不要**為了「看起來完整」把被推翻的述句標 sound。被後續推翻的**自己的新記錄**要標 `superseded_by`。
- `evidence[].artifact` 用**真實存在的路徑／行號／commit hash**；`episode_id` 只在來源節真的引用
  episode 時才填（見 §5 的副作用）。
- 內容的來源是日誌原文 ⇒ **逐字照抄數字，不轉述**。既有 38 筆的平均大小是 3–6 KB。

---

## 3. 寫入：片段檔 ＋ 一支 append 腳本（不要手改 `decisions.jsonl`）

**為什麼**：`decisions.jsonl` 是單行 JSON 的 append-only 共享檔。用 Edit 或 read-modify-write 會
（a）蓋掉別條線的並行 append、（b）在一個 10 萬行的檔案裡做字串替換。做法是：

1. 把記錄寫成**獨立的 JSONL 片段檔**（`/tmp/dec<date>_<a|b|c>.jsonl`），用 Write 工具一個一個檔寫。
   每筆一行、`ensure_ascii=False`。
2. 用 Write 工具寫一支 append 腳本（**不要用 shell heredoc** —— heredoc 會吃掉字串裡的 `\n`，
   症狀是 `SyntaxError: unterminated string literal` 指到字串開頭），腳本裡一律要 assert：
   ```python
   assert r["decision_id"] not in have                 # 不重複
   assert len(set(ids)) == len(ids)                    # 批次內也不重複
   assert not [i for i in ids if not i.startswith("dec-20260918-")]   # ★ 日期前綴
   ```
   ★ **日期前綴那一條是必要的**：打錯會靜默地把記錄歸到別天，而入口的「覆蓋」那一行看起來仍然正確。
3. 寫後**重讀並逐筆比對** `question` 與 `conclusion`（不要只信腳本自己印的）。
4. 驗 append-only：`git diff -U0 -- traces/decisions.jsonl` 必須是 **`-0 +N`**。
   用 **python 逐行數**，不要用 bash `grep` 過濾（那條管線在本機 sandbox 會靜默漏行）。

---

## 4. 改完之後要重生什麼（漏一個入口就紅）

```sh
python3 agent_harness/engine_loop/traces/validate.py        # 目標：decisions.jsonl: N records -> OK
python3 agent_harness/engine_loop/traces/selftest.py        # 必須 10/10
python3 agent_harness/engine_loop/sft_pi/build_sft_pi.py     # ★ 兩個投影都要重生
python3 agent_harness/engine_loop/sft_prime/build_sft_prime.py
python3 agent_harness/portal/build_portal.py                 # 入口：要 14/14 綠
```

★ **`decisions.jsonl` 不在 `MANIFEST.jsonl` 裡** ⇒ 編輯它**不會**讓 `index_assets` 漂移，
所以**不需要**重生索引；但它的**兩個衍生投影會 stale**。只重生一個的症狀是入口報
`derived-sft-pi` 紅燈 —— 那**不是**回歸，是入口在正常工作。

---

## 5. 一個必須揭露的副作用：兩個投影的投影方式不同

| 投影 | 讀什麼 | 這批決策的影響 |
|---|---|---|
| `sft_prime` | decision 的 `question` ＋ `evidence` ＋ `action` | `state_to_next_arm` 直線上升（+N） |
| `sft_pi` | **軌跡**，需要 `episode_id` | `train/valid` **位元不變**；而它印的 `decisions with no episode evidence` 會**等量上升** |

若決策記錄的 `evidence` 全是 artifact／原始碼行（引擎層的判斷通常如此），
`sft_pi` 的軌跡行**不會增加**，只有 `no episode evidence` 增加。
⇒ **把 `no episode evidence` 當成缺陷之前，要先做一個判準決定**（「決策記錄是否該強制回填 `episode_id`」），
不要在 commit message 裡含糊帶過。要寫的是「本批 N 筆全部落在 `no episode evidence`」這個事實。

---

## 6. 取證與邊界（寫進 commit message 與白皮書）

- **不要宣稱復算過。** 這些記錄是「日誌原文的結構化重排」：artifact 路徑、行號、數字照抄，
  但**沒有**重跑量測、**沒有**逐一開每個 `Backup/cgc_logs/*.log` 核對。要引用數字仍應回到原始產物。
- **不要宣稱覆蓋完整**，除非真的數過。缺口的計數方式：來源節數（`^## ` ＋ `^### `）− 已登錄數。
- 本 repo 的 D5 白皮書那一半**沒有豁免**：純 doc 的 commit 也要出一份白皮書；
  同一支儀器的連續推進 ⇒ **就地追加**（節號接續、開頭加取代標註框、原文一字不刪），
  而且**每次追加之後都要重跑整份自檢**（見 skill `cgc-whitepaper-delivery` §6.1.1）。
- 收尾常常是**兩個 commit**：交付本體，然後是「記憶 ＋ 快照 ＋ 索引」的 resync。

---

## 7. 已知的坑（都在實作時踩過）

| 症狀 | 真因 |
|---|---|
| 「§EN 編號斷了（102–111 之後直接跳 132）」 | **`§EN-111` 是 21 節的容器**，底下用 `###` 藏了 `§EN-112`–`§EN-131`。數節數時 `^## ` 與 `^### ` 都要算。 |
| 入口報 `derived-sft-pi` 紅燈 | 只重生了一個投影（§4）。 |
| `git diff` 的 `-0 +N` 數錯 | 用了 bash `grep` 過濾 diff ⇒ 靜默漏行。用 python。 |
| `SyntaxError: unterminated string literal` | 用 shell heredoc 寫 JSON ⇒ `\n` 被吃掉。改用 Write 工具寫實體檔。 |
| 記錄被歸到別天而入口看起來正常 | `decision_id` 的日期前綴打錯（§3 的第 2 條 assert 就是為此）。 |

## 8. 這條 skill 的邊界

它規範的是**格式與流程**，不是內容。一筆記錄的價值來自它的 `evidence` 是否指到真實產物
與 `conclusion` 是否真的被那些產物支持 —— 那要靠讀來源節，沒有辦法自動化。
