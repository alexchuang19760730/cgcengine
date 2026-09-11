# Chat scaffold 根因報告 — 2026-09-11

**Worktree**：`flashkv-devserver`（branch `feat/p1prime-knifeedge`）
**模型**：`Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf`（IQ3，生產載體）
**相關 commit**：`18ce36dd6` == `8eb570598` + 本報告描述的修復
**未提交 WIP 保護**：tag `wip/p1prime-20260911`、`wip/p1prime-20260911b`、`Backup/wip_p1prime_20260911/`

---

## 1. 根因：template 由 `enable_thinking` 驅動，但 launcher 從來不傳它

**決定性證據是把 GGUF 內嵌 template 抽出來看的結果**（不是推測）。內嵌 template 的尾端是：

```jinja
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}   ← 閉合的 scaffold（no-think 模式）
    {%- else %}
        {{- '<think>\n' }}                 ← 未閉合（thinking 模式）
    {%- endif %}
{%- endif %}
```

而 `run_server.sh` 在預設路徑**從不傳 `--chat-template-kwargs '{"enable_thinking": false}'`**。

所以每次請求都走 `else` 分支：generation prompt 尾端是**未閉合**的 `<think>`，模型自己補完 `</think>` 後又開一個 → marker 迴圈。
實測輸出：`<think>\n\n</think>\n\n<think>\n\n</think>\n…` 無限重複、`finish_reason=length`。

**關鍵點：這個缺陷同時存在於兩份 template**：
- GGUF 內嵌 template（Unsloth 出貨的原版）
- 本 branch 的 `Qwen3-nothink-ChatML.jinja`（被改成鏡像內嵌版的結構）

因此**只把 iq3 切到 embedded 並不能修好它**（已實測：切 embedded 後仍然 marker 迴圈）。
這也是為什麼先前「改 template 來源」的方向看起來合理、卻沒有解決問題。

### 1.1 修復驗證（A/B）

| 配置 | scaffold | 15+27 輸出 | finish |
|---|---|---|---|
| embedded，無 kwargs | 未閉合 `<think>\n` | marker 迴圈，無答案 | length |
| embedded + `enable_thinking=false` | 閉合 `<think>\n\n</think>\n\n` | **`15+27=42`**（英文題也給出 `42`） | length（仍不停） |

⇒ `enable_thinking=false` 是**必要**的一環，且已被證實能讓模型產出正確答案。

### 1.2 還有一個旗標語義陷阱

`CGC_LOOP_GUARD` 在 `server-context.cpp:1937` 的註解寫著「default off in code; **run_server.sh enables**」，
但 `run_server.sh` 這條 branch 上**完全沒有設定它**（grep 0 命中）。註解與事實不符，是典型的註解腐化。

---

## 2. 尚未解決：閉合 scaffold 之後仍然 echo + 不停止

即使 scaffold 已閉合、答案已正確，模型仍然**回顯 prompt 並無限重複**，`finish_reason=length`：

```
zh: <think>\n\n</think>\n\n15+27 等於多少？請只回答数字。\n\n15+27 等於多少？請只回答数字。\n\n…（重複）
en: <think>\n\n</think>\n\n15+27=42\nAnswer with the number only.\n42\n</think>\n\n15+27=42\n…（重複）
```

已排除 / 無效：
- `CGC_LOOP_GUARD=1` → 仍然 echo、仍然 `finish=length`（訊號：loop guard 的 window 與判準抓不到「prompt 前綴 + 重複」這種形狀）

這是本 branch 上**目前唯一擋住 48 題量測的缺陷**，而且它與 expert cache 無關。

註：白皮書記載的既有解法是「server 端對空 prefill 請求注入**內容匹配的 `assistant_prefill`**」來打破回顯——
而 `assistant_prefill` 只有 **custom jinja** 支援，embedded template 不支援。這與「iq3 用 embedded」的取向有張力，需要決定。

---

## 3. 這次修了什麼（commit `18ce36dd6`，只含 2 個檔案）

- `scripts/run_server.sh`：**model-aware template 選擇** —
  - `iq3`（含 `denseIQ4X` 生產載體）→ GGUF embedded template，不注入 custom jinja
  - `iq4` → custom jinja
  - 判別用 `IQ4_XS` 而非 `IQ4`，否則 `…-IQ3_XXS-denseIQ4X.gguf` 會被誤判為 iq4
  - 啟動時印出 `[chat] model_kind=…` 供驗證
- `scripts/check/assert_prompt_scaffold.py`（新）：scaffold 回歸守衛
  - A：template 每個分支都必須是**閉合**的 scaffold、不得使用非模型 marker 的字面字串
  - B：custom jinja 只能在 iq4 guard 之後被選用
  - C（`--base-url`）：實際打一題，偵測 marker 洩漏 / 退化重複 / 空燒預算
  - **已驗證**：對壞掉的 template 會 FAIL（A1/A2/A3 三條獨立診斷），對修好的會 PASS

`Qwen3-nothink-ChatML.jinja` 被還原成 HEAD 版（因此與 HEAD 無 diff、不在此 commit 內）。

---

## 4. P1/P2 稽核（回應「別人說 P1/P2 沒完成」）

| 說法 | 事實 |
|---|---|
| P1「已完成 9 項」（`LLAMA_EXPERT_CACHE_MMAP_STREAM`、`madvise_prefetch`、`expert_cache_mmap_stream`…） | **不在樹上**。三個分支（`feat/p1-expert-streaming-mmap`、`feat/p1prime-knifeedge`、`demo/sweet-spot-windows-fix`）**都指向同一個 commit `8eb570598`**，這些符號在磁碟上 grep 命中數為 **0**。那是未提交 WIP，在切分支時遺失（本次已用 tag 保護機制補上） |
| P1「Metal residency set 爆炸」是致命問題 | **正確**，且與獨立證偽一致：18GB 全量 expert 不縮小在 16GB 上無解，**shrink 是前提不是可優化開銷** |
| P2「6 項未完成、`madvise_lru_evict` 是空殼」 | **誇大**。功能其實在：`madvise(MADV_DONTNEED)` 淘汰（`:151`）、`F_RDADVISE` 預讀（`:2151`）、`prefetch_slot` 背景佇列（`:838`）、`mlock_dense`（`:1550`，且 `llama.cpp:449` 有呼叫）。缺的是**函式名**不是**行為** |
| 真正缺的 | parallel prefetch lanes、`mincore()` residency tracking、verify 路徑的 I/O 重疊（後者已寫在未提交 WIP，預設關閉） |

### 4.1 本輪對未提交 WIP 做的安全處理

- `CGC_PREROUTER_STAGED`（`llama-expert-cache.cpp:1613`）原本直接寫 `slot_owner`/`slot_table` 卻**不載權重** →
  會讓 slot 指向**別的 expert 的 bytes**（靜默污染）。已改為走 `prefetch_slot` enqueue，永不謊報 residency。
- P4 三個入口全部標註為 **STUB / 預設關閉**（`prerouter_predict` 是頻率排名、`recover_lora_*` 是 no-op）。
- 新增 `cgc_env_enabled()` 並把 4 個 `getenv(x) != nullptr`（非空即開）改成 value-aware —
  這正是先前 `L4_SKIP_LAYER0=0` 被讀成「開啟」的同一類陷阱。
- 6 個新旗標在 `run_server.sh` 維持 **0 命中**（刻意不接線），並在腳本內加註說明。

---

## 5. 對既有量測的影響（重要）

以下數字**不可引用**，因為它們是在 scaffold 故障（未閉合 `<think>`）的狀態下取得的：

- `Backup/knifeedge_results/knifeedge_iq4xs_pool4gb.json`（15:15）— 8 題全部 `content=''`（空字串，非品質問題）
- `/tmp/flip_8gb.json`（14:17）— 8/8 FAIL、`finish=length`、5 個 seed **byte-identical**、0 flips
  （「0 flips」被誤讀為「確定性」，實際上是「確定性地被 scaffold 弄壞」）
- `/tmp/pool_curve.json`（13:40）— 各 pool size 的品質欄位
- 白皮書 §10 的「記憶體下限 4GB」推論建立在上述數字上 → **需降級**

**尚未驗證的一個關鍵問題**：`enable_thinking` 是 template 層、與 pool 大小無關的變數；
所以先前觀測到的「pool 大小影響文字輸出」是否只是 scaffold 故障在不同 pool 下的不同表徵，仍待重測確認。

---

## 6. 下一步（建議順序）

1. **決定 iq3 的 scaffold 策略**：embedded + `enable_thinking=false`（需另解 echo）
   或 custom jinja + `assistant_prefill`（白皮書記載的既有 echo 解法）。
2. **解掉 echo/repeat**（`finish=length`）——目前唯一擋住量測的缺陷。候選：修 `cgc_detect_phrase_loop`
   讓它抓「prompt 前綴 + 重複」、或啟用 DRY/repeat-penalty、或注入 assistant_prefill。
3. 有了乾淨的 scaffold 之後，才重跑 4/6/8GB 的 48 題量測，重新定 pool 下限。
4. 更新白皮書 §10 與 P1 設計文檔（後者目前仍描述已被證偽的 mmap 設計）。

---

## 7. 追加驗證：在同一個 repo 的「別人的 server」上重現（同日稍晚）

上面 §1 的結論原本來自抽取 GGUF 內嵌 template。以下是**在一個活著的、非我啟動的 server 上**獨立重現的版本。

### 7.1 受測對象（不是我啟動的）

```
./src/llama.cpp/build/bin/llama-server -m models/gguf/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf \
  -expert-cache 8589934592 -ngl 99 --no-mmap -t 8 -c 2048 -np 1 \
  --cache-type-k q8_0 --cache-type-v q8_0 --jinja --reasoning off --reasoning-format none \
  --spec-type draft-mtp …
```

注意：**沒有 `--chat-template-file`**（所以走 GGUF 內嵌 template）、**沒有 `--chat-template-kwargs`**。

### 7.2 `GET /props` 的 `chat_template` 尾端（8057 chars，Unsloth 多模態原版）

```jinja
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}   ← 閉合（no-think）
    {%- else %}
        {{- '<think>\n' }}                 ← 未閉合（thinking）—— 實際走這條
    {%- endif %}
{%- endif %}
```

⇒ §1 的根因**在別人的 server 上完全一致**。`enable_thinking` 未被傳入，所以每個請求都拿到未閉合的 `<think>`。

### 7.3 實測輸出：確定性的純 prompt 回顯

同一題、`temperature=0`：

```
content = "15+27 等於多少？請只輸出答案\n\n15+27 等於多少？請只輸出答案\n\n15+27 等於多少？請只輸出答案\n\n15"
finish  = length      answers_42 = False      echoes_prompt = True      empty = False
```

**沒有任何答案、純回顯、永不停止。** 這就是「中文 echo」的本體。

### 7.4 `chat_template_kwargs` 在**請求層**無效（重要）

很多人（包括我先前）會想「那就每個請求帶 kwargs 就好」。實測**三個變體逐 byte 相同**：

| 請求 body | len | 內容 |
|---|---|---|
| 控制組（無） | 59 | 同上回顯字串 |
| `chat_template_kwargs={"enable_thinking": false}` | 59 | **完全相同** |
| 再加 `assistant_prefill: "答："` | 59 | **完全相同** |

⇒ 這個 build **不從請求 body 讀 `chat_template_kwargs`**。因此修復**只能在啟動層**（`--chat-template-kwargs`），
不能靠 client 補救。這也解釋了為什麼「換 template 來源」與「client 加參數」兩條路都沒救到。

### 7.5 推論：任何沒傳 `enable_thinking=false` 的 pool sweep，量到的都是 template

因為哪條 template 被選中與 pool 大小無關，而 scaffold 未閉合會**支配**輸出（純回顯、`finish=length`），
所以那些 sweep 的品質欄位量的是「template 壞掉」，不是「pool 太小」。

---

## 8. 評測 harness 改造（同日）

為了讓上面這類故障**不可能再被靜默量測**，評測層加了兩件事：

- `scripts/check/flip_rate.py`（既有，已擴充）：固定 seed × N 次重跑 + `--greedy-repeats` 同題重送，
  逐題算通過率與 **verdict-flip**，room 層算 suite 通過率的 mean/std/min/max，並記錄 tok/s。
  「0 flips」只有在答案本身正確時才代表確定性；若全數確定性失敗，它代表的是故障不是穩定。
- `scripts/check/knifeedge_matrix.py`（新）：**(model × pool) 矩陣驅動**
  - 模型維度：`iq3` / `iq4` 一鍵切換（不需手改腳本），並在結果記錄實際選中的 template
  - `--preflight {abort|warn|off}`：量測前先打一題 canonical probe，**必須含有 42** 才放行；
    否則記下 `answers_42/echoes_prompt/scaffold_leak/empty/finish` 並**拒絕產出品質數字**
  - `--attach`：量測**別人已在跑**的 server（不啟動、不殺），適合這種共用 checkout
  - `--preflight-only`：低成本健檢（1 個請求）——就是它在 7.3 抓到現行 server 壞掉
  - `running_servers()` 只認 argv[0] 真的是 `build/bin/llama-server` 的行程，
    避免 `pgrep -f` 誤判 bash wrapper（那會讓互斥閘門永遠拒絕量測）

**首次執行結果（`--preflight-only`）：`ok=false`，`echoes_prompt=true`，`answers_42=false`。**
閘門運作正常，且當下那台 server 的品質數字不可用。

---

## 9. `CGC_LOOP_GUARD` 與 `CGC_STRIP_SCAFFOLD` 實測（該不該留）

工具：`scripts/check/flag_ab.py`（新）——同一組 probe 跑不同 flag 組合。

設定：iq3 denseIQ4X、8GB pool、**embedded template + 不傳 `enable_thinking`**（＝重現出貨時的故障）、
`--reasoning off`。probe 分兩類：**該修的故障**（`math_zh`）與**不能被改壞的輸出**
（`en_short` 正常英文題、`code_marker` 要求輸出含 `<think>`/`</think>` 字面字的程式碼）。

### 9.1 `CGC_LOOP_GUARD`：**確實會 fire，建議保留**

server log 的直接證據（LG=1 的兩場有 3 次、LG=0 的三場 0 次）：

```
CGC-LOOP-GUARD: phrase loop detected at 125 generated chars, forcing stop (tail: 27 等於多少？請…)
CGC-LOOP-GUARD: phrase loop detected at  55 generated chars, forcing stop (tail: k>\n…)
CGC-LOOP-GUARD: phrase loop detected at  57 generated chars, forcing stop (tail: \n…)
```

輸出長度被切掉一半（148→78、265→74、475→76），但**`expect` 三個 probe 完全沒變**——
所以它是 **safety valve（防爆走、省 token 與時間），不是品質修復**。

**但它有一個結構性盲點，正是「超過閾值就抓不到」的真身：**

`window=256`、`min_repeat=3` ⇒ 可偵測的最長重複單元 = `min(max_unit=300, 256/3)` = **85 字元**。
`max_unit=300` 永遠碰不到。**單元超過 85 字元的迴圈在結構上不可能被偵測**（尾窗看不到 3 次完整重複）。
修法：`window` 提到 1024（→341 字元），或把 window 與 `max_unit * min_repeat` 綁定。

### 9.2 `CGC_STRIP_SCAFFOLD` Pass 1：**有效但不足以當品質修復；有一個真實回歸風險**

| probe | markers(off→on) | expect(正確答案是否保住) | 剝離後內容 |
|---|---|---|---|
| `math_zh` | 1 → **0** ✅ | 0（本來就沒有答案） | 仍是 prompt 回顯 ×3 |
| `en_short` | 1 → **0** ✅ | **1 → 1 保住** ✅ | `15+27=42\n\n\n15+27=42…` |
| `code_marker` | **1 → 1 ❌** | 0 | 仍是 3 個完整 `<think>` 區塊 |

兩個結論：

1. **對品質沒有貢獻。** 剝離後 `math_zh` 依舊是 prompt 回顯，`code_marker` 依舊是純 scaffold。
   它只讓 `content` 變乾淨，48 題分數不會因此上升。
2. **Pass 1 會刪中間內容，這是真實回歸風險。** 把 9.2 的實作套在原始輸出上模擬：
   `code_marker` 475 字 → **只剩 50 個空白字元**（標籤之間的東西全被刪）。
   任何**合法答案本身包含 `<think>`／`</think>` 字面字**的題目（程式、模板、文件類）都會被吃掉內文。
3. **行為不一致**：`code_marker` 這題客戶端收到的是 3 個完整 `<think>` 區塊，而模擬顯示該題剝離後應為空——
   代表剝離沒有統一作用，或內容被清空後又被回填。兩種都不可接受。

**建議改法**：Pass 1 只移除**標籤字串本身**（不刪中間內容），或只剝離**開頭**的 scaffold
（`^\s*<think>.*?</think>\s*`）而完全不碰正文內的標籤。目前的全域刪除不該預設開啟。

### 9.3 兩者都與真正的問題正交

兩個機制都只動「輸出後處理」。真正卡住的仍然是：
**custom jinja 路徑下 prompt 渲染完全正確（`<|im_start|>assistant答：`）但模型只吐 `15+27 等於 15+27`。**
開或關這兩個 flag 都不會改變那件事。
