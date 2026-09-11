# 評測一致性指標定義 + pool 不變性量測（2026-09-11）

## 1. 兩個獨立指標（永遠不合併）

同一組數字以前被兩邊各自引用成相反結論，所以把它定義成**兩個獨立指標**：

| 指標 | 定義 | 回答的問題 |
|---|---|---|
| **M1 numeric identity** | 每步 `row_fnv1a64` 逐位元組相同 | 「logits 有沒有飄」 |
| **M2 decision agreement** | 每步 `argmax_token` 相同 | 「選到的 token 一不一樣」 |
| **M3 top-k set agreement** | 每步 top-N id 集合相同（輔助） | 「候選集合一不一樣」 |

### 為什麼不能合併 — 2×2 交叉表必須一起讀

| | M2 same | M2 diff |
|---|---|---|
| **M1 same** | 真的是同一個計算 | **理論上不該發生**；出現就是 tie-break / 排序不穩定 → 當 bug 處理 |
| **M1 diff** | **最容易被誤讀的一格**：M1 說「有差」、M2 說「沒差」。真相是「有數值飄移，但還沒改變決策」——greedy 解碼是混沌的，隨時可能在下一個 token 翻面 | 真的分歧 |

**規則**：M2 單獨通過**不代表**「沒有差異」；M1 單獨通過也不代表「決策一致」。報告一律兩者並列。

### 用法

```bash
python3 scripts/check/cgc_logits_oracle_compare.py --a ref.jsonl --b cand.jsonl --report out.json
```

`--fail-on both`（預設）要求 M1 與 M2 都全同；`--fail-on numeric` 可重現舊的寬鬆規則。
`knifeedge_matrix.py --oracle-ref <ref.jsonl>` 會把這個比較變成每個 (model × pool) 的硬閘門。

---

## 2. 量測結果

模型 `Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X`（IQ3），同一題、`temperature=0`、
`CGC_LOGITS_ORACLE_DUMP` 取樣，`chat_template_kwargs={"enable_thinking": false}`。

### 2.1 四組配置

| 標籤 | pool | `CGC_POOL_MAX_TOKENS` | union 上限 | 53/106 slots | 結果 |
|---|---|---|---|---|---|
| `p8_pmax8` | 8GB | 8 | 64 | 106 → 塞得下 | OK，**nil=0**，dump 46 行 |
| `p8_pmax6` | 8GB | 6 | 48 | 106 → 塞得下 | OK，**nil=0**，dump 39 行 |
| `p4_pmax8` | 4GB | 8 | 64 | 53 → **塞不下 → gather path** | **120 次 `buffer is nil`** |
| `p4_pmax6` | 4GB | 6 | 48 | 53 → 塞得下 | OK，**nil=0**，dump 39 行 |

`pmax=2`（原本想用的值）**直接崩潰**：`llama-batch.cpp:609: GGML_ASSERT(n_ubatch > n_keep_tail) failed`。
所以可用下限是 `6`，不是 `2`。

### 2.2 三組對比

| 對比 | 隔離的變數 | M1 (bit-identical) | M2 (argmax) |
|---|---|---|---|
| `p8_pmax6` vs `p4_pmax6` | **只差 pool 大小**（形狀相同） | **7/39 (17.9%)** ❌ | **38/39 (97.4%)** ✅ |
| `p8_pmax8` vs `p8_pmax6` | **只差 batch 形狀**（pool 相同） | 3/39 (7.7%) ❌ | **6/39 (15.4%)** ❌ |
| 歷史 `oracle_8g` vs `oracle_4g`（皆 pmax=8） | pool 大小（且 4GB 有 nil） | 3/18 (16.7%) ❌ | 10/18 (55.6%) ❌ |

### 2.3 結論

**1. nil bug 解了。** `pmax=6` 讓 4GB 的 `buffer is nil` 從 **120 次變成 0 次**，
決策一致性同時從 **55.6% → 97.4%**。機制假設（union 塞不下 → 落到 L3-B gather path →
`wt->data` 被指到 host `std::vector` → Metal 層 nil）**獲得證實**。

**2. 但 bit-identical 沒有解。** 即使形狀相同、路徑相同（都走 pool path），
4GB vs 8GB 的 M1 仍只有 **17.9%**。所以「4/6/8 只是速度變慢、數值完全相同」
**用調整 cap 這條路是做不到的**——pool path 的算術本身依賴 slot 佈局 / 換出狀態。

**3. 更關鍵的新發現：cap 本身就是品質參數。** pool 固定 8GB、只把 cap 從 8 改成 6，
M2 就掉到 **15.4%**。也就是說 `CGC_POOL_MAX_TOKENS` 不是純效能旋鈕，改它會改變答案。
因此「不同 pool 用不同 cap」= 用不同形狀算 = 必然不只差速度。

**4. 真正要達成「只差速度」只有兩條路：**

   a. **固定 cap 並讓所有 pool 都塞得下**：所有大小都用同一個（較小的）cap，
      例如全用 `pmax=6`。形狀一致 → 只剩 pool 大小一個變數 → M2 97.4%（仍有 1/39 分歧）。
      代價：大 pool 也被降速。
   b. **讓歸約與 slot 佈局無關**：固定 `mul_mat_id` 的累加順序 / fp32 累加，
      讓 M1 真正收斂。這是唯一能拿到 **M1=100%** 的解法，也是「同一個實作不該有這種行為」的正解。

**5. 目前可用的結論**：4GB 的**退化**（nil + 決策分歧）可以用 `pmax=6` 消除到
「與 8GB 決策 97.4% 一致」；但**不要把 4/6/8 宣稱成 bit-identical**。

---

## 3. 4GB `buffer is nil`：已修復，且改成可證明

### 3.1 修復條件（不是常數，是不等式）

decode hook 只在「本 ubatch 的 expert union ≤ **該層** cap」時走 pool path，否則落到 L3-B
gather path（把 Metal-resident 的 FFN tensor 指到 host `std::vector` → `buffer is nil`）。
一個 ubatch 最多 `cap × topk` 個相異 expert，所以安全條件是：

```
CGC_POOL_MAX_TOKENS × n_expert_used  ≤  min_per_layer_slots − 1     (slot 0 = 保留 ZERO slot)
```

用 `min`（不是 `n_slots`、也不是平均）是因為 hook 比的是**當前層**的 cap。

### 3.2 引擎改動（讓上面那條可被外部驗證）

`llama-expert-cache.cpp` 的 LAYER_CAPS log 多印 `min N/layer`。沒有它，外部只能靠
「這支 prompt 剛好沒有走進溢出的層」來判斷，也就是說 nil 沒出現也不代表安全。

### 3.3 實測（同一 pool、同一模型、只改 cap）

IQ3 denseIQ4X、pool = 4 GiB、port 8091（私用埠）、`enable_thinking=false`：

| cap | union 上限 | min/cap 幾何 | `buffer is nil` | 判定 |
|---|---|---|---|---|
| 8 | 64 | 53 slots（usable 52，headroom **−12**） | **117 次 / 39 層** | ❌ 落 gather path |
| 6 | 48 | 53 slots（usable 52，headroom **+4**） | **0 次** | ✅ 全層 pool path |

所以 4GB 的 nil **是確定的**（117 → 0），而且**先由不等式預測、再由 log 觀察確認**——
這兩件事一致，才讓它從「某次跑起來沒事」變成「可證明不會發生」。

### 3.4 下限是兩個獨立限制

* **算術**：`cap × topk ≤ min_slots − 1`；
* **存活性**：`n_batch` 被夾到 `cap`，而 MTP verify 需要 `n_batch > n_keep_tail`。
  實測 `cap=4` 在啟動 warmup 就 abort：`llama-batch.cpp:609 GGML_ASSERT(n_ubatch > n_keep_tail)`。

因此 4GB 的可用下限是 **6**（不是 4，更不能是 2）。`knifeedge_matrix.py` 的 `MIN_CAP` 記錄此事。

---

## 4. Harness：讓「不同模型 × 不同記憶體」自動成立

`scripts/check/knifeedge_matrix.py` 現在對**每一個** (model × pool) 組合都做四道機器閘門，
不再靠人讀 log：

| 閘門 | 判定方式 | 失敗代表 |
|---|---|---|
| `gate-template` | `POST /apply-template` 看實際 render 出的 generation prompt 是否閉合 | scaffold 壞掉，數字量的是 template |
| `gate-union-fit` | `cap × topk ≤ min_slots − 1`（讀 launch log 的 `min N/layer`） | 算術上可達 gather path（nil 的前置條件） |
| `gate-buffer-nil` | 掃 launch log + CGC log 的 `buffer is nil`，**在真實流量之後**再掃一次 | 該組合已落 L3-B，數字不可用 |
| `gate-oracle` | 對 reference 比 M1/M2（`--oracle-ref`） | 不同 pool 不只差速度 |

`--pool-cap auto`（預設）：對**最小的** pool 先試預設 cap，讀出 `min N/layer` 後用不等式
直接跳到可證安全的 cap，再對**所有** pool 共用它。共用是必要的，不是方便：cap 會夾
`n_batch`，因此決定 ubatch 形狀與 prefill 切法（實測 8GB 下 cap 8 vs 6 的 argmax 只有 15.4% 一致）。

### 4.1 過程中抓到並修掉的 harness 自身缺陷

1. **誤量別人的 server**（最危險）：另一支 agent 用**同一個模型、同一個 8080** 在掃。
   命令列比對無法區分，於是 probe 在 launch 後 10 秒就「通過」——量到的是對方**已載入**的 server。
   現在要求四個事實同時成立：無其他 server、只有一個行程符合 model+port、**該行程持有該埠**
   （`lsof -ti`）、且它是 launch 之後才出現的。量測另用私用埠 8091。
2. **`[log]` 路徑解析**：該行結尾接中文括號，`\S+` 把括號一起吃進去 → `scan_nil` 開到不存在的檔案
   → nil 永遠回報「乾淨」。現在在第一個非 ASCII byte 截斷。
3. **pool 幾何取錯檔**：`n_slots`/`min` 是 C++ 寫進自己的 log，不在 run_server.sh 的 stdout，
   所以 union-fit 對每個組合都回 `unknown`。現在兩個檔都解析。
4. **nil 掃描太早**：在任何請求之前掃，只代表「還沒有 decode 建 union」，不能替組合背書。
   改為 traffic 之後再掃並覆蓋判定。
5. **cap 探測的接受條件太鬆**：`ok is not False` 讓「沒載入的 server」（無 nil、無幾何）也算通過。
   現在要求 `ok is True`：沒有證據 ≠ 通過。
6. **preflight 失敗時把 oracle 結果丟掉**：abort 分支組出的 record 不含 gates，於是「設定已經可疑」
   的時候反而看不到 M1/M2。現在 abort 也保存四道閘門的判定。

---

## 5. 六格矩陣：4/6/8 GiB × iq3/iq4（`--summary` 實際輸出）

固定 `CGC_POOL_MAX_TOKENS=6`（兩個模型在 4GB 下限的 auto 值），`--gates-only`，
`enable_thinking=false`，量測後另一個 agent 的 server 已全部清除。

```
config            cap    rss  free%   tmpl union-fit    nil  M1 bit-id    M2 argmax  preflt
----------------------------------------------------------------------------------------------
iq3_pool4gb         6   5.64     40   PASS PASS (48<=52, +4)   PASS 7/39 (17.9%) 38/39 (97.4%)    FAIL
iq3_pool6gb         6   7.02     29   PASS PASS (48<=78, +30)  PASS 7/39 (17.9%) 35/39 (89.7%)    FAIL
iq3_pool8gb         6   8.44     17   PASS PASS (48<=105, +57) PASS        n/a          n/a    FAIL
iq4_pool4gb         6   5.71     40   PASS PASS (48<=52, +4)   PASS 7/39 (17.9%) 31/39 (79.5%)    FAIL
iq4_pool6gb         6   7.01     29   PASS PASS (48<=78, +30)  PASS 7/39 (17.9%) 30/39 (76.9%)    FAIL
iq4_pool8gb         6   8.47     18   PASS PASS (48<=105, +57) PASS        n/a          n/a    FAIL
----------------------------------------------------------------------------------------------
cap is identical within each model -> pool size is the only variable. OK
```

（8GB 兩列是各自的 reference，所以 M1/M2 為 n/a。）

### 5.1 三個結構性閘門：跨模型、跨 pool 全部通過

* **template**：六格 render 出來的 generation prompt 都是
  `<|im_start|>assistant答：`——閉合、錨點正確。**scaffold 本身不再是問題。**
* **union-fit**：`6 × 8 = 48` 對 52 / 78 / 105 usable slots，headroom **+4 / +30 / +57**，
  六格全 PASS。同一條不等式自動涵藍了 4GB 這種最緊的幾何。
* **nil**：六格 **0 次** `buffer is nil`。4GB 的 nil 從 117 降到 0，iq3 與 iq4 都有相同結果。

### 5.2 兩個沒有過的

**M1（bit-identical）：四組對比全部剛好 7/39 = 17.9%。**
這個「剛好一樣的數字」本身就是最重要的訊息：相同的步數與 iq3/iq4、4GB/6GB 無關，也就是說
**前 7 步逐位元組相同、第 8 步必定分歧**——這是決定性的、與步數綁定的分歧點，不是隨機誤差。

**M2（argmax）：隨 pool 變小而下降**：iq3 97.4% → 89.7%；iq4 79.5% → 76.9%。
所以「4/6/8 只是速度不同」**目前只在結構層成立（路徑一致、無 nil），數值層不成立**。

**preflt（裸問）：六格全 FAIL**，且失敗形式與 pool 無關：

| pool | 模型輸出 | finish |
|---|---|---|
| 8GB | `15+27 等於 15+27` | stop |
| 4 / 6GB | `15+27 等於多少？請只輸出答案`（prompt 回顯） | stop |

也就是說：**scaffold 已經正確、pool 已經正確、路徑已經正確，但裸問仍然不回答。**
這是下一個要打的（48 題品質）的主題，而且它與 expert cache 正交。

---

## 6. M1 分歧：根因、修復、驗證（2026-09-11）

### 6.1 「7/39」不是修辭，它指明了觸發步

把兩份 dump 的**步序**對出來（不是只看總數）：

```
row-hash 相同 idx : [0, 1, 2, 3, 4, 5, 6]      ← 剛好是前綴 7 步
row-hash 不同 idx : [7, 8, 9, ...]
n_tokens 序列     : 1 1 1 1 1 1 1  4 4 4 4  1 1 1  4 4 4 4  1 1 ...
                                ↑ idx=7 是第一個 n_tokens=4
```

所以不是「前 7 步相同」而已，而是：**所有單 token 步都逐位元組相同，第一個多 token 步（n_tokens=4 = MTP verify batch）必定分歧。**

而這個分歧的幅度否定了「浮點序」這個解釋：

| | ref (8GB) | 4GB | 差異 |
|---|---|---|---|
| idx=7 `sum` | −531172.2515 | −593415.506793 | **相對 11.7%** |
| idx=9 `sum` | −489730.937273 | −586775.18508 | **相對 19.8%** |
| idx=7 top-4 | 20 / 22 / 21 / 17 | 20 / 22 / **15** / 21 | 候選集真的不同 |
| `n_cand` | 1334 | 1334 | 覆蓋相同，不是 collapse |

12–20% 不可能是舍入。**差的是某些 expert 根本被讀成零。**

### 6.2 根因：fast path 把 cold expert 映射到 ZERO slot

`run_server.sh` 設了 `CGC_VERIFY_DECODE=1` / `CGC_DRAFT_DECODE=1`，於是 MTP verify batch 走 fast path。
而 fast path 裡有一個開關：

```cpp
// CGC_SYNCFILL_COLD：對本步真正選到的 cold expert 做阻塞式精確填充
// 預設 OFF ⇒ 冷 expert 的 slot_table 仍是 -1 ⇒ remap 寫入 ZERO slot ⇒ 該 expert 貢獻 0
```

**這是「pool 大小影響數值」的機制**：某一步有多少 expert 是 cold，取決於 pool 有幾個 slot。
4GB 的 cold 比 8GB 多 → 被丟掉的 expert 多 → 輸出不同。單 token 步不走 fast path，所以完全不受影響——正好對上「前 7 步相同」。
程式碼自己的註解早就記過這個症狀（L4 pool + fast path 下 15+27 十連發 0/10），只是預設一直是 OFF。

### 6.3 修復

`llama-context.cpp`：`CGC_SYNCFILL_COLD` **預設改為 ON**，並修掉值的解析——舊寫法
`e[0] == '1'`／`!= nullptr` 會把 `=0` 當成「開」，所以現在 `nullptr` ⇒ ON、`'0'` 開頭 ⇒ OFF。
`CGC_SYNCFILL_COLD=0` 仍可還原舊行為，ZERO slot 保留作為 `ensure_slot` 失敗時的安全網。

### 6.4 驗證（iq3 denseIQ4X，cap=6，4/6GB 對 8GB）

| 對比 | 修復前 M1 | 修復前 M2 | **修復後 M1** | **修復後 M2** |
|---|---|---|---|---|
| iq3 4GB vs 8GB | 7/39 (17.9%) | 38/39 | **39/39 (100%)** | **39/39** |
| iq3 6GB vs 8GB | 7/39 (17.9%) | 35/39 | **39/39 (100%)** | **39/39** |

閘門從 `CHECK` 變成 `PASS`。**「4/6/8 只差速度」現在在 iq3 上成立到逐位元組層。**

代價：4GB 下 decode **6.13 vs 6.63 tok/s**（syncfill 開/關，單次樣本）——約 8%，
而且這 8% 買到的是「pool 大小不再改變答案」。

### 6.5 iq4 的空內容：不是 pool、也不是 M1，是 harness 把錨點關掉了

iq4 在 4GB 只吐 ~5 個 token 就 EOS（`content=''`、`finish=stop`），dump 裡完全沒有 n_tokens=4 的步。
先把它拆開：

| 診斷 | 結果 |
|---|---|
| `POST /completion`（不經 chat template，prompt `15+27 =`） | **` 42`** — 權重是對的 |
| chat 模式 | `completion_tokens: 1`、`content: ""`、`finish: stop` |
| 4GB vs 8GB | **完全一樣** — 所以與 pool 無關 |
| 渲染出的 prompt 尾端 | `assistant thinking\n\n</think>`，**後面什麼都沒有** |

而 `run_server.sh` 自己有一個錨點：`assistant_prefill:"答："`（來自 `CGC_SERVER_CHAT_AB=healthy-prefix`），
但它只在 chat-template-kwargs **為空**時才設定：

```bash
if [ -z "$SERVER_CHAT_TEMPLATE_KWARGS" ]; then
    SERVER_CHAT_TEMPLATE_KWARGS="{\"assistant_prefill\":\"答：\",...}"
fi
```

而這個 harness **永遠会傳 kwargs**，所以每次都在無意中把錨點關掉。
把兩把鑰匙放在同一個 kwargs 字串裡就是修復：

```json
{"enable_thinking": false, "assistant_prefill": "答："}
```

（`enable_thinking` 給 GGUF embedded template；`assistant_prefill` 給 custom jinja。iq3 的模板會忽略用不到的鍵，所以一個值對兩個模型都安全。）

效果：**`completion_tokens` 1 → 14、reference dump 5 行 → 39 行、渲染 prompt 變成 `assistant答：`**（與 iq3 一致）。

### 6.6 最終六格：M1/M2 全部 100%

> ⚠️ **更正（見 §7）**：這一節的 `iq3` 與 `iq4` 兩欄其實是**同一個檔案**——`iq3` 是 symlink 指向
> `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`。所以「兩個模型都 100%」實際上是**一個模型被量了兩次**。
> 逐位元組一致這件事本身仍然成立（它是 pool 不變性的證據），但**不能**當成跨模型證據。

```
config            cap    rss  free%   tmpl union-fit    nil  M1 bit-id    M2 argmax  preflt
----------------------------------------------------------------------------------------------
iq3_pool4gb         6   5.65     41   PASS PASS (48<=52, +4)   PASS 39/39 (100.0%) 39/39 (100.0%)    FAIL
iq3_pool6gb         6   7.04     30   PASS PASS (48<=78, +30)  PASS 39/39 (100.0%) 39/39 (100.0%)    FAIL
iq3_pool8gb         6   8.46     18   PASS PASS (48<=105, +57) PASS        n/a          n/a    FAIL
iq4_pool4gb         6   5.63     41   PASS PASS (48<=52, +4)   PASS 39/39 (100.0%) 39/39 (100.0%)    FAIL
iq4_pool6gb         6   7.13     29   PASS PASS (48<=78, +30)  PASS 39/39 (100.0%) 39/39 (100.0%)    FAIL
iq4_pool8gb         6   8.38     17   PASS PASS (48<=105, +57) PASS        n/a          n/a    FAIL
----------------------------------------------------------------------------------------------
cap is identical within each model -> pool size is the only variable. OK
```

**「4/6/8 只差速度」現在在 iq3 與 iq4 上都成立到逐位元組層。**

### 6.7 唯一沒過的：preflt（裸問不回答）

> ✅ **已修復，見 §7.2**：根因是模板把 think 區塊與錨點寫成互斥（`if prefill_text … else think`），
> 而這顆模型**兩者都需要**。改成兩者都輸出後，preflt 六格全 PASS。

六格都是同一個形狀：模型 **回顯 prompt** 而不是作答（`15+27 等於 15+27` / `15+27 等於多少？請只输出答案`），
但已不是空內容，也與 pool 無關。scaffold 已正確、路徑已正確、數值已逐位元組一致——
所以這是最後一個、也是與 expert cache 正交的問題：它決定 48 題能不能量出有意義的分數。

## 7. 更正與續報（2026-09-11 深夜）

### 7.1 「跨模型」從來沒有發生：矩陣的兩欄是同一個檔案

在追 4GB 的 M1 殘差（77/88）時，先做了最便宜的一步：**比對兩份 oracle dump 的 md5**。

```
0040238fa46a9f35d7ffb5fab2cb5e76  oracle_iq3_pool4gb.jsonl
0040238fa46a9f35d7ffb5fab2cb5e76  oracle_iq4_pool4gb.jsonl     <- 完全相同
7a608f16a808201122bc0cbc18a0aae3  ref_iq3_pool8gb.jsonl
7a608f16a808201122bc0cbc18a0aae3  ref_iq4_pool8gb.jsonl        <- 完全相同
```

原因在檔案系統，不在數字：

```
models/gguf/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf
    -> Qwen3.6-35B-A3B-UD-IQ4_XS.gguf     (symlink, 9/11 14:49 被改成指向 IQ4)
models/gguf/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf   18,209,036,576 bytes  (唯一的真檔)
```

真正的 IQ3_XXS-denseIQ4X（13.6 GB）只在**外接硬碟**上：
`/Volumes/AlexZhuang/Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf`
（內接只剩 10 GiB，放不下 13 GB）。

**為什麼這件事很值得記下來**：矩陣的「M1/M2 全部 100%」看起來是漂亮的跨模型證據，實際上
只是同一個檔案被量了兩次——它會產生完美自洽的數字。這跟先前的 false-passing cap probe 是
**同一類缺陷**：量測無法察覺自己沒有量到它宣稱的東西。

順帶更正一個我一直寫錯的前提：**chat template 不是由 model kind 決定的**。`run_server.sh` 只有
在 `CGC_SERVER_PROFILE` 或顯式 override 之下才換 template，否則一律用
`Qwen3-nothink-ChatML.jinja`。所以 MODELS 註解裡「iq3 → embedded template / iq4 → custom jinja」
是錯的；兩欄除了權重之外**沒有**任何差異。

**修復（hard gate，不是 warning）**：`gate_model_identity()` 在任何 cap probe 或啟動之前，
對每個欄位算 `realpath + size + 前 1 MiB 的 sha256`，兩欄指到同一組權重就**拒絕量測**：

```
$ python3 scripts/check/knifeedge_matrix.py --models iq3,iq4 --pools 8
[model-id] iq3: realpath=.../Qwen3.6-35B-A3B-UD-IQ4_XS.gguf exists=True size=18209036576
[model-id] iq4: realpath=.../Qwen3.6-35B-A3B-UD-IQ4_XS.gguf exists=True size=18209036576
REFUSING to measure: model identity check failed.
  - iq4 and iq3 resolve to the SAME weights (realpath=..., size=18209036576, head=5178a265e0aa4fde).
    A cross-model comparison would be two measurements of one file.
```

同時矩陣新增 `path` 覆寫（`model_path()`），讓外部硬碟上的真 IQ3 可以直接被量到，不需要
複製 13 GB 進內接。為此新增 `iq3ext` 一欄，指向 `/Volumes/AlexZhuang/...IQ3_XXS-denseIQ4X.gguf`。
外部讀取速度是乾淨量測的 **86 MB/s**（先前 7 MB/s 是與 `dd` 互搶造成的假象），所以小 pool 可以在
執行時從外接填滿。

### 7.2 preflt：模板把 think 區塊與錨點寫成互斥（已修復）

`Qwen3-nothink-ChatML.jinja` 的 generation prompt 是：

```jinja
{%- if prefill_text %}{{ prefill_text }}{%- else %} thinking\n\n</think>{%- endif %}
```

而裸問得不到答案的原因，是把兩個變體都餵給**同一個渲染 prompt** 後比出來的：

| 送進去的 generation prompt | 輸出 |
|---|---|
| `assistant\n\n`（無錨點） | `\n\n15+27 = 42\n</think>…` ← 算對了但重複 marker |
| `assistant答：`（有錨點、無 think 區塊） | 回顯 prompt |
| **`assistant\n\n答：`（兩者都有）** | **`15+27=42`** ✅ |

也就是說：**空 think 區塊與錨點不是替代關係，兩顆模型兩個都要**（`</think>` 標記 assistant turn 的
起點，錨點 `答：` 把輸出從 echo 拉回作答）。修法是把 generation prompt 改成兩者都輸出：

```jinja
{{- '<|im_start|>assistant\n\n</think>\n\n' -}}
{%- if (prefill_text | default('')) -%}{{- prefill_text -}}{%- endif -%}
```

結果：**preflt 六格全部 PASS**，模型穩定答 `15+27=42`。這是純模板（runtime data）改動，不需重建。

### 7.3 第三個 false-pass 類缺陷：把「啟動失敗」快取成「結果」

修正 `gate_model_identity` 之後第一次跑真跨模型矩陣，沒有任何一格跑起來，但過程完全靜默。
原因是**兩層缺陷疊在一起**：

1. `run_server.sh` 有 memory guard，在 full-MTP 模式下 `free < 40%` 就拒絕啟動：
   ```
   [guard] warning: 系統可用記憶體僅 15%（<25%）
   error: startup blocked by memory guard -> full-mtp: free=15%<40%
   ```
   這是正確的保護（機器當時 swap 已滿），不是 bug。
2. 矩陣把這次失敗**寫成結果檔** `knifeedge_<label>.json`（`{"up": false, ...}`），
   下一次執行看到檔案就印 `SKIP (result exists)`——於是**一個從未量過的配置被永久略過**。

這是同一個陷阱的第三次出現（前兩次是 false-passing cap probe 與「同一顆模型兩欄」）：
**量測系統把「沒量到」誤記成「量到了且通過」**。現在 skip 的條件改成必須有證據：

```python
prev = json.load(open(out_json))
if prev.get("up") and (prev.get("gates") or prev.get("skipped") is None):
    print(f"[{label}] SKIP (complete result exists; --force to redo)")
    return prev
print(f"[{label}] REDO: the cached result is not a measurement ...")
```

驗證：memory-guard 產生的 stub（`up=False`）現在走 `REDO`，而舊的完整結果（`up=True` 且有 gates）仍走 `SKIP`。

**配套：在啟動前就把機器的記憶體當成前置條件。** 單靠 skip 規則只修正了「事後快取」，仍然會白白做
六次注定失敗的啟動。現在 `--min-free-pct`（預設 40，對齊 run_server.sh 的 guard）在任何 launch 之前檢查：

```
[iq3ext_pool4gb] REFUSING to launch: machine is 9% free, below the 40.0% precondition
    (run_server.sh's memory guard would reject it: 'startup blocked by memory guard').
  This is a MACHINE precondition, not a result. Free memory ... and re-run;
  nothing was measured for this cell.
```

而且這種 cell **不寫任何結果檔**（所以下次一定會重試），摘要表也把它印成原因而不是「模型失敗」：

```
iq3ext_pool4gb      -- machine below --min-free-pct -> NOT measured, nothing cached
```

### 7.4 跨模型矩陣還沒跑完：機器記憶體耗盡（待處理）

真 IQ3 確實能起來且幾何明顯不同——**4 GiB pool 下 IQ3 是 71 slots/層，IQ4 只有 53**
（IQ3 較小且 denseIQ4X 的 expert 較省空間），所以 cap 探測結果也不同：**iq3ext → cap=8（union 64 ≤ usable 70）**、
**iq4 → cap=6（union 48 ≤ usable 52）**。這是目前唯一真正來自兩顆不同檔案的跨模型證據。

但矩陣跑不完，原因不是程式：

```
Pages wired down:                131702   ->  2.0 GB
Pages occupied by compressor:    759237   -> 11.6 GB   (壓縮了 2162799 頁 ≈ 33 GB)
Pages free:                        4121   ->   66 MB
swap: total = 16384 MB, used = 16384 MB, free = 0 MB
sum RSS(all processes)          = 1.76 GB
```

沒有任何單一行程佔大量 RSS（最大的只有 0.93 GB，是 Freebuff 自己），所以這 11.6 GB 是
**壓縮器裡的全域殘留**（反覆載入 13–18 GB 模型累積的結果）。`purge`（已以 root 執行）只清 file cache，
無法回收活行程的 anonymous 壓縮頁。在這種狀態下：

* memory guard 會（正確地）拒絕啟動；
* 即使繞過 guard（`CGC_SERVER_MEMORY_MODE=prod`），swap 已滿也會讓載入變成 thrash，速度數字不可信。

所以跨模型矩陣需要在記憶體恢復後（最乾淨的做法是重開機，清掉 compressor 與 swap）才能完成。
本次已備好：`gate_model_identity`、`path` 覆寫與 `iq3ext` 一欄、skip 規則修正、以及 detached 驅動
`/tmp/run_crossmodel2.py`（gates-only — 48 題套件是 48×(3 greedy + 5 seeded) 次生成／格，在 USB I/O 下要數小時，
所以它獨立於一致性表之外執行）。

