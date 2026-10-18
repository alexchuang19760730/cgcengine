# 判詞：replay 檢查 11 coding 質量退化實為「stop 開 fence 誤命中」（測試載荷缺陷，非引擎/模型退化）

- 日期：2026-10-19（JST）
- 觸發：commit2（budget v3 口徑 + 整組只讀診斷探針，env-gated 預設 OFF）提交時 pre-commit 檢查 11 replay 質量硬閘門
- 當前 HEAD：`58abe27ad`（commit2 兩次提交均被攔，未產生新 commit）
- 結論等級：**根因釘死（HEAD 二進制復現）**。coding 空回覆是 replay 測試載荷的 stop 字串缺陷（裸 `" ``` "` 在程式碼「開 fence」即命中），**不是模型質量退化、不是 commit2 引入、不是工作區未提交 dylib 引入**。修復方式為把 coding 專用 stop 改成閉 fence `"\n```"`，只修正停止位置、**不調整任何質量門檻**。

---

## 1. 現象

commit2 第二次提交，檢查 11 replay 跑完（`PASS 11 replay benchmark 跑完`），但質量硬閘門：

```
quality regression (hard gate): 1 profile(s) degraded
  coding: baseline 1.0 → current 0.6（scores [0.6,0.6,0.6]，variance 0，確定性復現）
          content=""  completion_tokens=1  finish_reason=stop
          length check fail（actual=0 < 50）、required_any fail（無 def/fibonacci/return）
```

其餘 profile（qa-zh / longform-zh / math / reasoning / writing / translation）質量均 neutral/improvement；speed 因 baseline 跨時代（MTP ON、decode 5–7）全標 unstable 不計；peak_rss 全改善。聚合 quality 1.0→0.94 中性，但 coding 單 profile 硬閘門 FAIL。

## 2. 載荷

`scripts/check/replay_server_profile.py`：

- coding prompt（斐波那契）：`用 Python 寫一個函數計算斐波那契數列的第 n 項。`
- `assistant_prefill`（`CODING_ANCHORS`）：`` ```python\ndef fibonacci(n):\n    `` —— **開頭就是開 fence ` ``` `**
- stop 列表（修復前，build_payload coding 分支）：`["```", "<|end|>", "<|output|>", "<|user|>"] + CHATML_STOPS`
- temperature=0（greedy）、seed=0、presence_penalty=1.5、max_tokens=512

## 3. 對照實驗（A/B/C，工作區 dylib = 10-17 構建，含 commit2 探針 + 未提交功能，全 env-gated OFF）

直接 import 腳本 `build_payload` 打 prod-new:8080，每個變體只動一個因子：

| 變體 | stop 含裸 ` ``` ` | assistant_prefill | 結果 |
|---|---|---|---|
| **A 原樣** | 是 | ` ```python\ndef fibonacci...` | finish=stop、**1 token、content 空**（復現 replay FAIL） |
| **B 去掉 fence stop** | 否 | 同上 | finish=stop、**88 token、完整正確 fibonacci**（`if n<0 raise.../for _ in range(2,n+1)...return curr`） |
| **C 去掉 prefill** | 是 | 無 | finish=stop、**1 token、content 空**（模型自然首 token 即開 fence，立即命中） |

**A/B/C 證明**：
1. 模型完全正常（B 生成完整正確程式碼）——無質量退化。
2. 致命因子是 **stop 串裸 ` ``` `**：無論開 fence 來自 prefill（A）還是模型首 token（C），都在程式碼「開 fence」當下命中 stop，返回空 content（stop 串被剝離、不計入 content）。

## 4. 決定性實驗：HEAD 已提交二進制同樣復現（排除 commit2 / 工作區）

為排除「工作區 dylib 未提交功能（affinity/NCMOE/metal 改動）引入」，做二進制對換：

1. 備份工作區 4 dylib 到 `/tmp/cur_dylib/`；從 git 對象提取 `HEAD:build/bin/*.dylib` 到 `/tmp/head_dylib/`（hash 均不同）。
2. 停 server → 僅覆蓋 build/bin 4 dylib 為 HEAD 版（**不動 git index**，staged 10 檔全程保留）→ 起 prod-new（PID 6675，`lsof` 確認載入的 4 dylib 全部 = HEAD）。
3. 對 HEAD 二進制打 coding A（原樣）：

```
HEAD 二進制（58abe27ad，G1 已提交，不含任何未提交功能）· coding A 原樣
finish: stop | completion_tokens: 1 | content: ''
```

**HEAD 同樣 1 token 空回覆。** 徹底排除：
- 不是 commit2（commit2 尚未提交，HEAD 不含它）；
- 不是工作區 dylib 未提交功能（HEAD dylib 不含 affinity/NCMOE/metal/探針）；
- 是 HEAD 已提交程式碼庫就存在的 stop 匹配行為。

程式碼範圍佐證：`tools/server`（chat completion / stop 序列處理）工作區**零改動**；`common/` 只有 `speculative.cpp` +132 行（MTP spec-tree，且 MTP OFF 路徑不啟用）；未提交 src 改動全在 MoE 計算/專家快取/Metal kernel 層，無一觸碰 stop 字串或 assistant_prefill 處理。

## 5. stop 串修復驗證（D/E，HEAD 二進制）

| 變體 | coding stop | 結果 |
|---|---|---|
| **D 閉 fence `"\n```"`** | `["\n```", "<|end|>", ...]` | finish=stop、**87 token、完整 fibonacci**、content 258 字元、末尾 `return curr`（閉 fence 被剝離，content 為純程式碼）、含 def/return/fib |
| **E 去掉 fence** | 無 fence，靠 eos | finish=stop、**88 token、完整 fibonacci**，末尾模型自收閉 fence |

**D 的 87 token 與 baseline（dbe5f8eb1）coding 的 87 token 完全一致**——閉 fence 正是 baseline 時代的隱性預期行為。採用 D（語義最精確：寫完整段程式碼後在「行首獨佔閉 fence」停止，開 fence ` ```python ` 因前無換行、後接語言標識而不命中）。

## 6. 為什麼 baseline（dbe5f8eb1）能過

baseline coding content 是完整程式碼（`if n < 0: ... return curr`，87 token，quality 1.0，content 從續寫部分起、不含 prefill 錨點），代表當時 stop 只在模型寫完程式碼、輸出**閉** fence 時才命中。`dbe5f8eb1..HEAD` 之間 server/chat 層僅有一個提交：

- `654ca6682 engine(spec-tree): 修 seq-1 同輪雙加 —— 播種那一步不再同時起草該 seq`

該提交（spec-tree / 種草 token 流）可能改變了 stop 串對 assistant_prefill / 首 token 的匹配邊界，使開 fence 提前命中。**未逐一二分該提交**（誠實邊界）；但它已在 HEAD、屬已提交歷史，commit2 無法回退也不為其負責。核心結論不變：此行為在 HEAD 確定復現，與本次 commit2 無關。

## 7. 修復（只動測試載荷，不動門檻）

`scripts/check/replay_server_profile.py` build_payload coding 分支：

```diff
-            "stop": ["```", "<|end|>", "<|output|>", "<|user|>"] + CHATML_STOPS,
+            # 裸 "```" 會在開 fence（assistant_prefill 開頭 ```python / 模型首 token 開 fence）即命中
+            # → 空回覆；改用閉 fence "\n```"（行首獨佔、前有換行），開 fence 不命中、閉 fence 才停。
+            "stop": ["\n```", "<|end|>", "<|output|>", "<|user|>"] + CHATML_STOPS,
```

- 僅影響 coding profile（其餘 profile 用 `stops_std`，不含 fence，未動）。
- **未改任何質量判定**：length（≥50）、required_any（def/fibonacci/return）、forbidden、finish_reason、loop_run/loop_phrase 全部保留原值。
- 這不是「讓失敗 commit 矇混過關」，而是修復一個**失效的檢查**——原裸 fence stop 使 coding 質量門在開 fence 就終止、永遠測不到任何程式碼（false negative）；修復後該門才真正去驗證「模型能否寫出完整正確程式碼」。

## 8. 復現命令

```bash
# 本機 curl/python 打 localhost 必須繞代理（--noproxy '*' / ProxyHandler({})）
cd /Users/alexchuang/Documents/flashkv-devserver
CGC_SERVER_PROFILE=prod-new bash scripts/run_server.sh     # 或 CGC_DETACHED=1
# A/B/C/D 變體：見本判詞 §3/§5，用 build_payload 構造後 POST http://127.0.0.1:8080/v1/chat/completions
```

## 9. 後續

- 重跑全 profile replay（工作區 dylib、修復後腳本）確認 coding 質量門恢復、其餘 profile 不退化，再讓 commit2 過檢查 11。
- baseline `.replay_bench_baseline.json` 建於 dbe5f8eb1（MTP ON、decode 5–7、老 pool），與當前 prod-new（MTP OFF、decode 17+、新 budget）跨時代，speed 全 unstable；是否在當前良好口徑上重建 baseline 為獨立決策（見當輪交接，不在本判詞範圍）。
- `654ca6682` 是否改變 stop 對 prefill 的匹配語義，可作為可選的單獨二分項（不阻塞 commit2）。

---

## 10. 第二道獨立門：compare reference 檔從未入庫（幽靈依賴，rc=2 ERROR）

stop 修復後重跑檢查 11，replay 本體成功，但**比較步驟** `replay_bench_compare.py` 以 rc=2 失敗：

```
ERROR: failed to load JSON: [Errno 2] No such file or directory:
  scripts/check/replay_server_bench_reference.json
```

### 10.1 三路顯式確認：該檔在 HEAD / index / 工作區全部不存在

```bash
git ls-files -s -- scripts/check/replay_server_bench_reference.json   # 空（index 無）
git cat-file -e HEAD:scripts/check/replay_server_bench_reference.json # 缺失（HEAD 無）
test -f scripts/check/replay_server_bench_reference.json; echo $?     # 1（工作區無）
```

它是 hook 檢查 11（`scripts/check_build_tracked.sh`）的**幽靈依賴**：hook 固定以 `REPLAY_REF=scripts/check/replay_server_bench_reference.json` 跑 `replay_bench_compare.py --reference`，該參數為 required，`_load` 失敗即 rc=2（ERROR）。該檔曾於 23:40 在工作區臨時存在、compare 因此跑通一次，後被刪除、**從未入庫**——乾淨 checkout 必撞此門。

### 10.2 為什麼不能只放 `_compare_thresholds`（會拆質量門）

compare 只從 reference 讀 `_compare_thresholds`；replay 對缺 profile 規則的容錯路徑是改用**三因子評分**（`evaluate_quality`：factor1 count-cold / factor2 zero-slot / factor3 memory，全部只讀 `/cgc_stats` 引擎狀態，**不看 content 是否為空**）。若 reference 只有 thresholds、無 per-profile 傳統規則，則 coding 空回覆（1 token、content 空）三因子仍可能 = 1.0 → **空回覆也判 1.0，等於把質量門拆了**。因此必須恢復 per-profile 的 length / required_any / forbidden 等傳統規則（空回覆正是靠 `length_min=50` fail 才被壓到 0.6）。

### 10.3 零轉寫恢復（內容全部來自真實來源，非編造）

恢復腳本 `/tmp/gen_reference.py`（一次性生成器），三個來源：

1. **prompts**：`replay_server_profile.py` 內建 `PROFILE_PROMPTS`（源碼註解明載「mirror reference v2: 3 prompts per profile」），用 `ast.literal_eval` 直接提取，不手敲。
2. **qa-zh / longform-zh / coding 傳統規則**：從 baseline（`.replay_bench_baseline.json`，HEAD=dbe5f8eb1 已知良好）的 `quality.checks` 回顯**反推**——`evaluate_quality` 會把 `length.min/max`、`required_any.keywords`、`forbidden.keywords`、`finish_reason.ok`、`loop_run.limit` 原樣寫進 check 結果。baseline 也證實 reference 本來就只覆蓋這 3 個核心 profile（math/reasoning/writing/translation 在 baseline 即 `no_reference` 三因子）。
3. **`_compare_thresholds`**：從 23:40 實際 gate 實錄 `/tmp/.replay_bench_report.json` 的 `summary.thresholds` 原樣（decode/prefill 30%、peak_rss 25%、quality 0.1，2026-09-06 校準，含 `_doc` 雜訊說明）。

產出 `scripts/check/replay_server_bench_reference.json`（合法 JSON）：3 核心 profile 帶 prompts + 完整傳統規則；其餘 4 profile 只帶 prompts（走三因子兜底，與 baseline 一致）。

### 10.4 驗證（新 reference、同口徑重跑）

以新 reference 重跑 `--all-profiles --runs 3 --warmup --seed 0`（工作區 dylib、修復後腳本）：

- **7 個 profile quality 全 1.0**，coding 為完整 fibonacci（傳統規則口徑，非三因子漏檢）。
- compare rc=0、**verdict: PASS**，`improvements=5 regressions=0 neutral=3 unstable=4`；quality 全 neutral（1.0 vs 1.0），speed 因跨時代 baseline 全 unstable 不計，peak_rss 全改善。

至此檢查 11 兩道獨立門（stop 開 fence + reference 幽靈依賴）皆修復且**未降低任何門檻**；reference 正式入庫後乾淨 checkout 可復現該 gate。
