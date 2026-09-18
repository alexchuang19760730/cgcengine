# http_duo 的 prefill 必 400，與一整排「全機獵殺」缺陷（2026-09-18）

本文記錄兩個缺陷的根因與修法。**它的性質是「修好了」而不是「量到了新性能」**：本文唯一可引用的性能數字
都標了「不可引用」，因為取得它們時機器是 HEAVY 熱態、並有另一條線在同一台機器上工作。

---

## ① prefill 軸在 `prod25` 上必 400

**現象（原文）**

```
srv  send_error: request (4500 tokens) exceeds the available context size (4096 tokens)
```

**根因不是 server，是這個工具的假設**：`http_duo.py` 的 prefill prompt 是寫死的 ~4500 token
（預設 profile 是 `prefill250`，`CTX` 8192/5632，塞得下）。換到 `prod25`（`run_server.sh:463`
的 `CTX=4096`）同一支 prompt 就一定溢出。所以「prefill 那半是為大 ctx profile 寫的」——不是
「`prod25` 不能量 prefill」。

**修法：讓 prompt 由 profile 自己的 ctx 反推，而不是改寫一支新的固定 prompt。**

| 步驟 | 來源 | 為什麼是它 |
|---|---|---|
| 取 ctx | `run_server.sh CGC_DUMP_ENV=1` → `CGCENV CTX` | 唯一直相源（與 `llama_bench_matrix.resolve()` 同一條路徑），零 GPU |
| 交叉驗證 | 活著的 server `GET /props` → `default_generation_settings.n_ctx` | dump 是「將要主機 starting」，props 是「現在真的在跑」 |
| 量 token | `POST /tokenize` → `{"tokens":[…]}` | 用 server 自己的 tokenizer，不用估 |
| 決定長度 | budget `= ctx − n_predict − reserve(64)`，二分求最大 unit 數，再向下修到 target | 任何 profile 都不會撞 ctx |

`--prefill-target-tokens` 預設 **2048**，刻意對齊 llama-bench 的 `prefill-house`（2048）⇒ 兩台儀器的
prefill 第一次落在**同一個 prompt 長度**上，可以直接互比（過去 4500 / 2873 / 2048 是三個不同的量，
被當成過「退步」）。

塞不下時（比 `--prefill-min-tokens` 還小），該軸報 **NOT APPLICABLE ＋ 算式**，**不發那支會 400 的請求**，
總結欄寫 `bar 250/12 -> NOT EVALUABLE on this profile`（不再把 `meets_bar=None` 讀成 PASS）。
想親眼看 400 就加 `--prefill-force`。

**live 驗證（`prod25`，2026-09-18 19:11，port 8080）**

```
resolved CTX (from run_server.sh CGC_DUMP_ENV=1) = 4096
n_ctx reported by the live server (/props) = 4096  -> using 4096
prefill prompt: 27 units = 2025 tokens (budget 4016 = ctx 4096 - n_predict 16 - reserve 64)
  rep1 prefill  30.67 t/s (2025 tokens)
  rep2 prefill  30.53 t/s (2025 tokens)
  rep3 prefill: REFUSED, still MODERATE
  rep3 decode    8.28 t/s (128 tokens)
=> bar 250/12 -> FAIL
```

**⚠ 那些 t/s 不可引用**：當下熱態 HEAVY→MODERATE、`free 14%`、同一台機器上有另一條線在做事，而且只剩
1 個可用樣本（reps=3 扣掉 warmup，又有 reps 被熱閘門拒）。這段輸出只能證明「**走得通**」——從必 400 變成
有數字；不能證明「prod25 prefill ≈ 30」。要引用請在 NOMINAL、空機、`reps≥4` 重跑。

---

## ② 一臂的 server 被外部 SIGTERM：假設被推翻，真因另有其人

原本的假設是「ABBA 用 `nohup` 起服務、沒新開 session，宿主回收兄弟命令時整個進程組連坐」。同一天稍後
的第二次死亡**發生在前景命令還活著時** ⇒ **直接推翻**（不是 cmdline(User) 的錯，也不是 `run_server.sh` 的錯，
是我前一天給自己的解釋太快）。

**真因：`run_server.sh` 的 preflight 是全機獵殺。**

- `CGC_PREFLIGHT_PATTERNS` + `pgrep -f`（**不看 port、不看 session**）：任何一條線起 `run_server.sh` 都會把
  別人的 `llama-*` SIGTERM 掉；
- 而且它**排在 memory guard 之前** ⇒ 就算自己隨後被 guard 擋下、沒能起來，**被殺的那一輪也已經死了**
  （對方的量測整天白跑，自己這輪也沒成果）；
- 更廣的是：**每次「查 env」（`CGC_DUMP_ENV=1`）都可能殺人** —— 那是 `llama_bench_matrix`／`prod_matrix`／
  `m123_oracle_gate`／`phase_split_ab` 的共同入口。

**改成：**

| 位置 | 舊 | 新 |
|---|---|---|
| preflight 預設 | 送 `SIGTERM`（→`-9`）清掉所有匹配行程 | `CGC_PREFLIGHT_KILL=1`：**只列出 `pid / etime / command`，不送任何訊號** |
| 真的要清場 | （不需要，預設就在清） | 明示 `CGC_PREFLIGHT_KILL=all` |
| `CGC_DUMP_ENV=1` | 會跑 preflight、會被 STALE/guard 擋住 | **跳過 preflight 與全部閘門** ⇒ 別人在跑時也查得到解析結果 |
| `http_duo.stop()` | `pkill -9 -f llama-server`（**全機通殺，同族缺陷**） | 只停自己的 pid；只**回報**自己 port 的佔用者 |
| 行程隔離 | 無 | `start_new_session=True`（另外補 `pool_curve.py`、`mtp_long_sequence_verify.py`） |
| port | 固定 8080，先到先得 | `--port auto`（8080 起第一個空的）⇒ 兩條線可各跑一台 |
| server 被殺 | 三排安靜的 `None` | 每 rep 重查存活；收尾掃 log 的 `[CGC] Received SIGTERM`（watchdog 走 `GGML_ABORT`，不會誤報）⇒ 印診斷、**exit 1**、json 記 `valid:false` |

列出來的第二欄是 **`etime`**：判斷「自己的殘留」還是「別人剛發的量測」，年齡是最有用的單一資訊。
後面的 STALE 硬檢查照舊擋住啟動 ⇒ **沒有拆掉 GPU OOM `ret=-3` 的安全網**，只是不再替你動手殺人。

---

## 體檢（可證偽，零 GPU）

```sh
python3 scripts/check/http_duo.py --self-test
```

11 項 PASS：prefill budget 算術（`ctx1024 + n_predict128` ⇒ NOT APPLICABLE、`n_predict > ctx` ⇒ budget ≤ 0）、
SIGTERM 掃描（健康日誌不誤報、`GGML_ABORT` 不算 SIGTERM）、`pick_port` 取到空 port。

**preflight 的二分對照**（用假行程，不碰 GPU）：編一支 `pause()` 放到 `.../build/bin/llama-server`

> 坑：路徑必須含 `build/bin/` **且 argv[0] 的 basename 要落在 `CGC_PREFLIGHT_NAMES`** —— 用
> `/bin/sh` 包 script（argv[0] 是 `/bin/sh`）或直接 `cp /bin/sleep`（arm64e，換 path 跑不起來）
> 兩種做法 hit 不到 preflight 的分類，會得到假的「通過」。

- 預設 ⇒ **假行程存活**，且 STALE 硬檢查把啟動擋住；
- `CGC_PREFLIGHT_KILL=all` ⇒ 假行程被清。

---

## 復產指令

```sh
python3 scripts/check/http_duo.py --profile prod25        # prefill 自動縮到 2025 tokens
python3 scripts/check/http_duo.py --profile prefill250 --reps 4 \
    --json Backup/prod_matrix/http_duo_prefill250.json
CGC_PREFLIGHT_KILL=all ./scripts/run_server.sh            # 確定只有自己時才清場
CGC_SERVER_PROFILE=prod25 CGC_DUMP_ENV=1 ./scripts/run_server.sh | grep '^CGCENV CTX'
```

未處理：`prod_matrix.cell_command()` ~line 319 的 batch 硬編碼（工具 `-b 512` vs server `-b 5632`）
—— 這是「服務路徑 vs bench 差多少」目前唯一沒對齊的一格。
