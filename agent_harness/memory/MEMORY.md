# flashkv-devserver — 專案長期筆記（索引入口）

> **這是快照，不是權威副本。**
> 權威位置：`.workbuddy/memory/MEMORY.md`（由 host 持續寫入）。
> 本檔於 2026-09-20 由 `agent_harness/scripts/import_harness_snapshot.py` 複製進 repo，唯一目的是讓 `agent_harness/`
> 底下的內容能被 `agent_harness/scripts/auto_git_push.ps1` 定時推送；原檔改了這裡**不會**自動跟上。
> 索引與漂移檢查見 `agent_harness/engine_loop/memory/INDEX.jsonl`。

llama.cpp 的 CGC fork：Metal ＋ **expert cache pool**（專家權重常駐 SSD→池）。根
`/Users/alexchuang/Documents/flashkv-devserver`（**git worktree**，`.git` 是檔案）。
逐日經過在 `.workbuddy/memory/YYYY-MM-DD.md`（append-only，本線用 `§EN-` 前綴）。

## 這個目錄怎麼讀（09-17 拆分；**本檔要維持小**）

- ⚠️ **「~10KB」的單位是「字元」，不是 bytes**（2026-09-20 實測）。本檔 **7791 字元 / 11990 bytes**
  （CJK 比例 1.54），而**它的尾巴在注入的記憶區塊裡是可見的** ⇒ 沒有被截斷。
  **不要把 11990 bytes 當成超限而去瘦身** —— 那會白動一個 7.8k 字元的脊椎檔（本線 09-20 差點就做了）。
  真正要維持小的理由是「它是每次動手前都要讀的那一份」。

**1 索引（本檔）＋ 3 主題檔**，三者是同一份長期記憶（不是歷史存檔），與本檔同齡。

| 檔 | 什麼時候讀 |
|---|---|
| **`MEMORY.md`**（本檔） | 每次動手前 |
| `MEMORY_PERF.md` | **要引用任何 t/s／profile／幾何／散熱數字之前** |
| `MEMORY_S1.md` | **要碰 S1／分歧定位／`CGC_TENSOR_CAPTURE`／池查表之前** |
| `MEMORY_FACTS.md` | 動 build／載入／預算／`-ub`／mmap、要碰 `agent_harness/`、或**要提交**之前 |

- 舊報告寫的「`MEMORY.md` 的 S1 節」＝ `MEMORY_S1.md`（dated 產物，不回改）。
- **新增／刪除這底下任何 `.md` 都要重生索引**（`build_memory_index.py:66` 掃本目錄）。
- ⚠️ `index_assets.py` 範圍＝`scripts/check/*`＋本目錄＋`docs/*` 的**顯式註冊表** ⇒ **新增 `docs/`
  檔案不會自動進 MANIFEST，也不會報漂移**；本線 09-17／09-18 那批 docs 都不在管轄內。

## 現在的一句話狀態（09-18 快照；**主題檔為權威**）

- **decode 可引用＝ 全程 NOMINAL 那兩臂 10.80（舊引擎）／7.74（新引擎）** ⇒ 範圍 **7.7–10.8**，不是一個點。
  ⚠️ 「9.12 vs 10.78」**不能說成退步**。HTTP 的 12.36/12.95 是另一台儀器，統一 llama-bench 後不再引用。

- **★ 2026-09-20 更正（原文未刪，就在上面那一條）**：上面那個 **7.7–10.8** 是 `profile_duo`／`prod_matrix` 的
  **`decode` cell** 的讀數，而**那個 cell 不是交付形狀**（沒有 `--spec-type` ⇒ MTP off、沒有 `--warm-skip`
  ⇒ 冷的時鐘、沒有 `--ctx-size` ⇒ llama-bench 自行推導 ~704）。**交付 decode＝ `12.57 t/s`**
  （2026-09-20、`NOMINAL` **全程**、±2.26；`Backup/prod_profile/prod_profile_20260920_1230.json`），
  落在 10.78–13.10 的既有帶裡。⇒ **要引用交付 decode，用 `scripts/check/prod_profile.py`，不要用 `profile_duo`。**
  同一命令的**單臂噪音底 ≈ ±27%**（9.90 vs 12.57，更低的那次 `worst=MODERATE`）⇒ 小於 ~27% 的效應單臂證明不出來。
  prefill 的 250 bar **在 09-20 未驗證**（三次 188.12／212.59／222.40，全部 `worst ≥ MODERATE`）。
  → `MEMORY_PERF.md` 的 profile 節；`docs/PRODUCTION_PROFILE_2026-09-20.md`。
- **「疑似 −15% decode 退步」未獲證實也未排除** ⇒ 見下方「量測衛生」。→ `MEMORY_PERF.md`
- **S1**＝第一個被 GPU table 服務的層的 MoE gather（層號由 `CGC_S1_MIN_IL` 定）→ `MEMORY_S1.md`
- **MTP／spec 的三份 dated 結論（m、RSL、residency thrash）已全部移到 `MEMORY_PERF.md` 的
  「## MTP／speculative」節** —— 每次要動 MTP 之前**先讀那一節**（含「別練 draft head」「別做
  dynamic-k」與一條被更正的舊結論）。本檔只留指標道德的結論：今天是虧的，但不虧在 accept。
- **里程碑**：M1 做一半卡住（數值閘門 117/117，但 decode 退步 0.72×）、M2 核心落地、M3/M4 未開始。
  ⚠️ 「M1/M2/M3」另有 **D5 判決指標**的意思，先確認問的是哪一個。

## 分工

**`agent_harness/` 歸另一條 session；引擎層（`src/`、`scripts/check/`、decode／prefill 量測）歸本線。**
動手前後各跑 `git status --porcelain -uall`；看到不是自己的 modified／staged 檔就停手、只 stage 自己的檔案。

- ⚠️ **09-20 起引擎層有兩條 session。命名已由 operator 裁定（11:5x）：本線 ＝ `線A (ace)`**
  （舊標籤「本線（line I）」與另一條的文件署名撞了）。**另一條請沿用 `線 I`** ——
  它的擁有物是 `cb`／快取命中儀器（`cb_miss_regression.py`、`docs/F1_CB_MISS_REGRESSION_RESULT_*`），
  正是「`線 I` ＝ 儀器／快取幾何線」的字面意思。
  **`線A (ace)` 的定義按擁有物**：S1／段邊界（`wait`／`gap`／S2）＋ 逐層 KIND×OP 儀器
  （`4fdfaa8de` 在本線祖先鏈上）。命名表與五個碰撞面在
  **`docs/ENGINE_LINE_ASSIGNMENT_AND_G1_LADDERS_2026-09-20.md`**，那份同時把 **G1 的 `to` 拆成兩條
  階梯**（`cb`→0 owner `線 I`；`wait`/段邊界→0 owner `線A`）⇒ 兩份 ceiling 不再互相否證。
- ⚠️ **共用碰撞面（09-20 實測）**：① `llama-context.cpp`（F2 在 hook `:5381`；本線在 `graph_compute`
  `:3348`）；② `ggml-backend.cpp`（F5 的每層 barrier 與本線 S2 切點、G4 儀器同在 `hook_seg`／submit 迴圈）；
  ③ `libggml-base` 與 8080／GPU 窗口；④ **`cb` 的口徑**（他們 60.22／74.18 ms 高 swap ＋ `CGC_HOOK_SPLIT`；
  本線 21.23 ms E2b 暖；差 3 倍但**兩個都可能對**）；⑤ `.workbuddy/memory/*.md` 的併發寫入
  （09-20 11:45–11:46 實測本線 `§EN-308` 標題被拆兩行；`.workbuddy/` gitignored ⇒ 無版控安全網）。
- **意圖不衝突**：F2 打 `cb`、S2 打 `wait`，兩者可加，且雙方天花板都自寫「單獨不足 25」。
  ⚠️ 但 `gap ⊆ cb+submit` 的內建交叉檢查以 10.09 ms 失敗 ⇒ **分量相加在該 regime 不閉合，別直接加總。**
  且雙方的天花板都自己寫明「單獨不足 25」。

## ★★ 並行 session 安全（2026-09-18 血的教訓，動手前必讀）

同一台 Mac 上有多條線同時量測。**任何用「binary 名字」當範圍的清理，都是在殺別人的受試對象。**

- **`run_server.sh` 的 preflight 曾是全機 cross-kill**（`'pattern + pgrep -f'` 不看 port、不看 session，
  且**排在 memory guard 之前** ⇒ 自己被 guard 擋下、別人卻已經死了）。09-18 已改：
  **`CGC_PREFLIGHT_KILL=1`（預設）只列出 pid+etime、不送任何訊號**；要清場得明確說
  `CGC_PREFLIGHT_KILL=all`。第二欄 etime 是用來分辨「自己的殘留」vs「別人剛發的量測」。
- **`stop()` 不得用 `pkill -9 -f llama-server`**（`http_duo.py` 就是這麼踩的）：清理要按**自己的 pid／port**。
- **「查 env」不能有副作用**：`CGC_DUMP_ENV=1` 會跳過 preflight 與所有閘門（STALE 硬檢查、memory guard）
  ⇒ 別人在跑時你仍然取得到解析結果。`llama_bench_matrix.resolve()` / `prod_matrix` / `m123_oracle_gate` /
  `phase_split_ab` 都靠它，**這是唯一的真相來源，不要在工具裡重打一份 env**。
- **工具自選 port**：`http_duo.py --port auto`（8080 起第一個空的）＋ `start_new_session=True`
  ⇒ 兩條線可以各跑一台 server 而不互害。
- **server 死於 SIGTERM 要當無效樣本**：日誌 `[CGC] Received SIGTERM` ＝ 外部獵殺（watchdog 走 `GGML_ABORT`、
  OOM 是另一回事）。`http_duo.py` 每 rep 重查存活＋掃該標記，中了就印診斷、**exit 1**。

## 入口（照抄）

```sh
cmake --build src/llama.cpp/build --target llama-server -j 8   # 產物 src/llama.cpp/build/bin
RUN_REPLAY_BENCH=0 python3 scripts/check/decode_sweep.py --profile prod25 \
  --arms p25-gputime,<臂> --rounds 3 --warmup 0 --n-predict 24 \
  --json Backup/phase_decomp/<名>.json --force
python3 scripts/check/m123_oracle_gate.py --tag <標籤>          # D5，自己起 server（先確認 8080 空）
python3 scripts/check/http_duo.py --profile prod25              # 服務路徑兩軸；prefill 依 ctx 自動縮
python3 scripts/check/profile_duo.py --profile prefill250       # llama-bench 兩軸（**交付口徑**）

# 索引重生（順序固定；有動 .workbuddy/memory/*.md、scripts/check/*、docs/*、agent_harness/engine_loop/* 就跑）
python3 agent_harness/engine_loop/memory/build_memory_index.py     # 先：寫 INDEX.jsonl
cd agent_harness/engine_loop && python3 index_assets.py && cd -    # 後：MANIFEST 記 INDEX 的 bytes/mtime
```

- **`run_server.sh` 有 env allowlist**：未列出的 `CGC_*` **靜默丟棄** ⇒「沒效果」與「沒設到」同形。
  現成的坑：`CGC_PREFETCH_SRC=hist` 在 C++ 有（`llama-context.cpp:2002`）但已被 allowlist 刪除（2026-09-13）⇒ 今天從任一條路設都不生效；可用的替代是 `CGC_SERVER_NO_PREFETCH=0`（`run_server.sh:2029`，bench 側用 `--extra CGC_SERVER_NO_PREFETCH=0`）。
- 索引重生**順序固定**：先 `build_memory_index.py` 再 `index_assets.py`（顛倒或只重生 manifest 都不會修）；
  **memory 寫完要在索引重生之前**；`index_assets.py` 不要加 `--out`；範圍不含 `agent_harness/memory/`
  與 `skills/`（另一條線手動 `import_harness_snapshot.py`，要跑就**先快照、後索引**）。

## 量測衛生

- **【使用者約定，2026-09-18 起】速度數字一律用 llama-bench，不用 HTTP**，且 **prefill ＋ decode 兩軸並列**、
  禁止單軸引用。工具 `profile_duo.py`；`http_duo.py` 只用來研究「儀器間差異」。
- **★ 單臂噪音 ≈ ±1.9 t/s（6 臂 7.03–10.80），比 15% 的效應還大。**「發射時 NOMINAL」「臂內 worst」
  「引擎版本」「記憶體水位」**四個已記錄變數都排不出順序**：
  decode 單 cell n=8 的 `|r|` 全在 0.18–0.32 ≪ 臨界 0.705（**「噪音源＝記憶體」已收回**；
  不限 cell 的 r=+0.63/−0.51 是把 prefill 行與 decode 行混在同一條軸上假造的 ⇒ **後設相關一定要限定同一
  cell**：`caliber_env.py --memory --cell-filter`）。
  ⇒ **走配對設計抵銷未知慢漂，而不是繼續找解釋變數。**
- **配對設計（`scripts/check/paired_ab.py`）**：逐 rep 證明 **arm 內** rep2/rep3 差 0.6–2%、**arm 間**散 1.54×
  ⇒ 噪音在臂開始前就定了。用 **AB／BA 交替 ＋ `median(A/B)`**；熱閘門**預設關**（`--thermal-gate` 開：
  等 NOMINAL 要 35 s–10 min，會把一對的兩半隔開幾分鐘，正好放大要消除的漂移）。
  **必跑第一步是 `--null`（兩槽同 binary）＝ 儀器噪音底**；若自身就散 ±10%，正解是「不可測」不是「沒測到」。
  附帶 `corr(|Δswapin|, |ratio-1|)`：近 0＝配對成功，近 +1＝配對失效。
- **記憶體壓力別想著「釋放」**（`purge` 要 root、沒 swapoff）。替代＝把「進入每一臂的記憶體狀態」設成
  enforced 條件（`--min-headroom-mb`），等不到就 abort。**「確認沒別的 session」不能用 `pgrep llama`**
  （09-18 的競爭者 argv 裡沒有 llama 字串）⇒ 用 `paired_ab.py preflight()`，每一對之前都重查。
- **★★ cell 型工具 ≠ run_server.sh：batch 差 11 倍**（`caliber_env.py --equiv`）—— ⚠ **只對 `prod_matrix`／`profile_duo` 的 cell 成立**：`prod_matrix.cell_command()` ~line 319 `b = ub = spec["batch"]` 蓋掉 profile（prefill250 = 5632）⇒ `-b 512`；**`spec_cost_curve.py` 走 `lbm.default_batch()`（優先用 profile）⇒ 它是 `-b/-ub 5632`，與 server 相同**（09-18 那 18 支 run 的 JSON 有記）。其餘旋鈕同源轉發。
  ⇒ **「服務路徑 vs bench 的差距是環境造成的」尚未被證明**（實測差 +13~30%，**且非常數**）。
- **`CGC_SERVER_MTP=0` 是一整組旋鈕，不是一個旗標**：它讓 **8 個 engine env 整塊消失**
  （`CGC_DRAFT_DECODE`、`CGC_MM_BITIDENT`、`CGC_MTP_NO_WARMUP`、`CGC_NO_PREFETCH`、`CGC_NO_SEQ_RM_PROBE`、
  `CGC_VERIFY_DECODE`、`CGC_WARM_NPAST`、`LLAMA_EXPERT_CACHE_LAYER_CAPS`）。⚠️ 但它**換的那份模型與預設
  逐視窗全同 bytes**（不同 inode）⇒ **模型檔不是混淆變數**（此句取代「MTP=0 ⇒ 不是同一受試對象」的舊述句）。
- **prefill 只比較同 prompt 長度**（bench 2048、舊 HTTP 臂 2873/4500 是三個不同的量，不是退步）。
  現在 `http_duo` 依 profile ctx 自動縮到 ~2048（prod25 ctx=4096 ⇒ 2025；4500 會 400）。
- **`prod25` 在 llama-bench 兩軸上都量不到**（零 GPU 可見）：`n_batch` 被池路徑 `cgc_pool_max_tokens()`
  夾在 8、`compat()` 拒 `-p 2048`，連 `-b 512` 的 decode 都不合法 ⇒ 高速候選只剩 prefill250 血統。
- **★ bisect 不必重建**：產物在版控裡 ⇒ `git archive <commit> src/llama.cpp/build/bin` 取舊 engine，再用
  `install_name_tool -rpath <真build/bin> <tmp/bin> <f>` ＋ `codesign -f -s -`（**`LC_RPATH` 是絕對路徑，
  不改就載到當前 dylib，bisect 等於沒做**）。`profile_duo.py --bin-dir`。**不要為 bisect 重建。**
- **md5 只在同 build 指紋 ＋ 同 `predicted_n` 下可比**；交錯 A/B ×3 ＋配對中位。
- **`CGC-MMID-ASSERT` 的 `id_oob` 不是「消費者讀到什麼」的證據**（encode 期由主機讀 `op->src[2]->data`）
  ⇒ 只能用內核側讀數。
- **建置新鮮度的判準是「輸出有沒有編譯行」**，不是 exit code／產物存在／mtime（`.metal` 另驗
  `autogenerated/ggml-metal-embed.s.o` 的 mtime）。
- **D5**：`ORACLE_PINNED_ENV` 釘在 `BATCH/UBATCH=6144`；重新基線＝`--write-ref` ＋ `--no-pin-oracle-env`；
  **先看 `comparable` 再讀 M1/M2/M3**。現行參考檔＝ `ref_iq3_pool8gb_M2_6144_bitident_v6_nbaware.jsonl`
  （md5 `72d82a33ad79e0e69bc935acd24228f2`；09-17 13:45 起取代 `v5_spac`）。
  ⚠ 該檔在 `Backup/`（**未受版控**），而每個判決都量它 ⇒ 09-19 起 **md5 釘在程式的 `REF_PINS`**
  （不符就 exit 4，要覆寫需 `--allow-ref-drift`），判定档同時記 `ref_md5` 與 `ref_pinned`。
  **重新基線後要在同一顆 commit 更新 `REF_PINS`**（`--write-ref` 不會自動改它）。
- lesson schema：`superseded_by` 必填（可 null）、`applies_to` 每項是**檔案路徑**。
- **環境坑**：BSD `grep` 不支援 `\|`（一律 `-E`）；**本 sandbox 的 bash `grep` 會靜默失效 ⇒ 用內建 Grep 工具**；
  zsh `*.log` 無匹配會不執行整條指令；**沒有 `timeout`**；`ps` 被擋、`pgrep` 可用；
  **`notifyutil -g com.apple.system.thermalpressurelevel` 連 key 一起印，要 `awk '{print $NF}'`**。
- **heredoc 陷阱**：工具呼叫裡的 shell heredoc 會把 `$VAR` 吃掉（寫 patch 腳本要先用 Write 落檔再執行）。

## Skill（動手前先讀）

`~/.workbuddy/skills/`：`cgc-commit-gate/`（閘門鏈、`BIN_DIR`、`RUN_REPLAY_BENCH=0`、索引順序、多段式收尾、
lesson 欄位陷阱）、`cgc-decode-attribution/`（decode 歸因與輸出擷取規則）、
`cgc-prefill-thermal-delivery/`（prefill t/s 的條件式交付）、`cgc-whitepaper-delivery/`（`docs/*.html` 版式）。
