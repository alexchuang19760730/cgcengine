# flashkv-devserver — 專案長期筆記

> **這是快照，不是權威副本。**
> 權威位置：`.workbuddy/memory/MEMORY.md`（由 host 持續寫入）。
> 本檔於 2026-09-16 手動複製進 repo，唯一目的是讓 `agent_harness/` 底下的內容
> 能被 `agent_harness/scripts/auto_git_push.ps1` 定時推送；原檔改了這裡**不會**自動跟上。
> 索引與漂移檢查見 `agent_harness/engine_loop/memory/INDEX.jsonl`。

## 這是什麼

llama.cpp 的 CGC fork：Metal 後端 + **expert cache pool**（專家權重常駐 SSD→池，decode 為 IO-bound）。
專案根：`/Users/alexchuang/Documents/flashkv-devserver`。

## 目前跑什麼模型（2026-09-15 更正）

**prod25 profile 目前用 `models/gguf/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`**
（符號連結 → `Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf`，13.6 GB，含 MTP）。
不是 Gemma 4 26B-A4B —— 那是更早的設定，舊筆記／舊記憶裡若還寫 Gemma 要忽略。
`ls models/gguf/` 看實際有哪些；另一個大檔是 `Ornith-1.5-35B-A3B-…-Compact-v2D-lite.gguf`。

## prod25 profile 關鍵幾何

- pool **8 GiB** → **143 slots/layer**；`LAYER_CAPS 40-40:256`（只把 nextn 層 cap 到 256）。
- CTX 4096、`SPEC_DRAFT_N_MAX=3`、MTP 在 sweep 裡一律 `CGC_SERVER_MTP=0`。
- 三支柱 bit-identical：`CGC_MM_BITIDENT=1` / `SERVER_MTP_NO_WARMUP=1` / `SERVER_NO_SEQ_RM_PROBE=1`。
- `worst_layer=2`（248 distinct 專家 vs 143 slots）——層 2 是池壓力熱點，但**已被排除**為
  S1 分歧的原因（見 `agent_harness/engine_loop/traces/decisions.jsonl`）。

## 建置與測試入口（照抄）

```sh
# 重建（增量通常 ~5 s；binary 是 src/llama.cpp/build/bin/llama-server）
cmake --build src/llama.cpp/build --target llama-server -j 8

# 取樣掃描（一律 RUN_REPLAY_BENCH=0；.replay_bench_baseline.json 是 stale）
RUN_REPLAY_BENCH=0 python3 scripts/check/decode_sweep.py --profile prod25 \
  --arms p25-gputime,<臂> --rounds 3 --warmup 0 --n-predict 24 \
  --json Backup/phase_decomp/<名>.json --force
```

- 閘門：`scripts/check/m123_oracle_gate.py`（M1/M2/M3 bit-identical + 最新 M2 oracle）。
- **`run_server.sh` 有 env allowlist**：未列出的 `CGC_*` **靜默丟棄** → 「沒效果」與「沒設到」同形。
  加新開關一定要同步加 `if [ -n "${VAR:-}" ]; then SERVER_ENV+=(VAR="$VAR"); fi`。

## agent_harness（雙 loop 訓練資料 + 痕跡）

- `agent_harness/CONVENTIONS.md` = **判準憲章**（A 數字可引用／B 診斷／C 讀原始碼／D 流程／E 自我約束）。
  每條都要指到具體證據。`tb_loop/` 三個入口腳本仍跑不起來（E1 待修）。
- `agent_harness/engine_loop/traces/`：`episodes.jsonl`（原子觀測）、`decisions.jsonl`（判斷點，
  含 `ruled_out`）、`lessons.jsonl`（可重用規訓）。`validate.py` 驗證器，`selftest.py` 證明它會擋錯。
  `emit_episodes.py` 是**純讀**的 T0 emitter（從不啟動引擎）。`index_assets.py` 產生 `MANIFEST.jsonl`。
- 硬規則：`build == null ⇒ usable_as_evidence == false`；有生成 ⇒ `answer_md5` 不得為空；
  `judgement == refuted ⇒ superseded_by` 非空。

## 量測衛生（踩過的）

- md5 **只在同 build 指紋 + 同實際生成長度（`predicted_n`）下可比**；長度不同時 digest 差異有混淆，
  與長度無關的證據是 `sample`（`texts[0][:160]`）前綴。
- `pkill -9 -f llama-server` 每次跑前；交錯 A/B ×3 + 配對中位。
- macOS BSD `grep` 不支援 `\|`（一律 `-E`）；zsh `*.log` 無匹配會不執行整條指令。
- **`CGC-MMID-ASSERT` 的 `id_oob` 不能當「消費者讀到什麼」的證據**：它在 `ggml_metal_mul_mat_id`
  的 **encode 期由主機**去讀 `op->src[2]->data`。ids 是主機寫的 leaf 時它有效（永遠 0）；
  ids 是裝置算出來的 gather 輸出時，它量到的是**寫入前的殘留**（實測同一臂在分段/非分段下分別讀到
  0 與 202000，而兩者輸出都是可讀法語）。同理：`POST` 是 **sync 之後**的讀回。
  → 任何「消費那一刻」的問題都必須用**內核側**讀數回答。`eng-mh-0008`。

## 2026-09-15 起生效的三條規則（都是閘門缺陷的產物）

1. **建置指紋改成 glob**（`decode_sweep.py:33`，`ab_interleave.py` 直接委派同一份實作）：
   原本只手動雜湊三檔（`llama-server` + `libggml-metal` + `libllama`），**漏了 `libggml-base`**
   —— 而 `ggml-backend.cpp`（排程器 / `CGC_OA_ASYNC` 閘門）就編在裡面。現在是 `libggml*.dylib` /
   `libllama*.dylib` 全掃，**8 鍵、鍵名不含版本號**。**v1（3 鍵）的歷史列與 v2 不可比**。
   `eng-mh-0007`。
2. **`run_server.sh` 的 `CGC_SERVER_OA_ASYNC` 預設是 `1`**（2026-09-15 從 0 改回來）。
   閘門修成 value-aware 之後，`0` 第一次真的生效 = **非分段 = decode 慢 ~10×**（同 build 實測
   6.95 → 0.72 t/s）。只有 `prod25` 顯式設 1，其餘 profile（含預設 `off`、`prefill250`、`qa-zh`、
   `longform-zh`）原本靠 bug 「被分段」。**`prefill250` 是 oracle 閘門與 bench matrix 的口徑。**
   顯式 `CGC_SERVER_OA_ASYNC=0` 仍可用（S1 診斷那一格）。`eng-gate-0006`。
   鐵律：**錨 digest 不變 ≠ 行為不變**——顯式設了那個現在才有意義的值的那個 profile，
   正是修正不可能移動的那一個，所以它是最差的證人。
3. **`index_assets.py` 不要用相對 `--out`**：`--out` 是 cwd 相對，`--check` 讀的是腳本自身目錄，
   所以從 repo 根跑 `--out MANIFEST.jsonl` 會在根目錄寫出第二份 manifest、原檔留成 stale
   （= 又一份安靜失效的第二真相）。**重建就是不加 `--out`**（工具的錯誤提示已修）。
4. **索引重生順序固定：先 `memory/build_memory_index.py`（寫 `memory/INDEX.jsonl`），
   再 `index_assets.py`（`MANIFEST.jsonl` 會記錄 `INDEX.jsonl` 的 bytes/mtime）。**
   顛倒順序 → `index_assets --check` 報漂移，而且只重生 manifest 不會修，必須兩者都重跑。
   同理，**任何改動 `traces/lessons.jsonl`、`traces/decisions.jsonl`、`CONVENTIONS.md`、
   或 `.workbuddy/memory/*.md`（含 `MEMORY.md` 本身）之後，都要重跑 `index_assets.py`**
   ——`MANIFEST.jsonl` 是 byte 級快照，不是內容摘要，改一個字就漂移。
   實務上這代表：**memory 寫完要放在索引重生之前**，否則會在最後一刻把已提交的索引弄成 stale。

## Skill

`~/.workbuddy/skills/cgc-decode-attribution/SKILL.md` —— decode 相位歸因的儀器清單、判準與陷阱。
動任何 decode 歸因工作前先讀它。

`~/.workbuddy/skills/cgc-prefill-thermal-delivery/SKILL.md` —— **prefill t/s 的條件式交付**：
什麼時候那個數字可以被引用、條件怎麼寫、三態標籤（COLD／HOT／UNKNOWN）怎麼讀、
以及為什麼「靜置 N 秒」與「swap 高低」都不是條件。要引用任何 prefill 數字前先讀它。

`~/.workbuddy/skills/cgc-commit-gate/SKILL.md` —— commit 前的閘門鏈與索引重生順序。

## prefill 250 的「設定」= 三層，只有一層在設定檔裡（2026-09-16 量到；同日 14:4x 更正第二層）

1. **引擎設定**：`CGC_SERVER_PROFILE=prefill250`（`-b/-ub 5632`（2026-09-16 由 6144 改過來；
   6144 在 req2 上 0/5 死）、`CGC_PREFILL_STREAM=1`、`CGC_GATHER_SLAB_CAP=256`、
   pool 8 GiB；ctx 8192）。**必要，但非充分。**
2. **散熱前提——原句「距上次持續 prefill ≥150 s」作為<u>充分</u>條件已被推翻。**
   2026-09-16 14:35:50 的臂安靜只有 **18 s**，req1 卻是 **289.86 t/s**（比安靜 32 分鐘那臂的
   254.29 更快）；安靜 **34 s** 的那臂只有 **166.95**。**安靜最長的最慢、最短的最快**
   ⇒ 自變數不是間隔長度，而是**同一序列裡這是第幾次啟動（累積負載）**：
   一次啟動（約 40 s prefill）不足以耗盡熱預算，三次以上會，之後秒級安靜買不回多少。
   機制本身沒變（`powermetrics` 逐樣本）：DVFS 階由 thermal governor 選定
   （1470→928→618 MHz），熱壓 `Nominal→Moderate→Heavy` 與時脈下降同秒、活躍駐留全程 ~98%。
   **[2026-09-16 15:2x 更正 —— 下面這句已被推翻，原文保留]** 原句是「此條件不可由非 root
   儀器觀測（`pmset -g therm` 只回 "No thermal warning level has been recorded"）⇒ 閘門只能
   **分類**（COLD／HOT／UNKNOWN-STATE），不能宣稱『已滿足前提』」。那是**關於一個工具的推論，
   被寫成了關於系統的事實**。真正的儀器是
   `notifyutil -g com.apple.system.thermalpressurelevel` ——**正是 `powermetrics` 的 thermal
   sampler 讀的那條 notify(3) key**，`0=Nominal 1=Moderate 2=Heavy 3=Trapping 4=Sleeping`，
   **不需要 root**、約 **11 ms/次**（20 次 0.225 s）、可 2 Hz 取樣；滿載讀 2、停止後約 47 秒回 0。
   **判準：發射驗收臂之前讀到 `0`（Nominal）。** 實測分離度（request 級、零重疊）：
   發射 0 → **6/6 ≥250**（253.42–271.64）；發射 1 或 2 → **0/21**（104.88–211.65）。
   讀的是**發射前那一刻**，不是全程：15:18:46 那臂發射 0、中途升 1→2，三請求仍全 ≥250
   （機制「governor 反應落後」是**假設**）。反面教材：`NSProcessInfo.thermalState`
   （JXA、非 root、四級刻度）**367/367 讀到 `fair`**，零區辨力 —— 找到介面 ≠ 找到儀器。
   閘門 `ARMS=2 bash Backup/run_thermal_gate.sh`（不成立 exit 3，fail closed）；
   2 Hz 序列 `Backup/thermal_pressure_probe.sh`；in-band 讀數已做進 `Backup/run_req2_retest.sh`。
   見 §11.15 與 `docs/PREFILL250_CONDITIONAL_DELIVERY_20260916.html`（＋ `.md`）。
   **§11.13 那 2.25× 已歸因（否證法）**：ABBA 檔案互換（`Backup/run_lib_ab.sh`，一臂暖機丟棄）
   ⇒ `libggml-metal` **不是**原因（req1 +0.8% / req2 −14.4% / req3 −0.6%，方向不一致；
   A 組內 span 51.56 = 組間均差 7.66 的 **6.7 倍**）。替代解釋由熱帶的一致性支持
   （268/274/257 落 Nominal 帶、118–155 落 Heavy 帶），但**那是一致，不是直接觀測**。
   **閘門只能「分類」而不能「強制」這句話本身已失效。**
3. **量測紀律**：**只有「安靜 ≥30 分鐘」那個序列的第一次啟動是冷態樣本**；
   接下來幾臂是熱態（run 2 繼承 run 1 的餘熱）。三次同一道命令可給 276.25 / 227.27 / 151.28 t/s。
4. **條件式交付（2026-09-16 14:31:36 在<u>最終產物</u>上量到，出廠預設、無 override）**：
   冷態臂 = **254.29 / 282.38 / 265.27 t/s**（req1–req3 全部 ≥250、`survived=yes`、0 crash report），
   同 A11 指紋 `a9c9dc104f057bf640e0132a83e01c12`。熱態平台 = **167.41–200.58**。
   13:45–13:57 那個 soak 序列的平台是 118.30–154.77（≈130）——**兩個平台差 55 t/s，未歸因**
   （中間夾著 13:49:11 / 13:50:16 兩次 D5 閘門與 12:44:53 的 8 核 CPU 重建，負載未量化）。
   **可交付的述句是「req1 ≥250 且該臂全程存活」；不能說「250 隨時可重現」，也不能說 req2/req3
   一定 ≥250（第 2 臂的 req3 是 233.51）。**
5. **`swap` 不是因，是果**：那 32 分 13 秒安靜期間它自己從 `used 5066.06M` 回到 `2743.00M`；
   反例的順序是**相反**的（289.86 t/s @ `4975.06M`、188.35 t/s @ `4549.69M`、13:51 慢臂 @ `5066.06M`）。

`t(token) = 0.5575 ms + 4226/f_eff(MHz)`（3 點擬合）：地板項不隨時脈縮放、冷態佔 15.4%，
是唯一不被散熱改善的部分。**250 不是上限**（1470 階純階預測 291.3；實測 294–308）。
純階反讀：1470 → 291 t/s、928 → 227、618 → 135。

## M1/W3 剩下的唯一 blocker 是 wide 路徑（設計不變量，不是難度）

`CGC_POOL_SPLIT` 要 wide 張量（`ne[2]=256`）⇒ Metal 得隔著**模型的權重 buffer** 讀 256-expert 張量
⇒ gather 路徑把 Metal buffer 裡的張量重新指向**未知 host 指標** ⇒
`ggml-metal-device.m:2026 buffer is nil`。實測：**abort 0 次、`buffer is nil` 234 次、M1 2/42**。
**0 次 abort 才是分界線**：不是報錯，是靜默算錯。出口恰好兩條，且 **(ii) 已經在跑**：
`CGC_PREFILL_STREAM=1` 用**暫時的 Metal slab** 讀同一個 wide 幾何而正確。
⇒ 下一步 = 把 slab 從「prefill chunk」一般化到「常駐 wide 區域」。
任何 `src/` 改動都要重建並同 commit stage `libllama.0.0.239.dylib`（檢查 8）。

## prefill250 的第二個請求會 SIGSEGV —— blast radius 已量定（2026-09-16 中午）

**MTP 在出貨配置是開的**（`run_server.sh:88` `SERVER_MTP:-1` 預設 1；`prefill250)` case 再確認一次；
`CGC_SERVER_DENSE_IQ4X=1` ⇒ 模型走 `-MTP-...-denseIQ4X.gguf`；argv 帶 `--spec-type draft-mtp
--spec-draft-n-max 3`）。所以這個崩潰不是「某個沒開的功能」。

**觸發條件 = `batch = ubatch = 6144` 那一格**，不是 MTP、也不是 slab/stream 路徑。
三臂單變數階梯（`Backup/run_mtp_crash_triage.sh`，同一 prompt 各連發 3 次、閘門全關、
每臂先 `CGC_DUMP_ENV=1` dump 解析後的環境）：`off` 全過 / `prefill250`+`2048` 全過（160/149/155 t/s）
/ `prefill250`+`6144` 在 req2 死。崩潰簽名三次逐格相同：
`objc_msgSend ← ggml_metal_synchronize ← ggml_backend_sched_synchronize ← llama_context::synchronize()
← llama_get_embeddings_nextn(ctx_tgt) ← common_speculative_impl_draft_mtp::process`
（`EXC_BAD_ACCESS`、`KERN_INVALID_ADDRESS at 0x10`）。**注意崩的是 `ctx_tgt`，不是 `ctx_dft`。**
**死點已在 2026-09-16 中午定位且已修（更正先前記的 `:727`）**：真正位置是 `cmd_bufs_ext` 的
**錯誤路徑**——舊碼「先釋放（釋放迴圈 + `removeAllObjects`）、後呼叫 `cgc_metal_record_error()`」，
而該函式在 `status == 5` 時讀 `[cmd_buf error]` ⇒ 對已釋放兩次的 ObjC 物件送訊息。
機械證據：崩潰位址 `ggml_metal_synchronize + 612` 的 `imageOffset = 0x13E04`，正好是 `0x13e00`
那條 `bl _objc_msgSend$error` 的返回位址（舊二進位反組譯，備份在
`Backup/pre_mtp_rebuild_20260916/`）。修正把記錄前移到任何 release 之前。

**可用退路**：`CGC_SERVER_PROFILE=prefill250 CGC_SERVER_BATCH=2048 CGC_SERVER_UBATCH=2048`
連三次不崩（代價是 prefill 從 250 級掉到 160 級）。**「單請求 prefill250」也成立**（292.07 t/s 冷態）。

**另一個獨立發現（與崩潰無已建立的因果）**：出貨的 `libllama-common.0.0.239.dylib`
**沒有編 `-DMTP_SUPPORT`**（`build/CMakeCache.txt` 的 `CMAKE_CXX_FLAGS` 空、`flags.make` 無此 define、
`strings -a` 正對照三缺一）。⇒ `common/speculative.cpp:1544-1554` 那段
`[CGC MTP fix] ... safely access each row` 不在當時的二進位裡。
**2026-09-16 中午已重建為 MTP-on**（`cmake -B build -DCMAKE_CXX_FLAGS="-DMTP_SUPPORT"` 全量重建，
411 TU／0 error；受 guard 字串 0/3 → 3/3），載入的是 `libllama-common.0.0.275.dylib`
（UUID `4ddc4e5d…`）。**12:12 再對齊四個 `GGML_*` 旗標**（`BLAS`/`ACCELERATE`/`CPU_REPACK`/`OPENMP`
= OFF，依 `build_fork_llama.sh:18-21`）並全量重建 ⇒ SONAME **`0.0.277`**，
D5 `--tag flagalign2` 四軸 **9/9 PASS**（對這個負載是數值 no-op）。
**仍未對齊的是刻意的部分**：腳本 `LLAMA_BUILD_SERVER/TESTS=OFF` 而 devserver 需要 `ON`
（因此**不跑** `REBUILD=1`，它在 `:53` 會 `rm -rf build`）。

## 長期事實（踩過就不該再踩）

- **`MTP_SUPPORT` 是編譯期開關，且必須定義。** macOS 沒有預設
  （`src/llama.cpp/CMakeLists.txt:63-72` 只給 `WIN32`），靠 `scripts/build_fork_llama.sh`
  `MTP_SUPPORT:-ON` 傳入。少了它被編掉的是**修正**（`qwen35moe.cpp:737` 的 `res->t_embd = cur;`
  與四處 `[CGC MTP fix]`），不只是除錯。`run_server.sh:325-341` 有可見性守門（量
  `MTPDBG mtp_ctor` 標記；好 build 0 warning）。
- **SONAME 內嵌 commit 數**（`libllama-common.0.0.<N>.dylib`）⇒ 每次全量重建都換檔名，舊檔變孤兒。
  載入器走 symlink 鏈：`libllama-common.dylib → .0.dylib → .0.0.<N>.dylib`（`otool -L` 可查）。
  **任何稽核都要從 `otool -L` 反推真實載入集再 realpath，不要寫死檔名**；`strings -a` 要一起印
  control（不受 guard 保護的）字串，否則分不出「開關沒開」與「artifact 壞了」。
- **`mtime` 不是 build 一致性證據。** 增量 `cmake --build` 只重編 CMake 判定 stale 的 TU，
  所以要問 CMake 自己、或用 `flags.make` 的內容比對。
- **錯誤路徑不得「先釋放、後記錄」。** 診斷用的物件必須在任何 release 之前讀完；錯誤路徑裡出現
  `EXC_BAD_ACCESS` 時先問「它遮蔽了什麼」。判準：指令序本身就是證據，所以**要保留當時的二進位**。
- **崩潰報告的 `imageOffset` 是位址，可以直接對二進位反組譯。** 另外 `usedImages` 的 image UUID
  可以證明「兩次 run 之間到底換了哪幾個 image」——這是驗證單變數對照最便宜的方法。
- **bash `grep` 在此 sandbox 對 `strings | grep -q` 這種用法是可用的**（已雙向量過：MTP-on rc=0、
  MTP-off rc=1）；先前的「grep 靜默失效」不是這個用法。
- **`-ub` 是 req2 OOM 的那根旋鈕，而且是單變數**（2026-09-16 12:00 量定）：同 profile、同
  `load_mode=none`，`-b/-ub 4096` 兩次獨立啟動共 6 次請求全過（0 錯誤行、無 crash report），
  `6144` 的 req1 過但 **req2 以 status 5 OOM 死**（`kIOGPUCommandBufferCallbackErrorOutOfMemory`）。
  **代價 −22%**（130-152 vs 166.8 t/s）。OOM 的**機制沒修**，只有繞道。
- **2026-09-16 13:4x 起，`prefill250` 的出廠預設 `BATCH/UBATCH` 是 `5632`（不再是 6144）。**
  存活表（同 profile、同 `load_mode=none`、同 `common_md5 7bceb3bd5320`）：
  `4096 5/5`、`5120 3/3`、`5632 4/4`、`6144 0/5`、`6144＋池 6 GiB 3/3`、`mmap` 全寬 `0/3`。
  兩個方向都指向同一句：**邊界是 (pool + compute) 的和，不是 `ub` 一個旋鈕**。
  回復舊寬度：`CGC_SERVER_UBATCH=6144 CGC_SERVER_BATCH=6144`。
  **若要引用 t/s，必須同時說出是哪一份產物與哪個機器狀態**——見下面「吞吐」那一條。
- **`--load-mode mmap` 與 expert cache 不相容，而且是載入階段的硬崩**：L4 pool 是
  「regions adopted from expert tensors」，直接寫進模型張量儲存；`mmap` 下那是唯讀 file-backed
  映射 ⇒ `fill_pool_direct` 的 zeroing 寫唯讀頁 ⇒ `SIGBUS`/`KERN_PROTECTION_FAILURE` 在 `__bzero`。
  `run_server.sh` 的 `[防護 2e]` 是硬閘門（單獨拒跑；`CGC_SERVER_EXPERT_CACHE_OFF=1` 或
  `CGC_SERVER_ALLOW_MMAP_EXPERT_CACHE=1` 才放行）⇒ **mmap 只在 cache-free 時安全**。
  **2026-09-16 更新（缺陷本體已修，但結論不變）**：修法**不動 expert cache 一行**，而是讓 L4 pool
  tensor 落到「不是 device default」的 `_Pool` buft（`ggml-metal.cpp` 新增、經
  `ggml_backend_dev_get_extra_bufts` 廣告；`supports_buft` 要白名單那些 `get_name`），
  這樣 `llama-model.cpp:1595` 的 mmap fast path 條件 `is_default_buft` 不成立 ⇒ 走可寫的 MTLBuffer。
  SIGBUS 消失（載完 40 層），**但 mmap 在任何量過的寬度仍是 0/5**，而且它會把 Metal 工作集撐大
  ⇒ **mmap 不是 OOM 的槓桿，不要再試**。`libllama-common.0.0.<N>` 的名字裡 `N` 是 commit 數。
- **`-ngl` 一律被顯式帶上 ⇒ `-fit`（`common_fit_params`）永遠是 no-op**：
  `common/common.h:477` 預設 `fit_params=true`，但 `common/fit.cpp:377-379` 在
  `n_gpu_layers != 預設(-1，llama-model.cpp:2479)` 時直接 throw "abort"。
  `CGC_SERVER_FIT=1` 會把 `-ngl` 拿掉讓 fit 真跑（預設關）。**argv 是位置比對 `ARG[i]`（D5 config
  stamp）⇒ 不可重排，只能用兩段式 append。**
- **KV 只有 10 層**（不是 40/41）：主 context 是 `llama_memory_hybrid`，attn filter =
  `il < n_layer() && !is_recr(il)`（`llama-model.cpp:2292-2295` → `llama-memory-hybrid.cpp:48-50`），
  `full_attention_interval=4` ⇒ 只有 `i=3,7,…,39` 帶 K/V ⇒ fp16 @8192 = **160 MiB**（q8_0 ≈ 85 MiB），
  **不是瓶頸**。MTP 那顆 context 走 plain `llama_kv_cache` 且 filter 是 `il >= n_layer()`（1 層）。
- **`-expert-cache $BUDGET` 的 8192 MiB 是上限，不是實配**：實配 ≈ `5976 槽`（`LAYER_CAPS`）÷ 40 層
  × `110 MiB/層@256 experts` ≈ **2.5 GiB**。啟動預算：model 13030 MiB（`load_mode=none` =
  匿名頁不可回收）+ 8192 ⇒ **21222 vs 16384 MiB = OVERSUBSCRIBED 4838 MiB**。
  `run_server.sh` 的 `[防護 2d]` 會把這個算式印出來（`CGC_SERVER_STRICT_BUDGET=1` 超額即 exit 1）。

## 建置新鮮度與交付驗證（2026-09-16 收尾量到，四條）

- **建置產物的新鮮度，判準是「建置輸出有沒有編譯行」，不是 exit code、也不是產物的存在或 mtime。**
  `cmake --build` 在「已經最新」與「剛剛編好」兩種情況都回 0。改了會被編進二元檔的檔案之後、
  跑任何測試之前，先對該 target 跑一次建置並**讀它的輸出**；一旦印出 `Building …`，
  先前在同一棵樹上取得的所有數字全部失效。
  實例：13:00:34 改了 `ggml-metal-context.m`（只改註解）沒重建 ⇒ 12:44–13:50 之間每一筆量測
  （含 13:45 的出廠驗收與 13:50 的 D5 PASS）都屬於舊產物；`libggml-metal` md5
  `968c36cf45742bb1667d5a02629c67be` → `f2d1c96193939bd15404ba713a6fa85d`。
  check 8 的 mtime 規則會擋這一格，但它在 commit 前才跑，而量測更早發生。
- **D5 閘門的「預設會重現 oracle 配置」不得寄生在生產預設值上。**
  `scripts/check/m123_oracle_gate.py` 現在有 `ORACLE_PINNED_ENV`
  （`CGC_SERVER_BATCH=6144`、`CGC_SERVER_UBATCH=6144`，放在 `DEFAULT_REF` 旁邊），resolve 前併入；
  `--env` 可覆蓋但有效集合會先印；重新基線＝`--write-ref` ＋ `--no-pin-oracle-env`。
  **殘留**：gate 認證的是 oracle 配置（6144）的數值；出廠預設 5632 沒有自己的參考檔，
  只能量到跨配置的 9/9，不被認證。
- **「兩次量測之間只有一個產物不同」≠ 單變數實驗。** 單變數的前提是**交錯**：
  機器狀態（頻率／swap／壓縮器）不在任何一方日誌裡。兩個候選都無法排除時，述句是「未歸因」。
- **吞吐不可重現的最強樣本（要引用就一起說出產物與狀態）**：同配置、同 A11 指紋
  `a9c9dc104f057bf640e0132a83e01c12`，12:44 產物 = `268.09/273.55/256.61` t/s，
  最終產物的熱態四臂 = `118.30–154.77` t/s。
  **「閒置 4 分鐘仍 126/124/126 ⇒ 不恢復」這個讀法是錯的**：4 分鐘不是復原時間尺度。
  2026-09-16 14:31:36，同一份最終產物、同指紋，在**安靜 32 分 13 秒**之後量到
  `254.29 / 282.38 / 265.27`（req1–req3 全部 ≥250）⇒ 復原存在，但尺度是**數十分鐘**，
  而且要連 CPU 密集建置一起算（12:44 那次是 8 核 rebuild）。
  兩份報告的 IO 統計逐位元相同（`jobs=47991 bytes=2784305152`）⇒ 慢的不是 streaming 路徑。
  已知同產物散佈可達 2.17×（`none:5120` = `284.58/157.95/130.88`）。
  狀態變數本身可觀測：`sysctl vm.swapusage`、`sysctl -n vm.loadavg`、`memory_pressure -Q`；
  **`ps` 在本 sandbox 被擋**、`pmset -g therm` 沒有紀錄（非 root 讀不到熱等級）。
  背景負載要點名：量測期間常駐兩個 `Xcasca`（Electron）renderer，idle 1 分鐘 load 在 1.71–6.47。
- **`CGC_METAL_LIB` 不是載入覆蓋**：`run_server.sh:809-812,943` 只拿它做 `_Pool` 的閘門偵測，
  不會改變載入器實際映射的庫。要做 metal 庫的 A/B 只能**互換檔案**。
- **判讀產物指紋的兩個陷阱**：(a) `llama-server`／`libllama-server-impl.dylib` 被重寫也可能
  與 HEAD **逐位元相同**（install name 沒動就不需 relink）；(b) `cmp -l` 的位元組差異數會被
  **內嵌建置戳記變長造成的整體位移**放大（`libggml-base` 顯示 47940 bytes 不同，實際只差
  `8af8c99bd` → `efeade7e2-dirty` 一行）⇒ 判準是 `strings` 差集，不是 `cmp` 的計數。
