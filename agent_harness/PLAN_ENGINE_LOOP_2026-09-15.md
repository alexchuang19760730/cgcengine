# 規劃：把「引擎調校 harness」收進 `agent_harness/`，做成 pi + prime agent 的訓練數據

日期：2026-09-15　狀態：**規劃，尚未動工**
前置讀物：`agent_harness/README.md`（既有 `tb_loop`）、`docs/REMAP_ROUNDTRIP_REMOVAL_PLAN_2026-09-15.md`、
`docs/PREFILL250_DECODE25_WHITEPAPER_20260915_1900.html`、`.workbuddy/memory/2026-09-15.md`

---

## 0. TL;DR

1. `agent_harness/` 目前只有**一個** loop：`tb_loop`（Terminal-Bench × gemma4 × prime-agent）。但今天真正
   在跑的是**第二個** loop：**引擎調校 loop**（`run_server.sh` + `decode_sweep.py` + `ab_interleave.py` +
   `m123_oracle_gate.py` + 白皮書）。兩者**同構**（evaluate → extract failures → refine → compare），
   差別只在「任務」與「獎勵」的定義：tb_loop 的獎勵是 tb 測試 pass/fail，engine loop 的獎勵是
   **bit-identical 閘門 + 吞吐 + 池計數不變量**。
2. 新增 `agent_harness/engine_loop/`。**不改動 `scripts/check/` 與 `scripts/run_server.sh`**——它們是生產
   腳本，pre-commit 閘門與白皮書都指著它們；engine_loop 只用 **wrapper + 索引**指向它們，避免兩份真相。
3. 訓練數據的**唯一出口**是 `agent_harness/traces/*.jsonl`，只有三種 record：`episode` / `decision` / `lesson`。
   今天**已經可以自動匯出**的只有 `episode`（純腳本、不依賴 LLM）；`decision` 半自動；`lesson` 人工定稿。
4. 同一批 traces 投影成兩份 SFT：`sft_pi/`（工具軌跡 run→observe→decide，messages + tool_calls）餵
   pi-coding-agent；`sft_prime/`（(證據 → 教訓) 對、以及 (狀態 → 下一個該跑哪個臂)）餵 prime-agent 的
   Continual Harness。
5. **最值錢的資產不是腳本，是今天判準數字的方法論**：交錯 A/B ×3 取 paired median、build fingerprint、
   md5 與數字並列、閘門可能自證、上界探針必須標註、rate-limited 斷言不是普查。這些收進
   `CONVENTIONS.md`，同時成為兩個 loop 的共同 system prompt。

---

## 1. 今天的 harness 盤點（現況，全部已存在）

| 層 | 資產 | 今天的作用 | 能否自動轉成 record |
|---|---|---|---|
| **啟動** | `scripts/run_server.sh`（84 KB） | profile `prod25` / `prefill250`；**env allowlist**（未列出的 `CGC_*` 被靜默丟棄，與「改了沒效果」不可區分）；preflight pkill；binary 選擇 | ✅ 讀 ctrl log |
| **建置** | `cmake --build src/llama.cpp/build -j8`；`scripts/build_turbofieldfare_server.sh` | 產生 build fingerprint（server / metal / llama 三個 md5 前 12 位） | ✅ |
| **臂定義** | `scripts/check/decode_sweep.py` 的 `ARMS` 表 | 每個臂 = env dict ＋ 一句「這臂在證明什麼」＋「不可用於吞吐」標註 | ✅ 直接讀 |
| **度量** | `decode_sweep.py`、`decode_bench.py`、`llama_bench_matrix.py` | decode/prefill t/s、池 hit%、capacity%、io MiB/s、per-step 相位 | ✅ |
| **A/B** | `scripts/check/ab_interleave.py` | 交錯 ×N、**paired per-rep ratio 的中位數**、build fingerprint、`--report-only` | ✅ |
| **閘門** | `scripts/check/m123_oracle_gate.py` ＋ `Backup/knifeedge_matrix/ref_iq3_pool8gb_M2_6144_bitident.jsonl(.cap)` | bit-identical；`DIAGNOSTIC_KEYS` 允許「被測命題」放行但記名 | ✅ |
| **閘門自測** | `oracle_truth_gate_selftest.py`、`feasibility_gate_selftest.py` | 證明閘門會擋錯（今天發現它其實**自證**） | ✅ |
| **探針** | `mmid_geometry_probe.sh`、`mmid_pool_vs_gguf.py`、`mmid_zero_row_triage.py`、`flip_rate.py`、`mtp_*` | 把 `CGC-MMID-ASSERT` 分類成 MODEL-ZERO vs ENGINE-ZERO | ⚠️ 半自動 |
| **證據** | `Backup/phase_decomp/*.json`（9 檔，52 KB）、`Backup/m123_oracle_gate/`（372 KB）、`Backup/knifeedge_matrix/`（6.3 MB）、`Backup/cgc_logs/*.log`（**463 MB / 1399 檔，今日 206 檔**） | 原始數字與 log | ✅（log 需抽樣） |
| **結論** | `docs/PREFILL250_DECODE25_WHITEPAPER_20260915_{1745,1810,1900}.html`、`docs/REMAP_ROUNDTRIP_REMOVAL_PLAN_2026-09-15.md`、`docs/MMID_GEOMETRY_PROBE_2026-09-15.md`、`docs/LATEST_COMMIT_GAP_ANALYSIS_2026-09-15.md` | 對外結論 | ⚠️ 半自動 |
| **決策日誌** | `.workbuddy/memory/2026-09-15.md`（56 KB） | 每一條「為什麼」——**資訊密度最高的一份** | ⚠️ 半自動 |

---

## 2. 為什麼這批東西值得當訓練數據

今天 90% 的工作不是打字，是**讀數字 → 排除假設 → 設計下一個探針**。這正是 agent 最缺的能力，而今天
恰好產生了成本明確的判斷點（每一條都能追到具體 log 行）：

| # | 判斷點 | 當天的代價 | 可訓練的形狀 |
|---|---|---|---|
| 1 | 單輪 `CGC_SUBMIT_AHEAD` ×1.78 不可對外引用，必須交錯 ×3 取 paired median | 一次錯的對外數字（實際 ×1.711，區間 1.327–1.924） | 「什麼時候一個數字可以引用」 |
| 2 | `host=0` 被讀成「表在 private buffer」——Metal 的 shared/private 都印 0 | 一次誤診 | 「結構性恆零的欄位不是證據」 |
| 3 | `build_moe_ffn` 的 `n_tokens ≠ ubatch.n_tokens`（hook 看到 2，ubatch 是 1） | 一輪白跑的 smoke | 「變數名稱不等於語意」 |
| 4 | `ggml_argsort_top_k` 輸出非連續 → `ggml_reshape_1d` 直接 abort | 一次編圖期崩潰 | 「reshape 前先確認連續性」 |
| 5 | `CGC-MMID-ASSERT` 是 **rate-limited**（前 8 筆 + 每 1000 筆 1 行），10 行輸出被當成「只有第 15 層」 | 一條錯的推理線索 | 「抽樣 ≠ 普查」 |
| 6 | 閘門是**自證的**（參考檔本身是 cache-ON 產物，只證 pool-size 不變性） | 差點誤信 19/19 | 「閘門要先證明自己會擋錯」 |
| 7 | S1 崩潰堆疊在 **`libggml-cpu` 的 `mul_mat_id`**——今天唯一一次，其餘都是 SIGBUS/SIGABRT。用 `GGML_SCHED_DEBUG=2` 的 per-node 傾印定位到**只有 layer 0**：它的 MoE FFN 在 CPU/BLAS（專家權重仍是全尺寸 82M vs 其餘 45M），ids 在 baseline 是 CPU backend 的 graph input，改 GPU 查表後變成 Metal 算出來 → 需要跨 backend 拷貝（dump 印成 `CPU#ffn_moe_slots-0# [NULL]`） | 1 小時定位、一次編圖期 SIGSEGV | 「同一個 graph 裡有兩個 backend 時，輸入的可見性就是契約」 |
| 8 | `scripts/check/mmid_zero_row_triage.py` 把斷言分成 MODEL-ZERO / ENGINE-ZERO（14/0） | 避免把模型性質當引擎缺陷 | 「先分類再修」 |
| 9 | **診斷本身要先被證明會印**：`GGML_SCHED_DEBUG=1` 因預設 verbosity（INFO）把 `GGML_LOG_DEBUG` 濾掉，log 裡 0 行 `## SPLIT`——與「只有 1 個 split」不可區分 | 一輪探針白跑 | 「沉默的儀器等於沒有儀器」 |
| 10 | **`tb_loop` 這個套件名今天根本不存在**：`run_round.sh` / `gen_sft.sh` / `finetune/eval_round.sh` 都寫 `--agent-import-path "tb_loop.agents.…"`，但 `PYTHONPATH=$TB_LOOP_DIR` 且 `TB_LOOP_DIR=agent_harness/` → `import tb_loop` 直接 ModuleNotFoundError（本輪已實測） | 三個入口腳本其實都跑不起來 | 「入口腳本也要有 smoke，不能只做 `bash -n`」 |
| 11 | **重寫一個張量時漏抄它的「不可被 ggml-alloc 疊用」保證**：leaf 靠 `ggml_set_output(remap)` 釘住每層的 buffer；S1 的 gather 輸出漏了同一行，於是 11 層裡 10 層的 `ids_data` 是同一個位址，每層讀到同一份 16 個 int | 兩天的猜測（table 內容？host 可見性？）全部無效 | 「重寫資料流時，逐項清點原實作依賴的**隱含契約**」——這次是一行 `ids_data=%p` 就結案 |

前 7 條已經寫在 `.workbuddy/memory/2026-09-15.md`，但格式是**散文**；要能訓練 agent，得轉成
**工具軌跡（messages）** 與 **(證據 → 結論) 對**。

---

## 3. 目標目錄結構

```
agent_harness/
├── README.md                      # 傘狀入口：2 個 loop、3 種 record、1 個匯出目錄
├── CONVENTIONS.md                 # ★ 判準憲章（兩 loop 共用；也是 SFT 的 system prompt 來源）
│
├── tb_loop/                       # ← 既有內容原樣 git mv 進來（見 E1；零內容改動）
│   ├── config.env  run_round.sh
│   ├── agents/ learning/ harness/ results/ sft_data/ sft_data_ft/ finetune/ datasets/ scripts/
│   └── docs/                      # 既有的 whittle-moe-whitepaper.md
│
├── engine_loop/                   # ★ 今天 harness 的新家
│   ├── README.md                  # 本 loop 的 evaluate / extract / refine / compare 定義
│   ├── MANIFEST.jsonl             # 資產清單：path, role, replayable, produces_record, notes
│   ├── config.env                 # ENGINE_BIN / ENGINE_MODEL / ENGINE_PROFILE / EVIDENCE_ROOT
│   ├── runners/
│   │   ├── rebuild.sh             # cmake --build + 三 md5 → build.json（fingerprint 單一來源）
│   │   ├── server.sh              # 薄包 scripts/run_server.sh（profile + allowlist 存在性檢查）
│   │   └── preflight.sh           # pkill / free mem / 殘留檢查
│   ├── wrappers/                  # 只放「指向 scripts/check/* 的呼叫 + 輸出正規化」，不放邏輯
│   │   ├── sweep.sh  ab.sh  bench.sh  gate.sh  triage.sh
│   ├── distill/
│   │   ├── refine_engine.sh       # prime-agent /refine --global（engine scope）
│   │   └── prompt/                # 餵 /refine 的模板：要求輸出 decision/lesson JSONL
│   ├── traces/                    # ★ 訓練數據唯一出口
│   │   ├── schema/{episode,decision,lesson}.schema.json
│   │   ├── emit_episodes.py       # T0：Backup/**/*.json → episodes.jsonl（純腳本）
│   │   ├── emit_decisions.py      # T1：memory/*.md + episodes → decisions.jsonl（半自動）
│   │   ├── lessons.jsonl          # T2：人工審核後定稿
│   │   ├── episodes.jsonl
│   │   └── validate.py            # schema + id 唯一 + superseded_by 完整性
│   ├── sft_pi/                    # tool-call 軌跡
│   │   ├── build_sft_pi.py
│   │   └── {train,valid}.jsonl
│   ├── sft_prime/                 # (證據→教訓) 對 + (狀態→下一臂)
│   │   ├── build_sft_prime.py
│   │   └── {train,valid}.jsonl
│   └── harness_engine/            # 第 2 份 PRIME_AGENT_CODING_AGENT_DIR
│       ├── extensions/            # 指向 tb_loop/harness/extensions（不複製）
│       └── memories/engine/       # 一 lesson 一檔（供 /refine 與注入）
│
└── shared/
    ├── trace_schema.md            # 三種 record 的權威定義（engine_loop/traces/schema 的說明）
    ├── sanitize.py                # 絕對路徑→$REPO、IP/hostname/金鑰遮罩
    └── pack_evidence.py           # log 抽樣 → evidence/<id>.txt.zst + sha256 索引
```

**紅線**：`scripts/check/*` 與 `scripts/run_server.sh` 是生產腳本，**不得複製**進 engine_loop。
（`knifeedge_matrix.py` 181 KB、`decode_sweep.py` 29 KB、`m123_oracle_gate.py` 31 KB——複製一份就等於
製造兩份真相，而且 pre-commit 閘門與白皮書引用的是 `scripts/` 那一份。）engine_loop 用 wrapper
呼叫它們，並把輸出正規化成 record。

---

## 4. 三種 record 的 schema（附今天真實樣本）

### 4.1 `episode` —— 一次 run、一個臂

```json
{
  "type": "episode",
  "episode_id": "eng-20260915-193150-p25-gputime-r1",
  "loop": "engine",
  "profile": "prod25",
  "goal": "measure decode t/s and the per-step phase split of the prod25 decode path with MTP off",
  "arm": {
    "name": "p25-gputime",
    "declared_purpose": "baseline",
    "usable_for_throughput": true,
    "env": {"CGC_SERVER_MTP": "0", "CGC_GPU_TIMING": "1", "CGC_DECODE_PROFILE": "1"}
  },
  "build": {"server": "131bc5316ebf", "metal": "fdfe6653b03c", "llama": "d8a1e17c571c"},
  "obs": {
    "decode_tps": {"median": 9.79, "min": 9.78, "max": 9.88, "n_rounds": 3},
    "answer_md5": ["28097996"], "answer_stable": true,
    "pool": {"hit_rate_pct": 91.0, "capacity_pct": 51.6, "io_effective_mib_s": 198.0},
    "phase_ms": {"wait": 75.27, "gpu_busy_sum": 95.73, "gpu_union": 68.00, "gap": 22.36},
    "asserts": {"mmid_oob": 0, "engine_zero": 0, "model_zero": 14}
  },
  "gate": {"name": "m123_oracle_gate", "verdict": "n/a",
           "ref": "Backup/knifeedge_matrix/ref_iq3_pool8gb_M2_6144_bitident.jsonl",
           "env_passthrough": {}},
  "artifact": {"log": "Backup/cgc_logs/llama_server_20260915_193150.log",
               "log_bytes": 28634, "sha256": "…"},
  "usable_as_evidence": true,
  "caveats": ["single-arm baseline; no bit-identical gate in this arm"]
}
```

關卡：`build` 三個欄位不得為空（沒有 fingerprint 的數字不可比）；`answer_md5` 集合大小 >1 時
`answer_stable=false`，且該 episode 預設 `usable_as_evidence=false`。

### 4.2 `decision` —— 一個判斷點（**最高價值**）

```json
{
  "type": "decision",
  "decision_id": "dec-20260915-2000-cause-is-serialization",
  "question": "Is the ~44% of decode step time removable serialization, or is the GPU genuinely busy?",
  "evidence": [
    {"episode_id": "eng-20260915-193150-p25-gputime-r1",
     "reading": "wait 57.7–75.3 ms (83–90% of the step); the CPU-side window (cb+submit) is only 8.5–15 ms"},
    {"episode_id": "eng-20260915-193322-p25-submit-ahead-r1",
     "reading": "9.45 → 16.82 t/s (×1.78) with a second segment in flight; md5 87647aec (degraded) vs 28097996"}
  ],
  "reasoning": "A second in-flight segment cannot make real GPU work faster, so the delta is dispatch/serialization, not compute. The GPU-clock-domain gap (12.3–22.4 ms/step) requires the queue to actually empty, so it is evidence; union/wait ≈ 87–95% is structural (a single-queue [min start, max end] spans the whole occupancy window) and is NOT.",
  "conclusion": "the cause is B (serialization), not A (GPU execution efficiency)",
  "confidence": "high",
  "judgement": "sound",
  "action": "write the D3 GPU-side slot-table design; note the 1.78× as an UPPER BOUND of an incorrect-output probe, not a real speedup",
  "ruled_out": [
    {"claim": "double-buffering the remap leaf recovers the 1.78×",
     "why_false": "the CPU round trip stays on the critical path, and n_segs is unchanged (still 39–40)"}
  ],
  "artifact": "docs/REMAP_ROUNDTRIP_REMOVAL_PLAN_2026-09-15.md",
  "superseded_by": null
}
```

`judgement ∈ {sound, refuted, unresolved}`——**這一欄是防止把錯誤當正例的閘門**（見 §10 R4）。

### 4.3 `lesson` —— 可重用的規訓（= harness 狀態）

```json
{
  "type": "lesson",
  "lesson_id": "eng-mh-0003",
  "class": "measurement-hygiene",
  "rule": "Quote a throughput delta only as the paired per-rep median of an interleaved A/B with ≥3 rounds, with the build fingerprint pinned and the md5 set printed next to the number.",
  "because": "A single round of CGC_SUBMIT_AHEAD measured ×1.78; the interleaved ×3 run measured ×1.711 (range 1.327–1.924) and the md5 differed on every rep — the single round was not quotable at all.",
  "counterexample_observed": "eng-20260915-1900-p25-submit-ahead (single round, degraded output)",
  "applies_to": ["scripts/check/ab_interleave.py", "scripts/check/decode_sweep.py"],
  "superseded_by": null
}
```

今天可直接定稿的高價值 lesson（每一條都已在本輪被證據支持）：

| id | class | rule 的一句話摘要 |
|---|---|---|
| `eng-mh-0003` | measurement-hygiene | 吞吐差只引用「交錯 ≥3 輪的 paired median + build fingerprint + md5」 |
| `eng-lf-0007` | log-forensics | rate-limited 斷言的輸出是**抽樣不是普查**，不得據此推「只有某幾層受影響」 |
| `eng-diag-0011` | diagnosis | 結構性恆零的診斷欄位不是證據（`host=0` 在 Metal 的 shared 與 private 都印 0） |
| `eng-diag-0012` | diagnosis | 同一 graph 混用多 backend 時，「誰算的、誰讀的」是契約；先印 `## SPLIT` + per-node backend（`GGML_SCHED_DEBUG=2`），再談內容（今天 S1 的 layer 0 就是這條） |
| `eng-gate-0002` | gate-integrity | 閘門必須先證明「它會擋錯」；參考檔若是被測配置自己產的，自比較只能抓非決定性 |
| `eng-src-0005` | source-reading | 變數名稱不等於語意（`build_moe_ffn` 的 `n_tokens ≠ ubatch.n_tokens`） |
| `eng-src-0006` | source-reading | `ggml_reshape_1d` 之前先確認連續性（`argsort_top_k` 的輸出是帶 stride 的 view） |
| `eng-bound-0001` | honest-bounds | 上界探針（故意輸出錯誤）的數字必須全程標註，不得進入任何對外表格 |
| `eng-diag-0013` | diagnosis | **診斷要先被證明會印**：預設 verbosity 會濾掉 `GGML_LOG_DEBUG`，0 行輸出與「沒有東西可報」不可區分 |
| `eng-smoke-0002` | smoke | 入口腳本不能只做 `bash -n`；`agent_harness` 的三個入口今天連 `import` 都失敗（套件名 `tb_loop` 不存在） |

---

## 5. 產生流水線（三層，T0 今天就能跑）

```
                     ┌──────────────────────── T0（純腳本，無 LLM）────────────────────────┐
Backup/phase_decomp/*.json ─┐
Backup/m123_oracle_gate/*   ├─▶ emit_episodes.py ─▶ traces/episodes.jsonl  (schema 驗證)
Backup/knifeedge_matrix/*   │        ▲
Backup/llama_bench/*        │        └─ build.json（runners/rebuild.sh 產）
cgc_logs/*.ctrl.log ────────┘
                     └───────────────────────────────────────────────────────────────────────┘

                     ┌──────── T1（半自動：prime-agent /refine --global）────────┐
traces/episodes.jsonl ＋ memory/YYYY-MM-DD.md ＋ docs/*.md/.html
        └─▶ distill/refine_engine.sh ─▶ traces/decisions.jsonl ＋ lessons.candidate.jsonl
                     └──────────────────────────────────────────────────────────┘

                     ┌──────── T2（人工審核 → 定稿）────────┐
lessons.candidate.jsonl ─▶ 人審 ─▶ traces/lessons.jsonl ＋ harness_engine/memories/engine/*.md
                     └─────────────────────────────────────┘

                     ┌──────── 投影 ────────┐
traces/*.jsonl ─▶ sft_pi/{train,valid}.jsonl    （工具軌跡）
              └─▶ sft_prime/{train,valid}.jsonl （(證據→教訓) 對；(狀態→下一臂)）
```

`distill/refine_engine.sh` 直接沿用 `tb_loop/learning/refine_harness.sh` 的三個既有設計（餵完整軌跡、
限定輸出類型、顯式 `--global`），只換兩件事：**scope 標記 `[engine]`** 與**要求輸出 JSONL 而非散文**。

---

## 6. 接到 pi + prime agent

### 6.1 `harness_engine/`（prime-agent 的第二份狀態）

prime-agent 的 Continual Harness 狀態是一個目錄（`PRIME_AGENT_CODING_AGENT_DIR`）。`tb_loop` 用
`tb_loop/harness/`；engine loop 用 `agent_harness/engine_loop/harness_engine/`，形狀相同：

- `extensions/` —— 指向 `tb_loop/harness/extensions/gemma4-provider.ts`（**不複製**，避免 provider 版本漂移）
- `memories/engine/<lesson_id>.md` —— 一 lesson 一檔，`class` 當 tag
- 第一行固定 `[engine] <rule>`，讓 `/refine --global` 的 scope 可辨

### 6.2 兩份 SFT 投影

| | `sft_pi/` | `sft_prime/` |
|---|---|---|
| target | pi-coding-agent（工具使用） | prime-agent（harness 自改進） |
| 一條樣本 | `messages`: system=CONVENTIONS.md 摘要 → user=「現況 + 可用臂 + 目前數字 + 閘門狀態」 → assistant=tool_calls（`bash: scripts/check/decode_sweep.py --arms …`）→ tool 結果 → assistant=結論 | `(evidence, lesson)` 對；以及 `(狀態, 下一個該跑的臂 + 會被什麼否證)` |
| 來源 | `episode` 序列 + `decision.action` | `decision` + `lesson` |
| 對應既有 | `tb_loop/learning/build_sft_dataset.py` 的同構版本 | `/refine` 的訓練面 |

`sft_pi` 的 system prompt **就是 `CONVENTIONS.md`**——這是刻意的：模型在推理時看到的判準，和它被訓練時
看到的判準逐字一致（沿用 `tb_loop/README.md` 已驗證的「訓練格式與運行時格式逐位元組一致」原則）。

### 6.3 閉環實驗（engine 版的 round1 vs round2）

用同一組「待答問題」（例如 S2 / S3 該先跑哪個臂、`n_segs` 該怎麼塌陷）跑兩次：

- **round1**：`harness_engine/memories/` 空的
- **round2**：注入 lesson

比較維度不是分數而是**決策品質**：是否先跑對照臂、是否標註 fingerprint、是否避開已知陷阱
（例如是否重犯「單輪就引用」）。`compare_rounds.py` 的形狀可直接沿用。

> **2026-09-16 19:5x 註記（E3 的執行器落地時）**：上面那個「8 條」是手寫的數字，而本計畫自己
> 已經不一致 —— §4.3（§11 指明它是那 8 條的來源）**現在列了十條**。「那 8 條」的忠實讀法是
> `lessons.jsonl` 的**檔案前 8 筆**（append-only 的寫入順序；它的前八筆正好是 §4.3 的前八列），
> 而 `lessons.jsonl` 現在有 110+ 條 live lesson。所以注入範圍做成了參數：
> `--memories-scope all|none|first:<k>|class:<abbr>|ids:<path>`（預設 `all`）。
> 選取**用權威的 `traces/lessons.jsonl`**、內容**用衍生的 `harness_engine/memories/engine/`**；
> 選到的 id 若沒有對應的 memory 檔，執行器**硬錯誤**而不是靜默跳過
> （「篩選與投影不同步」不可與「那些 lesson 不適用」同形）。
>
> 另兩個實作決定：`--reps` 預設 **3**（本表的驗收寫著「可重現」，單次採樣證明不了；reps 之間
> **交錯**、且每個 rep 的臂順序**輪轉**，避免序列中的位置成為候選原因），且**臂內不一致的題目，
> 其跨臂判斷在 `compare.json` 裡被標成 `comparable: false`**。
> 執行器：`distill/closed_loop.py`；離線自測：`distill/closed_loop_selftest.py`
> （33 項，含「儀器必須說得出『沒有差異』」的陰性對照）。

---

## 7. 命名與落盤約定

- `episode_id` = `eng-<YYYYMMDD>-<HHMMSS>-<arm>-r<rep>`（時間取自 ctrl log 的啟動時間，不用人工填）
- `decision_id` = `dec-<YYYYMMDD>-<HHMM>-<slug>`
- `lesson_id` = `eng-<class 縮寫>-<4 位序號>`；class 縮寫表：`mh`(measurement-hygiene)、`lf`(log-forensics)、
  `diag`、`gate`、`src`、`bound`、`perf`
- 一律 JSONL、UTF-8、一行一 record、欄位順序按 schema（diff 友善）
- 時間戳一律從檔案／log 取，**不手算**（今天的 ctrl log 檔名就是權威時間）

---

## 8. 體量與去隱私

- **log 政策**：`Backup/cgc_logs` 是 463 MB / 1399 檔（今日 206）。`shared/pack_evidence.py` 只保留命中
  `CGC-MMID-ASSERT|CGC-S1|CGC-HOOK|EXPECT-|CGC-GPUTIME|CGC-DECPROF|## SPLIT` 的行 ±8 行 ＋ 檔頭 40 行，
  上限 256 KB/episode，並寫 `sha256` 指向原文（原文**不進 repo**）。
- **去隱私**：絕對路徑 → `$REPO`；`192.168.101.90`、主機名、`sk-*` 一律遮罩。
  ⚠️ `run_server.sh` 會印「連線卡」（含區網 IP 與 port）→ `sanitize.py` 必擋。
- **納管策略**：`traces/*.jsonl`、`sft_*/*.jsonl`、`MANIFEST.jsonl`、`CONVENTIONS.md` **納管**（它們是
  產出物本體）；`Backup/cgc_logs` 原文、`harness_engine/logs/`、`sessions/` **不納管**。

---

## 9. 分階段執行與驗收

| 階段 | 內容 | 驗收（可機檢） |
|---|---|---|
| **E0**（零風險，今天可做） | 只做索引 + 第一版 episode 匯出：`engine_loop/README.md`、`MANIFEST.jsonl`、`traces/emit_episodes.py`、`traces/schema/*` | `emit_episodes.py --from Backup --out traces/episodes.jsonl --validate` 全綠；≥30 筆；**每筆都有 `build` fingerprint**；`git status` 除新檔外無變化 |
| **E1** | 結構 + 憲章：`git mv` 既有內容進 `tb_loop/`；修掉 import 路徑；寫 `CONVENTIONS.md` | `python -c "import tb_loop.agents.prime_agent_adapter"` 成功；`tb run --n-tasks 1` smoke 過；`CONVENTIONS.md` 每條都能指到今天的具體證據檔／log 行 |
| ↳ E1 實際結清狀況（2026-09-16 17:2x，`docs/AGENT_HARNESS_E1_RESTRUCTURE_20260916.html`） | `git mv` 完成（642 個 rename）；`PYTHONPATH` 改指 `tb_loop` 的父目錄；`tb_loop/__init__.py` 補上；`config.env` 新增 `TB_HARNESS_ROOT`／`TB_REPO_ROOT` 兩個錨點 | ✅ **套件解析通過**：`tb_loop` 為正規套件，四個模組解析到新位置，且對照組（`PYTHONPATH=tb_loop` 自己）仍正確報 `No module named 'tb_loop'`。⚠️ **smoke 未跑**：`tb_loop/.venv` 不存在、`terminal-bench` 未安裝、Docker daemon 未執行 ⇒ **驗收只結清一半**，`import tb_loop.agents.prime_agent_adapter` 目前止於 `No module named 'terminal_bench'` |
| **E2** | T1 蒸餾 + 兩個投影：`distill/refine_engine.sh`、`build_sft_pi.py`、`build_sft_prime.py`、`harness_engine/` | 跑一次 refine → `decisions.jsonl` 過 validate；`sft_prime/train.jsonl` ≥ 50 條；注入 `harness_engine` 前後同一臂的決策差異可讀；**＋ 結清 D6 的閉環欠帳（見本表下方）** |
| ↳ E2 實際結清狀況（2026-09-16 19:4x，`docs/AGENT_HARNESS_E2_TRAINING_PROJECTIONS_20260916.html`） | 四個目錄都建了：`harness_engine/`（106 條 memory，由 lessons 推導、`--check` 可驗；`extensions` 是符號連結不是複本）、`sft_pi/`（14 條軌跡／38 tool call）、`sft_prime/`（141 筆）、`distill/`（證據收集器 ＋ 驅動 ＋ prompt ＋ 假 prime-agent 自測 27 項全過 ＋ 四臂閉環對照器） | ✅ **機制層面通過**：`sft_prime/train.jsonl` = 120 條 ≥ 50；`sft_pi` 每筆帶 `_provenance.reconstructed`；`distill/selftest.py` 全過且抓到一個真的 prompt 設計缺陷（可用 id 被證據區的既有 id 撞掉）。⚠️ **兩件未結清**：① **T1 沒接真實模型跑過**（`REFINE_ENGINE_MODEL` 不給預設值，本機無端點）；② **D6 的閉環欠帳仍未結清**——對照器已建且 `--dry-run` 通過，但**模型半未跑**（跑它要起 13.6 GB 的 server，會與另一個 session 的 prefill/decode 量測互相干擾）。**依原文要求：在對照跑過並留有產物之前，不得聲稱 D6 修訂已結清。** |
| **E3** | 閉環：round1（無 lesson）vs round2（注入 lesson）在同一組待答問題上 | 至少 1 條 lesson 被證明改善且可重現；否則如實記為 negative（不美化） |
| ↳ E3 的執行器與實驗設計（2026-09-16 19:5x，`docs/AGENT_HARNESS_E3_DESIGN_20260916.html`） | 注入範圍做成參數（`--memories-scope`，預設 `all`；`first:8` 才是計畫原文的「那 8 條」）、`--reps` 預設 3 且**交錯＋輪轉**、臂內不一致的題目標 `comparable: false`、`manifest.json` 記下注入了哪些 id 與每個 rep 的臂順序 | ✅ **機械面通過**：`distill/closed_loop_selftest.py` **33 項全過**，含「同一個假模型、同一組題目，只換 scope ⇒ 一次必須答『相同』、一次必須答『不同』」的陰性／陽性對照。⚠️ **模型半未跑**（與 D6 欠帳同一個 blocker：本機無端點，而起 server 會干擾另一條線的量測）⇒ **E3 本體仍未結清** |
| **E4**（可選治理） | log 政策落地、`knifeedge_matrix.py`（181 KB）拆分、`scripts/check/` 40+ 檔分層、標記 stale（`replay_bench_*` 已 stale，`RUN_REPLAY_BENCH=0`） | `pack_evidence.py` 產物總量 < 20 MB；stale 資產有明確 banner |

**E2 承接一筆已認領的欠帳（2026-09-16 加入；來源：`CONVENTIONS.md` D6 的 dated 修訂）。**

`CONVENTIONS.md` §E 規定「改動它等於改動兩個 loop 的行為，每次改動都要走一次 §6.3 的閉環對照」，
因為它同時是 `engine_loop/sft_pi/` 的 system prompt 與 `harness_engine/memories/engine/` 的注入來源。
2026-09-16 對 D6 的修訂**改變不到任何執行時行為**（`sft_pi/` 與 `harness_engine/` 當時都還不存在），
但該閉環對照**尚未執行**。E2 一建立這兩個目錄，欠帳就變成**有消費者、可執行、可失敗**的檢查：

- **認領條件**：`distill/` 與 `harness_engine/` 一落地，就對「D6 修訂前後兩版 `CONVENTIONS.md`」
  各跑一次同一組待答問題，並如實記錄差異。**沒有差異也是結果**，不可美化成「通過」。
- **在對照跑過並留有產物之前，不得聲稱 D6 修訂已結清。**
- 這筆欠帳同時記在三處：本表、`CONVENTIONS.md` D6 修訂段、`.workbuddy/memory/MEMORY.md`。

**E0 是唯一今天就能做完且不會弄壞任何東西的一步。** 建議先只做 E0，看匯出的 episode 品質再決定 E1。

---

## 10. 誠實邊界與風險

- **R1 兩份真相**：把 `scripts/check/*` 複製進 engine_loop 會立刻分叉（`knifeedge_matrix.py` 一個 bug 修兩處）。
  → 硬規則：engine_loop 只放 wrapper，`MANIFEST.jsonl` 記錄「權威路徑」。
- **R2 tb_loop 的 import 已經是壞的（實測，不是假設）**：`run_round.sh:37`、`gen_sft.sh:27`、
  `finetune/eval_round.sh:34` 都寫 `--agent-import-path "tb_loop.agents.…"`，而三支腳本都用
  `PYTHONPATH="$TB_LOOP_DIR"`、`TB_LOOP_DIR` = 本目錄本身（`agent_harness/`）。
  本輪實測：`PYTHONPATH=agent_harness python -c "import tb_loop.agents.prime_agent_adapter"`
  → `ModuleNotFoundError: No module named 'tb_loop'`。**也就是說三個入口腳本現在都跑不起來。**
  修法（E1 一併處理，兩種都只動 1 行）：
  - **(a) 推薦**：`git mv` 既有內容進 `agent_harness/tb_loop/`，同時把三支腳本的 `PYTHONPATH` 改成
    「`TB_LOOP_DIR` 的父目錄」→ 匯入字串 `tb_loop.agents.…` **一行都不用改**，套件名與 import 路徑
    完全對齊。這也讓 `agent_harness/` 的頂層只留下 loop 目錄，不再是一堆散落的 sql/agent/learning。
  - **(b) 最小改動**：不動目錄，把 `PYTHONPATH` 從 `$TB_LOOP_DIR` 改成 repo 根，並把 `agent_harness`
    改名為 `tb_loop`（或加一個 `tb_loop -> .` 的符號連結）。不推薦：符號連結在打包／rsync 時容易斷。
  驗收必須是「真的 import 成功 ＋ 一次 `--n-tasks 1` smoke」，`bash -n` 抓不到這個。
  **已處置（2026-09-16 17:2x，E1 採路線 (a)）**：642 個 rename 完成；`PYTHONPATH` 改指
  `TB_HARNESS_ROOT`（= `tb_loop` 的父目錄）；補 `tb_loop/__init__.py` 讓它是**正規套件**
  而不是靠 PEP 420 的 namespace package。**套件解析層面已通過**（含一組對照組），
  但 **smoke 仍未跑**（`.venv`／`terminal-bench`／Docker 三者皆缺）⇒ 這條風險**只結清一半**。
  **本條不刪除**：它記錄的推理（「套件名與檔案所在地由同一次搬動決定 ⇒ 兩者會一起對、也一起錯」）
  在 E2 建 `sft_pi/`、`harness_engine/` 時會再被用到。
  另一個推論：`TB_LOOP_DIR` 這種「恰好等於另一個路徑」的定義是**定時炸彈**——
  它讓 `PYTHONPATH="$TB_LOOP_DIR"` 同時是對的又是錯的，只是在被移動之前無法區分。
  E1 因此把三個錨點（`TB_LOOP_DIR`／`TB_HARNESS_ROOT`／`TB_REPO_ROOT`）顯式分開。
- **R3 把已被推翻的結論寫進訓練集**：今天的「雙緩衝 remap 就能拿到 ×1.78」是**被推翻**的假設，若當正例訓練
  等於教模型重犯。→ `decision.judgement` 必填，`superseded_by` 由 `validate.py` 強制檢查。
- **R4 負樣本只做 SFT 會教壞模型**（`tb_loop/README.md` 已明確警告）：`--include-failed` 的 SFT 會讓模型
  模仿失敗行為；反例只能用來配 **偏好對（DPO/ORPO）** 或加顯式「這是錯誤示範」前綴。
- **R5 訓練數據 ≠ 對外數字**：`traces/` 裡的 ×1.78、`id_oob first=2143289344` 這些都是**未定案/錯誤輸出**
  的觀測。對外引用仍以 `docs/*.html` 白皮書為準，且 engine loop 的 record 一律保留 `usable_as_evidence`。
- **R6 這份規劃本身的證據等級**：S1 的 bit-identical 閘門**尚未通過**（見
  `docs/REMAP_ROUNDTRIP_REMOVAL_PLAN_2026-09-15.md` §5 與今天的 log）。規劃裡的任何 S1 敘述都只是
  「今天的觀測」，不是結論。

---

## 11. 建議的第一步

只做 **E0**，並且把範圍壓到最小：

1. `traces/schema/{episode,decision,lesson}.schema.json`（三份）
2. `traces/emit_episodes.py`（讀 `Backup/phase_decomp/*.json`、`Backup/m123_oracle_gate/*.json`、
   `Backup/llama_bench/*`、`Backup/knifeedge_matrix/*.json`，輸出 `episodes.jsonl`）
3. `MANIFEST.jsonl`（今天盤點表 → 機器可讀）
4. `engine_loop/README.md`（本 loop 的 evaluate/extract/refine/compare 定義）
5. `CONVENTIONS.md` 的第一版（§4.3 那 8 條 lesson 直接搬成憲章條文）

跑完 `emit_episodes.py --validate` 之後，再決定要不要動 tb_loop 的目錄結構（E1）。
