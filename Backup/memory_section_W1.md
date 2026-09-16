
### §W.1 收尾（commit 後才知道的事實）

- commit `0f9e7e5a3`（25 檔：19 A ＋ 6 M，`src/` **零變更**），已推 `origin` ＋ `cgcengine0907`，
  **三處 SHA 相同**。
- D5 `--tag instrcompare`：**M1/M2/M3 ＝ 9/9**、`comparable=True`、`config_diffs=[]`。
  產物 mtime 10:29–17:08（`libggml-metal` 17:08:55、`llama-server` 17:08:56），
  跑程 18:07–18:17 之間沒有任何 `Building …` 行 ⇒ 全部數字屬同一份產物，指紋見白皮書 §10。
- 驗證：`thermal_pressure --selftest` **15/15**（原 10）、`validate` OK（122/38/**106**）、
  `selftest` 10/10、`manifest OK: 76 assets`、`memory index OK: 58 sections`、
  `check_build_tracked.sh` 0 FAIL / 4 SKIP 各有理由、`whitepaper_selfcheck.py` OK(0)。
- **交接**：`agent_harness/skills/` 與 `agent_harness/memory/` 的快照**本輪沒有刷新**（分工是
  另一條線負責）。banner 已聲明不會自動跟上，故延後是設計允許的。本輪進入 `agent_harness/`
  的只有 `engine_loop/index_assets.py`（兩條 CURATED note）與 `traces/lessons.jsonl`（4 條）
  以及 `engine_loop/{memory/INDEX.jsonl, MANIFEST.jsonl}` 的重生——都是引擎層的索引機制。
- **`MEMORY.md` 現在 18,565 B / 213 行**（本輪 ＋約 2.3 KB）。它在 session 開始時曾因超過注入
  上限被截斷（當時 24,104 B），18.5 KB 應該安全，但下一次要加東西時**先壓縮**。
- **一個沒有結尾的線索**：`Backup/run_instrument_compare.sh` 第 7 臂（`prod25` 裸臂）**沒有**
  DECPROF 輸出，因為 `ARMS["prod25"]` 不帶 `CGC_DECODE_PROFILE`/`CGC_GPU_TIMING`。
  所以「MTP 模型」那一臂只有 t/s，沒有每步分解。要分解它得用
  `prod25:CGC_GPU_TIMING=1;CGC_DECODE_PROFILE=1`（**不要**加 `CGC_SERVER_MTP=0`）。
