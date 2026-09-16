
### §EN-8.2 準備工作與 skill 更新（2026-09-16 20:33，(b) 靜置中）

**(b) 進行中**：`Backup/run_spac_cold_arm.sh` 在背景等靜置。state file 用 **20:24:41**（本次最後一次
真正 GPU 負載＝D5 結束）**明確植入**；`COLD_QUIET=1800`（閘門預設）⇒ **20:54:41** 之後才發射，
之後閘門自己再 sleep `IDLE_BEFORE=180`（協議等待）。等待期間每 60 s 印剩餘秒數與讀數，
**不跑任何 GPU 工作**。

**(c) 的鏈驅動已備好**：`Backup/run_phase_c.sh`（**不用 `&&`**，每階段 echo rc ＋ 印哨兵行——
這是 §EN-6.2 那個「前階段靜默失敗會讓後階段一行都不跑」的直接對策）。
C1 = `SOLO=0 run_unified_depth_matrix.sh`（同行程 asc + desc，分離 depth 效應與順序效應）；
C2 = `run_spac_alpha_sweep.sh`（alpha 0.5/0.75/0.9 ＋ `CGC_SPAC=0` 對照，depth 512/1024，ctx 8192）。

**skill 更新（使用者層 `~/.workbuddy/skills/`，不在本 repo 內）**：

1. `cgc-decode-attribution` 新增一節 **「輸出擷取（kernel-side tensor capture）：三條規則」**
   —— 同臂對照是前提／命名規則是 `node(idx+n_fuse-1)`／`ABSENT` 不是 `SAME`；外加「發射順序不是計算順序」
   與「尾段 chunk 邊界」兩個比對陷阱、「先列舉再量測」、以及**混合堆疊的模型事實**
   （`full_attention_interval=4` ⇒ layer 2 是 delta-net，不是 flash attention）。
   並把 S1 現況那段接上本輪的定位表。
2. `cgc-commit-gate` 新增 **兩個欄位陷阱**：`superseded_by` 是 required（可為 null）、
   `applies_to` 的每一項是**檔案路徑**；並強調 **`-> OK` 不代表乾淨，要看 warning 數**。

**未提交的 skill 快照**：`agent_harness/skills/` 的那兩份快照仍落後（屬於另一條線的檔案，banner 已聲明
不會自動跟上）。要同步得由那一條線跑 `Backup/import_harness_snapshot.py`。
