
### §EN-8.1 (a) 的收尾：commit／推送結果，以及兩個我欠的更正（2026-09-16 20:33）

- **`648918b1bb39257a1a5302f236244bf6b121fbdd`** 已推 `origin` + `cgcengine0907`，**三處 SHA 相同**（34 檔，`src/` 3 檔含重建產物）。
- D5 `--tag capround2`：`comparable=True`、`config_diffs=[]`、M1/M2/M3 = 9/9；產物 20:09:32/33，閘門在其後。
- 白皮書：`docs/S1_CAPTURE_ROUND2_20260916_2035.html`（自檢 0 finding）。
- 新增 lesson：`eng-diag-0027`（定位）、`eng-src-0015`（命名規則）、`eng-mh-0043`（順序不穩）、`eng-mh-0044`（ABSENT／尾段邊界）。

**兩個更正（都在 commit 前修掉，沒有進版控）**：

1. **`eng-src-0015` 的機制寫錯了**：我寫「融合群末節點**匿名**時列被跳過（matcher 拒絕空名）」。
   實際上 ggml 給每個張量都取名，未命名者拿到自動名 **`node_NNN`**——所以列**有**被印出，
   只是名字不在明確清單裡 ⇒ **不匹配**。已改成正確描述（並用 `norm-2` 的 41/0 觀測當證據）。
2. **`applies_to` 的慣例是檔案路徑**，不是主題標籤（`CONVENTIONS.md` 57 次、`scripts/run_server.sh` 18 次…）。
   我第一版寫了 `s1-divergence` 之類的標籤 ⇒ 產生 12 條 WARN。已改成真實路徑，validate 回到 0 warning。

**另一個自我檢查**：白皮書草稿裡我寫「(a) 的第一半（`ffn_moe_down-2`）是 round 1」——對，
但**round 1 的 10 名單證據與 round 2 的 11 名單證據都必須進版控**，因為白皮書的表來自 round 2，
而 round 1 才是回答使用者 (a) 第一半（`ffn_moe_down-2` 的同構讀數）的那一輪。8 個臂日誌都提交了。

**（b）進行中**：COLD 臂的靜置視窗從 **20:24:41**（本次最後一次真正 GPU 負載＝D5 的結束，取 summary mtime）
起算，需 ≥1800 s ⇒ **20:54:41** 之後才發射。state file 用該 epoch **明確植入**（原本不存在，
不植入會是 `UNKNOWN-STATE`；植入現在時間則是假宣稱）。驅動 `Backup/run_spac_cold_arm.sh`，
等待期間每 60 s 印一次剩餘秒數與讀數。**等待期間不跑任何 GPU 工作**（rebuild、bench、sweep 都不行）。
