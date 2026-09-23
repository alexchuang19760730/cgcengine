# 下一步行動清單 — 2026-09-24

> 優先級排序後的待辦事項。做完手邊的事，照此順序。

> **⚠️ 強制前置（所有速度 A/B）**：讀 `docs/ABBA_MEASUREMENT_PROTOCOL_2026-09-23.md`
> 並照做（交錯、深冷卻 ≥300s、配對中位、log-space 校正、窗口守門、build 指紋）。
> 缺協議的數字不進報告。

> **🔴 commit 口徑統一**：commit 統一用 `prod-new`（單一介面）。任何寫進
> 報告/白皮書/commit 標題的數字一律 prod-new 口徑產出。

> **🔴 commit gate**：每次 commit 前跑 `python3 scripts/check/commit_bench.py`
> （prod-new，prefill≥120 / decode≥10，ABBA 協議）。標題帶 `[commit-gate]` 成績。

***

## 方向決策（2026-09-24 03:15，量 h 閉合）

### ❌ A（speculative slot binding / 跨 token 預指派）——從候選清單刪除
實測（docs/H_MEASURED_2026-09-24.md §四，prod-new / MTP off / 3721 層樣本）：
- **全中率 h_all = 0.0030**（每步 union 100% 被上一步 union 蓋住只有 0.3%）
- 贏 ρ 上界需 h ≥ 0.75；原門檻 0.65——實測差 200 倍以上
- 跨 token 猜語義上不可行：每次固定錯 ~2.7 個 expert
- **任何 agent 不要再碰 A 路線**（包括預指派 slot、跨 token 候選集加寬）

### ✅ ρ 路線 = 唯一正路（覆蓋率已飽和，剩 insert）
實測：cov_uni=0.8585 / rho_tok=0.8581（>0.70 飽和點，覆蓋率不是瓶頸）
剩餘槓桿（唯一正路）：
1. **ρ insert 實測**（白皮書 §10 下一步）——影子節點插入成本 4.76 ms/步 是估值，
   需實測歸因（CGC-RHO 影子節點 + gate+topk 的實際 GPU 時間）
2. cb 42–51 ms 遮蔽（CB_DELIVERY_SETTLED 已定讞，+14.5–19% 真實收益）

***

## 待辦（照序）

- [ ] **P0（正路）**：ρ insert 實測——量影子節點 + gate+topk 的實際成本，
      把 4.76 ms/步 從估值變成讀數；若 insert 不可壓，評估 ρ-batch 按層批次化（freebuff 的活）
- [ ] 環境治理：swap 8648MB 髒——重開機/purge 拿乾淨基線後再跑任何速度 A/B
- [ ] H_MEASURED §四 與 NEXT_ACTIONS 方向已 commit（含 h_all 儀器）
- [ ] freebuff：CGC_IDSEQ_DUMP 儀器不可達（CGC-CANON=0）——若要 per-call ids
      序列，移到真正執行的 topk hook 路徑；不急（h_all 已覆蓋判決需求）
