# **儀器對帳：m 的兩台儀器定標** m（每 draft token… — 技術白皮書　·　3a ③a 實驗目標達成（階段性）

> **一句話**：**儀器對帳：m 的兩台儀器定標** m（每 draft token 的邊際驗證成本）的兩台儀器為什麼給出不同的讀數？ - bench 儀器：m = **0.474** - 服務端儀器：m = **0.21** 這個落差是「量法／定義不同」還是「m 真的在飄」？ 為什麼要問：判準是 `S = (1+a·k)/(1+m·k)`、`k→∞` 天花板是 `a/m`。 ⇒ **轉正與否完全由 `a − m` 的符號決定**，m 的讀數不一致 ⇒ 這個符號目前**算不出來**。

- 主題：實驗　·　子目標：**M MTP on 加速**（活躍攻關軸：speculative 攤薄係數 m 與 accept rate a）
- 階段：實驗階段（階段性）　·　③a —— 機制／量測成立，但產物還不能進生產

---

## 1. 目標

**儀器對帳：m 的兩台儀器定標** m（每 draft token 的邊際驗證成本）的兩台儀器為什麼給出不同的讀數？ - bench 儀器：m = **0.474** - 服務端儀器：m = **0.21** 這個落差是「量法／定義不同」還是「m 真的在飄」？ 為什麼要問：判準是 `S = (1+a·k)/(1+m·k)`、`k→∞` 天花板是 `a/m`。 ⇒ **轉正與否完全由 `a − m` 的符號決定**，m 的讀數不一致 ⇒ 這個符號目前**算不出來**。

## 2. 判準

同一 session／同 seed 下，兩台儀器量到的 m **一致**（或差值可歸因於明確的定義差異）， 且 `a − m` 的符號可判定 ⇒ 待辦 ④ 有「可判定的下一步」。 若 `a > m` ⇒ **MTP-on 已轉正**，待辦 ①（修 seq-1 雙加）為唯一未套用缺陷修補。

## 3. 結果

實驗進行中（尚無讀數）

## 4. 判定

**3a · ③a 實驗目標達成（階段性）** — 達成實驗設計目的（機制／量測成立），但產物還不能放進生產級設置

> 假設：天花板是 `a/m`（k→∞）。`S(k) = (1+a·k)/(1+m·k)` 逐 k 逼近它。 ⇒ **「轉正與否」＝ `a > m` 的符號判定**，而 m 此刻有兩台儀器各說各話 （bench 0.474 ／ server 0.21），符號無法判定。 ⇒ 所以「持續優化」的第一件事是**定標 m**： 兩台儀器在同一 session／同 seed／同 shape／同 warm-skip 下各量一次， 把落差歸到「誰的量法不同」，而不是「m 真的在飄」。

---

## 5. 子目標綁定 · options · 測試 Log 報告

- **子目標**：`L25-5`　MTP 的產品化上限
- **來源**：子目標看板（唯一來源）　·　置信度 `high`
- **profile**：`prod-new`
- **被測 option**：`CGC_SERVER_MTP=1`
- **儀器開關**（不是被測 option）：`CGC_MTP_PERF=1`
- **CLI**：`--spec-type draft-mtp`；`--spec-draft-n-max 1`
- **arm 1（可複製）**：`prod-new:CGC_SERVER_MTP=1;CGC_SERVER_LAYER_CAPS=40-40:16;CGC_DRAFT_CTX_ALIGN=1;CGC_DRAFT_SMALL_BATCH=1;CGC_MTP_PERF=1`
- **結案狀態**：結案（§63，端點＝m 比值門檻）：乾淨窗口（attribution=none、NOMINAL、起跑 free 7159 MB）一趟 k=1 的 m ＝ 0.159（T_draft 13.70 ms/輪 ÷ 86.13；五條閘全過）≤ 目標 0.30。⇒ 卡片自己推的 k 無窮上界 24.59 是由 m=0.474 推出來的。⚠ 09-30 更正：原文寫「實測更小 ⇒ 上界仍成立」是方向反了 —— m 在分母，m 越小上界越高：1000/(85.8×0.159) ＝ 73.30（不是 24.59）。⚠ 兩件事要說清楚：(i) m 在劣化窗只會被吹大（單邊性）⇒ 乾淨窗量到的 0.159 是上界；(ii) prod-new 的 draft 成本比 09-24 舊 build 高 60%（0.159 vs 0.099）⇒ 但「天花板比舊帳面更低」講反了：draft 成本變高 ⇒ m 變大 ⇒ 上限變低；本格量到的 m 比舊帳面小（0.159 vs 0.474）⇒ 天花板反而被抬高到 73.30。兩者不衝突：09-24 舊 build 的 m≈0.099 是「同 cell、pool 3 GiB」的讀數，不能與 prod-new 的 0.159 直接比價；真正的帳應以同一 cell 為準。
- **逐條處置**：整合進子目標　→ `L25-5`　—　儀器對帳：定標 m（bench vs 服務端），使 verify 成本優化可判定。
- **測試 log**：無實跑 log（本條的證據是下面的「對應報告」，不是量測產物）
- ★ 09-29 起本格的量測口徑改為 prod-new ＋ harness bench（唯一認可入口），不再是 prod25／server ABBA。判準：m ≤ 0.30 且指名它的主項（draft 流量／verify 寬度／pool 預算）。⚠ 單邊性：窗劣化只把 wall 吹大 ⇒ 實測 t_draft ≥ 真值 ⇒ 由它算出的 m 是上界。

## 6. 與其它條目的關係（同軸／同階段，自動對照）

| 條目 | 級 | 結果（摘） |
|---|---|---|
| [MTP k-sweep（verify batch T 成本曲線）](mtp-ksweep.md) | 3a | step ≈ 14.61 + 42.03·T（max resid 11.5）；最佳 k=2 但那格不可移植（合成 accept 0.93~0.99 vs 交付 0.465~0.58 |
| [M：攤薄係數 m 0.474 → ≤0.073](exp-m-draft-cost.md) | 3a | **未跑（有卡、無讀數）**：t/s(k) = 1000·mean_len(k)/(S0·(1+m·k))；k=3 即使 a=1.0 也只 **19.25**、k→∞ 極限 **2 |
| [**「邊際 verify token ＝ 34.7 ms」由什麼構成…](exp-marginal-decomp.md) | 3a | decode 9.97 t/s |
| [M2 slab pool-reuse（整層填的 memcpy 省多少）](e-m2-slab-reuse.md) | 3a | decode 8.26 t/s |
| [同一趟（同 build、同格、同 session）裡： ① caps…](e-victim-classify-2026-10-05.md) | 3a | 尚無可引用讀數：1 個 run 全部非乾淨（attribution.verdict=swap（swap_growth=5369.81 MiB, max_swap=9644.69 M |
| [**寬度軸的因果量測（W1 vs W2，同 session 同 se…](exp-w1w2-paired-2026-10-06.md) | 3a | 實驗進行中（尚無讀數） |

## 7. 子目標分解（持續更新；1/5 完成）

- [x] 跑前立項（現狀/目標/假設/驗收） [證](../../../scripts/check/charters/exp-m-instrument-recon-2026-10-06.yaml)
- [~] 用生產級腳本 prod-new ＋ 自己 option 跑實驗臂，留存 log
- [ ] 驗收：同一 session／同 seed 下，兩台儀器量到的 m **一致**（或差值可歸因於明確的定義差異）， 且 `a − m` 的符號可判定 ⇒ 待辦 ④ 有「可判定的下一步」。 若 `a > m` ⇒ **MTP-on 已轉正**，待辦 ①（修 seq-1 雙加）為唯一未套用缺陷修補。
- [ ] 否證條件：同一條件下兩台儀器仍給不可調和的不同 m ⇒ 沒有可引用的 m ⇒ **verify 成本的優化目前無法判定有沒有效** ⇒ 待辦 ④ 退回「定標」，不得進入「優化」。
- [ ] 定標 m：同 session 同 seed 下，bench 儀器與服務端儀器各量一次並對帳

## 8. 依據 · 備註 · 對應報告

| 項目 | 內容 |
|---|---|
| 依據 | `docs/MTP_2X_BOUNDARY_2026-09-19.md §1–§2（公式與數字）； scripts/check/decode_board_2026-09-29.yaml:94／:1059（看板轉引）； scripts/check/charters/e-mtp-m-acc-2026-09-29.yaml（原始立項，owner: freebuff）` |
| 備註 | 假設：天花板是 `a/m`（k→∞）。`S(k) = (1+a·k)/(1+m·k)` 逐 k 逼近它。 ⇒ **「轉正與否」＝ `a > m` 的符號判定**，而 m 此刻有兩台儀器各說各話 （bench 0.474 ／ server 0.21），符號無法判定。 ⇒ 所以「持續優化」的第一件事是**定標 m**： 兩台儀器在同一 session／同 seed／同 shape／同 warm-skip 下各量一次， 把落差歸到「誰的量法不同」，而不是「m 真的在飄」。 |
| 軸性質 | 活躍攻關軸：speculative 攤薄係數 m 與 accept rate a |
| 對應報告 | 0 份 |

- （無）

---

← [**寬度軸的因果量測（W1 vs W2，同 session 同 se…](exp-w1w2-paired-2026-10-06.md)　·　[總目錄](index.md)　·　[HTML 版](exp-m-instrument-recon-2026-10-06.html)　·　[**拉 a：chain 路徑開 rejection sampling… →](exp-a-lever-rejection-2026-10-06.md)

本檔由 `scripts/check/mindmap_brief_build.py` 從 `docs/mindmap/mindmap.json` 機械生成；改內容請改 JSON 後重跑，勿直接編輯本檔。
