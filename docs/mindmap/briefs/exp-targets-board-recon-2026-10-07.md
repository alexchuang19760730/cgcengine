# 兩份權威檔對同一個里程碑的「是否已判死」講了相反的話，而沒有閘門會發… — 技術白皮書　·　3a ③a 實驗目標達成（階段性）

> **一句話**：兩份權威檔對同一個里程碑的「是否已判死」講了相反的話，而沒有閘門會發現。 能否加一支**只讀兩個檔**的對帳 checker，讓這種矛盾 fail-closed？ 以及：當兩邊衝突時，**哪一份是權威**？（此題需 operator 簽，本卡只提供依據）

- 主題：實驗　·　子目標：**不適用**（非攻關軸：約束／帳本／上界算術／入口索引／跨線產品）
- 階段：實驗階段（階段性）　·　③a —— 機制／量測成立，但產物還不能進生產

---

## 1. 目標

兩份權威檔對同一個里程碑的「是否已判死」講了相反的話，而沒有閘門會發現。 能否加一支**只讀兩個檔**的對帳 checker，讓這種矛盾 fail-closed？ 以及：當兩邊衝突時，**哪一份是權威**？（此題需 operator 簽，本卡只提供依據）

## 2. 判準

checker 存在並在既有三條矛盾上**如實 FAIL**（證明它會紅）， 且 operator 已指定權威來源 ⇒ 對帳後兩邊一致（不論是哪邊改）。

## 3. 結果

實驗進行中（尚無讀數）

## 4. 判定

**3a · ③a 實驗目標達成（階段性）** — 達成實驗設計目的（機制／量測成立），但產物還不能放進生產級設置

> 假設：共同根因是**宣告層與機械判準層分離**：人寫的結論在 A 檔、閘門讀的判準在 B 檔， 中間沒有單向對帳 ⇒ 兩邊可以同時「正確」而互相矛盾，且沒有人會發現。

---

## 5. 子目標綁定 · options · 測試 Log 報告

- **子目標**：`L25-1`　第六條軸：改「每 token 的 bytes 或 steps」
- **來源**：子目標看板（唯一來源）　·　置信度 `high`
- **profile**：`prod-new`
- **被測 option**：`--spec-type=ngram-simple`
- **候選（待指名）**：`--spec-ngram-simple-size-n=12（編譯期預設，不可覆寫）`；`--spec-ngram-simple-size-m=48（同上）`；`--spec-ngram-simple-min-hits=1（同上）`；`CGC_SERVER_LOAD_MODE=mmap`
- **arm 1（可複製）**：`prod-new`
- **結案狀態**：結案（排除）·不可測（operator 2026-09-30：不換盒子／不降模型）——PRIMARY 已否證（ngram 的 E 0.984）；SECONDARY＝load_mode=mmap：A 臂跑通（miss/step 中位 6.0、有 miss 層數中位 6.0/39、正規化 1.92%），B 臂在 pool 8 GiB 與 6 GiB 皆 Insufficient Memory、rc=−6 ⇒ 同形不可測。⛔ 這不是否證 ⇒ 本格自己的否證句「SECONDARY 否證 ⇒ 25 定案判死」不觸發；25 的定案現在只掛在 S1（is_mem_shared 的 draft 輪級掉線），而 S1 也要過同一個盒子（MTP-on 存活 ≥7703 MB、budget_gate 現判 OVERBUDGET）。（以下為結案前的舊註記，留檔）PRIMARY 否證；SECONDARY 09-30 首次實跑＝A 跑通、B fail-stop）：A 臂（delivery、load_mode=none）拿到 §8 兩量 —— ① miss 數/step 中位 6.0（IQR 4–12）② 有 miss 的層數/step 中位 6.0（min 1、max 39）、正規化 1.92%（與 e-fillbudget-m-decomp 的誠實臂 1.97% 獨立吻合；instrument ✅ BOUND：mm_pub 39/39、missmask_step=386、missmask_row=64）；B 臂（delivery-mmap）跑不完：command buffer 8 failed status 5 / Insufficient Memory、rc=−6、recovered 0 completed instance ⇒ 不是否證，是這台盒子量不到。
- （同條另掛：`L25-5`）
- **逐條處置**：整合進子目標　→ `L25-5`　—　「宣告與現判沒有對帳閘門」第三次：看板說 20+ 判死、無已命名路；targets.json 說『已判掉的三條路上界 ≈17–20』已被量測推翻（CGC_SUBMIT_AHEAD=1 ×1.702），且 prefill-250 在冷機發射 0 下已 6/6 ≥250（254–296）。本卡不量測，只做兩份權威檔的機械對帳；權威歸屬待 operator 簽。
- **測試 log**：無實跑 log（本條的證據是下面的「對應報告」，不是量測產物）
- ⚠ 本格的 `knobs.name` 有兩種：`CGC_*`＝env；`--*`＝**CLI 旗標**（`--spec-type` 不是 env，寫成 env 會被 allowlist 靜默丟掉 —— 那正是 MTP 修復臂空轉一趟的原因）。三趟：A 對照＝裸 `prod-new`（不帶任何 spec 旗標）、B＝A ＋ 上面那串、C＝B 重跑（驗 ON 自身 identical）。驗收三條：**① `need_n_rs_seq()==0` ② `E>1` ③ `answer_md5` 與 A 相同**。⛔ 三趟的 t/s 一律先過引用閘門；本格的第一個產物是「有沒有乘數」，不是速度。

## 6. 與其它條目的關係（同軸／同階段，自動對照）

| 條目 | 級 | 結果（摘） |
|---|---|---|
| [k=3 的 1.43× 飄移定位 ＋ 配對認證](k3-swing.md) | 3a | 飄移已定位；配對認證 n=5 時 t=2.08 未達 df=4 的 2.776 ⇒ 16.4% 仍未認證；上一輪「bench 配對 sd 更小」被自我撤回 |
| [口徑＋k=3 飄移認證（16.4% 噪聲底）](exp-caliber-calibration.md) | 3a | **未跑（有卡、無讀數）**：本節點的產出是**口徑校準**——記錄裡至少有四個不同的「步」（DECPROF work-row **167.81** ms／實付週期 step_ro |
| [2026-10-03 operator 裁定「移除所有具名格、pro…](exp-default-cell-first-2026-10-03.md) | 3a | 尚無可引用讀數：1 個 run 全部非乾淨（attribution.verdict=swap（swap_growth=3122.3099999999995 MiB, max_swa |
| [**跨 checkpoint 探針**：把 CGC 引擎（交付路徑＝…](exp-cgc-edge0gguf-2026-10-05.md) | 3a | 實驗進行中（尚無讀數） |
| [**交付口徑量測（生產級）**：spec-tree MTP-on（`…](exp-spectree-mtpon-default-2026-10-06.md) | 3a | decode 11.36 t/s |

## 7. 子目標分解（持續更新；1/5 完成）

- [x] 跑前立項（現狀/目標/假設/驗收） [證](../../../scripts/check/charters/exp-targets-board-recon-2026-10-07.yaml)
- [~] 用生產級腳本 prod-new ＋ 自己 option 跑實驗臂，留存 log
- [ ] 驗收：checker 存在並在既有三條矛盾上**如實 FAIL**（證明它會紅）， 且 operator 已指定權威來源 ⇒ 對帳後兩邊一致（不論是哪邊改）。
- [ ] 否證條件：checker 在已知矛盾上仍 PASS（沒抓到）⇒ 閘門無效，本卡結案為負。
- [ ] 寫一支只讀兩檔的對帳 checker，並用既有三條矛盾證明它會紅

## 8. 依據 · 備註 · 對應報告

| 項目 | 內容 |
|---|---|
| 依據 | `agent_harness/portal/targets.json` |
| 備註 | 假設：共同根因是**宣告層與機械判準層分離**：人寫的結論在 A 檔、閘門讀的判準在 B 檔， 中間沒有單向對帳 ⇒ 兩邊可以同時「正確」而互相矛盾，且沒有人會發現。 |
| 軸性質 | 非攻關軸：約束／帳本／上界算術／入口索引／跨線產品 |
| 對應報告 | 0 份 |

- （無）

---

← [**拉 a：chain 路徑開 rejection sampling…](exp-a-lever-rejection-2026-10-06.md)　·　[總目錄](index.md)　·　[HTML 版](exp-targets-board-recon-2026-10-07.html)　·　[LES 目標-router 預取移植（qwen35moe 同構 · G1–G5） →](e-les-router-prefetch-2026-10-17.md)

本檔由 `scripts/check/mindmap_brief_build.py` 從 `docs/mindmap/mindmap.json` 機械生成；改內容請改 JSON 後重跑，勿直接編輯本檔。
