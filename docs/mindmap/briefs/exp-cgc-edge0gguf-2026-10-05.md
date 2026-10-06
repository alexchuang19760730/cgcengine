# **跨 checkpoint 探針**：把 CGC 引擎（交付路徑＝… — 技術白皮書　·　3a ③a 實驗目標達成（階段性）

> **一句話**：**跨 checkpoint 探針**：把 CGC 引擎（交付路徑＝prod-new 格／8 GiB 池／load_mode none／無 MTP） 指到另一顆**同 base 的容器** models/gguf/Edge0-35B-Q4_0.gguf，它跑多少？ 為什麼值得問：現有全部交付錨點（tg 13.722 t/s 柵欄 ON／10.838 OFF）都長在 Nail （IQ3_XXS, 12.72 GiB）上。這顆是 18.19 GiB（x1.43 bytes）、top-8、**無 nextn head** （733 tensors，標頭已解析）。同 base（Qwen3.5 MoE 35B A3B、40 層、256 專家），不同容器與量化。 兩個機制指向相反： (1) flashkv 側 IO 已被池／async 藏在計算後（wait 是真的、IO 不在關鍵路徑）⇒ 更大的檔未必痛； (2) 同一顆 8 GiB 池在 Q4_0 上每個 bundle 更大 ⇒ 槽數更少、命中率更低、streaming 更多。 ⇒ **非量不可。** 口徑警告（必須與結論同時引用）：測試卡契約的頂層 `model:` 是 Nail，而 `check_cell()` 只比對 `cell` 區塊的維度（前向別名表裡沒有 -m）⇒ **換 checkpoint 過得了閘門、不會 fail-closed**。 這是已知的口徑缺口：本產物的 tg **不可直接與 13.722／10.838 做同模型比較**。

- 主題：實驗　·　子目標：**不適用**（非攻關軸：約束／帳本／上界算術／入口索引／跨線產品）
- 階段：實驗階段（階段性）　·　③a —— 機制／量測成立，但產物還不能進生產

---

## 1. 目標

**跨 checkpoint 探針**：把 CGC 引擎（交付路徑＝prod-new 格／8 GiB 池／load_mode none／無 MTP） 指到另一顆**同 base 的容器** models/gguf/Edge0-35B-Q4_0.gguf，它跑多少？ 為什麼值得問：現有全部交付錨點（tg 13.722 t/s 柵欄 ON／10.838 OFF）都長在 Nail （IQ3_XXS, 12.72 GiB）上。這顆是 18.19 GiB（x1.43 bytes）、top-8、**無 nextn head** （733 tensors，標頭已解析）。同 base（Qwen3.5 MoE 35B A3B、40 層、256 專家），不同容器與量化。 兩個機制指向相反： (1) flashkv 側 IO 已被池／async 藏在計算後（wait 是真的、IO 不在關鍵路徑）⇒ 更大的檔未必痛； (2) 同一顆 8 GiB 池在 Q4_0 上每個 bundle 更大 ⇒ 槽數更少、命中率更低、streaming 更多。 ⇒ **非量不可。** 口徑警告（必須與結論同時引用）：測試卡契約的頂層 `model:` 是 Nail，而 `check_cell()` 只比對 `cell` 區塊的維度（前向別名表裡沒有 -m）⇒ **換 checkpoint 過得了閘門、不會 fail-closed**。 這是已知的口徑缺口：本產物的 tg **不可直接與 13.722／10.838 做同模型比較**。

## 2. 判準

以統一入口（harness bench ＋ prod-new 格、單臂）給出 tg128 與 pp512，並標明不可與 Nail 錨點同模型比較

## 3. 結果

實驗進行中（尚無讀數）

## 4. 判定

**3a · ③a 實驗目標達成（階段性）** — 達成實驗設計目的（機制／量測成立），但產物還不能放進生產級設置

> 假設：檔大 1.43x 且無 MTP head ⇒ 每步搬的位元組更多、沒有投機攤薄。 若 IO 已完全隱藏 ⇒ 差距小（<15%）；若池命中率因 bundle 變大而降 ⇒ 差距大。

---

## 5. 子目標綁定 · options · 測試 Log 報告

- **子目標**：—（未歸屬看板 15 格中的任何一格）
- ⚠ 本條不屬於看板 15 個子目標中的任何一格（已認證／已定案／已作廢）⇒ 沒有待跑的臂，也就沒有要綁的 option。
- **逐條處置**：已定案（約束／基準）　—　不適用軸、無樣本（2026-10-05）：跨 checkpoint 探針 —— 把 CGC 引擎（prod-new 交付格、8 GiB 池、無 MTP）指到 models/gguf/Edge0-35B-Q4_0.gguf（同 base Qwen3.5-MoE 35B A3B 的另一個容器，18.19 GiB、top-8、無 nextn head）。① 交付口徑（load_mode none）：host 靜態需求 26,822 MiB（model resident 18,630 ＋ pool 8,192）vs 實體 16,384 ⇒ 超訂 10,438 MiB ⇒ budget gate 拒跑（MEASUREMENT_CONTRACT §3.4.1，rc=2）。② mmap：host 預檢過關（resident 0 ⇒ 8,192 MiB），但直呼 llama-bench 的診斷跑連第一個 prefill 都沒過：Insufficient Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory) ＋ CGC-METAL-FAIL，死在 il=39、RSS ≈ 9,127 MiB。⇒ mmap 讓 host 過關、卻讓 Metal 工作集爆掉（＝ L25-1 的反證在另一模型上第二次被量到）。③ 縮池＋縮 -ub（mmap ＋ pool 4 GiB（60 槽）＋ -ub 256）：同一種失敗、同一個爆點（kIOGPUCommandBufferCallbackErrorOutOfMemory／compute #41，死於 prefill 最後一層 il=39），RSS 僅 5,020 MiB ⇒ 池與 -ub 都不是綁定約束（各砍半、爆點不動）。⇒ 判詞：這顆 checkpoint（18.19 GiB）在 16 GB／M4（GPU 預算 11.45 GB）上沒有可跑的形狀 —— none 死在 host 算術（模型 18,630 ＞ 16,384，池再小都救不了）、mmap 死在 GPU，且 GPU 那條對 pool／ubatch 不敏感。⇒「CGC × 這顆 checkpoint 能跑多少速度」在此硬體上不存在答案；剩下只有換盒、或換 ≤ 約 Nail 尺寸（12.72 GiB）的 checkpoint。★ 儀器缺口（已驗）：budget gate 對 CGC_SERVER_MODEL 是盲的 —— llama_bench_matrix.py:1630 只用 resolve()["env"]，而該 env 既無 MODEL 也無 CGC_SERVER_MODEL（模型只進 argv 的 -m）⇒ budget_gate.sh:12 落到硬編碼 Nail 預設（本趟報 model 13030 ＝ Nail 13,030.1，真值 18,630.6）⇒ 換更大的 checkpoint 會低估超訂量（危險方向：可能放行其實超訂的臂）。charter scripts/check/charters/exp-cgc-edge0gguf-2026-10-05.yaml；產物 Backup/spec_tree/cgc_edge0gguf_2026-10-05/mmap_bench.{json,log} ＋ mmap4g_bench.{json,log}（兩份 json 皆 []）
- **測試 log**：無實跑 log（本條的證據是下面的「對應報告」，不是量測產物）
- 不適用軸、無樣本（2026-10-05）：跨 checkpoint 探針 —— 把 CGC 引擎（prod-new 交付格、8 GiB 池、無 MTP）指到 models/gguf/Edge0-35B-Q4_0.gguf（同 base Qwen3.5-MoE 35B A3B 的另一個容器，18.19 GiB、top-8、無 nextn head）。① 交付口徑（load_mode none）：host 靜態需求 26,822 MiB（model resident 18,630 ＋ pool 8,192）vs 實體 16,384 ⇒ 超訂 10,438 MiB ⇒ budget gate 拒跑（MEASUREMENT_CONTRACT §3.4.1，rc=2）。② mmap：host 預檢過關（resident 0 ⇒ 8,192 MiB），但直呼 llama-bench 的診斷跑連第一個 prefill 都沒過：Insufficient Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory) ＋ CGC-METAL-FAIL，死在 il=39、RSS ≈ 9,127 MiB。⇒ mmap 讓 host 過關、卻讓 Metal 工作集爆掉（＝ L25-1 的反證在另一模型上第二次被量到）。③ 縮池＋縮 -ub（mmap ＋ pool 4 GiB（60 槽）＋ -ub 256）：同一種失敗、同一個爆點（kIOGPUCommandBufferCallbackErrorOutOfMemory／compute #41，死於 prefill 最後一層 il=39），RSS 僅 5,020 MiB ⇒ 池與 -ub 都不是綁定約束（各砍半、爆點不動）。⇒ 判詞：這顆 checkpoint（18.19 GiB）在 16 GB／M4（GPU 預算 11.45 GB）上沒有可跑的形狀 —— none 死在 host 算術（模型 18,630 ＞ 16,384，池再小都救不了）、mmap 死在 GPU，且 GPU 那條對 pool／ubatch 不敏感。⇒「CGC × 這顆 checkpoint 能跑多少速度」在此硬體上不存在答案；剩下只有換盒、或換 ≤ 約 Nail 尺寸（12.72 GiB）的 checkpoint。★ 儀器缺口（已驗）：budget gate 對 CGC_SERVER_MODEL 是盲的 —— llama_bench_matrix.py:1630 只用 resolve()["env"]，而該 env 既無 MODEL 也無 CGC_SERVER_MODEL（模型只進 argv 的 -m）⇒ budget_gate.sh:12 落到硬編碼 Nail 預設（本趟報 model 13030 ＝ Nail 13,030.1，真值 18,630.6）⇒ 換更大的 checkpoint 會低估超訂量（危險方向：可能放行其實超訂的臂）。charter scripts/check/charters/exp-cgc-edge0gguf-2026-10-05.yaml；產物 Backup/spec_tree/cgc_edge0gguf_2026-10-05/mmap_bench.{json,log} ＋ mmap4g_bench.{json,log}（兩份 json 皆 []）

## 6. 與其它條目的關係（同軸／同階段，自動對照）

| 條目 | 級 | 結果（摘） |
|---|---|---|
| [k=3 的 1.43× 飄移定位 ＋ 配對認證](k3-swing.md) | 3a | 飄移已定位；配對認證 n=5 時 t=2.08 未達 df=4 的 2.776 ⇒ 16.4% 仍未認證；上一輪「bench 配對 sd 更小」被自我撤回 |
| [口徑＋k=3 飄移認證（16.4% 噪聲底）](exp-caliber-calibration.md) | 3a | **未跑（有卡、無讀數）**：本節點的產出是**口徑校準**——記錄裡至少有四個不同的「步」（DECPROF work-row **167.81** ms／實付週期 step_ro |
| [2026-10-03 operator 裁定「移除所有具名格、pro…](exp-default-cell-first-2026-10-03.md) | 3a | 尚無可引用讀數：1 個 run 全部非乾淨（attribution.verdict=swap（swap_growth=3122.3099999999995 MiB, max_swa |
| [**交付口徑量測（生產級）**：spec-tree MTP-on（`…](exp-spectree-mtpon-default-2026-10-06.md) | 3a | decode 11.36 t/s |

## 7. 子目標分解（持續更新；1/5 完成）

- [x] 跑前立項（現狀/目標/假設/驗收） [證](../../../scripts/check/charters/exp-cgc-edge0gguf-2026-10-05.yaml)
- [~] 用生產級腳本 prod-new ＋ 自己 option 跑實驗臂，留存 log
- [ ] 驗收：以統一入口（harness bench ＋ prod-new 格、單臂）給出 tg128 與 pp512，並標明不可與 Nail 錨點同模型比較
- [ ] 否證條件：若 tg128 >= 13.722 ⇒ 否證機制(2)：池對 bundle 大小不敏感、IO 已完全隱藏
- [ ] 量 CGC 引擎在 Edge0-35B-Q4_0.gguf 上的 tg128／pp512（prod-new 格、單臂）

## 8. 依據 · 備註 · 對應報告

| 項目 | 內容 |
|---|---|
| 依據 | `Backup/spec_tree/FENCE_DELIVERY_2026-10-05.md` |
| 備註 | 假設：檔大 1.43x 且無 MTP head ⇒ 每步搬的位元組更多、沒有投機攤薄。 若 IO 已完全隱藏 ⇒ 差距小（<15%）；若池命中率因 bundle 變大而降 ⇒ 差距大。 |
| 軸性質 | 非攻關軸：約束／帳本／上界算術／入口索引／跨線產品 |
| 對應報告 | 0 份 |

- （無）

---

← [同一趟（同 build、同格、同 session）裡： ① caps…](e-victim-classify-2026-10-05.md)　·　[總目錄](index.md)　·　[HTML 版](exp-cgc-edge0gguf-2026-10-05.html)　·　[**交付口徑量測（生產級）**：spec-tree MTP-on（`… →](exp-spectree-mtpon-default-2026-10-06.md)

本檔由 `scripts/check/mindmap_brief_build.py` 從 `docs/mindmap/mindmap.json` 機械生成；改內容請改 JSON 後重跑，勿直接編輯本檔。
