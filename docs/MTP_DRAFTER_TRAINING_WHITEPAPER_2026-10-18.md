# MTP 轉正條件白皮書：α 歸因 → slime 訓練 → 雙用途驗收

> 日期 2026-10-18 · 載體 `Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf`（M4 Air 16GB）
> 觸發問題：能否用 CGC_Phase2 訓練框架或 slime 把 MTP head 接受率 α 拉過 0.73，讓 MTP 轉正？
> 定位：**條件性預案，不是即時開發**。主路徑仍是 LES G1–G6（lossless、本地、不依賴訓練）；
> MTP 訓練是門控側路，只有「零成本歸因探針」證明有訓練空間才啟動。
> 數據口徑聲明：本檔成本模型（10.72 t/s 基線、α=0.654）產自 2026-09-22 的 prod25/髒窗口；
> 現最高紀錄 C17=20.79（MTP off + fence）。MTP 損益在 20.79 基線下需重算，本檔的 α 門檻與
> 天花板結論是方向性判據，不直接當 20.79 下的量測值。

---

## 0. 先給結論（三個硬事實）

1. **技術上能把 α 拉過 0.73**：slime（智譜 THUDM，Apache-2.0，SGLang+Megatron）原生支持
   在線訓練 MTP 層（`--mtp-num-layers`，Qwen3/GLM 專用 bridge）；外部論文實證自蒸餾/微調可把
   per-token α 從 ~0.6 拉到 ~0.8（k=1 80%、k=2 56%、k=3 36%）。DeepSeek-V3 MTP-1 開箱 α 就 80%+。
2. **但過 0.73 對「decode 25」沒有意義**：
   - 0.731 只是 **k=3 的損平線**（α=0.73、k=3 也才 10.78 t/s，不如 k=1）。
   - α=0.8 最佳仍是 k=1（12.43 t/s）；k=3 要顯著賺需 **α≥0.9**（13.87）；
   - **α=1 完美 drafter、k→∞ 天花板 19.39 t/s**，砍光整個 kernel 也只有 23.15——
     **MTP 投機解碼在物理上永遠到不了 25**。
3. **我們 α=0.654 異常低（對比 DeepSeek 開箱 0.80+），根因還沒分離**：可能是 IQ3_XXS 2-bit
   量化、Nail UD 轉換、或端側金屬數值路徑（與 bit-identical M1 1/14 同源）。**不先歸因就訓練，
   很可能在雲端把 α 訓上 0.8、量化回 IQ3 又被吃回 0.65，白做 12–18 天**。

⇒ 正確路徑不是「直接開發訓練」，而是：**階段 0 零成本歸因（證明訓練有空間、且量化不吃收益）
→ 階段 1 才上 slime 雲端訓練**。且 MTP 有兩個用途，訓練目標不同（見 §1）。

---

## 1. 關鍵：兩個「α」不是同一個東西

| 維度 | α_tok：token 接受率（投機解碼出 token） | ρ_expert：專家集合 recall（e-union 預取信號） |
|---|---|---|
| 用途 | draft token 直接當輸出（verify 接受） | draft token **逐層跑 router**，專家 union 只驅動後台預取 |
| 對應代碼/卡 | `speculative.cpp` CGC Strata、`--spec-draft-n-max` | `CGC-DRAFT-PF`/`draft_prefetch_ids` + charter **`e-union-cache-shrink-2026-10-14`** |
| 是否要求 bit-exact | **是**（量化已破壞，M1 1/14、MTP 被釘 off） | **否**。預測只是 hint，最終選擇走真實 router，錯了只浪費預取，不影響輸出 |
| 精度要求 | 嚴格（token 逐字一致） | 寬鬆（token 錯 ≠ 專家錯；union 去重，集合 recall 通常遠高於 token α） |
| 門檻 | k=3 損平 α=0.731；k=3 顯著需 α≥0.9 | **待測**（從未量過；G6 縮池暫定 union recall ≥0.85） |
| 目標 | 加速（但天花板 19.39，定位錦上添花） | **物理常駐 <6GB（G6 縮池），不是加速** |
| 不訓練是否可能夠 | 否（α=0.654 淨虧） | **可能**（現有 MTP head 的 ρ_expert 從未測過，可能本來就夠） |

**最重要的判斷**：即使 α_tok 因量化永遠救不回來（bit 翻臉），MTP head **仍可作為純預取信號**
用於 e-union——這條路完全 lossless，且專家集合比 token 寬鬆。**因此階段 0 最該先測的是
ρ_expert（不訓練），而不是先訓練 α_tok。**

---

## 2. 現狀事實（內部判詞 + 外部基準，均可追溯）

### 2.1 內部成本模型（`docs/DRAFT_BUDGET_K_2026-09-22.md`）
```
cost(T) = 41.74 + 51.56 * T  ms   （T = k+1，verify 每 token 全額讀自己的專家、無批量折扣）
c_tok = 8.36（MoE GEMV，16.2%）+ 43.20（其餘 83.8%，SSD miss/非 kernel）
```

| k | T | cost ms | 損平所需 α | 現 α=0.654 實際 t/s |
|---|---|---|---|---|
| 1 | 2 | 144.86 | **0.553** | **11.42（1.066×，最佳）** |
| 2 | 3 | 196.42 | 0.664 | 10.60（0.989×） |
| 3 | 4 | 247.98 | **0.731** | 9.53（0.889×，淨虧） |
| 7 | 8 | 454.22 | 0.851 | — |

- 現 α=0.654 反解自同 harness ON/OFF=0.889@k=3；α≤0.8 最佳 k 恆為 1；α≥0.9 才輪到 k=3。
- α=1、k→∞ 天花板 19.39（1.81×）；要 25 需 c_tok≤40.00ms（砍 22%）。

### 2.2 內部已有「練 draft head」判詞（三份，均指向先別做）
- `SPEC_COST_CURVE_2026-09-18.md:150`：「訓練 draft head（Phase 2-3，12–18 天）｜同樣成本，
  收益被 m 鎖在 ~1.5× 以內｜**先別做，先壓 m**」。← 這就是「CGC_Phase2 訓練」一說的來源，
  它是一個排期項，**repo 裡並無名為 CGC_Phase2 的現成訓練框架**；`R6_PHASE2_REDESIGN` 是引擎
  手術（推理側），不是訓練。
- `MTP_ON_OPTIMIZATION_PLAN_2026-09-25.md:244`：方案 A「提高接受率/練 draft head」判
  **❌ 不做、不必排程**，理由 draft head forward 只佔邊際 token 的 12%、已在自己 kernel 地板。
- `STEP_SERIALIZATION_2026-09-23.md:154`：「別練 draft head（m 攤薄係數不利）」。

> 這三份判詞針對的是「α_tok 投機加速」。本白皮書新增的 ρ_expert 維度（預取/縮池）此前未被
> 單獨評估，因此本檔不是推翻原判，而是把「訓練是否值得」的判斷補齊到第二個用途上。

### 2.3 外部基準（2026 年）
- **slime 官方文件**：在線訓練 MTP 層以對抗 RL 過程中 draft/target 分佈漂移（正是投機負收益的
  機制），`--mtp-num-layers 1`；MTP 層含 enorm/hnorm/final_layernorm，已適配 Qwen3 與 GLM。
  https://thudm.github.io/slime/advanced/speculative-decoding.html
  https://deepwiki.com/THUDM/slime/11.1-supported-model-architectures
- 微調 MTP 接受率（arXiv 2509.18362）：fine-tuned 後 k=1 80% / k=2 56% / k=3 36%，
  數學類從 vanilla k=3 的 3% 拉到 >50%。
- 自蒸餾 MTP（arXiv 2603.23911）：4 heads acceptance +7.5%、22.9% 加速；looped 擴 16 heads +35.1%。
- MTP-RL（ACL 2026 findings 1871）：SGLang 自蒸餾生成軌跡 + **Swift 從頭訓 MTP** + slime 跑 RL。
- DeepSeek-V3 MTP-1：開箱接受率 >80%（技術報告），約 1.8× 吞吐。
- 訓練基建：slime = Megatron 訓練 + SGLang rollout，fp8 rollout + DeepEP，**多卡雲端框架**。

---

## 3. α 為什麼只有 0.654：三個候選根因（階段 0 要分離）

| 根因 | 機制 | 若成立的對策 | 訓練是否有用 |
|---|---|---|---|
| **R1 量化** | IQ3_XXS 專家 2-bit、dense IQ4X，低比特扭曲 router logits/draft 數值；訓練後再量化會再吃一次 | 提高 drafter/MTP 量化檔位（IQ4_XS，task#22 已在做）、MTP head 單獨高檔保留 | 無用（先解量化） |
| **R2 Nail UD 轉換/head 質量** | UD 合併/轉換損傷 MTP head，或 head 本就針對別的分佈 | 自蒸餾/在線 SFT 重訓 MTP head | **有用** |
| **R3 端側 verify 數值路徑** | metal kernel/量化實現與標準框架不一致（與 bit-identical M1 1/14 同源） | 修後端數值一致性；或放棄 α_tok、只用 ρ_expert | α_tok 無用、ρ_expert 不受影響 |

---

## 4. 階段 0：零訓練歸因 + ρ_expert 探針（便宜，決定要不要訓練）

> 全部 harness 唯一入口、warm-skip 64、seed 1、三套 prompt；α 是質量指標，**不依賴速度**，
> 可在慢 offload / 雲端測，因此不受端側 16GB 限制。

- **0A 量化橫評（本地，可與 task#22 IQ4_XS 合併）**
  同 prompt 量 α_tok：IQ3_XXS（現 0.654）→ IQ4_XS（task#22 待跑）→ Q4_0 Edge0（本地有
  `Edge0-35B-Q4_0-MTP.gguf`，可慢跑只取接受率）。畫 α 對比特數曲線。
  *門檻*：若 α 隨比特數明顯回升（如 IQ4_XS/Q4_0 ≥0.75）⇒ R1 量化主因。
- **0B 雲端未量化基線（需雲資源）**
  用 SGLang/vLLM 標準 spec decode 跑未量化 Nail 基座（BF16/FP8，35B 約 70GB，16GB 端側放不下），
  得 α_tok 上限。
  *門檻*：BF16 α≥0.8 而 IQ3 0.654 ⇒ R1 確認；BF16 α 也 <0.7 ⇒ R2 訓練問題。
- **0C ρ_expert 探針（最關鍵、不訓練、可複用 G1 設施）**
  對 MTP 每個 draft token 逐層跑 router（離線，`CGC-DRAFT-PF` 已有雛形），把 N 個 draft × 每層
  專家集合做 union，與真實路由對拍，算 recall@d 與 union 去重增益（複用 G1 sidecar/A1 對賬設施）。
  *門檻*：
  - 不訓練 union recall ≥0.85 ⇒ **e-union 直接可上，無需訓練**（進 G6 疊加）；
  - recall 偏低但專家集合顯著寬鬆於 token α ⇒ 值得用 routing-aware loss 訓練（階段 1 方法③）；
  - recall 與 token α 一樣低 ⇒ MTP 預取線與 α_tok 同根因，等歸因結論。

**階段 0 決策表**

| 0B 結果 | 0A 結果 | 0C 結果 | 結論/動作 |
|---|---|---|---|
| BF16 α≥0.8 | IQ3 低、高比特回升 | — | R1 量化主因 → **不訓練**，先上 IQ4_XS/MTP 高檔保留，重測 α |
| BF16 α<0.7 | 各檔都低 | — | R2 訓練問題 → 進階段 1（α_tok 與 ρ_expert 一起訓） |
| 各檔 α_tok 都差 | — | ρ_expert≥0.85 | α_tok 放棄（R3 或不經濟），**MTP 僅作預取信號**，e-union 進 G6 |
| 各檔 α_tok 都差 | — | ρ_expert 低但寬鬆 | 階段 1 只訓 routing-aware head（不追求 α_tok/bit-exact） |

---

## 5. 階段 1：slime 雲端訓練（僅階段 0 開門）

- **工具**：slime（Apache-2.0，Megatron + SGLang，原生 MTP bridge，已支持 Qwen3/GLM）。
- **三種方法（按成本排序）**：
  1. **自蒸餾 MTP-D**：主模型生成軌跡 → 用 Swift/Megatron 從頭或繼續訓 MTP 層（ACL 2026 流程）；
  2. **在線 SFT during RL**：slime 原生，draft/target 同步防漂移（`--mtp-num-layers`）；
  3. **routing-aware 輔助 loss**：在 MTP loss 上加「draft token 逐層 router 專家集合」對齊項
     （直接優化 ρ_expert，服務 e-union；屬自研，需在 slime 加 reward/loss 掛鉤）。
- **資源**：雲端多卡（16×L20N 72GB / MI300X 等），M4 Air 只做端側部署與驗收；本地不訓 35B。
- **訓練後必須做的事**：把訓好的 head 量化回 IQ3_XXS / IQ4_XS，**重測 α_tok 與 ρ_expert 的淨值**
  （扣除量化損失），不可用雲端 BF16 數字直接外推端側。
- **訓練目標（不要把目標訂成 0.73）**：
  - α_tok 路：衝 α≥0.9（k=3 才有 13.87 量級），但天花板 19.39，預期淨收益 **+10~30%**
    （10.7→12–14，10.72 基線口徑；20.79 基線需重算），定位錦上添花；
  - ρ_expert 路：union recall ≥0.85，目標物理常駐 **<6GB（G6）**，這是訓練主要價值所在。

---

## 6. 階段 2：雙用途驗收（harness、env-gated、預設 OFF）

- **α_tok 臂（投機解碼）**：MTP on k=1/k=3，warm-skip64、seed1、三套 prompt，記 α_tok 與 t/s 配對；
  過 M1/M2/M3 bit-identical（222/222 口徑）。**若量化後仍 bit 翻臉，α_tok 不交付**，MTP 降級為
  純預取信號（這與 MTP_TRANSFERRED 結論一致）。
- **ρ_expert 臂（e-union 縮池）**：沿用 G6 矩陣 `LES/MTP-union on/off × 池{8G,6G,4G}`，每池一格；
  驗收 `t/s(on,6G) ≥ t/s(off,8G) × 0.95` 且物理峰值降 ≥1.5 GiB；
  LES 與 MTP-union 分別測、再測疊加（兩信號正交，見 LES charter §與既有預取線的關係）。
- 全部開關預設 OFF，出廠輸出與 MTP off 逐位一致。

---

## 7. 門控順序（與 LES 的關係，不可調換）

```
主路徑（lossless、本地、不依賴訓練）：
  G1 離線命中率 PASS（82%）→ G2 圖內 argsort（補丁已備，待 commit2 落地）
    → G3 preadv 重疊（攻 43.20ms SSD miss）→ G4 配對 ≥+5%
    → G5 MTP 共存/降級（n_tokens>1 不預測、兩檔 bit-exact）
    → G6 LES × 8/6/4G 縮池（權威基線，6G 無一手須重測）

MTP 訓練側路（條件並行，不搶 LES 的 GPU 窗口）：
  階段 0（0A 可與 IQ4_XS task#22 合併；0C 複用 G1 設施，排 LES 窗口後；0B 需雲資源）
    ├─ 開門才進階段 1 slime 雲端訓練
    └─ 不開門則關閉 α_tok，僅視 0C 結果決定 e-union 是否無訓練上線
```

- **decode 25 的槓桿仍是 LES（G3 砍 43.20ms 非 kernel）+ expert kernel 利用率（38–45%）**，
  不是 MTP（數學封頂 19.39）。
- MTP 訓練的最高性價比出口是 **ρ_expert 幫 G6 縮池（<6GB）**，其次才是投機解碼的 10~30% 加速。

---

## 8. 誠實風險

1. **量化歸零**：雲端訓練 α 上 0.8，量化回 IQ3 後可能跌回 0.65——階段 0 就是為了在燒訓練錢前排除。
2. **工期/資源**：slime 訓練 12–18 天起，需雲端多卡與未量化基座；端側無法本地完成。
3. **天花板硬**：α_tok 即便完美也到不了 25，只能錦上添花。
4. **bit-identical**：量化金屬路徑若無法修復，α_tok 路永久不交付，MTP 只能走預取信號。
5. 本檔成本模型來自 10.72 基線；20.79 基線下 MTP 損益與損平 α 必須在 LES 落地後重算，
   屆時非 kernel 成本下降會改變 k=1/k=3 的賠率。

---

## 9. 立即可做（不訓練、不搶 LES 窗口）

1. **0A**：task#22 等權 IQ4_XS ABBA 時，**順手記 α_tok**（量化橫評，零額外 GPU）。
2. **0C**：等 LES G2 落地後，基於同一 sidecar/A1 設施加一個 MTP draft union recall 離線腳本
   （不動引擎、一次 GPU），回答「不訓練的 ρ_expert 夠不夠 G6」。
3. **0B**：需要你提供雲端多卡與未量化 Nail 基座路徑（16×L20N / MI300X），我給 slime 跑 α 上限的命令。
4. 本檔在 mindmap 掛一個「MTP 轉正（條件性）」節點，**blocked-by 階段 0 歸因**，掛在 G5/e-union 下，
   不進當前活躍實驗（避免重蹈「無現狀、無目標、無驗收」的亂實驗）。

---

## 10. 一句話

**能把 α 訓過 0.73，但那條路又貴（雲端 12–18 天）、又被量化不確定性卡、又到不了 25；
最划算的動作是先做零成本歸因——尤其測一下不訓練的 MTP 專家集合 recall（ρ_expert），
它可能直接讓 e-union 縮池成立，而主線 LES 不等人。**
