# MTP 時間維 union（Ember 完整耦合）離線判詞 —— 0C-1 oracle + 0A α 基線

日期：2026-10-18　狀態：**0C-1 離線上界已完成（三套 prompt 一致）；0A 抽取工具就緒 + 舊模型 α 一手重抽**
關聯：白皮書 `docs/MTP_DRAFTER_TRAINING_WHITEPAPER_2026-10-18.md`、charter `e-union-cache-shrink-2026-10-14`、
`docs/MTP3_25TPS_ARITHMETIC_2026-09-18.md`、LES charter `e-les-router-prefetch-2026-10-17`。
工具：`docs/mtp_union/tools/mtp_union_recall.py`（離線 oracle，純 stdlib，~6s/套）、
`docs/mtp_union/tools/extract_mtp_acceptance.py`（α 抽取）。數據：`Backup/mtp_union_oracle_20261018/`。

---

## 一句話

**Ember 完整耦合（N draft 逐層 batch-router + union 聚合 + verify 前集合裁剪）工程上做得到，而且引擎已經在做**
（MTP verify fast path 就是 union 聚合，實測 verify 一步 union=19.05）；但它**對 decode 是結構性為負**
（每 token 多讀 11% 專家、長期 capacity miss ×10.2、verify 步 1.97×，淨增益約 1.09×），
對**縮池到 6G** 在真實 k=3 視野下也救不回（命中率仍差 8G 約 2.4pp）。
**它不是 decode 25 的槓桿**；唯一可能正面的「預取時序」與 LES G3 是同一批 fill IO、且 LES 信號更強
（G1 跨層 82% vs MTP 跨 token α=0.43）。結論：**不為 Ember union 投力，主線維持 LES**。

---

## 1. 0C-1 oracle 上界（假設預測全對，ρ_expert=1）

數據：G1 真實路由 `Backup/les_g1_hitrate_20261017/a1raw_{code,zh,math}.stderr.log`，
每 decode step 每層真實 top-8（641 pass × 40 層，warm-skip64 後 577 步）。純離線，不訓練、不動引擎、不需 GPU。

### 1.1 穩態工作集（577 步動員的 distinct 專家/層，256 上限）

| prompt | mean | min | max |
|---|---|---|---|
| code | 175 | 119 | 244 |
| zh | 176 | 128 | 241 |
| math | 172 | 122 | 250 |

每層 8G 池 143 槽 ≈ 覆蓋 81% 工作集；**6G 107 槽 ≈ 61%，4G 72 槽 ≈ 41%**（按池位元組等比換算，真實 LAYER_CAPS 以 G6 重測為準）。

### 1.2 union 擴大曲線（覆蓋「當前+未來 N 步」需多少專家/層，三套均值）

| N（draft 數） | 0 | 1 | 2 | 3 | 4 | 7 | 11 | 15 |
|---|---|---|---|---|---|---|---|---|
| union 專家數 | 8 | 12.8 | 16.7 | **20.1** | 23.2 | 30.9 | 39.3 | 46.3 |
| /k 倍數 | 1.0 | 1.6 | 2.1 | **2.5** | 2.9 | 3.9 | 4.9 | 5.8 |

> 即使 N=15，union 46 也遠小於 6G 池的 107 槽 ⇒ **容量放得下，瓶頸不是容量，是預測準度與時機**。

### 1.3 單點 overlap（不 union，只預測單一未來步；|S(p)∩S(p+n)|/k）

| n | 1 | 2 | 3 | 4 | 7 | 11 | 15 |
|---|---|---|---|---|---|---|---|
| overlap | 0.40 | 0.38 | 0.38 | 0.37 | 0.37 | 0.36 | 0.36 |

> 相鄰步只共享 40% 專家，n≥2 後約 0.36 且基本不隨距離衰減（≈隨機重疊）。**跨 token 單步專家相關性很弱**；
> 這正是 CGC-PREV-TOKEN-PF 註解「70-90% overlap」與實測不符的根因（以 G1/0C 實測 0.40 為準）。

### 1.4 命中率網格（列=預測視野 N，行=每層 cap；N=0 LRU 現狀，N 有限=預取保護 LRU，∞=完美 oracle）

| cap | 池 | LRU(N=0) | N=3 | N=7 | N=15 | oracle(N=∞) | Δ(oracle−LRU) |
|---|---|---|---|---|---|---|---|
| 143 | 8G | .949–.951 | .950–.952 | .950–.952 | .952–.953 | .960–.961 | **+1.0pp** |
| 107 | 6G | .920–.924 | .923–.927 | .926–.929 | .930–.933 | **.952–.953** | **+3.0pp** |
| 72 | 4G | .864–.867 | .873–.876 | .880–.884 | .892–.896 | .927–.929 | +6.3pp |

三個硬結論：

1. **8G 已飽和**：就算預測完美，命中率也只 +1.0pp。MTP union 在 8G 幾乎無空間。
2. **6G 只有「完美長視野」才勉強追平 8G**：6G oracle .952 比 8G 現狀 .949 高 +0.2~0.3pp；
   但現實 k=3（N=3）只到 .923–.927，仍差 8G 約 **−2.4pp**；N=15 才 .930–.933（僅拿到 oracle 差距的約 30%）。
   **集合裁剪的收益主要在長視野（N>7），而 k=3 給不了**。
3. **4G 即使完美也追不平**：oracle .927 仍比 8G 現狀低 2.2pp。

> 模型説明：有限視野用「預取保護 LRU」（union 提前載入、替換時保護未來 N 步視窗、不對視野外短視誤殺），
> 對 N 單調不降低；它對應 Ember 的「集合裁剪指導替換」。早期短視 Belady 會在 N=1 反而低於 LRU（誤殺視窗外即將復用者），
> 那正是「MTP 預測短、卻激進信任它」會造成的 cache churn，已改用不短視的版本。

---

## 2. 0C-1 與真實引擎實測交叉驗證（`MTP3_25TPS_ARITHMETIC_2026-09-18.md`，2026-09-18 server log 一手）

| 量 | 實測 | 與 0C-1 對照 |
|---|---|---|
| verify 一步 union（4 draft 全算） | **19.05 distinct expert**（9028/474） | 0C-1 oracle N=3 = 20.1，差 5% ⇒ **離線模型可信** |
| 單 token 一步 | 8.00 expert（288/36） | 0C-1 N=0 = 8 ✓ |
| 每 token 實際讀取 | **19.05/2.14 = 8.90**（比單步 8.0 **多 11%**） | union 沒省讀取，反而更費 |
| 長期 capacity miss | 218 → 2229（**×10.2**） | union 擴大把 cache 洗更髒（churn 加劇） |
| 命中率 | 96.0% → 87.1%；超額層 3 → 18 | 同上 |
| verify 步時 / 單步時 | 168.4ms / 85.65ms = **1.97×** | 實出 mean_len 2.14 |
| 淨增益 | 2.14 / 1.97 = **1.09×**（12.4–12.7 t/s） | 與白皮書 0.933× 同方向（都遠低於賠率） |
| α=1 天花板 | 4.0 / 0.1684 = **23.75 t/s < 25** | 與白皮書 19.39 一致封頂 |

**機制矛盾（verify 前集合裁剪為何無效）**：verify 必須先把 4 個 draft 對應的 union 全算過，才能確定接受哪個 token；
「哪些專家真的被走」這個資訊在 verify **之後**才存在。所以「verify 前集合裁剪」在物理上**無法減少 verify 讀取**，
只能決定「verify 完之後 cache 留什麼」——而 0C-1 顯示 k=3 那點未來資訊只夠 +0.3pp 命中，實測長期還淨負（×10.2 miss）。

---

## 3. 0A：現模型 α_tok 一手重抽（抽取工具已驗證）

從 2026-09-18 全天 server log（`Backup/cgc_logs/llama_server_20260918_*.log`）以
**「每 server 實例取最終累計、跨實例 pooled」**正確口徑重抽（`print_timing` 同 log 多 task 是累計值，不可逐行相加）：

| 樣本門檻 | 實例數 | pooled α | median α | mean_len 中位 |
|---|---|---|---|---|
| generated≥20 | 39 | 0.428 | 0.429 | 2.20 |
| generated≥40 | 20 | 0.429 | 0.438 | 2.25 |
| generated≥60 | 17 | **0.430** | 0.442 | 2.30 |
| generated≥100 | 15 | 0.420 | 0.442 | 2.31 |

大樣本實例 α 區間 **0.36–0.47，全部 < 0.553（k=1 break-even）** ⇒ **連最短 k=1 都不賺錢**。
白皮書轉引的 0.654 是另一口徑；一手 server log 更差。這對「α 低是不是量化造成（R1）」是更強的起點。

> 0A 剩餘步驟（需 GPU，等 task#22）：`models/imatrix/` 校準語料已備齊（train_0/1.parquet 各 150M、wiki 516M），
> 但尚無 imatrix.dat 與 IQ4_XS gguf。量化出 IQ4_XS 後，用 prod-new harness（warm-skip64、seed1、CGC_SERVER_MTP_N_MAX=3）
> 同場跑，用 `extract_mtp_acceptance.py` 抽 α，與 IQ3_XXS 的 0.43 對比，畫 α–位元數曲線判 R1。

---

## 4. Ember 三個動作逐個裁

| 動作 | 做得到嗎 | 證據判決 |
|---|---|---|
| N draft 逐層 batch-router | 做得到（draft 前向已有；CGC-DRAFT-PF 目前只覆蓋最後一層，需補逐層） | 只產出預測集合，準度受 α=0.43 限；專家集合級 recall 較寬鬆，但 0C-1 顯示 k=3 天花板低 |
| Union 聚合 | **已在做**（verify fast path remap 一次寫滿 union） | 實測 union 19.05、每 token 多讀 11%、capacity miss ×10.2 ⇒ **負債** |
| verify 前集合裁剪 | 邏輯可寫，但**物理上不能減 verify 讀取**（見 §2 矛盾） | 對單步延遲無幫助；對長期駐留 k=3 只 +0.3pp，實測淨負 |

---

## 5. 速度提升量級（誠實，不承諾）

- **decode 加速：期望貢獻 ≈ 0（實測為負）**。union/裁剪不動算力，反增讀取（+11%/token）與 1.97× verify 步；
  α=1 也只 23.75 t/s，MTP 投機軸數學封頂 19.39（白皮書成本模型）。
- **縮池到 6G：k=3 現實視野不成立**（差 8G 2.4pp）；只有又深又準的長視野才追平，而那要求 k>7、α>0.85，與 α=0.43 矛盾。
- **唯一可能正面的「預取時序」不是 union 的獨立收益**：把 cold 專家提前 load、藏進當前層 compute，
  正是 LES G3 preadv worker 的機制、同一批 fill IO；且 LES 用當前步跨層確定性信號（G1 82%），
  MTP 用跨 token 概率信號（α=0.43、N 短），**同一件事 LES 更值得做**。

---

## 6. 0C-2（真實 ρ_expert）的處置

唯一能給「訓練後真實專家集合 recall」的量：在 verify forward dump 每個 draft token 逐層真實 top-k，
算 ρ_expert（比 token α 寬鬆），再與 oracle 上界相乘。**但需 GPU，且按 §1/§2 先驗極低**
（ρ_expert=1 時 k=3 也只 +0.3pp；α 訓到 0.8 也只把 k=1 做到約 12.4 t/s）。
**決議：0C-2 不單獨開 GPU，等 LES G3/G6 乾淨窗口順手掛；不為它排隊。**

---

## 7. 回饋與排序

- **charter `e-union-cache-shrink-2026-10-14`：A1 判否**（oracle 上界即顯示 k=3 縮池不成立、實測 union 為負），
  不進 A2/A3，建議標記為 closed/rejected（附本判詞）。
- MTP 維持 LES charter **G5**：只驗證共存、n_tokens>1 安全降級、兩檔 bit-exact，不調加速。
- **主線不變**：LES G2（圖內 argsort，補丁已就緒待 commit2）→ G3（preadv worker，直接攻 43.20ms 非 kernel 裡的 SSD miss）
  → G6（8/6/4G 一手 ABBA，權威回答縮池）。
- 0A：抽取工具就緒，阻塞在 task#22 IQ4_XS 模型（需乾淨 GPU 窗口）。
- mindmap 節點：與白皮書一致，等 commit2 落地後一併掛（標 blocked-by 階段0），避免污染當前工作區。

## 8. 復現清單

```
# 0C-1 oracle（每套 ~6s，純離線）
python3 docs/mtp_union/tools/mtp_union_recall.py \
  --a1-log Backup/les_g1_hitrate_20261017/a1raw_code.stderr.log \
  --warm-skip 64 --label code --json Backup/mtp_union_oracle_20261018/mtp_union_code.json
# （zh/math 同法）

# 0A α 抽取
python3 docs/mtp_union/tools/extract_mtp_acceptance.py Backup/cgc_logs/llama_server_20260918_*.log
```
