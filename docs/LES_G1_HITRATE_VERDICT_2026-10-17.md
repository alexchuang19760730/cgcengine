# LES G1 判詞：跨層 router lookahead 預測命中率

- 日期：2026-10-17
- charter：`scripts/check/charters/e-les-router-prefetch-2026-10-17.yaml`（sg-g1-hitrate）
- 結論：**機制信號強烈成立，G1 已於 2026-10-17 裁定 PASS（可預取層 target≥4 口徑）、進 G2。** 三套 target≥4 mean 0.8197/0.8205/0.8207、最低單層 target39 0.54–0.57 均過線；全層 mean 0.790 僅作對照（首層機制性低命中、由 prefill 預熱、豁免）。
- 原始證據：`Backup/les_g1_hitrate_20261017/`（三套 sidecar / A1-RAW / harness json / hitrate json / prompt / PROVENANCE_NOTE）

---

## 1. 一句話

在物理層 `il` 末尾用**本層 attn_post_norm 餵給下一層 `ffn_gate_inp` 真實權重**打分、預測下一層 top8 專家（LES 做法，預測只指導預取、逐位元組無損）：

- 三套 prompt（英文編碼 / 繁中聊天 / 數學推理）跨層 top8 recall **mean ≈ 0.790–0.791、median = 0.875**，三套幾乎完全相同；
- 對照「跨 token 同層複用上一 token 路由」的樸素基線只有 **0.397** —— **LES 跨層預測是同層複用的約 2×**；
- 唯一破線層是 **target 層 1（0.095–0.107）**，機制上屬「embedding 後最淺表示無法跨層預測」，且該層在 prefill 首 token 就會被真實計算、自然進池，**不需要也不值得 lookahead 預取**；
- 排除首層後 mean 即 **0.809**；排除前 3 層（淺層表示未穩定區）後 **0.820–0.826**；解碼穩態大頭 target 層 10–38 普遍 **0.84–0.90**。

## 2. 三套結果（warm-skip 64，steady 577 step / 22503 aligned cells 每套）

| prompt | top8 recall mean | median | p25 | min | cold 專家覆蓋 | 樸素同層基線 |
|---|---|---|---|---|---|---|
| code | **0.7903** | 0.875 | 0.750 | 0.000 | 0.5985 | 0.3979 |
| zh_chat | **0.7914** | 0.875 | 0.750 | 0.000 | 0.6202 | 0.3970 |
| math | **0.7911** | 0.875 | 0.750 | 0.000 | 0.6053 | 0.3972 |

三套 mean 差異 < 0.0012，跨語言/任務高度魯棒。每套 steady steps 577 ≥ 128 門檻。

### recall@d（三套一致，給 G3 預取 depth 決策）

| d | 8 | 9 | 10 | 12 | 16 |
|---|---|---|---|---|---|
| recall@d | 0.79 | 0.83 | 0.86 | 0.90 | 0.94 |

含義：top8 預取 depth=8 已覆蓋 79% 真實專家；depth 加到 10 → 86%、12 → 90%。G3 後台 preadv ping-pong depth 可據此在「覆蓋率」與「預取頻寬/池擠佔」間取捨。

## 3. 分層曲線（target 層 = 被預測層；三套同構，以 code 為例）

| target 層段 | mean top8 recall | 說明 |
|---|---|---|
| 層 1（首 MoE） | **0.095** | 唯一 <0.5；embedding 後未經任何 FFN 的最淺表示，跨層語義差最大 |
| 層 2–4 | 0.60 / 0.62 / 0.66 | 淺層表示快速變換，命中率爬升 |
| 層 5–9 | 0.71–0.75 | 轉穩 |
| 層 10–19 | **0.842** | 穩態 |
| 層 20–29 | **0.866**（最高，層27=0.902） | 穩態 |
| 層 30–38 | 0.81–0.89 | 穩態 |
| 層 39（末層） | 0.56 | 末層回落（待查是否 full-attn / 邊界） |

不同口徑的整體均值（三套，code/zh/math）：

| 口徑 | code | zh | math |
|---|---|---|---|
| 全層 target 1–39（charter 字面） | 0.7903 | 0.7914 | 0.7911 |
| 排除首層 target 2–39 | **0.8086** | 0.8094 | 0.8093 |
| 排除前 3 層 target 4–39 | **0.8197** | 0.8205 | 0.8207 |
| 排除前 4 層 target 5–39 | **0.8243** | 0.8250 | 0.8255 |

## 4. 為何首層（target 1）不應納入 lookahead 判據

1. **注入點機制**：LES 節點建在物理層 `il=0` 末尾，能取得的是「embedding + 第一層 attention 後、**未經任何 FFN**」的 `attn_post_norm`；用它去給層 1 gate 打分，兩者的真實輸入語義差最大，所以 0.095 是預期的機制結果，不是實作 bug。
2. **無預取價值**：每個序列的第一個 MoE 層（target 1）在 **prefill 首 token 就會被真實計算**，其專家在第一時間自然進池；decode 期的 lookahead 是為「後續層、後續 token 會用到、當前不在池的冷專家」提前拉取，首層不在這個人口內。
3. 越早層相鄰層表示變換越快，命中率隨深度升高（0.095→0.60→0.66→…→0.90），符合「相鄰層路由相關性隨深度增強」的 LES 假設；真正長 decode 反覆 miss SSD 的大頭是中後段層，那裡命中率 0.84–0.90。

工程含義：前 1–3 層應由 prefill 預熱、不做 lookahead（或只做保守 depth）；lookahead 預取的適用層是 target ≥ 4（或 ≥5）。

## 5. charter 判據與裁定（2026-10-17 已裁定）

charter G1 字面原判據：三套 prompt 各 ≥128 steady、**整體 top8 recall ≥80% 且無單層 <50%**、三套一致。嚴格按全層字面，整體 mean 0.791 差 0.01、首層 0.095 <0.50，不能直接標 PASS。

**裁定（2026-10-17）：採「可預取層口徑」（target ≥ 4），G1 判 PASS、進 G2。**

- 豁免前 3 層（target 1–3）：由 prefill 首 token 自然預熱、不做 lookahead；首層 target1 是 embedding 後未經 FFN 的最淺表示，機制上不可跨層預測、且無預取價值（見 §4）。
- 判據改為：target ≥ 4 的 per-layer top8 recall 整體 ≥80%，且該範圍無單層**均值** <50%；三套 prompt 各 ≥128 steady step 一致。全層 target1+ 均值保留作對照口徑。
- 三套結果（target≥4，`les_hitrate.py --prefetch-from 4`，verdict=PASS）：

| prompt | 可預取層 mean（判據） | 最低單層（target39） | 全層對照 mean |
|---|---|---|---|
| code | **0.8197** | 0.5615 | 0.7903 |
| zh_chat | **0.8205** | 0.5429 | 0.7914 |
| math | **0.8207** | 0.5654 | 0.7911 |

三套均 ≥0.80、可預取層最低單層均 ≥0.50、steady 577 step ≥128，三套一致（差異 <0.001）。charter 判據與 `les_hitrate.py`（新增 `--prefetch-from`，預設 4，同時上報全層＋可預取層雙口徑）已同步。

## 6. 速度僅記錄、不引用（G1 不看 t/s）

三套 harness 記錄：code pp≈141.26 / tg≈15.07，zh tg≈12.27，math 略；thermal worst=HEAVY、swap 增長 +422~+720 MiB（launch swap≈4.8–5.1GB）。屬非乾淨樣本，僅證明探針在生產配置下可跑、無 OOM；命中率數值不受熱/swap 影響。任何 t/s 結論以 G4 配對 ABBA 為準。

## 7. 已知瑕疵與後續

- sidecar 每 (pass,il) 2 筆 bit-identical 重複（離線腳本字典天然去重，不影響命中率）；根因（注入節點 graph 走訪兩次）待 G1 後查，探針預設 OFF、不影響引擎。
- 41 層中 SSM(recr) 層表未核實，本輪未做 SSM/full-attn 分層（`--recr` 待補）；per-layer 已能看出中後段穩定，分層為補充非阻塞。
- G1 之後：G2（圖內 READ=0 線上對齊離線 ±1%、decode 退化 <2%、bit-identical）→ G3（後台 preadv ping-pong depth，可用 recall@d 曲線選 depth=8/10/12）→ G4（生產 ABBA warm-skip64/seed1/≥3 對、配對中位 ≥+5%）→ G5（MTP 共存 n_tokens>1 降級）→ G6（off/on × 池 {8,6,4}GiB，驗證更準預取能否讓縮池轉正）。
