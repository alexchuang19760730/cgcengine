# CGC Engine Embodied Validation / Benchmark Appendix v1.0
**版本**：v1.0  
**更新時間**：2026-06-12  
**對應主文**：`CGC_Engine_Embodied_Technical_Whitepaper_v1.0.md`

---

## V1. 附錄定位

本附錄聚焦 `CGC Engine Embodied 技術白皮書 v1.0` 的驗證與基準測試方法，內容包括：

- `view-invariance / one-shot` 的工程定義
- 驗證實驗矩陣
- 建議 gate / metrics
- benchmark 與驗收準則

主文負責架構與能力定位；本附錄負責說明如何證明系統真的具有 view-invariance、one-shot 與 execution robustness。

---

## V2. View-Invariance 與 One-Shot 的工程定義

CGC Engine Embodied v1.0 明確主張，不能用單次 demo 成功替代 view-invariance 或 one-shot 的證明。

### V2.1 View-Invariance

必須同時驗證：

- `canonical_bev` 表示跨視角是否穩定
- atom retrieval 是否能跨視角命中同語意 atom
- HoloMotion / Psi0 落地執行是否穩定

### V2.2 One-Shot

在本系統中的定義不是「看一次就能做所有相似動作」，而是：

> 對同一語意家族、有限擾動範圍內的新視角、新擺位、新軌跡細節，系統只依賴一次示教即可形成可重用的 primitive / atom / semantics，並成功完成任務。

### V2.3 當前工程主張

v1.0 已完成可支撐這類驗證的基礎設施：

- canonical BEV schema
- atom library schema
- bridge artifact
- runtime contract
- M7.3 conditioning metrics

v1.0 尚未完成大規模 view-invariance / one-shot 實驗閉環本身。

---

## V3. 驗證實驗矩陣

| 實驗編號 | 實驗名稱 | 目標 | 主要指標 | 建議通過標準 |
| :--- | :--- | :--- | :--- | :--- |
| E1 | Cross-view BEV Consistency | 驗證 `canonical_bev` 跨視角穩定性 | `ee pose alignment error`、`object pose alignment error` | 顯著優於 raw-frame baseline |
| E2 | Cross-view Retrieval Stability | 驗證跨視角檢索命中同語意 atom | `top-1 / top-k semantic hit rate` | `top-1 >= 0.85`、`top-k >= 0.95` |
| E3 | Cross-view Contact Consistency | 驗證接觸語意跨視角一致性 | `contact consistency`、`grasp type consistency` | >= `0.9` |
| E4 | Cross-view Retarget Robustness | 驗證 HoloMotion 在不同視角輸入下仍穩定 | `IK failure rate`、`contact slip rate` | 低於 baseline |
| E5 | Cross-view Psi0 Stability | 驗證 Psi0 在不同視角 primitive 下仍穩定 | `fallback rate`、`ZMP violation` | 不高於 baseline |
| E6 | Single-demo Same-family Generalization | 驗證一次示教後同家族泛化 | `task success rate`、`semantic hit rate` | 顯著高於 zero-shot |
| E7 | One-shot Cross-view Generalization | 驗證一次示教後換視角仍成功 | `cross-view success rate`、`performance drop` | 成功率下降 <= `10%` |
| E8 | Ablation: No Canonical BEV | 驗證 `canonical_bev` 的必要性 | `retrieval stability`、`execution success` | 明顯差於 GeoMirror v2 |
| E9 | Ablation: No Atom Library | 驗證 atom library 的必要性 | `one-shot success`、`cross-view robustness` | 明顯差於 full system |
| E10 | Ablation: No Semantics | 驗證 `holomotion_semantics / psi0_semantics` 的必要性 | `contact slip`、`fallback rate` | 明顯差於 full semantics |

---

## V4. 建議 Gate 指標

### V4.1 Representation / Retrieval

- `view_invariant_retrieval_top1 >= 0.85`
- `view_invariant_retrieval_topk >= 0.95`
- `contact_consistency >= 0.9`

### V4.2 Execution

- `cross_view_success_drop <= 10%`
- `retarget_failure_rate <= baseline`
- `psi0_fallback_rate <= baseline`

### V4.3 One-Shot

- `one_shot_success_rate >= 0.7`
- `same_family_generalization_rate >= target threshold`

---

## V5. Benchmark 實施建議

### V5.1 最小可行驗證集

建議先跑以下 6 組最小驗證：

- `E1 Cross-view BEV Consistency`
- `E2 Cross-view Retrieval Stability`
- `E4 Cross-view Retarget Robustness`
- `E6 Single-demo Same-family Generalization`
- `E7 One-shot Cross-view Generalization`
- `E8 Ablation: No Canonical BEV`

### V5.2 建議任務家族

建議優先從單一任務家族開始：

- `carry_box`

之後再擴展：

- `approach`
- `grasp`
- `place`

### V5.3 推薦對照組

至少保留以下 4 組：

- `GeoMirror v2`: `canonical_bev + ee_pose + atom + semantics`
- `Raw-frame baseline`
- `Joint-centric baseline`
- `Direct regression baseline`

如資源允許，可再增加：

- `No semantics`
- `No object pose`
- `No contact state`

---

## V6. Evidence 記錄建議

每次 episode 建議至少回寫以下欄位：

- `view_id`
- `source_frame_name`
- `canonical_frame_name`
- `atom top-1 / top-k`
- `retrieval score`
- `selected atom label`
- `contact consistency`
- `retarget error`
- `psi0 fallback count`
- `task success`
- `failure type`

這樣可區分失敗來源究竟來自：

- BEV 投影失敗
- retrieval 錯誤
- retarget 錯誤
- Psi0 安全過不了

---

## V7. 附錄結語

本附錄的目標，是把 `CGC Engine Embodied` 中最容易被口頭化的能力主張，轉成可量化、可重複、可 gate 的驗證框架。

真正要被證明的不是：

- demo 看起來像成功
- 某次視角切換剛好沒失敗

而是：

- `canonical_bev` 是否真的降低視角敏感性
- atom library 是否真的抓到語意而非攝影機外觀
- HoloMotion / Psi0 是否真的穩定執行
- 一次示教是否真的能支撐有限同家族泛化
