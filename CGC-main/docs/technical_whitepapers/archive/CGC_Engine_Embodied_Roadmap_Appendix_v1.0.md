# CGC Engine Embodied Roadmap Appendix v1.0
**版本**：v1.0  
**更新時間**：2026-06-12  
**對應主文**：`CGC_Engine_Embodied_Technical_Whitepaper_v1.0.md`

---

## R1. 附錄定位

本附錄聚焦 `CGC Engine Embodied 技術白皮書 v1.0` 的工程推進路線與 implementation checklist，內容包括：

- roadmap 與階段性優先級
- implementation checklist
- rule-based atom segmentation v0.1 checklist

主文負責技術定位與架構主張；本附錄負責將主文拆解為具體可追蹤的工程落地項目。

---

## R2. Roadmap：v1.0 之後的優先級

### R2.1 短期優先

- 完成 `rule-based atom segmentation`
- 完成 `single-episode export -> multi-atom export`
- 完成 bridge generator 對真實 atom library 的掛載
- 補齊 `structured conditioning smoke test`

### R2.2 中期優先

- 讓 `AtomRetrievalHead / PrimitiveComposer / ResidualActionHead` 真接 GeoMirror library
- 建立 view-invariance / one-shot 驗證矩陣
- 補齊 `holomotion / psi0` 實際執行 feedback 回寫

### R2.3 長期優先

- 舊 primitive runtime path 的高效 kernel 化實作
- learned segmentation
- 大規模多資料源 canonical BEV library
- 跨 provider comparative benchmark 體系

---

## R3. Implementation Checklist

### R3.1 GeoMirror 與資料層

- `episode_enriched.npz` 補齊 `source_frame_name`
- `episode_enriched.npz` 補齊 `canonical_frame_name`
- `episode_enriched.npz` 補齊 `transform_source_to_bev`
- `episode_enriched.npz` 補齊 `quat_order / unit_translation / ground_axis`
- `episode_enriched.npz` 補齊 `left_ee_pose_seq / right_ee_pose_seq`
- `episode_enriched.npz` 補齊 `object_pose_seq / contact_state_seq / support_state_seq`
- 補齊 `MANO / object_pose_6d` 空值策略一致性

### R3.2 Atom Library 與 Segmentation

- 落成 `rule-based atom segmentation` state machine
- 支援 `approach / grasp / carry / place`
- 輸出 `atom_index.json`
- 輸出 `atom_meta.json`
- 輸出 `bev_pose_seq.npz`
- 輸出 `retrieval_feature.npy`
- 輸出 `temporal_profile.json`
- 補齊 `segmentation_confidence / fallback_reason`

### R3.3 Bridge Artifact 與 Backend

- 讓 bridge generator 掛載真實 `08_bev_action_atom_lib`
- 確保 `psi0_bridge_schema v2.0` 欄位全部由真資料回填
- 確保 edge runner 對所有 provider 都能輸出一致 `runtime_contract`
- 完成現行 runtime host 對 structured conditioning 的更深層消費

### R3.4 Runtime / Execution

- 將 `AtomRetrievalHead` 真接 `08_bev_action_atom_lib`
- 將 `PrimitiveComposer` 真接 GeoMirror primitive
- 將 `ResidualActionHead` 改成末端位姿殘差輸出
- 打通 `HoloMotion / OmniRetarget -> Psi0 -> controller`
- 回寫 `Psi0 fallback`、`retarget error`、`task success`

### R3.5 Evidence / Gate

- 建立 `structured conditioning smoke test`
- 建立 `view-invariance / one-shot` evidence schema
- 把實驗結果接入 report
- 設立對應 gate threshold

---

## R4. Rule-Based Atom Segmentation v0.1 Checklist

### R4.1 前置訊號

- `timestamp_ns`
- `left_ee_pose_seq`
- `right_ee_pose_seq`
- `object_pose_seq` 或 `object_pose_6d`
- `contact_state_seq`
- `support_state_seq`
- `phase_t`

### R4.2 衍生特徵

- `ee_to_object_dist_l`
- `ee_to_object_dist_r`
- `dual_hand_contact`
- `object_linear_speed`
- `object_height_above_support`
- `ee_speed_l`
- `ee_speed_r`
- `contact_stable_window`
- `support_stable_window`

### R4.3 State Machine

```text
IDLE
 -> APPROACH when ee approaches object and no contact
 -> GRASP when contact starts
 -> CARRY when contact stable and object lifted/moving
 -> PLACE when object descends into target zone
 -> DONE when contact released and object stable
```

### R4.4 產出欄位

- `atom_id`
- `label`
- `sub_label`
- `source_episode_id`
- `source_frame_range`
- `fps`
- `duration_s`
- `representation`
- `contact_pattern`
- `phase_profile`
- `segmentation_confidence`
- `segmentation_rule_version`
- `fallback_reason`

---

## R5. 附錄結語

本附錄的目標不是重複主文，而是把主文已確立的技術主幹轉成可追蹤的工程里程碑與落地清單。

對 CGC Engine Embodied 而言，真正的推進標準不是只有：

- schema 已存在
- bridge 已能載入
- skeleton 已能跑通

而是：

- 可持續產生真實 artifact
- 可持續擴展 atom library
- 可持續接入 runtime 與 evidence
- 可持續朝 view-invariance / one-shot 的正式驗證前進
