# Qwen3.6 r3/r4 建置與驗證狀態

> 更新: 2026-08-09

## ✅ 已完成

### r4 (4-bit) — `prime-agent-worktrees/qwen36-r4.gturbo` (21GB, 本地)
- 40 層 (30 DeltaNet + 10 GatedAttn), 256 experts, stride 1.8MB
- **lm_head bug 已修復**：repack 的 GLOBAL_RESIDENT 用錯 key（`model.language_model.lm_head.weight` vs 官方 `lm_head.weight` 無前綴）→ lm_head 被靜默跳過 → logits 隨機 → ppl=228萬
- 用流式 patch 把 lm_head 附加進 model_weights.bin（613 entries, 4.9GB, 無需重跑 40 層）
- **驗證**：52-token quick ppl → **ppl=2.158, top1-acc=80.4%**（之前 0%）
- 排除的假設（全部驗證無誤）：fp16 dense 編碼、dequant/forward 數學（rel diff 0.13）、GatedAttn 權重齊全（60/60）、DeltaNet A_log/dt_bias 精確、RMSNorm 1+w 慣例、層類型判定（linear_attention/full_attention vs 每4層規則一致）

### r3 (3-bit) — `prime-agent-worktrees/qwen36-r3.gturbo` (18GB, 本地 ✓)
- repack 已修 lm_head key bug（r3 直接含 lm_head，不重蹈覆轍）
- 40 層完成、613 resident tensors（**lm_head present ✓**）、stride 1,376,256 bytes（比 r4 小 24% = 3-bit 預期）
- 3-bit pack/dequant roundtrip 驗證：max err 0.276（int3 量化誤差內）
- **已搬回本地**（刪除 gemma4-r2-attn3 8G + gemma4-r2 1.3G + WorkBuddy 快取 1.3G 騰出空間）
- 外接盤暫存副本已刪

## ⚠️ 待辦
- 完整 1077-token ppl 在 CPU DeltaNet fallback 下需 ~2.8 小時不現實 → 52-token sanity 已定案格式
- 速度測試走 Metal 引擎（turbo-fieldfare worktree）
- 本地磁碟剩 7.1Gi（r3 18G + r4 21G 都已本地）
