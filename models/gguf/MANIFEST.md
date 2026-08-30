# models/gguf — SHA256 資產清單

大檔（11–13GB）無法進 git/GitHub（單檔 100MB 硬限制、LFS 2GB 上限），
正式備份位於 Hugging Face：**https://huggingface.co/Alexchuang/cgcengine-models**

任何機器可用下表 URL 下載後以 SHA256 驗證，得到與本 repo 測試完全相同的模型位元組。

## 檔案清單

| 檔案 | 大小 (bytes) | 角色 | SHA256 |
|---|---|---|---|
| Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf | 13,663,116,512 | **MTP 生產線**（llama-speculative-simple，blk.40 MTP head Q6_K + dense IQ4_XS） | `13a43e06e7f491375bc00938b6272348f04bb0418d19a4d01c1c6bcebcd4f57d` |
| Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf | 13,211,155,424 | **非 MTP 生產線**（llama-simple，無 blk.40） | `9c964e657212fea1f24905dd7b0a89b82fd807d19fab0b41da14251b07b88fbe` |
| Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS.gguf | 14,069,275,872 | MTP 載體基底（blk.40 MTP head 同 trunk UD-IQ3_XXS） | `6275d06c6e1b0d0a4e07a69a5fbdc719dbaeaae87bc48e6c8377f4cd58ec369c` |
| gemma-4-26B-A4B-it-UD-IQ3_S.gguf | 11,289,671,136 | gemma4 家族（-ngl 30，無 MTP 路徑） | `878be93f9c238ea853b3fd1eb602637ce3cf1cddea56dc345d9a7bf2d6093e29` |
| Huihui-Qwen3.8-27B-abliterated-UD-IQ3_S.gguf | 11,952,901,024 | Qwen3.8 MoE 測試用 | `f785ad524b138cd53575410e07e93dfa6eff1cc92ca698264cb9c96973db6886` |

## 下載（Hugging Face）

```bash
# 單檔（hf CLI 內建斷點續傳與上傳時的 hash 校驗）
hf download Alexchuang/cgcengine-models Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf \
  --local-dir models/gguf

# 或 curl 直下（無續傳）
curl -L -o models/gguf/<檔名> \
  https://huggingface.co/Alexchuang/cgcengine-models/resolve/main/<檔名>
```

## 驗證

```bash
cd models/gguf
shasum -a 256 -c SHA256SUMS        # 全部 5 檔
shasum -a 256 -c <(grep denseIQ4X SHA256SUMS)   # 單檔
```

## 在地重生（替代下載）

- **denseIQ4X 載體**：不從 HF 下載也能從基底模型重生成（bit-identical by construction）：
  `python3 scripts/gen_denseiq4x_tt.py`（錨定 `^name$` per-tensor 釘選；dense Q6_K→IQ4_XS、
  head 釘 Q6_K、其餘 byte-copy）再以 `llama-quantize --tensor-type-file` 套用。
- 其餘 3 個外來模型（Qwen3.6 / gemma-4 / Huihui）為 Unsloth 態量化發佈物，以 SHA256 對齊上游版本。

## 紀律

- 新增/更換模型檔時：`shasum -a 256 <file>` 更新本表與 `SHA256SUMS`，
  上傳 `hf upload Alexchuang/cgcengine-models <file> <file>`，兩者同一 commit。
- `.gitignore` 的 `*.gguf` 排除規則**不得移除**（60GB blob 會毀掉 repo）。
