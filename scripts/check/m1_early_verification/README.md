# M1 早期三項關鍵驗證測試

日期：2026-09-13
分支：demo/sweet-spot-windows-fix @ c22d75395
機器：MacBook Air M4 16GB

## 測試目的

在 M1（解耦 pool 與圖形）實作之前，先驗證三個關鍵假設：

1. **Canonical Gather Order**：按 expert id 排序後的累加順序是否與原來的 slot index 順序一致（M1 不垮的基礎）
2. **Union > Slots 串流路徑**：當 union > slots 時，是否能優雅降級到串流路徑（pool size 可變的前提）
3. **Metal 整層 Slab 可行性**：Metal 是否能分配 280MB+ 連續 buffer 並做 batch matmul（prefill 整層路徑的硬體基礎）

## 測試結果摘要

| 測試 | 狀態 | 對 M1 的意義 |
|------|------|-------------|
| 1. Canonical Gather Order | ⚠️ 需代碼改動 | M1 核心改動，確保換 pool size 時 M1 不垮 |
| 2. Union > Slots 串流路徑 | ❌ 當前不支援（abort） | M1 核心功能，「pool size 只影響速度」的前提 |
| 3. Metal 整層 Slab | ✅ 通過 | 硬體可行，prefill 整層 slab 路徑可實作 |

## 文件說明

- `test_metal_slab.py`：測試 3 - Metal 整層 slab 可行性（Python/MLX）
- `test_union_slots.sh`：測試 2 - union > slots 串流路徑驗證（Bash）
- `test_canonical_gather.py`：測試 1 - canonical gather order 驗證框架（待 M1 代碼改動後使用）

## 關鍵發現

### 測試 2：Union > Slots 會 FATAL abort

當前代碼（llama-expert-cache.cpp line 798-803）：

```cpp
if (!in_flight) {
    fprintf(stderr,
            "llama_expert_cache: FATAL ensure_batch layer=%u: %zu distinct experts exceed the %u usable pool slots and no fill is in flight — cannot assign; aborting\n",
            layer, n, llama_expert_cache_usable_slots(cache, layer));
    abort();
}
```

這表示當前 `union-fit` 是硬性 PASS/FAIL，沒有串流降級路徑。M1 需要把它改成 `union-routable`：union ≤ slots **或** 串流路徑被走過。

### 測試 3：Metal 完全可行

- 280MB 連續 buffer 分配成功（299ms）
- Batch matmul [256,1024,512]×[256,512,256] 成功（70ms）
- 整層 MoE 三個 proj 共 768MB 分配成功
- Prefill 整層計算（256 tokens）成功（23.5ms）

## 使用方式

### 測試 3：Metal Slab

```bash
# 使用 Edge0 的 .venv（有 MLX）
/Users/alexchuang/Documents/edge0/.venv/bin/python scripts/check/m1_early_verification/test_metal_slab.py
```

### 測試 2：Union > Slots

```bash
# 需要先關閉其他 server 釋放內存
bash scripts/check/m1_early_verification/test_union_slots.sh
```

### 測試 1：Canonical Gather Order

```bash
# 待 M1 代碼改動後，用此框架驗證
python3 scripts/check/m1_early_verification/test_canonical_gather.py
```

## 結論

- 測試 3 通過 → 硬體可行，prefill 整層 slab 路徑可實作
- 測試 1 和 2 正是 M1 需要實作的核心改動 → 驗證了 roadmap 的判斷：M1 是最關鍵的里程碑
- M1 實作順序建議：先做 canonical gather order（M1 不垮的基礎），再做 union > slots 串流路徑（pool size 可變的前提）
