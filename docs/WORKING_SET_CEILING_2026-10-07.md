# 真正放開 10 GiB，一定要動 `recommendedMaxWorkingSetSize` 嗎？（2026-10-07 17:3x）

## 0. 答案：**不一定，而且大概率不該動它。**

三個查證結果，其中兩個指向「有更便宜的路」。

## 1. ★★ 引擎自己從來不用它擋配置 —— 超限只印 `GGML_LOG_WARN`

`ggml-metal-device.m:1574`：
```objc
if (device.currentAllocatedSize > device.recommendedMaxWorkingSetSize) {
    GGML_LOG_WARN("%s: warning: current allocated size is greater than the recommended max working set size\n", __func__);
}
```
★ **是 `WARN`，不是 return、不是 abort。**

全樹複核 `max_working_set_size` 的使用點，**只有三處**：
| 行 | 用途 |
|---|---|
| `ggml-metal-device.m:923` | **寫入**（`= mtl_device.recommendedMaxWorkingSetSize`） |
| `:981` | `LOG_INFO` 印出來 |
| `:1574` | `LOG_WARN` 比較 |

⇒ ★★ **沒有任何一處用它擋配置或中止。**
⇒ ★ **真正拒跑 10 GiB 的是我們自己的閘門，不是 Apple 的上限，也不是引擎。**

## 2. ★★★ 而且「放行」開關已經存在、已接好

`llama_bench_matrix.py:1059-1061`：
```python
refused_metal = mempress.metal_gate(
    _pre, _pool_mb, pool_source=_pool_src, phase="pre-launch",
    waived=os.environ.get("CGC_IGNORE_METAL_BUDGET") or None)
```
⇒ ★ **`CGC_IGNORE_METAL_BUDGET=<理由>` 就是放行閘門的方式**，而且它有既定用法：
`docs/SINGLESUBMIT_CORRECT_DESIGN_2026-09-28.md:273` 寫著
「要跑就得減壓或 `CGC_IGNORE_METAL_BUDGET=<理由>`（**理由會進產物**）」；
`scripts/check/charters/e-spec-tree-2026-10-02.yaml:165` 也用過它。

留痕機制完整（`memory_pressure.py:938-944`）：
```python
return {"ok": (not reasons) or bool(waived), ..., "waived": waived,
        "reasons": reasons, "over_by_mb": over, ...}
```
⇒ ★★ **放行不會抹掉紀錄**：`reasons`／`over_by_mb`／`ceiling`／`reserve` 全部照樣寫進產物，
`waived` 欄記下放行理由。⇒ **這正是 operator 立規要的那種繞過：能跑，但不可引用。**

## 3. ★★ 而且 10 GiB 沒有「引擎跑不動」的證據

`metal_fail_line()`（`memory_pressure.py:843-856`）抓的是**引擎自己回報的實際 OOM**
（`kIOGPUCommandBufferCallbackErrorOutOfMemory` / `Insufficient Memory` / `CGC-METAL-FAIL`）。
docstring 明寫：「**比預算算術硬：有這一行的那一輪已經沒有輸出可用，不是「跑得慢」。**」

實測兩臂（`pool10g_20261007_164641`）：
| 臂 | 引擎實際 OOM 行數 |
|---|---:|
| `mctl8g`（跑成） | **0** |
| `mtry10g`（被擋） | **0** |

⇒ ★★ **10 GiB 的 `fail_line` 是空的** ⇒ 它是**預算算術的預測**，不是「引擎說它會崩」。
⇒ ★ 這把問題變清楚：**我們要回答的是「它會不會真的 OOM」，而那只能靠真的跑一次來知道。**
   閘門目前是**用算術代替那次試驗**，而算式的兩個主要項（`pool` 10240 是宣告值、
   `wired` 1350 是起跑瞬時值）都不確定到能當否證依據。

## 4. ★ 那為什麼我仍不建議直接 waive

閘門註解（`budget_gate.py:32-42`）自己記著，本機 `deficit` **每一場皆正**
（wired 峰值 9.9–12.3 GiB ／ pool 8192 ／ 上限 11453 ⇒ 超 7.7–10.1 GiB），
**包含已認證的 11.703 錨點**；2026-09-30 登記 **IN-BUDGET 0 / 2629 場**。
⇒ ★★ **在這台機器上，這道閘對 8 GiB 也是「超」的。**
   ⇒ 它的實質不是「能不能跑」，而是「**池子必然有多少比例在 swap 上**」——
   而 10 GiB 的 1160 MiB 缺口比 8 GiB 的 887 MiB 餘裕更吃緊，這個**差別是真的**。
⇒ ★ 所以正確的取捨是：**waive 可以跑，但它給的數字必然比 8 GiB 那趟更不可引用。**
   值得跑的理由只能是「量機制列」，不是「量速度」。

## 5. ★★ 四條路的取捨（②③ 修好後重新排序）

`metal_ceiling()`（`memory_pressure.py:772-781`）的優先序本身就是答案的一部分：
1. `CGC_METAL_WORKING_SET_MB` —「人明確指定的真實（用於**換機器或除錯**）」
2. 本次 stderr 的引擎自報值
3. `sysctl iogpu.wired_limit_mb` — ★ **本機實測是 `0`**（⇒ 這條在本機無效）
4. 快取檔 `Backup/metal_working_set.json` — 註解明寫「**這台盒子上唯一能在 spawn 之前拿到它的方法（sysctl 是 0）**」
⇒ ★★ 換言之，**本機的 11453.25 是第 4 條（上一場的快取）**，而第 1 條本來就是為了
「換機器或除錯」而設的 override。

| 選項 | 動什麼 | 得到什麼 | 代價 |
|---|---|---|---|
| **① `--pool8g5-mtp`（已做好）** | 池 8 → 8.5 GiB | ★ **合法過閘**，回答「+512 MiB 會不會動 `cold_filled` 平台」 | 只多 512 MiB，效應可能看不出 |
| **② `CGC_METAL_WORKING_SET_MB=<N>`** | **只改「上限」這個讀值來源**（第 1 條優先序） | ★ **閘門正常運作**、`reasons`／`over_by` 照樣算、`overridden` 欄記下是人指定的 | ★ **必須給一個誠實的 N**。給大了＝自己關掉那個警告 |
| ③ `CGC_IGNORE_METAL_BUDGET=<理由>` | **直接跳過整道閘** | 能跑，且理由進產物 | ★ **`overridden` 不會記**、閘門的算式完全不跑 ⇒ 產物少一整層可稽核性 |
| ④ 動引擎／系統的 `recommendedMaxWorkingSetSize` | 上游 | — | ★ **最差**：不解決物理（`deficit` 仍超 1160），只移除誠實警告，且要 rebuild |

★ **② 比 ③ 好**（同樣是「人說了算」，但閘門照跑、數字照算、override 有記錄），
**④ 最差**（動上游卻不解決問題）。
⇒ ★★ **順序：① → ② →（④ 不做）**。而 ③ 保留給「① 與 ② 都用完仍要量速度」的最後手段。

### ★ ② 的誠實用法長這樣
```bash
# 依 16:55 實測的 wired=1350 與 8 GiB 臂的餘裕，給一個「我承認定這個上限」的數字
CGC_METAL_WORKING_SET_MB=13000   # 10 GiB 池需要 total ≤ 13000 ⇒ 10240+1350+1024=12614 通過
```
★ 這個數字**不是**「讓它過」而是「**我判斷 11453 對這台機器的實務狀態太保守**」——
**那個判斷屬於你（operator），我不替你簽。**

## 6. 誠實邊界
- 本輪**未跑量測**，全部靜態查證（源碼複核 ＋ 產物讀取 ＋ 算式）。
- ★ 「10 GiB 真的跑得起來」**至今未驗**。② 是唯一能驗的途徑，而它需要你決定是否值得。
- ⚠ §2 的 `CGC_IGNORE_METAL_BUDGET` **本輪未實跑驗證**（靜態讀到接線正確、
  且有 2026-10-02 的使用記錄）。若要用，**必須帶理由**，理由會進產物。
- ⚠ 引擎 `LOG_WARN` 那句是 **`#ifndef GGML_METAL_NDEBUG`** 內的，正式 build 可能編譯掉
  ⇒ 「引擎會警告」這句在 release build 上**未證**；但「引擎不擋配置」不受影響
  （那個 `if` 沒有任何阻斷行為）。