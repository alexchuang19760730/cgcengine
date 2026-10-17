# 10 GiB 實跑結果：**引擎真的 OOM 了**（2026-10-07 17:5x）

產物 `Backup/spec_tree/run10g_20261007_174727/`（腳本 `scripts/check/run_10g.sh`）
這一輪同時回答了 17:0x 留下的三個問題。

## 0. 一句話

★★ **閘門的預測是對的，方向錯了。**它低估 **7.2 倍**：用「起跑瞬時 `wired` 1684 MiB」算，
而整趟峰值是 **12172 MiB**。⇒ **10 GiB 在這台 16 GB 機器上真的跑不完**，
這是引擎自報的，不是我的推算：`kIOGPUCommandBufferCallbackErrorOutOfMemory` /
`error: Insufficient Memory (00000008:...)`。

## 1. 結果

| 量 | 值 |
|---|---|
| rows | **1**（只有 `pp512` = 201.69 t/s，非量測、不可引用：OOM 中止前唯一 prefill 單點、無 decode）**`tg` 一列都沒有** |
| 停止訊息 | `ggml_metal_synchronize: error: command buffer 8 failed with status 5` |
| 引擎自報 | **`Insufficient Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory)`** |
| `attribution` | **`swap`**（`swap_growth=4993 MiB`、`max_swap=5056 MiB`） |
| `thermal_worst` | NOMINAL（**不是熱的問題**） |
| launch `free` | **272.7 MiB**（8 GiB 那趟 4538、10 GiB 首趟 8852）⇒ 幾乎沒有了 |

★ **`thermal_worst` 是 NOMINAL** ⇒ 這次 OOM **與散熱無關**，是純記憶體。
⇒ ★ 這也回頭確認 16:5x 的結論正確：**髒（swap）不會讓 decode 變快，只會讓它崩。**

## 2. ★★★ 閘門低估 7.2 倍 —— 而它自己早就寫了這件事

| | `wired` | total | 結果 |
|---|---:|---:|---|
| **閘門用的**（起跑瞬時，`phase="pre-launch"`） | **1684** | 10240+1684+1024 = **12948** | 對 13000 ⇒ 剛好過（餘裕 52） |
| **實際**（整趟峰值，`phase="peak"`） | **12172** | 10240+12172+1024 = **23436** | ⇒ 超 10436 |

★ `memory_pressure.py:922-925` 的註解**逐字預告了這件事**：
> 起跑合格 ≠ 整趟合格：引擎自己的那一塊（峰值 Metal 駐留）在 spawn 之前不存在，
> 所以**起跑前那道閘永遠低估它**。

⇒ ★★ **我 17:3x 用「wired 1350」推算時，其實已經讀過這段註解，卻沒把它算進結論。**
⇒ 正確的讀法是：**這道閘的 `phase="peak"` 版本才是對的**（它抓 `worst().max_wired_mb`），
但 `:931` 明寫它「**只有警告，不拒跑**」——「必然換頁」被當成不是「不准跑」。

### 2a. 「Metal 駐留 12172」是什麼
log 自己指認：**「wired 峰值 ≈ 模型權重的 Metal 駐留：另一條線 2026-09-28 量到 12.4 GB
⇒ **縮 pool 解不掉這一項**，要從模型側或『同時間不要跑別的』下手」**
⇒ ★★ 這直接呼應記憶裡那條「async fill + per-expert 重算 CLOSED / 盒子 16GB 上無記憶體側槓桿」。

## 3. 三個問題的答覆（17:0x 留下的）

| 問題 | 答 |
|---|---|
| 10 GiB 能不能跑？ | ❌ **不能**。引擎自報 OOM，不是預測。 |
| 閘門的 `deficit > 0` 語意是「跑不成」嗎？ | ★ **不是，但這次是**。`budget_gate.py:32-42` 說的是「池子必然有一部分在 swap 上」；這一輪確實 swap 4993 MiB 後 OOM。 |
| 關掉 WorkBuddy 會不會有用？ | ❌ **無用**。起跑 `free` 已掉到 272 MiB，而那時 WorkBuddy 還開著——**WorkBuddy 不是那 2048 MiB 的主要來源**。 |

## 4. 判詞修正：10 GiB 這條**正式關上**，且理由比 17:1x 更硬

17:1x 我說「加池不可行、回填只佔 3%」。現在理由更強：
- ★ **不是「超預算所以不可引用」，而是「引擎真的跑不完」** —— 這是 fail-closed 的硬事實。
- ★ 就算把上限 override 到 13000（閘門算式過了、起跑 free 只剩 272 MiB）⇒ **一樣崩**。
  ⇒ ★★ **override 只改變「要不要事前擋」，不改變「能不能跑」。** 這是本輪最該記的一句。
- ⇒ **8.87 GiB 是上限**（17:1x 算的），而 **8.5 GiB 那格**（`--pool8g5-mtp`）仍在等實跑。

## 5. ★ 腳本抓到兩個 bug（都已修，值得記）

`scripts/check/run_10g.sh` 第一次實跑時：
1. ★ **`BUDGET_GATE=warn` 我只寫在說明裡、沒有 `export`** ⇒ rc=2、零樣本。
   （`pool10g_probe.sh:73` 有 export，16:55 那趟才跑得動；我的新腳本漏了。）
2. ★★ **判讀三路都漏了 rc**：只查「OOM 字樣」與「起跑閘不合格」兩句話，
   **沒先看 rc** ⇒ rc=2 時印出「✅ **起跑了**」。
   ⇒ 這是**最嚴重的一類 bug**：判讀工具在沒有資料時報有資料。

⇒ ★★ 通則（已寫入 hygiene）：**判讀必須先看 rc，再確認產物裡真的有 samples**。
「沒跑」與「跑了但慢」是不同的東西，而 rc=2 兩者都不是。

## 6. 誠實邊界
- ★ **tg 一列都沒有** ⇒ 10 GiB 的 decode 速度**至今未知**，且**不可知**（這台機器上它跑不完）。
- ★ 這一趟只跑一臂（無同場基線），但結論不依賴基線（引擎自報 OOM 是自足的）。
- ⚠ `CGC_METAL_WORKING_SET_MB=13000` 的 override **確實生效**（`overridden` 有記錄、
  `ceiling_source=CGC_METAL_WORKING_SET_MB（人指定）`）⇒ ★ ② 那條路本身是通的，
  通的時候機制列仍可讀 —— 只是這一趟機制列沒讀到（崩在 pp512）。