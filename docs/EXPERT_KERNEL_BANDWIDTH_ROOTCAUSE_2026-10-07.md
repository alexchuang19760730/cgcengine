# Expert id kernel 頻寬根因定位（38–45% 天花板）

日期：2026-10-07　工具：`scripts/check/shape_probe/cgc_shape_probe`（獨立探針、不蓋共享 dylib）
任務：#19　**本文件只做根因定位，不許諾優化收益。**

---

## 1. 一句話根因

expert GEMV（`kernel_mul_mv_id*`）跑不滿頻寬，**不是 id/gather/批次調度、也不是反量化算術**，
而是 **IQ2_S / IQ3_S 這種「2-bit/3-bit 查編碼表」格式在 kernel 內的訪存組織**：
每個權重 block 要從 **constant table（`iq2s_grid`）查錶**、並從 block 的**多個非連續欄位**
（`qs` 多段、`qh`、`scales`、`signs`、`d`）gather；這些查錶/gather 訪存打斷了權重的連續供給，
把有效頻寬壓到峰值的 ~30%。

連續編碼（F16、Q8_0）在**同一條 id 路徑**上能跑 82–91 GB/s，是決定性反證。

---

## 2. 判別矩陣（同一支 id kernel、只換權重編碼，batch 口徑）

| 編碼 | 反量化方式 | 形狀 k/out | used | 頻寬 GB/s | % 峰值（108.8） |
|---|---|---|---|---|---|
| F16 | 無（連續） | 2048/512 | 8 | **91.16** | 84% |
| Q8_0 | 線性、連續 block | 2048/512 | 8 | **86.05** | 79% |
| IQ2_S（gate/up） | **constant grid 查錶＋多欄位 gather** | 2048/512 | 8 | **36.80** | 34% |
| IQ2_S | 同上 | 2048/512 | 1 | 17.80 | 16% |
| F16（down） | 無 | 512/2048 | 8 | **81.70** | 75% |
| IQ3_S（down） | **grid 查錶＋多欄位 gather** | 512/2048 | 8 | **31.36** | 29% |

關鍵對比：

- gate/up：F16 91 → IQ2_S 37（**−59%**）
- down：F16 82 → IQ3_S 31（**−62%**）
- **Q8_0 也做反量化、卻 86 GB/s** ⇒ 反量化算術無辜、查表式訪存才是因。

---

## 3. 機制（第一性，為什麼 bandwidth-bound 的 kernel 卻餵不滿頻寬）

`kernel_mul_mv_id_glu_iq2_s_impl`（ggml-metal.metal:11416）內，32 個線程切 K 維，主循環每步：

1. 對每個 block 讀 `qs`（4×ib）、`qh`、`scales`、`signs`、`d` —— **多筆非連續小 load**；
2. 用 `qs/qh` 當索引查 **constant memory 的 `iq2s_grid`**（:11496-97）；
3. 查到的碼 × `yl` 做點積。

這些查錶/gather 與主權重 load **共享 load 管道**，且存在「load block 欄位 → 查 grid → 才能 dot」的
串行依賴；線程無法持續以滿頻寬發起權重 load，供給被查錶/gather 打斷。

- F16：連續大塊 load、無查錶 ⇒ 91。
- Q8_0：連續 block ＋ 每 block 一個線性 scale、無 grid 查錶 ⇒ 86。
- IQ2_S/IQ3_S：查錶/gather 主導 ⇒ 31–37。

**注意這不是「反量化算不過來」**：roofline 已證 ALU 極度寬裕（算術強度 4.55 vs 拐點 101、22×）；
是查錶式**訪存排程**問題，不是算力問題。

---

## 4. 逐項排除（皆有讀數）

| 假設 | 判決 | 證據 |
|---|---|---|
| id/gather/批次調度慢 | **排除** | 同 id 路徑 F16 91、Q8 86，id 結構能餵滿 |
| 反量化算術慢 | **排除** | Q8_0 同樣 dequant、86 GB/s |
| 專家讀得散（gather 散射） | **排除** | shared vs cycled ids 只差 2%（MOE_GATHER_BOUND §4） |
| NSG 並行度不夠 | **排除** | id/dense NSG 1..32 全距 <1pp（§4b、sweep_nsg） |
| 單 expert 平行不足 | 次要、非生產形狀 | used=1 18、used=8 37；生產固定 used=8 |
| **IQ2_S/IQ3_S 查錶式訪存** | **✅ 根因** | 連續編碼 82–91 vs 查錶編碼 31–37 |

---

## 5. 口徑與邊界（誠實）

- 以上為 shape_probe 的 **batch 口徑**（32 op 一批、ids cycle bank、cold），與真實 decode 單次 fused
  kernel 口徑略異；但**同口徑內 F16/Q8 vs IQ2 的相對對比穩健**，且 IQ2_S 37 / IQ3_S 31 復現了真實
  expert bank 的 30–45%（MOE_GATHER_BOUND：gate/up 45%、down 30.8%）。
- 真實 pool resident vs cold 不改變結論（§4 已證 resident/cold 對 kernel 頻寬只差 2%）。
- 本定位**不回答「能不能優化、能提多少」**；查錶是 IQ2_S/IQ3_S 高壓縮率的內在代價，能否重排訪存
  逼近 Q8_0 完全未證。

---

## 6. 若日後要優化（假設方向，非承諾、收益未知）

僅記錄可被實驗檢驗的方向，動手前需先做最小判別：

- 把權重 load 與 grid 查錶**解耦**（先批量預取/重排 block 欄位，再連續 dot），看供給能否連續；
- 增加每線程 outstanding load（軟體 prefetch / 多 buffer），打破 load→查錶→dot 串行鏈；
- 驗收必須 bit-exact，並用同口徑 ABBA 對比，**禁止用「Q8 有 86」直接外預期收益**。

### 條件式算術（純上界假設、非預期）

*若且唯若* IQ2_S gate/up 頻寬能從 45% 提到 Q8 的 79%、IQ3_S down 從 31% 提到 F16-down 的 75%：
T=1 expert bank 約可省 gate/up ~1.7ms + down ~2.4ms ≈ **4.1ms/步（T=1）**；T=4（verify）expert bank
基數更大、絕對值更高。此為**物理上界、未證可行**，不代表可交付收益。

---

## 7. 復現指令

```bash
cd /Users/alexchuang/Documents/flashkv-devserver/scripts/check/shape_probe
# 連續編碼對照（應 ~86-91）
./cgc_shape_probe --op mul_mat_id --type f16  --k 2048 --out 512 --experts 256 --used 8 --mode batch --pool-mib 2048
./cgc_shape_probe --op mul_mat_id --type q8_0 --k 2048 --out 512 --experts 256 --used 8 --mode batch
# 查錶編碼（應 ~31-37）
./cgc_shape_probe --op mul_mat_id --type iq2_s --k 2048 --out 512 --experts 256 --used 8 --mode batch
./cgc_shape_probe --op mul_mat_id --type iq3_s --k 512  --out 2048 --experts 256 --used 8 --mode batch
```
