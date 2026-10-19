# Expert kernel 多字段 gather — 最終判決（共有問題）

日期：2026-10-07
脈絡：通用 MoE engine（CGC Engine）目標——不同 MoE 模型在 16GB M4 Air 上 decode 25+。
本文是任務 #19/#20/#21 的收束：用受控變體把「expert GEMV 只有峰值 30–45%」的便宜工程解空間**逐面實測封死**，並給出問題的物理本質與僅存的真實路徑。

## 1. 判別矩陣（同一 id kernel、k=2048/out=512/used8/batch、bank 82 MiB、ids cycle bank）

| 變體 | 說明 | GB/s | vs base |
|---|---|---|---|
| **baseline** | constant grid 查表 + 多字段 gather（現況） | **36.3** | — |
| TG (`CGC_IQ2_TGGRID=1`) | grid 查表搬進 threadgroup、每 expert 複製 8KB + barrier | 29.3 | **−19%** |
| COOP (`CGC_IQ2_COOP=1`) | 整 block（82B）線性協作讀進 threadgroup、barrier、再解包 | 33.7 | **−7%** |
| NOGRID (`CGC_IQ2_NOGRID=1`) | 連續 device 讀 activation、去掉查表依賴、保留 gather | 40.5 | **+12%** |
| （對照）Q8_0 id | 直接讀 int8、線性 block 掃描 | **86.0** | +137% |
| （對照）F16 id | 直接讀 half | **91.2** | +151% |

並行度曲線（id iq2_s used8）：

| tokens | GB/s | vs t1 |
|---|---|---|
| 1 | 36.3 | — |
| 4 | 44.1 | **+21%** |

## 2. 規律 → 根因

- **加 barrier 就更慢**（TG −19%、COOP −7%）：threadgroup 中轉/同步在 decode（每 block 計算量小）是淨負擔。
- **去掉一層 load-use 依賴才略快**（NOGRID +12%）：依賴鏈深度有影響，但只佔小頭。
- **提高並行度僅 +21%**（tokens 1→4），仍只到 Q8 一半：延遲可被部分掩蓋、但主要損失不是並行度能消的。

逐項排除：id/批次框架（F16/Q8 同框架 86–91）、算術總量（roofline 比值 0.045、遠在 bandwidth 區）、NSG/散射 ids（無影響）、查表來源、block 連續性、並行度——**全部不是主因或不是便宜可解**。

### 物理本質

IQ2_S 把權重壓到 2.56 bpw，靠**執行期查表展開**恢復。每 K 權重的解碼鏈：

```
讀打包 qs/qh → 位拼接組裝 grid offset（數據依賴）
             → 從 8KB iq2s_grid 表 gather 8 個展開字節
             → 字節轉 float → 處理 signs → 乘 activation → 乘 scale
```

這條鏈的**指令數與 load-use 依賴數遠多於 Q8（直接讀 int8）/ F16（直接讀 half）**。decode batch=1 時，每 expert 由一個 threadgroup（2 simd 組 ×32 線程）處理，線程沒有足夠「已就緒的 load 請求」持續發給內存控制器去掩蓋這些 load-use 延遲 → DRAM 帶寬物理上有、有效帶寬卻腰斬。

**一句話：低比特壓縮的串行解壓依賴與 Apple GPU 的延遲層級不匹配——壓縮省下的字節時間，被解壓依賴鏈吃回去。這是 IQ2_S/IQ3_S 格式在單流 decode 下的固有性質，不是訪存重排可解。**

## 3. 對「通用引擎 decode 25+」的誠實判決

在**現成 IQ2_S/IQ3_S 格式 + 單流 decode batch=1** 下，expert kernel ~36–44 GB/s 是已被多面實測封死的硬約束，**沒有便宜的 kernel 工程解**。僅存的真實路徑（皆非樸素重排）：

| 路徑 | 內容 | 性質 | 預期 |
|---|---|---|---|
| **a. 格式–kernel 協同設計** | 為 Apple GPU SIMD/延遲結構設計新低比特編碼：展開無 gather、無數據依賴（固定位布局直接展開、或更優的一次寬讀+無依賴解包） | 核心研究、高價值 IP、通用引擎護城河 | 高風險、上界最大 |
| **b. 提高 decode 有效並行度** | 單 kernel 內把多 expert / 多 draft 的 GEMV 交錯（不改單流語義），用更多並發 load 掩蓋延遲 | 中等工程 | 上界有限（tokens 示 +21% 量級） |
| **c. 淺展開格式權衡** | 採用展開依賴淮的格式（Q4/k-quant 類），字節多但 kernel 能打滿，淨 t/s 可能更優；引擎按硬體選格式 | 量化格式選擇 | 需重新算 bpw↔帶寬淨帳 |

## 4. 這不是失敗，是收斂

- 用嚴格 A/B（每變體只動一處、其餘固定）把「便宜解」空間耗盡，避免再投入 TG/COOP 類改動。
- 回答了戰略問題：共有問題**確實不可迴避**，且其本質是**格式–硬件協同**層、不是調度小改——決定了引擎護城河要投在哪。
- 後續決策點（需掌舵人拍板）：是否啟動路徑 a（格式–kernel 協同設計的立項與探針），或先以 b/c 取得有限增量。

## 5. 複現

```bash
cd /Users/alexchuang/Documents/flashkv-devserver
# 四變體（每次只開一個 env）
CGC_IQ2_TGGRID=1 scripts/check/shape_probe/cgc_shape_probe --op mul_mat_id --type iq2_s \
  --k 2048 --out 512 --experts 256 --used 8 --tokens 1 --mode batch --iters 60
# CGC_IQ2_NOGRID=1 / CGC_IQ2_COOP=1 同理；對照去掉 env（baseline）
# 並行度：--tokens 1 vs --tokens 4
```

改動落點（未 commit）：
- ggml-metal.metal：iq2_s 模板加 `COOP_LOAD`、COOP inner 分支、id `_coop` 實例
- ggml-metal-device.cpp：`coop_load` flag、`CGC_IQ2_COOP` env、smem=768、base `_coop`
