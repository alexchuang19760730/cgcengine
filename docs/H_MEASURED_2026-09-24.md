# H 實測 + UNTAGGED 拆分（2026-09-24）

## 一、量 h（白皮書 §10 第 1 步）——完成，A 方案判死

### 方法
- 在 `llama-context.cpp` ρ block 加純儀器（三處，未 commit、不改行為）：
  - 跨 step 累計器 `s_rho_prev_uni[ul]`：每層存「本步真實 union」，下一步比對
  - `h_step = |prev_uni ∩ uni| / |uni|`（A 預指派的跨 token 命中率）
  - `CGC-RHO-SUM` 行新增 `h_step` 與 `h_layers`
- 環境：`CGC_SERVER_PROFILE=prod-new CGC_RHO_PROBE=1 CGC_PREBIND_PROBE_VERBOSE=1`，decode 24 token（temperature 0）

### 實測（server log Backup/cgc_logs/llama_server_20260924_015257.log）
```
CGC-RHO-SUM: steps=26 layers=1041 skip=0 rho_tok=0.8225 cov_uni=0.8243 h_step=0.3732 h_layers=1001
```

### 判決
| 量 | 值 | 白皮書門檻 | 結論 |
|---|---|---|---|
| **h_step（跨 token 預指派）** | **0.373** | ≥0.65 才值得 A（+16%） | **A 方案判死**：預期僅 ~13.6 t/s（+8%），不達門檻 |
| cov_uni（同一步提前 submodule） | 0.824 | ≥0.398 即 −10% | 覆蓋率足夠（但 ρ-batch 已因 DRAM 競爭判死，見下） |
| rho_tok（逐 token 重合） | 0.822 | — | 路由本身高度可預測 |

### 關鍵區分（這次量測分開了兩個口徑）
- **h = 0.373**：路由的**跨 step 時間局部性弱**（每步 top-8 大變）→「用上一步的 union 預綁 slot」買不到覆蓋率
- **cov_uni = 0.824**：路由的**同 step 空間局部性強**（殘差算出的近似 top-8 與真實高度重合）→ 覆蓋率不是問題
- ⇒ 白皮書 §10 的「先量 h」已閉合：**h=0.37 < 0.65，A（speculative slot binding）不值得做**。真正有覆蓋率的方向是 ρ（同一步提前），但它卡在 fill 的 DRAM 頻寬競爭（ρ-batch 5 臂淨負，已判死）——即「覆蓋率有了、搬不動」。

### 順帶
- llama-bench 直接跑 ρ probe 會 Metal OOM（影子節點 + 8GiB pool 推爆 working set 11.45GB）——**ρ probe 只能在 server 路徑（prod-new）跑**，bench 不可用。

## 二、拆 untagged 7970MB = pool vs dense —— 完成

### 方法
prod-new profile 兩次 load（同一 binary、只改 expert-cache 大小），`footprint -p` 直抓：

| pool 上限 | footprint | untagged | MALLOC | mapped file |
|---|---|---|---|---|
| 8GiB | 8460 MB | 7970 MB | ~474 MB | 880 KB |
| 4GiB | 5335 MB | 4850 MB | ~459 MB | ~1 MB |

### 推導
- **untagged 差異（4G→8G）= 3120 MB = pool 增量**（需求 >> 4GiB，4G 池被填滿 4GiB；8G 池配到 ~7.4GiB）
- **dense + 固定 ≈ 4G untagged − 4GiB pool = 4850 − 4295 ≈ 555 MB**（dense Metal + KV + 固定 overhead）
- **untagged 幾乎全是 pool**（8G 時 ~93% 是 pool）

### 對 SWAP 結構解的意義
- 模型 13.6GB 的**實際駐留 ≈ dense 555MB + pool（≤8GiB）**——13.6GB 從未全量駐留（L4 shrunk 143/256 + adopt 已證）
- **swap 壓力最大項 = pool 保留本身** ⇒ L1（pool 甜點）就是結構解：8G→4G footprint 省 3.1GB、swap 全程不升反降（5140→4533MB）
- dense 不是大頭（555MB），**「dense 常駐幾 GB」的舊推導作廢**

## 三、狀態
- llama-context.cpp 三處儀器改動：**未 commit**（純量測儀器；h 的 SUM 行已驗證會印）
- server 已停、swap 4533MB、無殘留行程
- 下一步：h=0.37 判死 A ⇒ 白皮書 §10 方向更新；L1 甜點曲線（4/6/8G × hit × swap）仍需 6G 一臂

---

## 四、h 全中率口徑實測（2026-09-24 03:10，白皮書 §10 第 1 步「量 h」閉合版）

### 背景：freebuff 的口徑糾錯
白皮書 §6 定價 A 用的是「**全中率**」h = P(本步 union 全被上一步 union 覆蓋)，不是覆蓋率 cov。
兩者可同時是 0.85 和 ~0：每次固定錯 ~2.8 個 expert（cov=0.854 而 h=0）。
硬上界：uni_avg≈19.4、cov=0.854 ⇒ 平均未覆蓋 2.83 ⇒ h ≤ 0.854，不足以定案；
**A 要贏過 ρ 上界（14.39 t/s）需 h ≥ 0.75**——這是唯一還沒量的判據。

### 儀器
llama-context.cpp :6260 塊加 `s_h_all` 計數器（h_step==1.0 的步數/總步數），
輸出到 CGC-RHO-SUM 的 `h_all=` 欄位。3 處改動、未 commit。

### 讀數（prod-new、MTP off、server 路徑、decode 128 tokens、swap 環境）
```
CGC-RHO-SUM: steps=94 layers=3761 skip=0 rho_tok=0.8581 cov_uni=0.8585 h_step=0.4318 h_layers=3721 h_all=0.0030
```
- **h_step = 0.4318**：上一步 union 平均蓋住本步 union 的 43%（時間局部性中等）
- **h_all = 0.0030**：**每步 100% 全中被上一步 union 蓋住的機率只有 0.3%**（3721 層樣本中僅 ~11 步）
- cov_uni = 0.8585、rho_tok = 0.8581（與先前 0.854 同量級，驗證可重現）

### 判決
- **A（speculative slot binding / 跨 token 預指派）徹底判死**：h_all=0.003 ≪ 0.75（贏 ρ 上界的門檻）≪ 0.65（原門檻）。
  跨 token 猜在語義上不可行——每次固定錯 ~2.7 個 expert，永遠有未覆蓋。
- **ρ 路線坐實**：cov_uni=0.859 已過 0.398 門檻且 >0.70 飽和點 ⇒ 覆蓋率不是瓶頸；
  剩餘槓桿是 insert 4.76 ms/步的實測與 cb 42–51 ms 的遮蔽（CB_DELIVERY_SETTLED 已定讞）。
- 與 freebuff 的口徑預測完全一致：「cov=0.854 而 h=0」——現在有實測。

### 順帶：CGC_IDSEQ_DUMP 不可達（freebuff 的儀器）
freebuff 的 CGC_IDSEQ_DUMP（llama-context.cpp :5464，A-gate 註解）放在 canonical
gather 塊之後，但**該函數在交付 build 下從未執行**：server log 與 bench log 的
CGC-CANON 計數皆為 0 ⇒ IDSEQ fopen 從不觸發、trace 永不生成（實測三次無檔案）。
要量 per-call ids 序列需移到真正執行的 topk hook 路徑；本輪用 h_all 計數器繞過。

### 狀態
- llama-context.cpp 儀器 3 處（h_all）未 commit；CGC_IDSEQ_DUMP 未 commit
- server 已停、無殘留行程
