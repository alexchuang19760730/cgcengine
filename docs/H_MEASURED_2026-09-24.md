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
