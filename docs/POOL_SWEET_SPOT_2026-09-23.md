# Pool 甜點曲線 — 首次可引用讀數（2026-09-23）

## 一、OOM 根因修復（本輪最重要產出）

**現象**：重開機後，8/6/5/4/2GB pool + 各種手動 env 全部 Metal OOM
（`command buffer 1 failed status 5 Insufficient Memory`，rc=134），連 `-p 2048` 完整形狀也崩。

**根因（實測定位，非猜測）**：測量命令缺完整 prod-new profile env。
env 的唯一正確來源是 `run_server.sh` 的 `CGC_DUMP_ENV=1` 解析輸出（
`llama_bench_matrix.py` 明載此規則——env 不來自手寫清單）。
手動只設 4 個 env（MTP/SPAC/OA_ASYNC/MM_BITIDENT）漏掉：
`CGC_DBUF=1`、`CGC_N_CB=8`、`CGC_PREFILL_STREAM=1`、`CGC_GATHER_SLAB_CAP=256`、
`CGC_GLU_FUSED_DOWN=1`、`CGC_LOOP_GUARD=1`、`LLAMA_EXPERT_CACHE_ALLOW_NGL=1`、
`CGC_WAKE_POLL_US=15`、`CGC_EVICTED_RING=0`、`CGC_SERVER_AUTO_ANCHOR=0`、
`CGC_SERVER_DEFAULT_MARKER_STOPS=1`、`CGC_SPAC_ALPHA=0.75`、`CGC_SOFT_POOL_L0/L1=0`、
`CGC_WATCHDOG=1`、`LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0=0`、`LLAMA_EXPERT_CACHE_WORKERS=8`。

**證據**：
- 同一顆 llama-bench（09-23 21:22 build）+ 同一形狀（p2048 n128 d512 r1）
- 手動 4 env → OOM；完整 dump env → rc=0，prefill 280.5 / decode 13.25 t/s
- rho 背景預取已排除（需 `CGC_PREROUTER` 才觸發，本線未設）
- 物理記憶體已澄清非瓶頸：page size = 16384B（vm_stat 必須 ×16KB），free 8.7GB

**修復**：`scripts/check/pool_sweet_spot.sh` 內建完整 prod-new env 陣列（
與 commit_bench 同口徑：-p 2048、PREFILL_STREAM=1 + slab）。

## 二、甜點曲線首輪讀數（完整 env、每臂 r=3、交錯 8/6/5/4、臂間 90s）

| pool | decode t/s | prefill t/s | hit% | resident(MiB) | dswap(MB) |
|---|---|---|---|---|---|
| 8G | 9.98 | 259.7 | 96.0 | 6197 | +540 |
| 6G | 6.56 | 196.5 | 93.3 | 4637 | +257 |
| 5G | 7.99 | 262.5 | 91.4 | 3857 | −184 |
| 4G | 8.34 | 257.7 | 87.6 | 3077 | −56 |

**讀法（相對曲線可信，絕對速度受 thermal 污染）**：
1. hit 隨 pool 單調上升（87.6→96.0）、resident 隨 pool 上升（3.08→6.20 GiB）——符合 L1 預算直覺。
2. **decode t/s 不隨 pool 單調**：8G 不優於 4G（9.98 vs 8.34），6G 異常慢（6.56）。
   8G 只駐留 6.2GiB（SSD 上 44%），加大 pool 的 hit 收益被記憶體/熱壓力抵消的證據不足但方向可疑。
3. dswap：8G/6G 增加 swap（+540/+257），5G/4G 反而下降（−184/−56）——**大 pool 不必然省 swap**。
4. 絕對速度全低於 12.57/13.25 錨點：本輪為連續 4 臂（M4 Air 無風扇，thermal 已漂）；
   錨點是乾淨窗口 + ABBA。**要可比速度必須冷卻 + 交錯。**

## 三、待辦（下一步）

- [ ] 冷卻後 ABBA 交錯 4G vs 8G（各 r=3 ×3 對），判定「4G 能否達到 8G 速度」——
      若能 ⇒ 平行雙實驗（兩 agent 各跑 4G）效率翻倍成立
- [ ] 6G 異常點重測（可能污染：該臂 swap 起點最高 4511）
- [ ] 把「env 必須來自 CGC_DUMP_ENV」固化進 skill / NEXT_ACTIONS（
      這次 OOM 浪費了 4 次嘗試 + 重開機，全部可避免）

## 四、命令復現鍵

```bash
cd /Users/alexchuang/Documents/flashkv-devserver
CGC_SERVER_PROFILE=prod-new CGC_DUMP_ENV=1 ./scripts/run_server.sh   # 取 ENV block
bash scripts/check/pool_sweet_spot.sh                                  # 完整 4 臂
```
