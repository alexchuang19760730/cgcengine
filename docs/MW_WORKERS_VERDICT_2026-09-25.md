> ⚠ **非生产口径（`--prompt 0` 冷池 ＋ `MTP on` ＋ `--batch 512`）的数据已于 2026-09-25 全部作废并删除。**
> 权威结论只看 `docs/IO_AXIS_VERDICT_2026-09-25.md`（生产口径）。

# M-W（`LLAMA_EXPERT_CACHE_WORKERS` 8→2）生产口径判决（2026-09-25）

形状：完全采用 `python3 scripts/check/commit_bench.py` 的 cell（未改任何形状参数）

```
--arms prod-new --prompt 2048 --gen 128 --depths 512 --reps 3 --ctx-size 0 --warm-skip 64
```

产物：`Backup/mw_ab/{mw_ctrl,mw_w2}.json` + `wd_ctrl/ wd_w2/`
判据用 `scripts/check/pair_ab.py`（门限 15%）。

## 1. 结论：**未分离**。IO 层确实改善，但不传导到 decode

| | W8（默认） | W2 | 差 |
|---|---|---|---|
| **decode t/s** | **12.195** (sd 0.272) | **12.123** (sd 0.355) | **−0.6%** |
| prefill t/s | 260.41 (sd 3.70) | 277.42 (sd 5.05) | +6.5% |
| **us / job** | 30891 | 27454 | **−11.1%** |
| **us_per_miss** | 541303 | 461641 | **−14.7%** |
| bytes / job | 28 KiB | 31 KiB | |
| hit% / misses | 96.4 / 4668 | 96.2 / 4907 | |
| swap_growth | 3446 MiB | 2044 MiB | |
| thermal | NOMINAL ×154 | NOMINAL ×150 | 两臂皆可引用 |

`pair_ab` 输出：`no separation (< 15%)`。

## 2. ★ 最有价值的不是结论，是这个推论：**生产口径下 IO 只占 step ≈ 9%**

IO 成本真的降了（us/job −11.1%、us_per_miss −14.7%），**但 t/s 完全没动**。

设 IO 占 step 的比例为 x，则 t/s 变化 ≈ x × 11%。实测约 1% ⇒ **x ≈ 9%**。

## 3. ⚠ 流程坑：env 前缀传的变量不进 json 的 `env` 字段

两臂 json 的 `env.LLAMA_EXPERT_CACHE_WORKERS` **都写 8**，但 W2 臂的 `us/job` 掉了 11%
⇒ 实际子进程拿到的是 **2**（`llama_bench_matrix.py:387` 的 `run_env = dict(os.environ)` 继承），
json 里记的是 `resolve()` 回传的 **profile 解析值**。

⇒ **验证 env 是否生效要看行为指标（`us/job`、`bytes/job`），不要看 json 的 `env` 字段。**
（这条坑会让「开关没生效」和「开关没效果」看起来一模一样。）

## 4. 顺带：两个 cell 的 IO 形态差 9 倍

⇒ 生产 cell 之所以该用，不只是流程：它**噪声低 9× 且对内存压力不敏感**，是稳健量具。

## 5. 复现

```sh
python3 scripts/check/pair_ab.py \
  --a Backup/mw_ab/mw_ctrl.json:Backup/mw_ab/wd_ctrl \
  --b Backup/mw_ab/mw_w2.json:Backup/mw_ab/wd_w2 \
  --alabel "W8(默认)" --blabel W2
```
