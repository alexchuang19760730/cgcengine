# expert IO / fill 轴结案（2026-09-25，**仅生产口径**）

> 本文只保留**生产口径**（`commit_bench.py` 的 cell）的结论。
> ⚠ 2026-09-25 凌晨那批**非生产口径**（`--prompt 0` 冷池 ＋ `--spec-type draft-mtp` MTP on ＋ `--batch 512`
> ＋ `--ctx-size 4096`）的数据**已全部作废并删除**，不进本文、不再讨论。若在别处看到那些数字，一律不要引用。

生产 cell（唯一权威，`python3 scripts/check/commit_bench.py --dry-run`）：

```sh
python3 scripts/check/llama_bench_matrix.py --arms prod-new --prompt 2048 --gen 128 \
        --depths 512 --reps 3 --ctx-size 0 --warm-skip 64
```

## 0. 一句话

**fill 的同步时间占 decode step 的 4.7%**（EBTIMER 直读，生产口径）；
**端到端影响待定案**（10:56 的 `nf2_nofill` 配对，reps=3）。
无论哪个，**上界都远低于"值得动 C++ 并 rebuild"的门槛**。

## 1. 几何：一个 expert 的三段在文件里**不连续**

`scripts/check/expert_file_layout.py`（selftest 7/7，只读 GGUF header）：

```
blk.0.ffn_down_exps  offset= 847,740,928  bytes=450,560
blk.0.ffn_gate_exps  offset= 963,641,344  bytes=335,872   gap = 115,449,856 B
blk.0.ffn_up_exps    offset=1,052,286,976  bytes=335,872   gap =  88,309,760 B
        useful = 1,122,304 B (1.07 MiB)   span = 204,881,920 B
        => merge cost = 182.6x extra bytes
```

⇒ 把三段合成一次 pread 要多读 **182.6 倍**字节 ⇒ **几何上判死**。
（`useful 1.07 MiB` 与实测 `bytes/miss ≈ 1.09 MiB` 吻合 ⇒ 一次 miss 就是读一个 expert 的三段。）

## 2. 物理：冷读是 device 限制，不是 per-call overhead

`Backup/phase_decomp/pread_cost_probe.py`（直接对抗，不拟合）：

| 场景 | 每 job | 折算 |
|---|---|---|
| pass1 冷读（1 绪） | **0.487 ms** | 760 MiB/s |
| pass2 热（page cache） | 0.021 ms | 17.6 GB/s |
| 8 绪冷读 | 2.013 ms | aggregate **1453 MiB/s** |

**pass2 ≪ pass1（23×）⇒ 冷读是 device 限制**⇒ 只有**减少字节**有用，
"更少更大的调用"不是这题的答案。

## 3. 「jobs/miss 17.5」的真相：分子含 **prefill 的 slab 流式读**

生产 cell 带 `-p 2048`，`CGC_PREFILL_STREAM=1` 走 whole-layer slab 流式读，
那批 `file_reads` **计入 `n_reads` 但不计入 `misses`** ⇒ 17.5 是**口径错位**，
不是"一个 miss 被切成 17.5 次"。

纯 demand 的切分是正常的 **约 3~4 个 job / miss**（＝一个 expert 的 3 个 segment，合并在几何上做不了更多）。

## 4. 合并逻辑**早已存在**

`fill_segments_pool`（`llama-expert-cache.cpp:3618-3676`）按 `(file_idx, file_offset)` 排序后，
把 **file-contiguous 的 run 合成一次 `preadv`**。实测输出写着
`(0.03 MiB/job as one contiguous run)`。
⇒「一次 pread 搬多个 expert」**已实现**，不是待做的改动。

**固定开销占 90%**：生产口径一个 job 共 **371 µs**（`fill_batch_usec / file_reads`），
其中传输 28 KiB @760 MiB/s 只需 **36 µs**。
⚠ 指标要用 `fill_batch_usec / file_reads`；`io_us_per_job` 是**跨 worker 累加**口径，会误导 30×。

⇒ 文件说的「瓶颈是每次 fill 的固定开销（pread syscall ＋ bg_cv 唤醒 ＋ 一次同步点）」**正确**。
但**可合并的机会已被几何锁死**（§1）：剩下的空间只有把 job 数从 3~4 压到 3，约 **−25~32%**。

## 5. 「fill wait 空转」：**已解决，且生产口径下未触发**

`Backup/pf_ab/wd_new/`（生产口径，09:20）：

```
read shape: jobs=82122 bytes=2481012736 (0.03 MiB/job as one contiguous run) us/job=30747
fill_wait_us=3396002   prefetch=0/0
slab fills: pool=30985.2 MiB disk=24484.8 MiB
```

- `as one contiguous run` ⇒ 每个 job 已是合并后的连续 run（`fill_segments_merged_serial` 生效）。
- **`prefetch=0/0` ⇒ 背景预取未发起** ⇒ 09-23 那个病态（8 万小读灌满装置、`fill_wait_us` 44 ms → 15.4 s）**没有触发条件**。
- `fill_wait_us` = 3.4 s，比病态值低 4.4×。
- 24.5 GiB 的 slab 流式读**就是 §3 里 17.5 的分子**。

## 6. 定案数字：fill 的**同步**时间 = **4.7%**

`Backup/nofill_prod/nf_fill.json`（生产口径，NOMINAL/NOMINAL，n=172）：

| 量 | 值 |
|---|---|
| decode | **11.793 t/s**（sd 0.217）⇒ **84.8 ms/token** |
| `CGC-EBTIMER` fill | **3.955 ms/step**（385 行；step ＝ 一个 token position 的 40 层） |
| **同步 fill 占 step** | **4.7%** |

⇒ **即使把同步 fill 完全消灭也只有 +5%。**

⚠ 但**同步时间 ≠ 端到端影响**：`nf_nofill` 用 reps=1 给过 `14.24 vs 11.79` ⇒ **端到端约 +20%**，
差额来自 fill 阻塞时 GPU 空转／IO 队列被占的**间接效应**，这些不出现在同步计时里。
⇒ **最终以 10:56 的 `nf2_nofill`（reps=3、同 build、同 cell、15~20 min 窗口）为准。**

## 7. 开关性质与自检

- **`CGC_EB_NOFILL=1`**：把 `fill_segments_pool` 变 no-op、`ok` 填 1 假装成功
  ⇒ **池里的字节根本没被写过，输出是垃圾**。**纯度量开关，永远不该进生产配置。**
- 自检通过：`nf_nofill` 的 EBTIMER ⇒ **fill 3.955 → 0.206 ms/step（−95%）**。
- ⚠ **env 前缀传的变量不进 json 的 `env` 字段**（那记的是 profile 解析值）
  ⇒ 验证开关是否生效要看**行为指标**，否则"没生效"和"没效果"长得一样。

## 8. 判定

| 项 | 判定 |
|---|---|
| **3b（合并 pread）** | **不做**：合并在几何上做不了更多（§1），剩余空间约 −25~32% 的 job 数 |
| **3c（per-expert 重算）** | **不做**：`STEP3_PER_EXPERT_RECOMPUTE` §3 已写死它**依赖 3b**；直接做＝白工 |
| **fill 生产口径重验** | **已完成**（§6 即为答案；端到端待 §6 的配对） |
| `CGC_PREFETCH_LEGACY_FILL` | **不需要**：它是回退开关（设了只会变差），而预取在生产口径 `0/0` |

## 9. 流程坑（本日新增）

1. **任何可能失败的 arm，stdout 一律写文件**（`> path/stdout.log 2>&1`），**不要 `| tail`**
   —— 会截掉 matrix 的 `!! arm exited rc=...`，那是唯一的死因线索。
2. `--workdir` **不自建目录**，且收尾时才读 log ⇒ 先 `mkdir -p`，否则跑完才 `FileNotFoundError`。
3. `llama_bench_matrix.py` 有**自动隔离闸门**：量具散度 >12% 会把 json quarantine 到
   `Backup/quarantine/`，原路径只留几百字节的指针文件 ⇒ **看到小 json 是被隔离，不是程序坏了**。
4. `fill_onpath_ab.py:decode_ts` 曾对**死在 decode 的满侧臂**取到 prefill 行，报 +2465% 的假 WIN
   ⇒ 已修（要求 `n_gen` 为真，selftest 8/8）。**看到几千 % 的 WIN 先怀疑取错行。**

## 10. 引用

- `docs/MW_WORKERS_VERDICT_2026-09-25.md`（M-W 配对）
- `scripts/check/expert_file_layout.py`（几何）
- `Backup/phase_decomp/pread_cost_probe.py`（物理）
- 日志 §EN-1330 ~ §EN-1338
