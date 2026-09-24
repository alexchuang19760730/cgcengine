# L3 neighbour prefetch：撤回 2026-09-21 的翻案（R=1 重新判为净亏）

日期：2026-09-21（承接 `docs/L3_PREFETCH_REPRICED_2026-09-21.md`）
状态：**撤回** §EN-409 的翻案。原 `NEIGHBOUR_PREFETCH_VERDICT` 的「净亏」结论**恢复有效**。
全部为 0 重建、未写 C++、未起 server。

---

## 0. 一句话

昨天我用「1 MiB → 3 MiB 快 2.7–4×」把 R=1 的 IO 时间算成 **×0.70（−30%）**，据此翻案。
两个错误：**真实读尺寸不是 1→3 MiB**（expert 在档内是 3 支独立张量，R=1 只会把每次读从
~0.21 MiB 变成 ~0.64 MiB，永远拿不到 3 MiB），**而且那支探针本身在不同读法下自相矛盾 2.3×**
⇒ 它的绝对 GB/s 一律不可引用。在同一支仪器内部自比，0.32→1.29 MiB 吞吐**基本持平**
⇒ R=1 的时间比是 **×1.87–3.05**，即「多读 3 倍字节 ≈ 多花 3 倍时间」。翻案撤回。

---

## 1. 真实布局（GGUF 头，精确到字节，非估计）

`Backup/phase_decomp/L3/gguf_expert_layout.py` 直接解析交付模型
`Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf`（12.72 GiB）：

```
blk.5.ffn_gate_exps.weight   dims=[2048, 512, 256]  type=22   82.0 MiB  -> 0.3203 MiB / expert
blk.5.ffn_up_exps.weight     dims=[2048, 512, 256]  type=22   82.0 MiB  -> 0.3203 MiB / expert
blk.5.ffn_down_exps.weight   dims=[512, 2048, 256]  type=21  110.0 MiB  -> 0.4297 MiB / expert
                                                    one expert = 1.0703 MiB
MoE exps 张量共 120 支 = 40 层 x 3，占全档 85.1%
```

GGUF 的 `dims` 是**逆序**存放的，所以 `dims[-1] = 256`（专家数）是**最外层**轴 ⇒
**expert e-1 / e / e+1 的切片在同一支张量内是文件连续的**。这一点对 R=1 有利，成立。

但**三支张量彼此是独立 GGUF tensor**，偏移相隔上亿字节：

```
down  off=2391667712
gate  off=2507568128      (+115.9 MiB)
up    off=2596213760      (+88.6 MiB)
```

⇒ **R=1 永远产生 3 次读，每次 (2R+1) 倍大**，绝不是一次 3 MiB 读：

| | 每次 pread 的大小 | 次数 |
|---|---|---|
| 现在 R=0 | 0.3203 / 0.3203 / 0.4297 MiB | 3 |
| R=1 | 0.9609 / 0.9609 / 1.2891 MiB | 3 |
| R=2 | 1.6015 / 1.6015 / 2.1485 MiB | 3 |

**而且引擎实测是 `reads/miss = 4.96`**（三支 run：35115/7081、34767/6961、35025/7021，
`prewarm miss = 0`），即实际每次读只有 **~220 KiB**，比 3 段更碎（kind 枚举是
`0=gate 1=up 2=down 3=gate_up`，多出来的 ~2 次来源未定，见 §5）。
⇒ 我昨天测的 1 MiB 与 3 MiB **两端都不在真实区间**。

---

## 2. 探针本身不可信（这是这次最重要的发现）

同一尺寸（3 MiB 冷读、2 绪、同一档案）在四次测量里得到四个数：

| 测量 | 读法 | 3 MiB @2 绪 | 3 MiB @8 绪 |
|---|---|---|---|
| `ssd_headroom.py`（昨天） | `os.pread`（每次新分配） | 4.78 | 4.09 |
| `ssd_shape_grid.py`（今天） | `os.preadv`（复用 buffer） | 1.93 | 1.59 |
| `pread_vs_preadv.py` A/B，交替 | `os.pread` | **2.36** | — |
| `pread_vs_preadv.py` A/B，交替 | `os.preadv` | **5.46** | — |

A/B 是**背靠背交替**的（排除漂移），单次内 `pread 2.19–2.59` / `preadv 5.27–5.49` 极其稳定，
但两者差 **2.31×**；而同一 `preadv` 在 grid 里只有 1.93。

最可能的根因：**16 GB 机器上放 12.72 GiB 的档**，页缓存能装下大半；
`F_NOCACHE` 在这台机器上显然没有把读彻底隔离（昨天 seq-1MiB 量到 5.39 GB/s、
今天 preadv 3 MiB 量到 5.46 GB/s，都超过这台机器 SSD 的物理上限，只能是内存命中）。

> **结论（方法论）**：`F_NOCACHE` 探针在这台机器上**不能再用来引用绝对吞吐**。
> 昨天所有 GB/s 数字（含「1.19 / 1.80 / 1.68 / 1.44」与「4.09 / 4.78」）**全部作废**。
> 要拿可信的冷读数字，得先解决「12.72 GiB 档 vs 16 GB RAM」这个根本冲突
> （例如用远大于 RAM 的档，或换一台/外接盘），否则不存在干净窗口。

---

## 3. 还能用的部分：同一仪器内部的**相对**比较

`ssd_shape_grid.py`（`preadv`，同一 session 内连续跑完）给出的吞吐矩阵（GB/s）：

| size\\threads | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| 0.320 MiB | 0.767 | 1.167 | 1.557 | 1.626 |
| 0.430 MiB | 0.927 | 1.306 | 1.672 | 1.691 |
| 0.961 MiB | 1.292 | 1.610 | 1.845 | 1.647 |
| 1.289 MiB | 1.366 | 1.672 | 1.809 | 1.598 |

补充扫描（同仪器）：0.961→1.601→2.148→3.000 MiB 在 2 绪是 1.616/1.734/1.783/1.930，
在 8 绪是 1.677/1.605/1.588/1.587 —— **没有 3 MiB 的跃升**，昨天的 4.78 是仪器假象。

按 `t(R) = Σ_k ((2R+1)·s_k) / thr((2R+1)·s_k, N)` 定价：

| 并发 | t(R=0) ms/expert | t(R=1) ms/expert | **时间比** | 判決 |
|---|---:|---:|---:|---|
| 1 | 1.3622 | 2.5499 | 1.872 | COSTS |
| 2 | 0.9203 | 2.0600 | 2.238 | COSTS |
| 4 | 0.7010 | 1.8396 | 2.624 | COSTS |
| 8 | 0.6797 | 2.0693 | 3.045 | COSTS |

**字节 ×3，时间 ×1.87–3.05** ⇒ 读尺寸的收益基本抵消不掉多读的字节。
这与 `NEIGHBOUR_PREFETCH_VERDICT` 原本的「时间 ∝ 字节」一致 —— 我不该推翻它。

### 步进层（跨 run，只当量级）

以 ~166 miss/步 × 1.0703 MiB：demand IO ≈ 114–229 ms/步（随并发），R=1 ≈ 309–429 ms/步。
GPU 阴影 159.77 ms ⇒ **R=1 的 IO 装不进阴影**（194–268%）。
即便乐观取并发 4 的 117.9 → 309.4 ms，也远超 159.77 ms。

---

## 4. 判决

1. **撤回** `docs/L3_PREFETCH_REPRICED_2026-09-21.md` 的翻案（「R=1 = −30% IO、上界 +7%」）。
   `NEIGHBOUR_PREFETCH_VERDICT` §3.3 的「净亏」**恢复有效**。
2. **L3 的 R=1 背景预取：不做。** 它不是「未被证明」，而是**被实测否证**
   （在唯一可用的仪器上，时间比 1.87–3.05；且读尺寸收益被页缓存污染，无从翻案）。
3. **L3 天花板回到 0**（`docs/L3_WINDOW_ZERO_2026-09-21.md` 的 async-fill 窗口 = 0
   ＋ 本文件的 R=1 净亏）。`18.59 t/s` 与 `13.5 t/s` 都不是可达值。
4. **唯一还活着的零风险改动**是并发旋钮 `LLAMA_EXPERT_CACHE_WORKERS=8 → 2`（见 §6），
   与 prefech 无关，且**不依赖任何探针的绝对数字**（µs/miss 是 run 级自比）。

---

## 5. 未决（不影响判决，但别当成已知）

- `reads/miss = 4.96` 与「每 expert 3 支张量、kind 只有 4 个」不符。多出的 ~2 次 read
  来源未定位（draft/MTP 路径、wide path、或 cold-path 的 `fill_segments_concurrent`
  都各计一次 `n_reads`）。不影响判决：无论 3 段还是 5 段，**R=1 都只是把每次读放大 (2R+1) 倍**。
- `pread_usec / n_reads` = 13–17 ms/次，对 220 KiB 读物理上不可能 ⇒ 该计数器口径有问题，
  **不要用它做任何每读定价**（已列入技能坑清单）。

---

## 6. WORKERS 8 → 2 的 A/B（与预取无关，唯一还站得住的收益）

见 `Backup/phase_decomp/L3/workers_ab.py` + `workers_ab_pool.py`。主指标 `pool_wait_us / misses`
（run 级累加量，比 t/s 紧得多；t/s 单臂 ±27%）。

**做法**：thermal 一到 MODERATE 闸门就拒跑，单串长跑只拿得到 2–3 臂 ⇒ 改成分批
（`--rounds 2` × 3 批，批间冷却 150 s），共 **6 臂 w=8 + 4 臂 w=2** 有效。

| 臂 | us/miss | misses | t/s | thermal（臂末） |
|---|---:|---:|---:|---|
| w=8 | 612.5 / 631.9 / 646.9 / 656.6 / 689.9 / 698.0 | 6379–7231 | 7.55–11.36 | 1 臂 HEAVY、1 臂 MODERATE |
| w=2 | 559.9 / 574.5 / 580.3 / **726.5** | 5518–7030 | 9.09–12.11 | 2 臂 MODERATE |

**中位比值（w=2 vs w=8）**：

| 指标 | 比值 | 读法 |
|---|---:|---|
| `pool_wait_us / misses` | **0.884** | <1 = w=2 的填充更快（−11.6%） |
| t/s | 1.105 | 方向一致，但幅度在 ±27% 噪音内，不可引用 |
| misses | 0.986 | — |

- **方向**：6 次配对里 **5 次偏向 w=2**；唯一反向的那次（726.5）是**臂末已进入 MODERATE** 的臂。
- **撤销一个我昨晚的说法**：昨天看到 w=2 的 miss 数 −21.5%（5518 vs 7033），推想「填得更快 ⇒
  step-ahead 预取赶得上」。补轮后 **misses 比值 0.986，该信号没有重现**，是单次样本，**不要引用**。
- `fill_batch` 比值 1.068（w=2 反而略差），与 `pool_wait` 不同向 ⇒ 两个计时器口径不同，
  以 `pool_wait/miss` 为准（它是 run 级累加量）。
- **判决**：`LLAMA_EXPERT_CACHE_WORKERS=8 → 2` 是**方向可信、幅度待定**的零风险 env 改动
  （不改语义、不写 C++）。要在**交付 cell**（`prod_profile.py`，reps=3）上验一次才算数；
  本 A/B 用的是 `dispatch_price_v2` 的 cell，t/s 只有 7.6–12.1，与交付的 12.57 不同口径。

---

## 7. 复现

```sh
python3 Backup/phase_decomp/L3/gguf_expert_layout.py --layer 5      # 真实布局（0 重建）
python3 Backup/phase_decomp/L3/ssd_shape_grid.py --sizes 0.3203 0.4297 0.9609 1.2891 \
        --threads 1 2 4 8 --reps 3 --total-gib 1.0                  # 形状 x 并发（相对可比）
python3 Backup/phase_decomp/L3/pread_vs_preadv.py --size-mib 3.0 --threads 2    # 仪器自证不可信
python3 Backup/phase_decomp/L3/workers_ab.py --rounds 4 --arms 8 2               # 并发旋钮
```
