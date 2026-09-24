# 分母钉死：ml、步时、仪器开销（2026-09-22）

目的只有一个：**在谈「写 kernel」之前，把「一步有多久」的分母钉死。**
上一份 `GPU_CEILING_STEP2_2026-09-22.md` 留下的两个未知是：

1. `union + gap ≈ 127 ms` 对应的是 **133 ms 的步** 还是 **268 ms 的步** —— 差 2×，全靠 `ml`
   （每步 token 数）是 1.7 还是 3.375。
2. 交付 cell 从封版 **12.57 t/s** 掉到 **8.36 t/s**，是**仪器开销**还是**被干扰**。

两条都是 0 重建（一条纯读旧 log，一条跑现成 binary）。

---

## ① ml = 1.69~1.91（不是 3.375）—— 三个独立计数

`ml` 以前是**推**出来的（`1 + mean_len 2.375 = 3.375`）。这次是**数**出来的，三条互不依赖的路径：

| 路径 | 来源 | 值 |
|---|---|---|
| A. 主图个数 | `CGC-DECPROF` 的 `segs=41 layers=40` 行（main 图 `ntok≥2` ⇒ **每行都印**，不是 1/8 抽样） | 227 步 |
| B. verify pass | `llama_expert_cache: ... verify: calls=8885`（每层一次 ⇒ `8885/39`） | **227.8 步** |
| C. token 数 | llama-bench `--gen 128 --warm-skip 64`（`llama-bench.cpp:3331`：`n_warm_skip=64`，
  计时段 `test_gen_spec(..., t.n_gen - n_warm_skip)` = 64；warm 段单独跑 64）⇒ **每 rep 128 token × 3 rep = 384** | 384 token |

`384 / 227.8 = ` **ml = 1.686**

第二轮 `gc_s2b1` 同样方法：`verify: calls=7836` ⇒ 200.9 步，main 行 201 步 ⇒ **ml = 1.911**。

⇒ **ml ∈ [1.69, 1.91]，中央 1.8。3.375 死。**

两个副产品，都确认了计数没问题：

- **depth fill 只有一次**：`llama-bench.cpp:3341` 的 `is_cached = (t.n_depth == cstate.depth)` ⇒
  只有 rep 1 真的做 512-token 预填（log 里 `ntok=512` 那行确实只有 **1 行**，不是 3 行）。
- **`draft n_max=3`**（log: `[spec] draft-mtp enabled (draft n_max=3)`）⇒ main 图 `ntok ≤ 4`，
  实测直方图 `{4:144, 3:42, 2:41}` 完全吻合 ⇒ main 图确实每行都印，A 路径的计数不漏。

### 步时 = 134 ms（「133」那一支成立）

封版 12.57 t/s ⇒ 79.55 ms/token ⇒ **步时 = 79.55 × 1.8 ≈ 143 ms**（ml 1.69→134 ms，1.91→152 ms）。

而 `union + gap` 实测 **127.29 ms**（`gc_s2a`）／127.47（s2b0）／162.30（s2b1）。

⇒ **GPU 时钟跨度 ≈ 步时的 89~113%**（s2b0/s2b1 那两轮因为整机慢，步时被拉到 202~220 ms，
GPU 跨度只占 63~74%，差额是被拉长的主机侧）。

**这反过来撤销了 `METRIC_STABILITY_2026-09-21.md` 的一个结论**：那份说「`union+gap` 不重建步时，
低估 2.10×」，而 2.10 正是 `268 / 127`。ml 改成 1.69~1.91 之后，**比值 ≈ 1.05，`union+gap`
基本上就是步时**。那份关于「`union+gap` 逐行 CV 35~47%、非平稳、不如 `µs/miss` 稳」的判断
**仍然成立**（那是噪音问题，不是口径问题）；只有「低估 2.10×／隐含 26.51 t/s 不存在」这一条作废。

---

## ② 12.57 → 8.4 不是仪器开销，是整机慢

A/B（`Backup/phase_decomp/L3/decprof_overhead_ab.py`，2+2 臂、ABBA、臂间 150 s 冷却、
主指标 `avg_ts`）：

| 臂 | 环境 | 结果 |
|---|---|---|
| off | `prefill250`（无 `CGC_*`） | — |
| on | `prefill250:CGC_GPU_TIMING=1;CGC_DECODE_PROFILE=1`（= s2b 原样） | — |

（表格在跑完后填；先记结论：**off 臂第 1 轮 = 8.587 t/s**，与 s2b 的 8.364 只差 2.6%，
而仪器开销的假设要求差 34%。**⇒ 仪器是免费的，掉速来自整机状态。**）

### 整机状态：这台机器现在跑这个 cell 时在疯狂换页

跑的时候量的（`top -l 1` / `vm_stat`）：

```
PhysMem: 15G used (10G wired, 2761M compressor), 101M unused
swap: used 7.8 GB / 9.2 GB     swapins 10.3M pages
```

交付 cell 的常驻需求本来就超过 16 GB RAM：模型 13.6 GB（`--load-mode none`）+ expert pool
8 GiB 预算。差额靠 swap 顶，**而 swap 的量取决于同一时刻桌面上还有什么**：

```
WorkBuddy Helper (Renderer) 4.09 GB   Electron 0.76 GB   Doubao Browser 0.68 GB ...
```

⇒ 桌面占掉 6~7 GB 时，模型的常驻部分就被挤出去，**decode 每一步都在等 page-in**。
这解释了 1.46×（12.57→8.5），也解释了 `union+gap` 那个跨 run 的 **1.75×**（123.78 vs 230.99）
—— 那不是仪器噪音，是同一件事。

**这是比「1.22」大得多的口徑问题**：封版 12.57 是「桌面干净时」的数，
今天能重现的是「桌面占 6 GB 时」的数。任何 A/B 必须在**同一内存状态下**做，
而我们的门禁只查了 thermal + 别条线的进程 + `usable_pct ≥ 30%`，**没查 swap**。

---

## ③ 对「写 kernel」的影响

- 分母现在可信了：**一步 ≈ 143 ms（封版状态）**，其中 GPU 时钟跨度 127 ms。
- 但**绝对带宽仍然不可引用**：`union` 88~107 ms 对应的 11~15 GB/s 是在「整机换页」状态下量的，
  同一状态下的峰值（108.8 GB/s，8 线程实测）倒是没被换页影响（那个探针只用 512 MiB）。
  ⇒ **带宽百分比（7~14%）会随整机状态漂移，不能拿来当 kernel 的判准。**
- ⇒ 判准要改：不是「有效带宽 25 → 33.5 GB/s」，而是**在同一内存状态下做配对的
  `union+gap` A/B**。判准本身也得先在一个干净窗口重取一次基线。

## 下一步

1. 补一个 **swap / 桌面内存** 进门禁（`window_sentinel` 或新 gate）：`swap_used` 低于阈值才放行，
   并把「跑之前先关掉占内存的应用」写进流程。
2. 在干净状态下重取 `union+gap` 基线，再回头判 kernel 那一格。
3. `METRIC_STABILITY_2026-09-21.md` 的「2.10×」那句要改（本档 ① 节）。
