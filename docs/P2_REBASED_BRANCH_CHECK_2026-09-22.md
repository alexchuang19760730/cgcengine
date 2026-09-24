# `p2/bit-identical-rebased` 分支检查报告

日期 2026-09-22 · 检查者：本线（线A / ace，引擎层）· 全程只读（0 重建、0 起 server）

## §1 一句话结论

**分支上只有 kernel，其他一样都没有。**
`p2/bit-identical-rebased` = 我们的 `demo/sweet-spot-windows-fix` + **1 个提交**（1f703c4d4），
内容只有 `ggml-metal.metal` 的 24 行（净改动 8 行）。他 claimed 的 M1 harness、shape inventory、
70 combos、-DMTP_SUPPORT 重建、以及 run4 31.42 / run5 31.30 的 raw —— **全部不在这个分支上**。

而且：**他自己在 12:29 和 13:41 已经两次撤回 31.5**（"Fix previous non-reproducible 31.5 t/s claim"），
但 13:50 建的这个 rebased 分支**没有带上那两次更正**，只带了 kernel。

## §2 分支拓扑（实测）

```
demo/sweet-spot-windows-fix  b54b83e80 (09-21 09:17)
  └─ p2/bit-identical-rebased  1f703c4d4 (09-22 13:50)  ← 只有这一个提交
       subject: perf: P2-C revised - separate partial sums + fixed-order combine
       diff --stat: src/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal | 24 ++++++++-----
                    1 file changed, 16 insertions(+), 8 deletions(-)
```

- merge-base(`rebased`, `p2/bit-identical`) = **b8a564d45**（P2-B/P2-C unroll，commit 日期 09-07）
- `git merge-base --is-ancestor b8a564d45 demo/sweet-spot-windows-fix` → **YES**
- `git merge-base --is-ancestor b8a564d45 c4e1c1778`（12.57 冻结 profile 提交）→ **YES**
  ⇒ **P2-B（vector load + unroll）早就合进主线，我们的 L0 = 12.57 基线里已经含 P2-B。**
- `git merge-base --is-ancestor 505b3a851(p2 tip) rebased` → **NO**（p2 最新两个提交不在 rebased 上）

reflog 显示分支是 13:48–13:50 之間建的：先建在 b8a564d45，13:49:57 `reset` 到
demo/sweet-spot-windows-fix，13:50:20 才 commit —— 所以它是**重做**出来的，不是 merge。

## §3 对照他 claimed 的清单

| 他说的 | 在 `p2/bit-identical-rebased` 上？ | 证据 |
|---|---|---|
| revised bit-exact kernel（8 行 Metal） | ✅ **有** | `git diff demo..rebased` 唯一文件 |
| M1 harness（7 shapes × 10 knobs = 70 combos） | ❌ **没有** | `git ls-tree -r HEAD \| grep m1_` 无 `m1_harness.py`；工作目录 `ls` 也不存在 |
| shape inventory S001–S007 | ❌ **没有** | `shape_inventory.py` 不在 HEAD 树 |
| run4 31.42 / run5 31.30 的 raw | ❌ **没有** | 全库 `grep -rlE '31\.42\|31\.30'` 只命中旧档与我们自己的 review 文档 |
| -DMTP_SUPPORT 重建 | ❌ **没有**（也不需要） | 是 build 配置不是代码；我们这棵树的产物已有 3 个 `MTPDBG mtp_ctor` |
| 自我更正文档 / repro 脚本 | ❌ **没有** | `P2_BITIDENTICAL_REVIEW_20260922.html` 与 `decode_bench_repro.py` 不在 HEAD 树 |

**harness 实际在哪**：`scripts/check/m1_harness.py`(238 行) + `shape_inventory.py`(115 行) 在
`autotuner/m1-shape-inventory`（dae4dbc1c 00:57 / 1ea38f5be 02:58）**和** p2 自己的
`p2/bit-identical@505b3a851`（13:41 "bring over oracle gate harness"）—— **两条都不在 rebased 上**。

## §4 harness 再追一层：它根本产不出 t/s

`m1_harness.py` 的 docstring 自己写：目标是 **µs/dispatch**、输出 `shape_knob_bench.json`、
gate 在 "M1 bit-exact"、形状是 **S001..S024**（不是 7 个）、knob 是 **5 个 env**
（CGC_MM_BITIDENT / CGC_DC_NSG / CGC_MMV_NSG / CGC_MMV_NR0 / CGC_MMV_FUSE）+ 2 个待补
（CGC_MM_NSG / CGC_MM_NXPSG），且 "M1 sweeps ONE knob at a time"。

⇒ **"7 shapes × 10 knobs = 70 combos 跑出 31.42 t/s" 与这支 harness 的口径对不上**：
它量的是 µs/dispatch，不是 t/s，也不能外推成端到端 t/s
（旧结论：同一 run 内 shape 级隐含 26.51 t/s，实测端到端 12.601）。

## §5 他自己已经撤回 31.5（12:29 / 13:41，都在 p2/bit-identical 上）

- **7dcb4f1c9 12:29** "fix: rebuild with -DMTP_SUPPORT + reproducible benchmark"
  → 明写 *"Fix previous non-reproducible 31.5 t/s claim"*
- **505b3a851 13:41** "docs: honest review + bring over oracle gate harness"
  → 明写 *"Correct numbers: 25.4 t/s decode (not 31.5)"*、*"only 8 lines Metal, not 2.4x speedup"*、
  *"Correct hardware: MacBook Air M4 16GB (not M4 Max)"*、*"预期收益 +0-5%，不是 2.4×"*

他的 repro 实测（`Backup/decode_bench_repro.json`，3 reps + 5 warmup，MTP on）：

| cell | t/s | 单次 raw | accept rate |
|---|---|---|---|
| short context **195 tok** | **25.41** (mean) | 24.32 / 26.44 / 25.46 | **100%**（draft 6/6） |
| long context **694 tok** | **12.63** (mean) | 9.60 / 13.98 / 14.30 | 56.9%（62/109） |

## §6 仍然错的三处（他的诚实版里还有）

1. **硬件**：`sysctl hw.model` = **Mac16,12 / Apple M4 / 8 GPU cores / 16 GB**
   （MacBook Air 级）。b8a564d45 的提交讯息写 "16GB **M4 Max**"、baseline 23.88 t/s —— 那是错的机器，
   那组 baseline 数字不可引用。
2. **bit-exact 的论证是错的**：他说 "#pragma unroll 讓 compiler 重排 FP 累加順序"。
   `#pragma unroll` 只展开循环，**不**允许 FP 重结合（无 fast-math）。真正改变 bits 的是他自己的
   新写法：原本是 16 项**串行**链，新版拆成 4 组各 4 项再 `((p0+p1)+p2)+p3` —— 这是**换累加顺序**，
   一般情况**不** bit-exact（FP 加法不结合）。它保证的是**确定性**（fixed combine order），
   而原版本来就是确定的。⇒ "bit-exact safe" 目前**未验证**，要用
   `scripts/check/cgc_logits_oracle_compare.py` 实跑才算数。
3. **25.17 的来源**：`--ancestry-path b8a564d45..c4e1c1778` 显示 P2-B 之后第一个提交是
   `9dde528a5 2026-09-07 release: v1.0.0 production - CGC Expert Cache **25.17 t/s**`。
   ⇒ 25.17 是 **09-07 那条 release note 的短 context 数字**，与今天 repro 的 25.41 是同一个 cell，
   **不是**新收益，也不是我们 delivery cell（`-d 512`）的数字。

## §7 与我们的 12.57 怎么对齐（关键 reconciling）

- 我们的 **L0 = 12.57** 是 llama-bench `prod_profile`，`-d 512`，**无 MTP**。
- 他的 **12.63** 是 server 路径、**MTP on**、694 tok、accept 56.9%。
- 两个数**落在同一格** ⇒ 在他自己可比的 cell 上，这个 kernel 的端到端 delta ≈ **0**。
- 他的 25.41 是 195 tok + MTP 6/6 全接的 cell（speculative 把 t/s 抬上去），
  与 12.57 不是同一格，**不可相减、不可比**。
- P2-B 的收益**已经在 12.57 里**（§2 已证）⇒ 今天"新增"的只有这个 sum_parts 改动。

⇒ 与他自己的估计 "+0–5%" 一致，也与我们既有的门槛判断一致：
**要做就走配对 A/B（ABBA、150 s 冷却、主指标 avg_ts），别引用任何单一数字。**

## §8 现场状态（动手动脚前必读）

- ⚠ **工作目录 HEAD 已经被切到 `p2/bit-identical-rebased`**（不是我们原来的分支），
  kernel 改动**已在源文件里**。
- ⚠ **产物已经重建过**：`libggml-metal.0.19.0.dylib` mtime **09-22 14:01** >
  `ggml-metal.metal` mtime **13:50** ⇒ 当前产物**含 P2-C revised**。
  从这棵树起的任何量测都是"新 kernel"，要 A/B 必须重建另一臂。
- ⚠ **8080 上现在有 llama-server 在听（PID 81094）** ⇒ 按门禁规则，**现在不能重建、不能量测**。

## §9 建议

1. 给他的回话（可直接转）：**"rebased 分支只有 8 行 kernel，harness / raw / MTP 重建 / 你 13:41 的
   诚实版都还没上去。先把 `p2/bit-identical` 的 505b3a851（含 honest review + oracle gate）一并带过来，
   再谈数字。"**
2. 别再引用 31.42 / 31.30（无 raw，且他本人已撤回）。
3. 若要验证这个 kernel：等 8080 空出来 → 干净窗口（mem_gate）→ 配对 A/B，主指标 avg_ts，
   比的是 **3% 门槛**，不是 25 t/s。
4. bit-exact 用 `cgc_logits_oracle_compare.py` 实跑判定，**不要接受"改 FP 顺序 = bit-exact"的说法**。

## §10 他的 25.4（195 tok）有没有参考价值

### 先说这个数字本身不完整

`decode_bench_repro.py` 只记 4 个字段：`decode_tps = timings.predicted_per_second`、`draft_n`、
`draft_n_accepted`、`accept_rate` —— **没有记 `predicted_n`（生成了几个 token）**，而 `max_tokens=100`。

`draft_n` 是**整请求累计**（`server-context.cpp:3237`：`slot.n_draft_total += draft.size()`），
配合 `--spec-draft-n-max 3`，短 cell 的 `draft_n=6` 意味着整个请求只发了 **6 个 draft token**（约 2 次投机步）。
⇒ 两种读法差一个量级，而 raw 里**无法区分**：

| 读法 | 生成长度 | 25.41 的含义 |
|---|---|---|
| A | ~8 token（2 步 × 4） | 被每请求固定开销主导，**不是吞吐数字** |
| B | ~100 token，spec 几乎没启动 | 短 context 真 decode ≈25 t/s，但 MTP 没贡献 |

⇒ **缺 `predicted_n` 就不可引用**。要救这个数字：补记 `predicted_n`，并要求 ≥64 token。

### 逐项判定

| 用法 | 判定 | 理由 |
|---|---|---|
| 当我们的基线 / 跟 12.57 比 | ❌ | 不同仪器（server vs llama-bench）、不同 context（195 vs `-d 512`）、MTP on 且 100% accept |
| 当 kernel 收益的证据 | ❌ | 25.17 早在 09-07 release note（`9dde528a5`）就是这格的数；25.41 − 25.17 = **+0.95%**，噪音内 |
| 证明 25 t/s 达成了 | ❌ | 我们 P(25) <3% 判的是**交付 cell**；这格不是交付 cell |
| 证明 MTP 有用 | ❌ | 同一脚本长 cell accept 56.9% → **12.63 ≈ 我们的 12.57** ⇒ 在我们这个 context 上 MTP 是打平的（与我们既有"今天 MTP 是亏的"一致） |
| **作内部对照否证 2.4×** | ✅ | 同一脚本、同一机器、同一模型，只差 context/accept：25.41 vs 12.63 ⇒ **差异由 context＋accept 决定，不是由 kernel 决定** |

### 唯一衍生的可测假设（用我们自己的仪器）

context 长度 / accept rate 是不是一根我们**还没测过**的杠杆？
建议：llama-bench 同 run 内加一个短 context cell 与 `-d 512` 配对（同一仪器、ABBA、主指标 avg_ts），
而不是引用他的数。**在这台机器上"context 长度"这一维我们目前没有自己的数据。**

## §11 现场状态（14:30 更新）

- ✅ **已切回 `demo/sweet-spot-windows-fix`（b54b83e80）**，与 `cgcengine0907` 同步；
  `ggml-metal.metal` 的 `sum_parts` 已归零；未追踪的 docs（含本报告）全部保留。
- ⚠ **源码与产物现在不一致**：`libggml-metal.0.19.0.dylib`（14:01 建）里仍有 **10 处 `sum_parts`**
  ⇒ 二进制还是 P2-C revised。**下次量测前必须重建 baseline 臂**（且 8080 仍被占，见 §8）。
