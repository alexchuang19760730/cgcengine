# 12.57 的 bit-identity 再确认 ＆ 「0907 knowhow 搬过来能否提升」

dated 产物 · 2026-09-22 · 0 重建（全部来自已落盘记录与 git 比对）
权威落盘：`Backup/m123_oracle_gate/summary_p2_rebased.json`、`oraclecmp_p2_rebased.json`

---

## §0. 一句话结论

| 问题 | 结论 |
|---|---|
| **1. 12.57 bit-identical？** | ✅ **实测已确认**：M1 = **9/9 (100%)**（13:51 那次 gate），且跑的那颗 dylib 与当前产物 **md5 相同** |
| **2. 0907 knowhow 搬过来能提升吗？** | ❌ **不能——它已经在里面了**。iq3_xxs glu kernel 与 0907 **212 行逐字节相同**；而 25.17 是**修 bug 之前**的数字 |

⚠ **我要更正自己前一份报告的一句话**：我在 `docs/FP_ORDER_SHAPE_2026-09-22.md:136`
写「我們從未驗證過 12.57 是不是 bit-exact」。**这句是错的**——同一天 13:51 已经跑过并落盘，
我在写那份报告时没查 `Backup/m123_oracle_gate/`。M1 是验证过的。

---

## §1. 问题 1：12.57 的 bit-identity（实测）

### 1.1 证据：今天 13:51 那次 gate

`Backup/m123_oracle_gate/summary_p2_rebased.json`（2026-09-22T13:51:15）

```
tag              : p2_rebased
ref              : Backup/knifeedge_matrix/ref_iq3_pool8gb_M2_6144_bitident_v6_nbaware.jsonl
ref_md5          : 72d82a33ad79e0e69bc935acd24228f2   (ref_pinned: true)
comparable       : true          config_diffs        : []
dump_records     : 9             coverage_pct        : 100.0
m1_numeric_identity    : 9/9     ← bit-identical logits
m2_decision_agreement  : 9/9
m3_topk_set_agreement  : 9/9
cross_tab        : {num_eq&dec_eq: 9, 其余 3 格: 0}
ok               : true
```

`cmp_p2_rebased.log` 的判定行：

```
VERDICT M1 (numeric identity)   : PASS (9/9)
VERDICT M2 (decision agreement) : PASS (9/9)
```

### 1.2 它跑的是哪颗产物 —— 与当前产物是同一颗

`summary` 的 `engine_digest` 记了 md5：

| 产物 | md5（前 16） | 记录 mtime |
|---|---|---|
| `libggml-metal.0.19.0.dylib`（13:51 那次跑的） | `430f9d315cbcf6c4` | 13:50:28 |
| **当前 `libggml-metal.0.19.0.dylib`** | **`430f9d315cbcf6c4`** | 14:01:39 |

⇒ **md5 相同**。当前这颗产物就是 13:51 通过 M1 的那颗（mtime 差只是重新 touch，内容一致）。
（当前 dylib 内 `sum_parts` 计数 = 10，即它含 P2-C revised 那 24 行。）

### 1.3 一个顺带被实测回答的问题

13:51 那次的 HEAD 是 `1f703c4d4`（p2/bit-identical-rebased，含 `sum_parts`），
而 ref v6 是 **09-17 13:45** dump 的（那时只有 P2-B、无 `sum_parts`）。

⇒ **含 `sum_parts` 的 build vs 无 `sum_parts` 的 ref ⇒ M1 仍然 9/9。**
即：**那 24 行在我们的 build 上确实没有改变 logits**——这是实测，不是论证。

我在上一份报告里从 fast-math（`ggml-metal-device.m:232` 的
`//[options setFastMathEnabled:false];` 被注释掉）推导出「只做 ① 拿不到 bit-exactness」。
**理论推导保留（它是对的风险提示），但实测结果是没飘。** 两条要一起记：
*理论上有权重排 ≠ 这个 kernel 上真的重排了。*

### 1.4 但这个确认的边界（必须一起讲）

| 能说 | 不能说 |
|---|---|
| 「相对 **09-17 v6 参考** 的 M1 = 100%」 | 「相对 **CPU 真值** 的 bit-exact」 |
| 「`sum_parts` 没改变 logits」 | 「P2-B 相对**无 P2-B 的上游**没改变 logits」 |

理由：

1. **ref v6 自己是 expert_cache ON 的 Metal dump**。它的 cap 原文：
   `"must never be used as ground truth (--oracle-abs refuses them)"`。
   ⇒ M1 是「两次 Metal run 逐步相同」，**不是**「与 CPU/上游对齐」。
2. **所有 ref 最早是 09-13 03:49，全部晚于 P2-B（`b8a564d45`，09-07）**：

   ```
   ref_iq3_pool8gb_STALE_cap6              2026-09-13T02:09
   ref_iq3_pool8gb                         2026-09-13T03:49
   ref_iq3_pool8gb_M2                      2026-09-14T16:13
   ref_iq3_pool8gb_M2_6144_STALE_pre-mmbitident  2026-09-15T01:13
   ...
   ref_iq3_pool8gb_M2_6144_bitident_v6_nbaware   2026-09-17T13:45   ← DEFAULT_REF
   ```
   ⇒ **手上没有任何一份「不含 P2-B/C」的参考**。要回答「P2-B 是否引入 FP 漂移」，
   **必须重建一颗去掉 P2-B/C 的 binary 再 dump 比对**，0 重建做不到。
3. 13:51 跑的是 `1f703c4d4`，不是当前 demo 分支 `b54b83e80`。
   demo 的 kernel = v6 ref 当时的 kernel（P2-B，无 `sum_parts`）⇒ 逻辑上必然也 9/9，
   **但没有单独跑过**，严格说仍是「待补一轮」。

### 1.5 补这一轮需要什么

① 重建（当前源码 demo 无 `sum_parts`、产物有 ⇒ 不一致，必须重建才能测 baseline 臂）
② 起 server 跑 gate（`BUDGET` 8 GiB + 13 GB 模型，在 16 GB 机器上本来就 OVERSUBSCRIBED）
③ 门禁：`window_sentinel --mem-gate` 当前 **usable 6.59 GiB < 8.0 → 硬拒**
   （唯一 ≥1 GiB 的进程是我们自己的 WorkBuddy Helper 1.18 GiB，杀不得）

⇒ **今天这口窗口补不了。** 列为待办，不阻塞结论（见 §0）。

---

## §2. 问题 2：0907 的 knowhow 搬过来能不能提升

### 2.1 逐项对照：它已经在树上了

| 0907 文档主张的 knowhow | 在我们树上的状态 | 证据 |
|---|---|---|
| **P2-B Vector Load**（32×float → 8×float4） | ✅ **已在，且是 12.57 基线的祖先** | `git merge-base --is-ancestor b8a564d45 c4e1c1778` = YES |
| **P2-B 的 kernel 本体** | ✅ **212 行逐字节相同** | `git show b8a564d45:...ggml-metal.metal` 的 `kernel_mul_mv_id_glu_iq3_xxs_impl` 与当前 `diff` → 无差异 |
| **P2-C** | ✅ 已在（`1f703c4d4` 是它的 revised 版） | 差异仅 17 个 `+` 行，且实测 M1 9/9 |
| SPAC（`CGC_SPAC=1`） | ✅ 已在 | ref v6 cap 的 ENV |
| DBUF（`CGC_DBUF=1`） | ✅ 已在 | 同上 |
| `draft_n_max = 3` | ✅ 已是默认 | 同上 / 0907 文档自述「預設」 |
| **MTP（spec）** | ✅ **12.57 那个 cell 本来就开着** | `prod_profile.py:29,113`：`--warm-skip 64 --spec-type draft-mtp` |

⇒ **可迁移的增量 = 0。** 「把 0907 的优化放到我们这里」这个动作**已经完成了，在 09-07 当天**。

### 2.2 那为什么我们是 12.57、他们是 25.17 —— 落差来自修 bug

0907 文档写：`25.17 t/s（coding profile，3 runs + warmup，seed 0）… 最佳 25.87，draft_accept 98.2%`

**三条独立理由说明这个数字不可引用：**

**① 我们自己在 09-16 就判过它「不该被当成结果引用」**（`.workbuddy/memory/2026-09-16.md:2052-2055`）

> 「…其中 **4 列標 NO**…以及 **25.17／27.71（舊日不同配置）**。**這四格不該被當成結果引用。**」

**② 25.17 量于 09-17 nb-aware 路由修正之前**（`8e9e830e5`，修的是
「修前每一臂都把 token≥1 路由到別的 token 的 experts」）。

09-17 14:3x 的**同 binary A/B**（build 14:12）把这件事钉死了：

| 臂 | accept | acc/gen | prefill t/s | decode t/s |
|---|---|---|---|---|
| MTP off（分母） | n/a | 0/0 | 19.24 | **9.82** |
| MTP on，`CGC_IDS_LINEAR_READ=1`（**修前**路径） | 73.81% | 155/210 | 7.30 | **8.62** |
| MTP on，nb-aware 修正（預設） | 58.25% | 180/309 | **21.74** | **12.62** |

同一份记忆明确写着：

> 「**修前記錄的每個 accept 數字都屬於那個 regime**（roadmap 的 19.9% 也是）」

⇒ 0907 文档里的 **98.2% accept 就是那个「被抬高」的 regime**；25.17 是在**错误路由**下量出来的。
修好之后的同一格 = **12.62**，而我们的封版 = **12.57**。**两者是同一个数字。**

**③ 那份文档的机器型号是错的**：文档自述「**M4 Max** 16GB…帶寬 ~100-120 GB/s…理論上限 ≈ 28.5 t/s」。
本机实测是 **Mac16,12 / Apple M4 / 8 GPU cores / 16 GB**。它的带宽与天花板估算建立在错误的型号上。

### 2.3 所以

**25.17 → 12.6 的落差 = 修正确性的代价，不是「没把优化搬过来」。**
把 0907 的东西再搬一次，得到的还是 12.57。

---

## §3. 待办（按优先级）

1. **【需要窗口】补一轮 demo 分支的 M1**：重建 → `m123_oracle_gate.py`（默认 ref = v6）。
   门禁当前拒（usable 6.59 GiB < 8.0）；8080 空、thermal NOMINAL，只差内存。
2. **【需要窗口】context-depth 干净重跑**：`context_depth_ab.py` 上一轮 14:55 掉到 HEAVY 被污染
   （`d=512` → 7.512/8.611，`d=0` → 9.988/9.300，4/4 分离但绝对值不可引用）。
   → 这一步回答的是「context 长度是不是杠杆」（约 +20%，不是 2×）。
3. **【0 重建，可做】** 若要回答「P2-B 相对上游是否 bit-exact」，唯一路径是重建一颗
   **去掉 P2-B/C** 的 binary 再 dump —— 当前无任何 ref 能回答，别再用推理代替。
4. **不建议**再花力气复现 25.17（理由见 §2.2）。

---

## §4. 本次用到的命令（可复现）

```sh
# ① 0907 kernel vs 当前 —— 逐字节
F=src/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal
git show b8a564d45:$F | awk '/kernel_mul_mv_id_glu_iq3_xxs_impl/,/^}/' > /tmp/old.txt
awk '/kernel_mul_mv_id_glu_iq3_xxs_impl/,/^}/' $F > /tmp/new.txt
diff /tmp/old.txt /tmp/new.txt          # 空 = 相同

# ② P2-B 是否 12.57 基线的祖先
git merge-base --is-ancestor b8a564d45 c4e1c1778 && echo YES

# ③ M1 判定（已有落盘）
python3 -c "import json;d=json.load(open('Backup/m123_oracle_gate/summary_p2_rebased.json'));print(d['m1_numeric_identity'],d['m2_decision_agreement'],d['comparable'])"

# ④ 产物身份
md5 -q src/llama.cpp/build/bin/libggml-metal.0.19.0.dylib
```

---

## §5. 附：12.93（+3%）这个优化有没有意义

### 5.1 它的唯一出处是一个 commit message，没有 raw

`git log -1 --format=%B da6903739`：

```
perf: P2-C revised - separate partial sums + fixed-order combine (bit-exact safe)
...
Result (server + MTP + DBUF+SPAC, 16GB M4 Max):
- baseline: 12.57 t/s
- revised: 12.93 t/s (+3%)
- P2-B+P2-C original: 25.17 t/s (bit-exact broken)
```

⇒ 全 repo（含 p2 worktree）grep `12.93` 只命中：这个 commit message、以及我们自己引用它的
review 文档。**没有任何 raw、臂数、reps、thermal/内存窗口记录。**

### 5.2 四个「不可引用」的理由

**① 跨仪器。** 它写的是 **server 路径**；我们的 12.57 是 **llama-bench**（`prod_profile.py`）。
两个 12.57 是**数字巧合，不是同一格**。我们自己的 server 路径实测（`server_path_ab.py`，
`CGC_DUMP_ENV=1` 证明两臂配置逐字节相同）：ours 干净臂 **7.79**、p2 臂 **5.72** —— 从未在
server 路径量到 12.57。

**② 12.93 落在 baseline 自己的离散区间里，出不去。**

| baseline 分布 | 区间 | 12.93 在内？ |
|---|---|---|
| 封版 12.57 ±2.26 | 10.31 – 14.83 | ✅ |
| 控制臂 n=7 中位 12.77（`MEMORY_PERF.md:764`） | **11.83 – 13.46** | ✅ |
| 单臂噪音底 ±27% | 9.18 – 15.96 | ✅ |

⇒ **+2.86% 连 baseline 自有离散都出不去**，更别提 ±27% 的单臂噪音底
（`MEMORY_PERF.md`：「同一個命令、同一次 session 讀 9.90 與 12.57」）。

**③ 低于我们自己的 3% 门槛。** `(12.93−12.57)/12.57 = **2.86%**` < 3%
⇒ 按本线规则（与 M-W 的判定同格：µs/miss −9.8% 确证、端到端 ≈+1.7%、低于 3% ⇒ 不可宣称）
**这个幅度不可对外宣称**。

**④ 机器型号是错的**：commit 写「16GB **M4 Max**」，本机实测 **Mac16,12 / Apple M4 / 8 GPU / 16GB**。
且同一个分支自己的产物给出 12.93 / 25.3–28.9 / 31.5（差 **2.4×**）⇒ 自相矛盾。

### 5.3 但「有意义」的那两层是真的

| 层次 | 判定 | 理由 |
|---|---|---|
| 作为**可引用的性能提升** | ❌ | 见 §5.2（跨仪器 / 噪音内 / <3% / 无 raw / 型号错） |
| 作为「**bit-exact 的 kernel 优化做得成**」的证明 | ✅ **有价值** | 但已被我们 **13:51 实测 M1 = 9/9 独立证明**（§1），**不需要引用他的数字** |
| 作为**通往 25 的路径** | ❌ | 本线已算（`P2_BITIDENT_REVIEW_2026-09-22.md` §12，原引
`P2BITIDENT_REVIEW_2026-09-22.md:332`，2026-09-24 去重後改指本處）：即便完美移植，落地 `12.57 × 1.054 = **13.25**`；与已指名杠杆总和 +5~10%（12.57 → **13.2–13.8**）同量级 ⇒ **不是 25 的路** |
| 作为「**要不要顺手做**」 | ⚠ | 与 M-W 同格：**零风险可顺手，但不可宣称收益** |

### 5.4 一个必须一起说的事实：M1 = 9/9 ⇒ 它在数值上是 no-op

§1.3 已实测：含 `sum_parts` 的 build vs 无 `sum_parts` 的 ref ⇒ **M1 = 9/9**。
⇒ 这 24 行**唯一可能的收益就是速度**；而速度增量（+2.86%）在噪音内。

### 5.5 真要判定它有没有提速，别在 t/s 上测

- t/s 单臂噪音 ±27% ⇒ 3% 效应需 **~50 臂/侧**（由 1.7% 需 ~165 臂换算），不现实。
- `gpu_union` 比 t/s 灵敏（0.2% vs ±27%），但 **`MEMORY_HYGIENE.md:143` 明写只对 prefill 配对成立**，
  decode 逐步的 `gpu_union` 臂内 p90/p10 = **2.27×** ⇒ 也不可用作这里的主指标。
- **正解**：先用 microbench（`autotuner/m1-shape-inventory` 的 `m1_harness.py`，量 µs/dispatch）
  单独测这个 kernel；再量它在 decode 步时里的占比 X%。
  若端到端真 +3%，则 kernel 本身提速 = `3/X`% —— **X 越小越值得做**。
  ⚠ 记忆警告：shape harness 的产出**仍须与 3% 门槛比**，且不能外推端到端。
