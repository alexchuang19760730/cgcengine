# `p2/bit-identical` 声称 prefill 285 / decode 31.5 —— 证据审查（2026-09-22）

审查对象：worktree `/Users/alexchuang/Documents/flashkv-p2bitident`，branch `p2/bit-identical` @ `512920c22`
文件：`docs/PREFILL285_DECODE31_WHITEPAPER_20260922.html`（03:47 产出）

**结论：这两个数字目前不可引用。** 不是因为它们一定是错的，而是因为（a）它们没有留下任何原始读数，
（b）**同一个作者的 commit message 与白皮书互相矛盾**，且（c）测量细胞与本项目约定的交付细胞不同。
下面每一条都可查。

---

## 1. 这条分支实际改了什么

| commit | 内容 |
|---|---|
| `b8a564d45` | P2-B：GEMV vector load + loop unroll（**已经是本线 HEAD 的祖先**，我们的 12.57 就含它） |
| `bf17325c8` | `CGC_DC_NSG` env |
| `be0d1c5eb`~`0eb5fbd9a` | 交叉平台文档、`run_server.sh` env 透传 |
| `da6903739` | **P2-C revised**：separate partial sums + 固定顺序合并 |
| `193adba66` | 「prefill250 + decode31 config」—— **diff 只有两个编好的 dylib，没有任何配置文件** |
| `813d72d35` | 白皮书 |
| `512920c22` | 从 demo 分支复制 harness |

**唯一的源码增量（相对 P2-B）只有 24 行**，在 `ggml-metal.metal` 的
`kernel_mul_mv_id_glu_iq3_xxs_impl`（IQ3_XXS 的 MoE GEMV）里：把

```metal
float2 sum = {0};  #pragma unroll for (l...) { for (j...) { sum[0] += ...; } }
```

改成 4 个独立的 `sum_parts[l]`，最后按固定顺序 `sum_parts[0]+[1]+[2]+[3]` 合并。
**flops 不变、搬运的 bytes 不变（`qq` 读取一模一样）**，改的只是 FP 累加顺序 —— 这正是它被称为
「bit-exact safe」的原因，也是它**几乎不可能带来 2.5×** 的原因。

## 2. 三处裂痕

### 裂痕 1：同一个作者的 commit 与白皮书对不上（最硬的一条）

`da6903739`（02:56）的 commit message 自己写：

```
Result (server + MTP + DBUF+SPAC, 16GB M4 Max):
- baseline: 12.57 t/s
- revised: 12.93 t/s (+3%)          <- P2-C revised，本白皮书宣称的版本
- P2-B+P2-C original: 25.17 t/s (bit-exact broken)
```

`813d72d35`（03:47，51 分钟后）的白皮书写 **P2-C revised = 31.5 t/s**，并把 25.17 写成「original」。

一个只做「锁死累加顺序」的版本（严格地比 original **少**一点优化自由），宣称跑赢 original 25%，
而它自己的 commit 说是 12.93。**两者不可能同时为真。**

顺带：他们表里「P2-B only = 9.59」也无法独立成立 —— P2-B 已在我们 HEAD 里，健康机器上含 P2-B 的
同一细胞是 **12.57**（今天凌晨在 swap 86% 的脏窗口里我们才量到 8.36–8.59）。9.59 很像在脏窗口取的样。

### 裂痕 2：这条分支里没有留下任何原始读数

整个 worktree 里，09-22 02:00 之后新增的资料只有：

```
docs/PREFILL285_DECODE31_WHITEPAPER_20260922.html
scripts/{run_server.sh,check/*}
Backup/cgc_logs/llama_server_20260922_023349.log     <- 134 行
src/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal
```

那支 log 的内容是：server **载入完成、`listening on http://0.0.0.0:8080`、结束** ——
全文没有一个 request、一次 generation、一个 `timings`。
在全 worktree 搜 `31.5` / `285.2`，命中的文件**只有白皮书自己**。

⇒ **没有 reps、没有逐轮样本、没有中位数/极值、没有 thermal 标签。** 而他们自己的白皮书 §4.3
承认「连续测试后 prefill 从 76 掉到 14 t/s」——恰恰是最需要逐轮 thermal 的情况。

### 裂痕 3：仪器与细胞都不是约定的那一个

- 他们 §4.2 自述：`llama-bench` 在 expert cache 下会 `GGML_ASSERT(n_tokens_all <= cparams.n_batch)` 崩，
  于是「workaround: 用 server API 測」。
- 本项目自 2026-09-18 起的约定：**server-derived 的 t/s 不可进 t/s 表**（等同我们的
  `http_duo.py` / `decode_sweep.py`，只做机制诊断）。
- 细胞也不同：他们用 `scripts/check/decode_bench.py`——**约 30 token 的短 prompt、n_predict 160、
  取 server 的 `timings.predicted_n / predicted_ms`**；我们的交付细胞是
  `--prompt 0 --gen 128 --depths 512 --batch 512 --ctx-size 4096 --warm-skip 64 --spec-type draft-mtp`。
- 而他们自己的 harness docstring 写着：**「the KV-cache size alone moves decode tok/s by ~5x」**。
  拿 30-token prompt 的 31.5 去比 `-d 512` 的 12.57，正是他们自己警告的那件事。
- 他们自己的 §3.1 表也在承认这件事：context 256/512/1k/2k → decode **6.5 / 11.1 / 10.4 / 15.4**，
  §4.1 直说「長 context 6–15 t/s」。31.5 只在「短 context warm」那一格。

### 附带一条：白皮书标错硬体

白皮书抬头写 **MacBook Pro M4 Max 16GB**；本机实际是
`Mac16,12` **MacBook Air** M4 16 GB（10 核 CPU）。commit message 也写「M4 Max」。
若数字来自本机 ⇒ 抬头错；若来自别的机器 ⇒ 与我们的天花板分析根本不是同一个硬件。**必须交代清楚。**

## 3. 但「同一个细胞上有没有真收益」这个问题仍然没答案

有一件事对本线是**好消息**：白皮书列的整套 env，我们的 `prefill250` profile **本来就已经全部包含**——

| 白皮书列的环境变数 | 本线 profile |
|---|---|
| `CGC_EXPERT_CACHE_BYTES=8589934592` / `LLAMA_EXPERT_CACHE_ALLOW_NGL=1` / `WORKERS=8` | ✅ |
| `CGC_DBUF=1`、`CGC_SPAC=1`、`CGC_SPAC_ALPHA=0.75`、`CGC_OA_ASYNC=1` | ✅ |
| `CGC_N_CB=8`、`CGC_GLU_FUSED_DOWN=1`、`CGC_MM_BITIDENT=1` | ✅ |
| `CGC_PREFILL_STREAM=1`、`CGC_GATHER_SLAB_CAP=256` | ✅ |
| `CGC_VERIFY_DECODE=1`、`CGC_DRAFT_DECODE=1`、`--spec-type draft-mtp` | ✅ |

⇒ 他们声称的那套「完整设定」**本来就是我们的交付细胞**。剩下的差异只有三项：
**① 那 24 行 kernel ② 仪器 llama-bench vs server ③ cell 深度 512 vs 30-token**。

⇒ 若 31.5 为真，剩下的差异里唯一可能的贡献者就是那 24 行 kernel（配置两侧相同），
而它对搬送的 bytes 是零改动 ——
这会直接推翻我们「第二步：有效频宽不是瓶颈」那天花板结论，**值得认真验一次**。
但请注意：**我们自己在 ② 得到的读数（union 有效频宽 11.9–15.1 GB/s，仅峰值 7–14%）
也是在今天凌晨那个 swap 86% 的脏窗口里量出来的** —— 我们自己同样需要一个干净重测。

## 4. 现在这台机器还给不出任何可引用数字

```
mem_gate → REFUSE
  usable 0.95 GiB < 8.0
  swap 5290 / 6144 MiB (86%)
others(): pid 54744 —— 另一条线的 llama-server 已跑 40 分钟，RSS 9.07 GB
```

这正是今天凌晨把我们自己的交付细胞从 12.57 打到 8.36 的同一个状态。
⇒ **在这个窗口里跑出来的任何数字（包括那条分支现在正在跑的）都不能用来判断 31.5 的真伪。**

## 5. 建议的验证顺序（先便宜后贵）

| 步骤 | 做什么 | 成本 | 判准 |
|---|---|---|---|
| 0 | 等干净窗口：那条 server 收掉 → `sudo purge`（或 GUI 授权）→ `mem_gate` PASS | operator | 门禁 PASS 才继续 |
| 1 | **先把我们自己的交付细胞重测一次**（reps=3）→ 确认 12.57 复得回来 | ~10 min | 拿不到 12.5±1 就别谈别人 |
| 2 | 用**同一个 server harness**（他们的 `decode_bench.py`，有 `--rounds/--json`、逐轮 thermal）对着**两支 binary** 各跑一轮：我们的 vs p2 的。`p2` 的 llama-bench 已在 10:19 编好，**0 重建** | ~20 min | 同仪器下的**相对**倍数才有意义（绝对值仍不是交付口径） |
| 3 | 若 2 显示有差 ⇒ 归因：把他们的 `-b 6144 -ub 6144` 与 n_cb 等逐项套到**我们的** binary（env 都在，0 重建），看是 kernel 赢还是 config 赢 | ~20 min | 单个旋钮 ≥3% 才算数 |
| 4 | 只有 1–3 都成立，才谈把这 24 行合进本线 | — | — |

另外值得告诉那条线一件事（省重复劳动）：白皮书 TODO 里的 **#4「K2 冗餘 CPY 上界 ≤22%」与
#5「K3 融合上界 ≤11.5%」在本线已经结案** —— 零字节 CPY 成本 = 0（`ggml_is_empty` 就过滤掉，不 dispatch）；
K3 真值 117 节点/步 × 0.0195% = **2.28% < 3% 门槛**，已判不写融合 kernel。

## 6. 一句话

**白皮书写的是 31.5，同一个作者的 commit 写的是 +3%，而支持这两个数字的资料一份都没有。**
在干净机器上按第 5 节跑一遍之前，prefill 285 / decode 31.5 只能当线索，不能当结论 ——
包括不能拿来回推我们的天花板（P(25 t/s) 那张表暂时不动）。

---

## 7. 追加（11:2x，仍然 0 起 server）：`CGC_DUMP_ENV=1` 把「配置差」与「binary 差」分开了

`run_server.sh` 有一个 `CGC_DUMP_ENV=1` 模式：**只把解析好的 env／argv 印出来，不起 server、
不做任何 kill**（`run_server.sh:762-765` 的注释明确写了这条不能有副作用）。所以两臂的**有效配置**
可以 0 风险、0 内存代价地完整比对。做法：两支 `run_server.sh` 各跑一次，带上本线 `prefill250`
profile 的 env（`llama_bench_matrix.resolve` 导入，不手抄）。

### 7.1 结论先行：配置**逐字节相同**，差异只有两处，都在编译期

`diff` 出来的全部差异（除路径与 log 之外）：

| 差异 | ours | p2 | 影响 |
|---|---|---|---|
| `-DMTP_SUPPORT` | ✅ 有 | ❌ **没有** | 见 7.2，这条最大 |
| pool/mmap 修复（[防護 2e]） | ✅ 有 | ❌ 没有 | `load_mode=none` 下不触发，本次无影响 |
| `BIN` / `LOG` / chat template 路径 | 本 worktree | p2 worktree | 无 |

其余全部一致：同一个模型 `Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf`、
`port=8080 ctx=8192 ngl=99`、同一个 budget（`model 13030 MiB + pool 10240 MiB`，
OVERSUBSCRIBED by 6886 MiB —— **两臂一样超订，这不是 p2 特有的问题**）、
`n_cb=8 glu_fused_down=1(mm_fuse=off) watchdog=1 oa_async=1 load_mode=none`、
`layer_caps=40-40:256`、`MTP ON (draft-mtp, n_max=3, denseIQ4X=1)`。

⇒ 白皮书里那张 env 表**不是他们的发现**，是我们的交付 profile 抄过去的（commit `512920c22`
「feat: copy harness from demo branch」，10:20，连 `decode_bench.py` 都是字节相同的同一份文件）。
**「配置不同」这个解释已经排除。**

### 7.2 p2 的 binary 没有 `-DMTP_SUPPORT`（他们自己的启动器在警告）

跑 p2 的 `run_server.sh` 时，它自己印出来：

```
warning: MTP=1 但 libllama-common.0.dylib 沒有 -DMTP_SUPPORT 編出來的程式碼。
         MTP 路徑的 [CGC MTP fix] 與 qwen35moe 的 t_embd 都會被編掉，產物與原始碼不一致。
         重建：cmake -B src/llama.cpp/build -DLLAMA_BUILD_SERVER=ON -DCMAKE_CXX_FLAGS=-DMTP_SUPPORT
```

本线的同一份检查（`run_server.sh:354-379`）是静默通过的 ⇒ **我们的 binary 有，他们的没有**。
`libllama-common` 是 09:48 重编的，所以这不是「旧的忘了编」，是**他们现在的产物也没有**。

这件事的方向性我不能空口猜，但它至少说明：**31.5 是在一个「产物与源码不一致」的 MTP 路径上量出来的**，
而 MTP 恰恰是决定 decode t/s 的最大单项（我们实测 ml ≈ 3.375 token/步）。**这条必须先排除，
否则「24 行 kernel 带来 2.5×」和「编掉的 MTP fix 带来 2.5×」无法区分。**

### 7.3 撤回一条我自己的假设

我在这一节之前怀疑过「p2 的 `libggml-metal` dylib mtime 02:43 早于 kernel commit 02:56，
所以编出来的东西不含那 24 行」——**这条是错的，撤回**：`.metal` 源文件的 mtime 也是 02:43
（先写文件、02:43 建好、02:56 才 commit；git 记的是 commit 时间不是写入时间）。
所以 **dylib 里大概率确实含那 24 行**。判据仍以实跑为准。

### 7.4 他们自己的白皮书更新已经把答案写了一半

他们在 10:26–11:13 连着五个 commit 更新了白皮书，其中关于 MTP 接受率的实测：

| context | decode t/s | MTP accept rate |
|---|---:|---:|
| 短（195 tokens） | 20 – 29（白皮书头条取 31.5） | **100%** |
| 长（694 tokens） | **4.6 – 6.0** | 49 – 83% |

并且 11:13 那条写了测量方法：「**跑 10 次 warmup 讓 expert cache 熱**」才量 decode。

⇒ 三个数字放在同一张表上就很清楚了：

- **31.5 是「短 context + 池已热 + MTP 全接受」那个角落的数字**；
- **同一个作者、同一个 binary，在 694 tokens 上只有 4.6–6.0 t/s**；
- 而我们的交付细胞是 `-d 512`，**落在他们表的下面那一半**。

⇒ 所以「31.5 vs 12.57」根本不是 2.5× 的差距，而是**两个不同角落**。
真正要问的问题变成：*在 512 深度、同一个 binary 下，我们和他们差多少* —— 这正是第 5 节第 2 步在做的事。

### 7.5 一个附带的好消息（关于我们自己）

`[budget]` 那几行也顺手确认了两件事，都对我们有利：

- **「OVERSUBSCRIBED by 6886 MiB」是两臂共同的常态**，不是 p2 的 bug，也不是我们今天量到 8.x 的原因；
- 本线 binary **含** pool/mmap 修复（[防護 2e]），所以载入不会 SIGBUS —— 我记忆里那条
  「mmap 会 SIGBUS」针对的是更早的构建，跑 server 路径时不再成立（但 mmap 仍然不是杠杆：
  4096/5120/6144 三个宽度**存活 0/3**）。

---

## 8. 真跑了一次 server 路径（11:21–11:34）：6.32 t/s，而这次跑测自己证明了它不可引用

按第 5 节第 2 步做了，只是只跑了一臂。新脚本 `scripts/check/server_path_ab.py`：
两支 `run_server.sh` 各自起 server → 用 **p2 那份字节相同的 `decode_bench.py`** 打 →
`--rounds 5 --warmup 10 --n-predict 160`（他们的 recipe：10 次 warmup 把池跑热）→ 收 server。

### 8.1 读数

| | 我们的 binary（p2 的 cell） |
|---|---|
| decode t/s（中位 / min–max） | **6.32** / 4.48 – 7.02 |
| prefill t/s | **6.2**（≈ 40 token 用掉 ~6.7 s） |
| 逐轮 thermal | NOMINAL → MODERATE → **HEAVY** |
| 答案一致性 | ✅ 5/5 完全相同（temp=0，确定性没问题） |

### 8.2 为什么这个数字不能用

不是「偏低」，是**量到了别的东西**。prefill 40 个 token 要把 13 GB 权重过一遍，
6.7 s ⇒ 有效 **≈1.9 GB/s**；健康的机器这个数应该是 ~14.8 GB/s（他们白皮书自己写 prefill 285 t/s
⇒ 250 token / 0.877 s ⇒ 14.8 GB/s）。**差 7.8 倍，而且差在读取速度上** —— 那是分页，不是 kernel。

跑测的代价也量到了（这解释了今天所有 8.x 的来源）：

| | 起跑前 | 收跑后 |
|---|---:|---:|
| usable | 6.44 GiB | **0.70 GiB** |
| swap used | 3417 / 5120 MiB (67%) | **5924 / 7168 MiB (83%)**（macOS 还自己加了一个 swap 档） |

附带一条：**他们的「10 次 warmup 把池跑热」在这台机器上没有出现** —— 第 0–9 轮（warmup）
与第 10–14 轮（计分）落在同一个区间（4.23–6.98 vs 4.48–7.02），**没有任何上升趋势**。
所以「31.5 是热池数字」这个解释，至少在今天的机器上不成立。

### 8.3 但有一个数字值得停下来看：31.5 ≈ 我们的 GPU 地板 × ml=4

用我们自己的 `union+gap`（一步的 GPU 时钟跨度）反推「主机侧开销全部归零」时的 t/s：

| run | union+gap 中位 | ml=3.375 | **ml=4.0** |
|---|---:|---:|---:|
| gc_s2a | 125.56 ms | 26.88 | **31.86** |
| gc_s2b0 | 132.47 ms | 25.48 | **30.20** |
| gc_s2b1 | 165.12 ms | 20.44 | 24.22 |

**他们的头条 31.5 正好落在这三条的中间。**

也就是说：31.5 = 「`wait` + `cb` + `submit` 全部为零」＋「MTP 每一步全接受 4 个 token」。
这两个条件我们各自都量过：

- 主机侧不可归零 —— **L3 可遮窗口实测为 0**（MoE 永远在 segment 第一个 buffer，5811/5811），
  `gap` 不可消除；
- ml=4 只在他们的**短 context** 出现（他们自己写：195 token 时 accept 100%，694 token 时 49–83%）。

⇒ 这两个数字不是「两个不同的 kernel 性能」，而是**同一个算术地板的两种取法**：
我们的 18.59（ml=3.375、主机归零）和他们的 31.5（ml=4、主机归零）是同一件不可达的事。
（提醒：`union+gap` 这三个读数本身也是在今天的脏窗口量的，只用来看量级，不当交付数字。）

### 8.4 现在的判决

**p2 那一臂今天不跑。** 不是懒，是跑出来也不能用：两臂的可用内存差 6.44 GiB vs 10.61 GiB，
这个 confound 比想测的效应大；而且再跑一次会把 swap 从 81% 继续往上推。

顺序改成（等干净窗口）：

1. `mem_gate` PASS（usable ≥ 8 GiB **且** swap 有 ≥2 GiB 余量）；
2. 跑 **ours → p2 → ours**（ABA，不是 AB）：两臂 ours 一致才说明窗口稳，中间那个 p2 才可引用；
3. 判准：**两臂比值**，不是绝对值 —— 绝对值永远不是交付口径（server 路径 + 短 context + 热池）。

脚本已经就位（`--arms ours,p2 --pairs 1`，ABBA 顺序、臂间冷却、逐臂记 gate/thermal/swap）；
顺手修了两个 bug：stdout 解析（`decode_bench` 在 json 后面还会印一行 `appended ->`，
直接 `json.loads` 会得到 "Extra data"）与「量测失败时 server 不会被收掉」（现在 `finally` 里收）。

---

## 9. 干净窗口重跑同一臂（11:41–11:52）：+23%，但**prefill 那半不动**

窗口：gate PASS（usable 9.86 GiB / 61.6%，swap 3975/5120 MiB = 78%，无 hog，无其他 session）、
thermal 全程 NOMINAL、无残留 server。同一脚本、同一 cell、同一 recipe（warmup 10 → 计分 5）。

| | §8 脏窗口 (a) | §9 干净窗口 (b) | 变化 |
|---|---:|---:|---:|
| decode t/s（中位） | 6.32 | **7.79** | **+23.3%** |
| prefill t/s（中位） | 6.22 | **6.51** | **+4.7%** |
| thermal worst | HEAVY | NOMINAL | 修好了 |
| 答案一致性 | 5/5 | 5/5 | — |

**decode 那半恢复了 23%，prefill 那半几乎不动。** 这一条推翻了 §8.2 里我自己的解释。

### 9.1 撤回 §8.2：「prefill 6.2 t/s 是分页」是错的

§8.2 我拿「健康值约 14.8 GB/s」当基准，算出「差 7.8 倍，差的正是读取速度 = 分页」。
**那个基准是拿他们的 285 t/s 反推的，而 285 t/s 根本不是这个 cell 的数字。**

正确的对照是本线自己的 llama-bench `prefill250` cell（`-p 2048`）：实测 **188.12 / 212.59 / 222.40 t/s**
（`MEMORY_PERF.md`，三次独立量测）。把它和 server 路径放在同一张算术上：

| cell | token | 实测 t/s | 一次 prefill 的墙钟 |
|---|---:|---:|---:|
| `prefill250` (llama-bench) | 2048 | ~200 | **10.2 s** |
| short (server) | ~40 | 6.51 | **6.15 s** |

token 数差 **51×**，墙钟只差 **1.7×** ⇒ **固定开销 ≈ 5.9 s，边际成本 ≈ 2 ms/token（≈490 t/s 边际）**。

也就是说：**30-token 这个 cell 的 prefill 数字 99% 是固定开销，不是带宽**。
它既不是内存脏（干净窗口 +4.7%），也不是 p2 独有（两臂 budget 逐字节相同）。

### 9.2 那个 5.9 s 是什么（结构性，不是环境）

两臂的 `[budget]` 都印同一行：

```
靜態需求 23270 MiB（model resident 13030 + pool 10240） vs 實體 16384 MiB
OVERSUBSCRIBED by 6886 MiB：這個啟動只能在 macOS 記憶體壓縮 + swap 之上執行。
```

一次 prefill 要把 39 层全过一遍 ⇒ 13.0 GB 的 dense 权重必须被 fault 回物理内存，而物理内存
装不下它（缺 6.9 GB）。所以每次请求都要付一次 swap-in。**这是配置级的，purge 和散热都救不了。**

⇒ 推论：**285.2 t/s 的 prefill 不可能在这台 16 GB 机器上、用这个 budget、从这支 `decode_bench.py`
（30-token prompt）量出来**。40 token 要跑 285 t/s 需要 prompt_ms = 140 ms，而光是把超载的
dense 权重 fault 回来就已经是秒级。

### 9.3 所以 §8 那次「复现」真正的收获是什么

**复现成功了，但复现出来的不是他们的数字 —— 这件事本身就是证据。**

- 配方复现✅：同一份 `decode_bench.py`（字节相同，他们 10:20 从本线复制）、30-token prompt、
  `n_predict 160`、warmup 10 → 计分 5、server API 路径。
- 配置复现✅：`CGC_DUMP_ENV=1` 逐字节 diff，两臂相同（含同一行 OVERSUBSCRIBED by 6886 MiB）。
- **数字复现❌**：decode 7.79（他们 31.5）、prefill 6.51（他们 285.2）。

两个数字差 4.0× 与 43.8×，而且 **prefill 那个差得比 decode 还大一个数量级** ⇒
他们白皮书上的「285.2 + 31.5」不是一次测量的两个字段，是**两个不同 cell / 不同仪器的并置**
（prefill 大概率来自 llama-bench 或长 prompt，decode 来自这支短 prompt 的 server 脚本）。
这正好是他们 §4.2 自述「llama-bench 会 GGML_ASSERT 崩，改用 server API」之后没有统一口径的后果。

### 9.4 对「31.5 vs 12.57」这个对比的修正

§8.3 我说 31.5 ≈ 我们的 GPU 地板 × ml=4。现在要加一层：**31.5 也不是同一个 cell 的数**。
他们自己白皮书更新（§7.4）已经写了短 context(195) 20–29 t/s、长 context(694) 4.6–6.0 t/s，
而我们这次在 ~40-token context 上拿到 **7.79 t/s（我们的 binary）** —— 落在他们长 context 那半的
下缘，**不是** 2.5× 差距，是两个不同的角落加一个不可比的仪器。

---

## 10. p2 那一臂也跑了（11:49–11:59）：5.72 t/s，而且**两臂生成的不是同一个东西**

`--arms p2,ours --tag c`，同一脚本、同一 cell、同一 recipe。gate 在 p2 起跑前 PASS
（usable 10.73 GiB / 67.1%，swap 5215/6144 MiB = **84.9%**，刚过门槛）。

| 臂 | decode 中位 | prefill 中位 | predicted_n | thermal worst | swap @launch |
|---|---:|---:|---:|---|---:|
| **b-ours**（干净窗口） | **7.79**（7.23–8.19） | 6.51 | **115** | NOMINAL | 78% |
| **c-p2**（他们的 binary） | **5.72**（5.55–6.01） | 8.45 | **160** | HEAVY | **84.9%** |

### 10.1 这个比值不能引用，三个 confound 全部同向

| confound | 方向 |
|---|---|
| swap @launch 78% vs 84.9%（跑完 87%） | 对 p2 不利 |
| thermal NOMINAL vs HEAVY（9/30 步） | 对 p2 不利 |
| **生成的 token 数 115 vs 160、内容类型不同** | **不是同一个测量** |

第三条是致命的，而且不是环境问题：

- ours 的 sample：`巴黎成為法國首都並非單一事件所致…`，115 token 自然结束（EOS）；
- p2 的 sample：以 **`<think>`** 开头，跑满 160 token 仍在 thinking 中间。

两臂的 chat template **是同一个文件**（`Qwen3-nothink-ChatML.jinja`，两臂 log 都印了同一路径名）、
`reasoning=off format=none` 也相同 ⇒ 差异来自 **binary 本身如何处理这个模板 / 如何停**，
也就是编译期差异的又一个表现（§7.2 已确认他们的 binary 缺 `-DMTP_SUPPORT`）。

⇒ **5.72 vs 7.79 不能断言「我们比较快」**，只能说**在这台机器上没有任何 2.5× 的迹象，方向甚至是反的**。
幅度一律不引用。

### 10.2 p2 跑完后，ours 对照臂被门禁挡下 —— 而这次挡对了

```
--- pair 0 arm ours | gate REFUSE | thermal NOMINAL
memory: usable 10.83 GiB (67.7%)  anon 2.51  cached 1.18
swap:   5348 / 6144 MiB (87%)
refusing (use --override to run anyway and label the numbers)
```

注意 `usable 10.83 GiB` 看起来**很干净**，旧门禁（绝对值 `SWAP_MAX_MIB = 6144`）在这个状态下会
**放行**（5348 < 6144）。这正是 §9 之前那个待修项：swap total 会变（观测到 5120 / 6144 / 7168 / 8192），
绝对值门槛在 total 缩小时形同虚设。已改为按比例（`SWAP_MAX_FRAC = 0.85`，绝对值降为 ≥8 GiB 时的地板）。

### 10.3 汇总：三次实跑，结论一致

| 批次 | 臂 | 窗口 | decode | 结论 |
|---|---|---|---:|---|
| a | ours | 脏（usable 6.44 → 0.70 GiB） | 6.32 | 不可引用（环境） |
| b | ours | 干净（9.86 GiB，NOMINAL） | **7.79** | 可引用（本线在 p2 cell 上的读数） |
| c | **p2** | 84.9%→87% swap，HEAVY | 5.72 | 不可引用（环境 + 生成任务不同） |

**没有任何一次接近 31.5。** 而他们声称的 31.5 与 285.2，按 §9.1–9.2 的算术，
不可能从这支 `decode_bench.py` 的这个 cell 量出来。

### 10.4 下一步（才有可比性）

1. **换 `long` cell**（~512-token prompt）—— 30-token cell 的 prefill 99% 是固定开销（§9.1），
   换掉之后 prefill 那一栏才有意义；
2. **统一生成任务**：把 `n_predict` 与停止条件固定成两边都能自然结束的同一段文本，
   并在脚本里断言两臂 `predicted_n` 相同 —— 不相同就作废（这是 §10.1 的教训，要写进断言而不是靠人看）；
3. **等 swap 降下来**（换出页是黏的，只能关掉持有者或重启）再跑 ABA。

---

## 11. 追加（12:1x）：「MTP enabled」的答案是**参数有、编译期没有**，实测 3 vs 0

那条线在 02:5x 问「mtp 有 enabled 嗎」，回答是看 server process 有 `--spec-type draft-mtp
--spec-draft-n-max 3` ⇒ ✅。**这个推论在 fork 里是明确被禁止的**，写检查的那个人把理由留在了
`scripts/run_server.sh:354`：

> `[2026-09-16] MTP_SUPPORT 是**編譯期**開關，不是執行期選項，所以「MTP 跑得起來」推不出
> 「MTP 是照原始碼那條路在跑」。少了 -DMTP_SUPPORT 時照樣會載入 draft context、draft acceptance
> 照樣可以是 1.00000，但被編掉的是三處 [CGC MTP fix]（M-RoPE 的 seq_rm、逐列安全讀取）與
> src/models/qwen35moe.cpp 的 res->t_embd 發佈。`

实测（判准用他们自己的字串 `MTPDBG mtp_ctor`，`strings -a libllama-common.0.dylib | grep -c`）：

| dylib | `MTPDBG mtp_ctor` | mtime |
|---|---:|---|
| **本线** `flashkv-devserver` | **3** | 09-20 23:05 |
| **p2** `flashkv-p2bitident` | **0** | **09-22 09:48**（最新） |

而 p2 那一臂的启动日志（我们 c 批 `launch_p2.log`）第 2–4 行与第 28 行是**并排**的：

```
2: warning: MTP=1 但 libllama-common.0.dylib 沒有 -DMTP_SUPPORT 編出來的程式碼。
3:          MTP 路徑的 [CGC MTP fix] 與 qwen35moe 的 t_embd 都會被編掉，產物與原始碼不一致。
28:[mode]  MTP ON (draft-mtp, n_max=3, denseIQ4X=1)
```

⇒ **第 28 行的「MTP ON」是启动器印的意图，第 2 行的 warning 是产物的事实。两者同时为真。**

这也解释了 §10 里那个 5.72 vs 7.79：我们那一臂的 MTP 是真的（3 hits），他们那一臂的 MTP 是被编掉的
（0 hits），而 MTP 是 decode t/s 的最大单项（本线 ml ≈ 3.375）。

### 11.1 31.42 / 31.30 没有任何落盘证据

全库搜（排除 `build/`、`versions/`、`CGC-main/`）：

- `31.42` / `31.30` 只命中 `CGC-main/.../gate_test_report_20260701_*.json`（是 `duration_ms`，
  2026-07 的旧报告，与本次无关）与 `llama-bench/README.md` 里的 `131.42`（上游文档）。
- **`scripts/check/` 里只有 11 个文件**；2026-09-22 02:00 之后新增的 `.py` 只有三个：
  `llama_bench_matrix.py`、`decode_bench.py`、`thermal_pressure.py` —— 都是从本线复制过去的。
- **「M1 harness」「shape inventory S001–S007」「7 shapes × 10 knobs = 70 combos」在该 worktree 里
  搜不到任何文件**（`grep -rl "knob"` 命中的都是文档与 `run_server.sh`，没有一支 harness）。

⇒ 那两个数字是**会话里声称的，没有落到任何可复现的脚本或 raw**。

### 11.2 「cold start → warm cache」解释不了 2.43×

那条线把 12.93 → 31.42（**2.43×**）归因于「之前是 cold start，现在是 warm cache（expert cache 热了）」。
本线三条独立实测都与这个量级冲突：

1. **我们今天在同一 cell 上跑过 warmup**：warmup 10 次 vs 计分 5 次，4.23–6.98 vs 4.48–7.02，**无趋势**
   —— 热池这件事在我们这台机器上根本没出现。
2. **pool_wait 只占步时 16.7%**（`M_W_DELIVERY_VERDICT`）⇒ 即使填池 IO 完全归零，端到端上限也只
   ≈ +20%，不是 +143%。
3. **L3 可遮窗口实测 = 0**（MoE 永远在 segment 第一个 command buffer，5811/5811）⇒ 结构上没有
   地方可以把 fill 藏起来。

### 11.3 但有一个数字是一致的，而且它才是可信区间

他们自己的表：**baseline 12.57 / revised 12.93（+2.9%）**。
本线封版的交付数也是 **12.57**，本线 M-W（WORKERS 8→2）实测 **≈+1.7%**。

⇒ 两条线在「慢的那一档」上**互相复现到 2% 以内**；分歧全部出现在「快的那一档」
（25.17 bit-exact ❌ → 31.42 无 raw）。**+3% 与 +1.7% 是同一量级，2.43× 不是。**

### 11.4 关于「做一个 shape harness 判定 shape→tps」

方向对（本线也建议过形状驱动的自适应调度表），但要先把两个陷阱写进设计：

1. **shape-level 的 t/s 不能外推成端到端 t/s**。本线已证 `union+gap` 不重建步时（同 run 内比值 **2.10×**，
   隐含 26.51 t/s 而 llama-bench 实测 12.601）—— 单 kernel 的等效吞吐乘回去会得到一个不存在的数。
   正确用法是**同 shape 内的相对比较**（A/B knob），不做绝对外推。
2. **shape 优化的可见上限很低**。本线实测：一个 dispatch 的固定成本 **18.6 µs**，102 个 elementwise
   = 1.9 ms，**只占 GPU busy 约 1%**；K3（117 节点/步 × 0.0195% = 2.28%）低于 3% 门槛已结案。
   ⇒ harness 值得做，但它的产出要拿去跟 3% 门槛比，不是跟 25 t/s 比。

---

## 12. 2026-09-24 補：同一件事曾有第二份檔案，已歸檔（含唯一的「13.25」計算）

曾經存在 `docs/P2BITIDENT_REVIEW_2026-09-22.md`（09-22 **14:48**，23966 B）與本檔
（`P2_BITIDENT_REVIEW`，09-22 **12:17**，29558 B）兩份只差一個底線的同名檔案。
兩份互相都沒有提及對方 ⇒ **不是草稿／正本，是各自獨立寫成的兩份**。

2026-09-24 去重時保留**本檔**（`MEMORY_PERF.md` 引用的就是這份，也是
`docs/OWNERSHIP_REVIEW_2026-09-24.html` 判定 leave 的那份），另一份移至
`Backup/P2BITIDENT_REVIEW_2026-09-22.md.draft-09-22-1448`（`.gitignore` 第 396 行含 `Backup/`
⇒ 不會再出現在 `git status` 裡，但檔案仍在磁碟上）。

⚠ **但那份不是冗餘**：`13.25 = 12.57 × 1.054` 這條結論只存在於它裡面，而
`docs/BITIDENTITY_AND_0907_KNOWHOW_2026-09-22.md` 正是引用它。為避免留下指向不存在內容的
斷鏈，把那段計算原樣搬來：

> | 每步 token 數（=1+3·accept） | 3.95（98.2%） | 2.77（58.9%） | **1.43×** |
> | 步時（推導） | 157 ms | ~220 ms | **1.40×** |
> | **乘積** | **25.17 t/s** | **12.57 t/s** | **2.00×** |
> | 那 24 行 | 23.88→25.17 | — | **1.054×** |
>
> （兩臂的 12.57 t/s / 58.9% accept 取自同一顆 build 的閘門 run；步時是從 t/s 與 mean_len
> 反推，不是獨立儀器，所以只當比例用。）
>
> **kernel 是 2.00× 裡的 1.054×。** 就算那 24 行完美移植、且可達，落地值是
> **12.57 × 1.054 = 13.25 t/s**。那不是「衝到 25 的路」，那是又一格 +0–5% ——
> 與它自己另一顆 commit 量的 **+3%**（12.57→12.93）同量級。

⇒ **這是「p2 那 24 行不是通往 25 的路」最直接的數字**，請引用本節，不要再去引那個 Backup 路徑。
完整原文（含第 13.5 節「那 24 行在本線上的可執行性」）保留在歸檔檔裡。
