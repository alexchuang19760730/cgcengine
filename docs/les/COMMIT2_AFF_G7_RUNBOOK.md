# LES commit2（含 D v3）→ 撤 affinity → G7 冷页只读探针　一条龙 Runbook

> 适用：worktree `/Users/alexchuang/Documents/flashkv-devserver`，分支 `demo/sweet-spot-windows-fix`。
> 模型：Qwen3.6-35B-A3B（M4 Air 16GB，IQ3_XXS）。系统命名「Embodibrain」。
> 范围：**只到 G7（C 阶段1 只读探针）**。不含 G2/G3a/G6 的执行（那些是后续链路，见 `~/LES_APPLY_RUNBOOK.md`）。
> 语言：繁體中文。所有脚本「给之前已自验」，**失败不要自行改参数重跑，整份日志回传**。

---

## 0. 这条链到底回答什么（不混目标）

| 阶段 | 动作 | 回答的问题 | 谁做 |
|---|---|---|---|
| 1 | commit2（含 D v3） | 预算口径修复 + 只读探针 A0/A1/B0/IO_SHAPE/BHEAD/LES 干净落地，pre-commit replay 背书无退化 | 用户（一条命令） |
| 2 | 撤 expert-affinity（8 文件） | 把已判死 **−0.87%** 的 affinity 功能从工作区干净移除，为 G2/G6 让路 | **agent（交接点，不盲删）** |
| 3 | apply G7 + 编译 | 编入两个**只读、默认 OFF** 探针（几何 + evict 线程） | agent |
| 4 | G7 探针 8G 一格 | **Q1** evict 是否只在 build 线程（distinct tid=1）；**Q2** 真实页/slot 几何 | 用户（一条命令） |

**Q1 是阶段2 冷页归还（`MADV_FREE_REUSABLE`）的前置闸门**：
- distinct tid = 1 → pool_ext 受害者只在 build 线程（图提交前、持 cache->m），fence 安全窗口成立，可据以设计阶段2；
- distinct tid > 1 → 阶段2 上 FREEREUS 前**必须先补 GPU fence**，禁止直接改 `cgc_discard_pages`。
- Q2 用真实 per-kind stride 复核静态合成的「99.3% 页独占、一页最多跨 2 slot」。

**这条链不预设 25 t/s，不许诺任何加速**；G7 只读、不改变任何决策。

---

## 1. 全景阶段表（谁自动、谁要 agent）

| 步骤 | 执行者 | 命令 | 耗时 |
|---|---|---|---|
| 0 重开机 + 预检 | 用户 | `bash ~/les_preflight.sh` | 2 min |
| 1 commit2（含 v3） | 用户 | `bash ~/les_commit2_v3_run.sh` | 30–60 min（含 replay 3×3） |
| **交接点 A** | 用户回传日志 → agent | — | — |
| 2 撤 affinity | **agent** | 逐文件（不盲删），见 §4 | 由 agent 完成 |
| 3 G7 apply + 编译 | **agent** | 见 §5（编译用 CPU，不占 GPU） | 编译约 5–10 min |
| 4 G7 探针（冷却后） | 用户 | `bash ~/les_g7_probe_run.sh` | 约 10 min |
| **交接点 B** | 用户回传产物 → agent 判 Q1/Q2 | — | — |

---

## 2. 阶段 0：重开机 + 预检（用户）

```bash
# 1) 重新开机（让 swap used 归零；不要只靠关 App）
# 2) 手动退出 WorkBuddy、Chrome、FreeBuff（可省 2–3 GB 以上）
#    WorkBuddy 是 Tencent 软件，脚本只检测、不代退，请手动退出。
# 3) 只开「终端机」，跑只读预检：
bash ~/les_preflight.sh
```

预检通过标准：
- §1 swap used = 0 MiB（或 < 1024）；
- §2 thermal nominal（`No thermal warning` 属正常，脚本已正确识别，不再误报）；
- §3 无 WorkBuddy/Chrome/FreeBuff、无残留 llama-server；
- §4 此刻**仍会报 8 文件 affinity**——正常，撤 affinity 在阶段 2 才做，**预检在这里先不用归零**；
- §5/§6 此时 G2/G3a/G7 未编入、二进制无标记——正常。

---

## 3. 阶段 1：commit2，含 D v3（用户，一条命令）

```bash
bash ~/les_commit2_v3_run.sh
```

**为什么是 wrapper（`_v3_run.sh`）而不是原 `les_commit2_run.sh`**：
budget_gate.sh / budget_preflight.py 的 **D v3 最终口径**（Metal working-set ceiling 10308、fixed=1396、
narrow/wide、wide_headroom、pool 匿名不可回收）先前只在工作区、没进 staged；直接跑原脚本会把旧 budget 提交、
D 根因修复丢失。wrapper 会先把这两个文件的 v3 补进 staged（**刻意不 add run_server.sh**，其工作区增量是
NCMOE/affinity/spec-gate/verify/dqhash，commit2 不含），做完七项自检再 `exec` 原脚本。

> v3 已在本机预先 staged 并验证过（仍 6 档、staged 零 affinity、零 ncmoe）；wrapper 幂等，重开机后再跑安全。

wrapper 自做：
1. 核对 HEAD = `58abe27ad`；
2. 确认两个 budget 工作区真含 v3（`10308/wide_headroom/workload`）；
3. 安全网：budget 差异不得混入 affinity/ncmoe/freereus/mmid/dqhash；
4. `git add scripts/check/budget_gate.sh scripts/check/budget_preflight.py`（**不 add run_server.sh**）；
5. 校验 staged 恰好 6 档、v3 确进 staged、staged 零 affinity、staged run_server 零 affinity/ncmoe；
6. `exec bash ~/les_commit2_run.sh`：swap/thermal/App/残留 server 检查 →
   `ALLOW_ENV_GATED_BIN=1 git commit` → **pre-commit 自动 replay（3 profile × 3 runs，30–60 min，不可跳）**。

**commit 边界（关键）**：
- pre-commit 检查 8（tracked binary）用 `ALLOW_ENV_GATED_BIN=1` 放行；
- replay gate **不可跳过**，**禁止 `--no-verify` / `--no-sentinel`**；
- 探针/功能默认 OFF，replay 对 baseline 口径无影响；
- commit2 刻意不含：affinity、CGC Strata NCMOE、exact-cache-verify、mmid-dqhash；不提交 dylib。

**验收**：脚本结束 `git commit exit=0`、HEAD 前进（不再是 `58abe27ad`）。
**回传**：`~/les_commit2_<TS>.log`、`/tmp/cgc_precommit_compare.json`（若有）。

---

## 4. 【交接点 A】阶段 2：撤 affinity（agent 做，不盲删）

> commit2 成功后，**把日志回传、叫我（agent）**。这一步不是给你盲跑的脚本——
> affinity 是跨 8 文件、深度嵌入 graph 构建的一整套未提交功能（indirection table、residency guard、
> logical→slot LUT），与**要保留的 bhead/les s2c、pool 路径、slab/prefill 交织**，盲删会破坏图、编译不过。

agent 流程（每个文件先 `git diff` 完整核对 → 只删 affinity hunk → 逐文件备份 `.bak-aff-<TS>` →
撤完编译验证）：

| 文件 | 撤什么（标记） | 必须保留 |
|---|---|---|
| `scripts/run_server.sh`（MM） | `CGC_EXPERT_AFFINITY` / `CGC_EXPERT_AFFINITY_FILE` / `CGC_RESIDENCY_GUARD` 透传（白名单 1 处） | budget 口径、**NCMOE、spec-gate、exact-verify、mmid-dqhash**、G7 透传 |
| `scripts/check/decode_sweep.py` | affinity-off/on 臂 ARMS（on 臂 `CGC_EXPERT_AFFINITY=1`） | 其它 sweep 臂 |
| `ggml/src/ggml-metal/ggml-metal-ops.cpp` | `cgc_affinity`、`n_logical_plane`、logical→slot LUT、find_by_name 索引 | 其它 Metal 改动 |
| `src/llama-context.cpp` | indirection table publish、`cgc_logical_ids`、residency 调用 | **保留 bhead/les s2c 块（G2/G6 需要）** |
| `src/llama-expert-cache.cpp` | `active_residency_guard()`、`load_affinity_perm`、`affinity_perm`、`cgc_affinity_prepop` | 其它字段、pool 路径、G7 探针 |
| `src/llama-expert-cache.h` | `affinity_perm` 字段、`load_affinity_perm` 声明 | 其它字段 |
| `src/llama-graph.cpp` | `cgc_affinity` lambda、remap-leaf 分支 | slab/prefill 扩展、其它改动 |
| `src/llama.cpp` | `CGC_EXPERT_AFFINITY_FILE` getenv、`load_affinity_perm` 调用、witness | 其它启动逻辑 |

关键词：`affinity_perm` / `CGC_EXPERT_AFFINITY` / `CGC_EXPERT_AFFINITY_FILE` / `CGC_RESIDENCY_GUARD` /
`active_residency_guard()` / `cgc_affinity` / `cgc_logical_ids` / `llama_expert_cache_load_affinity_perm` /
`perm_all_layers.txt` / `cgc_expert_affinity`。
（所列行号是 commit2 前定位；commit2 落地后行号会漂移，撤的时候以现 grep 为准。）

**验收（agent 自检）**：
- `bash ~/les_preflight.sh` §4 报 **0 文件含 affinity**；
- 全仓 grep 上述关键词归零（除 `.bak-aff-*` 备份）；
- **context.cpp 的 bhead/les s2c 块仍在**；
- 撤完先单独编译一次（确认移除功能后无悬空引用）：
  ```bash
  export PATH="/opt/homebrew/bin:$PATH"
  cmake --build src/llama.cpp/build --parallel 8    # BUILD_EXIT=0
  ```

---

## 5. 阶段 3：apply G7 + 编译（agent）

撤 affinity 编译通过后，再上 G7（两个只读探针）：

```bash
cd /Users/alexchuang/Documents/flashkv-devserver
# 1) 先 dry-run（撤 affinity 改過 run_server.sh，行號可能漂移；context 塊 CGC_POOL_SPLIT_DBG 唯一，
#    正常會自動偏移；若 --check 失敗，把輸出回傳，不要硬上）
git apply --check docs/les/G7_STAGE1_READONLY_PROBE.patch
# 2) 正式 apply
git apply docs/les/G7_STAGE1_READONLY_PROBE.patch
# 3) 編譯
export PATH="/opt/homebrew/bin:$PATH"
cmake --build src/llama.cpp/build --parallel 8
```

**G7 补丁内容（3 hunk、2 文件，已 `git apply --check` rc=0、C++ 用真实编译旗标 `-fsyntax-only`
零新增 warning、run_server `bash -n` 通过）**：
1. `llama-expert-cache.cpp`：`cgc_discard_pool_slot` 开头加 `CGC_L4_EVICT_TRACE`（记录 evict 的
   pthread_self tid，16 槽、每线程前 3 次 + 每 2048 次打印 `CGC-L4-EVICT-TRACE`）；
2. `llama-expert-cache.cpp`：`pool_split_adopt` 的 `CGC_POOL_SPLIT_DBG` 分支加 `CGC-POOL-SPLIT-GEOM`
   （page / base_mod_pg / stride_mod_pg / slice_mod_pg / stride_pg），供 `l4_slab_geometry.py` 解析；
3. `scripts/run_server.sh`：白名单透传 `CGC_L4_EVICT_TRACE`（否则 env 被 allowlist 静默丢弃）。

两个 env 默认 OFF，不改变任何决策；配套文档 `docs/les/G7_STAGE1_READONLY_PROBE.md`、
解析工具 `docs/les/tools/l4_slab_geometry.py`（`--self-test` 已过）。

**验收**：编译 BUILD_EXIT=0；
`strings build/bin/libllama-server-impl.dylib | grep -cE 'CGC-L4-EVICT-TRACE|CGC-POOL-SPLIT-GEOM'` ≥ 2。

---

## 6. 阶段 4：G7 探针（冷却后，用户，一条命令）

编译完成后等 thermal 冷却（编译用 CPU，会发热），确认无 WorkBuddy/Chrome/FreeBuff，然后：

```bash
bash ~/les_g7_probe_run.sh
```

脚本自带 9 道前置闸（swap / thermal / App / 残留 server / commit2 落地 / affinity 归零 / G7 已 apply /
二进制含标记且新于源码 / 冷却），任一不过直接中止、不跑 server。
过闸后用 prod-new 8 GiB 跑一格（C17 同口径：`CGC_OVERLAP_FENCE=41`、MTP off、warm-skip64、p512/g128、
r5、fixed-fill-seed1），arm 额外带 `CGC_POOL_SPLIT_DBG=1;CGC_L4_EVICT_TRACE=1`，约 10 min。
跑完自动从最新 server stderr 解析几何与 distinct tid。

**验收（脚本会直接打印）**：
- Q2：`g7_geometry.txt` 有 per-kind 几何（geometry rc=0，原始 GEOM 行 > 0）；
- Q1：`evict_verdict.txt` 印 distinct evict tid；
- 若 8G 一格全程没有 evict trace（capacity evict 未触发），脚本会明确提示，回传日志核对。

---

## 7. 【交接点 B】回传清单 + 判读

跑完把整包产物目录与日志回传，我做最终判读：

1. `~/les_g7_probe_<TS>.log`（屏幕总日志）；
2. `Backup/les_g7_probe_<TS>/g7_geometry.txt`（Q2 几何）；
3. `Backup/les_g7_probe_<TS>/evict_tid_counts.txt`、`evict_verdict.txt`（Q1）；
4. `Backup/les_g7_probe_<TS>/server_stderr.log`（原始 stderr，含全部 GEOM/ADOPT/EVICT 行）；
5. `Backup/les_g7_probe_<TS>/g7.json`（该格 harness 产物，供对照）。

判读规则（脚本先给机器提示，结论由 agent 下）：
- distinct tid = 1 → Q1 PASS，fence 安全窗口成立，阶段2 可按「整 kind buffer、按 16KiB 绝对页、
  仅回收整页覆盖 slot 全无 owner」设计 FREEREUS；
- distinct tid > 1 → Q1 FAIL，阶段2 先补 GPU fence；
- 几何：真实 stride 若非页整、跨 slot 页占比 > 合成的 0.7%，阶段2 回收粒度按真实几何调整。

---

## 8. Commit / Gate 边界（别再被 gate 拦）

| 场景 | 触发的 gate | 正确做法 | 禁止 |
|---|---|---|---|
| commit2（预算/探针） | pre-commit 检查 8（tracked binary）+ replay | wrapper 已内置 `ALLOW_ENV_GATED_BIN=1`，replay 自动跑 | `--no-verify`、跳 replay |
| 撤 affinity / G7（本链阶段 2–3） | **本链阶段不 commit**（只工作区编译验证） | 待 Q1/Q2 结论后，连同 G2/G3a 后续一并规划 commit | 现在就把未验证改动提交 |
| 纯文档 commit（本 runbook / docs/les） | replay（docs 镜像含 .cpp 会误触） | `RUN_REPLAY_BENCH=0 git commit …`（仅文档豁免性能 gate），quote-gate 不绕 | 拿文档豁免去混源码 |
| push | pre-push doc_claim_gate | push 前让 doc_claim_gate 全 PASS | 只在本机蒙混 |
| 任何实验 | charter gate + harness 唯一入口 | 一律 harness.py，带 charter | 直跑 matrix、无立项开跑 |

---

## 9. 回滚 / 应急

- **探针 env**：`CGC_L4_EVICT_TRACE`、`CGC_POOL_SPLIT_DBG` 默认 OFF，不设即完全关闭；
- **撤 affinity / G7 前都有 `.bak-*`**；`git checkout -- <文件>` 前先确认没混入他人未提交 hunk；
- G7 补丁可整体回退：`git apply -R docs/les/G7_STAGE1_READONLY_PROBE.patch`（先确认工作区无叠加）；
- **任何阶段失败**：不要重试、不要改参数，先 `bash ~/les_preflight.sh` 看现状，把对应日志整份回传；
- 本链不预设 25：G7 只回答「fence 安全窗口是否成立、真实几何如何」，结果如实，不做无依据承诺。

---

## 10. 当前已确认的全局事实（避免重复已判死的路）

- **LES 仅 G1 离线命中率跑通**（≈82%，三套 code/zh/math 一致，已入库 `79e5b24b0`）；G2/G3a/G6 在线未跑。
- **MTP 全线判否，交付口径钉死 off**（`CGC_SERVER_MTP=0`）：0C union 对 decode 结构性为负（每 token 多读 11%
  专家、capacity miss ×10.2、verify 步 1.97×、净 ~1.09×，α=0.43 < break-even 0.553，即使 α=1 天花板 19.39<25）；
  唯一转正路是训练更强 drafter（α>0.73），属云端多卡白皮书，不在本链。
- 已判死勿重复：10G 扩池（两轮 Metal OOM）、ubatch 5632→512、ρ layer-batch prefetch（−21%）、
  FORGE Clifford rotor（需 RGU 硬件、不适用 Apple Silicon）、扩散 LM、MegaTrain 调 tile、
  expert affinity（−0.87%）、S 序列化、A 查表重排、邻居 prebind、旧 8/6/4G 转述 12.06/10.48/7.67（不可引用）。
- C 阶段0 独立探针已证：固定 Metal shared buffer 的冷页，`MADV_FREE_REUSABLE` 能让 phys_footprint
  立即全额降（128MB→−128、256MB→−256），GPU 再读不 hang、重填 bit-exact；现网 `MADV_DONTNEED`
  （`CGC_POOL_MADVISE=2`）在 macOS 匿名/Metal 页近乎 no-op —— 这就是冷页归还从「vm_deallocate 大重构」
  降级为「最小改动」的依据。
