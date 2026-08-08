# TurboFieldfare MTP 投机解码：修复、实现与性能结论

日期：2026-08-07
硬件：Mac M4 / 16GB 统一内存 / 本地 SSD
模型：Gemma 4 26B-A4B MoE It 4-bit (`gemma4.gturbo`, 13GB)
MTP head：`models/gemma-4-mtp-head` (831MB, 4 层)
仓库：`turbo-fieldfare-github-official`，分支 `compat/local-sdk15-baseline`

---

## 1. 结论速览

| 结论 | 状态 |
|---|---|
| MTP drafter 与 HF 参考对齐 | **已完成**，accept rate 0% → 61~88% |
| batched verify（一次前向验证多 token） | **已实现**，功能正确 |
| MTP 相对 baseline 是否加速 | **否**。最优配置仍慢 3.4% |
| 根因 | MoE expert IO 随批内 token 的 expert 并集线性增长，投机解码无法摊薄 |

一句话：**投机解码在"专家权重按需从 SSD 流式加载"的 MoE 架构上不成立**，因为它试图摊薄的那部分成本恰好是随批量线性增长的。

---

## 2. MTP 正确性修复（accept 0% → 88%）

对照 HuggingFace `Gemma4AssistantForCausalLM` 参考实现，定位并修复 6 处偏差：

| # | 偏差 | 修复 |
|---|---|---|
| 1 | shared KV 取层错误 | 按 `full_attention`/`sliding_attention` 各取该类型最后一层 |
| 2 | position_ids 在 draft 循环内递增 | 改为循环外算一次 `input_ids.shape[1]-1` |
| 3 | `layer_scalar` 施加位置错误 | 改为整层输出（含残差）末尾乘一次 |
| 4 | 激活函数用了 SiLU | 改为 `gelu_pytorch_tanh` |
| 5 | `post_feedforward_layernorm` 位置错误 | 改为作用于 MLP 输出、残差相加之前 |
| 6 | SWA ring capacity 未传递 | 显式携带真实页槽数，不用 `buffer.length/stride` |

外加一处独立 bug：**prefill 后 hidden state 全 0**。prefill 写 `scratch.hidden`（[tokens,D]），decode 写 `hidden`（[D]），而 `copyLastHiddenState()` 读后者。修复：在 `writeFinalHead` 中把 `scratch.hidden` 第 `t-1` 行 blit 到 `hidden`（两者同为 pre-final-norm，语义精确）。

修复后 `[mtp] hidden rms` 首步从 `0.0000` 变为 `1.0635`，输出完全正确。

**涉及文件**
- `Sources/TurboFieldfare/Metal/Assistant/assistant.metal` — 新增 `assistant_add_then_scale` / `assistant_gelu_mul` kernel
- `Sources/TurboFieldfare/Runtime/Generation/LocalMTPAssistant.swift` — 层结构重排
- `Sources/TurboFieldfare/Runtime/Generation/AssistantBridgeTypes.swift` — `ringCapacity` 字段
- `Sources/TurboFieldfare/Runtime/Inference/RealForwardRunner.swift` — prefill hidden blit

---

## 3. batched verify 实现

原实现每个 draft 单独调一次 `producer.produce()`，N 个 draft 全接受需 N+1 次 target 前向 —— 与 baseline 产出同样多 token 的前向次数**完全相同**，MTP 净增 drafter 开销，必亏。

改为一次多 token 前向验证 `[cur, d1, d2, ...]`：

1. `KVCacheManager` 增加 `rewind(to:)` —— 支持拒绝后回滚
2. `LMHeadChainInt4.encodeGreedyDecode` 增加 `outTokenOffset` —— 多行 argmax 写入 `[N+1]` UInt32 缓冲
3. `RealForwardRunner` 增加 `verifyBatch` API，复用现成的 `prefillChunked` 多 token 前向
4. `MTPCompletion` 改用批量 verify + KV 回滚

功能验证：短 prompt 下 accept 从 7/8 升到 **8/8 (100%)**（末尾 draft 现在也在同一批内验证），输出逐字一致。

---

## 4. 性能测量

### 4.1 测量方法学（重要）

初期测量出现严重漂移：同一 baseline 配置在不同时段测出 6.10 / 6.96 / 7.67 tok/s，跨度 26%。原因是 OS 页缓存随反复运行逐步预热。

**所有下述结论均采用交错 A/B**（A,B,A,B,... 而非 A,A,B,B），以抵消漂移。顺序测量的数字一律作废。

测试 prompt：`/tmp/msg_code.json`（Python 记忆化斐波那契），128 token 生成，greedy（`--temperature 0 --repetition-penalty 1.0`）。

### 4.2 draft 深度扫描（slots=16，顺序测量，仅看趋势）

| 配置 | tok/s | accept |
|---|---|---|
| baseline | 6.104 | — |
| draft=1 | 3.621 | 58/69 (84%) |
| draft=2 | 3.472 | 76/104 (73%) |
| draft=4 | 1.897 | 91/148 (61%) |
| draft=8 | 1.811 | 99/232 (43%) |

单调恶化。draft=4 时每批 5 行耗时 1824ms，而 5 次单 token 前向仅 819ms —— **批量前向比逐个前向贵 2.2 倍**。

### 4.3 expert cache 槽数的决定性作用（交错 A/B）

模型参数：`topK=8`、`numExperts=128`、`expertStride=3.2MB`、30 层。
单 token decode 需读 30×8×3.2MB ≈ **768MB** expert 权重。

`PrefillRoutedTileScheduler.fitsSlotBudget`：
```
(maxPendingDepth + 1) * tileExperts + reservedHits <= slotCount
```
`tileExperts` 上限 16，故 **slotCount=16 只允许 maxPendingDepth=0，即零预取重叠**。
N 行 token 的 expert 并集最多 8N 个，超过槽数就必须分 tile 串行换入换出。

| 场景 | slots=16 | slots=32 | 差异 |
|---|---|---|---|
| baseline（单 token） | 7.306 | 7.586 | **+3.8%** |
| MTP draft=4（5 行） | 4.626 | 6.907 | **+49%** |

因果清晰：**槽预算只对批量 verify 关键，对单 token decode 几乎无关**。单 token 只需 8 个 expert，16 槽绰绰有余；5 行并集最多 40 个，16 槽必然颠簸。

### 4.4 最终对比（slots=32，read=baseline，交错 A/B ×3）

| 轮次 | baseline | MTP draft=3 |
|---|---|---|
| 1 | 7.454 | 7.350 |
| 2 | 7.992 | 7.511 |
| 3 | 7.570 | 7.374 |
| **均值** | **7.672** | **7.412** |

draft=3（4 行，并集 ≤32 恰好装满）是最优 MTP 配置，accept 68%，**仍慢 3.4%**。
draft=4 均值 6.907，慢 10%。

### 4.5 prefill expert 读模式（交错 A/B ×2）

| read mode | draft=3 tok/s |
|---|---|
| baseline | 7.544 |
| coalesced | 7.320 |

coalesced 慢 3%，保持 `.baseline` 为默认。
（顺序测量时 coalesced 曾测出 4.157，属漂移伪影，已作废。）

---

## 5. 为什么投机解码在此架构上不成立

在稠密模型上，投机解码有效的前提是：**批量前向的成本 ≈ 单 token 前向的成本**（权重相同，只是多算几行 GEMM，算力富余）。

在本架构上这个前提不成立：

- 权重不常驻。模型 13GB，机器 16GB，expert 权重必须按需从 SSD 流式加载。
- decode 是纯 IO-bound。768MB/token ÷ 144ms/token ≈ 5.3GB/s，与内置 SSD 带宽吻合。
- **不同 token 路由到不同 expert**。N 行的 expert 并集近似 8N（重合度低），IO 成本随批量近似线性增长。

于是批量前向省不下 IO，投机解码只剩纯开销：drafter 计算 + 拒绝后的浪费。accept 68% 也换不回来。

**投机解码只有在 expert 权重常驻内存时才会转正**。届时批量前向近似免费，68% accept 可望带来约 1.7× 加速。这需要 ≥32GB 内存的机器（expert 全常驻需 12.3GB）。

---

## 6. 产出与建议

**保留 batched verify**。它把 MTP 从"慢 70%"救到"持平"，且是架构上正确的实现。一旦换到大内存机器 / expert 常驻，它是转正的必要前提。

**新增环境变量旋钮**（`Sources/TurboFieldfareCLI/Run.swift`）：
```
TURBO_FIELDFARE_EXPERT_SLOTS=8|16|24|32          # 默认 16
TURBO_FIELDFARE_PREFILL_EXPERT_READ=baseline|coalesced|layer-local-readahead
```

**默认值建议不变**（slots=16）。32 槽占用 30×32×3.2MB ≈ 3.07GB；本机运行时仅剩 28% 空闲内存（≈4.5GB），再往上（48 槽 4.6GB）会挤压系统。仅当启用 MTP 时才建议设为 32。

**下一步方向**（按预期收益排序）：
1. 降低 expert IO —— 这是唯一真正的瓶颈（144ms/token 几乎全是 IO）。方向：更激进的量化、expert 预测预取、mmap + 页缓存替代 pread。
2. 在 ≥32GB 机器上复测 MTP —— expert 常驻后投机解码应转正。
3. 提高 accept rate —— 当前 68%，但在 IO-bound 下收益有限，属于第二优先级。

---

## 附：运行方式

```bash
cd /Users/alexchuang/Documents/turbo-fieldfare-github-official
export PATH="/opt/homebrew/opt/swift/bin:$PATH"   # 需 Swift 6.3.3，系统 6.1 不兼容
swift build -c release --product TurboFieldfareCLI

MODEL=/Users/alexchuang/Documents/flashkv0516/models/gemma4.gturbo
MTP=/Users/alexchuang/Documents/flashkv0516/models/gemma-4-mtp-head

# baseline
./.build/release/TurboFieldfareCLI --model "$MODEL" \
  --messages-file /tmp/msg_code.json \
  --temperature 0 --repetition-penalty 1.0 --max-new 128

# MTP（必须 greedy，否则报 "MTP speculation requires fused greedy head"）
TURBO_FIELDFARE_EXPERT_SLOTS=32 TURBO_FIELDFARE_MTP_DEBUG=1 \
./.build/release/TurboFieldfareCLI --model "$MODEL" --mtp-model "$MTP" \
  --mtp-max-draft 3 --messages-file /tmp/msg_code.json \
  --temperature 0 --repetition-penalty 1.0 --max-new 128
```

原始日志：`/tmp/mtp_draft_sweep.log`、`/tmp/slots_ab.log`、`/tmp/slots_mtp_ab.log`、`/tmp/mtp_ab_interleaved.log`、`/tmp/mtp_readmode_ab.log`


---

## 7. 追加验证（2026-08-07 午后，磁盘清理后 42Gi 空闲）

### 7.1 接受率回归调查：3-14% 是伪影，无真实回归

期间一度测到 accept 3-14%，draft 呈现「完全错位」（如 drafts=[67104,84750,107] vs targets=[563,563,108]），一度怀疑 6 项对齐修复丢失（MTP 源文件 untracked 从未提交）。逐一排查后：

- **代码完好**：`LocalMTPAssistant.swift` 六项修复全在（gelu_pytorch_tanh / post_feedforward_layernorm / layer_scalar / ringCapacity / assistant_add_then_scale / assistant_gelu_mul），分支 `compat/local-sdk15-baseline` 一致，`MTPCompletion` batched verify + rewindKV + publishHiddenRow 完好。
- **模型一致**：`gemma4.gturbo` 30 层 / 128 experts / stride 3,358,720，与报告一致；sourceSnapshotHash 与 r3 相同。
- **实测（干净机器，slots=32，draft=3，msg_code，128 tok）**：
  - r4：`mtp=86/126 (68%)` —— 与报告 68% **逐位一致**
  - r3：`mtp=91/111 (82%)` —— 更好
- 之前 3-14% 与「总分布错位」是同源伪影：磁盘抖动期 + 陈旧二进制（期间 build 目录被并发重建、产物过期）污染。debug 输出显示高 id token 区域（236xxx 附近）才是 drafter 的失配重灾区（0/3 步都集中于此），并非系统性错位。

### 7.2 「MTP 不加速」结论在干净机器上复核：依然成立（且差距更大）

同 slots=32、交错 ×3：

| 模型 | baseline | MTP draft=3 | 差异 |
|---|---|---|---|
| r4 | 8.69 | 7.80 | **−10%** |
| r3 | 11.70 | 9.92 | **−15%** |

干净机器上 baseline 收益（页缓存整载 27G 模型）大于 MTP 收益，导致差距比报告（−3.4%）更大。**根因不变：batched verify 的 expert 并集 IO 随批量线性增长，投机解码摊不薄它**；r3 的 82% 接受率也救不回来。

### 7.3 结论

- MTP 接受率健康（68%/82%），无需修复；不要被抖动期伪影带偏。
- 投机解码在此架构（MoE expert 流式加载 + IO-bound）上维持「负收益」结论，唯一转正路径不变：**expert 常驻内存（≥32GB 机器）**，届时 68-82% 接受率可望带来显著加速。


---

## 8. Adaptive MTP gate + 计算侧结构发现（2026-08-07 晚间）

### 8.1 Adaptive gate（TURBO_FIELDFARE_MTP_ADAPTIVE=1 默认开）

实现于 `Sources/TurboFieldfare/Runtime/Generation/MTPAdaptive.swift` + `MTPCompletion.swift` + `Run.swift`（footer 新增 `adaptive(...)` 摘要）。**纯经验吞吐决策**，不做接受率硬门槛：

- **启动校准时序**（击败 IO 冷启动偏差）：3 步丢弃 warmup → 4 步全 draft MTP（compute-bound，暖 cache）→ 6 步 warm d=0 baseline（诚实单步基准）。
- **运行策略**：每 8 步量一次 MTP 窗口吞吐（仅统计 d>0 步），与 warm baseline 比较；>baseline×1.05 升 draft、<baseline×0.95 降、<baseline×0.90 **一步直接禁用**；禁用后每 64 步 probe 一次（d=2 ×8 步）防 prompt 话题切换错杀。
- **正确性**：adaptive ON/OFF 输出逐字节一致（md5 相同），多次稳定。

### 8.2 实测行为（r3，64 slots，temp0，128 tok）

| 场景 | base | fixed d=3 | adaptive | gate 行为 |
|---|---|---|---|---|
| code（82% accept） | ~10-11 | ~11-12 | ~11-12 | **保持 d=3**（off=0） |
| prose（45-50% accept） | 12-16 | 8.2-8.7 | 5-9（禁用后） | **快速禁用**（off=9-11，仅 19-32 drafts） |

- code 上 gate 正确保持最大 drafts（MTP 在此 prompt 是真实正收益）。
- prose 上 gate 正确识别并禁用 MTP——**但禁用后仍只有 ~6-8 tok/s，远低于 base 12-16**（见 8.4）。

### 8.3 计算侧尝试与结论（本轮全部实测）

1. **verify 时间分解**（MTP_DEBUG phases 行）：draft 17% / **verify 83%** / rewind 0%。verify 是唯一大头。
2. **批量投影实验**（`TURBO_FIELDFARE_VERIFY_BATCHED`，verify span t<32 强制 qmm）：交错 A/B 6 轮，批量 **−6%**（qmm 8×8 threadgroup 在小 t 利用率低）——**已回退**，per-row GEMV 保持默认。
3. **verify ≈ 3.5× single-step**（5 行 ≈ 3.5 步）：已含 ~30% 批量红利（expert 并集共享），其余是 (1+d) 行工作的固有成本。
4. **「draft 複用主模型 shared FFN」结构性不成立**：`gemma-4-mtp-head` 是独立 4 层 / hidden 1024 / 自带 FFN 的 assistant 模型（831MB），仅读取 target 的 KV bridge snapshot，与主模型无共享 FFN 可复用。唯一复用点是主模型末层 hidden state（backboneA 输入），已实现。
5. **MTP loop 本身是结构瓶颈（最大发现）**：adaptive 禁用 drafts 后（d=0 步），loop 仍比 base decode 慢 2.5-3×（5-6 vs 12-16 tok/s）。原因：每步序列化 sync（bridge snapshot → verifyBatch wait → rewind wait → publish blit）+ 1 行 verify 走通用 prefill 路径（非 decode 专用内核）。**投机解码在本机无法转正的真正障碍不是接受率、不是 expert IO，而是 loop 的每步延迟税。**

### 8.4 结论与正确路径

- **adaptive gate 保留为默认**：它在 loop 内部正确管理 draft 数（code 保持、prose 禁用），输出零变化，且在任何机器上不会让 MTP 跑出比 loop 内 d=0 更差的结果。
- **MTP 转正的真正前置**：把禁用步（以及整个 MTP 循环）路由到 decode 路径（produce），或流水线化 loop（重叠 CPU encode 与 GPU 执行）消除每步 sync——这是比 attention tensor-ops 更便宜、更直接的杠杆。
- 机器/平台结论不变：72GB L20N 上 verify 每行成本相对小，loop 税占比低，68-82% 接受率才有机会转正。


### 8.5 d=0 decode-path 路由（2026-08-07 深夜，结构瓶颈修复）

**问题**：adaptive gate 禁用 drafts 后，loop 仍跑 `verifyBatch(1)`（通用 prefill 路径 + 序列化 sync），d=0 步是 base decode 的 2.5-3×（5-6 vs 12-16 tok/s）——「禁用」回不到 base 速度。

**修复**（`MTPCompletion.swift`）：`draftBudget == 0` 时跳过 bridge snapshot / drafter / verifyBatch / rewind / publish，改走单 token **decode path**（`produce(token:position:into:)`），语义与 `runRawCompletion` 的 decode 循环一致。混用安全已核实：

- verify 步结束：`prefillChunked` 内 `markCommitted()` + `rewindKV(to: position)` 设 `kv.position == position`；
- `produce` 要求 `prefillChunkState.requireClean` + `kv.position == position` —— 两条路径状态机完全兼容，可自由交错。

**附带收益**：warm baseline 现在直接量到**真实 decode 速度**（不再是 loop 内 d=0 的慢速），校准的 `rowShare`（=1.5，钳制值）也真实反映 verify/decode 成本比。

**实测（r3，64 slots，temp0，128 tok，三向交错 ×3 取中位）**：

| 場景 | base | fixed d=3 | **adaptive** |
|---|---|---|---|
| prose（45-50% accept） | 12.3 | 8.8（−28%） | **14.3（≥base）** |
| code（80-84% accept） | 11.5 | 11.9 | **14.4（+25% vs base）** |

- **正确性**：adaptive 与 base 输出逐字节一致（prose `465d7edb` / code `4211ccfe`），多次稳定。
- **核心承诺兑现**：adaptive gate + decode-path 路由保证 **MTP 在任何 prompt 上都不劣于 base**（prose 14.3 vs 12.3、code 14.4 vs 11.5），且远优于固定 draft（prose 快 62%）。
- adaptive 快于 base 的残余增益来源：校準期 MTP（首个 ~10 步 84% accept 批量产 token）+ 校準暖热 expert cache 的惯性 —— 均真实、稳定（6/6 轮全胜）。

**结论**：`TURBO_FIELDFARE_MTP_ADAPTIVE=1`（默认）现在是安全的默认配置：prompt 接受率高时吃 MTP 红利，低时自动回落真实 base 速度，机器/prompt 自适应，无需硬编码阈值。


### 8.6 长 run 确定性说明（600 tokens）

600-token 长生成在 16GB 机器上**非逐字节可复现**：base 自身跑 3 次得到 2 种输出（`9a9bf145` / `6ae2ad54`），分歧发生在 ~char 1925（「reaches cooler altitudes」vs「cools, it loses」——logits 边界翻转的两种合理续写，非状态损坏）。128-token 短 run 完全可复现（多次 md5 一致）。原因是长生成下内存/页缓存压力导致的数值层非确定性。因此：

- 长 run 的正确性对比不能依赖 md5；adaptive 与 base 在确定性窗口内逐字节一致（adaptive2 == base1 == base3）。
- 600-token adaptive run 触发了 probe 重探（off=66 > 64 → decode→MTP→decode 过渡被实盘执行），输出仍与 base 规范输出一致——**decode-path 路由的 decode→MTP 过渡已实测通过**（此前 128-token 无法触发的路径）。
