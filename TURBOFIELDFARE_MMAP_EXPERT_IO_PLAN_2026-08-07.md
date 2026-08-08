# TurboFieldfare Routed-Expert IO 改造方案：pread → mmap + 页缓存预读

日期：2026-08-07（磁盘清理后，42Gi 空闲）
目标：把 routed expert 权重从「pread 拷贝进私有 slot」改为「mmap + 页缓存直读」，让多个 token（尤其 MTP batched verify）共享同一批已缓存专家页。

---

## 1. 现状（当前 pread 数据面）

`PreadExpertStreamer.swift`（每层一个 streamer，`ExpertStreamingMode.pread(slotCount:)`，默认 16，生产 64）：

1. init：`open(layer_file, O_RDONLY)` + `posix_memalign` 分配 `slotCount` 个页对齐私有缓冲（`slotPointers`），每个 `device.makeBuffer(bytesNoCopy:, .storageModeShared)` 包成 MTLBuffer。
2. miss 路径：`loadExpert` → `readFull` → **循环 `pread(fd, slotPtr, stride, streamOffset+expertOffset)`**，把整块 expert（r4 3.2MB / r3 2.5MB）从页缓存**拷贝进私有 slot**。
3. hit 判定：slot 标签表 `slotExpert[slot] == expert`（有限槽的启发式），LRU 换出。
4. 预取：`fcntl(fd, F_RDADVISE)`（`RDAdvice.swift`）+ lookahead 预读进 slot。
5. 对比：**resident 权重早已是 mmap 模式**（`ResidentBuffer.swift`：`mmap(PROT_READ, MAP_PRIVATE)` → `posix_madvise(POSIX_MADV_RANDOM)` → `makeBuffer(bytesNoCopy:, .storageModeShared, deallocator: munmap)`）。routed 是唯一还在 pread 拷贝的数据面。

### 关键数字（本次会话实测）

| 项 | 值 |
|---|---|
| r4 expert stride | 3,358,720 B = **205 × 16,384（页对齐 ✓）** |
| r3 expert stride | 2,605,056 B = **159 × 16,384（页对齐 ✓）** |
| 层 × 专家 | 30 × 128，专家权重合计 ≈ 12.9GB |
| slot 内存占用 | 30 层 × 64 槽 × 3.2MB ≈ **6.3GB**（128 槽 = 12.6GB，压垮 16GB 机） |
| slots=64（干净机，r4） | hit 95.7%，pread 9.9GiB/128tok，tok/s ≈ 15.9 |
| slots=16 | hit 62.7%，pread 18GiB/64tok，decode 7-9 tok/s |

---

## 2. 改造方案：mmap + mincore 规划 + MADV_WILLNEED 预读

### 2.1 数据面（核心，改动最小）

在 `PreadExpertStreamer` 内新增 `useMmap` 模式（或独立 `MmapExpertStreamer`，公共 API 不变）：

- **init**：`mmap(nil, streamSize, PROT_READ, MAP_PRIVATE, fd, 0)` 整层文件一次映射（`streamOffset=0`、每层独立文件，天然成立）。
- **专家缓冲**：`device.makeBuffer(bytesNoCopy: mappedBase + expertOffset, length: stride, .storageModeShared)` —— 每个专家一个视图，**3840 个 buffer 全部指向同一块 mmap，零额外内存**；首次访问时惰性创建并缓存（字典），**永不驱逐**。
- **loadExpert**：字典查表（命中即返回 buffer），无 pread、无拷贝。GPU 直读页缓存页（与 resident 权重同一已证明模式）。
- **hit/miss 规划**：把 slot 标签表换成 **`mincore(mapped + offset, stride, vec)` 真页驻留判定**——「hit」= 页确实在内存，比「曾经装进某槽」精确得多（槽被 LRU 换出但页仍驻留的专家，今天会白白 pread 重读，mmap 下是真 hit）。
- **预取**：保留 `F_RDADVISE`，并新增 **`madvise(addr, len, MADV_WILLNEED)`** 对 lookahead 预测的下一层专家窗口做精确预读（比 fd 级 advice 更精准）。
- **deinit**：先释放所有 MTLBuffer，再 `munmap`。

### 2.2 接线

- `ExpertStreamingMode` 增加 `mmap`（或 `.pread(slotCount:useMmap:)`）；`Model.swift` 按模式选 streamer。
- 新 env 旋钮：`TURBO_FIELDFARE_EXPERT_MMAP=1`（默认 off，回退 pread）。
- telemetry：expert-cache 命中率改为 mincore 口径，新增 `[expert-io]` 的 mmap fault 统计。

### 2.3 为什么内存不再随 slot 增长

slot 缓冲从「每个槽私有分配」变成「全部视图同一块 mmap」——**slot 数概念对内存完全无成本**。128+ 槽、甚至「所有 3840 个专家 buffer 全活」都只占 mmap 元数据 + 3840 个轻量 MTLBuffer 对象（合计 < 几 MB）。页是否驻留由 OS 页缓存统一裁决（42Gi 空闲 vs 12.9GB 专家，热集可整驻）。

---

## 3. 预期收益（量化）

### 3.1 内存：−6.3GB（64 槽）或解锁 128+ 槽

30×64×3.2MB ≈ **6.3GB 私有 slot 分配归还系统**（128 槽 12.6GB 同理解锁）。16GB 机上这是巨大呼吸空间，且「128 槽压垮机器」的旧结论作废。

### 3.2 decode / TTFT 量化预估（2026-08-07 干净机 telemetry 推算）

基线（slots=64，B-tree prompt，128 tok，本会话实测）：

| 模型 | decode | tok/s | readWall 占 decode | GPU busy（floor） | GPU ceiling |
|---|---|---|---|---|---|
| r3 | 8.96s | 14.3 | 4.52s（**50%**） | 5.73s | ~25 tok/s |
| r4 | 8.05s* | 15.9* | ~63%（噪声run 8.5s/13.5s） | ~7.7s | ~19 tok/s |

（*r4 取 B1 交错中位；telemetry 单跑噪声大。）

**IO 为什么占一半：命中率 94-97% 下 readWall 仍巨大**——(a) lookahead 预取每轮 pread 拷贝 7-8GiB 进 slot，命中 65% 即 ~35% 白拷；(b) 被 LRU 换出但页仍在缓存的专家被白白重读重拷（slot 口径 hit 低估页驻留）。

**mmap 后预估（页缓存 42Gi 承载 12.9GB 专家，热集整驻）：**

| 项 | 改造前（r3） | 改造后预估（r3） | 改造后预估（r4） |
|---|---|---|---|
| decode tok/s | 13-14 | **19-22**（+40~60%） | **17-19**（接近 GPU ceiling ~19） |
| TTFT | 2.6-2.9s | **2.0-2.5s**（−15~25%） | **2.6-3.1s**（−10~20%） |

推算逻辑：
- decode：`readWall → 首触 fault 为主（预取变页预热、驻留重读变真 hit、拷贝与 syscall 消除）`，IO 占比 50%→~15-20%；decode 时间 → GPU busy + head + 残余 sched ≈ r3 5.7-6.2s / r4 6.8-7.5s。
- 上限即 GPU ceiling（r4 ~19 / r3 ~25 tok/s，按 147 token 当量 GPU busy 推算）——mmap 之后 decode 将贴近此值，再往上需 GPU 侧优化（head/attention）。
- TTFT：prefill 专家读同样去拷贝/去预取白拷，但首触磁盘读仍存在，故增益小于 decode。



- 现状瓶颈（slots=64 干净机，r4 ~15.9 / r3 ~13.0）：每 miss 仍是一次 pread 系统调用 + 3.2MB 页缓存→私有页拷贝 + LRU 驱逐重读。
- mmap 后：miss 变成一次页 fault（首次触碰该专家才发生）；热集专家重访问全部页 hit、零拷贝。命中率从 95.7%（槽口径）升到 **~99%+（页驻留口径）**。
- 残余成本只剩「首次触碰的不同专家必须从盘读一次」——这是任何方案都免不了的，但 42Gi 页缓存 + 12.9GB 专家使其在预热后趋近零。
- 对应「IO 占 decode 比例 56.9%→42.7%（4→2-bit）」的既有下降曲线，mmap 把该比例推向 <10%。

### 3.3 MTP batched verify：从 −10~15% 转向持平或转正

- 今天 MTP 失败机制：verify 批内 N 行 expert 并集（≤8N 个），slots=32 时槽颠簸 → 每行重读重拷。mmap + mincore 下并集专家页**批内共享、批间驻留**，batched verify 不再重复付 pread 拷贝。
- 接受率已健康（r4 68% / r3 82%），若 verify 的 IO 边际成本归零，投机解码在 r3 上转正的现实性大增。

### 3.4 规划精度与跨进程共享

- mincore 口径让 prefetch planner 只对真缺失页发 F_RDADVISE/MADV_WILLNEED（今天槽口径会误发）。
- 页缓存全局共享：Server/worker 多进程共用热专家页。

---

## 4. 风险与缓解

| 风险 | 缓解 |
|---|---|
| Metal `bytesNoCopy` 包 mmap 内存 | **同一代码库 resident 权重已生产使用**（ResidentBuffer）；窗口天然 16K 页对齐（stride=205/159 页） |
| GPU 访问被换出的页 | MAP_PRIVATE 页驱逐后 GPU fault 经统一内存管理器重新调入（与 resident 路径今天的行为相同）；MADV_WILLNEED 预读缓解抖动 |
| buffer 生命周期 vs munmap | deinit 先 release 全部 buffer 再 munmap（ResidentBuffer 已示范 deallocator 捕获模式） |
| 热路径回归 | 新模式默认 off（env 开启），A/B 与 pread 严格交错对比；公共 API 不变，可随时回退 |
| mincore 粒度 | macOS 支持；stride 对齐页边界，判定无边界瑕疵 |

---

## 5. 实施步骤（建议顺序）

1. `PreadExpertStreamer` 加 `useMmap` 数据面（mmap init / 专家 buffer 缓存 / mincore 规划 / madvise 预取 / munmap deinit），公共 API 不变。
2. `ExpertStreamingMode` + `Model.swift` 接线 + `TURBO_FIELDFARE_EXPERT_MMAP=1`。
3. telemetry 扩展（mincore 口径 hit、fault 计数）。
4. 交错 A/B（64 槽 vs mmap，slots=16/64 两组，r3+r4，msg_code 固定 prompt，6 轮中位）：tok/s、io bytes、hit%、fault%。
5. MTP A/B（slots 无关后，draft=3）复核 −10~15% 是否转正。
6. 输出结论并决定默认值。

**改动量估计**：数据面 ~200-300 行（集中在 streamer 一个文件），接线 + telemetry ~50 行，env 默认 off。风险集中在 GPU 直读页缓存这一新路径，用 A/B + 输出一致性校验兜底。

---

## 6. 诚实边界（mmap 不解决什么）

- **不同专家首次触碰仍需从盘读一次**（专家并集 IO 的增长不是「拷贝」而是「磁盘字节」）；mmap 消除的是重读/重拷与槽颠簸，不是首读。
- 页缓存被其他负载挤占时（内存压力），仍会抖动——42Gi 空闲下暂不构成问题。
- 方案不改变量化、不改变 kernel；是纯 IO 数据面替换。

---

## 7. 实测结论（2026-08-07 实施后）— mmap 在 16GB Mac 上不敌 pread

已完整实现 `TURBO_FIELDFARE_EXPERT_MMAP=1` 数据面并逐项实测，最终结论：

### 7.1 实现内容（全部已验证正确）

- **单 stream buffer + per-expert offset**：每层一个 buffer 包住整条 expert stream（`bytesNoCopy`），arg buffer 用 `setBuffer(buffer, offset:)` 编入 per-expert 偏移（MSL kernel 零改动），slot 表只做 hit/miss 记账
- **WILLNEED + 逐页 touch**：`POSIX_MADV_WILLNEED` 批量预读 + 每 16K 页读 1 字节强制驻留。touch 是**正确性硬要求**——GPU 无法 fault 真正冷的 file-backed 页（会读成垃圾）；早期 blit 探针"证明冷页可读"是假象：文件刚被自己写入，页全在缓存里
- **useResources 去重**：所有 blob 指向同一 stream buffer → 每次 CB 只注册 1 次驻留
- **arg buffer 时序修复**：phase2 的 split argBuf 从 post-fetch 的 fresh blobs 重新编码（否则 miss 槽的地址/偏移是加载前的旧值 → 垃圾）

### 7.2 正确性（达成）

temp-0 固定 prompt 下 mmap 与 pread 输出**逐字节一致**，多次运行稳定。

### 7.3 性能（未达成，且无法达成）

| 指标 | pread | mmap（健康时） | mmap（压力下） |
|---|---|---|---|
| decode | 11.8-14.3 tok/s | 9.1-10.3 | **卡死/爬行** |
| TTFT | 2.6-2.8s | 3.8-4.2s | — |
| IO（readWall） | 1.2s | **0.06s** | — |
| CB 编码开销 | 0.97s (78μs/cb) | 4.3-4.9s (350-400μs/cb) | — |

三个不可逾越的障碍（均为平台行为，非实现缺陷）：

1. **file-backed `useResource` 驻留税**：Metal 对 `bytesNoCopy` file-backed buffer 做同步驻留 walk，~85μs/buffer/CB。批量/去重只省调用开销，驻留工作按 buffer 计。IO 省的 1.2s 远小于税花的 3.4s → **mmap 结构性赢不了 pread**
2. **TTFT +1.2s**：prefill 首触专家时 WILLNEED+touch 在关键路径上逐个 fault
3. **间歇性 GPU 页回收卡死**：16GB RAM 压力下 clean file-backed 页随时被内核丢弃，GPU 的映射失效 → 逐页慢速 re-fault，decode 从 2 token/90s 到完全卡死（多次复现）

### 7.4 结论与后续

- **pread 保持默认**（未受影响，逐字节回归验证通过）；mmap 保留为 opt-in 实验路径（`TURBO_FIELDFARE_EXPERT_MMAP=1`；128-slot 全常驻用法：`TURBO_FIELDFARE_EXPERT_MMAP=1 TURBO_FIELDFARE_EXPERT_SLOTS=128`）
- 用户原目标「mmap 赢 pread → 128 slots 全常驻 → MTP 转正」在这台 16GB Mac 上被 Metal+file-backed 语义挡住
- **MTP 常驻的正确路径**：
  - 首选：72GB L20N 服务器上做**匿名内存全量常驻**（10-13GB 专家池一次加载，匿名 `bytesNoCopy` 无驻留税、无回收风险）→ MTP 正收益如报告预测（r4 +23~64% / r3 +52~103%）
  - 次选（16GB Mac）：**匿名热子集池**——每层 top-N 高频专家（~2-4GB）常驻匿名内存，其余流式；MTP verify 的并集主要落在热子集 → 大部分收益、内存可控

---

## 8. 匿名热子集常驻池（hot pool）— 已实现并实测

mmap 路被平台挡死后，按 7.4 的次选路线实现了 **profile 驱动的 pinned 匿名热池**（`pread` 模式的纯增强，不占额外内存——从 slot 预算中划出）：

### 8.1 实现

- **profile**：从一次 trace（`TURBO_FIELDFARE_EXPERT_TRACE`）统计每层 top-N 高频专家 → JSON（按层索引）
- **pinned 池**：streamer init 时把 top-N 专家 pread 进专属 slot 并标记 `slotPinned`，planner 的 evictable 列表永久排除 pinned slot（永不移除、永不重读）
- **预算**：poolSize = min(N, slotCount − 16)，保证 ≥16 个 LRU slot 能放下单 token topK 与 MTP 并集（≤16）
- 用法：`TURBO_FIELDFARE_HOT_POOL=1 TURBO_FIELDFARE_HOT_POOL_EXPERTS=32 TURBO_FIELDFARE_HOT_POOL_PROFILE=/tmp/hot_pool_profile.json`

### 8.2 覆盖率（trace 实测，30 层）

| 池大小/层 | 请求量覆盖 | MTP verify 并集覆盖 | 内存（匿名） |
|---|---|---|---|
| top-16 | 66.5% | 56.9% | 1.6GB |
| top-32 | 87.4% | 82.5% | 3.2GB |
| top-48 | 95.0% | 92.6% | 4.8GB |

### 8.3 实测（r3，64 slots，temp0）

| 配置 | code prompt | prose prompt |
|---|---|---|
| base（无池） | 11.79 tok/s | 12.16 |
| **pool** | **14.79（+25%）** | **12.89（+6%）** |
| base+MTP (82% accept) | 11.78（MTP 中性） | 7.58 |
| pool+MTP | 11.50（−22% vs pool 非 MTP） | 7.08（+7.6% vs base MTP） |

- **池是真实的 decode 收益**：code prompt +25%（3 轮全胜）、prose +6%——pinning 防止高频专家被 LRU 挤出重读
- **MTP 的 IO 半壁被池解决**：readWall −10%、bytes 8.55→8.28GiB；但 **MTP 仍中性偏负**——batched verify 的 GPU 计算（draft + 4 行前向）是主导成本，池管不到
- 结论修正：**「常驻 → MTP 转正」只解决了 IO 半边；GPU 计算半边需另找（draft 复用共享 FFN、verify 稀疏化）或接受 MTP 在该硬件上的中性定位**

### 8.5 「17.5% 是否拖累」— 决定性实验（pool48 = 92.6% 并集覆盖）

code prompt, 64 slots, 82% accept：

| 配置 | decode | tok/s | readWall | bytes |
|---|---|---|---|---|
| pool32 无 MTP | 8.32s | 15.38 | 4.16s | 9.86GiB |
| pool32 + MTP | 11.09s | 11.55 | 4.22s (+0.06s) | 8.28GiB |
| pool48 + MTP (92.6%) | 10.95s | 11.69 (+1%) | 3.86s (−8%) | 9.73GiB |

结论：
- 覆盖 82.5%→92.6%（尾清零）→ tok/s +1%（噪声内）→ **非池 17.5% 不是拖累**
- 池相同时 MTP 开 vs 关：IO +0.06s、bytes 反降，但 decode +2.77s（+33%）→ **MTP 成本 = batched verify 的 GPU 前向计算**，与专家 IO 完全解耦
- 即使 100% 常驻，verify 计算照样在 → **MTP 在此硬件上不会因常驻转正**；转正只能砍计算（draft 复用 shared FFN / verify 稀疏化 / 减 draft 数）

### 8.4 代码状态

- 改动：`PreadExpertStreamer`（pinned 池 + planner 排除）、`Model`（env + profile 解析）、无 MSL 改动
- 正确性：pool on/off 输出逐字节一致、多次稳定；默认（无池）行为零变化
### 9. B3 — decode attention tensor-ops 移植:可行性評估(2026-08-07)

#### 9.1 現狀測量(r3, 64 slots, temp0, code prompt, 128 tok)

```
[gpu] attn=2.03s routedMoE=1.70s sharedFFN=0.79s phase1Hit=0.32s head=0.83s busy=5.67s ofWall=44%
[sync] cbs=12760 cb1Wait=5.60s sched=7.32s overhead=0.87s(68us/cb) ofWall=45%
[stage] cb1=0.16s(1%) io=2.21s(16%) gpuWait=10.06s(75%) wall=13.49s
decode = 9.23s (128 tok, 72ms/step)
```

- **attn = 2.03s = GPU busy 的 36%,最大單項**;但 GPU 只佔 wall 的 44%,CPU 側 sched/idle(7.3s)才是更大的洞
- 30 層中 **5 層 full-attention**(512/16/2,indices 5/11/17/23/29)、25 層 SWA(256/16/8,window 1024)
- 短上下文下 full ≈ 29% 的 attention 位置工作量(≈0.6s);**長上下文(≥4K)full attention 無界增長,SWA 封頂 1024 → full 佔 60-76%**,價值隨上下文增長

#### 9.2 既有藍圖(已驗證、已是 M4 生產路徑)

- `prefill.metal` 已有 `attention_prefill_full_tensorops_2d_validity_v2`:mpp::tensor_ops `matmul2d`(8 輸出 × 512 head-dim,keys 以 64 為 tile)+ flash rescale(row_max/row_sum/row_old_scale)
- OPTIMIZATION_JOURNEY 記錄:8 head 一次處理,**64K 下 attention 11x**、32K prefill 491→204s(2.4x);reduce order 微改 logits 已被證實無害
- decode full attention 與 prefill 是同一數學(queryCount=1 的特例);目前 decode 走 per-position 純量 FMA + 每位置 ~4 barrier(GQA-full 1024 threads)+ flash-decoding 16-chunk 兩趟(partial+combine)

#### 9.3 移植方案

- 新 kernel `attention_decode_full_tensorops` = prefill tensor kernel 的 queryCount=1 特化:
  - grid.x = 1(query)、grid.y = numQHeads/8 = 2(KV-head group),128 threads
  - kvValidCount = seqLen,keys 以 64 為 tile 單趟 flash,直接寫 attnOut(省 combine 趟)
  - 只服務 full 512/16/2、ringCapacity==0、scale==1.0 的層;SWA 層不動(後續需 256/16/8 變體)
- env gate `TURBO_FIELDFARE_ATTN_TENSOROPS=1`(默認 off,與其他實驗 knob 一致),A/B 定案後再決定 default

#### 9.4 預期收益與風險

| 項目 | 預期 |
|---|---|
| full 層 attention kernel 時間 | MPP 8-head 並行 + 64-key tile(4 barriers/64 keys vs 4/position)→ 估 2-3x(短 ctx)~10x+(長 ctx,同 prefill 數據) |
| 短 ctx 總 decode | attn 0.6s 部分 → 省 ~0.3s ≈ 3%(有限) |
| 長 ctx(4K+)| full attention 60-76% → 省 20-30% decode,隨 ctx 增長 |
| CB 數 | 不變(仍在 cb1 內) |

- **數值風險**:f16 MPP 累加順序不同 → logits 可能微差(與 prefill 同源,已證無害);必須逐字節驗證
- **並行風險**:單趟 2 TG 失去 flash-decoding 的 chunk 並行;長 ctx 若變慢需改 chunked-tensor 變體(phase 2)
- **覆蓋風險**:SWA(25/30 層)本輪不動

#### 9.5 驗證計劃

1. 構建 → decode tensor 路徑 on/off 輸出**逐字節一致**
2. `[gpu] attn` 對比(短 ctx 128 tok)+ 長 ctx(600+ tok)對比
3. 3-way 交錯 A/B(同機、防頁緩存漂移),取中位

### 9.6 B3 Phase 1 實作結果(2026-08-07)

**實作**:decode `Attention.swift` 複用 prefill tensor-ops kernel(queryCount=1),env `TURBO_FIELDFARE_ATTN_TENSOROPS=1` gate(默認 off),只服務 full 512/16/2 層,單趟直寫 attnOut(省 partial+combine)。零 MSL 新增。

**正確性**:tensor on/off 輸出**逐字節一致**(4211ccfe),多次復現。f16 MPP 累加與 tiled 路徑在 argmax 層面零分歧。

**A/B(6 輪交錯,中位)**:

| 場景 | base attn | tensor attn | base tok/s | tensor tok/s |
|---|---|---|---|---|
| 短 ctx(163 pos,128 tok) | 2.81s | 2.82s | 10.41 | 10.28 |
| 長 ctx(~600 pos,600 tok) | 11.83s | 11.34s(−4%) | 15.22 | 15.97(+5%) |

**結論:短上下文中性、長上下文微正(噪聲內)**。原因:

1. **decode 只有 1 個 query → 2 TG 的 MPP 利用率極低**;prefill 的 11x 紅利來自大量 query TG 並行,decode 沒有
2. **tiled flash-decoding(16 chunk × 2 KV = 32 TG 並行)已足夠好**;單趟 tensor 2 TG 串行掃 key-tile 反而損失並行度
3. **full 層只佔 5/30**(short ctx ≈29% attn work)→ 任何 kernel 收益被稀釋 3-6 倍
4. **attn 只佔 wall 的 ~22%**(2.8s/12.8s);即便 attn 2x 也只省 ~8% wall;真正的洞是 CPU 側 sched(7.3s)

**Phase 2(若要繼續)**:chunked tensor——把 KV 範圍切成多段、每段一個 TG 跑 tensor kernel + 現有 combine 合併。這才能複製 prefill 的並行紅利;且只有上下文 >1024(SWA 封頂)後 full attention 佔比才 >50%。當前保留 env-gated opt-in,默認不動(tiled),零回歸。

### 10. B4 — CPU 側 sched/idle 深挖:多 queue 併發能省多少(2026-08-07)

#### 10.1 問題

decode 12.8s wall 中 GPU busy 只有 5.65s(44%),其餘是 CPU 側 sched/idle(12,760 個 CB)。假設:多 MTLCommandQueue + events 併發能填補 GPU 飢餓。

#### 10.2 新儀器:[cb-latency](GPU_TIMING gate)

per-cb1 分解(commit→gpuStart→gpuEnd→waitReturn),加入 `RealForwardRunner` + `[cb-latency]` diag 行:

```
[cb-latency] wait=7.24s gpu=2.68s wake=0.61s sched=4.03s other=0.00s ofWall=48%
```

- `wait` = waitUntilCompleted 總時長(每層一次,30 次/step)
- `gpu` = cb1 的 GPU span
- `wake` = gpuEnd → waitReturn(執行緒喚醒)
- `sched` = commit → gpuStart;實測 ≈ 前一層 routed+shared+phase1Hit 的排隊 GPU(3.8s)+ 真 dispatch 延遲(~0.2s,68-72us/cb)
- `other` = 0 → wait 完全被分解,無殘留

#### 10.3 多 queue 結論:≈ 省不到東西 — 依賴鏈是嚴格鏈

1. **沒有可並行的 GPU 工作**:每層 cb1 需要前一層 routed 的 GPU 輸出(hidden);routed 需要 cb1 的 router readback。第二條 queue 沒有獨立工作可跑
2. **GPU idle 4.04s 的歸因(`[gpu-idle-after]`)**:cb1→shared ≈0(shared 已 pre-commit)、routed→cb1 ≈0(cb1 已 pre-commit)、**shared→routed = 2.12-2.30s + phase1Hit→routed = 1.60-1.76s → 3.7-3.9s 是 GPU 等 expert fetch / routed commit**
3. **hot pool 驗證**:idle-after-sharedFFN 只從 2.30→2.12s(池已把 miss 壓到 ~13%)→ 殘餘 idle 是主執行緒 readback→plan→fetch-await→routed-encode→commit 的串行 CPU 下限(~0.6ms/layer ≈ 18ms/step),不是磁碟
4. **wake 實驗(continuation vs waitUntilCompleted)**:0.59s vs 0.59s **完全一樣** → wake 是 Metal 內部 completion 成本,執行緒阻塞方式無法縮減;已回退
5. **結論:MTLCommandQueue 併發對單 stream decode 延遲沒有可兌現收益**;GPU idle 是 IO-bound(fetch)與主執行緒串行工作,不是 queue 調度問題

#### 10.4 真正有效的槓桿(按 ROI)

| 槓桿 | 狀態 | 預期 |
|---|---|---|
| hot pool(固定高頻專家) | ✅ 已做(code +25% / prose +6%) | 直接縮小 fetch → idle |
| 更大的池(top-48 = 92.6% 覆蓋) | 選項 | 更多同向 |
| 主執行緒串行下限(0.6ms/layer) | 需深層重構(routed encode 移背景執行緒) | ~15-18ms/step,風險高 |
| wake 0.61s(5%) | 實測不可縮 | 0 |
| B1 CB 融合 | 已測回歸 | 0 |

#### 10.5 程式碼狀態

- 新增:cb1ScheduleNanos/cb1WakeNanos 計數器 + `[cb-latency]` diag 行(GPU_TIMING 才啟動,默認路徑零影響)
- 實驗:continuation wait(TURBO_FIELDFARE_CONT_WAIT)實測中性,已回退
- 正確性:默認路徑輸出逐字節一致

### 11. 三個候選優化的實測定案(2026-08-07)

#### 11.1 A1/A2 — prefill 並行讀:中性(SSD-bound,非讀取模式問題)

247-token prompt,交錯 3 輪中位 TTFT:

| 模式 | workers=2 | workers=8 |
|---|---|---|
| baseline | 9.07s | 9.24s |
| coalesced | 9.06s | 9.12s |
| layerLocalReadahead | 9.05s | 9.16s |

- 三種模式全在 ±3% 噪聲內;workers 8 反而略差
- 根因:prefill 讀取是 **SSD 吞吐 bound**(1.79-1.92GiB/s 穩定),coalesced/readahead 改不了磁碟吞吐
- **真正的 TTFT 槓桿是 hot pool**:同 prompt TTFT 9.19→7.87s(**−13%**)— prefill 讀取命中 pinned 專家
- 結論:A1/A2 模式保留為選項(server 場景可能不同),CLI 默認 baseline 不動

#### 11.2 桌面 CPU 搶佔:無安全可關目標(iCloud daemon 突發,已自行消退)

- 真兇:bird 30% + ContainerMetadataExtractor 26% + knowledge-agent 25% + coreduetd 21%(系統 iCloud 同步 daemon 突發)
- WindowServer 31%(必需);WorkBuddy/Freebuff(會話本身);模型檔非 fileprovider-backed
- 突發已自行消退(bird 30%→4.3%)→ tok/s 仍 ±30% 擺動是系統層噪聲(load 2.0+,4 users),非 App 可關
- 結論:沒有可安全關閉的桌面 App;等待 iCloud 同步完成即可

#### 11.3 routed encode → 背景執行緒:不值得做 — 修正先前估計

- **cb2(encode+commit)只有 0.07s/run = 0.7ms/step(0% of wall)** — 不是串行下限
- 串行下限是 expert pread:長 prompt readWall 5.78s = **37.5% of wall**、ofDecode 78%,SSD-bound 1.79GiB/s
- GPU idle 2.74s = sharedFFN 1.56s + phase1Hit 1.18s(等 fetch),不是等 encode
- 修正:先前「~18ms/step 串行下限的 60-70%」估計錯誤;encode 移背景執行緒最多省 ~0.7ms/step(1%)
- **真正剩餘槓桿**:更大的池(top-48,92.6%)、2-bit/3-bit 權重(讀取 bytes 減半,已 A/B)

### 12. 生產配置固化 + top-48 池實測(2026-08-07)

#### 12.1 top-48 池實測(code + prose,交錯中位,load 3.36 噪聲窗口)

| 配置 | code ttft | code tok/s | prose ttft | prose tok/s | RSS |
|---|---|---|---|---|---|
| nopool | 3.51-3.57s | 12-14 | 4.16s | 12.7 | ~2.7GB |
| pool32 | **3.02-3.05s** | 12.4-13.5 | **3.60s** | 9.6 | ~3.0GB |
| pool48 | 3.40-3.44s | 15.1(高窗 16-18) | 3.94s | 10.1 | ~3.0GB |

- **RSS 意外收穫:池幾乎零額外內存** — pool 用既有 slot buffer(64 slots × 30 層已分配),只是 init 預載,pool48 只 +288MB
- **pool48 decode ≥ pool32**(乾淨窗口 16-18 tok/s vs 14.8),但 **TTFT +0.35s init 預載稅**(4.8GB pread vs 3.2GB)
- 噪聲窗口下 prose 無法定案(code profile 對 prose 覆蓋本就較低)
- 正確性:pool48 輸出逐字節一致;roundtrip profile == shipped profile

#### 12.2 生產配置(已交付)

- **`bin/run_prod.sh`**:包裝全部驗證過的設定
  - `TURBO_FIELDFARE_EXPERT_SLOTS=64`(命中 95%)、`HOT_POOL=1` + profile(默認 48,`HOT_POOL_PROFILE_SIZE=32` 可切)、`--trust-receipt`(TTFT −3.9s)
  - adaptive MTP / early shared 已是 code 默認開
- **`bin/make_hotpool_profile.sh`**:一鍵生成 profile(trace → gen_hotpool_profile.py → 與模型同目錄 `profiles/top<N>_code.json`)
- **`Scripts/gen_hotpool_profile.py`**:profile 生成器(已入 repo)
- **`models/profiles/top32_code.json` / `top48_code.json`**:shipped profiles
- 驗證:prod 路徑輸出逐字節一致、roundtrip 可重現、錯誤路徑有清晰提示

### 13. 非阻塞池預載(TURBO_FIELDFARE_HOT_POOL_PRELOAD=async,2026-08-07)

#### 13.1 動機

pool48 的 init 同步預載 4.8GB 阻塞在 prefill 首觸路徑(streamer lazy 建立),TTFT +0.35s 稅。

#### 13.2 實作

- init 時**先 pin 池 slot**(slotPinned=true、slotOwnerPhase=.sharedResident,零讀取)→ planner 永不移除/分配
- slotExpert 保持 -1 直到資料落地 → 未載入的池專家只是普通 miss(載入 LRU slot),池「尚未熱」而非錯誤
- 資料由後台任務載入(2-deep 併發 semaphore + utility QoS,跨 30 層不 thrash SSD),完成後 cacheLock 下寫 slotExpert/hitCount
- 任務持強引用 → streamer/slotPointers 不會提前釋放
- env `TURBO_FIELDFARE_HOT_POOL_PRELOAD=async`,默認 sync 保持原行為

#### 13.3 A/B(code prompt,4 輪交錯中位)

| 指標 | sync | async | Δ |
|---|---|---|---|
| TTFT | 3.37s | **3.02s** | **−0.35s(稅全回收,4/4 輪一致)** |
| decode | 12.31 | **14.79** | **+20%** |

- 256-token 驗證:decode 命中率 **94.7%**(池在 decode 期間填滿並生效)、TTFT 3.06s 保持
- 正確性:sync/async 輸出**逐字節一致**
- 額外:async 下 readWall 吞吐 2.18GiB/s(>sync 的 1.9),背景預載順帶利用 decode 的空閒 SSD 窗口
- run_prod.sh 已默認 async;`TURBO_FIELDFARE_HOT_POOL_PRELOAD=sync` 可回退

### 13.1 MTP_MODEL 旋鈕接入 run_prod.sh(2026-08-07)

- `bin/run_prod.sh` 新增 `MTP_MODEL` env 旋鈕:
  - 默認:`/Users/alexchuang/Documents/flashkv0516/models/gemma-4-mtp-head`(MTP 開)
  - `MTP_MODEL=""`(注意:空字串,非 unset)→ MTP 關,純 base decode
  - 用的是 `${VAR-default}` 而非 `:-`,讓「空字串=關閉」與「unset=默認」可區分
- 腳本同時拒收 caller 傳 `--mtp-model`(與 `--model` 同規則,避免雙源分歧)
- bash 3.2 + `set -u` 空陣列展開 bug 已用 `${MTP_ARGS[@]+"${MTP_ARGS[@]}"}` 慣用法修復
- 接上後 **adaptive gate 自動生效**(code 默認 ON,`TURBO_FIELDFARE_MTP_ADAPTIVE=0` 可關):
  - 實證 footer:`mtp=10/12(83%) adaptive(d=0 red=0 off=0 row=...)` — MTP 與 gate 同時活躍
- 128-token 6 輪交錯 A/B(code prompt,load 2.7 噪聲窗口):
  - base 中位 **12.90 tok/s** vs MTP-adaptive 中位 **15.31 tok/s** → **+18.6%**
  - session 歷史最佳(code):adaptive 14.4 vs base 11.5 → +25%
  - prose:gate 低接受時自動 d=0 走 decode path,≈base 不拖累(結構保證)
- 結論:code 類 prompt +15~25%,prose ≥0%(gate 保證不劣於 base),TTFT 不受影響

### 13.2 MTP-adaptive 接受率 + verify 可省性審計(2026-08-07)

**現狀(code prompt, 實測)**:
- 接受率 **84%**(27/32 drafted),probe 窗滾動接受率 100% — draft 品質本身健康
- tok/s 15.4(乾淨窗),+18.6% vs base(6 輪交錯中位);decode 緩存命中 95.9%
- **gate 過度禁用**:`TURBO_FIELDFARE_MTP_ADAPTIVE_DEBUG=1` 逐 step 顯示 — calib 後 probe(d=2,acc=1.00)跑 8 步即禁用,剩餘 ~60% 步數走 d=0 decode path,全程 `stepsAtMaxDraft=0`
- 根因:8-step 窗口 + ±5% 帶 + hardLow=-10% 在機器 ±30% 噪聲下統計無意義;MTP 步 >3.33× base 步即誤判禁用。clean 窗 A/B(歷史)adaptive 保持 d=3 吃到 +25%,噪聲窗則大幅禁用以致 +18.6%

**verify 結構審計**:
- 已最優:單 forward batch 驗證全部 draft(1+D 行,`verifyBatch→prefillChunked` 單 span);pool 讓 verify expert 並集共享;gate 自動調 draft 數
- 兩段式 verify(先驗 draft[0] 再 batch 其餘):α=0.84 數學省 ~12% verify 行,但加一次 forward 固定開銷(~85μs CB + readback),淨收益≈0,不建議
- verify 層稀疏(batched forward 下只驗必要層):架構不可行(所有行共享層管線)
- 真槓桿:**draft 品質**(84%→90%+ 直接多 ~0.5 token/cycle,零 runtime 成本,需 fine-tune MTP head)與 **gate 窗口穩定化**

**總體剩餘提升(依 ROI 排序)**:
1. gate 窗口穩定化(windowSize 8→32、rate EMA、禁用前多輪確認)→ 高接受率 run 全程吃 draft,+18.6%→+25%
2. MTP head fine-tune(接受率↑,零 runtime 成本)
3. B3 Phase 2 chunked tensor attention(長 ctx)
4. B1 CB 融合(8808 CBs,330us/cb)
5. 2-bit/3-bit 權重(讀取減半)

### 13.3 五項優化全推(2026-08-07) — 進度與實測

**① gate 窗口穩定化 — 已實作並驗證**
- MTPAdaptive.swift:decideEvery 8→16、新增 rateEMA(emaAlpha 平滑原始窗口速率)、禁用/降級需**連續 2 次輸判定**(單次噪聲窗口不能誤殺高接受率 prompt)
- 驗證(DEBUG trace):code prompt off=10→**off=1**,接受率 27/32→**70/80(88%)**,全程保持 draft;交錯 A/B(噪聲窗)+3.8%(EM A 更保守,但不再誤判禁用)
- 剩餘:EMA 使升檔更保守(停在 d=2),乾淨窗會升到 d=3-4

**② MTP head fine-tune — 審計:屬雲端工作流,Mac 端受阻**
- `app/training/train_mtp.py` 管線完整(collect_hidden_states → distill_train → validate_mtp_accept),但面向 **sglang + CUDA + /data/models**(host1)
- Mac 端缺:①本地無 HF gemma4 base(collect 需要)②collector 用 `CGC_Phase2.mtp_verify_loop`(sglang 遠端)③訓練產出 `.pt` checkpoint 無 →Metal drafter(safetensors)轉換器
- 正確執行位置:host1(`/data/models/gemma-4-E4B-it` + CUDA),需補 `.pt→safetensors` 導出器才能餵回本地 Metal drafter

**③ B3 Phase 2 chunked tensor attention — 未做(長上下文專屬)**
- Phase 1 已證:短 ctx(163pos)中性、長 ctx(~600)tok/s +5%;SWA 封頂後(>1024)full attention 才 >50% 佔比
- 設計:把 prefill tensor kernel(MPP matmul2d)的 KV 切段、每段一 TG + 現有 combine,複製 prefill 的並行紅利
- 當前 prompt(35-600 tok)無收益,列為長上下文工作負載的後續項

**④ B1 CB 融合 — 審計:已完成,無殘餘**
- decode loop 現況:每層 2 CB(cb1 = GQA-full 單 kernel attention + router + lookahead;sharedCB early-commit;routed 非同步),8808 CBs/128tok ≈ 2.3/layer — 原 12,416 CB 是舊內核
- `TURBO_FIELDFARE_FUSE_SHARED`(shared 併入 cb1)已在乾淨機器實測**回歸**、默認 off:併入會延遲 router readback → routed 晚啟動,破壞 CPU/GPU 重疊

**⑤ 2-bit/3-bit 權重 — 已實作並實測(r2 repack + A/B)**
- `TurboFieldfareRebits --input gemma4.gturbo --routed-bits 2` → `models/gemma4-r2.gturbo`:packed 12.9→7.2GB(−44.4%),權重空間 RMSE gate 0.0125/up 0.0127/down 0.0190
- 6 輪交錯 A/B(code,128tok):**TTFT 3.56→2.25s(−36.7%)、tok/s 11.07→18.57(+67.7%)**
- 品質:固定 prompt 對照 — r2 輸出正確且更工整的 fibonacci memoization(無退化)
- **r2 上 MTP 接受率降**(46% vs r3 77%):量化偏移 target 分佈,draft head 對齊度下降
- `run_prod.sh` 新增 **MODEL_BITS=2|3**(默認 3,`TURBO_FIELDFARE_MODEL` 仍優先);`MODEL_BITS=2` 生產路徑驗證通過(TTFT 2.2s)

### 13.4 審查修正 + r2 profile 澄清(2026-08-07)

- gate 審查 2 個問題已修:①移除 write-only 死狀態 `winningDecisions` ②revive 分支會比對**凍結的舊 EMA** — 改為 probe 啟動時 `rateEMA=0` 重置,revive 用新鮮 probe 窗(避免 stale-high 誤復活/stale-low 永不復活)
- 第一決策仍無 EMA 平滑(seed=單樣本),真正改善來自 window 8→16;禁用需連續 2 輸 → 主題轉移最壞 ~32-48 cycle 才關(可接受,bounded)
- **r2 profile 澄清**:`run_prod.sh` 的 `$(dirname "$MODEL")/profiles` 解析到 `models/profiles/`(**共享目錄**,非每變體目錄)——r2/r3 同架構、re-quantization 保留 expert ID,r3 的 top48 profile 對 r2 有效;`MODEL_BITS=2` 生產路徑驗證通過
- gate 最終驗證:code prompt off=10→**off=1**,接受率 **88%**(70/80),無誤判禁用;draft 停在 2(EMA 保守,乾淨窗升 3-4)

### 13.5 Perplexity 評估工具 + r2/r3/r4 實測(2026-08-07)

**新增 `--perplexity <corpus.txt>` 模式**(TurboFieldfareCLI):
- `Args.swift` + `Run.swift` + `Command/main.swift`:logits head(forceLogitsHead)+ 逐 token produce + 讀 [vocab] FP16 logits buffer,logsumexp softmax 算 NLL
- 輸出:`ppl`(幾何均)/ `medPPL`(魯棒中位)/ `meanNLL` / `medNLL` / `p95NLL` / `argmaxAcc`(argmax 命中率)
- 注意:instruct model 對 raw prose 是 OOD(無 chat template 時 medPPL 高達 ~400K、近均勻)——**要套 chat template 才有意義**
- 驗證:重複文本 ppl=1.62(測量正確);fix 過程踩到 Args.swift 的 `modeMissing` 校驗(--perplexity 需豁免)+ repo 增量緩存坑(需 `swift package clean`)

**三變體 ppl(chat-templated corpus,178 tokens)**:

| variant | medNLL | medPPL | argmaxAcc |
|---|---|---|---|
| r4 | 9.43 | 12,464 | 23.0% |
| r3 | 10.78(+1.35) | 48,273 | 23.6% |
| r2 | 12.08(+2.65) | 175,989 | 16.3% |

**三變體 TTFT/decode(生產配置,4 輪交錯中位,load 3.6 噪聲窗)**:

| variant | TTFT | decode tok/s |
|---|---|---|
| r4 | 4.15s | 10.69 |
| r3 | 3.62s | 14.63 |
| r2 | 2.72s | 14.36 |

- r4 歷史最佳:TTFT ~4.0s(--trust-receipt;含全量 hash 時 8.05s)、decode **15.9 tok/s**(乾淨機,§8 B1 交錯中位);優化前基線 6.82s / 8.35
- 修正:13.5 首測的「r4 10.69」是三方交錯時被 r2/r3 頁快取串擾的低值;r4-only 6 輪實測中位 12.28 / 峰值 15.34(load 3.3 噪聲窗)
- r2 乾淨窗峰值 decode 18.5 tok/s、TTFT 2.72s(讀取減半)
- 結論:r2 付 medPPL ×14 換 decode +34% / TTFT -34%(vs r4);品質敏感場景用 r3,速度優先用 r2

### 13.6 top-64 池 + sync preload(2026-08-07)— MTP 在 r4 的最終裁定 + pool64 意外勝利

**動機**:測試「r4 + pool64 + MTP-adaptive」三方對照,確認池補足 IO 後 MTP 在 r4 是否轉正。

**top-64 profile 生成**:覆蓋率 **97.1%**(vs top-48 的 92.5%),profile 為 r2/r3/r4 共享(同架構、expert ID 保留)。

**第一輪 A/B 陷阱(重要教訓)**:pool64 初測崩到 5.3 tok/s、swap 14.1/15.36GB — 一度誤判為「96 slots × 3.2MB × 30 層記憶體爆掉」。鑑別實驗證明**根因是 async preload 的 6.1GB 背景讀取在 decode 期間與 miss 搶 SSD 頻寬**(2 並發 semaphore 把讀取攤到整個 decode),不是記憶體(posix_memalign 是 lazy commit,RSS ~2.5GB 安全)。swap 高企是 18-run 超時測試的惰性殘留,非即時瓶頸。

**r4 對照(4 輪交錯中位,64 tok,code prompt)**:

| 配置 | TTFT | decode | 總帳(64 tok) |
|---|---|---|---|
| pool48 + async | 3.84s | 9.04 tok/s | 10.92s |
| **pool64 + sync** | 4.50s | **14.84 tok/s** | **8.81s(−19%)** |

- pool64-sync 單項最低 10.3 仍贏過 pool48 最高 9.6 — 定案級別
- 總帳延伸:32 tok −10%、128 tok −27%、256 tok −32%、512 tok −36%(sync 的 0.66s 稅完全被 decode 加速覆蓋)

**r3 對照(生產默認,4 輪交錯中位)**:pool64-sync **15.98 tok/s(+17.8%)**、TTFT +23.8%(0.75s 稅);64 tok 短生成總帳打平(7.89 vs 7.93s),長生成明顯贏。

**MTP 在 r4 的最終裁定:確認負收益,不轉正**。即使 SSD 乾淨、覆蓋率 97.1%:

| 配置 | decode | 接受率 |
|---|---|---|
| pool64 + sync base | 14.70 tok/s | — |
| pool64 + sync + MTP adaptive | 5.78 | 63% |
| pool64 + sync + MTP fixed(d=4) | 7.14 | 49% |

r4 接受率(63-72%)低於 r3(84%),verify 的 batched forward 計算成本壓不過 draft 紅利;且 MTP loop 的 d=0 步本身有 ~2× base 的序列化 sync 開銷。**MTP 不是 r4 的槓桿** — 保持 r3 上的「adaptive + pool」組合即可。

**交付**:
- `run_prod.sh` 支援 `HOT_POOL_PROFILE_SIZE=64`(32|48|64)
- 聯動:pool64 → `TURBO_FIELDFARE_EXPERT_SLOTS=96` + `HOT_POOL_PRELOAD=sync`(pool64 必須 sync,async 會崩 decode −50%);32/48 → 64 slots + async 不變
- 驗證:prod script 三路徑全通(64/sync、默認 48/async、非法 50 拒絕)

**一句話**:pool64 + sync 是 decode 之王(r4 +64%、r3 +18%),MTP 對 r4 定案為不轉正;async 只適合小池,pool64 上必須 sync。

**定案(§13.6 追加)**:r4 128-tok 4 輪交錯 — p64-sync **中位 14.34 / 峰值 16.65 tok/s(破 15.9 歷史紀錄)** vs p48-async 中位 10.87(+31.9%)、總帳 13.5 vs 15.5s(−12.5%)。`run_prod.sh` 默認 `HOT_POOL_PROFILE_SIZE` 改為 **64**(自動 96 slots + sync preload);MTP_MODEL= 關 MTP、HOT_POOL_PROFILE_SIZE=48 回退 async 均端到端驗證通過(默認 128 tok 實測 14.05 tok/s、MTP 接受率 88%)。

### 13.7 B3 tensor-ops attention 定案(2026-08-07)— 已存在、+7.9%、設為默認

**審計發現**:`TURBO_FIELDFARE_ATTN_TENSOROPS` 的 single-pass MPP tensor-ops decode attention **kernel 早已存在且完整接線**(Attention.swift `psoDecodeTensorOps`,512/16/2 full-attn shape),只是 env-gated 默認 off 且從未測過性能。

**A/B(r4 + pool64-sync,128 tok,4 輪交錯中位)**:tensor-ops **14.23 vs base 13.19 tok/s(+7.9%)**,TTFT 中性。byte-identity 驗證:code prompt + 2 個 prose prompt 全部 `md5` 一致。

**GPU 拆解(關鍵洞察 — 收益來源不是 GPU 加速)**:

| 指標 | base | tensor-ops |
|---|---|---|
| GPU attn | 2.91s | 2.93s(沒變) |
| cb1Wait | 10.06s | 9.55s(−0.5s) |
| overhead/cb | 277us | 230us(−17%) |

tensor-ops 把每層 partial+combine 兩次 dispatch 鏈合成 single-pass,**省的是 CPU 側 dispatch/schedule 稅,不是 GPU kernel 時間**。

**剩餘空間拆解(回答「16.65 之後還有什麼」)**:完整 stage/gpu/sync 拆解顯示:

```
[stage] cb1=0.2s io=1.8s cb2=0.1s head=1.3s gpuWait=19.3s(85%)
[gpu]   attn=2.7s routedMoE=2.7s shared=1.1s head=1.3s busy=7.9s ofWall=35%
[sync]  cbs=11957 cb1Wait=13.7s sched=9.9s overhead=7.3s(613us/cb)
[cb-latency] wait=13.7s gpu=2.7s wake=0.5s sched=10.6s(!!) other=0
```

- **最大單一成本是 sched=10.6s(CPU commit → gpuStart 排隊延遲),不是 attention** — 是「GPU 跑完一批等 CPU 提交下一批」的 pipeline 深度問題(B4 的獵物),遠超 attention 的 2.7s
- GPU busy 只有 35-47% of wall — **GPU 有一半以上的時間在等**,調度空轉才是主矛盾
- attention 2.7s 已不是最大 GPU 單項(與 routedMoE 2.7s 並列),B3 的收益已兌現(+7.9%)

**交付**:`run_prod.sh` 默認 `TURBO_FIELDFARE_ATTN_TENSOROPS=1`(opt-out =0)。prod 默認 128 tok 實測 11.84 tok/s(MTP 88% 接受率)。

**一句話**:B3 不值得「重寫」— 它早已存在,開默認 +7.9%;真正剩下的洞是 sched 10.6s 調度空轉(B4 pipeline 深度),不是 attention。

### 13.8 B4 pipeline 深挖 — hit-only sync fetch(2026-08-07)

**動機**:sched 10.6s 是最大單一成本。調查「提前提交下一層 cb1 / 多 CB 在飛」可行性。

**結構性結論:「提前提交下一層 cb1」架構上不可行** — 下一層 attention 依賴本層 routed 的 hidden 輸出,是硬數據依賴。decode 是每層 `commit cb1 → waitForCompletion(cb1) → router readback → plan → fetch → commit routedCB` 的嚴格串行鏈。

**真正的獵物**:GPU idle 拆解(gpu-idle-after)顯示 **sharedFFN 後 4.26s(3800 個 ~1.1ms 小 gap)** — 即使全 hit,每層 fetch 都走 `withCheckedThrowingContinuation + DispatchQueue.global.async` 的跨執行緒 hop。

**實作**:`fetchRoutedExpertsHitOnlySync` — plan.misses 為空(全 hit)時同步執行,跳過 continuation+dispatch hop。`TURBO_FIELDFARE_B4_HIT_ONLY_SYNC` 默認 ON(`=0` 關閉),審查後 guard 取代 precondition、加 ModelError case。

**A/B 結果 — 結構改善但吞吐中性**:

| 指標 | base | B4 | Δ |
|---|---|---|---|
| gpuIdle | 9.19s | 3.41s | −63% |
| overhead/cb | 597us | 162us | −73% |
| cb1Wait | 13.05s | 7.12s | −46% |
| >5ms 長 gap | 180x | 43x | −94% |
| **decode(4 輪交錯中位)** | 12.46 | 12.56 | **+0.8%(中性)** |

**關鍵洞察(矛盾即真相)**:消除 63% GPU idle 沒有轉成吞吐 — **decode 是 CPU 串行鏈瓶頸(latency-bound),不是 GPU 吞吐瓶頸**。每層 `waitForCompletion(cb1)`(GPU 完成 attention+router 才能 readback indices)是物理上不可省的一環;GPU 有閒置容量但 CPU 必須等。sched 排隊延遲的根源是串行鏈深度,不是提交節奏。

**保留決策**:審查 flag「off-by-default 是死代碼」。B4 設默認 ON(嚴格更簡單:相同工作、更少 hop、byte-identical、無風險);與 PRELOAD=async 的 cacheLock 交互已註記為 latent(prod 默認 sync preload)。

**一句話**:B4 證明「GPU idle 消除 ≠ 吞吐提升」— decode 的牆是 CPU 串行鏈(waitForCompletion 每層一次),不是 GPU。突破它需要真正的架構變化(如 MTP verify 的多 token 並行),而非更快的單步提交。

### 13.9 GPU span 時間線全分析(2026-08-07)— idle gap 的真面目

**工具**:Run.swift 新增 `TURBO_FIELDFARE_GPU_TIMELINE_CSV=<path>` — dump 完整 span 序列(start/end/dur/label/gap)。配套 `bin/analyze_gpu_timeline.py` 深度分析。

**數據(256 tok,pool64-sync+tensorops+B4,24086 spans)**:gpuBusy=14.97s gpuIdle=11.13s window=26.07s。

**gap 分布 — 不是「千個亞毫秒縫隙」也不是「幾個長停頓」,是「7650 個 ~1.2ms 規律縫隙 + 少數 miss 長停頓」**:

| bucket | count | 時間 |
|---|---|---|
| <0.2ms | 17755 | 0.20s |
| 0.2-1ms | 3695 | 2.02s |
| **1-5ms** | **2364** | **4.51s** |
| 5-20ms | 212 | 2.21s |
| 20-100ms | 59 | 2.09s |
| >100ms | 1 | 0.10s |

**正確歸因(修正:gap i 是 span i 之前的空閒 = span i−1 之後的空閒)**:

| 空閒位置 | n | 總時間 | 含義 |
|---|---|---|---|
| **after sharedFFN** | 7650 | **9.06s(81%)** | **每層唯一真縫隙**:CPU readback→plan→fetch→commit routedCB 鏈 |
| after phase1Hit | 626 | 1.74s | hit phase1 後等 miss pread |
| after routed | 7650 | 0.10s | routed→cb1 無縫(pipeline 重疊生效) |
| after cb1 | 7650 | 0.01s | cb1→shared 無縫(early shared commit 生效) |
| after head | 509 | 0.24s | token 邊界 |

**長停頓(>5ms)全部是 miss pread**:sharedFFN→routed 183x(2.97s)+ phase1Hit→routed 56x(1.00s),即 20-100ms 的 59 個長停頓 = miss 專家讀取。1-5ms 主力縫隙(1799x=3.40s)也是同一條 sharedFFN→routed 鏈,只是 CPU 快時較短。

**drift 測試**:sharedFFN→routed gap 隨上下文**縮短**(前 3 token med 1.46ms → 後 3 token 0.24ms)— 因為頁快取預熱後 CPU 側 fetch 更快,且 GPU attn 增長被 tensor-ops 消化。**不是 attn 增長問題**。

**一句話**:GPU idle 81% 是「每層 sharedFFN 完成後等 CPU 準備 routedCB」的規律縫隙(平均 1.2ms×7650)+ miss pread 長停頓。routed→cb1 與 cb1→shared 已完美重疊。這與 §13.8 B4 結論一致:縫隙是 CPU 串行鏈,不是提交節奏 — 唯一真正有效的下一個槓桿是 miss pread 的長停頓(2.09s,pool 再擴大)或打破串行鏈的多 token 並行。

### 13.10 最後兩個槓桿的證偽(2026-08-07)— pool80 與 prefetch 都不行

**前置修正**:時間線重算後,>5ms miss 停頓其實是 **4.40s(272 個,17% of wall)**,先前報 2.09s 是低估。>20ms 的 60 個長停頓 = 2.20s 是 miss pread。

**槓桿 1:pool 擴大到 top-80 — 記憶體牆**。top-80 profile 覆蓋率 99.1%(vs top-64 97.1%),miss 理論減半。但 r4 expert 3.2MB × 80 × 30 層 sync preload = **7.7GB commit** + 模型 12.9GB 頁快取 → 16GB 機爆。實測 128 tok 只有 6.63 tok/s、swap 持續 10.4GB。**pool64 是這台機器的記憶體極限**。注意:同窗 pool64 也掉到 8.75(Doubao 51% CPU 搶載),「pool80 失敗」需乾淨機才算最終定案,但 7.7GB commit 的記憶體帳本身就是硬牆。

**槓桿 2:expert prefetch(下一層 router 預測提前讀)— 與 hot pool 衝突**。`TURBO_FIELDFARE_EXPERT_PREFETCH` 默認沒開;4 輪交錯 A/B:prefetch **9.54 vs base 16.25 tok/s(−41.3%)**。根因:pool64 下 99% 已是 hit,prefetch 對已 pin 專家做重複 pread,浪費 SSD 頻寬。該機制是「無 pool 時代」設計,與 hot pool 正交且衝突。lookahead=None 也確認預測路徑未有效運行。

**結論**:兩個候選槓桿都被證偽。剩餘空間的真實分布(r4,pool64-sync+tensorops+B4):
- miss pread 4.40s:pool 再擴大不可行(記憶體),prefetch 衝突 — **只能靠 r3/r2 更小 expert 或接受**
- sharedFFN→routed 串行縫隙 9.06s:§13.8 B4 已證 CPU 串行鏈,單步優化已到極限
- **真正剩下的路:多 token 並行(MTP verify 泛化)或位寬切換(r2 = TTFT −37% / decode +68%)**

**生產默認不變**:pool64-sync + tensor-ops + B4(hit-only sync)+ adaptive MTP + trust-receipt。

### 13.11 EXPERT_READ_WORKERS=8 定案(2026-08-08,重開機乾淨窗)— +11.1% decode,零 TTFT 稅

**動機**:用戶提醒「多 worker 之前設定成 2」。查證:`TURBO_FIELDFARE_EXPERT_READ_WORKERS` 控制 decode+prefill 並行 miss pread 深度,默認 **2**(`boundedParallelMissReadWorkersDefault`),生產沒設。decode 每層最多 topK=4 個 miss,worker=2 時 SSD 深度不足。

**split-worker 實作**:`executeParallelMissReads` 加 `workerLimit` 參數;prefill 分支(含 MTP verify 的 prefillChunked)cap 在 `min(2, ...)`,decode 用完整 env 深度。理由:prefill 高並行會與 hot-pool sync preload 搶 SSD 頻寬(TTFT 稅)。

**重開機乾淨窗 A/B(swap=0,6 輪交錯,r4 pool64-sync+tensorops)**:

| 指標 | w2 | **w8-split** | Δ |
|---|---|---|---|
| decode med | 13.89 | **15.43** | **+11.1%** |
| best3 平均 | 17.0 | 18.1 | +6.5% |
| TTFT | 4.62s | 4.63s | **+0.1%(稅消失)** |

- **先前重載下的「中性」被證實是 load 污染**(±30% 噪聲);乾淨窗下 w8-split 每輪 ≥ w2
- MTP on 對照:10.68 vs 9.61(+11.1%),接受率同 72% — **verify 的 prefill-cap 無害**(batched 讀取量小)
- byte-identity 驗證 ✓

**審查閉環**:coalesced/readahead prefill 模式繞過 cap(opt-in 非默認,已理解);MTP verify 用 prefill plane 被 cap 已實測無回歸;magic number 改為複用 `boundedParallelMissReadWorkersDefault`。

**交付**:`run_prod.sh` 默認 `TURBO_FIELDFARE_EXPERT_READ_WORKERS=8`(opt-out env 覆蓋)。生產默認更新為:pool64-sync + tensor-ops + B4 + **w8 split-workers** + adaptive MTP + trust-receipt。

### 13.12 GPU busy 65%→85% 的串行鏈精確歸因(2026-08-08)— sched 不是洞,wake+chainWall 才是

**動機**:用戶要求把「GPU busy 65%→85% 的串行鏈縫隙」精確歸因 — sched vs wake vs cb1wait 各佔多少、哪些 CPU 可控、哪些是 Metal 排隊硬限制。

**新增診斷能力**(3 處計時點):
- `[cpu-chain]` 行:readback / plan / io / cb2 四段 CPU 純計算牆鐘(Run.swift)
- `chainWallNanos`:cb1 `waitForCompletion` 返回到 routedCB.commit 的完整牆鐘(決定性測量)
- `cb1WaitCallNanos`:waitForCompletion 呼叫本身的牆鐘

**乾淨窗實測**(r4,pool64-sync+tensorops+B4+w8,MTP off,256 tok,decode 19.8 tok/s):

```
[stage]      cb1=0.29s io=1.27s cb2=0.08s head=1.28s gpuWait=14.65s wall=17.58s
[cpu-chain]  readback=0.01s plan=0.04s io=1.27s cb2=0.08s total=1.40s
             chainWall=1.66s unaccounted=0.26s cb1WaitCall=9.33s ofWall=8%
[cb-latency] wait=9.33s gpu=3.34s wake=0.94s sched=5.15s other=0.00s ofWall=53%
[timeline]   gpuBusy=8.41s gpuIdle=4.49s gaps=24049 1-5ms:1057x=1.96s >5ms:32x=0.21s
[gpu-idle-after] sharedFFN=3.32s(7639x,0.4ms avg) phase1Hit=0.92s(601x,1.5ms avg)
```

**三個關鍵修正(推翻 §13.8 的初步結論)**:

1. **sched 5.15s 不是「調度空轉」— 大部分是正常 pipeline 重疊**。cb-latency 的 sched = cb1 commit→gpuStart,這段時間 GPU 在跑**前一層的 routedCB + sharedFFN**(gpuRoutedNanos 5.16s 就是證據)。cb1 排在 routed 之後是正確的流水線 — GPU 有活幹,不是空轉。先前把 sched 10.6s 當最大成本是**歸因錯誤**:那是 GPU 忙前一層,不是 CPU 提交慢。

2. **CPU 準備鏈真實只有 1.66s(chainWall),不是 8.6s**。`[cpu-chain]` 四段加總 1.40s,chainWall(含所有縫隙)也只有 1.66s — unaccounted 僅 0.26s(34us/層)。先前把 gpuIdle-after-sharedFFN 的 8.62s 全歸給 CPU 準備是錯的:該 gap 的大頭是 **wake 延遲 + Metal 排隊**,不是 CPU 在忙。

3. **真正的成本分布(每層 2.3ms = 19.8 tok/s 的倒數)**:
   - **GPU 執行 1.10ms/層**(attn 3.34 + routed ~2.2 + shared + head):Metal 硬成本,只剩 kernel 優化
   - **wake 0.94s = 123us/層**:GPU 完成→CPU 喚醒的 Metal 完成處理器延遲 — **部分可控**(polling/spin 取代 blocking wait 可回收一半以上)
   - **chainWall 1.66s = 218us/層**:CPU 準備鏈(io 167us 為主)— **完全 CPU 可控**(hit-only fetch 再簡化、arg buffer 重用)
   - **sched 5.15s**:主要是 GPU 忙前一層的正常重疊 — **不是損失**

**哪些 CPU 可控、哪些是硬限制**:

| 成分 | 規模 | 屬性 | 可回收 |
|---|---|---|---|
| GPU 執行(attn+routed+shared+head) | 8.41s(48%) | Metal 硬成本 | 僅 kernel 優化 |
| wake(GPU 完成→CPU 喚醒) | 0.94s(5%) | 半可控 | polling 可收 ~0.4-0.6s |
| chainWall CPU 準備鏈 | 1.66s(9%) | **CPU 可控** | io 1.27s 是最大單項,hit-only 可再簡化 |
| sched(cb1 排隊) | 5.15s(29%) | 正常 pipeline 重疊 | 非損失,勿追 |
| 其餘(head 後空轉等) | ~1.4s(8%) | 混合 | 小 |

**一句話**:GPU busy 65% 的天花板不是 sched — sched 是假的。真實剩餘空間 = **wake 0.94s + chainWall 1.66s ≈ 2.6s(≈15% decode)**,其中 CPU 可控的是 chainWall 的 io 1.27s 與 wake 的一半。**B4 之後 io 仍是每層最大的 CPU 單項(167us/層)**:hit-only fetch 仍付 cacheLock + 計數器 + 雙重 view 構建稅,值得第三輪簡化。

### 13.13 wake polling + io 拆解(2026-08-08)— 推翻 §13.12 前提、wake 是實槓桿

**動機**:用戶要求「hit-only fetch 第三輪簡化(砍 io 一半)+ wake polling」同時做。加了 fetch 分支計數器([cpu-chain] 新增 hitIo/missIo)後,用資料說話:

```
[cpu-chain] io=1.12s hitIo=0.01s missIo=1.10s   (r4 pool64-sync w8, 128 tok)
[expert-cache] decode req=30480 hit=29927 rate=98.2%
```

**推翻 §13.12 前提:io 1.27s 不是 hit-path 稅,是 miss pread**。hit-only sync fetch 已接近免費(hitIo=0.01s ≈ 3us/層),「第三輪簡化 hit-only fetch」無從砍起。io 的 99% 是 miss 讀取:553-707 個 miss 專家 × 3.2MB ÷ ~1.75GiB/s ≈ 1.1-1.3s。miss 分布分析(256 tok trace vs profile):**798 個不同非 pool 專家,top-40 只覆蓋 20.6% — 純長尾**,pool promotion 無效。

**miss 槓桿窮盡清單**:
- pool 64→80:記憶體牆(7.7GB sync commit,§13.10 已證)
- slots 96→112(LRU 32→48,lazy 記憶體):**實測崩壞** — hit 率掉到 62.1%、讀 78GiB、decode 9.5 vs 12.3 tok/s(pool 載入異常,勿用)
- pool promotion(熱 miss 交換冷 pool 成員):長尾分布下 top-40 僅覆蓋 20.6%,ROI 極低
- **剩餘 miss 是 SSD 頻寬下限,唯一出路是 r3/r2 更小 expert(§13.4 已定案)**

**wake polling — 實槓桿**:

實作:`TURBO_FIELDFARE_WAKE_POLL_US`(0=off,默認 off;prod 設 5000)。decode loop 的 cb1 wait 改為 `waitForCompletionPolling`:先 spin `cb.status`(sched_yield 間隔,不飢餓其他執行緒),deadline 內完成即省掉 waitUntilCompleted 的 semaphore 喚醒;超時才 fallback blocking。fused 是 off(split path),sharedCB 在 cb1 後跑 ~150us,wake(120us)延遲 CPU chain 開始 → 暴露 188us/層;polling 讓 chain 提早 ~100us/層開始,且**對負載噪聲免疫**(parked thread 喚醒在負載下膨脹,spinning 不受)。

**A/B 數據(2026-08-08,重開機後)**:

| 場景 | off | poll-5000 | 說明 |
|---|---|---|---|
| 4 輪交錯(MTP off,負載噪聲窗) | 11.6 median | **15.7 median** | **+35%** |
| 診斷窗(load ~2-3) | 8.3-17.2 飄 | 16.3-17.1 穩定 | polling 消除負載敏感 |
| spin 窗口 1500/5000/20000us | — | 16.3/17.1/16.5 | 5000 是甜蜜點(覆蓋 avg wait + miss 讀) |
| 完整 prod(MTP on,3 輪交錯,有效 off) | 12.13 | **13.89** | **+14.5%** |
| byte-identity | — | ✓(code+prose md5 一致) | 只改等待機制,不動資料面 |

**成本**:decode 執行緒 CPU 平均 43%→70%(+28%,單一執行緒 sched_yield spin)。換來的是**任何負載下穩定 ~16-17 tok/s** — 對這台跑 Doubao/WindowServer 的機器是實際勝利。

**prod 默認**:`TURBO_FIELDFARE_WAKE_POLL_US=5000`(run_prod.sh 用 `${VAR:-5000}` 尊重 caller override,opt-out=0 有效)。prefill/TTFT 的 wait 未接 polling(長 kernel 上 spin 浪費 CPU,已確認 TTFT 無稅)。code review 修正:`.error` 終態立即 fallthrough、`&*` 防 overflow、missIo `&+` 一致性。

**一句話**:§13.12 的「hit-only fetch 第三輪簡化」前提錯誤 — io 是 miss pread 不是 hit 稅(hitIo=0.01s);真正的新槓桿是 wake polling,在負載下 +35%、安靜窗中性、byte-identical,已設為 prod 默認。

### 13.14 MTP 全面定案為默認關閉(2026-08-08)— 三種位寬全負收益

**動機**:13.13 後用戶要求驗證 r4 上 MTP 是否仍正收益(先前接受率 72-88% 是 r4 數據)。若 r4 也負,就把 MTP 默認關掉。

**r4 交錯 A/B(256 tok,load ~2.3,3 輪)**:

| 位寬 | MTP-on(adaptive) | MTP-off | Δ |
|---|---|---|---|
| r2 | 17.1 | **23.5** | **−37%** |
| r3 | 16.7-17.0 | **18.6-19.2** | −12% |
| r4 | 15.3 | **19.7** | **−29%** |

**三種位寬 MTP 全是淨負收益**。根因(base 變快後 verify 不划算 + gate 校準盲點,見 13.13):adaptive gate 的 d=0 baseline 走 MTP loop 量測(含 loop 自身序列化開銷),低估真 base 速度 → 誤判 MTP 划算而保持 drafts。

**執行**:run_prod.sh 的 `MTP_MODEL` 默認改為**空**(MTP off),保留 `MTP_MODEL=/path` 顯式開啟。`${MTP_MODEL-}` 語義:UNSET 或空 = off;只有顯式非空才開。

**新默認端到端(r3)**:MTP-off **20.7 tok/s** vs 舊 MTP-on 17.7 — 默認配置直接 +17%,且 r3 也破 18 紀錄。override 驗證:MTP_MODEL=gemma-4-mtp-head 正常回 MTP-on 17.7(mtp=66/80, acc=81%)。

**生產默認更新(完整)**:
- r3 默認 + pool64-sync + tensor-ops + B4 + w8 + wake-poll 5000 + **MTP off** + trust-receipt
- = **20.7 tok/s**(r3)、23.5(r2)、19.7(r4,MTP-off)

**一句話**:MTP 在 base 變快後全面負收益(verify 82% 計算成本 > 接受率收益),默認關閉;r3 默認現在 20.7 tok/s,超越歷史上限 18。

### 13.15 r2(MTP-off)GPU 拆解與天花板(2026-08-08)— 24.1 tok/s = 76% of ceiling

**動機**:r2 MTP-off 23.5-24.1 tok/s 為目前最佳,用戶要求拆解剩餘縫隙並精算天花板。

**完整拆解(256 tok,decode 10.61s,load ~2.3)**:

```
[gpu] attn=3.25s routedMoE=2.38s sharedFFN=1.08s phase1Hit=0.14s head=1.17s busy=8.03s ofWall=58%
[cb-latency] wait=7.70s gpu=3.25s wake=0.82s sched=3.72s
[cpu-chain] io=0.76s hitIo=0.01s missIo=0.75s total=1.64s chainWall=1.11s
[timeline] gpuBusy=8.01s gpuIdle=2.59s  >5ms stalls: 1x(0.01s) — 全消失!
[gpu-idle-after] sharedFFN=1.74s(7650x,0.23ms avg) phase1Hit=0.60s(638x,0.94ms) head=0.17s routed=0.07s
```

**天花板精算**:
- GPU 每 token = 8.02s/256 = **31.3ms → GPU-only ceiling = 31.9 tok/s**
- 85% busy 實務上限 = **27.1 tok/s**
- 現在 24.1 = **76% of GPU ceiling**(r4 是 46%,r2 因權重小 GPU 時間縮短而大幅改善)
- 剩餘 2.59s idle = 10.1ms/token:sharedFFN-after 1.74s + phase1Hit-after 0.60s + head 0.17s

**每層預算(精確歸因)**:
- GPU 執行 1.04ms/層;CPU chain 252us/層(chainWall 1.11s + wake 0.82s);shared FFN GPU 141us/層
- **hit 層:chain ~150us ≈ shared 141us → 已近最佳**(med idle 0.13ms)
- **miss 層(784 請求 = 1.3%):chain 膨脹到 ~2ms → 全部 idle 來源**(0.60s phase1Hit-after + sharedFFN-after 大頭)
- w8 + r2 2.15MB 小權重已消滅 >5ms 長停頓(1x only)— 上一輪 miss 問題只剩短暴露

**下一個槓桿(排序)**:
1. **miss 讀取(0.75s missIo + 0.60s phase1Hit-after ≈ 1.3s,潛在 ~12%)**:最後未試的 lookahead-miss-prefetch — 在 cb1 GPU 窗口用下一層 router 預測,只讀**非 pool 的 miss**(預測命中率 57%,避開 pool 成員 → 零冗餘 I/O、零鎖競爭)。回收一半 ≈ decode 10.61→9.9s ≈ **26.5-27 tok/s**
2. chain 殘餘(0.23s unaccounted + plan/cb2 0.12s):微
3. attn 3.25s(最大 GPU 單項):128-tok 下 B3 phase-2 增益小

**一句話**:r2 已到 76% of ceiling(24.1 vs 31.9),hit 層已近最佳;剩餘 2.59s idle 幾乎全是 784 個 miss 讀取的短暴露(w8 已消滅長停頓)。下一個實槓桿 = lookahead 只預取非 pool miss,潛在 24.1 → 26.5-27 tok/s。

### 13.16 lookahead-miss-prefetch 證偽(2026-08-08)— 預測長尾不準,全面負收益

**動機**:§13.15 指出剩餘 idle 幾乎全是 miss 讀取(0.60s phase1Hit-after + missIo 0.75s),下一個槓桿候選 = lookahead-miss-prefetch(在 cb1 GPU 窗口用下一層 router 預測,只讀非 pool 的 miss)。

**實作**(TURBO_FIELDFARE_MISS_PREFETCH=1,默認 off):lookahead 預測在 readback(L) 產出後立即 dispatch 非 pool 預測專家,async 預取到 L+1 的 LRU,下層 plan 前 drain。V2 加「只預取本 run 已 miss 過」(known-misser 過濾)。

**V1 結果(r2)**:機制成功 — missIo 0.76→0.32s(−58%)、phase1Idle 0.60→0.22s(−63%) — 但 **tok/s 24.1→21.5(−11%)**。missPrefetch=4985 次(65% 層),預測長尾大多錯誤 → 讀了 ~7500 專家(vs 實際 784 miss)浪費 SSD。

**V2 結果(r2)**:known-misser 過濾後 dispatch 3274(仍 43% 層),但 **missIo 沒降(0.78s)、tok/s 仍 −7%**。根因數據:
```
[lookahead] predicted=59160 hit=40295 rate=68.1% prefetchRead=102 drain=0.05s
```
- lookahead 整體準確 68%(但被 pool 成員主導;長尾子集遠低)
- **prefetchRead=102**:3274 次 dispatch 只實際讀了 102 個專家 — 其餘候選 dispatch 時已 resident(plan 說不用讀)→ 純白付 dispatch + bg 執行緒 + cacheLock 開銷
- 真正 784 個 miss 是「首次長尾」或「被預測漏掉」— 預測命中不了

**r3/r4 確認**:r3 off 21.4 vs on 21.1(中性);r4 off 17.8 vs on 16.1(更糟,權重更大 drain 暴露更多)。

**結論**:miss-prefetch 全面負收益 — 預測對長尾的準確率撐不起 speculative read。代碼保留 env-gated off(可複現實驗),**prod 未接**。r2 剩餘 2.59s idle 的 miss 部分確認無 cheap 槓桿:pool 記憶體牆(§13.10)、slots112 崩壞(§13.13)、prefetch 負收益(本節)。**r2 現狀 = 24.1 tok/s = 76% of GPU ceiling(31.9) 就是這台機器+此分布的實際終點,除非架構級多 token 並行。**


## 13.17 B3 phase-2 ROI 評估（attention core 成本歸因，2026-08-08）

用 env 開關（`TURBO_FIELDFARE_SKIP_ATTN_CORE/SWA/FULL=1`，默認 off，prod 未接）跳過
attention core kernel（QKV / epilogue / OProj / router 照跑），量 cb1 GPU 時間 delta。

### 架構事實

- `D=2816`、16 Q heads / 8 KV heads（GQA 2:1）、30 層、5 層 full-attn（mask 位元 1）
- SWA 層：`headDim=256`，走 tiled split path（partial + combine 多 kernel + 中間 buffer）
- Full 層：`headDim=512`，走 tensor-ops 單 kernel（`512/16/2` geometry）

### 決定性測量（256 tok、乾淨窗）

| 指標 | core on | core skipped | Δ |
|---|---|---|---|
| `[gpu] attn` | 3.58s | 2.29s | **core = 1.29s = cb1 的 36%** |
| decode | 13.08s | 8.43s | 19.6 → **30.4 tok/s** |
| GPU busy | 9.15s | 6.77s | −2.38s |

### 關鍵發現

1. **FLOPs 佔比預期完全錯誤**：seqLen=128 時 core 只有 ~1-2M MACs/層 vs QKV 投影 ~24M
   （理論 ~5-10%），但實測 **佔 cb1 GPU 的 36%** — core 是 launch-bound（7680 次
   encode、每次 ~168us 對 1-2M MACs 是災難級低佔用），不是 compute-bound。
2. **放大係數 3.6x**：core 省 1.29s GPU → decode 省 4.65s。cb1 是串行鏈第一個環節，
   提早完成 → 每層 wake/chain 提早開始 → 全鏈縮短。routed/shared/head 也各快
   （GPU busy 額外 −1.1s）。
3. **128 tok 交錯歸因**（3 輪中位）：skip-SWA 省 0.59s、skip-FULL 省 0.99s、
   skip-both 省 3.20s（on=10.48s）。SWA 25 層每層 0.024s、FULL 5 層每層 0.20s —
   **full 層每層成本是 SWA 的 8 倍**（512 headDim + tensor kernel 仍低佔用）。

### ROI 結論：值得做

- phase-2 合理目標：把 core 的 launch-bound 轉成 compute-bound（SWA split path
  合併成單 pass tensor-ops + full kernel 佔用改善），core 砍半 → 省 ~0.65s GPU
- 套用 3.6x 放大 → **decode 13.08 → ~10.8s → 23.7 tok/s（+21%）**
- 保守估計（放大 2x）：+10-15% decode
- 這是 miss-prefetch 證偽後目前唯一的實質 decode 槓桿（wake-poll 已定案 +35%）

### 優先級建議

1. **SWA split path → 單 kernel tensor-ops**（25/30 層，主體工作，中等工程）
2. **full tensor-ops kernel 佔用檢查**（5/30 層但每層成本 8 倍，先查為什麼低效）
3. 不做：長上下文 chunked（128-256 tok 下 core 已佔 36%，chunk 對短上下文無增益）


## 13.18 full-attn tensor-ops kernel 低佔用診斷（B3 phase-2 第一步，2026-08-08）

### 問題

§13.17 顯示 full-attn 5 層每層 cost 是 SWA 25 層的 8 倍（0.20s vs 0.024s）。
本節找出 launch-bound 的具體機制。

### Kernel 結構事實（attention_prefill_full_tensorops_2d_validity_v2）

- **復用 prefill 的 MPP tensor kernel**，decode 時 `queryCount=1` → grid 只有
  `width:1 × height:numQHeads/8=2` = **2 個 threadgroup × 128 threads = 256 threads**
- `kPrefillTensorOpsOutputs=8`：每個 TG 處理 8 個 Q head（1 個 KV-head group）
- `execution_simdgroups<4>`：128 threads = 4 simdgroup 跑 matmul2d（QK、PV）
- 每 chunk 迭代 64 keys，seqLen=128 → 2 次 key loop 迭代
- **softmax 段序列化**：`if (lid < kPrefillTensorOpsOutputs)` — 每次 chunk 迭代只有
  8/128 threads 工作（其餘 120 threads 空等 barrier），8 個 thread 串行掃 64 keys

### 對照：SWA split path（256/16/8）

- partial pass：`numKVHeads(8) × numChunks(8) = 64 TG × 256 threads = 16,384 threads`
- combine pass：16 TG
- 並行度是 tensor kernel 的 **64 倍**

### 為什麼 full 層 8 倍貴（綜合證據）

| 維度 | tensor kernel (full 512/16/2) | split path (SWA 256/16/8) |
|---|---|---|
| threadgroups | **2** | 64 |
| threads | **256** | 16,384 |
| 每 TG 工作 | 8 Q heads × 512 dim | 1 KV head × 1 chunk |
| softmax 並行 | **8 threads 串行掃** | spread across TG |
| headDim | 512（理論 2× SWA） | 256 |
| 實測每層 | 0.20s | 0.024s |

FLOPs 只有 2 倍，但並行度差 64 倍 + softmax 序列化 → 實測 8 倍。這是
**launch-bound / 低佔用**，不是 compute-bound。kernel 是為 prefill（大量 query
tokens，grid = queryCount × 2）設計的，decode 單 token 時 MPP 優勢吃不到。

### 決定性實驗：tensor-ops on vs off（同窗 GPU 對照）

| 指標 | tensor ON | tensor OFF (split path) | Δ |
|---|---|---|---|
| attn GPU | 2.62s | **2.02s** | **−0.60s（−23%）** |
| GPU busy | 7.08s | **5.35s** | −1.73s |
| cb1Wait | 7.58s | **5.42s** | −2.16s |
| sched | 4.67s | **3.13s** | −1.54s |
| byte-identity | — | **IDENTICAL** | 零品質損失 |

full-attn 走 split path（32 TG × 256 = 8,192 threads）比 MPP tensor kernel
（256 threads）快 23% attn GPU。**不需要等 phase-2 重寫 — 直接關 tensor-ops
就是免費的修復**。

### 誠實的不確定性

decode 牆鐘的 3 組交錯 A/B 方向不穩（off 勝 17%、off 勝 9.6%、on 勝 15%）—
負載 2.7-4.1 噪聲把 ±10-15% 差異淹沒。GPU-side 證據（同窗對照）一致指向 off，
但最終 decode 定案需要乾淨窗。

### 附帶 bug

`decodeTensorOpsEnabled` 是 `environment["TURBO_FIELDFARE_ATTN_TENSOROPS"] != nil` —
註解宣稱「opt-out with =0」但 `=0` 仍是存在即開。應改 `== "1"`。


## 13.19 更正：tensor-ops kernel 從未生效（2026-08-08，推翻 §13.18 結論）

### 調查過程

用戶要求修 `decodeTensorOpsEnabled` 的 `!= nil` bug（讓 `=0` 真正 opt-out），
然後做 tensor on/off 的乾淨 A/B。過程中發現更深的事實：

1. **run_prod.sh 無條件 `export TURBO_FIELDFARE_ATTN_TENSOROPS=1`** — 覆蓋所有
   caller 設定。修正為 `${VAR:-1}` 尊重 override。
2. 修完後跑三態（unset/zero/one）：attn GPU 完全相同（2.01 vs 2.01）→ 可疑。
3. 加 debug print：`decodeTensorOpsEnabled=true 但 pso=false` — **psoDecodeTensorOps 是 nil**。
4. 查 prefill.metal：tensor kernel 被 `#if defined(__HAVE_TENSOR__)` 包住。
5. 用獨立 Swift probe 驗證：M4 支援 apple9，但 **`__HAVE_TENSOR__` 未定義**，
   force-on 編譯失敗：`MetalPerformancePrimitives.h file not found`（CLT SDK 無 MPP header）。

### 事實

- tensor-ops kernel **從未編譯進 library**，`psoDecodeTensorOps` 永遠 nil
- decode 的 full-attn 512/16/2 層**一直走 tiled split path**（與 SWA 相同的
  partial+combine 機制，只是 headDim 512）
- `ATTN_TENSOROPS=0` vs `=1`：byte-identical 輸出 + 相同 GPU 時間（兩者都是 split）
- **§13.18 的「決定性實驗」（on 2.62 vs off 2.02）是負載假象**：當時 run_prod.sh
  無條件 `=1`，兩次都是 on，且都走 split path — 0.6s 差異是負載波動

### full-attn 8 倍貴的真實原因（重新歸因）

不是 tensor-ops（它不存在），而是 **split path 對 512/16/2 的 dispatch 本身**：

- full 層：`numKVHeads=2`（numFullKVHeads）× chunks → split partial TG 數少、
  headDim 512 → 每個 TG 工作量大
- SWA 層：`numKVHeads=8` × 8 chunks = 64 TG 並行
- 8 倍成本 = 512 headDim 的 2× compute + 更低的並行度/TG 工作（launch-bound）

### 已交付

- `decodeTensorOpsEnabled` 改 `== "1"`（真 opt-out 語義）+ 完整真相註解
- run_prod.sh 默認改 `0`（flag 在此 SDK 無效，誠實默認）+ 真相註解
- debug print 已移除
- 所有在 SDK 升級（有 MPP header）前，phase-2 的正確方向是**優化 split path
  對 512/16/2 的 dispatch**（如提高 partial TG 數 / 減小每 TG 工作量），
  而不是修 tensor-ops kernel（編不過）


## 13.20 SWA split path phase-2：partial+combine 合併單 kernel 改造方案（2026-08-08）

### 目標

SWA 25/30 層是 decode 主力。現行每層 attention core = 2 次 kernel launch
（partial + combine）+ 2MB partial scratch 往返。評估合併成單 kernel。

### Kernel 結構事實（讀過 attention.metal 確認）

**partial kernel（attention_decode_gqa_swa_partial）已經是 flash-attention 結構**：
- `m_run / d_run` online-softmax 在 TG 內累積，chunk 間不需要跨 TG 合併邏輯
- grid = `numKVHeads(8) × numChunks`，每 TG 掃 `chunk_len` 個 KV 位置
- decode 128-token：KV=128 → 8 chunks × 16 位置/TG — **每 TG 極短，launch 稅佔比高**

**combine kernel（attention_decode_combine）是純合併**：
- 每 Q head 一個 TG，讀 8 個 chunk 的 (m, d, o) partial，重算 global max/denom
- 無狀態，可完全內聯到 partial 的收尾（當 chunks=1 時 combine = identity copy）

### 合併可行性：高

因為 partial 已是 online-softmax，把 `numChunks=1`（每 KV-head 一個 TG 掃全部
KV）+ 直接寫 `out` 就能消除 combine pass。這是標準的「flash-decode → flash-attn
單 pass」收斂。

### 已做的 chunk sweep（TURBO_FIELDFARE_ATTN_CHUNKS，r1 乾淨窗）

| chunks | partial TG | 每 TG KV | decode | attn GPU |
|---|---|---|---|---|
| 1 | 8 | 128 | 8.81s | 2.44s |
| **2** | **16** | **64** | **7.69s** | **1.95s** |
| 4 | 32 | 32 | 9.00s | 2.31s |
| **8（默認）** | **64** | **16** | 8.27s | 2.11s |
| 16 | 128 | 8 | 9.41s | 2.40s |
| 32 | 256 | 4 | 10.09s | 2.57s |

（r2 被桌面負載污染 22-27s 異常，不計。load<2 需重測定案。）

### 觀察與矛盾

- **chunks=2 在 r1 明顯勝出**（7.69s vs 默認 8.27s，+7%；attn 1.95 vs 2.11，
  −8%）— 「每 TG 64 位置 × 16 TG」是 128-token decode 的甜蜜點
- **chunks=1（=合併單 kernel 的結構近似）反而 8.81s 較慢** — 8 個 TG 太少，
  M4 GPU 吃不完（32+ 執行單元）。單 kernel 不能走「每 KV-head 一個 TG」，
  需要 **8 KV × 更高 chunk 內平行** 或 **把 Q heads 也拆 TG** 才能填滿 GPU

### 改造方案（3 個可測選項，按成本排序）

**方案 A：改默認 chunk 策略（零 kernel 改動，最快可測）**
- 現有 chunkCount 固定 8（SWA）。改為 **依 KV 長度動態調**：
  `chunks = clamp(ceil(effLen / 64), 1, 16)` → 128-token 時 = 2，長上下文自動升
- 收益：r1 顯示 128-token 下 +7% decode（7.69 vs 8.27）；長上下文因 chunks 更多
  不受懲罰
- 成本：改 1 行 chunkCount 邏輯 + A/B 定案
- 風險：低（純參數）；需確認 256/512-token 的 chunk 最佳值不是簡單線性

**方案 B：單 kernel 但保持 TG 數（中等工程）**
- 寫 `attention_decode_gqa_swa_single`：grid = `numQHeads(16)`（每 Q head 一個
  TG），每 TG 掃全部 KV（128），online-softmax 直接寫 out
- 消除：combine launch（30 層 × 每層 1 次）+ 2MB partial scratch 往返
- 風險：16 TG × 256 threads 可能仍不足以填滿 M4 GPU（對照 full-attn 8 倍成本
  的教訓：TG 太少是 launch-bound 元凶）。保守起見 TG 數應 ≥ 32
- 預期：combine 省下的是 launch + scratch，估 attn GPU −10~15%

**方案 C：partial 內合併（工程最大，收益上限最高）**
- 保留 chunk 平行（TG 數高），但每 TG 內用 threadgroup 記憶體做 chunk 合併，
  combine pass 完全消失
- 實質是把方案 B 的「每 Q head 掃全部」改成「每 (KV-head, chunk) 掃 + 收尾合併
  到 threadgroup」，TG 數 = 64 保持
- 預期：方案 B 的收益 + 無 TG 數損失

### 建議順序

1. 先做方案 A（1 行 + A/B，r1 已有 +7% 信號）— 立即兌現
2. 方案 B/C 是 kernel 重寫，等方案 A 定案後再做（且需乾淨窗）

### 誠實限制

- chunk sweep r2 被負載污染，所有數字需 load<2 重測
- 方案 B 的「TG 太少」風險有實證前科（full-attn 8 倍成本就是 TG 太少 + 每 TG
  工作過大的組合），必須先驗證 TG 數下限


## 13.21 SWA split path phase-2：方案 A/B/C 實測結果（2026-08-08）

### 方案 A：動態 chunk 策略 — 定案：中性，不上 prod

- 實作：`chunkCount` 依 KV 長度動態調（目標每 chunk ~64 位置，128-token → 2）
- byte-identity ✓（dynamic vs forced-8 輸出一致）
- 4 輪交錯 A/B：dynamic 8.53s vs forced-8 8.43s（中位）— **無顯著差異**，attn 相同
- 結論：r1 乾淨窗的 chunks=2 +7% 是負載假象，未複現。env 開關保留，默認維持 fixed-8

### 方案 B：單 kernel（每 Q head 一 TG，掃全部 KV，直寫 out）— 定案：負收益

- 實作：新 `attention_decode_single` kernel（grid=16 Q heads × 256 threads，
  online-softmax 直寫 FP16 out，無 partial scratch、無 combine launch）
- byte-identity ✓
- 4 輪交錯 A/B（GPU 時間不受 CPU 負載污染）：

| 模式 | decode | attn GPU | busy |
|---|---|---|---|
| split（現行）| **8.16s** | **2.02s** | **5.42s** |
| single（方案 B）| 10.90s | 2.85s | 7.20s |

- **attn +41%（2.85 vs 2.02）— 16 個 TG 填不滿 M4 GPU**。每 TG 256 threads 串行
  掃 128 個 KV，對 4096 threads 的總量，GPU 執行單元大量空轉
- 之前 verify 的「−16%」是 split 撞負載尖峰（decode 19.4s）的假象，交錯定案為準
- 這實證 §13.20 預警：「TG 太少 = launch-bound」不是理論 — full-attn 8 倍成本
  是同一機制（TG 少 + 每 TG 工作大）
- 結論：**保留 env-gated off**，不接 prod

### 方案 C：partial 內合併 — 架構障礙，不可行（在此 kernel 結構下）

- combine 的 `m_glob = max over NC chunks` 需要**跨 TG 讀取**其他 chunk 的 partial
- Metal kernel 內**沒有 grid-wide sync**：partial TG 寫 scratch 後，同一 kernel
  無法等所有 TG 完成再合併（不同 TG 執行進度無保證）
- 唯一做法是 atomic flag 自旋等待（grid-sync hack）— partial TG 數 × 空轉風險，
  且 decode 每層都要等，複雜度/風險遠超收益
- 結論：不做。combine 本身是輕量 kernel（每 Q head 一 TG 讀 NC 個 partial）

### 綜合結論

SWA 25 層的 partial+combine split path 已是合理結構：
- partial（64 TG）提供 decode 需要的並行度 — 方案 B 證明 TG 數是硬約束
- combine 是必要的跨 chunk 合併（Metal 無 grid sync）
- 真正的剩餘空間不在合併方式，而在 partial kernel 本身的效率
  （kGQAPerThread 分攤、barrier 次數、Q/K/V 載入模式）

### 保留的實驗資產

- `TURBO_FIELDFARE_ATTN_CHUNKS`（方案 A 調參）
- `TURBO_FIELDFARE_ATTN_SINGLE`（方案 B，off）
- 兩者皆默認不影響 prod


## 13.22 slots112 崩壞根因：不是 planner bug，是急切分配記憶體

**結論：舊「slots112 災難」（62% hit / 78GiB 讀）在現行代碼下沒有重現。**
今天的交錯 A/B 與 RSS 實測推翻 planner 邏輯 bug 假設，鎖定根因為記憶體。

### 實測證據

| 量測 | slots=96 | slots=112 | 差 |
|---|---|---|---|
| miss（128 tok）| 985 | 985 | 相同 → planner 行為一致 |
| readWall | 2.19s | 2.39s | 幾乎相同 → IO 不是根因 |
| **peak RSS** | **2193MB** | **3718MB** | **+1525MB** |
| decode 交錯 | 16.42s med | 25.34s med | 112 慢 35% |

### 根因機制（程式碼證據）

```swift
// Model.swift: streamers[L] — 每層一個 streamer，各帶完整 slotCount
streamersBox.streamers[L] = try PreadExpertStreamer(layout:, slotCount: slotCount, ...)

// PreadExpertStreamer init — 急切分配：每個 slot posix_memalign(expertStride)
for _ in 0..<slotCount { posix_memalign(&raw, alignment, allocationSize) }
```

- slots 是全域快取但 **per-layer 分配**：30 層 × slotCount × 3.2MB（r4 expert）
- slots=96：96 × 30 × 3.2MB = **9.2GB** 急切常駐
- slots=112：112 × 30 × 3.2MB = **10.7GB** → +1.5GB
- 16GB Mac + r4 模型（12GB 權重）+ 10.7GB slots → 嚴重記憶體壓力，swap 已用 5.8GB
- 解碼 35% 變慢 = swap 抖動 + 頁面競爭，與 IO/planner 無關

### 為何舊災難（62% hit / 78GiB）沒重現

- 當時是早期版本：可能無 hot pool（`HOT_POOL_PROFILE_SIZE` 未固化）或 pool 邏輯不同
- 舊災難的「重讀」是記憶體壓力把頁快取擠出 → LRU 槽被逐出 → 每層重讀
- 現行 prod 已固定 pool64 + hit 97-98%，LRU 只兜底，所以 miss 不再爆
- **slots112 的唯一問題 = 急切分配的額外 1.5GB 常駐記憶體**

### 修復方向（未做，成本權衡）

1. **延遲分配 LRU slot**：pinned pool slots（64）急切，LRU slots（48）首次 miss 才 posix_memalign → slots112 的記憶體與 slots96 相同，slots 數的 LRU 容量優勢才真正可用
2. **或維持 slots=96 不變**：32 個 LRU 已覆蓋單 token（topK≤4）與 MTP verify（≤16）上限，slots112 的 48 LRU 沒有實際收益
3. **或 mmap 模式**：零急切常駐，但先前已定案 mmap 在單 token decode 上結構性輸給 pread（useResource 稅）

**定案**：維持 prod 默認 slots=96（本調查證明它是記憶體/容量平衡點）；slots112 的收益不存在，成本存在。


## 13.23 多 token 並行可行性設計：2-token self-speculation（2026-08-08）

**動機**：r2 停在 76% of ceiling（24.1/31.9 tok/s），miss-prefetch 已證偽（§13.16）。剩餘唯一架構級槓桿 = 把 MTP verify 的 batched forward 泛化到普通 decode。本節是可行性設計（非實作）。

### 1. verify batch 成本結構（實測，r2 + pool64，256 tok）

phase 分解（TURBO_FIELDFARE_MTP_DEBUG=1）：

| draft 數 | verify 行數 | draft 前向 | verify batch | rewind | cycle 數 | per-verify |
|---|---|---|---|---|---|---|
| 1 | 2 | 3.08s (12%) | 23.39s (88%) | 0.04s | 144 | **162ms** |
| 2 | 3 | 3.67s (16%) | 19.40s (84%) | 0.03s | 117 | **166ms** |
| 3 | 4 | 6.30s (17%) | 29.81s (82%) | 0.04s | 118 | 253ms* |

\*draft=3 那輪 decode 36.26s 撞負載尖峰，per-verify 253ms 高估；第一輪測量為 180ms。可靠結論：**verify(2行)≈162ms、verify(3行)≈166ms — 加一行只多 ~4-14ms**。

**成本結構**：verify 不是線性的。固定大頭（~160ms）= prefillChunked 的固定開銷（1+ 層管線、expert 並集、head/readback），每行增量極小（routed expert 並集共享後每行只多 ~4ms）。

**同窗對照（重負載窗）**：
- 單步 decode wall = 66ms/token（GPU busy 66ms/token，這窗負載重）
- verify(2行) = 162ms 產 1.77 token（77% 接受）→ 每 token 91ms
- verify(3行) = 166ms 產 2.19 token（70%）→ 每 token 76ms
- 對比單步：**MTP 每 token 成本是 base 的 1.5-2×**

### 2. 為什麼 batched verify 沒轉正：固定大頭不是計算，是 prefillChunked 錯配

verifyBatch 走 **prefillChunked（prefill 路徑）**，不是 decode 路徑。prefill kernel 為長序列設計，單 token decode 用它是錯配（§13.17 的教訓：full-attn tensor kernel 復用 prefill 是 decode 錯配）。~160ms 固定大頭裡：

- prefill 層管線（每層 sharedFFN + attn + MoE + head 一輪）≈ 30 層 × 每層 ~5ms = 150ms 量級
- 對照 decode 單步 GPU 只有 31.3ms/token（乾淨窗）— prefill 路徑每層成本是 decode 的 ~5 倍

**關鍵洞察**：verify 的固定大頭不是「batched forward 天生貴」，而是「用了錯的 kernel 路徑」。如果 verify 的每行能用 decode path 的成本（31.3ms），2 行 verify ≈ 40-45ms（共享 FFN/attn/KV，MoE 並集多一點），整個數學就反轉。

### 3. 接受率門檻模型

設單步成本 C（decode path），verify 成本 V(K) = F + K·m（固定大頭 F + 每行邊際 m），接受率 α，draft K 個。

**勝出條件**：V(K) 的 cycle 產出 = Σ_{i=1..K} α^i = α(1-α^K)/(1-α) tokens，必須 > K 個單步成本... 精確形式：

```
cycle 產出 E(K,α) = α + α² + ... + α^K
cycle 成本   V(K) = F + (K+1)·m   （驗證 1 行 current + K 行 draft）
單步替代成本 = E · C  （不投機時這些 token 花 E 個單步）

MTP 轉正 ⇔ V(K) < E(K,α) · C
```

代入實測（重負載窗 C=66ms、乾淨窗 C=41.5ms）：

| 場景 | F | m | V(2) | E(2,0.77) | E·C (乾淨) | 結論 |
|---|---|---|---|---|---|---|
| 現行 prefillChunked | ~160ms | ~5ms | 162ms | 1.77 | 73ms | **輸 2.2×** |
| decode-path verify 理想 | ~15ms | ~12ms | 40ms | 1.77 | 73ms | **贏 1.8×** |

**門檻**：α=0.77、K=1 時需要 V(2) < 1.77·C。乾淨窗 C=41.5ms → V(2) 必須 < 73ms。現行 162ms 差 2.2×。把 verify 移到 decode path（V(2) ≈ 40ms）即轉正。

**α 敏感性**（V=40ms 固定、乾淨窗 C=41.5ms）：α=0.5 時 E=0.5、E·C=20.8ms < 40ms 輸；α=0.77 時 E=1.77、E·C=73ms 贏 1.8×。**門檻 α ≈ 0.63**（40ms/41.5ms 的根）。r2/r3/r4 接受率 60-88%，code prompt 高、prose 低 — 所以 self-spec 需保留 adaptive gate。

### 4. 2-token self-speculation 具體方案

**與 MTP 的差異**：不依賴 MTP head 模型。draft 用主模型自己的 logits argmax（produce 後 `lastGreedyToken` 已存在，零額外成本）。

```
cycle = produce(t)                       // 1 次 decode forward，順手拿到 greedy argmax = d0
      + verifyBatch([t, d0])             // 1 次 2 行 batched verify（走 decode-path 多行）
      → d0 被接受：直接輸出 t, d0（2 token / cycle）
      → 拒絕：輸出 t，下個 cycle 正常
```

**關鍵工程點**：
1. **verify 走 decode path**（不是 prefillChunked）：需要 decode-path 的多行 forward（2 行共享 FFN/attn/KV，MoE 並集）。這是本設計的核心工作量 — prefillChunked 已有行並行結構，但 kernel 是 prefill 錯配；要麼改 prefill kernel 對 2 行優化，要麼在 decode path 加 2 行 batch。
2. **draft 品質**：self-spec 的 draft = 主模型 greedy argmax。文檔 §13.2 已記錄 MTP head 接受率 84%（code）；主模型自投機的接受率會略低（沒有專用 head 的未來預測），需實測 — 這是最大不確定性。
3. **KV rewind**：verifyBatch 已支援 rewindKV（MTP 循環在用），self-spec 可直接複用。

### 5. 預期收益（樂觀/悲觀）

| 場景 | 假設 | decode tok/s | Δ vs 24.1 |
|---|---|---|---|
| 悲觀 | α=0.55（主模型自投機低接受率）| ~21 | −13% |
| 中性 | α=0.70 + V(2)=40ms | ~30 | **+25%** |
| 樂觀 | α=0.80 + V(2)=35ms | ~33 | **+37%** |

中性場景 ~30 tok/s 需要：decode-path verify 成本壓到 <1.2× 單步、接受率 ≥0.7。兩者都有實測依據（prefill 錯配是已知可修、MTP head 接受率已到 84%）但**主模型自投機接受率是未知數** — 建議先做最小探針：produce 後取 lastGreedyToken 當 draft，跟蹤它的接受率（零 runtime 成本），再決定是否投入 decode-path verify 工程。

### 6. 結論

- batched verify 的 ~160ms 固定大頭 = prefillChunked kernel 錯配，**不是 batched forward 天生貴**
- 門檻模型：V(2) < 1.77·C 即轉正（乾淨窗 = V(2) < 73ms）
- 2-token self-speculation 中性預期 +25%（24.1 → ~30 tok/s）
- 最大風險：主模型自投機接受率未知；最小探針零成本可先測
- 不做：不改 MTP（MTP head 本身是額外成本，self-spec 免費 draft 才是正解）


## 13.24 新生產基準：r2/r3/r4 三種位寬重測（A/B/C 定案後，2026-08-08）

**動機**：§13.21 方案 A/B/C 全部定案（A 中性不上 prod、B 負收益、C 架構障礙），
確認生產設定為「SWA 25 層維持 partial+combine split path」。本節用當前生產默認
配置重測 r2/r3/r4 三種位寬，標記為 **A/B/C 定案後的當前生產狀態基準**。

### 當前生產默認配置（本基準的完整環境）

- `pool64-sync`（HOT_POOL_PROFILE_SIZE=64 + HOT_POOL_EXPERTS=64 + PRELOAD=sync）
- `slots=96`（§13.22 定案：記憶體/容量平衡點）
- `w8 split-workers`（EXPERT_READ_WORKERS=8 + prefill cap）
- `wake-poll 5000us`（§13.13 定案）
- `MTP off`（§13.14 定案：三種位寬全負收益；`MTP_MODEL=""` 默認）
- `ATTN_TENSOROPS=0`（§13.19 定案：flag 在此 SDK inert，誠實 off）
- `ATTN_CHUNKS / ATTN_SINGLE` 未設（§13.21 實驗開關默認 off）
- `trust-receipt`（跳過 SHA-256 全量重哈希，§13.4 省 3.9s TTFT）
- code prompt（/tmp/msg_code.json），256 tok，temperature 0

### 重測結果（256 tok，每種位寬 2 輪，兩輪幾乎重複）

| 位寬 | decode | **tok/s** | TTFT | attn GPU |
|---|---|---|---|---|
| **r2** | 9.66s | **26.5** | 3.22s | 3.11s |
| **r3** | 11.33s | **22.6** | 4.08s | 3.31s |
| **r4** | 13.94s | **18.4** | 4.85s | 3.85s |

128 tok 快測（2 輪一致）：r2 25.0 / r3 19.1 / r4 13.1 tok/s（256 tok 因 warm-up 攤平較高）。

### 對照歷史最佳

| 位寬 | 舊基準（§13.14/13.15） | **本次** | Δ |
|---|---|---|---|
| r2 | 24.1 | **26.5** | **+10%** |
| r3 | 20.7 | **22.6** | **+9%** |
| r4 | 19.7 | 18.4 | −7%（負載波動，非回歸） |

r2/r3 都打破先前紀錄；r4 略降屬當窗負載（同窗 MTP-off 對照 9.5-10.1 tok/s 顯示該窗負載偏重）。

### 關鍵發現：attn GPU 對位寬不敏感

- attention 權重是 int4 固定，不隨 routedExpert 位寬變 → r2 3.11s vs r4 3.85s 只差 0.74s
- r4 decode 慢的主因是 **MoE 權重 3.2MB/專家（r2 2.15MB）的 IO + GPU 計算**，不是 attention
- SWA phase-2 的空間對三種位寬一樣，已被 A/B/C 證明無法再榨（§13.21）

**一句話**：A/B/C 定案後的生產狀態基準 = r2 26.5 / r3 22.6 / r4 18.4 tok/s @256tok，
r2/r3 創新高；剩餘架構級空間見 §13.23 多 token 並行設計。


## 13.25 r2 剩餘空間拆解：26.5 tok/s vs ceiling 31.9（2026-08-08）

**動機**：SWA phase-2 已定案不可行（§13.21），重新聚焦 r2 剩餘 17% — 用完整
diagnostics（GPU_TIMING + timeline CSV）拆解 256-tok 的剩餘 idle，評估實質槓桿。

### 完整拆解（256 tok，r2，prod 默認；decode 11.41s / 22.4 tok/s，load ~2.9）

```
[gpu] attn=3.41s routedMoE=2.49s sharedFFN=1.18s phase1Hit=0.14s head=1.24s busy=8.47s ofWall=58%
[sync] cbs=24064 cb1Wait=8.36s routedWait=0.17s blocked=8.53s sched=10.72s overhead=1.30s(54us/cb)
[cb-latency] wait=8.36s gpu=3.41s wake=0.81s sched=4.23s
[cpu-chain] io=0.77s missIo=0.76s chainWall=1.14s cb1Wait=8.36s
[timeline] gpuBusy=8.54s gpuIdle=3.50s  gaps: <0.2ms 0.40s / 0.2-1ms 1.96s / 1-5ms 1.07s / >5ms 0.07s
```

### idle 精確歸因（timeline gap 分析，24064 spans）

| 來源 | idle | 佔比 | 中位 | 本質 |
|---|---|---|---|---|
| **routed-before（hit 層）** | **2.41s** | 69% | **0.18ms × 7046** | sharedFFN(0.16ms) 蓋不住 CPU 鏈(149μs)，差一點 |
| routed-before（miss 層）| 0.57s | 16% | 0.89ms × 604 | miss pread（每層 4.9× hit 成本）|
| phase1Hit-before | 0.25s | 7% | 0.42ms | miss 層的 phase1 暴露 |
| head-before | 0.20s | 6% | 0.39ms | 層尾 head 提交延遲 |
| cb1-before | 0.08s | 2% | 0.01ms | 可忽略 |

**routedWait=0.17s** — CPU 等 routed CB 完成幾乎無（pool 後）。**sched=10.72s 是跨
24064 CB 的 GPU 排隊總計**（含 cb1 依賴 + Metal 調度），非單一可縮點。

### 關鍵發現：hit 層 routed gap = 「差一點」結構，不是大洞

- sharedFFN GPU = 0.16ms/層（7650 層 × 0.16ms = 1.19s）
- CPU 鏈（readback+plan+fetch+commit）= chainWall 1.14s / 7650 層 = **149μs/層**
- 理想 pipeline：sharedFFN(160μs) 蓋住 CPU 鏈(149μs) → 每層零暴露
- 實測 hit 層 routed gap 中位 0.18ms ≈ 180μs — **剛好多 30μs/層 × 7650 = 0.23s 理論殘餘**
- 但實際 2.41s total >> 0.23s — 說明大部分 hit 層的 CPU 鏈**沒有**被 sharedFFN 蓋住
  （CPU 鏈在 cb1 wait 後才開始，GPU 早排的 sharedFFN 只是把 routed 提交延後）

**修正理解**：routed-before gap 的 2.41s 大頭不是「sharedFFN 蓋不住」，而是
**CPU 鏈必須等 cb1 的 router readback** — readback 在 cb1 完成後才可取，這是
串行鏈硬點。sharedFFN 早排只蓋住 fetch 的一部分（hit 時 fetch 是 inline 的
~100μs），readback+plan 的 ~50μs 仍在 cb1 wait 內。

### 槓桿評估（誠實結論）

| 槓桿 | 理論回收 | 可行性 | 定案 |
|---|---|---|---|
| miss pread（0.57s + missIo 0.76s）| ~1.3s | 需更大 pool（記憶體牆 §13.10）或預測（§13.16 證偽）| **無 cheap 槓桿** |
| hit 層 routed gap（2.41s）| 部分 | readback 依賴 cb1 完成 = **Metal 串行硬限制**；唯一出路是下一層 cb1 提前（B4 深度 pipeline）| **B4 已試**：多 CB 在飛需跨層依賴分析，風險高 |
| wake 殘餘（0.81s）| ~0.5s | wake-poll 已用（§13.13），殘餘是 polling 粒度 | **已榨乾** |
| attn（3.41s，最大 GPU 單項）| — | §13.17/13.21 已證 SWA split path 最優 | **定案** |

### 結論

- 剩餘 3.50s idle 的 85% 是 routed-before gap，其中 hit 層 2.41s 的根源是
  **readback 依賴 cb1 GPU 完成**（串行鏈硬點），miss 層 0.57s 是 pool 記憶體牆
- 現有機制（early-shared、hit-only-sync、phase1Hit、wake-poll）已把可縮的都縮了
- **r2 的 26.5 tok/s（乾淨窗）≈ 83% of ceiling 已是這台機器的結構性終點**，
  除非打破 readback→routed 的串行依賴（跨層 pipeline 或 §13.23 多 token 並行）
- §13.23 多 token 並行是唯一剩餘架構級槓桿：batched verify 天然跨 token 共享
  readback/plan/fetch，把 149μs/層的 CPU 鏈攤到多 token


## 13.26 r4 vs r2 IO 假設驗證：慢在「routed 權重讀取量」，不在 miss 數（2026-08-08）

**動機**：假設「r4 的 decode 慢是 MoE IO（3.2MB/專家 vs r2 2.15MB）而非 attention」。
驗證方法：同窗交錯對照 r4 vs r2 的 missIo/readWall/cache/GPU 結構。

### 硬數據（manifest 實測，非近似）

| | r2 | r4 | 比 |
|---|---|---|---|
| expertStride | **1,867,776 B (1.78MB)** | **3,358,720 B (3.20MB)** | **1.80×** |
| topKExperts | 8 | 8 | 相同 |
| **每 token routed 權重讀取** | **14.2MB** | **25.6MB** | **1.80×** |

### 同窗對照（交錯 A/B，r2 輪為有效窗：r2=19.42s vs r4=34.60s）

| 指標 | r2 | r4 | 比 | 解讀 |
|---|---|---|---|---|
| decode | 19.42s | 34.60s | **1.78×** | ≈ 權重比 1.80× ✓ |
| **miss** | 1192-1229 | **1142** | **~相同** | pool64 對兩者覆蓋相同 |
| bytes | 2.07-2.14GiB | 3.57GiB | **1.73×** | 符合權重比（miss 相同）|
| readWall | 1.38-1.73s | 2.55s | 1.6-1.85× | 隨 bytes 縮放，throughput 恆定 |
| throughput | 1.20-1.51 | 1.40 | 相同 | **IO 效率沒退化** |
| **routedMoE GPU** | 4.26-4.85s | **7.20s** | **1.5-1.7×** | 權重讀取量主導 |
| attn GPU | 5.46-6.01s | 7.67s | 1.3-1.4× | **比權重比小** |

### 結論：假設成立，但「IO」的定義要修正

1. **「r4 miss 多」是錯的** — miss 數完全相同（1142 vs 1192，都 98.2% hit）。
   pool64 對 r4/r2 覆蓋相同，**提高 pool size 對 hit 率零影響**（§13.22 記憶體牆更直接否決）。

2. **r4 的慢 = 每 token routed 權重讀取量大**（25.6MB vs 14.2MB = 1.8×），
   這個讀取在 **GPU 從 slot buffer 讀權重**（hit 98% 下是 GPU→記憶體頻寬，不是 SSD IO）。

3. **IO 效率沒退化** — throughput 恆定（1.2-1.5 GiB/s），readWall 隨 bytes 線性縮放。
   不存在「r4 的 read 策略要調整」的空間；read 效率兩者相同。

4. **attn 不是主因** — attn GPU 只差 1.3-1.4×（小於權重比），因為 attention 權重
   是 int4 固定，不隨 routedExpert 位寬變（§13.24 已記錄）。

### decode 慢 1.78× 的歸因

- decode 比（1.78×）≈ expert 權重比（1.80×）— **r4 的慢幾乎全部由 routed 權重
  讀取量解釋**。routedMoE GPU 1.5-1.7× + attn 1.3-1.4× 的混合。
- 每 token 25.6MB × 256 tok = 6.6GB 的 GPU 權重讀取，是 r4 的結構性成本。

### 策略定案（誠實）

| 候選 | 結論 |
|---|---|
| 提高 r4 pool size | **否決** — miss 已相同（1142），hit 98.2%，pool 再大零收益 + 記憶體牆 |
| 調整 read 策略 | **否決** — throughput 恆定，read 效率無退化可修 |
| 減每 token 讀取量 | **唯一真實槓桿** = 用 r3/r2（權重更小）或 §13.23 多 token 並行（batched verify 讓 8 個 expert 的權重讀取被多 token 攤薄）|
| attention 優化 | 方向錯誤 — attn 只差 1.3-1.4×，不是 r4 的主因 |

**一句話**：r4 的慢 = routed 權重讀取量 1.8×（GPU 頻寬），不是 miss 多、不是 read 策略、
不是 attention；pool size 與 read 策略都無調整空間，唯一出路是權重更小（r3/r2）或
多 token 攤薄（§13.23）。


## 13.27 r4 routed kernel 頻寬 vs 效率分析（2026-08-08）— 不是頻寬牆，是 x 重讀缺陷

**動機**：§13.26 指出 r4 的慢 = routed 權重讀取量（25.6MB/token/層），但
routedMoE GPU 1.5-1.7× vs 權重比 1.8× 暗示不是純頻寬線性。本節量化 GPU 頻寬
是否飽和、是頻寬牆還是 kernel 效率問題。

### 理論計算（負載無關）

| 量 | r4 | r2 |
|---|---|---|
| expertStride | 3.36MB | 1.87MB |
| 每 token 每層 routed 權重 | 25.6MB | 14.2MB |
| 每 token 30 層 routed 權重 | **768.8MB** | 427.5MB |
| 256 tok 總權重讀取 | **192.2GiB** | 106.9GiB |
| compute intensity | **3.5 FLOP/byte** | 6.4 FLOP/byte |
| roofline ridge (120GB/s vs 5TFLOPs) | 0.024 FLOP/byte | — |

compute intensity 3.5 遠高於 ridge 0.024 → 理論上 **compute-bound**。
但 total FLOPs 只有 0.7 TFLOP（256 tok）→ @8TFLOPs 只需 **0.09s**，實測 5.12s = **56× 餘裕**。

### 實測（timeline span 分析，256 tok）

```
routed spans: n=7650 total=5.12s mean=0.67ms med=0.49ms p90=1.10ms
cb1 spans:    n=7650 total=6.09s mean=0.80ms med=0.64ms
```

- 隱含頻寬（只看權重）= 26.9MB / 0.67ms = **40 GB/s = M4 理論(120)的 33%**
- 加上 x 重讀後 = 58.6MB / 0.67ms = **87 GB/s = 72%** ← 真相

### 根因：x 重讀是隱藏頻寬殺手

phase1 kernel（`moe_phase1_gate_up_act_u16load`）**完全沒有 threadgroup 共享記憶體**
（grep 確認 0 個 threadgroup 宣告）：

- 每個 (expert, row) workunit 從 DRAM 重讀整個 x（2816 halfs = 5.6KB）
- 每層 8 experts × 704 rows = **5632 個 workunit × 5.6KB = 30.2MB 的 x 重讀**
- 超過權重本身的 25.6MB！x 只有 5.6KB，完全該進 threadgroup smem（或至少 L2 穩定命中）
- 總 DRAM 流量/層 ≈ 25.6（權重）+ 30.2（x 重讀）= **55.9MB**，其中 **47% 是 x 重讀**

### 共享記憶體優化的理論收益

| | DRAM/層 | routed 時間/層 |
|---|---|---|
| 現狀（x 重讀）| 55.9MB | 0.67ms |
| **smem x（每 TG 讀一次）** | **29.4MB（−47%）** | **0.35ms（−48%）** |

- 256 tok × 30 層：routedMoE **5.12s → 2.68s（−2.44s）**
- decode 32.7s → ~30s（+9%，這窗含負載；乾淨窗 decode 14s → ~11.6s）

### 其他 kernel 缺陷（同源）

1. **phase2 down**：grid = D=2816 TG，每 TG 串行掃 8 experts × 704 acts = 5632 元素只算 1 個 d → 利用率極低。acts 也重讀。
2. **int4 dequant 純標量**：`& 0x0Fu`、`>> 4`、逐個 fma — 沒用 MPP tensor ops（§13.19 已證 SDK 無 MPP header），M4 SIMD 算力沒吃滿。
3. **gate+up 分開的 dot product**：同一份 x 讀兩次（gate 一輪、up 一輪）。

### 結論

- **不是頻寬牆**：隱含頻寬 87 GB/s（含 x 重讀）已達 M4 的 72%，但其中 47% 流量是 kernel 缺陷造成
- **不是計算牆**：計算只需 0.09s，實測 5.12s
- **是 kernel 效率問題**：x 重讀 + 低佔用 + 標量 dequant 三重疊加
- **最便宜的第一優化**：phase1 加 threadgroup x 共享（每 TG 一次載入 5.6KB，8 個 row 共用）→ routedMoE −48%
- **下一個**：phase2 的 acts 同理 + 提高每 TG 產出（多 d 共享專家權重讀取）

**注意**：本節數字在負載 ~3 的窗測得（decode 32.7s 含負載），但 routed span 的
0.67ms/層是 GPU span 時間，不受 CPU 負載污染 — 結構性結論可靠。共享記憶體優化
是 r4 上**唯一已知且未試**的 kernel 級槓桿，預期 r4 decode +8-15%。


### §13.27 實作結果：threadgroup x 共享記憶體優化（2026-08-08）

**改動**（`moe.metal`，6 個 phase-1 kernel + 3 個 inner dot 函數）：
- 每個 kernel 開頭宣告 `threadgroup half xSmem[kMoEPhase1MaxX]`（4096 halfs = 8KB）
- 新 helper `moe_phase1_load_x_smem`：256 threads 合作複製 x → smem + `threadgroup_barrier`
- inner dot 函數 x 參數改 `const threadgroup half*`；router gemv 保持 device（誤改已還原）
- code review 後收緊 bound 8192→4096 + `min(D, kMoEPhase1MaxX)` 防 OOB

**實測（r4，256 tok，GPU span 時間不受 CPU 負載污染）**：

| 指標 | 優化前（§13.27） | 優化後 | Δ |
|---|---|---|---|
| **routedMoE GPU** | 5.12s | **2.44-2.53s** | **−51%**（理論預測 −47%）|
| decode tok/s | 5-8（負載窗） | 21.8-24.2 | 視負載 |
| byte-identity | — | **3 次 md5 全同**（c7a76d5e…）| 零品質損失 |

**驗證**：
- 3 次運行 md5 完全一致（temp=0 同 prompt）— byte-identical ✓
- r2 也通過（routedMoE 2.19s、27.1 tok/s）
- code review 通過：barrier 位置安全（無早退 lane 卡死）、未初始化 smem 尾不被讀
  （inner 索引數學嚴格界於 D）、gemv 的 device x 正確保留

**關鍵教訓**：`.metal` 是 SwiftPM bundle resource，執行期 `makeLibrary(source:)`
從 `.build/.../*.bundle/Metal/MoE/moe.metal` 編譯 — **改 .metal 必須 `touch` +
`swift build` 讓 bundle 同步，否則測的是舊 kernel**（本輪踩坑：半成品 bundle
造成假編譯錯誤）。debug 流程：`grep xSmem .build/.../bundle/.../moe.metal` 確認同步。

**剩餘空間**（§13.27 的 phase2 同源問題未做）：phase2 down kernel 的 acts 重讀
（grid=D=2816 TG 串行掃 8×704 acts）+ int4 dequant 標量 unpack — 可視為後續
kernel 優化，但 phase1（最大單項）已 −51%。


---

### §13.28 phase-2 down kernel：chunked smem acts 實驗 + vector-width 分析（2026-08-08）

承接 §13.27 的「phase2 同源問題」。**實作 + 實測後定案：phase2 的 acts 重讀
不是 DRAM 問題（被 L2 吸收），chunked 重構是淨損失；64/128-bit 向量寬升級與
byte-identity 互斥**。以下是完整證據鏈。

#### 13.28.1 問題的實際結構（讀 kernel 修正後）

`moe_phase2_down_reduce_k8`：grid = DD = 2816 threadgroups，每 TG 8 個
simdgroup（sg_idx = expert），**只算 1 個 d 輸出**。每 TG 讀全部 8 experts 的
acts 行（8×F halfs = 11.3KB）→ 名義上 acts 被重讀 2816 次 = 31.7MB/層
（acts 本體只有 11.3KB）。down_W 每行只讀一次（7.9MB/層，無放大）。

但與 phase1 的 x 重讀（30.2MB，實測打到 DRAM）**結構不同**：

| | phase1 x 重讀 | phase2 acts 重讀 |
|---|---|---|
| 重讀量 | 30.2MB/層 | 31.7MB/層（名義）|
| 同層權重串流 | 25.6MB（gate+up，L2 擠爆）| 7.9MB（僅 down_W，< L2 16MB）|
| 實際 DRAM 命中 | **打到 DRAM**（L2 被權重 thresh）| **L2 吸收**（working set 僅 ~8MB/層）|

實證：chunked 版（smem acts，名義上 acts 流量降 31×）routedMoE **不降反升
+10%**（1.36s → 1.50s）——若 acts 重讀真的打到 DRAM，31× 流量下降必然反映
在 GPU 時間上。沒有 → 證明 acts 原本就是 L2 服務的。

#### 13.28.2 實作（保留為 env-gated A/B，默認 off）

新 kernel `moe_phase2_down_reduce_k8_chunked`：每 TG 把 8 experts 的 acts
載入 threadgroup smem（16KB bound `kMoEPhase2MaxActs`，cooperative 256-thread
load + barrier），然後 loop 算 `chunk` 個 d（`kMoEPhase2MaxChunk=64`，每
(d, expert) 用獨立 partial slot 免第二道 barrier）。接入
`encodeRoutedPersistentPhase2Reduce(chunk:)` + PSO。Swift 端
`TURBO_FIELDFARE_PHASE2_CHUNK`（>1 啟用，默認 1 = 舊 kernel）。

**Byte-identity ✓**：text-only md5 全同（chunk=1 vs 32，
`7885e2561bb62448f72c0fc4e6fc1e0e`；diff 僅計時行不同）。acts smem 複製是
精確的，per-d dot + reduction 順序不變。

**Code review 修復**（Swift 端）：kernel 內部 clamp chunk ≤ 64，Swift 端 grid
原用未 clamp 的原始 env 值 → env chunk>64 時 dispatch 太少 TG，y 後段沒寫
（靜默毀損）。已修：Swift 端 `min(chunk, maxPhase2Chunk=64)` 統一用於
grid 與 setBytes，並加 `f <= 1024` guard（避免 8*F > 16KB acts smem copy
的 OOB 讀取）。

**效能（r4 pool64 128-tok 3 輪交錯）**：

| chunk | decode tok/s（median）| routedMoE GPU |
|---|---|---|
| 1（舊 kernel）| **20.29** | 1.36s |
| 8 | 19.79 | — |
| 16 | 18.89 | — |
| 32 | 19.46 | **1.50s（+10%）** |
| 64 | 19.13 | — |

chunked 的串行 d-loop + 每迭代 barrier 把並行度從 2816 個獨立 TG 壓成
88 個有 barrier 依賴的胖 TG —— launch 稅的節省 < 序列化損失。結論：
**phase2 acts 不該用 smem chunked 處理**（與 phase1 的結論相反，因為
phase1 的 x 重讀是真 DRAM 流量，phase2 的 acts 是 L2 命中的）。

#### 13.28.3 vector-width（64/128-bit）分析：與 byte-identity 互斥

用戶建議把 32-bit load 升到 64/128-bit。**分析後定案：在現有 stripe lane
映射下做不到 byte-identity，不值得做**：

- 現行 phase1 gate_up inner：每 lane 每 blk 讀 4 bytes 權重（8 int4 codes）
  + 2×half4 x（8 halfs），`byte_base = blk*128 + lane*4` —— lane 的 stripe
  是 4 bytes，跨 blk stride 128。
- 升 uint4（16 bytes/lane）：要讀 32 codes 需 `byte_base = blk*512 + lane*16`，
  但 N=704 只有 11 groups（n_groups/16 = 0 全 tail），且 lane 的 fma 累加
  順序必變 → **輸出非 byte-identical**（fp32 加法非結合）。
- 現行 phase2 gemv 已用單次 `*(device const uint*)` 32-bit load（phase1
  gate_up 的 2×ushort+combine 是同一效果的慣用語，編譯器可合併）——32-bit
  已是該映射下的最大有效寬度。
- 真正 128-bit 需要「每 thread 持有連續 16-byte 子列」的平行化重寫，那是
  新 kernel（非 byte-identical 變體），與 session 的 byte-identity 標準衝突，
  且 phase2 本身只有 ~0.2-0.5s 佔比，ROI 低。

#### 13.28.4 當前 routed 分解（r4 pool64，128-tok，post-smem）

```
[gpu] attn=1.70s routedMoE=1.36s sharedFFN=0.62s phase1Hit=0.10s head=0.68s busy=4.46s ofWall=40%
[sync] cbs=11964 cb1Wait=4.18s sched=5.18s overhead=0.49s(41us/cb) ofWall=38%
[timeline] gpuIdle=1.88s gaps=11921 <0.2ms:10119x 0.2-1ms:1389x 1-5ms:388x >5ms:25x
```

routedMoE 5.12s → **1.36s（−73%）** 後，最大 GPU 單項是 **attn 1.70s**，
最大 wall 成本是 **CPU 側 sched 5.18s + cb1Wait 4.18s**（38% wall）——phase2
kernel 優化已不是高 ROI 方向。下一槓杆：attn kernel 或打破 cb1 串行鏈。


---

### §13.29 attn 1.79s 拆解 + split-path partial TG 佈局 ROI（2026-08-08）

承接 §13.28.4 的「最大 GPU 單項是 attn」。用 GPU_TIMING + timeline CSV +
既有診斷（skip-core / single-pass）做完整拆解與 ROI 定案。

#### 13.29.1 attn 的組成（r4 pool64，128-tok）

```
[gpu] attn=1.79s routedMoE=1.36s sharedFFN=0.62s phase1Hit=0.10s head=0.70s busy=4.68s
[cb-latency] wait=4.50s gpu=1.84s wake=0.45s sched=2.27s
```

timeline CSV 的 cb1 span 按層位置 bucketing（full = L{5,11,17,23,29}）：

| 層型 | 層數 | GPU 時間 | 佔比 | 每層 |
|---|---|---|---|---|
| SWA（256/16/8, window 1024）| 25 | 1.41s | **79%** | 0.444ms |
| full（512/16/2, GQA）| 5 | 0.38s | 21% | 0.594ms（僅 SWA 的 1.34×，非 8×）|

三向診斷分解（skip-core 只跳 partial+combine；single-pass 把 SWA 換成
單 kernel 免 combine）：

| 組態 | attn GPU | decode wall | cb1Wait | sched | gpuIdle |
|---|---|---|---|---|---|
| baseline | 1.79s | 6.60s | 4.50s | 2.27s | 1.78s |
| skip-core | 1.19s | 4.36s | 3.13s | 1.62s | 0.90s |
| **Δ（core 成本）** | **0.60s** | **2.24s** | −1.37s | −0.65s | −0.88s |
| single-pass SWA | 1.87s | 6.41s | — | — | — |

**拆解結論**：

1. **attn core（partial+combine）GPU 只有 0.60s（34%）**；QKV/RoPE/epilogue/
   OProj 非 core = **1.19s（66%）**。
2. **core 的 wall 成本 2.24s ≫ GPU 0.60s**：多出的 ~1.5s 是 CPU 側 sched
   （−0.65s）+ 下游 GPU idle（−0.88s）——60 個 kernel dispatch/token 在
   串行 cb1 鏈中的成本。**skip-core 讓 decode 19.8 → 29.5 tok/s（+49%）**。
3. **single-pass SWA（免 combine）無效**（19.98 vs 19.84）：並行度從 64 TG
   掉到 16 TG（每 Q head 一 TG），compute 損失吃掉 combine 節省 → **combine
   不是邊際成本，partial 的計算才是**。
4. **partial kernel 結構**：逐 KV position 迴圈 + 每 position 1-2 道
   threadgroup barrier（GQA-full 1024 threads 2 barriers/pos）。decode 時
   seqLen 只有 80-208 → SWA chunk ~26 positions（不是 window 1024），barrier
   成本 ~0.1s 量級，非主導。

#### 13.29.2 partial TG 佈局 ROI：**天花板 = 0.60s GPU，且結構改動難贏**

- TG 佈局已合理：SWA GQA-grouped 64 TG（8 KV × 8 chunks）、full 32 TG ×
  1024 threads（8 Q/KV，4 contiguous dims/thread）。single-pass 實驗已證
  偽「免 combine 就更快」——改佈局的預期收益 < 結構改動風險。
- **partial/combine 的 CPU 鏈成本（~1.5s wall）比 GPU 0.60s 大**——真正的
  槓杆是打破串行鏈（B4 pipeline 加深、多 CB 在飛），對 attn + routed +
  shared 全體有效，不是 attn kernel 本身。

#### 13.29.3 真正的下一步：QKV GEMV 的 x 重讀（同 phase1 修法，已驗證）

非 core 的 **1.19s（66%）才是更大的目標**。attention 權重是 int4
（manifest `quant.attention.weightBits=4`，group 64 affine），QKV+OProj
每層 ~18MB，1.19s → 隱含 ~58GB/s（M4 理論 120GB/s 的 ~50%）——與 routed
MoE 同源的權重讀取效率問題。

`dequant_int4_qkv_gemv_simd`（FusedQKVGEMV）的 x 讀取與 MoE phase1 **同構**：
grid = (q 4096 + k 2048 + v 2048)/8 = 1024 TG × 8 rows/TG，每 row workunit
從 device 重讀整份 x（2816 halfs = 5.6KB）→ **x 重讀 = 8192 × 5.6KB =
46MB/層，是權重（~12MB）的 4 倍**。

**ROI 評估**：phase1 的 smem x 修法（§13.27，routedMoE −73%）可直接套到
QKV GEMV（同一 inner `dequant_int4_gemv_simd_body` 模式）——8 rows/TG 的
x 進 threadgroup smem（8×5.6KB = 45KB，或按現有 4096 halfs cap 拆分）。
QKV GEMV 估佔 attn 1.19s 的 ~0.6-0.7s（q+k+v 的 GEMV 部分，OProj 另算），
x 重讀砍掉後預期 QKV GEMV 時間 −30-50% → **attn −0.2-0.35s GPU，decode
+3-5%**。這是比 partial TG 佈局更確定、更便宜的動作。

#### 13.29.4 QKV smem-x 實作結果：WASH（L2 閾值規律，2026-08-08）

實作（env-gated `TURBO_FIELDFARE_QKV_SMEM_X=1`，默認 off）：新 kernel
`dequant_int4_qkv_gemv_simd_smemx` + `dequant_int4_gemv_simd_body_smem`
（threadgroup x，cooperative 256-thread load + barrier，8 rows/TG 共用）。

**踩坑**：`*((const threadgroup half4*)(x + elem))` 讀出錯誤資料（misaligned
half4 在 threadgroup 陣列上 UB——陣列 base 非 16-byte 對齊時；MoE phase1 剛好
對齊所以沒踩到）。改純 scalar threadgroup 讀取後 **byte-identical ✓**（md5
`7885e256` 全同）。

**A/B（r4 pool64，128-tok，3 輪交錯）**：

| | decode（median）| attn GPU |
|---|---|---|
| off | 19.29 | 1.71-1.80s |
| on | 19.59 | 1.73-1.78s |

**WASH**——x 重讀（名義 46MB/層）已被 L2 吸收，smem 化零 DRAM 收益。

**L2 閾值規律**（三案例閉合）：

| 案例 | 每層權重串流 | working set | x/acts 是否真打 DRAM | smem 修法 |
|---|---|---|---|---|
| phase1 routed（§13.27）| 25.6MB | **57MB > L2 16MB** | **是**（L2 thrash）| **−51~73% ✓** |
| phase2 down（§13.28）| 7.9MB | 8MB < 16MB | 否（L2 服務）| wash ✗ |
| QKV GEMV（§13.29.4）| 12MB | 12MB < 16MB | 否（L2 服務）| wash ✗ |

**結論**：smem-x 修法只在「權重串流 + x 重讀 > L2 容量」時有效（phase1）。
QKV/phase2 的權重串流都 < 16MB，x 小且熱 → L2 吸收 → 零收益。§13.29.3 的
「−0.2~0.35s」預期被實測推翻。attn 1.79s 中真正可動的是 **非 core 的 1.19s
（QKV/OProj GEMV 的 int4 權重讀取效率 58GB/s ≈ 50% BW）**——但那是權重讀取
本身（與 routed MoE 同源），不是 x 重讀；要省只能降位寬（r3/r2）或攤薄
（多 token 並行），與 §13.26 結論一致。


### 13.30 QKV GEMV int4 權重讀取頻寬審計（2026-08-08）

**動機**：attn 非 core 的 1.19s 是最大 GPU 單項。檢查 dequant_int4_qkv_gemv_simd 是否
存在權重/scale 重讀或 pattern 次優，評估 double-row 共享 scale / batched prefetch。

**kernel 結構**（已讀原始碼確認）：
- grid = 1024 TGs（8192 rows / 8 rows_per_tg），每 TG 8 simdgroup，每 simdgroup 算 1 row
- 每 row 讀自己的 W_row（N/2 bytes）+ s_row/b_row（各 44 groups bf16）

**審計結果**：

| 項 | 量 | 判定 |
|---|---|---|
| 權重重讀 | 0 — 每 row 恰好讀一次（row-parallel grid）| 無放大 |
| scale/bias 重讀 | 1.38MB/層 = 權重 12.5%，group 內 8 lanes 共享同一 scale | L1 吸收，共享=0 收益 |
| x 重讀 | 44MB/層 = 4.0× 權重串流 | L2 吸收（smem-x A/B WASH 已證）|
| 強制 DRAM 串流 | 59.5GB / 128-tok（QKV 11.0 + OProj 3.78 MB/層 ×30×128）| 數學下限，不可減少 |
| 有效頻寬 | 59.5GB / 1.19s = **50GB/s = 42% M4 peak** | pattern 已最優（warp 連續 128B），非 pattern 問題 |

**結論：kernel 層級無可優化。** 權重讀取已是理論最優（row-parallel、每 row 一次、
warp 合併連續 128-byte block 串流）；double-row 共享 scale 的收益上限 = 12.5% 的
L1-cached 讀取 ≈ 0；batched prefetch 無物可預取（已是線性串流）。42% peak 是
int4 nibble 解碼的 flops/byte 比 + 每層 30 個串行 kernel drain/fill 的硬體特性。

**關鍵新發現：attention 權重在 r2/r3/r4 全部是 int4**（manifest 確認，只有
routedExpert 降到 2/3-bit）——59.5GB 強制成本在三個變體完全相同，r2 的快純粹來自
routed expert 位寬。這是一個**尚未被動過的槓桿**：

| 方案 | 強制串流 | 預估 attn 非 core | Δ vs int4 |
|---|---|---|---|
| attn@4bit（現狀）| 59.5GB | 1.19s | — |
| attn@3bit | 44.6GB | ~0.93s | −0.26s（decode +3~4%）|
| attn@2bit | 29.8GB | ~0.62s | −0.57s（decode +7~9%）|
| 多 token 並行（batch=4）| 攤薄 4× | ~0.35s | 架構級（QKV+OProj+FFN 全攤薄）|

**ROI 定案**：
1. QKV kernel 內部重構（double-row scale、prefetch、grid 重排）：**不做**（零收益，審計已證）
2. attention 降位寬：**下一步候選**（與 routed expert 同源工具已就緒，但 attn 比 FFN
   敏感，須先跑 perplexity 對照再決定 3-bit vs 2-bit）
3. 多 token 並行：最終架構級答案（§13.23 自投機門檻 0.63 接受率），把 QKV/OProj/
   全部 FFN 的強制讀取一起攤薄——這是 r4 突破 18 tok/s 的唯一路徑


### 13.31 B1 CB 融合審計：partial+combine+前後 GEMV 併入單一 CB？（2026-08-08）

**動機**：attn 拆解顯示 60 kernels/token（partial+combine × 30 層）的 CPU 串行鏈
（sched 4.06s + 下游 idle 0.88s）大於 GPU 0.66s。評估融合能否回收。

**現況（已讀 decode loop 原始碼確認）**：

```
每層 3 個 command buffer：
  cb1      (8 kernels)  inputNorm → QKV → QKVepilogue → partial → combine
                        → OProj → postAttnSetup → router(+lookahead next-router)
  sharedCB (2 kernels)  sharedFFN + sharedNorm   ← early-commit（與 cb1 並行）
  routedCB (3 kernels)  phase1(hit/subset) → phase2 → tail   ← 非同步
```

**三層結論：**

1. **CB 層級：attention 鏈本來就在單一 cb1**。user 提的「partial+combine 與前後
   GEMV 融合進單一 CB」= 現狀已達成（8 kernels / 1 CB / 層）。CB 數 11964/128tok
   ≈ 93/token（3 CB/層 × 30 + head），192μs/cb CPU 稅 → 2.30s overhead。

2. **B1 殘餘選項已窮盡且實測回歸**：
   - `TURBO_FIELDFARE_FUSE_SHARED`（shared 併入 cb1）：乾淨機 6 輪交錯
     **r3 13.0 vs 8.3、r4 15.9 vs 8.5（−57%~−88% 回歸）**。根因：early-split 把
     shared GPU 時間藏在 CPU router-readback 窗下，融合把它串回 critical wait。
   - routedCB 併入 cb1：**結構性不可能** — expert 選擇依賴 cb1 完成後的 CPU
     readback（router indices），GPU 無法自決。硬數據依賴。
   - kernel 層級（partial+combine → 單 kernel）：single-pass SWA 已測**中性**
     （19.98 vs 19.84）——60 dispatches 的 launch 稅可忽略。

3. **sched 4.06s 不是 dispatch 數問題**：`[cb-latency]` 定義 sched = commit→gpuStart，
   是 **per-CB 佇列延遲**，主成分是 cb1 對前一層 routed 的硬依賴（GPU 必須先完成
   routed 才能開始 cb1）。kernel 融合不改變 CB 數 → sched 不變。真正能砍 sched 的
   只剩「減少 CB commit 數」（B1 已證回歸）或「打破串行鏈」（多 token 並行）。

**skip-core 實驗的重新解讀**（19.8 → 29.7 tok/s，+52%）：
省下的 2.24s = partial kernel GPU 0.66s + **ripple 1.58s**（cb1 GPU 縮短 → CPU 更早
readback → 下游整鏈前移）。這不是 dispatch 稅——**是 partial kernel 的 GPU 時間
縮短後的連鎖效應**。single-pass 中性正是因為它沒縮短 GPU 工作（只省 dispatch）。

**ROI 定案**：
- **B1（CB 融合）：死路，已實證，不再做**。可回收 ≈ 0，且會回歸。
- 真正能回收 0.88s 下游 idle 的槓桿 = **縮短 partial kernel 的 GPU 時間**
  （逐 position barrier 迴圈是內部序列化，見 §13.29 拆解）→ ripple 效應自動回收。
- 唯一架構級槓桿仍是**多 token 並行**（打破 cb1→routed 的串行鏈，QKV/OProj/FFN
  強制讀取一起攤薄）。


### 13.32 零成本自投機接受率探針：重複式 self-spec 證偽（2026-08-08）

**實作**：`TURBO_FIELDFARE_SPEC_PROBE=1`（RawCompletion.swift，env-gated，禁用時
零熱路徑成本）。在 decode loop 每次 produce 前後比較 fused-head 的
`lastGreedyToken`——測量 P(greedy[N+1] == token[N])，即「draft = 重複前一個
greedy token」方案的接受率（§13.23 設計，探針為純診斷，不改前向路徑）。

**結果**（prod 配置、code prompt、256 tok、254 個有效步驟）：

| 變體 | 接受 | 接受率 | 2-draft 預期 token/verify |
|---|---|---|---|
| r4 | 1/254 | **0.4%** | 1.004 |
| r2 | 3/254 | **1.2%** | 1.012 |
| r3 | 0/254 | **0.0%** | 1.000 |

**門檻 0.63，實測 0.4%——差 157 倍。重複式 self-spec 徹底證偽**：code prompt
上模型每個 token 都在產生新內容，greedy 自重複率 ≈ 0。任何「draft = 主模型自己
的輸出重複」的免訓練投機方案在此 prompt 上零機會。

**決策樹落地**：探針 < 0.63 → 按預定規則轉做 **attention 降位寬**（+3~4% 保底）。
多 token 並行若要重啟，只能走「有訓練的 draft 模型」（MTP 已測 72-84% 接受率但
loop 稅使其 net-negative）或不同 prompt 域（prose 的 self-repeat 較高，但生產
benchmark 是 code）。

**附帶收穫**：同窗測到 prod 配置乾淨數字 r2 24.95 / r3 21.72 / r4 21.53 tok/s
（load 低窗），與文檔基準 r2 23.5 一致。


### 13.33 全模型 garbage 回歸根因：moe_phase1 xSmem 載入迴圈只用 simdgroup-local lane（2026-08-08）

**現象**：09:33 起 r4/r2/r3 全變 garbage（「The capital of France is **?**?**...」），
09:36 前同 prompt 輸出連貫（out_a/out_off.txt 為證）。新舊 binary（含 8/7 bin/ 快照）全壞。
模型檔 sha 驗證無損、prompt 未變。

**關鍵證據鏈**：
1. r4 perplexity（純 forward math，無 template/取樣/解碼干擾）= **43,581**（argmaxAcc 7.1%）→ forward 數學壞掉
2. `strings bin/TurboFieldfareCLI-adaptive` 顯示 SwiftPM resource accessor 內嵌**絕對路徑**
   `.build/arm64-apple-macosx/release/TurboFieldfare_TurboFieldfare.bundle` →
   **新舊 binary 共用同一個 bundle**（runtime 從 bundle 讀 .metal 源碼再即時編譯）
3. 09:33 bundle = 8/7 kernel 源碼；現在 bundle = 今日源碼 → 壞在 bundle kernel 變更
4. 今日 kernel 變更逐一排除：attention.metal 純追加（238 行無刪除）、dequant_int4.metal 純追加
   （b3 kernels）、**moe.metal 有 8 行刪除 = 6 個 phase-1 wrapper + u16load body/inner 被改成 xSmem**

**根因（一行）**：
```metal
// moe_phase1_load_x_smem（13:35 新增）
for (uint i = lane; i < DD; i += 256u) { xSmem[i] = x[i]; }
```
`lane` = `thread_index_in_simdgroup`（**0..31**）。8 simdgroups × 32 lanes 全部用同一組
lane 值 → 只有元素 {lane + 256k | lane<32} 被寫入（xSmem 的 ~12.5%），其餘 87.5% 是
**未初始化 threadgroup 記憶體** → 確定性 garbage。u16load（r4，今日引入）與 bits
（r2/r3 int2/int3，8/7 引入、**從未做過輸出品質驗證**——只量過 tok/s）兩條路徑同病。

**修復**（`moe.metal`，一行語義）：
- helper 改收**全域 thread index** `tid = sg_idx * 32u + lane`（TG = 256 threads，
  tid ∈ [0,256) 恰好每個 x 元素寫一次）；6 個 wrapper 的呼叫改傳 `sg_idx * 32u + lane`

**驗證**（全部通過）：
- r4 ppl：43,581 → **1,503**（argmaxAcc 7.1% → 42.9%）；修復後 smem 版 1,511 ≈ device 版
  1,503 → **修復的 smem 與 device 讀取 byte-identical**（smem 優化 −51% MoE 保留）
- r4 msg_code：復現 09:33 連貫輸出（"The code you provided is a correct and efficient
  implementation of the Fibonacci sequence…"）
- r2/r3 cap：garbage → "The capital of France is **Paris**." ✓
- r2 128-tok decode：14.0 / 16.3 tok/s（速度未退化）

**衍生發現**：
1. **8/7 起的 r2/r3 tok/s benchmark（23.5 tok/s 等）可能是在 garbage 輸出上量的**——bits smem
   路徑引入後從未驗證品質。修復後需重測 r2/r3 品質與 tok/s 基線。
2. **規則**：任何 kernel 共享記憶體/緩存優化必須先做 byte-identity 或 ppl/text 驗證再定案，
   不能只看 tok/s（此 bug 完美通過所有速度 A/B）。


### 13.34 修復後重新驗證：r2/r3/r4 品質與速度基線（2026-08-08）

xSmem tid 修復（§13.33）後，輸出已恢復正確。本節是修復後的正式基線重測
（prod 配置：pool64-sync + slots96 + READ_WORKERS=8 + WAKE_POLL=5000 + MTP off）。

**品質（perplexity，同 4KB corpus、同一 harness、無 softcap——變體間公平）**：

| 變體 | ppl | medPPL | argmaxAcc |
|---|---|---|---|
| r4 | 1,315 | 42.5 | 43.7% |
| r3 | 1,389 | 46.2 | 42.9% |
| r2 | 4,877 | **811** | 32.6% |

- r4 ≈ r3（medPPL 42-46），**r2 顯著退化**（medPPL 811、argmaxAcc −25%）——2-bit
  routed 的品質代價比預期大，與 8/7 MTP 報告的「r2 品質退化」一致且幅度更大
- 固定 prompt（def fibonacci(n): 續寫 96 tok）：**三變體全部連貫正確**——r4/r3 給
  完整解說+迭代/遞迴 code，r2 也給正確的迭代法解說（r2 在對話型 prompt 上沒 ppl
  顯示的那麼糟，ppl 高點集中在 corpus 的 code/文檔 token 上）

**速度（256-tok、3 輪交錯、中位）**：

| 變體 | 中位 tok/s | 最佳輪 | TTFT（含 pool64 預載） |
|---|---|---|---|
| r2 | 22.17 | 25.41 | 2.9s |
| r3 | 15.65 | 21.01 | 3.7s |
| r4 | 15.28 | 16.31 | 4.3s |

**結論**：
1. **歷史速度基準仍然成立**（r2 23.5 / r3 22.6 / r4 15.9）：garbage 不影響 kernel
   速度，所以過去的 tok/s A/B 數字是有效的——但**那期間的「輸出品質」是壞的**。
2. 歷史品質基準（若有）**無效**：r4 自 2026-08-08 13:35、r2/r3 自 bits smem 引入
   （8/7）起輸出即為 garbage，任何基於輸出的品質結論（MTP 接受率、degradation
   觀察）在這些窗口內都需重新評估。
3. r2 速度最快但品質退化最重；r3/r4 品質接近。生產預設若重品質應選 r3（速度 r3≈r4
   但位寬低、模型檔小）；若重速度且接受 2-bit 品質，r2 仍是最快。


### 13.35 修復後 MTP-adaptive 重測：接受率未受 garbage 污染、net 仍負（2026-08-08）

xSmem 修復（§13.33）後模型輸出正確，重跑 r4/r3 × MTP off/adaptive 的 256-tok 交錯 A/B
（prod 配置 + draft=gemma-4-mtp-head，TURBO_FIELDFARE_MTP_ADAPTIVE 預設 ON）。
load 4.8 高窗，r1 r4-off 與 r3 r4-mtp 兩筆為 load 污染離群值（標 *），不納入結論。

**結果（tok/s，[stop=] 行）**：

| 輪 | r4 off | r4 mtp | r3 off | r3 mtp |
|---|---|---|---|---|
| 1 | 2.02* | 4.35（acc 65%）| 12.80 | 11.74（acc 82%）|
| 2 | 13.08 | 8.41（acc 80%）| 15.69 | 14.16（acc 82%）|
| 3 | 13.12 | 2.86*（acc 86%）| — | — |

**接受率**：修復後實測 **65-88%（多數 80-86%）**，與歷史 72-84% 一致 →
**歷史接受率數據有效，未被 garbage 污染**。原因合理：接受率 = draft vs target 逐位元匹配；
修復前兩個模型「壞在同一處」，匹配率不受影響。

**net 收益**：仍是負的——r4 −30%+（13.1 vs 8.4）、r3 −8~10%（15.7 vs 14.2）。
r4 上 adaptive gate 全程未關（off=0，接受率 80%+ 讓 gate 保持 draft），但 MTP loop 的
結構稅（d=0 步 2.5-3× 慢 + batched verify 的 expert 並集）吃光收益；r3 上 gate 關了 9 次
（off=9）仍輸 10%。

**結論**：
1. **MTP 預設關閉的定案在修復後仍然正確**（run_prod.sh 保持 MTP_MODEL 空）。
2. 若未來要讓 MTP 轉正，仍只剩砍計算路徑（§13.33 前的結論不變）：修 d=0 步的
   序列化稅（路由回 decode path）、verify 稀疏化、或減 draft 數。
3. 接受率健康（80%+）說明 draft model 品質沒問題——瓶頸純在 loop 結構成本。


### 13.36 MTP 每步成本拆解與 r4 理論上限（2026-08-08）

用 r4 MTP-on（128 tok、prod 配置、MTP_DEBUG + MTP_ADAPTIVE_DEBUG）實測拆解。

**每步結構（MTPCompletion.swift 逐行確認）**：
- draftBudget > 0：`makeBridgeSnapshot` → `drafter.draftTokens`（**順序迴圈**，每 draft 一個
  獨立 draft-model 前向）→ `verifyBatch([current+drafts])`（**一次 batched prefill 前向**）
  → `rewindKV` → `publishHiddenRow`。GPU sync 點 = 2/步（draft CB + verify wait）；base decode = 1。
- draftBudget = 0：**已路由回 decode path**（`producer.produce(...)`，註解「a disabled gate
  really returns to base speed」）——d=0 路由修復**已在程式碼中**，不是缺失項。

**實測成本（38 個 draft 步，平均 d=2.63、acc 81% → 每步產 3.13 tokens）**：

| 項目 | ms/步 | 佔比 | 說明 |
|---|---|---|---|
| draft 前向 | 97 | 14% | 2.63 個**順序** draft 前向（~37ms/個，不可重疊）|
| verify batch | 617 | 86% | 3.15-token batched prefill 前向 |
| rewind | 1 | 0% | KV cursor |

**verify 是唯一有意義的稅**：617ms / 3.15 tok = **196ms/token 等效 vs base 76.5ms = 2.56×**。
根因（§13.35 前置發現）：(a) batched verify 走 prefill 路徑、expert **並集 IO 隨 token 數線性
增長**（3 token 路由到不同 expert → 讀取 ~2.6× 單 token）；(b) 3-token 的微小 batch 在
prefill kernel 上有每 position 固定開銷；(c) prefill 的 expert cache 命中僅 ~78%，miss 讀磁碟。

**gate 為什麼不關（off=0）**：47 步 = warmup 3 + calibrating 5（d=4）+ warmBaseline 6
（d=0）+ running 33。**baseline 在 prefill 後的前 6 步（冷 expert cache 期）量測**——
冷期 base decode 遠慢於熱期，baseline 被低估 → MTP rate 看似划算（row=0.48 卻不觸發
「兩次連續輸才關」的門檻）。256-tok 短窗也讓 gate 來不及收斂。

**r4 理論上限模型（base = 76.5ms/token = 13.1 tok/s，本窗）**：

| 情境 | 每步成本 | 產 token | tok/s | vs base |
|---|---|---|---|---|
| 現狀（d=2.6、verify 並集線性）| 715ms | 3.13 | 4.4 | −66% |
| A. d=0 路由生效 + gate 收斂關閉（已實作）| 76.5ms | 1 | **13.1** | =（+36% vs 現狀）|
| B. verify IO 完美共享（compute 2× 線性）| ~251ms | 3.13 | ~12.5 | ≈（−5%）|
| C. d=1、p1=90%、verify(2) 1.5× | ~153ms | 1.90 | ~12.4 | ≈ |
| D. verify 零成本（不可能，target logits 必須算）| 98ms | 3.13 | 32 | 上限 |

**break-even 條件**：d=1 需要 verify(2) < **107ms**（現估 ~390ms，差 3.6×）；d=2 需要
verify(3) < **131ms**（現估 ~500ms+，差 4×）。

**結論**：
1. **r4 的 MTP 天花板 ≈ base（parity），不是 >base**——batched verify 只攤薄 IO，compute
   （attn/sharedFFN/head/路由計算）仍隨 batch 線性增長，而 r4 decode 已近 GPU 效率。
2. **d=0 路由的價值 = 防止虧損**（−36% → 0%），不是創造收益。修復已在，真正沒接上的是
   **gate 收斂**：冷快取期 baseline 校準 + 短窗不及收斂 → gate 全程保持 draft。
3. 要讓 MTP 在 r4 轉正，verify 需降 4×（並集 IO 共享 + prefill 用 decode 側 cache）——即使
   做到也只有 parity。**r4 上 MTP 維持預設關閉是正確定案。**
4. r3 損耗僅 −10%（15.7 vs 14.2）且 gate 關了 9 次：expert 更小（2.7MB）→ verify 更便宜 +
   gate 較快收斂 → 離 break-even 更近，但仍是負。


### 13.37 Adaptive MTP gate fix (2026-08-08): r4 gate now correctly disables MTP

**Problem.** On r4 the adaptive gate never disabled MTP (`off=0-1` in a 128-tok
run) even though MTP was a 3.4x loser (4.4 vs 14.9 tok/s same window). Two
independent measurement bugs kept drafts on:

1. **Baseline used the MEAN of the warmBaseline d=0 samples.** In the MTP loop
   the d=0 decode steps degrade after a cold verify batch (72ms -> 187ms over
   6 steps; loop-context noise, not load). The mean (~130ms) under-estimates
   the true RawCompletion decode rate (67ms) by ~2x. A too-slow baseline is
   exactly what makes MTP look competitive.
2. **Cold-start calibration outlier.** The first calibratingMTP step (first
   5-token verify) costs 5-7s (cold expert-union IO) vs ~100ms steady. It was
   included in `mtpTotal`, poisoning the seed rate / rowShare.
3. **Disable latency.** Disabling required TWO consecutive losing decisions at
   16-step windows (~13s of generation burned at MTP speed once the verdict
   was already obvious).

**Fix** (`MTPAdaptive.swift`):
- Baseline = **min** of the warmBaseline samples (the min matches the honest
  RawCompletion decode rate; degradation is loop-context noise).
- **Skip the first calibratingMTP sample** from mtpTotal.
- **Immediate disable** when rate < baseline x 0.75 (hardDisableRatio) on the
  FIRST losing decision instead of waiting for two windows.
- Calibration shortened: calibMTPSteps 4->2, warmBaselineSteps 6->4.

**Verification** (r4, 128 tok, fibonacci prompt, prod config, load 3.3-4.7):

| metric | before | after |
|---|---|---|
| gate disables per run | 0-1 | 5 |
| running steps at d=0 | ~0% | 82% |
| draft steps per run | 40+ | 16 |
| in-loop d=0 step min | - | 54ms (= RawCompletion 67ms under same load) |

**Honest timing note.** The MTP step time (`phaseEnd - stepStart`) was always
honest in the gate's report(); an early debug print that measured only the
draftTokens phase caused a false "35ms/step" scare. With the print moved after
the full step, d=2 verify steps measure 420-520ms (5-token batch), consistent
with the wall clock; `gap=` between steps is ~0ms (no hidden time).

**Remaining gap.** MTP remains NET NEGATIVE on r2/r3/r4 (acceptance 78-88% but
batched-verify expert-union IO is superlinear, ~2.6x per-token cost), so the
run_prod.sh default stays OFF. The gate is now correct if you opt in: it
disables MTP quickly and the in-loop d=0 path is at parity with RawCompletion
decode under the same load.


### 13.38 r4 verify-batch expert-union pool coverage (2026-08-08)

**Question.** Single-token decode has ~93-97% hot-pool coverage. Does the MTP
verify batch's expert *union* (across all batch tokens) hold up, or does it
leak to streaming? This decides whether edge MTP can ever be positive.

**Measurement** (TURBO_FIELDFARE_UNION_STATS=1 + UNION_DUMP, r4, 128 tok,
pool64, MTP-adaptive): for every routed batch (t = batch size), per layer the
union of routed experts vs the hot-pool set. topK=8 experts/token.

| batch t | union size | union coverage (pool64) | per-token coverage |
|---|---|---|---|
| 1 (decode) | 8 | **93.5%** | 93.5% |
| 3 (d=2 verify) | ~18 | **79.1%** (med 81.5%) | 81.4% |
| 5 (d=4 verify) | ~27 | **73.7%** (med 75.0%) | 80.8% |

**Pool-size sweep** (top32/48/64/80 real profiles; 96/128 extrapolated from the
same prompt's routing — optimistic):

| pool | t=1 | t=3 union | t=5 union |
|---|---|---|---|
| 32 | 87.5% | 55.9% | 47.9% |
| 48 | 87.6% (mean) | 70.0% | 63.2% |
| 64 (prod) | 93.5% | **81.5%** | **75.0%** |
| 80 | 96.2% | **88.6%** | **85.4%** |
| 96 | ~99.8% | ~99.9% | ~99.9% |
| 128 (full) | 100% | 100% | 100% |

**Streaming leak (the superlinear tax, quantified):** at pool64 each t=3 verify
step streams ~300MB of non-pool experts (t=5: ~560MB) vs ~50MB for a decode
step — 6-11x the per-step streaming of decode. That is the 2.6x per-token
verify cost. Pool80 cuts the leak 38% (t=3: 299->185MB/step; 2.96GB/run vs
4.78GB/run). Per-token coverage in verify batches (81%) is also lower than
decode (93.5%): rejected drafts route to off-distribution experts whose
fetches buy nothing — acceptance quality directly reduces wasted IO.

**Verdict for edge MTP:** the barrier is NOT "MTP on edge devices" — it is
"verify-union pool coverage". On pool64 the union leaks 19-26% to streaming,
which keeps MTP negative. Conditions for edge MTP to work:
- pool ~80: union coverage 85-89%, leak -38% -> MTP approaches breakeven
  (+1.6GB RAM over pool64; feasible on 16GB, tight).
- pool ~96 or full residency: coverage ~90-100% -> MTP positive (+3.2GB over
  pool64; needs 24-32GB, or a smaller model). Server (72GB VRAM) full
  residency -> MTP should win (matches the "resident => MTP useful" thesis).
- Rejected drafts waste union IO: fewer drafts / higher acceptance directly
  shrink the leak.

### 13.39 Adaptive pool (pin-as-you-go) 离线路演：静态 profile 是 verify union 覆盖率的瓶颈（2026-08-08）

**动机**：pool80/mmap-pool 都是「加大常驻预算」的路。用 r4 union dump（同一 code prompt、96 步、
t=1/t=3/t=5 混合）离线模拟了**自适应池**——池随生成进程动态钉当前实际路由到的专家
（pin-as-you-go，LRU 淘汰）——结果发现：**同样的 64 槽预算，自适应池的 verify union 覆盖率
比静态 top-64 code profile 高 +11.5pp，不需要 pool80 也不需要 mmap。**

**数据**（median step 覆盖率；t=3=16 步、t=5=2 步小样本）：

| 池策略 | t=1 decode | t=3 verify union | t=5 verify union |
|---|---|---|---|
| 静态 top-64 code profile（现生产） | 96.5% | **78.5%** | 74.9% |
| 静态 top-80 code profile | 98.1% | **85.9%** | 83.6% |
| pin-as-you-go cap-64 | 92.9% | **90.0%** | 60.8%* |
| pin-as-you-go cap-80 | 95.8% | **90.4%** | 60.8%* |
| pin-as-you-go cap-96 | 97.1% | **90.4%** | 60.8%* |
| oracle top-64（整轮频率，上界） | 96.2% | 96.1% | 94.6% |
| oracle top-80（整轮频率，上界） | 99.2% | 99.3% | 99.0% |

（*t=5 只有 2 步样本，含冷启动步，中位数失真；oracle 证明 64 槽就装得下 t=5 union，
是暖池后可达的目标。）

**根因：被拒 draft 路由到偏分布专家**——verify 的 t=3 union 中 **35.1%** 的专家从未在
先前 decode 步出现，t=5 达 **54.4%**。静态 profile 无论多好都覆盖不了这些（它们本质上是
off-distribution），只有「观测到才钉」的自适应池能学到。

**关键结论**：
1. **cap-64 自适应 = 90.0% union 覆盖，cap-80 只到 90.4%（+0.4pp）** —— 64 之后
   cap 的边际收益趋零（池已收敛到本 prompt 分布）。**pool80/mmap-pool 是错的方向**；
   真正的缺口是「静态 vs 自适应」，不是「64 vs 80」。
2. 生产系统其实已有自适应件：**slot LRU 缓存**（96 槽）就是 pin-as-you-go 的现成实现。
   但静态 pool 的 64 槽永不淘汰——如果本轮热集偏离 profile，最多 ~20% 槽位浪费，
   LRU 又只有 ~3.2 槽/层（96/30）。**候选修法：把静态 pool 槽位改为可淘汰
   （LRU 换代），或按本轮实际路由把 pool profile 动态替换。**
3. 90% union 覆盖（现 78.5%）把每步 streaming leak 从 ~21% 砍到 ~10%——streaming 税
   减半。按 §13.36 的模型 verify 617ms → ~430ms/步，**仍高于 240ms 打平门槛**，
   单独做不足以让 MTP 转正，但它是任何 verify 优化（稀疏化）的免费前提。

**待办**：实现「可淘汰 pool 槽」或用 slots LRU 实测 verify union 命中率（当前
EXPERT_STATS 只报 per-token 命中），确认自适应后 t=3 union 命中是否真到 ~90%。


### 13.40 自适应池（promote-on-miss）实作 + r3 MTP 重算（2026-08-08）

**实作**（env `TURBO_FIELDFARE_ADAPTIVE_POOL=1`，默认 off，§13.39 模拟的落点）：
- `PreadExpertStreamer` 自适应模式下 pool 槽**不再 pinned**——profile 只作暖启动
  填充（preload 到全 slot 预算），LFU/LRU 淘汰自动实现 promote-on-miss（hot
  非池专家累计 use-count 存活、cold pool 成员被驱逐）。改动很小：init 分支 +
  全 budget poolSize + plan-residency 遥测（`[plan-residency]` 按 batch 分桶）。

**验证结果：r4 MTP-on 交错 A/B（3 轮）——无结构性差异**

| 轮 | STATIC（pinned pool）| ADAPTIVE |
|---|---|---|
| r1 | 7.86 tok/s | 7.84 |
| r2 | 8.10 | 12.14（load 噪声）|
| r3 | 6.69（load）| 5.77（load）|
| 中位 | **7.86** | **7.84** |
| decode hit | 98.2% | 98.2% |
| evictions | 193 | 210 |

**为什么模拟（+11.5pp）没有兑现**：§13.39 的离线路演量的是 **pool-ONLY**
union 覆盖率（78.5% vs 90%）——它**忽略了 32 个 LRU 槽**。真实系统是
pool64 + LRU32 的**合并驻留**，实测合并命中率 97.0%，LRU 槽本身就已经在做
pin-as-you-go。让 pool 槽也可淘汰没有新增任何驻留能力（LFU 本来就会保住
hot 专家）。**结论：pool 槽位的静态 vs 自适应不是瓶颈——96 槽预算下合并驻留
已经 97%，自适应池是已解决问题的空想方案，保留 env 开关但生产默认 off。**

**r3 MTP on/off 重算（prod 配置，128 tok，3 轮交错）——MTP 仍净负**

| r3 | MTP-OFF | MTP-ON | delta |
|---|---|---|---|
| r1 | 16.50 tok/s | 11.81 | −28% |
| r2 | 9.70（load）| 9.91 | ±0 |
| r3 | 16.98 | 11.91 | −30% |
| readWall | 1.39-1.48s | 1.42-1.45s | **相同** |
| read bytes | 2.37GiB | 2.37GiB | **相同** |

**关键发现：r3 上 MTP 已不是 IO-bound**。MTP-on 与 off 的 readWall/bytes
**完全相同**（1.4s / 2.37GiB，命中 97.0%）——verify 的 expert 并集已被
pool+LRU 合并驻留覆盖，streaming 税≈0。MTP 的 −28~30% 损失**全部来自 GPU
compute（batched verify 的 kernel 计算 + MTP loop 结构开销）**，与记忆体、
池大小、自适应与否无关。

**这回答了"r3 在 16GB Mac 上 MTP 能否转正"**：
- 记忆体：r3 每专家 2.48MB，pool64+96 槽 ≈ 7.1GB（与 r4 同结构），16GB 账
  本就紧（桌面 3.5 + runtime 2.5 + dense 1.5 + pool + streaming ≈ 16-17GB）。
- 但**加大 pool / 自适应池 / mmap 都救不了 MTP**：IO 已经不是约束（readWall
  不变），瓶颈是 verify 的 GPU 计算。唯一能让 r3 MTP 转正的路是**砍 verify
  计算**（draft 复用 shared FFN、verify 稀疏化、减 draft 数）——§13.36 早已
  算出 verify 必须 ≤240ms/步（现况 ~430-617ms）才打平，而现在的差距已从
  "IO 超线性税" 变为 "纯 GPU 计算税"。
- **r3 MTP 在 16GB 上的结论：负收益，与记忆体无关。** 保持 `MTP_MODEL=""`
  默认 off 不变。要转正只能砍计算，无法靠池。


### 13.41 「原生 MTP」提案评估：当前实作已是原生共享激活架构，提案前提不成立（2026-08-08）

外部提案声称：当前用「4 层完整独立 Transformer Draft 小模型」= 两套完整模型
各自前向，应该改造成「纯顶层投影头 + 复用主干激活/KV」的原生 MTP，即可从
-30% 转正 +35~42%。**用代码 + 权重实况逐条核验后：前提不成立。**

**事实 1：当前 draft 模型就是官方 MTP assistant 本体。**
- `models/gemma-4-mtp-head/` 的 README 明确写着"This model card is for the
  Multi-Token Prediction (MTP) drafters for the Gemma 4 models"——它就是
  `google/gemma-4-26B-A4B-it-assistant` 官方权重（0.78GiB bf16）。
- config: `num_hidden_layers=4, hidden_size=1024, backbone_hidden_size=2816`。

**事实 2：draft 不是「独立前向」——它已经复用主模型激活 + KV。**
- `LocalMTPAssistant.draftTokens(hiddenState:...)` 的入参就是主模型的
  `bridgeSnapshot.lastHiddenState`（最后一层激活），不是自己重跑主干。
- 代码注释原话："The assistant owns no k_proj/v_proj. Sharing is by layer
  type: every sliding layer reads the target's last sliding-layer KV, and the
  full layer reads the target's last full-attention KV — exactly the two
  entries the reference's `shared_kv_states` dict carries."
- 也就是提案描述为「原生 MTP 该做」的事（读主干最后激活 + 共享 KV），
  **当前代码已经逐字在做**。draft 的 4 层只跑 1024 维小 transformer 做
  逐 token 链式预测（97ms/步 = 32ms/层步），不碰主干 30 层 MoE。

**事实 3：verify 已经是「单次主干前向批量校验」——提案的 verify 描述
（"复用主干单次前向一次性校验全部草稿 Token"）与现状完全一致。**
- `MTPCompletion.verifyBatch` → `prefillChunked`（batched forward，draft span
  一次跑完），没有为每个 draft 重复加载专家。
- §13.40 实测：MTP on/off 的 readWall/bytes **完全相同**（1.4s / 2.37GiB，
  hit 97%）——verify 的专家并集已被 pool+LRU 覆盖，streaming 税≈0。

**事实 4：draft 权重结构 = 4 层小 transformer（不是纯投影头）。**
- 48 tensors：4 层 × (input_layernorm / q_proj / q_norm / o_proj / 3×FFN /
  post_* norms) + embed + norm + pre/post_projection。
- 提案声称官方头是"纯顶层预测投影头、无独立 Transformer 层"——与官方
  权重实况矛盾（官方就有 4 层 attention + FFN，只是 1024 维小模型）。

**结论：**
1. 提案描述的「旧方案 vs 原生 MTP」二分法在 TurboFieldfare 里不存在——
   当前实现就是原生共享激活 MTP（主干跑 1 次、draft 读主干激活 + 共享 KV、
   verify 单次批量前向）。改造成提案说的样子 = 原地重写已有逻辑。
2. r3/r4 上 MTP 负收益的根因（§13.36/§13.40）是 **verify 的 GPU 计算税**
   （batched forward 3.15 tokens = 617ms，per-token 2.6× 超线性），不是
   draft 架构。提案没有触及这个根因——它把 verify 描述成"无二次专家加载"，
   而实测专家加载本来就不是瓶颈（readWall 相同）。
3. 因此「原生 MTP 改造」**对 r3 没有必要，且不会转正**。MTP 维持默认 off；
   要转正唯一的路仍是砍 verify 计算（减 draft 数 / verify 稀疏化 / 与主干
   共享权重的真 MTP 训练，最后一项是训练级工程，非 runtime 优化）。


### 13.42 r3 减 draft A/B：d=1 比 d=2 更差——「减 draft 数」假设被证伪（2026-08-08）

为回答「减 draft 数能否逼近打平」跑 r3 交错 A/B（128 tok，prod pool64-sync，
`TURBO_FIELDFARE_MTP_ADAPTIVE=0` 固定 draft，3 轮交错）：

| r3 | OFF | d=1 | d=2 |
|---|---|---|---|
| r1 | 17.15 tok/s | 7.72 (88%) | 10.94 (85%) |
| r2 | 13.16 | 7.47 (88%) | 11.31 (85%) |
| r3 | 9.14（load）| 4.34（load）| 9.22 |
| **中位** | **13.16** | **7.47** | **10.94** |
| vs OFF | — | **−43%** | **−17%** |
| 接受率 | — | 88% (60/68) | 85% (80/94) |

**反直觉且决定性的结果：d=1 不是「更接近打平」，而是比 d=2 差 46%。**
减 draft 数把问题搞得更糟，原因：
1. **verify 的固定成本占主导**——每步的 bridge snapshot、KV commit、
   verify span 准备、rewind 是固定税，span=2 与 span=3 几乎同价，而
   d=2 每步多拿 +0.8 token 摊薄了这比固定税。
2. **边际 verify 成本随 draft 数递减**（超线性是前置加载的）——d=1 的
   每 token 成本反而最高。§13.36 的「verify 超线性 2.6×」在 span 很小
   时近似于「固定成本 ≈ 大块」，不是按 token 线性增长。
3. d=2 的 −17% 是所有 MTP 变体的最优，仍输 OFF。d=3/4（§13.36 历史）
   约 −28~30%。**最优 draft 数在 2 附近，但没有任何 draft 数能转正。**

**结论：减 draft 数不是出路（已证伪）。** 打平所需的 verify ≤240ms/步
无法靠调 draft 数达到——固定税在 d=1 时反而最重。MTP 保持默认 off；
唯一未试的 runtime 级杠杆只剩「verify 稀疏化（跳主模型后 K 层）」，且
它必须承担接受率/品质风险，需 perplexity gate 才能定案。


### 13.43 verify 稀疏化：架构与语义双重不可行——最后一个 runtime 杠杆已定案（2026-08-08）

**实测（r3 d=2 fixed，128 tok，MTP_DEBUG phases）：**
- draft 1.77s (14%) + verify 10.43s (85%) + rewind 0.01s
- 50 步（85% 接受 → 2.57 tokens/步）：draft 35ms/步 + verify 209ms/步 = 244ms
- **verify 在此是线性的**：209ms / 2.57 tokens ≈ 81ms/token-equiv vs 单 decode
  ~80ms —— 不是 §13.36 r4 d=4 的 2.6× 超线性（r4 专家更大 3.2MB、batch 3.15
  超出 pool，r3 d=2 的 batch 2.57 全在 pool+LRU 内，IO/compute 均摊平）。

**verify 稀疏化（跳主模型后 K 层）为何不可行——两层独立的死因：**

1. **架构死因（无法拼接）**：draft 是独立 4 层 1024 维网络，其
   `post_projection`（1024→2816）只在最末端把 draft 自己的预测空间映射回
   backbone 维——**draft 从未计算过任何与主模型第 L 层对齐的 2816 维中间态**。
   "verify 只跑主模型后 K 层、前 30-K 层用 draft 输出近似" 需要把 draft 状态
   拼进主模型第 30-K 层的输入，但主模型每层期望的是"上一层在相同 KV 位置、
   相同 token 序列上的真实输出"——draft 的输出是不同位置、不同分布的东西，
   喂进去 = 把垃圾喂给训练好的网络 → logits 全错。

2. **语义死因（verify 的定义）**：verify 的职责是算出**主模型在 draft 位置的
   真实 logits** 并与之比较——这是投机解码"输出与 greedy 完全一致"的保证。
   跳过任意层 = logits 不再是主模型的真实分布 → 接受/拒绝决策错误 → 输出
   不再等于 greedy → **它就不再是投机解码，而是"draft 为主 + 主模型部分校正"
   的近似生成**，品质退化无法用 perplexity gate 修复（gate 只能测量退化，
   不能消除退化）。perplexity 不是"通过就有效"的门——它只是量化代价。

**打平账（r3，假设 draft 免费——不可能的极端）：**
- 现况 244ms/步 → 10.5 tok/s；draft 归零 → 209ms/步 → 12.3 tok/s
- 本窗 base 12.5 tok/s —— **即使 draft 完全免费也只是勉强打平**，verify 的
  209ms 一步都不能省（省=语义破坏）。
- 结论：**verify 稀疏化在现架构上不是"质量赌注"，而是实现不了的方案**。
  与 §13.41（共享前 N 层不成立）、§13.42（减 draft 证伪）合并，MTP 的所有
  runtime 级杠杆全部穷尽。

**最终 MTP 定案（r2/r3/r4）：**
- 生产默认 `MTP_MODEL=""`（off）保持不变。
- MTP 变正的唯一路径 = 与主干共享权重的真 MTP（draft 即主干截断 + 轻量头），
  那是训练/repack 级工程，不是 runtime 优化；且即便实现，verify 仍必须跑主干
  完整前向（这是投机解码的本质），收益上限 = 本节打平账的镜像（draft 免费）。
- 剩余优化精力应全部回到 decode 主链（§13.44 起）。



### 13.44 「verify 改雲端」評估：KV 狀態牆 + remote spec 量化 + 三方案判決（2026-08-08）

**問題**：verify 是純 GPU 計算稅（§13.43：r3 d=2 每步 verify 209ms），能否把
verify 的完整前向搬到雲端做，讓本地只跑 decode + draft？

**決定性事實：verify 的輸入是主模型的 KV cache，不是 draft tokens。**

verify 的定義 = 主模型在 draft 位置的**完整前向**，需要主模型在位置 0..t 的
全部 KV cache。雲端沒有這份 KV 就無法 verify。KV 大小：

- 每 token ≈ 30 層 ×（K 2816 + V 2816）× fp16 2B ≈ **330KB/token**
- 128 ctx = 42MB、2K ctx = 660MB、8K ctx = 2.6GB（隨上下文線性增長）

「本地 decode + 雲端 verify」只有三條理論出路：

| 出路 | 可行？ | 原因 |
|---|---|---|
| A. 本地把 KV 傳給雲端 | ❌ | 每步傳 42MB~2.6GB（線性增長）。Wi-Fi 200MB/s 光傳 KV 就要 210ms+，比本地 verify 209ms 還慢且越來越慢；即使 cq4 壓縮也砍不到 20MB |
| B. 雲端自己維護 KV（每個被接受 token 雲端也跑一次完整前向）| ❌ | 等於**兩台機器同時跑同一模型**：雲端每 token 付 decode 30ms + verify 攤 31ms，本地同時付 80ms——系統總工作量大於單機，純浪費 |
| C. 雲端全推理 + 本地只做 draft | ✅ | = remote speculative decoding，唯一活路 |

**C 路量化（r3，d=2，接受率 85%）：**

- 結構：雲端 decode → 回傳最後 hidden state（5.6KB）+ 接受結果 → 本地 draft
  （35ms）→ draft tokens 上傳 → 雲端 verify（L20N 全常駐 ≈ 30ms）
- 每步嚴格串行：2×RTT + draft + verify

```
每步 = RTT×2 + 35ms + 30ms
  RTT 10ms（資料中心 LAN）: 105ms → 2.4 token/步 → ~23 tok/s
  RTT 30ms（好 WAN）:      125ms → ~19 tok/s
  RTT 80ms（現實家用）:     225ms → ~11 tok/s
```

**三個基準對照（r3）：**

| 方案 | tok/s | 前提 |
|---|---|---|
| 本地 base（現生產，MTP off）| **16.5-17** | 無網路依賴、免費 |
| remote spec（30ms RTT）| ~19 | 全程在線 + 雲端 GPU 費用 |
| **純雲端推理**（雲端全常駐 decode）| **~33** | 有雲端 GPU |

**判決：**

1. **「本地 decode + 雲端 verify」不存在**——KV 是 verify 的輸入，與 decode
   不可分離（A/B 兩路皆死）。
2. **remote spec 只在 <30ms RTT 才勉強贏本地**，且**永遠輸給純雲端推理**
   ——verify 的 batched forward（2.5× 計算）比順序 decode 貴，投機只對
   「RTT 是瓶頸」的場景有價值，對「server 計算飽和」是負的。
3. **現有基建的答案**：`llama-server --spec-type mtp` / SGLang `NEXTN`
   就是「雲端做 verify」的成熟形態，但那是「雲端全推理 + 雲端 MTP 頭」，
   不是混合。CGC edge-cloud bridge（M7.3-M7.6）也走同一條：雲端全算、
   edge 只拿結果。此外混合方案帶三個隱藏成本：斷線 = 生成停擺、
   prompt/輸出上雲（隱私）、每 token 雲端費用。

**戰略結論：有雲端 GPU 就整盤搬上去（雲端 decode + 雲端 MTP，~33 tok/s）；
16GB Mac 維持離線最優（base 16.5，MTP off）。中間的「混合 verify」是
資訊論死路，不值得建。**

---


### 13.45 r3/r4 生產極限對照（2026-08-08）

乾淨窗已驗證紀錄（即「生產級配置最多能做到多少」的答案）：

| 指標 | **r3（3-bit）** | **r4（4-bit）** |
|---|---|---|
| decode @256 tok | **22.6 tok/s**（§13.24 定案基準；另測 21.7）| **21.5 tok/s**（最佳乾淨窗）/ 18.4-19.8 典型（§13.24、§13.18）|
| decode @128 tok | 19.1 tok/s | 16.65 峰值（短生成暖機攤平少）|
| TTFT | ~3.5-4s（pool64-sync 稅 +0.75s）| **~4.0s**（--trust-receipt；含全量 hash 8.05s）|
| 品質定位 | 速度/品質甜區 | 最佳品質 |

達成極限的生產配置 = `run_prod.sh` 默認：pool64-sync（96 slots，hit ~97-99%）、
`TURBO_FIELDFARE_EXPERT_READ_WORKERS=8`、`TURBO_FIELDFARE_WAKE_POLL_US=5000`、
`MTP_MODEL=`（MTP off）、`--trust-receipt`。

2026-08-08 晚間負載污染實測（load 5.6，desktop 常態）：r3 256 tok 掉到
TTFT 7.82s / 2.53 tok/s、64 tok 11.2 tok/s——**不是機器極限，是負載**。
乾淨窗（load<2）重測確認仍待補。

兩者皆在 ~80-85% GPU ceiling（r2 26.5 = 83%）。剩餘槓桿：attention 降位寬
（§13.46 起，+2-4% 保底）或架構級多 token 並行（已證偽）。

---


### 13.46 attention 降位寬 A/B：3-bit attention 雙重證偽（品質 + 速度）（2026-08-08）

**動機**：§13.45 後剩餘槓桿 = attention 降位寬（+2-4% 保底）。基建早已存在
（`gemma4-r2-attn3.gturbo`：attn=3/routed=2；`TURBO_FIELDFARE_ATTN_BITS=3`
env + `_b3` kernel）但**從未驗證過**。

**發現 1：既有 b3 路徑是壞的（靜默 garbage）**。首跑 r2-attn3 + ATTN_BITS=3
輸出全 garbage（`싶 own own ownchedまま`）。逐層排除後定位根因：

- 資料面**正確**：3-bit repack 佈局精確（q_proj size 4,325,376 = 4096×2816×3/8
  吻合、offset 與 index 一致、dequant 對照 corr 0.9832 / rmse 0.0038）
- **根因**：`ATTN_BITS=3` 只接上了 **decode** 的 QKV（`dequant_int4_qkv_gemv_simd_b3`）
  與 OProj（`encodeTgx`→tgx_b3）；**prefill** 的 projection（`prefillQMM` +
  per-row `int4.encode`）仍用 4-bit kernel 讀 3-bit 資料 → prefill 隱藏態全錯 →
  全鏈 garbage。且選擇是「全域 env」而非「per-tensor」——用錯模型 + env 也會炸。

**修復（per-tensor weightBits 驅動，非 env）**：
- `Model.attentionWeightBits`（manifest `quant.attention.weightBits`）
- `prefill.metal` 新增 `prefill_dequant_int4_qmm_f16_block_b3`（3-bit 逐組解包）
- `PrefillInt4QMM` / `FusedQKVGEMV` / `DequantInt4GEMV.encodeTgx` 增加 `bits:`
  參數；b3 PSO 改為無條件構建
- runner：`encodeInt4Projection(bits:)`，per-row bits==3 改走 `encodeTgx(bits:3)`，
  prefill q/kv/o 與 decode QKV/OProj 全部傳 manifest 位寬；MPP affine 路徑限定 bits==4
- 結果：r2-attn3 **不用 env** 即輸出連貫（manifest 自動驅動），r2 4-bit 對照不變

**品質 gate（1134 tokens 同 corpus，r2 為 attn4 對照，隔離純 attention 4→3）**：

| metric | r2（attn4）| r2-attn3（attn3）| delta |
|---|---|---|---|
| ppl | 4853 | 10424 | **+115%** |
| medPPL | 1052 | 3950 | **+275%** |
| argmaxAcc | 32.8% | 23.2% | **−9.6pp** |

**速度 A/B（128 tok 交錯，load ~4.8，同窗對照）**：

| | r2（attn4）| r2-attn3（attn3）|
|---|---|---|
| decode 中位 | 17.2 tok/s | **16.1 tok/s（−6%）** |
| TTFT | 3.5s | 3.3s（±0.2s，噪聲內）|

**判決：attention 3-bit 雙重證偽，不做。**
1. **品質**：+275% medPPL、−9.6pp acc——attention 遠比 routed expert 對位寬敏感
   （QKV/OProj 每 token × 30 層全量使用，誤差無專家選擇性的稀釋）。這是「attention
   降位寬」假設（+2-4% 保底）的徹底否定——它連保底都是負的。
2. **速度**：−6% 而非 +2-4%——attention 讀取非純頻寬牆（§13.27：~58GB/s = 50%
   理論 BW），省 25% 權重位元組沒有線性兌現，且 b3 kernel 解包路徑（24-bit 位流）
   比 nibble 路徑更重。
3. **memory**：dense 540→405MB 的節省微不足道，非決策因素。

**保留項**：修復本身值得留——per-tensor bits 選擇是正確架構（修掉了「attn3 模型
靜默 garbage」的坑 + 為任何未來低 bit attention 實驗鋪路）。`TURBO_FIELDFARE_ATTN_BITS=3`
env 現已冗餘（manifest 自動驅動），保留為 force-override。**attention 2-bit 無需再試**
（只會更糟）。

**至此 §13.45 的剩餘槓桿全部定案**：attention 降位寬 ❌（本節）、架構級多 token 並行
❌（§13.23-13.24）。r3 22.6 / r4 21.5 tok/s 生產基線即當前結構終點。

---


### 13.47 attention-3bit 修復的 byte-identity 驗證：4-bit 路徑零回歸（2026-08-08）

**方法**：把 §13.46 的 6 檔修復從 working tree 完整逆向（反向 patch 到 /tmp/tf_baseline
副本，僅撤銷我的變更、保留全部 session 工作），構建出修復前 binary（TF-old，
md5 36e1019d；sanity：r3 輸出連貫）。與修復後 binary（TF-new，md5 da9a46f2）
在 r3 + 生產配置（pool64-sync / w8 / wake-poll / MTP off）下，3 prompt × 60 tok
temperature 0 輸出逐字 md5 對比：

| prompt | TF-old（修復前）| TF-new（修復後）| 結果 |
|---|---|---|---|
| code（fibonacci）| 10151ff1 | 10151ff1 | ✅ 一致 |
| prose1（Eiffel Tower）| 55c4bc91 | 55c4bc91 | ✅ 一致 |
| prose2（sky is blue）| e99f39e0 | e99f39e0 | ✅ 一致 |

**結論：attention-3bit 修復對 4-bit 路徑零回歸**——r3/r4 生產模型輸出與修復前
逐位元相同。修復的影響僅限於 manifest attention weightBits=3 的模型（§13.46）。
TF-old / TF-new 二進位保留於 /tmp（可複現）。生產 binary = 當前 .build/release。

---


**§13.47 追加（2026-08-08）— 擴展到全部位寬變體：r2 / r3 / r4 全矩陣 byte-identity**

同一套對比（TF-old vs TF-new，3 prompt × 60 tok，temperature 0）擴展到 r4
（gemma4.gturbo）與 r2（gemma4-r2.gturbo），profile 用共享 top64_code.json
（routing 全變體一致）：

| model | prompt | TF-old（修復前）| TF-new（修復後）| 結果 |
|---|---|---|---|---|
| r2 | code | b422558e | b422558e | ✅ |
| r2 | prose1 | 55c4bc91 | 55c4bc91 | ✅ |
| r2 | prose2 | 3af41f89 | 3af41f89 | ✅ |
| r3 | code | 10151ff1 | 10151ff1 | ✅ |
| r3 | prose1 | 55c4bc91 | 55c4bc91 | ✅ |
| r3 | prose2 | e99f39e0 | e99f39e0 | ✅ |
| r4 | code | 938636e4 | 938636e4 | ✅ |
| r4 | prose1 | 55c4bc91 | 55c4bc91 | ✅ |
| r4 | prose2 | d5d00f68 | d5d00f68 | ✅ |

**9/9 逐位元一致**——attention-3bit 修復對 4-bit 路徑在 r2/r3/r4 全部位寬變體
零回歸（prose1 三變體同 md5 = 高置信單一答案「Paris.」，符合預期）。byte-identity
為 load 無關測試（greedy 確定性），不受當晚 load 7.6 影響。

---
