# L4 冷页归还 · 探针判词与技术方案（C 线）

- 日期：2026-10-18（Apple M4 Air / 16GB，page = 16 KiB，统一内存带宽 ~108.8 GB/s，SSD ~1760 MiB/s）
- 模型：`Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf`，L=41（40 路由层），E=256 / top-8
- 目标：让 L4 expert pool 的**冷页**在内存压力下被真正归还（**不写 swap**），把「物理常驻」压到 6GB 以下，同时**保留大容量池的命中率**（不靠缩池牺牲 t/s）
- 性质：**探针先行、不许诺 t/s 收益**。本文档记录阶段 0 已完成的一手实测，并给出阶段 1–4 的落地与验证路径
- 配套探针：`docs/les/tools/l4_reclaim_probe.mm`（Metal 主探针）、`docs/les/tools/l4_reclaim_host_controls.mm`（host 侧 VM 语义对照）

---

## 0. 结论速览（TL;DR）

1. **判词（好消息，比预期乐观）**：在现有「4 个固定大 Metal shared buffer + 内 offset 切 slot」的 L4 架构上，**无需释放 MTLBuffer、无需 vm_deallocate、无需重构分配粒度**，只要对冷 slot 所在页范围调用 **`MADV_FREE_REUSABLE`（macOS advise #7）**，物理页就**立即退出进程 `phys_footprint`**（内存压力/swap 预算口径），压测下直接回收、**不写 swap**；无压力时页保留为免费缓存。
2. **现有代码用错了 syscall**：`cgc_discard_pages`（`llama-expert-cache.cpp:948-960`）用的是 `MADV_DONTNEED`，实测在 macOS 匿名/Metal shared 页上**近乎无动作**（resident/dirty/footprint 全不变；与 Linux 立即归零不同）。这是为什么 `CGC_POOL_MADVISE=2` 开了却看不到常驻下降。
3. **而且现有钩子只覆盖 L3，不覆盖 L4**：`cgc_discard_pool_slot`（`1001-1015`）只遍历 `cache->pool`（malloc 池）；`cgc_in_malloc_pool`（`965-983`）注释明确写 pool_ext 是 Metal allocation、不可 discard。**阶段 0 实测推翻了这个「不可 discard」的假设**——Metal shared 页可以用 `MADV_FREE_REUSABLE`（不是 DONTNEED）安全降级。
4. **安全性（探针已证）**：GPU 仍持有 MTLBuffer、在 FREEREUS 之后再次 dispatch 读取该 buffer，**不 hang、不 crash、status=completed**；重填（先 `MADV_FREE_REUSE` 再全覆写）后 host/GPU 头尾 **bit-exact**。
5. **最小改动落地（阶段 2）**：① `cgc_discard_pages` 的 `MADV_DONTNEED` → `MADV_FREE_REUSABLE`（macOS）；② 把回收从 L3 扩展到 L4（经 `pool_region()` 取 slot 页范围，按页对齐的 slab 粒度）；③ fill 前 `MADV_FREE_REUSE`；④ env-gated，默认 OFF，先 bit-identical 自证。
6. **收益不许诺**：探针在无压力小程序中证明「footprint 立即降」，但**在真实引擎高频 evict/refill 下的 madvise 系统调用开销、页对齐粒度、压力下是否真不 swap、refill 重故障延迟对 t/s 的影响**，必须用阶段 3 ABBA 量化。
7. **兜底**：`vm_deallocate` 能彻底归还（256 MiB 实测 footprint 301→44MB），但需把池拆成独立 region 的动态 slab（大重构），**仅在 FREEREUS 落地后实测不达标时才启动**。

---

## 1. 背景：为什么盯 L4 冷页

### 1.1 vmmap 铁数字（`Backup/steady_ub512_20261018_205300/vmmap_ub512.txt`）

- 进程独占脏足迹 ~12.9–13.2 GB；常驻 7-profile 宽集 8G 池超订、swap 增长 4.55 GB（tier D）。
- L4 expert pool（`SM=SHM`，64 个 ~128MB 共享块）：8359 MB vsize / 8236 MB resident / **0 swap**——但它是**匿名不可回收**的 Metal shared 页。
- `MallocHelperZone`（L3/L4 malloc 侧）：6185 MB vsize / 0 resident / **4552 MB swapped**，其中 `MALLOC_LARGE_REUSABLE` 已有 898/889 MB（libc 自己已经在走 reusable）。
- pool owner-set 5720（40×143 全满）、capacity miss 21204、evict 30170、non-resident share 44.1%、hit 95.86%。

**含义**：池里有大量被 evict / 不再热的槽，其物理页在压力下被换出或顶高 footprint；但它们在被重新 fill 时本就会被 SSD 全覆写，**这些冷页没有任何理由继续占用物理内存预算**。

### 1.2 判据口径：为什么以 `phys_footprint` 为准

- 系统判断「内存压力 / 是否 swap / 是否 Jetsam」看的是任务的 `phys_footprint`（`task_vm_info`），不是某段 region 的 `pages_resident`。
- 探针证明这两个口径**会打架**：`vm_region_extended_info` 显示 Metal shared 页回收后 `share_mode=5`、`resident` 不降、`reusable` 不升（页物理上仍作缓存）；**但同一时刻 `phys_footprint` 已全额下降**。只看 region_stat 会错误地判定「回收无效」（阶段 0 中途一度因此误判，后用 phys_footprint + 严格 A/B 对照纠正）。
- 因此 C 线所有「归还成功」的判据统一为：**`phys_footprint` 下降 + 压力下 swapped 不增长 + GPU 不 hang + 重填 bit-exact**。

### 1.3 关键内存事实（budget v3 口径）

- Metal ceiling：`recommendedMaxWorkingSetSize`，独立小程序报 **10923 MiB**；`Backup/metal_working_set.json` 引擎 stderr 抓 **11453.25 MiB**（差 530MB，下次干净窗口用引擎 stderr 重抓、保守取小）。budget 现采 11453.25×0.90=**10308 MiB**。
- 名义常驻（fixed=1396 + 池）：8G≈9.4G / 6G≈7.4G / 4G≈5.4G；**只有 4G 池名义常驻 <6GB**。C 线要做的是：**让 8G 池的「物理常驻」也接近热工作集（远小于 8G），即「大池容量、小物理常驻」**。

---

## 2. 阶段 0 探针证据（已完成，一手实测）

### 2.1 探针与复现

| 文件 | 角色 | 编译/运行 |
|---|---|---|
| `docs/les/tools/l4_reclaim_probe.mm` | Metal 主探针：vm_allocate+NoCopy 复刻 ggml 路径、runtime GPU kernel 读头尾 sentinel；6 case（NOMETAL/DONTNEED/MADV_FREE/FREEREUS/PURGEABLE/RELEASE-THEN-REUSE） | `clang++ -std=c++17 -fobjc-arc -O2 -framework Foundation -framework Metal docs/les/tools/l4_reclaim_probe.mm -o /tmp/l4_reclaim_probe && /tmp/l4_reclaim_probe [128]` |
| `docs/les/tools/l4_reclaim_host_controls.mm` | host 侧 VM 语义对照：T1–T4，region_stat + phys_footprint | 同上，文件换 `l4_reclaim_host_controls.mm`，产物 `/tmp/l4_host_controls` |

### 2.2 实测矩阵（GPU dispatch 过的 shared region；✅=达到目的）

| 动作 | region_stat 现象 | `phys_footprint` | GPU | 判定 |
|---|---|---|---|---|
| `MADV_DONTNEED`（#6，含纯 vm_allocate） | resident/dirty/head 全不变 | 不变 | — | ❌ macOS 对匿名页近乎无动作 |
| `MADV_FREE`（#5） | dirty 降、resident 留、reusable 0 | 基本不变 | — | ❌ 去脏不归还 |
| **`MADV_FREE_REUSABLE`（#7），GPU 持有 buffer** | share=5、resident 暂留、reusable 不升（**误导**） | **立即降（128MB→−128；256MB→−256）** | **再读不 hang、读旧值** | ✅ **主路径** |
| `MADV_FREE_REUSABLE`，纯 vm_allocate（未碰 Metal） | reusable 立即升、dirty 0 | 降 | — | ✅（L3 适用） |
| `setPurgeableState(Volatile)` 整 buffer | 无压力时 resident 不降（prev=2） | 无压力不降 | — | △ 粒度=整 buffer、压力下才回收，待压力验证 |
| 仅释放 MTLBuffer（drain），不 madvise | share 一度回 2 / 刚 dispatch 仍卡 5 | 严格 A/B：**等 300ms 不降** | — | ❌ 释放 buffer 不归还 |
| `vm_deallocate` | region 消失 | 256MB 301→44（降 257） | — | ✅ 彻底归还（兜底，需动态 slab） |

### 2.3 严格 A/B 对照（排除时序假象）

独立小程序（`/tmp/adv_ctl.mm`，256 MiB、dispatch 后）：

- **A 组（不 madvise，仅 usleep 300ms）**：footprint 299→299，变化 **0**——IOAccelerator 不会自然释放。
- **B 组（madvise FREEREUS）**：footprint 299→43，变化 **−256**。
- 因果成立：**footprint 下降是 FREEREUS 造成，不是 buffer 释放的延迟副作用。**

### 2.4 GPU 持有安全性（`/tmp/adv_hold.mm`）

MTLBuffer 全程持有（模拟引擎 `pool_owned_bufs` 常驻）：

```
GPU dispatch 完、buffer 仍持有，phys=299
GPU 持有中 FREEREUS rc=0 phys=43
FREEREUS 後 GPU 再讀 status=4 head=0x77777777 tail=0x77777777（不 hang）
```

重填路径（主探针 Q4）：先 `MADV_FREE_REUSE`（#8）→ 全覆写新内容 → GPU 读 → host/GPU 头尾均为新 sentinel，**PASS bit-exact**。

### 2.5 机制解释

- ggml Metal host 分配走 `vm_allocate(VM_FLAGS_ANYWHERE)` + `newBufferWithBytesNoCopy(shared, deallocator:nil)`（`ggml-metal-device.m:1643` 等）。
- NoCopy 后 region `share_mode` 由 2(私有)→5(shared)；GPU 一旦 dispatch 过，IOAccelerator 建立持久共享映射。
- `MADV_FREE_REUSABLE` 是 macOS libc malloc 真正的「归还页」契约：调用方承诺「这些页内容我不再需要，回收前我会用 REUSE 撤销」。内核对 shared 页的处理是**把它从 footprint 预算摘除、标记为可丢弃**（压力下直接回收，无需回写 swap，因为调用方已声明可丢弃）；在无压力窗口里物理页可暂时保留作缓存（所以仍读到旧值、resident 暂不降）。
- `MADV_DONTNEED` 在 macOS 上不像 Linux 那样立即零页/释放，对匿名页基本是 no-op——这是现有钩子失效的根因。

---

## 3. 阶段 1：引擎内「evict 安全窗口」只读探针（下一步，不改生产）

目标：把小程序结论平移到引擎真实数据结构前，先只读确认 4 件事。

1. **evict 点是否已有 fence**：`llama-expert-cache.cpp:878-885` 的 evict 决策点，在调用 discard 前，目标 slot 是否可能仍在 in-flight GPU 命令里？现有的 `CGC_OVERLAP_FENCE=41` / DBUF 双缓冲能否保证「evict 即无 GPU 引用」？
   - **静态读码已釘死（2026-10-18）**：L4 pool_ext 的回收 `cgc_discard_pool_slot` 唯一调用点是 `pick_slot`（:884），而 `pick_slot` 只在 build 线程的 `ensure_slot`(:1248)/`ensure_batch`(:1602)（图提交前）触发；bg 线程只填 build 预分配的 slot，其 `evict_lru`(:3902) 只碰旧 malloc `map`、不碰 pool_ext。且 victim 与当前 dispatch 同层、该层 FFN 等 fill 完成后才 dispatch，decode 串行依赖保证上一步 GPU 已完成。
   - 只读插桩（实际采用，比 fence 序号更简单）：在 discard 钩子用 `pthread_self()` 记录调用线程 id，统计一整轮 decode 的 **distinct tid 数（期望 1）**。补丁见 `docs/les/G7_STAGE1_READONLY_PROBE.md`（Hunk B，env `CGC_L4_EVICT_TRACE`）。
2. **L4 slot 的真实 stride / 页对齐**：经 `pool_region(layer,kind,&stride,&slots)`（`916-930`）打印各 kind 的 base、stride、slots，核对 stride 是否为 16 KiB 页的整数倍。
   - 单 slot 1.07–1.43 MiB 量级；若 stride 非整页，**相邻 slot 会共享一个 16KiB 页**，对单 slot madvise 会误伤邻居。
   - 对策：以「**页对齐 slab**」为回收单位（连续 N 个 slot 恰好凑整页边界），或在 split_adopt 时把 stride 向上 round 到页（会改变池几何，需谨慎评估容量）。
3. **base 地址是否页对齐**：`pool_split_alloc` 用 `ggml_backend_buft_alloc_buffer` 分配，需确认其底层 host 地址页对齐（vm_allocate 通常整页对齐，NoCopy 后 base 应页对齐）。
4. **fill 时序**：确认 `4042-4047` fill 前 madvise 点在所有 refill 路径上都会先执行 `MADV_FREE_REUSE` 且随后是**全覆写**（不能只覆写部分 offset，否则回收后读到零页会污染）。

阶段 1 产物：只读插桩补丁 + 一次 GPU 运行日志，不改变任何数值结果。

> **阶段 1 补丁已就绪（2026-10-18，待干净窗口 apply）**：
> - `docs/les/G7_STAGE1_READONLY_PROBE.patch`（1 档 2 hunk，env-gated 默认 OFF；`git apply --check` rc=0，真实 build 旗标 `-fsyntax-only` rc=0、零新增 warning）。
> - `docs/les/G7_STAGE1_READONLY_PROBE.md`（apply/编译/运行/判定门槛）。
> - `docs/les/tools/l4_slab_geometry.py`（解析 GEOM 行，算 per-kind 页/slot 交叠与回收 slab 粒度；合成自测：stride=68.5 页时 99.3% 页独占单 slot、0.7% 共享页、同层每 2 slot 凑整页）。
> - 静态已钉死：②stride 非页整（loader 无 round-up）、④fill 全覆写+短读清零；①只剩运行时验 distinct tid=1，③取真实 per-kind 几何。前置：commit2 落地 + 撤 affinity 后才 apply。

---

## 4. 阶段 2：FREEREUS 落地补丁（最小改动，env-gated 默认 OFF）

新增 env（命名建议）：`CGC_POOL_FREEREUS=0/1`（默认 0）。仅在显式开启时启用，保证主线 bit-identical。

### 4.1 改动点

1. **`cgc_discard_pages`（948-960）**：
   - macOS：`MADV_DONTNEED` → `MADV_FREE_REUSABLE`（#7）；保持页边界向内对齐（不可跨 region 首/尾页）。
   - 保留 Linux 路径（DONTNEED）。用 `#ifdef __APPLE__` 分支。
2. **扩展 `cgc_discard_pool_slot`（1001-1015）到 L4**：
   - 新增 L4 分支：经 `pool_region(layer,kind,&stride,&slots)` 取 `base`，`slot_ptr = base + slot*stride`，对「该 slot 覆盖的页范围」（按阶段 1 确认的 slab 对齐）调 `cgc_discard_pages`。
   - 推翻 `cgc_in_malloc_pool`（965-983）里「pool_ext 不可 discard」的旧假设；更新该注释，引用本判词。
3. **fill 前（4042-4047）**：macOS 改 `MADV_FREE_REUSE`（#8），随后保持现有全覆写 pread。
4. **evict 钩子（878-885）**：将 L4 slot 纳入 `CGC_POOL_MADVISE>=2`（或新 `CGC_POOL_FREEREUS`）触发范围。
5. **L3 同步**：L3 malloc 池（非 Metal shared）同样把 DONTNEED 换 FREEREUS（探针 T2 已证纯 malloc reusable 立即升）——**低风险、可能立竿见影的副产物**。

### 4.2 正确性自证（合入前必过）

- 全关（`CGC_POOL_FREEREUS=0`）与未补丁版本 **bit-identical**（`CGC_MM_BITIDENT=1`；stderr 不得出现新标记）。
- 开启后，跑 warm-skip 64 / seed 1 固定 cell，输出 token bit-exact；无 GPU hang、无 `buffer is nil`。
- 全程 vmmap 采样：`phys_footprint` 随 evict 下降；`SM=SHM` 段在压力下 swapped 不增长。

### 4.3 不做的事（边界）

- 不改池几何（slots/stride）除非阶段 1 证明必须页对齐上取整。
- 不释放 MTLBuffer、不 vm_deallocate（主路径不需要）。
- 不碰他人未提交的工作区（context.cpp bhead/les s2c、affinity 等）。

---

## 5. 阶段 3：ABBA 量化（收益不许诺，先证伪）

在干净窗口（重启 swap=0、thermal nominal、退 WorkBuddy/Chrome/FreeBuff）同场 ABBA，warm-skip 64 / seed 1 / C17 口径。

| 指标 | 采集 | 关注 |
|---|---|---|
| 物理常驻峰值 | 周期 `task_vm_info.phys_footprint` + vmmap | 8G 池 + FREEREUS 能否把常驻峰值压到 <6GB（fixed 1396 + 热工作集） |
| swapped | vmmap 各段 swapped / 系统 swapused | 冷页在压力下**不写 swap**（核心承诺） |
| decode t/s | harness C17（cell p512 n128 d0 r5） | 不能因 madvise/refill 抖动掉速；记录 p50/p95 而非只看均值 |
| 冷页占比/归还量 | discard 计数 × slot 字节 vs footprint 实际下降 | 归因「归还了多少、多少被立刻 refill」 |
| madvise 开销 | 每次 discard 计时（µs）× 频率 | 高频 evict 下 syscall 累积是否可接受 |
| refill 延迟 | fill 路径 REUSE→pread 计时 | 页已被回收后重故障 + 全覆写是否增加 SSD 等待 |
| 命中率 | pool hit % | 池容量不变 → hit 应基本持平（95.86% 基线） |

**诚实的成功/失败判据**：
- 成功：8G 池物理常驻峰值显著下降（目标 <6GB 量级），swapped 不增长，decode t/s 落在基线 ±5%（无显著退化），bit-exact。
- 失败/降级：若高频 madvise 或 refill 抖动导致 t/s 明显退化，则改为「更粗 slab 粒度 + 仅在 evict 风暴时批量回收」；若仍不达标，才评估阶段 4 动态 slab（vm_deallocate）。

---

## 6. 阶段 4：压力与并发验证、动态 slab（兜底）

1. **真实压力**：在后台制造可控内存压力（或跑常驻 7-profile 宽集），确认 FREEREUS 冷页被直接回收、`swapped` 不增长（而非只在无压力小程序里退出 footprint）。
2. **prefill / OA_ASYNC 并发**：确认 prefill 流式与 decode 异步预取并发时，回收与 fill 不竞争、无页被「回收后又被 GPU 读到零页」。
3. **动态 slab（仅当 FREEREUS 不达标）**：把固定大 buffer 改为按回收粒度的独立 `vm_allocate` region + 独立 NoCopy MTLBuffer（几十~几百个 slab，平衡句柄数，避免 per-slot 40×143×3≈1.7 万 buffer）；evict 时 fence→释放 slab buffer→`vm_deallocate`（探针证 301→44），refill 时重新 alloc+pread+NoCopy。这是大重构（GPU 绑定从固定 buffer+offset 改为句柄表查指针、地址稳定性、fence 时序），**单独立项，不与 FREEREUS 混做**。

---

## 7. 与其他路线的关系

- **G6 缩池（8/6/4G）**：FREEREUS 是「**不缩池也降物理常驻**」的互补路线。若成功，可保留 8G 大池的高命中率、同时拿到 <6GB 物理常驻——比缩池更优。建议 G6 增加一档 **8G+FREEREUS** 观察臂，与 6G/4G 物理常驻/t/s 直接对比。
- **LES G3 preadv（砍 SSD miss）**：C 线省内存、G3 砍等待，两者正交。FREEREUS 回收后若 refill 抖动上升，恰好由 G3 preadv 把 SSD 等待藏进当前层 compute。
- **decode 25**：FREEREUS 本身不直接加 t/s（省内存，不是砍 43.20ms 非 kernel）；间接收益是消除 swap 抖动、可保大池降 capacity miss。25 的主线仍是 LES（G3）+ expert kernel 利用率（38–45%）+ b 重叠。
- **<6GB 物理常驻**：这是 C 线的直接目标。固定大 Metal shared buffer 上原地 madvise 不可行的旧判断（D 判词）**被阶段 0 修正为：DONTNEED 不可行，但 FREEREUS 可行且最小改动**。

---

## 8. 风险与未决诚实边界

- 探针是 128/256MB 整 region、低频；引擎是成千上万 slot、高频 evict/refill，**syscall 开销与页对齐粒度未在引擎内验证**。
- 「压力下不 swap」目前由内核语义保证 + 无压力小程序 footprint 证据，**未在真实超订压力下复现**（阶段 4）。
- Metal ceiling 10923 vs 11453 的 530MB 差异待引擎 stderr 重抓（保守取小）。
- `SM=SHM` 8.2G resident 的归属（dense/compute 还是第二份 expert Metal 副本）未 100% 拆清（若双副本，修掉比缩池杠杆更大）；干净窗口用启动日志 `CGC-POOL-SPLIT` / `GATHER-SLAB` base 对 vmmap 实时核对。
- `MADV_FREE_REUSABLE`/`MADV_FREE_REUSE` 是 macOS 私有（libc 已公开符号）；Linux 分支保留 DONTNEED。
- 本文档不承诺 t/s 数字；<6GB 物理常驻的可达性以阶段 3 实测为准。

---

## 9. 下一步清单（按序）

1. **阶段 1**：写只读插桩（fence 差值、per-kind stride/页对齐、base 对齐、fill 全覆写），一次干净窗口运行落数据。
2. **阶段 2**：在 commit2 落地、撤 affinity 之后，基于阶段 1 事实写 `CGC_POOL_FREEREUS` 补丁（默认 OFF），编译 + bit-identical 自证。
3. **阶段 3**：G6 增加 8G+FREEREUS 臂，ABBA 量化常驻/swap/t/s/开销。
4. **阶段 4**：真实压力 + 并发验证；不达标才启动动态 slab 立项。
5. 结果回写 mindmap LES 节点 HTML 与排名 panel（附检验档）。
