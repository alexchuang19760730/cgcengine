# CGC TrueOrthoKDA + R-SWA 统一技术白皮书（重构版）

**版本**: v1.0  
**状态**: 设计 / 重构规范版（Design & Refactor Spec）  
**定位**: 规范「重构后的 TrueOrthoKDA（窗口作用域 O(1) KV + 局部正交压缩器）」与「R-SWA（Reference / recency 管理器）」的分工边界，并定义 `inject_unified_ir_for_role` 统一注入入口，使其可挂接到 sglang / vllm / cloud_sglang active path 等多个 backend。

---

## 〇、本文与既有白皮书的关系（必须先读）

本文**不是**对以下既有文档的重复，而是对其**过度宣称**的纠偏 + 对缺失零件的补充：

| 既有文档 | 过去宣称 | 本文核实后的真实状态 |
|---|---|---|
| `CGC_Gate_2.3_unlimited_rswa_prefill_pool` | R-SWA / PrefillPool 已 396 / 354 行完整实现 | ❌ **R-SWA 与 PrefillPool 全树无任何实现代码**，仅白皮书伪代码 + 测试桩（且测试 import 的 `rswa_integration` 模块不存在） |
| `CGC_M75_TRUEORTHOKDA_ACTIVE_RUNTIME` | TrueOrthoKDA 已接入 active runtime | ✅ **属实**：`true_ortho_kda.py` 已实现，M7.5 实测 `compression_ratio = 0.9259`（小状态几乎无压缩，仅长上下文 ≫32 才显收益） |
| 历史宣称「256x 压缩」「CGC env var 一键开关 RSWA/256x」 | 存在魔法压缩比与开关 | ❌ **代码内无任何 CGC env var 开关**；TrueOrthoKDA 当前为固定容量(32)环形淘汰压缩器，理论极限取决于 `ortho_base_dim` 与真实序列长度之比，并非恒定 256x |

**本文核心主张**：把 TrueOrthoKDA 的角色从「被误读的内积优化器」纠正为「**窗口作用域的正交 KV 压缩器**」，让它与 R-SWA 在**互不重叠的两个维度**上分工，并通过一个真正的注入入口挂到各 backend。

> **v1.0 后续交付（已落地为 `rswaengine/`）**：R-SWA 现已以 C/C++ 实现（`rswa_manager.cpp`，含 Reference 永久区 + 滑动窗口环形 KV + 可见性掩码 + 混合层策略）；`inject_unified_ir_for_role` 已实现（`unified_ir.cpp`）并接入 `cgc_cpp` 引擎核心，与 `cgc_inject_strategy` 双向联动（注入后自动启用 KDA replace mode）；另提供 sglang / vllm 的 Python monkey-patch adapter 与窗口压缩器的 CUDA kernel。详见 §13。

---

## 一、目标与可达成性声明

### 1.1 业务目标

为「LLM 长对话无限生成」提供：

1. **恒定 KV 显存上限** —— KV 总量不随对话轮数无限上涨；
2. **Reference 信息不衰减** —— 系统 Prompt / 关键事实锁定为永久可见；
3. **窗口内注意力精度提升** —— 缓解滑动窗口常见的注意力弥散；
4. **跨 backend 可落地** —— sglang / vllm / cloud_sglang active path 共用同一套 IR 与注入逻辑。

### 1.2 可达成性结论（明确）

| 子目标 | 能否达成 | 依据 |
|---|---|---|
| 无限长对话 + 恒定 KV | ✅ 能 | 滑动窗口注意力的成熟范式 |
| Reference 永久可见 | ✅ 能 | Reference 旁路隔离（方案 A） |
| 窗口内数值稳定 / 精度 | ✅ 能 | TrueOrthoKDA 已实现，局部正交基胜任 |
| 复杂度 O(n²)→O(n) 线性 | ✅ 能，但**只靠 R-SWA** | 正交基不降阶，仅常数系数加速 |
| R-SWA + OrthoKDA「按维度正交叠加」 | ⚠️ 需纠正后成立 | 见 §2、§4 |

> ⚠️ **诚实边界（v1.0 交付后更新）**：R-SWA 与 `inject_unified_ir_for_role` 现已以 C/C++ 实现（见 §13）。剩余工程缺口为：① sglang / vllm 的**真实注意力层 monkey-patch 接线**（adapter 已提供集成模板，需按目标框架版本落地）；② 窗口压缩器 CUDA kernel 的**实际编译/运行验证**（本机无 nvcc，kernel 已对齐 v4 风格编写，待 GPU 环境编译）。

---

## 二、分工原则：两个维度完全正交，各管一摊

### 2.1 作用域划分

```
R-SWA  (Reference Sliding Window Attention)
  作用域：序列层面 / Token 选择层面
  - 把序列切分为：Reference 永久区 + 滑动窗口环形 KV 缓存
  - 决定「哪些 K/V 有资格参与注意力计算」
  - 提供恒定 KV 显存上限（窗口容量固定）
  - 复杂度：O(M + W)，M=Reference 长度，W=窗口尺寸

TrueOrthoKDA (Gram-Schmidt 正交基 KV 压缩器)
  作用域：向量空间层面 / KV 表征层面（仅作用于窗口内部）
  - 在「R-SWA 已选定的窗口 K/V」集合内部做变换
  - 对窗口 K 执行 Gram-Schmidt 正交化，构造局部正交基
  - 用 ≤ ortho_base_dim 个基向量替代完整窗口 KV 参与 attention
  - 复杂度：窗口内 O(ortho_base_dim²) = 常数（与窗口原始长度无关）
```

### 2.2 数据流顺序

```
输入序列
  └─> R-SWA 掩码：划分 Reference(永久) + Window(滚动)
        ├─ Reference K/V ──> 标准注意力（不进正交基，见 §4.4）
        └─ Window  K/V ──> 灌入 WindowOrthoKVCompressor
                                └─> 局部正交基 attention ──> 输出
```

### 2.3 「各管一摊、不抢活」的精确含义

历史误读认为「窗口侧算力双重缩减（R-SWA 减 Token 数 + OrthoKDA 减内积）」。在**真实实现**下该说法**不成立**：

- TrueOrthoKDA 的 `attention()` 永远只与 ≤ `ortho_base_dim`(默认 32) 个基向量算内积，窗口原始长度不影响其计算量；
- 因此窗口侧的「算力缩减」由 TrueOrthoKDA **独占**，R-SWA 在窗口上的边际算力收益 ≈ 0；
- R-SWA 的真正价值是 **Reference / recency 管理 + 作为非正交层的兜底**（当某层选择不使用正交压缩时，仍能靠窗口维持恒定 KV）。

→ 结论：R-SWA 管「可见性（哪些 Token）」，OrthoKDA 管「窗口 KV 的紧凑表征」。两者职责不重叠。

---

## 三、真实 TrueOrthoKDA 机制核实（基于 `cgc_engine/cgc/true_ortho_kda.py`）

> 本节所有接口均来自真实源码，非设计臆测。

### 3.1 核心类 `TrueOrthoBasisAccumulator`

```python
class TrueOrthoBasisAccumulator:
    def __init__(self, num_heads, head_dim, ortho_base_dim: int = 32,
                 eps: float = 1e-8, decay_rate: float = 0.01):
        # K, V 形状固定: [num_heads, ortho_base_dim, head_dim]
        # decay 形状:    [num_heads, ortho_base_dim]
        # current_dim   : 当前已填充的基数量
        # total_updates : 累计 update 次数

    def gram_schmidt(self, v, basis, idx):
        # 对 v 减去其在 basis[:idx] 上的投影并归一化，返回正交向量

    def update(self, k_new, v_new):
        # k_new, v_new: [num_heads, head_dim]
        # 若 current_dim < ortho_base_dim:
        #     Gram-Schmidt 后 K[:,i] += k_ortho; V[:,i] += v_new; current_dim++
        # 否则（满载）:
        #     环形移位淘汰: K[:,j] = K[:,j-1]（最旧被踢）
        #     新向量塞入位置 0
        # 返回 {K, V, decay, current_dim}

    def attention(self, Q):
        # Q: [batch, num_heads, head_dim] -> out: [batch, num_heads, head_dim]
        # score = Q · K(≤ortho_base_dim); attn = score * decay; out = Σ attn * V
        # 计算量恒为 O(batch * heads * ortho_base_dim²)

    def get_state(self) -> Dict:   # {K, V, decay, current_dim, total_updates}
    def set_state(self, state):    # 用于跨节点 state 传输 / resume

    def memory_footprint(self) -> Dict:
        # 固定: num_heads * ortho_base_dim * (head_dim*2 + 1) 个元素
        # 与 seq_len 完全无关 -> O(1) 显存
```

### 3.2 关键事实（决定重构方向）

1. **它是压缩器，不是内积优化器**：`ortho_base_dim=32` 是其 KV 容量上限。超过 32 即 **ring 移位淘汰最旧向量**，不保留全部 Token。这与「保留全部选中 Token」的误读相反。
2. **O(1) KV 已天然成立**：`memory_footprint` 与 seq_len 无关，正是「恒定 KV 上限」所需。
3. **ring 淘汰会破坏正交性**：满载后 `K[:,j]=K[:,j-1]` 直接复制向量，**未重新正交化**，基间正交性会漂移。这是 §6 风险 1 的根源，必须在重构中修复。
4. **TimeDecay + NoPE 已存在**：`TimeDecayAttention`（`exp(-i*decay_rate)`）与 `NoPEPositionEmbedding` 已融合进 `TrueOrthoBasisKDA`。

### 3.3 实测证据（M7.5）

`2026-06-18` active rerun：

- `raw_state_bytes = 264613`，`compressed_state_bytes = 244993`
- `compression_ratio = 0.9259`（小状态几乎无压缩）
- `state_kind = kda_state_v1`，`resume_decode_executed = true`，`cpu_copy_count = 0`，`uma_buffer_used = true`

→ 说明：压缩收益只在**真实长上下文 ≫ ortho_base_dim** 时显现；同时 eviction 会带来质量损失，需消融验证。

---

## 四、重构方案：TrueOrthoKDA → 窗口作用域正交 KV 压缩器

### 4.1 重构目标

把现有 `TrueOrthoBasisAccumulator` 重新定位为 **WindowOrthoKVCompressor**：

1. **窗口对齐的淘汰语义**：淘汰由 R-SWA 窗口推进触发，而非独立 32 计数器；
2. **禁止 Reference 进入基**：Reference 走标准注意力，避免被 ring 踢掉；
3. **淘汰后重正交化**：每轮窗口滚动后，对保留的基子集重新 Gram-Schmidt，抑制漂移；
4. **保留 get_state/set_state**：继续支撑 M7.5 的 `kda_state_v1` 跨节点传输与 resume。

### 4.2 目标接口（重构后）

```python
class WindowOrthoKVCompressor:
    """
    R-SWA 窗口作用域内的正交 KV 压缩器。
    只接收窗口 K/V，绝不接收 Reference。
    """
    def __init__(self, num_heads, head_dim,
                 ortho_base_dim: int = 32,   # 窗口压缩预算，可随 window_size 配置
                 window_size: int = 4096,    # 与 R-SWA 窗口一致，仅用于容量断言
                 eps=1e-8, decay_rate=0.01):

    def feed_window_token(self, k, v):
        """
        接收窗口内一个新 Token 的 K/V（[num_heads, head_dim]）。
        内部执行 Gram-Schmidt 正交化后累积进基；
        若基已满，按 R-SWA 窗口淘汰顺序（最旧窗口 Token）替换，
        并触发 _reorthogonalize()。
        """

    def on_window_advance(self, evicted_positions: List[int]):
        """
        由 R-SWA 在窗口滚动时回调：通知哪些窗口 Token 被淘汰，
        从基中移除对应贡献并重新正交化，保持基间正交。
        """

    def window_attention(self, Q):
        """Q: [batch, num_heads, head_dim] -> 与 ≤ortho_base_dim 基向量算 attention"""

    def get_state(self):   # 继承原有，支撑 kda_state_v1 传输
    def set_state(self, s):
```

### 4.3 与 R-SWA 窗口对齐（关键集成点）

```
R-SWA 窗口推进一个 Token
   └─> 调用 compressor.on_window_advance([evicted_pos])
         └─> 从基中扣除该 Token 贡献 / 重新 Gram-Schmidt
               └─> 基容量恒定，显存 O(1)
```

这样 **淘汰顺序由 R-SWA 统一掌控**，正交基不再有「独立 32 计数器与窗口错位」的问题。

### 4.4 Reference 绝不进基（规避风险 1 + 风险 2）

```
Reference 永久区  ──> 标准 Full/Rope Attention（不调用 compressor）
Window 滚动区    ──> WindowOrthoKVCompressor（局部正交基）
```

理由（双重）：

- **风险 1（ring 淘汰破坏正交）**：若 Reference 进基，会被 32 容量 ring 踢掉，关键指令衰减；
- **风险 2（分布偏移）**：Reference（长期固定）与 Window（滚动）特征分布不同，合并正交会被大量旧 Reference 主导，压制窗口新 Token。

→ 分层隔离是**强制**设计约束，不是可选优化。

---

## 五、R-SWA 设计（从零实现）：Reference / recency 管理

> R-SWA 现已实现于 `rswaengine/cpp/src/kernels/rswa_manager.cpp`（C/C++），本节为其接口与约束规范（已落地）。

### 5.1 职责

1. 划分 **Reference 永久区**（系统 Prompt、关键事实）—— 永不淘汰、走标准注意力；
2. 维护 **滑动窗口环形 KV**（近期对话）—— 容量固定 `W`，超出即淘汰最旧；
3. 产出 **可见性掩码**：`visible = Reference ∪ Window`；
4. 在窗口滚动时回调 `WindowOrthoKVCompressor.on_window_advance`。

### 5.2 掩码逻辑（伪代码）

```python
class RSWAManager:
    def __init__(self, reference_tokens, window_size: int):
        self.ref = reference_tokens          # 永久可见，不进压缩基
        self.W = window_size
        self.window = deque(maxlen=W)        # 环形窗口

    def step(self, new_kv):
        evicted = None
        if len(self.window) == self.W:
            evicted = self.window.popleft()  # 淘汰最旧窗口 Token
        self.window.append(new_kv)
        return evicted                        # 回调给 compressor

    def visibility_mask(self, seq_len):
        # Reference 位 = 1，Window 内 = 1，Window 外 = 0
        return mask
```

### 5.3 cuda-graph 兼容（高优先工程约束）

R-SWA 动态淘汰会破坏 cuda-graph 固定计算图。落地须：

- **固定 `window_size` + 预分配最大 KV 缓冲**；
- decode 阶段窗口位置用索引偏移而非动态形状；
- 或：R-SWA 仅用于 **prefill**（非 cuda-graph），decode 用标准注意力 + cuda-graph。

---

## 六、三大风险与缓解（基于真实实现）

### 风险 1：ring 淘汰破坏正交基

- **现象**：`true_ortho_kda.py` 满载后 `K[:,j]=K[:,j-1]` 直接复制，**不重新正交化**，基间正交性漂移。
- **缓解（§4.3）**：淘汰由 R-SWA 窗口推进触发，每次滚动后 `_reorthogonalize()`。
- **残余**：窗口内局部正交仍优于跨窗口全局基，但理论上无法完全消除漂移；靠消融确认可接受阈值。

### 风险 2：Reference / Window 分布不一致

- **现象**：合并正交时旧 Reference 主导基，压制窗口新 Token。
- **缓解（§4.4）**：**强制分层隔离**——Reference 走标准注意力，绝不进压缩基。本白皮书将其定为硬约束。

### 风险 3：训练 / 推理 Gap 叠加

- **现象**：预训练 Full Attention；推理同时开稀疏掩码 + 正交投影 → PPL 上升、长对话逻辑断裂。
- **缓解（二选一）**：
  1. **训练期同时启用「R-SWA 掩码 + 窗口内 OrthoKDA」LoRA 微调**（效果最优，成本高）；
  2. **混合层策略**：偶数层完整 R-SWA+OrthoKDA；奇数层仅 R-SWA、不启用正交变换，保留原生特征通路。

---

## 七、`inject_unified_ir_for_role`：统一注入入口

> 该函数现已实现于 `rswaengine/cpp/src/unified_ir.cpp`（C/C++），并接入 `cgc_cpp` 引擎核心：实现在构建 `cgc_strategy_t` 后回调 `cgc_inject_strategy`，而 `cgc_inject_strategy` 检测到 `unified_ir:` 元数据即自动 `cgc_set_kda_replace_mode(true)`，形成真正的双向联动（见 §13.3）。

### 7.1 职责

把「R-SWA + 窗口 OrthoKDA」的组合注意力，**以 backend 无关的中间表示（IR）描述，再分别适配**到各推理框架，避免为每个 backend 重写一遍注意力逻辑。

### 7.2 接口签名（提案）

```python
def inject_unified_ir_for_role(
    role: str,                      # "prefill" | "decode" | "edge_resume"
    model,                         # 目标 backend 的模型对象
    config: dict,                  # {reference_tokens, window_size, ortho_base_dim, hybrid_layers, ...}
    backend: str = "sglang",       # "sglang" | "vllm" | "cloud_sglang"
) -> "InjectedModel":
    """
    1. 构造 UnifiedIR：layers 列表，逐层标记
         {type: "rswa_reference_std" | "rswa_window_ortho" | "rswa_window_std", ...}
       并实例化 RSWAManager + 每窗口 WindowOrthoKVCompressor。
    2. 选择 backend adapter，执行 monkey-patch（替换原生 attention）。
    3. 返回注入后的模型（保留 get_state/set_state 供 M7.5 传输）。
    """
```

### 7.3 IR 层标记语义

| 层标记 | 含义 | 是否进压缩基 |
|---|---|---|
| `rswa_reference_std` | Reference 区，标准注意力 | 否（硬约束 §4.4） |
| `rswa_window_ortho` | 窗口区，启用局部正交压缩 | 是 |
| `rswa_window_std` | 窗口区，标准注意力（混合层策略奇数层） | 否 |

### 7.4 Backend Adapter 策略

```
sglang:
  - 替换 RadixAttention / 自定义 attention kernel
  - 固定 window_size 预分配 KV，兼容 cuda-graph
vllm:
  - 替换 attention backend（如 FlashAttention 封装层）
  - 注入 RSWAManager 可见性掩码
cloud_sglang (M7.5 active path):
  - 复用现有 kda_state_v1 传输
  - cloud prefill 端用注入后的 window_ortho 产出 state
  - edge 端 set_state 后 resume decode（沿用 m75 已验证路径）
```

---

## 八、多 backend 注入适配清单

| Backend | 注入点 | 复用能力 | 状态 |
|---|---|---|---|
| sglang | `sglang_adapter.patch_sglang` 替换 attention（RSWAOrthoAttention） | cuda-graph 兼容窗口 | ✅ adapter 已提供（需 sglang+torch 落地） |
| vllm | `vllm_adapter.patch_vllm` 替换 attention backend（RSWAOrthoBackend） | 掩码注入 | ✅ adapter 已提供（需 vllm+torch 落地） |
| cloud_sglang (M7.5) | `kda_state_v1` 生产/消费 | `get_state/set_state` 已验证 | 半就绪（TrueOrthoKDA 部分）+ wokdc 兼容 |

---

## 九、实证验收与 MVP 顺序

### 9.1 MVP 顺序（强制）

1. **先单独上线 R-SWA**，打通「无限长对话 + 恒定 KV」基线（不接 OrthoKDA）；
2. **稳定后**，在滑动窗口分支接入 `WindowOrthoKVCompressor`（禁止跨窗口维护全局基）；
3. **消融对照**：纯 R-SWA VS R-SWA + 窗口 OrthoKDA。

### 9.2 验收指标

| 指标 | 说明 |
|---|---|
| TTFT | prefill 首 token 延迟 |
| 持续多轮困惑度（PPL） | 长对话逻辑是否断裂 |
| 任务成功率 | 端到端对话任务 |
| 显存占用 | 是否恒定（验证 O(1) KV） |
| 单步 decode 耗时 | 窗口正交基是否带来常数加速 |
| `compression_ratio` | 仅长上下文 ≫ ortho_base_dim 时显著（如 M7.5 小状态仅 0.926） |

### 9.3 回退策略

若精度下滑 → 立即切**分层方案**：Reference 关闭正交优化（§4.4 已是默认硬约束，只需关闭窗口侧 `rswa_window_ortho`、退回 `rswa_window_std`）。

---

## 十、实际状态 vs 设计目标（诚实声明，必读）

| 组件 | 实际状态 | 本文定位 |
|---|---|---|
| `TrueOrthoBasisAccumulator` / `TrueOrthoBasisKDA` | ✅ 已实现（517 行 + C++ kernel 参考） | 重构为 `WindowOrthoKVCompressor` |
| `get_state/set_state` + M7.5 active runtime | ✅ 已实现并验证（resume decode、0-copy） | 直接复用，支撑跨节点 |
| GDS (`cuFileWrapper`) | ✅ 已实现（需真实 cuFile 库） | 可选：窗口 KV 落盘加速 |
| **R-SWA** | ✅ **已实现 C/C++**（`rswa_manager.cpp`） | 已落地（§5）；待 sglang/vllm 真实 patch 接线 |
| **`inject_unified_ir_for_role`** | ✅ **已实现 C/C++**（`unified_ir.cpp`）并联动 `cgc_inject_strategy` | 已落地（§7/§13） |
| 多 backend adapter（sglang/vllm） | ✅ **Python monkey-patch 模板已提供** | 已提供（§8/§13）；待按框架版本落地 |
| Reference/Window 分层隔离 | ✅ **C/C++ 已强制**（Reference 恒标准注意力，绝不进压缩基） | 已实现为硬约束 |
| 256x 压缩 / CGC env var 开关 | ❌ 不存在 | 本文**撤销**该宣称，改为按 `ortho_base_dim` 与真实 seq_len 比值计算 |
| 窗口压缩器 CUDA kernel | ⚠️ 已编写（对齐 v4 风格），本机无 nvcc 未编译 | 待 GPU 环境编译验证（§13.4） |

---

## 十一、最小改造文件清单（提案）

### 11.1 重构 / 新建

- `cgc_engine/cgc/window_ortho_kv_compressor.py`（由 `true_ortho_kda.py` 重构而来，新增窗口对齐淘汰 + 重正交化 + Reference 排斥）
- `cgc_engine/cgc/rswa_manager.py`（**新建**，R-SWA Reference/recency 管理）
- `cgc_engine/compiler/inject_unified_ir_for_role.py`（**新建**，统一注入入口 + IR 构造）
- `cgc_engine/compiler/backends/{sglang,vllm,cloud_sglang}.py`（**新建** adapter）

### 11.2 复用（不改动）

- `cgc_engine/cgc/true_ortho_kda.py` 的 `gram_schmidt` / `attention` / `get_state` / `set_state`
- `app/servers/cloud_socket_server.py` + `app/edge_engine/local_infer.py` 的 M7.5 `kda_state_v1` 路径

---

## 十二、结论

1. ✅ **目标可达**：无限长对话 + 恒定 KV + 窗口精度提升，路径清晰、风险可管理；
2. ⚠️ **前提纠正**：TrueOrthoKDA 是「固定容量(32)环形淘汰正交基压缩器」，不是「保留全部 Token 的内积优化器」；重构后它专司**窗口 KV 压缩**，R-SWA 专司 **Reference/recency 管理**，两者维度正交、各管一摊；
3. 🚧 **剩余工程缺口（v1.0 交付后）**：R-SWA 与 `inject_unified_ir_for_role` 的 C/C++ 实现已完成（见 §13），剩余为 sglang/vllm 真实注意力层 monkey-patch 接线（adapter 模板已提供）与窗口压缩器 CUDA kernel 的 GPU 编译验证；
4. 📌 **硬约束**：Reference 绝不进压缩基（规避风险 1、2）；训练/推理 Gap 靠混合层策略或 LoRA 缓解（风险 3）；
5. 📉 **撤销夸大宣称**：256x 压缩与 CGC env var 一键开关不成立，改为按真实 `ortho_base_dim` / seq_len 比值计。

> 本文为**设计 / 重构规范**。**v1.0 的交付已落地为 `rswaengine/` 自包含引擎目录**（C/C++ 实现 + Python adapter + 白皮书），详见 §13。

---

## 十三、rswaengine 交付物、构建与目录结构

### 13.1 目录结构

```
rswaengine/
├── README.md
├── docs/
│   └── CGC_TrueOrthoKDA_RSWA_Unified_Technical_Whitepaper_v1.0_zh_CN.md
├── cpp/
│   ├── include/
│   │   ├── cgc_cpp.h                 # 引擎核心 API（含统一 IR 联动 include）
│   │   ├── unified_ir.h              # inject_unified_ir_for_role / IR 层类型
│   │   └── kernels/
│   │       ├── rswa_manager.h        # R-SWA 管理器接口
│   │       ├── window_ortho_kv_compressor.h  # 窗口作用域正交 KV 压缩器
│   │       └── ortho_kda_v4.cuh      # 既有 TrueOrthoKDA v4（CUDA/CPU 双路径）
│   ├── src/
│   │   ├── cgc_cpp.cpp               # 引擎核心 + cgc_inject_strategy <-> unified IR 联动
│   │   ├── unified_ir.cpp            # 统一注入实现（回调 cgc_inject_strategy）
│   │   └── kernels/
│   │       ├── rswa_manager.cpp
│   │       ├── window_ortho_kv_compressor.cpp   # CPU 路径
│   │       ├── window_ortho_kv_compressor.cu    # CUDA kernel（对齐 v4 风格）
│   │       ├── ortho_kda_v4.cpp / .cu / _binding.cpp  # 既有 v4
│   ├── examples/
│   │   ├── main_unified.cpp          # R-SWA + 窗口 OrthoKDA 功能演示
│   │   └── main_engine.cpp           # 引擎联动验证（SDPA->KDA 重路由）
│   ├── build_unified.sh              # 纯 CPU / C++17 构建（产出 libcgc_unified.so）
│   └── build_unified_cuda.sh         # nvcc 构建（CUDA kernel）
└── python/
    ├── cgc_unified_injection.py      # ctypes 桥接（libcgc_unified.so）
    ├── sglang_adapter.py             # sglang monkey-patch 消费者
    └── vllm_adapter.py               # vllm monkey-patch 消费者
```

### 13.2 构建与验证

```bash
cd rswaengine/cpp && bash build_unified.sh      # 编译 libcgc_unified.so + 两个 demo 并运行
bash build_unified_cuda.sh                      # 若有 nvcc，编译 CUDA kernel
cd rswaengine/python && python3 cgc_unified_injection.py  # ctypes 自检
python3 sglang_adapter.py                        # sglang 消费路径（numpy 独立 demo）
python3 vllm_adapter.py                          # vllm 消费路径（numpy 独立 demo）
```

验证结果（CPU 路径已通过）：
- 引擎联动：调用 `cgc_inject_unified_ir_for_role` → 回调 `cgc_inject_strategy` → 自动 `cgc_set_kda_replace_mode(true)`；`cgc_execute_opcode(0x10)` 重路由到 `0x11(KDA)` ✅；
- R-SWA：喂入 20 token（>窗口 8）正确触发环形淘汰，visible 恒为 ref+win=12 ✅；
- 窗口 OrthoKDA：`current_dim` 恒为 ortho_base_dim=8（O(1) 压缩表征）✅；
- `kda_state_v1` round-trip：`max|K diff|=0`（兼容 M7.5 跨节点传输）✅；
- Python adapter：经 ctypes 调用同一 C 库，combined_attention 数值有限、state 可序列化 ✅。

### 13.3 统一注入与引擎核心的联动（cgc_inject_strategy 双向联动）

```
cgc_inject_unified_ir_for_role(role, model, config, adapter)
   └─> 构建 cgc_strategy_t（metadata 含 "unified_ir:role=...;adapter=..."）
         └─> cgc_inject_strategy(&strat)                ← 联动①（unified_ir.cpp 回调）
               └─> 检测到 "unified_ir:" 元数据
                     └─> cgc_set_kda_replace_mode(true) ← 联动②
                           └─> cgc_execute_opcode(0x10) -> 0x11 重路由生效
```

即「统一 IR 注入」真正成为「引擎全局策略」的生产者，二者不再是各自为政的两套状态。

### 13.4 窗口压缩器 CUDA kernel（对齐 v4）

`window_ortho_kv_compressor.cu` 沿用 `ortho_kda_v4.cu` 的约定：device Gram-Schmidt、
单 kernel 完成「窗口环追加 + 局部正交基重建（re-orthogonalize）」、注意力 kernel，
并以 `extern "C"` host 包装（`window_ortho_create_cuda / feed_cuda / attention_cuda /
get_state_cuda / set_state_cuda / destroy_cuda`）暴露。本机无 nvcc，kernel 已编写待
GPU 环境编译验证（`bash build_unified_cuda.sh`）。

### 13.5 Python monkey-patch adapter（消费统一 IR）

`cgc_unified_injection.py` 经 ctypes 加载 `libcgc_unified.so`，暴露
`inject_unified_ir_for_role()` 与 `RSWAManager`（封装 rswa_manager + 窗口压缩器 C API）。
`sglang_adapter.patch_sglang()` / `vllm_adapter.patch_vllm()` 据此注册 IR 并把框架注意力
层替换为 `RSWAOrthoAttention` / `RSWAOrthoBackend`（每请求/序列一个 RSWAManager，
Reference 标准注意力、窗口走 OrthoKDA）。真实 patch 需 sglang/vllm + torch，并以目标
框架版本的注意力 API 为准落地；无框架时提供 numpy 独立 demo 验证同一条计算路径。

---

*附录：本文所有接口签名均依据 `ComputeGraphCompiler-main/cgc_engine/cgc/true_ortho_kda.py` 真实源码与 `CGC_M75_TRUEORTHOKDA_ACTIVE_RUNTIME` 实测证据撰写，未经证实的宣称已在 §十、§〇 明确标注。*
