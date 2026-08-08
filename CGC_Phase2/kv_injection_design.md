# CGC step8 KV cache 注入设计 (layer-split resume)

## 1. 问题复盘

**现状**:layer-split 反向(Mac→cloud)。Mac 跑前 P=8 层 forward → emit `hidden_P` →
cloud 从 layer P EXTEND forward 到 end。

**崩溃点**:cloud EXTEND 时 `store_cache(layer 8..47)`,但 sglang KV cache 期望
layer 0..47 连续(layer 0..7 空槽)→ `tvm.error.InternalError Tensor match failed
kvcache.cuh:177`。

**根因**:cloud resume 跳过前 P 层(`loop_start = max(start_layer, _resume_cut+1)`),
前 P 层 KV cache 完全未写。后续 decode 阶段 attention 读取 layer 0..7 的 KV 时,
读到空槽/垃圾值 → 索引错乱 → kernel assert。

**见** `qwen3_vl_resume_patch.py:539-558`:
```
loop_start = max(self.start_layer, int(_resume_cut) + 1)   # = P, 跳过 0..P-1
for layer_idx in range(loop_start, self.end_layer):
    hidden_states, residual = self.layers[layer_idx](...)  # 只写 layer P..end 的 KV
```

**修法**:Mac 捕获前 P 层 RoPE 后的 K/V → 传输 → cloud 在 layer loop 前注入
KV cache(layer 0..P-1),再 EXTEND forward layer P..end。

---

## 2. sglang KV cache 接口 (云端实测)

### 2.1 存储 API: `MHATokenToKVPool.set_kv_buffer`

文件: `/usr/local/lib/python3.12/dist-packages/sglang/srt/mem_cache/memory_pool.py:1199`

```python
def set_kv_buffer(
    self,
    layer: RadixAttention,            # 带 .layer_id
    loc: torch.Tensor,                # token slot indices, shape (seq,)
    cache_k: torch.Tensor,            # (seq, n_kv_heads*head_dim) 或 (seq, n_kv_heads, head_dim)
    cache_v: torch.Tensor,
    k_scale=None, v_scale=None,
    layer_id_override: Optional[int] = None,   # ← 关键: 可覆盖 layer_id
):
    layer_id = layer_id_override if layer_id_override is not None else layer.layer_id
    ...
    _set_kv_buffer_impl(
        cache_k, cache_v,
        self.k_buffer[layer_id - self.start_layer],   # 目标 K buffer
        self.v_buffer[layer_id - self.start_layer],   # 目标 V buffer
        loc,                                            # 写入位置
        row_dim=self.row_dim,                          # = n_kv_heads * head_dim
        store_dtype=self.store_dtype,
        ...
    )
```

### 2.2 底层写入: `_set_kv_buffer_impl` / `store_cache`

`memory_pool.py:97` → CUDA 路径走 JIT kernel `store_cache` (`jit_kernel/kvcache.py:51`):
```
k_cache[loc] = k    (本质: 按 token slot 散写)
v_cache[loc] = v
```
- `k` shape `(batch, H*D)`, `k_cache` shape `(num_pages, H*D)`, `indices=loc` shape `(batch,)`
- fallback: `k_cache[indices] = k`(naive)

### 2.3 loc 与 kv_pool 获取路径

- `loc = forward_batch.out_cache_loc`  (EXTEND 时已由 sglang 分配好 token slot)
- `kv_pool = forward_batch.model_runner.token_to_kv_pool`  (MHATokenToKVPool 实例)
- `RadixAttention.forward` (`radix_attention.py:109`) 内部把 K/V reshape 成
  `(-1, tp_k_head_num, head_dim)` 交给 attention backend,backend 调
  `kv_pool.set_kv_buffer(self, loc, k, v)`。

**结论**:`set_kv_buffer` 天然支持按 layer_id 注入任意层的 KV,接口零侵入。

---

## 3. Mac MLX KV 捕获设计

### 3.1 MLX Qwen3 Attention K/V 位置

文件: `.venv-cgc/lib/python3.13/site-packages/mlx_lm/models/qwen3.py:59`

```python
def __call__(self, x, mask=None, cache=None):
    queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
    keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
    values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
    if cache is not None:
        keys = self.rope(keys, offset=cache.offset)     # RoPE 后 K
        keys, values = cache.update_and_fetch(keys, values)
    else:
        keys = self.rope(keys)                           # RoPE 后 K (EXTEND 首次)
    # keys/values: (B, n_kv_heads, L, head_dim)
```

**捕获点**:EXTEND 首次(`cache is None`)时,`keys`(RoPE 后)与 `values` 即
cloud 需要的 KV — sglang KV cache 存的正是 RoPE 后的 K。

### 3.2 捕获方案: monkey-patch Attention.__call__

Mac emitter forward 前 P 层时,wrap `Attention.__call__`,把 RoPE 后的 K/V 存到
外部 list:

```python
# mac_kv_capture.py (Mac 端新增)
_captured_kv = []  # list[(K, V)] per layer, K/V shape (B, n_kv_heads, L, head_dim)
_capture_layers = set(range(0, P))   # 前 P 层
_orig_attn_call = Attention.__call__

def _capturing_attn_call(self, x, mask=None, cache=None):
    B, L, D = x.shape
    queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
    keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
    values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
    if cache is not None:
        queries = self.rope(queries, offset=cache.offset)
        keys = self.rope(keys, offset=cache.offset)
        keys, values = cache.update_and_fetch(keys, values)
    else:
        queries = self.rope(queries)
        keys = self.rope(keys)
    # === 捕获: RoPE 后 K + V ===
    if getattr(self, "_cap_layer_idx", -1) in _capture_layers:
        _captured_kv.append((keys, values))   # 原始 (B, n_kv_heads, L, head_dim)
    output = scaled_dot_product_attention(queries, keys, values, cache=cache, scale=self.scale, mask=mask)
    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return self.o_proj(output)

# 安装时给每层 self_attn 标 layer_idx, 然后 patch
```

### 3.3 传输格式对齐 (MLX → sglang)

| 维度 | MLX 输出 | sglang set_kv_buffer 期望 |
|------|----------|---------------------------|
| K shape | `(B, n_kv_heads, L, head_dim)` | `(seq, n_kv_heads, head_dim)` 或 `(seq, n_kv_heads*head_dim)` |
| 转换 | `keys.transpose(0,2,1,3).reshape(B*L, n_kv_heads, head_dim)` | view 成 `(seq, row_dim)` |
| dtype | mlx float16/bf16 | torch bf16 (cloud) |
| RoPE | 已应用 ✓ | 期望 RoPE 后 ✓ |

```python
# Mac 端序列化: (K, V) → (seq, n_kv_heads*head_dim) numpy/torch bytes
def serialize_kv(kv_list):
    out = []
    for k, v in kv_list:   # (B, n_kv_heads, L, head_dim)
        k_np = np.array(k.transpose(0,2,1,3).reshape(-1, k.shape[1]*k.shape[3]))  # (B*L, n_kv_heads*head_dim)
        v_np = np.array(v.transpose(0,2,1,3).reshape(-1, v.shape[1]*v.shape[3]))
        out.append((k_np, v_np))
    return out
```

### 3.4 GQA / TP 对齐 (关键工程点)

- Qwen3 用 GQA: `n_kv_heads < n_heads`。MLX `self.n_kv_heads` = 全量。
- cloud 若 TP>1,sglang `tp_k_head_num = n_kv_heads // tp_size`,KV cache 只存本 rank 的 KV 子集。
- **方案**:Mac 传全量 `n_kv_heads` 的 KV,cloud 注入前按 `tp_rank` 切片:
  `mac_k[:, tp_rank*n_kv_per:(tp_rank+1)*n_kv_per, :]`。
- 单 GPU(cloud TP=1)时直接全量,无需切片。

---

## 4. cloud KV 注入设计

### 4.1 注入点: `_qwen3_llm_model_forward_resume` layer loop 之前

文件: `CGC_Phase2/qwen3_vl_resume_patch.py`,在 `loop_start` 计算之后、layer loop
之前插入 KV 注入循环。

```python
# === 新增: 注入前 P 层 KV cache (CGC_KV_INJECT) ===
def _cgc_inject_prefill_kv(self, forward_batch, mac_kv_layers, resume_cut):
    """把 Mac 捕获的前 P 层 KV 注入 sglang KV cache (layer 0..resume_cut)。

    mac_kv_layers: list[(k_t, v_t)] 长度 resume_cut+1
                   k_t/v_t shape (seq, n_kv_heads*head_dim), dtype/cloud device
    只在 EXTEND 注入; decode 时前 P 层 KV 已在 pool 中。
    """
    if not mac_kv_layers:
        return
    if forward_batch is None or not forward_batch.forward_mode.is_extend():
        return   # decode 不重注
    kv_pool = forward_batch.model_runner.token_to_kv_pool
    loc = forward_batch.out_cache_loc          # (seq,) token slot indices
    if loc is None:
        print("[CGC_KV_INJECT] WARN out_cache_loc is None, skip", flush=True)
        return
    for layer_idx in range(0, resume_cut + 1):  # 0..P-1
        k_t, v_t = mac_kv_layers[layer_idx]
        attn = self.layers[layer_idx].self_attn   # RadixAttention, .layer_id=layer_idx
        # TP 切片 (cloud TP>1 时)
        # k_t = k_t.view(seq, n_kv_heads, head_dim)[:, attn.tp_k_head_num_slice]
        kv_pool.set_kv_buffer(attn, loc, k_t, v_t)   # layer_id = attn.layer_id
    print(f"[CGC_KV_INJECT] injected KV for layer 0..{resume_cut} "
          f"(seq={k_t.shape[0]}, loc={tuple(loc.shape)})", flush=True)
```

### 4.2 patch 接入位置

在 `_qwen3_llm_model_forward_resume` 现有代码中(`qwen3_vl_resume_patch.py:535` 后):

```python
    hidden_states = _hs
    # ... residual 处理 ...

    # === 新增: 注入前 P 层 KV ===
    _mac_kv = _dec.get("mac_kv_layers") or _dec["abi_descriptor"].get("mac_kv_layers")
    if _mac_kv:
        _cgc_inject_prefill_kv(self, forward_batch, _mac_kv, _resume_cut)

    # --- layer loop (不变) ---
    loop_start = max(self.start_layer, int(_resume_cut) + 1)
    for layer_idx in range(loop_start, self.end_layer):
        ...
```

### 4.3 transport 扩展

`DOPDResumePayloadV2.abi_descriptor` 增字段:
- `mac_kv_layers_b64`: list[base64] — 每层 K/V 的 torch bytes(P 层)
- 或独立 TCP 流:`mac_emit` transport recv hidden_P 后,再 recv P 层 KV
  (复用 `HandoffTransport`,step 不变,额外 pop P 次)

**推荐**:复用现有 `mac_emit` transport,P 层 KV 与 hidden_P 同一会话连续 PUT,
cloud 连续 recv P+1 次(hidden_P + P 层 KV)。零 transport 改动。

### 4.4 decode 阶段处理

- EXTEND(首 token):注入前 P 层 KV ✓
- decode(后续 token):前 P 层 KV 已在 pool,cloud 正常 forward layer P..end
  即可,但 **decode 时 layer 0..P-1 的单 token KV 也需写入** — 否则 decode
  增量 KV 缺失。
- **方案 B(推荐)**:decode 时 cloud 也需跑前 P 层(轻量,单 token)。
  即 `loop_start` 在 decode 时改为 0(全层 forward),只有 EXTEND 走 resume+注入。
  这避开 decode 增量 KV 同步难题。
- **方案 A**:decode 时 Mac 持续传前 P 层单 token KV,cloud 注入。代价:每
  decode step 都有 Mac→cloud 往返,延迟高。不推荐。

---

## 5. 工程量评估

| 模块 | 工作量 | 说明 |
|------|--------|------|
| Mac KV 捕获 | 1-1.5 天 | monkey-patch MLX Attention + 序列化 + 单测 |
| transport 扩展 | 0.5 天 | 复用 mac_emit,连续 recv P+1 次 |
| cloud 注入 patch | 0.5-1 天 | `_cgc_inject_prefill_kv` + 接入 layer loop 前 |
| dtype/shape 对齐 | 1 天 | MLX→torch 转换、RoPE 对齐验证 |
| TP/GQA 切片 | 1 天 | cloud TP>1 时 KV 切片(若需要) |
| decode 方案 B | 1 天 | decode 时 loop_start=0,验证不崩 |
| 端到端测试 | 1-2 天 | step8 复现 + 修复验证 |
| **合计** | **6-8 天** | 关键路径:捕获 + 注入 + decode |

### 风险点

1. **KV 传输带宽**:Qwen3-8B, P=8, n_kv_heads=8, head_dim=128, seq=4096, bf16
   - 每层 K+V = 4096×8×128×2×2 = 16 MB,×8 层 = **128 MB**
   - cq4 压缩后 ~32 MB,家用带宽可接受(~1-2s)
   - 若 seq 更长需流式传输或量化
2. **RoPE 一致性**:MLX `rope(traditional=False)` vs sglang RoPE 实现 — 需数值
   对齐验证(前 P 层 K 的 RoPE 必须与 cloud 等价)。
3. **store_dtype**:sglang KV cache 可能用 fp8 存储(`store_dtype != dtype`),
   注入时需走 `cache_k.view(self.store_dtype)` 路径,与 set_kv_buffer 内部一致。
4. **vl 的 vision token**:VL 模型 EXTEND 含 vision embed,seq 含图像 token,
   `out_cache_loc` 已覆盖,无需特殊处理。

---

## 6. 可行性结论

**高可行性**。sglang `set_kv_buffer` 的 `layer_id_override` 参数天然支持按层注入,
接口零侵入;MLX Qwen3 Attention 的 K/V 在 RoPE 后可直接捕获,格式可干净转换到
sglang 期望。主要工作量在 MLX→sglang 的 dtype/shape/RoPE 对齐与 decode 阶段处理
(方案 B: decode 全层 forward 最稳)。预计 6-8 天可完成 step8 修复。

---

## 7. 实施顺序(建议)

1. cloud 注入 patch + mock Mac KV(本地测 set_kv_buffer 不崩)— 1 天
2. Mac MLX KV 捕获 + 序列化(离线验证 K/V 数值)— 1.5 天
3. transport 传 KV + 端到端 EXTEND 注入 — 1 天
4. decode 方案 B(loop_start=0 for decode)— 1 天
5. RoPE/dtype 对齐 + 端到端 step8 修复验证 — 2 天
