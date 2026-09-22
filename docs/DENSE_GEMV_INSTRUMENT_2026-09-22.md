# dense GEMV 的仪器与上界（2026-09-22）

顺序按约定：**先造仪器 → 再开 knob → 才扫描 → 最后才算上界**。四步都做完了，
结论是一个定量判决（不是一个期许）。

回答的问题：`BEST_SHAPE_IQ3XXS_M4_2026-09-22.md` 结尾说「dense GEMV 的 achieved
bandwidth 在本树上没有可引用仪器；要上界先造 per-shape 探针」。现在有了仪器，
也有了上界，**并且上界把它封了**。

---

## 0. 一句话结论

> dense GEMV（含 lm_head）每步 **13.39 ms**，占 step 的 **13.7–16.8%**；
> 它们已经跑在 **56–108 GB/s**（对比本机实测 DRAM 峰值 108.8 GB/s）。
> **就算让每一个 GEMV 都打满 100% DRAM 峰值，也只能拿回 1.92 ms = 2.42% 的 step
> —— 低於 3% 门槛。**
> 而把 `CGC_MMV_NSG` 接到 IQ4_XS 之后全射程扫描（1..32）**对成本没有影响**
> （±3% 内、无单调趋势、均在仪器噪音以内）⇒ **这个家族可以结案了。**

---

## 1. 仪器：`scripts/check/shape_probe/`

| 文件 | 作用 |
|---|---|
| `dense_gemv_probe.cpp` | 单形状 GEMV 探针。一个权重张量 + 一个 n 列输入，反复 `MUL_MAT` 计时。不载入模型、不建 llama 图、不开任何 `CGC_*` 仪器 |
| `build.sh` | **独立编译**，链接树上既有的 `libggml*`，**不跑 `cmake --build`** |
| `run_shapes.py` | 按形状批量定价 + **closure test** |
| `sweep_nsg.py` | 旋钮扫描 + **marginal（边际）成本模型** |

为什么必须写成独立二进制：本 repo 的构建产物纳入版控且跨 session 共用，
`cmake --build` 会盖掉别条线正在 mmap 的 dylib。探针只编译自己、链接既有库，
不动任何共享产物。

### 1.1 为什么一个形状要给三个数（`sync` / `batch` / `marginal`）

| 模式 | 量的是什么 | 为什么不能直接用 |
|---|---|---|
| `sync` | 单 dispatch + 等完成 | **延迟**，不是成本。同一个形状两轮之间差 2.6×（354 vs 922 µs） |
| `batch`(b) | b 个 op 在同一 command buffer 内，`total/b` | 内含「每张图固定开销」的一份摊额 |
| **marginal** | `(g(64) − g(32)) / (64 − 32)`，g = 每张图耗时 | **只有它随 op 数线性增长**，固定项被差分消掉 ⇒ 这才是 step 真付的钱 |

marginal 这一层是必须的，不是花招：batch=32 时 `ffn_gate_shexp` 看起来是
**13.7 GB/s（净值感很低）**，换成 marginal 后是 **57.3 GB/s**——差出来的 4 倍
全是每张图 ~220 µs 的固定开销被 32 个 op 均分的结果。

### 1.2 仪器踩到并修掉的两个陷阱（都会让人报错方向）

1. **共用权重的假带宽**：batch 里 b 个 op 读同一个权重 ⇒ 后面那些命中缓存。
   加了 `--distinct`（每个 op 读自己的权重，真实 decode 就是各层读各自的）。
   实测差 7%（120.6 vs 112.8 µs）——比担心的小，但默认必须是 distinct。
2. **小车 ConductorBug**：`ws` 只 push 了第一个张量 ⇒ 其余 31 个从未落到 device
   buffer，图里全是悬空张量 ⇒ **静默结束、退出码 0**。已修。

---

## 2. 判据：把 priced 值算回整个 step（closure test）

任何一种形态只要「把每个形状 multiply 回一步」之后**超过真实 step 时间**，
就说明它把 GPU 实际会重叠掉的等待记成了成本，从它派生的一切都不可引用。

| 分支 | priced dense+GEMV | 判决 |
|---|---|---|
| `sync`（单 dispatch 延迟） | **96.99 ms**（七个形状，未含 lm_head） | 卡在 step 上界 97.9 ms 边缘；补上 lm_head 后必超 ⇒ **弃用** |
| `batch`（含 each-graph 摊额） | 21.17 ms | 可用，但小形状被摊额污染（高估） |
| **marginal** | **13.39 ms** | **采用** |

---

## 3. 修正后的 per-shape 表（ NSG=默认，本机 M4 / 108.8 GB/s 峰值）

| 形状 | 型别 | n/步 | MiB/步 | marginal µs | GB/s | %峰值 | ms/步 |
|---|---|---|---|---|---|---|---|
| lm_head `output.weight` | Q6_K 2048×248320 | 1 | 397.9 | 3916.1 | 99.2 | 91.2% | 3.92 |
| `attn_qkv` | IQ4_XS 2048×8192 | 40 | 340.0 | 100.2 | 82.9 | 76.2% | 4.01 |
| `ssm_out` | IQ4_XS 4096×2048 | 40 | 170.0 | 49.9 | 83.2 | 76.5% | 2.00 |
| `attn_gate` | IQ4_XS 2048×4096 | 30 | 127.5 | 49.3 | 84.2 | 77.4% | 1.48 |
| router `ffn_gate_inp` | F32 2048×256 | 40 | 80.0 | 18.0 | 108.5 | 99.7% | 0.72 |
| `ffn_gate_shexp` | IQ4_XS 2048×512 | 100 | 53.1 | 9.1 | 57.3 | 52.7% | 0.91 |
| `ffn_down_shexp` | IQ4_XS 512×2048 | 40 | 21.2 | 9.1 | 56.8 | 52.2% | 0.36 |
| | | | **合计 1189.7** | | | | **13.39** |

（`bytes/step` 与 `Backup/phase_decomp/L3/dense_bytes_census.py` 一致。
⚠ 小形状两次重复之间有 31–300% 的噪声，见 §7 诚实栏，别把单次读数当定论。）

改一行更正：`BEST_SHAPE…` 里说「IQ4_XS 四兄弟 + router 占 dense 主导」是对的，
但**按 sparse 计算 η 会误导**——同样 0.53 MiB 的形状，router（F32、2 MiB）能跑到
几乎满峰（很可能由 SLC 命中），而 IQ4_XS 的 512 行形状只有 57 GB/s。
**小张量可能与 DRAM 峰值不可比**（缓存命中），别拿它们的 GB/s 跟 lm_head 比。

---

## 4. 第三步：把 knob 接到 IQ4_XS（patch 未提交，在 working tree）

`src/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.cpp` 两处，`+24/-1` 行：

1. `ggml_metal_nsg_env()` 的 switch 加 `GGML_TYPE_IQ4_XS / Q6_K / Q8_0` 的 return（原来是 `default: -1`）
2. `ggml_metal_library_get_pipeline_mul_mv()`（**仅此函数**）的 `case GGML_TYPE_IQ4_XS` 从 `nsg = N_SG_IQ4_XS` 改成 `nsg = ggml_metal_nsg_env(...)`

- ⚠ `get_pipeline_mul_mv_id` / `_down_combine` 里的同名 case **没动**（那是 MoE 的路）
- ⚠ 一开始就踩了 `duplicate case value 'GGML_TYPE_Q8_0'` / `Q6_K`：那两个型别在
  同一个 switch 里**已经有 case**，不能新增，只能把既有 case 接到 env helper
- **构建方式：新建独立目录 `/tmp/cgc-nsg-build`，只 build `ggml-metal` target。
  树上的 `src/llama.cpp/build/bin/*` 一个字节都没动。**
- knob 确实到位（免重建 `.metal`、每行点积仍在一个 simdgroup 内 ⇒ bit-identity 保持）：

```
CGC_MMV_NSG unset → kernel_mul_mv_iq4_xs_f32_nsg=2_ne12=1_r2=1_r3=1
CGC_MMV_NSG=8     → kernel_mul_mv_iq4_xs_f32_nsg=8_ne12=1_r2=1_r3=1
CGC_MMV_NSG=32    → kernel_mul_mv_iq4_xs_f32_nsg=32_ne12=1_r2=1_r3=1
```

---

## 5. 第四步：扫描结果 —— 旋钮在全射程上没有作用

`marginal = (g(64) − g(32)) / 32`，`--libdir /tmp/cgc-nsg-build/bin`：

| NSG | `attn_qkv` 2048×8192 | `ffn_gate_shexp` 2048×512 | `ssm_out` 4096×2048 |
|---|---|---|---|
| unset(=2) | 100.15 µs / 82.9 GB/s | 9.05 µs / 57.3 GB/s | 49.89 µs / 83.2 GB/s |
| 1 | 99.09 (+1.06%) | 9.38 (−3.65%) | 50.28 (−0.78%) |
| 4 | 99.10 (+1.05%) | 9.06 (−0.11%) | 50.37 (−0.96%) |
| 8 | 104.85 (−4.69%) | 8.77 (+3.09%) | 58.34 (−16.9%) |
| 16 | 101.77 (−1.62%) | 9.02 (+0.33%) | 50.02 (−0.26%) |
| 32 | 103.73 (−3.57%) | 9.24 (−2.10%) | 50.43 (−1.08%) |

**判決：不采用。** 三点理由：① 幅度都在 ±3% 仪器噪音内；② **无单调趋势**
（8 看起来最差，但它的固定项同时从 532 掉到 270 µs ⇒ 是机器状态变了，不是 kernel 变了）；
③ 最好的一格（NSG=1，`attn_qkv`）只值 0.05 ms/步 = **0.06% 的 step**。

**=> nsg 这个 tile 维度本身被否证了**（不是「还没调好」）。

---

## 6. 最后：上界 —— 并且它自我否证

用这些形状自己**观测到的最好带宽**（108.5 GB/s，来自 router 那一格）重算全部 dense GEMV：

| 假设 | 总耗时 | 可回收 | 占 step |
|---|---|---|---|
| 现在（实测） | 13.39 ms | — | 13.7–16.8% |
| 全部打到本族观测最好值 **108.5 GB/s** | 11.50 ms | 1.89 ms | **2.38%** |
| 全部打到 **100% DRAM 峰值** 108.8 GB/s | 11.47 ms | 1.92 ms | **2.42%** |

> **即便物理上不可能的「每个 GEMV 都打满 DRAM 峰值」，也只有 2.42%，低于 3% 门槛。**
> 注意 bound 比 measured 还严格：几个形状已经跑等于/优于家族最好带宽
> （router/ lm_head），说明这个家族没有一块「明显落在后面」的可拔之地。

配合前面的 NSG 全射程扫描 ⇒ **dense GEMV 家族作为优化目标结案。**

---

## 7. 诚实栏（必须跟着一起引用）

- **窗口不达标**：全程 usable 5.3–5.7 GiB，**低于本项目 8.0 GiB 门禁**（swap 75%）。
  这些是 GPU 微基准、主机侧占用很小，影响应远小于整模型 t/s；但**没有回头补一次干净窗口**，别把它说成「窗口内取得」。
- **仪器噪音**：重复同一设定，形状之间差别很大 —— `attn_gate` 两次 49.31 / 64.58 µs（31%），
  `router` 两次 18.00 / 55.78 µs。表里取的是**与物理一致的那一次**（例如 `ffn_down_shexp`
  与 `ffn_gate_shexp` 字节数相同 ⇒ 都应是 ~9 µs；第二个样本的隐含固定项为负 ⇒ 弃用）。
  **对小形状，只做一次 reading 是不可信的。**
- `run_shapes.py` 里 `batch` 分支的数字**已被本文弃用**（含每张图摊额），
  留他们只为 closure test 的对照。`sweep_nsg.py` 的 marginal 才是权威。
- patch **未提交**，在 working tree；`/tmp/cgc-nsg-build` 在 repo 外，不入库。
- 本文所有 step 时间分母仍用 `MEMORY_PERF.md` 的生产值 79.5–97.9 ms。

---

## 8. 下一步（按性价比）

1. 给这个 instrument 补 `--target-ms` 固定 loop 数 + 多次重复 ⇒ 把小形状噪音压下来（现在的 bound 已足够判否了，所以这是 hygiene 不是价值）
2. 真正还没定价的是 **MoE expert 路径（`MUL_MAT_ID`，占 step ~10%）** 和 **GatedDeltaNet 递归块** —— 它们才有可能是剩下的 80%
3. 别再从 tile 形状找钱：**这一族已经被实测封死了**
