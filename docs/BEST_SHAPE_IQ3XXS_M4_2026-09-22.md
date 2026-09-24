# 「最佳 shape」是啥：本模型 × Mac16,12(M4/8GPU/16GB) × Metal backend

日期 2026-09-22（17:5x）｜延续 §EN-431/432 autotuner 线

## 0. 一句话

**在这颗模型 + 这台硬件 + 这个 backend 上，"最佳 shape"目前不是一个可选项 —— 支配每步 81%
字节的 IQ4_XS GEMV 家族没有任何 runtime tile 维度（tile 是编译期宏）。**
但这一次把 plumbing 挖到了底：**nsg 早已以 Metal function constant 编译并按值缓存，
只差 plain `mul_mv` 的 switch 没接 env**。所以这不是「判死」，是**一处 switch case 就能开一条还活着的通道**
—— 也是这条线上到目前为止唯一没被结论覆盖的方向。

---

## 1. 先把「shape」摆出来：每步 1.96 GB 长什么样

静态普查（`Backup/phase_decomp/L3/dense_bytes_census.py`，GGUF header，无需 GPU，可引用）：

| 组成 | MiB/步 | 占 subtotal |
|---|---|---|
| trunk dense（40 层） | 827.0 | 42.2% |
| **lm_head `output.weight` Q6_K 2048×248320** | **397.9** | **20.3%** |
| MTP draft head（blk.40） | 303.0 | 15.5% |
| experts（top-8 × 40 层） | 342.5 | 17.5% |
| subtotal | 1870.3 | 100% |

⇒ **dense 占 81.7%**，不是 MEMORY_PERF 里记的 70% —— 之前那个数只算了「trunk dense + experts」，
**漏了 lm_head**。这是本日要更正的一条。

### trunk dense 852 MiB 里只有 5 个形状，占 90.5%

| K×N | 型别 | 张量 | MiB/步 | 占 dense | 是什么 |
|---|---|---|---|---|---|
| 2048×8192 | IQ4_XS | 40 | 340.00 | 39.9% | `attn_qkv` |
| 4096×2048 | IQ4_XS | 40 | 170.00 | 20.0% | `ssm_out` |
| 2048×4096 | IQ4_XS | 30 | 127.50 | 15.0% | `attn_gate` |
| 2048×256 | **F32** | 40 | 80.00 | 9.4% | `ffn_gate_inp`（router） |
| 2048×512 | IQ4_XS | 100 | 53.12 | 6.2% | `ffn_*_shexp` |

**IQ4_XS 四兄弟 = 690.6 MiB = trunk dense 的 81.1%。**
外加 lm_head 一张 Q6_K 张量 = 397.9 MiB，**比 40 层全部专家（342.5 MiB）还多**。

几何（GGUF，权威）：41 层／experts 256／top-8／hidden 2048／expert_ff 512。
MoE 投影：gate/up = IQ2_S (K=2048,N=512)×39 层；down = IQ3_S (K=512,N=2048)×37 层。

---

## 2. 为什么这三个形状「没有 tile 可选」：三条源码事实交叉

| # | 事实 | 出处 |
|---|---|---|
| 1 | 生产配置 **`CGC_MM_BITIDENT=1`** ⇒ small-batch ext family **整体被绕过**（M≤8 全回落 `mul_mv`） | `ggml-metal-ops.cpp:2565`；配置见于 `dense_bw/row.json` env |
| 2 | 就算关掉 BITIDENT：**IQ4_XS 不在 ext family 的型别清单**。清单只有两组：(F32,F16,BF16,Q1_0,Q2_0,Q4_0,Q4_1,Q5_0,Q5_1,Q8_0,MXFP4,IQ4_NL) 需 ne11∈[2,8]；(Q4_K,Q5_K,Q6_K,Q2_K,Q3_K) 需 ne11∈[4,8] | `ggml-metal-ops.cpp:2566-2589` |
| 3 | **`CGC_MMV_NSG` / `CGC_MMV_NR0` 到不了 IQ4_XS**：两个 env 函数的 switch 都有 `default: return -1`，只有 IQ2_S / IQ3_XXS 有 case | `ggml-metal-device.cpp:802-836` |

⇒ 三条一起：占 trunk dense **81.1% 字节**的那 4 个 IQ4_XS 形状，
**既进不了 ext family，也拿不到任何一个 runtime knob**。tile 全部是编译期宏：

```
ggml-metal-impl.h:87  #define N_R0_IQ4_XS 2
ggml-metal-impl.h:88  #define N_SG_IQ4_XS 2
```

对照：关掉 BITIDENT 后**唯一**能进 ext family 的 dense 项是 **F32 router 80 MiB（9.4%）**；
而 ext family 自己的 tile 选择也藏着一句自白 —— `nsg = 2` 硬编码，源码注释写
`// note: not sure how optimal are those across all different hardware. there might be something cleverer`
（`ggml-metal-ops.cpp:2604`），且 nxpsg 只按 **`ne00 % 256` / `ne00 % 128` 的可整除性**分档，不做带宽/占用率优化。

---

## 3. ★ 真正的发现：plumbing 早就通了，缺的是一处 switch

`CGC_MMV_NSG` 现在打不到 IQ4_XS，不是因为信道不存在。跑一次就有证据 —— metal 真的把它编进了 pipeline 名：

```
compiling pipeline: base = 'kernel_mul_mv_iq4_xs_f32',
                    name = 'kernel_mul_mv_iq4_xs_f32_nsg=2_ne12=1_r2=1_r3=1'
```

而它是 **function constant**，按值编译缓存：

```
ggml-metal-device.cpp:1013   ggml_metal_cv_set_int16(cv, nsg, FC_MUL_MV + 0);
ggml-metal-device.cpp:1007   snprintf(name, ..., "%s_nsg=%d_ne12=%d_r2=%d_r3=%d", base, nsg, ...);
```

⇒ **改 nsg：免重建、免改 `.metal`、runtime 按值缓存**；且按 `ggml_metal_nsg_env` 的注释，
每行点积在单一 simdgroup 内完成 ⇒ **nsg 不改每行归约顺序 ⇒ bit-identity 保持，M1 不必重 baseline**。

唯一缺的一环 —— plain `mul_mv` 的 type switch **写死宏，没走 env**：

```
ggml-metal-device.cpp:989    case GGML_TYPE_IQ4_XS:
                                 nsg = N_SG_IQ4_XS;      // ← 这里
                                 nr0 = N_R0_IQ4_XS;
```

而 `impl` 本身**已经是 NR0 模板**（改 NR0 也只是常数→function constant 的同款动作）：

```
ggml-metal.metal:9801  kernel_mul_mv_iq4_xs_f32_impl<N_R0_IQ4_XS, ...>(...);
```

**⇒ Patch = 一处 switch case**（照 IQ2_S 在 `ggml_metal_nsg_env` 里已有的分支抄），
`CGC_MMV_NSG` 就能射到占 81% dense 字节的 IQ4_XS GEMV，然后 `scripts/check/autotune_mmv.py`
直接能扫。**这是新建 knob，不是扫一个既有 knob。**

---

## 4. 诚实栏：上界目前不能引用，因为仪器不存在

想把上面的 actor 排期望值，需要 dense GEMV 的 **achieved GB/s**。目前树上拿不到：

- **`CGC-GPUOPS` 全数是「开仪器」状态**：所有含它的轮次 step total = 138.99 / 140.99 / 150.94 / … / 210.18 ms，
  而生产 busy 是 79.5–97.9 ms（同一枚 PAT）。用它算 GB/s 会得到 router = **258 GB/s**，
  是实测峰值 108.8 GB/s 的 2.4 倍 ⇒ **荒谬 ⇒ 不可用**。
- **`test-backend-ops` 的 `MUL_MAT` 内置 case 是 4096×14336**，**不含本模型任何一个支配形状**
  （实测 `-p 'type_a=iq4_xs,m=8192,n=1,k=2048'` ⇒ **0 命中**）
  ⇒ autotuner 的接线层对 dense **无 case 可接**，这是 §EN-431 那层的覆盖缺口，不是脚本 bug。

⇒ **dense GEMV 的 achieved bandwidth 在本树上没有可引用仪器**。
要上界，先要有 per-shape 探针（给 test-backend-ops 注册本模型形状，或写形状隔离 probe）。
在那之前，**任何「dense 能拿回 X%」的数字都是猜的，本项目口径下不得引用。**

（结构可参考但绝对值同样不可引用 —— dense_bw 轮 MUL_MAT 按 node name 的 top：
`z-` 23.4%／`linear_attn` 13.1%／`ffn_gate`+`ffn_up`+`shared_expert_gate` 各 8.5%／
`dnqkv_proj`+`dnbeta_proj` 各 6.6%。该轮 t/s=3.19、thermal HEAVY、hit 57.8%，已被判不可引用。）

---

## 5. 已经判死的，别重复作业

| 方向 | 结论 | 出处 |
|---|---|---|
| MoE gate/up IQ2_S 的 NSG 全射程 | **no evidence**：1→32 只有 ~4%，噪音底实测 2.4%，端到端 ≈0.22%（差 3% 门槛 13 倍） | `Backup/phase_decomp/L3/autotune_nsg.json`、`docs/AUTOTUNER_WIRING_2026-09-22.md` |
| MoE down 的 fused down-combine | 先验 measured_worse + 777/800 分歧；且**只覆盖 expert down**（IQ3_S/IQ4_XS/Q3_K），**碰不到 dense attention/GDN** | `MEMORY.md` autotuner 行、`ggml-metal.metal:11901` |
| 关 MTP 省 CPY / k 步长 | MTP off 9.82 vs on 12.62 ⇒ **−28.5%，别动** | `MEMORY_PERF.md` |

---

## 6. 下一步（按「先有仪器，才有结论」排序）

1. **造仪器**（前置）：给 dense 支配形状做可引用探针。
   `test-backend-ops` 的 MUL_MAT case 是写死的 ⇒ 加 case 需改源码并重建（会碰别线的 build 产物，动前先看 `git status`）。
2. **开 knob**：`ggml-metal-device.cpp:989` 的 IQ4_XS case 接上 env override（照 IQ2_S 抄）。
   ⇒ 免重建扫描，bit-identity 保持。
3. **扫**：`autotune_mmv.py --knob CGC_MMV_NSG --values 1,2,4,8,16,32 --run --reps 3`。
   ⚠ selftest/plan 必须先重跑一次确认新 case 存在性；没 `--share` 就不许印端到端数字。
4. **算上界**：拿到 §4 的 per-shape GB/s 之后才算，之前不算。

## 7. 环境 / 口径

- 期间 purge 了一次（`window_sentinel.py --purge`），purge 后 usable 8.78 GiB、swap 4125/5120 (81%)。
- `models/gguf/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf` 与 `Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf`
  是 **hardlink 到同一份 13.66 GB**（`ls -laL` 已验证）⇒ 两个名字节数都有效，别当成两个模型。
- macOS 无 `timeout` 命令，别在命令里用（本日踩过一次）。
