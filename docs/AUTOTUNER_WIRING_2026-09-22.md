# autotuner 配方：把三块接起来的那一层（2026-09-22 §EN-431）

结论先行：**三块本来就都在，缺的确实只是接线层。** 接线层 = `scripts/check/autotune_mmv.py`
（新建，0 重建、0 改 kernel）。**「线是通的」已被实测证明**：设 `CGC_MMV_NSG=2` 时 Metal
**按值编译出新 pipeline** 并打出

```
ggml_metal_library_compile_pipeline: compiling pipeline: base = 'kernel_mul_mv_id_iq2_s_f32', name = 'kernel_mul_mv_id_iq2_s_f32_nsg=2'
  MUL_MAT_ID(type_a=iq2_s,type_b=f32,n_mats=256,n_used=8,b=0,m=512,n=1,k=2048):  11922 runs - 86.69 us/run - 193.53 GFLOPS
```

⇒ env 真的到达 kernel，且 `us/run` 可读。**但这一格只有单点（且 NSG=2 恰好等于编译期默认值
`N_SG_IQ2_S=2`），不是比较，不能当结果引用。**

## 三块各自在哪（以及各自为什么）

| 块 | 谁 | 为什么用它 | 状态 |
|---|---|---|---|
| ① 形状清单 | `shape_roofline.carrier_projections(gguf)`（读 GGUF header） | 唯一诚实的「这颗模型有哪些形状」来源；不靠猜 | ✅ 直接复用 |
| ② 量测半 | `test-backend-ops perf -o <op> -p <regex>` | **取代 `m1_harness` 的量测半**：每形状 µs、真编译的 pipeline | ✅ 已验证可用 |
| ③ 旋钮 | `CGC_MMV_NSG` / `CGC_MMV_NR0` / `CGC_MMV_FUSE` | NSG 是 **Metal function constant**（`FC_MUL_MV+0`）⇒ runtime 按值编译＋快取 | ✅ 真名已订正 |

**② 为什么必须换掉 `m1_harness` 的量测半**（`shape_roofline.py` docstring 已写，这里补判据）：
`m1_harness` 的时间路径是 shell 出去跑 llama-bench、regex 一个整模型 t/s 栏 ⇒
(a) **没有形状隔离**，一个 kernel 的赢无法归因到形状；(b) **单臂噪音 ±27%**，是 3% 门槛的 9 倍
⇒ 一个把某 kernel 改快 5% 的旋钮，在那里**结构上不可见**，在这里一眼可见。
（⚠ `m1_harness.py` 在本 repo 里**找不到实体档**，只出现在 `shape_roofline.py` 的 docstring 与
`MEMORY.md`；它 docstring 里的 TODO 写的是**不存在**的 `CGC_MM_NSG`/`NXPSG` ⇒ 别照抄。）

## 旋钮表（唯一知道「旋钮是什么」的地方）

| 旋钮 | 只作用型别 | 合法值 | 预设 | bit-identity | 备注 |
|---|---|---|---|---|---|
| `CGC_MMV_NSG` | `iq2_s` `iq3_xxs` | 1..32（clamp = M4 threadgroup 1024/32） | unset = `N_SG_IQ2_S` = **2**（`ggml-metal-impl.h:76`） | **保持**（每行点积在单一 simdgroup 内 ⇒ 不改每行归约顺序 ⇒ **M1 不用重新 baseline**） | `ggml-metal-device.cpp:818-833`；**免重建、免改 .metal** |
| `CGC_MMV_NR0` | 同上 | 只有字串 `"8"` | unset = `N_R0_IQ2_S` = **4**（`:75`） | 保持 | `ggml-metal-device.cpp:801-816`；**只认 `strcmp(e,"8")==0`，其它值静默回落 4** ⇒ 别以为扫了 1..8 |
| `CGC_MMV_FUSE` | 同上 | `1` | unset = off | **宣称 bit-identical，实测 777/800 行分歧**（CGC bit-bisect v8） | `ggml-metal-ops.cpp:2800-2830`；**先验是 measured_worse**：15.94 vs 22.07 t/s、draft accept 崩 ⇒ 采用前**必须过 M1 闸门**；且需要 GLU 形状的 case（裸 `MUL_MAT_ID` 到不了它） |

## 接线层做了什么（5 件事，全是「拒绝而不是猜」）

1. **①→② 的 join**：形状 → `-p` regex，钉死 **n=1**（decode GEMV 行數）。
   本模型：`type_a=iq2_s,type_b=\w+,n_mats=256,n_used=8,b=\d+,m=512,n=1,k=2048`。
   （n≥2 的 case 在 harness 裡也有，但那是 prefill 侧的形状，扫它＝扫一个步里没有的形状。）
   ⚠ **两个已踩的坑**：`-p` 是 `std::regex_search((*it)->vars(), ...)`（`test-backend-ops.cpp:10469`），
   `vars()` **不含** `MUL_MAT_ID(...)` 外殼，但**含全部字段** ⇒ 漏写 `type_b=\w+` 会**静默 0 命中**
   （浪费一整轮）；并且这是 **ECMAScript** 正则 ⇒ **不要用 `(?:...)`**。
2. **资格（eligibility）**：旋钮到不了型别就 **REFUSE**，不静默扫。
   实测本载体：gate/up = `iq2_s` ⇒ **ELIGIBLE**；down = **`iq3_s` ⇒ REFUSED**
   （扫一个 kernel 忽略的旋钮＝量噪音，而且看起来像结果）。少数派层（gate/up 各有 1 层 `iq3_s`、
   1 层 `q2_K`；down 有 3 层 `iq4_xs`、1 层 `q3_K`）**明示列出**，不假装代表。
3. **存在性**：只 tuning **在 baseline capture 里真的出现** 的 case；没出现 ⇒ exit 1。
4. **判决规则**（沿用 M-W）：成对 ABBA 顺序、取中位数、赢幅要过 `--threshold`（默认 3%）
   **且每一 rep 同号** ⇒ 否则印 `no evidence`，绝不印「best guess」。
   ⚠ 同号是对**中位数方向**判的：一致变慢 = `LOSS`，不是「no evidence」（selftest 抓到并修掉这个 bug）。
5. **换算**：per-shape 赢 ≠ 端到端赢。没给 `--share`（该 kernel 族占步时的比例）就**不印**端到端数字。

## 用法

```sh
PY=/Users/alexchuang/.workbuddy/binaries/python/versions/3.13.12/bin/python3

# 不碰 GPU：列出会扫什么
$PY scripts/check/autotune_mmv.py --gguf models/gguf/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf --plan

# 不碰 GPU、不读模型：规则自检（parser／旋钮表／资格／判决）
$PY scripts/check/autotune_mmv.py --selftest          # 0 failed

# 实扫（每个形状 × 每个旋钮值 × reps，免重建）
$PY scripts/check/autotune_mmv.py --gguf models/gguf/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf \
    --knob CGC_MMV_NSG --values 1,2,4,8,16,32 --reps 3 --share 0.085 \
    --json Backup/phase_decomp/L3/autotune_nsg.json
```

## 第一次实扫的结果（17:0x，purge 后窗口：usable 9.38 GiB、swap 84%）

盖 md5：`dylib 544b6c94bc7a372d49a5a890bc81a425` ｜ `probe d96c3e19dea0864f592e07426860bd98`
（`--json Backup/phase_decomp/L3/autotune_nsg.json`）。21 次量测（7 设定 × 3 rep，ABBA），
case = 唯一的 decode 形状 `MUL_MAT_ID(iq2_s, m=512, n=1, k=2048, 256 mats, top-8)`：

| 设定 | µs 中位 | vs baseline | 样本 |
|---|---|---|---|
| **unset**（= `N_SG_IQ2_S`=2） | 59.45 | — | 57.64 / 60.17 / 59.45 |
| NSG=1 | 59.62 | −0.29% | 57.72 / 62.04 / 59.62 |
| **NSG=2**（显式） | 58.01 | **+2.42%** | 57.70 / 58.01 / 59.17 |
| NSG=4 | 58.70 | +1.26% | 58.70 / 58.17 / 60.42 |
| NSG=8 | 58.19 | +2.12% | 56.55 / 58.19 / 60.00 |
| NSG=16 | 57.17 | +3.84% | 57.17 / 57.16 / 59.11 |
| NSG=32 | 56.93 | **+4.24%** | 56.93 / 56.76 / 60.65 |

**判決：`no evidence`（NSG=32 那格符号只 2/3 rep 同向）⇒ 不采用。**

- **★ 最重要的副产品＝这颗仪器的噪音底**：`unset` 与 `NSG=2` **按定义是同一个值**
  （unset 就是走 `N_SG_IQ2_S=2`），两者差 **2.42%** ⇒ **这台机器上这个 case 的单格噪音 ≈ 2.4%**
  （对照：llama-bench 单臂 ±27%，差一个数量级 ⇒ ② 换掉 `m1_harness` 量测半是对的、且已量化）。
- **趋势存在但压不过噪音**：1→32 单调 0% → 4.24%，幅度 ≈ 2× 噪音底，且最佳格不过 3-rep 同号规则
  ⇒ 按预先登记的判决规则，结论是「没有证据」，**不是「NSG=32 快 4%」**。
- **纵使 4.24% 全为真，端到端也不可引用**：gate/up 只占 MoE GEMV 全族的
  `2×82 / (2×82+110) = 59.9%`（bpw：iq2_s 82/256、iq3_s 110/256）⇒ gate/up ≈ 步時
  `8.5% × 0.599 = 5.1%` ⇒ **端到端 4.24% × 5.1% ≈ 0.22%**。离 3% 门槛差 13 倍。
- ⚠ **16:50 那次单点 86.69 µs 作废**：它跑的是 4 个 case（n=1..4）的那一轮，与本次的 57–60 µs
  差 50%，原因未定位（多 pipeline 编译／当时窗口）——**单点、未配对，别引用**。
- 结论不变：**autotuner 通道是通的、噪音底从 27% 降到 2.4%，但这颗旋钮的整个射程只有 ~4%，
  端到端 ~0.2%** ⇒ 与 `SHAPE_ROOFLINE_REVIEW` 预估的「全族 2–3%、只扫 gate/up 1–2%」一致（实测更小）。

## 上界（别期待太高；`docs/SHAPE_ROOFLINE_REVIEW_2026-09-22.md`）

- MoE GEMV **全族** = 步时 8.5%（8.355 ms / 97.90 ms）⇒ 把全族写到完美，端到端 **2–3%**；
  **只扫得到 gate/up（`iq2_s`）** ⇒ 剩 **~1–2%**。**都低于 3% 门槛。**
- `--share 0.085` 用的是全族 8.5%；若只想算 gate/up，share 更小 ⇒ 端到端更小。
- in situ 12.0 GB/s ＝ 峰值 12%，居住性惩罚 3.4× ⇒ 「先造 kernel」的顺序是反的；
  autotuner 的价值是**把已有的 knob 钉到最优值**，不是开创 25 的路。

## 顺手修的 bug

- `scripts/check/window_sentinel.py:272`：`mem_state()` 回的是 **3-tuple**，被当 dict 用
  ⇒ 整个 sentinel 一跑就 `TypeError`。已修（并顺手加了 usable < 门槛就拒绝，避免把退化状态记成参考）。
